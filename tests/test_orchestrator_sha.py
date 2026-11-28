"""Orchestrator.optimize_sha() の CPU モックテスト (GPU 不要)。

run_in_worker をモックして Successive Halving のラウンド制御・絞り込み・
キャッシュ HIT/MISS・全失敗・Discord 通知を検証する。
"""

from __future__ import annotations

import tempfile
import threading
import warnings
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from forge.benchmark.statistics import BenchmarkResult
from forge.cache.key import CacheKey
from forge.cache.repository import CachedKernel, KernelRepository
from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.orchestrator import Orchestrator
from forge.runtime.worker import WorkerResult
from forge.search.grid import GridSearch
from forge.search.space import SearchSpace

# --- ヘルパー ---


def _rmsnorm_spec() -> KernelSpec:
    return KernelSpec(
        op_type="rmsnorm",
        input_specs=(
            TensorSpec((4, 64), torch.float32, True),
            TensorSpec((64,), torch.float32, True),
        ),
        output_specs=(TensorSpec((4, 64), torch.float32, True),),
        constants={"eps": 1e-6},
        graph_hash="rmsnorm_sha_v1",
        constraints=(),
    )


def _tiny_grid(n: int = 4) -> GridSearch:
    """n 候補を生成する最小グリッド。"""
    space = SearchSpace(
        block_sizes=[64, 128],
        num_warps=[4, 8],
        acc_dtypes=["fp32"],
        variants=["single_row"],
    )
    return GridSearch(space)


def _ok_result(median_us: float = 10.0) -> WorkerResult:
    bench = BenchmarkResult(median_us=median_us, p20_us=median_us - 1, p80_us=median_us + 1)
    baseline = BenchmarkResult(median_us=20.0, p20_us=19.0, p80_us=21.0)
    return WorkerResult(
        success=True,
        correct=True,
        candidate=bench,
        baseline=baseline,
        baseline_name="F.rms_norm",
    )


def _fail_result() -> WorkerResult:
    return WorkerResult(success=False, error="simulated crash")


def _incorrect_result() -> WorkerResult:
    return WorkerResult(success=True, correct=False)


# --- テスト ---


class TestOptimizeSHACPU:
    def test_cache_miss_returns_search_result(self):
        """キャッシュ MISS → SHA → SearchResult を返す。"""
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo)
            spec = _rmsnorm_spec()

            with patch("forge.orchestrator.run_in_worker", return_value=_ok_result()):
                result = orch.optimize_sha(
                    spec, initial_budget=4, halving_rounds=2, search=_tiny_grid()
                )

            assert not result.cache_hit
            assert result.best_params is not None
            assert result.best_benchmark is not None
            repo.close()

    def test_cache_hit_skips_search(self):
        """1 回目で best がキャッシュされ、2 回目は HIT してスキップする。"""
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo)
            spec = _rmsnorm_spec()

            with patch("forge.orchestrator.run_in_worker", return_value=_ok_result()):
                first = orch.optimize_sha(
                    spec, initial_budget=4, halving_rounds=1, search=_tiny_grid()
                )
            assert not first.cache_hit

            with patch("forge.orchestrator.run_in_worker") as mock_worker:
                second = orch.optimize_sha(
                    spec, initial_budget=4, halving_rounds=1, search=_tiny_grid()
                )
                mock_worker.assert_not_called()

            assert second.cache_hit
            assert second.best_params == first.best_params
            repo.close()

    def test_cache_hit_ignores_entry_slower_than_eager(self):
        """eager より遅い (0.9x < min_speedup) 保存済みエントリは HIT せず再探索する (#526)。"""
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo)
            spec = _rmsnorm_spec()
            key = CacheKey.from_spec_and_env(spec)
            repo.put(
                key,
                CachedKernel(
                    cache_key=key,
                    params={
                        "block_size": 256,
                        "num_warps": 2,
                        "num_stages": 1,
                        "acc_dtype": "fp32",
                    },
                    kernel_code="# stale",
                    benchmark_json=BenchmarkResult(
                        median_us=10.0, p20_us=10.0, p80_us=10.0
                    ).to_dict(),
                    created_at=datetime.now(UTC),
                    baseline_us=9.0,
                ),
            )
            with patch(
                "forge.orchestrator.run_in_worker", return_value=_ok_result()
            ) as mock_worker:
                result = orch.optimize_sha(
                    spec, initial_budget=4, halving_rounds=1, search=_tiny_grid()
                )
            assert not result.cache_hit
            mock_worker.assert_called()
            repo.close()

    def test_all_fail_returns_no_best(self):
        """全候補が FAIL のとき best_params=None。"""
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo)
            spec = _rmsnorm_spec()

            with patch("forge.orchestrator.run_in_worker", return_value=_fail_result()):
                result = orch.optimize_sha(
                    spec, initial_budget=4, halving_rounds=2, search=_tiny_grid()
                )

            assert result.best_params is None
            assert result.best_benchmark is None
            assert len(result.experiments) > 0
            assert all(not e.success for e in result.experiments)
            repo.close()

    def test_all_incorrect_returns_no_best(self):
        """全候補が INCORRECT のとき best_params=None。"""
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo)
            spec = _rmsnorm_spec()

            with patch("forge.orchestrator.run_in_worker", return_value=_incorrect_result()):
                result = orch.optimize_sha(
                    spec, initial_budget=4, halving_rounds=2, search=_tiny_grid()
                )

            assert result.best_params is None
            assert all(not e.correct for e in result.experiments)
            repo.close()

    def test_halving_reduces_candidates_each_round(self):
        """ラウンドを重ねるごとに _eval_one の呼び出し数が減る（上位 50% 絞り込み）。"""

        def _counting_worker(*args, **kwargs):  # type: ignore[no-untyped-def]
            return _ok_result()

        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo)
            spec = _rmsnorm_spec()

            with patch("forge.orchestrator.run_in_worker", side_effect=_counting_worker) as mock:
                orch.optimize_sha(spec, initial_budget=4, halving_rounds=3, search=_tiny_grid())
                # 4 (r1) + 2 (r2) + 1 (r3) = 7 回以下
                assert mock.call_count <= 7
            repo.close()

    def test_warmup_repeat_change_per_round(self):
        """各ラウンドで run_in_worker に渡す warmup/repeat が増加する。"""
        seen: list[tuple[int, int]] = []

        def _spy_worker(*args, **kwargs):  # type: ignore[no-untyped-def]
            seen.append((kwargs.get("warmup", -1), kwargs.get("repeat", -1)))
            return _ok_result()

        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo)
            spec = _rmsnorm_spec()

            with patch("forge.orchestrator.run_in_worker", side_effect=_spy_worker):
                orch.optimize_sha(spec, initial_budget=4, halving_rounds=3, search=_tiny_grid())

        # 各ラウンドの warmup は 5 → 10 → 25 と増えるはず
        warmups = [w for w, _ in seen]
        assert warmups[:4] == [5] * min(4, len(warmups))  # round 1
        if len(seen) > 4:
            r2_start = 4
            assert seen[r2_start][0] == 10  # round 2 warmup

    def test_discord_notified_on_success(self):
        """ベスト候補が見つかったとき Discord 通知が呼ばれる。"""
        notifier = MagicMock()
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo, notifier=notifier)
            spec = _rmsnorm_spec()

            with patch("forge.orchestrator.run_in_worker", return_value=_ok_result()):
                orch.optimize_sha(spec, initial_budget=4, halving_rounds=2, search=_tiny_grid())

        notifier.send_optimization_complete.assert_called_once()
        assert notifier.send_optimization_complete.call_args.kwargs["op_name"] == "rmsnorm"

    def test_discord_notified_on_all_fail(self):
        """全候補が失敗したとき Discord エラー通知が呼ばれる。"""
        notifier = MagicMock()
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo, notifier=notifier)
            spec = _rmsnorm_spec()

            with patch("forge.orchestrator.run_in_worker", return_value=_fail_result()):
                orch.optimize_sha(spec, initial_budget=4, halving_rounds=2, search=_tiny_grid())

        notifier.send_optimization_error.assert_called_once()
        notifier.send_optimization_complete.assert_not_called()

    def test_returns_search_result_type(self):
        """戻り値が SearchResult であり既存 optimize() と同じ API を持つ。"""
        from forge.orchestrator import SearchResult

        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo)
            spec = _rmsnorm_spec()

            with patch("forge.orchestrator.run_in_worker", return_value=_ok_result()):
                result = orch.optimize_sha(
                    spec, initial_budget=4, halving_rounds=1, search=_tiny_grid()
                )

        assert isinstance(result, SearchResult)
        assert result.speedup is not None
        repo.close()

    def test_small_budget_emits_warning(self):
        """initial_budget < 64 のとき UserWarning が発行される。"""
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo)
            spec = _rmsnorm_spec()

            with patch("forge.orchestrator.run_in_worker", return_value=_ok_result()):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    orch.optimize_sha(
                        spec, initial_budget=32, halving_rounds=1, search=_tiny_grid()
                    )

        user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warns) == 1
        assert "initial_budget=32" in str(user_warns[0].message)
        repo.close()

    def test_large_budget_no_warning(self):
        """initial_budget >= 64 のとき警告なし。"""
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo)
            spec = _rmsnorm_spec()

            with patch("forge.orchestrator.run_in_worker", return_value=_ok_result()):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    orch.optimize_sha(
                        spec, initial_budget=64, halving_rounds=1, search=_tiny_grid()
                    )

        user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warns) == 0
        repo.close()

    def test_zero_candidates_returns_no_best(self):
        """search が候補を 0 件返した場合、round_results が未定義にならず best_params=None を返す。

        回帰テスト: #193 — round_results をループ外で初期化せずに参照すると NameError になる。
        """

        class _EmptySearch:
            def generate(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                return []

        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo)
            spec = _rmsnorm_spec()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                result = orch.optimize_sha(
                    spec, initial_budget=4, halving_rounds=2, search=_EmptySearch()
                )

        assert result.best_params is None
        assert result.best_benchmark is None
        assert result.experiments == []
        repo.close()

    def test_halving_rounds_zero_returns_no_best(self):
        """halving_rounds=0 のとき outer ループが回らず round_results が未定義にならない。

        回帰テスト: #193 — range(1, 0+1) は空なのでループ本体が未実行になる。
        """
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo)
            spec = _rmsnorm_spec()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                result = orch.optimize_sha(
                    spec, initial_budget=4, halving_rounds=0, search=_tiny_grid()
                )

        assert result.best_params is None
        assert result.best_benchmark is None
        repo.close()

    def test_round_results_empty_when_all_rounds_break_no_deadcode(self):
        """全ラウンドが break（round_results 空）のとき SearchResult が不整合にならない。

        回帰テスト: #268 — all_experiments を走査する中間ループ（デッドコード）を削除後、
        round_results が空のまま _finalize に到達しても best_params=best_bench=None が
        一貫して返され、incorrect_count が all_experiments から正しく集計される。
        """
        # ラウンド 1 のみ実行され、全候補が INCORRECT → round_results が空 → break
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo)
            spec = _rmsnorm_spec()

            with patch("forge.orchestrator.run_in_worker", return_value=_incorrect_result()):
                result = orch.optimize_sha(
                    spec, initial_budget=4, halving_rounds=3, search=_tiny_grid()
                )

        # round_results 空なので best は None
        assert result.best_params is None
        assert result.best_benchmark is None
        # experiments は評価済みのものが入っている
        assert len(result.experiments) > 0
        # incorrect_count は all_experiments から一貫して集計される
        assert result.incorrect_count == len(result.experiments)
        assert result.failed_count == 0
        repo.close()

    def test_cancel_at_first_candidate_of_round2_keeps_round1_best(self):
        """ラウンド 2 の 1 件目で打ち切られても、ラウンド 1 の best を捨てない。

        回帰テスト: ラウンド先頭で round_results を空に再初期化していたため、
        ラウンド 2 で有効結果が 0 件だと best_params=None になっていた。
        """
        cancel_event = threading.Event()

        def _progress(event):  # type: ignore[no-untyped-def]
            # ラウンド 2 の開始ログ直後（候補ループ先頭の _should_stop 前）にキャンセル
            if "[SHA round 2/" in event.label:
                cancel_event.set()

        notifier = MagicMock()
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(
                repo=repo, cancel_event=cancel_event, notifier=notifier, progress=_progress
            )

            with patch("forge.orchestrator.run_in_worker", return_value=_ok_result(10.0)) as mock:
                result = orch.optimize_sha(
                    _rmsnorm_spec(), initial_budget=4, halving_rounds=3, search=_tiny_grid()
                )

            assert mock.call_count == 4  # ラウンド 1 の 4 候補のみ
            assert result.best_params is not None
            assert result.best_benchmark is not None
            assert result.best_benchmark.median_us == 10.0
            best = [e for e in result.experiments if e.is_best]
            assert len(best) == 1
            assert result.experiments.index(best[0]) < 4
            notifier.send_failure.assert_not_called()
            repo.close()

    def test_round2_all_failed_falls_back_to_round1_best(self):
        """ラウンド 2 が全件失敗でも、ラウンド 1 の best にフォールバックする。"""

        def _worker(*args, **kwargs):  # type: ignore[no-untyped-def]
            if kwargs.get("warmup") == 5:
                return _ok_result(10.0)
            return _fail_result()

        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo)

            with patch("forge.orchestrator.run_in_worker", side_effect=_worker):
                result = orch.optimize_sha(
                    _rmsnorm_spec(), initial_budget=4, halving_rounds=3, search=_tiny_grid()
                )

            assert result.best_params is not None
            assert result.best_benchmark is not None
            assert result.best_benchmark.median_us == 10.0
            best = [e for e in result.experiments if e.is_best]
            assert len(best) == 1
            assert result.experiments.index(best[0]) < 4
            repo.close()

    def test_partial_round2_does_not_override_complete_round1(self):
        """ラウンド 2 が途中で打ち切られたら、その部分結果は前ラウンドの完全な結果を上書きしない (#503)。

        ラウンド 2 の 1 件目だけ評価してキャンセルし、その 1 件がラウンド 1 の best より
        遅い。忠実度（warmup/repeat）が違うので直接比べず、完全なラウンド 1 の best と、
        それを測ったときの baseline を採る。
        """
        cancel_event = threading.Event()
        logs: list[str] = []
        calls = {"n": 0}

        def _worker(*args, **kwargs):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            if kwargs.get("warmup") == 5:  # ラウンド 1: 候補ごとに 10, 11, 12, 13us / baseline 20us
                return _ok_result(10.0 + (calls["n"] - 1))
            # ラウンド 2 の 1 件目で打ち切る。部分結果は 15us / baseline 40us
            cancel_event.set()
            bench = BenchmarkResult(median_us=15.0, p20_us=14.0, p80_us=16.0)
            baseline = BenchmarkResult(median_us=40.0, p20_us=39.0, p80_us=41.0)
            return WorkerResult(
                success=True, correct=True, candidate=bench, baseline=baseline, baseline_name="F"
            )

        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(
                repo=repo,
                cancel_event=cancel_event,
                progress=lambda ev: logs.append(ev.label),
            )
            with patch("forge.orchestrator.run_in_worker", side_effect=_worker) as mock:
                result = orch.optimize_sha(
                    _rmsnorm_spec(), initial_budget=4, halving_rounds=3, search=_tiny_grid()
                )

            assert mock.call_count == 5  # ラウンド 1 の 4 件 + ラウンド 2 の 1 件
            assert result.best_benchmark is not None
            assert result.best_benchmark.median_us == 10.0
            assert result.baseline_benchmark is not None
            assert result.baseline_benchmark.median_us == 20.0
            assert result.adopted is True  # 10us vs 20us = 2x
            best = [e for e in result.experiments if e.is_best]
            assert len(best) == 1
            assert result.experiments.index(best[0]) < 4
            assert any("partial" in m and "round 1" in m for m in logs)
            repo.close()

    def test_is_best_marks_final_round_experiment(self):
        """正常終了時、is_best は最終ラウンド（高忠実度）の実験に付く。"""
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo)

            with patch("forge.orchestrator.run_in_worker", return_value=_ok_result(10.0)):
                result = orch.optimize_sha(
                    _rmsnorm_spec(), initial_budget=4, halving_rounds=3, search=_tiny_grid()
                )

            exps = result.experiments
            assert len(exps) == 7  # 4 (r1) + 2 (r2) + 1 (r3)
            best_idx = [i for i, e in enumerate(exps) if e.is_best]
            assert best_idx == [6]
            assert exps[6].params == result.best_params
            repo.close()
