"""Orchestrator.optimize_sha() の CPU モックテスト (GPU 不要)。

run_in_worker をモックして Successive Halving のラウンド制御・絞り込み・
キャッシュ HIT/MISS・全失敗・Discord 通知を検証する。
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from forge.benchmark.statistics import BenchmarkResult
from forge.cache.repository import KernelRepository
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
