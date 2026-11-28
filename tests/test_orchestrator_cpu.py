"""Orchestrator.optimize() の CPU モックテスト (GPU 不要)。

run_in_worker をモックして、キャッシュ HIT/MISS・全失敗・全不正確・Discord 通知を検証する。
"""

from __future__ import annotations

import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from forge.benchmark.statistics import BenchmarkResult
from forge.cache.key import CacheKey
from forge.cache.repository import CachedKernel, KernelRepository
from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.orchestrator import Orchestrator
from forge.runtime.worker import WorkerResult
from forge.search.grid import GridSearch
from forge.search.params import SearchParams
from forge.search.space import SearchSpace

# --- ヘルパー ---

_STALE_PARAMS = SearchParams(block_size=128, num_warps=2, num_stages=1, acc_dtype="fp32")


def _rmsnorm_spec() -> KernelSpec:

    return KernelSpec(
        op_type="rmsnorm",
        input_specs=(
            TensorSpec((4, 64), torch.float32, True),
            TensorSpec((64,), torch.float32, True),
        ),
        output_specs=(TensorSpec((4, 64), torch.float32, True),),
        constants={"eps": 1e-6},
        graph_hash="rmsnorm_v1",
        constraints=(),
    )


def _tiny_grid() -> GridSearch:
    space = SearchSpace(
        block_sizes=[64],
        num_warps=[4],
        acc_dtypes=["fp32"],
        variants=["single_row"],
    )
    return GridSearch(space)


def _ok_worker_result() -> WorkerResult:
    bench = BenchmarkResult(median_us=10.0, p20_us=9.0, p80_us=11.0)
    baseline = BenchmarkResult(median_us=20.0, p20_us=19.0, p80_us=21.0)
    return WorkerResult(
        success=True,
        correct=True,
        candidate=bench,
        baseline=baseline,
        baseline_name="F.rms_norm",
    )


def _fail_worker_result() -> WorkerResult:
    return WorkerResult(success=False, error="simulated CUDA crash")


def _incorrect_worker_result() -> WorkerResult:
    return WorkerResult(success=True, correct=False)


# --- テスト ---


class TestOptimizeCPU:
    """run_in_worker をモックした CPU テスト群。"""

    def test_cache_miss_saves_best_params(self):
        """キャッシュ MISS → 探索 → ベスト候補を保存して返す。"""
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo)
            spec = _rmsnorm_spec()

            with patch("forge.orchestrator.run_in_worker", return_value=_ok_worker_result()):
                result = orch.optimize(spec, budget=1, search=_tiny_grid())

            assert not result.cache_hit
            assert result.best_params is not None
            assert result.best_benchmark is not None
            assert result.best_benchmark.median_us == pytest.approx(10.0)
            repo.close()

    def test_cache_hit_skips_search(self):
        """1 回目で best がキャッシュされ、2 回目は HIT してスキップする。"""
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo)
            spec = _rmsnorm_spec()

            with patch("forge.orchestrator.run_in_worker", return_value=_ok_worker_result()):
                first = orch.optimize(spec, budget=1, search=_tiny_grid())
            assert not first.cache_hit

            with patch("forge.orchestrator.run_in_worker") as mock_worker:
                second = orch.optimize(spec, budget=1, search=_tiny_grid())
                mock_worker.assert_not_called()

            assert second.cache_hit
            assert second.best_params == first.best_params
            assert second.experiments == []
            repo.close()

    @pytest.mark.parametrize(
        ("baseline_us", "median_us", "adopted"),
        [
            (9.0, 10.0, False),  # eager より遅い (0.9x < min_speedup 1.03): #475 より前の遺物
            (None, 10.0, False),  # baseline 無しの旧エントリ: 比べられないので採用しない
            (20.0, 10.0, True),  # 2.0x: 従来どおり HIT
        ],
        ids=["slow", "no-baseline", "fast"],
    )
    def test_cache_hit_adoption_by_stored_speedup(self, baseline_us, median_us, adopted, caplog):
        """保存済みの速度比が min_speedup 未満（比べられない場合を含む）なら HIT せず再探索する (#526)。"""
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo)
            spec = _rmsnorm_spec()
            key = CacheKey.from_spec_and_env(spec)
            stale_params = _STALE_PARAMS.to_dict()
            repo.put(
                key,
                CachedKernel(
                    cache_key=key,
                    params=stale_params,
                    kernel_code="# stale",
                    benchmark_json=BenchmarkResult(
                        median_us=median_us, p20_us=median_us, p80_us=median_us
                    ).to_dict(),
                    created_at=datetime.now(UTC),
                    baseline_us=baseline_us,
                ),
            )

            with (
                caplog.at_level(logging.WARNING, logger="forge.orchestrator"),
                patch(
                    "forge.orchestrator.run_in_worker", return_value=_ok_worker_result()
                ) as mock_worker,
            ):
                result = orch.optimize(spec, budget=1, search=_tiny_grid())

            if adopted:
                assert result.cache_hit
                mock_worker.assert_not_called()
                assert result.best_params is not None
                assert result.best_params.to_dict() == stale_params
                assert "cache entry ignored" not in caplog.text
            else:
                assert not result.cache_hit
                mock_worker.assert_called()
                assert "cache entry ignored" in caplog.text
                # 再探索の結果 (speedup 2.0x) で上書きされ、次回は HIT する
                stored = repo.get(key)
                assert stored is not None
                assert stored.baseline_us == pytest.approx(20.0)
                assert stored.benchmark_json["median_us"] == pytest.approx(10.0)
            repo.close()

    @pytest.mark.parametrize(
        ("baseline_us", "median_us"),
        [
            (float("nan"), 10.0),
            (float("inf"), 10.0),  # inf / 10 = inf >= min_speedup と誤って採用されうる
            (-20.0, 10.0),
            ("abc", 10.0),  # SQLite は数値に直せない文字列もそのまま保存する
            ("inf", 10.0),
            (20.0, float("nan")),
            (20.0, float("inf")),
            (20.0, -10.0),
            (20.0, "abc"),
        ],
        ids=[
            "baseline-nan",
            "baseline-inf",
            "baseline-negative",
            "baseline-str",
            "baseline-str-inf",
            "median-nan",
            "median-inf",
            "median-negative",
            "median-str",
        ],
    )
    def test_cache_hit_rejects_non_finite_or_unreadable_values(self, baseline_us, median_us):
        """baseline_us / median_us が有限の正の数でなければ、例外にせず不採用にする (#526)。"""
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo)
            spec = _rmsnorm_spec()
            key = CacheKey.from_spec_and_env(spec)
            bench = BenchmarkResult(median_us=10.0, p20_us=10.0, p80_us=10.0).to_dict()
            bench["median_us"] = median_us
            repo.put(
                key,
                CachedKernel(
                    cache_key=key,
                    params=_STALE_PARAMS.to_dict(),
                    kernel_code="# stale",
                    benchmark_json=bench,
                    created_at=datetime.now(UTC),
                    baseline_us=baseline_us,
                ),
            )
            with patch(
                "forge.orchestrator.run_in_worker", return_value=_ok_worker_result()
            ) as mock_worker:
                result = orch.optimize(spec, budget=1, search=_tiny_grid())
            assert not result.cache_hit
            mock_worker.assert_called()
            repo.close()

    def test_cache_hit_ignored_stale_entry_is_kept_when_research_not_adopted(self):
        """再探索でも eager に勝てなければ古いエントリは上書きされず残り、毎回 HIT 判定で弾かれる。"""
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo)
            spec = _rmsnorm_spec()
            key = CacheKey.from_spec_and_env(spec)
            repo.put(
                key,
                CachedKernel(
                    cache_key=key,
                    params=_STALE_PARAMS.to_dict(),
                    kernel_code="# stale",
                    benchmark_json=BenchmarkResult(
                        median_us=10.0, p20_us=10.0, p80_us=10.0
                    ).to_dict(),
                    created_at=datetime.now(UTC),
                    baseline_us=9.0,
                ),
            )
            slow = WorkerResult(
                success=True,
                correct=True,
                candidate=BenchmarkResult(median_us=10.0, p20_us=10.0, p80_us=10.0),
                baseline=BenchmarkResult(median_us=9.0, p20_us=9.0, p80_us=9.0),
                baseline_name="F.rms_norm",
            )
            with patch("forge.orchestrator.run_in_worker", return_value=slow) as mock_worker:
                for _ in range(2):
                    result = orch.optimize(spec, budget=1, search=_tiny_grid())
                    assert not result.cache_hit
                    assert not result.adopted
            assert mock_worker.call_count == 2
            assert repo.get(key) is not None
            repo.close()

    def test_all_fail_returns_no_best(self):
        """全候補が FAIL のとき best_params=None。"""
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo)
            spec = _rmsnorm_spec()

            with patch("forge.orchestrator.run_in_worker", return_value=_fail_worker_result()):
                result = orch.optimize(spec, budget=1, search=_tiny_grid())

            assert result.best_params is None
            assert result.best_benchmark is None
            assert len(result.experiments) == 1
            assert not result.experiments[0].success
            repo.close()

    def test_all_incorrect_returns_no_best(self):
        """全候補が INCORRECT のとき best_params=None。"""
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo)
            spec = _rmsnorm_spec()

            with patch("forge.orchestrator.run_in_worker", return_value=_incorrect_worker_result()):
                result = orch.optimize(spec, budget=1, search=_tiny_grid())

            assert result.best_params is None
            assert len(result.experiments) == 1
            assert result.experiments[0].correct is False
            repo.close()

    def test_discord_notified_on_success(self):
        """ベスト候補が見つかったとき Discord 通知が呼ばれる。"""
        notifier = MagicMock()
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo, notifier=notifier)
            spec = _rmsnorm_spec()

            with patch("forge.orchestrator.run_in_worker", return_value=_ok_worker_result()):
                orch.optimize(spec, budget=1, search=_tiny_grid())

        notifier.send_optimization_complete.assert_called_once()
        call_kwargs = notifier.send_optimization_complete.call_args.kwargs
        assert call_kwargs["op_name"] == "rmsnorm"
        assert call_kwargs["num_candidates"] >= 1

    def test_discord_notified_on_all_fail(self):
        """全候補が失敗したとき Discord エラー通知が呼ばれる。"""
        notifier = MagicMock()
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo, notifier=notifier)
            spec = _rmsnorm_spec()

            with patch("forge.orchestrator.run_in_worker", return_value=_fail_worker_result()):
                orch.optimize(spec, budget=1, search=_tiny_grid())

        notifier.send_optimization_error.assert_called_once()
        notifier.send_optimization_complete.assert_not_called()
