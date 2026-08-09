"""Orchestrator.optimize() の CPU モックテスト (GPU 不要)。

run_in_worker をモックして、キャッシュ HIT/MISS・全失敗・全不正確・Discord 通知を検証する。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
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
