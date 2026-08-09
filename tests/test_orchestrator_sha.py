"""Orchestrator.optimize_sha() の CPU モックテスト (GPU 不要)。

run_in_worker をモックして SHA ラウンドの絞り込み・warmup/repeat 変化・
キャッシュ HIT を検証する。
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
from forge.orchestrator import Orchestrator, SearchResult
from forge.runtime.worker import WorkerResult
from forge.search.candidate import CandidateGenerator
from forge.search.grid import GridSearch
from forge.search.params import SearchParams
from forge.search.space import SearchSpace

# ── helpers ───────────────────────────────────────────────────────────────────


def _spec() -> KernelSpec:
    return KernelSpec(
        op_type="rmsnorm",
        input_specs=(
            TensorSpec((4, 64), torch.float32, True),
            TensorSpec((64,), torch.float32, True),
        ),
        output_specs=(TensorSpec((4, 64), torch.float32, True),),
        constants={"eps": 1e-6},
        graph_hash="rmsnorm_sha_test",
        constraints=(),
    )


def _tiny_grid(n: int = 2) -> GridSearch:
    """block_sizes=[64] num_warps=[4] acc_dtypes=['fp32'] variants=['single_row'] → 1 candidate."""
    space = SearchSpace(
        block_sizes=[64],
        num_warps=[4],
        acc_dtypes=["fp32"],
        variants=["single_row"],
    )
    return GridSearch(space)


def _ok_result(median_us: float = 10.0) -> WorkerResult:
    bench = BenchmarkResult(median_us=median_us, p20_us=median_us * 0.9, p80_us=median_us * 1.1)
    baseline = BenchmarkResult(median_us=20.0, p20_us=18.0, p80_us=22.0)
    return WorkerResult(
        success=True,
        correct=True,
        candidate=bench,
        baseline=baseline,
        baseline_name="F.rms_norm",
    )


def _fail_result() -> WorkerResult:
    return WorkerResult(success=False, error="mock CUDA crash")


def _incorrect_result() -> WorkerResult:
    return WorkerResult(success=True, correct=False)


def _orch(tmp: str) -> Orchestrator:
    return Orchestrator(repo=KernelRepository(Path(tmp) / "cache.db"), warmup=25, repeat=200)


# ── tests ─────────────────────────────────────────────────────────────────────


class TestOptimizeSHA:
    def test_returns_search_result(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            orch = _orch(d)
            with patch("forge.orchestrator.run_in_worker", return_value=_ok_result()):
                result = orch.optimize_sha(_spec(), initial_budget=1, halving_rounds=1)
        assert isinstance(result, SearchResult)
        assert not result.cache_hit

    def test_cache_hit_returns_immediately(self) -> None:
        """キャッシュ HIT 時は run_in_worker を呼ばずに返す。"""
        with tempfile.TemporaryDirectory() as d:
            orch = _orch(d)
            with patch("forge.orchestrator.run_in_worker", return_value=_ok_result(5.0)) as mock_w:
                # warm the cache via optimize()
                orch.optimize(_spec(), budget=1, search=_tiny_grid())
                # second call should hit cache
                result = orch.optimize_sha(_spec(), initial_budget=1, halving_rounds=1)
            assert result.cache_hit
            assert result.experiments == []
            # run_in_worker called once for optimize(), zero for optimize_sha()
            assert mock_w.call_count == 1

    def test_best_params_set_when_candidate_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            orch = _orch(d)
            with patch("forge.orchestrator.run_in_worker", return_value=_ok_result(8.0)):
                result = orch.optimize_sha(_spec(), initial_budget=1, halving_rounds=1)
        assert result.best_params is not None
        assert result.best_benchmark is not None
        assert result.best_benchmark.median_us == pytest.approx(8.0)

    def test_best_params_none_when_all_fail(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            orch = _orch(d)
            with patch("forge.orchestrator.run_in_worker", return_value=_fail_result()):
                result = orch.optimize_sha(_spec(), initial_budget=1, halving_rounds=2)
        assert result.best_params is None
        assert result.best_benchmark is None

    def test_incorrect_candidate_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            orch = _orch(d)
            with patch("forge.orchestrator.run_in_worker", return_value=_incorrect_result()):
                result = orch.optimize_sha(_spec(), initial_budget=1, halving_rounds=1)
        assert result.best_params is None
        exp = result.experiments[0]
        assert exp.correct is False

    def test_warmup_repeat_scale_per_round(self) -> None:
        """各ラウンドで warmup/repeat が _SHA_ROUND_CONFIGS に沿って変化する。"""
        with tempfile.TemporaryDirectory() as d:
            orch = _orch(d)
            calls_warmup: list[int] = []
            calls_repeat: list[int] = []

            def capture(*args: object, **kwargs: object) -> WorkerResult:  # type: ignore[misc]
                calls_warmup.append(int(kwargs["warmup"]))  # type: ignore[arg-type]
                calls_repeat.append(int(kwargs["repeat"]))  # type: ignore[arg-type]
                return _ok_result(10.0)

            # 2 candidates → round 1 evaluates both, round 2 evaluates top 1
            space2 = SearchSpace(
                block_sizes=[64, 1024],
                num_warps=[4],
                acc_dtypes=["fp32"],
                variants=["single_row"],
            )
            with patch("forge.orchestrator.run_in_worker", side_effect=capture):
                orch.optimize_sha(
                    _spec(),
                    initial_budget=2,
                    halving_rounds=2,
                    search=GridSearch(space2),
                )

        # round 1: warmup=5, repeat=20 for each candidate
        assert 5 in calls_warmup
        assert 20 in calls_repeat
        # round 2: warmup=10, repeat=50
        assert 10 in calls_warmup
        assert 50 in calls_repeat

    def test_top_half_survives_halving(self) -> None:
        """round 1 で速い候補がラウンド 2 へ進む。"""
        with tempfile.TemporaryDirectory() as d:
            orch = _orch(d)
            results_iter = iter([
                _ok_result(5.0),   # fast — should survive
                _ok_result(50.0),  # slow — should be cut
                _ok_result(5.0),   # round 2 survivor re-evaluated
            ])
            space = SearchSpace(
                block_sizes=[64, 1024],
                num_warps=[4],
                acc_dtypes=["fp32"],
                variants=["single_row"],
            )
            with patch(
                "forge.orchestrator.run_in_worker", side_effect=lambda *a, **kw: next(results_iter)
            ):
                result = orch.optimize_sha(
                    _spec(),
                    initial_budget=2,
                    halving_rounds=2,
                    search=GridSearch(space),
                )
        # 2 exps in round 1 + 1 exp in round 2
        assert len(result.experiments) == 3
        # best is the fast one
        assert result.best_benchmark is not None
        assert result.best_benchmark.median_us == pytest.approx(5.0)

    def test_custom_search_is_used(self) -> None:
        """search= で渡した CandidateGenerator が使われる。"""
        with tempfile.TemporaryDirectory() as d:
            orch = _orch(d)

            mock_gen = MagicMock(spec=CandidateGenerator)
            mock_gen.generate.return_value = [
                SearchParams(
                    block_size=64,
                    num_warps=4,
                    num_stages=1,
                    acc_dtype="fp32",
                    variant="single_row",
                    rows_per_program=1,
                )
            ]
            with patch("forge.orchestrator.run_in_worker", return_value=_ok_result()):
                orch.optimize_sha(_spec(), initial_budget=1, halving_rounds=1, search=mock_gen)

        mock_gen.generate.assert_called_once()

    def test_baseline_benchmark_populated(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            orch = _orch(d)
            with patch("forge.orchestrator.run_in_worker", return_value=_ok_result()):
                result = orch.optimize_sha(_spec(), initial_budget=1, halving_rounds=1)
        assert result.baseline_benchmark is not None
        assert result.baseline_name is not None

    def test_no_candidates_returns_empty_result(self) -> None:
        """初期候補 0 件の場合でもクラッシュしない。"""
        with tempfile.TemporaryDirectory() as d:
            orch = _orch(d)
            mock_gen = MagicMock(spec=CandidateGenerator)
            mock_gen.generate.return_value = []
            with patch("forge.orchestrator.run_in_worker") as mock_w:
                result = orch.optimize_sha(
                    _spec(), initial_budget=0, halving_rounds=2, search=mock_gen
                )
            mock_w.assert_not_called()
        assert result.best_params is None
        assert result.experiments == []
