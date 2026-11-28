"""最良候補が eager に min_speedup 倍で勝たないときは採用しない (#475)。GPU 不要。

run_in_worker をモックして、Orchestrator がキャッシュに書かないこと、decorator の _build が
eager(None) に倒れること、cache hit は判定し直さないことを確かめる。
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from forge import decorator as dec
from forge.benchmark.statistics import BenchmarkResult
from forge.cache.repository import KernelRepository
from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.orchestrator import Orchestrator
from forge.runtime.worker import WorkerResult
from forge.search.grid import GridSearch
from forge.search.space import SearchSpace


def _spec() -> KernelSpec:
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


def _grid() -> GridSearch:
    return GridSearch(
        SearchSpace(block_sizes=[64], num_warps=[4], acc_dtypes=["fp32"], variants=["single_row"])
    )


def _worker(candidate_us: float, baseline_us: float) -> WorkerResult:
    return WorkerResult(
        success=True,
        correct=True,
        candidate=BenchmarkResult(
            median_us=candidate_us, p20_us=candidate_us - 0.1, p80_us=candidate_us + 0.1
        ),
        baseline=BenchmarkResult(
            median_us=baseline_us, p20_us=baseline_us - 0.1, p80_us=baseline_us + 0.1
        ),
        baseline_name="F.rms_norm",
    )


class _OneShotLLM:
    """optimize_rounds 用の最小の候補生成器（常にグリッドの 1 候補を返す）。"""

    def generate(self, spec: Any, cc: str, budget: int | None = None, history: Any = None) -> Any:
        return _grid().generate(spec, cc, budget=1)

    def reset_usage(self) -> None:
        return None


def _run(method: str, orch: Orchestrator, spec: KernelSpec) -> Any:
    if method == "optimize":
        return orch.optimize(spec, budget=1, search=_grid())
    if method == "optimize_sha":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return orch.optimize_sha(spec, initial_budget=4, halving_rounds=1, search=_grid())
    return orch.optimize_rounds(spec, _OneShotLLM(), n_rounds=1, candidates_per_round=1)


METHODS = ["optimize", "optimize_sha", "optimize_rounds"]


@pytest.mark.parametrize("method", METHODS)
class TestOrchestratorRejectsSlower:
    def test_slower_candidate_is_not_cached_or_adopted(self, method: str) -> None:
        """candidate が eager と同等(1.00x)なら採用せず、キャッシュにも書かない。"""
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo)
            spec = _spec()
            with patch("forge.orchestrator.run_in_worker", return_value=_worker(100.0, 100.0)):
                first = _run(method, orch, spec)
                second = _run(method, orch, spec)
            # 報告用に最良候補は残るが、採用されない
            assert first.best_params is not None
            assert first.adopted is False
            # キャッシュに書かれていないので 2 回目も cache hit にならず再探索する
            assert getattr(second, "cache_hit", False) is False
            assert second.adopted is False
            repo.close()

    def test_faster_candidate_is_adopted_and_cached(self, method: str) -> None:
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo)
            spec = _spec()
            with patch("forge.orchestrator.run_in_worker", return_value=_worker(50.0, 100.0)):
                first = _run(method, orch, spec)
            assert first.adopted is True
            with patch("forge.orchestrator.run_in_worker") as mock_worker:
                second = _run(method, orch, spec)
                mock_worker.assert_not_called()
            assert second.adopted is True
            repo.close()

    def test_boundary_exactly_min_speedup_is_adopted(self, method: str) -> None:
        """速度比がちょうど min_speedup(1.03) なら採用、わずかに下回れば不採用。"""
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo, min_speedup=1.03)
            spec = _spec()
            with patch("forge.orchestrator.run_in_worker", return_value=_worker(100.0, 102.9)):
                below = _run(method, orch, spec)
            assert below.adopted is False
            with patch("forge.orchestrator.run_in_worker", return_value=_worker(100.0, 103.0)):
                at = _run(method, orch, spec)
            assert at.adopted is True
            repo.close()

    def test_lower_min_speedup_adopts_slower_candidate(self, method: str) -> None:
        """戻し方: min_speedup を下げれば従来に近づく（0.9 なら 0.95x でも採用）。"""
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo, min_speedup=0.9)
            with patch("forge.orchestrator.run_in_worker", return_value=_worker(100.0, 95.0)):
                result = _run(method, orch, _spec())
            assert result.adopted is True
            repo.close()


def test_finalize_without_speedup_does_not_cache() -> None:
    """baseline と比べられない(speedup=None)ときは採用せず、キャッシュにも書かない。"""
    with tempfile.TemporaryDirectory() as d:
        repo = KernelRepository(Path(d) / "c.db")
        orch = Orchestrator(repo=repo)
        spec = _spec()
        ctx, _ = orch._prepare(spec, True)
        params = _grid().generate(spec, ctx.key.compute_capability, budget=1)[0]
        bench = BenchmarkResult(median_us=10.0, p20_us=9.0, p80_us=11.0)
        adopted = orch._finalize(ctx, params, bench, num_candidates=1, speedup=None)
        assert adopted is False
        assert repo.get(ctx.key) is None
        repo.close()


# --- decorator._build ---


class _FakeOrch:
    result: Any = None

    def __init__(self, **_: Any) -> None:
        pass

    def __enter__(self) -> _FakeOrch:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def optimize(self, spec: Any, budget: int, search: Any) -> Any:
        return self.result


def _build_with(result: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    _FakeOrch.result = result
    monkeypatch.setattr(dec, "Orchestrator", _FakeOrch)
    monkeypatch.setattr(dec, "generate", lambda spec, params: "code")
    sentinel = object()
    monkeypatch.setattr(dec, "load_kernel_fn", lambda code: sentinel)
    out = dec._build(
        lambda x: x,
        "rmsnorm",
        [torch.zeros(2)],
        {},
        dec._BuildConfig(
            budget=1,
            repo=None,
            search=None,
            min_speedup=1.03,
            python_executable=None,
            progress=lambda _e: None,
        ),
    )
    return out, sentinel


def test_build_returns_none_when_not_adopted(monkeypatch: pytest.MonkeyPatch) -> None:
    result = SimpleNamespace(best_params="p", adopted=False, speedup=0.95, cache_hit=False)
    out, _ = _build_with(result, monkeypatch)
    assert out is None


def test_build_uses_kernel_when_adopted(monkeypatch: pytest.MonkeyPatch) -> None:
    result = SimpleNamespace(best_params="p", adopted=True, speedup=2.0, cache_hit=False)
    out, sentinel = _build_with(result, monkeypatch)
    assert out is sentinel


def test_build_uses_cache_hit_even_though_speedup_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cache hit は baseline を持たず speedup=None だが、採用済みなので eager に倒さない。"""
    from forge.orchestrator import SearchResult

    result = SearchResult(
        spec=_spec(),
        cache_hit=True,
        best_params=SimpleNamespace(),  # type: ignore[arg-type]
        best_benchmark=None,
        baseline_benchmark=None,
        baseline_name=None,
        experiments=[],
    )
    assert result.speedup is None
    out, sentinel = _build_with(result, monkeypatch)
    assert out is sentinel
