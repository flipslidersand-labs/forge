"""Orchestrator の GPU end-to-end テスト。CUDA 非対応環境ではスキップ。"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from forge.cache.repository import KernelRepository
from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.orchestrator import Orchestrator, SearchResult
from forge.search.grid import GridSearch
from forge.search.space import SearchSpace

pytestmark = pytest.mark.gpu
_SKIP = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")


def _spec() -> KernelSpec:
    return KernelSpec(
        op_type="rmsnorm",
        input_specs=(
            TensorSpec((512, 4096), torch.float16, True),
            TensorSpec((4096,), torch.float16, True),
        ),
        output_specs=(TensorSpec((512, 4096), torch.float16, True),),
        constants={"eps": 1e-6},
        graph_hash="rmsnorm_v1",
        constraints=(),
    )


def _orch(repo: KernelRepository) -> Orchestrator:
    return Orchestrator(repo=repo, warmup=5, repeat=30)


def _search(acc_dtypes: list[str] | None = None) -> GridSearch:
    # 計測軸を絞って探索を高速化
    space = SearchSpace(num_warps=[4, 8], acc_dtypes=acc_dtypes or ["fp32"])
    return GridSearch(space)


@_SKIP
def test_search_finds_correct_faster_kernel_and_caches() -> None:
    with tempfile.TemporaryDirectory() as d:
        repo = KernelRepository(Path(d) / "cache.db")
        orch = _orch(repo)
        spec = _spec()

        result = orch.optimize(spec, budget=10, search=_search())
        assert not result.cache_hit
        assert result.best_params is not None
        assert result.best_params.acc_dtype == "fp32"
        assert result.best_benchmark is not None
        # 全候補が正確性を通ること（fp32 のみなので）
        assert all(e.correct for e in result.experiments)
        # 融合カーネルは F.rms_norm baseline より速い
        assert result.speedup is not None and result.speedup > 1.0
        repo.close()


@_SKIP
def test_second_run_is_cache_hit() -> None:
    with tempfile.TemporaryDirectory() as d:
        repo = KernelRepository(Path(d) / "cache.db")
        orch = _orch(repo)
        spec = _spec()

        first = orch.optimize(spec, budget=6, search=_search())
        assert not first.cache_hit

        second = orch.optimize(spec, budget=6, search=_search())
        assert second.cache_hit
        assert second.experiments == []  # 探索していない
        assert second.best_params == first.best_params
        repo.close()


@_SKIP
def test_search_across_variants_all_correct() -> None:
    # single_row / multi_row / two_pass を横断し、全 fp32 候補が正確であること
    with tempfile.TemporaryDirectory() as d:
        repo = KernelRepository(Path(d) / "cache.db")
        space = SearchSpace(
            num_warps=[8],
            num_stages=[1],
            acc_dtypes=["fp32"],
            variants=["single_row", "multi_row", "two_pass"],
            rows_per_program=[2],
        )
        orch = Orchestrator(repo=repo, warmup=5, repeat=20)
        result = orch.optimize(_spec(), budget=20, search=GridSearch(space))

        tried_variants = {e.params.variant for e in result.experiments}
        assert tried_variants == {"single_row", "multi_row", "two_pass"}
        assert all(e.correct for e in result.experiments)
        assert result.best_params is not None
        repo.close()


def test_economic_objective_records_search_cost_s() -> None:
    """objective="economic" 時に search_cost_s = len(candidates) * per_candidate_s が記録される。GPU 不要。"""
    spec = _spec()

    # 候補生成だけ行い、GPU 実行は行わない mock
    mock_candidates = [MagicMock() for _ in range(5)]
    mock_search = MagicMock()
    mock_search.generate.return_value = mock_candidates

    messages: list[str] = []

    with tempfile.TemporaryDirectory() as d:
        repo = KernelRepository(Path(d) / "cache.db")
        orch = Orchestrator(repo=repo, progress=messages.append)

        with patch("forge.orchestrator.generate", side_effect=ValueError("skip")) as _mock_gen:
            result = orch.optimize(
                spec,
                budget=5,
                search=mock_search,
                use_cache=False,
                objective="economic",
                per_candidate_s=3.5,
            )

    assert result.search_cost_s == pytest.approx(5 * 3.5)
    assert any("economic" in m for m in messages)
    assert any("17.5" in m for m in messages)


@_SKIP
def test_economic_objective_search_cost_none_for_latency() -> None:
    """objective="latency"（デフォルト）では search_cost_s は None のまま。"""
    with tempfile.TemporaryDirectory() as d:
        repo = KernelRepository(Path(d) / "cache.db")
        orch = _orch(repo)
        result = orch.optimize(_spec(), budget=4, search=_search(), objective="latency")
        assert result.search_cost_s is None
        repo.close()


@_SKIP
def test_fp16_accumulator_rejected_as_incorrect() -> None:
    # fp16 accumulator は縮約精度不足で tolerance 超過 → 不採用になるはず
    with tempfile.TemporaryDirectory() as d:
        repo = KernelRepository(Path(d) / "cache.db")
        space = SearchSpace(num_warps=[8], acc_dtypes=["fp16"])
        orch = Orchestrator(repo=repo, warmup=5, repeat=20)
        result = orch.optimize(_spec(), budget=4, search=GridSearch(space))
        assert result.best_params is None  # fp16 acc は全滅
        assert all(not e.correct for e in result.experiments)
        repo.close()
