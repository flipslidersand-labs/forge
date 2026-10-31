"""PR トリガー GPU smoke test — rmsnorm / softmax を最小 budget で検証。

budget=3, warmup=5, repeat=20 の軽量設定で 10 分以内完了を目標にする。
GPU が利用不可の場合はスキップする（ローカル CPU 環境での誤実行防止）。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from forge.cache.repository import KernelRepository
from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.orchestrator import Orchestrator
from forge.search.grid import GridSearch
from forge.search.space import SearchSpace

pytestmark = [pytest.mark.gpu, pytest.mark.smoke]

_SKIP = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")

# smoke 専用の軽量 Orchestrator 設定
_SMOKE_WARMUP = 5
_SMOKE_REPEAT = 20
_SMOKE_BUDGET = 3

# 探索空間を絞って高速化
_SMOKE_GRID = GridSearch(
    SearchSpace(
        block_sizes=[128],
        num_warps=[4],
        acc_dtypes=["fp32"],
        variants=["single_row"],
    )
)


def _make_orch(repo: KernelRepository) -> Orchestrator:
    return Orchestrator(repo=repo, warmup=_SMOKE_WARMUP, repeat=_SMOKE_REPEAT)


@_SKIP
def test_smoke_rmsnorm() -> None:
    """rmsnorm の smoke: カーネルが生成・実行されて best_params が返ること。"""
    spec = KernelSpec(
        op_type="rmsnorm",
        input_specs=(
            TensorSpec((32, 512), torch.float16, True),
            TensorSpec((512,), torch.float16, True),
        ),
        output_specs=(TensorSpec((32, 512), torch.float16, True),),
        constants={"eps": 1e-6},
        graph_hash="rmsnorm_v1",
        constraints=(),
    )
    with tempfile.TemporaryDirectory() as d:
        with Orchestrator(
            repo=KernelRepository(Path(d) / "smoke.db"),
            warmup=_SMOKE_WARMUP,
            repeat=_SMOKE_REPEAT,
        ) as orch:
            result = orch.optimize(spec, budget=_SMOKE_BUDGET, search=_SMOKE_GRID)

    assert not result.cache_hit
    assert result.best_params is not None, "rmsnorm smoke: no valid kernel found"


@_SKIP
def test_smoke_softmax() -> None:
    """softmax の smoke: カーネルが生成・実行されて best_params が返ること。"""
    spec = KernelSpec(
        op_type="softmax",
        input_specs=(TensorSpec((64, 256), torch.float16, True),),
        output_specs=(TensorSpec((64, 256), torch.float16, True),),
        constants={},
        graph_hash="softmax_v1",
        constraints=(),
    )
    with tempfile.TemporaryDirectory() as d:
        with Orchestrator(
            repo=KernelRepository(Path(d) / "smoke.db"),
            warmup=_SMOKE_WARMUP,
            repeat=_SMOKE_REPEAT,
        ) as orch:
            result = orch.optimize(spec, budget=_SMOKE_BUDGET, search=_SMOKE_GRID)

    assert not result.cache_hit
    assert result.best_params is not None, "softmax smoke: no valid kernel found"
