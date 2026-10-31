"""Orchestrator の GPU end-to-end テスト。CUDA 非対応環境ではスキップ。"""

import tempfile
from pathlib import Path

import pytest
import torch

from forge.cache.repository import KernelRepository
from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.orchestrator import MultiRoundResult, Orchestrator
from forge.search.candidate import HistoryEntry
from forge.search.grid import GridSearch
from forge.search.llm_generator import LLMGenerator
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


# --- optimize_rounds GPU テスト ---


@_SKIP
def test_optimize_rounds_finds_best_and_accumulates_history() -> None:
    """3 ラウンドで LLM (fake) が history を受け取りながら探索し、有効な結果を返す。"""
    history_per_round: list[int] = []
    _call = [0]

    def _propose(spec, cc, n, history):
        history_per_round.append(len(history))
        # ラウンドごとに num_warps を変えて重複スキップを防ぐ
        # （single_row は block_size >= hidden_size が必須なので block_size は 4096 固定）
        num_warps_seq = [4, 8, 16]
        nw = num_warps_seq[_call[0] % len(num_warps_seq)]
        _call[0] += 1
        return [
            dict(
                base_variant="single_row",
                block_size=4096,
                num_warps=nw,
                num_stages=1,
                acc_dtype="fp32",
                rows_per_program=1,
                hypothesis="test",
            )
        ]

    with tempfile.TemporaryDirectory() as d:
        repo = KernelRepository(Path(d) / "cache.db")
        orch = Orchestrator(repo=repo, warmup=3, repeat=20)
        llm = LLMGenerator(propose_fn=_propose)

        result = orch.optimize_rounds(_spec(), llm=llm, n_rounds=3, candidates_per_round=1)

        assert isinstance(result, MultiRoundResult)
        assert len(result.rounds) == 3
        # history は前ラウンドの成功結果が積み上がる
        assert history_per_round[0] == 0  # round1: 空
        assert history_per_round[1] >= 1  # round1 の成功 1 件が history に入っているはず
        assert history_per_round[2] >= history_per_round[1]
        # 少なくとも 1 つの有効な結果があるはず
        assert result.best_params is not None
        assert result.total_candidates_evaluated == 3
        repo.close()


@_SKIP
def test_optimize_rounds_history_grows_with_successful_evals() -> None:
    """成功した eval が次ラウンドの history に渡されることを確認。"""
    captured = {}

    def _propose(spec, cc, n, history):
        if history:
            captured["saw_history"] = history
        return [
            dict(
                base_variant="single_row",
                block_size=4096,
                num_warps=8,
                num_stages=1,
                acc_dtype="fp32",
                rows_per_program=1,
                hypothesis="test",
            )
        ]

    with tempfile.TemporaryDirectory() as d:
        repo = KernelRepository(Path(d) / "cache.db")
        orch = Orchestrator(repo=repo, warmup=3, repeat=20)
        llm = LLMGenerator(propose_fn=_propose)
        orch.optimize_rounds(_spec(), llm=llm, n_rounds=2, candidates_per_round=1)

        assert "saw_history" in captured
        assert isinstance(captured["saw_history"][0], HistoryEntry)
        repo.close()


# --- ExtendedBaselineResult GPU テスト ---


@_SKIP
def test_optimize_with_extended_baselines() -> None:
    """measure_extended=True のとき extended_baselines に torch.compile 結果が入る。"""
    with tempfile.TemporaryDirectory() as d:
        repo = KernelRepository(Path(d) / "cache.db")
        space = SearchSpace(num_warps=[8], num_stages=[1], acc_dtypes=["fp32"])
        orch = Orchestrator(repo=repo, warmup=3, repeat=20, measure_extended=True)
        result = orch.optimize(_spec(), budget=2, search=GridSearch(space))

        assert isinstance(result.extended_baselines, list)
        assert len(result.extended_baselines) >= 1
        eb = result.extended_baselines[0]
        assert "torch.compile" in eb.name
        if eb.error:
            pytest.skip(f"torch.compile unsupported on this GPU: {eb.error[:80]}")
        assert eb.benchmark.median_us > 0
        assert eb.benchmark.p95_us > 0
        assert eb.compile_time_s >= 0
        repo.close()
