"""Orchestrator の GPU end-to-end テスト。CUDA 非対応環境ではスキップ。"""

import tempfile
from pathlib import Path

import pytest
import torch

from forge.cache.repository import KernelRepository
from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.orchestrator import MultiRoundResult, Orchestrator, RoundResult
from forge.search.candidate import HistoryEntry
from forge.search.grid import GridSearch
from forge.search.llm_generator import LLMGenerator
from forge.search.params import SearchParams
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


# --- MultiRoundResult / RoundResult 構造テスト（GPU 不要）---


def _make_params(**kw) -> SearchParams:
    base = dict(block_size=4096, num_warps=8, num_stages=1, acc_dtype="fp32")
    base.update(kw)
    return SearchParams(**base)


class TestMultiRoundResult:
    def test_speedup_none_without_benchmarks(self) -> None:
        from forge.search.llm_generator import TokenUsage

        r = MultiRoundResult(
            spec=_spec(),
            rounds=[],
            best_params=None,
            best_benchmark=None,
            baseline_benchmark=None,
            baseline_name=None,
            token_usage=TokenUsage(),
        )
        assert r.speedup is None

    def test_all_experiments_flattens_rounds(self) -> None:
        from forge.orchestrator import ExperimentResult
        from forge.search.llm_generator import TokenUsage

        p1 = _make_params(num_warps=4)
        p2 = _make_params(num_warps=8)
        exp1 = ExperimentResult(p1, True, True, 50.0, None)
        exp2 = ExperimentResult(p2, True, True, 45.0, None)
        r = MultiRoundResult(
            spec=_spec(),
            rounds=[
                RoundResult(1, [exp1], p1, 50.0),
                RoundResult(2, [exp2], p2, 45.0),
            ],
            best_params=p2,
            best_benchmark=None,
            baseline_benchmark=None,
            baseline_name=None,
            token_usage=TokenUsage(),
            total_candidates_evaluated=2,
        )
        assert len(r.all_experiments) == 2

    def test_best_round_returns_correct_round(self) -> None:
        from forge.orchestrator import ExperimentResult
        from forge.search.llm_generator import TokenUsage

        p1 = _make_params(num_warps=4)
        p2 = _make_params(num_warps=8)
        r = MultiRoundResult(
            spec=_spec(),
            rounds=[
                RoundResult(1, [], p1, 50.0),
                RoundResult(2, [], p2, 45.0),
            ],
            best_params=p2,
            best_benchmark=None,
            baseline_benchmark=None,
            baseline_name=None,
            token_usage=TokenUsage(),
        )
        assert r.best_round == 2


# --- optimize_rounds GPU テスト ---


@_SKIP
def test_optimize_rounds_finds_best_and_accumulates_history() -> None:
    """3 ラウンドで LLM (fake) が history を受け取りながら探索し、有効な結果を返す。"""
    history_per_round: list[int] = []

    def _propose(spec, cc, n, history):
        history_per_round.append(len(history))
        # 単一の valid な候補を毎回返す
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

        result = orch.optimize_rounds(_spec(), llm=llm, n_rounds=3, candidates_per_round=1)

        assert isinstance(result, MultiRoundResult)
        assert len(result.rounds) == 3
        # history は前ラウンドの成功結果が積み上がる
        assert history_per_round[0] == 0   # round1: 空
        assert history_per_round[1] >= 0   # round2: round1 の結果
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
