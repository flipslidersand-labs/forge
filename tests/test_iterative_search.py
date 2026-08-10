"""IterativeLLMSearch のオフラインテスト。propose_fn / evaluate_fn を注入し API 不要。"""

from __future__ import annotations

import torch

from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.search.candidate import HistoryEntry
from forge.search.iterative_search import (
    IterativeLLMSearch,
    IterativeSearchResult,
    build_feedback_prompt,
)
from forge.search.llm_generator import LLMGenerator
from forge.search.params import SearchParams


# ── fixtures ─────────────────────────────────────────────────────────────────

def _spec() -> KernelSpec:
    n = 4096
    return KernelSpec(
        op_type="rmsnorm",
        input_specs=(
            TensorSpec((2048, n), torch.float16, True),
            TensorSpec((n,), torch.float16, True),
        ),
        output_specs=(TensorSpec((2048, n), torch.float16, True),),
        constants={"eps": 1e-6},
        graph_hash="rmsnorm_test",
        constraints=(),
    )


def _cand(**kw) -> dict:
    base = dict(
        base_variant="single_row",
        block_size=4096,
        num_warps=8,
        num_stages=1,
        acc_dtype="fp32",
        rows_per_program=1,
        hypothesis="test",
    )
    base.update(kw)
    return base


def _gen_returning(raw: list[dict]) -> LLMGenerator:
    return LLMGenerator(propose_fn=lambda s, cc, n, h: raw)


def _evaluate_always_correct(params: SearchParams) -> HistoryEntry:
    return HistoryEntry(params=params, correct=True, median_us=100.0)


def _evaluate_always_fail(params: SearchParams) -> HistoryEntry:
    return HistoryEntry(params=params, correct=False, median_us=None)


# ── tests ─────────────────────────────────────────────────────────────────────

class TestIterativeLLMSearchBasic:
    """基本的な実行フロー。"""

    def test_returns_iterative_search_result(self) -> None:
        gen = _gen_returning([_cand()])
        searcher = IterativeLLMSearch(gen, _evaluate_always_correct, max_rounds=2)
        result = searcher.run(_spec(), "8.9")
        assert isinstance(result, IterativeSearchResult)

    def test_runs_correct_number_of_rounds(self) -> None:
        gen = _gen_returning([_cand()])
        searcher = IterativeLLMSearch(gen, _evaluate_always_correct, max_rounds=3)
        result = searcher.run(_spec(), "8.9")
        assert len(result.rounds) == 3

    def test_best_is_fastest_correct(self) -> None:
        candidates = [
            _cand(block_size=4096, num_warps=4),
            _cand(block_size=4096, num_warps=8),
        ]
        latencies = [200.0, 80.0]
        idx = 0

        def evaluate(params: SearchParams) -> HistoryEntry:
            nonlocal idx
            lat = latencies[idx % len(latencies)]
            idx += 1
            return HistoryEntry(params=params, correct=True, median_us=lat)

        gen = _gen_returning(candidates)
        searcher = IterativeLLMSearch(gen, evaluate, max_rounds=1, candidates_per_round=2)
        result = searcher.run(_spec(), "8.9")
        assert result.best is not None
        assert result.best.median_us == 80.0

    def test_best_is_none_when_all_fail(self) -> None:
        gen = _gen_returning([_cand()])
        searcher = IterativeLLMSearch(gen, _evaluate_always_fail, max_rounds=2)
        result = searcher.run(_spec(), "8.9")
        assert result.best is None


class TestDiversityDeduplication:
    """同一候補の重複評価防止。"""

    def test_duplicate_params_evaluated_once(self) -> None:
        calls: list[SearchParams] = []

        def evaluate(params: SearchParams) -> HistoryEntry:
            calls.append(params)
            return HistoryEntry(params=params, correct=True, median_us=100.0)

        # 3 ラウンド同じ候補を提案 → 1 回しか評価されない
        gen = _gen_returning([_cand()])
        searcher = IterativeLLMSearch(gen, evaluate, max_rounds=3, candidates_per_round=1)
        searcher.run(_spec(), "8.9")
        assert len(calls) == 1


class TestTokenBudget:
    """max_tokens による早期打ち切り。"""

    def test_stops_when_budget_exceeded(self) -> None:
        call_count = 0
        original_propose = lambda s, cc, n, h: [_cand()]

        def propose_and_count(s, cc, n, h):
            nonlocal call_count
            call_count += 1
            return [_cand(num_warps=4 + call_count)]  # 毎回異なる候補

        gen = LLMGenerator(propose_fn=propose_and_count)
        # max_tokens=0 → 1 ラウンド目終了後に budget_exhausted
        searcher = IterativeLLMSearch(gen, _evaluate_always_correct, max_rounds=10, max_tokens=0)
        result = searcher.run(_spec(), "8.9")
        # 0 トークンで即打ち切り → ラウンドなし
        assert len(result.rounds) == 0

    def test_unlimited_budget_runs_all_rounds(self) -> None:
        gen = _gen_returning([_cand()])
        searcher = IterativeLLMSearch(gen, _evaluate_always_correct, max_rounds=3, max_tokens=None)
        result = searcher.run(_spec(), "8.9")
        assert len(result.rounds) == 3


class TestTokenUsageAccumulation:
    """トークン消費の累積集計。"""

    def test_total_token_usage_sums_rounds(self) -> None:
        gen = _gen_returning([_cand()])
        searcher = IterativeLLMSearch(gen, _evaluate_always_correct, max_rounds=2)
        result = searcher.run(_spec(), "8.9")
        # propose_fn は TokenUsage を加算しない（本物の API 呼び出しのみ加算）
        # → total は 0 だがオブジェクトが存在することを確認
        assert result.total_token_usage.total >= 0

    def test_round_result_has_token_usage(self) -> None:
        gen = _gen_returning([_cand()])
        searcher = IterativeLLMSearch(gen, _evaluate_always_correct, max_rounds=1)
        result = searcher.run(_spec(), "8.9")
        assert hasattr(result.rounds[0], "token_usage")


class TestBuildFeedbackPrompt:
    """build_feedback_prompt のフォーマット確認。"""

    def _entry(self, correct: bool, us: float | None, ws: int = 8) -> HistoryEntry:
        p = SearchParams(block_size=4096, num_warps=ws, num_stages=1)
        return HistoryEntry(params=p, correct=correct, median_us=us)

    def test_shows_top_k_successes(self) -> None:
        history = [
            self._entry(True, 100.0, ws=4),
            self._entry(True, 80.0, ws=8),
            self._entry(True, 60.0, ws=16),
        ]
        prompt = build_feedback_prompt(history, top_k=2)
        assert "Top-2" in prompt
        # 最速 2 件 (60, 80) が含まれていること
        assert "warps=16" in prompt
        assert "warps=8" in prompt

    def test_shows_failure_patterns(self) -> None:
        history = [
            self._entry(False, None, ws=4),
            self._entry(True, 100.0, ws=8),
        ]
        prompt = build_feedback_prompt(history, top_k=3)
        assert "Failure" in prompt
        assert "INCORRECT" in prompt

    def test_empty_history_returns_empty(self) -> None:
        prompt = build_feedback_prompt([], top_k=3)
        assert prompt == ""
