"""コスト考慮判定のテスト。"""

from __future__ import annotations

import pytest

from forge.benchmark.pareto import (
    CandidateWithCost,
    ParetoFrontier,
    tokens_to_cost,
    time_to_cost_s,
)
from forge.search.params import SearchParams


class TestTokensAndTimeCost:
    """token/time コスト換算。"""

    def test_tokens_to_cost_positive(self) -> None:
        cost = tokens_to_cost(1_000_000)
        assert cost > 0
        assert 2.0 < cost < 4.0  # 1M tokens で ~$3

    def test_time_to_cost_s_positive(self) -> None:
        cost = time_to_cost_s(100.0)  # 100 秒
        assert cost > 0
        assert cost < 0.01  # 100s で 1 セント未満


class TestCandidateWithCost:
    """候補＋コスト表現。"""

    def test_candidate_with_cost_correct(self) -> None:
        params = SearchParams(
            block_size=256,
            num_warps=8,
            num_stages=2,
            acc_dtype="fp32",
            variant="single_row",
            rows_per_program=1,
        )
        cand = CandidateWithCost(
            params=params,
            median_us=100.0,
            tokens_for_proposal=5000,
            benchmark_time_ms=1000.0,
            correct=True,
        )
        assert cand.correct is True
        assert cand.median_us == 100.0

    def test_cost_per_1x_speedup_positive(self) -> None:
        params = SearchParams(
            block_size=256,
            num_warps=8,
            num_stages=2,
            acc_dtype="fp32",
            variant="single_row",
            rows_per_program=1,
        )
        cand = CandidateWithCost(
            params=params,
            median_us=50.0,  # baseline 100.0 から 2x speedup
            tokens_for_proposal=5000,
            benchmark_time_ms=1000.0,
            correct=True,
        )
        # speedup = 2.0 なので cost_per_1x_speedup は正の値
        cost = cand.cost_per_1x_speedup(baseline_us=100.0)
        assert cost is not None
        assert cost > 0

    def test_cost_per_1x_speedup_no_improvement(self) -> None:
        params = SearchParams(
            block_size=256,
            num_warps=8,
            num_stages=2,
            acc_dtype="fp32",
            variant="single_row",
            rows_per_program=1,
        )
        cand = CandidateWithCost(
            params=params,
            median_us=150.0,  # baseline 100.0 より遅い
            tokens_for_proposal=5000,
            benchmark_time_ms=1000.0,
            correct=True,
        )
        cost = cand.cost_per_1x_speedup(baseline_us=100.0)
        assert cost is None


class TestParetoOptimal:
    """パレート最適性判定。"""

    def test_pareto_optimal_simple(self) -> None:
        params1 = SearchParams(
            block_size=256,
            num_warps=8,
            num_stages=2,
            acc_dtype="fp32",
            variant="single_row",
            rows_per_program=1,
        )
        params2 = SearchParams(
            block_size=512,
            num_warps=16,
            num_stages=3,
            acc_dtype="fp32",
            variant="single_row",
            rows_per_program=1,
        )
        # cand1: 高速だが高コスト
        cand1 = CandidateWithCost(
            params=params1,
            median_us=50.0,
            tokens_for_proposal=10000,
            benchmark_time_ms=2000.0,
            correct=True,
        )
        # cand2: 低速だが低コスト
        cand2 = CandidateWithCost(
            params=params2,
            median_us=80.0,
            tokens_for_proposal=5000,
            benchmark_time_ms=500.0,
            correct=True,
        )
        candidates = [cand1, cand2]
        # どちらもパレート最適（cand1 は速度、cand2 はコスト）
        assert cand1.is_pareto_optimal(candidates)
        assert cand2.is_pareto_optimal(candidates)

    def test_pareto_dominated(self) -> None:
        params1 = SearchParams(
            block_size=256,
            num_warps=8,
            num_stages=2,
            acc_dtype="fp32",
            variant="single_row",
            rows_per_program=1,
        )
        params2 = SearchParams(
            block_size=512,
            num_warps=16,
            num_stages=3,
            acc_dtype="fp32",
            variant="single_row",
            rows_per_program=1,
        )
        # cand1: 高速 + 低コスト（完全に優れている）
        cand1 = CandidateWithCost(
            params=params1,
            median_us=50.0,
            tokens_for_proposal=5000,
            benchmark_time_ms=500.0,
            correct=True,
        )
        # cand2: 低速 + 高コスト（完全に劣っている）
        cand2 = CandidateWithCost(
            params=params2,
            median_us=100.0,
            tokens_for_proposal=10000,
            benchmark_time_ms=2000.0,
            correct=True,
        )
        candidates = [cand1, cand2]
        # cand1 は pareto 最適、cand2 は dominated
        assert cand1.is_pareto_optimal(candidates)
        assert not cand2.is_pareto_optimal(candidates)


class TestParetoFrontier:
    """パレート前線の管理。"""

    def test_pareto_frontier_construction(self) -> None:
        params1 = SearchParams(
            block_size=256,
            num_warps=8,
            num_stages=2,
            acc_dtype="fp32",
            variant="single_row",
            rows_per_program=1,
        )
        params2 = SearchParams(
            block_size=512,
            num_warps=16,
            num_stages=3,
            acc_dtype="fp32",
            variant="single_row",
            rows_per_program=1,
        )
        cand1 = CandidateWithCost(
            params=params1,
            median_us=50.0,
            tokens_for_proposal=10000,
            benchmark_time_ms=2000.0,
            correct=True,
        )
        cand2 = CandidateWithCost(
            params=params2,
            median_us=80.0,
            tokens_for_proposal=5000,
            benchmark_time_ms=500.0,
            correct=True,
        )
        frontier = ParetoFrontier([cand1, cand2])
        assert len(frontier.frontier) == 2

    def test_pareto_frontier_recommend(self) -> None:
        params1 = SearchParams(
            block_size=256,
            num_warps=8,
            num_stages=2,
            acc_dtype="fp32",
            variant="single_row",
            rows_per_program=1,
        )
        params2 = SearchParams(
            block_size=512,
            num_warps=16,
            num_stages=3,
            acc_dtype="fp32",
            variant="single_row",
            rows_per_program=1,
        )
        cand1 = CandidateWithCost(
            params=params1,
            median_us=50.0,
            tokens_for_proposal=10000,
            benchmark_time_ms=2000.0,
            correct=True,
        )
        cand2 = CandidateWithCost(
            params=params2,
            median_us=80.0,
            tokens_for_proposal=5000,
            benchmark_time_ms=500.0,
            correct=True,
        )
        frontier = ParetoFrontier([cand1, cand2])
        rec = frontier.recommend()
        assert rec is not None
