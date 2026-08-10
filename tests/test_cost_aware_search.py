"""コスト考慮判定のテスト。"""

from __future__ import annotations

from forge.benchmark.pareto import (
    CandidateWithCost,
    ParetoFrontier,
    time_to_cost_s,
    tokens_to_cost,
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


# ── CostModel / BudgetTracker / scalarize ────────────────────────────────────

import tempfile
from pathlib import Path

from forge.search.cost_model import BudgetTracker, CostModel, scalarize


class TestCostModel:
    """CostModel の predict / record。"""

    def _make_params(self, block_size: int = 256) -> SearchParams:
        return SearchParams(block_size=block_size, num_warps=4, num_stages=2)

    def test_predict_returns_default_on_cache_miss(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with CostModel(db_path=Path(d) / "cost.db", default_ms=50.0) as model:
                p = self._make_params()
                est = model.predict_cost(p)
                assert not est.is_cached
                assert est.estimated_bench_ms > 0

    def test_record_then_predict_returns_cached(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with CostModel(db_path=Path(d) / "cost.db") as model:
                p = self._make_params()
                model.record(p, bench_time_ms=123.0)
                est = model.predict_cost(p)
                assert est.is_cached
                assert est.estimated_bench_ms == 123.0

    def test_upsert_overwrites_old_value(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with CostModel(db_path=Path(d) / "cost.db") as model:
                p = self._make_params()
                model.record(p, bench_time_ms=100.0)
                model.record(p, bench_time_ms=42.0)
                assert model.predict_cost(p).estimated_bench_ms == 42.0


class TestBudgetTracker:
    """BudgetTracker の早期打ち切り判定。"""

    def test_no_budget_never_exhausted(self) -> None:
        tracker = BudgetTracker(max_total_s=None)
        assert not tracker.budget_exhausted
        assert not tracker.should_skip(999_999.0)

    def test_should_skip_when_estimate_exceeds_remaining(self) -> None:
        tracker = BudgetTracker(max_total_s=1.0)
        # 即座に予算を使い切る想定: 残り ~1s、推定 2000ms
        assert tracker.should_skip(2000.0)

    def test_record_tracks_elapsed(self) -> None:
        tracker = BudgetTracker(max_total_s=60.0)
        tracker.record(100.0)
        tracker.record(200.0)
        assert not tracker.budget_exhausted


class TestScalarize:
    """scalarize のスコア計算。"""

    def test_high_speedup_scores_higher(self) -> None:
        s1 = scalarize(speedup=2.0, cost=0.1, lam=0.1)
        s2 = scalarize(speedup=1.5, cost=0.1, lam=0.1)
        assert s1 > s2

    def test_high_cost_penalizes_score(self) -> None:
        s_low = scalarize(speedup=2.0, cost=0.1, lam=1.0)
        s_high = scalarize(speedup=2.0, cost=10.0, lam=1.0)
        assert s_low > s_high

    def test_zero_lambda_ignores_cost(self) -> None:
        s1 = scalarize(speedup=2.0, cost=1000.0, lam=0.0)
        s2 = scalarize(speedup=2.0, cost=0.0, lam=0.0)
        assert s1 == s2
