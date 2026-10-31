"""adoption モジュールのオフラインテスト。GPU 不要。"""

import inspect

from forge.decorator import optimize
from forge.search.adoption import AdoptionDecision, should_run_search


class TestShouldRunSearch:
    def test_high_invocation_count_should_search(self) -> None:
        # 100s 探索コスト、50µs 基準、1.3x スピードアップ
        # saved_per_call = 50 * (1 - 1/1.3) ≈ 11.5µs
        # break_even = 100e6 / 11.5 ≈ 8.7M 回
        # 10M > 8.7M → should_search=True
        decision = should_run_search(
            expected_invocations=10_000_000,
            search_cost_s=100.0,
            baseline_us=50.0,
            expected_speedup=1.3,
        )
        assert decision.should_search is True
        assert decision.break_even_invocations < 10_000_000

    def test_low_invocation_count_skip(self) -> None:
        # 1000s 探索コスト、50µs 基準、1.3x スピードアップ
        # break_even = 1000e6 / 11.5 ≈ 87M 回
        # 100 < 87M → should_search=False
        decision = should_run_search(
            expected_invocations=100,
            search_cost_s=1000.0,
            baseline_us=50.0,
            expected_speedup=1.3,
        )
        assert decision.should_search is False
        assert decision.break_even_invocations > 100

    def test_speedup_le_1_returns_false(self) -> None:
        decision = should_run_search(
            expected_invocations=10_000_000,
            search_cost_s=10.0,
            baseline_us=50.0,
            expected_speedup=0.9,
        )
        assert decision.should_search is False
        assert decision.break_even_invocations == float("inf")

    def test_speedup_exactly_1_returns_false(self) -> None:
        decision = should_run_search(
            expected_invocations=10_000_000,
            search_cost_s=10.0,
            baseline_us=50.0,
            expected_speedup=1.0,
        )
        assert decision.should_search is False

    def test_break_even_formula(self) -> None:
        # 確定値でブレークイーブンを確認
        # baseline=100µs, speedup=2.0 → saved=50µs/call
        # search_cost=1s → 1e6µs
        # break_even = 1e6 / 50 = 20000
        decision = should_run_search(
            expected_invocations=20000,
            search_cost_s=1.0,
            baseline_us=100.0,
            expected_speedup=2.0,
        )
        assert abs(decision.break_even_invocations - 20000) < 1
        assert decision.should_search  # ちょうど break-even

    def test_expected_total_gain_positive_when_should_search(self) -> None:
        decision = should_run_search(
            expected_invocations=1_000_000,
            search_cost_s=1.0,
            baseline_us=100.0,
            expected_speedup=2.0,
        )
        assert decision.should_search
        assert decision.expected_total_gain_us > 0

    def test_expected_total_gain_negative_when_not_worth_it(self) -> None:
        decision = should_run_search(
            expected_invocations=1,
            search_cost_s=100.0,
            baseline_us=100.0,
            expected_speedup=2.0,
        )
        assert not decision.should_search
        assert decision.expected_total_gain_us < 0

    def test_returns_adoption_decision_type(self) -> None:
        decision = should_run_search(100, 10.0, 50.0)
        assert isinstance(decision, AdoptionDecision)
        assert isinstance(decision.reason, str)
        assert len(decision.reason) > 0

    def test_reason_mentions_break_even(self) -> None:
        decision = should_run_search(100, 10.0, 50.0)
        assert "break" in decision.reason.lower() or "skip" in decision.reason.lower()


class TestPerCandidateS:
    """optimize() デコレータの per_candidate_s パラメータのオフラインテスト。"""

    def test_optimize_accepts_per_candidate_s(self) -> None:
        sig = inspect.signature(optimize)
        assert "per_candidate_s" in sig.parameters

    def test_per_candidate_s_default_is_2(self) -> None:
        sig = inspect.signature(optimize)
        assert sig.parameters["per_candidate_s"].default == 2.0

    def test_search_cost_scales_with_per_candidate_s(self) -> None:
        # budget=10, baseline=100µs, speedup=2.0 → saved=50µs/call
        # per_candidate_s=0.1 → search_cost_s=1.0 → break_even=1e6/50=20000
        # per_candidate_s=50.0 → search_cost_s=500 → break_even=500e6/50=10M
        budget = 10
        baseline_us = 100.0
        invocations = 50_000  # 20000 < 50000 < 10M なので分岐が明確

        d_cheap = should_run_search(invocations, budget * 0.1, baseline_us, expected_speedup=2.0)
        d_expensive = should_run_search(
            invocations, budget * 50.0, baseline_us, expected_speedup=2.0
        )

        assert d_cheap.should_search is True
        assert d_expensive.should_search is False
