from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdoptionDecision:
    """探索を実行すべきか否かの判断結果。"""

    should_search: bool
    break_even_invocations: float
    expected_total_gain_us: float
    search_cost_us: float
    saved_per_call_us: float
    reason: str


def should_run_search(
    expected_invocations: int,
    search_cost_s: float,
    baseline_us: float,
    expected_speedup: float = 1.3,
) -> AdoptionDecision:
    """探索コストを回収できるか判断する。

    Parameters
    ----------
    expected_invocations:
        このカーネルが今後呼ばれる見込み回数。
    search_cost_s:
        探索にかかる推定時間（秒）。``budget * per_candidate_s`` の見積もりを渡す。
    baseline_us:
        PyTorch eager 実装の実測レイテンシ（マイクロ秒）。
    expected_speedup:
        探索で期待できるスピードアップ倍率（デフォルト 1.3 = 30% 高速化）。

    Returns
    -------
    AdoptionDecision
        ``should_search=True`` なら探索を実行する価値がある。
    """
    search_cost_us = search_cost_s * 1e6
    saved_per_call_us = baseline_us * (1.0 - 1.0 / max(expected_speedup, 1e-9))

    if saved_per_call_us <= 0:
        return AdoptionDecision(
            should_search=False,
            break_even_invocations=float("inf"),
            expected_total_gain_us=-search_cost_us,
            search_cost_us=search_cost_us,
            saved_per_call_us=saved_per_call_us,
            reason="expected_speedup <= 1.0: no gain possible",
        )

    break_even = search_cost_us / saved_per_call_us
    expected_total_gain = expected_invocations * saved_per_call_us - search_cost_us
    should = float(expected_invocations) >= break_even

    if should:
        reason = (
            f"break-even at {break_even:.0f} calls; "
            f"expected {expected_invocations} → net gain {expected_total_gain / 1e6:.3f}s"
        )
    else:
        reason = (
            f"need {break_even:.0f} calls to break even, "
            f"only expect {expected_invocations} → skip search"
        )

    return AdoptionDecision(
        should_search=should,
        break_even_invocations=break_even,
        expected_total_gain_us=expected_total_gain,
        search_cost_us=search_cost_us,
        saved_per_call_us=saved_per_call_us,
        reason=reason,
    )
