"""@forge.optimize の budget/min_speedup/per_candidate_s バリデーション (#320)。GPU 不要。"""

from __future__ import annotations

import math

import pytest

import forge


def test_budget_zero_raises_value_error() -> None:
    with pytest.raises(ValueError, match="budget"):
        forge.optimize(budget=0)


def test_budget_negative_raises_value_error() -> None:
    with pytest.raises(ValueError, match="budget"):
        forge.optimize(budget=-1)


def test_min_speedup_zero_raises_value_error() -> None:
    with pytest.raises(ValueError, match="min_speedup"):
        forge.optimize(min_speedup=0)


def test_min_speedup_negative_raises_value_error() -> None:
    with pytest.raises(ValueError, match="min_speedup"):
        forge.optimize(min_speedup=-0.5)


def test_per_candidate_s_zero_raises_value_error() -> None:
    with pytest.raises(ValueError, match="per_candidate_s"):
        forge.optimize(per_candidate_s=0)


def test_per_candidate_s_negative_raises_value_error() -> None:
    with pytest.raises(ValueError, match="per_candidate_s"):
        forge.optimize(per_candidate_s=-2.0)


def test_valid_values_do_not_raise() -> None:
    forge.optimize(budget=1, min_speedup=1.03, per_candidate_s=2.0)


def test_budget_nan_raises_value_error() -> None:
    """`nan < 1` は False のため `<` 比較では素通りしてしまう罠の回帰テスト。"""
    with pytest.raises(ValueError, match="budget"):
        forge.optimize(budget=math.nan)


def test_min_speedup_nan_raises_value_error() -> None:
    with pytest.raises(ValueError, match="min_speedup"):
        forge.optimize(min_speedup=math.nan)


def test_per_candidate_s_nan_raises_value_error() -> None:
    with pytest.raises(ValueError, match="per_candidate_s"):
        forge.optimize(per_candidate_s=math.nan)
