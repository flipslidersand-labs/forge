"""objective="economic" + min_invocations の採算判定分岐の検証 (#477)。GPU 不要。

decorator.wrapper の _time_eager → should_run_search → (skip: eager 固定 /
run: _build_or_eager) の配線を、_time_eager と _build_or_eager を差し替えて確認する。
CUDA テンソルは MagicMock(spec=torch.Tensor) で代用する。

採算の目安: budget=50, per_candidate_s=2.0 → 探索コスト 1e8 µs。
baseline_us=1000 のとき 1 呼び出しあたりの節約は約 230.8 µs なので、
損益分岐は約 43 万回。min_invocations=10 は NG、10**9 は OK になる。
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
import torch

from forge.decorator import optimize

_BASELINE_US = 1000.0
_SKIP_INVOCATIONS = 10
_RUN_INVOCATIONS = 10**9
_SENTINEL = object()


def _cuda_tensor() -> MagicMock:
    t = MagicMock(spec=torch.Tensor)
    t.is_cuda = True
    t.shape = (4, 4)
    t.dtype = torch.float32
    t.device = torch.device("cuda:0")
    t.is_contiguous.return_value = True
    return t


def _make_fn() -> MagicMock:
    """呼び出し回数を数える eager 関数(固定の sentinel を返す)。"""
    eager = MagicMock(return_value=_SENTINEL)
    eager.__qualname__ = "eager_fn"
    return eager


def _decorate(eager: MagicMock, **opt_kwargs):
    def fn(x):
        return eager(x)

    return optimize(**opt_kwargs)(fn)


@pytest.fixture
def patched():
    """identify を固定し、_time_eager / _build_or_eager を差し替える。"""
    with (
        patch("forge.decorator.identify", return_value="softmax"),
        patch("forge.decorator._time_eager", return_value=_BASELINE_US) as time_eager,
        patch("forge.decorator._build_or_eager", return_value=None) as build,
    ):
        yield time_eager, build


class TestEconomicSkip:
    def test_unprofitable_skips_build_and_returns_eager(self, patched) -> None:
        """採算 NG: _build_or_eager を呼ばず eager の結果を返す。"""
        time_eager, build = patched
        eager = _make_fn()
        wrapped = _decorate(eager, objective="economic", min_invocations=_SKIP_INVOCATIONS)

        out = wrapped(_cuda_tensor())

        assert out is _SENTINEL
        eager.assert_called_once()
        time_eager.assert_called_once()
        build.assert_not_called()

    def test_skip_pins_eager_and_does_not_remeasure(self, patched) -> None:
        """採算 NG の判定は shape ごとに eager 固定され、2 回目以降は再計測しない。"""
        time_eager, build = patched
        eager = _make_fn()
        wrapped = _decorate(eager, objective="economic", min_invocations=_SKIP_INVOCATIONS)
        x = _cuda_tensor()

        wrapped(x)
        wrapped(x)

        assert time_eager.call_count == 1
        build.assert_not_called()
        assert eager.call_count == 2
        # キャッシュキーは (shape, dtype)。eager 固定は None として保存される。
        compiled = wrapped._forge_compiled
        assert list(compiled.values()) == [None]

    def test_skip_is_per_shape(self, patched) -> None:
        """別 shape は別キーなので改めて計測される。"""
        time_eager, _build = patched
        wrapped = _decorate(_make_fn(), objective="economic", min_invocations=_SKIP_INVOCATIONS)
        a = _cuda_tensor()
        b = _cuda_tensor()
        b.shape = (8, 8)

        wrapped(a)
        wrapped(b)

        assert time_eager.call_count == 2


class TestEconomicRun:
    def test_profitable_calls_build_once(self, patched) -> None:
        """採算 OK: _build_or_eager が 1 回だけ呼ばれ、2 回目は再ビルドしない。"""
        time_eager, build = patched
        kernel = MagicMock(return_value="kernel-out")
        build.return_value = kernel
        eager = _make_fn()
        wrapped = _decorate(eager, objective="economic", min_invocations=_RUN_INVOCATIONS)
        x = _cuda_tensor()

        assert wrapped(x) == "kernel-out"
        assert wrapped(x) == "kernel-out"

        time_eager.assert_called_once()
        build.assert_called_once()
        assert kernel.call_count == 2
        eager.assert_not_called()  # _time_eager は差し替え済みなので eager も呼ばれない

    def test_profitable_but_build_fails_returns_eager(self, patched) -> None:
        """採算 OK でも _build_or_eager が None(探索失敗)なら eager で実行する。"""
        _time_eager, build = patched
        build.return_value = None
        eager = _make_fn()
        wrapped = _decorate(eager, objective="economic", min_invocations=_RUN_INVOCATIONS)

        assert wrapped(_cuda_tensor()) is _SENTINEL
        build.assert_called_once()
        eager.assert_called_once()

    def test_search_cost_uses_budget_and_per_candidate_s(self, patched) -> None:
        """search_cost_s = budget * per_candidate_s で should_run_search が呼ばれる。"""
        _time_eager, _build = patched
        wrapped = _decorate(
            _make_fn(),
            objective="economic",
            min_invocations=7,
            budget=5,
            per_candidate_s=3.0,
        )
        with patch("forge.decorator.should_run_search") as sr:
            sr.return_value = MagicMock(should_search=False, reason="stub")
            wrapped(_cuda_tensor())

        sr.assert_called_once_with(7, 15.0, _BASELINE_US)


class TestEconomicProgress:
    def test_progress_receives_adoption_message(self, patched) -> None:
        """progress に 'adoption: ...' が渡る(skip 側)。"""
        messages: list[str] = []
        wrapped = _decorate(
            _make_fn(),
            objective="economic",
            min_invocations=_SKIP_INVOCATIONS,
            progress=messages.append,
        )

        wrapped(_cuda_tensor())

        assert len(messages) == 1
        assert messages[0].startswith("adoption: ")
        assert "skip search" in messages[0]

    def test_progress_receives_adoption_message_when_running(self, patched) -> None:
        """progress に 'adoption: ...' が渡る(search 側)。"""
        _time_eager, build = patched
        messages: list[str] = []
        wrapped = _decorate(
            _make_fn(),
            objective="economic",
            min_invocations=_RUN_INVOCATIONS,
            progress=messages.append,
        )

        wrapped(_cuda_tensor())

        build.assert_called_once()
        assert len(messages) == 1
        assert messages[0].startswith("adoption: break-even at")

    def test_progress_none_is_noop_but_logged(self, patched, caplog) -> None:
        """progress=None でも落ちず、判断理由は INFO ログに残る。"""
        wrapped = _decorate(
            _make_fn(), objective="economic", min_invocations=_SKIP_INVOCATIONS, progress=None
        )

        with caplog.at_level(logging.INFO, logger="forge.decorator"):
            assert wrapped(_cuda_tensor()) is _SENTINEL

        assert any("adoption fn=" in r.getMessage() for r in caplog.records)


class TestEconomicNotEngaged:
    def test_min_invocations_zero_skips_timing(self, patched) -> None:
        """economic でも min_invocations=0(既定)なら _time_eager は呼ばれず直接ビルド。"""
        time_eager, build = patched
        wrapped = _decorate(_make_fn(), objective="economic", min_invocations=0)

        wrapped(_cuda_tensor())

        time_eager.assert_not_called()
        build.assert_called_once()

    def test_latency_objective_ignores_min_invocations(self, patched) -> None:
        """objective=latency では min_invocations>0 でも _time_eager は呼ばれない。"""
        time_eager, build = patched
        wrapped = _decorate(_make_fn(), objective="latency", min_invocations=_SKIP_INVOCATIONS)

        wrapped(_cuda_tensor())

        time_eager.assert_not_called()
        build.assert_called_once()


class TestTimeEagerFailureContract:
    """_time_eager の例外は _build_or_eager の外で起きるため、wrapper に伝播する。

    現状の挙動を固定するテスト。eager フォールバックへ契約を変えるなら別 Issue で
    このテストごと更新すること。
    """

    def test_exception_propagates_and_does_not_pin(self, patched) -> None:
        time_eager, build = patched
        time_eager.side_effect = RuntimeError("timing boom")
        wrapped = _decorate(_make_fn(), objective="economic", min_invocations=_SKIP_INVOCATIONS)
        x = _cuda_tensor()

        with pytest.raises(RuntimeError, match="timing boom"):
            wrapped(x)

        build.assert_not_called()
        # 失敗はキャッシュされず、次の呼び出しで再計測される。
        assert wrapped._forge_compiled == {}
        with pytest.raises(RuntimeError, match="timing boom"):
            wrapped(x)
        assert time_eager.call_count == 2

    def test_eager_fn_exception_during_timing_propagates(self) -> None:
        """実 _time_eager 経由: fn 自身の例外(warmup 中)も wrapper に伝播する。"""

        def boom(x):
            raise ValueError("fn boom")

        with (
            patch("forge.decorator.identify", return_value="softmax"),
            patch("forge.decorator._build_or_eager") as build,
        ):
            wrapped = optimize(objective="economic", min_invocations=_SKIP_INVOCATIONS)(boom)
            with pytest.raises(ValueError, match="fn boom"):
                wrapped(_cuda_tensor())

        build.assert_not_called()
