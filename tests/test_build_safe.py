"""_build_safe / eager フォールバック契約の検証 (#319)。GPU 不要。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from forge.decorator import optimize


def _cuda_tensor() -> MagicMock:
    t = MagicMock(spec=torch.Tensor)
    t.is_cuda = True
    t.shape = (4, 4)
    t.dtype = torch.float32
    return t


def _bind_single_tensor(t: MagicMock):
    """inspect.Signature.bind をパッチし、wrapper 内の tensors リストを [t] に固定する。"""
    bound = MagicMock()
    bound.arguments = {"x": t}
    bound.apply_defaults.return_value = None
    return patch("inspect.Signature.bind", return_value=bound)


class TestBuildExceptionEagerFallback:
    def test_build_exception_falls_back_to_eager_instead_of_raising(self) -> None:
        """_build() が例外を送出しても wrapper は落ちず eager 関数の結果を返す。"""
        t = _cuda_tensor()
        calls: list[str] = []

        def fn(x):
            calls.append("eager")
            return "eager-result"

        with (
            patch("forge.decorator.identify", return_value="rmsnorm"),
            patch("forge.decorator._build", side_effect=RuntimeError("codegen exploded")),
            _bind_single_tensor(t),
        ):
            wrapped = optimize(budget=1)(fn)
            result = wrapped(t)

        assert result == "eager-result"
        assert calls == ["eager"]

    def test_build_exception_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        t = _cuda_tensor()

        def fn(x):
            return x

        with (
            patch("forge.decorator.identify", return_value="rmsnorm"),
            patch("forge.decorator._build", side_effect=RuntimeError("codegen exploded")),
            _bind_single_tensor(t),
            caplog.at_level("WARNING", logger="forge.decorator"),
        ):
            wrapped = optimize(budget=1)(fn)
            wrapped(t)

        assert any("build failed" in r.message for r in caplog.records)

    def test_keyboard_interrupt_is_not_swallowed(self) -> None:
        """KeyboardInterrupt/SystemExit は eager フォールバックせず再送出する。"""
        t = _cuda_tensor()

        def fn(x):
            return x

        with (
            patch("forge.decorator.identify", return_value="rmsnorm"),
            patch("forge.decorator._build", side_effect=KeyboardInterrupt),
            _bind_single_tensor(t),
            pytest.raises(KeyboardInterrupt),
        ):
            wrapped = optimize(budget=1)(fn)
            wrapped(t)

    def test_compiled_dict_records_none_after_exception(self) -> None:
        """例外後、同一 shape の再呼び出しでも _build を再試行せず None（eager）を再利用する。"""
        t = _cuda_tensor()
        build_calls = 0

        def failing_build(*args, **kwargs):
            nonlocal build_calls
            build_calls += 1
            raise RuntimeError("codegen exploded")

        def fn(x):
            return "eager-result"

        with (
            patch("forge.decorator.identify", return_value="rmsnorm"),
            patch("forge.decorator._build", side_effect=failing_build),
            _bind_single_tensor(t),
        ):
            wrapped = optimize(budget=1)(fn)
            wrapped(t)
            wrapped(t)

        assert build_calls == 1
