"""wrapper のテンソル間デバイス一貫性検証 (#328)。GPU 不要。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from forge.decorator import optimize


def _cuda_tensor(device: str = "cuda:0") -> MagicMock:
    t = MagicMock(spec=torch.Tensor)
    t.is_cuda = True
    t.shape = (4, 4)
    t.dtype = torch.float32
    t.device = torch.device(device)
    return t


def _cpu_tensor() -> MagicMock:
    t = MagicMock(spec=torch.Tensor)
    t.is_cuda = False
    t.shape = (4, 4)
    t.dtype = torch.float32
    t.device = torch.device("cpu")
    return t


def _bind_two_tensors(a: MagicMock, b: MagicMock):
    bound = MagicMock()
    bound.arguments = {"x": a, "y": b}
    bound.apply_defaults.return_value = None
    return patch("inspect.Signature.bind", return_value=bound)


class TestDeviceConsistency:
    def test_mismatched_cuda_devices_falls_back_to_eager(self) -> None:
        """2つの引数がそれぞれ別GPU上にある場合、_build を呼ばず eager にフォールバック。"""
        a = _cuda_tensor("cuda:0")
        b = _cuda_tensor("cuda:1")
        calls: list[str] = []

        def fn(x, y):
            calls.append("eager")
            return "eager-result"

        with (
            patch("forge.decorator.identify", return_value="rmsnorm"),
            patch("forge.decorator._build") as mock_build,
            _bind_two_tensors(a, b),
        ):
            wrapped = optimize(budget=1)(fn)
            result = wrapped(a, b)

        mock_build.assert_not_called()
        assert result == "eager-result"
        assert calls == ["eager"]

    def test_mismatched_cpu_gpu_falls_back_to_eager(self) -> None:
        """1番目がGPU・2番目がCPUの混在デバイスでも _build を呼ばない。"""
        a = _cuda_tensor("cuda:0")
        b = _cpu_tensor()

        def fn(x, y):
            return "eager-result"

        with (
            patch("forge.decorator.identify", return_value="rmsnorm"),
            patch("forge.decorator._build") as mock_build,
            _bind_two_tensors(a, b),
        ):
            wrapped = optimize(budget=1)(fn)
            result = wrapped(a, b)

        mock_build.assert_not_called()
        assert result == "eager-result"

    def test_mismatched_devices_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        a = _cuda_tensor("cuda:0")
        b = _cuda_tensor("cuda:1")

        def fn(x, y):
            return "eager-result"

        with (
            patch("forge.decorator.identify", return_value="rmsnorm"),
            patch("forge.decorator._build"),
            _bind_two_tensors(a, b),
            caplog.at_level("WARNING", logger="forge.decorator"),
        ):
            wrapped = optimize(budget=1)(fn)
            wrapped(a, b)

        assert any("デバイスが一致していません" in r.message for r in caplog.records)

    def test_matching_devices_proceeds_to_build(self) -> None:
        """全テンソルが同一デバイスなら通常どおり _build が呼ばれる。"""
        a = _cuda_tensor("cuda:0")
        b = _cuda_tensor("cuda:0")

        def fn(x, y):
            return "eager-result"

        with (
            patch("forge.decorator.identify", return_value="rmsnorm"),
            patch("forge.decorator._build", return_value=None) as mock_build,
            _bind_two_tensors(a, b),
        ):
            wrapped = optimize(budget=1)(fn)
            wrapped(a, b)

        mock_build.assert_called_once()
