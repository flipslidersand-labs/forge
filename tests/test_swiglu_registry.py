"""swiglu OpDefinition の CPU 単体テスト。GPU 不要。"""

from __future__ import annotations

import torch
import pytest

from forge.ops.registry import OP_REGISTRY


@pytest.fixture
def swiglu_def():
    return OP_REGISTRY["swiglu"]


class TestSwigluReference:
    def test_basic_fp32(self, swiglu_def) -> None:
        x = torch.randn(4, 16)
        gate = torch.randn(4, 16)
        out = swiglu_def.reference_fn(x, gate)
        expected = torch.nn.functional.silu(gate.float()) * x.float()
        torch.testing.assert_close(out, expected.to(x.dtype))

    def test_fp16_cast(self, swiglu_def) -> None:
        x = torch.randn(4, 16, dtype=torch.float16)
        gate = torch.randn(4, 16, dtype=torch.float16)
        out = swiglu_def.reference_fn(x, gate)
        assert out.dtype == torch.float16

    def test_zeros_input(self, swiglu_def) -> None:
        x = torch.zeros(4, 16)
        gate = torch.zeros(4, 16)
        out = swiglu_def.reference_fn(x, gate)
        torch.testing.assert_close(out, torch.zeros(4, 16))


class TestSwigluBaseline:
    def test_matches_reference_fp32(self, swiglu_def) -> None:
        x = torch.randn(8, 32, generator=torch.Generator().manual_seed(42))
        gate = torch.randn(8, 32, generator=torch.Generator().manual_seed(43))
        ref = swiglu_def.reference_fn(x, gate)
        base = swiglu_def.baseline_fn(x, gate)
        torch.testing.assert_close(ref, base, atol=1e-5, rtol=1e-5)

    def test_baseline_display_name(self, swiglu_def) -> None:
        assert swiglu_def.baseline_display_name == "F.silu(gate) * x"


class TestSwigluInputs:
    def test_returns_two_specs(self, swiglu_def) -> None:
        from forge.ops.registry import _swiglu_inputs

        specs = _swiglu_inputs(4, 16, "fp32")
        assert len(specs) == 2

    def test_same_shape(self, swiglu_def) -> None:
        from forge.ops.registry import _swiglu_inputs

        specs = _swiglu_inputs(4, 16, "fp32")
        assert specs[0]["shape"] == specs[1]["shape"] == [4, 16]

    def test_different_seeds(self, swiglu_def) -> None:
        from forge.ops.registry import _swiglu_inputs

        specs = _swiglu_inputs(4, 16, "fp32", seed=10)
        assert specs[0]["seed"] != specs[1]["seed"]


class TestSwigluRegistry:
    def test_in_op_registry(self) -> None:
        assert "swiglu" in OP_REGISTRY

    def test_op_type_field(self, swiglu_def) -> None:
        assert swiglu_def.op_type == "swiglu"

    def test_tolerance(self, swiglu_def) -> None:
        assert swiglu_def.tolerance.atol == pytest.approx(2e-3)
        assert swiglu_def.tolerance.rtol == pytest.approx(1e-2)
