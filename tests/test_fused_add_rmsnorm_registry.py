"""fused_add_rmsnorm OpDefinition の CPU 単体テスト。GPU 不要。"""

from __future__ import annotations

import pytest
import torch

from forge.ops.registry import OP_REGISTRY


@pytest.fixture
def op_def():
    return OP_REGISTRY["fused_add_rmsnorm"]


class TestFusedAddRmsnormReference:
    def test_basic_fp32(self, op_def) -> None:
        m, n = 4, 16
        x = torch.randn(m, n)
        residual = torch.randn(m, n)
        weight = torch.ones(n)
        out = op_def.reference_fn(x, residual, weight)

        hidden = (x + residual).float()
        rms = torch.rsqrt(torch.mean(hidden * hidden, dim=-1, keepdim=True) + 1e-6)
        expected = (hidden * rms * weight.float()).to(x.dtype)
        torch.testing.assert_close(out, expected)

    def test_fp16_preserves_dtype(self, op_def) -> None:
        x = torch.randn(4, 16, dtype=torch.float16)
        residual = torch.randn(4, 16, dtype=torch.float16)
        weight = torch.ones(16, dtype=torch.float16)
        out = op_def.reference_fn(x, residual, weight)
        assert out.dtype == torch.float16

    def test_zero_residual_matches_rmsnorm(self, op_def) -> None:
        x = torch.randn(4, 16)
        residual = torch.zeros(4, 16)
        weight = torch.ones(16)
        out = op_def.reference_fn(x, residual, weight)

        # x + 0 = x → should equal plain rmsnorm(x)
        xf = x.float()
        rms = torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + 1e-6)
        expected = (xf * rms).to(x.dtype)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


class TestFusedAddRmsnormBaseline:
    def test_matches_reference_fp32(self, op_def) -> None:
        g = torch.Generator().manual_seed(42)
        x = torch.randn(8, 32, generator=g)
        residual = torch.randn(8, 32, generator=g)
        weight = torch.ones(32)
        ref = op_def.reference_fn(x, residual, weight)
        base = op_def.baseline_fn(x, residual, weight)
        torch.testing.assert_close(ref, base, atol=1e-4, rtol=1e-4)

    def test_baseline_display_name(self, op_def) -> None:
        name = op_def.baseline_display_name.lower()
        assert "rms_norm" in name or "rmsnorm" in name


class TestFusedAddRmsnormInputs:
    def test_returns_three_specs(self) -> None:
        from forge.ops.registry import _fused_add_rmsnorm_inputs

        specs = _fused_add_rmsnorm_inputs(4, 16, "fp32")
        assert len(specs) == 3

    def test_shapes(self) -> None:
        from forge.ops.registry import _fused_add_rmsnorm_inputs

        specs = _fused_add_rmsnorm_inputs(4, 16, "fp32")
        assert specs[0]["shape"] == [4, 16]  # x
        assert specs[1]["shape"] == [4, 16]  # residual
        assert specs[2]["shape"] == [16]  # weight

    def test_weight_init_ones(self) -> None:
        from forge.ops.registry import _fused_add_rmsnorm_inputs

        specs = _fused_add_rmsnorm_inputs(4, 16, "fp32")
        assert specs[2]["init"] == "ones"


class TestFusedAddRmsnormRegistry:
    def test_in_op_registry(self) -> None:
        assert "fused_add_rmsnorm" in OP_REGISTRY

    def test_op_type_field(self, op_def) -> None:
        assert op_def.op_type == "fused_add_rmsnorm"

    def test_tolerance(self, op_def) -> None:
        assert op_def.tolerance.atol == pytest.approx(2e-3)
        assert op_def.tolerance.rtol == pytest.approx(1e-2)
