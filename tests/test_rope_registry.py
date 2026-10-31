"""rope OpDefinition の CPU 単体テスト。GPU 不要。"""

from __future__ import annotations

import pytest
import torch

from forge.ops.registry import OP_REGISTRY


@pytest.fixture
def rope_def():
    return OP_REGISTRY["rope"]


class TestRopeReference:
    def test_basic_fp32(self, rope_def) -> None:
        m, d = 4, 16
        x = torch.randn(m, d)
        cos = torch.randn(m, d)
        sin = torch.randn(m, d)
        out = rope_def.reference_fn(x, cos, sin)
        # manual: rotate_half(x) = cat([-x[8:], x[:8]], dim=-1)
        h = d // 2
        rot = torch.cat([-x[:, h:], x[:, :h]], dim=-1)
        expected = (x.float() * cos.float() + rot.float() * sin.float()).to(x.dtype)
        torch.testing.assert_close(out, expected)

    def test_fp16_preserves_dtype(self, rope_def) -> None:
        x = torch.randn(4, 16, dtype=torch.float16)
        cos = torch.randn(4, 16, dtype=torch.float16)
        sin = torch.randn(4, 16, dtype=torch.float16)
        out = rope_def.reference_fn(x, cos, sin)
        assert out.dtype == torch.float16

    def test_zeros_x(self, rope_def) -> None:
        x = torch.zeros(4, 16)
        cos = torch.randn(4, 16)
        sin = torch.randn(4, 16)
        out = rope_def.reference_fn(x, cos, sin)
        torch.testing.assert_close(out, torch.zeros(4, 16))

    def test_identity_when_cos1_sin0(self, rope_def) -> None:
        x = torch.randn(4, 16)
        cos = torch.ones(4, 16)
        sin = torch.zeros(4, 16)
        out = rope_def.reference_fn(x, cos, sin)
        torch.testing.assert_close(out, x)


class TestRopeBaseline:
    def test_matches_reference_fp32(self, rope_def) -> None:
        x = torch.randn(8, 32, generator=torch.Generator().manual_seed(42))
        cos = torch.randn(8, 32, generator=torch.Generator().manual_seed(43))
        sin = torch.randn(8, 32, generator=torch.Generator().manual_seed(44))
        ref = rope_def.reference_fn(x, cos, sin)
        base = rope_def.baseline_fn(x, cos, sin)
        torch.testing.assert_close(ref, base, atol=1e-5, rtol=1e-5)

    def test_baseline_display_name(self, rope_def) -> None:
        assert "rotate_half" in rope_def.baseline_display_name


class TestRopeInputs:
    def test_returns_three_specs(self, rope_def) -> None:
        from forge.ops.registry import _rope_inputs

        specs = _rope_inputs(4, 16, "fp32")
        assert len(specs) == 3

    def test_all_same_shape(self) -> None:
        from forge.ops.registry import _rope_inputs

        specs = _rope_inputs(4, 16, "fp32")
        for s in specs:
            assert s["shape"] == [4, 16]

    def test_different_seeds(self) -> None:
        from forge.ops.registry import _rope_inputs

        specs = _rope_inputs(4, 16, "fp32", seed=10)
        seeds = [s["seed"] for s in specs]
        assert len(set(seeds)) == 3


class TestRopeRegistry:
    def test_in_op_registry(self) -> None:
        assert "rope" in OP_REGISTRY

    def test_op_type_field(self, rope_def) -> None:
        assert rope_def.op_type == "rope"

    def test_tolerance(self, rope_def) -> None:
        assert rope_def.tolerance.atol == pytest.approx(2e-3)
        assert rope_def.tolerance.rtol == pytest.approx(1e-2)
