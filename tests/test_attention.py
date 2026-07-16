"""Attention op のユニットテスト（CPU / GPU 不要の範囲）。"""

from __future__ import annotations

import torch

from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.ops import OP_INFO, get_op_info
from forge.search.params import SearchParams
from forge.search.space import SearchSpace
from forge.validation.test_cases import correctness_cases, primary_input
from forge.validation.tolerance import get_tolerance


def _attention_spec(B=2, H=8, S=64, D=64, causal=True) -> KernelSpec:
    shape = (B, H, S, D)
    ts = TensorSpec(shape, torch.float16, True)
    return KernelSpec(
        op_type="attention",
        input_specs=(ts, ts, ts),
        output_specs=(ts,),
        constants={"causal": causal},
        graph_hash="attention_v1",
        constraints=(),
    )


class TestOpInfo:
    def test_attention_in_op_info(self) -> None:
        assert "attention" in OP_INFO

    def test_attention_kind_is_attention(self) -> None:
        assert get_op_info("attention").kind == "attention"

    def test_attention_n_tensor_inputs(self) -> None:
        assert get_op_info("attention").n_tensor_inputs == 3


class TestAttentionSearchSpace:
    def test_enumerate_yields_attention_variant(self) -> None:
        space = SearchSpace()
        params = list(space.enumerate(_attention_spec(), "8.0"))
        variants = {p.variant for p in params}
        assert variants == {"attention"}

    def test_enumerate_uses_attention_blocks(self) -> None:
        space = SearchSpace(attention_blocks=[16, 32])
        params = list(space.enumerate(_attention_spec(), "8.0"))
        block_sizes = {p.block_size for p in params}
        assert block_sizes == {16, 32}

    def test_enumerate_no_reduction_variants(self) -> None:
        space = SearchSpace()
        params = list(space.enumerate(_attention_spec(), "8.0"))
        for p in params:
            assert p.variant not in ("single_row", "multi_row", "two_pass", "elementwise")

    def test_pascal_limits_stages(self) -> None:
        space = SearchSpace()
        params = list(space.enumerate(_attention_spec(), "6.1"))
        for p in params:
            assert p.num_stages == 1


class TestAttentionSearchParams:
    def test_attention_variant_valid(self) -> None:
        p = SearchParams(block_size=64, num_warps=4, num_stages=2, variant="attention")
        assert p.variant == "attention"

    def test_rows_per_program_1_for_attention(self) -> None:
        p = SearchParams(block_size=64, num_warps=4, num_stages=2, variant="attention")
        assert p.rows_per_program == 1


class TestAttentionCodegen:
    def test_generate_causal_template(self) -> None:
        from forge.codegen.triton_codegen import generate

        spec = _attention_spec(causal=True)
        params = SearchParams(block_size=32, num_warps=4, num_stages=2, variant="attention")
        code = generate(spec, params)
        assert "CAUSAL=True" in code
        assert "kernel_fn" in code
        assert "_attention_kernel" in code

    def test_generate_non_causal_template(self) -> None:
        from forge.codegen.triton_codegen import generate

        spec = _attention_spec(causal=False)
        params = SearchParams(block_size=32, num_warps=4, num_stages=2, variant="attention")
        code = generate(spec, params)
        assert "CAUSAL=False" in code

    def test_generate_block_size_in_code(self) -> None:
        from forge.codegen.triton_codegen import generate

        spec = _attention_spec()
        params = SearchParams(block_size=64, num_warps=8, num_stages=3, variant="attention")
        code = generate(spec, params)
        assert "BLOCK_M = 64" in code


class TestAttentionTestCases:
    def test_primary_input_returns_3_tensors(self) -> None:
        spec = _attention_spec(B=2, H=8, S=64, D=64)
        inp = primary_input(spec)
        assert len(inp) == 3

    def test_primary_input_shapes_match_spec(self) -> None:
        spec = _attention_spec(B=2, H=8, S=64, D=64)
        inp = primary_input(spec)
        for tensor_spec in inp:
            assert tensor_spec["shape"] == [2, 8, 64, 64]

    def test_correctness_cases_has_required_names(self) -> None:
        spec = _attention_spec()
        names = {c["name"] for c in correctness_cases(spec)}
        assert {"basic", "odd_seq", "large_values", "small_values"} <= names

    def test_correctness_cases_each_has_3_inputs(self) -> None:
        spec = _attention_spec()
        for case in correctness_cases(spec):
            assert len(case["input_specs"]) == 3


class TestAttentionTolerance:
    def test_tolerance_defined(self) -> None:
        tol = get_tolerance("attention")
        assert tol.atol >= 1e-2

    def test_tolerance_rtol(self) -> None:
        tol = get_tolerance("attention")
        assert tol.rtol >= 1e-2
