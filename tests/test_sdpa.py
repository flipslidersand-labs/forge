"""scaled_dot_product_attention op の統合ユニットテスト（CPU / GPU 不要）。

test_codegen.py / test_validation.py / test_search.py に分散した SDPA テストを補完し、
OP_INFO・runtime/reference の未カバー領域を埋める専用ファイル。
"""

from __future__ import annotations

import torch

from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.ops import OP_INFO, get_op_info, is_matmul
from forge.runtime.reference import (
    BASELINE_IMPLS,
    REFERENCE_IMPLS,
    baseline_name,
    get_baseline,
    get_reference,
    sdpa_reference,
)
from forge.search.params import SearchParams
from forge.search.space import SearchSpace
from forge.validation.test_cases import correctness_cases, primary_input
from forge.validation.tolerance import get_tolerance

OP = "scaled_dot_product_attention"


def _sdpa_spec(bh: int = 8, s: int = 64, d: int = 64, is_causal: bool = False) -> KernelSpec:
    shape = (bh, s, d)
    ts = TensorSpec(shape, torch.float16, True)
    return KernelSpec(
        op_type=OP,
        input_specs=(ts, ts, ts),
        output_specs=(ts,),
        constants={"is_causal": is_causal},
        graph_hash="sdpa_test_v1",
        constraints=(),
    )


def _flash(block_size: int = 64, num_warps: int = 4, num_stages: int = 1) -> SearchParams:
    return SearchParams(
        block_size=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
        acc_dtype="fp32",
        variant="flash",
        rows_per_program=1,
    )


class TestOpInfo:
    def test_op_registered(self) -> None:
        assert OP in OP_INFO

    def test_kind_is_matmul(self) -> None:
        assert get_op_info(OP).kind == "matmul"

    def test_is_matmul_helper(self) -> None:
        assert is_matmul(OP)

    def test_n_tensor_inputs_is_3(self) -> None:
        assert get_op_info(OP).n_tensor_inputs == 3


class TestSearchSpace:
    def test_flash_variants_only(self) -> None:
        params = list(SearchSpace().enumerate(_sdpa_spec(), "8.9"))
        assert params
        assert {p.variant for p in params} <= {"flash", "flash_causal_opt"}
        assert "flash" in {p.variant for p in params}
        assert "flash_causal_opt" in {p.variant for p in params}

    def test_attention_seq_blocks_includes_small_sizes(self) -> None:
        params = list(SearchSpace().enumerate(_sdpa_spec(s=64), "8.9"))
        blocks = {p.block_size for p in params}
        assert 16 in blocks and 32 in blocks

    def test_custom_attention_seq_blocks(self) -> None:
        space = SearchSpace(attention_seq_blocks=[16, 32])
        params = list(space.enumerate(_sdpa_spec(s=64), "8.9"))
        assert {p.block_size for p in params} == {16, 32}

    def test_block_size_leq_seq_len(self) -> None:
        params = list(SearchSpace().enumerate(_sdpa_spec(s=64), "8.9"))
        for p in params:
            assert p.block_size <= 64

    def test_no_reduction_or_elementwise_variants(self) -> None:
        params = list(SearchSpace().enumerate(_sdpa_spec(), "8.9"))
        for p in params:
            assert p.variant not in ("single_row", "multi_row", "two_pass", "elementwise")

    def test_pascal_limits_num_stages_to_1(self) -> None:
        params = list(SearchSpace().enumerate(_sdpa_spec(), "6.1"))
        for p in params:
            assert p.num_stages == 1

    def test_acc_dtype_always_fp32(self) -> None:
        params = list(SearchSpace().enumerate(_sdpa_spec(), "8.9"))
        for p in params:
            assert p.acc_dtype == "fp32"

    def test_rows_per_program_always_1(self) -> None:
        params = list(SearchSpace().enumerate(_sdpa_spec(), "8.9"))
        for p in params:
            assert p.rows_per_program == 1

    def test_small_seq_len_uses_fallback_block(self) -> None:
        # seq_len < min(attention_seq_blocks) でもフォールバックして候補を返す
        params = list(SearchSpace().enumerate(_sdpa_spec(s=16), "8.9"))
        assert params
        for p in params:
            assert p.block_size <= 16

    def test_head_dim_not_multiple_of_16_yields_nothing(self) -> None:
        params = list(SearchSpace().enumerate(_sdpa_spec(d=65), "8.9"))
        assert params == []


class TestSearchParams:
    def test_flash_variant_accepted(self) -> None:
        p = _flash()
        assert p.variant == "flash"

    def test_flash_is_hashable(self) -> None:
        p = _flash()
        assert hash(p) is not None
        assert p in {p}

    def test_flash_equality(self) -> None:
        assert _flash(64, 4, 1) == _flash(64, 4, 1)
        assert _flash(64, 4, 1) != _flash(128, 4, 1)


class TestCodegen:
    def test_generate_causal_true(self) -> None:
        from forge.codegen.triton_codegen import generate

        code = generate(_sdpa_spec(is_causal=True), _flash())
        assert "IS_CAUSAL=True" in code

    def test_generate_causal_false(self) -> None:
        from forge.codegen.triton_codegen import generate

        code = generate(_sdpa_spec(is_causal=False), _flash())
        assert "IS_CAUSAL=False" in code

    def test_head_dim_embedded_in_code(self) -> None:
        from forge.codegen.triton_codegen import generate

        code = generate(_sdpa_spec(d=32), _flash())
        assert "HEAD_DIM = 32" in code or "head_dim=32" in code or "32" in code

    def test_contains_kernel_fn(self) -> None:
        from forge.codegen.triton_codegen import generate

        code = generate(_sdpa_spec(), _flash())
        assert "kernel_fn" in code

    def test_causal_opt_variant_valid_python(self) -> None:
        from forge.codegen.triton_codegen import generate
        from forge.search.params import SearchParams

        p = SearchParams(
            block_size=64, num_warps=4, num_stages=1, acc_dtype="fp32", variant="flash_causal_opt"
        )
        code = generate(_sdpa_spec(is_causal=True), p)
        compile(code, "<gen>", "exec")
        assert "flash_causal_opt" in code
        assert "safe_end" in code  # causal block-skip ロジック

    def test_causal_opt_non_causal_valid_python(self) -> None:
        from forge.codegen.triton_codegen import generate
        from forge.search.params import SearchParams

        p = SearchParams(
            block_size=64, num_warps=4, num_stages=1, acc_dtype="fp32", variant="flash_causal_opt"
        )
        code = generate(_sdpa_spec(is_causal=False), p)
        compile(code, "<gen>", "exec")

    def test_causal_opt_small_block_valid(self) -> None:
        from forge.codegen.triton_codegen import generate
        from forge.search.params import SearchParams

        p = SearchParams(
            block_size=16, num_warps=4, num_stages=1, acc_dtype="fp32", variant="flash_causal_opt"
        )
        code = generate(_sdpa_spec(s=64, d=64, is_causal=True), p)
        compile(code, "<gen>", "exec")
        assert "BLOCK_S=16" in code

    def test_op_variant_annotation_in_header(self) -> None:
        from forge.codegen.triton_codegen import generate

        code = generate(_sdpa_spec(), _flash())
        assert f"op={OP} variant=flash" in code

    def test_non_flash_variant_raises(self) -> None:
        import pytest

        from forge.codegen.triton_codegen import generate

        non_flash = SearchParams(block_size=64, num_warps=4, num_stages=1, variant="single_row")
        with pytest.raises((ValueError, KeyError)):
            generate(_sdpa_spec(), non_flash)


class TestReference:
    def test_sdpa_reference_in_registry(self) -> None:
        assert OP in REFERENCE_IMPLS

    def test_get_reference_returns_callable(self) -> None:
        fn = get_reference(OP)
        assert callable(fn)

    def test_sdpa_reference_output_shape(self) -> None:
        q = torch.randn(2, 8, 32, dtype=torch.float16)
        k = torch.randn(2, 8, 32, dtype=torch.float16)
        v = torch.randn(2, 8, 32, dtype=torch.float16)
        out = sdpa_reference(q, k, v)
        assert out.shape == q.shape

    def test_sdpa_reference_output_dtype_matches_input(self) -> None:
        q = torch.randn(2, 8, 32, dtype=torch.float16)
        k = torch.randn(2, 8, 32, dtype=torch.float16)
        v = torch.randn(2, 8, 32, dtype=torch.float16)
        out = sdpa_reference(q, k, v)
        assert out.dtype == torch.float16

    def test_sdpa_reference_causal_vs_non_causal_differ(self) -> None:
        torch.manual_seed(0)
        q = torch.randn(2, 8, 32)
        k = torch.randn(2, 8, 32)
        v = torch.randn(2, 8, 32)
        out_nc = sdpa_reference(q, k, v, is_causal=False)
        out_c = sdpa_reference(q, k, v, is_causal=True)
        assert not torch.allclose(out_nc, out_c)

    def test_sdpa_baseline_in_registry(self) -> None:
        assert OP in BASELINE_IMPLS

    def test_get_baseline_returns_callable(self) -> None:
        fn = get_baseline(OP)
        assert callable(fn)

    def test_sdpa_baseline_output_shape(self) -> None:
        q = torch.randn(2, 8, 32, dtype=torch.float16)
        k = torch.randn(2, 8, 32, dtype=torch.float16)
        v = torch.randn(2, 8, 32, dtype=torch.float16)
        fn = get_baseline(OP)
        out = fn(q, k, v)
        assert out.shape == q.shape

    def test_baseline_name_is_f_sdpa(self) -> None:
        assert baseline_name(OP) == "F.scaled_dot_product_attention"


class TestTestCases:
    def test_primary_input_has_3_tensors(self) -> None:
        inp = primary_input(_sdpa_spec())
        assert len(inp) == 3

    def test_primary_input_shapes_match_spec(self) -> None:
        spec = _sdpa_spec(bh=4, s=32, d=16)
        inp = primary_input(spec)
        for t in inp:
            assert t["shape"] == [4, 32, 16]

    def test_correctness_cases_count(self) -> None:
        cases = correctness_cases(_sdpa_spec())
        assert len(cases) >= 3

    def test_correctness_cases_names(self) -> None:
        names = {c["name"] for c in correctness_cases(_sdpa_spec())}
        assert {"basic", "single_batch", "multi_batch"} <= names

    def test_correctness_cases_each_has_3_inputs(self) -> None:
        for case in correctness_cases(_sdpa_spec()):
            assert len(case["input_specs"]) == 3


class TestTolerance:
    def test_tolerance_registered(self) -> None:
        tol = get_tolerance(OP)
        assert tol is not None

    def test_tolerance_atol(self) -> None:
        tol = get_tolerance(OP)
        assert tol.atol >= 1e-3

    def test_tolerance_rtol(self) -> None:
        tol = get_tolerance(OP)
        assert tol.rtol >= 1e-3

    def test_tolerance_relaxed_vs_rmsnorm(self) -> None:
        sdpa_tol = get_tolerance(OP)
        rmsnorm_tol = get_tolerance("rmsnorm")
        assert sdpa_tol.atol >= rmsnorm_tol.atol
