"""Linear op の統合ユニットテスト（GPU 不要）。"""

from __future__ import annotations

import torch

from forge.codegen.triton_codegen import generate
from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.lowering import identify
from forge.ops import OP_INFO, get_op_info, is_gemm
from forge.runtime.reference import BASELINE_IMPLS, REFERENCE_IMPLS, baseline_name
from forge.search.params import SearchParams
from forge.search.space import SearchSpace
from forge.validation.test_cases import correctness_cases, primary_input
from forge.validation.tolerance import get_tolerance

OP = "linear"


def _linear_spec(m: int = 4096, k: int = 4096, n: int = 4096) -> KernelSpec:
    x_spec = TensorSpec(shape=(m, k), dtype=torch.float16, is_contiguous=True)
    w_spec = TensorSpec(shape=(n, k), dtype=torch.float16, is_contiguous=True)
    b_spec = TensorSpec(shape=(n,), dtype=torch.float16, is_contiguous=True)
    return KernelSpec(
        op_type=OP,
        input_specs=(x_spec, w_spec, b_spec),
        output_specs=(TensorSpec(shape=(m, n), dtype=torch.float16, is_contiguous=True),),
        constants={},
        graph_hash="linear_test_v1",
        constraints=(),
    )


def _gemm_params(block_size: int = 128, num_warps: int = 4) -> SearchParams:
    return SearchParams(
        block_size=block_size,
        num_warps=num_warps,
        num_stages=1,
        acc_dtype="fp32",
        variant="gemm",
        rows_per_program=1,
    )


class TestOpInfo:
    def test_op_registered(self) -> None:
        assert OP in OP_INFO

    def test_kind_is_gemm(self) -> None:
        assert get_op_info(OP).kind == "gemm"

    def test_is_gemm_helper(self) -> None:
        assert is_gemm(OP)

    def test_n_tensor_inputs(self) -> None:
        assert get_op_info(OP).n_tensor_inputs == 3


class TestLowering:
    def test_f_linear_with_bias_identified(self) -> None:
        import torch.nn.functional as F

        def fn(x, weight, bias):
            return F.linear(x, weight, bias)

        assert identify(fn) == OP

    def test_f_linear_no_bias_identified(self) -> None:
        import torch.nn.functional as F

        def fn(x, weight):
            return F.linear(x, weight)

        assert identify(fn) == OP


class TestCodegen:
    def test_generate_valid_python(self) -> None:
        code = generate(_linear_spec(), _gemm_params())
        compile(code, "<gen>", "exec")

    def test_contains_kernel_fn(self) -> None:
        code = generate(_linear_spec(), _gemm_params())
        assert "def kernel_fn(x, weight, bias=None)" in code

    def test_block_size_embedded(self) -> None:
        code = generate(_linear_spec(), _gemm_params(block_size=64))
        assert "BLOCK_M=64" in code
        assert "BLOCK_N=64" in code

    def test_block_k_default(self) -> None:
        code = generate(_linear_spec(), _gemm_params())
        assert "BLOCK_K=32" in code

    def test_has_bias_logic(self) -> None:
        code = generate(_linear_spec(), _gemm_params())
        assert "HAS_BIAS" in code

    def test_acc_fp32(self) -> None:
        code = generate(_linear_spec(), _gemm_params())
        assert "float32" in code

    def test_op_annotation(self) -> None:
        code = generate(_linear_spec(), _gemm_params())
        assert "op=linear variant=gemm" in code


class TestSearchSpace:
    def test_gemm_variant_only(self) -> None:
        params = list(SearchSpace().enumerate(_linear_spec(), "8.9"))
        assert params
        assert all(p.variant == "gemm" for p in params)

    def test_pascal_restricts_stages(self) -> None:
        params = list(SearchSpace().enumerate(_linear_spec(), "6.1"))
        assert all(p.num_stages == 1 for p in params)

    def test_no_duplicates(self) -> None:
        params = list(SearchSpace().enumerate(_linear_spec(), "8.9"))
        keys = [(p.block_size, p.block_k, p.num_warps, p.num_stages, p.acc_dtype) for p in params]
        assert len(keys) == len(set(keys))

    def test_block_sizes_are_gemm_output_blocks(self) -> None:
        space = SearchSpace()
        params = list(space.enumerate(_linear_spec(), "8.9"))
        assert {p.block_size for p in params} == set(space.gemm_output_blocks)


class TestReference:
    def test_reference_registered(self) -> None:
        assert OP in REFERENCE_IMPLS

    def test_baseline_registered(self) -> None:
        assert OP in BASELINE_IMPLS

    def test_baseline_name(self) -> None:
        assert baseline_name(OP) == "F.linear"


class TestValidation:
    def test_primary_input_shape(self) -> None:
        spec = _linear_spec(m=2048, k=4096, n=4096)
        inputs = primary_input(spec)
        assert len(inputs) == 3
        assert inputs[0]["shape"] == [2048, 4096]  # x
        assert inputs[1]["shape"] == [4096, 4096]  # weight [N, K]
        assert inputs[2]["shape"] == [4096]  # bias

    def test_correctness_cases_count(self) -> None:
        cases = correctness_cases(_linear_spec())
        assert len(cases) == 5

    def test_tolerance_registered(self) -> None:
        tol = get_tolerance(OP)
        assert tol.atol <= 1e-1
