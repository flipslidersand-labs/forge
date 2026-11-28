import ast

import torch

from forge.codegen.triton_codegen import generate, generate_rmsnorm
from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.search.params import SearchParams


def softmax_spec(n: int = 4096) -> KernelSpec:
    return KernelSpec(
        op_type="softmax",
        input_specs=(TensorSpec(shape=(2048, n), dtype=torch.float16, is_contiguous=True),),
        output_specs=(TensorSpec(shape=(2048, n), dtype=torch.float16, is_contiguous=True),),
        constants={"dim": -1},
        graph_hash="softmax_v1",
        constraints=(),
    )


class TestSoftmaxCodegen:
    def test_single_row_valid_python(self) -> None:
        code = generate(softmax_spec(), default_params())
        compile(code, "<gen>", "exec")
        assert "op=softmax variant=single_row" in code
        assert "tl.exp" in code and "tl.max" in code  # safe softmax
        assert "def kernel_fn(x, dim=-1)" in code

    def test_multi_row_valid_python(self) -> None:
        p = default_params(variant="multi_row", rows_per_program=4)
        code = generate(softmax_spec(), p)
        compile(code, "<gen>", "exec")
        assert "ROWS=4" in code
        assert "triton.cdiv(M, 4)" in code

    def test_two_pass_valid_python_and_loops(self) -> None:
        p = default_params(variant="two_pass", block_size=1024)
        code = generate(softmax_spec(), p)
        compile(code, "<gen>", "exec")
        assert "variant=two_pass" in code
        assert "for start in range" in code  # streaming loops
        assert "tl.maximum" in code  # online max update
        assert "tl.exp" in code  # exp in both passes

    def test_two_pass_tile_size_in_header(self) -> None:
        p = default_params(variant="two_pass", block_size=512)
        code = generate(softmax_spec(), p)
        assert "tile=512" in code


def layernorm_spec(n: int = 4096) -> KernelSpec:
    return KernelSpec(
        op_type="layernorm",
        input_specs=(
            TensorSpec(shape=(2048, n), dtype=torch.float16, is_contiguous=True),
            TensorSpec(shape=(n,), dtype=torch.float16, is_contiguous=True),
            TensorSpec(shape=(n,), dtype=torch.float16, is_contiguous=True),
        ),
        output_specs=(TensorSpec(shape=(2048, n), dtype=torch.float16, is_contiguous=True),),
        constants={"eps": 1e-5},
        graph_hash="layernorm_v1",
        constraints=(),
    )


def gelu_spec(n: int = 4096) -> KernelSpec:
    return KernelSpec(
        op_type="gelu",
        input_specs=(TensorSpec(shape=(2048, n), dtype=torch.float16, is_contiguous=True),),
        output_specs=(TensorSpec(shape=(2048, n), dtype=torch.float16, is_contiguous=True),),
        constants={},
        graph_hash="gelu_v1",
        constraints=(),
    )


class TestLayerNormGeluCodegen:
    def test_layernorm_single_row_valid(self) -> None:
        code = generate(layernorm_spec(), default_params())
        compile(code, "<gen>", "exec")
        assert "op=layernorm" in code
        assert "tl.where" in code  # マスク付き分散
        assert "def kernel_fn(x, weight, bias, eps=1e-5)" in code

    def test_layernorm_multi_row_valid(self) -> None:
        code = generate(layernorm_spec(), default_params(variant="multi_row", rows_per_program=2))
        compile(code, "<gen>", "exec")
        assert "ROWS=2" in code

    def test_layernorm_two_pass_valid(self) -> None:
        p = default_params(variant="two_pass", block_size=1024)
        code = generate(layernorm_spec(), p)
        compile(code, "<gen>", "exec")
        assert "op=layernorm variant=two_pass" in code
        assert "for start in range(0, N, BLOCK_SIZE)" in code
        assert "def kernel_fn(x, weight, bias, eps=1e-5)" in code

    def test_layernorm_welford_valid(self) -> None:
        code = generate(layernorm_spec(), default_params(variant="welford"))
        compile(code, "<gen>", "exec")
        assert "op=layernorm variant=welford" in code
        assert "sum_x" in code and "sum_x2" in code
        assert "tl.where" not in code  # welford は xc 中間ベクトル不要
        assert "def kernel_fn(x, weight, bias, eps=1e-5)" in code

    def test_layernorm_welford_num_warps_32(self) -> None:
        code = generate(layernorm_spec(), default_params(variant="welford", num_warps=32))
        compile(code, "<gen>", "exec")
        assert "num_warps=32" in code

    def test_gelu_elementwise_valid(self) -> None:
        p = default_params(variant="elementwise", block_size=1024)
        code = generate(gelu_spec(), p)
        compile(code, "<gen>", "exec")
        assert "op=gelu variant=elementwise" in code
        assert "tl.erf" in code
        assert "x.numel()" in code
        assert "def kernel_fn(x)" in code

    def test_gelu_erf_argument_is_fp32_for_fp16_acc(self) -> None:
        # tl.erf は fp32/fp64 しか受けない。acc=fp16 でも erf の引数は fp32 に上げ、
        # 結果を acc の型へ戻す (#459)。
        for acc in ("fp16", "fp32"):
            code = generate(gelu_spec(), default_params(variant="elementwise", acc_dtype=acc))
            ast.parse(code)
            erf_lines = [ln for ln in code.splitlines() if "tl.erf(" in ln]
            assert len(erf_lines) == 1, acc
            line = erf_lines[0]
            assert "tl.erf(x.to(tl.float32) *" in line, f"acc={acc}: {line}"
            tl_acc = "float16" if acc == "fp16" else "float32"
            assert f"tl.erf(x.to(tl.float32) * 0.7071067811865476).to(tl.{tl_acc})" in line, (
                f"acc={acc}: {line}"
            )


def make_spec(out_dtype: torch.dtype = torch.float16) -> KernelSpec:
    return KernelSpec(
        op_type="rmsnorm",
        input_specs=(
            TensorSpec(shape=(2048, 4096), dtype=torch.float16, is_contiguous=True),
            TensorSpec(shape=(4096,), dtype=torch.float16, is_contiguous=True),
        ),
        output_specs=(TensorSpec(shape=(2048, 4096), dtype=out_dtype, is_contiguous=True),),
        constants={"eps": 1e-6},
        graph_hash="rmsnorm_v1",
        constraints=(),
    )


def default_params(**kw: object) -> SearchParams:
    base = dict(block_size=4096, num_warps=8, num_stages=1, acc_dtype="fp32", variant="single_row")
    base.update(kw)
    return SearchParams(**base)  # type: ignore[arg-type]


class TestCodegen:
    def test_generates_valid_python(self) -> None:
        code = generate_rmsnorm(make_spec(), default_params())
        # 構文として valid であること（GPU なしで検証可能）
        compile(code, "<generated>", "exec")

    def test_contains_kernel_fn(self) -> None:
        code = generate_rmsnorm(make_spec(), default_params())
        assert "def kernel_fn(" in code
        assert "@triton.jit" in code

    def test_block_size_embedded(self) -> None:
        code = generate_rmsnorm(make_spec(), default_params(block_size=2048))
        assert "BLOCK_SIZE=2048" in code

    def test_num_warps_embedded(self) -> None:
        code = generate_rmsnorm(make_spec(), default_params(num_warps=16))
        assert "num_warps=16" in code

    def test_acc_dtype_fp32(self) -> None:
        code = generate_rmsnorm(make_spec(), default_params(acc_dtype="fp32"))
        assert "tl.float32" in code

    def test_acc_dtype_fp16(self) -> None:
        code = generate_rmsnorm(make_spec(), default_params(acc_dtype="fp16"))
        assert ".to(tl.float16)" in code

    def test_output_dtype_respected(self) -> None:
        code = generate_rmsnorm(make_spec(out_dtype=torch.bfloat16), default_params())
        assert "out.to(tl.bfloat16)" in code


class TestVariantCodegen:
    def test_multi_row_valid_python_and_grid(self) -> None:
        p = default_params(variant="multi_row", rows_per_program=4)
        code = generate_rmsnorm(make_spec(), p)
        compile(code, "<gen>", "exec")
        assert "ROWS=4" in code
        assert "triton.cdiv(M, 4)" in code
        assert "tl.static_range" in code

    def test_two_pass_valid_python_and_loops(self) -> None:
        p = default_params(variant="two_pass", block_size=1024)
        code = generate_rmsnorm(make_spec(), p)
        compile(code, "<gen>", "exec")
        assert "BLOCK_SIZE=1024" in code
        assert "for start in range(0, N, BLOCK_SIZE)" in code

    def test_two_pass_forces_fp32_reduction(self) -> None:
        # pass1 の縮約は acc_dtype に関わらず fp32 で安定化される
        code = generate_rmsnorm(make_spec(), default_params(variant="two_pass", acc_dtype="fp16"))
        assert "tl.zeros([BLOCK_SIZE], dtype=tl.float32)" in code

    def test_each_variant_uses_distinct_template(self) -> None:
        s = generate_rmsnorm(make_spec(), default_params(variant="single_row"))
        m = generate_rmsnorm(make_spec(), default_params(variant="multi_row", rows_per_program=2))
        t = generate_rmsnorm(make_spec(), default_params(variant="two_pass", block_size=1024))
        assert "variant=multi_row" in m
        assert "variant=two_pass" in t
        assert "ROWS" not in s and "ROWS" not in t

    def test_deterministic(self) -> None:
        a = generate_rmsnorm(make_spec(), default_params())
        b = generate_rmsnorm(make_spec(), default_params())
        assert a == b


def sdpa_spec(bh: int = 8, s: int = 64, d: int = 64, is_causal: bool = False) -> KernelSpec:
    return KernelSpec(
        op_type="scaled_dot_product_attention",
        input_specs=(
            TensorSpec(shape=(bh, s, d), dtype=torch.float16, is_contiguous=True),
            TensorSpec(shape=(bh, s, d), dtype=torch.float16, is_contiguous=True),
            TensorSpec(shape=(bh, s, d), dtype=torch.float16, is_contiguous=True),
        ),
        output_specs=(TensorSpec(shape=(bh, s, d), dtype=torch.float16, is_contiguous=True),),
        constants={"is_causal": is_causal},
        graph_hash="sdpa_v1",
        constraints=(),
    )


def flash_params(block_size: int = 64, num_warps: int = 4, num_stages: int = 1) -> SearchParams:
    return SearchParams(
        block_size=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
        acc_dtype="fp32",
        variant="flash",
    )


class TestSdpaCodegen:
    def test_non_causal_valid_python(self) -> None:
        code = generate(sdpa_spec(is_causal=False), flash_params())
        compile(code, "<gen>", "exec")
        assert "op=scaled_dot_product_attention variant=flash" in code
        assert "IS_CAUSAL=False" in code

    def test_causal_valid_python(self) -> None:
        code = generate(sdpa_spec(is_causal=True), flash_params())
        compile(code, "<gen>", "exec")
        assert "IS_CAUSAL=True" in code

    def test_head_dim_baked_in(self) -> None:
        code = generate(sdpa_spec(d=64), flash_params())
        assert "HEAD_DIM=64" in code

    def test_block_size_embedded(self) -> None:
        code = generate(sdpa_spec(), flash_params(block_size=128))
        assert "BLOCK_S=128" in code
        assert "cdiv(S, 128)" in code

    def test_contains_flash_attention_structure(self) -> None:
        code = generate(sdpa_spec(), flash_params())
        assert "tl.dot" in code
        assert "tl.exp" in code
        assert "tl.max" in code
        assert "def kernel_fn(q, k, v" in code

    def test_unknown_template_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="No codegen template"):
            generate(sdpa_spec(), default_params(variant="single_row"))

    def test_store_casts_use_tl_namespace(self) -> None:
        """全 SDPA テンプレート × 出力 dtype で、`.to()` の引数が `tl.<dtype>` であること。

        `compile()` では裸名 (`float16`) は通ってしまい、Triton の JIT 時に
        NameError になる (#460)。
        """
        import ast

        for variant in ("flash", "flash_causal_opt"):
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                spec = sdpa_spec(is_causal=True)
                spec = KernelSpec(
                    op_type=spec.op_type,
                    input_specs=spec.input_specs,
                    output_specs=(TensorSpec(shape=(8, 64, 64), dtype=dtype, is_contiguous=True),),
                    constants=spec.constants,
                    graph_hash=spec.graph_hash,
                    constraints=spec.constraints,
                )
                params = SearchParams(
                    block_size=64,
                    num_warps=4,
                    num_stages=1,
                    acc_dtype="fp32",
                    variant=variant,
                )
                tree = ast.parse(generate(spec, params))
                casts = [
                    n
                    for n in ast.walk(tree)
                    if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "to"
                    and n.args
                ]
                assert casts, (variant, dtype)
                for call in casts:
                    arg = call.args[0]
                    assert (
                        isinstance(arg, ast.Attribute)
                        and isinstance(arg.value, ast.Name)
                        and arg.value.id == "tl"
                    ), (variant, dtype, ast.unparse(call))

    def test_flash_variant_accepted_by_search_params(self) -> None:
        p = flash_params()
        assert p.variant == "flash"
        assert SearchParams.from_dict(p.to_dict()) == p


class TestSearchParams:
    def test_rejects_non_power_of_2_block_size(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="power of 2"):
            default_params(block_size=3000)

    def test_rejects_unknown_variant(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="variant"):
            default_params(variant="nonexistent")

    def test_roundtrip_dict(self) -> None:
        p = default_params()
        assert SearchParams.from_dict(p.to_dict()) == p
