import pytest
import torch

from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.search.grid import GridSearch
from forge.search.params import SearchParams
from forge.search.random_search import RandomSearch
from forge.search.space import SearchSpace, _cc_to_int, _next_pow2


def spec(n: int = 4096) -> KernelSpec:
    return KernelSpec(
        op_type="rmsnorm",
        input_specs=(
            TensorSpec((2048, n), torch.float16, True),
            TensorSpec((n,), torch.float16, True),
        ),
        output_specs=(TensorSpec((2048, n), torch.float16, True),),
        constants={"eps": 1e-6},
        graph_hash="rmsnorm_v1",
        constraints=(),
    )


def single(variant: str, **kw) -> SearchSpace:
    return SearchSpace(variants=[variant], **kw)


def gelu_spec(n: int = 4096) -> KernelSpec:
    return KernelSpec(
        op_type="gelu",
        input_specs=(TensorSpec((2048, n), torch.float16, True),),
        output_specs=(TensorSpec((2048, n), torch.float16, True),),
        constants={},
        graph_hash="gelu_v1",
        constraints=(),
    )


class TestElementwiseSpace:
    def test_gelu_uses_elementwise_variant_only(self) -> None:
        params = list(SearchSpace().enumerate(gelu_spec(4096), "8.9"))
        assert params and {p.variant for p in params} == {"elementwise"}

    def test_gelu_block_not_tied_to_hidden(self) -> None:
        # elementwise は N=4096 でも小さい block(256 等)を許す
        params = list(SearchSpace().enumerate(gelu_spec(4096), "8.9"))
        assert any(p.block_size < 4096 for p in params)
        assert all(p.rows_per_program == 1 for p in params)


class TestHelpers:
    def test_cc_to_int(self) -> None:
        assert _cc_to_int("8.9") == 89
        assert _cc_to_int("6.1") == 61
        assert _cc_to_int("7.0") == 70

    def test_next_pow2(self) -> None:
        assert _next_pow2(4096) == 4096
        assert _next_pow2(4000) == 4096
        assert _next_pow2(5000) == 8192


class TestSearchSpaceBlocks:
    def test_single_row_block_at_least_hidden(self) -> None:
        params = list(single("single_row").enumerate(spec(4096), "8.9"))
        assert params and all(p.block_size >= 4096 for p in params)

    def test_two_pass_allows_smaller_tiles(self) -> None:
        params = list(single("two_pass").enumerate(spec(4096), "8.9"))
        # two_pass のタイルは N 以下
        assert params and all(p.block_size <= 4096 for p in params)
        assert any(p.block_size < 4096 for p in params)

    def test_multi_row_has_rows_gt_1(self) -> None:
        params = list(single("multi_row").enumerate(spec(4096), "8.9"))
        assert params and {p.rows_per_program for p in params} == {2, 4, 8, 16}

    def test_non_multi_row_keeps_rows_1(self) -> None:
        for v in ("single_row", "two_pass"):
            params = list(single(v).enumerate(spec(4096), "8.9"))
            assert all(p.rows_per_program == 1 for p in params)

    def test_no_valid_block_falls_back_to_next_pow2(self) -> None:
        params = list(single("single_row").enumerate(spec(6000), "8.9"))
        assert all(p.block_size == 8192 for p in params)


class TestSearchSpaceGpu:
    def test_pascal_restricts_num_stages(self) -> None:
        params = list(SearchSpace().enumerate(spec(4096), "6.1"))
        assert {p.num_stages for p in params} == {1}

    def test_volta_allows_pipelining(self) -> None:
        params = list(SearchSpace().enumerate(spec(4096), "8.9"))
        assert {p.num_stages for p in params} == {1, 2, 3}

    def test_no_duplicates(self) -> None:
        params = list(SearchSpace().enumerate(spec(4096), "8.9"))
        keys = [
            (p.variant, p.block_size, p.rows_per_program, p.num_warps, p.num_stages, p.acc_dtype)
            for p in params
        ]
        assert len(keys) == len(set(keys))

    def test_all_three_variants_present(self) -> None:
        params = list(SearchSpace().enumerate(spec(4096), "6.1"))
        assert {"single_row", "multi_row", "two_pass"} <= {p.variant for p in params}


def sdpa_spec(bh: int = 8, s: int = 64, d: int = 64, is_causal: bool = False) -> KernelSpec:
    return KernelSpec(
        op_type="scaled_dot_product_attention",
        input_specs=(
            TensorSpec((bh, s, d), torch.float16, True),
            TensorSpec((bh, s, d), torch.float16, True),
            TensorSpec((bh, s, d), torch.float16, True),
        ),
        output_specs=(TensorSpec((bh, s, d), torch.float16, True),),
        constants={"is_causal": is_causal},
        graph_hash="sdpa_v1",
        constraints=(),
    )


class TestAttentionSpace:
    def test_uses_flash_variants_only(self) -> None:
        params = list(SearchSpace().enumerate(sdpa_spec(), "8.9"))
        assert params
        assert {p.variant for p in params} <= {"flash", "flash_causal_opt"}
        assert "flash" in {p.variant for p in params}
        assert "flash_causal_opt" in {p.variant for p in params}

    def test_block_sizes_from_attention_seq_blocks(self) -> None:
        params = list(SearchSpace().enumerate(sdpa_spec(s=64), "8.9"))
        assert all(p.block_size <= 64 for p in params)

    def test_acc_dtype_is_fp32(self) -> None:
        params = list(SearchSpace().enumerate(sdpa_spec(), "8.9"))
        assert all(p.acc_dtype == "fp32" for p in params)

    def test_no_duplicates(self) -> None:
        params = list(SearchSpace().enumerate(sdpa_spec(), "8.9"))
        # variant も含めた完全なキーで重複チェック
        keys = [(p.variant, p.block_size, p.num_warps, p.num_stages) for p in params]
        assert len(keys) == len(set(keys))

    def test_pascal_restricts_stages(self) -> None:
        params = list(SearchSpace().enumerate(sdpa_spec(), "6.1"))
        assert {p.num_stages for p in params} == {1}

    def test_yields_results_for_common_head_dims(self) -> None:
        for d in (32, 64, 128):
            params = list(SearchSpace().enumerate(sdpa_spec(d=d), "8.9"))
            assert params, f"No params for head_dim={d}"


class TestGridSearch:
    def test_budget_caps_results(self) -> None:
        cands = GridSearch().generate(spec(4096), "8.9", budget=5)
        assert len(cands) == 5

    def test_single_row_pascal_count(self) -> None:
        # single_row, cc6.1: blocks{4096,8192} x warps{4,8,16,32} x stages{1} x acc{fp32,fp16} = 16
        cands = GridSearch(single("single_row")).generate(spec(4096), "6.1")
        assert len(cands) == 16

    def test_variants_expand_space(self) -> None:
        single_only = GridSearch(single("single_row")).generate(spec(4096), "6.1")
        all_variants = GridSearch().generate(spec(4096), "6.1")
        assert len(all_variants) > len(single_only)


class TestRandomSearch:
    def test_deterministic_with_seed(self) -> None:
        a = RandomSearch(seed=42).generate(spec(4096), "8.9", budget=10)
        b = RandomSearch(seed=42).generate(spec(4096), "8.9", budget=10)
        assert a == b

    def test_different_seed_differs(self) -> None:
        a = RandomSearch(seed=1).generate(spec(4096), "8.9", budget=10)
        b = RandomSearch(seed=2).generate(spec(4096), "8.9", budget=10)
        assert a != b

    def test_budget_caps(self) -> None:
        cands = RandomSearch().generate(spec(4096), "8.9", budget=7)
        assert len(cands) == 7

    def test_samples_are_valid_subset(self) -> None:
        full = set(SearchSpace().enumerate(spec(4096), "8.9"))
        sample = RandomSearch(seed=3).generate(spec(4096), "8.9", budget=15)
        assert set(sample) <= full


def _f16(shape: tuple[int, ...]) -> TensorSpec:
    return TensorSpec(shape, torch.float16, True)


def _make_spec(op_type: str, **kw: int) -> KernelSpec:
    """各 op_type 用の最小 KernelSpec を返す（GPU 不要）。"""
    if op_type in ("rmsnorm", "softmax", "layernorm", "gelu"):
        m, n = kw["m"], kw["n"]
        n_in = {"rmsnorm": 2, "layernorm": 3, "softmax": 1, "gelu": 1}[op_type]
        inputs = [_f16((m, n))] * n_in
        return KernelSpec(
            op_type=op_type, input_specs=tuple(inputs),
            output_specs=(_f16((m, n)),), constants={"eps": 1e-6},
            graph_hash=f"{op_type}_v1", constraints=(),
        )
    if op_type == "swiglu":
        m, n = kw["m"], kw["n"]
        return KernelSpec(
            op_type="swiglu",
            input_specs=(_f16((m, n)), _f16((m, n))),
            output_specs=(_f16((m, n)),), constants={},
            graph_hash="swiglu_v1", constraints=(),
        )
    if op_type == "rope":
        m, n = kw["m"], kw["n"]
        return KernelSpec(
            op_type="rope",
            input_specs=(_f16((m, n)), _f16((m, n)), _f16((m, n))),
            output_specs=(_f16((m, n)),), constants={},
            graph_hash="rope_v1", constraints=(),
        )
    if op_type == "fused_add_rmsnorm":
        m, n = kw["m"], kw["n"]
        return KernelSpec(
            op_type="fused_add_rmsnorm",
            input_specs=(_f16((m, n)), _f16((m, n)), _f16((n,))),
            output_specs=(_f16((m, n)),), constants={"eps": 1e-6},
            graph_hash="fused_add_rmsnorm_v1", constraints=(),
        )
    if op_type == "linear":
        m, k, n = kw["m"], kw["k"], kw["n"]
        return KernelSpec(
            op_type="linear",
            input_specs=(_f16((m, k)), _f16((n, k)), _f16((n,))),
            output_specs=(_f16((m, n)),), constants={},
            graph_hash="linear_v1", constraints=(),
        )
    if op_type == "scaled_dot_product_attention":
        bh, s, d = kw["bh"], kw["s"], kw["d"]
        return KernelSpec(
            op_type="scaled_dot_product_attention",
            input_specs=(_f16((bh, s, d)), _f16((bh, s, d)), _f16((bh, s, d))),
            output_specs=(_f16((bh, s, d)),), constants={},
            graph_hash="sdpa_v1", constraints=(),
        )
    raise ValueError(f"unknown op_type: {op_type}")


@pytest.mark.parametrize(
    "op_type,shape_kw",
    [
        ("rmsnorm", {"m": 16, "n": 4096}),
        ("softmax", {"m": 16, "n": 4096}),
        ("layernorm", {"m": 16, "n": 4096}),
        ("gelu", {"m": 16, "n": 4096}),
        ("swiglu", {"m": 16, "n": 4096}),
        ("rope", {"m": 16, "n": 128}),
        ("fused_add_rmsnorm", {"m": 16, "n": 4096}),
        ("linear", {"m": 16, "k": 4096, "n": 4096}),
        ("scaled_dot_product_attention", {"bh": 4, "s": 64, "d": 64}),
    ],
)
def test_enumerate_returns_candidates(op_type: str, shape_kw: dict) -> None:
    """全 op_type で SearchSpace.enumerate() が最低1件以上の候補を返すこと。"""
    spec_ = _make_spec(op_type, **shape_kw)
    candidates = list(SearchSpace().enumerate(spec_, "8.0"))
    assert len(candidates) >= 1, f"{op_type}: candidates={candidates}"
    assert all(isinstance(c, SearchParams) for c in candidates)


@pytest.mark.parametrize(
    "op_type,shape_kw",
    [
        ("rmsnorm", {"m": 16, "n": 4096}),
        ("gelu", {"m": 16, "n": 4096}),
        ("swiglu", {"m": 16, "n": 4096}),
        ("rope", {"m": 16, "n": 128}),
        ("linear", {"m": 16, "k": 4096, "n": 4096}),
        ("scaled_dot_product_attention", {"bh": 4, "s": 64, "d": 64}),
    ],
)
def test_pascal_cc_restricts_stages_to_one(op_type: str, shape_kw: dict) -> None:
    """cc=6.1 (Pascal) では num_stages=1 のみ列挙される。"""
    spec_ = _make_spec(op_type, **shape_kw)
    candidates = list(SearchSpace().enumerate(spec_, "6.1"))
    assert len(candidates) >= 1
    assert all(c.num_stages == 1 for c in candidates), (
        f"{op_type}: non-stage-1 found in Pascal candidates"
    )
