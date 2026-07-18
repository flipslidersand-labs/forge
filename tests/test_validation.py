import torch

from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.validation.test_cases import correctness_cases, primary_input
from forge.validation.tolerance import get_tolerance


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


class TestSdpaValidation:
    def test_primary_input_three_tensors(self) -> None:
        inp = primary_input(sdpa_spec())
        assert len(inp) == 3
        assert all(s["shape"] == [8, 64, 64] for s in inp)
        assert all(s["init"] == "randn" for s in inp)

    def test_correctness_cases_has_four_cases(self) -> None:
        cases = correctness_cases(sdpa_spec())
        assert len(cases) == 4

    def test_correctness_cases_names(self) -> None:
        names = {c["name"] for c in correctness_cases(sdpa_spec())}
        assert {"basic", "single_batch", "multi_batch", "large_values"} == names

    def test_correctness_cases_all_have_three_inputs(self) -> None:
        for case in correctness_cases(sdpa_spec()):
            assert len(case["input_specs"]) == 3

    def test_correctness_cases_keep_seq_and_head_dim(self) -> None:
        cases = correctness_cases(sdpa_spec(s=64, d=64))
        for case in cases:
            for inp in case["input_specs"]:
                assert inp["shape"][1] == 64  # seq_len preserved
                assert inp["shape"][2] == 64  # head_dim preserved

    def test_sdpa_tolerance_defined(self) -> None:
        tol = get_tolerance("scaled_dot_product_attention")
        assert tol.atol >= 5e-3

    def test_sdpa_tolerance_relaxed_vs_rmsnorm(self) -> None:
        sdpa_tol = get_tolerance("scaled_dot_product_attention")
        rmsnorm_tol = get_tolerance("rmsnorm")
        assert sdpa_tol.atol >= rmsnorm_tol.atol


class TestTolerance:
    def test_rmsnorm_atol_relaxed(self) -> None:
        # #4 実測 1.95e-3 を通すため atol >= 2e-3
        assert get_tolerance("rmsnorm").atol >= 2e-3

    def test_to_dict(self) -> None:
        d = get_tolerance("rmsnorm").to_dict()
        assert set(d) == {"atol", "rtol", "equal_nan"}

    def test_unknown_op_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="No tolerance"):
            get_tolerance("nope")


class TestTestCases:
    def test_primary_input_matches_spec_shape(self) -> None:
        inp = primary_input(spec(4096))
        assert inp[0]["shape"] == [2048, 4096]
        assert inp[1]["shape"] == [4096]
        assert inp[0]["init"] == "randn"
        assert inp[1]["init"] == "ones"

    def test_correctness_cases_keep_hidden_constant(self) -> None:
        cases = correctness_cases(spec(4096))
        for c in cases:
            assert c["input_specs"][0]["shape"][1] == 4096  # N 固定
            assert c["input_specs"][1]["shape"] == [4096]

    def test_correctness_cases_cover_edge_categories(self) -> None:
        names = {c["name"] for c in correctness_cases(spec())}
        assert {"single_row", "zeros", "large_values", "weight_randn"} <= names

    def test_zeros_case_uses_zeros_init(self) -> None:
        cases = {c["name"]: c for c in correctness_cases(spec())}
        assert cases["zeros"]["input_specs"][0]["init"] == "zeros"
