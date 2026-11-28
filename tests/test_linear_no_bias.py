"""bias=None の linear 形関数(F.linear(x, w, bias=None))の入力数の整合。GPU 不要。

decorator は bias=None を constants に入れる(テンソルは x, weight の 2 つ)が、linear の
検証入力は常に 3 テンソル(bias の zeros 付き)だった。ワーカーは
kernel_fn(*tensors, **constants) を呼ぶため、bias が位置引数とキーワード引数で重複して
TypeError になっていた。
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import torch

from forge.decorator import optimize
from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.ops.registry import OP_REGISTRY
from forge.validation.test_cases import correctness_cases, primary_input


def _mock_cuda_tensor(shape: tuple[int, ...]) -> MagicMock:
    t = MagicMock(spec=torch.Tensor)
    t.is_cuda = True
    t.shape = shape
    t.dtype = torch.float16
    t.device = torch.device("cuda:0")
    t.is_contiguous.return_value = True
    return t


def _cpu_tensor(spec: dict[str, Any]) -> torch.Tensor:
    """ワーカーの _build_tensor の CPU 版(shape / dtype / init のみ再現)。"""
    shape = tuple(spec["shape"])
    dtype = getattr(torch, spec["dtype"])
    if spec["init"] == "zeros":
        return torch.zeros(shape, dtype=torch.float32).to(dtype)
    return (torch.randn(shape, dtype=torch.float32) * spec["scale"]).to(dtype)


def _kernel_fn_like_template(x, weight, bias=None):
    """codegen/templates/linear.py.jinja の kernel_fn と同じシグネチャ。"""
    return torch.nn.functional.linear(x.float(), weight.float(), bias).to(x.dtype)


def _spec(n_inputs: int, constants: dict[str, Any]) -> KernelSpec:
    shapes = [(8, 16), (32, 16), (32,)][:n_inputs]
    return KernelSpec(
        op_type="linear",
        input_specs=tuple(TensorSpec(s, torch.float16, True) for s in shapes),
        output_specs=(TensorSpec((8, 32), torch.float16, True),),
        constants=constants,
        graph_hash="h",
        constraints=(),
    )


def _capture_build_args(fn, *args: Any) -> tuple[list[Any], dict[str, Any]]:
    """decorator が _build_or_eager に渡す tensors / constants を取り出す。"""
    captured: dict[str, Any] = {}

    def fake_build(_fn, _op, tensors, constants, *_rest):
        captured["tensors"] = tensors
        captured["constants"] = constants
        return None  # eager 扱い

    with (
        patch("forge.decorator.identify", return_value="linear"),
        patch("forge.decorator._build_or_eager", side_effect=fake_build),
    ):
        optimize(budget=1)(fn)(*args)
    return captured["tensors"], captured["constants"]


def test_decorator_passes_two_tensors_and_none_bias_constant() -> None:
    """bias=None は tensors に入らず constants に入る(検証入力の数はこれに合わせる)。"""

    def f(x, w, bias=None):
        return "eager"

    x, w = _mock_cuda_tensor((8, 16)), _mock_cuda_tensor((32, 16))
    tensors, constants = _capture_build_args(f, x, w)
    assert len(tensors) == 2
    assert constants == {"bias": None}


def test_decorator_keeps_tensor_bias_as_tensor() -> None:
    def f(x, w, bias=None):
        return "eager"

    x, w, b = (_mock_cuda_tensor(s) for s in [(8, 16), (32, 16), (32,)])
    tensors, constants = _capture_build_args(f, x, w, b)
    assert len(tensors) == 3
    assert constants == {}


def test_worker_style_call_has_no_duplicate_bias() -> None:
    """decorator の出力 → 検証入力 → ワーカー相当の呼び出し(*tensors, **constants)が重複引数にならない。"""

    def f(x, w, bias=None):
        return "eager"

    x, w = _mock_cuda_tensor((8, 16)), _mock_cuda_tensor((32, 16))
    tensors, constants = _capture_build_args(f, x, w)
    spec = _spec(len(tensors), constants)

    defn = OP_REGISTRY["linear"]
    cases = [{"name": "primary", "input_specs": primary_input(spec)}, *correctness_cases(spec)]
    for case in cases:
        real = [_cpu_tensor(s) for s in case["input_specs"]]
        assert len(real) == 2, case["name"]
        out_c = _kernel_fn_like_template(*real, **constants)
        out_r = defn.reference_fn(*real, **constants)
        assert out_c.shape == out_r.shape


def test_primary_and_cases_follow_input_spec_count_without_bias() -> None:
    spec = _spec(2, {})
    assert len(primary_input(spec)) == 2
    cases = correctness_cases(spec)
    assert cases, "ケースが空"
    assert all(len(c["input_specs"]) == 2 for c in cases)
    assert "zeros_bias" not in {c["name"] for c in cases}


def test_primary_and_cases_unchanged_with_bias() -> None:
    spec = _spec(3, {})
    assert len(primary_input(spec)) == 3
    cases = correctness_cases(spec)
    assert [c["name"] for c in cases] == [
        "basic",
        "single_row",
        "odd_rows",
        "large_values",
        "zeros_bias",
    ]
    assert all(len(c["input_specs"]) == 3 for c in cases)


def test_non_none_constants_unaffected_for_other_ops() -> None:
    """linear 以外(rmsnorm の eps 等)の constants は変えない。"""

    def f(x, w, eps=1e-6):
        return "eager"

    captured: dict[str, Any] = {}

    def fake_build(_fn, _op, tensors, constants, *_rest):
        captured["constants"] = constants
        return None

    x, w = _mock_cuda_tensor((8, 16)), _mock_cuda_tensor((16,))
    with (
        patch("forge.decorator.identify", return_value="rmsnorm"),
        patch("forge.decorator._build_or_eager", side_effect=fake_build),
    ):
        optimize(budget=1)(f)(x, w)
    assert captured["constants"] == {"eps": 1e-6}


# --- #507: bias 以外の引数名 / テンソル数の不一致は eager に倒す ---


def _run_decorated(op: str, fn, *args: Any) -> tuple[Any, MagicMock]:
    build = MagicMock(return_value=None)
    with (
        patch("forge.decorator.identify", return_value=op),
        patch("forge.decorator._build_or_eager", build),
    ):
        out = optimize(budget=1)(fn)(*args)
    return out, build


def test_other_arg_name_none_falls_back_to_eager_without_build() -> None:
    """b=None 形は kernel_fn(bias=...) に対応づけられないので、候補生成せず eager。"""

    def f(x, w, b=None):
        return "eager"

    x, w = _mock_cuda_tensor((8, 16)), _mock_cuda_tensor((32, 16))
    out, build = _run_decorated("linear", f, x, w)
    assert out == "eager"
    build.assert_not_called()


def test_unknown_extra_constant_falls_back_to_eager_without_build() -> None:
    def f(x, w, bias=None, activation="relu"):
        return "eager"

    x, w = _mock_cuda_tensor((8, 16)), _mock_cuda_tensor((32, 16))
    out, build = _run_decorated("linear", f, x, w)
    assert out == "eager"
    build.assert_not_called()


def test_tensor_bias_with_extra_constant_falls_back_to_eager() -> None:
    def f(x, w, b, scale=2.0):
        return "eager"

    x, w, b = (_mock_cuda_tensor(s) for s in [(8, 16), (32, 16), (32,)])
    out, build = _run_decorated("linear", f, x, w, b)
    assert out == "eager"
    build.assert_not_called()


def test_non_none_bias_constant_falls_back_to_eager() -> None:
    def f(x, w, bias=0.0):
        return "eager"

    x, w = _mock_cuda_tensor((8, 16)), _mock_cuda_tensor((32, 16))
    out, build = _run_decorated("linear", f, x, w)
    assert out == "eager"
    build.assert_not_called()


def test_supported_linear_forms_still_build() -> None:
    def f2(x, w, bias=None):
        return "eager"

    def f3(x, w, b):
        return "eager"

    def f2_plain(x, w):
        return "eager"

    x, w, b = (_mock_cuda_tensor(s) for s in [(8, 16), (32, 16), (32,)])
    for fn, args in [(f2, (x, w)), (f2, (x, w, b)), (f3, (x, w, b)), (f2_plain, (x, w))]:
        _, build = _run_decorated("linear", fn, *args)
        build.assert_called_once()


def test_tensor_count_mismatch_falls_back_to_eager_for_other_ops() -> None:
    """OpDefinition.n_tensor_inputs と実テンソル数が違う呼び出しは候補生成の前に eager。"""

    def f(x, w, extra, eps=1e-6):
        return "eager"

    x, w, extra = (_mock_cuda_tensor((8, 16)) for _ in range(3))
    out, build = _run_decorated("rmsnorm", f, x, w, extra)
    assert out == "eager"
    build.assert_not_called()


def test_matching_tensor_count_still_builds_for_other_ops() -> None:
    def f(x, w, eps=1e-6):
        return "eager"

    x, w = _mock_cuda_tensor((8, 16)), _mock_cuda_tensor((16,))
    _, build = _run_decorated("rmsnorm", f, x, w)
    build.assert_called_once()
