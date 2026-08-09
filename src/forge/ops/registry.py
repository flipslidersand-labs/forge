"""OpDefinition レジストリ — op ごとのメタデータを 1 箇所に集約する。

新 op を追加する際はこのファイルに `OpDefinition` を追加するだけでよい。
他のファイルは `OP_REGISTRY` から参照するため、追加漏れがランタイムエラーになる。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from forge.ir.kernel_spec import KernelSpec
from forge.validation.tolerance import Tolerance


@dataclass
class OpDefinition:
    """1つの op に関するすべてのメタデータ。"""

    op_type: str
    reference_fn: Callable[..., torch.Tensor]
    baseline_fn: Callable[..., torch.Tensor]
    baseline_display_name: str
    tolerance: Tolerance
    primary_input_fn: Callable[[KernelSpec], list[dict[str, Any]]]
    correctness_cases_fn: Callable[[KernelSpec], list[dict[str, Any]]]
    render_extra_kwargs_fn: Callable[[KernelSpec], dict[str, Any]] = field(
        default_factory=lambda: lambda _spec: {}
    )


# --- reference / baseline 関数 ---


def _rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.float()
    rms = torch.rsqrt(torch.mean(x32 * x32, dim=-1, keepdim=True) + eps)
    return (x32 * rms * weight.float()).to(x.dtype)


def _softmax_reference(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.softmax(x.float(), dim=dim).to(x.dtype)


def _layernorm_reference(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    n = x.shape[-1]
    return torch.nn.functional.layer_norm(x.float(), (n,), weight.float(), bias.float(), eps).to(
        x.dtype
    )


def _swiglu_reference(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return (torch.nn.functional.silu(gate.float()) * x.float()).to(x.dtype)


def _gelu_reference(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x.float()).to(x.dtype)


def _sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    return torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), scale=scale, is_causal=is_causal
    ).to(q.dtype)


def _linear_reference(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    b = bias.float() if bias is not None else None
    return torch.nn.functional.linear(x.float(), weight.float(), b).to(x.dtype)


def _rmsnorm_baseline(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if hasattr(torch.nn.functional, "rms_norm"):
        return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, eps)  # type: ignore[attr-defined]  # PyTorch 2.4 追加; 旧スタブに定義なし
    x32 = x.float()
    rms = torch.rsqrt(torch.mean(x32 * x32, dim=-1, keepdim=True) + eps)
    return (x32 * rms * weight.float()).to(x.dtype)


def _softmax_baseline(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.nn.functional.softmax(x, dim=dim)


def _layernorm_baseline(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps)


def _swiglu_baseline(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(gate) * x


def _gelu_baseline(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x)


def _sdpa_baseline(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, scale=scale, is_causal=is_causal
    )


def _linear_baseline(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    return torch.nn.functional.linear(x, weight, bias)


# --- primary_input / correctness_cases ヘルパ ---


def _rmsnorm_inputs(m: int, n: int, dtype: str, **kw: Any) -> list[dict[str, Any]]:
    seed = kw.get("seed", 0)
    return [
        {
            "shape": [m, n],
            "dtype": dtype,
            "init": kw.get("x_init", "randn"),
            "scale": kw.get("x_scale", 1.0),
            "seed": seed,
        },
        {
            "shape": [n],
            "dtype": dtype,
            "init": kw.get("w_init", "ones"),
            "scale": 1.0,
            "seed": seed + 1,
        },
    ]


def _softmax_inputs(m: int, n: int, dtype: str, **kw: Any) -> list[dict[str, Any]]:
    seed = kw.get("seed", 0)
    return [
        {
            "shape": [m, n],
            "dtype": dtype,
            "init": kw.get("x_init", "randn"),
            "scale": kw.get("x_scale", 1.0),
            "seed": seed,
        }
    ]


def _swiglu_inputs(m: int, n: int, dtype: str, **kw: Any) -> list[dict[str, Any]]:
    seed = kw.get("seed", 0)
    return [
        {
            "shape": [m, n],
            "dtype": dtype,
            "init": kw.get("x_init", "randn"),
            "scale": kw.get("x_scale", 1.0),
            "seed": seed,
        },
        {
            "shape": [m, n],
            "dtype": dtype,
            "init": kw.get("g_init", "randn"),
            "scale": kw.get("g_scale", 1.0),
            "seed": seed + 1,
        },
    ]


def _gelu_inputs(m: int, n: int, dtype: str, **kw: Any) -> list[dict[str, Any]]:
    seed = kw.get("seed", 0)
    return [
        {
            "shape": [m, n],
            "dtype": dtype,
            "init": kw.get("x_init", "randn"),
            "scale": kw.get("x_scale", 1.0),
            "seed": seed,
        }
    ]


def _layernorm_inputs(m: int, n: int, dtype: str, **kw: Any) -> list[dict[str, Any]]:
    seed = kw.get("seed", 0)
    return [
        {
            "shape": [m, n],
            "dtype": dtype,
            "init": kw.get("x_init", "randn"),
            "scale": kw.get("x_scale", 1.0),
            "seed": seed,
        },
        {
            "shape": [n],
            "dtype": dtype,
            "init": kw.get("w_init", "ones"),
            "scale": 1.0,
            "seed": seed + 1,
        },
        {
            "shape": [n],
            "dtype": dtype,
            "init": kw.get("b_init", "zeros"),
            "scale": 1.0,
            "seed": seed + 2,
        },
    ]


def _sdpa_inputs(bh: int, s: int, d: int, dtype: str, **kw: Any) -> list[dict[str, Any]]:
    seed = kw.get("seed", 0)
    scale = kw.get("scale", 1.0)
    return [
        {"shape": [bh, s, d], "dtype": dtype, "init": "randn", "scale": scale, "seed": seed},
        {"shape": [bh, s, d], "dtype": dtype, "init": "randn", "scale": scale, "seed": seed + 1},
        {"shape": [bh, s, d], "dtype": dtype, "init": "randn", "scale": scale, "seed": seed + 2},
    ]


def _linear_inputs(m: int, k: int, n: int, dtype: str, **kw: Any) -> list[dict[str, Any]]:
    seed = kw.get("seed", 0)
    return [
        {
            "shape": [m, k],
            "dtype": dtype,
            "init": kw.get("x_init", "randn"),
            "scale": kw.get("x_scale", 0.1),
            "seed": seed,
        },
        {
            "shape": [n, k],
            "dtype": dtype,
            "init": kw.get("w_init", "randn"),
            "scale": 0.1,
            "seed": seed + 1,
        },
        {
            "shape": [n],
            "dtype": dtype,
            "init": kw.get("b_init", "zeros"),
            "scale": 0.0,
            "seed": seed + 2,
        },
    ]


# --- primary_input_fn per op ---


def _rmsnorm_primary(spec: KernelSpec) -> list[dict[str, Any]]:
    x = spec.input_specs[0]
    m, n = x.shape
    return _rmsnorm_inputs(m, n, x.dtype_str())


def _softmax_primary(spec: KernelSpec) -> list[dict[str, Any]]:
    x = spec.input_specs[0]
    m, n = x.shape
    return _softmax_inputs(m, n, x.dtype_str())


def _swiglu_primary(spec: KernelSpec) -> list[dict[str, Any]]:
    x = spec.input_specs[0]
    m, n = x.shape
    return _swiglu_inputs(m, n, x.dtype_str())


def _gelu_primary(spec: KernelSpec) -> list[dict[str, Any]]:
    x = spec.input_specs[0]
    m, n = x.shape
    return _gelu_inputs(m, n, x.dtype_str())


def _layernorm_primary(spec: KernelSpec) -> list[dict[str, Any]]:
    x = spec.input_specs[0]
    m, n = x.shape
    return _layernorm_inputs(m, n, x.dtype_str())


def _sdpa_primary(spec: KernelSpec) -> list[dict[str, Any]]:
    x = spec.input_specs[0]
    bh, s, d = x.shape
    return _sdpa_inputs(bh, s, d, x.dtype_str())


def _linear_primary(spec: KernelSpec) -> list[dict[str, Any]]:
    x = spec.input_specs[0]
    m, k = x.shape
    n = spec.input_specs[1].shape[0]
    return _linear_inputs(m, k, n, x.dtype_str())


# --- correctness_cases_fn per op ---


def _rmsnorm_cases(spec: KernelSpec) -> list[dict[str, Any]]:
    x = spec.input_specs[0]
    _, n = x.shape
    dt = x.dtype_str()
    return [
        {"name": "basic", "input_specs": _rmsnorm_inputs(2048, n, dt)},
        {"name": "single_row", "input_specs": _rmsnorm_inputs(1, n, dt)},
        {"name": "odd_rows", "input_specs": _rmsnorm_inputs(7, n, dt, seed=3)},
        {"name": "weight_randn", "input_specs": _rmsnorm_inputs(64, n, dt, w_init="randn", seed=5)},
        {"name": "large_values", "input_specs": _rmsnorm_inputs(64, n, dt, x_scale=100.0, seed=7)},
        {"name": "small_values", "input_specs": _rmsnorm_inputs(64, n, dt, x_scale=1e-3, seed=9)},
        {"name": "zeros", "input_specs": _rmsnorm_inputs(8, n, dt, x_init="zeros")},
    ]


def _softmax_cases(spec: KernelSpec) -> list[dict[str, Any]]:
    x = spec.input_specs[0]
    _, n = x.shape
    dt = x.dtype_str()
    return [
        {"name": "basic", "input_specs": _softmax_inputs(2048, n, dt)},
        {"name": "single_row", "input_specs": _softmax_inputs(1, n, dt)},
        {"name": "odd_rows", "input_specs": _softmax_inputs(7, n, dt, seed=3)},
        {"name": "large_values", "input_specs": _softmax_inputs(64, n, dt, x_scale=50.0, seed=7)},
        {"name": "small_values", "input_specs": _softmax_inputs(64, n, dt, x_scale=1e-3, seed=9)},
        {"name": "zeros", "input_specs": _softmax_inputs(8, n, dt, x_init="zeros")},
    ]


def _layernorm_cases(spec: KernelSpec) -> list[dict[str, Any]]:
    x = spec.input_specs[0]
    _, n = x.shape
    dt = x.dtype_str()
    return [
        {"name": "basic", "input_specs": _layernorm_inputs(2048, n, dt)},
        {"name": "single_row", "input_specs": _layernorm_inputs(1, n, dt)},
        {"name": "odd_rows", "input_specs": _layernorm_inputs(7, n, dt, seed=3)},
        {
            "name": "wb_randn",
            "input_specs": _layernorm_inputs(64, n, dt, w_init="randn", b_init="randn", seed=5),
        },
        {
            "name": "large_values",
            "input_specs": _layernorm_inputs(64, n, dt, x_scale=100.0, seed=7),
        },
        {"name": "zeros", "input_specs": _layernorm_inputs(8, n, dt, x_init="zeros")},
    ]


def _swiglu_cases(spec: KernelSpec) -> list[dict[str, Any]]:
    x = spec.input_specs[0]
    _, n = x.shape
    dt = x.dtype_str()
    return [
        {"name": "basic", "input_specs": _swiglu_inputs(2048, n, dt)},
        {"name": "single_row", "input_specs": _swiglu_inputs(1, n, dt)},
        {"name": "odd_rows", "input_specs": _swiglu_inputs(7, n, dt, seed=3)},
        {"name": "large_values", "input_specs": _swiglu_inputs(64, n, dt, x_scale=10.0, seed=7)},
        {"name": "zeros", "input_specs": _swiglu_inputs(8, n, dt, x_init="zeros", g_init="zeros")},
    ]


def _gelu_cases(spec: KernelSpec) -> list[dict[str, Any]]:
    x = spec.input_specs[0]
    _, n = x.shape
    dt = x.dtype_str()
    return [
        {"name": "basic", "input_specs": _gelu_inputs(2048, n, dt)},
        {"name": "single_row", "input_specs": _gelu_inputs(1, n, dt)},
        {"name": "odd_rows", "input_specs": _gelu_inputs(7, n, dt, seed=3)},
        {"name": "large_values", "input_specs": _gelu_inputs(64, n, dt, x_scale=10.0, seed=7)},
        {"name": "negative", "input_specs": _gelu_inputs(64, n, dt, x_scale=-5.0, seed=9)},
        {"name": "zeros", "input_specs": _gelu_inputs(8, n, dt, x_init="zeros")},
    ]


def _sdpa_cases(spec: KernelSpec) -> list[dict[str, Any]]:
    x = spec.input_specs[0]
    bh, s, d = x.shape
    dt = x.dtype_str()
    return [
        {"name": "basic", "input_specs": _sdpa_inputs(bh, s, d, dt)},
        {"name": "single_batch", "input_specs": _sdpa_inputs(1, s, d, dt, seed=3)},
        {"name": "multi_batch", "input_specs": _sdpa_inputs(4, s, d, dt, seed=5)},
        {"name": "large_values", "input_specs": _sdpa_inputs(bh, s, d, dt, scale=10.0, seed=7)},
    ]


def _linear_cases(spec: KernelSpec) -> list[dict[str, Any]]:
    x = spec.input_specs[0]
    m, k = x.shape
    n_out = spec.input_specs[1].shape[0]
    dt = x.dtype_str()
    return [
        {"name": "basic", "input_specs": _linear_inputs(m, k, n_out, dt)},
        {"name": "single_row", "input_specs": _linear_inputs(1, k, n_out, dt)},
        {"name": "odd_rows", "input_specs": _linear_inputs(7, k, n_out, dt, seed=3)},
        {
            "name": "large_values",
            "input_specs": _linear_inputs(64, k, n_out, dt, x_scale=10.0, seed=7),
        },
        {
            "name": "zeros_bias",
            "input_specs": _linear_inputs(64, k, n_out, dt, b_init="zeros", seed=9),
        },
    ]


# --- render_extra_kwargs_fn per op ---


def _sdpa_render_kwargs(spec: KernelSpec) -> dict[str, Any]:
    return {
        "head_dim": spec.input_specs[0].shape[-1],
        "is_causal": bool(spec.constants.get("is_causal", False)),
    }


def _linear_render_kwargs(spec: KernelSpec) -> dict[str, Any]:
    return {"block_k": int(str(spec.constants.get("block_k", 32)))}


# --- baseline_display_name helpers ---


def _rmsnorm_baseline_name() -> str:
    if hasattr(torch.nn.functional, "rms_norm"):
        return "F.rms_norm"
    return "fp32_reference"


# --- OP_REGISTRY ---

_TOL = Tolerance  # alias for brevity

OP_REGISTRY: dict[str, OpDefinition] = {
    "rmsnorm": OpDefinition(
        op_type="rmsnorm",
        reference_fn=_rmsnorm_reference,
        baseline_fn=_rmsnorm_baseline,
        baseline_display_name=_rmsnorm_baseline_name(),
        tolerance=_TOL(atol=2e-3, rtol=1e-2),
        primary_input_fn=_rmsnorm_primary,
        correctness_cases_fn=_rmsnorm_cases,
    ),
    "softmax": OpDefinition(
        op_type="softmax",
        reference_fn=_softmax_reference,
        baseline_fn=_softmax_baseline,
        baseline_display_name="F.softmax",
        tolerance=_TOL(atol=2e-3, rtol=1e-2),
        primary_input_fn=_softmax_primary,
        correctness_cases_fn=_softmax_cases,
    ),
    "layernorm": OpDefinition(
        op_type="layernorm",
        reference_fn=_layernorm_reference,
        baseline_fn=_layernorm_baseline,
        baseline_display_name="F.layer_norm",
        tolerance=_TOL(atol=2e-3, rtol=1e-2),
        primary_input_fn=_layernorm_primary,
        correctness_cases_fn=_layernorm_cases,
    ),
    "gelu": OpDefinition(
        op_type="gelu",
        reference_fn=_gelu_reference,
        baseline_fn=_gelu_baseline,
        baseline_display_name="F.gelu",
        tolerance=_TOL(atol=2e-3, rtol=1e-2),
        primary_input_fn=_gelu_primary,
        correctness_cases_fn=_gelu_cases,
    ),
    "swiglu": OpDefinition(
        op_type="swiglu",
        reference_fn=_swiglu_reference,
        baseline_fn=_swiglu_baseline,
        baseline_display_name="F.silu(gate) * x",
        tolerance=_TOL(atol=2e-3, rtol=1e-2),
        primary_input_fn=_swiglu_primary,
        correctness_cases_fn=_swiglu_cases,
    ),
    "scaled_dot_product_attention": OpDefinition(
        op_type="scaled_dot_product_attention",
        reference_fn=_sdpa_reference,
        baseline_fn=_sdpa_baseline,
        baseline_display_name="F.scaled_dot_product_attention",
        tolerance=_TOL(atol=5e-3, rtol=1e-2),
        primary_input_fn=_sdpa_primary,
        correctness_cases_fn=_sdpa_cases,
        render_extra_kwargs_fn=_sdpa_render_kwargs,
    ),
    "linear": OpDefinition(
        op_type="linear",
        reference_fn=_linear_reference,
        baseline_fn=_linear_baseline,
        baseline_display_name="F.linear",
        tolerance=_TOL(atol=1e-2, rtol=1e-2),
        primary_input_fn=_linear_primary,
        correctness_cases_fn=_linear_cases,
        render_extra_kwargs_fn=_linear_render_kwargs,
    ),
}
