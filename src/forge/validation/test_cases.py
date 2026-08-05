from __future__ import annotations

from typing import Any

from forge.ir.kernel_spec import KernelSpec


def _rmsnorm_inputs(
    m: int,
    n: int,
    dtype: str,
    x_init: str = "randn",
    x_scale: float = 1.0,
    w_init: str = "ones",
    seed: int = 0,
) -> list[dict[str, Any]]:
    return [
        {"shape": [m, n], "dtype": dtype, "init": x_init, "scale": x_scale, "seed": seed},
        {"shape": [n], "dtype": dtype, "init": w_init, "scale": 1.0, "seed": seed + 1},
    ]


def _softmax_inputs(
    m: int,
    n: int,
    dtype: str,
    x_init: str = "randn",
    x_scale: float = 1.0,
    seed: int = 0,
) -> list[dict[str, Any]]:
    # softmax / gelu は入力 x のみ
    return [{"shape": [m, n], "dtype": dtype, "init": x_init, "scale": x_scale, "seed": seed}]


def _layernorm_inputs(
    m: int,
    n: int,
    dtype: str,
    x_init: str = "randn",
    x_scale: float = 1.0,
    w_init: str = "ones",
    b_init: str = "zeros",
    seed: int = 0,
) -> list[dict[str, Any]]:
    # layernorm は x, weight, bias の 3 入力
    return [
        {"shape": [m, n], "dtype": dtype, "init": x_init, "scale": x_scale, "seed": seed},
        {"shape": [n], "dtype": dtype, "init": w_init, "scale": 1.0, "seed": seed + 1},
        {"shape": [n], "dtype": dtype, "init": b_init, "scale": 1.0, "seed": seed + 2},
    ]


def _sdpa_inputs(
    bh: int,
    s: int,
    d: int,
    dtype: str,
    scale: float = 1.0,
    seed: int = 0,
) -> list[dict[str, Any]]:
    # Q, K, V の 3 入力。shape は [B*H, S, D]。
    return [
        {"shape": [bh, s, d], "dtype": dtype, "init": "randn", "scale": scale, "seed": seed},
        {"shape": [bh, s, d], "dtype": dtype, "init": "randn", "scale": scale, "seed": seed + 1},
        {"shape": [bh, s, d], "dtype": dtype, "init": "randn", "scale": scale, "seed": seed + 2},
    ]


def _linear_inputs(
    m: int,
    k: int,
    n: int,
    dtype: str,
    x_init: str = "randn",
    x_scale: float = 0.1,
    w_init: str = "randn",
    b_init: str = "zeros",
    seed: int = 0,
) -> list[dict[str, Any]]:
    # linear は x[M,K], weight[N,K], bias[N] の 3 入力
    return [
        {"shape": [m, k], "dtype": dtype, "init": x_init, "scale": x_scale, "seed": seed},
        {"shape": [n, k], "dtype": dtype, "init": w_init, "scale": 0.1, "seed": seed + 1},
        {"shape": [n], "dtype": dtype, "init": b_init, "scale": 0.0, "seed": seed + 2},
    ]


def primary_input(spec: KernelSpec) -> list[dict[str, Any]]:
    """ベンチマークに使う代表入力（spec の shape そのまま）。"""
    x = spec.input_specs[0]
    dt = x.dtype_str()
    if spec.op_type == "scaled_dot_product_attention":
        bh, s, d = x.shape
        return _sdpa_inputs(bh, s, d, dt)
    if spec.op_type == "linear":
        m, k = x.shape
        n = spec.input_specs[1].shape[0]
        return _linear_inputs(m, k, n, dt)
    m, n = x.shape
    if spec.op_type == "rmsnorm":
        return _rmsnorm_inputs(m, n, dt)
    if spec.op_type in ("softmax", "gelu"):
        return _softmax_inputs(m, n, dt)
    if spec.op_type == "layernorm":
        return _layernorm_inputs(m, n, dt)
    raise ValueError(f"primary_input: unsupported op_type {spec.op_type!r}")


def correctness_cases(spec: KernelSpec) -> list[dict[str, Any]]:
    """正確性検証のエッジケース群。

    block_size は hidden_size(N) に対して調律されるため N は固定し、行数 M と
    数値分布を変える。各ケースは worker が tensor を組み立てる input_specs を持つ。
    """
    x = spec.input_specs[0]
    dt = x.dtype_str()

    if spec.op_type == "scaled_dot_product_attention":
        bh, s, d = x.shape
        return [
            {"name": "basic", "input_specs": _sdpa_inputs(bh, s, d, dt)},
            {"name": "single_batch", "input_specs": _sdpa_inputs(1, s, d, dt, seed=3)},
            {"name": "multi_batch", "input_specs": _sdpa_inputs(4, s, d, dt, seed=5)},
            # 大きい値：softmax の数値安定性（online max 更新）を検証
            {
                "name": "large_values",
                "input_specs": _sdpa_inputs(bh, s, d, dt, scale=10.0, seed=7),
            },
        ]

    _, n = x.shape

    if spec.op_type == "rmsnorm":
        return [
            {"name": "basic", "input_specs": _rmsnorm_inputs(2048, n, dt)},
            {"name": "single_row", "input_specs": _rmsnorm_inputs(1, n, dt)},
            {"name": "odd_rows", "input_specs": _rmsnorm_inputs(7, n, dt, seed=3)},
            {
                "name": "weight_randn",
                "input_specs": _rmsnorm_inputs(64, n, dt, w_init="randn", seed=5),
            },
            {
                "name": "large_values",
                "input_specs": _rmsnorm_inputs(64, n, dt, x_scale=100.0, seed=7),
            },
            {
                "name": "small_values",
                "input_specs": _rmsnorm_inputs(64, n, dt, x_scale=1e-3, seed=9),
            },
            {"name": "zeros", "input_specs": _rmsnorm_inputs(8, n, dt, x_init="zeros")},
        ]
    if spec.op_type == "softmax":
        return [
            {"name": "basic", "input_specs": _softmax_inputs(2048, n, dt)},
            {"name": "single_row", "input_specs": _softmax_inputs(1, n, dt)},
            {"name": "odd_rows", "input_specs": _softmax_inputs(7, n, dt, seed=3)},
            # 大きい値は exp のオーバーフロー耐性（safe softmax）を検証する
            {
                "name": "large_values",
                "input_specs": _softmax_inputs(64, n, dt, x_scale=50.0, seed=7),
            },
            {
                "name": "small_values",
                "input_specs": _softmax_inputs(64, n, dt, x_scale=1e-3, seed=9),
            },
            {"name": "zeros", "input_specs": _softmax_inputs(8, n, dt, x_init="zeros")},
        ]
    if spec.op_type == "layernorm":
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
    if spec.op_type == "gelu":
        return [
            {"name": "basic", "input_specs": _softmax_inputs(2048, n, dt)},
            {"name": "single_row", "input_specs": _softmax_inputs(1, n, dt)},
            {"name": "odd_rows", "input_specs": _softmax_inputs(7, n, dt, seed=3)},
            {
                "name": "large_values",
                "input_specs": _softmax_inputs(64, n, dt, x_scale=10.0, seed=7),
            },
            {
                "name": "negative",
                "input_specs": _softmax_inputs(64, n, dt, x_scale=-5.0, seed=9),
            },
            {"name": "zeros", "input_specs": _softmax_inputs(8, n, dt, x_init="zeros")},
        ]
    if spec.op_type == "linear":
        m, k = x.shape
        n_out = spec.input_specs[1].shape[0]
        return [
            {"name": "basic", "input_specs": _linear_inputs(m, k, n_out, dt)},
            {"name": "single_row", "input_specs": _linear_inputs(1, k, n_out, dt)},
            {"name": "odd_rows", "input_specs": _linear_inputs(7, k, n_out, dt, seed=3)},
            {"name": "large_values", "input_specs": _linear_inputs(64, k, n_out, dt, x_scale=10.0, seed=7)},  # noqa: E501
            {"name": "zeros_bias", "input_specs": _linear_inputs(64, k, n_out, dt, b_init="zeros", seed=9)},  # noqa: E501
        ]
    raise ValueError(f"correctness_cases: unsupported op_type {spec.op_type!r}")
