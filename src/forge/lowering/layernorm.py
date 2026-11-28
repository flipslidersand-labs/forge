from __future__ import annotations

from typing import Any

from .registry import OpPattern, is_dynamic, literal_arg, register


def _normalizes_last_dim_only(node: Any) -> bool:
    # reference は (x.shape[-1],) 固定。normalized_shape が 1 要素でないリテラル
    # （複数次元の正規化）は対応外。その 1 要素が実入力の最終次元と一致するかは
    # trace 時に入力形状が無いため判定できない（既知の限界）。
    shape = literal_arg(node, "normalized_shape", 1, None)
    if is_dynamic(shape):
        return True
    return isinstance(shape, (tuple, list)) and len(shape) == 1


# F.layer_norm(x, (N,), weight, bias, eps) を torch.fx で trace すると {layer_norm:1}。
LAYERNORM_PATTERN = OpPattern(
    op_type="layernorm",
    op_counts={"layer_norm": 1},
    node_guards={"layer_norm": _normalizes_last_dim_only},
)

register(LAYERNORM_PATTERN)
