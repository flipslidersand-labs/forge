from __future__ import annotations

from typing import Any

from .registry import OpPattern, is_dynamic, literal_arg, register


def _dim_is_last(node: Any) -> bool:
    # reference は dim=-1 固定。リテラルの dim が -1 以外（正の dim は ndim が trace 時に
    # 不明なので判定不能）なら対応外。dim 省略・None も対応外。Node（引数経由）は対象外。
    dim = literal_arg(node, "dim", 1, None)
    return is_dynamic(dim) or dim == -1


# torch.softmax(x, dim=-1) を torch.fx で trace すると call_function は {softmax:1}。
SOFTMAX_PATTERN = OpPattern(
    op_type="softmax",
    op_counts={"softmax": 1},
    node_guards={"softmax": _dim_is_last},
)

register(SOFTMAX_PATTERN)
