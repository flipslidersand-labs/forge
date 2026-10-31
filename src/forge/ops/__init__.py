from __future__ import annotations

import warnings
from dataclasses import dataclass

# 演算ごとのメタデータ。SearchSpace（block 制約）と validation（入力構成）が参照する。
#   reduction:   行ごとに last-dim を縮約。BLOCK は N 以上（single/multi_row）。出力は入力同形。
#   elementwise: 要素ごと。flat に numel をタイル分割するため BLOCK は N に縛られない。


@dataclass(frozen=True)
class OpInfo:
    """非推奨。OP_REGISTRY の kind / n_tensor_inputs を直接参照してください。"""

    kind: str  # "reduction" | "elementwise"
    n_tensor_inputs: int  # kernel_fn に渡す tensor 入力数


from forge.ops.registry import OP_REGISTRY, OpDefinition  # noqa: E402

# OP_INFO は非推奨。OP_REGISTRY から動的に生成する（後方互換）。
OP_INFO: dict[str, OpInfo] = {
    op: OpInfo(kind=defn.kind, n_tensor_inputs=defn.n_tensor_inputs)
    for op, defn in OP_REGISTRY.items()
}


def get_op_info(op_type: str) -> OpInfo:
    """非推奨。OP_REGISTRY[op_type].kind / n_tensor_inputs を直接参照してください。"""
    warnings.warn(
        "get_op_info() は非推奨です。OP_REGISTRY[op_type] を直接参照してください。",
        DeprecationWarning,
        stacklevel=2,
    )
    if op_type not in OP_REGISTRY:
        raise ValueError(f"Unknown op_type: {op_type!r}")
    d = OP_REGISTRY[op_type]
    return OpInfo(kind=d.kind, n_tensor_inputs=d.n_tensor_inputs)


def is_elementwise(op_type: str) -> bool:
    return OP_REGISTRY[op_type].kind == "elementwise"


def is_matmul(op_type: str) -> bool:
    return OP_REGISTRY[op_type].kind == "matmul"


def is_gemm(op_type: str) -> bool:
    return OP_REGISTRY[op_type].kind == "gemm"


__all__ = [
    "OpInfo",
    "OP_INFO",
    "get_op_info",
    "is_elementwise",
    "is_matmul",
    "is_gemm",
    "OP_REGISTRY",
    "OpDefinition",
]
