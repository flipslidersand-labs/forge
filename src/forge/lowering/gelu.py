from __future__ import annotations

from typing import Any

from .registry import OpPattern, is_dynamic, literal_arg, register


def _is_exact(node: Any) -> bool:
    # reference / codegen は exact(erf)。approximate='tanh' のリテラルは対応外。
    approximate = literal_arg(node, "approximate", 1, "none")
    return is_dynamic(approximate) or approximate == "none"


# F.gelu(x) を torch.fx で trace すると {gelu:1}（tanh 近似も同じ target 名）。
# codegen は exact(erf) を生成するため、approximate='tanh' のリテラルを持つ関数は
# node_guards で検出して eager にフォールバックさせる。
GELU_PATTERN = OpPattern(
    op_type="gelu",
    op_counts={"gelu": 1},
    node_guards={"gelu": _is_exact},
)

register(GELU_PATTERN)
