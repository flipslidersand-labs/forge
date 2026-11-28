from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OpPattern:
    """既知の演算を torch.fx グラフの call_function 名の多重集合で識別する。

    MVP の素朴なパターンマッチ。標準的な式の形のみを認識し、それ以外は None を返して
    eager 実行にフォールバックする（任意グラフのコンパイラ化は範囲外 — 総評参照）。
    """

    op_type: str
    # 期待する call_function の __name__ → 個数（例: rmsnorm = {mul:3, mean:1, add:1, rsqrt:1}）
    op_counts: dict[str, int]
    # call_function 名 → 「このノードのリテラル引数はカーネルの前提と合うか」の述語（#472）。
    # False を返したノードは graph_op_counts で UNSUPPORTED_SUFFIX 付きの別名で数えられ、
    # どのパターンにも一致しなくなる（= op_type=None で eager へ落ちる）。
    node_guards: dict[str, Callable[[Any], bool]] = field(default_factory=dict)

    def matches(self, counts: Counter[str]) -> bool:
        return counts == Counter(self.op_counts)


_REGISTRY: dict[str, OpPattern] = {}
_NODE_GUARDS: dict[str, Callable[[Any], bool]] = {}

# 対応外リテラルを持つノードの名前に付ける印。どの op_counts にも現れないので不一致になる。
UNSUPPORTED_SUFFIX = "!literal"

_MISSING: Any = object()


def register(pattern: OpPattern) -> None:
    _REGISTRY[pattern.op_type] = pattern
    _NODE_GUARDS.update(pattern.node_guards)


def literal_arg(node: Any, name: str, index: int, default: Any = _MISSING) -> Any:
    """fx ノードから引数 `name`（位置 `index`）を取り出す。無ければ `default`。

    値が torch.fx.Node（= 関数の引数など実行時に決まる値）の場合はそのまま Node を返す。
    呼び出し側は Node なら「リテラルではない」として判定対象外にする。
    """
    if name in node.kwargs:
        return node.kwargs[name]
    if index < len(node.args):
        return node.args[index]
    return default


def is_dynamic(value: Any) -> bool:
    """値が fx Node（実行時に決まる値）か。"""
    from torch.fx import Node

    return isinstance(value, Node)


def match_counts(counts: Counter[str]) -> str | None:
    for pattern in _REGISTRY.values():
        if pattern.matches(counts):
            return pattern.op_type
    return None


def graph_op_counts(graph: Any) -> Counter[str]:
    """torch.fx Graph から call_function ノードの __name__ 多重集合を作る。

    リテラル引数が対応外のノード（node_guards が False）は名前に UNSUPPORTED_SUFFIX を付ける。
    """
    names: list[str] = []
    for node in graph.nodes:
        if node.op == "call_function":
            name = getattr(node.target, "__name__", str(node.target))
            guard = _NODE_GUARDS.get(name)
            if guard is not None and not guard(node):
                name += UNSUPPORTED_SUFFIX
            names.append(name)
    return Counter(names)


def _matcher(op_type: str) -> Callable[[Counter[str]], bool]:
    return _REGISTRY[op_type].matches
