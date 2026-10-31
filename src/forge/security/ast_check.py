"""LLM 生成 Triton コードの簡易 AST チェック。

subprocess 隔離の手前で明らかに危険なコードを拒否する多層防御。
import は許可リスト方式、呼び出し・属性アクセスは禁止リスト方式。
"""

from __future__ import annotations

import ast

ALLOWED_MODULES = frozenset({"triton", "torch", "math"})
FORBIDDEN_NAMES = frozenset(
    {"eval", "exec", "compile", "__import__", "open", "input", "breakpoint"}
)


def _module_allowed(name: str) -> bool:
    return name.split(".")[0] in ALLOWED_MODULES


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[str] = []

    def _add(self, node: ast.AST, msg: str) -> None:
        self.violations.append(f"line {getattr(node, 'lineno', '?')}: {msg}")

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if not _module_allowed(alias.name):
                self._add(node, f"禁止された import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level or not node.module or not _module_allowed(node.module):
            self._add(node, f"禁止された import: {'.' * node.level}{node.module or ''}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        # 呼び出しだけでなく別名束縛 (f = eval) などの参照も拒否する
        if node.id in FORBIDDEN_NAMES and isinstance(node.ctx, ast.Load):
            self._add(node, f"禁止された名前の参照: {node.id}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self._add(node, f"dunder 属性アクセス: {node.attr}")
        self.generic_visit(node)


def check_generated_code(source: str) -> list[str]:
    """違反メッセージのリストを返す。空なら通過。構文エラーも違反として返す。"""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"構文エラー: {e.msg} (line {e.lineno})"]
    v = _Visitor()
    v.visit(tree)
    return v.violations
