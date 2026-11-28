"""LLM 生成 Triton コードの簡易 AST チェック。

subprocess 隔離の手前で明らかに危険なコードを拒否する多層防御。
import は許可リスト方式、呼び出し・属性アクセス・import するシンボルは禁止リスト方式。

注意: この AST チェックはサンドボックスではなく隔離ではない。禁止リストは既知の
バイパス (許可モジュール内側の os 等への経路、getattr / globals 等) を塞ぐだけで、
未知の経路を網羅できない。隔離の本線は subprocess ワーカー (docs/adr/ADR-003)。
"""

from __future__ import annotations

import ast

ALLOWED_MODULES = frozenset({"triton", "torch", "math"})
FORBIDDEN_NAMES = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "breakpoint",
        # 属性名を文字列で組み立てて禁止検査を迂回する / 名前空間を直接触る組み込み
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
    }
)
# 許可モジュールの内側 (torch.os など) から抜けられる標準ライブラリの危険モジュール名。
# torch 自身のサブモジュール (torch.hub / torch.multiprocessing など) は含めない
# (許可範囲を変えない。#344 で決める)。
# `from torch import os`、`import torch.os`、`torch.os.system(...)` のように
# import するシンボル・モジュール path・属性のどこに現れても拒否する。
FORBIDDEN_MODULE_ATTRS = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "builtins",
        "importlib",
        "ctypes",
        "shutil",
        "socket",
        "pickle",
        "marshal",
        "runpy",
        "inspect",
        "threading",
        "pathlib",
        "io",
        "gc",
        "posix",
        "nt",
    }
)


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _module_allowed(name: str) -> bool:
    return name.split(".")[0] in ALLOWED_MODULES


def _forbidden_part(dotted: str) -> str | None:
    """dotted path の中に危険モジュール名 / dunder があればその部分を返す。"""
    for part in dotted.split("."):
        if part in FORBIDDEN_MODULE_ATTRS or _is_dunder(part):
            return part
    return None


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[str] = []

    def _add(self, node: ast.AST, msg: str) -> None:
        self.violations.append(f"line {getattr(node, 'lineno', '?')}: {msg}")

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if not _module_allowed(alias.name):
                self._add(node, f"禁止された import: {alias.name}")
            elif (part := _forbidden_part(alias.name)) is not None:
                self._add(node, f"禁止されたモジュール経路の import: {alias.name} ({part})")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level or not node.module or not _module_allowed(node.module):
            self._add(node, f"禁止された import: {'.' * node.level}{node.module or ''}")
        else:
            if (part := _forbidden_part(node.module)) is not None:
                self._add(node, f"禁止されたモジュール経路の import: {node.module} ({part})")
            for alias in node.names:
                if alias.name == "*":
                    self._add(node, f"import * は禁止: from {node.module} import *")
                elif (part := _forbidden_part(alias.name)) is not None:
                    self._add(node, f"禁止されたシンボルの import: {node.module}.{alias.name}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        # 呼び出しだけでなく別名束縛 (f = eval) などの参照も拒否する
        if node.id in FORBIDDEN_NAMES and isinstance(node.ctx, ast.Load):
            self._add(node, f"禁止された名前の参照: {node.id}")
        elif node.id in ("__builtins__", "__loader__", "__spec__"):
            self._add(node, f"dunder 名の参照: {node.id}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if _is_dunder(node.attr):
            self._add(node, f"dunder 属性アクセス: {node.attr}")
        elif node.attr in FORBIDDEN_MODULE_ATTRS:
            self._add(node, f"禁止された属性アクセス: {node.attr}")
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
