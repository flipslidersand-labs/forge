"""op 定義の 3 者（OP_REGISTRY / lowering 登録 / codegen テンプレート）の整合性テスト（Issue #373）。

新 op の追加は (1) lowering/<op>.py の OpPattern 登録 (2) ops/registry.py の OP_REGISTRY
(3) codegen/triton_codegen.py の _TEMPLATES + jinja の同期編集が必要で、漏れても import エラーに
ならず静かに eager フォールバックや ValueError になる。ここで CI 上の不一致を検出する。
GPU 不要。
"""

from __future__ import annotations

import importlib
import pkgutil

import forge.lowering
from forge.codegen.triton_codegen import _TEMPLATE_DIR, _TEMPLATES
from forge.lowering.registry import _REGISTRY as LOWERING_REGISTRY
from forge.ops.registry import OP_REGISTRY

# 意図的に 3 者が一致しない op_type。理由を書いたうえで追加する。現状は空（不一致なし）。
LOWERING_EXEMPT: set[str] = set()  # OP_REGISTRY にあるが lowering パターンを持たない op
TEMPLATE_EXEMPT: set[str] = set()  # OP_REGISTRY にあるが codegen テンプレートを持たない op

TEMPLATE_OPS = {op for op, _variant in _TEMPLATES}


def test_op_registry_ops_have_lowering_pattern() -> None:
    missing = set(OP_REGISTRY) - set(LOWERING_REGISTRY) - LOWERING_EXEMPT
    assert not missing, (
        f"OP_REGISTRY にあるが lowering 未登録（identify() が None になる）: {missing}"
    )


def test_lowering_patterns_are_in_op_registry() -> None:
    extra = set(LOWERING_REGISTRY) - set(OP_REGISTRY)
    assert not extra, f"lowering 登録済みだが OP_REGISTRY に無い: {extra}"


def test_op_registry_ops_have_codegen_template() -> None:
    missing = set(OP_REGISTRY) - TEMPLATE_OPS - TEMPLATE_EXEMPT
    assert not missing, (
        f"OP_REGISTRY にあるが _TEMPLATES に無い（generate() が ValueError）: {missing}"
    )


def test_template_ops_are_in_op_registry() -> None:
    extra = TEMPLATE_OPS - set(OP_REGISTRY)
    assert not extra, f"_TEMPLATES にあるが OP_REGISTRY に無い: {extra}"


def test_templates_point_to_existing_files() -> None:
    missing = {name for name in _TEMPLATES.values() if not (_TEMPLATE_DIR / name).is_file()}
    assert not missing, f"_TEMPLATES が参照するテンプレートファイルが無い: {missing}"


def test_no_orphan_template_files() -> None:
    on_disk = {p.name for p in _TEMPLATE_DIR.glob("*.jinja")}
    orphans = on_disk - set(_TEMPLATES.values())
    assert not orphans, f"_TEMPLATES から参照されない jinja: {orphans}"


def test_lowering_modules_are_all_imported() -> None:
    """lowering パッケージ内の全モジュールが `import forge.lowering` で読み込まれる。"""
    import sys

    names = [m.name for m in pkgutil.iter_modules(forge.lowering.__path__)]
    assert names, "lowering モジュールが 1 つも見つからない"
    for name in names:
        full = f"forge.lowering.{name}"
        assert full in sys.modules, f"{full} が forge.lowering から import されていない"
        importlib.import_module(full)
