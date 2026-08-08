from __future__ import annotations

from forge.codegen.triton_codegen import template_hash


def test_hash_is_stable() -> None:
    assert template_hash("rmsnorm") == template_hash("rmsnorm")


def test_hash_is_12_hex() -> None:
    h = template_hash("rmsnorm")
    assert len(h) == 12
    assert all(c in "0123456789abcdef" for c in h)


def test_different_ops_have_different_hashes() -> None:
    assert template_hash("rmsnorm") != template_hash("softmax")
    assert template_hash("gelu") != template_hash("layernorm")


def test_hash_invalidates_when_template_content_changes(tmp_path) -> None:
    # gelu は単一テンプレート (gelu.py.jinja)。内容を書き換えるとハッシュが変わる。
    (tmp_path / "gelu.py.jinja").write_text("# v1\n")
    h1 = template_hash("gelu", template_dir=tmp_path)
    (tmp_path / "gelu.py.jinja").write_text("# v2 — 別実装\n")
    h2 = template_hash("gelu", template_dir=tmp_path)
    assert h1 != h2


def test_missing_template_dir_is_deterministic(tmp_path) -> None:
    # ファイルが無くてもファイル名だけで決定的（例外を投げない）。
    assert template_hash("rmsnorm", template_dir=tmp_path) == template_hash(
        "rmsnorm", template_dir=tmp_path
    )
