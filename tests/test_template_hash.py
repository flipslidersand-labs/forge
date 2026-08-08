from __future__ import annotations

import pytest

from forge.codegen.triton_codegen import graph_hash_for, template_hash


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


def test_missing_template_raises(tmp_path) -> None:
    # テンプレート欠損は破損インストール — 静かにフォールバックせず即例外。
    with pytest.raises(FileNotFoundError):
        template_hash("rmsnorm", template_dir=tmp_path)


def test_length_prefix_framing_disambiguates(tmp_path) -> None:
    # rmsnorm は複数テンプレート。境界をまたぐ内容移動で同一ダイジェストに
    # ならないこと（長さプレフィックスの検証）。
    names = ["rmsnorm.py.jinja", "rmsnorm_multi_row.py.jinja", "rmsnorm_two_pass.py.jinja"]
    for n in names:
        (tmp_path / n).write_text("")
    (tmp_path / names[0]).write_text("AB")
    h1 = template_hash("rmsnorm", template_dir=tmp_path)
    (tmp_path / names[0]).write_text("A")
    (tmp_path / names[1]).write_text("B")
    h2 = template_hash("rmsnorm", template_dir=tmp_path)
    assert h1 != h2


def test_graph_hash_for_format() -> None:
    h = graph_hash_for("rmsnorm")
    assert h.startswith("rmsnorm_")
    assert h == f"rmsnorm_{template_hash('rmsnorm')}"
