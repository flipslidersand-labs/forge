"""loader.py のユニットテスト。Triton 不要。"""

from __future__ import annotations

import tempfile
from pathlib import Path

_DUMMY_MODULE = """\
def kernel_fn(*args, **kwargs):
    return None
"""

_RAISE_MODULE = """\
def kernel_fn(*args, **kwargs):
    return None

raise RuntimeError("module load error")
"""


def test_file_is_created_and_importable():
    """load_kernel_fn が kernel_fn を返す。"""
    from forge.runtime import loader

    fn = loader.load_kernel_fn(_DUMMY_MODULE)
    assert callable(fn)


def test_tmp_file_deleted_after_load():
    """load_kernel_fn が返った後、一時ファイルが削除されている。"""
    from forge.runtime import loader

    tmp_dir = Path(tempfile.gettempdir()) / "forge_kernels"
    files_before = set(tmp_dir.glob("kernel_*.py")) if tmp_dir.exists() else set()

    loader.load_kernel_fn(_DUMMY_MODULE)

    files_after = set(tmp_dir.glob("kernel_*.py")) if tmp_dir.exists() else set()
    new_files = files_after - files_before
    assert new_files == set(), f"一時ファイルが残っている: {new_files}"


def test_tmp_file_deleted_on_exec_error():
    """exec_module が例外を送出しても一時ファイルが削除される。"""
    import pytest

    from forge.runtime import loader

    tmp_dir = Path(tempfile.gettempdir()) / "forge_kernels"
    files_before = set(tmp_dir.glob("kernel_*.py")) if tmp_dir.exists() else set()

    with pytest.raises(RuntimeError, match="module load error"):
        loader.load_kernel_fn(_RAISE_MODULE)

    files_after = set(tmp_dir.glob("kernel_*.py")) if tmp_dir.exists() else set()
    new_files = files_after - files_before
    assert new_files == set(), f"エラー後も一時ファイルが残っている: {new_files}"


def test_returns_callable():
    """load_kernel_fn の戻り値が callable。"""
    from forge.runtime import loader

    fn = loader.load_kernel_fn(_DUMMY_MODULE)
    assert callable(fn)
    assert fn() is None


# --- HMAC sign / verify tests (Issue #269) ---


def test_sign_and_verify_roundtrip():
    """sign_kernel_code → verify_kernel_code がコードを完全復元する。"""
    from forge.runtime.loader import sign_kernel_code, verify_kernel_code

    original = _DUMMY_MODULE
    signed = sign_kernel_code(original)
    assert verify_kernel_code(signed) == original


def test_sign_adds_hmac_prefix():
    """署名済み文字列には HMAC タグが含まれる。"""
    from forge.runtime.loader import _SEP, sign_kernel_code

    signed = sign_kernel_code(_DUMMY_MODULE)
    assert _SEP in signed


def test_verify_raises_on_tampered_code():
    """コード本文を改ざんすると ValueError が発生する。"""
    import pytest

    from forge.runtime.loader import _SEP, sign_kernel_code, verify_kernel_code

    signed = sign_kernel_code(_DUMMY_MODULE)
    tag, _, code = signed.partition(_SEP)
    tampered = f"{tag}{_SEP}{code}import os; os.system('id')"
    with pytest.raises(ValueError, match="HMAC 検証に失敗"):
        verify_kernel_code(tampered)


def test_verify_raises_on_missing_tag():
    """タグなし文字列を渡すと ValueError が発生する。"""
    import pytest

    from forge.runtime.loader import verify_kernel_code

    with pytest.raises(ValueError, match="HMAC タグが見つかりません"):
        verify_kernel_code(_DUMMY_MODULE)


def test_verify_raises_on_wrong_tag():
    """タグが正しくない（別のコードのタグ）場合も ValueError が発生する。"""
    import pytest

    from forge.runtime.loader import _SEP, sign_kernel_code, verify_kernel_code

    signed_a = sign_kernel_code("code_a")
    signed_b = sign_kernel_code("code_b")
    tag_a, _, _ = signed_a.partition(_SEP)
    _, _, code_b = signed_b.partition(_SEP)
    cross = f"{tag_a}{_SEP}{code_b}"
    with pytest.raises(ValueError, match="HMAC 検証に失敗"):
        verify_kernel_code(cross)


# --- HMAC 鍵の永続化（#327） ---


def _reset_key(monkeypatch, key_path: Path) -> None:
    from forge.runtime import loader

    monkeypatch.setenv("FORGE_HMAC_KEY_PATH", str(key_path))
    monkeypatch.setattr(loader, "_HMAC_KEY", None, raising=False)


def test_hmac_key_shared_across_processes(tmp_path, monkeypatch):
    """別プロセスで署名したコードを、同じ鍵ファイルを使う親プロセスで検証できる。"""
    import os
    import subprocess
    import sys

    from forge.runtime import loader

    key_path = tmp_path / "sub" / "hmac.key"
    env = {**os.environ, "FORGE_HMAC_KEY_PATH": str(key_path)}
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "from forge.runtime.loader import sign_kernel_code;"
            "print(sign_kernel_code('def k(): pass'), end='')",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    _reset_key(monkeypatch, key_path)
    assert loader.verify_kernel_code(out) == "def k(): pass"


def test_hmac_key_file_mode_is_0600(tmp_path, monkeypatch):
    """鍵ファイルは 0600、親ディレクトリは 0700 で作られる。"""
    import stat

    from forge.runtime import loader

    key_path = tmp_path / "newdir" / "hmac.key"
    _reset_key(monkeypatch, key_path)
    loader.sign_kernel_code("x")
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(key_path.parent.stat().st_mode) == 0o700
    assert len(key_path.read_bytes()) == 32


def test_hmac_key_invalid_file_falls_back(tmp_path, monkeypatch, caplog):
    """長さ不正の鍵ファイルでもクラッシュせず、メモリ上の鍵で動作し、ファイルは上書きしない。"""
    from forge.runtime import loader

    key_path = tmp_path / "hmac.key"
    key_path.write_bytes(b"short")
    _reset_key(monkeypatch, key_path)
    with caplog.at_level("WARNING"):
        signed = loader.sign_kernel_code("code")
        assert loader.verify_kernel_code(signed) == "code"
    assert key_path.read_bytes() == b"short"
    assert sum("hmac" in r.getMessage().lower() for r in caplog.records) == 1


def test_hmac_key_reused_within_process(tmp_path, monkeypatch):
    """同一プロセス内で署名→検証が通り、再読み込みしても同じ鍵になる。"""
    from forge.runtime import loader

    key_path = tmp_path / "hmac.key"
    _reset_key(monkeypatch, key_path)
    signed = loader.sign_kernel_code("code")
    monkeypatch.setattr(loader, "_HMAC_KEY", None, raising=False)  # 再読み込み
    assert loader.verify_kernel_code(signed) == "code"
