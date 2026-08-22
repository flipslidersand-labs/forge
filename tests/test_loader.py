"""loader.py のユニットテスト。Triton 不要。"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

_DUMMY_MODULE = """\
def kernel_fn(*args, **kwargs):
    return None
"""

_BROKEN_MODULE = """\
raise RuntimeError("broken")
def kernel_fn(): pass
"""


def test_file_is_created_and_importable():
    """load_kernel_fn が一時ファイルを作成し kernel_fn を返す。"""
    from forge.runtime import loader

    fn = loader.load_kernel_fn(_DUMMY_MODULE)
    assert callable(fn)


def test_tmpfile_deleted_after_successful_load():
    """ロード成功後に /tmp/forge_kernels/ に .py ファイルが残らない。"""
    from forge.runtime import loader

    tmp_dir = Path(tempfile.gettempdir()) / "forge_kernels"
    before = set(tmp_dir.glob("kernel_*.py")) if tmp_dir.exists() else set()

    loader.load_kernel_fn(_DUMMY_MODULE)

    after = set(tmp_dir.glob("kernel_*.py")) if tmp_dir.exists() else set()
    new_files = after - before
    assert new_files == set(), f"残存ファイル: {new_files}"


def test_tmpfile_deleted_after_load_failure():
    """exec_module が例外を投げた場合もファイルが削除される。"""
    import uuid

    from forge.runtime import loader

    tmp_dir = Path(tempfile.gettempdir()) / "forge_kernels"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []

    original_write = Path.write_text

    def _track_write(self: Path, *args, **kwargs):  # type: ignore[override]
        result = original_write(self, *args, **kwargs)
        if "forge_kernels" in str(self) and self.suffix == ".py":
            created.append(self)
        return result

    with patch.object(Path, "write_text", _track_write):
        try:
            loader.load_kernel_fn(_BROKEN_MODULE)
        except (RuntimeError, AssertionError):
            pass

    for p in created:
        assert not p.exists(), f"失敗時にも削除されるべき: {p}"


def test_tmpfile_missing_ok_on_load_failure():
    """exec_module が例外を投げたときに unlink(missing_ok=True) がエラーを出さない。"""
    from forge.runtime import loader

    try:
        loader.load_kernel_fn(_BROKEN_MODULE)
    except (RuntimeError, AssertionError):
        pass  # 例外は伝播するが unlink で追加エラーは出ない
