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
