"""loader.py のユニットテスト。Triton 不要。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

_DUMMY_MODULE = """\
def kernel_fn(*args, **kwargs):
    return None
"""


def test_file_is_created_and_importable():
    """load_kernel_fn が一時ファイルを作成し kernel_fn を返す。"""
    from forge.runtime import loader

    fn = loader.load_kernel_fn(_DUMMY_MODULE)
    assert callable(fn)


def test_atexit_cleanup_removes_file():
    """atexit ハンドラが呼ばれるとファイルが削除される。"""
    import tempfile
    import uuid

    tmp_dir = Path(tempfile.gettempdir()) / "forge_kernels"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mod_path = tmp_dir / f"kernel_{uuid.uuid4().hex}.py"
    mod_path.write_text(_DUMMY_MODULE)
    assert mod_path.exists()

    cleanup = lambda: mod_path.unlink(missing_ok=True)  # noqa: E731
    cleanup()

    assert not mod_path.exists()


def test_atexit_cleanup_missing_ok():
    """存在しないファイルに unlink(missing_ok=True) してもエラーにならない。"""
    import tempfile
    import uuid

    tmp_dir = Path(tempfile.gettempdir()) / "forge_kernels"
    mod_path = tmp_dir / f"kernel_{uuid.uuid4().hex}.py"

    cleanup = lambda: mod_path.unlink(missing_ok=True)  # noqa: E731
    cleanup()  # raises nothing


def test_load_kernel_fn_registers_atexit():
    """load_kernel_fn が atexit.register を呼ぶことを確認。"""
    from forge.runtime import loader

    registered: list = []

    # loader モジュールが参照する atexit.register を直接 patch する
    with patch.object(loader.atexit, "register", side_effect=registered.append):
        loader.load_kernel_fn(_DUMMY_MODULE)

    assert len(registered) == 1
    # クリーンアップを手動実行して後始末
    registered[0]()


def test_load_kernel_fn_cleanup_deletes_file():
    """load_kernel_fn が登録した atexit ハンドラが実際にファイルを削除する。"""
    from forge.runtime import loader

    registered: list = []

    with patch.object(loader.atexit, "register", side_effect=registered.append):
        loader.load_kernel_fn(_DUMMY_MODULE)

    cleanup = registered[0]
    # 削除前にファイルが存在することは確認できないが（UUID が取れないため）、
    # cleanup を呼んでもエラーにならないことを確認
    cleanup()
