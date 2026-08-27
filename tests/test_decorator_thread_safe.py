"""decorator.wrapper のスレッドセーフ検証 (#267)。GPU 不要。"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from forge.decorator import optimize


def _make_cpu_fn():
    """CPU テンソルを受け取るだけのダミー関数（GPU チェックで eager fallback）。"""

    @optimize(budget=1)
    def fn(x):
        return x

    return fn


def test_lock_attribute_exists():
    """wrapper の freevars に _lock が含まれること（コード変数名で確認）。

    functools.wraps 後 fn 自体が wrapper 関数を指す（__wrapped__ は元の fn）。
    """
    fn = _make_cpu_fn()
    freevars = fn.__code__.co_freevars
    assert "_lock" in freevars, f"_lock が wrapper freevars に無い: {freevars}"


def test_compiled_dict_built_once_under_concurrent_access():
    """同一 shape で複数スレッドが同時呼び出しても _build は 1 回だけ実行される (#267)。"""
    import torch

    build_count = 0
    build_lock = threading.Lock()

    def fake_build(*args, **kwargs):
        nonlocal build_count
        import time

        time.sleep(0.02)  # _build の GPU 探索を模倣
        with build_lock:
            build_count += 1
        return None  # eager fallback を返す

    # identify が "rmsnorm" を返し、_build が fake_build に差し替えられる状態を作る
    with (
        patch("forge.decorator.identify", return_value="rmsnorm"),
        patch("forge.decorator._build", side_effect=fake_build),
    ):

        @optimize(budget=1)
        def fn(x):
            return x

        # CUDA が無い環境では is_cuda=False で eager fallback になるため、
        # CUDA テンソルのふりをした MagicMock を使う
        t = MagicMock(spec=torch.Tensor)
        t.is_cuda = True
        t.shape = (4, 4)
        t.dtype = torch.float32

        # パッチした fn(*args) で tensors リストが [t] になるよう bind を迂回
        with patch("inspect.Signature.bind") as mock_bind:
            bound = MagicMock()
            bound.arguments = {"x": t}
            bound.apply_defaults.return_value = None
            mock_bind.return_value = bound

            errors: list[Exception] = []

            def call_fn():
                try:
                    fn(t)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=call_fn) for _ in range(8)]
            for th in threads:
                th.start()
            for th in threads:
                th.join()

    assert not errors, f"スレッド実行中に例外: {errors}"
    assert build_count == 1, f"_build が {build_count} 回呼ばれた（期待: 1 回）"
