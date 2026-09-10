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
    """wrapper の freevars に shape 別ビルドロック取得関数が含まれること
    （コード変数名で確認、#329でop_type用ロックとビルド用ロックを分離）。

    functools.wraps 後 fn 自体が wrapper 関数を指す（__wrapped__ は元の fn）。
    """
    fn = _make_cpu_fn()
    freevars = fn.__code__.co_freevars
    assert "_get_build_lock" in freevars, f"_get_build_lock が wrapper freevars に無い: {freevars}"


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


def test_different_shapes_build_concurrently():
    """#329: 無関係な shape のビルドは単一ロックで直列化されず並行に進行できる。

    2つの異なる shape を同時にビルドさせ、両方が「開始」した後でなければ
    どちらも完了できないバリアを仕込む。単一ロックのままなら一方のビルドが
    もう一方の開始をブロックし、必ずタイムアウトする。
    """
    import time

    import torch

    build_started: dict[tuple[int, ...], bool] = {}
    release = threading.Event()

    def fake_build(fn, op_type, tensors, *rest):
        shape = tuple(tensors[0].shape)
        build_started[shape] = True
        if not release.wait(timeout=2):
            raise TimeoutError(f"shape={shape} のビルドが並行開始されなかった（直列化の疑い）")
        return None

    def _fake_bind(*args, **kwargs):
        bound = MagicMock()
        bound.arguments = {"x": args[0]}
        bound.apply_defaults.return_value = None
        return bound

    with (
        patch("forge.decorator.identify", return_value="rmsnorm"),
        patch("forge.decorator._build", side_effect=fake_build),
        patch("inspect.Signature.bind", side_effect=_fake_bind),
    ):

        @optimize(budget=1)
        def fn(x):
            return x

        def _make_tensor(shape: tuple[int, ...]) -> MagicMock:
            t = MagicMock(spec=torch.Tensor)
            t.is_cuda = True
            t.shape = shape
            t.dtype = torch.float32
            return t

        t_a = _make_tensor((4, 4))
        t_b = _make_tensor((8, 8))

        errors: list[Exception] = []

        def call_fn(t: MagicMock) -> None:
            try:
                fn(t)
            except Exception as e:
                errors.append(e)

        th_a = threading.Thread(target=call_fn, args=(t_a,))
        th_b = threading.Thread(target=call_fn, args=(t_b,))
        th_a.start()
        th_b.start()

        deadline = time.monotonic() + 2
        while len(build_started) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        release.set()

        th_a.join(timeout=5)
        th_b.join(timeout=5)

    assert not errors, f"スレッド実行中に例外: {errors}"
    assert len(build_started) == 2, f"両方の shape が並行ビルドされなかった: {build_started}"
