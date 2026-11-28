"""採用前にユーザー関数そのものと採用カーネルを実入力 1 回で照合する (#472)。

CPU テスト(クラス TestBuildVerification)は _build を直接呼び、Orchestrator と
load_kernel_fn を差し替える。GPU テスト(TestEndToEnd)は実カーネルで確かめる。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F

import forge
from forge import decorator as dec
from forge.cache.repository import KernelRepository
from forge.ops.registry import OP_REGISTRY, accepts_call
from forge.search.grid import GridSearch
from forge.search.space import SearchSpace

_CFG = dec._BuildConfig(
    budget=1,
    repo=None,
    search=None,
    min_speedup=1.0,
    python_executable=None,
    progress=lambda _e: None,
)


class _FakeOrch:
    def __init__(self, **_: Any) -> None:
        pass

    def __enter__(self) -> _FakeOrch:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def optimize(self, spec: Any, budget: int, search: Any) -> Any:
        return SimpleNamespace(best_params="p", adopted=True, speedup=2.0, cache_hit=False)


def _build(
    monkeypatch: pytest.MonkeyPatch,
    user_fn: Any,
    kernel: Any,
    args: tuple,
    op_type: str = "softmax",
    kwargs: dict | None = None,
    constants: dict | None = None,
) -> Any:
    monkeypatch.setattr(dec, "Orchestrator", _FakeOrch)
    monkeypatch.setattr(dec, "generate", lambda spec, params: "code")
    monkeypatch.setattr(dec, "load_kernel_fn", lambda code: kernel)
    tensors = [a for a in (*args, *(kwargs or {}).values()) if isinstance(a, torch.Tensor)]
    return dec._build(
        user_fn,
        op_type,
        tensors,
        constants or {},
        _CFG,
        (args, kwargs or {}),
    )


def _softmax_fn(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


class TestBuildVerification:
    def test_matching_kernel_is_adopted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        x = torch.randn(4, 8)

        def kernel(t: torch.Tensor) -> torch.Tensor:
            return torch.softmax(t, dim=-1)

        assert _build(monkeypatch, _softmax_fn, kernel, (x,)) is kernel

    def test_mismatching_kernel_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        x = torch.randn(4, 8)

        def kernel(t: torch.Tensor) -> torch.Tensor:
            return torch.softmax(t, dim=0)  # ユーザー関数は dim=-1

        assert _build(monkeypatch, _softmax_fn, kernel, (x,)) is None

    def test_user_fn_is_compared_not_the_reference(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """kernel が registry の reference(dim=-1)と一致しても、ユーザー関数と違えば不採用。"""
        x = torch.randn(4, 8)

        def user_fn(t: torch.Tensor) -> torch.Tensor:
            return torch.softmax(t, dim=0)

        def kernel(t: torch.Tensor) -> torch.Tensor:
            return torch.softmax(t, dim=-1)

        assert _build(monkeypatch, user_fn, kernel, (x,)) is None

    def test_shape_mismatch_is_rejected_even_if_broadcastable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        x = torch.randn(4, 8)

        def kernel(t: torch.Tensor) -> torch.Tensor:
            return torch.softmax(t, dim=-1)[:1]  # allclose はブロードキャストで通ってしまう形

        assert _build(monkeypatch, lambda t: torch.softmax(t, dim=-1), kernel, (x,)) is None

    def test_dtype_mismatch_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        x = torch.randn(4, 8)

        def kernel(t: torch.Tensor) -> torch.Tensor:
            return torch.softmax(t, dim=-1).double()

        assert _build(monkeypatch, _softmax_fn, kernel, (x,)) is None

    def test_non_tensor_result_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        x = torch.randn(4, 8)
        assert _build(monkeypatch, _softmax_fn, lambda t: None, (x,)) is None

    def test_kernel_exception_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        x = torch.randn(4, 8)

        def kernel(t: torch.Tensor) -> torch.Tensor:
            raise TypeError("unexpected keyword argument")

        assert _build(monkeypatch, _softmax_fn, kernel, (x,)) is None

    def test_constants_are_passed_to_kernel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        x = torch.randn(4, 8)
        seen: dict[str, Any] = {}

        def user_fn(t: torch.Tensor, scale: float) -> torch.Tensor:
            return t * scale

        def kernel(t: torch.Tensor, **kw: Any) -> torch.Tensor:
            seen.update(kw)
            return t * kw["scale"]

        out = _build(
            monkeypatch, user_fn, kernel, (x, 2.0), constants={"scale": 2.0}, op_type="softmax"
        )
        assert out is kernel
        assert seen == {"scale": 2.0}

    def test_tolerance_is_the_registry_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """許容誤差はワーカー検証と同じ OP_REGISTRY の値(softmax は atol 2e-3, rtol 1e-2)。"""
        tol = OP_REGISTRY["softmax"].tolerance
        x = torch.ones(2, 4)

        def inside(t: torch.Tensor) -> torch.Tensor:
            return t + tol.atol * 0.5

        def outside(t: torch.Tensor) -> torch.Tensor:
            return t + tol.atol * 3 + tol.rtol * 3

        assert _build(monkeypatch, lambda t: t.clone(), inside, (x,)) is inside
        assert _build(monkeypatch, lambda t: t.clone(), outside, (x,)) is None

    def test_inplace_user_fn_does_not_corrupt_inputs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """fn が入力をその場で書き換えても、照合はユーザーの実テンソルに触れない。"""
        x = torch.randn(4, 8)
        orig = x.clone()

        def user_fn(t: torch.Tensor) -> torch.Tensor:
            y = torch.softmax(t, dim=-1)
            t.zero_()
            return y

        def kernel(t: torch.Tensor) -> torch.Tensor:
            return torch.softmax(t, dim=-1)

        assert _build(monkeypatch, user_fn, kernel, (x,)) is kernel
        assert torch.equal(x, orig)

    def test_inplace_kernel_does_not_corrupt_inputs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """カーネルが入力を書き換えても、ユーザーの実テンソルは変わらない。"""
        x = torch.randn(4, 8)
        orig = x.clone()

        def kernel(t: torch.Tensor) -> torch.Tensor:
            y = torch.softmax(t, dim=-1)
            t.zero_()
            return y

        assert _build(monkeypatch, _softmax_fn, kernel, (x,)) is kernel
        assert torch.equal(x, orig)

    def test_inplace_user_fn_with_kwargs_tensor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        x = torch.randn(4, 8)
        orig = x.clone()

        def user_fn(t: torch.Tensor) -> torch.Tensor:
            y = torch.softmax(t, dim=-1)
            t.zero_()
            return y

        def kernel(t: torch.Tensor) -> torch.Tensor:
            return torch.softmax(t, dim=-1)

        assert _build(monkeypatch, user_fn, kernel, (), kwargs={"t": x}) is kernel
        assert torch.equal(x, orig)

    def test_kernel_exception_warning_has_traceback(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        def kernel(t: torch.Tensor) -> torch.Tensor:
            raise TypeError("boom")

        with caplog.at_level("WARNING", logger="forge.decorator"):
            _build(monkeypatch, _softmax_fn, kernel, (torch.randn(2, 4),))
        assert any(r.exc_info for r in caplog.records)

    def test_build_without_user_call_skips_verification(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """user_call を渡さない呼び出し(内部テスト用)では照合しない。decorator は必ず渡す。"""
        monkeypatch.setattr(dec, "Orchestrator", _FakeOrch)
        monkeypatch.setattr(dec, "generate", lambda spec, params: "code")
        sentinel = object()
        monkeypatch.setattr(dec, "load_kernel_fn", lambda code: sentinel)
        out = dec._build(_softmax_fn, "softmax", [torch.zeros(2)], {}, _CFG)
        assert out is sentinel


class TestWrapperPassesUserCall:
    def test_wrapper_gives_build_the_real_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """wrapper が _build に実際の args / kwargs を渡す(照合に使われる)。"""
        from unittest.mock import MagicMock, patch

        def _fake_cuda() -> Any:
            t = MagicMock(spec=torch.Tensor)
            t.is_cuda = True
            t.device = "cuda:0"
            t.shape = (4, 8)
            t.dtype = torch.float16
            t.is_contiguous.return_value = True
            return t

        got: dict[str, Any] = {}

        def fake_build(fn, op_type, tensors, constants, config, user_call=None):
            got["user_call"] = user_call
            return None

        x = _fake_cuda()
        with (
            patch("forge.decorator.identify", return_value="softmax"),
            patch("forge.decorator._build", side_effect=fake_build),
        ):

            @forge.optimize(budget=1)
            def f(t, dim=-1):
                return "eager"

            f(x, dim=-1)
        args, kwargs = got["user_call"]
        assert args == (x,)
        assert kwargs == {"dim": -1}


class TestSdpaAcceptsCall:
    @pytest.mark.parametrize(
        "constants",
        [{}, {"is_causal": True}, {"scale": 0.5}, {"scale": None, "is_causal": False}],
    )
    def test_supported_constants(self, constants: dict[str, Any]) -> None:
        assert accepts_call("scaled_dot_product_attention", 3, constants)

    @pytest.mark.parametrize(
        "constants",
        [{"dropout_p": 0.0}, {"enable_gqa": False}, {"causal": True}, {"attn_mask": None}],
    )
    def test_unsupported_constants_fall_back(self, constants: dict[str, Any]) -> None:
        """kernel_fn が受け取らない引数名(dropout_p 等)は探索せず eager。"""
        assert not accepts_call("scaled_dot_product_attention", 3, constants)

    def test_wrong_tensor_count(self) -> None:
        assert not accepts_call("scaled_dot_product_attention", 2, {})


# --- GPU e2e ---


def _fast_search(**kw: Any) -> GridSearch:
    return GridSearch(
        SearchSpace(
            attention_seq_blocks=[16],
            num_warps=[4],
            num_stages=[1],
            acc_dtypes=["fp32"],
            variants=["single_row"],
            **kw,
        )
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")
class TestEndToEnd:
    def test_rmsnorm_lookalike_is_not_adopted(self, caplog: pytest.LogCaptureFixture) -> None:
        """rmsnorm と同じ演算の多重集合で式が違う関数は、照合で弾かれて eager の結果を返す。"""
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")

            def lookalike(x, weight, eps=1e-6):
                # mean(x)（x*x ではない）。{mul:3, mean:1, add:1, rsqrt:1} は rmsnorm と同じ
                return x * x * weight * torch.rsqrt(torch.mean(x, dim=-1, keepdim=True) + eps)

            f = forge.optimize(
                budget=1,
                repo=repo,
                search=GridSearch(
                    SearchSpace(
                        block_sizes=[256],
                        num_warps=[4],
                        num_stages=[1],
                        acc_dtypes=["fp32"],
                        variants=["single_row"],
                    )
                ),
                min_speedup=1e-6,  # 速度では落とさない(照合だけで落とすことを確かめる)
            )(lookalike)

            x = torch.rand(8, 256, device="cuda", dtype=torch.float32) + 0.5  # mean(x) > 0
            w = torch.rand(256, device="cuda", dtype=torch.float32) + 0.5
            out = f(x, w)
            assert torch.equal(out, lookalike(x, w))
            assert list(f._forge_compiled.values()) == [None]

            # 誤カーネルは SQLite キャッシュに残る。新しい decorator(= 別プロセス相当)が
            # cache hit で読んでも、同じ照合で弾かれる。
            from unittest.mock import patch

            f2 = forge.optimize(budget=1, repo=repo, min_speedup=1e-6)(lookalike)
            with patch("forge.orchestrator.run_in_worker") as worker:
                out2 = f2(x, w)
                worker.assert_not_called()  # cache hit(探索していない)
            assert torch.equal(out2, lookalike(x, w))
            assert list(f2._forge_compiled.values()) == [None]
            repo.close()

    def test_inplace_user_fn_first_call_returns_correct_result(self) -> None:
        """入力をその場で書き換える関数でも、初回の戻り値は eager と同じ(照合が入力を壊さない)。"""
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")

            def f(x):
                y = torch.softmax(x, dim=-1)
                x.zero_()
                return y

            g = forge.optimize(
                budget=1,
                repo=repo,
                min_speedup=1e-6,
                search=GridSearch(
                    SearchSpace(
                        block_sizes=[256],
                        num_warps=[4],
                        num_stages=[1],
                        acc_dtypes=["fp32"],
                        variants=["single_row"],
                    )
                ),
            )(f)
            x = torch.randn(8, 256, device="cuda")
            ref = torch.softmax(x.clone(), dim=-1)
            out = g(x)
            assert torch.allclose(out, ref, atol=1e-3)
            repo.close()

    def test_matching_function_is_still_adopted(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")

            def rmsnorm(x, weight, eps=1e-6):
                return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * weight

            f = forge.optimize(
                budget=1,
                repo=repo,
                search=GridSearch(
                    SearchSpace(
                        block_sizes=[256],
                        num_warps=[4],
                        num_stages=[1],
                        acc_dtypes=["fp32"],
                        variants=["single_row"],
                    )
                ),
                min_speedup=1e-6,
            )(rmsnorm)
            x = torch.randn(8, 256, device="cuda", dtype=torch.float32)
            w = torch.rand(256, device="cuda", dtype=torch.float32) + 0.5
            f(x, w)
            assert all(k is not None for k in f._forge_compiled.values())
            assert len(f._forge_compiled) == 1
            repo.close()

    def test_sdpa_scale_argument_is_honored(self) -> None:
        """引数で渡した scale はカーネルに反映される(#472 の持ち越しの確認)。"""
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")

            @forge.optimize(budget=1, repo=repo, search=_fast_search(), min_speedup=1e-6)
            def attn(q, k, v, scale=None):
                return F.scaled_dot_product_attention(q, k, v, scale=scale)

            q, k, v = (torch.randn(2, 64, 32, device="cuda", dtype=torch.float16) for _ in range(3))
            out = attn(q, k, v, scale=3.0)
            ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=3.0)
            assert torch.allclose(out.float(), ref, atol=5e-3, rtol=1e-2)
            assert all(kfn is not None for kfn in attn._forge_compiled.values())
            repo.close()

    def test_sdpa_dropout_p_argument_does_not_search(self) -> None:
        """dropout_p を引数で渡す関数は探索せず eager(kernel_fn が dropout_p を受けないため)。"""
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")

            @forge.optimize(budget=1, repo=repo, search=_fast_search(), min_speedup=1e-6)
            def attn(q, k, v, dropout_p=0.0):
                return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

            q, k, v = (torch.randn(2, 64, 32, device="cuda", dtype=torch.float16) for _ in range(3))
            with patch("forge.decorator._build") as build:
                attn(q, k, v, dropout_p=0.0)
            build.assert_not_called()
            repo.close()
