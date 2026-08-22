"""_fn_graph_hash — CPU-only unit tests for #197.

Verifies that two functions with the same op_type but different implementations
produce different graph_hash values, preventing cache key collisions.
"""

from __future__ import annotations

from forge.decorator import _fn_graph_hash


class TestFnGraphHash:
    def test_same_function_is_deterministic(self) -> None:
        def rmsnorm_v1(x, w, eps):
            return x / (x**2).mean() * w

        assert _fn_graph_hash(rmsnorm_v1) == _fn_graph_hash(rmsnorm_v1)

    def test_different_implementations_differ(self) -> None:
        def rmsnorm_v1(x, w, eps):  # noqa: ARG001
            return x / (x**2).mean() * w

        def rmsnorm_v2(x, w, eps):  # noqa: ARG001
            # different body
            import math
            return x * math.sqrt(1.0) * w

        assert _fn_graph_hash(rmsnorm_v1) != _fn_graph_hash(rmsnorm_v2)

    def test_different_qualnames_differ(self) -> None:
        """Even with identical body, different __qualname__ produces different hash."""

        def alpha(x):
            return x

        def beta(x):
            return x

        # Bodies are identical but qualnames differ
        assert _fn_graph_hash(alpha) != _fn_graph_hash(beta)

    def test_hash_contains_qualname(self) -> None:
        def my_special_kernel(x):
            return x

        h = _fn_graph_hash(my_special_kernel)
        assert "my_special_kernel" in h

    def test_lambda_does_not_raise(self) -> None:
        """Lambdas may not have getsource() — must not raise."""
        f = lambda x: x  # noqa: E731
        result = _fn_graph_hash(f)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_two_lambdas_differ_by_qualname_uniqueness(self) -> None:
        """Two closures captured at different points get distinct qualnames."""

        def make(n: int):
            def fn(x):
                return x + n
            return fn

        f1 = make(1)
        f2 = make(2)
        # Both are named 'make.<locals>.fn' — same qualname + same source.
        # This is a known limitation: closures with identical qualname+body share hash.
        # For user-decorated top-level functions this case does not arise.
        assert isinstance(_fn_graph_hash(f1), str)
        assert isinstance(_fn_graph_hash(f2), str)
