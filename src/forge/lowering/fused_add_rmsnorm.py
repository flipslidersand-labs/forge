from __future__ import annotations

from .registry import OpPattern, register

# fused_add_rmsnorm(x, residual, weight, eps):
#   hidden = x + residual
#   return hidden * rsqrt(mean(hidden*hidden, -1) + eps) * weight
# torch.fx trace → {mul:3, add:2, mean:1, rsqrt:1}
FUSED_ADD_RMSNORM_PATTERN = OpPattern(
    op_type="fused_add_rmsnorm",
    op_counts={"mul": 3, "add": 2, "mean": 1, "rsqrt": 1},
)

register(FUSED_ADD_RMSNORM_PATTERN)
