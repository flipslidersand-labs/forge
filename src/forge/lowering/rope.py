from __future__ import annotations

from .registry import OpPattern, register

# apply_rope(x, cos, sin) = x*cos + rotate_half(x)*sin
# rotate_half: h = x.shape[-1]//2; cat([-x[...,h:], x[...,:h]])
# torch.fx trace → {getitem:3, mul:2, getattr:1, floordiv:1, neg:1, cat:1, add:1}
ROPE_PATTERN = OpPattern(
    op_type="rope",
    op_counts={"getitem": 3, "mul": 2, "getattr": 1, "floordiv": 1, "neg": 1, "cat": 1, "add": 1},
)

register(ROPE_PATTERN)
