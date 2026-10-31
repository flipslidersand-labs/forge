from __future__ import annotations

from .registry import OpPattern, register

# F.silu(gate) * x を torch.fx で trace すると {silu:1, mul:1}
SWIGLU_PATTERN = OpPattern(
    op_type="swiglu",
    op_counts={"silu": 1, "mul": 1},
)

register(SWIGLU_PATTERN)
