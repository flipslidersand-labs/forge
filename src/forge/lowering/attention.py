from __future__ import annotations

from .registry import OpPattern, register

SDPA_PATTERN = OpPattern(
    op_type="scaled_dot_product_attention",
    op_counts={"scaled_dot_product_attention": 1},
)

register(SDPA_PATTERN)
