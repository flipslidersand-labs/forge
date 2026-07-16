from __future__ import annotations

from .registry import OpPattern, register

ATTENTION_PATTERN = OpPattern(
    op_type="attention",
    op_counts={"scaled_dot_product_attention": 1},
)
register(ATTENTION_PATTERN)
