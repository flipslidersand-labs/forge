from __future__ import annotations

from .registry import OpPattern, register

# F.scaled_dot_product_attention を torch.fx でトレースすると
# call_function ノードが 1 個（target.__name__ == "scaled_dot_product_attention"）になる。
SDPA_PATTERN = OpPattern(
    op_type="scaled_dot_product_attention",
    op_counts={"scaled_dot_product_attention": 1},
)

register(SDPA_PATTERN)
