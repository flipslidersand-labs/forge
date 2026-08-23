from __future__ import annotations

from .registry import OpPattern, register

# F.scaled_dot_product_attention を torch.fx でトレースすると
# call_function ノードが 1 個（target.__name__ == "scaled_dot_product_attention"）になる。
#
# forge が最適化できる SDPA の制約:
#   - head_dim (最終次元) は 2 のべき乗 かつ ≥ 16 であること
#   - attn_mask は None のみ対応（任意マスクは fx トレース不能）
#   - dropout_p は 0.0 のみ（> 0.0 は eager フォールバック）
#   - enable_gqa=True は非対応（GQA / MQA は異なる op パターン）
#   - is_causal / scale は任意値対応
SDPA_PATTERN = OpPattern(
    op_type="scaled_dot_product_attention",
    op_counts={"scaled_dot_product_attention": 1},
)

register(SDPA_PATTERN)
