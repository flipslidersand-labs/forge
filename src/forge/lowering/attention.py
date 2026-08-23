from __future__ import annotations

from .registry import OpPattern, register

# F.scaled_dot_product_attention を torch.fx でトレースすると
# call_function ノードが 1 個（target.__name__ == "scaled_dot_product_attention"）になる。
#
# 対応制約（違反時は識別失敗 → eager フォールバック）:
#   - head_dim >= 16 かつ head_dim % 16 == 0（2 のべき乗推奨）
#   - attn_mask=None のみ（任意マスクはグラフパターンが変わり認識不可）
#   - dropout_p=0.0 のみ
#   - enable_gqa=False のみ（GQA/MQA は別パターン）
#   - is_causal は True / False どちらも対応（causal_opt バリアントを自動選択）
SDPA_PATTERN = OpPattern(
    op_type="scaled_dot_product_attention",
    op_counts={"scaled_dot_product_attention": 1},
)

register(SDPA_PATTERN)
