from __future__ import annotations

from typing import Any

from .registry import OpPattern, is_dynamic, literal_arg, register


def _supported_args(node: Any) -> bool:
    # sdpa のシグネチャ: (q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa)
    # 既定外の値はカーネルに反映されず結果が静かに変わるので、一致させず eager に落とす。
    #
    # attn_mask: カーネルは受け取らない。fx 上は引数でもクロージャのテンソル(get_attr)でも
    #   Node になるので、Node かどうかに関わらず None 以外は対応外。
    if literal_arg(node, "attn_mask", 3, None) is not None:
        return False
    # dropout_p / scale / enable_gqa: リテラルのみ検査する。引数経由（Node）の扱いは #472。
    # scale は既定 (1/sqrt(head_dim)) と等しい値でも静的には判定できないので None のみ対応。
    dropout_p = literal_arg(node, "dropout_p", 4, 0.0)
    if not is_dynamic(dropout_p) and dropout_p != 0:
        return False
    scale = literal_arg(node, "scale", 6, None)
    if not is_dynamic(scale) and scale is not None:
        return False
    enable_gqa = literal_arg(node, "enable_gqa", 7, False)
    if not is_dynamic(enable_gqa) and enable_gqa:
        return False
    # is_causal は引数経由（Node）なら constants としてカーネルに渡るので対象外。
    # 関数本体のリテラル True は constants に入らず非 causal カーネルが採用されてしまうため対応外。
    is_causal = literal_arg(node, "is_causal", 5, False)
    return is_dynamic(is_causal) or not is_causal


# F.scaled_dot_product_attention を torch.fx でトレースすると
# call_function ノードが 1 個（target.__name__ == "scaled_dot_product_attention"）になる。
#
# forge が最適化できる SDPA の制約:
#   - head_dim (最終次元) は 2 のべき乗 かつ ≥ 16 であること
#   - attn_mask は None のみ対応（引数・クロージャのテンソルも eager）
#   - dropout_p は 0 のみ（リテラルで非 0 は eager フォールバック）
#   - enable_gqa=True は非対応（GQA / MQA は異なる op パターン）
#   - is_causal は関数の引数として渡された場合のみ対応（本体のリテラル True は対応外 → eager）
#   - scale は None のみ対応（リテラルで値を渡すと eager。constants への取り込みは #472）
SDPA_PATTERN = OpPattern(
    op_type="scaled_dot_product_attention",
    op_counts={"scaled_dot_product_attention": 1},
    node_guards={"scaled_dot_product_attention": _supported_args},
)

register(SDPA_PATTERN)
