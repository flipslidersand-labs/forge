from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from forge.ir.kernel_spec import KernelSpec

from ._base_generator import _BaseGenerator
from .candidate import HistoryEntry
from .params import SUPPORTED_ACC_DTYPES, SUPPORTED_VARIANTS

# LLM に候補を出させる際の戻り値型: 構造化された dict のリスト。
ProposeFn = Callable[[KernelSpec, str, int, list[HistoryEntry]], list[dict[str, Any]]]

DEFAULT_MODEL = "claude-opus-4-8"


@dataclass
class TokenUsage:
    """Anthropic API トークン使用量の累積カウンタ。"""

    input_tokens: int = field(default=0)
    output_tokens: int = field(default=0)

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def _add_from_response(self, resp: Any) -> None:
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.input_tokens += getattr(usage, "input_tokens", 0)
            self.output_tokens += getattr(usage, "output_tokens", 0)


_SYSTEM = (
    "You are a GPU kernel autotuning assistant. You propose Triton kernel "
    "configurations for a given operation and a one-line hypothesis for each. "
    "You do NOT write code — only structured parameters. Favor configurations "
    "likely to be both correct and fast, and learn from the provided history."
)


class LLMGenerator(_BaseGenerator):
    """Claude に構造化された探索候補を出させる CandidateGenerator 実装。

    LLM には自由な Triton コードを書かせず、変更命令（variant + パラメータ + 仮説）
    のみを JSON で出力させる（ADR-005）。実際のコード生成はテンプレートが担う。

    API 呼び出しは ``propose_fn`` で差し替え可能。テストや非ネットワーク環境では
    canned な dict を返す関数を注入する。省略時は Anthropic SDK を遅延 import して
    ``claude-opus-4-8`` を呼ぶ（ANTHROPIC_API_KEY が必要）。
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        client: Any | None = None,
        default_n: int = 12,
        propose_fn: ProposeFn | None = None,
    ) -> None:
        try:
            import typing

            from anthropic.types import ModelParam

            # ModelParam は Union[Literal["claude-..."], str] 型。
            # Literal の引数だけ取り出して既知モデルセットを構築する。
            _known: set[str] = set()
            for arg in ModelParam.__args__:  # type: ignore[union-attr]
                if hasattr(arg, "__args__"):
                    _known.update(a for a in typing.get_args(arg) if isinstance(a, str))
            if _known and model not in _known:
                raise ValueError(
                    f"Unknown Anthropic model: {model!r}. Known models: {sorted(_known)}"
                )
        except ImportError:
            pass  # anthropic SDK 未インストール時はスキップ
        self.model = model
        self.client = client
        self.default_n = default_n
        self._propose_fn = propose_fn
        self.token_usage = TokenUsage()

    def reset_usage(self) -> None:
        """token_usage を初期化する。複数 spec にまたがって再利用する場合に呼ぶ。"""
        self.token_usage = TokenUsage()

    def _propose(
        self,
        spec: KernelSpec,
        compute_capability: str,
        n: int,
        history: list[HistoryEntry],
    ) -> list[dict[str, Any]]:
        if self._propose_fn is not None:
            return self._propose_fn(spec, compute_capability, n, history)
        return self._propose_via_claude(spec, compute_capability, n, history)

    # --- 実際の Claude 呼び出し（遅延 import） ---

    def _propose_via_claude(
        self,
        spec: KernelSpec,
        compute_capability: str,
        n: int,
        history: list[HistoryEntry],
    ) -> list[dict[str, Any]]:
        import anthropic  # type: ignore[import]  # anthropic SDK は py.typed 未対応

        from ._proposal_models import Proposal

        client = self.client or anthropic.Anthropic()
        prompt = build_prompt(spec, compute_capability, n, history)
        resp = client.messages.parse(
            model=self.model,
            max_tokens=8192,
            thinking={"type": "adaptive"},
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_format=Proposal,
        )
        self.token_usage._add_from_response(resp)
        proposal = resp.parsed_output
        if proposal is None:
            return []
        return [c.model_dump() for c in proposal.candidates]


def build_prompt(
    spec: KernelSpec,
    compute_capability: str,
    n: int,
    history: list[HistoryEntry],
) -> str:
    x = spec.input_specs[0]
    op_name = spec.op_type.replace("_", " ").title()
    input_shape_desc = ", ".join(str(s.shape) for s in spec.input_specs)
    lines = [
        f"Propose {n} distinct Triton {op_name} kernel configurations.",
        "",
        f"GPU compute capability: {compute_capability}",
        f"Input shapes: {input_shape_desc}, dtype: {x.dtype_str()}",
    ]
    if spec.constants:
        lines.append(f"Constants: {dict(spec.constants)}")
    lines += [
        "",
        "Each candidate must specify:",
        f"- base_variant: one of {list(SUPPORTED_VARIANTS)}",
        "- block_size: power of 2. For single_row/multi_row it must be >= hidden "
        f"size ({x.shape[-1]}); for two_pass it is a tile and must be <= hidden size.",
        "- num_warps: typically 4, 8, or 16",
        "- num_stages: 1 on pre-Volta (cc < 7.0), else 1-3",
        f"- acc_dtype: one of {list(SUPPORTED_ACC_DTYPES)}. Note: fp16 accumulation "
        "in the reduction is usually NOT numerically correct for single_row/multi_row.",
        "- rows_per_program: >1 only for multi_row, else 1",
        "- hypothesis: one line explaining why this config may be fast",
    ]
    if history:
        lines += ["", "History of already-evaluated configs (learn from these):"]
        lines += [f"  {h.summary()}" for h in history[-30:]]
    return "\n".join(lines)
