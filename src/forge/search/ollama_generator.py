from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from forge.ir.kernel_spec import KernelSpec
from forge.search.llm_generator import LLMGenerator, build_prompt
from forge.search.params import SearchParams

if TYPE_CHECKING:
    from forge.search.candidate import HistoryEntry

_DEFAULT_MODEL = "qwen2.5-coder:latest"
_DEFAULT_HOST = "http://localhost:11434"

_SYSTEM = (
    "You are a GPU kernel autotuning assistant. "
    "Propose Triton kernel configurations as structured JSON. "
    "Do NOT write code — only structured parameters."
)


class _Candidate(BaseModel):
    base_variant: str
    block_size: int
    num_warps: int
    num_stages: int
    acc_dtype: str
    rows_per_program: int
    hypothesis: str


class _Proposal(BaseModel):
    candidates: list[_Candidate]


class OllamaGenerator:
    """ローカル ollama サーバーを使った CandidateGenerator。

    ANTHROPIC_API_KEY 不要。`qwen2.5-coder` など JSON 出力を得意とするモデルを推奨。
    ollama の ``format`` パラメータで Pydantic スキーマを強制するため、
    JSON パース失敗のリスクが低い。

    ``generate()`` のシグネチャは ``LLMGenerator`` と同一なので、
    ``Orchestrator.optimize_rounds()`` にそのまま渡せる。

    Args:
        model: ollama モデル名（``ollama list`` で確認）。
        host: ollama サーバーアドレス。デフォルトは ``http://localhost:11434``。
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        host: str = _DEFAULT_HOST,
    ) -> None:
        self.model = model
        self.host = host

    def generate(
        self,
        spec: KernelSpec,
        compute_capability: str,
        budget: int | None = None,
        history: list[HistoryEntry] | None = None,
    ) -> list[SearchParams]:
        n = budget or 12
        prompt = build_prompt(spec, compute_capability, n, history or [])
        raw = self._propose(prompt)
        out: list[SearchParams] = []
        seen: set[SearchParams] = set()
        for d in raw:
            params = LLMGenerator._coerce(d, spec)
            if params is not None and params not in seen:
                seen.add(params)
                out.append(params)
        return out[:n]

    def _propose(self, prompt: str) -> list[dict[str, Any]]:
        import ollama

        try:
            resp = ollama.Client(host=self.host).chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                format=_Proposal.model_json_schema(),
            )
            content = resp.message.content or ""
            proposal = _Proposal.model_validate_json(content)
            return [c.model_dump() for c in proposal.candidates]
        except Exception:
            return []
