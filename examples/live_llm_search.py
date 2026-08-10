"""ライブ LLM 反復探索の使用例（モック評価）。

ANTHROPIC_API_KEY が設定されていれば実際に Claude Opus を呼ぶが、
未設定の場合は propose_fn を注入してオフラインで動作確認できる。

Usage:
    python examples/live_llm_search.py          # オフライン（モック）
    ANTHROPIC_API_KEY=sk-... python examples/live_llm_search.py --live
"""

from __future__ import annotations

import os
import sys

import torch

from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.search.candidate import HistoryEntry
from forge.search.iterative_search import IterativeLLMSearch, build_feedback_prompt
from forge.search.llm_generator import LLMGenerator
from forge.search.params import SearchParams

SPEC = KernelSpec(
    op_type="rmsnorm",
    input_specs=(
        TensorSpec((2048, 4096), torch.float16, True),
        TensorSpec((4096,), torch.float16, True),
    ),
    output_specs=(TensorSpec((2048, 4096), torch.float16, True),),
    constants={"eps": 1e-6},
    graph_hash="rmsnorm_demo",
    constraints=(),
)

_MOCK_POOL = [
    dict(base_variant="single_row", block_size=4096, num_warps=4,  num_stages=1, acc_dtype="fp32", rows_per_program=1, hypothesis="narrow warps"),
    dict(base_variant="single_row", block_size=4096, num_warps=8,  num_stages=1, acc_dtype="fp32", rows_per_program=1, hypothesis="balanced"),
    dict(base_variant="single_row", block_size=4096, num_warps=16, num_stages=2, acc_dtype="fp32", rows_per_program=1, hypothesis="wide warps"),
    dict(base_variant="two_pass",   block_size=1024, num_warps=8,  num_stages=2, acc_dtype="fp32", rows_per_program=1, hypothesis="two-pass tiling"),
    dict(base_variant="multi_row",  block_size=4096, num_warps=8,  num_stages=2, acc_dtype="fp32", rows_per_program=4, hypothesis="multi-row fused"),
]
_mock_idx = 0


def _mock_propose(spec, cc, n, history):
    global _mock_idx
    start = _mock_idx % len(_MOCK_POOL)
    out = (_MOCK_POOL * 2)[start : start + n]
    _mock_idx += n
    return out


def _mock_evaluate(params: SearchParams) -> HistoryEntry:
    """疑似ベンチマーク: block_size × num_warps でスコアを決める。"""
    import random
    latency = 300.0 / (params.num_warps * (params.block_size / 4096))
    latency += random.uniform(-5, 5)
    return HistoryEntry(params=params, correct=True, median_us=latency)


def main() -> None:
    live = "--live" in sys.argv and os.getenv("ANTHROPIC_API_KEY")

    if live:
        gen = LLMGenerator(model="claude-opus-4-8")
        print("mode: live (Claude Opus)")
    else:
        gen = LLMGenerator(propose_fn=_mock_propose)
        print("mode: offline (mock propose_fn)")

    searcher = IterativeLLMSearch(
        generator=gen,
        evaluate_fn=_mock_evaluate,
        max_rounds=3,
        candidates_per_round=5,
        max_tokens=50_000 if live else None,
        top_k_feedback=3,
    )

    result = searcher.run(SPEC, compute_capability="8.9")

    print(f"\n=== Results ({len(result.rounds)} rounds) ===")
    for r in result.rounds:
        print(f"Round {r.round_num}: {len(r.evaluated)} evaluated, tokens={r.token_usage.total}")
        for entry in r.evaluated:
            status = f"{entry.median_us:.1f}us" if entry.correct else "FAIL"
            print(f"  {entry.params.variant} bs={entry.params.block_size} w={entry.params.num_warps} -> {status}")

    if result.best:
        print(f"\nbest: {result.best.params} @ {result.best.median_us:.1f}us")
    else:
        print("\nbest: none (all failed)")

    print(f"total tokens: {result.total_token_usage.total}")

    if len(result.rounds) > 0:
        all_entries = [e for r in result.rounds for e in r.evaluated]
        fb = build_feedback_prompt(all_entries, top_k=3)
        if fb:
            print("\n--- Feedback summary ---")
            print(fb)


if __name__ == "__main__":
    main()
