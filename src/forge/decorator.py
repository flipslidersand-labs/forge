from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable
from typing import Any

_log = logging.getLogger("forge.decorator")

from forge.cache.repository import KernelRepository
from forge.codegen.triton_codegen import generate
from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.lowering import identify
from forge.orchestrator import Orchestrator
from forge.runtime.loader import load_kernel_fn
from forge.search.adoption import should_run_search
from forge.search.candidate import CandidateGenerator


def optimize(
    budget: int = 50,
    backend: str = "triton",
    objective: str = "latency",
    *,
    repo: KernelRepository | None = None,
    search: CandidateGenerator | None = None,
    min_speedup: float = 1.03,
    min_invocations: int = 0,
    per_candidate_s: float = 2.0,
    python_executable: str | None = None,
    progress: Callable[..., None] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """純粋な PyTorch 演算を最速の Triton 実装へ自動置換するデコレータ。

    初回（shape ごと）: torch.fx で op_type を判定 → KernelSpec を組み立て →
    探索・検証・ベンチ・キャッシュ（Orchestrator）→ 最速カーネルを in-process ロード。
    2 回目以降（同 shape）: in-process キャッシュから直接実行。新しい shape は
    再探索するが SQLite キャッシュにヒットすれば即座に返る。

    判定不能・最適化で速くならない場合は元の eager 関数にフォールバックする。

    objective="economic" かつ min_invocations > 0 のとき、探索コストが回収できない
    と判断した場合は探索をスキップして eager にフォールバックする。
    per_candidate_s は 1 候補あたりの探索コスト見積もり（秒）。GPU の速度に応じて調整する。
    """

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(fn)
        # shape シグネチャ -> in-process カーネル (or None=eager)
        compiled: dict[tuple[Any, ...], Callable[..., Any] | None] = {}
        op_type_box: list[
            str | None
        ] = []  # 一度だけ判定（[] 未判定 / [None] 不可 / [str] 判定済み）

        def _resolve_op_type() -> str | None:
            if not op_type_box:
                op_type_box.append(identify(fn) if backend == "triton" else None)
            return op_type_box[0]

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import torch

            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            tensors = [v for v in bound.arguments.values() if isinstance(v, torch.Tensor)]
            constants = {
                k: v for k, v in bound.arguments.items() if not isinstance(v, torch.Tensor)
            }

            # GPU テンソルが無い / op 判定不能 → eager
            op_type = _resolve_op_type()
            if not tensors or op_type is None or not tensors[0].is_cuda:
                _log.debug("eager fallback fn=%s op=%s", fn.__qualname__, op_type)
                return fn(*args, **kwargs)

            key = tuple((tuple(t.shape), str(t.dtype)) for t in tensors)
            if key not in compiled:
                # economic: eager を 1 回タイムしてから探索判断
                if objective == "economic" and min_invocations > 0:
                    baseline_us = _time_eager(fn, args, kwargs)
                    search_cost_s = budget * per_candidate_s
                    decision = should_run_search(min_invocations, search_cost_s, baseline_us)
                    _progress_fn = progress or (lambda _m: None)
                    _progress_fn(f"adoption: {decision.reason}")
                    _log.info("adoption fn=%s: %s", fn.__qualname__, decision.reason)
                    if not decision.should_search:
                        compiled[key] = None
                    else:
                        compiled[key] = _build(
                            op_type,
                            tensors,
                            constants,
                            budget,
                            repo,
                            search,
                            min_speedup,
                            python_executable,
                            progress,
                        )
                else:
                    compiled[key] = _build(
                        op_type,
                        tensors,
                        constants,
                        budget,
                        repo,
                        search,
                        min_speedup,
                        python_executable,
                        progress,
                    )
            kfn = compiled[key]
            if kfn is None:
                return fn(*args, **kwargs)
            return kfn(*tensors, **constants)

        wrapper._forge_compiled = compiled  # type: ignore[attr-defined]  # テスト/内省用
        return wrapper

    return deco


def _build(
    op_type: str,
    tensors: list[Any],
    constants: dict[str, Any],
    budget: int,
    repo: KernelRepository | None,
    search: CandidateGenerator | None,
    min_speedup: float,
    python_executable: str | None,
    progress: Callable[[str], None] | None,
) -> Callable[..., Any] | None:
    """この shape に対して最速カーネルを探索し、in-process 実行関数を返す（無ければ None）。"""
    input_specs = tuple(TensorSpec.from_tensor(t) for t in tensors)
    out = TensorSpec.from_tensor(tensors[0])
    spec = KernelSpec(
        op_type=op_type,
        input_specs=input_specs,
        output_specs=(out,),
        constants=constants,
        graph_hash=f"{op_type}_v1",  # 計算の識別子。テンプレ無効化は CacheKey.template_hash が担う
        constraints=(),
    )

    _log.info("build start op=%s", op_type)
    orch = Orchestrator(
        repo=repo,
        min_speedup=min_speedup,
        python_executable=python_executable,
        progress=progress or (lambda _m: None),
    )
    result = orch.optimize(spec, budget=budget, search=search)
    if result.best_params is None:
        _log.info("build no best op=%s — falling back to eager", op_type)
        return None
    _log.info("build complete op=%s best=%s", op_type, result.best_params)
    code = generate(spec, result.best_params)
    return load_kernel_fn(code)


def _time_eager(
    fn: Callable[..., Any],
    args: tuple,
    kwargs: dict,
    warmup: int = 3,
    repeat: int = 10,
) -> float:
    """eager 関数を warmup + repeat 回 CUDA Event で計測し中央値（µs）を返す。

    単一計測は ±10-20% のノイズがあるため、中央値を使って安定した推定を得る。
    """
    import statistics

    import torch

    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)  # ms -> µs

    return statistics.median(samples)
