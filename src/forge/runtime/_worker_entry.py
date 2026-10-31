"""subprocess worker のエントリポイント。

親から stdin に JSON を受け取り、生成カーネルをコンパイル・複数ケースで正確性
検証・ベンチマークして結果を stdout に JSON で返す。CUDA illegal memory access 等で
このプロセスが死んでも親は生き残る（ADR-003）。

@triton.jit は inspect でソースをファイルから読むため、カーネルコードは必ず実在する
一時 .py ファイルとして import する（Issue #3 の申し送り）。

実行: <python> -m forge.runtime._worker_entry  < payload.json
"""

from __future__ import annotations

import json
import sys
from typing import Any


def _build_tensor(spec: dict[str, Any], torch: Any):
    dtype = getattr(torch, spec["dtype"])
    shape = tuple(spec["shape"])
    init = spec.get("init", "randn")
    scale = float(spec.get("scale", 1.0))
    seed = int(spec.get("seed", 0))
    if init == "randn":
        gen = torch.Generator(device="cuda").manual_seed(seed)
        return torch.randn(shape, dtype=dtype, device="cuda", generator=gen) * scale
    if init == "ones":
        return torch.ones(shape, dtype=dtype, device="cuda") * scale
    if init == "zeros":
        return torch.zeros(shape, dtype=dtype, device="cuda")
    raise ValueError(f"unknown init: {init}")


def _measure_extended_baselines(payload: dict, torch, measure, get_reference) -> None:
    """task="extended_baseline": torch.compile(reference) を 1 回計測して JSON 出力。"""
    import time

    op_type = payload["op_type"]
    constants = payload.get("constants", {})
    warmup = int(payload.get("warmup", 25))
    repeat = int(payload.get("repeat", 200))

    bench_tensors = [_build_tensor(s, torch) for s in payload["benchmark_input"]]
    ref_fn = get_reference(op_type)
    baselines = []

    # torch.compile(reference)
    try:
        compiled_fn = torch.compile(ref_fn)
        t0 = time.perf_counter()
        compiled_fn(*bench_tensors, **constants)  # 初回: compile が走る
        torch.cuda.synchronize()
        compile_time_s = time.perf_counter() - t0

        result = measure(lambda: compiled_fn(*bench_tensors, **constants), warmup, repeat)
        baselines.append(
            {
                "name": "torch.compile(reference)",
                "benchmark": result.to_dict(),
                "compile_time_s": compile_time_s,
            }
        )
    except (RuntimeError, OSError, ValueError) as e:
        baselines.append(
            {
                "name": "torch.compile(reference)",
                "benchmark": {"median_us": 0.0, "p20_us": 0.0, "p80_us": 0.0, "p95_us": 0.0},
                "compile_time_s": 0.0,
                "error": str(e),
            }
        )

    print(json.dumps({"success": True, "task": "extended_baseline", "baselines": baselines}))


def main() -> None:
    payload = json.loads(sys.stdin.read())
    try:
        import torch

        from forge.benchmark.timer import measure
        from forge.ops.registry import OP_REGISTRY
        from forge.runtime.loader import load_kernel_fn
        from forge.runtime.reference import baseline_name, get_baseline, get_reference

        op_type = payload["op_type"]

        if op_type not in OP_REGISTRY:
            print(json.dumps({"success": False, "error": f"unknown op_type: {op_type!r}"}))
            return

        constants = payload.get("constants", {})
        task = payload.get("task", "full")
        tol = payload.get("tolerance", {"atol": 2e-3, "rtol": 1e-2, "equal_nan": False})

        if task == "extended_baseline":
            _measure_extended_baselines(payload, torch, measure, get_reference)
            return

        kernel_fn = load_kernel_fn(payload["kernel_code"])
        reference = get_reference(op_type)

        # --- 正確性検証（複数ケース） ---
        cases = payload.get("correctness_cases") or [
            {"name": "primary", "input_specs": payload["benchmark_input"]}
        ]
        failures: list[dict[str, Any]] = []
        max_abs_diff = 0.0
        for case in cases:
            tensors = [_build_tensor(s, torch) for s in case["input_specs"]]
            out_c: Any = kernel_fn(*tensors, **constants)
            out_r: Any = reference(*tensors, **constants)
            torch.cuda.synchronize()
            diff = (out_c.float() - out_r.float()).abs().max().item()
            max_abs_diff = max(max_abs_diff, diff)
            ok = bool(
                torch.allclose(
                    out_c.float(),
                    out_r.float(),
                    atol=float(tol["atol"]),
                    rtol=float(tol["rtol"]),
                    equal_nan=bool(tol.get("equal_nan", False)),
                )
            )
            if not ok:
                failures.append({"case": case["name"], "max_diff": diff})

        correct = len(failures) == 0
        result: dict[str, Any] = {
            "success": True,
            "correct": correct,
            "max_abs_diff": max_abs_diff,
            "failures": failures,
        }

        # --- ベンチマーク（正確な候補のみ） ---
        if task in ("benchmark", "full") and correct:
            warmup = int(payload.get("warmup", 25))
            repeat = int(payload.get("repeat", 200))
            bench_tensors = [_build_tensor(s, torch) for s in payload["benchmark_input"]]
            baseline = get_baseline(op_type)
            cand = measure(lambda: kernel_fn(*bench_tensors, **constants), warmup, repeat)
            base = measure(lambda: baseline(*bench_tensors, **constants), warmup, repeat)
            result["candidate"] = cand.to_dict()
            result["baseline"] = base.to_dict()
            result["baseline_name"] = baseline_name(op_type)

        print(json.dumps(result))
    except (RuntimeError, OSError, ValueError, ImportError, AttributeError) as e:
        import traceback

        error_type = "cuda_oom" if type(e).__name__ == "OutOfMemoryError" else type(e).__name__
        print(
            json.dumps(
                {
                    "success": False,
                    "error": f"{type(e).__name__}: {e}",
                    "error_type": error_type,
                    "traceback": traceback.format_exc(),
                }
            )
        )


if __name__ == "__main__":
    main()
