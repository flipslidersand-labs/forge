"""RMSNorm の基準実装の比較（#413: README 表で Eager が食い違った原因の切り分け）。

`F.rms_norm`（融合）・非融合の手書き版・`torch.compile(backend="eager")` を、
`examples/bench_all.py` と同じ計測方式（CUDA Event・warmup=25・repeat=200・中央値）で並べる。

    python examples/bench_rmsnorm_baselines.py
"""

import statistics
import sys

import torch
import torch.nn.functional as F

WARMUP, REPEAT = 25, 200


def measure_us(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(REPEAT):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(samples)


def manual(x, w, eps=1e-6):
    """非融合の手書き RMSNorm（forge の入力関数と同じ書き方）。"""
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * w


def bench_shape(m: int, n: int, compiled) -> None:
    x = torch.randn(m, n, dtype=torch.float16, device="cuda")
    w = torch.ones(n, dtype=torch.float16, device="cuda")

    def fused() -> torch.Tensor:
        return F.rms_norm(x, (n,), w)

    def unfused() -> torch.Tensor:
        return manual(x, w)

    def traced() -> torch.Tensor:
        return compiled(x, w)

    diff = (fused().float() - unfused().float()).abs().max().item()
    print(f"({m},{n}) max|F.rms_norm - manual| = {diff:.4g}")
    for round_ in range(2):  # 測定順の影響も見る
        r_f, r_m, r_c = measure_us(fused), measure_us(unfused), measure_us(traced)
        print(
            f"({m},{n}) round{round_}: F.rms_norm={r_f:.1f}us  "
            f"manual(unfused)={r_m:.1f}us  compile(eager)={r_c:.1f}us"
        )


def main() -> None:
    print(f"torch {torch.__version__} / {torch.cuda.get_device_name(0)}", file=sys.stderr)
    compiled = torch.compile(manual, backend="eager")
    for m, n in [(2048, 4096), (1024, 8192)]:
        bench_shape(m, n, compiled)


if __name__ == "__main__":
    main()
