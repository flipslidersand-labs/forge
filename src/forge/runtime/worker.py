"""Subprocess ワーカープロセスの結果型・JSON (de)serialization。

Triton カーネルの検証・ベンチマークは子プロセス（ワーカー）で隔離して実行。
親プロセス（orchestrator）はワーカーの結果を JSON で受け取り、
BenchmarkResult や ExtendedBaselineResult に変換する。

このモジュールは結果型と serialization ロジックを集約。
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger("forge.runtime")

from forge.benchmark.statistics import BenchmarkResult


@dataclass
class ExtendedBaselineResult:
    """torch.compile 等の重い baseline を 1 回だけ計測した結果。

    計測に失敗した場合は ``error`` フィールドに理由が入り、``benchmark`` の値は
    すべて 0 になる。利用側は ``eb.failed`` を確認してから数値を使うこと。
    """

    name: str
    benchmark: BenchmarkResult
    compile_time_s: float = 0.0
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "benchmark": self.benchmark.to_dict(),
            "compile_time_s": self.compile_time_s,
        }
        if self.error is not None:
            d["error"] = self.error
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExtendedBaselineResult:
        return cls(
            name=str(d["name"]),
            benchmark=BenchmarkResult.from_dict(d["benchmark"]),
            compile_time_s=float(d.get("compile_time_s", 0.0)),
            error=d.get("error"),
        )


@dataclass
class WorkerResult:
    success: bool
    correct: bool = False
    max_abs_diff: float | None = None
    failures: list[dict[str, Any]] = field(default_factory=list)
    candidate: BenchmarkResult | None = None
    baseline: BenchmarkResult | None = None
    baseline_name: str | None = None
    error: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorkerResult:
        if not d.get("success"):
            return cls(success=False, error=d.get("error"))
        return cls(
            success=True,
            correct=bool(d.get("correct", False)),
            max_abs_diff=d.get("max_abs_diff"),
            failures=list(d.get("failures", [])),
            candidate=BenchmarkResult.from_dict(d["candidate"]) if "candidate" in d else None,
            baseline=BenchmarkResult.from_dict(d["baseline"]) if "baseline" in d else None,
            baseline_name=d.get("baseline_name"),
        )


def run_in_worker(
    kernel_code: str,
    op_type: str,
    benchmark_input: list[dict[str, Any]],
    constants: dict[str, Any],
    correctness_cases: list[dict[str, Any]] | None = None,
    task: str = "full",
    warmup: int = 25,
    repeat: int = 200,
    tolerance: dict[str, Any] | None = None,
    timeout_s: float = 60.0,
    python_executable: str | None = None,
) -> WorkerResult:
    """生成カーネルを使い捨て subprocess で実行する。

    benchmark_input はタイミング計測に使う代表入力。correctness_cases を渡すと
    各ケースで正確性を検証する（省略時は benchmark_input で 1 回検証）。
    CUDA エラー・タイムアウト・コンパイル失敗のいずれも WorkerResult(success=False)
    として返し、親プロセスは生き残る。python_executable 省略時は現在のインタプリタ。
    """
    payload = json.dumps(
        {
            "kernel_code": kernel_code,
            "op_type": op_type,
            "benchmark_input": benchmark_input,
            "correctness_cases": correctness_cases,
            "constants": constants,
            "task": task,
            "warmup": warmup,
            "repeat": repeat,
            "tolerance": tolerance or {"atol": 2e-3, "rtol": 1e-2, "equal_nan": False},
        }
    )
    exe = python_executable or sys.executable
    try:
        proc = subprocess.run(
            [exe, "-m", "forge.runtime._worker_entry"],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        _log.warning("worker timeout op=%s after %.1fs", op_type, timeout_s)
        return WorkerResult(success=False, error=f"timeout after {timeout_s}s")

    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        detail = stderr[:4096] if stderr else f"exit {proc.returncode}"
        _log.warning("worker crashed op=%s rc=%d", op_type, proc.returncode)
        return WorkerResult(success=False, error=f"worker crashed: {detail}")

    stdout = proc.stdout.strip()
    if not stdout:
        stderr = proc.stderr.strip()
        detail = stderr[:4096] if stderr else "empty stdout"
        _log.warning("bad worker output op=%s: empty stdout", op_type)
        return WorkerResult(success=False, error=f"bad worker output: {detail}")
    try:
        return WorkerResult.from_dict(json.loads(stdout.splitlines()[-1]))
    except (json.JSONDecodeError, IndexError, KeyError) as exc:
        snippet = stdout[:4096]
        _log.warning("bad worker output op=%s: %r", op_type, exc)
        return WorkerResult(success=False, error=f"bad worker output: {exc!r} stdout={snippet!r}")


def run_extended_baseline_in_worker(
    op_type: str,
    benchmark_input: list[dict[str, Any]],
    constants: dict[str, Any],
    warmup: int = 25,
    repeat: int = 200,
    timeout_s: float = 180.0,
    python_executable: str | None = None,
) -> list[ExtendedBaselineResult]:
    """torch.compile(reference) を subprocess で 1 回だけ計測する。

    GPU 必須。torch.compile の初回コンパイル時間も記録する。
    失敗した場合は空リストを返す（探索は継続できる）。
    """
    payload = json.dumps(
        {
            "op_type": op_type,
            "benchmark_input": benchmark_input,
            "constants": constants,
            "task": "extended_baseline",
            "warmup": warmup,
            "repeat": repeat,
        }
    )
    exe = python_executable or sys.executable
    try:
        proc = subprocess.run(
            [exe, "-m", "forge.runtime._worker_entry"],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return []

    if proc.returncode != 0:
        return []

    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        if not result.get("success"):
            return []
        return [ExtendedBaselineResult.from_dict(b) for b in result.get("baselines", [])]
    except (json.JSONDecodeError, IndexError, KeyError):
        return []
