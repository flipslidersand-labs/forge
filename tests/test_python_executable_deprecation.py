"""python_executable 引数の非推奨化 (#364)。GPU 不要。

削除は次の major。それまでは従来どおり効かせたうえで DeprecationWarning を出す。
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest
import torch

import forge
import forge.decorator as decorator_mod
from forge.cache.repository import KernelRepository
from forge.orchestrator import Orchestrator


def test_optimize_warns_when_python_executable_given() -> None:
    with pytest.warns(DeprecationWarning, match="python_executable") as rec:
        forge.optimize(python_executable=sys.executable)
    assert rec[0].filename == __file__  # 警告の発生位置が呼び出し元（stacklevel の回帰防止）


def test_optimize_silent_without_python_executable() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        forge.optimize()


def test_orchestrator_warns_and_still_honors_value(tmp_path: Path) -> None:
    repo = KernelRepository(tmp_path / "a.db")
    with pytest.warns(DeprecationWarning, match="python_executable") as rec:
        orch = Orchestrator(repo=repo, python_executable="/x/python")
    assert rec[0].filename == __file__
    assert orch.python_executable == "/x/python"  # 非推奨でも動作は変えない
    orch.close()
    repo.close()


def test_orchestrator_silent_without_python_executable(tmp_path: Path) -> None:
    repo = KernelRepository(tmp_path / "a.db")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        orch = Orchestrator(repo=repo)
    assert orch.python_executable is None
    orch.close()
    repo.close()


def test_build_passes_value_to_orchestrator_without_second_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """optimize -> _build -> Orchestrator の経路: 値は渡り、警告は二重に出ない。"""
    seen: dict[str, object] = {}

    class FakeOrchestrator:
        python_executable: str | None = None

        def __init__(self, **_kw: object) -> None:
            pass

        def __enter__(self) -> FakeOrchestrator:
            return self

        def __exit__(self, *_a: object) -> None:
            seen["python_executable"] = self.python_executable

        def optimize(self, *_a: object, **_kw: object) -> object:
            return type("R", (), {"best_params": None})()

    monkeypatch.setattr(decorator_mod, "Orchestrator", FakeOrchestrator)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        out = decorator_mod._build(
            lambda x: x, "elementwise", [torch.zeros(4)], {}, 1, None, None, 1.03, "/x/python", None
        )
    assert out is None  # best_params なし -> eager フォールバック
    assert seen["python_executable"] == "/x/python"
