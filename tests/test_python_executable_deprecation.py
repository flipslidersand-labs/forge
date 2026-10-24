"""python_executable 引数の非推奨化 (#364)。GPU 不要。

削除は次の major。それまでは従来どおり効かせたうえで DeprecationWarning を出す。
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

import forge
from forge.cache.repository import KernelRepository
from forge.orchestrator import Orchestrator


def test_optimize_warns_when_python_executable_given() -> None:
    with pytest.warns(DeprecationWarning, match="python_executable"):
        forge.optimize(python_executable=sys.executable)


def test_optimize_silent_without_python_executable() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        forge.optimize()


def test_orchestrator_warns_and_still_honors_value(tmp_path: Path) -> None:
    with pytest.warns(DeprecationWarning, match="python_executable"):
        orch = Orchestrator(repo=KernelRepository(tmp_path / "a.db"), python_executable="/x/python")
    assert orch.python_executable == "/x/python"  # 非推奨でも動作は変えない
    orch.close()


def test_orchestrator_silent_without_python_executable(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        orch = Orchestrator(repo=KernelRepository(tmp_path / "a.db"))
    assert orch.python_executable is None
    orch.close()
