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
        config = decorator_mod._BuildConfig(
            budget=1,
            repo=None,
            search=None,
            min_speedup=1.03,
            python_executable="/x/python",
            progress=None,
        )
        out = decorator_mod._build(lambda x: x, "elementwise", [torch.zeros(4)], {}, config)
    assert out is None  # best_params なし -> eager フォールバック
    assert seen["python_executable"] == "/x/python"


def test_build_forwards_all_config_fields_to_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_BuildConfig の各フィールドが Orchestrator(コンストラクタ/optimize)へ渡る（#357）。"""
    seen: dict[str, object] = {}
    repo, search, progress = object(), object(), print

    class FakeOrchestrator:
        python_executable: str | None = None

        def __init__(self, **kw: object) -> None:
            seen["init"] = kw

        def __enter__(self) -> FakeOrchestrator:
            return self

        def __exit__(self, *_a: object) -> None:
            pass

        def optimize(self, _spec: object, **kw: object) -> object:
            seen["optimize"] = kw
            return type("R", (), {"best_params": None})()

    monkeypatch.setattr(decorator_mod, "Orchestrator", FakeOrchestrator)
    config = decorator_mod._BuildConfig(
        budget=7,
        repo=repo,  # type: ignore[arg-type]
        search=search,  # type: ignore[arg-type]
        min_speedup=1.5,
        python_executable=None,
        progress=progress,
    )
    assert decorator_mod._build(lambda x: x, "elementwise", [torch.zeros(4)], {}, config) is None
    assert seen["init"] == {"repo": repo, "min_speedup": 1.5, "progress": progress}
    assert seen["optimize"] == {"budget": 7, "search": search}
