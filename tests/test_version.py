"""版数はインストール済みメタデータ(git タグ由来)から得る(#390)。"""

from __future__ import annotations

import pytest

import forge


def test_version_is_non_empty_string() -> None:
    assert isinstance(forge.__version__, str) and forge.__version__


@pytest.mark.parametrize(
    ("ver", "expected"),
    [
        ("1.20.5", ("1", "20", "5")),
        ("1.20.6.dev28+g8ac54d28a", ("1", "20", "6")),
        ("0.0.0+unknown", ("0", "0", "0")),
        ("2.0", ("2", "0", "0")),
    ],
)
def test_version_info_tolerates_dev_and_unknown(
    monkeypatch: pytest.MonkeyPatch, ver: str, expected: tuple[str, str, str]
) -> None:
    monkeypatch.setattr(forge, "__version__", ver)
    info = forge.version_info()
    assert (info["major"], info["minor"], info["patch"]) == expected
