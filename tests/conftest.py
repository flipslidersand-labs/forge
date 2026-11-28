"""テスト共通 fixture。"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_hmac_key(tmp_path_factory, monkeypatch):
    """HMAC 鍵ファイルを実環境（~/.forge）でなくテスト専用 tmp に向ける（#327）。"""
    from forge.runtime import loader

    key_path = tmp_path_factory.mktemp("hmac") / "hmac.key"
    monkeypatch.setenv("FORGE_HMAC_KEY_PATH", str(key_path))
    monkeypatch.setattr(loader, "_HMAC_KEY", None, raising=False)
