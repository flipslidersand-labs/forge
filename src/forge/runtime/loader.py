from __future__ import annotations

import hashlib
import hmac
import importlib.util
import logging
import os
import tempfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path

_log = logging.getLogger("forge.runtime")

_KEY_LEN = 32

# HMAC 鍵のキャッシュ。初回使用時に ``_get_hmac_key`` が設定する（import 時は読まない）。
# テストからは直接差し替え可能（None に戻すと再読み込みされる）。
_HMAC_KEY: bytes | None = None


def _hmac_key_path() -> Path:
    """鍵ファイルのパス。``FORGE_HMAC_KEY_PATH`` で上書き可（既定 ``~/.forge/hmac.key``）。"""
    env = os.environ.get("FORGE_HMAC_KEY_PATH")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".forge" / "hmac.key"


def _read_key_file(path: Path) -> bytes:
    key = path.read_bytes()
    if len(key) != _KEY_LEN:
        raise ValueError(f"鍵ファイルの長さが {_KEY_LEN} バイトではありません: {len(key)}")
    return key


def _load_or_create_key(path: Path) -> bytes:
    """鍵ファイルを読む。無ければ 0600 で原子的に作成する（競合時は読み直す）。"""
    try:
        return _read_key_file(path)
    except FileNotFoundError:
        pass
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    key = os.urandom(_KEY_LEN)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        # 他プロセスが先に作成した。書き込み完了前に読むと短い場合があるので少し待つ。
        for _ in range(20):
            try:
                return _read_key_file(path)
            except ValueError:
                time.sleep(0.05)
        return _read_key_file(path)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(key)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return key


def _get_hmac_key() -> bytes:
    """HMAC 鍵を返す（初回使用時に鍵ファイルから読み込む）。

    鍵ファイルが読めない・長さ不正などの場合は警告を一度だけ出し、
    このプロセス限りのメモリ上の乱数鍵にフォールバックする。
    """
    global _HMAC_KEY
    if _HMAC_KEY is not None:
        return _HMAC_KEY
    path = _hmac_key_path()
    try:
        _HMAC_KEY = _load_or_create_key(path)
    except (OSError, ValueError) as exc:
        _log.warning(
            "HMAC 鍵ファイル %s を使えないため、このプロセス限りの鍵を使います"
            "（永続キャッシュは再利用されません）: %s",
            path,
            exc,
        )
        _HMAC_KEY = os.urandom(_KEY_LEN)
    return _HMAC_KEY


# HMAC タグと本文を区切るセパレータ。コード中に出現しない文字列を選択。
_SEP = "||forge-hmac||"


def sign_kernel_code(code: str) -> str:
    """カーネルコードに HMAC タグを付与して返す。

    返却値は ``<hmac_hex>||forge-hmac||<code>`` 形式。
    キャッシュへ書き込む前に呼び出すこと。

    Args:
        code: 署名対象のカーネルソース文字列。

    Returns:
        ``<hmac_hex>||forge-hmac||<code>`` 形式の署名済み文字列。
    """
    tag = hmac.new(_get_hmac_key(), code.encode(), hashlib.sha256).hexdigest()
    return f"{tag}{_SEP}{code}"


def verify_kernel_code(signed: str) -> str:
    """HMAC タグを検証し、元のカーネルコードを返す。

    タグが不正・欠損の場合は ``ValueError`` を送出する。
    キャッシュから読み出した直後に呼び出すこと。

    Args:
        signed: ``sign_kernel_code`` が返した署名済み文字列。

    Returns:
        検証通過後の元のカーネルソース文字列。

    Raises:
        ValueError: HMAC タグが不正または欠損している場合。
    """
    if _SEP not in signed:
        raise ValueError(
            "kernel_code に HMAC タグが見つかりません — 改ざんまたは未署名のエントリです"
        )
    tag_hex, _, code = signed.partition(_SEP)
    expected = hmac.new(_get_hmac_key(), code.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(tag_hex, expected):
        raise ValueError("kernel_code の HMAC 検証に失敗しました — 改ざんの可能性があります")
    return code


def load_kernel_fn(code: str) -> Callable[..., object]:
    """生成された Triton モジュール文字列を一時 .py として import し kernel_fn を返す。

    @triton.jit は inspect でソースをファイルから読むため、実在するファイル経由で
    import する必要がある（インライン exec は不可 — Issue #3）。worker（subprocess）と
    デコレータ（in-process）の両方がこれを使う。in-process 実行は、キャッシュ済みの
    検証通過カーネルのみを対象とすること。

    一時ファイルはモジュールロード直後に削除する（SIGKILL 耐性・ディスク蓄積防止）。
    """
    tmp_dir = Path(tempfile.gettempdir()) / "forge_kernels"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mod_path = tmp_dir / f"kernel_{uuid.uuid4().hex}.py"
    mod_path.write_text(code)
    try:
        spec = importlib.util.spec_from_file_location(mod_path.stem, str(mod_path))
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.kernel_fn
    finally:
        mod_path.unlink(missing_ok=True)
