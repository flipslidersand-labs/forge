from __future__ import annotations

import hashlib
import hmac
import importlib.util
import os
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path

# プロセス起動時に一度だけ生成する秘密鍵。
# サブプロセスや永続化ストレージには共有しない（in-process 用）。
_HMAC_KEY: bytes = os.urandom(32)

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
    tag = hmac.new(_HMAC_KEY, code.encode(), hashlib.sha256).hexdigest()
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
    expected = hmac.new(_HMAC_KEY, code.encode(), hashlib.sha256).hexdigest()
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
