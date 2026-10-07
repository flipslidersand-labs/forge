---
title: "optional extra の import が pyright reportMissingImports で落ちる"
tags: [pyright, python, ci, typecheck]
severity: medium
date: "2026-08-07"
---

## 症状

`triton` を `[gpu]` optional extra に移動した後、CI の pyright が失敗した。

```
/home/runner/work/forge/forge/src/forge/cache/key.py:29:20 - error: Import "triton" could not be resolved (reportMissingImports)
1 error, 0 warnings, 0 informations
```

## 原因

CI では `pip install -e ".[dev]"` のみ実行するため `triton` が入らない。
pyright strict モードは `try: import triton` でも `reportMissingImports` を報告する
（try/except は型チェックをバイパスしない）。

## 解決策

import 行に `# type: ignore[import-untyped]` を追加する。

```python
try:
    import triton  # type: ignore[import-untyped]
    triton_ver = triton.__version__
except ImportError:
    triton_ver = "none"
```

## 予防

optional extra に移動したモジュールを使う箇所には `type: ignore` を付ける。
または pyproject.toml の `[tool.pyright]` に `ignore = ["path/to/file.py"]` で
ファイル単位の型チェックを除外できる（ただし他エラーも隠れるため最終手段）。
