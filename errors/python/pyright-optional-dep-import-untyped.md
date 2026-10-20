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

import 行に Pyright 固有の `# pyright: ignore[reportMissingImports]` を追加する。
（`# type: ignore[import-untyped]` の `import-untyped` は mypy のコード。Pyright は
PEP 484 の汎用 `# type: ignore` として扱い、その行の全診断を黙らせてしまう。）

```python
try:
    import triton  # pyright: ignore[reportMissingImports]
    triton_ver = triton.__version__
except ImportError:
    triton_ver = "none"
```

## 予防

optional extra に移動したモジュールを使う箇所には `# pyright: ignore[reportMissingImports]` を付ける（rule 指定で他の診断を隠さない）。
または pyproject.toml の `[tool.pyright]` に `ignore = ["path/to/file.py"]` で
ファイル単位の型チェックを除外できる（ただし他エラーも隠れるため最終手段）。
