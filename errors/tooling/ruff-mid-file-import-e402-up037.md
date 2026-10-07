---
title: "ruff E402/UP037: ファイル途中 import と quoted 型注釈"
tags: [ruff, python, ci, forge]
severity: low
date: "2026-08-11"
---

## 症状

CI lint (`ruff check`) が以下で失敗:

```
E402 Module level import not at top of file
  --> tests/test_cost_aware_search.py:246:1

UP037 Remove quotes from type annotation
  --> src/forge/search/cost_model.py:108:28
```

## 原因

### E402

テストファイルでセクション区切りコメント（`# ── Section ──`）の後に import を書いた。
ruff はファイル内のどこに書かれた import も「先頭以外 → E402」と判定する。

```python
# NG: コメントで区切っても mid-file import はダメ
class TestA: ...

# ── 追加テスト ──
import tempfile  # E402!
from forge.x import Y
```

### UP037

`from __future__ import annotations` が有効な場合、戻り値型注釈の文字列クォートは不要。

```python
# NG
def __enter__(self) -> "CostModel":

# OK
def __enter__(self) -> CostModel:
```

## 解決策

### E402

全 import を必ずファイル先頭（docstring の直後）にまとめる。

```python
"""モジュールのdocstring。"""

from __future__ import annotations

import tempfile
from pathlib import Path

from forge.x import Y  # ← 全部ここに
```

### UP037

`from __future__ import annotations` を書いている場合は型注釈を素直に書く。
`ruff check --fix` で自動修正される。

## 予防

```bash
# push 前に必ず実行
ruff check --fix src/ tests/
ruff format src/ tests/
```

`--fix` が直せない違反（E731 など）は手動修正が必要。
