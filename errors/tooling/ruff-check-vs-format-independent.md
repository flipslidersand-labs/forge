---
title: "ruff check と ruff format --check は独立したチェック"
tags: [ruff, python, ci, lint]
severity: medium
date: "2026-08-07"
---

## 症状

`ruff check src/ tests/` が pass しても、CI の `ruff format --check src/ tests/` が fail する。
特に `# noqa: E501` コメントで行長を抑制したとき、formatter が行を自動分割しようとしてコンフリクトが起きる。

```
unformatted: File would be reformatted
   --> src/forge/validation/test_cases.py:205:14
```

## 原因

`ruff check` は lint ルールを検査する（E501 等）。
`ruff format` はコードフォーマットを検査・適用する（Black スタイルの整形）。
両者は **独立** しており、`# noqa: E501` は lint エラーを消すが formatter の動作には影響しない。
CI が両方実行している場合、片方 pass でも落ちる。

## 解決策

`# noqa: E501` で抑制するのではなく、formatter が望む形（複数行に分割）に実際に書き換える。
または `ruff format src/ tests/` を実行してから `ruff check src/ tests/` を確認する。

```bash
ruff format src/ tests/   # フォーマット適用
ruff check src/ tests/    # lint チェック
ruff format --check src/ tests/  # CI と同じ検証
```

## 予防

ローカルで `ruff format` → `ruff check` → `ruff format --check` の順に実行して CI と同じ状態を確認してから push する。
CI の lint job が両方実行している場合は特に注意。
