---
title: "スタック PR をリベースすると同一ファイルで繰り返しコンフリクト"
tags: [git, rebase, pr, conflict]
severity: medium
date: "2026-08-07"
---

## 症状

PR #62/#63/#64/#65 のスタック（依存チェーン）を master にマージした後、
後続ブランチをリベースすると `src/forge/search/space.py` や `params.py` で
毎回同じコンフリクトが発生した。

```
CONFLICT (content): Merge conflict in src/forge/search/space.py
error: 3a737d7 を適用できませんでした...
```

## 原因

上流ブランチのコミットが squash merge されると、コミット SHA が変わる。
リベース対象ブランチの履歴には旧 SHA のコミットが残っているため、
git が「upstream に無い変更」と判断して再適用しようとする。
特に Lint 修正コミット（E501 折り返し）は上流にも含まれており、
`patch contents already upstream` でスキップされる場合と、
コンテンツが微妙に異なる（single-line vs multi-line）場合でコンフリクトになる場合がある。

## 解決策

コンフリクト発生時は HEAD（master 側）を `--ours` で採用するのが基本。
その後 `ruff format && ruff check && pyright` で整合性を確認してから push。

```bash
git checkout --ours src/forge/search/space.py
git add src/forge/search/space.py
git rebase --continue
```

gemm など「downstream 側が新たに追加した内容」が必要な場合は手動マージ。

## 予防

スタック PR を作る場合、Lint 修正コミットはなるべくベースブランチにまとめて
コミット数を最小化する。squash merge 後のリベースは衝突が多い——できれば
PR を独立させて master に直接 PR を出す設計にする。
