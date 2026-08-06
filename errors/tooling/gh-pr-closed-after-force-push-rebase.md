---
title: "rebase + force-push 後に PR が CLOSED になり別番号でマージされた"
tags: [git, github, pr, rebase]
severity: medium
date: "2026-07-18"
---

## 症状

`docs/benchmark-results` ブランチを `git rebase origin/master` + `git push --force-with-lease` した後、
PR #43 の状態が `mergeable: CONFLICTING / mergeStateStatus: DIRTY` のまま変わらず、
バックグラウンドの `until ... gh pr merge 43` が完了すると PR #43 が CLOSED（mergedAt=null）になっており、
代わりに PR #44 が MERGED になっていた。

## 原因

GitHub が force-push 後の SHA を再計算する間、PR の merge state が古い状態にキャッシュされる。
`until` ループの `grep -q "mergeable"` 条件が、CLOSED 状態の PR の JSON でも `"mergeable"` という文字列にマッチして
早期終了し、CONFLICTING のまま `gh pr merge` が走った可能性がある。
GitHub 側でスカッシュマージが別 PR として処理された可能性もある。

## 解決策

rebase 後は `gh pr view <N> --json state,mergedAt` で実際の state を確認してからマージする。
`until` ループで待つ場合は `mergeable` 文字列マッチではなく `"MERGEABLE"` を明示的に grep する:

```bash
until gh pr view 43 --repo ... --json mergeable | grep -q '"MERGEABLE"'; do sleep 5; done
gh pr merge 43 --squash --delete-branch
```

## 予防

- force-push 後は数十秒待って `gh pr view --json mergeable,mergeStateStatus` で CLEAN になったことを確認
- `grep -q "mergeable"` は文字列として "mergeable" にマッチするため、CONFLICTING でも通過する。必ず値を絞ること
- PR が意図せず CLOSED になった場合、`gh pr list --state merged` で実際にどの番号でマージされたか確認
