# ADR-011: コスト考慮型探索 (Cost-Aware Multiobjective Search)

## ステータス

Accepted

## 背景

探索ループが候補を増やすほど、評価コスト（ベンチマーク実行時間 + LLM トークン）が
無制限に増大する。特に DS1（RTX 4060）での長時間ベンチマークは探索予算を急速に
消費するため、コストを明示的にモデル化する必要がある。

Issue #168 では「速度倍率」と「評価コスト」の二目的最適化を要件とし、
既存の `ParetoFrontier` をコスト軸で補完する形で設計を進めた。

## 決定

### 1. CostModel: SQLite キャッシュによるコスト予測

- `~/.cache/forge/cost_profile.db` に shape/variant ごとの実測ベンチマーク時間を保存
- キャッシュミス時は `default_ms × (block_size/256) × (num_warps/4)` の簡易ヒューリスティックを使用
- UPSERT により最新の実測値が常に参照される
- ADR-002（SQLite キャッシュ基盤）と同じ設計方針を踏襲

### 2. BudgetTracker: 壁時計ベースの早期打ち切り

- `max_total_s`（秒）で探索全体の上限を設定し、`time.monotonic()` で経過を追跡
- `should_skip(estimated_ms)` が True を返したら当該候補の評価をスキップ
- `max_total_s=None` で無制限（後方互換）

### 3. scalarize(): 線形スカラー化

```
score = speedup - λ × cost
```

- `λ=0.1` をデフォルトとして設定（コスト感度を低く抑え、速度改善を優先）
- 呼び出し元が `λ` を調整することで予算/性能のトレードオフを制御可能

## 却下した代替案

| 案                                 | 却下理由                                                             |
| ---------------------------------- | -------------------------------------------------------------------- |
| ε-制約法（コストを制約として固定） | ε 設定が ad-hoc になりやすく、`λ` で連続的に調整できる方が扱いやすい |
| Hypervolume 最大化                 | 2 目的なら `scalarize` で十分。計算コストに見合わない                |
| コスト予測に ML モデルを使用       | 過学習リスクあり。SQLite 実績ベースのヒューリスティックで精度十分    |

## 影響

- `src/forge/search/cost_model.py` を新規追加（既存モジュールへの影響なし）
- 探索ループ（`Orchestrator`）が使うのは `BudgetTracker.budget_exhausted`（壁時計による
  打ち切り）だけ。`CostModel.predict_cost()` → `BudgetTracker.should_skip()` による
  候補ごとのガードは、探索ループに配線しない（2026-11-28 改訂。下の改訂履歴を参照）
- `CostModel`・`BudgetTracker.should_skip()`・`scalarize()` は、呼び出し側が単体で使う部品
  として残す。使い方は `examples/cost_aware_optimization.py`
- `forge.benchmark.pareto`（`ParetoFrontier`・`CandidateWithCost`）も探索ループには統合しない。
  計測結果を呼び出し側で比較するための、単体で使う部品とする
- テストは `tempfile.TemporaryDirectory` を使い DB を分離（副作用なし）

## 改訂履歴

### 2026-11-28: 候補ごとのコストガードを探索ループに配線しない

当初の「影響」は、探索ループが `predict_cost()` → `should_skip()` の順に呼んでガードすると
書いていたが、実装では配線されていなかった。配線せず、記述を実装に合わせる。

- 理由 1: 予測のもとになる実測値を保存する処理（`CostModel.record()`）を呼ぶ箇所が `src` に無い。
  配線しても予測は常に簡易ヒューリスティックの値になり、実測に基づくガードにならない
- 理由 2: `src` には `CostModel` を生成する箇所が無い(利用例は `examples/`、ほかはテストだけ)。必要とする利用の実績が無い
- 理由 3: 探索全体の上限（`max_total_s`）は `budget_exhausted` ですでに効いている
- 却下した案: 探索ループに配線し、`record()` で実測値を保存する（探索結果と予算の挙動が変わる。
  利用の要望が出てから行う）
- 戻し方: 利用の要望が出た時点で、配線を新しい Issue として起こす
- 方針の記録: [ADR-012](ADR-012-development-direction.md) の決定 5
