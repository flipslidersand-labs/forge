# ADR-006: 探索戦略の二層分離 — 生成器（Bayesian TPE）と実行戦略（Successive Halving）

- **日付**: 2026-08-09
- **状態**: Accepted

## 背景

Phase 5 完了時点での探索器は GridSearch / RandomSearch / LLMGenerator の 3 種。
いずれも「候補リストを返す生成器」として `CandidateGenerator` Protocol に適合している。

2 つの要望があった：

1. **Bayesian 最適化** — 過去の評価結果（history）をサロゲートモデルで活用し、次候補を賢く選ぶ
2. **Successive Halving (SHA)** — 候補を多数生成し、warmup/repeat を段階的に増やして絞り込む

これらを実装するにあたり、「1 つの抽象に統合するか、層を分けるか」を決定する必要があった。

## 検討した選択肢

### 案 A: 両方を CandidateGenerator として実装

SHA も `generate()` の内部で軽量ベンチを走らせ、上位候補だけを返す。

**メリット**: API 統一、既存 `Orchestrator.optimize()` をそのまま使える。

**デメリット**:
- `generate()` は pure な候補生成の責務を持つべきで、GPU 実行を含むと副作用が大きい
- SHA 内部で `run_in_worker` を呼ぶと `Orchestrator` のライフサイクル（キャッシュ書き込み、通知）を迂回してしまう
- テストで GPU を避けにくくなる

### 案 B: 両方を Orchestrator メソッドとして実装

`optimize()` / `optimize_rounds()` と同様に `optimize_bayesian()` / `optimize_sha()` を追加。

**デメリット**: Bayesian は「候補の選び方」が本質であり、GridSearch と同じ `optimize()` に渡せない。探索器の差し替えができない。

### 案 C（採択）: 責務で分割 — 生成器 と 実行戦略

| 概念 | 担う責務 | 実装場所 |
|------|----------|----------|
| **BayesianGenerator** | 何を試すか（候補選定） | `_BaseGenerator` サブクラス |
| **optimize_sha()** | どう測るか（warmup/repeat の段階的拡大） | `Orchestrator` 新メソッド |

`BayesianGenerator` は `CandidateGenerator` Protocol に適合し、`GridSearch` の完全な drop-in 代替になる。
`optimize_sha()` は任意の `CandidateGenerator`（GridSearch / Bayesian / LLM）を受け取り、SHA の実行戦略を適用する。

組み合わせ例:
```python
orch.optimize_sha(spec, search=BayesianGenerator())   # Bayesian 提案 + SHA フィルタ
orch.optimize_sha(spec, search=GridSearch())           # 全列挙 + SHA フィルタ
orch.optimize(spec, search=BayesianGenerator())        # Bayesian のみ（SHA なし）
```

## 決定

**案 C を採択**。

### BayesianGenerator の設計

- `_BaseGenerator` を継承し、`_propose()` のみ実装する
- Optuna の `TPESampler` をサロゲートモデルとして使用
- `history` の `(SearchParams, median_us)` を Optuna `trial.suggest_*` + `study.tell()` で注入
- `n_startup_trials` 回未満はランダムサンプリング（cold start 期）
- `optuna` は optional extra `[search]` に追加し、本体に依存させない

```python
# pyproject.toml
[project.optional-dependencies]
search = ["optuna>=3.0"]
```

cold start 問題（履歴なしの初回）は `RandomSearch` にフォールバックせず、
Optuna 自身の startup ランダムサンプリングに任せる（Optuna デフォルト動作）。

### optimize_sha() の設計

```
initial_budget 候補
    ↓ warmup=5, repeat=20（quick bench）
上位 ceil(n/2) を選択
    ↓ warmup=10, repeat=50
上位 ceil(n/4) を選択
    ↓ warmup=25, repeat=200（正式ベンチ）
best_params を確定
```

- ラウンド数とハーフィング比率はパラメータ化（デフォルト: 3 ラウンド / 上位 50%）
- `_prepare()` / `_finalize()` を流用してキャッシュ・通知を維持
- `SearchResult` を返して既存 API と互換

## 却下した代替技術

| 技術 | 却下理由 |
|------|----------|
| GPyOpt | メンテナンス停止 |
| scikit-optimize (skopt) | Python 3.11+ での型問題、開発停滞 |
| HyperBand（Successive Halving + 複数 bracket） | bracket 管理が複雑。forge の探索空間サイズ（〜数百候補）では SHA で十分 |
| SMAC3 | 依存が重い、Triton パラメータ空間との統合が煩雑 |

## 結果として生まれるアーキテクチャ

```
CandidateGenerator (Protocol)
├── GridSearch          ← 既存
├── RandomSearch        ← 既存
├── LLMGenerator        ← 既存（_BaseGenerator 継承）
├── OllamaGenerator     ← 既存（_BaseGenerator 継承）
└── BayesianGenerator   ← 新規（_BaseGenerator 継承、optuna TPE）

Orchestrator
├── optimize()          ← 既存（任意 CandidateGenerator を受け取る）
├── optimize_rounds()   ← 既存（LLMGenerator 専用）
└── optimize_sha()      ← 新規（任意 CandidateGenerator + SHA 実行戦略）
```

## 関連

- ADR-005: LLM 候補生成の構造化（`_BaseGenerator` の設計根拠）
- Issue #126: この ADR の Issue
- Issue #127: BayesianGenerator 実装
- Issue #128: Orchestrator.optimize_sha() 実装
- Issue #129: 性能比較ベンチマーク
