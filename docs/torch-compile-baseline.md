# torch.compile Baseline 比較フレーム

## 概要

forge の相対的な性能を位置づけるため、PyTorch 標準の baseline（eager）と torch.compile との比較フレーム。

## 測定方法論

### 計測パラメータ

```python
WARMUP = 25       # GPU cache warming
REPEAT = 200      # number of runs
BUDGET = 50       # forge exploration budget (seconds)
```

**採択理由:**

- **Warmup=25**: Triton、PyTorch 標準実装（5-100 推奨範囲）に準拠
- **Repeat=200**: 中央値統計に 200 回のサンプルを確保（CI 内に収まる）
- **Budget=50**: 実用的な探索時間（本番は 30-120s が一般的）

### 統計判定基準

**測定値採択**: 中央値（Median）

- **理由**: ADR-004 で明示的に選択。ノイズ耐性と外れ値除外を両立
- **代替案**: P25/P75 信頼区間（未実装、今後 Phase 4）

**Speedup 判定**:

```
forge / baseline >= 1.1  → ✅ 確実な改善
0.95 <= ratio < 1.1      → → 同等
ratio < 0.95             → ⚠️ 遅延
```

### 計測対象

**演算種別** (5種類):

1. **RMSNorm** (reduction) — 標準化層で頻出
2. **Softmax** (reduction) — Attention 機構
3. **LayerNorm** (normalization) — 毎レイヤーで実行
4. **GELU** (activation) — 非線形層
5. **SDPA** (attention) — Transformer の核

**Shape 組み合わせ** (10個):

- Dense: (2048, 4096), (1024, 8192)
- Attention: (8, 256, 64), (8, 512, 64)

**データ型**: float16 のみ（Phase 5 で float32/bfloat16 対応予定）

## 既知の制約・環境依存性

### GPU 環境

**テスト環境**: GTX 1080 (cc6.1, Triton 公式サポート外)

**他の環境への外挿**:

- RTX 4090 (cc8.9): より高速（L2 cache 容量 2倍）
- H100 (cc9.0): さらに高速（Tensor Core 最適化）
- Triton 3.x は cc7.0+ が公式サポート → cc6.1 での動作は互換性モード

### torch.compile Backend 制限

**実装状況**:

- ✅ `backend="eager"` — 実装済み（比較基準用）
- ❌ `backend="inductor"` — GTX 1080 未サポート（cc6.1）
- ❌ `backend="xpu"` — Intel GPU 環境なし

**設計**: eager backend は torch.compile の最小オーバーヘッド版として機能。本来の JIT 最適化（inductor）は環境制約で未測定。

### cudnn.benchmark 状態

**現状**: `torch.backends.cudnn.benchmark` 未設定（default=False）

**選定理由**:

- False: 再現性重視。benchmark=True だと実行ごとに異なるアルゴリズム選択
- 測定の安定性と結果の再現性を優先

**今後**: Phase 5 で `--benchmark-cudnn` フラグ追加予定

## 計測結果の解釈

### タイプ A: Decisive Win (forge >= 1.1x)

```
RMSNorm (2048, 4096): eager 815.1µs → forge 192.9µs (4.23x)
```

**意味**: forge が構造的に優れている（手書き Triton カーネル vs PyTorch 汎用実装）

### タイプ B: Marginal Loss (0.95x >= forge < 1.0x)

```
LayerNorm (2048, 4096): eager 926.5µs → forge 973.1µs (0.95x)
```

**意味**:

- forge がカーネルをフォールバック（eager 実装で計測）
- または探索予算不足で探索が十分ではない

**対策**:

- [x] `BUDGET=50` では LayerNorm 探索が不十分 → Phase 5 で動的予算配分

### タイプ C: torch.compile Anomaly

```
RMSNorm (2048, 4096): eager 815.1µs → torch.compile 1704.4µs (2.09x 遅延)
```

**意味**: torch.compile (eager backend) がトレース・コンパイル オーバーヘッド（実行 1 回目）を計測している可能性

## 統計的有意性

### 現在の実装

- **単一計測** — 各 case で中央値 1 個のみ記録
- **信頼区間なし** — P25/P75 などの分散情報なし
- **複数実行なし** — 日替わり / 環境変動の検証なし

### リスク評価

**低リスク**:

- Warmup=25 により GPU キャッシュ状態は安定
- Repeat=200 で十分なサンプル収集
- 中央値は外れ値耐性が高い

**中リスク**:

- ノイズ内の変動（±5-10%）を「改善」と誤認する可能性
- GPU 温度、他プロセスの干渉を制御していない

**推奨対策**:

- Phase 4 で複数実行（n=3-5）フレーム実装
- CI で定期実行（日次/週次）して傾向監視

## Phase ロードマップ

| Phase      | 項目                                   | 工数 | 期限       |
| ---------- | -------------------------------------- | ---- | ---------- |
| 1 (完了)   | 基本フレーム（eager vs torch.compile） | 3h   | 2026-08-30 |
| 2 (進行中) | README 整合性 + ドキュメント           | 1h   | 2026-08-30 |
| 3 (計画)   | dtype 拡張（float32/bfloat16）         | 1h   | 2026-09-15 |
| 4 (計画)   | 複数実行フレーム (n=3-5)               | 2h   | 2026-09-15 |
| 5 (計画)   | inductor backend 対応（新環境時）      | TBD  | TBD        |

## 運用ガイド

### ローカル実行

```bash
python examples/benchmark_torch_compile_autotune.py
```

**出力**: JSON + Markdown テーブル（stdout）

### CI/CD 統合

```yaml
# .github/workflows/ci.yml（計画中）
- name: "Benchmark: torch.compile baseline"
  if: contains(github.event.pull_request.labels.*.name, 'benchmark')
  run: python examples/benchmark_torch_compile_autotune.py > /tmp/results.json
  # → Artifact 保存 / 前回結果との比較
```

### 結果の見方

**条件分岐**:

1. forge/eager >= 1.1x → ✅ forge 採用推奨
2. 0.95x <= forge/eager < 1.1x → → フォールバック または個別分析
3. forge/eager < 0.95x → ⚠️ バグ報告推奨

## 既知の Future Work

- [ ] inductor backend 対応（新環境時）
- [ ] cudnn.benchmark=True 計測
- [ ] dtype 拡張（float32/bfloat16）
- [ ] 複数 GPU マトリクス実行
- [ ] Performance regression CI
- [ ] Pareto frontier 構築（複数指標: speedup vs latency）

## 参考資料

- **ADR-004** — 中央値採択の設計決定
- **Issue #166** — 本フレーム実装記録
- **torch.compile ドキュメント** — https://pytorch.org/docs/stable/generated/torch.compile.html
