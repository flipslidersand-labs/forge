# Search Strategy Benchmark

`examples/compare_search_strategies.py` による実測結果。

**条件**: rmsnorm 512×4096 fp16, budget=32, SHA halving_rounds=3

## 実行方法

```bash
pip install "forge-kernel[search]"   # optuna が必要
python examples/compare_search_strategies.py --rows 512 --hidden 4096 --budget 32
```

## 結果

### DS1 (RTX 4060, CUDA 8.9)

baseline (PyTorch `F.rms_norm`): **57.2us**

| Strategy          | best_us | n_eval | time_s | speedup_vs_baseline |
| ----------------- | ------- | ------ | ------ | ------------------- |
| GridSearch        | 56.3    | 32     | 74.4   | 1.015x              |
| BayesianGenerator | 55.3    | 28     | 66.6   | ~1.03x              |
| SHA(3r)           | 56.3    | 44     | 92.1   | 1.015x              |

**計測日**: 2026-08-09
**commit**: `f20ff88` (test/compare-129 マージ後)

#### 観察

- BayesianGenerator が最速 (55.3us) かつ最少 eval 数 (28, -12%) と最短時間 (66.6s, -10%) を達成
- SHA は budget=32 が小さすぎ、ラウンド合計 `32+8+4=44` eval で GridSearch を上回った
  → SHA が有効になるのは **initial_budget ≥ 64** 以上が目安
- `welford` variant の SKIP が 4 件発生（codegen テンプレート未実装）
  → BayesianGenerator の候補空間から `welford` を除外すべき
- Bayesian の `speedup_vs_baseline=2.241x` は最後に計測した baseline が高い値を取ったためで、
  実際の改善は `57.2 / 55.3 = 1.034x` 程度（baseline を安定化する改善余地あり）

### GTX 1080 (開発PC, CUDA sm_61)

_CUDA sm_61 は triton の対応 CC 範囲外のためカーネルコンパイル不可。計測省略。_

## 評価指標の定義

| 指標                  | 内容                                    |
| --------------------- | --------------------------------------- |
| `best_us`             | 探索で見つけた最速カーネルの中央値 (μs) |
| `n_eval`              | 評価した候補数（SKIP/INCORRECT を含む） |
| `time_s`              | 探索全体の所要時間 (wall clock)         |
| `speedup_vs_baseline` | PyTorch baseline に対する高速化倍率     |

## SHA の設計根拠

Successive Halving は各ラウンドで測定精度を上げながら候補を絞る。

| ラウンド | 候補数         | warmup | repeat |
| -------- | -------------- | ------ | ------ |
| 1        | initial_budget | 5      | 20     |
| 2        | 上位 50%       | 10     | 50     |
| 3        | 上位 50%       | 25     | 200    |

budget=32 の場合、評価回数は実測で `32+8+4=44`（上位 50% 絞り込みで減少）。
budget が小さいとき SHA のオーバーヘッドが支配的になるため、**budget ≥ 64** での使用を推奨。

## 今後の改善 Issue

| 優先度 | 内容                                                                        |
| ------ | --------------------------------------------------------------------------- |
| High   | BayesianGenerator の候補空間から `welford` 等の未実装 variant を除外 (#137) |
| Medium | baseline 計測の安定化（複数回計測の中央値を使う）                           |
| Low    | SHA 推奨 budget のドキュメント化 / `--budget < 64` 時の警告追加             |
