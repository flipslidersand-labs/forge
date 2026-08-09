# Search Strategy Benchmark

`examples/compare_search_strategies.py` による実測結果。

**条件**: rmsnorm 512×4096 fp16, budget=32, SHA halving_rounds=3

## 実行方法

```bash
pip install "forge-kernel[search]"   # optuna が必要
python examples/compare_search_strategies.py --rows 512 --hidden 4096 --budget 32
```

## 結果

### DS1 (RTX 4060)

| Strategy          | best_us | n_eval | time_s | speedup_vs_baseline |
| ----------------- | ------- | ------ | ------ | ------------------- |
| GridSearch        | —       | —      | —      | —                   |
| BayesianGenerator | —       | —      | —      | —                   |
| SHA(3r)           | —       | —      | —      | —                   |

_未計測。GPU 環境で `examples/compare_search_strategies.py` を実行後にこのテーブルを更新すること。_

### GTX 1080 (開発PC)

| Strategy          | best_us | n_eval | time_s | speedup_vs_baseline |
| ----------------- | ------- | ------ | ------ | ------------------- |
| GridSearch        | —       | —      | —      | —                   |
| BayesianGenerator | —       | —      | —      | —                   |
| SHA(3r)           | —       | —      | —      | —                   |

_未計測。GTX 1080 は CUDA sm_61 のため triton カーネルのコンパイルに失敗する場合あり。_

## 評価指標の定義

| 指標                  | 内容                                                      |
| --------------------- | --------------------------------------------------------- |
| `best_us`             | 探索で見つけた最速カーネルの中央値 (μs)                   |
| `n_eval`              | 評価した候補数（budget と一致しない場合は無効候補を除外） |
| `time_s`              | 探索全体の所要時間 (wall clock)                           |
| `speedup_vs_baseline` | PyTorch baseline に対する高速化倍率                       |

## SHA の設計根拠

Successive Halving は各ラウンドで測定精度を上げながら候補を絞る。

| ラウンド | 候補数         | warmup | repeat |
| -------- | -------------- | ------ | ------ |
| 1        | initial_budget | 5      | 20     |
| 2        | 上位 50%       | 10     | 50     |
| 3        | 上位 50%       | 25     | 200    |

budget=32 の場合、評価回数は 32+16+8=56 回だが、後半ほど精度が高い。
GridSearch は同一 budget=32 を全て repeat=200 で評価するため、正確だが遅い。
