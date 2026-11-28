# forge

[![CI](https://github.com/flipslidersand-labs/forge/actions/workflows/ci.yml/badge.svg)](https://github.com/flipslidersand-labs/forge/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/flipslidersand-labs/forge/graph/badge.svg)](https://codecov.io/gh/flipslidersand-labs/forge)

PyTorch 演算に対し、Triton カーネルの実装方式とパラメータを自動探索し、
正確性・測定ノイズ・環境差を考慮した上で最速実装をキャッシュ再利用する
GPU カーネル自動最適化エンジン。

```python
import forge
import torch


@forge.optimize(budget=50)
def rmsnorm(x, weight, eps=1e-6):
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * weight


x = torch.randn(2048, 4096, dtype=torch.float16, device="cuda")
w = torch.ones(4096, dtype=torch.float16, device="cuda")
y = rmsnorm(x, w)  # 初回: 探索してキャッシュ / 2回目以降: 最速カーネルを即実行
```

## ベンチマーク

### forge vs PyTorch Eager

forge が探索した最速カーネルと PyTorch Eager の比較（実測値）。

> **注記（#413）**: 以前の表 1 は RMSNorm の Eager が 1672.9µs / 1675.3µs で、表 2
> （815.1µs / 714.4µs）と食い違っていた。同じ環境（GTX 1080・PyTorch 2.13.0+cu126）で 2 つのスクリプトを測り直すと、
> `F.rms_norm` の Eager は 704〜727µs で、表 2 の値に近かった（旧表 1 の値は再現しなかった）。一方、非融合の手書き
> RMSNorm（`x * rsqrt(mean(x*x)) * w`）は 1633〜1663µs、`torch.compile(backend="eager")` は 1690〜1735µs で、
> 旧表 1 の値に近い。旧表 1 の RMSNorm の Eager は、`F.rms_norm` ではなく非融合実装に相当する時間だった可能性が高い
> が、旧表 1 がその値になった理由（当時の PyTorch の挙動など）は未確認。旧値の 11.19x / 7.92x は、`F.rms_norm` 基準では
> 再現しないため、RMSNorm の 2 行を再測定値に差し替えた。
> 表 2 の 815.1µs と今回の 704〜727µs の差（約 10%）は、同じ関数の最初の測定だけ 800µs 台になる回がある
> （切り分けスクリプトの再実行 1 回で 817.1µs、定常は 705〜711µs）ことと矛盾しないが、これも未確認。
> Eager 列について、Softmax・LayerNorm・GELU の再測定は旧値と ±4% 以内（forge 列は探索結果によって動く）。
> SDPA (8, 64, 64) は今回の再測定の対象外（shape が変わったため）。

| op          | shape        | dtype | PyTorch Eager (µs) | forge (µs) |    speedup |
| ----------- | ------------ | ----- | -----------------: | ---------: | ---------: |
| RMSNorm     | (2048, 4096) | fp16  |              704.2 |      162.5 |  **4.33x** |
| RMSNorm     | (1024, 8192) | fp16  |              726.7 |      166.4 |  **4.37x** |
| Softmax     | (2048, 4096) | fp16  |              791.2 |      193.5 |  **4.09x** |
| Softmax     | (1024, 8192) | fp16  |              819.2 |      146.4 |  **5.59x** |
| LayerNorm   | (2048, 4096) | fp16  |              949.0 |      961.5 |      0.99x |
| GELU        | (2048, 4096) | fp16  |              241.7 |      181.9 |  **1.33x** |
| SDPA        | (8, 64, 64)  | fp16  |              193.6 |      206.5 |      0.94x |
| SDPA causal | (8, 64, 64)  | fp16  |              248.9 |      278.9 |      0.89x |

> SDPA causal の値は、`is_causal=True` を本体にリテラルで書く旧形の `examples/bench_all.py` で測った可能性が高い（推測。未確認）。リテラル形は現在 eager にフォールバックするため、引数形に直したベンチで再測定が必要（未実施）。

### forge vs torch.compile (eager backend)

torch.compile と PyTorch Eager との比較。中央値計測。

| Op        | Shape        | Eager (µs) | torch.compile (µs) | forge (µs) | forge/eager | forge/compiled |
| --------- | ------------ | ---------: | -----------------: | ---------: | ----------: | -------------: |
| RMSNorm   | (2048, 4096) |      815.1 |             1704.4 |      192.9 |   **4.23x** |      **8.84x** |
| Softmax   | (2048, 4096) |      776.9 |              825.2 |      186.4 |   **4.17x** |      **4.43x** |
| LayerNorm | (2048, 4096) |      926.5 |              970.2 |      973.1 |       0.95x |          1.00x |
| GELU      | (2048, 4096) |      245.5 |              278.0 |      181.2 |   **1.35x** |      **1.53x** |
| RMSNorm   | (1024, 8192) |      714.4 |             1686.5 |      189.5 |   **3.77x** |      **8.90x** |
| Softmax   | (1024, 8192) |      811.5 |              890.8 |      178.2 |   **4.55x** |      **5.00x** |
| LayerNorm | (1024, 8192) |      934.8 |              987.6 |      949.2 |       0.98x |          1.04x |
| GELU      | (1024, 8192) |      243.7 |              283.6 |      181.9 |   **1.34x** |      **1.56x** |
| SDPA      | (8, 256, 64) |      205.4 |              250.9 |      222.2 |       0.92x |          1.13x |
| SDPA      | (8, 512, 64) |      551.6 |              596.0 |      560.0 |       0.98x |          1.06x |

**主な発見**:

- **RMSNorm・Softmax**: forge は eager の 3.77-4.55x、torch.compile の 8.84-8.90x （torch.compile が特に遅い）。
  RMSNorm の torch.compile が遅いのは、`backend="eager"` が演算を融合しないためとみられる
  （非融合の手書き版 1633〜1663µs に近く、compile はそれより約 3〜5% 遅い）。融合された `F.rms_norm`（Eager）より遅くなる（#413）
- **GELU**: forge は eager の 1.34-1.35x、torch.compile の 1.53-1.56x
- **LayerNorm・SDPA**: 小さいサイズでは forge が eagerと同等以下（GPU メモリレイアウト最適化の余地あり）

> **測定環境**: GTX 1080 (cc6.1, Triton 公式サポート外), PyTorch 2.13.0+cu126, Triton 3.7.1。  
> **計測設定**: budget=50, warmup=25, repeat=200 の中央値。baseline は PyTorch eager で動作確認。  
> torch.compile は backend="eager" を使用（他の backend は GTX 1080 未サポート）。  
> LayerNorm・SDPA は本 GPU・shape では eager と同等以下。  
> **この表は報告用の計測結果で、採用の可否とは別です。** 探索で得た最良カーネルの計測値を載せており、
> `@forge.optimize` は最良カーネルが eager に `min_speedup`（既定 1.03）倍で勝たない場合は採用せず、
> キャッシュにも書かずに eager で実行します（上の eager 以下の行はこれに当たります）。  
> **自環境での計測**（表 1・表 2 を同じ環境で続けて測り、食い違いを確認できる）:
>
> ```bash
> python examples/bench_all.py && python examples/benchmark_torch_compile_autotune.py
> ```
>
> - `examples/bench_all.py` — forge vs eager（表 1）
> - `examples/benchmark_torch_compile_autotune.py` — forge vs torch.compile (eager backend)（表 2）
> - `examples/bench_rmsnorm_baselines.py` — RMSNorm の `F.rms_norm` / 非融合の手書き版 / `torch.compile(eager)` を同条件で比較（#413 の切り分け）

## 対応演算

RMSNorm / Softmax / LayerNorm / GELU / Fused Add+RMSNorm / RoPE / SwiGLU / Linear / ScaledDotProductAttention（Flash Attention 2 スタイル・causal マスク対応）

## 仕組み

1. `@forge.optimize` が関数を torch.fx で trace し op を判定（lowering）
2. 入力テンソルから `KernelSpec` を構築
3. 探索器（Grid / Random / LLM）が候補を生成
4. 各候補を **使い捨て subprocess** でコンパイル・正確性検証・ベンチマーク
   （CUDA エラーで親プロセスが死なない）
5. 統計的に最速（`p80 < p20/1.03`）かつ正確な実装を SQLite にキャッシュ
6. 2 回目以降は環境込み `CacheKey`（torch/triton/cuda/compute-capability）で
   ヒット → 探索ゼロ

## インストール

```bash
# PyPI からインストール（パッケージ名は kernelsmith、import 名は forge。GPU 実行には [gpu] extra が必要）
pip install kernelsmith          # コアのみ（import だけなら GPU 不要）
pip install "kernelsmith[gpu]"   # GPU カーネル実行フル機能（triton 含む）
```

> GPU なし環境では `import forge` は成功します。GPU 実行（`@forge.optimize` 実行）時のみエラーになります。

## 必要環境

- Python 3.11+
- GPU 実行: NVIDIA GPU + PyTorch CUDA ビルド + Triton 3.x
  - Triton 公式サポートは compute capability 7.0+（GTX 1080 / cc6.1 でも動作確認済み）

## 開発環境セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,gpu]"   # コア + GPU + 開発ツール（pytest, ruff, pyright）
pip install -e ".[llm]"       # LLM 候補生成を使う場合（任意。anthropic + pydantic）
```

> システム Python が externally-managed の場合は venv 必須。
> PyTorch は使用 GPU のドライバに合う CUDA ビルドを入れること
> （例: ドライバが CUDA 12.x なら
> `pip install torch --index-url https://download.pytorch.org/whl/cu121`）。

## Discord 通知（オプション）

最適化完了時や エラー時に Discord に通知するには、環境変数を設定：

```bash
export DISCORD_WEBHOOK_COMPLETION="https://discordapp.com/api/webhooks/..."
export DISCORD_WEBHOOK_ERRORS="https://discordapp.com/api/webhooks/..."
```

設定しない場合は通知されません（処理に影響なし）。

## 使い方

### デコレータ

```python
@forge.optimize(budget=50)  # budget = 探索候補数の上限
def softmax(x):
    return torch.softmax(x, dim=-1)  # dim は定数で書く（trace 要件）
```

判定できない場合、および最良カーネルが eager に `min_speedup`（既定 1.03）倍で勝てない（速くならない）
場合は、元の eager 関数にフォールバックする。後者はキャッシュにも書かない。
`min_speedup` を下げれば、遅いカーネルも採用される（従来の挙動に近づく）。

### 探索 API（直接）

```python
from forge.ir.kernel_spec import KernelSpec
from forge.orchestrator import Orchestrator

# KernelSpec を組み立てて orch.optimize(spec, budget=50)
```

動かせる例は `examples/decorator_demo.py` / `examples/rmsnorm_search.py` を参照。

```bash
.venv/bin/python examples/decorator_demo.py
```

### LLM 候補生成 + マルチラウンド探索（任意）

`forge.search.llm_generator.LLMGenerator` は Claude（`claude-opus-4-8`）に
構造化された候補パラメータを出させる探索器。実 API 利用には `ANTHROPIC_API_KEY`
が必要（`pip install -e ".[llm]"`）。テストは `propose_fn` 注入でオフライン実行できる。

マルチラウンド探索は `Orchestrator.optimize_rounds()` で実装：

```python
from forge.orchestrator import Orchestrator, MultiRoundResult
from forge.search.llm_generator import LLMGenerator

llm = LLMGenerator()  # claude-opus-4-8 を使用
orch = Orchestrator()
result: MultiRoundResult = orch.optimize_rounds(
    spec=spec,
    llm=llm,
    n_rounds=3,  # 3 ラウンド
    candidates_per_round=12,  # 各ラウンドで 12 候補を提案
)
# result.best_params で最速カーネルを取得
# result.token_usage で Anthropic API の総トークン数を確認
# result.total_benchmark_time_s で GPU ベンチマーク時間を確認
```

### コスト考慮判定（パレート最適化）

> **注意（未配線）**: `forge.benchmark.pareto` は現状 `Orchestrator` の探索ループには
> 組み込まれていない（単体で呼び出して使う API）。統合は未実施で、統合計画は未定。
> 探索ループ側で使われているのは `BudgetTracker`（予算超過の判定）のみ（ADR-011）。

`forge.benchmark.pareto` モジュールは、速度と探索コスト（API トークン数・GPU 時間）
のトレードオフを可視化：

```python
from forge.benchmark.pareto import CandidateWithCost, ParetoFrontier

# 複数候補をコスト情報付きで評価（p1, p2 は SearchParams）
candidates = [
    CandidateWithCost(
        params=p1, median_us=50.0, tokens_for_proposal=5000,
        benchmark_time_ms=300.0, correct=True,
    ),
    CandidateWithCost(
        params=p2, median_us=80.0, tokens_for_proposal=2000,
        benchmark_time_ms=200.0, correct=True,
    ),
]
frontier = ParetoFrontier(candidates)
# frontier.frontier に パレート最適な候補のみ
recommended = frontier.recommend()  # スコアが最高の候補を推奨
```

**利用例**:

- 候補 A: `7.8µs`（高速、token cost 高）
- 候補 B: `8.2µs`（中速、token cost 低）
  → 両者とも Pareto 最適。用途に応じて選択可能

#### λ（lambda）パラメータによるトレードオフ制御

`forge.search.cost_model.scalarize()` で複数目的を線形結合し、単一スコアに変換します
（`cost` の単位は任意だが、秒を想定）。これは単体で呼べる関数で、上の
`ParetoFrontier.recommend()` は使いません（`recommend()` は λ を取らず、固定の重み付けで選びます）。
使い方は `examples/cost_aware_optimization.py` と ADR-011 を参照：

```python
from forge.search.cost_model import scalarize

# 速度重視（デフォルト）
score_speed = scalarize(speedup=4.0, cost=2.0, lam=0.1)  # 3.8

# コスト重視
score_cost = scalarize(speedup=4.0, cost=2.0, lam=1.0)   # 2.0

# 速度優先・コスト無視
score_nopenalty = scalarize(speedup=4.0, cost=2.0, lam=0.0)  # 4.0
```

**λ の意味**:

```
score = speedup - λ × cost
```

- **λ=0.0**: 速度優先（コスト無視）→ 最も高速な候補選択
- **λ=0.1** (デフォルト): 速度重視・コスト補助 → 実用的なバランス
- **λ=1.0**: 同等ウェイト → 速度とコストを等価扱い
- **λ > 1.0**: コスト重視 → 計測時間が限られた環境向け

**推奨値**:

- `local-GPU`: λ=0.0（GPU 時間制限なし） → 最高速求める
- `cloud-API` (LLM利用): λ=0.1（API 費用を考慮）
- `embedded`: λ=0.5-1.0（計測時間・メモリ節約）

## 環境変数

| 変数名 | 既定値 | 説明 |
|---|---|---|
| `FORGE_DB_PATH` | `~/.forge/cache.db` | キャッシュ DB のパス |
| `FORGE_HMAC_KEY_PATH` | `~/.forge/hmac.key` | カーネル署名用 HMAC 鍵ファイルのパス |
| `DISCORD_WEBHOOK_COMPLETION` | ― | 最適化完了の Discord 通知用 Webhook URL |
| `DISCORD_WEBHOOK_ERRORS` | ― | エラー時の Discord 通知用 Webhook URL |

```bash
# CI/CD やテスト時に一時 DB を使う例
FORGE_DB_PATH=/tmp/forge_test.db pytest

# 複数プロジェクトでキャッシュを分離する例
FORGE_DB_PATH=~/projects/myapp/.forge/cache.db python train.py
```

CLI の `--db` オプションは環境変数より優先されます。

### HMAC 鍵ファイル

キャッシュ済みカーネルの改ざん検知用 HMAC 鍵は、初回使用時に `~/.forge/hmac.key`（`FORGE_HMAC_KEY_PATH` で変更可）へ
32 バイトで作成され、プロセス間で共有されます（ファイル 0600。ディレクトリは新規作成時のみ 0700 で、既存の `~/.forge` の権限は変更しません）。
このファイルを削除・変更するか別マシンへ DB だけ持ち出すと、既存エントリの検証に失敗してキャッシュは全ミスになります（再生成されます）。
鍵ファイルが読めない・長さ不正の場合は警告を出し、そのプロセス限りの鍵で動作します。

## CLI（キャッシュ管理）

インストールすると `forge` コマンドが使えます（`forge --version` でバージョン表示）。
キャッシュ DB の中身を確認・削除します。

```bash
forge cache list [--json] [--db PATH]                       # キャッシュ済みカーネル一覧
forge cache clear [--force] [--db PATH]                     # 全件削除（--force なしは確認あり）
forge cache prune [--before YYYY-MM-DD] [--keep-latest N] [--force] [--db PATH]
                                                            # 条件に合う古いエントリを削除
```

- `prune` は `--before`（この日付より前に作成）か `--keep-latest`（最新 N 件を残す）の
  少なくとも一方が必要。どちらも無く DB がある場合は、件数だけ表示して終了コード 1 になる。
  `--keep-latest` は 1 以上（全件削除は `clear`）。
- `--force` を付けない削除は `[y/N]` で確認する。非対話（EOF）・Ctrl-C は中止になる。
- DB ファイルが無いときは何も作らず「キャッシュは空です」と表示する。

## テスト

```bash
.venv/bin/python -m pytest tests/ -m "not gpu"   # GPU 不要（CPU のみ）
.venv/bin/python -m pytest tests/                # GPU を含む全テスト
.venv/bin/ruff check src/ tests/                 # Lint
```

## ディレクトリ構成

```
src/forge/
  ops/              op メタデータ（OP_REGISTRY: reduction / elementwise / matmul / gemm）
  ir/               TensorSpec / KernelSpec / hashing
  lowering/         torch.fx グラフ → op_type 判定
  codegen/          KernelSpec + params → Triton コード（Jinja2 テンプレート）
  search/           SearchSpace / GridSearch / RandomSearch / LLMGenerator
  runtime/          subprocess worker / kernel ローダ / 参照実装
  validation/       正確性スイート / 許容誤差
  benchmark/        CUDA Event タイマー / 統計的採用判定
  cache/            CacheKey / SQLite リポジトリ
  security/         生成 Triton コードの AST セキュリティチェック
  notifiers/        通知先の Protocol と Discord 通知
  orchestrator.py   探索 → 検証 → ベンチ → キャッシュの統括
  decorator.py      @forge.optimize
  prewarm.py        デプロイ前のキャッシュ事前生成
  cli.py            `forge cache list/clear/prune`
docs/               spec / data-model / implementation-guide / tech-stack /
                    benchmark-search-strategies / torch-compile-baseline / adr/
examples/           実行デモ
tests/              CPU テスト + GPU テスト（@pytest.mark.gpu）
```

設計判断は `docs/adr/` を参照（Triton 採用、SQLite、subprocess 隔離、
統計的ベンチ判定、LLM 構造化生成）。

## 対応演算と制約

forge が最適化を試みる演算の一覧と、各演算固有の制約です。
未対応・torch.fx トレース不能（動的 dim 等）な場合は元の eager 実装に透過フォールバックします。

| 演算 | op_type | 制約・備考 |
| --- | --- | --- |
| RMSNorm | `rmsnorm` | `eps` は定数で記述すること |
| LayerNorm | `layernorm` | `normalized_shape` は定数 |
| GELU | `gelu` | exact（erf）のみ。tanh 近似は許容誤差を超えて eager になり得る |
| Softmax | `softmax` | `dim` は定数で記述すること |
| Fused Add+RMSNorm | `fused_add_rmsnorm` | `(x + residual)` → RMSNorm の融合 |
| RoPE | `rope` | `x*cos + rotate_half(x)*sin` パターン |
| SwiGLU | `swiglu` | — |
| SDPA | `scaled_dot_product_attention` | **[下記参照](#sdpa-制約)** |
| Linear | `linear` | bias は任意 |

### SDPA 制約

`torch.nn.functional.scaled_dot_product_attention` に `@forge.optimize` を適用する際は
以下の条件を満たす必要があります。条件を満たさない場合は eager にフォールバックします。

| 項目 | 要件 |
| --- | --- |
| `head_dim` (テンソルの最終次元) | **2 のべき乗 かつ ≥ 16**（例: 16, 32, 64, 128） |
| `attn_mask` | **非対応**。`None` のみ受け付ける |
| `dropout_p` | **非対応**（`> 0.0` は eager フォールバック） |
| `enable_gqa` | **非対応**（GQA / MQA は異なる op パターン） |
| `is_causal` | `True` / `False` どちらも対応。ただし**関数の引数として受け取る形**に限る。本体にリテラル（`is_causal=True`）で書くと eager にフォールバック |
| `scale` | 任意（`None` = デフォルトスケール） |
| テンソル形状 | `(batch_heads, seq_len, head_dim)` の 3-D |

```python
# ✅ 動作する（is_causal は引数で受け取る）
@forge.optimize()
def attention(q, k, v, is_causal=True):
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

# ❌ is_causal=True を本体にリテラルで書く → eager フォールバック
#    （op の同定が関数名の個数だけでは is_causal の値を区別できず、非 causal カーネルが
#     選ばれる誤りを防ぐため、リテラル引数を持つ呼び出しは対応外として eager に落とす）
# ❌ head_dim=12 は 2 のべき乗でない → eager フォールバック
# ❌ attn_mask を渡す → torch.fx トレース不能 → eager フォールバック
# ❌ dropout_p=0.1 → eager フォールバック
```

## 既知の制約

- 演算は標準的な式の形のみ認識（torch.fx の call_function 多重集合でマッチ）
- 開発・検証は GTX 1080（compute capability 6.1、Triton 公式サポート外）で実施

## ロードマップ

進行中・予定の作業は [GitHub Issues](https://github.com/flipslidersand-labs/forge/issues) を参照。

## 設計判断 / Architecture decisions

[ADR 一覧](docs/adr/README.md) — 設計判断、根拠資料、未決定事項。
