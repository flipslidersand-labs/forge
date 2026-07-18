# forge

[![CI](https://github.com/flipslidersand/forge/actions/workflows/ci.yml/badge.svg)](https://github.com/flipslidersand/forge/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/flipslidersand/forge/graph/badge.svg)](https://codecov.io/gh/flipslidersand/forge)

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

forge が探索した最速カーネルと PyTorch Eager の比較（実測値）。

| op          | shape        | dtype | PyTorch Eager (µs) | forge (µs) |    speedup |
| ----------- | ------------ | ----- | -----------------: | ---------: | ---------: |
| RMSNorm     | (2048, 4096) | fp16  |             1672.9 |      149.5 | **11.19x** |
| RMSNorm     | (1024, 8192) | fp16  |             1675.3 |      211.5 |  **7.92x** |
| Softmax     | (2048, 4096) | fp16  |              791.2 |      193.5 |  **4.09x** |
| Softmax     | (1024, 8192) | fp16  |              819.2 |      146.4 |  **5.59x** |
| LayerNorm   | (2048, 4096) | fp16  |              949.0 |      961.5 |      0.99x |
| GELU        | (2048, 4096) | fp16  |              241.7 |      181.9 |  **1.33x** |
| SDPA        | (8, 64, 64)  | fp16  |              193.6 |      206.5 |      0.94x |
| SDPA causal | (8, 64, 64)  | fp16  |              248.9 |      278.9 |      0.89x |

> **測定環境**: GTX 1080 (cc6.1, Triton 公式サポート外), PyTorch 2.13.0+cu126, Triton 3.7.1。  
> budget=50, warmup=25, repeat=200 の中央値。Baseline は `F.rms_norm` / `F.softmax` / `F.layer_norm` / `F.gelu` / `F.scaled_dot_product_attention`。  
> LayerNorm・SDPA は本 GPU・shape では eager と同等以下（forge はフォールバックせず正直に報告）。  
> 自環境での計測: `.venv/bin/python examples/bench_all.py`

## 対応演算

RMSNorm / Softmax / LayerNorm / GELU / ScaledDotProductAttention（Flash Attention 2 スタイル・causal マスク対応）

## 仕組み

1. `@forge.optimize` が関数を torch.fx で trace し op を判定（lowering）
2. 入力テンソルから `KernelSpec` を構築
3. 探索器（Grid / Random / LLM）が候補を生成
4. 各候補を **使い捨て subprocess** でコンパイル・正確性検証・ベンチマーク
   （CUDA エラーで親プロセスが死なない）
5. 統計的に最速（`p80 < p20/1.03`）かつ正確な実装を SQLite にキャッシュ
6. 2 回目以降は環境込み `CacheKey`（torch/triton/cuda/compute-capability）で
   ヒット → 探索ゼロ

## 必要環境

- NVIDIA GPU（compute capability は問わないが、Triton は 7.0+ が公式サポート）
- Python 3.11+ / PyTorch 2.x（CUDA 対応ビルド）/ Triton 3.x

## セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"   # コア + 開発ツール（pytest, ruff, pyright）
pip install -e ".[llm]"   # LLM 候補生成を使う場合（任意。anthropic + pydantic）
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

判定できない・最適化で速くならない場合は、元の eager 関数にフォールバックする。

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

### LLM 候補生成（任意）

`forge.search.llm_generator.LLMGenerator` は Claude（`claude-opus-4-8`）に
構造化された候補パラメータを出させる探索器。実 API 利用には `ANTHROPIC_API_KEY`
が必要（`pip install -e ".[llm]"`）。テストは `propose_fn` 注入でオフライン実行できる。

## テスト

```bash
.venv/bin/python -m pytest tests/ -m "not gpu"   # GPU 不要（CPU のみ）
.venv/bin/python -m pytest tests/                # GPU を含む全テスト
.venv/bin/ruff check src/ tests/                 # Lint
```

## ディレクトリ構成

```
src/forge/
  ops.py            op メタデータ（reduction / elementwise）
  ir/               TensorSpec / KernelSpec / hashing
  lowering/         torch.fx グラフ → op_type 判定
  codegen/          KernelSpec + params → Triton コード（Jinja2 テンプレート）
  search/           SearchSpace / GridSearch / RandomSearch / LLMGenerator
  runtime/          subprocess worker / kernel ローダ / 参照実装
  validation/       正確性スイート / 許容誤差
  benchmark/        CUDA Event タイマー / 統計的採用判定
  cache/            CacheKey / SQLite リポジトリ
  orchestrator.py   探索 → 検証 → ベンチ → キャッシュの統括
  decorator.py      @forge.optimize
docs/               spec / data-model / implementation-guide / adr/
examples/           実行デモ
tests/              CPU テスト + GPU テスト（@pytest.mark.gpu）
```

設計判断は `docs/adr/` を参照（Triton 採用、SQLite、subprocess 隔離、
統計的ベンチ判定、LLM 構造化生成）。

## 既知の制約

- 判定できる演算は上記 5 種のみ。未対応・trace 不能（動的 dim 等）は eager フォールバック
- SDPA は attn_mask / dropout / enable_gqa 非対応。head_dim は 2 のべき乗 ≥ 16 が必須
- GELU は exact（erf）のみ。tanh 近似の関数は許容誤差を超えて eager になり得る
- 演算は標準的な式の形のみ認識（torch.fx の call_function 多重集合でマッチ）
- 開発・検証は GTX 1080（compute capability 6.1、Triton 公式サポート外）で実施

## ロードマップ

GitHub Issues を参照:

- #28 pyright 型エラー解消
- #29 pytest-cov カバレッジ計測・バッジ追加
- #30 対応 op 拡張（FlashAttention 系・大 SDPA shape での最適化）
