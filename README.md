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

### forge vs PyTorch Eager

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

### forge vs torch.compile / autotune

torch.compile（eager backend）との比較。中央値計測。

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

- **RMSNorm・Softmax**: forge は eager の 3.77-4.55x、torch.compile の 8.84-8.90x （torch.compile が特に遅い）
- **GELU**: forge は ease の 1.34-1.35x、torch.compile の 1.53-1.56x
- **LayerNorm・SDPA**: 小さいサイズでは forge が eagerと同等以下（GPU メモリレイアウト最適化の余地あり）

> **測定環境**: GTX 1080 (cc6.1, Triton 公式サポート外), PyTorch 2.13.0+cu126, Triton 3.7.1。  
> **計測設定**: budget=50, warmup=25, repeat=200 の中央値。baseline は PyTorch eager で動作確認。  
> torch.compile は backend="eager" を使用（他の backend は GTX 1080 未サポート）。  
> LayerNorm・SDPA は本 GPU・shape では eager と同等以下（forge はフォールバックせず正直に報告）。  
> **自環境での計測**:
>
> - `examples/bench_all.py` — forge vs eager
> - `examples/benchmark_torch_compile_autotune.py` — forge vs torch.compile vs autotune

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

## インストール

```bash
# PyPI からインストール（GPU 実行には [gpu] extra が必要）
pip install forge-kernel          # コアのみ（import だけなら GPU 不要）
pip install "forge-kernel[gpu]"   # GPU カーネル実行フル機能（triton 含む）
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

`forge.benchmark.pareto` モジュールは、速度と探索コスト（API トークン数・GPU 時間）
のトレードオフを可視化：

```python
from forge.benchmark.pareto import CandidateWithCost, ParetoFrontier

# 複数候補をコスト情報付きで評価
candidates = [
    CandidateWithCost(params=p1, median_us=50.0, tokens_for_proposal=5000, ...),
    CandidateWithCost(params=p2, median_us=80.0, tokens_for_proposal=2000, ...),
]
frontier = ParetoFrontier(candidates)
# frontier.frontier に パレート最適な候補のみ
recommended = frontier.recommend()  # スコアが最高の候補を推奨
```

**利用例**:

- 候補 A: `7.8µs`（高速、token cost 高）
- 候補 B: `8.2µs`（中速、token cost 低）
  → 両者とも Pareto 最適。用途に応じて選択可能

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
