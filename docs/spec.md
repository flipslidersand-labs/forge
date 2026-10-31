# Spec — Forge

## 目的

PyTorch で書いた演算（RMSNorm / Softmax 等）に対して、Triton カーネルの実装方式とパラメータを自動探索し、正確性・測定ノイズ・環境差を考慮した上で最速実装をキャッシュ再利用するライブラリ。

## 解決する問題

- 手書き Triton カーネルは速いが、`BLOCK_SIZE` や `num_warps` の最適値は GPU・dtype・shape によって異なる
- `@triton.autotune` は手動でヒントを書く必要があり、実装方式の探索はできない
- LLM に自由コード生成させると検証が難しく再現性がない
- GPU ベンチマークは測定ノイズが大きく、1 回比較では信頼できない

## 対応演算

RMSNorm / Softmax / LayerNorm / GELU / ScaledDotProductAttention / Linear（Matmul）

新 op の追加方法は [CONTRIBUTING.md](../CONTRIBUTING.md) を参照。

## 実装範囲

### 現行（Phase 1〜6 完了）

- `@forge.optimize` デコレータ — torch.fx trace → KernelSpec 自動生成
- グリッド探索（`GridSearch`）/ ランダム探索（`RandomSearch`）
- 複数実装バリアント（single_row / multi_row / two_pass / welford / flash / gemm 等）
- 正確性検証（エッジケース含む / subprocess 隔離）
- 統計的に安全なベンチマーク（warmup + 中央値 + p20/p80）
- SQLite キャッシュ（環境差込み `CacheKey`）
- LLM 探索（`LLMGenerator` — Claude）/ ローカル LLM（`OllamaGenerator`）
- `OpDefinition` レジストリ — op 追加を 1 ファイル 1 箇所で完結

### スコープ外

- 任意 Python 関数の自動 GPU 化
- PyTorch モデル全体の最適化
- Backward / 勾配計算
- マルチ GPU
- CUDA C++ 直接生成

### 将来候補（Issue 化済み）

- Bayesian / Successive Halving 探索 → Issue #123

## API

### デコレータ（推奨）

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

### Orchestrator（プログラム制御）

```python
from forge.cache.repository import KernelRepository
from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.orchestrator import Orchestrator
from forge.search.grid import GridSearch
import torch

spec = KernelSpec(
    op_type="rmsnorm",
    input_specs=(
        TensorSpec((2048, 4096), torch.float16, True),
        TensorSpec((4096,), torch.float16, True),
    ),
    output_specs=(TensorSpec((2048, 4096), torch.float16, True),),
    constants={"eps": 1e-6},
    graph_hash="rmsnorm_v1",
    constraints=(),
)

with Orchestrator() as orch:
    result = orch.optimize(spec, budget=50, search=GridSearch())

print(result.best_params)       # SearchParams(block_size=2048, num_warps=8, ...)
print(f"speedup: {result.speedup:.2f}x")
```

### LLM 探索（Claude）

```python
from forge.orchestrator import Orchestrator
from forge.search.llm_generator import LLMGenerator

llm = LLMGenerator(model="claude-opus-4-8")

with Orchestrator() as orch:
    result = orch.optimize_rounds(spec, llm=llm, n_rounds=3, candidates_per_round=12)

print(f"best: {result.best_params}, rounds: {len(result.rounds)}")
print(f"tokens used: {result.token_usage.total}")
```

### ローカル LLM 探索（Ollama）

```python
from forge.search.ollama_generator import OllamaGenerator

gen = OllamaGenerator(model="qwen2.5-coder:7b", host="http://localhost:11434")
# GridSearch と同様に Orchestrator.optimize() に渡す
result = orch.optimize(spec, budget=12, search=gen)
```

### CLI

```bash
forge cache list          # キャッシュ済みカーネル一覧
forge cache clear         # キャッシュ全削除
```

## 仕組み

1. `@forge.optimize` が関数を `torch.fx` で trace し op type を判定（lowering）
2. 入力テンソルから `KernelSpec` を構築
3. 探索器（Grid / Random / LLM）が `SearchParams` 候補を生成
4. 各候補を **使い捨て subprocess** でコンパイル・正確性検証・ベンチマーク
   （CUDA エラーで親プロセスが死なない）
5. 統計的に最速（`p80 < best_p20 / min_speedup`）かつ正確な実装を SQLite にキャッシュ
6. 2 回目以降は `CacheKey`（torch/triton/cuda/compute-capability 込み）でヒット → 探索ゼロ

## キー設計決定

| 決定                                  | 理由                                                  |
| ------------------------------------- | ----------------------------------------------------- |
| subprocess 隔離                       | CUDA エラーで親プロセスを守る                         |
| `CacheKey` に環境情報を含む           | torch/triton バージョン違いでキャッシュ誤ヒットしない |
| LLM にコード生成させない              | 構造化パラメータのみ → 検証・再現が容易               |
| `OpDefinition` レジストリ             | op 追加時の変更箇所を 1 ファイルに集約                |
| `BenchmarkResultDict` split TypedDict | 必須/オプションキーの型安全                           |

## 成功条件

| Phase           | 条件                                                        |
| --------------- | ----------------------------------------------------------- |
| Phase 1 完了 ✅ | KernelSpec → CacheKey → SQLite の往復が動く                 |
| Phase 2 完了 ✅ | RMSNorm 50 候補探索、PyTorch Eager 比 +20% 以上をキャッシュ |
| Phase 3 完了 ✅ | 複数バリアントを含む探索で autotune 相当以上の性能          |
| Phase 4 完了 ✅ | `@forge.optimize` デコレータが 2 回目以降ゼロ探索で動く     |
| Phase 5 完了 ✅ | `OpDefinition` レジストリで op 追加経路を一本化             |
| Phase 6 完了 ✅ | CPU モックテスト・GPU smoke test・pyright 0 errors          |
