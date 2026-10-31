# Contributing to forge

## 開発環境セットアップ

```bash
git clone https://github.com/flipslidersand-labs/forge.git
cd forge
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,llm,ollama]"
```

GPU 実行（triton コンパイル）が必要なテストは `pytest -m gpu` で実行します。

## テスト

```bash
# CPU のみ（CI と同じ）
pytest -k "not gpu" -q

# GPU テスト（CUDA 環境が必要）
pytest -m gpu -q

# PR 向け smoke テスト（GPU, budget=3）
pytest -m smoke -q

# 型チェック
pyright src/

# lint / format
ruff check src/ tests/
ruff format src/ tests/
```

## 新しい op を追加する手順

新しい演算（例: `layer_scale`）を追加するときは以下の順で変更します。
**変更が必要なファイルは 4 箇所**で、それ以外は自動的にカバーされます。

### ステップ 1 — `OpDefinition` をレジストリに登録

`src/forge/ops/registry.py` に `OpDefinition` インスタンスを追加します。

```python
# src/forge/ops/registry.py

def _layer_scale_reference(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """正確性検証用 ground truth（fp32 で計算）。"""
    return (x.float() * scale.float()).to(x.dtype)

def _layer_scale_baseline(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """速度比較の baseline（PyTorch 最適化実装）。"""
    return x * scale

OP_REGISTRY["layer_scale"] = OpDefinition(
    reference_fn=_layer_scale_reference,
    baseline_fn=_layer_scale_baseline,
    baseline_display_name="x * scale",
    tolerance=Tolerance(atol=1e-3, rtol=1e-3),
    primary_input_fn=lambda spec: [...],        # ベンチマーク入力テンソルを返す
    correctness_cases_fn=lambda spec: [...],    # 正確性検証ケースを返す
    render_extra_kwargs_fn=lambda spec: {},     # テンプレートへの追加変数（不要なら省略可）
)
```

`primary_input_fn` / `correctness_cases_fn` の書き方は既存 op（`rmsnorm` 等）を参照してください。

### ステップ 2 — lowering パターンを追加

`src/forge/lowering/layer_scale.py` を新規作成し、`registry.py` に登録します。

```python
# src/forge/lowering/layer_scale.py
from forge.lowering.registry import OpPattern, register_pattern
import torch

register_pattern(
    OpPattern(
        op_type="layer_scale",
        match_fn=lambda node: (
            node.op == "call_function"
            and node.target == torch.ops.aten.mul.Tensor
            # 必要に応じてより詳細なパターンマッチを実装
        ),
        extract_spec_fn=lambda node, inputs: dict(
            input_specs=(...),
            output_specs=(...),
            constants={},
        ),
    )
)
```

次に `src/forge/lowering/__init__.py` の import リストに追加します。

```python
# src/forge/lowering/__init__.py
from forge.lowering import layer_scale  # noqa: F401
```

### ステップ 3 — Triton テンプレートを追加

`src/forge/codegen/templates/layer_scale.py.jinja` を作成します。
`kernel_fn(*args)` を公開する完全スタンドアロンな Triton モジュールにしてください。

```jinja2
import triton
import triton.language as tl

@triton.jit
def _layer_scale_kernel(
    x_ptr, scale_ptr, out_ptr, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask)
    s = tl.load(scale_ptr + cols, mask=mask)
    tl.store(out_ptr + row * n_cols + cols, x * s, mask=mask)

def kernel_fn(x, scale):
    out = x.clone()
    n_rows, n_cols = x.shape
    grid = (n_rows,)
    _layer_scale_kernel[grid](x, scale, out, n_cols, BLOCK_SIZE={{ block_size }})
    return out
```

テンプレートと `SearchParams.variant` の対応を `src/forge/codegen/triton_codegen.py` の `_TEMPLATES` dict に追加します。

```python
_TEMPLATES = {
    ...
    ("layer_scale", "elementwise"): "layer_scale.py.jinja",
}
```

### ステップ 4 — 動作確認

```bash
# レジストリ登録を確認
python3 -c "from forge.ops.registry import OP_REGISTRY; print('layer_scale' in OP_REGISTRY)"

# tolerance / reference が引けることを確認
python3 -c "
from forge.validation.tolerance import get_tolerance
from forge.runtime.reference import get_reference
print(get_tolerance('layer_scale'))
print(get_reference('layer_scale'))
"

# CPU テスト（GPU なしで実行可能）
pytest -k "not gpu" -q

# GPU smoke（GPU が必要）
pytest -m smoke -q
```

### チェックリスト

新 op を PR する前に以下を確認してください。

- [ ] `src/forge/ops/registry.py` に `OpDefinition` を追加した
- [ ] `src/forge/lowering/<op>.py` を作成し `__init__.py` に import を追加した
- [ ] `src/forge/codegen/templates/<op>*.py.jinja` を作成した
- [ ] `src/forge/codegen/triton_codegen.py` の `_TEMPLATES` に追加した
- [ ] `pytest -k "not gpu" -q` が全通過する
- [ ] `pyright src/` が 0 errors
- [ ] `ruff check src/ tests/` が 0 errors

## コミットメッセージ規約

[Conventional Commits](https://www.conventionalcommits.org/) に従います。

```
feat(ops): add layer_scale op

Adds LayerScale (element-wise multiply with a learned scale vector)
as a new supported op in OP_REGISTRY with single-row Triton template.

Closes #NN
```

## ブランチ命名

| 種別         | プレフィックス | 例                     |
| ------------ | -------------- | ---------------------- |
| 新機能       | `feat/`        | `feat/layer-scale-op`  |
| バグ修正     | `fix/`         | `fix/sdpa-tolerance`   |
| ドキュメント | `docs/`        | `docs/contributing`    |
| テスト       | `test/`        | `test/layer-scale-cpu` |
| CI           | `ci/`          | `ci/gpu-smoke`         |
| リファクタ   | `refactor/`    | `refactor/op-registry` |
