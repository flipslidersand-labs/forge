# GPU Test Execution Guide

Forge GPU カーネルテストの実行手順。

## 前提条件

- GPU: RTX 4060 以上（Compute Capability 8.6+）
- CUDA 12.1+
- Python 3.11+
- PyTorch 2.3+ with CUDA support

## 環境確認

```bash
# GPU 確認
nvidia-smi

# PyTorch CUDA サポート確認
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name()}')"

# Triton インストール確認
python3 -c "import triton; print(f'Triton: {triton.__version__}')"
```

## テスト実行

### 1. オフラインテスト（GPU 不要）

```bash
# GPU 不要なテストだけを実行
python3 -m pytest tests/test_adoption.py tests/test_benchmark.py tests/test_cache.py tests/test_ir.py -v

# 結果: 53/53 通過が目標
```

### 2. GPU テスト（一部）

```bash
# Triton kernel registry テスト（GPU 必須）
python3 -m pytest tests/test_fused_add_rmsnorm_registry.py::TestFusedAddRmsnormReference::test_basic_fp32 -v

# または個別実行
python3 -m pytest tests/test_rope_registry.py -v --tb=short
```

### 3. 完全テスト

```bash
# すべてのテストを実行（GPU 必須、10-15分）
python3 -m pytest tests/ -v --tb=short -x

# 並列実行（高速、4 workers）
python3 -m pytest tests/ -v -n 4
```

## トラブルシューティング

### CUDA Compute Capability エラー

**エラー例:**

```
The following list shows the CCs this version of PyTorch was built for:
  - 7.5 which supports hardware CC >=7.5,<8.0
Current GPU CC: 6.1 (not supported)
```

**対応:**

- PyTorch を正しいバージョンに更新
- または CPU mode でテスト実行

### Triton import エラー

```bash
# Triton 再インストール
pip install --upgrade triton

# または triton version 確認
python3 -c "import triton; print(triton.__version__)"
```

### タイムアウトエラー

テスト実行時間が長い場合、タイムアウトを延長：

```bash
python3 -m pytest tests/ -v --timeout=300  # 5 分
```

## マシン別実行手順

### DS1 (Windows, RTX4060)

```powershell
# PowerShell
cd C:\path\to\forge
python -m pytest tests\test_adoption.py tests\test_benchmark.py -v

# または Git Bash
bash
python3 -m pytest tests/ -v
```

### YUKI-PRIVATE002 (Linux, A100)

```bash
ssh yuki-private
cd /path/to/forge
python3 -m pytest tests/ -v --tb=short
```

### Dev PC (Linux, GTX1080 - CPU mode only)

```bash
# CPU mode でのみテスト可能
export CUDA_VISIBLE_DEVICES=""
python3 -m pytest tests/test_adoption.py tests/test_benchmark.py tests/test_cache.py tests/test_ir.py -v
```

## CI/CD での自動テスト

GitHub Actions では以下のテストが自動実行されます：

- **Offline tests**: Always runs (53/53)
- **GPU tests**: Linux runner で実行（GPU なし）

本格的な GPU テスト（Triton kernel validation）は手動実行推奨。

## レポート記録

テスト結果を記録してください：

```bash
# テスト実行 & ログ保存
python3 -m pytest tests/ -v --tb=short 2>&1 | tee test-results.log

# 結果サマリー
tail -20 test-results.log
```

## 関連リンク

- [PyTorch Testing Guide](https://pytorch.org/docs/stable/testing.html)
- [Triton Documentation](https://triton-lang.org/)
- forge Issue #172, #173, #174 (quality improvements)
