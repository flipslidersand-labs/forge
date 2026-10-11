#!/bin/bash
# GPU kernel テスト実行スクリプト
# GPU 搭載の self-hosted runner / ローカル GPU マシンでの実行用

set -euo pipefail

echo "=== GPU Environment Check ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    cc = torch.cuda.get_device_capability(0)
    print(f'Compute Capability: {cc[0]}.{cc[1]}')
"

echo ""
echo "=== Running pytest ==="
python3 -m pytest tests/ -v --tb=short -x 2>&1 | tee test-results.log

echo ""
echo "=== Test Summary ==="
tail -20 test-results.log | grep -E "^(tests/|PASSED|FAILED|ERROR|=)"
