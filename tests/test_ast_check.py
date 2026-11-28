import pytest

from forge.security.ast_check import check_generated_code

OK = """
import math
import torch
import triton
import triton.language as tl
from triton import jit

@triton.jit
def k(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    x = tl.load(x_ptr + pid * BLOCK + tl.arange(0, BLOCK))
    tl.store(x_ptr + pid, x * math.pi)
"""


def test_valid_triton_passes():
    assert check_generated_code(OK) == []


@pytest.mark.parametrize(
    "src",
    [
        "import os",
        "import subprocess as sp",
        "import sys, math",
        "import socket",
        "import shutil",
        "import ctypes",
        "from os import path",
        "from . import x",
        "import torch.x\nimport numpy",
        "eval('1')",
        "exec('x=1')",
        "compile('1', 'f', 'eval')",
        "__import__('os')",
        "open('/etc/passwd')",
        "f = eval\nf('1')",
        "x = (1).__class__",
        "torch.__dict__",
    ],
)
def test_forbidden(src):
    assert check_generated_code(src)


def test_syntax_error_is_violation_not_exception():
    out = check_generated_code("def (:")
    assert len(out) == 1 and "構文エラー" in out[0]


def test_submodule_of_allowed_ok():
    assert check_generated_code("import torch.nn.functional as F") == []


# 許可モジュール (torch など) の内側から os 等へ抜ける経路と、禁止名リストの迂回経路 (#474)。
# AST チェックは完全なサンドボックスではない (ast_check.py の docstring / ADR-003 参照)
# が、既知のバイパスは禁止リストで塞ぐ。
# ここでは文字列を check_generated_code に渡すだけで、コードは実行しない。
@pytest.mark.parametrize(
    "src",
    [
        pytest.param(
            "import torch\nfrom torch import os\nos.system('id')",
            id="from-torch-import-os",
        ),
        pytest.param("from torch import os as o\no.system('id')", id="from-torch-import-os-alias"),
        pytest.param("from torch import sys", id="from-torch-import-sys"),
        pytest.param("from triton import subprocess", id="from-triton-import-subprocess"),
        pytest.param("from torch import (nn, os)", id="from-torch-import-tuple-os"),
        pytest.param("from torch import __dict__", id="from-torch-import-dunder"),
        pytest.param("import torch.os", id="import-torch-dot-os"),
        pytest.param("import torch.os as o", id="import-torch-dot-os-alias"),
        pytest.param("from torch.os import system", id="from-torch-dot-os-import"),
        pytest.param("from torch import *", id="from-torch-import-star"),
        pytest.param("from triton import *", id="from-triton-import-star"),
        pytest.param(
            "import torch\ngetattr(torch, 'os').system('id')",
            id="getattr-torch-os",
        ),
        pytest.param("import torch\ntorch.os.system('id')", id="torch-dot-os"),
        pytest.param("import torch\ntorch.sys.modules", id="torch-dot-sys"),
        pytest.param("import torch\ntorch.subprocess.run(['id'])", id="torch-dot-subprocess"),
        pytest.param("import torch\nx = torch.builtins", id="torch-dot-builtins"),
        pytest.param("g = globals()", id="globals"),
        pytest.param("g = locals()", id="locals"),
        pytest.param("g = vars()", id="vars"),
        pytest.param("import torch\nsetattr(torch, 'x', 1)", id="setattr"),
        pytest.param("import torch\ndelattr(torch, 'x')", id="delattr"),
        pytest.param("f = getattr\nf(1, 'real')", id="getattr-alias"),
        pytest.param("b = __builtins__", id="dunder-builtins-name"),
    ],
)
def test_known_bypass_is_rejected(src):
    assert check_generated_code(src)


@pytest.mark.parametrize(
    "src",
    [
        "from torch import nn",
        "from torch import Tensor, float16",
        "from torch.nn import functional as F",
        "from triton import language as tl",
        "from triton import jit, cdiv",
        "from math import pi",
        "import torch.nn.functional as F\nF.relu(1)",
    ],
)
def test_legit_symbol_imports_still_pass(src):
    assert check_generated_code(src) == []


def _all_template_cases():
    from forge.codegen.triton_codegen import _TEMPLATES

    return sorted(_TEMPLATES)


# 既存の全テンプレートが生成するコードが、強化後の検査を通ること。
@pytest.mark.parametrize(("op", "variant"), _all_template_cases())
def test_all_template_outputs_pass(op, variant):
    import torch

    from forge.codegen.triton_codegen import generate
    from forge.ir.kernel_spec import KernelSpec
    from forge.ir.tensor_spec import TensorSpec
    from forge.search.params import SearchParams

    shapes = {
        "fused_add_rmsnorm": [(8, 64), (8, 64), (64,)],
        "layernorm": [(8, 64), (64,), (64,)],
        "rmsnorm": [(8, 64), (64,)],
        "linear": [(8, 64), (32, 64), (32,)],
        "scaled_dot_product_attention": [(2, 32, 64)] * 3,
        "rope": [(8, 64)] * 3,
        "swiglu": [(8, 64)] * 2,
    }.get(op, [(8, 64)])
    ins = tuple(TensorSpec(s, torch.float16, True) for s in shapes)
    for causal in (False, True) if op == "scaled_dot_product_attention" else (False,):
        spec = KernelSpec(
            op_type=op,
            input_specs=ins,
            output_specs=(ins[0],),
            constants={"is_causal": causal} if causal else {},
            graph_hash=f"ast_check_{op}_{variant}",
            constraints=(),
        )
        params = SearchParams(
            block_size=128,
            num_warps=4,
            num_stages=1,
            acc_dtype="fp32",
            variant=variant,
            rows_per_program=2 if variant == "multi_row" else 1,
        )
        assert check_generated_code(generate(spec, params)) == []


# torch 自身のサブモジュール経由の経路は、許可範囲 (torch を絞るか) の決定 (#344) まで
# 塞がない。現状は違反なしで通る。塞いだら XPASS(strict) で落ちるので xfail を外すこと。
@pytest.mark.xfail(strict=True, reason="残存バイパス: torch 自身のサブモジュール経由 (#344 で決定)")
@pytest.mark.parametrize(
    "src",
    [
        pytest.param("from torch.utils import cpp_extension", id="torch-utils-cpp-extension"),
        pytest.param("import torch.hub", id="torch-hub"),
    ],
)
def test_remaining_bypass_via_torch_submodule(src):
    assert check_generated_code(src)
