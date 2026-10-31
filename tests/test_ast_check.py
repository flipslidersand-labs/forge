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
