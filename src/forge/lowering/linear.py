from __future__ import annotations

from .registry import OpPattern, register

# F.linear(x, weight, bias) を torch.fx で trace すると {linear:1}。
# bias=None の場合も同じパターンになる。その場合 spec.input_specs は x, weight の 2 テンソルで、
# ops/registry の検証入力も同じ数で生成する(bias ありは 3 テンソル)。
LINEAR_PATTERN = OpPattern(
    op_type="linear",
    op_counts={"linear": 1},
)

register(LINEAR_PATTERN)
