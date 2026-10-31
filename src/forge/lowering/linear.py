from __future__ import annotations

from .registry import OpPattern, register

# F.linear(x, weight, bias) を torch.fx で trace すると {linear:1}。
# bias=None の場合も同じパターンになるが、n_tensor_inputs=3 の spec では bias を渡す。
LINEAR_PATTERN = OpPattern(
    op_type="linear",
    op_counts={"linear": 1},
)

register(LINEAR_PATTERN)
