from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float
    equal_nan: bool = False

    def to_dict(self) -> dict[str, float | bool]:
        return {"atol": self.atol, "rtol": self.rtol, "equal_nan": self.equal_nan}


def get_tolerance(op_type: str) -> Tolerance:
    from forge.ops.registry import OP_REGISTRY

    if op_type not in OP_REGISTRY:
        raise ValueError(f"No tolerance defined for op_type={op_type!r}")
    return OP_REGISTRY[op_type].tolerance
