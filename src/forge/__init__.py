"""Forge — Automatic GPU kernel optimizer for PyTorch operations."""

from forge._version import __version__
from forge.decorator import optimize
from forge.prewarm import prewarm


def version_info() -> dict[str, str]:
    """Return structured version information."""
    # dev 版 (1.2.3.dev4+gabc) や未インストール版 (0.0.0+unknown) でも壊れないよう先頭3要素を数値部で取る
    core = __version__.split("+")[0].split(".")[:3]
    major, minor, patch = (core + ["0", "0", "0"])[:3]
    return {"version": __version__, "major": major, "minor": minor, "patch": patch}


__all__ = ["optimize", "prewarm", "__version__", "version_info"]
