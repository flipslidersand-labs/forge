"""Forge — Automatic GPU kernel optimizer for PyTorch operations."""

from forge._version import __version__
from forge.decorator import optimize
from forge.prewarm import prewarm


def version_info() -> dict[str, str]:
    """Return structured version information."""
    major, minor, patch = __version__.split(".")
    return {"version": __version__, "major": major, "minor": minor, "patch": patch}


__all__ = ["optimize", "prewarm", "__version__", "version_info"]
