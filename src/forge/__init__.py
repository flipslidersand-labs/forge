"""Forge — Automatic GPU kernel optimizer for PyTorch operations."""

from forge._version import __version__
from forge.decorator import optimize

__all__ = ["optimize", "__version__"]
