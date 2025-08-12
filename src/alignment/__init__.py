"""Embedding space alignment methods."""

from .linear_mapping import LinearMapping
from .procrustes import ProcrustesAlignment

__all__ = ["LinearMapping", "ProcrustesAlignment"]