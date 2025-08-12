"""Linguistic Distance Measurement Tools

This package provides tools to measure distance between embedding spaces
of different monolingual corpora.
"""

__version__ = "0.1.0"
__author__ = "Linguistic Distance Project"

from . import data, embeddings, alignment, distance, utils

__all__ = ["data", "embeddings", "alignment", "distance", "utils"]