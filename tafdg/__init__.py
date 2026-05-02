"""TAFDG clean implementation."""

from .config import TAFDGConfig
from .trainer import TAFDGTrainer
from .data import build_benchmark

__all__ = ["TAFDGConfig", "TAFDGTrainer", "build_benchmark"]
