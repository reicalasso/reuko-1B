"""
Command Line Interface for Reuko-1B
"""

from .train import main as train_main
from .test import main as test_main
from .serve import main as serve_main

__all__ = ["train_main", "test_main", "serve_main"]
