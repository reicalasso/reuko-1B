"""
Reuko-1B: Professional T5-based NLP Pipeline

A production-ready pipeline for Question Answering and Text Summarization
using T5 transformer models with modular architecture.
"""

__version__ = "0.1.0"
__author__ = "Rei Calasso"
__email__ = "rei@example.com"

from .models.t5_model import T5QAModel, T5SummarizationModel
from .pipeline.inference import ReukoInference
from .pipeline.trainer import ReukoTrainer
from .utils.config import ConfigManager
from .utils.logger import get_logger

__all__ = [
    "T5QAModel",
    "T5SummarizationModel", 
    "ReukoInference",
    "ReukoTrainer",
    "ConfigManager",
    "get_logger",
]
