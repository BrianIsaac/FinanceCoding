"""
LSTM-based portfolio optimization models.

This package implements LSTM networks with multi-head attention for temporal pattern
recognition and portfolio weight prediction with memory-optimized training.
"""

from .architecture import (
    LSTMConfig,
    LSTMNetwork,
    MultiHeadAttention,
    SharpeRatioLoss,
    create_lstm_network,
)
from .model import LSTMModelConfig, LSTMPortfolioModel, create_lstm_model
from .training import MemoryEfficientTrainer, TimeSeriesDataset, TrainingConfig, create_trainer

__all__ = [
    # Architecture components
    "LSTMConfig",
    "LSTMNetwork",
    "MultiHeadAttention",
    "SharpeRatioLoss",
    "create_lstm_network",
    # Main model
    "LSTMModelConfig",
    "LSTMPortfolioModel",
    "create_lstm_model",
    # Training components
    "MemoryEfficientTrainer",
    "TimeSeriesDataset",
    "TrainingConfig",
    "create_trainer",
]
