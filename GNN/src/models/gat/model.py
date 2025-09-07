"""
Main GAT model interface for portfolio optimization.

This module provides the high-level interface for the Graph Attention Network
implementation, integrating the GAT architecture with graph construction utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch

from .gat_model import HeadCfg
from .graph_builder import GraphBuildConfig

__all__ = [
    "GATPortfolioModel",
    "GATModelConfig",
]


@dataclass
class GATModelConfig:
    """Configuration for GAT portfolio model."""

    # GAT model configuration
    head_config: HeadCfg
    # Graph construction configuration
    graph_config: GraphBuildConfig
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 100


class GATPortfolioModel:
    """
    High-level interface for GAT-based portfolio optimization.

    This class integrates graph construction, GAT model training, and portfolio
    weight prediction into a unified interface.
    """

    def __init__(self, config: GATModelConfig):
        """
        Initialize GAT portfolio model.

        Args:
            config: Model configuration including GAT and graph parameters
        """
        self.config = config
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        returns: pd.DataFrame,
        universe: list[str],
        fit_period: tuple[pd.Timestamp, pd.Timestamp],
    ) -> None:
        """
        Train GAT model on historical data.

        Args:
            returns: Historical returns DataFrame
            universe: List of asset tickers
            fit_period: (start_date, end_date) for training period
        """
        # This will be implemented in future stories
        # For now, preserve existing GAT functionality
        pass

    def predict_weights(self, date: pd.Timestamp, universe: list[str]) -> pd.Series:
        """
        Generate portfolio weights for rebalancing date.

        Args:
            date: Rebalancing date
            universe: List of asset tickers

        Returns:
            Portfolio weights as pandas Series
        """
        # This will be implemented in future stories
        # For now, return equal weights to maintain functionality
        equal_weights = 1.0 / len(universe)
        return pd.Series(equal_weights, index=universe)

    def get_model_info(self) -> dict[str, Any]:
        """
        Return model metadata for analysis.

        Returns:
            Dictionary containing model information
        """
        return {
            "model_type": "GAT",
            "head_config": self.config.head_config,
            "graph_config": self.config.graph_config,
            "device": str(self.device),
        }
