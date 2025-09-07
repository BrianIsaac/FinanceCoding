"""
Abstract base model interface for portfolio optimization models.

This module defines the unified interface that all portfolio models must implement,
ensuring consistent APIs across different ML approaches (HRP, LSTM, GAT).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class PortfolioConstraints:
    """
    Portfolio constraints configuration.

    Defines constraints that apply to all portfolio optimization models
    to ensure consistent risk management and regulatory compliance.
    """

    long_only: bool = True
    top_k_positions: Optional[int] = None
    max_position_weight: float = 0.10
    max_monthly_turnover: float = 0.20
    transaction_cost_bps: float = 10.0
    min_weight_threshold: float = 0.01


class PortfolioModel(ABC):
    """
    Abstract base class for portfolio optimization models.

    All portfolio models (HRP, LSTM, GAT, baselines) must inherit from this class
    and implement the required methods. This ensures a unified interface for
    backtesting, evaluation, and production deployment.
    """

    def __init__(self, constraints: PortfolioConstraints):
        """
        Initialize portfolio model with constraints.

        Args:
            constraints: Portfolio constraints configuration
        """
        self.constraints = constraints
        self.is_fitted = False

    @abstractmethod
    def fit(
        self,
        returns: pd.DataFrame,
        universe: List[str],
        fit_period: Tuple[pd.Timestamp, pd.Timestamp],
    ) -> None:
        """
        Train model on historical data.

        Args:
            returns: Historical returns DataFrame with datetime index and asset columns
            universe: List of asset tickers to include in optimization
            fit_period: (start_date, end_date) tuple defining training period

        Raises:
            ValueError: If returns data is insufficient or invalid
        """
        pass

    @abstractmethod
    def predict_weights(self, date: pd.Timestamp, universe: List[str]) -> pd.Series:
        """
        Generate portfolio weights for rebalancing date.

        Args:
            date: Rebalancing date for which to generate weights
            universe: List of asset tickers (must be subset of fitted universe)

        Returns:
            Portfolio weights as pandas Series with asset tickers as index.
            Weights must sum to 1.0 and satisfy all portfolio constraints.

        Raises:
            ValueError: If model is not fitted or universe is invalid
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Return model metadata for analysis and reproducibility.

        Returns:
            Dictionary containing model type, hyperparameters, constraints,
            and other relevant metadata for performance analysis.
        """
        pass

    def validate_weights(self, weights: pd.Series) -> pd.Series:
        """
        Validate and enforce portfolio constraints on weights.

        Args:
            weights: Raw portfolio weights

        Returns:
            Constrained weights that satisfy all portfolio constraints
        """
        # Ensure long-only constraint
        if self.constraints.long_only:
            weights = weights.clip(lower=0.0)

        # Apply minimum weight threshold
        weights = weights.where(weights >= self.constraints.min_weight_threshold, 0.0)

        # Apply maximum position weight constraint
        if self.constraints.max_position_weight < 1.0:
            weights = weights.clip(upper=self.constraints.max_position_weight)

        # Apply top-k positions constraint
        if self.constraints.top_k_positions is not None:
            top_k_weights = weights.nlargest(self.constraints.top_k_positions)
            weights = weights.where(weights.isin(top_k_weights), 0.0)

        # Normalize to sum to 1.0
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            # If all weights are zero, use equal weights
            weights = pd.Series(1.0 / len(weights), index=weights.index)

        return weights
