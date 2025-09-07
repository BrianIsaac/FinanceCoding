"""
Abstract base model interface for portfolio optimization models.

This module defines the unified interface that all portfolio models must implement,
ensuring consistent APIs across different ML approaches (HRP, LSTM, GAT).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class PortfolioConstraints:
    """
    Portfolio constraints configuration.

    Defines constraints that apply to all portfolio optimization models
    to ensure consistent risk management and regulatory compliance.
    """

    long_only: bool = True
    top_k_positions: int | None = None
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
        universe: list[str],
        fit_period: tuple[pd.Timestamp, pd.Timestamp],
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
    def predict_weights(self, date: pd.Timestamp, universe: list[str]) -> pd.Series:
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
    def get_model_info(self) -> dict[str, Any]:
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

        # Apply top-k positions constraint first
        if self.constraints.top_k_positions is not None:
            top_k_weights = weights.nlargest(self.constraints.top_k_positions)
            weights = weights.where(weights.isin(top_k_weights), 0.0)

        # Final normalization to ensure weights sum to 1.0
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights / weight_sum

        # Handle max position weight constraint after normalization
        if self.constraints.max_position_weight < 1.0:
            # Iteratively adjust weights that exceed the constraint
            max_iters = 10
            for _ in range(max_iters):
                violating_mask = weights > self.constraints.max_position_weight
                if not violating_mask.any():
                    break

                # Clip violating weights to max allowed
                excess_weight = (
                    weights[violating_mask] - self.constraints.max_position_weight
                ).sum()
                weights[violating_mask] = self.constraints.max_position_weight

                # Redistribute excess to non-violating assets
                non_violating_mask = ~violating_mask
                if non_violating_mask.any():
                    available_capacity = (
                        self.constraints.max_position_weight - weights[non_violating_mask]
                    ).clip(lower=0)
                    total_capacity = available_capacity.sum()

                    if total_capacity > 0:
                        # Redistribute proportionally based on available capacity
                        redistribution = excess_weight * (available_capacity / total_capacity)
                        weights[non_violating_mask] += redistribution
                    else:
                        # No capacity left - the remaining excess represents cash
                        break
        else:
            # If all weights are zero, use constrained equal weights
            n_assets = len(weights)
            if self.constraints.max_position_weight * n_assets >= 1.0:
                # Equal weights respect constraint
                weights = pd.Series(1.0 / n_assets, index=weights.index)
            else:
                # Use max allowed weight and leave remainder as cash
                weights = pd.Series(self.constraints.max_position_weight, index=weights.index)

        return weights
