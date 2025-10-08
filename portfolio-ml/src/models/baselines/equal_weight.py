"""
Equal-weight portfolio allocation model.

This module implements a simple equal-weight baseline that distributes
capital equally across the top-k securities in the investment universe.
Serves as a baseline for comparison with more sophisticated ML models.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.models.base.portfolio_model import PortfolioConstraints, PortfolioModel


class EqualWeightModel(PortfolioModel):
    """
    Equal-weight portfolio allocation model.

    Distributes capital equally across top-k securities based on
    configurable selection criteria. Provides a simple baseline
    for evaluating more sophisticated portfolio optimization approaches.
    """

    def __init__(self, constraints: PortfolioConstraints, top_k: int = 50):
        """
        Initialize equal-weight model.

        Args:
            constraints: Portfolio constraints configuration
            top_k: Number of top securities to include (default: 50)
        """
        super().__init__(constraints)
        self.top_k = top_k
        self.fitted_universe: list[str] | None = None
        self.selection_method = "alphabetical"  # Simple deterministic selection

    def fit(
        self,
        returns: pd.DataFrame,
        universe: list[str],
        fit_period: tuple[pd.Timestamp, pd.Timestamp],
    ) -> None:
        """
        Fit the equal-weight model.

        For equal-weight allocation, fitting simply stores the universe
        and validates the data. No actual training is required.

        Args:
            returns: Historical returns DataFrame with datetime index and asset columns
            universe: List of asset tickers to include in optimization
            fit_period: (start_date, end_date) tuple defining training period

        Raises:
            ValueError: If returns data is insufficient or invalid
        """
        if returns.empty:
            raise ValueError("Returns data cannot be empty")

        if len(universe) == 0:
            raise ValueError("Universe cannot be empty")

        # Validate that fit_period has valid dates
        start_date, end_date = fit_period
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")

        # Filter returns to fit period and universe
        period_mask = (returns.index >= start_date) & (returns.index <= end_date)
        available_returns = returns.loc[period_mask, universe]

        if available_returns.empty:
            raise ValueError(f"No return data available for period {start_date} to {end_date}")

        # Store fitted universe (assets with sufficient data)
        min_observations = 20  # Minimum 20 trading days of data
        sufficient_data = available_returns.count() >= min_observations
        self.fitted_universe = [asset for asset in universe if sufficient_data[asset]]

        if len(self.fitted_universe) == 0:
            raise ValueError("No assets have sufficient data for fitting")

        self.is_fitted = True

    def predict_weights(self, date: pd.Timestamp, universe: list[str]) -> pd.Series:
        """
        Generate equal-weight portfolio weights for rebalancing date.

        Args:
            date: Rebalancing date for which to generate weights
            universe: List of asset tickers (must be subset of fitted universe)

        Returns:
            Portfolio weights as pandas Series with asset tickers as index.
            Weights are equal for top-k assets and 0 for others.

        Raises:
            ValueError: If model is not fitted or universe is invalid
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating predictions")

        if len(universe) == 0:
            raise ValueError("Universe cannot be empty")

        # Filter universe to assets that were in fitted universe
        if self.fitted_universe is None:
            raise ValueError("Model must be fitted before generating predictions")

        valid_assets = [asset for asset in universe if asset in self.fitted_universe]

        if len(valid_assets) == 0:
            raise ValueError("No assets in current universe were present during fitting")

        # Select top-k assets using deterministic method
        selected_assets = self._select_top_k_assets(valid_assets)

        # Create equal weights for selected assets
        weights = pd.Series(0.0, index=universe)

        if len(selected_assets) > 0:
            equal_weight = 1.0 / len(selected_assets)
            weights[selected_assets] = equal_weight

        # Apply portfolio constraints
        constrained_weights = self.validate_weights(weights)

        return constrained_weights

    def get_model_info(self) -> dict[str, Any]:
        """
        Return model metadata for analysis and reproducibility.

        Returns:
            Dictionary containing model type, hyperparameters, constraints,
            and other relevant metadata for performance analysis.
        """
        return {
            "model_type": "EqualWeight",
            "model_class": self.__class__.__name__,
            "top_k": self.top_k,
            "selection_method": self.selection_method,
            "fitted_universe_size": len(self.fitted_universe) if self.fitted_universe else 0,
            "is_fitted": self.is_fitted,
            "constraints": {
                "long_only": self.constraints.long_only,
                "top_k_positions": self.constraints.top_k_positions,
                "max_position_weight": self.constraints.max_position_weight,
                "max_monthly_turnover": self.constraints.max_monthly_turnover,
                "transaction_cost_bps": self.constraints.transaction_cost_bps,
                "min_weight_threshold": self.constraints.min_weight_threshold,
            },
        }

    def _select_top_k_assets(self, universe: list[str]) -> list[str]:
        """
        Select top-k assets from universe using configured selection method.

        Args:
            universe: List of available asset tickers

        Returns:
            List of selected asset tickers (length <= top_k)
        """
        # Use constraint's top_k_positions if available, otherwise model's top_k
        k = self.constraints.top_k_positions or self.top_k

        # Ensure k doesn't exceed universe size
        k = min(k, len(universe))

        if self.selection_method == "alphabetical":
            # Simple deterministic selection for reproducibility
            sorted_universe = sorted(universe)
            return sorted_universe[:k]
        else:
            # Future: could add market cap, liquidity, or other selection criteria
            sorted_universe = sorted(universe)
            return sorted_universe[:k]
