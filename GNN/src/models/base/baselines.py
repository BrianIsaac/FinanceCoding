"""
Baseline portfolio models for comparative analysis.

This module provides simple baseline models that serve as benchmarks
for comparison against more sophisticated ML approaches (HRP, LSTM, GAT).
These models implement common portfolio construction strategies.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import PortfolioModel
from src.models.base.constraints import PortfolioConstraints

logger = logging.getLogger(__name__)


class EqualWeightModel(PortfolioModel):
    """
    Equal weight baseline model.

    Allocates equal weights to all assets in the universe.
    This serves as a naive baseline for comparison.
    """

    def __init__(self, constraints: PortfolioConstraints | None = None):
        """Initialize equal weight model."""
        if constraints is None:
            constraints = PortfolioConstraints()
        super().__init__(constraints)
        self.name = "EqualWeight"

    def fit(
        self,
        returns: pd.DataFrame,
        universe: list[str],
        fit_period: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    ) -> None:
        """
        Fit equal weight model (no actual training required).

        Args:
            returns: Historical returns data
            universe: Asset universe
            fit_period: Training period (not used for equal weight)
        """
        self.universe = universe
        self.is_fitted = True
        logger.debug(f"EqualWeight model fitted with {len(universe)} assets")

    def predict_weights(
        self,
        date: pd.Timestamp,
        universe: list[str] | None = None,
    ) -> pd.Series:
        """
        Generate equal weights for universe.

        Args:
            date: Prediction date
            universe: Asset universe (uses fitted universe if None)

        Returns:
            Equal weights for all assets
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        target_universe = universe or self.universe
        if not target_universe:
            return pd.Series(dtype=float)

        equal_weight = 1.0 / len(target_universe)
        weights = pd.Series(
            equal_weight,
            index=target_universe,
            name="equal_weights"
        )

        return weights

    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata for analysis."""
        return {
            "model_type": "EqualWeight",
            "model_name": self.name,
            "description": "Equal weight allocation across all assets",
            "hyperparameters": {},
            "constraints": self.constraints.__dict__ if hasattr(self.constraints, '__dict__') else {},
        }


class MarketCapWeightedModel(PortfolioModel):
    """
    Market capitalization weighted baseline model.

    Allocates weights proportional to market capitalisation.
    Uses returns volatility as a proxy for market cap when actual
    market cap data is not available.
    """

    def __init__(self, lookback_days: int = 252, constraints: PortfolioConstraints | None = None):
        """
        Initialize market cap weighted model.

        Args:
            lookback_days: Days to use for volatility calculation
            constraints: Portfolio constraints
        """
        if constraints is None:
            constraints = PortfolioConstraints()
        super().__init__(constraints)
        self.name = "MarketCapWeighted"
        self.lookback_days = lookback_days
        self.returns_data = None

    def fit(
        self,
        returns: pd.DataFrame,
        universe: list[str],
        fit_period: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    ) -> None:
        """
        Fit market cap weighted model.

        Args:
            returns: Historical returns data
            universe: Asset universe
            fit_period: Training period
        """
        self.universe = universe
        self.returns_data = returns
        self.is_fitted = True
        logger.debug(f"MarketCapWeighted model fitted with {len(universe)} assets")

    def predict_weights(
        self,
        date: pd.Timestamp,
        universe: list[str] | None = None,
    ) -> pd.Series:
        """
        Generate market cap weighted portfolio.

        Uses inverse volatility as proxy for market cap when actual
        market cap data is unavailable.

        Args:
            date: Prediction date
            universe: Asset universe (uses fitted universe if None)

        Returns:
            Market cap weighted portfolio
        """
        if not self.is_fitted or self.returns_data is None:
            raise ValueError("Model must be fitted before prediction")

        target_universe = universe or self.universe
        if not target_universe:
            return pd.Series(dtype=float)

        # Get historical returns up to prediction date
        end_date = date
        start_date = end_date - pd.Timedelta(days=self.lookback_days)

        historical_returns = self.returns_data.loc[
            start_date:end_date, target_universe
        ]

        if historical_returns.empty:
            # Fallback to equal weights if no data
            equal_weight = 1.0 / len(target_universe)
            return pd.Series(
                equal_weight,
                index=target_universe,
                name="market_cap_weights"
            )

        # Calculate volatilities
        volatilities = historical_returns.std()

        # Use inverse volatility as proxy for market cap
        # (lower volatility = larger, more stable companies)
        inverse_vol = 1.0 / (volatilities + 1e-8)  # Add small constant to avoid division by zero

        # Normalise to get weights
        weights = inverse_vol / inverse_vol.sum()
        weights.name = "market_cap_weights"

        return weights

    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata for analysis."""
        return {
            "model_type": "MarketCapWeighted",
            "model_name": self.name,
            "description": "Market cap weighted allocation using inverse volatility proxy",
            "hyperparameters": {"lookback_days": self.lookback_days},
            "constraints": self.constraints.__dict__ if hasattr(self.constraints, '__dict__') else {},
        }


class MeanReversionModel(PortfolioModel):
    """
    Mean reversion baseline model.

    Allocates higher weights to assets that have underperformed
    recently, based on mean reversion principle.
    """

    def __init__(self, lookback_days: int = 21, reversion_strength: float = 1.0, constraints: PortfolioConstraints | None = None):
        """
        Initialize mean reversion model.

        Args:
            lookback_days: Days to look back for performance calculation
            reversion_strength: Strength of mean reversion signal (1.0 = full reversion)
            constraints: Portfolio constraints
        """
        if constraints is None:
            constraints = PortfolioConstraints()
        super().__init__(constraints)
        self.name = "MeanReversion"
        self.lookback_days = lookback_days
        self.reversion_strength = reversion_strength
        self.returns_data = None

    def fit(
        self,
        returns: pd.DataFrame,
        universe: list[str],
        fit_period: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    ) -> None:
        """
        Fit mean reversion model.

        Args:
            returns: Historical returns data
            universe: Asset universe
            fit_period: Training period
        """
        self.universe = universe
        self.returns_data = returns
        self.is_fitted = True
        logger.debug(f"MeanReversion model fitted with {len(universe)} assets")

    def predict_weights(
        self,
        date: pd.Timestamp,
        universe: list[str] | None = None,
    ) -> pd.Series:
        """
        Generate mean reversion weighted portfolio.

        Assets with worse recent performance receive higher weights.

        Args:
            date: Prediction date
            universe: Asset universe (uses fitted universe if None)

        Returns:
            Mean reversion weighted portfolio
        """
        if not self.is_fitted or self.returns_data is None:
            raise ValueError("Model must be fitted before prediction")

        target_universe = universe or self.universe
        if not target_universe:
            return pd.Series(dtype=float)

        # Get recent returns for mean reversion signal
        end_date = date
        start_date = end_date - pd.Timedelta(days=self.lookback_days)

        recent_returns = self.returns_data.loc[
            start_date:end_date, target_universe
        ]

        if recent_returns.empty:
            # Fallback to equal weights if no data
            equal_weight = 1.0 / len(target_universe)
            return pd.Series(
                equal_weight,
                index=target_universe,
                name="mean_reversion_weights"
            )

        # Calculate cumulative returns over lookback period
        cumulative_returns = (1 + recent_returns).prod() - 1

        # Generate mean reversion signal
        # Lower returns get higher weights (mean reversion)
        reversion_signal = -cumulative_returns * self.reversion_strength

        # Convert to weights using softmax-like transformation
        # Shift signal to make all values positive
        shifted_signal = reversion_signal - reversion_signal.min() + 0.1

        # Apply exponential transformation
        exp_signal = np.exp(shifted_signal)
        weights = exp_signal / exp_signal.sum()

        weights.name = "mean_reversion_weights"

        return weights

    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata for analysis."""
        return {
            "model_type": "MeanReversion",
            "model_name": self.name,
            "description": "Mean reversion allocation favouring underperforming assets",
            "hyperparameters": {
                "lookback_days": self.lookback_days,
                "reversion_strength": self.reversion_strength
            },
            "constraints": self.constraints.__dict__ if hasattr(self.constraints, '__dict__') else {},
        }


class MinimumVarianceModel(PortfolioModel):
    """
    Minimum variance baseline model.

    Constructs a portfolio that minimises variance based on
    historical covariance matrix.
    """

    def __init__(self, lookback_days: int = 252, regularization: float = 1e-4, constraints: PortfolioConstraints | None = None):
        """
        Initialize minimum variance model.

        Args:
            lookback_days: Days to use for covariance estimation
            regularization: Regularization parameter for covariance matrix
            constraints: Portfolio constraints
        """
        if constraints is None:
            constraints = PortfolioConstraints()
        super().__init__(constraints)
        self.name = "MinimumVariance"
        self.lookback_days = lookback_days
        self.regularization = regularization
        self.returns_data = None

    def fit(
        self,
        returns: pd.DataFrame,
        universe: list[str],
        fit_period: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    ) -> None:
        """
        Fit minimum variance model.

        Args:
            returns: Historical returns data
            universe: Asset universe
            fit_period: Training period
        """
        self.universe = universe
        self.returns_data = returns
        self.is_fitted = True
        logger.debug(f"MinimumVariance model fitted with {len(universe)} assets")

    def predict_weights(
        self,
        date: pd.Timestamp,
        universe: list[str] | None = None,
    ) -> pd.Series:
        """
        Generate minimum variance portfolio.

        Args:
            date: Prediction date
            universe: Asset universe (uses fitted universe if None)

        Returns:
            Minimum variance weighted portfolio
        """
        if not self.is_fitted or self.returns_data is None:
            raise ValueError("Model must be fitted before prediction")

        target_universe = universe or self.universe
        if not target_universe:
            return pd.Series(dtype=float)

        # Get historical returns for covariance estimation
        end_date = date
        start_date = end_date - pd.Timedelta(days=self.lookback_days)

        historical_returns = self.returns_data.loc[
            start_date:end_date, target_universe
        ]

        if historical_returns.empty or len(historical_returns) < 10:
            # Fallback to equal weights if insufficient data
            equal_weight = 1.0 / len(target_universe)
            return pd.Series(
                equal_weight,
                index=target_universe,
                name="min_variance_weights"
            )

        # Calculate covariance matrix
        cov_matrix = historical_returns.cov()

        # Add regularization to improve numerical stability
        n_assets = len(cov_matrix)
        cov_matrix += self.regularization * np.eye(n_assets)

        try:
            # Compute minimum variance weights
            # w = (Σ^-1 * 1) / (1^T * Σ^-1 * 1)
            inv_cov = np.linalg.inv(cov_matrix.values)
            ones = np.ones((n_assets, 1))

            numerator = inv_cov @ ones
            denominator = ones.T @ inv_cov @ ones

            weights = (numerator / denominator).flatten()

            # Ensure weights are positive and sum to 1
            weights = np.maximum(weights, 0)
            weights = weights / weights.sum()

            weights_series = pd.Series(
                weights,
                index=target_universe,
                name="min_variance_weights"
            )

            return weights_series

        except np.linalg.LinAlgError:
            # Fallback to equal weights if matrix is singular
            logger.warning("Singular covariance matrix, falling back to equal weights")
            equal_weight = 1.0 / len(target_universe)
            return pd.Series(
                equal_weight,
                index=target_universe,
                name="min_variance_weights"
            )

    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata for analysis."""
        return {
            "model_type": "MinimumVariance",
            "model_name": self.name,
            "description": "Minimum variance portfolio optimisation",
            "hyperparameters": {
                "lookback_days": self.lookback_days,
                "regularization": self.regularization
            },
            "constraints": self.constraints.__dict__ if hasattr(self.constraints, '__dict__') else {},
        }


class MomentumModel(PortfolioModel):
    """
    Momentum baseline model.

    Allocates higher weights to assets with strong recent performance,
    based on momentum principle.
    """

    def __init__(self, lookback_days: int = 63, momentum_strength: float = 1.0, constraints: PortfolioConstraints | None = None):
        """
        Initialize momentum model.

        Args:
            lookback_days: Days to look back for momentum calculation
            momentum_strength: Strength of momentum signal
            constraints: Portfolio constraints
        """
        if constraints is None:
            constraints = PortfolioConstraints()
        super().__init__(constraints)
        self.name = "Momentum"
        self.lookback_days = lookback_days
        self.momentum_strength = momentum_strength
        self.returns_data = None

    def fit(
        self,
        returns: pd.DataFrame,
        universe: list[str],
        fit_period: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    ) -> None:
        """
        Fit momentum model.

        Args:
            returns: Historical returns data
            universe: Asset universe
            fit_period: Training period
        """
        self.universe = universe
        self.returns_data = returns
        self.is_fitted = True
        logger.debug(f"Momentum model fitted with {len(universe)} assets")

    def predict_weights(
        self,
        date: pd.Timestamp,
        universe: list[str] | None = None,
    ) -> pd.Series:
        """
        Generate momentum weighted portfolio.

        Assets with better recent performance receive higher weights.

        Args:
            date: Prediction date
            universe: Asset universe (uses fitted universe if None)

        Returns:
            Momentum weighted portfolio
        """
        if not self.is_fitted or self.returns_data is None:
            raise ValueError("Model must be fitted before prediction")

        target_universe = universe or self.universe
        if not target_universe:
            return pd.Series(dtype=float)

        # Get recent returns for momentum signal
        end_date = date
        start_date = end_date - pd.Timedelta(days=self.lookback_days)

        recent_returns = self.returns_data.loc[
            start_date:end_date, target_universe
        ]

        if recent_returns.empty:
            # Fallback to equal weights if no data
            equal_weight = 1.0 / len(target_universe)
            return pd.Series(
                equal_weight,
                index=target_universe,
                name="momentum_weights"
            )

        # Calculate cumulative returns over lookback period
        cumulative_returns = (1 + recent_returns).prod() - 1

        # Generate momentum signal
        momentum_signal = cumulative_returns * self.momentum_strength

        # Convert to weights using softmax-like transformation
        # Shift signal to make all values positive
        shifted_signal = momentum_signal - momentum_signal.min() + 0.1

        # Apply exponential transformation
        exp_signal = np.exp(shifted_signal)
        weights = exp_signal / exp_signal.sum()

        weights.name = "momentum_weights"

        return weights

    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata for analysis."""
        return {
            "model_type": "Momentum",
            "model_name": self.name,
            "description": "Momentum allocation favouring recent outperformers",
            "hyperparameters": {
                "lookback_days": self.lookback_days,
                "momentum_strength": self.momentum_strength
            },
            "constraints": self.constraints.__dict__ if hasattr(self.constraints, '__dict__') else {},
        }
