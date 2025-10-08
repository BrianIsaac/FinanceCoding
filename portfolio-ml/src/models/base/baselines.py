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
        self._is_baseline = True  # Mark as baseline model for rolling engine

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
        self._is_baseline = True  # Mark as baseline model for rolling engine
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
            logger.error(f"MarketCapWeighted model not fitted properly: is_fitted={self.is_fitted}, has_returns_data={self.returns_data is not None}")
            raise ValueError("Model must be fitted before prediction")

        target_universe = universe or self.universe
        if not target_universe:
            logger.warning(f"MarketCapWeighted: Empty target universe")
            return pd.Series(dtype=float)

        # Handle dynamic universe - filter to available assets
        available_assets = [asset for asset in target_universe if asset in self.returns_data.columns]
        unavailable_assets = [asset for asset in target_universe if asset not in self.returns_data.columns]

        if not available_assets:
            logger.warning(f"MarketCapWeighted: No assets from target universe available in fitted data. Using equal weights.")
            equal_weight = 1.0 / len(target_universe)
            return pd.Series(equal_weight, index=target_universe, name="market_cap_weights")

        if unavailable_assets:
            logger.debug(f"MarketCapWeighted: {len(unavailable_assets)} assets not in fitted data: {unavailable_assets[:5]}...")

        # Get historical returns up to prediction date
        end_date = date
        start_date = end_date - pd.Timedelta(days=self.lookback_days)

        historical_returns = self.returns_data.loc[
            start_date:end_date, available_assets
        ]

        logger.debug(f"MarketCapWeighted prediction for {date}: lookback {start_date} to {end_date}, data shape {historical_returns.shape}")

        if historical_returns.empty:
            # Fallback to equal weights if no data
            logger.warning(f"MarketCapWeighted: NO DATA available for period {start_date} to {end_date}, falling back to equal weights")
            equal_weight = 1.0 / len(target_universe)
            return pd.Series(
                equal_weight,
                index=target_universe,
                name="market_cap_weights"
            )

        # Clean outliers before volatility calculation to prevent extreme weights
        # First, identify and remove extreme outliers (likely data errors)
        # Any return > 200% or < -80% in a single day is likely erroneous
        extreme_threshold_upper = 2.0  # 200% daily return
        extreme_threshold_lower = -0.8  # -80% daily return

        # Replace extreme outliers with NaN first, then forward fill
        cleaned_returns = historical_returns.copy()
        cleaned_returns = cleaned_returns.where(
            (cleaned_returns <= extreme_threshold_upper) &
            (cleaned_returns >= extreme_threshold_lower),
            np.nan
        )

        # Forward fill NaN values from outlier removal
        cleaned_returns = cleaned_returns.ffill().bfill()

        # Secondary clipping to reasonable bounds for remaining values
        cleaned_returns = cleaned_returns.clip(lower=-0.5, upper=1.0)

        # Calculate volatilities on cleaned data with additional robustness
        # Use robust standard deviation (MAD-based) if available
        volatilities = cleaned_returns.std()

        # Additional safety: exclude assets with extreme volatilities
        # Use a more reasonable threshold based on financial theory
        # Most stocks have volatility between 15-60% annualised (0.01-0.04 daily)
        # Allow up to 3x median or 5% daily volatility, whichever is higher
        vol_median = volatilities.median()
        vol_threshold = max(vol_median * 3, 0.05)  # More reasonable threshold
        valid_assets = volatilities[volatilities <= vol_threshold].index

        if len(valid_assets) < len(volatilities):
            logger.warning(f"MarketCapWeighted: Excluding {len(volatilities) - len(valid_assets)} assets with extreme volatility")
            volatilities = volatilities[valid_assets]

        if volatilities.empty:
            # Fallback to equal weights if no valid assets
            logger.warning(f"MarketCapWeighted: No valid assets after volatility filtering, falling back to equal weights")
            equal_weight = 1.0 / len(target_universe)
            return pd.Series(
                equal_weight,
                index=target_universe,
                name="market_cap_weights"
            )

        # Use inverse volatility as proxy for market cap
        # (lower volatility = larger, more stable companies)
        # Add larger constant and cap inverse values to prevent extreme weights
        min_volatility = max(volatilities.min(), 0.001)  # At least 0.1% daily volatility
        adjusted_volatilities = volatilities.clip(lower=min_volatility)

        # Use inverse with reasonable bounds
        inverse_vol = 1.0 / adjusted_volatilities

        # Additional capping of extreme inverse volatilities
        inverse_vol_median = inverse_vol.median()
        inverse_vol_cap = inverse_vol_median * 5  # Cap at 5x median
        inverse_vol = inverse_vol.clip(upper=inverse_vol_cap)

        # Normalise to get weights (only for valid assets)
        weights = inverse_vol / inverse_vol.sum()

        # Expand weights back to full universe
        full_weights = pd.Series(0.0, index=target_universe, name="market_cap_weights")

        # For dynamic universe: allocate 95% to market cap weights, 5% to new assets
        if unavailable_assets:
            market_cap_allocation = 0.95
            new_asset_allocation = 0.05

            # Scale down market cap weights
            full_weights[weights.index] = weights * market_cap_allocation

            # Equal allocation to new assets
            new_asset_weight = new_asset_allocation / len(unavailable_assets)
            full_weights[unavailable_assets] = new_asset_weight
        else:
            # No new assets, use full market cap weights
            full_weights[weights.index] = weights

        weights = full_weights

        logger.debug(f"MarketCapWeighted generated weights: sum={weights.sum():.6f}, std={weights.std():.6f}, range=[{weights.min():.6f}, {weights.max():.6f}]")

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
        self._is_baseline = True  # Mark as baseline model for rolling engine
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
            logger.error(f"MeanReversion model not fitted properly: is_fitted={self.is_fitted}, has_returns_data={self.returns_data is not None}")
            raise ValueError("Model must be fitted before prediction")

        target_universe = universe or self.universe
        if not target_universe:
            logger.warning(f"MeanReversion: Empty target universe")
            return pd.Series(dtype=float)

        # Handle dynamic universe - filter to available assets
        available_assets = [asset for asset in target_universe if asset in self.returns_data.columns]
        unavailable_assets = [asset for asset in target_universe if asset not in self.returns_data.columns]

        if not available_assets:
            logger.warning(f"MeanReversion: No assets from target universe available in fitted data. Using equal weights.")
            equal_weight = 1.0 / len(target_universe)
            return pd.Series(equal_weight, index=target_universe, name="mean_reversion_weights")

        if unavailable_assets:
            logger.debug(f"MeanReversion: {len(unavailable_assets)} assets not in fitted data: {unavailable_assets[:5]}...")

        # Get recent returns for mean reversion signal
        end_date = date
        start_date = end_date - pd.Timedelta(days=self.lookback_days)

        recent_returns = self.returns_data.loc[
            start_date:end_date, available_assets
        ]

        logger.debug(f"MeanReversion prediction for {date}: lookback {start_date} to {end_date}, data shape {recent_returns.shape}")

        if recent_returns.empty:
            # Fallback to equal weights if no data
            logger.warning(f"MeanReversion: NO DATA available for period {start_date} to {end_date}, falling back to equal weights")
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

        # Expand weights to full universe
        full_weights = pd.Series(0.0, index=target_universe, name="mean_reversion_weights")

        # For dynamic universe: allocate 95% to mean reversion weights, 5% to new assets
        if unavailable_assets:
            mean_reversion_allocation = 0.95
            new_asset_allocation = 0.05

            # Scale down mean reversion weights
            full_weights[weights.index] = weights * mean_reversion_allocation

            # Equal allocation to new assets
            new_asset_weight = new_asset_allocation / len(unavailable_assets)
            full_weights[unavailable_assets] = new_asset_weight
        else:
            # No new assets, use full mean reversion weights
            full_weights[weights.index] = weights

        weights = full_weights

        logger.debug(f"MeanReversion generated weights: sum={weights.sum():.6f}, std={weights.std():.6f}, range=[{weights.min():.6f}, {weights.max():.6f}]")

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

        # Handle dynamic universe - filter to available assets
        available_assets = [asset for asset in target_universe if asset in self.returns_data.columns]
        unavailable_assets = [asset for asset in target_universe if asset not in self.returns_data.columns]

        if not available_assets:
            logger.warning(f"MinimumVariance: No assets from target universe available in fitted data. Using equal weights.")
            equal_weight = 1.0 / len(target_universe)
            return pd.Series(equal_weight, index=target_universe, name="min_variance_weights")

        if unavailable_assets:
            logger.debug(f"MinimumVariance: {len(unavailable_assets)} assets not in fitted data: {unavailable_assets[:5]}...")

        # Get historical returns for covariance estimation
        end_date = date
        start_date = end_date - pd.Timedelta(days=self.lookback_days)

        historical_returns = self.returns_data.loc[
            start_date:end_date, available_assets
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

            # Create weights series for available assets
            available_weights = pd.Series(
                weights,
                index=available_assets,
                name="min_variance_weights"
            )

            # Expand weights to full universe
            full_weights = pd.Series(0.0, index=target_universe, name="min_variance_weights")

            # For dynamic universe: allocate 95% to min variance weights, 5% to new assets
            if unavailable_assets:
                min_var_allocation = 0.95
                new_asset_allocation = 0.05

                # Scale down minimum variance weights
                full_weights[available_assets] = available_weights * min_var_allocation

                # Equal allocation to new assets
                new_asset_weight = new_asset_allocation / len(unavailable_assets)
                full_weights[unavailable_assets] = new_asset_weight
            else:
                # No new assets, use full minimum variance weights
                full_weights[available_assets] = available_weights

            return full_weights

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
