"""
Abstract base model interface for portfolio optimization models.

This module defines the unified interface that all portfolio models must implement,
ensuring consistent APIs across different ML approaches (HRP, LSTM, GAT).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from ...evaluation.backtest.transaction_costs import (
    TransactionCostCalculator,
    TransactionCostConfig,
)
from .constraint_engine import UnifiedConstraintEngine
from .constraints import PortfolioConstraints


class PortfolioModel(ABC):
    """
    Abstract base class for portfolio optimization models.

    All portfolio models (HRP, LSTM, GAT, baselines) must inherit from this class
    and implement the required methods. This ensures a unified interface for
    backtesting, evaluation, and production deployment with consistent constraint
    enforcement across all models.
    """

    def __init__(self, constraints: PortfolioConstraints):
        """
        Initialize portfolio model with unified constraint system.

        Args:
            constraints: Portfolio constraints configuration
        """
        self.constraints = constraints
        self.is_fitted = False
        self.universe_df: pd.DataFrame | None = None  # For membership-aware imputation

        # Initialize unified constraint engine with transaction cost integration
        transaction_config = TransactionCostConfig(
            linear_cost_bps=constraints.transaction_cost_bps,
            bid_ask_spread_bps=5.0,
        )
        transaction_calculator = TransactionCostCalculator(transaction_config)

        self.constraint_engine = UnifiedConstraintEngine(
            constraints=constraints,
            transaction_cost_calculator=transaction_calculator,
        )

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

    def rolling_fit(
        self,
        returns: pd.DataFrame,
        universe: list[str],
        rebalance_date: pd.Timestamp,
        lookback_months: int = 36,
        min_observations: int = 252,
    ) -> None:
        """
        Perform rolling fit for monthly retraining.

        This method enables models to retrain on a rolling window of data,
        updating model parameters based on recent market conditions while
        maintaining computational efficiency.

        Args:
            returns: Full historical returns DataFrame
            universe: Dynamic universe of assets for this rebalancing period
            rebalance_date: Date for which we're rebalancing (training uses data up to this date - 1)
            lookback_months: Number of months to look back for training data
            min_observations: Minimum number of observations required for training

        Raises:
            NotImplementedError: If model doesn't support rolling retraining
            ValueError: If insufficient data for rolling window
        """
        # Default implementation - call regular fit with rolling window
        end_date = rebalance_date - pd.Timedelta(days=1)
        start_date = end_date - pd.Timedelta(days=lookback_months * 30)

        # Filter returns for rolling window
        mask = (returns.index >= start_date) & (returns.index <= end_date)
        rolling_returns = returns[mask]

        if len(rolling_returns) < min_observations:
            raise ValueError(
                f"Insufficient data for rolling fit: {len(rolling_returns)} < {min_observations}"
            )

        # Call regular fit with rolling window
        self.fit(rolling_returns, universe, (start_date, end_date))

    def supports_rolling_retraining(self) -> bool:
        """
        Check if model supports rolling retraining.

        Returns:
            True if model supports efficient rolling retraining,
            False if model should use static training only
        """
        # Models can override this to indicate rolling support
        return False

    def validate_weights(
        self,
        weights: pd.Series,
        previous_weights: pd.Series | None = None,
        model_scores: pd.Series | None = None,
        date: pd.Timestamp | None = None,
        use_soft_constraints: bool = True,
    ) -> pd.Series:
        """
        Validate and enforce portfolio constraints using unified constraint engine.

        Now supports soft constraints for better real-world flexibility.

        Args:
            weights: Raw portfolio weights
            previous_weights: Previous period weights for turnover calculation
            model_scores: Model confidence scores (optional)
            date: Current date for adaptive constraints
            use_soft_constraints: Use soft penalties instead of hard limits
            model_scores: Model confidence scores for position ranking
            date: Current rebalancing date for logging

        Returns:
            Constrained weights that satisfy all portfolio constraints
        """
        # Use unified constraint engine for comprehensive constraint enforcement
        constrained_weights, violations, cost_analysis = (
            self.constraint_engine.enforce_all_constraints(
                weights=weights,
                previous_weights=previous_weights,
                model_scores=model_scores,
                date=date,
                use_soft_constraints=use_soft_constraints,
            )
        )

        # Store latest violations and cost analysis for reporting
        self._latest_violations = violations
        self._latest_cost_analysis = cost_analysis

        return constrained_weights

    def get_constraint_violations(self):
        """Get latest constraint violations from validation."""
        return getattr(self, "_latest_violations", [])

    def get_transaction_cost_analysis(self):
        """Get latest transaction cost analysis from validation."""
        return getattr(self, "_latest_cost_analysis", {})

    def validate_portfolio_feasibility(
        self, weights: pd.Series, previous_weights: pd.Series | None = None
    ):
        """
        Validate overall portfolio feasibility using unified constraint engine.

        Args:
            weights: Portfolio weights to validate
            previous_weights: Previous weights for comparison

        Returns:
            Feasibility assessment with recommendations
        """
        return self.constraint_engine.validate_portfolio_feasibility(weights, previous_weights)

    def create_constraint_report(
        self, weights: pd.Series, previous_weights: pd.Series | None = None
    ):
        """
        Create comprehensive constraint enforcement report.

        Args:
            weights: Current portfolio weights
            previous_weights: Previous weights for comparison

        Returns:
            Comprehensive constraint report
        """
        return self.constraint_engine.create_enforcement_report(weights, previous_weights)

    def set_universe_schedule(self, universe_df: pd.DataFrame | None) -> None:
        """
        Set the universe schedule for membership-aware data handling.

        This method should be called by the backtest engine before running backtests
        to enable membership-aware cross-sectional mean imputation. If not called,
        models will fall back to standard imputation without membership awareness.

        Args:
            universe_df: DataFrame with columns ['ticker', 'start', 'end'] defining
                when each asset is in the investable universe
                - 'ticker': str, asset ticker symbol
                - 'start': pd.Timestamp, date asset enters universe
                - 'end': pd.Timestamp, date asset exits universe

        Raises:
            ValueError: If universe_df has incorrect format

        Example:
            >>> universe_df = pd.DataFrame({
            ...     'ticker': ['AAPL', 'MSFT', 'GOOGL'],
            ...     'start': pd.Timestamp('2020-01-01'),
            ...     'end': pd.Timestamp('2025-01-01'),
            ... })
            >>> model.set_universe_schedule(universe_df)

        Note:
            This is optional. If not provided, models will fall back to standard
            imputation without membership awareness. The accuracy improvement from
            membership-aware imputation is typically 2-5% in cross-sectional mean
            estimation.
        """
        if universe_df is not None:
            # Validate format
            required_cols = ['ticker', 'start', 'end']
            if not all(col in universe_df.columns for col in required_cols):
                raise ValueError(
                    f"universe_df must have columns {required_cols}, "
                    f"got {list(universe_df.columns)}"
                )

            # Ensure date columns are timestamps
            if not pd.api.types.is_datetime64_any_dtype(universe_df['start']):
                universe_df['start'] = pd.to_datetime(universe_df['start'])
            if not pd.api.types.is_datetime64_any_dtype(universe_df['end']):
                universe_df['end'] = pd.to_datetime(universe_df['end'])

        self.universe_df = universe_df

    def expand_weights_to_universe(
        self,
        model_weights: pd.Series,
        full_universe: list[str],
        strategy: str = "zero_invalid",
    ) -> pd.Series:
        """
        Expand model weights to full universe.

        Models may generate weights for a filtered subset of the universe
        (assets meeting quality criteria). This method expands to full universe.

        Args:
            model_weights: Weights for valid assets (may be subset of universe)
            full_universe: Complete target universe
            strategy: How to handle invalid assets:
                - 'zero_invalid': Set invalid assets to 0
                - 'small_allocation': Give invalid assets 2% total, scale valid to 98%

        Returns:
            Weights series indexed by full universe, sums to 1.0

        Examples:
            >>> # Model generated weights for [A, B] but universe is [A, B, C]
            >>> model_weights = pd.Series([0.6, 0.4], index=['A', 'B'])
            >>> full_weights = expand_weights_to_universe(model_weights, ['A', 'B', 'C'])
            >>> # Returns Series([0.6, 0.4, 0.0], index=['A', 'B', 'C'])
        """
        import numpy as np

        full_weights = pd.Series(0.0, index=full_universe)

        # Copy model weights for valid assets
        valid_assets = [a for a in model_weights.index if a in full_universe]
        full_weights[valid_assets] = model_weights[valid_assets]

        if strategy == "small_allocation":
            # Give invalid assets small allocation
            invalid_assets = [a for a in full_universe if a not in valid_assets]

            if len(invalid_assets) > 0 and full_weights.sum() > 0:
                # Scale valid assets to 98%
                full_weights[valid_assets] *= 0.98

                # Distribute 2% among invalid assets
                full_weights[invalid_assets] = 0.02 / len(invalid_assets)

        # Renormalise to ensure sum=1.0
        weight_sum = full_weights.sum()
        if weight_sum > 1e-6:
            full_weights = full_weights / weight_sum
        else:
            # Fallback: equal weights
            full_weights = pd.Series(1.0 / len(full_universe), index=full_universe)

        return full_weights

    def get_last_data_quality_metrics(self) -> dict[str, Any]:
        """
        Get data quality metrics from last prediction.

        Returns:
            Dictionary with keys:
            - requested_assets: int
            - valid_assets: int
            - coverage_ratio: float
            - na_ratio: float (if applicable)
        """
        if not hasattr(self, '_last_data_quality_metrics'):
            return {
                'requested_assets': 0,
                'valid_assets': 0,
                'coverage_ratio': 0.0,
            }

        return self._last_data_quality_metrics
