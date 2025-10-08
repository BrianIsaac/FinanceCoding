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
