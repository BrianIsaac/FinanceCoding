"""
Abstract base model interface for portfolio optimization models.

This module defines the unified interface that all portfolio models must implement,
ensuring consistent APIs across different ML approaches (HRP, LSTM, GAT).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

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

    def validate_weights(
        self,
        weights: pd.Series,
        previous_weights: Optional[pd.Series] = None,
        model_scores: Optional[pd.Series] = None,
        date: Optional[pd.Timestamp] = None,
    ) -> pd.Series:
        """
        Validate and enforce portfolio constraints using unified constraint engine.

        Args:
            weights: Raw portfolio weights
            previous_weights: Previous period weights for turnover calculation
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
        self, weights: pd.Series, previous_weights: Optional[pd.Series] = None
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
        self, weights: pd.Series, previous_weights: Optional[pd.Series] = None
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
