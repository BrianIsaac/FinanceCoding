"""
Main backtesting engine for portfolio optimization models.

This module provides the core backtesting functionality for evaluating
portfolio models with realistic constraints, transaction costs, and
rolling validation procedures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from src.evaluation.backtest.rebalancing import PortfolioRebalancer, RebalancingConfig
from src.evaluation.backtest.transaction_costs import (
    TransactionCostCalculator,
    TransactionCostConfig,
)
from src.evaluation.metrics.returns import PerformanceAnalytics
from src.models.base import PortfolioModel

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting engine."""

    start_date: datetime
    end_date: datetime
    rebalance_frequency: str = "M"  # Monthly
    initial_capital: float = 1000000.0
    transaction_cost_bps: float = 10.0
    benchmark_ticker: str = "SPY"
    min_history_days: int = 252  # Minimum history required for training


class BacktestEngine:
    """
    Core backtesting engine for portfolio models.

    Provides rolling validation, realistic transaction costs,
    and comprehensive performance analytics for comparing
    different portfolio optimization approaches.
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize backtest engine.

        Args:
            config: Backtesting configuration parameters
        """
        self.config = config

        # Initialize transaction cost calculator
        cost_config = TransactionCostConfig(linear_cost_bps=config.transaction_cost_bps)
        self.cost_calculator = TransactionCostCalculator(cost_config)

        # Initialize rebalancer
        rebalance_config = RebalancingConfig(
            frequency=config.rebalance_frequency, transaction_cost_config=cost_config
        )
        self.rebalancer = PortfolioRebalancer(rebalance_config)

        # Initialize performance analytics
        self.performance_analytics = PerformanceAnalytics()

    def run_backtest(
        self,
        model: PortfolioModel,
        returns_data: pd.DataFrame,
        universe_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """
        Execute complete backtest for a portfolio model.

        Args:
            model: Portfolio model to backtest
            returns_data: Historical returns data with DatetimeIndex and asset columns
            universe_data: Optional dynamic universe membership data

        Returns:
            Dictionary containing backtest results including returns,
            weights, turnover, and performance metrics
        """
        logger.info(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")

        # Generate rebalancing dates
        rebalance_dates = self.get_rebalance_dates(returns_data)

        if len(rebalance_dates) == 0:
            logger.warning("No rebalancing dates generated")
            return self._empty_results()

        # Initialize tracking variables
        portfolio_weights = []
        portfolio_returns = []
        transaction_costs = []
        previous_weights = None

        for i, rebalance_date in enumerate(rebalance_dates):
            try:
                # Get training window
                train_start = max(
                    self.config.start_date,
                    rebalance_date - pd.Timedelta(days=self.config.min_history_days + 30),
                )

                # Get current universe
                if universe_data is not None:
                    universe = (
                        universe_data.loc[rebalance_date]
                        if rebalance_date in universe_data.index
                        else returns_data.columns.tolist()
                    )
                else:
                    universe = returns_data.columns.tolist()

                # Filter returns data for available assets
                train_returns = returns_data.loc[train_start:rebalance_date, universe].dropna(
                    axis=1, how="all"
                )

                if train_returns.empty or len(train_returns) < 60:  # Minimum 60 days for LSTM
                    logger.warning(f"Insufficient data for {rebalance_date}, skipping")
                    continue

                # Train model
                model.fit(
                    returns=train_returns,
                    universe=train_returns.columns.tolist(),
                    fit_period=(train_start, rebalance_date),
                )

                # Generate portfolio weights
                current_weights = model.predict_weights(
                    date=rebalance_date, universe=train_returns.columns.tolist()
                )

                # Calculate transaction costs if not first rebalancing
                if previous_weights is not None:
                    costs = self.cost_calculator.calculate_transaction_costs(
                        current_weights,
                        previous_weights,
                        returns_data.loc[rebalance_date, current_weights.index],
                    )
                else:
                    costs = 0.0

                # Store results
                portfolio_weights.append(current_weights)
                transaction_costs.append(costs)
                previous_weights = current_weights

                # Calculate returns for next period
                if i < len(rebalance_dates) - 1:
                    next_date = rebalance_dates[i + 1]
                    period_returns = returns_data.loc[
                        rebalance_date:next_date, current_weights.index
                    ]
                    if not period_returns.empty:
                        portfolio_return = (period_returns * current_weights).sum(axis=1).iloc[1:]
                        portfolio_returns.extend(portfolio_return.tolist())

                logger.info(f"Completed rebalancing for {rebalance_date}")

            except Exception as e:
                logger.error(f"Error during rebalancing on {rebalance_date}: {e}")
                continue

        if not portfolio_weights:
            logger.warning("No successful rebalancing periods")
            return self._empty_results()

        # Compile results
        weights_df = pd.DataFrame(
            portfolio_weights, index=rebalance_dates[: len(portfolio_weights)]
        )
        returns_series = pd.Series(portfolio_returns, name="portfolio_returns")
        costs_series = pd.Series(
            transaction_costs,
            index=rebalance_dates[: len(transaction_costs)],
            name="transaction_costs",
        )

        # Calculate performance metrics
        performance_metrics = self.performance_analytics.calculate_portfolio_metrics(returns_series)

        return {
            "portfolio_returns": returns_series,
            "portfolio_weights": weights_df,
            "turnover": self._calculate_turnover(weights_df),
            "transaction_costs": costs_series,
            "benchmark_returns": pd.Series(dtype=float),  # TODO: Implement benchmark
            "performance_metrics": performance_metrics,
            "rebalance_dates": rebalance_dates[: len(portfolio_weights)],
        }

    def _empty_results(self) -> dict[str, Any]:
        """Return empty results structure."""
        return {
            "portfolio_returns": pd.Series(dtype=float),
            "portfolio_weights": pd.DataFrame(),
            "turnover": pd.Series(dtype=float),
            "transaction_costs": pd.Series(dtype=float),
            "benchmark_returns": pd.Series(dtype=float),
            "performance_metrics": {},
            "rebalance_dates": [],
        }

    def _calculate_turnover(self, weights_df: pd.DataFrame) -> pd.Series:
        """Calculate portfolio turnover."""
        if len(weights_df) < 2:
            return pd.Series(dtype=float)

        turnover = []
        for i in range(1, len(weights_df)):
            prev_weights = weights_df.iloc[i - 1].fillna(0)
            curr_weights = weights_df.iloc[i].fillna(0)
            # Align indices
            common_assets = prev_weights.index.intersection(curr_weights.index)
            prev_aligned = prev_weights.reindex(common_assets, fill_value=0)
            curr_aligned = curr_weights.reindex(common_assets, fill_value=0)
            turnover.append(abs(curr_aligned - prev_aligned).sum() / 2)

        return pd.Series(turnover, index=weights_df.index[1:], name="turnover")

    def calculate_transaction_costs(
        self, current_weights: pd.Series, previous_weights: pd.Series, prices: pd.Series
    ) -> float:
        """
        Calculate transaction costs for portfolio rebalancing.

        Args:
            current_weights: Target portfolio weights
            previous_weights: Previous portfolio weights
            prices: Asset prices for the rebalancing date

        Returns:
            Total transaction costs as percentage of portfolio value

        Note:
            This is a stub implementation. Transaction cost modeling
            will be implemented in future stories.
        """
        # Stub implementation
        return 0.0

    def get_rebalance_dates(self, returns_data: pd.DataFrame) -> list[pd.Timestamp]:
        """
        Generate rebalancing dates based on frequency configuration.

        Args:
            returns_data: Historical returns data for date range

        Returns:
            List of rebalancing dates
        """
        # Get date range from returns data
        start_date = max(pd.to_datetime(self.config.start_date), returns_data.index.min())
        end_date = min(pd.to_datetime(self.config.end_date), returns_data.index.max())

        # Generate rebalancing dates based on frequency
        if (
            self.config.rebalance_frequency == "M"
            or self.config.rebalance_frequency.lower() == "monthly"
        ):
            # Monthly rebalancing - first business day of each month
            dates = pd.date_range(start=start_date, end=end_date, freq="MS")  # Month Start
            # Ensure dates are business days and exist in returns data
            rebalance_dates = []
            for date in dates:
                # Find the first business day on or after this date that exists in returns data
                candidate_dates = returns_data.index[returns_data.index >= date]
                if len(candidate_dates) > 0:
                    rebalance_dates.append(candidate_dates[0])
        elif (
            self.config.rebalance_frequency == "Q"
            or self.config.rebalance_frequency.lower() == "quarterly"
        ):
            # Quarterly rebalancing
            dates = pd.date_range(start=start_date, end=end_date, freq="QS")  # Quarter Start
            rebalance_dates = []
            for date in dates:
                candidate_dates = returns_data.index[returns_data.index >= date]
                if len(candidate_dates) > 0:
                    rebalance_dates.append(candidate_dates[0])
        else:
            # Default to monthly if frequency not recognized
            logger.warning(
                f"Unknown frequency {self.config.rebalance_frequency}, defaulting to monthly"
            )
            dates = pd.date_range(start=start_date, end=end_date, freq="MS")
            rebalance_dates = []
            for date in dates:
                candidate_dates = returns_data.index[returns_data.index >= date]
                if len(candidate_dates) > 0:
                    rebalance_dates.append(candidate_dates[0])

        # Filter dates to ensure we have enough history for training
        min_date = start_date + pd.Timedelta(days=self.config.min_history_days)
        rebalance_dates = [d for d in rebalance_dates if d >= min_date]

        logger.info(
            f"Generated {len(rebalance_dates)} rebalancing dates from {start_date} to {end_date}"
        )
        return rebalance_dates
