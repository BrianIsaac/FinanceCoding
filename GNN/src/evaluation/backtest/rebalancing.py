"""
Portfolio rebalancing logic for backtesting framework.

This module provides monthly rebalancing functionality that integrates
with the universe calendar and portfolio models to execute systematic
rebalancing with transaction cost consideration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.evaluation.backtest.transaction_costs import (
    TransactionCostCalculator,
    TransactionCostConfig,
)
from src.models.base.portfolio_model import PortfolioModel


@dataclass
class RebalancingConfig:
    """Configuration for portfolio rebalancing."""

    frequency: str = "monthly"  # Rebalancing frequency
    transaction_cost_config: TransactionCostConfig | None = None
    min_rebalance_threshold: float = 0.01  # Minimum turnover to trigger rebalance
    enable_transaction_costs: bool = True
    rebalance_tolerance: float = 0.001  # Weight tolerance for rebalancing


class PortfolioRebalancer:
    """
    Portfolio rebalancing engine for backtesting.

    Handles systematic rebalancing with transaction cost integration,
    universe changes, and constraint enforcement. Works with any
    portfolio model implementing the PortfolioModel interface.
    """

    def __init__(self, config: RebalancingConfig):
        """
        Initialize portfolio rebalancer.

        Args:
            config: Configuration for rebalancing behavior
        """
        self.config = config

        # Initialize transaction cost calculator
        if self.config.enable_transaction_costs:
            cost_config = config.transaction_cost_config or TransactionCostConfig()
            self.cost_calculator = TransactionCostCalculator(cost_config)
        else:
            self.cost_calculator = None

        # Track rebalancing history
        self.rebalancing_history = []

    def generate_rebalancing_schedule(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        universe_calendar: Any  # UniverseCalendar from Story 1.2
    ) -> list[pd.Timestamp]:
        """
        Generate rebalancing dates based on configuration and universe calendar.

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            universe_calendar: Universe calendar for active trading dates

        Returns:
            List of rebalancing dates
        """
        rebalance_dates = []

        if self.config.frequency == "monthly":
            # Generate month-end dates
            date_range = pd.date_range(start=start_date, end=end_date, freq='ME')

            # Filter to dates with active universe
            for date in date_range:
                if hasattr(universe_calendar, 'is_active_date') and universe_calendar.is_active_date(date):
                    rebalance_dates.append(date)
                else:
                    # Fallback: use the date if universe_calendar doesn't have is_active_date
                    rebalance_dates.append(date)

        elif self.config.frequency == "quarterly":
            # Generate quarter-end dates
            date_range = pd.date_range(start=start_date, end=end_date, freq='Q')
            rebalance_dates.extend(date_range.tolist())

        else:
            raise ValueError(f"Unsupported rebalancing frequency: {self.config.frequency}")

        return rebalance_dates

    def execute_rebalancing(
        self,
        rebalance_date: pd.Timestamp,
        model: PortfolioModel,
        current_weights: pd.Series,
        universe: list[str],
        portfolio_value: float = 1.0
    ) -> tuple[pd.Series, dict[str, Any]]:
        """
        Execute portfolio rebalancing for a specific date.

        Args:
            rebalance_date: Date for rebalancing
            model: Portfolio model for weight generation
            current_weights: Current portfolio weights
            universe: Available universe for rebalancing
            portfolio_value: Current portfolio value

        Returns:
            Tuple of (new_weights, rebalancing_info)
        """
        # Generate target weights using the model
        target_weights = model.predict_weights(rebalance_date, universe)

        # Check if rebalancing is needed
        if self._should_rebalance(current_weights, target_weights):
            new_weights = target_weights.copy()

            # Calculate transaction costs if enabled
            if self.cost_calculator:
                net_return, cost_info = self.cost_calculator.apply_transaction_costs(
                    returns=0.0,  # Will be calculated elsewhere
                    current_weights=current_weights,
                    target_weights=target_weights,
                    portfolio_value=portfolio_value
                )
                transaction_costs = cost_info
            else:
                transaction_costs = {"total_cost": 0.0, "turnover": 0.0}

            rebalancing_info = {
                "date": rebalance_date,
                "rebalanced": True,
                "universe_size": len(universe),
                "active_positions": (new_weights > 0).sum(),
                "transaction_costs": transaction_costs,
                "model_info": model.get_model_info(),
            }

        else:
            # No rebalancing needed
            new_weights = current_weights.copy()
            rebalancing_info = {
                "date": rebalance_date,
                "rebalanced": False,
                "reason": "Below minimum threshold",
                "universe_size": len(universe),
                "active_positions": (current_weights > 0).sum(),
                "transaction_costs": {"total_cost": 0.0, "turnover": 0.0},
            }

        # Record rebalancing history
        self.rebalancing_history.append(rebalancing_info)

        return new_weights, rebalancing_info

    def backtest_with_rebalancing(
        self,
        model: PortfolioModel,
        returns_data: pd.DataFrame,
        universe_calendar: Any,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        initial_portfolio_value: float = 1.0
    ) -> dict[str, Any]:
        """
        Execute full backtest with systematic rebalancing.

        Args:
            model: Portfolio model to use for rebalancing
            returns_data: Historical returns data
            universe_calendar: Universe calendar for active dates and assets
            start_date: Backtest start date
            end_date: Backtest end date
            initial_portfolio_value: Starting portfolio value

        Returns:
            Dictionary containing backtest results, performance metrics,
            and rebalancing history

        Note:
            This is a stub implementation. Full backtesting integration
            will be implemented when universe calendar interface is finalized.
        """
        # Generate rebalancing schedule
        rebalance_dates = self.generate_rebalancing_schedule(
            start_date, end_date, universe_calendar
        )

        # Initialize portfolio state
        portfolio_value = initial_portfolio_value
        current_weights = pd.Series(dtype=float)

        # Track results
        weight_history = []
        performance_history = []

        for rebalance_date in rebalance_dates:
            # Get universe for this date
            if hasattr(universe_calendar, 'get_active_tickers'):
                universe = universe_calendar.get_active_tickers(rebalance_date)
            else:
                # Fallback: use all available assets
                universe = returns_data.columns.tolist()

            # Execute rebalancing
            new_weights, rebalancing_info = self.execute_rebalancing(
                rebalance_date, model, current_weights, universe, portfolio_value
            )

            # Update portfolio state
            current_weights = new_weights
            weight_history.append({
                "date": rebalance_date,
                "weights": new_weights.to_dict(),
            })

            # Calculate period performance (stub)
            period_return = 0.0  # Will be calculated with actual returns
            portfolio_value *= (1 + period_return)

            performance_history.append({
                "date": rebalance_date,
                "portfolio_value": portfolio_value,
                "period_return": period_return,
            })

        return {
            "weight_history": weight_history,
            "performance_history": performance_history,
            "rebalancing_history": self.rebalancing_history,
            "final_portfolio_value": portfolio_value,
            "total_rebalances": len([r for r in self.rebalancing_history if r.get("rebalanced", False)]),
        }

    def _should_rebalance(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series
    ) -> bool:
        """
        Determine if rebalancing is necessary based on weight differences.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights

        Returns:
            True if rebalancing should be executed, False otherwise
        """
        if current_weights.empty:
            return True  # First rebalancing

        # Align weights for comparison
        all_assets = current_weights.index.union(target_weights.index)
        current_aligned = current_weights.reindex(all_assets, fill_value=0.0)
        target_aligned = target_weights.reindex(all_assets, fill_value=0.0)

        # Calculate turnover
        weight_differences = target_aligned - current_aligned
        turnover = abs(weight_differences).sum()

        return turnover > self.config.min_rebalance_threshold

    def get_rebalancing_statistics(self) -> dict[str, Any]:
        """
        Calculate rebalancing statistics from history.

        Returns:
            Dictionary with rebalancing statistics and insights
        """
        if not self.rebalancing_history:
            return {"error": "No rebalancing history available"}

        executed_rebalances = [r for r in self.rebalancing_history if r.get("rebalanced", False)]

        if not executed_rebalances:
            return {"total_periods": len(self.rebalancing_history), "executed_rebalances": 0}

        # Calculate statistics
        total_transaction_costs = sum(
            r["transaction_costs"]["total_cost"] for r in executed_rebalances
        )

        average_turnover = sum(
            r["transaction_costs"]["turnover"] for r in executed_rebalances
        ) / len(executed_rebalances)

        average_positions = sum(
            r["active_positions"] for r in executed_rebalances
        ) / len(executed_rebalances)

        return {
            "total_periods": len(self.rebalancing_history),
            "executed_rebalances": len(executed_rebalances),
            "rebalancing_rate": len(executed_rebalances) / len(self.rebalancing_history),
            "total_transaction_costs": total_transaction_costs,
            "average_turnover": average_turnover,
            "average_active_positions": average_positions,
            "cost_per_rebalance": total_transaction_costs / len(executed_rebalances),
        }
