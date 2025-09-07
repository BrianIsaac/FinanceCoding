"""
Transaction cost modeling for portfolio backtesting.

This module provides transaction cost calculations based on portfolio
turnover, enabling realistic performance evaluation that accounts
for trading costs in portfolio rebalancing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TransactionCostConfig:
    """Configuration for transaction cost calculations."""

    linear_cost_bps: float = 10.0  # Linear cost in basis points (0.1%)
    fixed_cost_per_trade: float = 0.0  # Fixed cost per trade
    market_impact_coefficient: float = 0.0  # Market impact (future feature)
    bid_ask_spread_bps: float = 5.0  # Bid-ask spread cost
    enable_cost_decay: bool = False  # Decay costs over time (future feature)


class TransactionCostCalculator:
    """
    Calculator for portfolio transaction costs.

    Provides linear transaction cost modeling with configurable
    cost parameters. Integrates with portfolio rebalancing to
    calculate realistic net returns after trading costs.
    """

    def __init__(self, config: TransactionCostConfig):
        """
        Initialize transaction cost calculator.

        Args:
            config: Configuration for cost calculations
        """
        self.config = config

    def calculate_turnover(
        self, current_weights: pd.Series, target_weights: pd.Series, method: str = "one_way"
    ) -> float:
        """
        Calculate portfolio turnover between current and target weights.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights after rebalancing
            method: Turnover calculation method ("one_way" or "two_way")

        Returns:
            Portfolio turnover as fraction (0.0 to 2.0 for one-way)

        Raises:
            ValueError: If method is not recognized
        """
        if method not in ["one_way", "two_way"]:
            raise ValueError(f"Unknown turnover method: {method}")

        # Align weights to common index
        all_assets = current_weights.index.union(target_weights.index)
        current_aligned = current_weights.reindex(all_assets, fill_value=0.0)
        target_aligned = target_weights.reindex(all_assets, fill_value=0.0)

        # Calculate weight changes
        weight_changes = target_aligned - current_aligned

        if method == "one_way":
            # One-way turnover: sum of absolute weight changes
            return np.abs(weight_changes).sum()
        else:
            # Two-way turnover: sum of buys and sells separately
            buys = weight_changes[weight_changes > 0].sum()
            sells = np.abs(weight_changes[weight_changes < 0]).sum()
            return buys + sells

    def calculate_transaction_costs(
        self, current_weights: pd.Series, target_weights: pd.Series, portfolio_value: float = 1.0
    ) -> dict[str, float]:
        """
        Calculate transaction costs for rebalancing.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights after rebalancing
            portfolio_value: Total portfolio value (default: 1.0)

        Returns:
            Dictionary containing cost breakdown:
                - total_cost: Total transaction cost as fraction of portfolio
                - linear_cost: Linear cost component
                - fixed_cost: Fixed cost component
                - turnover: Portfolio turnover
                - cost_per_turnover: Cost per unit of turnover
        """
        # Calculate turnover
        turnover = self.calculate_turnover(current_weights, target_weights)

        # Linear transaction costs (proportional to turnover)
        linear_cost = turnover * (self.config.linear_cost_bps / 10000.0)

        # Fixed costs (per number of trades)
        num_trades = self._count_trades(current_weights, target_weights)
        fixed_cost = num_trades * self.config.fixed_cost_per_trade / portfolio_value

        # Total cost
        total_cost = linear_cost + fixed_cost

        return {
            "total_cost": total_cost,
            "linear_cost": linear_cost,
            "fixed_cost": fixed_cost,
            "turnover": turnover,
            "num_trades": num_trades,
            "cost_per_turnover": total_cost / turnover if turnover > 0 else 0.0,
        }

    def apply_transaction_costs(
        self,
        returns: pd.Series,
        current_weights: pd.Series,
        target_weights: pd.Series,
        portfolio_value: float = 1.0,
    ) -> tuple[float, dict[str, float]]:
        """
        Apply transaction costs to portfolio returns.

        Args:
            returns: Period returns (before transaction costs)
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights after rebalancing
            portfolio_value: Total portfolio value

        Returns:
            Tuple of (net_return_after_costs, cost_breakdown)
        """
        # Calculate transaction costs
        cost_breakdown = self.calculate_transaction_costs(
            current_weights, target_weights, portfolio_value
        )

        # Apply costs to returns
        gross_return = returns.sum() if isinstance(returns, pd.Series) else returns
        net_return = gross_return - cost_breakdown["total_cost"]

        cost_breakdown["gross_return"] = gross_return
        cost_breakdown["net_return"] = net_return

        return net_return, cost_breakdown

    def estimate_annual_costs(
        self, monthly_turnover: float, rebalancing_frequency: int = 12
    ) -> dict[str, float]:
        """
        Estimate annual transaction costs based on turnover patterns.

        Args:
            monthly_turnover: Average monthly portfolio turnover
            rebalancing_frequency: Number of rebalances per year (default: 12)

        Returns:
            Dictionary with annual cost estimates
        """
        # Annual turnover
        annual_turnover = monthly_turnover * rebalancing_frequency

        # Annual costs
        annual_linear_cost = annual_turnover * (self.config.linear_cost_bps / 10000.0)

        # Estimate trades per rebalance (rough approximation)
        avg_trades_per_rebalance = monthly_turnover * 50  # Assume 50 assets average
        annual_fixed_cost = (
            avg_trades_per_rebalance * rebalancing_frequency * self.config.fixed_cost_per_trade
        )

        return {
            "annual_turnover": annual_turnover,
            "annual_linear_cost": annual_linear_cost,
            "annual_fixed_cost": annual_fixed_cost,
            "annual_total_cost": annual_linear_cost + annual_fixed_cost,
            "cost_drag_bps": (annual_linear_cost + annual_fixed_cost) * 10000,
        }

    def _count_trades(self, current_weights: pd.Series, target_weights: pd.Series) -> int:
        """
        Count number of individual trades required for rebalancing.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights

        Returns:
            Number of individual trades (buy or sell orders)
        """
        # Align weights to common index
        all_assets = current_weights.index.union(target_weights.index)
        current_aligned = current_weights.reindex(all_assets, fill_value=0.0)
        target_aligned = target_weights.reindex(all_assets, fill_value=0.0)

        # Count assets with weight changes (accounting for minimum trade size)
        min_trade_threshold = 0.0001  # 0.01% minimum trade size
        weight_changes = np.abs(target_aligned - current_aligned)

        return (weight_changes > min_trade_threshold).sum()

    def create_cost_report(
        self, rebalancing_history: pd.DataFrame, returns_history: pd.Series
    ) -> dict[str, Any]:
        """
        Create comprehensive transaction cost report.

        Args:
            rebalancing_history: DataFrame with columns for dates, weights, costs
            returns_history: Time series of portfolio returns

        Returns:
            Dictionary containing comprehensive cost analysis

        Note:
            This is a stub implementation. Full cost reporting
            will be expanded in future iterations.
        """
        # Stub implementation - returns placeholder report
        return {
            "total_periods": len(rebalancing_history),
            "average_turnover": 0.0,
            "average_cost_bps": 0.0,
            "total_cost_drag": 0.0,
            "cost_efficiency": 0.0,
        }
