"""
Turnover tracking and enforcement for portfolio backtesting.

This module provides comprehensive turnover tracking capabilities with
rolling windows, budgeting systems, and enforcement mechanisms that
integrate with the unified constraint system.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class TurnoverBudget:
    """Turnover budget allocation for assets."""

    total_budget: float
    allocated: Dict[str, float]
    remaining: float

    def allocate(self, asset: str, amount: float) -> bool:
        """
        Allocate turnover budget to an asset.

        Args:
            asset: Asset identifier
            amount: Turnover amount to allocate

        Returns:
            True if allocation successful, False if insufficient budget
        """
        if amount <= self.remaining:
            self.allocated[asset] = self.allocated.get(asset, 0) + amount
            self.remaining -= amount
            return True
        return False

    def reset(self, new_budget: float) -> None:
        """Reset budget for new period."""
        self.total_budget = new_budget
        self.allocated.clear()
        self.remaining = new_budget


class TurnoverTracker:
    """
    Comprehensive turnover tracking and enforcement system.

    Tracks turnover across multiple time horizons with budgeting,
    enforcement, and detailed analytics for portfolio optimization.
    """

    def __init__(
        self,
        max_monthly_turnover: float = 0.20,
        lookback_months: int = 1,
        enable_budgeting: bool = True,
    ):
        """
        Initialize turnover tracker.

        Args:
            max_monthly_turnover: Maximum allowed monthly turnover
            lookback_months: Number of months for rolling turnover calculation
            enable_budgeting: Whether to enable turnover budgeting system
        """
        self.max_monthly_turnover = max_monthly_turnover
        self.lookback_months = lookback_months
        self.enable_budgeting = enable_budgeting

        # Turnover history storage
        self.turnover_history: List[Dict[str, Any]] = []
        self.rolling_turnovers: deque = deque(maxlen=lookback_months)

        # Asset-level tracking
        self.asset_turnovers: Dict[str, List[float]] = defaultdict(list)

        # Budgeting system
        self.current_budget: Optional[TurnoverBudget] = None

    def calculate_turnover(
        self, current_weights: pd.Series, previous_weights: pd.Series, method: str = "one_way"
    ) -> Dict[str, float]:
        """
        Calculate comprehensive turnover metrics.

        Args:
            current_weights: Current portfolio weights
            previous_weights: Previous portfolio weights
            method: Turnover calculation method ("one_way", "two_way")

        Returns:
            Dictionary with turnover metrics
        """
        # Align weights to common universe
        all_assets = current_weights.index.union(previous_weights.index)
        current_aligned = current_weights.reindex(all_assets, fill_value=0.0)
        previous_aligned = previous_weights.reindex(all_assets, fill_value=0.0)

        # Calculate weight changes
        weight_changes = current_aligned - previous_aligned

        # Portfolio-level turnover
        if method == "one_way":
            portfolio_turnover = np.abs(weight_changes).sum()
        else:  # two_way
            buys = weight_changes[weight_changes > 0].sum()
            sells = np.abs(weight_changes[weight_changes < 0]).sum()
            portfolio_turnover = buys + sells

        # Asset-level turnovers
        asset_turnovers = np.abs(weight_changes).to_dict()

        # Additional metrics
        num_trades = (np.abs(weight_changes) > 0.0001).sum()  # Trades above threshold
        avg_trade_size = portfolio_turnover / num_trades if num_trades > 0 else 0.0

        return {
            "portfolio_turnover": portfolio_turnover,
            "asset_turnovers": asset_turnovers,
            "num_trades": num_trades,
            "avg_trade_size": avg_trade_size,
            "buys": weight_changes[weight_changes > 0].sum(),
            "sells": np.abs(weight_changes[weight_changes < 0]).sum(),
        }

    def track_rebalancing(
        self,
        date: pd.Timestamp,
        current_weights: pd.Series,
        previous_weights: pd.Series,
        portfolio_value: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Track a rebalancing event with comprehensive metrics.

        Args:
            date: Rebalancing date
            current_weights: Current portfolio weights
            previous_weights: Previous portfolio weights
            portfolio_value: Total portfolio value

        Returns:
            Dictionary with tracking results and constraint status
        """
        # Calculate turnover metrics
        turnover_metrics = self.calculate_turnover(current_weights, previous_weights)

        # Create tracking record
        tracking_record = {
            "date": date,
            "portfolio_turnover": turnover_metrics["portfolio_turnover"],
            "num_trades": turnover_metrics["num_trades"],
            "avg_trade_size": turnover_metrics["avg_trade_size"],
            "buys": turnover_metrics["buys"],
            "sells": turnover_metrics["sells"],
            "portfolio_value": portfolio_value,
            "constraint_compliant": turnover_metrics["portfolio_turnover"]
            <= self.max_monthly_turnover,
        }

        # Update history
        self.turnover_history.append(tracking_record)
        self.rolling_turnovers.append(turnover_metrics["portfolio_turnover"])

        # Update asset-level tracking
        for asset, turnover in turnover_metrics["asset_turnovers"].items():
            self.asset_turnovers[asset].append(turnover)

        # Budget tracking if enabled
        budget_status = {}
        if self.enable_budgeting and self.current_budget is not None:
            budget_status = self._update_budget_tracking(turnover_metrics)

        return {
            "tracking_record": tracking_record,
            "rolling_average": self.get_rolling_average_turnover(),
            "budget_status": budget_status,
            "enforcement_needed": turnover_metrics["portfolio_turnover"]
            > self.max_monthly_turnover,
        }

    def enforce_turnover_constraint(
        self,
        current_weights: pd.Series,
        previous_weights: pd.Series,
        method: str = "proportional_blend",
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Enforce turnover constraint by adjusting weights.

        Args:
            current_weights: Target portfolio weights
            previous_weights: Previous portfolio weights
            method: Enforcement method ("proportional_blend", "priority_based", "budget_aware")

        Returns:
            Tuple of (adjusted_weights, enforcement_details)
        """
        # Calculate initial turnover
        initial_turnover = self.calculate_turnover(current_weights, previous_weights)[
            "portfolio_turnover"
        ]

        if initial_turnover <= self.max_monthly_turnover:
            # No enforcement needed
            return current_weights, {
                "enforcement_applied": False,
                "initial_turnover": initial_turnover,
                "final_turnover": initial_turnover,
            }

        # Apply enforcement based on method
        if method == "proportional_blend":
            adjusted_weights = self._proportional_blend_enforcement(
                current_weights, previous_weights
            )
        elif method == "priority_based":
            adjusted_weights = self._priority_based_enforcement(current_weights, previous_weights)
        elif method == "budget_aware":
            adjusted_weights = self._budget_aware_enforcement(current_weights, previous_weights)
        else:
            raise ValueError(f"Unknown enforcement method: {method}")

        # Calculate final turnover
        final_turnover = self.calculate_turnover(adjusted_weights, previous_weights)[
            "portfolio_turnover"
        ]

        return adjusted_weights, {
            "enforcement_applied": True,
            "method": method,
            "initial_turnover": initial_turnover,
            "final_turnover": final_turnover,
            "turnover_reduction": initial_turnover - final_turnover,
            "constraint_satisfied": final_turnover <= self.max_monthly_turnover,
        }

    def _proportional_blend_enforcement(
        self, current_weights: pd.Series, previous_weights: pd.Series
    ) -> pd.Series:
        """Enforce turnover by proportional blending with previous weights."""
        # Calculate required blend factor
        initial_turnover = self.calculate_turnover(current_weights, previous_weights)[
            "portfolio_turnover"
        ]
        blend_factor = self.max_monthly_turnover / initial_turnover if initial_turnover > 0 else 1.0

        # Align weights
        all_assets = current_weights.index.union(previous_weights.index)
        current_aligned = current_weights.reindex(all_assets, fill_value=0.0)
        previous_aligned = previous_weights.reindex(all_assets, fill_value=0.0)

        # Apply blending
        adjusted_weights = blend_factor * current_aligned + (1 - blend_factor) * previous_aligned

        # Normalize
        weight_sum = adjusted_weights.sum()
        if weight_sum > 0:
            adjusted_weights = adjusted_weights / weight_sum

        return adjusted_weights

    def _priority_based_enforcement(
        self, current_weights: pd.Series, previous_weights: pd.Series
    ) -> pd.Series:
        """Enforce turnover by prioritizing most important changes."""
        # Calculate weight changes
        all_assets = current_weights.index.union(previous_weights.index)
        current_aligned = current_weights.reindex(all_assets, fill_value=0.0)
        previous_aligned = previous_weights.reindex(all_assets, fill_value=0.0)
        weight_changes = current_aligned - previous_aligned

        # Sort changes by absolute magnitude (priority)
        change_priorities = np.abs(weight_changes).sort_values(ascending=False)

        # Allocate turnover budget to highest priority changes
        remaining_budget = self.max_monthly_turnover
        adjusted_changes = pd.Series(0.0, index=weight_changes.index)

        for asset in change_priorities.index:
            desired_change = weight_changes[asset]
            required_turnover = abs(desired_change)

            if required_turnover <= remaining_budget:
                # Full change can be accommodated
                adjusted_changes[asset] = desired_change
                remaining_budget -= required_turnover
            else:
                # Partial change within remaining budget
                if remaining_budget > 0:
                    proportion = remaining_budget / required_turnover
                    adjusted_changes[asset] = desired_change * proportion
                break

        # Create adjusted weights
        adjusted_weights = previous_aligned + adjusted_changes

        # Normalize
        weight_sum = adjusted_weights.sum()
        if weight_sum > 0:
            adjusted_weights = adjusted_weights / weight_sum

        return adjusted_weights

    def _budget_aware_enforcement(
        self, current_weights: pd.Series, previous_weights: pd.Series
    ) -> pd.Series:
        """Enforce turnover using budget allocation system."""
        if self.current_budget is None:
            self._initialize_budget()

        # Use priority-based enforcement with budget constraints
        return self._priority_based_enforcement(current_weights, previous_weights)

    def _initialize_budget(self) -> None:
        """Initialize turnover budget for current period."""
        self.current_budget = TurnoverBudget(
            total_budget=self.max_monthly_turnover,
            allocated={},
            remaining=self.max_monthly_turnover,
        )

    def _update_budget_tracking(self, turnover_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Update budget tracking with current turnover usage."""
        if self.current_budget is None:
            return {}

        total_used = turnover_metrics["portfolio_turnover"]
        budget_utilization = (
            total_used / self.current_budget.total_budget
            if self.current_budget.total_budget > 0
            else 0
        )

        return {
            "budget_utilization": budget_utilization,
            "remaining_budget": max(0, self.current_budget.total_budget - total_used),
            "over_budget": total_used > self.current_budget.total_budget,
        }

    def get_rolling_average_turnover(self, periods: Optional[int] = None) -> float:
        """Get rolling average turnover over specified periods."""
        if not self.rolling_turnovers:
            return 0.0

        periods = periods or self.lookback_months
        recent_turnovers = list(self.rolling_turnovers)[-periods:]
        return np.mean(recent_turnovers) if recent_turnovers else 0.0

    def get_turnover_statistics(self) -> Dict[str, Any]:
        """Get comprehensive turnover statistics."""
        if not self.turnover_history:
            return {"error": "No turnover history available"}

        turnovers = [record["portfolio_turnover"] for record in self.turnover_history]

        return {
            "total_rebalances": len(self.turnover_history),
            "avg_turnover": np.mean(turnovers),
            "median_turnover": np.median(turnovers),
            "std_turnover": np.std(turnovers),
            "min_turnover": np.min(turnovers),
            "max_turnover": np.max(turnovers),
            "constraint_violations": sum(
                1 for record in self.turnover_history if not record["constraint_compliant"]
            ),
            "violation_rate": np.mean(
                [not record["constraint_compliant"] for record in self.turnover_history]
            ),
            "rolling_average": self.get_rolling_average_turnover(),
        }

    def reset_tracking(self) -> None:
        """Reset all tracking data for new backtest."""
        self.turnover_history.clear()
        self.rolling_turnovers.clear()
        self.asset_turnovers.clear()
        self.current_budget = None

    def create_turnover_report(self) -> Dict[str, Any]:
        """Create comprehensive turnover tracking report."""
        statistics = self.get_turnover_statistics()

        # Asset-level analysis
        asset_stats = {}
        for asset, turnovers in self.asset_turnovers.items():
            if turnovers:
                asset_stats[asset] = {
                    "avg_turnover": np.mean(turnovers),
                    "total_turnover": np.sum(turnovers),
                    "num_trades": len([t for t in turnovers if t > 0.0001]),
                }

        return {
            "configuration": {
                "max_monthly_turnover": self.max_monthly_turnover,
                "lookback_months": self.lookback_months,
                "budgeting_enabled": self.enable_budgeting,
            },
            "portfolio_statistics": statistics,
            "asset_analysis": asset_stats,
            "recent_history": self.turnover_history[-10:] if self.turnover_history else [],
            "budget_status": (
                {
                    "current_budget": self.current_budget.total_budget,
                    "remaining": self.current_budget.remaining,
                    "allocated": dict(self.current_budget.allocated),
                }
                if self.current_budget
                else None
            ),
        }
