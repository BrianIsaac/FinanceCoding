"""
Operational performance metrics for portfolio evaluation.

This module provides comprehensive operational metrics including
turnover analysis, implementation shortfall, constraint compliance,
and operational efficiency measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..backtest.turnover_tracker import TurnoverTracker


@dataclass
class OperationalMetricsConfig:
    """Configuration for operational metrics calculation."""

    transaction_cost_bps: float = 10.0  # 10 basis points default transaction cost
    max_monthly_turnover: float = 0.20  # 20% max monthly turnover
    implementation_delay_days: int = 1  # Days between signal and execution
    constraint_tolerance: float = 0.01  # 1% tolerance for constraint violations


class OperationalAnalytics:
    """
    Comprehensive operational analytics for portfolio performance.

    Provides metrics for turnover, implementation costs, constraint compliance,
    and operational efficiency across all portfolio models.
    """

    def __init__(self, config: OperationalMetricsConfig = None):
        """
        Initialize operational analytics.

        Args:
            config: Configuration for operational metrics calculation
        """
        self.config = config or OperationalMetricsConfig()
        self.turnover_tracker = TurnoverTracker(
            max_monthly_turnover=self.config.max_monthly_turnover
        )

    def calculate_monthly_turnover_metrics(
        self, portfolio_weights: pd.DataFrame, portfolio_returns: pd.Series
    ) -> dict[str, Any]:
        """
        Create monthly turnover tracking and analysis across all models.

        Args:
            portfolio_weights: DataFrame with dates as index, assets as columns
            portfolio_returns: Portfolio returns series

        Returns:
            Dictionary containing comprehensive turnover metrics
        """
        if portfolio_weights.empty:
            return {"error": "No portfolio weights data provided"}

        turnover_records = []
        monthly_turnovers = []

        # Calculate turnover for each rebalancing period
        for i in range(1, len(portfolio_weights)):
            current_date = portfolio_weights.index[i]
            current_weights = portfolio_weights.iloc[i]
            previous_weights = portfolio_weights.iloc[i - 1]

            # Calculate turnover for this period
            turnover_result = self.turnover_tracker.track_rebalancing(
                date=current_date,
                current_weights=current_weights,
                previous_weights=previous_weights,
            )

            turnover_records.append(turnover_result["tracking_record"])
            monthly_turnovers.append(turnover_result["tracking_record"]["portfolio_turnover"])

        # Calculate summary statistics
        if monthly_turnovers:
            turnover_stats = {
                "avg_monthly_turnover": np.mean(monthly_turnovers),
                "median_monthly_turnover": np.median(monthly_turnovers),
                "std_monthly_turnover": np.std(monthly_turnovers),
                "min_monthly_turnover": np.min(monthly_turnovers),
                "max_monthly_turnover": np.max(monthly_turnovers),
                "total_rebalances": len(monthly_turnovers),
                "constraint_violations": sum(
                    1 for t in turnover_records if not t["constraint_compliant"]
                ),
                "violation_rate": np.mean(
                    [not r["constraint_compliant"] for r in turnover_records]
                ),
            }
        else:
            turnover_stats = {
                "avg_monthly_turnover": 0.0,
                "median_monthly_turnover": 0.0,
                "std_monthly_turnover": 0.0,
                "min_monthly_turnover": 0.0,
                "max_monthly_turnover": 0.0,
                "total_rebalances": 0,
                "constraint_violations": 0,
                "violation_rate": 0.0,
            }

        # Asset-level turnover analysis
        asset_turnover_analysis = self._analyze_asset_level_turnover(portfolio_weights)

        return {
            "turnover_statistics": turnover_stats,
            "turnover_history": turnover_records,
            "asset_analysis": asset_turnover_analysis,
            "annual_turnover_estimate": turnover_stats["avg_monthly_turnover"] * 12,
        }

    def calculate_implementation_shortfall(
        self,
        target_weights: pd.DataFrame,
        actual_weights: pd.DataFrame,
        returns: pd.DataFrame,
        benchmark_returns: pd.Series = None,
    ) -> dict[str, float]:
        """
        Calculate implementation shortfall with transaction cost impact.

        Args:
            target_weights: Target portfolio weights
            actual_weights: Actual implemented weights
            returns: Asset returns during implementation period
            benchmark_returns: Optional benchmark returns

        Returns:
            Dictionary containing implementation shortfall metrics
        """
        if target_weights.empty or actual_weights.empty:
            return {"implementation_shortfall": 0.0, "transaction_cost_impact": 0.0}

        # Align data
        common_dates = target_weights.index.intersection(actual_weights.index)
        if common_dates.empty:
            return {"implementation_shortfall": 0.0, "transaction_cost_impact": 0.0}

        target_aligned = target_weights.loc[common_dates]
        actual_aligned = actual_weights.loc[common_dates]

        # Calculate weight differences
        weight_diffs = actual_aligned - target_aligned

        # Calculate implementation shortfall
        shortfalls = []
        transaction_costs = []

        for date in common_dates:
            if date in returns.index:
                date_returns = returns.loc[date]
                date_weight_diff = weight_diffs.loc[date]

                # Calculate opportunity cost (shortfall)
                aligned_returns = date_returns.reindex(date_weight_diff.index, fill_value=0.0)
                shortfall = (date_weight_diff * aligned_returns).sum()
                shortfalls.append(shortfall)

                # Calculate transaction costs
                turnover = np.abs(date_weight_diff).sum()
                transaction_cost = turnover * (self.config.transaction_cost_bps / 10000)
                transaction_costs.append(transaction_cost)

        avg_shortfall = np.mean(shortfalls) if shortfalls else 0.0
        avg_transaction_cost = np.mean(transaction_costs) if transaction_costs else 0.0

        # Annualize if needed
        trading_days_per_year = 252
        annualized_shortfall = avg_shortfall * trading_days_per_year
        annualized_transaction_cost = avg_transaction_cost * trading_days_per_year

        return {
            "implementation_shortfall": annualized_shortfall,
            "transaction_cost_impact": annualized_transaction_cost,
            "total_implementation_cost": annualized_shortfall + annualized_transaction_cost,
            "avg_daily_shortfall": avg_shortfall,
            "avg_daily_transaction_cost": avg_transaction_cost,
        }

    def calculate_constraint_compliance(
        self, portfolio_weights: pd.DataFrame, constraint_definitions: dict[str, Any] = None
    ) -> dict[str, Any]:
        """
        Add constraint compliance monitoring and violation reporting.

        Args:
            portfolio_weights: DataFrame with portfolio weights over time
            constraint_definitions: Dictionary defining constraints to check

        Returns:
            Dictionary containing constraint compliance metrics
        """
        if constraint_definitions is None:
            constraint_definitions = self._get_default_constraints()

        compliance_results = {}
        violations_by_type = {}

        for constraint_name, constraint_config in constraint_definitions.items():
            violations = []

            if constraint_name == "turnover_limit":
                # Already handled by turnover tracker
                violations = self._check_turnover_violations(portfolio_weights, constraint_config)

            elif constraint_name == "position_limits":
                violations = self._check_position_limit_violations(
                    portfolio_weights, constraint_config
                )

            elif constraint_name == "sector_limits":
                violations = self._check_sector_limit_violations(
                    portfolio_weights, constraint_config
                )

            elif constraint_name == "leverage_limit":
                violations = self._check_leverage_violations(portfolio_weights, constraint_config)

            # Store results
            compliance_results[constraint_name] = {
                "total_violations": len(violations),
                "violation_rate": (
                    len(violations) / len(portfolio_weights) if len(portfolio_weights) > 0 else 0.0
                ),
                "violations": violations[:10],  # Store first 10 violations for analysis
            }

            violations_by_type[constraint_name] = len(violations)

        # Overall compliance metrics
        total_violations = sum(violations_by_type.values())
        total_periods = len(portfolio_weights)
        total_possible_violations = total_periods * len(constraint_definitions)

        # Calculate compliance rate (should be between 0 and 1)
        if total_possible_violations > 0:
            violation_rate = total_violations / total_possible_violations
            overall_compliance_rate = max(0.0, 1.0 - violation_rate)
        else:
            overall_compliance_rate = 1.0

        return {
            "constraint_compliance": compliance_results,
            "overall_compliance_rate": overall_compliance_rate,
            "violations_by_constraint": violations_by_type,
            "total_violations": total_violations,
            "most_violated_constraint": (
                max(violations_by_type, key=violations_by_type.get) if violations_by_type else None
            ),
        }

    def calculate_operational_efficiency(
        self,
        portfolio_weights: pd.DataFrame,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series = None,
    ) -> dict[str, float]:
        """
        Create operational efficiency comparison framework.

        Args:
            portfolio_weights: Portfolio weights over time
            portfolio_returns: Portfolio returns
            benchmark_returns: Optional benchmark returns for comparison

        Returns:
            Dictionary containing operational efficiency metrics
        """
        # Calculate turnover efficiency
        turnover_metrics = self.calculate_monthly_turnover_metrics(
            portfolio_weights, portfolio_returns
        )
        avg_turnover = turnover_metrics["turnover_statistics"]["avg_monthly_turnover"]

        # Calculate return per unit turnover
        if avg_turnover > 0 and not portfolio_returns.empty:
            avg_return = portfolio_returns.mean() * 252  # Annualized
            return_per_turnover = avg_return / (
                avg_turnover * 12
            )  # Return per unit annual turnover
        else:
            return_per_turnover = 0.0

        # Calculate Sharpe ratio per unit turnover
        if avg_turnover > 0 and not portfolio_returns.empty:
            portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = avg_return / portfolio_volatility if portfolio_volatility > 0 else 0.0
            sharpe_per_turnover = sharpe_ratio / (avg_turnover * 12)
        else:
            sharpe_per_turnover = 0.0

        # Calculate implementation efficiency
        num_rebalances = turnover_metrics["turnover_statistics"]["total_rebalances"]
        avg_implementation_cost = avg_turnover * (self.config.transaction_cost_bps / 10000)

        # Calculate active share efficiency (if benchmark provided)
        active_share_efficiency = 0.0
        if benchmark_returns is not None:
            active_return = (portfolio_returns.mean() - benchmark_returns.mean()) * 252
            active_share_efficiency = active_return / avg_turnover if avg_turnover > 0 else 0.0

        return {
            "return_per_turnover": return_per_turnover,
            "sharpe_per_turnover": sharpe_per_turnover,
            "avg_implementation_cost": avg_implementation_cost * 12,  # Annualized
            "rebalancing_frequency": (
                num_rebalances / (len(portfolio_weights) / 252)
                if len(portfolio_weights) > 252
                else 0.0
            ),
            "active_share_efficiency": active_share_efficiency,
            "turnover_adjusted_return": avg_return - (avg_implementation_cost * 12),
            "efficiency_score": (
                return_per_turnover * sharpe_per_turnover if sharpe_per_turnover > 0 else 0.0
            ),
        }

    def generate_operational_report(
        self,
        portfolio_weights: pd.DataFrame,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series = None,
        constraint_definitions: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """
        Generate comprehensive operational performance report.

        Args:
            portfolio_weights: Portfolio weights over time
            portfolio_returns: Portfolio returns
            benchmark_returns: Optional benchmark returns
            constraint_definitions: Optional constraint definitions

        Returns:
            Comprehensive operational metrics report
        """
        report = {}

        # Turnover analysis
        report["turnover_analysis"] = self.calculate_monthly_turnover_metrics(
            portfolio_weights, portfolio_returns
        )

        # Implementation shortfall (simplified version without actual vs target)
        if not portfolio_weights.empty:
            # Calculate pseudo implementation shortfall using weight changes
            weight_changes = portfolio_weights.diff().abs().sum(axis=1).dropna()
            avg_weight_change = weight_changes.mean()
            estimated_shortfall = (
                avg_weight_change * (self.config.transaction_cost_bps / 10000) * 252
            )

            report["implementation_metrics"] = {
                "estimated_implementation_shortfall": estimated_shortfall,
                "avg_weight_change": avg_weight_change,
                "implementation_frequency": len(weight_changes),
            }

        # Constraint compliance
        report["constraint_compliance"] = self.calculate_constraint_compliance(
            portfolio_weights, constraint_definitions
        )

        # Operational efficiency
        report["operational_efficiency"] = self.calculate_operational_efficiency(
            portfolio_weights, portfolio_returns, benchmark_returns
        )

        # Summary metrics
        report["summary"] = {
            "avg_monthly_turnover": report["turnover_analysis"]["turnover_statistics"][
                "avg_monthly_turnover"
            ],
            "constraint_compliance_rate": report["constraint_compliance"][
                "overall_compliance_rate"
            ],
            "operational_efficiency_score": report["operational_efficiency"]["efficiency_score"],
            "estimated_annual_transaction_costs": report["operational_efficiency"][
                "avg_implementation_cost"
            ],
        }

        return report

    def _analyze_asset_level_turnover(self, portfolio_weights: pd.DataFrame) -> dict[str, Any]:
        """Analyze turnover at the asset level."""
        if portfolio_weights.empty:
            return {}

        # Calculate asset-level turnover for each period
        asset_turnovers = {}

        for asset in portfolio_weights.columns:
            asset_weights = portfolio_weights[asset]
            asset_weight_changes = asset_weights.diff().abs().dropna()

            if not asset_weight_changes.empty:
                asset_turnovers[asset] = {
                    "avg_turnover": asset_weight_changes.mean(),
                    "total_turnover": asset_weight_changes.sum(),
                    "max_turnover": asset_weight_changes.max(),
                    "num_changes": (asset_weight_changes > 0.001).sum(),  # Changes > 0.1%
                }

        # Find most/least active assets
        if asset_turnovers:
            most_active = max(
                asset_turnovers.keys(), key=lambda x: asset_turnovers[x]["avg_turnover"]
            )
            least_active = min(
                asset_turnovers.keys(), key=lambda x: asset_turnovers[x]["avg_turnover"]
            )

            return {
                "asset_turnovers": asset_turnovers,
                "most_active_asset": most_active,
                "least_active_asset": least_active,
                "num_assets_analyzed": len(asset_turnovers),
            }

        return {}

    def _get_default_constraints(self) -> dict[str, Any]:
        """Get default constraint definitions."""
        return {
            "turnover_limit": {"max_monthly": self.config.max_monthly_turnover},
            "position_limits": {"max_weight": 0.10, "min_weight": 0.0},
            "leverage_limit": {"max_leverage": 1.0},
        }

    def _check_turnover_violations(
        self, portfolio_weights: pd.DataFrame, constraint_config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Check turnover constraint violations."""
        violations = []
        max_turnover = constraint_config.get("max_monthly", self.config.max_monthly_turnover)

        for i in range(1, len(portfolio_weights)):
            current_weights = portfolio_weights.iloc[i]
            previous_weights = portfolio_weights.iloc[i - 1]

            turnover = np.abs(current_weights - previous_weights).sum()

            if turnover > max_turnover:
                violations.append(
                    {
                        "date": portfolio_weights.index[i],
                        "violation_type": "turnover_limit",
                        "actual_value": turnover,
                        "limit": max_turnover,
                        "excess": turnover - max_turnover,
                    }
                )

        return violations

    def _check_position_limit_violations(
        self, portfolio_weights: pd.DataFrame, constraint_config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Check position limit violations."""
        violations = []
        max_weight = constraint_config.get("max_weight", 0.10)
        min_weight = constraint_config.get("min_weight", 0.0)

        for date in portfolio_weights.index:
            weights = portfolio_weights.loc[date]

            # Check max weight violations
            max_violations = weights[weights > max_weight]
            for asset, weight in max_violations.items():
                violations.append(
                    {
                        "date": date,
                        "asset": asset,
                        "violation_type": "max_position_limit",
                        "actual_value": weight,
                        "limit": max_weight,
                        "excess": weight - max_weight,
                    }
                )

            # Check min weight violations (if applicable)
            if min_weight > 0:
                min_violations = weights[(weights > 0) & (weights < min_weight)]
                for asset, weight in min_violations.items():
                    violations.append(
                        {
                            "date": date,
                            "asset": asset,
                            "violation_type": "min_position_limit",
                            "actual_value": weight,
                            "limit": min_weight,
                            "shortfall": min_weight - weight,
                        }
                    )

        return violations

    def _check_sector_limit_violations(
        self, portfolio_weights: pd.DataFrame, constraint_config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Check sector limit violations (placeholder implementation)."""
        # This would require sector mapping data which isn't available in this context
        return []

    def _check_leverage_violations(
        self, portfolio_weights: pd.DataFrame, constraint_config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Check leverage violations."""
        violations = []
        max_leverage = constraint_config.get("max_leverage", 1.0)

        for date in portfolio_weights.index:
            weights = portfolio_weights.loc[date]
            total_weight = weights.abs().sum()  # Total absolute weight for leverage calculation

            if total_weight > max_leverage + self.config.constraint_tolerance:
                violations.append(
                    {
                        "date": date,
                        "violation_type": "leverage_limit",
                        "actual_value": total_weight,
                        "limit": max_leverage,
                        "excess": total_weight - max_leverage,
                    }
                )

        return violations
