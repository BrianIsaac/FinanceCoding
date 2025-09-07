"""
Unified constraint engine for portfolio optimization models.

This module provides the main ConstraintEngine class that integrates with
transaction cost calculations and provides comprehensive constraint enforcement
across all portfolio models (HRP, LSTM, GAT, baselines).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ...evaluation.backtest.transaction_costs import TransactionCostCalculator
from .constraints import ConstraintEngine as BaseConstraintEngine
from .constraints import ConstraintViolation, PortfolioConstraints


class UnifiedConstraintEngine:
    """
    Unified constraint engine with transaction cost integration.

    This class combines the base constraint enforcement with transaction
    cost modeling to provide comprehensive portfolio optimization constraint
    handling across all models.
    """

    def __init__(
        self,
        constraints: PortfolioConstraints,
        transaction_cost_calculator: Optional[TransactionCostCalculator] = None,
    ):
        """
        Initialize unified constraint engine.

        Args:
            constraints: Portfolio constraints configuration
            transaction_cost_calculator: Optional transaction cost calculator
        """
        self.base_engine = BaseConstraintEngine(constraints)
        self.transaction_cost_calculator = transaction_cost_calculator
        self.constraints = constraints

    def enforce_all_constraints(
        self,
        weights: pd.Series,
        previous_weights: Optional[pd.Series] = None,
        model_scores: Optional[pd.Series] = None,
        date: Optional[pd.Timestamp] = None,
        portfolio_value: float = 1.0,
    ) -> Tuple[pd.Series, List[ConstraintViolation], Dict[str, Any]]:
        """
        Apply all constraints including transaction cost optimization.

        Args:
            weights: Raw portfolio weights from model
            previous_weights: Previous period weights
            model_scores: Model confidence scores for ranking
            date: Current rebalancing date
            portfolio_value: Total portfolio value

        Returns:
            Tuple of (constrained_weights, violations, cost_analysis)
        """
        # Step 1: Apply base constraints
        constrained_weights, violations = self.base_engine.enforce_constraints(
            weights, previous_weights, model_scores, date
        )

        # Step 2: Calculate transaction costs if available
        cost_analysis = {}
        if (
            self.transaction_cost_calculator is not None
            and previous_weights is not None
            and self.constraints.cost_aware_enforcement
        ):
            cost_analysis = self.transaction_cost_calculator.calculate_transaction_costs(
                previous_weights, constrained_weights, portfolio_value
            )

            # Step 3: Cost-aware constraint adjustment if needed
            if cost_analysis["total_cost"] > (self.constraints.transaction_cost_bps / 10000.0):
                constrained_weights = self._apply_cost_aware_adjustment(
                    constrained_weights, previous_weights, cost_analysis, violations
                )

        return constrained_weights, violations, cost_analysis

    def _apply_cost_aware_adjustment(
        self,
        weights: pd.Series,
        previous_weights: pd.Series,
        cost_analysis: Dict[str, Any],
        violations: List[ConstraintViolation],
    ) -> pd.Series:
        """
        Apply cost-aware adjustments to reduce transaction costs.

        Args:
            weights: Current constrained weights
            previous_weights: Previous period weights
            cost_analysis: Transaction cost analysis
            violations: Current violations list

        Returns:
            Cost-adjusted portfolio weights
        """
        # If transaction costs are too high, blend more with previous weights
        if cost_analysis["turnover"] > 0:
            # Calculate cost-efficiency ratio
            cost_per_turnover = cost_analysis["cost_per_turnover"]
            max_acceptable_cost = self.constraints.transaction_cost_bps / 10000.0

            if cost_per_turnover > max_acceptable_cost:
                # Reduce turnover by blending more with previous weights
                target_cost_ratio = max_acceptable_cost / cost_per_turnover
                blend_factor = min(0.9, target_cost_ratio)  # Cap at 90% previous weights

                # Align weights
                all_assets = weights.index.union(previous_weights.index)
                current_aligned = weights.reindex(all_assets, fill_value=0.0)
                previous_aligned = previous_weights.reindex(all_assets, fill_value=0.0)

                # Apply cost-aware blending
                adjusted_weights = (
                    blend_factor * previous_aligned + (1 - blend_factor) * current_aligned
                )

                # Renormalize
                weight_sum = adjusted_weights.sum()
                if weight_sum > 0:
                    adjusted_weights = adjusted_weights / weight_sum

                return adjusted_weights

        return weights

    def validate_portfolio_feasibility(
        self, weights: pd.Series, previous_weights: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Validate overall portfolio feasibility and constraint adherence.

        Args:
            weights: Portfolio weights to validate
            previous_weights: Previous weights for comparison

        Returns:
            Dictionary with feasibility assessment and recommendations
        """
        # Get base constraint metrics
        metrics = self.base_engine.calculate_constraint_metrics(weights, previous_weights)

        # Check violations
        violations = self.base_engine.check_violations(weights, previous_weights)

        # Calculate severity score
        severity_scores = {"warning": 1, "minor": 2, "major": 3, "critical": 4}
        total_severity = sum(severity_scores.get(v.severity.value, 0) for v in violations)

        # Determine feasibility status
        if total_severity == 0:
            feasibility_status = "fully_compliant"
        elif total_severity <= 3:
            feasibility_status = "acceptable_with_warnings"
        elif total_severity <= 6:
            feasibility_status = "requires_adjustment"
        else:
            feasibility_status = "infeasible"

        # Generate recommendations
        recommendations = []
        if violations:
            violation_types = {v.violation_type.value for v in violations}
            if "long_only" in violation_types:
                recommendations.append("Enable long-only constraint enforcement")
            if "top_k_positions" in violation_types:
                recommendations.append("Reduce number of positions or increase top-k limit")
            if "monthly_turnover" in violation_types:
                recommendations.append("Increase turnover limit or enable turnover blending")
            if "max_position_weight" in violation_types:
                recommendations.append("Increase max position weight or enable redistribution")

        return {
            "feasibility_status": feasibility_status,
            "total_violations": len(violations),
            "severity_score": total_severity,
            "constraint_metrics": metrics,
            "violations": [
                {
                    "type": v.violation_type.value,
                    "severity": v.severity.value,
                    "description": v.description,
                }
                for v in violations
            ],
            "recommendations": recommendations,
        }

    def get_constraint_configuration(self) -> Dict[str, Any]:
        """Get current constraint configuration for reporting."""
        return {
            "basic_constraints": {
                "long_only": self.constraints.long_only,
                "top_k_positions": self.constraints.top_k_positions,
                "max_position_weight": self.constraints.max_position_weight,
                "min_weight_threshold": self.constraints.min_weight_threshold,
            },
            "turnover_constraints": {
                "max_monthly_turnover": self.constraints.max_monthly_turnover,
                "enable_turnover_penalty": self.constraints.enable_turnover_penalty,
                "turnover_lookback_months": self.constraints.turnover_lookback_months,
            },
            "transaction_cost_settings": {
                "transaction_cost_bps": self.constraints.transaction_cost_bps,
                "cost_aware_enforcement": self.constraints.cost_aware_enforcement,
            },
            "enforcement_settings": {
                "violation_handling": self.constraints.violation_handling,
                "position_ranking_method": self.constraints.position_ranking_method,
            },
        }

    def create_enforcement_report(
        self, weights: pd.Series, previous_weights: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Create comprehensive constraint enforcement report.

        Args:
            weights: Current portfolio weights
            previous_weights: Previous weights for comparison

        Returns:
            Comprehensive enforcement report
        """
        # Get base constraint report
        base_report = self.base_engine.create_constraint_report()

        # Add feasibility assessment
        feasibility = self.validate_portfolio_feasibility(weights, previous_weights)

        # Add transaction cost analysis if available
        cost_analysis = {}
        if self.transaction_cost_calculator and previous_weights is not None:
            cost_analysis = self.transaction_cost_calculator.calculate_transaction_costs(
                previous_weights, weights
            )

        return {
            "constraint_enforcement": base_report,
            "portfolio_feasibility": feasibility,
            "transaction_cost_analysis": cost_analysis,
            "configuration": self.get_constraint_configuration(),
            "summary": {
                "total_positions": (weights > 0).sum(),
                "max_weight": weights.max(),
                "weight_sum": weights.sum(),
                "is_feasible": feasibility["feasibility_status"] != "infeasible",
                "has_violations": len(feasibility["violations"]) > 0,
                "estimated_transaction_cost": cost_analysis.get("total_cost", 0.0),
            },
        }
