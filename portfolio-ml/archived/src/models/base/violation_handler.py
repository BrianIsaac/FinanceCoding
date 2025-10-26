"""
Constraint violation handling and remediation system.

This module provides hierarchical remediation strategies, violation severity
classification, and portfolio feasibility validation for the unified
constraint system.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd

from .constraints import ConstraintViolation, ConstraintViolationType, ViolationSeverity


class RemediationStrategy(Enum):
    """Available remediation strategies for constraint violations."""

    IGNORE = "ignore"  # Log but don't adjust
    WARN = "warn"  # Issue warning and log
    ADJUST = "adjust"  # Automatically adjust weights
    REJECT = "reject"  # Reject the portfolio entirely


@dataclass
class RemediationAction:
    """Remediation action for a constraint violation."""

    strategy: RemediationStrategy
    description: str
    adjustments_made: dict[str, Any]
    success: bool
    fallback_used: bool = False


class ViolationHandler:
    """
    Hierarchical constraint violation handler with multiple remediation strategies.

    Provides sophisticated violation handling including warnings, automatic
    adjustments, and fallback strategies to maintain portfolio feasibility.
    """

    def __init__(
        self,
        default_strategy: RemediationStrategy = RemediationStrategy.ADJUST,
        severity_strategies: dict[ViolationSeverity, RemediationStrategy] | None = None,
        enable_fallbacks: bool = True,
    ):
        """
        Initialize violation handler.

        Args:
            default_strategy: Default remediation strategy
            severity_strategies: Strategy mapping by violation severity
            enable_fallbacks: Whether to use fallback strategies
        """
        self.default_strategy = default_strategy
        self.severity_strategies = severity_strategies or {
            ViolationSeverity.WARNING: RemediationStrategy.WARN,
            ViolationSeverity.MINOR: RemediationStrategy.ADJUST,
            ViolationSeverity.MAJOR: RemediationStrategy.ADJUST,
            ViolationSeverity.CRITICAL: RemediationStrategy.REJECT,
        }
        self.enable_fallbacks = enable_fallbacks

        # Tracking
        self.remediation_history: list[dict[str, Any]] = []

    def handle_violations(
        self,
        violations: list[ConstraintViolation],
        weights: pd.Series,
        previous_weights: pd.Series | None = None,
    ) -> tuple[pd.Series, list[RemediationAction]]:
        """
        Handle all violations with appropriate remediation strategies.

        Args:
            violations: List of constraint violations
            weights: Current portfolio weights
            previous_weights: Previous weights for context

        Returns:
            Tuple of (adjusted_weights, remediation_actions)
        """
        if not violations:
            return weights, []

        adjusted_weights = weights.copy()
        remediation_actions = []

        # Sort violations by severity (critical first)
        severity_order = {
            ViolationSeverity.CRITICAL: 0,
            ViolationSeverity.MAJOR: 1,
            ViolationSeverity.MINOR: 2,
            ViolationSeverity.WARNING: 3,
        }
        sorted_violations = sorted(violations, key=lambda v: severity_order[v.severity])

        # Handle each violation
        for violation in sorted_violations:
            strategy = self._get_strategy_for_violation(violation)

            if strategy == RemediationStrategy.REJECT:
                # Reject entire portfolio - return equal weights as fallback
                if self.enable_fallbacks:
                    fallback_weights = self._create_fallback_portfolio(weights)
                    action = RemediationAction(
                        strategy=strategy,
                        description=f"Portfolio rejected due to {violation.violation_type.value} violation, using fallback",
                        adjustments_made={"fallback_portfolio": True},
                        success=True,
                        fallback_used=True,
                    )
                    return fallback_weights, [action]
                else:
                    action = RemediationAction(
                        strategy=strategy,
                        description=f"Portfolio rejected due to {violation.violation_type.value} violation",
                        adjustments_made={},
                        success=False,
                    )
                    remediation_actions.append(action)
                    break

            elif strategy == RemediationStrategy.ADJUST:
                adjusted_weights, action = self._apply_adjustment(
                    violation, adjusted_weights, previous_weights
                )
                remediation_actions.append(action)

            elif strategy == RemediationStrategy.WARN:
                action = RemediationAction(
                    strategy=strategy,
                    description=f"Warning: {violation.description}",
                    adjustments_made={},
                    success=True,
                )
                remediation_actions.append(action)

            else:  # IGNORE
                action = RemediationAction(
                    strategy=strategy,
                    description=f"Ignored: {violation.description}",
                    adjustments_made={},
                    success=True,
                )
                remediation_actions.append(action)

        # Log remediation actions
        self._log_remediation(violations, remediation_actions)

        return adjusted_weights, remediation_actions

    def _get_strategy_for_violation(self, violation: ConstraintViolation) -> RemediationStrategy:
        """Get appropriate remediation strategy for a violation."""
        return self.severity_strategies.get(violation.severity, self.default_strategy)

    def _apply_adjustment(
        self,
        violation: ConstraintViolation,
        weights: pd.Series,
        previous_weights: pd.Series | None,
    ) -> tuple[pd.Series, RemediationAction]:
        """Apply adjustment for specific violation type."""
        if violation.violation_type == ConstraintViolationType.LONG_ONLY:
            return self._adjust_long_only_violation(violation, weights)
        elif violation.violation_type == ConstraintViolationType.TOP_K_POSITIONS:
            return self._adjust_top_k_violation(violation, weights)
        elif violation.violation_type == ConstraintViolationType.MAX_POSITION_WEIGHT:
            return self._adjust_max_weight_violation(violation, weights)
        elif violation.violation_type == ConstraintViolationType.MONTHLY_TURNOVER:
            return self._adjust_turnover_violation(violation, weights, previous_weights)
        else:
            # Generic adjustment - normalize weights
            return self._generic_adjustment(violation, weights)

    def _adjust_long_only_violation(
        self, violation: ConstraintViolation, weights: pd.Series
    ) -> tuple[pd.Series, RemediationAction]:
        """Adjust long-only constraint violation."""
        negative_mask = weights < 0
        negative_weight = weights[negative_mask].sum()

        # Clip negative weights to zero
        adjusted_weights = weights.clip(lower=0.0)

        # Renormalize
        weight_sum = adjusted_weights.sum()
        if weight_sum > 0:
            adjusted_weights = adjusted_weights / weight_sum

        action = RemediationAction(
            strategy=RemediationStrategy.ADJUST,
            description=f"Clipped {negative_mask.sum()} negative weights (total: {negative_weight:.4f})",
            adjustments_made={
                "clipped_assets": negative_mask.sum(),
                "clipped_weight": negative_weight,
                "renormalized": True,
            },
            success=True,
        )

        return adjusted_weights, action

    def _adjust_top_k_violation(
        self, violation: ConstraintViolation, weights: pd.Series
    ) -> tuple[pd.Series, RemediationAction]:
        """Adjust top-k positions violation."""
        k = int(violation.constraint_value)

        if len(weights) <= k:
            # No adjustment needed
            action = RemediationAction(
                strategy=RemediationStrategy.ADJUST,
                description="No adjustment needed for top-k constraint",
                adjustments_made={},
                success=True,
            )
            return weights, action

        # Select top-k positions by weight
        top_k_assets = weights.nlargest(k).index
        adjusted_weights = pd.Series(0.0, index=weights.index)
        adjusted_weights.loc[top_k_assets] = weights.loc[top_k_assets]

        # Renormalize
        weight_sum = adjusted_weights.sum()
        if weight_sum > 0:
            adjusted_weights = adjusted_weights / weight_sum

        excluded_positions = len(weights) - k
        action = RemediationAction(
            strategy=RemediationStrategy.ADJUST,
            description=f"Excluded {excluded_positions} positions to satisfy top-{k} constraint",
            adjustments_made={
                "excluded_positions": excluded_positions,
                "selected_assets": list(top_k_assets),
                "renormalized": True,
            },
            success=True,
        )

        return adjusted_weights, action

    def _adjust_max_weight_violation(
        self, violation: ConstraintViolation, weights: pd.Series
    ) -> tuple[pd.Series, RemediationAction]:
        """Adjust maximum position weight violation."""
        max_weight = violation.constraint_value
        violating_mask = weights > max_weight

        if not violating_mask.any():
            action = RemediationAction(
                strategy=RemediationStrategy.ADJUST,
                description="No max weight violations found",
                adjustments_made={},
                success=True,
            )
            return weights, action

        adjusted_weights = weights.copy()
        excess_weight = (adjusted_weights[violating_mask] - max_weight).sum()

        # Cap violating weights
        adjusted_weights[violating_mask] = max_weight

        # Redistribute excess to non-violating assets
        non_violating_mask = ~violating_mask
        if bool(non_violating_mask.any()) and excess_weight > 0:
            available_capacity = (max_weight - adjusted_weights[non_violating_mask]).clip(lower=0)
            total_capacity = available_capacity.sum()

            if total_capacity > 0:
                redistribution = excess_weight * (available_capacity / total_capacity)
                adjusted_weights[non_violating_mask] += redistribution

        action = RemediationAction(
            strategy=RemediationStrategy.ADJUST,
            description=f"Capped {violating_mask.sum()} positions at {max_weight:.1%}, redistributed {excess_weight:.4f}",
            adjustments_made={
                "capped_positions": violating_mask.sum(),
                "redistributed_weight": excess_weight,
                "max_weight": max_weight,
            },
            success=True,
        )

        return adjusted_weights, action

    def _adjust_turnover_violation(
        self,
        violation: ConstraintViolation,
        weights: pd.Series,
        previous_weights: pd.Series | None,
    ) -> tuple[pd.Series, RemediationAction]:
        """Adjust turnover constraint violation."""
        if previous_weights is None:
            action = RemediationAction(
                strategy=RemediationStrategy.ADJUST,
                description="Cannot adjust turnover violation without previous weights",
                adjustments_made={},
                success=False,
            )
            return weights, action

        max_turnover = violation.constraint_value
        current_turnover = violation.violation_value

        # Calculate required blend factor
        if current_turnover > 0:
            blend_factor = max_turnover / current_turnover
        else:
            blend_factor = 1.0

        # Align weights and blend
        all_assets = weights.index.union(previous_weights.index)
        current_aligned = weights.reindex(all_assets, fill_value=0.0)
        previous_aligned = previous_weights.reindex(all_assets, fill_value=0.0)

        adjusted_weights = blend_factor * current_aligned + (1 - blend_factor) * previous_aligned

        # Renormalize
        weight_sum = adjusted_weights.sum()
        if weight_sum > 0:
            adjusted_weights = adjusted_weights / weight_sum

        action = RemediationAction(
            strategy=RemediationStrategy.ADJUST,
            description=f"Blended with previous weights (factor: {blend_factor:.3f}) to reduce turnover",
            adjustments_made={
                "blend_factor": blend_factor,
                "original_turnover": current_turnover,
                "target_turnover": max_turnover,
            },
            success=True,
        )

        return adjusted_weights, action

    def _generic_adjustment(
        self, violation: ConstraintViolation, weights: pd.Series
    ) -> tuple[pd.Series, RemediationAction]:
        """Apply generic weight normalization adjustment."""
        # Simple normalization
        weight_sum = weights.sum()
        if weight_sum > 0:
            adjusted_weights = weights / weight_sum
        else:
            adjusted_weights = pd.Series(1.0 / len(weights), index=weights.index)

        action = RemediationAction(
            strategy=RemediationStrategy.ADJUST,
            description=f"Applied generic normalization for {violation.violation_type.value} violation",
            adjustments_made={"normalized": True, "original_sum": weight_sum},
            success=True,
        )

        return adjusted_weights, action

    def _create_fallback_portfolio(self, weights: pd.Series) -> pd.Series:
        """Create fallback portfolio when original is rejected."""
        # Equal weight portfolio as ultimate fallback
        if len(weights) > 0:
            return pd.Series(1.0 / len(weights), index=weights.index)
        else:
            return weights

    def _log_remediation(
        self, violations: list[ConstraintViolation], actions: list[RemediationAction]
    ) -> None:
        """Log remediation actions for tracking and analysis."""
        remediation_record = {
            "timestamp": pd.Timestamp.now(),
            "num_violations": len(violations),
            "violation_types": [v.violation_type.value for v in violations],
            "severities": [v.severity.value for v in violations],
            "num_actions": len(actions),
            "strategies_used": [a.strategy.value for a in actions],
            "successful_actions": sum(1 for a in actions if a.success),
            "fallbacks_used": sum(1 for a in actions if a.fallback_used),
        }
        self.remediation_history.append(remediation_record)

    def get_remediation_statistics(self) -> dict[str, Any]:
        """Get statistics about remediation actions."""
        if not self.remediation_history:
            return {"total_remediations": 0}

        total_violations = sum(record["num_violations"] for record in self.remediation_history)
        total_actions = sum(record["num_actions"] for record in self.remediation_history)
        successful_actions = sum(
            record["successful_actions"] for record in self.remediation_history
        )
        fallbacks_used = sum(record["fallbacks_used"] for record in self.remediation_history)

        # Count strategy usage
        strategy_counts = {}
        for record in self.remediation_history:
            for strategy in record["strategies_used"]:
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        # Count violation types
        violation_type_counts = {}
        for record in self.remediation_history:
            for violation_type in record["violation_types"]:
                violation_type_counts[violation_type] = (
                    violation_type_counts.get(violation_type, 0) + 1
                )

        return {
            "total_remediations": len(self.remediation_history),
            "total_violations_handled": total_violations,
            "total_actions_taken": total_actions,
            "success_rate": successful_actions / total_actions if total_actions > 0 else 0.0,
            "fallback_rate": fallbacks_used / total_actions if total_actions > 0 else 0.0,
            "strategy_usage": strategy_counts,
            "violation_type_distribution": violation_type_counts,
            "avg_violations_per_remediation": total_violations / len(self.remediation_history),
        }

    def create_remediation_report(self) -> dict[str, Any]:
        """Create comprehensive remediation report."""
        statistics = self.get_remediation_statistics()

        return {
            "handler_configuration": {
                "default_strategy": self.default_strategy.value,
                "severity_strategies": {
                    severity.value: strategy.value
                    for severity, strategy in self.severity_strategies.items()
                },
                "fallbacks_enabled": self.enable_fallbacks,
            },
            "statistics": statistics,
            "recent_history": self.remediation_history[-10:] if self.remediation_history else [],
        }
