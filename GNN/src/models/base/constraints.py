"""
Portfolio constraint system for unified risk management.

This module provides constraint enforcement and validation utilities
that ensure all portfolio models comply with risk management requirements
and regulatory constraints. Implements unified constraint system from Story 2.4.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class ConstraintViolationType(Enum):
    """Types of portfolio constraint violations."""

    LONG_ONLY = "long_only"
    TOP_K_POSITIONS = "top_k_positions"
    MAX_POSITION_WEIGHT = "max_position_weight"
    MONTHLY_TURNOVER = "monthly_turnover"
    TRANSACTION_COST = "transaction_cost"
    WEIGHT_THRESHOLD = "weight_threshold"


class ViolationSeverity(Enum):
    """Severity levels for constraint violations."""

    WARNING = "warning"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


@dataclass
class ConstraintViolation:
    """Record of a constraint violation."""

    violation_type: ConstraintViolationType
    severity: ViolationSeverity
    description: str
    violation_value: float
    constraint_value: float
    remediation_action: str


@dataclass
class PortfolioConstraints:
    """Unified portfolio constraints configuration."""

    # Basic constraints
    long_only: bool = True
    top_k_positions: int | None = None
    max_position_weight: float = 0.10
    min_weight_threshold: float = 0.01

    # Turnover and transaction cost constraints
    max_monthly_turnover: float = 0.20
    transaction_cost_bps: float = 10.0
    turnover_lookback_months: int = 1

    # Enhanced constraint parameters
    violation_handling: str = "adjust"  # "warn", "adjust", "reject"
    position_ranking_method: str = "weight"  # "weight", "score", "mixed"
    cost_aware_enforcement: bool = True
    enable_turnover_penalty: bool = True

    # Risk-based constraints (for future use)
    max_portfolio_volatility: float | None = None
    max_asset_correlation: float | None = None
    min_diversification_ratio: float | None = None

    # Regulatory constraints
    max_sector_concentration: float | None = None
    max_single_issuer_weight: float = 0.10
    min_liquidity_threshold: float | None = None


class ConstraintEngine:
    """
    Unified constraint enforcement engine for all portfolio models.

    This class provides centralized constraint enforcement to ensure
    consistent risk management across HRP, LSTM, GAT, and baseline models.
    Supports violation tracking, remediation strategies, and comprehensive
    constraint adherence monitoring.
    """

    def __init__(self, constraints: PortfolioConstraints):
        """
        Initialize constraint engine.

        Args:
            constraints: Unified portfolio constraints configuration
        """
        self.constraints = constraints
        self.violation_log: list[dict[str, Any]] = []
        self.turnover_history: dict[str, list[float]] = {}

    def enforce_constraints(
        self,
        weights: pd.Series,
        previous_weights: pd.Series | None = None,
        model_scores: pd.Series | None = None,
        date: pd.Timestamp | None = None,
    ) -> tuple[pd.Series, list[ConstraintViolation]]:
        """
        Apply all constraints to portfolio weights with comprehensive violation tracking.

        Args:
            weights: Raw portfolio weights from model
            previous_weights: Previous period weights for turnover calculation
            model_scores: Model confidence scores for position ranking
            date: Current rebalancing date for logging

        Returns:
            Tuple of (constrained_weights, violations_list)
        """
        violations = []
        constrained_weights = weights.copy()

        # Step 1: Check all violations before applying constraints
        violations.extend(self.check_violations(constrained_weights, previous_weights))

        # Step 2: Apply constraints in priority order
        # Apply turnover constraint first to preserve asset universe continuity
        constrained_weights = self._apply_turnover_constraint(
            constrained_weights, previous_weights, violations
        )
        constrained_weights = self._apply_long_only_constraint(constrained_weights, violations)
        constrained_weights = self._apply_weight_threshold_constraint(
            constrained_weights, violations
        )
        constrained_weights = self._apply_top_k_constraint(
            constrained_weights, model_scores, violations
        )
        constrained_weights = self._apply_max_weight_constraint(constrained_weights, violations)

        # Step 3: Final iterative constraint application and normalization
        # Apply max weight constraint and normalize iteratively until convergence
        max_iters = 5
        for _iteration in range(max_iters):
            # Check if max weight constraint is violated
            if (constrained_weights > self.constraints.max_position_weight).any():
                constrained_weights = self._apply_max_weight_constraint(constrained_weights, violations)

            # Normalize weights
            constrained_weights = self._normalize_weights(constrained_weights)

            # Check convergence: if weights sum to 1.0 and no max weight violations, we're done
            weight_sum = constrained_weights.sum()
            max_weight_violated = (constrained_weights > self.constraints.max_position_weight + 1e-10).any()
            weights_normalized = abs(weight_sum - 1.0) < 1e-10

            if weights_normalized and not max_weight_violated:
                break

        # Step 4: Log violations if date provided
        if date is not None and violations:
            self._log_violations(date, violations)

        return constrained_weights, violations

    def check_violations(
        self, weights: pd.Series, previous_weights: pd.Series | None = None
    ) -> list[ConstraintViolation]:
        """
        Check for constraint violations without applying adjustments.

        Args:
            weights: Portfolio weights to check
            previous_weights: Previous weights for turnover checks

        Returns:
            List of detected constraint violations
        """
        violations = []

        # Long-only violations
        if self.constraints.long_only and (weights < 0).any():
            negative_count = (weights < 0).sum()
            min_weight = weights.min()
            violations.append(
                ConstraintViolation(
                    violation_type=ConstraintViolationType.LONG_ONLY,
                    severity=ViolationSeverity.MAJOR,
                    description=f"{negative_count} assets have negative weights",
                    violation_value=min_weight,
                    constraint_value=0.0,
                    remediation_action="Clip negative weights to zero and redistribute",
                )
            )

        # Max position weight violations
        if (weights > self.constraints.max_position_weight).any():
            violating_assets = weights[weights > self.constraints.max_position_weight]
            max_violation = violating_assets.max()
            violations.append(
                ConstraintViolation(
                    violation_type=ConstraintViolationType.MAX_POSITION_WEIGHT,
                    severity=ViolationSeverity.MINOR,
                    description=f"{len(violating_assets)} assets exceed max weight",
                    violation_value=max_violation,
                    constraint_value=self.constraints.max_position_weight,
                    remediation_action="Cap weights and redistribute excess",
                )
            )

        # Top-k constraint violations
        if self.constraints.top_k_positions is not None:
            num_positions = (weights > 0).sum()
            if num_positions > self.constraints.top_k_positions:
                violations.append(
                    ConstraintViolation(
                        violation_type=ConstraintViolationType.TOP_K_POSITIONS,
                        severity=ViolationSeverity.MINOR,
                        description=f"Portfolio has {num_positions} positions, max allowed: {self.constraints.top_k_positions}",
                        violation_value=float(num_positions),
                        constraint_value=float(self.constraints.top_k_positions),
                        remediation_action="Select top-k positions by ranking method",
                    )
                )

        # Turnover violations
        if previous_weights is not None and self.constraints.enable_turnover_penalty:
            turnover = self.calculate_turnover(weights, previous_weights)
            if turnover > self.constraints.max_monthly_turnover:
                violations.append(
                    ConstraintViolation(
                        violation_type=ConstraintViolationType.MONTHLY_TURNOVER,
                        severity=ViolationSeverity.WARNING,
                        description=f"Turnover {turnover:.1%} exceeds limit {self.constraints.max_monthly_turnover:.1%}",
                        violation_value=turnover,
                        constraint_value=self.constraints.max_monthly_turnover,
                        remediation_action="Blend with previous weights to reduce turnover",
                    )
                )

        return violations

    def _apply_long_only_constraint(
        self, weights: pd.Series, violations: list[ConstraintViolation]
    ) -> pd.Series:
        """Apply long-only constraint by clipping negative weights."""
        if not self.constraints.long_only:
            return weights

        # Clip negative weights to zero
        return weights.clip(lower=0.0)

    def _apply_weight_threshold_constraint(
        self, weights: pd.Series, violations: list[ConstraintViolation]
    ) -> pd.Series:
        """Apply minimum weight threshold constraint."""
        return weights.where(weights >= self.constraints.min_weight_threshold, 0.0)

    def _apply_top_k_constraint(
        self,
        weights: pd.Series,
        model_scores: pd.Series | None,
        violations: list[ConstraintViolation],
    ) -> pd.Series:
        """Apply top-k positions constraint based on ranking method."""
        if self.constraints.top_k_positions is None:
            return weights

        # Choose ranking method
        if self.constraints.position_ranking_method == "weight":
            ranking_values = weights
        elif self.constraints.position_ranking_method == "score" and model_scores is not None:
            ranking_values = model_scores
        elif self.constraints.position_ranking_method == "mixed" and model_scores is not None:
            # Combine weights and scores (weighted average)
            normalized_weights = weights / weights.max() if weights.max() > 0 else weights
            normalized_scores = (
                model_scores / model_scores.max() if model_scores.max() > 0 else model_scores
            )
            ranking_values = 0.7 * normalized_weights + 0.3 * normalized_scores
        else:
            # Fallback to weight-based ranking
            ranking_values = weights

        # Select top-k positions
        if len(ranking_values) > self.constraints.top_k_positions:
            top_k_assets = ranking_values.nlargest(self.constraints.top_k_positions).index
            weights = weights.reindex(top_k_assets, fill_value=0.0)

        return weights

    def _apply_max_weight_constraint(
        self, weights: pd.Series, violations: list[ConstraintViolation]
    ) -> pd.Series:
        """Apply maximum position weight constraint with redistribution."""
        max_weight = self.constraints.max_position_weight

        # Only apply constraint if there are actual violations
        if not (weights > max_weight).any():
            return weights

        # Iterative adjustment to handle cascading violations
        max_iters = 10
        for _iteration in range(max_iters):
            violating_mask = weights > max_weight
            if not violating_mask.any():
                break

            # Calculate excess weight to redistribute
            excess_weight = (weights[violating_mask] - max_weight).sum()
            weights[violating_mask] = max_weight

            # Redistribute excess to non-violating assets
            non_violating_mask = ~violating_mask
            if non_violating_mask.any():
                # Calculate available capacity for each asset
                available_capacity = (max_weight - weights[non_violating_mask]).clip(lower=0)
                total_capacity = available_capacity.sum()

                if total_capacity > 0 and excess_weight > 0:
                    # Distribute proportionally based on available capacity
                    redistribution = excess_weight * (available_capacity / total_capacity)
                    weights[non_violating_mask] += redistribution
                else:
                    # No capacity available - stop iteration
                    break
            else:
                # All assets are at max weight - stop iteration
                break

        return weights

    def _apply_turnover_constraint(
        self,
        weights: pd.Series,
        previous_weights: pd.Series | None,
        violations: list[ConstraintViolation],
    ) -> pd.Series:
        """Apply turnover constraint by blending with previous weights."""
        if previous_weights is None or not self.constraints.enable_turnover_penalty:
            return weights

        turnover = self.calculate_turnover(weights, previous_weights)

        if turnover > self.constraints.max_monthly_turnover:
            # Calculate correct blending factor to achieve target turnover
            # If target_turnover = α * current_turnover, then α = target/current
            target_turnover = self.constraints.max_monthly_turnover

            # Align weights to common universe
            all_assets = weights.index.union(previous_weights.index)
            current_aligned = weights.reindex(all_assets, fill_value=0.0)
            previous_aligned = previous_weights.reindex(all_assets, fill_value=0.0)

            # Calculate the scaling factor needed
            # If we move α fraction towards new weights: (1-α)*prev + α*new
            # The turnover will be α * original_turnover
            alpha = target_turnover / turnover if turnover > 0 else 0.0
            alpha = min(alpha, 1.0)  # Cap at 1.0

            # Apply constrained blending
            blended_weights = (1 - alpha) * previous_aligned + alpha * current_aligned

            # Only keep assets that were in the original weights (preserve other constraints)
            weights = blended_weights.reindex(weights.index, fill_value=0.0)

            # Ensure weights sum to 1.0 to prevent subsequent normalization from undoing turnover constraint
            weight_sum = weights.sum()
            if weight_sum > 0:
                weights = weights / weight_sum

        return weights

    def _normalize_weights(self, weights: pd.Series) -> pd.Series:
        """
        Normalize weights to sum to 1.0 while respecting max position weight constraint.

        If equal normalization would violate max position constraint, redistribute
        excess to maintain constraint compliance.
        """
        weight_sum = weights.sum()
        if weight_sum > 0:
            normalized = weights / weight_sum

            # Check if normalization violates max position weight
            max_weight = self.constraints.max_position_weight
            if (normalized > max_weight).any():
                # Cap weights at max and redistribute excess
                capped_weights = normalized.copy()

                # Iteratively cap and redistribute
                for _ in range(10):  # Prevent infinite loops
                    violating_mask = capped_weights > max_weight
                    if not violating_mask.any():
                        break

                    # Calculate excess from violating positions
                    excess = (capped_weights[violating_mask] - max_weight).sum()
                    capped_weights[violating_mask] = max_weight

                    # Redistribute to non-violating positions with capacity
                    non_violating_mask = ~violating_mask
                    if non_violating_mask.any():
                        available_capacity = (max_weight - capped_weights[non_violating_mask]).clip(lower=0)
                        total_capacity = available_capacity.sum()

                        if total_capacity > 0:
                            redistribution = excess * (available_capacity / total_capacity)
                            capped_weights[non_violating_mask] += redistribution
                        else:
                            # No capacity - accept sub-unity sum
                            break
                    else:
                        # All positions at max - accept sub-unity sum
                        break

                return capped_weights
            else:
                return normalized
        else:
            # Equal weights fallback - respect max position constraint
            if len(weights) > 0:
                equal_weight = 1.0 / len(weights)
                max_weight = self.constraints.max_position_weight

                if equal_weight <= max_weight:
                    return pd.Series(equal_weight, index=weights.index)
                else:
                    # Equal weights would violate constraint - use max weight
                    return pd.Series(max_weight, index=weights.index)
            else:
                return weights

    def calculate_turnover(self, current_weights: pd.Series, previous_weights: pd.Series) -> float:
        """Calculate one-way portfolio turnover."""
        # Align weights to common index
        all_assets = current_weights.index.union(previous_weights.index)
        current_aligned = current_weights.reindex(all_assets, fill_value=0.0)
        previous_aligned = previous_weights.reindex(all_assets, fill_value=0.0)

        # Calculate one-way turnover
        return np.abs(current_aligned - previous_aligned).sum()

    def _log_violations(self, date: pd.Timestamp, violations: list[ConstraintViolation]) -> None:
        """Log constraint violations with timestamps."""
        for violation in violations:
            violation_record = {
                "timestamp": date,
                "violation_type": violation.violation_type.value,
                "severity": violation.severity.value,
                "description": violation.description,
                "violation_value": violation.violation_value,
                "constraint_value": violation.constraint_value,
                "remediation_action": violation.remediation_action,
            }
            self.violation_log.append(violation_record)

    def get_violation_summary(self) -> dict[str, Any]:
        """Get summary of all recorded constraint violations."""
        if not self.violation_log:
            return {"total_violations": 0, "by_type": {}, "by_severity": {}}

        # Count violations by type
        by_type: dict[str, int] = {}
        by_severity: dict[str, int] = {}

        for record in self.violation_log:
            violation_type = record["violation_type"]
            severity = record["severity"]

            by_type[violation_type] = by_type.get(violation_type, 0) + 1
            by_severity[severity] = by_severity.get(severity, 0) + 1

        return {
            "total_violations": len(self.violation_log),
            "by_type": by_type,
            "by_severity": by_severity,
            "recent_violations": self.violation_log[-10:] if self.violation_log else [],
        }

    def calculate_constraint_metrics(
        self, weights: pd.Series, previous_weights: pd.Series | None = None
    ) -> dict[str, Any]:
        """
        Calculate comprehensive constraint adherence metrics.

        Args:
            weights: Current portfolio weights
            previous_weights: Previous period weights

        Returns:
            Dictionary of constraint metrics for monitoring and reporting
        """
        metrics = {}

        # Basic weight metrics
        metrics["max_weight"] = weights.max() if len(weights) > 0 else 0.0
        metrics["min_weight"] = weights[weights > 0].min() if (weights > 0).any() else 0.0
        metrics["num_positions"] = (weights > 0).sum()
        metrics["weight_concentration"] = (weights**2).sum()  # Herfindahl index
        metrics["effective_positions"] = (
            1.0 / metrics["weight_concentration"] if metrics["weight_concentration"] > 0 else 0.0
        )

        # Constraint adherence metrics
        metrics["long_only_compliant"] = not (weights < 0).any()
        metrics["max_weight_violations"] = (weights > self.constraints.max_position_weight).sum()

        if self.constraints.top_k_positions is not None:
            metrics["top_k_compliant"] = (
                metrics["num_positions"] <= self.constraints.top_k_positions
            )
        else:
            metrics["top_k_compliant"] = True

        # Turnover metrics
        if previous_weights is not None:
            turnover = self.calculate_turnover(weights, previous_weights)
            metrics["turnover"] = turnover
            metrics["turnover_compliant"] = turnover <= self.constraints.max_monthly_turnover
        else:
            metrics["turnover"] = 0.0
            metrics["turnover_compliant"] = True

        # Weight distribution metrics
        metrics["weight_sum"] = weights.sum()
        metrics["weight_std"] = weights.std()
        metrics["weight_gini"] = self._calculate_gini_coefficient(weights)

        return metrics

    def _calculate_gini_coefficient(self, weights: pd.Series) -> float:
        """Calculate Gini coefficient for weight distribution inequality."""
        if len(weights) <= 1:
            return 0.0

        # Sort weights in ascending order
        sorted_weights = np.sort(weights.values)
        n = len(sorted_weights)

        # Calculate Gini coefficient
        cumsum = np.cumsum(sorted_weights)
        gini = (2 * np.sum(np.arange(1, n + 1) * sorted_weights)) / (n * cumsum[-1]) - (n + 1) / n

        return max(0.0, gini)  # Ensure non-negative

    def create_constraint_report(self) -> dict[str, Any]:
        """Create comprehensive constraint system report."""
        return {
            "constraint_config": {
                "long_only": self.constraints.long_only,
                "top_k_positions": self.constraints.top_k_positions,
                "max_position_weight": self.constraints.max_position_weight,
                "max_monthly_turnover": self.constraints.max_monthly_turnover,
                "transaction_cost_bps": self.constraints.transaction_cost_bps,
                "violation_handling": self.constraints.violation_handling,
                "position_ranking_method": self.constraints.position_ranking_method,
            },
            "violation_summary": self.get_violation_summary(),
            "enforcement_statistics": {
                "total_enforcements": len(self.violation_log),
                "successful_remediations": len(
                    [v for v in self.violation_log if "remediation_action" in v]
                ),
            },
        }
