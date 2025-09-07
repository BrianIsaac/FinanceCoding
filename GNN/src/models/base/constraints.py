"""
Portfolio constraint system for unified risk management.

This module provides constraint enforcement and validation utilities
that ensure all portfolio models comply with risk management requirements
and regulatory constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class TurnoverConstraints:
    """Configuration for turnover-based constraints."""

    max_monthly_turnover: float = 0.20
    transaction_cost_bps: float = 10.0
    enable_turnover_penalty: bool = True


@dataclass
class RiskConstraints:
    """Configuration for risk-based constraints."""

    max_portfolio_volatility: Optional[float] = None
    max_asset_correlation: Optional[float] = None
    min_diversification_ratio: Optional[float] = None


@dataclass
class RegulatoryConstraints:
    """Configuration for regulatory compliance constraints."""

    max_sector_concentration: Optional[float] = None
    max_single_issuer_weight: float = 0.10
    min_liquidity_threshold: Optional[float] = None


class ConstraintEngine:
    """
    Engine for enforcing portfolio constraints across all models.

    This class provides centralized constraint enforcement to ensure
    consistent risk management across HRP, LSTM, GAT, and baseline models.
    """

    def __init__(
        self,
        turnover_constraints: Optional[TurnoverConstraints] = None,
        risk_constraints: Optional[RiskConstraints] = None,
        regulatory_constraints: Optional[RegulatoryConstraints] = None,
    ):
        """
        Initialize constraint engine.

        Args:
            turnover_constraints: Turnover and transaction cost constraints
            risk_constraints: Risk-based portfolio constraints
            regulatory_constraints: Regulatory compliance constraints
        """
        self.turnover_constraints = turnover_constraints or TurnoverConstraints()
        self.risk_constraints = risk_constraints or RiskConstraints()
        self.regulatory_constraints = regulatory_constraints or RegulatoryConstraints()

    def apply_constraints(
        self,
        weights: pd.Series,
        previous_weights: Optional[pd.Series] = None,
        returns_data: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Apply all constraints to portfolio weights.

        Args:
            weights: Raw portfolio weights
            previous_weights: Previous period weights for turnover calculation
            returns_data: Historical returns data for risk calculations

        Returns:
            Constrained portfolio weights
        """
        # Apply basic weight constraints
        constrained_weights = self._apply_basic_constraints(weights)

        # Apply turnover constraints if previous weights available
        if previous_weights is not None:
            constrained_weights = self._apply_turnover_constraints(
                constrained_weights, previous_weights
            )

        # Apply risk constraints if returns data available
        if returns_data is not None:
            constrained_weights = self._apply_risk_constraints(constrained_weights, returns_data)

        return constrained_weights

    def _apply_basic_constraints(self, weights: pd.Series) -> pd.Series:
        """Apply basic weight and regulatory constraints."""
        # Handle empty weights edge case
        if len(weights) == 0:
            return weights
            
        # Ensure non-negative weights (long-only)
        weights = weights.clip(lower=0.0)

        # Apply maximum single issuer weight
        max_weight = self.regulatory_constraints.max_single_issuer_weight
        if max_weight < 1.0:
            weights = weights.clip(upper=max_weight)

        # Normalize weights to sum to 1.0
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            # Equal weights fallback
            weights = pd.Series(1.0 / len(weights), index=weights.index)

        return weights

    def _apply_turnover_constraints(
        self, weights: pd.Series, previous_weights: pd.Series
    ) -> pd.Series:
        """Apply turnover-based constraints."""
        if not self.turnover_constraints.enable_turnover_penalty:
            return weights

        # Calculate turnover
        common_assets = weights.index.intersection(previous_weights.index)
        if len(common_assets) == 0:
            return weights

        # Align weights for common assets
        current_aligned = weights.reindex(common_assets, fill_value=0.0)
        previous_aligned = previous_weights.reindex(common_assets, fill_value=0.0)

        # Calculate one-way turnover
        turnover = np.abs(current_aligned - previous_aligned).sum()

        # If turnover exceeds limit, blend with previous weights
        max_turnover = self.turnover_constraints.max_monthly_turnover
        if turnover > max_turnover:
            # Calculate blending factor to achieve target turnover
            blend_factor = max_turnover / turnover

            # Blend weights
            blended_weights = blend_factor * current_aligned + (1 - blend_factor) * previous_aligned

            # Update weights for common assets
            weights.update(blended_weights)

        return weights

    def _apply_risk_constraints(self, weights: pd.Series, returns_data: pd.DataFrame) -> pd.Series:
        """Apply risk-based constraints."""
        # This is a stub - risk constraints will be implemented in future stories
        # For now, return weights unchanged
        return weights

    def calculate_constraint_metrics(
        self, weights: pd.Series, previous_weights: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Calculate constraint adherence metrics.

        Args:
            weights: Current portfolio weights
            previous_weights: Previous period weights

        Returns:
            Dictionary of constraint metrics for monitoring
        """
        metrics = {}

        # Basic weight metrics
        metrics["max_weight"] = weights.max()
        metrics["min_weight"] = weights[weights > 0].min() if (weights > 0).any() else 0.0
        metrics["num_positions"] = (weights > 0).sum()
        metrics["weight_concentration"] = (weights**2).sum()  # Herfindahl index

        # Turnover metrics
        if previous_weights is not None:
            common_assets = weights.index.intersection(previous_weights.index)
            if len(common_assets) > 0:
                current_aligned = weights.reindex(common_assets, fill_value=0.0)
                previous_aligned = previous_weights.reindex(common_assets, fill_value=0.0)
                metrics["turnover"] = np.abs(current_aligned - previous_aligned).sum()
            else:
                metrics["turnover"] = 1.0  # Complete turnover for new universe

        return metrics
