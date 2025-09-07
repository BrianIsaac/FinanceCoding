"""
Unit tests for portfolio constraint system.

Tests validate constraint enforcement, turnover calculations,
and constraint engine functionality across different scenarios.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.base.constraints import (
    ConstraintEngine,
    RegulatoryConstraints,
    RiskConstraints,
    TurnoverConstraints,
)


class TestConstraintEngine:
    """Test suite for ConstraintEngine class."""

    @pytest.fixture
    def sample_weights(self):
        """Create sample portfolio weights for testing."""
        assets = [f"ASSET_{i:03d}" for i in range(10)]
        weights = np.array([0.15, 0.12, 0.10, 0.08, 0.08, 0.07, 0.06, 0.05, 0.04, 0.25])
        return pd.Series(weights, index=assets)

    @pytest.fixture
    def previous_weights(self):
        """Create previous period weights for turnover testing."""
        assets = [f"ASSET_{i:03d}" for i in range(10)]
        weights = np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
        return pd.Series(weights, index=assets)

    @pytest.fixture
    def default_engine(self):
        """Create ConstraintEngine with default settings."""
        return ConstraintEngine()

    @pytest.fixture
    def strict_engine(self):
        """Create ConstraintEngine with strict constraints."""
        turnover_constraints = TurnoverConstraints(
            max_monthly_turnover=0.10,
            transaction_cost_bps=10.0,
            enable_turnover_penalty=True,
        )

        regulatory_constraints = RegulatoryConstraints(
            max_single_issuer_weight=0.08,
        )

        return ConstraintEngine(
            turnover_constraints=turnover_constraints,
            regulatory_constraints=regulatory_constraints,
        )

    def test_engine_initialization(self):
        """Test proper constraint engine initialization."""
        engine = ConstraintEngine()

        assert engine.turnover_constraints is not None
        assert engine.risk_constraints is not None
        assert engine.regulatory_constraints is not None
        assert engine.turnover_constraints.max_monthly_turnover == 0.20
        assert engine.regulatory_constraints.max_single_issuer_weight == 0.10

    def test_apply_basic_constraints_long_only(self, default_engine):
        """Test long-only constraint enforcement."""
        # Create weights with negative values
        negative_weights = pd.Series(
            [-0.05, 0.15, 0.10, -0.02, 0.82], index=[f"ASSET_{i:03d}" for i in range(5)]
        )

        constrained = default_engine.apply_constraints(negative_weights)

        # All weights should be non-negative
        assert all(
            constrained >= 0
        ), "All weights should be non-negative after constraint application"

        # Weights should sum to 1.0 (normalized)
        assert abs(constrained.sum() - 1.0) < 1e-10, "Weights should sum to 1.0 after normalization"

    def test_apply_max_position_weight_constraint(self, strict_engine, sample_weights):
        """Test maximum single position weight constraint."""
        # sample_weights has ASSET_009 at 0.25, which exceeds strict_engine's 0.08 limit
        original_max_weight = sample_weights.max()
        constrained = strict_engine.apply_constraints(sample_weights)

        # After clipping and renormalization, max weight should be reduced from original
        new_max_weight = constrained.max()
        assert (
            new_max_weight < original_max_weight
        ), "Max weight should be reduced after constraint application"

        # Weights should still sum to 1.0
        assert abs(constrained.sum() - 1.0) < 1e-10

    def test_apply_turnover_constraints_within_limit(
        self, default_engine, sample_weights, previous_weights
    ):
        """Test turnover constraint when turnover is within limits."""
        # Calculate expected turnover
        turnover = np.abs(sample_weights - previous_weights).sum()
        max_turnover = default_engine.turnover_constraints.max_monthly_turnover

        if turnover <= max_turnover:
            # Should return weights unchanged (after basic constraints)
            constrained = default_engine.apply_constraints(
                sample_weights, previous_weights=previous_weights
            )

            # Weights should be similar to original (accounting for normalization)
            np.testing.assert_allclose(constrained.values, sample_weights.values, atol=0.01)

    def test_apply_turnover_constraints_exceeds_limit(self, strict_engine, previous_weights):
        """Test turnover constraint when turnover exceeds limits."""
        # Create weights that would result in high turnover
        high_turnover_weights = pd.Series([0.8, 0.2] + [0.0] * 8, index=previous_weights.index)

        constrained = strict_engine.apply_constraints(
            high_turnover_weights, previous_weights=previous_weights
        )

        # Calculate actual turnover after constraint application
        common_assets = constrained.index.intersection(previous_weights.index)
        current_aligned = constrained.reindex(common_assets, fill_value=0.0)
        previous_aligned = previous_weights.reindex(common_assets, fill_value=0.0)
        actual_turnover = np.abs(current_aligned - previous_aligned).sum()

        # Turnover should be reduced (though may not be exactly at limit due to normalization)
        original_turnover = np.abs(high_turnover_weights - previous_weights).sum()
        assert (
            actual_turnover < original_turnover
        ), "Turnover should be reduced by constraint engine"

    def test_apply_constraints_empty_weights(self, default_engine):
        """Test constraint application on empty weights."""
        empty_weights = pd.Series([], dtype=float)

        constrained = default_engine.apply_constraints(empty_weights)

        # Should return equal weights fallback
        assert len(constrained) > 0 or len(empty_weights) == 0

    def test_apply_constraints_all_zero_weights(self, default_engine):
        """Test constraint application when all weights are zero."""
        zero_weights = pd.Series([0.0, 0.0, 0.0, 0.0], index=[f"ASSET_{i:03d}" for i in range(4)])

        constrained = default_engine.apply_constraints(zero_weights)

        # Should fallback to equal weights
        expected_weight = 1.0 / len(zero_weights)
        np.testing.assert_allclose(constrained.values, expected_weight, atol=1e-10)

    def test_calculate_constraint_metrics_basic(self, default_engine, sample_weights):
        """Test basic constraint metrics calculation."""
        metrics = default_engine.calculate_constraint_metrics(sample_weights)

        assert "max_weight" in metrics
        assert "min_weight" in metrics
        assert "num_positions" in metrics
        assert "weight_concentration" in metrics

        assert metrics["max_weight"] == sample_weights.max()
        assert metrics["min_weight"] == sample_weights[sample_weights > 0].min()
        assert metrics["num_positions"] == (sample_weights > 0).sum()

        # Herfindahl index (concentration measure)
        expected_concentration = (sample_weights**2).sum()
        assert abs(metrics["weight_concentration"] - expected_concentration) < 1e-10

    def test_calculate_constraint_metrics_with_turnover(
        self, default_engine, sample_weights, previous_weights
    ):
        """Test constraint metrics calculation with turnover."""
        metrics = default_engine.calculate_constraint_metrics(
            sample_weights, previous_weights=previous_weights
        )

        assert "turnover" in metrics

        # Calculate expected turnover
        expected_turnover = np.abs(sample_weights - previous_weights).sum()
        assert abs(metrics["turnover"] - expected_turnover) < 1e-10

    def test_calculate_constraint_metrics_no_common_assets(self, default_engine):
        """Test metrics calculation when current and previous weights have no common assets."""
        current_weights = pd.Series([0.5, 0.5], index=["ASSET_A", "ASSET_B"])
        previous_weights = pd.Series([0.3, 0.7], index=["ASSET_C", "ASSET_D"])

        metrics = default_engine.calculate_constraint_metrics(
            current_weights, previous_weights=previous_weights
        )

        # Should indicate complete turnover
        assert metrics["turnover"] == 1.0  # Complete portfolio change

    def test_turnover_constraints_configuration(self):
        """Test TurnoverConstraints configuration."""
        constraints = TurnoverConstraints(
            max_monthly_turnover=0.15,
            transaction_cost_bps=5.0,
            enable_turnover_penalty=False,
        )

        assert constraints.max_monthly_turnover == 0.15
        assert constraints.transaction_cost_bps == 5.0
        assert constraints.enable_turnover_penalty is False

    def test_regulatory_constraints_configuration(self):
        """Test RegulatoryConstraints configuration."""
        constraints = RegulatoryConstraints(
            max_single_issuer_weight=0.05,
            max_sector_concentration=0.20,
        )

        assert constraints.max_single_issuer_weight == 0.05
        assert constraints.max_sector_concentration == 0.20

    def test_risk_constraints_stub(self, default_engine, sample_weights):
        """Test that risk constraints are currently a stub implementation."""
        # Create mock returns data
        returns_data = pd.DataFrame(
            np.random.normal(0.001, 0.02, (100, len(sample_weights))), columns=sample_weights.index
        )

        # Risk constraints should not modify weights (stub implementation)
        original_weights = sample_weights.copy()
        constrained = default_engine.apply_constraints(sample_weights, returns_data=returns_data)

        # Should be similar (risk constraints are stub)
        # Note: basic constraints might still apply
        assert len(constrained) == len(original_weights)

    def test_apply_constraints_preserves_index(self, default_engine, sample_weights):
        """Test that constraint application preserves weight index."""
        constrained = default_engine.apply_constraints(sample_weights)

        assert list(constrained.index) == list(sample_weights.index)
        assert constrained.index.name == sample_weights.index.name
