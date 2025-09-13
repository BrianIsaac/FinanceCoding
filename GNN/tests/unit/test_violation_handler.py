"""
Unit tests for constraint violation handler.

Tests validate violation handling strategies, remediation actions,
and fallback mechanisms across different scenarios.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.base.constraints import (
    ConstraintViolation,
    ConstraintViolationType,
    ViolationSeverity,
)
from src.models.base.violation_handler import (
    RemediationAction,
    RemediationStrategy,
    ViolationHandler,
)


class TestViolationHandler:
    """Test suite for ViolationHandler class."""

    @pytest.fixture
    def sample_weights(self):
        """Create sample portfolio weights for testing."""
        assets = [f"ASSET_{i:03d}" for i in range(5)]
        weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        return pd.Series(weights, index=assets)

    @pytest.fixture
    def previous_weights(self):
        """Create previous period weights for turnover testing."""
        assets = [f"ASSET_{i:03d}" for i in range(5)]
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        return pd.Series(weights, index=assets)

    @pytest.fixture
    def default_handler(self):
        """Create ViolationHandler with default settings."""
        return ViolationHandler()

    @pytest.fixture
    def strict_handler(self):
        """Create ViolationHandler with strict settings."""
        return ViolationHandler(
            default_strategy=RemediationStrategy.REJECT,
            severity_strategies={
                ViolationSeverity.WARNING: RemediationStrategy.WARN,
                ViolationSeverity.MINOR: RemediationStrategy.ADJUST,
                ViolationSeverity.MAJOR: RemediationStrategy.REJECT,
                ViolationSeverity.CRITICAL: RemediationStrategy.REJECT,
            },
            enable_fallbacks=True,
        )

    @pytest.fixture
    def no_fallback_handler(self):
        """Create ViolationHandler without fallbacks."""
        return ViolationHandler(enable_fallbacks=False)

    def test_handler_initialization(self):
        """Test proper violation handler initialization."""
        handler = ViolationHandler()

        assert handler.default_strategy == RemediationStrategy.ADJUST
        assert handler.enable_fallbacks is True
        assert len(handler.severity_strategies) == 4
        assert handler.remediation_history == []

    def test_handle_no_violations(self, default_handler, sample_weights):
        """Test handling when no violations present."""
        adjusted_weights, actions = default_handler.handle_violations(
            [], sample_weights
        )

        assert adjusted_weights.equals(sample_weights)
        assert actions == []

    def test_handle_long_only_violation(self, default_handler):
        """Test handling of long-only constraint violation."""
        # Create weights with negative values
        weights_with_negatives = pd.Series(
            [-0.1, 0.4, 0.3, 0.2, 0.2], index=[f"ASSET_{i:03d}" for i in range(5)]
        )

        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.LONG_ONLY,
            severity=ViolationSeverity.MAJOR,
            description="Negative weights detected",
            constraint_value=0.0,
            violation_value=-0.1,
            remediation_action="Clip negative weights",
        )

        adjusted_weights, actions = default_handler.handle_violations(
            [violation], weights_with_negatives
        )

        # All weights should be non-negative
        assert (adjusted_weights >= 0).all()
        # Should sum to 1.0
        assert abs(adjusted_weights.sum() - 1.0) < 1e-10
        # Should have one adjustment action
        assert len(actions) == 1
        assert actions[0].strategy == RemediationStrategy.ADJUST
        assert actions[0].success is True

    def test_handle_top_k_violation(self, default_handler, sample_weights):
        """Test handling of top-k positions violation."""
        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.TOP_K_POSITIONS,
            severity=ViolationSeverity.MINOR,
            description="Too many positions",
            constraint_value=3,
            violation_value=5,
            remediation_action="Select top K positions",
        )

        adjusted_weights, actions = default_handler.handle_violations(
            [violation], sample_weights
        )

        # Should have exactly 3 non-zero positions
        assert (adjusted_weights > 0).sum() == 3
        # Should sum to 1.0
        assert abs(adjusted_weights.sum() - 1.0) < 1e-10
        # Should have one adjustment action
        assert len(actions) == 1
        assert actions[0].strategy == RemediationStrategy.ADJUST
        assert actions[0].success is True

    def test_handle_max_weight_violation(self, default_handler, sample_weights):
        """Test handling of maximum weight violation."""
        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.MAX_POSITION_WEIGHT,
            severity=ViolationSeverity.MINOR,
            description="Position weight too high",
            constraint_value=0.25,
            violation_value=0.3,
            remediation_action="Cap and redistribute excess weight",
        )

        adjusted_weights, actions = default_handler.handle_violations(
            [violation], sample_weights
        )

        # No weight should exceed max constraint
        assert (adjusted_weights <= 0.25).all()
        # Should sum to 1.0
        assert abs(adjusted_weights.sum() - 1.0) < 1e-10
        # Should have one adjustment action
        assert len(actions) == 1
        assert actions[0].strategy == RemediationStrategy.ADJUST
        assert actions[0].success is True

    def test_handle_turnover_violation(self, default_handler, sample_weights, previous_weights):
        """Test handling of turnover violation."""
        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.MONTHLY_TURNOVER,
            severity=ViolationSeverity.MINOR,
            description="Turnover too high",
            constraint_value=0.2,
            violation_value=0.6,
            remediation_action="Blend with previous weights",
        )

        adjusted_weights, actions = default_handler.handle_violations(
            [violation], sample_weights, previous_weights
        )

        # Should sum to 1.0
        assert abs(adjusted_weights.sum() - 1.0) < 1e-10
        # Should have one adjustment action
        assert len(actions) == 1
        assert actions[0].strategy == RemediationStrategy.ADJUST
        assert actions[0].success is True

    def test_handle_turnover_violation_no_previous_weights(self, default_handler, sample_weights):
        """Test handling turnover violation without previous weights."""
        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.MONTHLY_TURNOVER,
            severity=ViolationSeverity.MINOR,
            description="Turnover too high",
            constraint_value=0.2,
            violation_value=0.6,
            remediation_action="Blend with previous weights",
        )

        adjusted_weights, actions = default_handler.handle_violations(
            [violation], sample_weights, None
        )

        # Should return original weights unchanged
        assert adjusted_weights.equals(sample_weights)
        # Should have one failed action
        assert len(actions) == 1
        assert actions[0].strategy == RemediationStrategy.ADJUST
        assert actions[0].success is False

    def test_handle_warn_strategy(self, default_handler, sample_weights):
        """Test warning strategy handling."""
        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.LONG_ONLY,
            severity=ViolationSeverity.WARNING,
            description="Minor issue",
            constraint_value=0.0,
            violation_value=-0.01,
            remediation_action="Generic action",
        )

        # Override strategy for this test
        default_handler.severity_strategies[ViolationSeverity.WARNING] = RemediationStrategy.WARN

        adjusted_weights, actions = default_handler.handle_violations(
            [violation], sample_weights
        )

        # Weights should be unchanged
        assert adjusted_weights.equals(sample_weights)
        # Should have one warning action
        assert len(actions) == 1
        assert actions[0].strategy == RemediationStrategy.WARN
        assert actions[0].success is True

    def test_handle_ignore_strategy(self, default_handler, sample_weights):
        """Test ignore strategy handling."""
        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.LONG_ONLY,
            severity=ViolationSeverity.WARNING,
            description="Minor issue",
            constraint_value=0.0,
            violation_value=-0.01,
            remediation_action="Generic action",
        )

        # Override strategy for this test
        default_handler.severity_strategies[ViolationSeverity.WARNING] = RemediationStrategy.IGNORE

        adjusted_weights, actions = default_handler.handle_violations(
            [violation], sample_weights
        )

        # Weights should be unchanged
        assert adjusted_weights.equals(sample_weights)
        # Should have one ignore action
        assert len(actions) == 1
        assert actions[0].strategy == RemediationStrategy.IGNORE
        assert actions[0].success is True

    def test_handle_reject_strategy_with_fallback(self, default_handler, sample_weights):
        """Test reject strategy with fallback enabled."""
        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.LONG_ONLY,
            severity=ViolationSeverity.CRITICAL,
            description="Critical violation",
            constraint_value=0.0,
            violation_value=-0.5,
            remediation_action="Generic action",
        )

        # Override strategy for this test
        default_handler.severity_strategies[ViolationSeverity.CRITICAL] = RemediationStrategy.REJECT

        adjusted_weights, actions = default_handler.handle_violations(
            [violation], sample_weights
        )

        # Should return equal weight fallback
        expected_weight = 1.0 / len(sample_weights)
        np.testing.assert_allclose(adjusted_weights.values, expected_weight, atol=1e-10)
        # Should have one reject action with fallback
        assert len(actions) == 1
        assert actions[0].strategy == RemediationStrategy.REJECT
        assert actions[0].success is True
        assert actions[0].fallback_used is True

    def test_handle_reject_strategy_no_fallback(self, no_fallback_handler, sample_weights):
        """Test reject strategy without fallback."""
        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.LONG_ONLY,
            severity=ViolationSeverity.CRITICAL,
            description="Critical violation",
            constraint_value=0.0,
            violation_value=-0.5,
            remediation_action="Generic action",
        )

        # Override strategy for this test
        no_fallback_handler.severity_strategies[ViolationSeverity.CRITICAL] = RemediationStrategy.REJECT

        adjusted_weights, actions = no_fallback_handler.handle_violations(
            [violation], sample_weights
        )

        # Should return original weights
        assert adjusted_weights.equals(sample_weights)
        # Should have one failed reject action
        assert len(actions) == 1
        assert actions[0].strategy == RemediationStrategy.REJECT
        assert actions[0].success is False
        assert actions[0].fallback_used is False

    def test_handle_multiple_violations_priority(self, default_handler, sample_weights):
        """Test handling multiple violations with priority order."""
        violations = [
            ConstraintViolation(
                violation_type=ConstraintViolationType.LONG_ONLY,
                severity=ViolationSeverity.MINOR,
                description="Minor violation",
                constraint_value=0.0,
                violation_value=-0.01,
                remediation_action="Generic action",
        ),
            ConstraintViolation(
                violation_type=ConstraintViolationType.MAX_POSITION_WEIGHT,
                severity=ViolationSeverity.CRITICAL,
                description="Critical violation",
                constraint_value=0.1,
                violation_value=0.3,
                remediation_action="Generic action",
        ),
        ]

        # Set critical to reject for test
        default_handler.severity_strategies[ViolationSeverity.CRITICAL] = RemediationStrategy.REJECT

        adjusted_weights, actions = default_handler.handle_violations(
            violations, sample_weights
        )

        # Should handle critical violation first and return fallback
        assert len(actions) == 1
        assert actions[0].strategy == RemediationStrategy.REJECT
        assert actions[0].fallback_used is True

    def test_generic_adjustment(self, default_handler):
        """Test generic adjustment for unknown violation types."""
        # Create a violation type that doesn't have specific handling
        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.WEIGHT_THRESHOLD,
            severity=ViolationSeverity.MINOR,
            description="Generic violation type",
            constraint_value=0.3,
            violation_value=0.5,
            remediation_action="Generic action",
        )

        weights = pd.Series([0.6, 0.4], index=["ASSET_A", "ASSET_B"])

        adjusted_weights, actions = default_handler.handle_violations(
            [violation], weights
        )

        # Should normalise weights
        assert abs(adjusted_weights.sum() - 1.0) < 1e-10
        assert len(actions) == 1
        assert actions[0].strategy == RemediationStrategy.ADJUST
        assert actions[0].success is True

    def test_top_k_adjustment_no_change_needed(self, default_handler):
        """Test top-k adjustment when no change is needed."""
        weights = pd.Series([0.5, 0.5], index=["ASSET_A", "ASSET_B"])

        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.TOP_K_POSITIONS,
            severity=ViolationSeverity.MINOR,
            description="Too many positions",
            constraint_value=5,  # More than current positions
            violation_value=2,
            remediation_action="Select top positions",
        )

        adjusted_weights, actions = default_handler.handle_violations(
            [violation], weights
        )

        # Should return unchanged weights
        assert adjusted_weights.equals(weights)
        assert len(actions) == 1
        assert actions[0].success is True

    def test_max_weight_adjustment_no_violations(self, default_handler):
        """Test max weight adjustment when no violations exist."""
        weights = pd.Series([0.2, 0.3, 0.5], index=["ASSET_A", "ASSET_B", "ASSET_C"])

        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.MAX_POSITION_WEIGHT,
            severity=ViolationSeverity.MINOR,
            description="Max weight exceeded",
            constraint_value=0.6,  # Higher than any weight
            violation_value=0.5,
            remediation_action="Check max weight constraint",
        )

        adjusted_weights, actions = default_handler.handle_violations(
            [violation], weights
        )

        # Should return unchanged weights
        assert adjusted_weights.equals(weights)
        assert len(actions) == 1
        assert actions[0].success is True

    def test_create_fallback_portfolio_empty(self, default_handler):
        """Test fallback portfolio creation with empty weights."""
        empty_weights = pd.Series([], dtype=float)

        fallback = default_handler._create_fallback_portfolio(empty_weights)

        assert len(fallback) == 0
        assert fallback.equals(empty_weights)

    def test_generic_adjustment_zero_sum(self, default_handler):
        """Test generic adjustment with zero sum weights."""
        weights = pd.Series([0.0, 0.0, 0.0], index=["ASSET_A", "ASSET_B", "ASSET_C"])

        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.WEIGHT_THRESHOLD,
            severity=ViolationSeverity.MINOR,
            description="Test violation",
            constraint_value=0.3,
            violation_value=0.5,
            remediation_action="Generic action",
        )

        adjusted_weights, actions = default_handler.handle_violations(
            [violation], weights
        )

        # Should create equal weight portfolio
        expected_weight = 1.0 / len(weights)
        np.testing.assert_allclose(adjusted_weights.values, expected_weight, atol=1e-10)
        assert len(actions) == 1
        assert actions[0].success is True

    def test_remediation_statistics_empty(self, default_handler):
        """Test statistics when no remediations have occurred."""
        stats = default_handler.get_remediation_statistics()

        assert stats["total_remediations"] == 0

    def test_remediation_statistics_with_history(self, default_handler, sample_weights):
        """Test statistics calculation with remediation history."""
        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.LONG_ONLY,
            severity=ViolationSeverity.MINOR,
            description="Test violation",
            constraint_value=0.0,
            violation_value=-0.1,
            remediation_action="Generic action",
        )

        # Process some violations to build history
        default_handler.handle_violations([violation], sample_weights)
        default_handler.handle_violations([violation], sample_weights)

        stats = default_handler.get_remediation_statistics()

        assert stats["total_remediations"] == 2
        assert stats["total_violations_handled"] == 2
        assert stats["total_actions_taken"] == 2
        assert stats["success_rate"] == 1.0
        assert stats["fallback_rate"] == 0.0
        assert "strategy_usage" in stats
        assert "violation_type_distribution" in stats

    def test_create_remediation_report(self, default_handler, sample_weights):
        """Test comprehensive remediation report creation."""
        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.LONG_ONLY,
            severity=ViolationSeverity.MINOR,
            description="Test violation",
            constraint_value=0.0,
            violation_value=-0.1,
            remediation_action="Generic action",
        )

        # Process violation to build history
        default_handler.handle_violations([violation], sample_weights)

        report = default_handler.create_remediation_report()

        assert "handler_configuration" in report
        assert "statistics" in report
        assert "recent_history" in report

        config = report["handler_configuration"]
        assert config["default_strategy"] == "adjust"
        assert config["fallbacks_enabled"] is True
        assert "severity_strategies" in config
