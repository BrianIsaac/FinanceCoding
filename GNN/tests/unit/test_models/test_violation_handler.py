"""
Comprehensive tests for ViolationHandler class to improve code coverage.
"""

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

    def test_initialization_defaults(self):
        """Test handler initialization with defaults."""
        handler = ViolationHandler()

        assert handler.default_strategy == RemediationStrategy.ADJUST
        assert handler.enable_fallbacks is True
        assert len(handler.severity_strategies) == 4
        assert handler.severity_strategies[ViolationSeverity.CRITICAL] == RemediationStrategy.REJECT
        assert handler.remediation_history == []

    def test_initialization_custom(self):
        """Test handler initialization with custom parameters."""
        custom_strategies = {
            ViolationSeverity.WARNING: RemediationStrategy.IGNORE,
            ViolationSeverity.CRITICAL: RemediationStrategy.WARN,
        }

        handler = ViolationHandler(
            default_strategy=RemediationStrategy.WARN,
            severity_strategies=custom_strategies,
            enable_fallbacks=False,
        )

        assert handler.default_strategy == RemediationStrategy.WARN
        assert handler.enable_fallbacks is False
        assert handler.severity_strategies == custom_strategies

    def test_handle_no_violations(self):
        """Test handling empty violations list."""
        handler = ViolationHandler()
        weights = pd.Series([0.4, 0.6], index=['A', 'B'])

        result_weights, actions = handler.handle_violations([], weights)

        assert result_weights.equals(weights)
        assert actions == []

    def test_reject_strategy_with_fallback(self):
        """Test REJECT strategy with fallback enabled."""
        handler = ViolationHandler(enable_fallbacks=True)
        weights = pd.Series([0.8, 0.2], index=['A', 'B'])

        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.MAX_POSITION_WEIGHT,
            severity=ViolationSeverity.CRITICAL,
            description="Critical violation",
            constraint_value=0.5,
            violation_value=0.8,
            remediation_action="reject"
        )

        result_weights, actions = handler.handle_violations([violation], weights)

        # Should return equal weight fallback
        expected_weights = pd.Series([0.5, 0.5], index=['A', 'B'])
        assert result_weights.equals(expected_weights)
        assert len(actions) == 1
        assert actions[0].strategy == RemediationStrategy.REJECT
        assert actions[0].success is True
        assert actions[0].fallback_used is True

    def test_reject_strategy_without_fallback(self):
        """Test REJECT strategy with fallback disabled."""
        handler = ViolationHandler(enable_fallbacks=False)
        weights = pd.Series([0.8, 0.2], index=['A', 'B'])

        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.MAX_POSITION_WEIGHT,
            severity=ViolationSeverity.CRITICAL,
            description="Critical violation",
            constraint_value=0.5,
            violation_value=0.8,
            remediation_action="reject"
        )

        result_weights, actions = handler.handle_violations([violation], weights)

        # Should return original weights with failed action
        assert result_weights.equals(weights)
        assert len(actions) == 1
        assert actions[0].strategy == RemediationStrategy.REJECT
        assert actions[0].success is False
        assert actions[0].fallback_used is False

    def test_warn_strategy(self):
        """Test WARN strategy handling."""
        handler = ViolationHandler()
        weights = pd.Series([0.4, 0.6], index=['A', 'B'])

        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.MAX_POSITION_WEIGHT,
            severity=ViolationSeverity.WARNING,
            description="Warning violation",
            constraint_value=0.5,
            violation_value=0.6,
            remediation_action="warn"
        )

        result_weights, actions = handler.handle_violations([violation], weights)

        assert result_weights.equals(weights)
        assert len(actions) == 1
        assert actions[0].strategy == RemediationStrategy.WARN
        assert actions[0].success is True
        assert "Warning:" in actions[0].description

    def test_ignore_strategy(self):
        """Test IGNORE strategy handling."""
        custom_strategies = {ViolationSeverity.WARNING: RemediationStrategy.IGNORE}
        handler = ViolationHandler(severity_strategies=custom_strategies)
        weights = pd.Series([0.4, 0.6], index=['A', 'B'])

        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.MAX_POSITION_WEIGHT,
            severity=ViolationSeverity.WARNING,
            description="Ignored violation",
            constraint_value=0.5,
            violation_value=0.6,
            remediation_action="warn"
        )

        result_weights, actions = handler.handle_violations([violation], weights)

        assert result_weights.equals(weights)
        assert len(actions) == 1
        assert actions[0].strategy == RemediationStrategy.IGNORE
        assert actions[0].success is True
        assert "Ignored:" in actions[0].description

    def test_long_only_violation_adjustment(self):
        """Test long-only constraint violation adjustment."""
        handler = ViolationHandler()
        weights = pd.Series([0.7, -0.2, 0.5], index=['A', 'B', 'C'])

        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.LONG_ONLY,
            severity=ViolationSeverity.MINOR,
            description="Negative weights detected",
            constraint_value=0.0,
            violation_value=-0.2,
            remediation_action="warn"
        )

        result_weights, actions = handler.handle_violations([violation], weights)

        # Should clip negative weights and renormalize
        assert (result_weights >= 0).all()
        assert abs(result_weights.sum() - 1.0) < 1e-10
        assert result_weights['B'] == 0.0
        assert len(actions) == 1
        assert actions[0].strategy == RemediationStrategy.ADJUST
        assert actions[0].success is True

    def test_top_k_violation_adjustment(self):
        """Test top-k positions violation adjustment."""
        handler = ViolationHandler()
        weights = pd.Series([0.3, 0.25, 0.2, 0.15, 0.1], index=['A', 'B', 'C', 'D', 'E'])

        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.TOP_K_POSITIONS,
            severity=ViolationSeverity.MINOR,
            description="Too many positions",
            constraint_value=3,
            violation_value=5,
            remediation_action="adjust"
        )

        result_weights, actions = handler.handle_violations([violation], weights)

        # Should keep only top 3 positions
        assert (result_weights > 0).sum() == 3
        assert abs(result_weights.sum() - 1.0) < 1e-10
        assert result_weights[['D', 'E']].sum() == 0.0  # Smallest weights should be zero
        assert len(actions) == 1
        assert actions[0].success is True

    def test_top_k_no_adjustment_needed(self):
        """Test top-k when no adjustment is needed."""
        handler = ViolationHandler()
        weights = pd.Series([0.5, 0.5], index=['A', 'B'])

        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.TOP_K_POSITIONS,
            severity=ViolationSeverity.MINOR,
            description="Position count ok",
            constraint_value=5,
            violation_value=2,
            remediation_action="adjust"
        )

        result_weights, actions = handler.handle_violations([violation], weights)

        assert result_weights.equals(weights)
        assert len(actions) == 1
        assert "No adjustment needed" in actions[0].description

    def test_max_weight_violation_adjustment(self):
        """Test maximum weight violation adjustment."""
        handler = ViolationHandler()
        weights = pd.Series([0.6, 0.4], index=['A', 'B'])

        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.MAX_POSITION_WEIGHT,
            severity=ViolationSeverity.MINOR,
            description="Weight too high",
            constraint_value=0.5,
            violation_value=0.6,
            remediation_action="reject"
        )

        result_weights, actions = handler.handle_violations([violation], weights)

        # Should cap A at 0.5 and redistribute excess to B
        assert result_weights['A'] == 0.5
        assert result_weights['B'] > 0.4  # Should receive redistributed weight
        assert abs(result_weights.sum() - 1.0) < 1e-10
        assert len(actions) == 1
        assert actions[0].success is True

    def test_max_weight_no_violations(self):
        """Test max weight when no violations exist."""
        handler = ViolationHandler()
        weights = pd.Series([0.4, 0.6], index=['A', 'B'])

        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.MAX_POSITION_WEIGHT,
            severity=ViolationSeverity.MINOR,
            description="No violations",
            constraint_value=0.7,
            violation_value=0.6,
            remediation_action="adjust"
        )

        result_weights, actions = handler.handle_violations([violation], weights)

        assert result_weights.equals(weights)
        assert len(actions) == 1
        assert "No max weight violations" in actions[0].description

    def test_turnover_violation_adjustment(self):
        """Test turnover violation adjustment."""
        handler = ViolationHandler()
        current_weights = pd.Series([0.8, 0.2], index=['A', 'B'])
        previous_weights = pd.Series([0.2, 0.8], index=['A', 'B'])

        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.MONTHLY_TURNOVER,
            severity=ViolationSeverity.MINOR,
            description="High turnover",
            constraint_value=0.3,
            violation_value=1.2,  # High turnover
            remediation_action="adjust"
        )

        result_weights, actions = handler.handle_violations(
            [violation], current_weights, previous_weights
        )

        # Should blend current and previous weights
        assert result_weights['A'] < current_weights['A']
        assert result_weights['B'] > current_weights['B']
        assert abs(result_weights.sum() - 1.0) < 1e-10
        assert len(actions) == 1
        assert "Blended with previous weights" in actions[0].description

    def test_turnover_without_previous_weights(self):
        """Test turnover violation without previous weights."""
        handler = ViolationHandler()
        weights = pd.Series([0.8, 0.2], index=['A', 'B'])

        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.MONTHLY_TURNOVER,
            severity=ViolationSeverity.MINOR,
            description="High turnover",
            constraint_value=0.3,
            violation_value=1.2,
            remediation_action="adjust"
        )

        result_weights, actions = handler.handle_violations([violation], weights, None)

        assert result_weights.equals(weights)
        assert len(actions) == 1
        assert actions[0].success is False
        assert "Cannot adjust turnover" in actions[0].description

    def test_generic_adjustment(self):
        """Test generic adjustment for unknown violation type."""
        handler = ViolationHandler()
        weights = pd.Series([0.3, 0.7], index=['A', 'B'])  # Sums to 1.0

        # Create violation with type that triggers generic adjustment
        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.TRANSACTION_COST,  # Not handled specifically
            severity=ViolationSeverity.MINOR,
            description="Generic violation",
            constraint_value=1.0,
            violation_value=1.5,
            remediation_action="adjust"
        )

        result_weights, actions = handler.handle_violations([violation], weights)

        # Should normalize weights (which are already normalized)
        assert result_weights.equals(weights)
        assert len(actions) == 1
        assert "generic normalization" in actions[0].description

    def test_generic_adjustment_zero_sum(self):
        """Test generic adjustment with zero sum weights."""
        handler = ViolationHandler()
        weights = pd.Series([0.0, 0.0], index=['A', 'B'])

        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.TRANSACTION_COST,
            severity=ViolationSeverity.MINOR,
            description="Zero sum violation",
            constraint_value=1.0,
            violation_value=0.0,
            remediation_action="adjust"
        )

        result_weights, actions = handler.handle_violations([violation], weights)

        # Should create equal weights
        expected_weights = pd.Series([0.5, 0.5], index=['A', 'B'])
        assert result_weights.equals(expected_weights)
        assert len(actions) == 1
        assert actions[0].success is True

    def test_create_fallback_portfolio(self):
        """Test fallback portfolio creation."""
        handler = ViolationHandler()
        weights = pd.Series([0.8, 0.2], index=['A', 'B'])

        fallback = handler._create_fallback_portfolio(weights)

        expected = pd.Series([0.5, 0.5], index=['A', 'B'])
        assert fallback.equals(expected)

    def test_create_fallback_empty_portfolio(self):
        """Test fallback portfolio with empty weights."""
        handler = ViolationHandler()
        weights = pd.Series([], dtype=float)

        fallback = handler._create_fallback_portfolio(weights)

        assert fallback.equals(weights)

    def test_multiple_violations_sorted_by_severity(self):
        """Test handling multiple violations sorted by severity."""
        handler = ViolationHandler()
        weights = pd.Series([0.6, 0.4], index=['A', 'B'])

        violations = [
            ConstraintViolation(
                violation_type=ConstraintViolationType.MAX_POSITION_WEIGHT,
                severity=ViolationSeverity.WARNING,  # Lower priority
                description="Warning violation",
                constraint_value=0.5,
                violation_value=0.6,
                remediation_action="reject"
            ),
            ConstraintViolation(
                violation_type=ConstraintViolationType.LONG_ONLY,
                severity=ViolationSeverity.CRITICAL,  # Higher priority
                description="Critical violation",
                constraint_value=0.0,
                violation_value=-0.1,
                remediation_action="warn"
            )
        ]

        result_weights, actions = handler.handle_violations(violations, weights)

        # Should be rejected due to critical violation and return fallback
        expected_weights = pd.Series([0.5, 0.5], index=['A', 'B'])
        assert result_weights.equals(expected_weights)
        assert len(actions) == 1  # Only rejection action, warning skipped
        assert actions[0].strategy == RemediationStrategy.REJECT

    def test_remediation_statistics_empty(self):
        """Test statistics with no remediation history."""
        handler = ViolationHandler()

        stats = handler.get_remediation_statistics()

        assert stats == {"total_remediations": 0}

    def test_remediation_statistics_with_history(self):
        """Test statistics calculation with remediation history."""
        handler = ViolationHandler()
        weights = pd.Series([0.4, 0.6], index=['A', 'B'])

        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.MAX_POSITION_WEIGHT,
            severity=ViolationSeverity.WARNING,
            description="Test violation",
            constraint_value=0.5,
            violation_value=0.6,
            remediation_action="warn"
        )

        # Process violation to create history
        handler.handle_violations([violation], weights)

        stats = handler.get_remediation_statistics()

        assert stats["total_remediations"] == 1
        assert stats["total_violations_handled"] == 1
        assert stats["total_actions_taken"] == 1
        assert stats["success_rate"] == 1.0
        assert stats["fallback_rate"] == 0.0
        assert "warn" in stats["strategy_usage"]
        assert "max_position_weight" in stats["violation_type_distribution"]

    def test_create_remediation_report(self):
        """Test comprehensive remediation report creation."""
        handler = ViolationHandler()

        report = handler.create_remediation_report()

        assert "handler_configuration" in report
        assert "statistics" in report
        assert "recent_history" in report
        assert report["handler_configuration"]["default_strategy"] == "adjust"
        assert report["handler_configuration"]["fallbacks_enabled"] is True

    def test_strategy_selection_fallback_to_default(self):
        """Test strategy selection falls back to default for unknown severity."""
        custom_strategies = {ViolationSeverity.WARNING: RemediationStrategy.IGNORE}
        handler = ViolationHandler(
            default_strategy=RemediationStrategy.WARN,
            severity_strategies=custom_strategies
        )

        # Create violation with severity not in custom_strategies
        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.MAX_POSITION_WEIGHT,
            severity=ViolationSeverity.CRITICAL,  # Not in custom_strategies
            description="Test violation",
            constraint_value=0.5,
            violation_value=0.6,
            remediation_action="reject"
        )

        strategy = handler._get_strategy_for_violation(violation)
        assert strategy == RemediationStrategy.WARN  # Should use default

    def test_max_weight_redistribution_edge_case(self):
        """Test max weight redistribution when no capacity available."""
        handler = ViolationHandler()
        # All weights at max limit
        weights = pd.Series([0.5, 0.5], index=['A', 'B'])

        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.MAX_POSITION_WEIGHT,
            severity=ViolationSeverity.MINOR,
            description="All at max",
            constraint_value=0.5,
            violation_value=0.5,
            remediation_action="adjust"
        )

        result_weights, actions = handler.handle_violations([violation], weights)

        # Should handle the edge case gracefully
        assert len(actions) == 1
        assert actions[0].success is True

    def test_turnover_zero_current_turnover(self):
        """Test turnover adjustment with zero current turnover."""
        handler = ViolationHandler()
        current_weights = pd.Series([0.5, 0.5], index=['A', 'B'])
        previous_weights = pd.Series([0.5, 0.5], index=['A', 'B'])

        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.MONTHLY_TURNOVER,
            severity=ViolationSeverity.MINOR,
            description="Zero turnover case",
            constraint_value=0.1,
            violation_value=0.0,  # Zero current turnover
            remediation_action="adjust"
        )

        result_weights, actions = handler.handle_violations(
            [violation], current_weights, previous_weights
        )

        # Should handle zero turnover case
        assert len(actions) == 1
        assert actions[0].success is True
        assert actions[0].adjustments_made["blend_factor"] == 1.0
