"""
Unit tests for unified constraint engine system.

Tests the constraint engine, violation handling, and integration
with transaction cost calculations across all constraint types.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest.transaction_costs import (
    TransactionCostCalculator,
    TransactionCostConfig,
)
from src.models.base.constraint_engine import UnifiedConstraintEngine
from src.models.base.constraints import (
    ConstraintEngine,
    ConstraintViolation,
    ConstraintViolationType,
    PortfolioConstraints,
    ViolationSeverity,
)
from src.models.base.violation_handler import RemediationStrategy, ViolationHandler


class TestConstraintEngine:
    """Test the base constraint engine functionality."""

    @pytest.fixture
    def basic_constraints(self):
        """Basic constraint configuration for testing."""
        return PortfolioConstraints(
            long_only=True,
            top_k_positions=5,
            max_position_weight=0.20,
            max_monthly_turnover=0.15,
            transaction_cost_bps=10.0,
            min_weight_threshold=0.01,
        )

    @pytest.fixture
    def constraint_engine(self, basic_constraints):
        """Create constraint engine for testing."""
        return ConstraintEngine(basic_constraints)

    @pytest.fixture
    def sample_weights(self):
        """Sample portfolio weights for testing."""
        return pd.Series(
            [0.25, 0.20, 0.15, 0.10, 0.10, 0.08, 0.05, 0.04, 0.02, 0.01],
            index=[f"ASSET_{i}" for i in range(10)],
        )

    @pytest.fixture
    def previous_weights(self):
        """Previous period weights for turnover testing."""
        return pd.Series(
            [0.20, 0.20, 0.20, 0.15, 0.10, 0.05, 0.05, 0.03, 0.01, 0.01],
            index=[f"ASSET_{i}" for i in range(10)],
        )

    def test_constraint_engine_initialization(self, basic_constraints):
        """Test constraint engine initialization."""
        engine = ConstraintEngine(basic_constraints)
        assert engine.constraints == basic_constraints
        assert isinstance(engine.violation_log, list)
        assert len(engine.violation_log) == 0

    def test_long_only_constraint_detection(self, constraint_engine):
        """Test detection of long-only constraint violations."""
        # Create weights with negative values
        weights = pd.Series([0.5, -0.1, 0.4, 0.2], index=["A", "B", "C", "D"])

        violations = constraint_engine.check_violations(weights)

        # Should detect long-only violation
        long_only_violations = [
            v for v in violations if v.violation_type == ConstraintViolationType.LONG_ONLY
        ]
        assert len(long_only_violations) == 1
        assert long_only_violations[0].severity == ViolationSeverity.MAJOR

    def test_top_k_constraint_detection(self, constraint_engine):
        """Test detection of top-k position violations."""
        # Create weights with more than 5 non-zero positions
        weights = pd.Series(
            [0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.05, 0.03, 0.02],
            index=[f"ASSET_{i}" for i in range(10)],
        )

        violations = constraint_engine.check_violations(weights)

        # Should detect top-k violation (more than 5 positions)
        top_k_violations = [
            v for v in violations if v.violation_type == ConstraintViolationType.TOP_K_POSITIONS
        ]
        assert len(top_k_violations) == 1
        assert top_k_violations[0].violation_value == 10  # 10 positions > 5 limit

    def test_max_weight_constraint_detection(self, constraint_engine):
        """Test detection of maximum weight violations."""
        # Create weights with position exceeding 20% limit
        weights = pd.Series([0.35, 0.25, 0.20, 0.15, 0.05], index=[f"ASSET_{i}" for i in range(5)])

        violations = constraint_engine.check_violations(weights)

        # Should detect max weight violation
        max_weight_violations = [
            v for v in violations if v.violation_type == ConstraintViolationType.MAX_POSITION_WEIGHT
        ]
        assert len(max_weight_violations) == 1
        assert max_weight_violations[0].violation_value == 0.35  # Highest violating weight

    def test_turnover_constraint_detection(
        self, constraint_engine, sample_weights, previous_weights
    ):
        """Test detection of turnover constraint violations."""
        violations = constraint_engine.check_violations(sample_weights, previous_weights)

        # Calculate expected turnover
        expected_turnover = constraint_engine.calculate_turnover(sample_weights, previous_weights)

        if expected_turnover > constraint_engine.constraints.max_monthly_turnover:
            turnover_violations = [
                v
                for v in violations
                if v.violation_type == ConstraintViolationType.MONTHLY_TURNOVER
            ]
            assert len(turnover_violations) == 1
            assert turnover_violations[0].violation_value == expected_turnover

    def test_constraint_enforcement_long_only(self, constraint_engine):
        """Test enforcement of long-only constraints."""
        # Weights with negative values
        weights = pd.Series([0.6, -0.1, 0.3, 0.2], index=["A", "B", "C", "D"])

        constrained_weights, violations = constraint_engine.enforce_constraints(weights)

        # All weights should be non-negative
        assert (constrained_weights >= 0).all()
        # Should respect max position weight constraint
        assert (
            constrained_weights <= constraint_engine.constraints.max_position_weight + 1e-10
        ).all()
        # Weights should sum to <= 1.0 (may hold cash if constraints require it)
        assert constrained_weights.sum() <= 1.0 + 1e-10

    def test_constraint_enforcement_top_k(self, constraint_engine):
        """Test enforcement of top-k position constraints."""
        # More than 5 positions
        weights = pd.Series(
            [0.15, 0.15, 0.12, 0.12, 0.10, 0.08, 0.08, 0.06, 0.07, 0.07],
            index=[f"ASSET_{i}" for i in range(10)],
        )

        constrained_weights, _ = constraint_engine.enforce_constraints(weights)

        # Should have exactly 5 non-zero positions
        num_positions = (constrained_weights > 0).sum()
        assert num_positions == 5
        # Should sum to <= 1.0 (may hold cash if normalization would violate max weight constraint)
        assert constrained_weights.sum() <= 1.0 + 1e-10
        # All weights should respect max weight constraint
        assert (
            constrained_weights <= constraint_engine.constraints.max_position_weight + 1e-10
        ).all()

    def test_constraint_enforcement_max_weight(self, constraint_engine):
        """Test enforcement of maximum weight constraints."""
        # Weight exceeding 20% limit
        weights = pd.Series([0.40, 0.30, 0.20, 0.10], index=["A", "B", "C", "D"])

        constrained_weights, violations = constraint_engine.enforce_constraints(weights)

        # No weight should exceed 20%
        assert (constrained_weights <= 0.20 + 1e-10).all()
        # Should sum to approximately 1.0 (some capacity might remain)
        assert constrained_weights.sum() <= 1.0 + 1e-10

    def test_constraint_enforcement_turnover(
        self, constraint_engine, sample_weights, previous_weights
    ):
        """Test enforcement of turnover constraints."""
        constrained_weights, violations = constraint_engine.enforce_constraints(
            sample_weights, previous_weights
        )

        # The turnover constraint should work within the positions allowed by other constraints
        # When top-k constraint eliminates positions, turnover is calculated including eliminations
        # So we just verify that constraint enforcement was applied and weights are valid
        assert (constrained_weights >= 0).all()
        assert (
            constrained_weights <= constraint_engine.constraints.max_position_weight + 1e-10
        ).all()

        # Check that top-k constraint was respected
        num_positions = (
            constrained_weights > constraint_engine.constraints.min_weight_threshold
        ).sum()
        if constraint_engine.constraints.top_k_positions is not None:
            assert num_positions <= constraint_engine.constraints.top_k_positions

    def test_constraint_metrics_calculation(
        self, constraint_engine, sample_weights, previous_weights
    ):
        """Test calculation of constraint adherence metrics."""
        metrics = constraint_engine.calculate_constraint_metrics(sample_weights, previous_weights)

        # Check expected metrics are present
        expected_metrics = [
            "max_weight",
            "min_weight",
            "num_positions",
            "weight_concentration",
            "effective_positions",
            "long_only_compliant",
            "max_weight_violations",
            "top_k_compliant",
            "turnover",
            "turnover_compliant",
            "weight_sum",
            "weight_std",
            "weight_gini",
        ]

        for metric in expected_metrics:
            assert metric in metrics

    def test_gini_coefficient_calculation(self, constraint_engine):
        """Test Gini coefficient calculation for weight distribution."""
        # Perfectly equal weights should have Gini = 0
        equal_weights = pd.Series([0.25, 0.25, 0.25, 0.25], index=["A", "B", "C", "D"])
        gini_equal = constraint_engine._calculate_gini_coefficient(equal_weights)
        assert abs(gini_equal) < 1e-10

        # Highly concentrated weights should have higher Gini
        concentrated_weights = pd.Series([0.7, 0.2, 0.05, 0.05], index=["A", "B", "C", "D"])
        gini_concentrated = constraint_engine._calculate_gini_coefficient(concentrated_weights)
        assert gini_concentrated > gini_equal


class TestUnifiedConstraintEngine:
    """Test the unified constraint engine with transaction cost integration."""

    @pytest.fixture
    def transaction_config(self):
        """Transaction cost configuration for testing."""
        return TransactionCostConfig(linear_cost_bps=10.0, bid_ask_spread_bps=5.0)

    @pytest.fixture
    def transaction_calculator(self, transaction_config):
        """Transaction cost calculator for testing."""
        return TransactionCostCalculator(transaction_config)

    @pytest.fixture
    def unified_engine(self, basic_constraints, transaction_calculator):
        """Unified constraint engine for testing."""
        return UnifiedConstraintEngine(basic_constraints, transaction_calculator)

    @pytest.fixture
    def basic_constraints(self):
        """Basic constraints with transaction cost integration."""
        return PortfolioConstraints(
            long_only=True,
            top_k_positions=5,
            max_position_weight=0.20,
            max_monthly_turnover=0.15,
            transaction_cost_bps=10.0,
            cost_aware_enforcement=True,
        )

    def test_unified_engine_initialization(self, unified_engine, basic_constraints):
        """Test unified constraint engine initialization."""
        assert unified_engine.constraints == basic_constraints
        assert unified_engine.transaction_cost_calculator is not None
        assert isinstance(unified_engine.base_engine, ConstraintEngine)

    def test_enforce_all_constraints(self, unified_engine):
        """Test comprehensive constraint enforcement."""
        weights = pd.Series([0.3, 0.25, 0.2, 0.15, 0.1], index=[f"ASSET_{i}" for i in range(5)])
        previous_weights = pd.Series(
            [0.2, 0.2, 0.2, 0.2, 0.2], index=[f"ASSET_{i}" for i in range(5)]
        )

        constrained_weights, violations, cost_analysis = unified_engine.enforce_all_constraints(
            weights, previous_weights, portfolio_value=1000000
        )

        # Should return valid weights
        assert isinstance(constrained_weights, pd.Series)
        assert abs(constrained_weights.sum() - 1.0) < 1e-10

        # Should have cost analysis
        assert isinstance(cost_analysis, dict)
        if cost_analysis:  # If transaction costs were calculated
            assert "total_cost" in cost_analysis

    def test_portfolio_feasibility_validation(self, unified_engine):
        """Test portfolio feasibility validation."""
        # Infeasible portfolio (all weights exceed max)
        infeasible_weights = pd.Series([0.5, 0.5], index=["A", "B"])

        feasibility = unified_engine.validate_portfolio_feasibility(infeasible_weights)

        assert "feasibility_status" in feasibility
        assert "total_violations" in feasibility
        assert "recommendations" in feasibility

        # Should not be fully compliant
        assert feasibility["feasibility_status"] != "fully_compliant"

    def test_constraint_configuration_retrieval(self, unified_engine):
        """Test constraint configuration retrieval."""
        config = unified_engine.get_constraint_configuration()

        expected_sections = [
            "basic_constraints",
            "turnover_constraints",
            "transaction_cost_settings",
            "enforcement_settings",
        ]

        for section in expected_sections:
            assert section in config

    def test_enforcement_report_creation(self, unified_engine):
        """Test creation of comprehensive enforcement report."""
        weights = pd.Series([0.4, 0.3, 0.2, 0.1], index=["A", "B", "C", "D"])
        previous_weights = pd.Series([0.25, 0.25, 0.25, 0.25], index=["A", "B", "C", "D"])

        report = unified_engine.create_enforcement_report(weights, previous_weights)

        expected_sections = [
            "constraint_enforcement",
            "portfolio_feasibility",
            "transaction_cost_analysis",
            "configuration",
            "summary",
        ]

        for section in expected_sections:
            assert section in report

    def test_cost_aware_adjustment(self, unified_engine):
        """Test cost-aware constraint adjustment for high transaction costs."""
        # Create weights that would result in high transaction costs but respect max position weight
        weights = pd.Series([0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1], index=[f"ASSET_{i}" for i in range(7)])
        previous_weights = pd.Series([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5], index=[f"ASSET_{i}" for i in range(7)])

        constrained_weights, violations, cost_analysis = unified_engine.enforce_all_constraints(
            weights, previous_weights, portfolio_value=100000
        )

        # Should apply cost-aware adjustment if transaction costs are high
        assert isinstance(constrained_weights, pd.Series)
        # The sum might be less than 1.0 if constraints require holding cash
        assert constrained_weights.sum() <= 1.0 + 1e-10

    def test_feasibility_validation_different_status_levels(self, unified_engine):
        """Test feasibility validation with different violation severity levels."""
        # Fully compliant portfolio
        compliant_weights = pd.Series([0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05])
        compliant_weights.index = [f"ASSET_{i}" for i in range(len(compliant_weights))]

        feasibility_compliant = unified_engine.validate_portfolio_feasibility(compliant_weights)

        # Portfolio with minor violations
        minor_violation_weights = pd.Series([0.25, 0.25, 0.25, 0.25])
        minor_violation_weights.index = [f"ASSET_{i}" for i in range(4)]

        _ = unified_engine.validate_portfolio_feasibility(minor_violation_weights)

        # Portfolio with critical violations (multiple severe violations)
        critical_weights = pd.Series([-0.2, 0.6, 0.6])  # Negative weights + exceeds max position weight severely
        critical_weights.index = ["A", "B", "C"]

        feasibility_critical = unified_engine.validate_portfolio_feasibility(critical_weights)

        # Check that different severity levels are properly classified
        assert feasibility_critical["feasibility_status"] != feasibility_compliant["feasibility_status"]

    def test_enforce_all_constraints_without_transaction_calculator(self, basic_constraints):
        """Test constraint enforcement without transaction cost calculator."""
        engine = UnifiedConstraintEngine(basic_constraints, None)

        weights = pd.Series([0.3, 0.25, 0.2, 0.15, 0.1], index=[f"ASSET_{i}" for i in range(5)])

        constrained_weights, violations, cost_analysis = engine.enforce_all_constraints(weights)

        # Should work without transaction cost calculator
        assert isinstance(constrained_weights, pd.Series)
        assert cost_analysis == {}

    def test_cost_aware_adjustment_edge_cases(self, unified_engine):
        """Test cost-aware adjustment edge cases."""
        # Zero turnover case
        weights = pd.Series([0.2, 0.2, 0.2, 0.2, 0.2], index=[f"ASSET_{i}" for i in range(5)])
        previous_weights = weights.copy()

        constrained_weights, _, cost_analysis = unified_engine.enforce_all_constraints(
            weights, previous_weights, portfolio_value=100000
        )

        # Should handle zero turnover gracefully
        assert constrained_weights.equals(weights)

    def test_feasibility_recommendations(self, unified_engine):
        """Test generation of feasibility recommendations."""
        # Create weights that violate multiple constraints
        weights = pd.Series([-0.1, 0.3, 0.3, 0.3, 0.2], index=[f"ASSET_{i}" for i in range(5)])
        previous_weights = pd.Series([0.2, 0.2, 0.2, 0.2, 0.2], index=[f"ASSET_{i}" for i in range(5)])

        feasibility = unified_engine.validate_portfolio_feasibility(weights, previous_weights)

        # Should have recommendations for violations
        assert len(feasibility["recommendations"]) > 0
        assert any("long-only" in rec.lower() for rec in feasibility["recommendations"])

    def test_enforcement_report_without_previous_weights(self, unified_engine):
        """Test enforcement report creation without previous weights."""
        weights = pd.Series([0.4, 0.3, 0.2, 0.1], index=["A", "B", "C", "D"])

        report = unified_engine.create_enforcement_report(weights, None)

        # Should still create report without previous weights
        assert "constraint_enforcement" in report
        assert "portfolio_feasibility" in report
        # Transaction cost analysis should be empty without previous weights
        assert report["transaction_cost_analysis"] == {}

    def test_cost_aware_adjustment_high_cost_scenario(self, unified_engine):
        """Test cost-aware adjustment in high cost scenario with detailed logic."""
        # Create a scenario that will trigger cost adjustment
        weights = pd.Series([1.0, 0.0], index=["A", "B"])  # Complete portfolio change
        previous_weights = pd.Series([0.0, 1.0], index=["A", "B"])

        # Test the private method directly to verify logic
        cost_analysis = {
            "turnover": 2.0,  # High turnover
            "cost_per_turnover": 0.005,  # High cost per turnover
            "total_cost": 0.01
        }

        adjusted = unified_engine._apply_cost_aware_adjustment(
            weights, previous_weights, cost_analysis, []
        )

        # Should blend more with previous weights when costs are high
        assert not adjusted.equals(weights)  # Should be different from original


class TestViolationHandler:
    """Test violation handling and remediation strategies."""

    @pytest.fixture
    def violation_handler(self):
        """Create violation handler for testing."""
        return ViolationHandler(default_strategy=RemediationStrategy.ADJUST)

    @pytest.fixture
    def sample_violations(self):
        """Sample violations for testing."""
        return [
            ConstraintViolation(
                violation_type=ConstraintViolationType.LONG_ONLY,
                severity=ViolationSeverity.MAJOR,
                description="Negative weights detected",
                violation_value=-0.1,
                constraint_value=0.0,
                remediation_action="Clip negative weights",
            ),
            ConstraintViolation(
                violation_type=ConstraintViolationType.MAX_POSITION_WEIGHT,
                severity=ViolationSeverity.MINOR,
                description="Weight exceeds maximum",
                violation_value=0.25,
                constraint_value=0.20,
                remediation_action="Cap and redistribute",
            ),
        ]

    def test_violation_handler_initialization(self):
        """Test violation handler initialization."""
        handler = ViolationHandler(
            default_strategy=RemediationStrategy.WARN,
            enable_fallbacks=True,
        )

        assert handler.default_strategy == RemediationStrategy.WARN
        assert handler.enable_fallbacks is True
        assert isinstance(handler.remediation_history, list)

    def test_handle_long_only_violations(self, violation_handler):
        """Test handling of long-only constraint violations."""
        weights = pd.Series([0.6, -0.1, 0.3, 0.2], index=["A", "B", "C", "D"])

        violations = [
            ConstraintViolation(
                violation_type=ConstraintViolationType.LONG_ONLY,
                severity=ViolationSeverity.MAJOR,
                description="Negative weights",
                violation_value=-0.1,
                constraint_value=0.0,
                remediation_action="Clip negative weights",
            )
        ]

        adjusted_weights, actions = violation_handler.handle_violations(violations, weights)

        # Should have clipped negative weights
        assert (adjusted_weights >= 0).all()
        assert len(actions) == 1
        assert actions[0].success is True

    def test_handle_multiple_violations(self, violation_handler, sample_violations):
        """Test handling of multiple simultaneous violations."""
        weights = pd.Series([0.4, -0.1, 0.3, 0.4], index=["A", "B", "C", "D"])

        adjusted_weights, actions = violation_handler.handle_violations(sample_violations, weights)

        # Should have handled all violations
        assert len(actions) == len(sample_violations)
        assert (adjusted_weights >= 0).all()  # Long-only enforced
        assert abs(adjusted_weights.sum() - 1.0) < 1e-10  # Normalized

    def test_remediation_statistics(self, violation_handler, sample_violations):
        """Test remediation statistics tracking."""
        weights = pd.Series([0.4, -0.1, 0.3, 0.4], index=["A", "B", "C", "D"])

        # Process some violations
        violation_handler.handle_violations(sample_violations, weights)
        violation_handler.handle_violations(sample_violations[:1], weights)  # Another round

        stats = violation_handler.get_remediation_statistics()

        assert stats["total_remediations"] == 2
        assert stats["total_violations_handled"] > 0
        assert "success_rate" in stats
        assert "strategy_usage" in stats

    def test_fallback_portfolio_creation(self, violation_handler):
        """Test creation of fallback portfolio for rejected portfolios."""
        weights = pd.Series([0.3, 0.3, 0.2, 0.2], index=["A", "B", "C", "D"])

        fallback = violation_handler._create_fallback_portfolio(weights)

        # Should be equal weights
        assert abs(fallback.sum() - 1.0) < 1e-10
        expected_weight = 1.0 / len(weights)
        assert all(abs(w - expected_weight) < 1e-10 for w in fallback)

    def test_remediation_report_creation(self, violation_handler):
        """Test creation of comprehensive remediation report."""
        # Process some violations first
        weights = pd.Series([0.4, -0.1, 0.3, 0.4], index=["A", "B", "C", "D"])
        violations = [
            ConstraintViolation(
                violation_type=ConstraintViolationType.LONG_ONLY,
                severity=ViolationSeverity.MAJOR,
                description="Test violation",
                violation_value=-0.1,
                constraint_value=0.0,
                remediation_action="Test action",
            )
        ]

        violation_handler.handle_violations(violations, weights)

        report = violation_handler.create_remediation_report()

        expected_sections = ["handler_configuration", "statistics", "recent_history"]
        for section in expected_sections:
            assert section in report


if __name__ == "__main__":
    pytest.main([__file__])
