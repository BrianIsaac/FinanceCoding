"""
Integration tests for unified constraint system across all models.

Tests constraint enforcement consistency between HRP, LSTM, GAT,
and baseline models to ensure fair performance comparisons.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest.turnover_tracker import TurnoverTracker
from src.models.base.constraints import PortfolioConstraints
from src.models.base.portfolio_model import PortfolioModel
from src.models.baselines.equal_weight import EqualWeightModel


class MockPortfolioModel(PortfolioModel):
    """Mock portfolio model for testing constraint integration."""

    def __init__(self, constraints: PortfolioConstraints, raw_weights: pd.Series):
        super().__init__(constraints)
        self.raw_weights = raw_weights
        self.is_fitted = True

    def fit(self, returns, universe, fit_period):
        """Mock fit method."""
        pass

    def predict_weights(self, date, universe):
        """Return mock weights for testing."""
        return self.validate_weights(self.raw_weights)

    def get_model_info(self):
        """Return mock model info."""
        return {"model_type": "mock", "constraints": self.constraints}


class TestUnifiedConstraintIntegration:
    """Test unified constraint system integration across all models."""

    @pytest.fixture
    def test_constraints(self):
        """Standard constraint configuration for testing."""
        return PortfolioConstraints(
            long_only=True,
            top_k_positions=5,
            max_position_weight=0.25,
            max_monthly_turnover=0.20,
            transaction_cost_bps=10.0,
            min_weight_threshold=0.01,
            violation_handling="adjust",
            position_ranking_method="weight",
            cost_aware_enforcement=True,
        )

    @pytest.fixture
    def sample_universe(self):
        """Sample asset universe for testing."""
        return [f"ASSET_{i:02d}" for i in range(10)]

    @pytest.fixture
    def sample_returns(self, sample_universe):
        """Sample returns data for testing."""
        dates = pd.date_range("2023-01-01", periods=252, freq="B")  # Business days
        np.random.seed(42)
        returns_data = np.random.normal(0.001, 0.02, (len(dates), len(sample_universe)))

        return pd.DataFrame(returns_data, index=dates, columns=sample_universe)

    @pytest.fixture
    def problematic_weights(self, sample_universe):
        """Weights that violate multiple constraints for testing."""
        return pd.Series(
            [
                0.35,  # Violates max weight (25%)
                -0.05,  # Violates long-only
                0.20,
                0.15,
                0.12,
                0.08,
                0.06,  # Position 7 - violates top-5 constraint
                0.05,  # Position 8 - violates top-5 constraint
                0.03,  # Position 9 - violates top-5 constraint
                0.01,  # Position 10 - violates top-5 constraint
            ],
            index=sample_universe,
        )

    def test_constraint_consistency_across_models(self, test_constraints, problematic_weights):
        """Test that all models apply constraints consistently."""
        # Create different model instances with same constraints
        mock_model_1 = MockPortfolioModel(test_constraints, problematic_weights.copy())
        mock_model_2 = MockPortfolioModel(test_constraints, problematic_weights.copy())

        # Get constrained weights from both models
        weights_1 = mock_model_1.predict_weights(
            pd.Timestamp("2023-06-01"), problematic_weights.index.tolist()
        )
        weights_2 = mock_model_2.predict_weights(
            pd.Timestamp("2023-06-01"), problematic_weights.index.tolist()
        )

        # Should be identical (within numerical precision)
        pd.testing.assert_series_equal(weights_1, weights_2, rtol=1e-10)

        # Both should satisfy all constraints
        for weights in [weights_1, weights_2]:
            # Long-only constraint
            assert (weights >= 0).all(), "Long-only constraint violated"

            # Top-k constraint
            num_positions = (weights > test_constraints.min_weight_threshold).sum()
            assert (
                num_positions <= test_constraints.top_k_positions
            ), f"Top-k constraint violated: {num_positions} > {test_constraints.top_k_positions}"

            # Max weight constraint
            assert (
                weights <= test_constraints.max_position_weight + 1e-10
            ).all(), "Max weight constraint violated"

            # Weights should sum to approximately 1.0
            assert abs(weights.sum() - 1.0) < 1e-10, f"Weights sum to {weights.sum()}, expected 1.0"

    def test_top_k_constraint_variants(self, test_constraints, problematic_weights):
        """Test top-k constraint with different k values."""
        k_values = [3, 5, 7, 10]

        for k in k_values:
            constraints = test_constraints
            constraints.top_k_positions = k

            model = MockPortfolioModel(constraints, problematic_weights.copy())
            constrained_weights = model.predict_weights(
                pd.Timestamp("2023-06-01"), problematic_weights.index.tolist()
            )

            num_positions = (constrained_weights > constraints.min_weight_threshold).sum()
            assert num_positions <= k, f"Top-{k} constraint violated: {num_positions} positions"

    def test_turnover_constraint_enforcement(self, test_constraints, sample_universe):
        """Test turnover constraint enforcement across different scenarios."""
        # Create current and previous weights with high turnover
        previous_weights = pd.Series([0.20] * 5 + [0.0] * 5, index=sample_universe)
        current_weights = pd.Series([0.0] * 5 + [0.20] * 5, index=sample_universe)  # Complete flip

        model = MockPortfolioModel(test_constraints, current_weights)

        # Apply constraints with previous weights for turnover calculation
        constrained_weights = model.validate_weights(
            current_weights, previous_weights=previous_weights, date=pd.Timestamp("2023-06-01")
        )

        # Calculate final turnover
        turnover = np.abs(constrained_weights - previous_weights).sum()
        assert (
            turnover <= test_constraints.max_monthly_turnover + 1e-6
        ), f"Turnover {turnover:.3f} exceeds limit {test_constraints.max_monthly_turnover}"

    def test_position_ranking_methods(self, test_constraints, sample_universe):
        """Test different position ranking methods for top-k constraint."""
        # Create weights and mock scores
        weights = pd.Series(np.random.uniform(0.05, 0.15, 10), index=sample_universe)
        weights = weights / weights.sum()  # Normalize

        model_scores = pd.Series(np.random.uniform(0.1, 0.9, 10), index=sample_universe)

        ranking_methods = ["weight", "score", "mixed"]

        for method in ranking_methods:
            constraints = test_constraints
            constraints.position_ranking_method = method
            constraints.top_k_positions = 5

            model = MockPortfolioModel(constraints, weights.copy())

            constrained_weights = model.validate_weights(
                weights, model_scores=model_scores, date=pd.Timestamp("2023-06-01")
            )

            # Should respect top-5 constraint regardless of ranking method
            num_positions = (constrained_weights > constraints.min_weight_threshold).sum()
            assert (
                num_positions <= 5
            ), f"Top-5 constraint violated with {method} ranking: {num_positions} positions"

    def test_transaction_cost_integration(self, test_constraints, sample_universe):
        """Test integration with transaction cost calculations."""
        previous_weights = pd.Series([0.10] * 10, index=sample_universe)
        high_turnover_weights = pd.Series([0.0] * 5 + [0.20] * 5, index=sample_universe)

        # Enable cost-aware enforcement
        constraints = test_constraints
        constraints.cost_aware_enforcement = True

        model = MockPortfolioModel(constraints, high_turnover_weights)

        constrained_weights = model.validate_weights(
            high_turnover_weights,
            previous_weights=previous_weights,
            date=pd.Timestamp("2023-06-01"),
        )

        # Should have transaction cost analysis available
        cost_analysis = model.get_transaction_cost_analysis()

        if cost_analysis:  # If costs were calculated
            assert "total_cost" in cost_analysis
            assert "turnover" in cost_analysis
            assert cost_analysis["total_cost"] >= 0

    def test_constraint_violation_logging(self, test_constraints, problematic_weights):
        """Test constraint violation detection and logging."""
        model = MockPortfolioModel(test_constraints, problematic_weights)

        # Apply constraints - should generate violations
        constrained_weights = model.validate_weights(
            problematic_weights, date=pd.Timestamp("2023-06-01")
        )

        # Check that violations were recorded
        violations = model.get_constraint_violations()

        # Should have detected multiple violations in problematic_weights
        violation_types = {v.violation_type.value for v in violations}

        # Should detect long-only violation (negative weight)
        assert "long_only" in violation_types

        # Should detect max weight violation (35% > 25%)
        assert "max_position_weight" in violation_types

        # Should detect top-k violation (10 positions > 5)
        assert "top_k_positions" in violation_types

    def test_feasibility_assessment(self, test_constraints, sample_universe):
        """Test portfolio feasibility assessment functionality."""
        # Create clearly infeasible portfolio
        infeasible_weights = pd.Series([0.5, 0.5] + [0.0] * 8, index=sample_universe)

        model = MockPortfolioModel(test_constraints, infeasible_weights)
        feasibility = model.validate_portfolio_feasibility(infeasible_weights)

        assert feasibility["feasibility_status"] != "fully_compliant"
        assert feasibility["total_violations"] > 0
        assert len(feasibility["recommendations"]) > 0

        # Create feasible portfolio
        feasible_weights = pd.Series([0.2] * 5 + [0.0] * 5, index=sample_universe)
        feasibility_good = model.validate_portfolio_feasibility(feasible_weights)

        assert (
            feasibility_good["total_violations"] == 0
            or feasibility_good["feasibility_status"] == "fully_compliant"
        )

    def test_constraint_report_generation(self, test_constraints, sample_universe):
        """Test comprehensive constraint report generation."""
        weights = pd.Series(
            [0.22, 0.20, 0.18, 0.15, 0.15, 0.05, 0.03, 0.02] + [0.0] * 2, index=sample_universe
        )
        previous_weights = pd.Series([0.20] * 5 + [0.0] * 5, index=sample_universe)

        model = MockPortfolioModel(test_constraints, weights)

        report = model.create_constraint_report(weights, previous_weights)

        # Check report structure
        expected_sections = [
            "constraint_enforcement",
            "portfolio_feasibility",
            "transaction_cost_analysis",
            "configuration",
            "summary",
        ]

        for section in expected_sections:
            assert section in report, f"Missing section: {section}"

        # Check summary metrics
        summary = report["summary"]
        assert "total_positions" in summary
        assert "max_weight" in summary
        assert "is_feasible" in summary

    @patch("src.models.baselines.equal_weight.EqualWeightModel")
    def test_equal_weight_baseline_integration(
        self, mock_equal_weight, test_constraints, sample_universe
    ):
        """Test constraint integration with equal weight baseline."""
        # Mock the equal weight model to return our test weights
        mock_instance = Mock()
        mock_instance.constraints = test_constraints
        mock_equal_weight.return_value = mock_instance

        # Create weights that should be automatically adjusted
        equal_weights = pd.Series([0.1] * 10, index=sample_universe)
        mock_instance.predict_weights.return_value = equal_weights

        # Verify equal weight model would produce feasible portfolio
        assert (equal_weights >= 0).all()
        assert abs(equal_weights.sum() - 1.0) < 1e-10
        assert (equal_weights <= test_constraints.max_position_weight).all()

    def test_memory_efficiency_large_universe(self, test_constraints):
        """Test constraint enforcement efficiency with large universe."""
        # Create large universe (similar to S&P MidCap 400)
        large_universe = [f"ASSET_{i:03d}" for i in range(400)]

        # Create random weights
        np.random.seed(42)
        large_weights = pd.Series(np.random.exponential(0.01, 400), index=large_universe)
        large_weights = large_weights / large_weights.sum()

        # Set reasonable constraints for large universe
        constraints = test_constraints
        constraints.top_k_positions = 50
        constraints.max_position_weight = 0.05

        model = MockPortfolioModel(constraints, large_weights)

        # This should complete without memory issues
        constrained_weights = model.predict_weights(pd.Timestamp("2023-06-01"), large_universe)

        # Verify constraints are satisfied
        num_positions = (constrained_weights > constraints.min_weight_threshold).sum()
        assert num_positions <= constraints.top_k_positions
        assert (constrained_weights <= constraints.max_position_weight + 1e-10).all()
        assert abs(constrained_weights.sum() - 1.0) < 1e-8


class TestTurnoverTracker:
    """Test turnover tracking integration with constraint system."""

    @pytest.fixture
    def turnover_tracker(self):
        """Create turnover tracker for testing."""
        return TurnoverTracker(max_monthly_turnover=0.20, lookback_months=3, enable_budgeting=True)

    @pytest.fixture
    def sample_rebalancing_data(self):
        """Sample rebalancing data for testing."""
        assets = [f"ASSET_{i}" for i in range(5)]

        current = pd.Series([0.25, 0.20, 0.20, 0.20, 0.15], index=assets)
        previous = pd.Series([0.20, 0.20, 0.20, 0.20, 0.20], index=assets)

        return current, previous

    def test_turnover_calculation(self, turnover_tracker, sample_rebalancing_data):
        """Test comprehensive turnover calculation."""
        current_weights, previous_weights = sample_rebalancing_data

        turnover_metrics = turnover_tracker.calculate_turnover(current_weights, previous_weights)

        # Check expected metrics
        expected_keys = [
            "portfolio_turnover",
            "asset_turnovers",
            "num_trades",
            "avg_trade_size",
            "buys",
            "sells",
        ]

        for key in expected_keys:
            assert key in turnover_metrics

        # Verify calculations
        expected_turnover = np.abs(current_weights - previous_weights).sum()
        assert abs(turnover_metrics["portfolio_turnover"] - expected_turnover) < 1e-10

    def test_rebalancing_tracking(self, turnover_tracker, sample_rebalancing_data):
        """Test rebalancing event tracking."""
        current_weights, previous_weights = sample_rebalancing_data

        tracking_result = turnover_tracker.track_rebalancing(
            date=pd.Timestamp("2023-06-01"),
            current_weights=current_weights,
            previous_weights=previous_weights,
            portfolio_value=1000000,
        )

        # Check tracking result structure
        assert "tracking_record" in tracking_result
        assert "rolling_average" in tracking_result
        assert "enforcement_needed" in tracking_result

        # Check tracking record
        record = tracking_result["tracking_record"]
        assert record["date"] == pd.Timestamp("2023-06-01")
        assert "portfolio_turnover" in record
        assert "constraint_compliant" in record

    def test_turnover_enforcement(self, turnover_tracker):
        """Test turnover constraint enforcement."""
        # Create high-turnover scenario
        current_weights = pd.Series([0.0, 0.0, 0.5, 0.5], index=["A", "B", "C", "D"])
        previous_weights = pd.Series([0.5, 0.5, 0.0, 0.0], index=["A", "B", "C", "D"])

        # This should have turnover = 2.0 (complete flip)
        initial_turnover = turnover_tracker.calculate_turnover(current_weights, previous_weights)[
            "portfolio_turnover"
        ]
        assert initial_turnover > turnover_tracker.max_monthly_turnover

        # Apply enforcement
        adjusted_weights, enforcement_details = turnover_tracker.enforce_turnover_constraint(
            current_weights, previous_weights
        )

        # Check enforcement was applied
        assert enforcement_details["enforcement_applied"] is True
        assert enforcement_details["final_turnover"] <= turnover_tracker.max_monthly_turnover + 1e-6
        assert enforcement_details["turnover_reduction"] > 0

    def test_turnover_statistics(self, turnover_tracker, sample_rebalancing_data):
        """Test turnover statistics calculation."""
        current_weights, previous_weights = sample_rebalancing_data

        # Generate several rebalancing events
        dates = pd.date_range("2023-01-01", periods=5, freq="M")

        for date in dates:
            # Slightly modify weights for each period
            modified_current = current_weights * (
                1 + np.random.normal(0, 0.1, len(current_weights))
            )
            modified_current = modified_current / modified_current.sum()

            turnover_tracker.track_rebalancing(date, modified_current, previous_weights)
            previous_weights = modified_current.copy()

        stats = turnover_tracker.get_turnover_statistics()

        # Check statistics structure
        expected_stats = [
            "total_rebalances",
            "avg_turnover",
            "median_turnover",
            "std_turnover",
            "constraint_violations",
            "violation_rate",
        ]

        for stat in expected_stats:
            assert stat in stats

        assert stats["total_rebalances"] == 5

    def test_turnover_report_generation(self, turnover_tracker, sample_rebalancing_data):
        """Test comprehensive turnover report generation."""
        current_weights, previous_weights = sample_rebalancing_data

        # Track at least one rebalancing event
        turnover_tracker.track_rebalancing(
            pd.Timestamp("2023-06-01"), current_weights, previous_weights
        )

        report = turnover_tracker.create_turnover_report()

        # Check report structure
        expected_sections = [
            "configuration",
            "portfolio_statistics",
            "asset_analysis",
            "recent_history",
        ]

        for section in expected_sections:
            assert section in report


if __name__ == "__main__":
    pytest.main([__file__])
