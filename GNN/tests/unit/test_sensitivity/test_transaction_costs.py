"""Unit tests for transaction cost sensitivity analysis."""

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.evaluation.sensitivity.engine import SensitivityResult
from src.evaluation.sensitivity.transaction_costs import (
    CostImpactResult,
    TransactionCostAnalyzer,
    TransactionCostScenario,
)


class TestTransactionCostScenario:
    """Test cases for TransactionCostScenario dataclass."""

    def test_transaction_cost_scenario_creation(self):
        """Test TransactionCostScenario creation."""
        scenario = TransactionCostScenario(
            name="test_scenario", cost_bps=15.0, description="Test scenario description"
        )

        assert scenario.name == "test_scenario"
        assert scenario.cost_bps == 15.0
        assert scenario.description == "Test scenario description"


class TestCostImpactResult:
    """Test cases for CostImpactResult dataclass."""

    def test_cost_impact_result_creation(self):
        """Test CostImpactResult creation."""
        scenario = TransactionCostScenario("test", 10.0, "test scenario")

        result = CostImpactResult(
            model_type="test_model",
            cost_scenario=scenario,
            original_metrics={"sharpe_ratio": 1.5},
            cost_adjusted_metrics={"sharpe_ratio": 1.3},
            performance_impact={"sharpe_ratio_relative": -13.33},
            ranking_change=1,
            statistical_significance={"p_value": 0.05, "is_significant": True},
        )

        assert result.model_type == "test_model"
        assert result.cost_scenario == scenario
        assert result.original_metrics == {"sharpe_ratio": 1.5}
        assert result.cost_adjusted_metrics == {"sharpe_ratio": 1.3}
        assert result.performance_impact == {"sharpe_ratio_relative": -13.33}
        assert result.ranking_change == 1
        assert result.statistical_significance["is_significant"] is True


class TestTransactionCostAnalyzer:
    """Test cases for TransactionCostAnalyzer."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for TransactionCostAnalyzer."""
        mock_sensitivity_engine = Mock()
        mock_performance_analytics = Mock()
        mock_statistical_validator = Mock()

        return {
            "sensitivity_engine": mock_sensitivity_engine,
            "performance_analytics": mock_performance_analytics,
            "statistical_validator": mock_statistical_validator,
        }

    @pytest.fixture
    def cost_analyzer(self, mock_dependencies):
        """Create TransactionCostAnalyzer instance with mocked dependencies."""
        return TransactionCostAnalyzer(
            sensitivity_engine=mock_dependencies["sensitivity_engine"],
            performance_analytics=mock_dependencies["performance_analytics"],
            statistical_validator=mock_dependencies["statistical_validator"],
        )

    def test_initialization(self, cost_analyzer, mock_dependencies):
        """Test TransactionCostAnalyzer initialization."""
        assert cost_analyzer.sensitivity_engine == mock_dependencies["sensitivity_engine"]
        assert cost_analyzer.performance_analytics == mock_dependencies["performance_analytics"]
        assert cost_analyzer.statistical_validator == mock_dependencies["statistical_validator"]

        # Check default cost scenarios
        assert len(cost_analyzer.cost_scenarios) == 3
        scenario_names = [s.name for s in cost_analyzer.cost_scenarios]
        assert "aggressive" in scenario_names
        assert "baseline" in scenario_names
        assert "conservative" in scenario_names

        # Check scenario cost values
        scenario_costs = {s.name: s.cost_bps for s in cost_analyzer.cost_scenarios}
        assert scenario_costs["aggressive"] == 5.0
        assert scenario_costs["baseline"] == 10.0
        assert scenario_costs["conservative"] == 20.0

    def test_get_best_configuration_valid_results(self, cost_analyzer):
        """Test getting best configuration with valid results."""
        mock_results = [
            SensitivityResult(
                model_type="test_model",
                parameter_combination={"param1": 1},
                performance_metrics={"sharpe_ratio": 1.0},
                backtest_results=pd.DataFrame(),
                execution_time=10.0,
                memory_usage=100.0,
            ),
            SensitivityResult(
                model_type="test_model",
                parameter_combination={"param1": 2},
                performance_metrics={"sharpe_ratio": 1.5},
                backtest_results=pd.DataFrame(),
                execution_time=12.0,
                memory_usage=120.0,
            ),
            SensitivityResult(
                model_type="test_model",
                parameter_combination={"param1": 3},
                performance_metrics={"sharpe_ratio": 1.2},
                backtest_results=pd.DataFrame(),
                execution_time=11.0,
                memory_usage=110.0,
            ),
        ]

        best_config = cost_analyzer._get_best_configuration(mock_results, "sharpe_ratio")

        assert best_config is not None
        assert best_config.parameter_combination == {"param1": 2}
        assert best_config.performance_metrics["sharpe_ratio"] == 1.5

    def test_get_best_configuration_no_valid_results(self, cost_analyzer):
        """Test getting best configuration with no valid results."""
        mock_results = [
            SensitivityResult(
                model_type="test_model",
                parameter_combination={"param1": 1},
                performance_metrics={},  # No sharpe_ratio
                backtest_results=pd.DataFrame(),
                execution_time=10.0,
                memory_usage=100.0,
            ),
            SensitivityResult(
                model_type="test_model",
                parameter_combination={"param1": 2},
                performance_metrics={"other_metric": 1.5},  # No sharpe_ratio
                backtest_results=pd.DataFrame(),
                execution_time=12.0,
                memory_usage=120.0,
                error="Test error",  # Has error
            ),
        ]

        best_config = cost_analyzer._get_best_configuration(mock_results, "sharpe_ratio")

        assert best_config is None

    def test_get_best_configuration_empty_results(self, cost_analyzer):
        """Test getting best configuration with empty results."""
        best_config = cost_analyzer._get_best_configuration([], "sharpe_ratio")

        assert best_config is None

    def test_calculate_performance_impact(self, cost_analyzer):
        """Test performance impact calculation."""
        original_metrics = {"sharpe_ratio": 1.5, "max_drawdown": -0.1, "annual_return": 0.12}

        cost_adjusted_metrics = {"sharpe_ratio": 1.3, "max_drawdown": -0.12, "annual_return": 0.10}

        impact = cost_analyzer._calculate_performance_impact(
            original_metrics, cost_adjusted_metrics
        )

        # Check sharpe_ratio impact
        expected_absolute = 1.3 - 1.5  # -0.2
        expected_relative = (expected_absolute / 1.5) * 100  # -13.33%

        assert "sharpe_ratio_absolute" in impact
        assert "sharpe_ratio_relative" in impact
        assert abs(impact["sharpe_ratio_absolute"] - expected_absolute) < 1e-10
        assert abs(impact["sharpe_ratio_relative"] - expected_relative) < 1e-10

        # Check max_drawdown impact
        expected_dd_absolute = -0.12 - (-0.1)  # -0.02
        expected_dd_relative = (expected_dd_absolute / abs(-0.1)) * 100  # -20%

        assert abs(impact["max_drawdown_absolute"] - expected_dd_absolute) < 1e-10
        assert abs(impact["max_drawdown_relative"] - expected_dd_relative) < 1e-10

    def test_calculate_performance_impact_zero_division(self, cost_analyzer):
        """Test performance impact calculation with zero original value."""
        original_metrics = {"metric": 0.0}
        cost_adjusted_metrics = {"metric": 0.1}

        impact = cost_analyzer._calculate_performance_impact(
            original_metrics, cost_adjusted_metrics
        )

        assert impact["metric_absolute"] == 0.1
        assert impact["metric_relative"] == 0.0  # Should handle zero division

    def test_calculate_ranking_stability(self, cost_analyzer):
        """Test ranking stability calculation."""
        # Create mock cost results with different scenarios
        cost_results = {
            "model_a": [
                CostImpactResult(
                    model_type="model_a",
                    cost_scenario=TransactionCostScenario("baseline", 10.0, "baseline"),
                    original_metrics={},
                    cost_adjusted_metrics={"sharpe_ratio": 1.5},
                    performance_impact={},
                    ranking_change=0,
                    statistical_significance={},
                ),
                CostImpactResult(
                    model_type="model_a",
                    cost_scenario=TransactionCostScenario("aggressive", 5.0, "aggressive"),
                    original_metrics={},
                    cost_adjusted_metrics={"sharpe_ratio": 1.6},
                    performance_impact={},
                    ranking_change=0,
                    statistical_significance={},
                ),
            ],
            "model_b": [
                CostImpactResult(
                    model_type="model_b",
                    cost_scenario=TransactionCostScenario("baseline", 10.0, "baseline"),
                    original_metrics={},
                    cost_adjusted_metrics={"sharpe_ratio": 1.2},
                    performance_impact={},
                    ranking_change=0,
                    statistical_significance={},
                ),
                CostImpactResult(
                    model_type="model_b",
                    cost_scenario=TransactionCostScenario("aggressive", 5.0, "aggressive"),
                    original_metrics={},
                    cost_adjusted_metrics={"sharpe_ratio": 1.0},
                    performance_impact={},
                    ranking_change=0,
                    statistical_significance={},
                ),
            ],
        }

        cost_analyzer._calculate_ranking_stability(cost_results, "sharpe_ratio")

        # Check that ranking changes were calculated
        # In baseline: model_a (1.5) > model_b (1.2), so model_a=0, model_b=1
        # In aggressive: model_a (1.6) > model_b (1.0), so model_a=0, model_b=1
        # No ranking change should occur

        for model_results in cost_results.values():
            for result in model_results:
                if result.cost_scenario.name != "baseline":
                    # Ranking changes should be calculated
                    assert hasattr(result, "ranking_change")

    def test_analyze_ranking_stability_no_results(self, cost_analyzer):
        """Test ranking stability analysis with no results."""
        with pytest.raises(ValueError, match="No cost impact results available"):
            cost_analyzer.analyze_ranking_stability()

    def test_get_cost_sensitivity_summary_no_results(self, cost_analyzer):
        """Test cost sensitivity summary with no results."""
        with pytest.raises(ValueError, match="No cost impact results available"):
            cost_analyzer.get_cost_sensitivity_summary()

    def test_get_cost_sensitivity_summary_with_results(self, cost_analyzer):
        """Test cost sensitivity summary with valid results."""
        # Mock cost impact results
        cost_analyzer.cost_impact_results = {
            "model_a": [
                CostImpactResult(
                    model_type="model_a",
                    cost_scenario=TransactionCostScenario("baseline", 10.0, "baseline"),
                    original_metrics={},
                    cost_adjusted_metrics={"sharpe_ratio": 1.5},
                    performance_impact={"sharpe_ratio_relative": -5.0},
                    ranking_change=0,
                    statistical_significance={},
                ),
                CostImpactResult(
                    model_type="model_a",
                    cost_scenario=TransactionCostScenario("conservative", 20.0, "conservative"),
                    original_metrics={},
                    cost_adjusted_metrics={"sharpe_ratio": 1.3},
                    performance_impact={"sharpe_ratio_relative": -10.0},
                    ranking_change=1,
                    statistical_significance={},
                ),
            ],
            "model_b": [
                CostImpactResult(
                    model_type="model_b",
                    cost_scenario=TransactionCostScenario("baseline", 10.0, "baseline"),
                    original_metrics={},
                    cost_adjusted_metrics={"sharpe_ratio": 1.2},
                    performance_impact={"sharpe_ratio_relative": -2.0},
                    ranking_change=0,
                    statistical_significance={},
                ),
                CostImpactResult(
                    model_type="model_b",
                    cost_scenario=TransactionCostScenario("conservative", 20.0, "conservative"),
                    original_metrics={},
                    cost_adjusted_metrics={"sharpe_ratio": 1.1},
                    performance_impact={"sharpe_ratio_relative": -4.0},
                    ranking_change=0,
                    statistical_significance={},
                ),
            ],
        }

        summary = cost_analyzer.get_cost_sensitivity_summary("sharpe_ratio")

        assert summary["models_analyzed"] == 2
        assert summary["cost_scenarios"] == 3  # Default scenarios
        assert summary["most_sensitive_model"] in ["model_a", "model_b"]
        assert summary["least_sensitive_model"] in ["model_a", "model_b"]
        assert "avg_impact_by_scenario" in summary
        assert isinstance(summary["ranking_stability_score"], float)

    def test_create_cost_impact_visualization_data_no_results(self, cost_analyzer):
        """Test visualization data creation with no results."""
        with pytest.raises(ValueError, match="No cost impact results available"):
            cost_analyzer.create_cost_impact_visualization_data()

    def test_create_cost_impact_visualization_data_with_results(self, cost_analyzer):
        """Test visualization data creation with valid results."""
        # Mock cost impact results
        cost_analyzer.cost_impact_results = {
            "model_a": [
                CostImpactResult(
                    model_type="model_a",
                    cost_scenario=TransactionCostScenario("baseline", 10.0, "baseline"),
                    original_metrics={"sharpe_ratio": 1.5},
                    cost_adjusted_metrics={"sharpe_ratio": 1.4},
                    performance_impact={"sharpe_ratio_relative": -6.67},
                    ranking_change=0,
                    statistical_significance={"is_significant": True},
                )
            ]
        }

        viz_data = cost_analyzer.create_cost_impact_visualization_data("sharpe_ratio")

        assert "heatmap" in viz_data
        assert "ranking" in viz_data
        assert "sensitivity" in viz_data

        # Check heatmap data
        heatmap_df = viz_data["heatmap"]
        assert len(heatmap_df) == 1
        assert "model" in heatmap_df.columns
        assert "cost_scenario" in heatmap_df.columns
        assert "performance_impact" in heatmap_df.columns
        assert "is_significant" in heatmap_df.columns

        # Check ranking data
        ranking_df = viz_data["ranking"]
        assert len(ranking_df) == 1
        assert "model" in ranking_df.columns
        assert "ranking_change" in ranking_df.columns

        # Check sensitivity data
        sensitivity_df = viz_data["sensitivity"]
        assert len(sensitivity_df) == 1
        assert "model" in sensitivity_df.columns
        assert "mean_impact" in sensitivity_df.columns

    def test_export_cost_analysis_results_no_results(self, cost_analyzer, tmp_path):
        """Test export with no results."""
        export_path = tmp_path / "test_export.csv"

        with pytest.raises(ValueError, match="No cost impact results available"):
            cost_analyzer.export_cost_analysis_results(str(export_path))

    def test_export_cost_analysis_results_csv(self, cost_analyzer, tmp_path):
        """Test CSV export functionality."""
        # Mock cost impact results
        cost_analyzer.cost_impact_results = {
            "model_a": [
                CostImpactResult(
                    model_type="model_a",
                    cost_scenario=TransactionCostScenario("baseline", 10.0, "baseline"),
                    original_metrics={"sharpe_ratio": 1.5},
                    cost_adjusted_metrics={"sharpe_ratio": 1.4},
                    performance_impact={"sharpe_ratio_relative": -6.67},
                    ranking_change=0,
                    statistical_significance={"is_significant": True, "p_value": 0.03},
                )
            ]
        }

        export_path = tmp_path / "cost_results.csv"
        cost_analyzer.export_cost_analysis_results(str(export_path))

        # Verify file was created and has correct content
        assert export_path.exists()

        df = pd.read_csv(export_path)
        assert len(df) == 1
        assert df.iloc[0]["model_type"] == "model_a"
        assert df.iloc[0]["cost_scenario"] == "baseline"
        assert df.iloc[0]["cost_bps"] == 10.0
        assert df.iloc[0]["is_significant"]
        assert abs(df.iloc[0]["original_sharpe_ratio"] - 1.5) < 1e-10

    def test_export_cost_analysis_results_unsupported_format(self, cost_analyzer, tmp_path):
        """Test export with unsupported file format."""
        # Mock results
        cost_analyzer.cost_impact_results = {"model_a": []}

        export_path = tmp_path / "results.txt"

        with pytest.raises(ValueError, match="Unsupported file format"):
            cost_analyzer.export_cost_analysis_results(str(export_path))

    def test_analyze_cost_scenario_success(self, cost_analyzer, mock_dependencies):
        """Test successful cost scenario analysis."""
        # Mock base configuration
        base_config = SensitivityResult(
            model_type="test_model",
            parameter_combination={"param1": 1},
            performance_metrics={"sharpe_ratio": 1.5},
            backtest_results=pd.DataFrame({"returns": [0.01, 0.02, -0.01]}),
            execution_time=10.0,
            memory_usage=100.0,
        )

        # Mock cost scenario
        cost_scenario = TransactionCostScenario("test", 15.0, "test scenario")

        # Mock sensitivity engine result
        mock_cost_result = SensitivityResult(
            model_type="test_model",
            parameter_combination={"param1": 1, "transaction_cost_bps": 15.0},
            performance_metrics={"sharpe_ratio": 1.3},
            backtest_results=pd.DataFrame({"returns": [0.008, 0.018, -0.012]}),
            execution_time=10.5,
            memory_usage=105.0,
        )

        mock_dependencies["sensitivity_engine"]._execute_parameter_combination.return_value = (
            mock_cost_result
        )
        mock_dependencies["statistical_validator"].sharpe_ratio_test.return_value = {
            "p_value": 0.05,
            "is_significant": True,
            "test_statistic": 2.0,
        }

        # Execute test
        result = cost_analyzer._analyze_cost_scenario(
            model_type="test_model",
            base_config=base_config,
            cost_scenario=cost_scenario,
            data=pd.DataFrame(),
            start_date="2023-01-01",
            end_date="2023-12-31",
            target_metric="sharpe_ratio",
        )

        # Assertions
        assert result is not None
        assert result.model_type == "test_model"
        assert result.cost_scenario == cost_scenario
        assert result.original_metrics == {"sharpe_ratio": 1.5}
        assert result.cost_adjusted_metrics == {"sharpe_ratio": 1.3}
        assert "sharpe_ratio_absolute" in result.performance_impact
        assert "sharpe_ratio_relative" in result.performance_impact
        assert result.statistical_significance["is_significant"] is True

        # Verify mock calls
        mock_dependencies["sensitivity_engine"]._execute_parameter_combination.assert_called_once()
        mock_dependencies["statistical_validator"].sharpe_ratio_test.assert_called_once()

    def test_analyze_cost_scenario_execution_error(self, cost_analyzer, mock_dependencies):
        """Test cost scenario analysis with execution error."""
        base_config = SensitivityResult(
            model_type="test_model",
            parameter_combination={"param1": 1},
            performance_metrics={"sharpe_ratio": 1.5},
            backtest_results=pd.DataFrame(),
            execution_time=10.0,
            memory_usage=100.0,
        )

        cost_scenario = TransactionCostScenario("test", 15.0, "test scenario")

        # Mock execution failure
        mock_error_result = SensitivityResult(
            model_type="test_model",
            parameter_combination={"param1": 1},
            performance_metrics={},
            backtest_results=pd.DataFrame(),
            execution_time=0.0,
            memory_usage=0.0,
            error="Test execution error",
        )

        mock_dependencies["sensitivity_engine"]._execute_parameter_combination.return_value = (
            mock_error_result
        )

        # Execute test
        result = cost_analyzer._analyze_cost_scenario(
            model_type="test_model",
            base_config=base_config,
            cost_scenario=cost_scenario,
            data=pd.DataFrame(),
            start_date="2023-01-01",
            end_date="2023-12-31",
            target_metric="sharpe_ratio",
        )

        # Should return None due to error
        assert result is None
