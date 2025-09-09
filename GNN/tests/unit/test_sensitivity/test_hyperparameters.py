"""Unit tests for hyperparameter sensitivity testing."""

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.evaluation.sensitivity.engine import ParameterGrid, SensitivityAnalysisEngine
from src.evaluation.sensitivity.hyperparameters import (
    HyperparameterTester,
    HyperparameterTestResult,
)


class TestHyperparameterTestResult:
    """Test cases for HyperparameterTestResult dataclass."""

    def test_hyperparameter_test_result_creation(self):
        """Test HyperparameterTestResult creation."""
        result = HyperparameterTestResult(
            model_type="test_model",
            parameter_name="param1",
            parameter_values=[1, 2, 3],
            metric_values=[0.5, 1.0, 1.5],
            statistical_significance={"p_value": 0.03},
            effect_size=0.25,
            optimal_value=3,
            confidence_interval=(1.2, 1.8),
        )

        assert result.model_type == "test_model"
        assert result.parameter_name == "param1"
        assert result.parameter_values == [1, 2, 3]
        assert result.metric_values == [0.5, 1.0, 1.5]
        assert result.statistical_significance == {"p_value": 0.03}
        assert result.effect_size == 0.25
        assert result.optimal_value == 3
        assert result.confidence_interval == (1.2, 1.8)


class TestHyperparameterTester:
    """Test cases for HyperparameterTester class."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for HyperparameterTester."""
        mock_sensitivity_engine = Mock(spec=SensitivityAnalysisEngine)
        mock_statistical_validator = Mock()
        mock_bootstrap_methodology = Mock()
        mock_multiple_comparison_corrector = Mock()

        return {
            "sensitivity_engine": mock_sensitivity_engine,
            "statistical_validator": mock_statistical_validator,
            "bootstrap_methodology": mock_bootstrap_methodology,
            "multiple_comparison_corrector": mock_multiple_comparison_corrector,
        }

    @pytest.fixture
    def hyperparameter_tester(self, mock_dependencies):
        """Create HyperparameterTester instance with mocked dependencies."""
        return HyperparameterTester(
            sensitivity_engine=mock_dependencies["sensitivity_engine"],
            statistical_validator=mock_dependencies["statistical_validator"],
            bootstrap_methodology=mock_dependencies["bootstrap_methodology"],
            multiple_comparison_corrector=mock_dependencies["multiple_comparison_corrector"],
        )

    def test_initialization(self, hyperparameter_tester, mock_dependencies):
        """Test HyperparameterTester initialization."""
        assert hyperparameter_tester.sensitivity_engine == mock_dependencies["sensitivity_engine"]
        assert (
            hyperparameter_tester.statistical_validator
            == mock_dependencies["statistical_validator"]
        )
        assert (
            hyperparameter_tester.bootstrap_methodology
            == mock_dependencies["bootstrap_methodology"]
        )
        assert (
            hyperparameter_tester.multiple_comparison_corrector
            == mock_dependencies["multiple_comparison_corrector"]
        )
        assert hyperparameter_tester.test_results == {}

    def test_create_hrp_parameter_grid(self, hyperparameter_tester):
        """Test HRP parameter grid creation."""
        grid = hyperparameter_tester.create_hrp_parameter_grid()

        assert grid.model_type == "hrp"
        assert "lookback_days" in grid.parameters
        assert "linkage_method" in grid.parameters
        assert "distance_metric" in grid.parameters
        assert "min_weight" in grid.parameters
        assert "max_weight" in grid.parameters

        # Check parameter values
        assert grid.parameters["lookback_days"] == [252, 504, 756, 1008]
        assert grid.parameters["linkage_method"] == ["single", "complete", "average", "ward"]
        assert grid.parameters["distance_metric"] == [
            "correlation",
            "angular",
            "absolute_correlation",
        ]

        # Check constraints
        assert grid.constraints["rebalance_frequency"] == "monthly"
        assert grid.constraints["transaction_cost_bps"] == 10.0

    def test_create_lstm_parameter_grid(self, hyperparameter_tester):
        """Test LSTM parameter grid creation."""
        grid = hyperparameter_tester.create_lstm_parameter_grid()

        assert grid.model_type == "lstm"
        assert "sequence_length" in grid.parameters
        assert "hidden_size" in grid.parameters
        assert "num_layers" in grid.parameters
        assert "dropout" in grid.parameters
        assert "learning_rate" in grid.parameters
        assert "batch_size" in grid.parameters
        assert "epochs" in grid.parameters

        # Check parameter values
        assert grid.parameters["sequence_length"] == [30, 45, 60, 90]
        assert grid.parameters["hidden_size"] == [64, 128, 256]
        assert grid.parameters["num_layers"] == [1, 2, 3]
        assert grid.parameters["dropout"] == [0.1, 0.3, 0.5]
        assert grid.parameters["learning_rate"] == [0.0001, 0.001, 0.01]

        # Check constraints
        assert grid.constraints["optimizer"] == "adam"

    def test_create_gat_parameter_grid(self, hyperparameter_tester):
        """Test GAT parameter grid creation."""
        grid = hyperparameter_tester.create_gat_parameter_grid()

        assert grid.model_type == "gat"
        assert "attention_heads" in grid.parameters
        assert "hidden_dim" in grid.parameters
        assert "dropout" in grid.parameters
        assert "learning_rate" in grid.parameters
        assert "graph_construction" in grid.parameters
        assert "k_neighbors" in grid.parameters
        assert "edge_threshold" in grid.parameters
        assert "num_layers" in grid.parameters

        # Check parameter values
        assert grid.parameters["attention_heads"] == [2, 4, 8]
        assert grid.parameters["hidden_dim"] == [64, 128, 256]
        assert grid.parameters["graph_construction"] == ["k_nn", "mst", "tmfg"]
        assert grid.parameters["k_neighbors"] == [5, 10, 15, 20]
        assert grid.parameters["edge_threshold"] == [0.1, 0.2, 0.3]
        assert grid.parameters["num_layers"] == [2, 3, 4]

    def test_kruskal_wallis_test_valid_groups(self, hyperparameter_tester):
        """Test Kruskal-Wallis test with valid groups."""
        groups = [[1.0, 1.2, 1.1], [1.5, 1.7, 1.6], [0.8, 0.9, 0.85]]

        result = hyperparameter_tester._kruskal_wallis_test(groups)

        assert "statistic" in result
        assert "p_value" in result
        assert isinstance(result["statistic"], float)
        assert isinstance(result["p_value"], float)
        assert 0 <= result["p_value"] <= 1

    def test_kruskal_wallis_test_insufficient_groups(self, hyperparameter_tester):
        """Test Kruskal-Wallis test with insufficient groups."""
        groups = [[1.0, 1.2]]  # Only one group

        result = hyperparameter_tester._kruskal_wallis_test(groups)

        assert result["statistic"] == 0.0
        assert result["p_value"] == 1.0

    def test_kruskal_wallis_test_empty_groups(self, hyperparameter_tester):
        """Test Kruskal-Wallis test with empty groups."""
        groups = [[], []]

        result = hyperparameter_tester._kruskal_wallis_test(groups)

        assert result["statistic"] == 0.0
        assert result["p_value"] == 1.0

    def test_mannwhitney_u_test_valid_groups(self, hyperparameter_tester):
        """Test Mann-Whitney U test with valid groups."""
        group1 = [1.0, 1.2, 1.1, 1.3]
        group2 = [1.5, 1.7, 1.6, 1.8]

        result = hyperparameter_tester._mannwhitney_u_test(group1, group2)

        assert "statistic" in result
        assert "p_value" in result
        assert isinstance(result["statistic"], float)
        assert isinstance(result["p_value"], float)
        assert 0 <= result["p_value"] <= 1

    def test_mannwhitney_u_test_empty_groups(self, hyperparameter_tester):
        """Test Mann-Whitney U test with empty groups."""
        result = hyperparameter_tester._mannwhitney_u_test([], [1, 2, 3])

        assert result["statistic"] == 0.0
        assert result["p_value"] == 1.0

        result = hyperparameter_tester._mannwhitney_u_test([1, 2], [])

        assert result["statistic"] == 0.0
        assert result["p_value"] == 1.0

    def test_bootstrap_confidence_interval_valid_data(self, hyperparameter_tester):
        """Test bootstrap confidence interval with valid data."""
        with patch("numpy.random.choice") as mock_choice, patch("numpy.random.seed"):

            # Mock bootstrap samples
            mock_choice.side_effect = lambda data, size, replace: data  # Return original data

            data = [1.0, 1.2, 1.1, 1.3, 1.15]
            result = hyperparameter_tester._bootstrap_confidence_interval(
                data, confidence_level=0.95
            )

            assert len(result) == 2
            lower_bound, upper_bound = result
            assert isinstance(lower_bound, float)
            assert isinstance(upper_bound, float)
            assert lower_bound <= upper_bound

    def test_bootstrap_confidence_interval_empty_data(self, hyperparameter_tester):
        """Test bootstrap confidence interval with empty data."""
        result = hyperparameter_tester._bootstrap_confidence_interval([])

        assert result == (0.0, 0.0)

    def test_calculate_effect_size_multiple_groups(self, hyperparameter_tester):
        """Test effect size calculation with multiple groups."""
        metric_groups = [[1.0, 1.1, 1.2], [1.5, 1.6, 1.7], [0.8, 0.9, 1.0]]
        stat_result = {"p_value": 0.05}

        effect_size = hyperparameter_tester._calculate_effect_size(metric_groups, stat_result)

        assert isinstance(effect_size, float)
        assert 0 <= effect_size <= 1  # Eta-squared should be between 0 and 1

    def test_calculate_effect_size_insufficient_data(self, hyperparameter_tester):
        """Test effect size calculation with insufficient data."""
        metric_groups = [[1.0], [1.1]]
        stat_result = {"p_value": 0.05}

        effect_size = hyperparameter_tester._calculate_effect_size(metric_groups, stat_result)

        assert effect_size == 0.0

    def test_calculate_effect_size_zero_variance(self, hyperparameter_tester):
        """Test effect size calculation with zero variance."""
        metric_groups = [
            [1.0, 1.0, 1.0],  # No variance within groups
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
        stat_result = {"p_value": 0.05}

        effect_size = hyperparameter_tester._calculate_effect_size(metric_groups, stat_result)

        assert effect_size == 0.0

    def test_analyze_single_hyperparameter_insufficient_data(self, hyperparameter_tester):
        """Test single hyperparameter analysis with insufficient data."""
        from src.evaluation.sensitivity.engine import SensitivityResult

        sensitivity_results = [
            SensitivityResult(
                model_type="test_model",
                parameter_combination={"param1": 1},
                performance_metrics={"sharpe_ratio": 1.0},
                backtest_results=pd.DataFrame(),
                execution_time=10.0,
                memory_usage=100.0,
            )
        ]

        result = hyperparameter_tester._analyze_single_hyperparameter(
            model_type="test_model",
            parameter_name="param1",
            sensitivity_results=sensitivity_results,
            target_metric="sharpe_ratio",
            significance_level=0.05,
        )

        assert result.model_type == "test_model"
        assert result.parameter_name == "param1"
        assert result.parameter_values == [1]
        assert result.metric_values == []
        assert result.statistical_significance == {}
        assert result.effect_size == 0.0
        assert result.optimal_value is None
        assert result.confidence_interval == (0.0, 0.0)

    def test_compare_hyperparameter_importance_no_results(self, hyperparameter_tester):
        """Test hyperparameter importance comparison with no results."""
        with pytest.raises(ValueError, match="No hyperparameter results for test_model"):
            hyperparameter_tester.compare_hyperparameter_importance("test_model")

    def test_compare_hyperparameter_importance_with_results(self, hyperparameter_tester):
        """Test hyperparameter importance comparison with results."""
        # Mock test results
        mock_results = [
            HyperparameterTestResult(
                model_type="test_model",
                parameter_name="param1",
                parameter_values=[1, 2, 3],
                metric_values=[1.0, 1.5, 2.0],
                statistical_significance={"p_value": 0.01},
                effect_size=0.8,
                optimal_value=3,
                confidence_interval=(1.8, 2.2),
            ),
            HyperparameterTestResult(
                model_type="test_model",
                parameter_name="param2",
                parameter_values=["a", "b"],
                metric_values=[1.2, 1.3],
                statistical_significance={"p_value": 0.3},
                effect_size=0.2,
                optimal_value="b",
                confidence_interval=(1.1, 1.5),
            ),
        ]

        hyperparameter_tester.test_results["test_model"] = mock_results

        importance_df = hyperparameter_tester.compare_hyperparameter_importance("test_model")

        # Assertions
        assert len(importance_df) == 2
        assert "parameter" in importance_df.columns
        assert "effect_size" in importance_df.columns
        assert "p_value" in importance_df.columns
        assert "optimal_value" in importance_df.columns
        assert "metric_range" in importance_df.columns
        assert "significance" in importance_df.columns

        # Should be sorted by effect size (descending)
        assert importance_df.iloc[0]["effect_size"] >= importance_df.iloc[1]["effect_size"]
        assert importance_df.iloc[0]["parameter"] == "param1"  # Higher effect size
        assert importance_df.iloc[0]["significance"] == "Yes"  # p < 0.05
        assert importance_df.iloc[1]["significance"] == "No"  # p > 0.05

    def test_get_optimal_hyperparameters_no_results(self, hyperparameter_tester):
        """Test getting optimal hyperparameters with no results."""
        with pytest.raises(ValueError, match="No hyperparameter results for test_model"):
            hyperparameter_tester.get_optimal_hyperparameters("test_model")

    def test_get_optimal_hyperparameters_with_results(self, hyperparameter_tester):
        """Test getting optimal hyperparameters with results."""
        # Mock test results
        mock_results = [
            HyperparameterTestResult(
                model_type="test_model",
                parameter_name="param1",
                parameter_values=[1, 2, 3],
                metric_values=[1.0, 1.5, 2.0],
                statistical_significance={"p_value": 0.01},  # Significant
                effect_size=0.8,
                optimal_value=3,
                confidence_interval=(1.8, 2.2),
            ),
            HyperparameterTestResult(
                model_type="test_model",
                parameter_name="param2",
                parameter_values=["a", "b"],
                metric_values=[1.2, 1.3],
                statistical_significance={"p_value": 0.3},  # Not significant
                effect_size=0.2,
                optimal_value="b",
                confidence_interval=(1.1, 1.5),
            ),
        ]

        hyperparameter_tester.test_results["test_model"] = mock_results

        optimal_params = hyperparameter_tester.get_optimal_hyperparameters("test_model")

        # Only significant parameters should be included
        assert len(optimal_params) == 1
        assert "param1" in optimal_params
        assert optimal_params["param1"] == 3
        assert "param2" not in optimal_params  # Not significant

    def test_get_optimal_hyperparameters_custom_threshold(self, hyperparameter_tester):
        """Test getting optimal hyperparameters with custom significance threshold."""
        # Mock test results
        mock_results = [
            HyperparameterTestResult(
                model_type="test_model",
                parameter_name="param1",
                parameter_values=[1, 2],
                metric_values=[1.0, 1.5],
                statistical_significance={"p_value": 0.08},  # Not significant at 0.05 but at 0.1
                effect_size=0.6,
                optimal_value=2,
                confidence_interval=(1.3, 1.7),
            )
        ]

        hyperparameter_tester.test_results["test_model"] = mock_results

        # With default threshold (0.05) - should be empty
        optimal_params_strict = hyperparameter_tester.get_optimal_hyperparameters("test_model")
        assert len(optimal_params_strict) == 0

        # With relaxed threshold (0.1) - should include param1
        optimal_params_relaxed = hyperparameter_tester.get_optimal_hyperparameters(
            "test_model", significance_threshold=0.1
        )
        assert len(optimal_params_relaxed) == 1
        assert "param1" in optimal_params_relaxed
        assert optimal_params_relaxed["param1"] == 2

    def test_test_multiple_hyperparameters_with_correction(
        self, hyperparameter_tester, mock_dependencies
    ):
        """Test multiple hyperparameter testing with correction."""
        # Mock sensitivity engine results
        from src.evaluation.sensitivity.engine import SensitivityResult

        mock_sensitivity_results = {
            "test_model": [
                SensitivityResult(
                    model_type="test_model",
                    parameter_combination={"param1": 1, "param2": "a"},
                    performance_metrics={"sharpe_ratio": 1.0},
                    backtest_results=pd.DataFrame(),
                    execution_time=10.0,
                    memory_usage=100.0,
                ),
                SensitivityResult(
                    model_type="test_model",
                    parameter_combination={"param1": 2, "param2": "b"},
                    performance_metrics={"sharpe_ratio": 1.5},
                    backtest_results=pd.DataFrame(),
                    execution_time=12.0,
                    memory_usage=120.0,
                ),
            ]
        }

        mock_dependencies["sensitivity_engine"].run_sensitivity_analysis.return_value = (
            mock_sensitivity_results
        )

        # Mock multiple comparison correction
        mock_correction_result = {
            "corrected_p_values": np.array([0.02, 0.06]),
            "rejected": np.array([True, False]),
        }
        mock_dependencies["multiple_comparison_corrector"].bonferroni_correction.return_value = (
            mock_correction_result
        )

        # Mock the test_hyperparameter_sensitivity method to return simple results
        original_results = [
            HyperparameterTestResult(
                model_type="test_model",
                parameter_name="param1",
                parameter_values=[1, 2],
                metric_values=[1.0, 1.5],
                statistical_significance={"p_value": 0.01},
                effect_size=0.5,
                optimal_value=2,
                confidence_interval=(1.2, 1.8),
            ),
            HyperparameterTestResult(
                model_type="test_model",
                parameter_name="param2",
                parameter_values=["a", "b"],
                metric_values=[1.0, 1.5],
                statistical_significance={"p_value": 0.03},
                effect_size=0.3,
                optimal_value="b",
                confidence_interval=(1.2, 1.8),
            ),
        ]

        with patch.object(
            hyperparameter_tester, "test_hyperparameter_sensitivity", return_value=original_results
        ):

            corrected_results = hyperparameter_tester.test_multiple_hyperparameters_with_correction(
                model_type="test_model",
                data=pd.DataFrame(),
                start_date="2023-01-01",
                end_date="2023-12-31",
                correction_method="bonferroni",
            )

        # Assertions
        assert len(corrected_results) == 2

        # Check that corrected p-values and significance flags were added
        assert "corrected_p_value" in corrected_results[0].statistical_significance
        assert "significant_after_correction" in corrected_results[0].statistical_significance
        assert corrected_results[0].statistical_significance["corrected_p_value"] == 0.02
        assert corrected_results[0].statistical_significance["significant_after_correction"]

        assert corrected_results[1].statistical_significance["corrected_p_value"] == 0.06
        assert (
            corrected_results[1].statistical_significance["significant_after_correction"] is False
        )

        # Verify mock calls
        mock_dependencies[
            "multiple_comparison_corrector"
        ].bonferroni_correction.assert_called_once()

    def test_create_hyperparameter_sensitivity_report(self, hyperparameter_tester, tmp_path):
        """Test creating hyperparameter sensitivity report."""
        # Mock test results
        mock_results = [
            HyperparameterTestResult(
                model_type="test_model",
                parameter_name="param1",
                parameter_values=[1, 2, 3],
                metric_values=[1.0, 1.5, 2.0],
                statistical_significance={"p_value": 0.0005},
                effect_size=0.9,
                optimal_value=3,
                confidence_interval=(1.8, 2.2),
            ),
            HyperparameterTestResult(
                model_type="test_model",
                parameter_name="param2",
                parameter_values=["a", "b"],
                metric_values=[1.2, 1.3],
                statistical_significance={"p_value": 0.02},
                effect_size=0.4,
                optimal_value="b",
                confidence_interval=(1.1, 1.5),
            ),
        ]

        hyperparameter_tester.test_results["test_model"] = mock_results

        report_path = tmp_path / "hyperparameter_report.md"

        hyperparameter_tester.create_hyperparameter_sensitivity_report(
            model_type="test_model", output_path=str(report_path)
        )

        # Verify report was created
        assert report_path.exists()

        # Read and check report content
        report_content = report_path.read_text()

        assert "Hyperparameter Sensitivity Analysis Report: TEST_MODEL" in report_content
        assert "Total parameters analyzed: 2" in report_content
        assert "Significant parameters (p < 0.05): 2" in report_content
        assert "param1 ***" in report_content  # p = 0.0005 < 0.001 threshold
        assert "param2 *" in report_content  # p = 0.02 < 0.05 threshold
        assert "Effect size: 0.9000" in report_content
        assert "Optimal value: 3" in report_content
        assert "Recommended Configuration" in report_content
