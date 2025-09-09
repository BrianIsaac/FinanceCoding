"""Unit tests for sensitivity analysis engine."""

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.evaluation.sensitivity.engine import (
    ParameterGrid,
    SensitivityAnalysisEngine,
    SensitivityResult,
)


class TestParameterGrid:
    """Test cases for ParameterGrid class."""

    def test_parameter_grid_initialization(self):
        """Test basic parameter grid initialization."""
        parameters = {"param1": [1, 2, 3], "param2": ["a", "b"]}
        constraints = {"constraint1": True}

        grid = ParameterGrid(
            model_type="test_model", parameters=parameters, constraints=constraints
        )

        assert grid.model_type == "test_model"
        assert grid.parameters == parameters
        assert grid.constraints == constraints

    def test_generate_combinations(self):
        """Test parameter combination generation."""
        parameters = {"param1": [1, 2], "param2": ["a", "b"]}
        constraints = {"fixed_param": True}

        grid = ParameterGrid("test_model", parameters, constraints)
        combinations = grid.generate_combinations()

        expected_combinations = [
            {"param1": 1, "param2": "a", "fixed_param": True},
            {"param1": 1, "param2": "b", "fixed_param": True},
            {"param1": 2, "param2": "a", "fixed_param": True},
            {"param1": 2, "param2": "b", "fixed_param": True},
        ]

        assert len(combinations) == 4
        for combo in expected_combinations:
            assert combo in combinations

    def test_validate_parameters_valid(self):
        """Test parameter validation with valid parameters."""
        grid = ParameterGrid("test_model", {"param1": [1, 2, 3], "param2": ["a", "b"]})

        valid_params = {"param1": 2, "param2": "a", "extra": "value"}
        assert grid.validate_parameters(valid_params) is True

    def test_validate_parameters_invalid(self):
        """Test parameter validation with invalid parameters."""
        grid = ParameterGrid("test_model", {"param1": [1, 2, 3], "param2": ["a", "b"]})

        invalid_params = {"param1": 5, "param2": "a"}  # 5 not in [1, 2, 3]
        assert grid.validate_parameters(invalid_params) is False

    def test_empty_parameter_grid(self):
        """Test behavior with empty parameter grid."""
        grid = ParameterGrid("test_model", {})
        combinations = grid.generate_combinations()

        assert len(combinations) == 1
        assert combinations[0] == {}


class TestSensitivityResult:
    """Test cases for SensitivityResult dataclass."""

    def test_sensitivity_result_creation(self):
        """Test SensitivityResult creation."""
        backtest_data = pd.DataFrame({"returns": [0.01, 0.02, -0.01]})

        result = SensitivityResult(
            model_type="test_model",
            parameter_combination={"param1": 1},
            performance_metrics={"sharpe_ratio": 1.5},
            backtest_results=backtest_data,
            execution_time=10.5,
            memory_usage=100.0,
        )

        assert result.model_type == "test_model"
        assert result.parameter_combination == {"param1": 1}
        assert result.performance_metrics == {"sharpe_ratio": 1.5}
        assert result.execution_time == 10.5
        assert result.memory_usage == 100.0
        assert result.error is None

    def test_sensitivity_result_with_error(self):
        """Test SensitivityResult with error."""
        result = SensitivityResult(
            model_type="test_model",
            parameter_combination={},
            performance_metrics={},
            backtest_results=pd.DataFrame(),
            execution_time=0.0,
            memory_usage=0.0,
            error="Test error",
        )

        assert result.error == "Test error"


class TestSensitivityAnalysisEngine:
    """Test cases for SensitivityAnalysisEngine."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for SensitivityAnalysisEngine."""
        mock_backtest_engine = Mock()
        mock_performance_analytics = Mock()
        mock_statistical_validator = Mock()

        return {
            "backtest_engine": mock_backtest_engine,
            "performance_analytics": mock_performance_analytics,
            "statistical_validator": mock_statistical_validator,
        }

    @pytest.fixture
    def sensitivity_engine(self, mock_dependencies):
        """Create SensitivityAnalysisEngine instance with mocked dependencies."""
        return SensitivityAnalysisEngine(
            backtest_engine=mock_dependencies["backtest_engine"],
            performance_analytics=mock_dependencies["performance_analytics"],
            statistical_validator=mock_dependencies["statistical_validator"],
            max_workers=2,
            memory_limit_gb=8,
        )

    def test_initialization(self, sensitivity_engine, mock_dependencies):
        """Test SensitivityAnalysisEngine initialization."""
        assert sensitivity_engine.backtest_engine == mock_dependencies["backtest_engine"]
        assert (
            sensitivity_engine.performance_analytics == mock_dependencies["performance_analytics"]
        )
        assert (
            sensitivity_engine.statistical_validator == mock_dependencies["statistical_validator"]
        )
        assert sensitivity_engine.max_workers == 2
        assert sensitivity_engine.memory_limit_gb == 8
        assert sensitivity_engine.results == []
        assert sensitivity_engine.parameter_grids == {}

    def test_register_parameter_grid(self, sensitivity_engine):
        """Test parameter grid registration."""
        grid = ParameterGrid("test_model", {"param1": [1, 2]})

        sensitivity_engine.register_parameter_grid("test_model", grid)

        assert "test_model" in sensitivity_engine.parameter_grids
        assert sensitivity_engine.parameter_grids["test_model"] == grid

    def test_create_model_config(self, sensitivity_engine):
        """Test model configuration creation."""
        parameters = {"param1": 1, "param2": "a"}

        config = sensitivity_engine._create_model_config("test_model", parameters)

        expected_config = {"model_type": "test_model", "parameters": parameters}
        assert config == expected_config

    @patch("time.time")
    @patch("psutil.Process")
    def test_execute_parameter_combination_success(
        self, mock_process_class, mock_time, sensitivity_engine, mock_dependencies
    ):
        """Test successful parameter combination execution."""
        # Mock time progression
        mock_time.side_effect = [0.0, 10.0]  # start and end times

        # Mock memory usage
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 100  # 100 MB in bytes
        mock_process.memory_info.return_value = mock_memory_info
        mock_process_class.return_value = mock_process

        # Mock backtest results
        backtest_data = pd.DataFrame({"returns": [0.01, 0.02, -0.01]})
        mock_dependencies["backtest_engine"].run_rolling_backtest.return_value = backtest_data

        # Mock performance metrics
        mock_dependencies["performance_analytics"].calculate_portfolio_metrics.return_value = {
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.1,
        }

        # Execute test
        result = sensitivity_engine._execute_parameter_combination(
            model_type="test_model",
            parameters={"param1": 1},
            data=pd.DataFrame(),
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        # Assertions
        assert result.model_type == "test_model"
        assert result.parameter_combination == {"param1": 1}
        assert result.performance_metrics == {"sharpe_ratio": 1.5, "max_drawdown": -0.1}
        assert result.execution_time == 10.0
        assert result.memory_usage == 0.0  # difference between same memory values
        assert result.error is None

        # Verify mock calls
        mock_dependencies["backtest_engine"].run_backtest.assert_called_once()
        mock_dependencies["performance_analytics"].calculate_all_metrics.assert_called_once_with(
            backtest_data
        )

    @patch("time.time")
    @patch("psutil.Process")
    def test_execute_parameter_combination_error(
        self, mock_process_class, mock_time, sensitivity_engine, mock_dependencies
    ):
        """Test parameter combination execution with error."""
        # Mock time
        mock_time.side_effect = [0.0, 5.0]

        # Mock process
        mock_process = Mock()
        mock_process_class.return_value = mock_process

        # Mock backtest engine to raise exception
        mock_dependencies["backtest_engine"].run_rolling_backtest.side_effect = Exception(
            "Test error"
        )

        # Execute test
        result = sensitivity_engine._execute_parameter_combination(
            model_type="test_model",
            parameters={"param1": 1},
            data=pd.DataFrame(),
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        # Assertions
        assert result.model_type == "test_model"
        assert result.parameter_combination == {"param1": 1}
        assert result.performance_metrics == {}
        assert result.execution_time == 5.0
        assert result.error == "Test error"
        assert isinstance(result.backtest_results, pd.DataFrame)
        assert len(result.backtest_results) == 0

    def test_analyze_parameter_sensitivity_no_results(self, sensitivity_engine):
        """Test parameter sensitivity analysis with no results."""
        with pytest.raises(ValueError, match="No sensitivity analysis results available"):
            sensitivity_engine.analyze_parameter_sensitivity()

    def test_analyze_parameter_sensitivity_with_results(self, sensitivity_engine):
        """Test parameter sensitivity analysis with valid results."""
        # Create mock results
        mock_results = [
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

        sensitivity_engine.results = mock_results

        # Analyze sensitivity
        result_df = sensitivity_engine.analyze_parameter_sensitivity("sharpe_ratio")

        # Assertions
        assert len(result_df) == 2
        assert "model_type" in result_df.columns
        assert "param1" in result_df.columns
        assert "param2" in result_df.columns
        assert "sharpe_ratio" in result_df.columns
        assert "execution_time" in result_df.columns
        assert "memory_usage" in result_df.columns

        assert result_df.iloc[0]["sharpe_ratio"] == 1.0
        assert result_df.iloc[1]["sharpe_ratio"] == 1.5

    def test_calculate_parameter_stability(self, sensitivity_engine):
        """Test parameter stability calculation."""
        # Register parameter grid
        grid = ParameterGrid("test_model", {"param1": [1, 2], "param2": ["a", "b"]})
        sensitivity_engine.register_parameter_grid("test_model", grid)

        # Create mock results with varying performance
        mock_results = [
            SensitivityResult(
                "test_model",
                {"param1": 1, "param2": "a"},
                {"sharpe_ratio": 1.0},
                pd.DataFrame(),
                10.0,
                100.0,
            ),
            SensitivityResult(
                "test_model",
                {"param1": 1, "param2": "b"},
                {"sharpe_ratio": 1.2},
                pd.DataFrame(),
                10.0,
                100.0,
            ),
            SensitivityResult(
                "test_model",
                {"param1": 2, "param2": "a"},
                {"sharpe_ratio": 0.8},
                pd.DataFrame(),
                10.0,
                100.0,
            ),
            SensitivityResult(
                "test_model",
                {"param1": 2, "param2": "b"},
                {"sharpe_ratio": 1.5},
                pd.DataFrame(),
                10.0,
                100.0,
            ),
        ]

        sensitivity_engine.results = mock_results

        # Calculate stability
        stability_scores = sensitivity_engine.calculate_parameter_stability("sharpe_ratio")

        # Assertions
        assert "test_model" in stability_scores
        assert "param1" in stability_scores["test_model"]
        assert "param2" in stability_scores["test_model"]

        # Stability scores should be between 0 and 1
        for _param_name, score in stability_scores["test_model"].items():
            assert 0 <= score <= 1

    def test_get_best_parameters(self, sensitivity_engine):
        """Test getting best parameter combinations."""
        # Register parameter grid
        grid = ParameterGrid("test_model", {"param1": [1, 2]})
        sensitivity_engine.register_parameter_grid("test_model", grid)

        # Create mock results
        mock_results = [
            SensitivityResult(
                "test_model", {"param1": 1}, {"sharpe_ratio": 1.0}, pd.DataFrame(), 10.0, 100.0
            ),
            SensitivityResult(
                "test_model", {"param1": 2}, {"sharpe_ratio": 1.5}, pd.DataFrame(), 12.0, 120.0
            ),
        ]

        sensitivity_engine.results = mock_results

        # Get best parameters
        best_params = sensitivity_engine.get_best_parameters("sharpe_ratio", top_k=2)

        # Assertions
        assert "test_model" in best_params
        assert len(best_params["test_model"]) == 2

        # First result should be the best (highest Sharpe ratio)
        assert best_params["test_model"][0]["parameters"] == {"param1": 2}
        assert best_params["test_model"][0]["performance"]["sharpe_ratio"] == 1.5
        assert best_params["test_model"][0]["execution_time"] == 12.0

        # Second result should be the second best
        assert best_params["test_model"][1]["parameters"] == {"param1": 1}
        assert best_params["test_model"][1]["performance"]["sharpe_ratio"] == 1.0

    def test_export_results_csv(self, sensitivity_engine, tmp_path):
        """Test exporting results to CSV format."""
        # Create mock results
        mock_results = [
            SensitivityResult(
                model_type="test_model",
                parameter_combination={"param1": 1},
                performance_metrics={"sharpe_ratio": 1.0},
                backtest_results=pd.DataFrame(),
                execution_time=10.0,
                memory_usage=100.0,
            )
        ]

        sensitivity_engine.results = mock_results

        # Export to CSV
        csv_path = tmp_path / "results.csv"
        sensitivity_engine.export_results(str(csv_path))

        # Verify file exists and has correct content
        assert csv_path.exists()

        df = pd.read_csv(csv_path)
        assert len(df) == 1
        assert df.iloc[0]["model_type"] == "test_model"
        assert df.iloc[0]["param1"] == 1
        assert df.iloc[0]["sharpe_ratio"] == 1.0
        assert df.iloc[0]["execution_time"] == 10.0
        assert df.iloc[0]["memory_usage"] == 100.0

    def test_export_results_unsupported_format(self, sensitivity_engine, tmp_path):
        """Test exporting results to unsupported format."""
        sensitivity_engine.results = []

        unsupported_path = tmp_path / "results.txt"

        with pytest.raises(ValueError, match="Unsupported file format"):
            sensitivity_engine.export_results(str(unsupported_path))


class TestIntegration:
    """Integration tests for sensitivity analysis components."""

    def test_full_workflow_mock(self):
        """Test complete sensitivity analysis workflow with mocked dependencies."""
        # Create mocks
        mock_backtest_engine = Mock()
        mock_performance_analytics = Mock()
        mock_statistical_validator = Mock()

        # Setup mock returns
        mock_backtest_engine.run_rolling_backtest.return_value = pd.DataFrame(
            {"returns": [0.01, 0.02, -0.01]}
        )
        mock_performance_analytics.calculate_portfolio_metrics.return_value = {"sharpe_ratio": 1.2}

        # Create engine
        engine = SensitivityAnalysisEngine(
            backtest_engine=mock_backtest_engine,
            performance_analytics=mock_performance_analytics,
            statistical_validator=mock_statistical_validator,
            max_workers=1,
        )

        # Register parameter grid
        grid = ParameterGrid("test_model", {"param1": [1, 2]})
        engine.register_parameter_grid("test_model", grid)

        # Mock data
        data = pd.DataFrame({"price": [100, 101, 102]})

        # Mock time and process for parameter execution
        with (
            patch("time.time", side_effect=[0.0, 5.0, 5.0, 10.0]),
            patch("psutil.Process") as mock_process_class,
        ):

            mock_process = Mock()
            mock_memory_info = Mock()
            mock_memory_info.rss = 1024 * 1024 * 100  # 100 MB
            mock_process.memory_info.return_value = mock_memory_info
            mock_process_class.return_value = mock_process

            # Run sensitivity analysis
            results = engine.run_sensitivity_analysis(
                model_types=["test_model"],
                data=data,
                start_date="2023-01-01",
                end_date="2023-12-31",
            )

        # Assertions
        assert "test_model" in results
        assert len(results["test_model"]) == 2  # Two parameter combinations

        for result in results["test_model"]:
            assert result.model_type == "test_model"
            assert result.error is None
            assert "sharpe_ratio" in result.performance_metrics
            assert result.performance_metrics["sharpe_ratio"] == 1.2

        # Verify backtest was called for each parameter combination
        assert mock_backtest_engine.run_rolling_backtest.call_count == 2
        assert mock_performance_analytics.calculate_portfolio_metrics.call_count == 2
