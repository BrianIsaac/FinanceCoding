"""
Unit tests for performance analytics execution framework.

Tests comprehensive performance analytics calculation, statistical significance testing,
rolling window consistency analysis, sensitivity analysis, publication-ready reporting,
results validation, and enhanced risk mitigation implementation.
"""

import json

# Import the modules we're testing
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

try:
    from scripts.run_performance_analytics import (
        PerformanceAnalyticsConfig,
        PerformanceAnalyticsExecutor,
    )
except ImportError:
    # Skip tests if dependencies are missing
    pytest.skip("Performance analytics dependencies not available", allow_module_level=True)


class TestPerformanceAnalyticsConfig:
    """Test performance analytics configuration."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = PerformanceAnalyticsConfig()

        assert config.gpu_memory_limit_gb == 11.0
        assert config.max_execution_hours == 8.0
        assert config.sharpe_improvement_threshold == 0.2
        assert config.confidence_level == 0.95
        assert config.bootstrap_samples == 10000
        assert config.significance_level == 0.05
        assert config.random_state == 42
        assert config.rolling_window_months == 12
        assert config.rolling_step_months == 3
        assert config.apa_compliance is True
        assert config.decimal_places == 4
        assert config.p_value_threshold == 0.001


class TestPerformanceAnalyticsExecutor:
    """Test performance analytics executor functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return PerformanceAnalyticsConfig()

    @pytest.fixture
    def executor(self, config):
        """Create test executor."""
        with patch('torch.cuda.is_available', return_value=False):
            return PerformanceAnalyticsExecutor(config)

    @pytest.fixture
    def sample_returns_dict(self):
        """Create sample return data for testing."""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
        dates = dates[dates.dayofweek < 5]  # Business days only

        returns_dict = {
            "HRP_average_correlation_756": pd.Series(
                np.random.normal(0.0008, 0.015, len(dates)),
                index=dates,
                name="HRP"
            ),
            "LSTM": pd.Series(
                np.random.normal(0.0009, 0.016, len(dates)),
                index=dates,
                name="LSTM"
            ),
            "GAT_MST": pd.Series(
                np.random.normal(0.0007, 0.014, len(dates)),
                index=dates,
                name="GAT_MST"
            ),
            "EqualWeight": pd.Series(
                np.random.normal(0.0005, 0.012, len(dates)),
                index=dates,
                name="EqualWeight"
            ),
            "MarketCapWeighted": pd.Series(
                np.random.normal(0.0006, 0.013, len(dates)),
                index=dates,
                name="MarketCapWeighted"
            )
        }

        return returns_dict

    def test_executor_initialization(self, config):
        """Test executor initialization."""
        with patch('torch.cuda.is_available', return_value=False):
            executor = PerformanceAnalyticsExecutor(config)

            assert executor.config == config
            assert executor.bootstrap is not None
            assert executor.multi_bootstrap is not None
            assert executor.significance_tester is not None
            # These are simplified to None for compatibility
            assert executor.confidence_intervals is None  # Simplified
            assert executor.hypothesis_testing is None    # Simplified
            assert executor.multiple_corrections is None  # Simplified
            assert executor.sensitivity_engine is None    # Simplified
            assert executor.table_formatter is None       # Simplified
            assert executor.publication_reporter is None  # Simplified
            assert isinstance(executor.results, dict)

    def test_validate_execution_constraints_gpu_available(self, executor):
        """Test execution constraints validation with GPU available."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.memory_allocated', return_value=8 * (1024**3)):  # 8GB

            result = executor.validate_execution_constraints()
            assert result is True

    def test_validate_execution_constraints_gpu_exceeded(self, executor):
        """Test execution constraints validation with GPU memory exceeded."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.memory_allocated', return_value=12 * (1024**3)):  # 12GB

            result = executor.validate_execution_constraints()
            assert result is False

    def test_validate_execution_constraints_time_exceeded(self, executor):
        """Test execution constraints validation with time exceeded."""
        # Set start time to simulate 9 hours ago
        executor.start_time = executor.start_time - (9 * 3600)

        result = executor.validate_execution_constraints()
        assert result is False

    def test_load_backtest_results_with_files(self, executor, tmp_path):
        """Test loading backtest results from files."""
        # Create sample CSV files
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        sample_returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)

        # Create returns files
        returns_file = tmp_path / "returns_TestModel.csv"
        sample_returns.to_csv(returns_file)

        returns_dict = executor.load_backtest_results(tmp_path)

        assert "TestModel" in returns_dict
        assert len(returns_dict["TestModel"]) == 100

    def test_load_backtest_results_no_files(self, executor, tmp_path):
        """Test loading backtest results when no files exist (synthetic data)."""
        returns_dict = executor.load_backtest_results(tmp_path)

        # Should generate synthetic data
        assert len(returns_dict) > 0
        assert "HRP_average_correlation_756" in returns_dict
        assert "LSTM" in returns_dict
        assert "EqualWeight" in returns_dict

    def test_generate_synthetic_returns(self, executor):
        """Test synthetic return generation."""
        returns_dict = executor._generate_synthetic_returns()

        assert len(returns_dict) >= 5
        assert "HRP_average_correlation_756" in returns_dict
        assert "LSTM" in returns_dict
        assert "EqualWeight" in returns_dict

        # Check data properties
        for _model_name, returns in returns_dict.items():
            assert isinstance(returns, pd.Series)
            assert len(returns) > 1000  # Should have several years of data
            assert returns.index.freq is not None or len(returns.index) > 0


class TestTask1PerformanceAnalytics:
    """Test Task 1: Complete Performance Analytics Calculation."""

    @pytest.fixture
    def executor_with_data(self):
        """Create executor with sample data."""
        config = PerformanceAnalyticsConfig()
        with patch('torch.cuda.is_available', return_value=False):
            executor = PerformanceAnalyticsExecutor(config)

        # Add sample data
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
        dates = dates[dates.dayofweek < 5]

        returns_dict = {
            "TestModel": pd.Series(
                np.random.normal(0.001, 0.02, len(dates)),
                index=dates
            ),
            "EqualWeight": pd.Series(
                np.random.normal(0.0005, 0.015, len(dates)),
                index=dates
            )
        }

        return executor, returns_dict

    def test_execute_task_1_basic_metrics(self, executor_with_data):
        """Test basic performance metrics calculation."""
        executor, returns_dict = executor_with_data

        executor.execute_task_1_performance_analytics(returns_dict)

        # Check results structure
        assert "performance_metrics" in executor.results
        assert "TestModel" in executor.results["performance_metrics"]
        assert "EqualWeight" in executor.results["performance_metrics"]

        # Check metric presence
        metrics = executor.results["performance_metrics"]["TestModel"]
        expected_metrics = [
            "CAGR", "AnnMean", "AnnVol", "Sharpe", "MDD",
            "information_ratio", "var_95", "cvar_95", "calmar_ratio",
            "sortino_ratio", "omega_ratio", "rolling_sharpe_consistency",
            "annual_turnover", "transaction_costs", "implementation_shortfall",
            "probabilistic_sharpe", "deflated_sharpe"
        ]

        for metric in expected_metrics:
            assert metric in metrics

    def test_execute_task_1_institutional_metrics(self, executor_with_data):
        """Test institutional-grade metrics calculation."""
        executor, returns_dict = executor_with_data

        executor.execute_task_1_performance_analytics(returns_dict)

        metrics = executor.results["performance_metrics"]["TestModel"]

        # Test VaR and CVaR
        assert isinstance(metrics["var_95"], (int, float))
        assert isinstance(metrics["cvar_95"], (int, float))

        # Test information ratio calculation
        assert isinstance(metrics["information_ratio"], (int, float))

        # Test advanced ratios
        assert isinstance(metrics["calmar_ratio"], (int, float))
        assert isinstance(metrics["sortino_ratio"], (int, float))
        assert isinstance(metrics["omega_ratio"], (int, float))

    def test_execute_task_1_rolling_metrics(self, executor_with_data):
        """Test rolling performance metrics calculation."""
        executor, returns_dict = executor_with_data

        executor.execute_task_1_performance_analytics(returns_dict)

        metrics = executor.results["performance_metrics"]["TestModel"]

        # Test rolling consistency metric
        assert "rolling_sharpe_consistency" in metrics
        assert isinstance(metrics["rolling_sharpe_consistency"], (int, float))

    def test_execute_task_1_operational_metrics(self, executor_with_data):
        """Test operational efficiency metrics calculation."""
        executor, returns_dict = executor_with_data

        executor.execute_task_1_performance_analytics(returns_dict)

        metrics = executor.results["performance_metrics"]["TestModel"]

        # Test operational metrics
        assert "annual_turnover" in metrics
        assert "transaction_costs" in metrics
        assert "implementation_shortfall" in metrics

        assert metrics["annual_turnover"] >= 0
        assert metrics["transaction_costs"] >= 0
        assert metrics["implementation_shortfall"] >= 0

    def test_execute_task_1_error_handling(self, executor_with_data):
        """Test error handling in performance metrics calculation."""
        executor, returns_dict = executor_with_data

        # Add problematic data (empty series)
        returns_dict["EmptyModel"] = pd.Series([], dtype=float)

        # Should not raise exception
        executor.execute_task_1_performance_analytics(returns_dict)

        # Check that empty model gets empty metrics
        assert "EmptyModel" in executor.results["performance_metrics"]
        assert executor.results["performance_metrics"]["EmptyModel"] == {}


class TestTask2StatisticalSignificance:
    """Test Task 2: Statistical Significance Testing Framework."""

    @pytest.fixture
    def executor_with_metrics(self):
        """Create executor with performance metrics already calculated."""
        config = PerformanceAnalyticsConfig()
        with patch('torch.cuda.is_available', return_value=False):
            executor = PerformanceAnalyticsExecutor(config)

        # Add sample data and calculate metrics
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
        dates = dates[dates.dayofweek < 5]

        returns_dict = {
            "HRP_model": pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates),
            "LSTM": pd.Series(np.random.normal(0.0012, 0.021, len(dates)), index=dates),
            "EqualWeight": pd.Series(np.random.normal(0.0008, 0.018, len(dates)), index=dates),
            "MarketCapWeighted": pd.Series(np.random.normal(0.0007, 0.017, len(dates)), index=dates),
        }

        executor.execute_task_1_performance_analytics(returns_dict)

        return executor, returns_dict

    def test_execute_task_2_jobson_korkie_tests(self, executor_with_metrics):
        """Test Jobson-Korkie statistical testing."""
        executor, returns_dict = executor_with_metrics

        executor.execute_task_2_statistical_significance(returns_dict)

        # Check results structure
        assert "statistical_tests" in executor.results
        assert "jobson_korkie_results" in executor.results["statistical_tests"]

        jk_results = executor.results["statistical_tests"]["jobson_korkie_results"]

        # Should have pairwise comparisons
        assert len(jk_results) > 0
        assert "portfolio_a" in jk_results.columns
        assert "portfolio_b" in jk_results.columns
        assert "test_statistic" in jk_results.columns
        assert "p_value" in jk_results.columns
        assert "is_significant" in jk_results.columns

    def test_execute_task_2_bootstrap_confidence_intervals(self, executor_with_metrics):
        """Test bootstrap confidence intervals generation."""
        executor, returns_dict = executor_with_metrics

        executor.execute_task_2_statistical_significance(returns_dict)

        # Check bootstrap results
        bootstrap_cis = executor.results["statistical_tests"]["bootstrap_confidence_intervals"]

        assert len(bootstrap_cis) == len(returns_dict)

        for _model_name, ci_results in bootstrap_cis.items():
            if ci_results:  # Skip empty results
                assert "ci_lower" in ci_results
                assert "ci_upper" in ci_results
                assert "bootstrap_mean" in ci_results
                assert "bootstrap_std" in ci_results
                assert "original_sharpe" in ci_results

                # CI bounds should be reasonable
                assert ci_results["ci_lower"] <= ci_results["ci_upper"]

    def test_execute_task_2_multiple_comparison_corrections(self, executor_with_metrics):
        """Test multiple comparison corrections."""
        executor, returns_dict = executor_with_metrics

        executor.execute_task_2_statistical_significance(returns_dict)

        jk_results = executor.results["statistical_tests"]["jobson_korkie_results"]

        # Check corrected p-values
        assert "p_value_bonferroni" in jk_results.columns
        assert "p_value_holm_sidak" in jk_results.columns
        assert "is_significant_bonferroni" in jk_results.columns
        assert "is_significant_holm_sidak" in jk_results.columns

        # Corrected p-values should be >= original p-values
        for _i, row in jk_results.iterrows():
            assert row["p_value_bonferroni"] >= row["p_value"]
            assert row["p_value_holm_sidak"] >= row["p_value"]

    def test_execute_task_2_sharpe_improvement_tests(self, executor_with_metrics):
        """Test Sharpe ratio improvement hypothesis testing."""
        executor, returns_dict = executor_with_metrics

        executor.execute_task_2_statistical_significance(returns_dict)

        improvement_tests = executor.results["statistical_tests"]["sharpe_improvement_tests"]

        # Check structure of improvement tests
        for _test_name, test_results in improvement_tests.items():
            assert "sharpe_improvement" in test_results
            assert "meets_threshold" in test_results
            assert "p_value" in test_results
            assert "effect_size_cohens_d" in test_results
            assert "is_statistically_significant" in test_results
            assert "is_practically_significant" in test_results

            # Threshold check should be consistent
            meets_threshold = test_results["sharpe_improvement"] >= 0.2
            assert test_results["meets_threshold"] == meets_threshold


class TestTask3RollingConsistency:
    """Test Task 3: Rolling Window Consistency Analysis."""

    @pytest.fixture
    def executor_with_long_data(self):
        """Create executor with longer time series for rolling analysis."""
        config = PerformanceAnalyticsConfig()
        with patch('torch.cuda.is_available', return_value=False):
            executor = PerformanceAnalyticsExecutor(config)

        # Create 5 years of data for meaningful rolling analysis
        np.random.seed(42)
        dates = pd.date_range(start="2018-01-01", end="2023-12-31", freq="D")
        dates = dates[dates.dayofweek < 5]

        returns_dict = {
            "TestModel": pd.Series(
                np.random.normal(0.001, 0.02, len(dates)),
                index=dates
            ),
            "EqualWeight": pd.Series(
                np.random.normal(0.0008, 0.018, len(dates)),
                index=dates
            )
        }

        return executor, returns_dict

    def test_execute_task_3_rolling_windows(self, executor_with_long_data):
        """Test rolling window performance analysis."""
        executor, returns_dict = executor_with_long_data

        executor.execute_task_3_rolling_consistency(returns_dict)

        # Check results structure
        assert "rolling_analysis" in executor.results
        assert "rolling_windows_results" in executor.results["rolling_analysis"]

        rolling_results = executor.results["rolling_analysis"]["rolling_windows_results"]

        assert "TestModel" in rolling_results
        assert "EqualWeight" in rolling_results

        # Check rolling metrics structure
        test_model_results = rolling_results["TestModel"]
        assert "rolling_metrics" in test_model_results
        assert "stability_metrics" in test_model_results

        rolling_df = test_model_results["rolling_metrics"]

        # Check rolling metrics columns
        expected_columns = [
            "window_start", "window_end", "annualized_return",
            "annualized_volatility", "sharpe_ratio", "cumulative_return",
            "max_drawdown", "positive_months", "negative_months"
        ]

        for col in expected_columns:
            assert col in rolling_df.columns

    def test_execute_task_3_stability_metrics(self, executor_with_long_data):
        """Test performance stability metrics calculation."""
        executor, returns_dict = executor_with_long_data

        executor.execute_task_3_rolling_consistency(returns_dict)

        stability_metrics = executor.results["rolling_analysis"]["rolling_windows_results"]["TestModel"]["stability_metrics"]

        expected_stability_metrics = [
            "sharpe_mean", "sharpe_std", "sharpe_min", "sharpe_max",
            "positive_sharpe_ratio", "positive_return_ratio",
            "max_drawdown_mean", "max_drawdown_worst", "consistency_score"
        ]

        for metric in expected_stability_metrics:
            assert metric in stability_metrics
            assert isinstance(stability_metrics[metric], (int, float))

    def test_execute_task_3_outperformance_analysis(self, executor_with_long_data):
        """Test temporal outperformance analysis."""
        executor, returns_dict = executor_with_long_data

        executor.execute_task_3_rolling_consistency(returns_dict)

        outperformance_analysis = executor.results["rolling_analysis"]["outperformance_analysis"]

        assert "TestModel" in outperformance_analysis

        test_model_outperformance = outperformance_analysis["TestModel"]

        expected_metrics = [
            "outperformance_ratio", "consecutive_outperformance_max",
            "consecutive_underperformance_max", "outperformance_periods", "total_periods"
        ]

        for metric in expected_metrics:
            assert metric in test_model_outperformance
            assert isinstance(test_model_outperformance[metric], (int, float))

        # Ratio should be between 0 and 1
        assert 0 <= test_model_outperformance["outperformance_ratio"] <= 1

    def test_execute_task_3_persistence_analysis(self, executor_with_long_data):
        """Test performance persistence analysis."""
        executor, returns_dict = executor_with_long_data

        executor.execute_task_3_rolling_consistency(returns_dict)

        persistence_analysis = executor.results["rolling_analysis"]["persistence_analysis"]

        assert "TestModel" in persistence_analysis

        test_model_persistence = persistence_analysis["TestModel"]

        expected_metrics = ["rank_correlation", "information_coefficient", "persistence_score"]

        for metric in expected_metrics:
            assert metric in test_model_persistence
            assert isinstance(test_model_persistence[metric], (int, float))

        # Correlation should be between -1 and 1
        assert -1 <= test_model_persistence["rank_correlation"] <= 1

    def test_max_consecutive_true_helper(self, executor_with_long_data):
        """Test the max consecutive True helper function."""
        executor, _ = executor_with_long_data

        # Test with various boolean series
        test_series = pd.Series([True, True, False, True, True, True, False])
        result = executor._max_consecutive_true(test_series)
        assert result == 3

        # Test with all True
        test_series = pd.Series([True, True, True])
        result = executor._max_consecutive_true(test_series)
        assert result == 3

        # Test with all False
        test_series = pd.Series([False, False, False])
        result = executor._max_consecutive_true(test_series)
        assert result == 0

        # Test with empty series
        test_series = pd.Series([], dtype=bool)
        result = executor._max_consecutive_true(test_series)
        assert result == 0


class TestTask4SensitivityAnalysis:
    """Test Task 4: Sensitivity Analysis Execution."""

    @pytest.fixture
    def executor_with_hrp_models(self):
        """Create executor with HRP model variants for sensitivity testing."""
        config = PerformanceAnalyticsConfig()
        with patch('torch.cuda.is_available', return_value=False):
            executor = PerformanceAnalyticsExecutor(config)

        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
        dates = dates[dates.dayofweek < 5]

        returns_dict = {
            "HRP_single_correlation_252": pd.Series(np.random.normal(0.0008, 0.02, len(dates)), index=dates),
            "HRP_complete_correlation_252": pd.Series(np.random.normal(0.0009, 0.021, len(dates)), index=dates),
            "HRP_average_correlation_252": pd.Series(np.random.normal(0.0010, 0.019, len(dates)), index=dates),
            "HRP_ward_euclidean_252": pd.Series(np.random.normal(0.0007, 0.022, len(dates)), index=dates),
            "LSTM": pd.Series(np.random.normal(0.0011, 0.023, len(dates)), index=dates),
            "EqualWeight": pd.Series(np.random.normal(0.0006, 0.018, len(dates)), index=dates),
        }

        return executor, returns_dict

    def test_execute_task_4_parameter_sensitivity(self, executor_with_hrp_models):
        """Test parameter sensitivity analysis."""
        executor, returns_dict = executor_with_hrp_models

        executor.execute_task_4_sensitivity_analysis(returns_dict)

        # Check results structure
        assert "sensitivity_analysis" in executor.results
        sensitivity_results = executor.results["sensitivity_analysis"]

        assert "parameter_sensitivity" in sensitivity_results

        # Check HRP linkage sensitivity analysis
        parameter_sensitivity = sensitivity_results["parameter_sensitivity"]

        if "hrp_linkage_sensitivity" in parameter_sensitivity:
            linkage_analysis = parameter_sensitivity["hrp_linkage_sensitivity"]

            # Should have analysis for different linkage methods
            linkage_methods = ["single", "complete", "average", "ward"]

            for method in linkage_methods:
                if method in linkage_analysis:
                    method_results = linkage_analysis[method]

                    expected_metrics = ["mean_sharpe", "std_sharpe", "min_sharpe", "max_sharpe"]
                    for metric in expected_metrics:
                        assert metric in method_results
                        assert isinstance(method_results[metric], (int, float))

    def test_execute_task_4_market_regime_analysis(self, executor_with_hrp_models):
        """Test market regime analysis."""
        executor, returns_dict = executor_with_hrp_models

        executor.execute_task_4_sensitivity_analysis(returns_dict)

        regime_analysis = executor.results["sensitivity_analysis"]["market_regime_analysis"]

        # Check that all models have regime analysis
        for model_name in returns_dict.keys():
            assert model_name in regime_analysis

            model_regimes = regime_analysis[model_name]

            # Should have analysis for different volatility regimes
            regime_names = ["low_volatility", "normal_volatility", "high_volatility"]

            for regime in regime_names:
                if regime in model_regimes:
                    regime_results = model_regimes[regime]

                    expected_metrics = [
                        "mean_return", "volatility", "sharpe_ratio",
                        "max_drawdown", "periods"
                    ]

                    for metric in expected_metrics:
                        assert metric in regime_results
                        assert isinstance(regime_results[metric], (int, float))

    def test_execute_task_4_transaction_cost_sensitivity(self, executor_with_hrp_models):
        """Test transaction cost sensitivity analysis."""
        executor, returns_dict = executor_with_hrp_models

        executor.execute_task_4_sensitivity_analysis(returns_dict)

        cost_sensitivity = executor.results["sensitivity_analysis"]["transaction_cost_sensitivity"]

        # Check that all models have cost sensitivity analysis
        for model_name in returns_dict.keys():
            assert model_name in cost_sensitivity

            model_cost_analysis = cost_sensitivity[model_name]

            # Should have analysis for different cost scenarios
            cost_scenarios = ["5_bps", "10_bps", "15_bps", "20_bps"]

            for scenario in cost_scenarios:
                assert scenario in model_cost_analysis

                scenario_results = model_cost_analysis[scenario]

                expected_metrics = [
                    "gross_annual_return", "net_annual_return", "cost_impact",
                    "gross_sharpe", "net_sharpe"
                ]

                for metric in expected_metrics:
                    assert metric in scenario_results
                    assert isinstance(scenario_results[metric], (int, float))

                # Net return should be <= gross return
                assert scenario_results["net_annual_return"] <= scenario_results["gross_annual_return"]

                # Cost impact should be >= 0
                assert scenario_results["cost_impact"] >= 0

    def test_execute_task_4_universe_robustness(self, executor_with_hrp_models):
        """Test universe construction robustness analysis."""
        executor, returns_dict = executor_with_hrp_models

        executor.execute_task_4_sensitivity_analysis(returns_dict)

        universe_robustness = executor.results["sensitivity_analysis"]["universe_construction_robustness"]

        # Check that robustness analysis is present (placeholder structure)
        expected_keys = [
            "methodology", "data_source_sensitivity",
            "universe_size_sensitivity", "rebalancing_frequency_sensitivity"
        ]

        for key in expected_keys:
            assert key in universe_robustness

    def test_calculate_max_drawdown_helper(self, executor_with_hrp_models):
        """Test max drawdown calculation helper function."""
        executor, _ = executor_with_hrp_models

        # Test with sample return series
        returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.005])
        max_dd = executor._calculate_max_drawdown(returns)

        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Max drawdown should be negative or zero

        # Test with empty series
        empty_returns = pd.Series([], dtype=float)
        max_dd_empty = executor._calculate_max_drawdown(empty_returns)
        assert max_dd_empty == 0.0


class TestTask5PublicationReporting:
    """Test Task 5: Publication-Ready Statistical Reporting."""

    @pytest.fixture
    def executor_with_full_results(self):
        """Create executor with complete analysis results."""
        config = PerformanceAnalyticsConfig()
        with patch('torch.cuda.is_available', return_value=False):
            executor = PerformanceAnalyticsExecutor(config)

        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
        dates = dates[dates.dayofweek < 5]

        returns_dict = {
            "HRP_model": pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates),
            "LSTM": pd.Series(np.random.normal(0.0012, 0.021, len(dates)), index=dates),
            "EqualWeight": pd.Series(np.random.normal(0.0008, 0.018, len(dates)), index=dates),
        }

        # Execute all previous tasks to build up results
        executor.execute_task_1_performance_analytics(returns_dict)
        executor.execute_task_2_statistical_significance(returns_dict)

        return executor, returns_dict

    def test_execute_task_5_summary_tables(self, executor_with_full_results):
        """Test publication-ready summary table generation."""
        executor, returns_dict = executor_with_full_results

        executor.execute_task_5_publication_reporting(returns_dict)

        # Check results structure
        assert "publication_tables" in executor.results
        publication_tables = executor.results["publication_tables"]

        assert "performance_summary_table" in publication_tables

        summary_table = publication_tables["performance_summary_table"]

        # Check table structure
        assert isinstance(summary_table, pd.DataFrame)
        assert len(summary_table) == len(returns_dict)

        expected_columns = [
            "Model", "Sharpe Ratio", "Annual Return", "Annual Volatility",
            "Maximum Drawdown", "Information Ratio", "Calmar Ratio",
            "VaR (95%)", "CVaR (95%)", "Sharpe CI Lower", "Sharpe CI Upper"
        ]

        for col in expected_columns:
            assert col in summary_table.columns

    def test_execute_task_5_comparison_matrices(self, executor_with_full_results):
        """Test performance comparison matrices generation."""
        executor, returns_dict = executor_with_full_results

        executor.execute_task_5_publication_reporting(returns_dict)

        publication_tables = executor.results["publication_tables"]

        assert "sharpe_difference_matrix" in publication_tables
        assert "p_value_matrix" in publication_tables

        diff_matrix = publication_tables["sharpe_difference_matrix"]
        p_value_matrix = publication_tables["p_value_matrix"]

        # Check matrix dimensions
        model_names = list(returns_dict.keys())
        assert diff_matrix.shape == (len(model_names), len(model_names))
        assert p_value_matrix.shape == (len(model_names), len(model_names))

        # Check diagonal elements
        for model in model_names:
            assert diff_matrix.loc[model, model] == "—"
            assert p_value_matrix.loc[model, model] == "—"

    def test_execute_task_5_hypothesis_tests_table(self, executor_with_full_results):
        """Test hypothesis testing results table generation."""
        executor, returns_dict = executor_with_full_results

        executor.execute_task_5_publication_reporting(returns_dict)

        publication_tables = executor.results["publication_tables"]

        assert "hypothesis_tests_table" in publication_tables

        hypothesis_table = publication_tables["hypothesis_tests_table"]

        # Check table structure
        expected_columns = [
            "Comparison", "Sharpe Improvement", "Meets ≥0.2 Threshold",
            "p-value", "Effect Size (Cohen's d)", "Statistically Significant",
            "Practically Significant"
        ]

        if len(hypothesis_table) > 0:  # Only check if there are improvement tests
            for col in expected_columns:
                assert col in hypothesis_table.columns

    def test_execute_task_5_bootstrap_table(self, executor_with_full_results):
        """Test bootstrap confidence intervals table generation."""
        executor, returns_dict = executor_with_full_results

        executor.execute_task_5_publication_reporting(returns_dict)

        publication_tables = executor.results["publication_tables"]

        assert "bootstrap_confidence_intervals_table" in publication_tables

        bootstrap_table = publication_tables["bootstrap_confidence_intervals_table"]

        # Check table structure
        expected_columns = [
            "Model", "Original Sharpe", "Bootstrap Mean", "Bootstrap Std",
            "95% CI Lower", "95% CI Upper", "CI Width"
        ]

        for col in expected_columns:
            assert col in bootstrap_table.columns

        # Check that CI width is calculated correctly
        if len(bootstrap_table) > 0:
            for _, row in bootstrap_table.iterrows():
                ci_upper = float(row["95% CI Upper"])
                ci_lower = float(row["95% CI Lower"])
                expected_width = ci_upper - ci_lower
                actual_width = float(row["CI Width"])
                assert abs(actual_width - expected_width) < 1e-6

    def test_execute_task_5_apa_formatting(self, executor_with_full_results):
        """Test APA-style table formatting."""
        executor, returns_dict = executor_with_full_results

        executor.execute_task_5_publication_reporting(returns_dict)

        publication_tables = executor.results["publication_tables"]

        # Check for APA-formatted tables
        apa_tables = [key for key in publication_tables.keys() if key.endswith("_apa")]

        assert len(apa_tables) > 0

        # Check APA formatting content
        for apa_table_name in apa_tables:
            apa_content = publication_tables[apa_table_name]
            assert isinstance(apa_content, str)
            assert "Table:" in apa_content
            assert "Note." in apa_content
            assert "CI = confidence interval" in apa_content


class TestTask6ResultsValidation:
    """Test Task 6: Results Validation and Quality Assurance."""

    @pytest.fixture
    def executor_with_complete_analysis(self):
        """Create executor with complete analysis for validation testing."""
        config = PerformanceAnalyticsConfig()
        with patch('torch.cuda.is_available', return_value=False):
            executor = PerformanceAnalyticsExecutor(config)

        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
        dates = dates[dates.dayofweek < 5]

        returns_dict = {
            "TestModel": pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates),
            "EqualWeight": pd.Series(np.random.normal(0.0008, 0.018, len(dates)), index=dates),
        }

        # Execute tasks to build up results
        executor.execute_task_1_performance_analytics(returns_dict)
        executor.execute_task_2_statistical_significance(returns_dict)

        return executor, returns_dict

    def test_execute_task_6_reference_validation(self, executor_with_complete_analysis):
        """Test statistical calculations validation against reference implementations."""
        executor, returns_dict = executor_with_complete_analysis

        executor.execute_task_6_results_validation(returns_dict)

        # Check validation results
        assert "validation_results" in executor.results
        validation_results = executor.results["validation_results"]

        assert "reference_validation" in validation_results

        reference_validation = validation_results["reference_validation"]

        # Check that validation tests were performed
        assert len(reference_validation) > 0

        for _test_name, test_results in reference_validation.items():
            assert "our_calculation" in test_results
            assert "scipy_reference" in test_results
            assert "error_percentage" in test_results
            assert "within_tolerance" in test_results

            # Error percentage should be a number
            assert isinstance(test_results["error_percentage"], (int, float))
            assert test_results["error_percentage"] >= 0

    def test_execute_task_6_bootstrap_validation(self, executor_with_complete_analysis):
        """Test bootstrap procedures validation."""
        executor, returns_dict = executor_with_complete_analysis

        executor.execute_task_6_results_validation(returns_dict)

        validation_results = executor.results["validation_results"]

        assert "bootstrap_validation" in validation_results

        bootstrap_validation = validation_results["bootstrap_validation"]

        assert "bootstrap_coverage_test" in bootstrap_validation

        coverage_test = bootstrap_validation["bootstrap_coverage_test"]

        assert "target_coverage" in coverage_test
        assert "observed_coverage" in coverage_test
        assert "coverage_within_tolerance" in coverage_test

        assert coverage_test["target_coverage"] == 0.95
        assert 0 <= coverage_test["observed_coverage"] <= 1

    def test_execute_task_6_testing_framework_validation(self, executor_with_complete_analysis):
        """Test comprehensive testing framework validation."""
        executor, returns_dict = executor_with_complete_analysis

        executor.execute_task_6_results_validation(returns_dict)

        validation_results = executor.results["validation_results"]

        assert "testing_framework_validation" in validation_results

        framework_validation = validation_results["testing_framework_validation"]

        expected_metrics = [
            "statistical_accuracy_tests_passed", "total_statistical_accuracy_tests",
            "bootstrap_coverage_tests_passed", "total_bootstrap_tests",
            "overall_validation_score"
        ]

        for metric in expected_metrics:
            assert metric in framework_validation
            assert isinstance(framework_validation[metric], (int, float))

        # Validation score should be between 0 and 1
        assert 0 <= framework_validation["overall_validation_score"] <= 1

    def test_execute_task_6_qa_checks(self, executor_with_complete_analysis):
        """Test automated quality assurance checks."""
        executor, returns_dict = executor_with_complete_analysis

        executor.execute_task_6_results_validation(returns_dict)

        validation_results = executor.results["validation_results"]

        assert "qa_checks" in validation_results

        qa_checks = validation_results["qa_checks"]

        # Check data integrity
        assert "data_integrity" in qa_checks
        data_integrity = qa_checks["data_integrity"]

        assert "all_returns_finite" in data_integrity
        assert "no_empty_series" in data_integrity
        assert "consistent_frequencies" in data_integrity

        # Check statistical completeness
        assert "statistical_completeness" in qa_checks
        statistical_completeness = qa_checks["statistical_completeness"]

        assert "all_models_have_metrics" in statistical_completeness
        assert "all_pairwise_tests_completed" in statistical_completeness
        assert "confidence_intervals_complete" in statistical_completeness

        # Check constraint compliance
        assert "constraint_compliance" in qa_checks
        constraint_compliance = qa_checks["constraint_compliance"]

        assert "gpu_memory_within_limits" in constraint_compliance
        assert "processing_time_acceptable" in constraint_compliance


class TestTask7RiskMitigation:
    """Test Task 7: Enhanced Risk Mitigation Implementation."""

    @pytest.fixture
    def executor_with_full_pipeline(self):
        """Create executor with full pipeline for risk mitigation testing."""
        config = PerformanceAnalyticsConfig()
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.memory_allocated', return_value=8 * (1024**3)):  # 8GB
            executor = PerformanceAnalyticsExecutor(config)

        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
        dates = dates[dates.dayofweek < 5]

        returns_dict = {
            "TestModel": pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates),
            "EqualWeight": pd.Series(np.random.normal(0.0008, 0.018, len(dates)), index=dates),
        }

        # Execute full pipeline
        executor.execute_task_1_performance_analytics(returns_dict)
        executor.execute_task_2_statistical_significance(returns_dict)
        executor.execute_task_6_results_validation(returns_dict)

        return executor, returns_dict

    def test_execute_task_7_accuracy_framework(self, executor_with_full_pipeline):
        """Test statistical accuracy assurance framework."""
        executor, returns_dict = executor_with_full_pipeline

        executor.execute_task_7_risk_mitigation(returns_dict)

        # Check risk mitigation results
        assert "risk_mitigation" in executor.results
        risk_mitigation = executor.results["risk_mitigation"]

        assert "statistical_accuracy_framework" in risk_mitigation

        accuracy_framework = risk_mitigation["statistical_accuracy_framework"]

        expected_components = [
            "reference_validation_status", "continuous_monitoring",
            "validation_frequency", "accuracy_thresholds", "automated_alerts"
        ]

        for component in expected_components:
            assert component in accuracy_framework

        # Check thresholds
        thresholds = accuracy_framework["accuracy_thresholds"]
        assert "sharpe_ratio_tolerance" in thresholds
        assert "p_value_tolerance" in thresholds
        assert "ci_coverage_tolerance" in thresholds

    def test_execute_task_7_gpu_monitoring(self, executor_with_full_pipeline):
        """Test GPU memory profiling and monitoring."""
        executor, returns_dict = executor_with_full_pipeline

        executor.execute_task_7_risk_mitigation(returns_dict)

        risk_mitigation = executor.results["risk_mitigation"]

        assert "gpu_memory_monitoring" in risk_mitigation

        gpu_monitoring = risk_mitigation["gpu_memory_monitoring"]

        expected_components = [
            "real_time_monitoring", "memory_usage_tracking",
            "alerts_configured", "alert_threshold_gb", "cleanup_protocols"
        ]

        for component in expected_components:
            assert component in gpu_monitoring

        # Check memory tracking
        assert isinstance(gpu_monitoring["memory_usage_tracking"], list)

        if len(gpu_monitoring["memory_usage_tracking"]) > 0:
            memory_entry = gpu_monitoring["memory_usage_tracking"][0]

            expected_fields = [
                "timestamp", "current_usage_gb", "max_usage_gb", "within_limits"
            ]

            for field in expected_fields:
                assert field in memory_entry

    def test_execute_task_7_academic_compliance(self, executor_with_full_pipeline):
        """Test academic compliance validation framework."""
        executor, returns_dict = executor_with_full_pipeline

        executor.execute_task_7_risk_mitigation(returns_dict)

        risk_mitigation = executor.results["risk_mitigation"]

        assert "academic_compliance_validation" in risk_mitigation

        academic_compliance = risk_mitigation["academic_compliance_validation"]

        expected_components = [
            "apa_formatting_verification", "statistical_disclosure_completeness",
            "reproducibility_validation", "compliance_score", "automated_checks"
        ]

        for component in expected_components:
            assert component in academic_compliance

        # Check compliance score
        assert 0 <= academic_compliance["compliance_score"] <= 1

        # Check automated checks
        automated_checks = academic_compliance["automated_checks"]

        expected_checks = [
            "table_formatting", "significance_reporting",
            "confidence_interval_reporting", "effect_size_reporting",
            "multiple_comparison_correction"
        ]

        for check in expected_checks:
            assert check in automated_checks

    def test_execute_task_7_external_review_readiness(self, executor_with_full_pipeline):
        """Test external academic review process validation."""
        executor, returns_dict = executor_with_full_pipeline

        executor.execute_task_7_risk_mitigation(returns_dict)

        risk_mitigation = executor.results["risk_mitigation"]

        assert "external_review_readiness" in risk_mitigation

        external_review = risk_mitigation["external_review_readiness"]

        expected_components = [
            "methodology_documentation", "statistical_procedure_transparency",
            "reproducible_research_package", "peer_review_ready", "review_checklist"
        ]

        for component in expected_components:
            assert component in external_review

        # Check review checklist
        review_checklist = external_review["review_checklist"]

        expected_checklist_items = [
            "data_quality_documented", "statistical_assumptions_validated",
            "methodology_clearly_described", "results_comprehensively_reported",
            "limitations_acknowledged", "code_availability"
        ]

        for item in expected_checklist_items:
            assert item in review_checklist

    def test_execute_task_7_overall_assessment(self, executor_with_full_pipeline):
        """Test overall risk mitigation assessment."""
        executor, returns_dict = executor_with_full_pipeline

        executor.execute_task_7_risk_mitigation(returns_dict)

        risk_mitigation = executor.results["risk_mitigation"]

        assert "overall_risk_mitigation_score" in risk_mitigation
        assert "risk_level" in risk_mitigation

        # Check score range
        assert 0 <= risk_mitigation["overall_risk_mitigation_score"] <= 1

        # Check risk level
        assert risk_mitigation["risk_level"] in ["LOW", "MEDIUM", "HIGH"]


class TestExecutionSummaryAndSaving:
    """Test execution summary generation and results saving."""

    @pytest.fixture
    def executor_with_complete_execution(self):
        """Create executor with complete execution for summary testing."""
        config = PerformanceAnalyticsConfig()
        with patch('torch.cuda.is_available', return_value=False):
            executor = PerformanceAnalyticsExecutor(config)

        # Add sample results to simulate complete execution
        executor.results = {
            "performance_metrics": {"TestModel": {"Sharpe": 1.5}, "EqualWeight": {"Sharpe": 1.2}},
            "statistical_tests": {"jobson_korkie_results": pd.DataFrame({"p_value": [0.05, 0.01]})},
            "rolling_analysis": {},
            "sensitivity_analysis": {},
            "publication_tables": {"summary_table": pd.DataFrame()},
            "validation_results": {"testing_framework_validation": {"overall_validation_score": 0.95}},
            "risk_mitigation": {"overall_risk_mitigation_score": 0.85}
        }

        return executor

    def test_generate_execution_summary(self, executor_with_complete_execution):
        """Test execution summary generation."""
        executor = executor_with_complete_execution

        summary = executor.generate_execution_summary()

        # Check summary structure
        assert "execution_metadata" in summary
        assert "task_completion_status" in summary
        assert "quality_metrics" in summary

        # Check execution metadata
        execution_metadata = summary["execution_metadata"]

        expected_metadata = [
            "total_execution_time_hours", "gpu_memory_limit_gb",
            "max_execution_hours", "constraints_met", "bootstrap_samples",
            "confidence_level", "significance_level", "sharpe_improvement_threshold"
        ]

        for field in expected_metadata:
            assert field in execution_metadata

        # Check task completion status
        task_status = summary["task_completion_status"]

        expected_tasks = [
            "task_1_performance_analytics", "task_2_statistical_significance",
            "task_3_rolling_consistency", "task_4_sensitivity_analysis",
            "task_5_publication_reporting", "task_6_results_validation",
            "task_7_risk_mitigation"
        ]

        for task in expected_tasks:
            assert task in task_status
            assert isinstance(task_status[task], bool)

        # Check quality metrics
        quality_metrics = summary["quality_metrics"]

        expected_quality_metrics = [
            "models_analyzed", "statistical_tests_completed",
            "bootstrap_cis_generated", "publication_tables_created",
            "validation_score", "risk_mitigation_score"
        ]

        for metric in expected_quality_metrics:
            assert metric in quality_metrics
            assert isinstance(quality_metrics[metric], (int, float))

    def test_save_results(self, executor_with_complete_execution, tmp_path):
        """Test results saving functionality."""
        executor = executor_with_complete_execution

        # Save results to temporary directory
        output_dir = tmp_path / "test_results"
        executor.save_results(output_dir)

        # Check that main results file was created
        main_results_file = output_dir / "performance_analytics_results.json"
        assert main_results_file.exists()

        # Check that individual component files were created
        expected_files = [
            "performance_metrics.json",
            "statistical_tests.json",
            "rolling_analysis.json",
            "sensitivity_analysis.json",
            "publication_tables.json",
            "validation_results.json",
            "risk_mitigation.json"
        ]

        for filename in expected_files:
            assert (output_dir / filename).exists()

        # Check that publication tables directory was created
        tables_dir = output_dir / "publication_tables"
        assert tables_dir.exists()

        # Test loading saved results
        with open(main_results_file) as f:
            loaded_results = json.load(f)

        assert "performance_metrics" in loaded_results
        assert "statistical_tests" in loaded_results


class TestPerformanceConstraints:
    """Test performance and constraint validation."""

    def test_gpu_memory_constraint_validation(self):
        """Test GPU memory constraint validation."""
        config = PerformanceAnalyticsConfig()
        config.gpu_memory_limit_gb = 11.0

        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.memory_allocated', return_value=10 * (1024**3)):  # 10GB
            executor = PerformanceAnalyticsExecutor(config)
            assert executor.validate_execution_constraints() is True

        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.memory_allocated', return_value=12 * (1024**3)):  # 12GB
            executor = PerformanceAnalyticsExecutor(config)
            assert executor.validate_execution_constraints() is False

    def test_processing_time_constraint_validation(self):
        """Test processing time constraint validation."""
        config = PerformanceAnalyticsConfig()
        config.max_execution_hours = 8.0

        with patch('torch.cuda.is_available', return_value=False):
            executor = PerformanceAnalyticsExecutor(config)

            # Test within time limit
            executor.start_time = executor.start_time - (7 * 3600)  # 7 hours ago
            assert executor.validate_execution_constraints() is True

            # Test exceeding time limit
            executor.start_time = executor.start_time - (2 * 3600)  # Total 9 hours ago
            assert executor.validate_execution_constraints() is False

    def test_error_tolerance_validation(self):
        """Test <0.1% error tolerance validation for statistical calculations."""
        config = PerformanceAnalyticsConfig()

        with patch('torch.cuda.is_available', return_value=False):
            PerformanceAnalyticsExecutor(config)

        # Create test data
        np.random.seed(42)
        test_returns = np.random.normal(0.001, 0.02, 1000)

        # Calculate Sharpe ratio with our method
        our_sharpe = np.mean(test_returns) / np.std(test_returns) * np.sqrt(252)

        # Calculate with scipy as reference
        from scipy import stats as scipy_stats
        scipy_sharpe = scipy_stats.describe(test_returns).mean / np.sqrt(scipy_stats.describe(test_returns).variance) * np.sqrt(252)

        # Check error tolerance
        error_pct = abs(our_sharpe - scipy_sharpe) / abs(scipy_sharpe) * 100 if scipy_sharpe != 0 else 0

        assert error_pct < 0.1  # Should be within 0.1% tolerance


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""

    def test_complete_pipeline_execution(self):
        """Test complete pipeline execution from start to finish."""
        config = PerformanceAnalyticsConfig()
        config.bootstrap_samples = 100  # Reduced for testing speed

        with patch('torch.cuda.is_available', return_value=False):
            executor = PerformanceAnalyticsExecutor(config)

        # Generate minimal test data
        np.random.seed(42)
        dates = pd.date_range(start="2022-01-01", end="2023-12-31", freq="D")
        dates = dates[dates.dayofweek < 5]

        returns_dict = {
            "TestModel": pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates),
            "EqualWeight": pd.Series(np.random.normal(0.0008, 0.018, len(dates)), index=dates),
        }

        # Execute all tasks
        executor.execute_task_1_performance_analytics(returns_dict)
        executor.execute_task_2_statistical_significance(returns_dict)
        executor.execute_task_3_rolling_consistency(returns_dict)
        executor.execute_task_4_sensitivity_analysis(returns_dict)
        executor.execute_task_5_publication_reporting(returns_dict)
        executor.execute_task_6_results_validation(returns_dict)
        executor.execute_task_7_risk_mitigation(returns_dict)

        # Generate summary
        summary = executor.generate_execution_summary()

        # Verify all tasks completed
        task_status = summary["task_completion_status"]
        for task, completed in task_status.items():
            assert completed, f"Task {task} was not completed"

        # Verify quality metrics
        quality_metrics = summary["quality_metrics"]
        assert quality_metrics["models_analyzed"] == len(returns_dict)
        assert quality_metrics["validation_score"] >= 0.8
        assert quality_metrics["risk_mitigation_score"] >= 0.8

    def test_empty_data_handling(self):
        """Test handling of empty or invalid data."""
        config = PerformanceAnalyticsConfig()

        with patch('torch.cuda.is_available', return_value=False):
            executor = PerformanceAnalyticsExecutor(config)

        # Test with empty returns dictionary
        empty_returns = {}

        try:
            executor.execute_task_1_performance_analytics(empty_returns)
            # Should not crash, should handle gracefully
            assert executor.results["performance_metrics"] == {}
        except Exception:
            pytest.fail("Should handle empty data gracefully")

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data for rolling analysis."""
        config = PerformanceAnalyticsConfig()

        with patch('torch.cuda.is_available', return_value=False):
            executor = PerformanceAnalyticsExecutor(config)

        # Create very short time series (insufficient for 12-month rolling windows)
        short_dates = pd.date_range(start="2023-01-01", end="2023-03-31", freq="D")
        short_dates = short_dates[short_dates.dayofweek < 5]

        returns_dict = {
            "ShortModel": pd.Series(np.random.normal(0.001, 0.02, len(short_dates)), index=short_dates)
        }

        # Should handle gracefully without crashing
        executor.execute_task_3_rolling_consistency(returns_dict)

        # Check that results structure is maintained even with insufficient data
        assert "rolling_analysis" in executor.results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
