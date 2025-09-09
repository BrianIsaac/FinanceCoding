"""
Tests for comprehensive performance analysis framework.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.evaluation.reporting.performance_analysis import (
    ComprehensivePerformanceAnalyzer,
    PerformanceAnalysisConfig,
)


@pytest.fixture
def performance_analyzer():
    """Create performance analyzer for testing."""
    config = PerformanceAnalysisConfig(
        confidence_levels=[0.90, 0.95, 0.99],
        bootstrap_iterations=1000,  # Reduced for testing
        decimal_places=4,
        risk_free_rate=0.02,
    )
    return ComprehensivePerformanceAnalyzer(config)


@pytest.fixture
def sample_performance_data():
    """Sample performance data for testing."""
    np.random.seed(42)  # For reproducible tests
    dates = pd.date_range("2020-01-01", periods=1000, freq="D")

    return {
        "HRP": pd.DataFrame(
            {
                "returns": np.random.normal(0.0008, 0.012, 1000),
                "rolling_sharpe": np.random.normal(1.2, 0.3, 1000),
                "rolling_returns": np.random.normal(0.12, 0.05, 1000),
            },
            index=dates,
        ),
        "LSTM": pd.DataFrame(
            {
                "returns": np.random.normal(0.001, 0.015, 1000),
                "rolling_sharpe": np.random.normal(1.8, 0.4, 1000),
                "rolling_returns": np.random.normal(0.18, 0.08, 1000),
            },
            index=dates,
        ),
        "GAT": pd.DataFrame(
            {
                "returns": np.random.normal(0.0009, 0.013, 1000),
                "rolling_sharpe": np.random.normal(1.5, 0.35, 1000),
                "rolling_returns": np.random.normal(0.15, 0.06, 1000),
            },
            index=dates,
        ),
    }


@pytest.fixture
def sample_statistical_data():
    """Sample statistical test results."""
    return {
        "HRP": {
            "pvalue_vs_equal_weight": 0.08,
            "pvalue_vs_market_cap": 0.12,
            "jobson_korkie_pvalue": 0.09,
            "bootstrap_pvalue": 0.10,
            "effect_size": 0.3,
            "bonferroni_adjusted_pvalue": 0.16,
            "fdr_adjusted_pvalue": 0.10,
            "bootstrap_results": {
                "sharpe_90_ci": [1.0, 1.4],
                "sharpe_95_ci": [0.9, 1.5],
                "sharpe_99_ci": [0.8, 1.6],
                "return_95_ci": [0.10, 0.14],
                "volatility_95_ci": [0.13, 0.17],
                "drawdown_95_ci": [0.06, 0.12],
                "iterations": 1000,
            },
        },
        "LSTM": {
            "pvalue_vs_equal_weight": 0.01,
            "pvalue_vs_market_cap": 0.02,
            "jobson_korkie_pvalue": 0.015,
            "bootstrap_pvalue": 0.012,
            "effect_size": 0.8,
            "bonferroni_adjusted_pvalue": 0.03,
            "fdr_adjusted_pvalue": 0.018,
            "bootstrap_results": {
                "sharpe_90_ci": [1.6, 2.0],
                "sharpe_95_ci": [1.5, 2.1],
                "sharpe_99_ci": [1.4, 2.2],
                "return_95_ci": [0.16, 0.20],
                "volatility_95_ci": [0.14, 0.18],
                "drawdown_95_ci": [0.08, 0.16],
                "iterations": 1000,
            },
        },
        "GAT": {
            "pvalue_vs_equal_weight": 0.04,
            "pvalue_vs_market_cap": 0.06,
            "jobson_korkie_pvalue": 0.05,
            "bootstrap_pvalue": 0.045,
            "effect_size": 0.5,
            "bonferroni_adjusted_pvalue": 0.08,
            "fdr_adjusted_pvalue": 0.055,
            "bootstrap_results": {
                "sharpe_90_ci": [1.3, 1.7],
                "sharpe_95_ci": [1.2, 1.8],
                "sharpe_99_ci": [1.1, 1.9],
                "return_95_ci": [0.13, 0.17],
                "volatility_95_ci": [0.12, 0.16],
                "drawdown_95_ci": [0.07, 0.13],
                "iterations": 1000,
            },
        },
    }


class TestComprehensivePerformanceAnalyzer:
    """Test cases for ComprehensivePerformanceAnalyzer."""

    def test_initialization(self):
        """Test performance analyzer initialization."""
        config = PerformanceAnalysisConfig(bootstrap_iterations=5000)
        analyzer = ComprehensivePerformanceAnalyzer(config)

        assert analyzer.config.bootstrap_iterations == 5000
        assert analyzer.config.confidence_levels == [0.90, 0.95, 0.99]
        assert analyzer.config.risk_free_rate == 0.02

    def test_summary_metrics_calculation(self, performance_analyzer, sample_performance_data):
        """Test summary metrics calculation."""
        hrp_data = sample_performance_data["HRP"]
        metrics = performance_analyzer._calculate_summary_metrics(hrp_data)

        # Check required metrics are calculated
        required_metrics = [
            "annual_return",
            "volatility",
            "sharpe_ratio",
            "information_ratio",
            "max_drawdown",
            "calmar_ratio",
            "sortino_ratio",
            "win_rate",
            "avg_win_loss",
        ]
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert not np.isnan(metrics[metric])

        # Check metric ranges
        assert metrics["win_rate"] >= 0 and metrics["win_rate"] <= 1
        assert metrics["volatility"] > 0
        assert metrics["max_drawdown"] <= 0

    def test_performance_summary_table_creation(
        self, performance_analyzer, sample_performance_data, sample_statistical_data
    ):
        """Test performance summary table creation."""
        summary_table = performance_analyzer._create_performance_summary_table(
            sample_performance_data, sample_statistical_data
        )

        assert not summary_table.empty
        assert len(summary_table) == 3  # Three models

        # Check required columns
        required_cols = [
            "Model",
            "Sharpe Ratio",
            "Information Ratio",
            "Annual Return (%)",
            "Volatility (%)",
            "Max Drawdown (%)",
            "Calmar Ratio",
            "Sortino Ratio",
        ]
        for col in required_cols:
            assert col in summary_table.columns

        # Check statistical significance indicators
        lstm_sharpe = summary_table[summary_table["Model"] == "LSTM"]["Sharpe Ratio"].iloc[0]
        assert "*" in lstm_sharpe  # Should have significance indicator

    def test_risk_adjusted_analysis_table(
        self, performance_analyzer, sample_performance_data, sample_statistical_data
    ):
        """Test risk-adjusted analysis table creation."""
        risk_table = performance_analyzer._create_risk_adjusted_analysis_table(
            sample_performance_data, sample_statistical_data
        )

        assert not risk_table.empty
        assert len(risk_table) == 3

        # Check risk metrics columns
        risk_cols = [
            "VaR 95% (%)",
            "CVaR 95% (%)",
            "VaR 99% (%)",
            "CVaR 99% (%)",
            "Tail Ratio",
            "Skewness",
            "Kurtosis",
        ]
        for col in risk_cols:
            assert col in risk_table.columns

        # Check confidence intervals
        assert "Sharpe 95% CI" in risk_table.columns

    def test_statistical_significance_table(
        self, performance_analyzer, sample_performance_data, sample_statistical_data
    ):
        """Test statistical significance table creation."""
        sig_table = performance_analyzer._create_statistical_significance_table(
            sample_performance_data, sample_statistical_data
        )

        assert not sig_table.empty
        assert len(sig_table) == 3

        # Check statistical test columns
        stat_cols = [
            "vs Equal Weight p-value",
            "vs Market Cap p-value",
            "Jobson-Korkie Test",
            "Bootstrap p-value",
            "Effect Size",
            "Significance Level",
        ]
        for col in stat_cols:
            assert col in sig_table.columns

        # Check significance classification
        lstm_significance = sig_table[sig_table["Model"] == "LSTM"]["Significance Level"].iloc[0]
        assert "Significant" in lstm_significance

    def test_consistency_analysis_table(self, performance_analyzer, sample_performance_data):
        """Test consistency analysis table creation."""
        consistency_table = performance_analyzer._create_consistency_analysis_table(
            sample_performance_data
        )

        assert not consistency_table.empty
        assert len(consistency_table) == 3

        # Check consistency metrics
        consistency_cols = [
            "Avg Rolling Sharpe",
            "Sharpe Volatility",
            "Consistency Score",
            "Positive Periods (%)",
            "Performance Persistence",
        ]
        for col in consistency_cols:
            assert col in consistency_table.columns

        # Check metric ranges
        for _, row in consistency_table.iterrows():
            consistency_score = float(row["Consistency Score"])
            assert consistency_score >= 0  # Consistency score should be non-negative

    def test_confidence_intervals_table(
        self, performance_analyzer, sample_performance_data, sample_statistical_data
    ):
        """Test confidence intervals table creation."""
        ci_table = performance_analyzer._create_confidence_intervals_table(
            sample_performance_data, sample_statistical_data
        )

        assert not ci_table.empty
        assert len(ci_table) == 3

        # Check confidence interval columns
        ci_cols = [
            "Sharpe Ratio 90% CI",
            "Sharpe Ratio 95% CI",
            "Sharpe Ratio 99% CI",
            "Annual Return 95% CI",
            "Volatility 95% CI",
        ]
        for col in ci_cols:
            assert col in ci_table.columns

        # Check interval formatting
        lstm_ci = ci_table[ci_table["Model"] == "LSTM"]["Sharpe Ratio 95% CI"].iloc[0]
        assert "[" in lstm_ci and "]" in lstm_ci

    def test_comprehensive_table_generation(
        self, performance_analyzer, sample_performance_data, sample_statistical_data, tmp_path
    ):
        """Test comprehensive performance table generation."""
        tables = performance_analyzer.generate_comprehensive_performance_tables(
            sample_performance_data, sample_statistical_data, output_dir=tmp_path
        )

        # Check all expected tables are generated
        expected_tables = [
            "performance_summary",
            "risk_adjusted_analysis",
            "statistical_significance",
            "consistency_analysis",
            "confidence_intervals",
        ]
        for table_name in expected_tables:
            assert table_name in tables
            assert not tables[table_name].empty

        # Check files are saved
        for table_name in expected_tables:
            assert (tmp_path / f"{table_name}.csv").exists()
            assert (tmp_path / f"{table_name}.html").exists()

    def test_significance_indicator_logic(self, performance_analyzer, sample_statistical_data):
        """Test statistical significance indicator logic."""
        # LSTM should get strong significance indicator
        lstm_indicator = performance_analyzer._get_significance_indicator(
            "LSTM", sample_statistical_data
        )
        assert lstm_indicator == "**"  # p < 0.01

        # HRP should get no indicator
        performance_analyzer._get_significance_indicator("HRP", sample_statistical_data)
        assert lstm_indicator == ""  # p > 0.05

    def test_significance_classification(self, performance_analyzer):
        """Test significance level classification."""
        assert performance_analyzer._classify_significance(0.0005) == "Highly Significant (p<0.001)"
        assert performance_analyzer._classify_significance(0.005) == "Very Significant (p<0.01)"
        assert performance_analyzer._classify_significance(0.03) == "Significant (p<0.05)"
        assert (
            performance_analyzer._classify_significance(0.08) == "Marginally Significant (p<0.10)"
        )
        assert performance_analyzer._classify_significance(0.15) == "Not Significant (p≥0.10)"

    def test_confidence_interval_formatting(self, performance_analyzer):
        """Test confidence interval formatting."""
        # Regular formatting
        ci_regular = performance_analyzer._format_confidence_interval([1.2, 1.8])
        assert ci_regular == "[1.2000, 1.8000]"

        # Percentage formatting
        ci_percent = performance_analyzer._format_confidence_interval([0.12, 0.18], percentage=True)
        assert ci_percent == "[12.00%, 18.00%]"

        # Invalid interval
        ci_invalid = performance_analyzer._format_confidence_interval([np.nan, 1.5])
        assert ci_invalid == "N/A"

    def test_empty_data_handling(self, performance_analyzer):
        """Test handling of empty or invalid data."""
        empty_data = {}
        empty_statistical = {}

        tables = performance_analyzer.generate_comprehensive_performance_tables(
            empty_data, empty_statistical
        )

        # Should return empty tables gracefully
        for _table_name, table_df in tables.items():
            assert isinstance(table_df, pd.DataFrame)

    def test_attribution_analysis_with_sector_data(self, performance_analyzer):
        """Test attribution analysis with sector data."""
        # Create performance data with sector information
        perf_data_with_sectors = {
            "HRP": pd.DataFrame(
                {
                    "returns": np.random.normal(0.0008, 0.012, 100),
                    "sector_returns": np.random.normal(0.0006, 0.008, 100),
                    "benchmark_returns": np.random.normal(0.0005, 0.010, 100),
                }
            )
        }

        attribution_table = performance_analyzer._create_attribution_analysis_table(
            perf_data_with_sectors
        )

        assert not attribution_table.empty
        assert "Sector Contribution" in attribution_table.columns
        assert "Alpha (%)" in attribution_table.columns
        assert "Beta" in attribution_table.columns

    def test_config_validation(self):
        """Test configuration validation and defaults."""
        # Test default config
        config = PerformanceAnalysisConfig()
        assert config.confidence_levels == [0.90, 0.95, 0.99]
        assert config.bootstrap_iterations == 10000
        assert config.risk_free_rate == 0.02

        # Test custom config
        custom_config = PerformanceAnalysisConfig(
            confidence_levels=[0.95], bootstrap_iterations=5000, risk_free_rate=0.025
        )
        assert custom_config.confidence_levels == [0.95]
        assert custom_config.bootstrap_iterations == 5000
        assert custom_config.risk_free_rate == 0.025

    def test_metrics_calculation_edge_cases(self, performance_analyzer):
        """Test edge cases in metrics calculation."""
        # Zero volatility case
        zero_vol_data = pd.DataFrame({"returns": np.zeros(100)})
        metrics = performance_analyzer._calculate_summary_metrics(zero_vol_data)
        assert metrics["volatility"] == 0
        assert metrics["sharpe_ratio"] == 0

        # All negative returns case
        negative_data = pd.DataFrame({"returns": np.random.normal(-0.001, 0.005, 100)})
        metrics = performance_analyzer._calculate_summary_metrics(negative_data)
        assert metrics["annual_return"] < 0
        assert metrics["win_rate"] < 0.5

    def test_table_saving_functionality(
        self, performance_analyzer, sample_performance_data, tmp_path
    ):
        """Test table saving functionality."""
        tables = {"test_table": pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})}

        performance_analyzer._save_performance_tables(tables, tmp_path)

        assert (tmp_path / "test_table.csv").exists()
        assert (tmp_path / "test_table.html").exists()

        # Verify content
        saved_df = pd.read_csv(tmp_path / "test_table.csv")
        assert len(saved_df) == 3
        assert list(saved_df.columns) == ["A", "B"]
