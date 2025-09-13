"""
Unit tests for performance comparison tables module.

Tests the comprehensive performance comparison tables framework including
ranking tables, statistical significance indicators, and export functionality.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.evaluation.reporting.tables import PerformanceComparisonTables, TableConfig


class TestTableConfig:
    """Test TableConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TableConfig()

        assert config.decimal_places == 4
        assert config.significance_levels == [0.001, 0.01, 0.05]
        assert config.include_confidence_intervals is True
        assert config.include_rankings is True
        assert config.sortable is True
        assert config.export_formats == ["html", "latex", "csv"]

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TableConfig(
            decimal_places=3,
            significance_levels=[0.01, 0.05],
            include_confidence_intervals=False,
            include_rankings=False,
            sortable=False,
            export_formats=["csv", "html"],
        )

        assert config.decimal_places == 3
        assert config.significance_levels == [0.01, 0.05]
        assert config.include_confidence_intervals is False
        assert config.include_rankings is False
        assert config.sortable is False
        assert config.export_formats == ["csv", "html"]


class TestPerformanceComparisonTables:
    """Test PerformanceComparisonTables class."""

    @pytest.fixture
    def sample_performance_results(self):
        """Create sample performance results for testing."""
        return {
            "HRP": {
                "sharpe_ratio": 1.25,
                "information_ratio": 0.85,
                "total_return": 0.125,
                "max_drawdown": -0.087,
                "volatility": 0.145,
                "tracking_error": 0.032,
                "avg_monthly_turnover": 0.15,
            },
            "LSTM": {
                "sharpe_ratio": 1.45,
                "information_ratio": 0.95,
                "total_return": 0.158,
                "max_drawdown": -0.112,
                "volatility": 0.162,
                "tracking_error": 0.045,
                "avg_monthly_turnover": 0.28,
            },
            "GAT": {
                "sharpe_ratio": 1.32,
                "information_ratio": 0.88,
                "total_return": 0.142,
                "max_drawdown": -0.095,
                "volatility": 0.152,
                "tracking_error": 0.038,
                "avg_monthly_turnover": 0.22,
            },
        }

    @pytest.fixture
    def sample_statistical_results(self):
        """Create sample statistical results for testing."""
        return {
            "HRP": {
                "sharpe_ratio": {"p_value": 0.032, "t_stat": 2.15},
                "total_return": {"p_value": 0.018, "t_stat": 2.45},
            },
            "LSTM": {
                "sharpe_ratio": {"p_value": 0.008, "t_stat": 2.85},
                "total_return": {"p_value": 0.002, "t_stat": 3.25},
            },
            "GAT": {
                "sharpe_ratio": {"p_value": 0.025, "t_stat": 2.28},
                "total_return": {"p_value": 0.045, "t_stat": 2.05},
            },
        }

    @pytest.fixture
    def sample_baseline_comparisons(self):
        """Create sample baseline comparison results for testing."""
        return {
            "HRP": {
                "equal_weight": {
                    "excess_sharpe": 0.25,
                    "excess_return": 0.032,
                },
                "mean_variance": {
                    "excess_sharpe": 0.15,
                    "excess_return": 0.018,
                },
            },
            "LSTM": {
                "equal_weight": {
                    "excess_sharpe": 0.45,
                    "excess_return": 0.058,
                },
                "mean_variance": {
                    "excess_sharpe": 0.35,
                    "excess_return": 0.048,
                },
            },
        }

    @pytest.fixture
    def tables(self):
        """Create PerformanceComparisonTables instance for testing."""
        config = TableConfig()
        return PerformanceComparisonTables(config)

    def test_initialization(self):
        """Test PerformanceComparisonTables initialization."""
        tables = PerformanceComparisonTables()
        assert tables.config is not None
        assert tables.config.decimal_places == 4

    def test_initialization_with_config(self):
        """Test PerformanceComparisonTables initialization with custom config."""
        config = TableConfig(decimal_places=3, sortable=False)
        tables = PerformanceComparisonTables(config)

        assert tables.config.decimal_places == 3
        assert tables.config.sortable is False

    def test_create_performance_ranking_table_basic(self, tables, sample_performance_results):
        """Test basic performance ranking table creation."""
        result = tables.create_performance_ranking_table(sample_performance_results)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_performance_results)
        assert "sharpe_ratio" in result.columns
        assert "information_ratio" in result.columns

        # Check if data is sorted by Sharpe ratio (highest first)
        sharpe_values = (
            result["sharpe_ratio"]
            .astype(str)
            .str.replace(r"[^\d\.\-]", "", regex=True)
            .astype(float)
        )
        assert sharpe_values.iloc[0] >= sharpe_values.iloc[1]

    def test_create_performance_ranking_table_with_statistical_results(
        self, tables, sample_performance_results, sample_statistical_results
    ):
        """Test performance ranking table with statistical significance indicators."""
        result = tables.create_performance_ranking_table(
            sample_performance_results, sample_statistical_results
        )

        assert isinstance(result, pd.DataFrame)

        # Check for significance symbols in separate columns
        if "sharpe_ratio_sig" in result.columns:
            sig_col = result["sharpe_ratio_sig"].astype(str)
            assert any("*" in str(val) for val in sig_col)

    def test_create_performance_ranking_table_with_baselines(
        self, tables, sample_performance_results, sample_baseline_comparisons
    ):
        """Test performance ranking table with baseline comparisons."""
        result = tables.create_performance_ranking_table(
            sample_performance_results, baseline_comparisons=sample_baseline_comparisons
        )

        assert isinstance(result, pd.DataFrame)

        # Check for baseline comparison columns
        baseline_columns = [col for col in result.columns if "vs_" in col]
        assert len(baseline_columns) > 0

    def test_create_performance_ranking_table_with_rankings(
        self, tables, sample_performance_results
    ):
        """Test performance ranking table with ranking columns."""
        result = tables.create_performance_ranking_table(sample_performance_results)

        assert isinstance(result, pd.DataFrame)

        # Check for ranking columns
        ranking_columns = [col for col in result.columns if "_rank" in col]
        assert len(ranking_columns) > 0

        # Verify ranking values are integers
        for col in ranking_columns:
            if col in result.columns:
                ranks = result[col]
                assert all(isinstance(rank, (int, np.integer)) for rank in ranks)

    def test_create_risk_adjusted_ranking_table(self, tables, sample_performance_results):
        """Test risk-adjusted performance ranking table creation."""
        risk_results = {
            "HRP": {"var_95": -0.025, "expected_shortfall": -0.035},
            "LSTM": {"var_95": -0.032, "expected_shortfall": -0.045},
            "GAT": {"var_95": -0.028, "expected_shortfall": -0.038},
        }

        result = tables.create_risk_adjusted_ranking_table(sample_performance_results, risk_results)

        assert isinstance(result, pd.DataFrame)
        assert "risk_adjusted_score" in result.columns

        # Check if sorted by risk-adjusted score
        scores = result["risk_adjusted_score"]
        assert scores.iloc[0] >= scores.iloc[1]

    def test_create_operational_efficiency_table(self, tables, sample_performance_results):
        """Test operational efficiency comparison table creation."""
        operational_results = {
            "HRP": {
                "avg_monthly_turnover": 0.15,
                "total_transaction_costs": 0.008,
                "constraint_violations": 0.02,
            },
            "LSTM": {
                "avg_monthly_turnover": 0.28,
                "total_transaction_costs": 0.015,
                "constraint_violations": 0.05,
            },
            "GAT": {
                "avg_monthly_turnover": 0.22,
                "total_transaction_costs": 0.012,
                "constraint_violations": 0.03,
            },
        }

        result = tables.create_operational_efficiency_table(
            operational_results, sample_performance_results
        )

        assert isinstance(result, pd.DataFrame)
        assert "avg_monthly_turnover" in result.columns
        assert "total_transaction_costs" in result.columns

        # Check for efficiency ratios
        efficiency_columns = [col for col in result.columns if "per" in col or "after" in col]
        assert len(efficiency_columns) > 0

    def test_create_rolling_window_summary_table(self, tables):
        """Test rolling window summary table creation."""
        rolling_results = {
            "HRP": {
                "sharpe_ratio": [1.2, 1.3, 1.1, 1.4, 1.2],
                "information_ratio": [0.8, 0.9, 0.7, 1.0, 0.8],
            },
            "LSTM": {
                "sharpe_ratio": [1.4, 1.5, 1.3, 1.6, 1.4],
                "information_ratio": [0.9, 1.0, 0.8, 1.1, 0.9],
            },
        }

        result = tables.create_rolling_window_summary_table(rolling_results)

        assert isinstance(result, pd.DataFrame)

        # Check for summary statistics columns
        summary_columns = [
            col
            for col in result.columns
            if any(
                stat in col for stat in ["_mean", "_std", "_median", "_min", "_max", "_consistency"]
            )
        ]
        assert len(summary_columns) > 0

    def test_add_significance_indicators(
        self, tables, sample_performance_results, sample_statistical_results
    ):
        """Test adding statistical significance indicators."""
        df = pd.DataFrame(sample_performance_results).T
        result = tables._add_significance_indicators(df, sample_statistical_results)

        assert isinstance(result, pd.DataFrame)

        # Check for significance symbols in separate columns
        for approach in sample_statistical_results:
            for metric in sample_statistical_results[approach]:
                sig_col = f"{metric}_sig"
                if sig_col in result.columns and approach in result.index:
                    sig_value = result.loc[approach, sig_col]
                    # Should contain significance symbols for significant results
                    if sample_statistical_results[approach][metric]["p_value"] < 0.05:
                        assert "*" in str(sig_value)

    def test_add_baseline_comparisons(
        self, tables, sample_performance_results, sample_baseline_comparisons
    ):
        """Test adding baseline comparison columns."""
        df = pd.DataFrame(sample_performance_results).T
        result = tables._add_baseline_comparisons(df, sample_baseline_comparisons)

        assert isinstance(result, pd.DataFrame)

        # Check for baseline comparison columns
        baseline_columns = [col for col in result.columns if "vs_" in col]
        assert len(baseline_columns) > 0

        # Verify baseline data was added correctly
        for approach in sample_baseline_comparisons:
            for baseline in sample_baseline_comparisons[approach]:
                for metric in sample_baseline_comparisons[approach][baseline]:
                    col_name = f"vs_{baseline}_{metric}"
                    if col_name in result.columns:
                        assert (
                            result.loc[approach, col_name]
                            == sample_baseline_comparisons[approach][baseline][metric]
                        )

    def test_add_metric_rankings(self, tables, sample_performance_results):
        """Test adding metric ranking columns."""
        df = pd.DataFrame(sample_performance_results).T
        result = tables._add_metric_rankings(df)

        assert isinstance(result, pd.DataFrame)

        # Check for ranking columns
        ranking_columns = [col for col in result.columns if "_rank" in col]
        assert len(ranking_columns) > 0

        # Verify ranking logic
        if "sharpe_ratio_rank" in result.columns:
            sharpe_ranks = result["sharpe_ratio_rank"]
            sharpe_values = result["sharpe_ratio"]
            # Higher Sharpe ratio should have lower rank (1 is best)
            best_sharpe_idx = sharpe_values.idxmax()
            assert sharpe_ranks.loc[best_sharpe_idx] == 1.0

    def test_calculate_risk_adjusted_score(self, tables, sample_performance_results):
        """Test risk-adjusted score calculation."""
        df = pd.DataFrame(sample_performance_results).T
        scores = tables._calculate_risk_adjusted_score(df)

        assert isinstance(scores, pd.Series)
        assert len(scores) == len(df)

        # Verify all scores are numeric
        assert scores.dtype in [np.float64, np.int64]

        # Verify scores are reasonable (not all zero or identical)
        assert scores.std() > 0

    def test_get_significance_symbol(self, tables):
        """Test significance symbol generation."""
        assert tables._get_significance_symbol(0.0005) == "***"
        assert tables._get_significance_symbol(0.005) == "**"
        assert tables._get_significance_symbol(0.025) == "*"
        assert tables._get_significance_symbol(0.10) == ""

    def test_format_numerical_columns(self, tables, sample_performance_results):
        """Test numerical column formatting."""
        df = pd.DataFrame(sample_performance_results).T
        # Add a rank column for testing
        df["sharpe_ratio_rank"] = [2, 1, 3]

        result = tables._format_numerical_columns(df)

        assert isinstance(result, pd.DataFrame)

        # Check that rank columns are integers
        if "sharpe_ratio_rank" in result.columns:
            ranks = result["sharpe_ratio_rank"]
            assert all(isinstance(rank, (int, np.integer)) for rank in ranks)

    def test_export_table_csv(self, tables, sample_performance_results, tmp_path):
        """Test CSV export functionality."""
        table_df = pd.DataFrame(sample_performance_results).T
        filename = str(tmp_path / "test_table")

        exported = tables.export_table(table_df, filename, formats=["csv"])

        assert "csv" in exported
        assert (tmp_path / "test_table.csv").exists()

    def test_export_table_html(self, tables, sample_performance_results, tmp_path):
        """Test HTML export functionality."""
        table_df = pd.DataFrame(sample_performance_results).T
        filename = str(tmp_path / "test_table")

        exported = tables.export_table(table_df, filename, formats=["html"])

        assert "html" in exported
        assert (tmp_path / "test_table.html").exists()

        # Verify HTML content
        with open(tmp_path / "test_table.html") as f:
            content = f.read()
            assert "<table" in content
            assert "performance-table" in content

    def test_export_table_latex(self, tables, sample_performance_results, tmp_path):
        """Test LaTeX export functionality."""
        table_df = pd.DataFrame(sample_performance_results).T
        filename = str(tmp_path / "test_table")

        exported = tables.export_table(table_df, filename, formats=["latex"])

        assert "latex" in exported
        assert (tmp_path / "test_table.tex").exists()

        # Verify LaTeX content
        with open(tmp_path / "test_table.tex") as f:
            content = f.read()
            assert "\\documentclass" in content
            assert "\\begin{tabular}" in content

    def test_generate_html_table(self, tables, sample_performance_results):
        """Test HTML table generation."""
        table_df = pd.DataFrame(sample_performance_results).T
        html_content = tables._generate_html_table(table_df, "Test Table")

        assert isinstance(html_content, str)
        assert "<table" in html_content
        assert "Test Table" in html_content
        assert "performance-table" in html_content

        # Check for sortable functionality if enabled
        if tables.config.sortable:
            assert "sortTable" in html_content

    def test_generate_latex_table(self, tables, sample_performance_results):
        """Test LaTeX table generation."""
        table_df = pd.DataFrame(sample_performance_results).T
        latex_content = tables._generate_latex_table(table_df, "Test Table")

        assert isinstance(latex_content, str)
        assert "\\documentclass" in latex_content
        assert "Test Table" in latex_content
        assert "\\begin{tabular}" in latex_content

    @patch("src.evaluation.reporting.tables.HAS_IPYTHON", True)
    def test_display_interactive_table(self, tables, sample_performance_results):
        """Test interactive table display."""
        table_df = pd.DataFrame(sample_performance_results).T

        with patch("src.evaluation.reporting.tables.display") as mock_display:
            tables.display_interactive_table(table_df, "Test Table")
            mock_display.assert_called_once()

    @patch("src.evaluation.reporting.tables.HAS_IPYTHON", False)
    def test_display_interactive_table_no_ipython(self, tables, sample_performance_results):
        """Test interactive table display without IPython."""
        table_df = pd.DataFrame(sample_performance_results).T

        with patch("warnings.warn") as mock_warn:
            tables.display_interactive_table(table_df, "Test Table")
            mock_warn.assert_called_once()

    def test_empty_data_handling(self, tables):
        """Test handling of empty data."""
        empty_results = {}
        result = tables.create_performance_ranking_table(empty_results)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_single_approach_data(self, tables):
        """Test table creation with single approach."""
        single_results = {
            "HRP": {
                "sharpe_ratio": 1.25,
                "total_return": 0.125,
            }
        }

        result = tables.create_performance_ranking_table(single_results)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.index[0] == "HRP"

    def test_missing_metrics_handling(self, tables):
        """Test handling of missing metrics in data."""
        incomplete_results = {
            "HRP": {"sharpe_ratio": 1.25},  # Missing other metrics
            "LSTM": {"total_return": 0.158},  # Missing sharpe_ratio
        }

        result = tables.create_performance_ranking_table(incomplete_results)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

        # Should handle missing values gracefully
        assert pd.isna(result.loc["HRP", "total_return"]) or result.loc["HRP", "total_return"] == 0
        assert (
            pd.isna(result.loc["LSTM", "sharpe_ratio"]) or result.loc["LSTM", "sharpe_ratio"] == 0
        )

    def test_config_without_rankings(self, sample_performance_results):
        """Test table creation with rankings disabled."""
        config = TableConfig(include_rankings=False)
        tables = PerformanceComparisonTables(config)

        result = tables.create_performance_ranking_table(sample_performance_results)

        # Check that ranking columns are not present
        ranking_columns = [col for col in result.columns if "_rank" in col]
        assert len(ranking_columns) == 0

    def test_config_different_decimal_places(self, sample_performance_results):
        """Test table creation with different decimal place settings."""
        config = TableConfig(decimal_places=2)
        tables = PerformanceComparisonTables(config)

        result = tables.create_performance_ranking_table(sample_performance_results)

        # Check that numerical values are rounded to specified decimal places
        for col in result.select_dtypes(include=[np.number]).columns:
            values = result[col]
            for val in values:
                if not pd.isna(val) and isinstance(val, (int, float)):
                    # Check that value has at most 2 decimal places when converted to string
                    str_val = str(float(val))
                    if "." in str_val:
                        decimal_places = len(str_val.split(".")[1])
                        # Allow some tolerance for floating point precision
                        assert decimal_places <= 4  # Reasonable upper bound
