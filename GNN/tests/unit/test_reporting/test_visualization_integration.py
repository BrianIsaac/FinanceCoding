"""
Comprehensive test suite for the visualization framework.

Tests all visualization modules including tables, charts, risk-return analysis,
heatmaps, operational analysis, and regime analysis.
"""

import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.evaluation.reporting.charts import ChartConfig, TimeSeriesCharts
from src.evaluation.reporting.heatmaps import HeatmapConfig, PerformanceHeatmaps
from src.evaluation.reporting.operational_analysis import (
    OperationalConfig,
    OperationalEfficiencyAnalysis,
)
from src.evaluation.reporting.regime_analysis import MarketRegimeAnalysis, RegimeAnalysisConfig
from src.evaluation.reporting.risk_return import RiskReturnAnalysis, RiskReturnConfig

# Import visualization modules
from src.evaluation.reporting.tables import PerformanceComparisonTables, TableConfig


# Module-level fixtures
@pytest.fixture
def sample_returns_data():
    """Create sample returns data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=252, freq="D")

    returns_data = {}
    for approach in ["HRP", "LSTM", "GAT"]:
        # Generate sample returns with different characteristics
        if approach == "HRP":
            returns = np.random.normal(0.0005, 0.012, len(dates))  # Low vol, modest return
        elif approach == "LSTM":
            returns = np.random.normal(0.0008, 0.015, len(dates))  # Higher vol, higher return
        else:  # GAT
            returns = np.random.normal(0.0007, 0.013, len(dates))  # Medium characteristics

        returns_data[approach] = pd.Series(returns, index=dates)

    return returns_data


@pytest.fixture
def sample_performance_metrics():
    """Create sample performance metrics."""
    return {
        "HRP": {
            "sharpe_ratio": 1.25,
            "information_ratio": 0.85,
            "total_return": 0.12,
            "annualized_return": 0.125,
            "volatility": 0.18,
            "max_drawdown": -0.08,
            "win_rate": 0.58,
        },
        "LSTM": {
            "sharpe_ratio": 1.45,
            "information_ratio": 1.02,
            "total_return": 0.15,
            "annualized_return": 0.158,
            "volatility": 0.21,
            "max_drawdown": -0.12,
            "win_rate": 0.62,
        },
        "GAT": {
            "sharpe_ratio": 1.35,
            "information_ratio": 0.95,
            "total_return": 0.138,
            "annualized_return": 0.142,
            "volatility": 0.19,
            "max_drawdown": -0.095,
            "win_rate": 0.61,
        },
    }


@pytest.fixture
def sample_turnover_data():
    """Create sample turnover data."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=252, freq="D")

    turnover_data = {}
    for approach in ["HRP", "LSTM", "GAT"]:
        # Generate sample turnover with different levels
        if approach == "HRP":
            turnover = np.random.gamma(2, 0.02, len(dates))  # Low turnover
        elif approach == "LSTM":
            turnover = np.random.gamma(3, 0.03, len(dates))  # Higher turnover
        else:  # GAT
            turnover = np.random.gamma(2.5, 0.025, len(dates))  # Medium turnover

        turnover_data[approach] = pd.Series(turnover, index=dates)

    return turnover_data


@pytest.fixture
def sample_statistical_results():
    """Create sample statistical significance results."""
    return {
        "HRP": {
            "sharpe_ratio": {"p_value": 0.025, "t_stat": 2.15},
            "information_ratio": {"p_value": 0.08, "t_stat": 1.75},
        },
        "LSTM": {
            "sharpe_ratio": {"p_value": 0.008, "t_stat": 2.68},
            "information_ratio": {"p_value": 0.003, "t_stat": 3.12},
        },
        "GAT": {
            "sharpe_ratio": {"p_value": 0.015, "t_stat": 2.45},
            "information_ratio": {"p_value": 0.045, "t_stat": 2.02},
        },
    }


class TestVisualizationFramework:
    """Test suite for the complete visualization framework."""


class TestPerformanceComparisonTables:
    """Test performance comparison tables functionality."""

    def test_table_creation(self, sample_performance_metrics, sample_statistical_results):
        """Test basic table creation."""
        tables = PerformanceComparisonTables()

        result_table = tables.create_performance_ranking_table(
            sample_performance_metrics, sample_statistical_results
        )

        assert isinstance(result_table, pd.DataFrame)
        assert len(result_table) == 3  # Three approaches
        assert "sharpe_ratio" in result_table.columns
        assert result_table.index.tolist() == ["LSTM", "GAT", "HRP"]  # Sorted by Sharpe ratio

    def test_risk_adjusted_ranking(self, sample_performance_metrics):
        """Test risk-adjusted ranking table creation."""
        tables = PerformanceComparisonTables()

        # Create some risk metrics
        risk_metrics = {
            approach: {
                "tracking_error": np.random.uniform(0.02, 0.06),
                "downside_deviation": np.random.uniform(0.08, 0.15),
            }
            for approach in sample_performance_metrics.keys()
        }

        result_table = tables.create_risk_adjusted_ranking_table(
            sample_performance_metrics, risk_metrics
        )

        assert isinstance(result_table, pd.DataFrame)
        assert "risk_adjusted_score" in result_table.columns
        assert len(result_table) == 3

    def test_table_export(self, sample_performance_metrics, tmp_path):
        """Test table export functionality."""
        tables = PerformanceComparisonTables()

        result_table = tables.create_performance_ranking_table(sample_performance_metrics)

        # Change to temporary directory for export
        original_cwd = Path.cwd()
        try:
            import os

            os.chdir(tmp_path)

            exported_files = tables.export_table(
                result_table, "test_performance_table", formats=["csv"]
            )

            assert "csv" in exported_files
            assert Path(exported_files["csv"]).exists()

        finally:
            os.chdir(original_cwd)

    def test_configuration(self):
        """Test table configuration options."""
        config = TableConfig(
            decimal_places=3, significance_levels=[0.01, 0.05], include_confidence_intervals=False
        )

        tables = PerformanceComparisonTables(config)

        assert tables.config.decimal_places == 3
        assert tables.config.significance_levels == [0.01, 0.05]
        assert not tables.config.include_confidence_intervals


class TestTimeSeriesCharts:
    """Test time series charts functionality."""

    @patch("src.evaluation.reporting.charts.HAS_MATPLOTLIB", True)
    def test_cumulative_returns_plot(self, sample_returns_data):
        """Test cumulative returns plot creation."""
        charts = TimeSeriesCharts()

        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            result = charts.plot_cumulative_returns(sample_returns_data, interactive=False)

            # Should return matplotlib figure
            assert result is mock_fig
            mock_subplots.assert_called_once()

    @patch("src.evaluation.reporting.charts.HAS_MATPLOTLIB", True)
    def test_drawdown_analysis(self, sample_returns_data):
        """Test drawdown analysis chart creation."""
        charts = TimeSeriesCharts()

        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            mock_fig = Mock()
            mock_axes = [Mock(), Mock()]
            mock_subplots.return_value = (mock_fig, mock_axes)

            result = charts.plot_drawdown_analysis(sample_returns_data, interactive=False)

            assert result is mock_fig

    @patch("src.evaluation.reporting.charts.HAS_PLOTLY", True)
    def test_interactive_charts(self, sample_returns_data):
        """Test interactive chart creation."""
        charts = TimeSeriesCharts()

        with patch("plotly.graph_objects.Figure") as mock_figure_class:
            mock_fig = Mock()
            mock_figure_class.return_value = mock_fig

            result = charts.plot_cumulative_returns(sample_returns_data, interactive=True)

            # Should return plotly figure
            assert result is mock_fig

    def test_chart_configuration(self):
        """Test chart configuration options."""
        config = ChartConfig(figsize=(10, 6), dpi=200, confidence_level=0.90)

        charts = TimeSeriesCharts(config)

        assert charts.config.figsize == (10, 6)
        assert charts.config.dpi == 200
        assert charts.config.confidence_level == 0.90


class TestRiskReturnAnalysis:
    """Test risk-return analysis functionality."""

    def test_risk_return_scatter_creation(self, sample_performance_metrics):
        """Test risk-return scatter plot creation."""
        risk_return = RiskReturnAnalysis()

        with patch("src.evaluation.reporting.risk_return.HAS_MATPLOTLIB", True):
            with patch("matplotlib.pyplot.subplots") as mock_subplots:
                mock_fig = Mock()
                mock_ax = Mock()
                mock_subplots.return_value = (mock_fig, mock_ax)

                result = risk_return.plot_risk_return_scatter(
                    sample_performance_metrics, interactive=False
                )

                assert result is mock_fig

    def test_confidence_ellipse_calculation(self, sample_returns_data):
        """Test confidence ellipse calculation."""
        risk_return = RiskReturnAnalysis()

        ellipses = risk_return.calculate_confidence_ellipses(sample_returns_data)

        # Should return ellipse parameters for each approach
        assert isinstance(ellipses, dict)
        for approach in sample_returns_data.keys():
            if approach in ellipses:  # May not have ellipse if insufficient data
                assert "center_x" in ellipses[approach]
                assert "center_y" in ellipses[approach]
                assert "width" in ellipses[approach]
                assert "height" in ellipses[approach]

    def test_regime_specific_analysis(self, sample_performance_metrics):
        """Test regime-specific risk-return analysis."""
        risk_return = RiskReturnAnalysis()

        # Create regime performance data
        regime_data = {
            "bull": sample_performance_metrics,
            "bear": {
                k: {**v, "sharpe_ratio": v["sharpe_ratio"] * 0.7}
                for k, v in sample_performance_metrics.items()
            },
            "sideways": {
                k: {**v, "sharpe_ratio": v["sharpe_ratio"] * 0.85}
                for k, v in sample_performance_metrics.items()
            },
        }

        with patch("src.evaluation.reporting.risk_return.HAS_MATPLOTLIB", True):
            with patch("matplotlib.pyplot.subplots") as mock_subplots:
                mock_fig = Mock()
                mock_axes = np.array([[Mock(), Mock()], [Mock(), Mock()]])
                mock_subplots.return_value = (mock_fig, mock_axes)

                result = risk_return.create_regime_specific_analysis(regime_data, interactive=False)

                assert result is mock_fig


class TestPerformanceHeatmaps:
    """Test performance heatmaps functionality."""

    def test_monthly_heatmap_creation(self, sample_returns_data):
        """Test monthly performance heatmap creation."""
        heatmaps = PerformanceHeatmaps()

        with patch("src.evaluation.reporting.heatmaps.HAS_MATPLOTLIB", True):
            with patch("matplotlib.pyplot.subplots") as mock_subplots:
                mock_fig = Mock()
                mock_axes = [Mock(), Mock(), Mock()]
                mock_subplots.return_value = (mock_fig, mock_axes)

                result = heatmaps.create_monthly_performance_heatmap(
                    sample_returns_data, interactive=False
                )

                assert result is mock_fig

    def test_relative_performance_heatmap(self, sample_returns_data):
        """Test relative performance heatmap creation."""
        heatmaps = PerformanceHeatmaps()

        # Create baseline returns
        baseline_returns = {
            "equal_weight": sample_returns_data["HRP"] * 0.8,
            "mean_variance": sample_returns_data["HRP"] * 0.9,
        }

        with patch("src.evaluation.reporting.heatmaps.HAS_MATPLOTLIB", True):
            with patch("matplotlib.pyplot.subplots") as mock_subplots:
                mock_fig = Mock()
                mock_axes = np.array([[Mock(), Mock()]])
                mock_subplots.return_value = (mock_fig, mock_axes)

                result = heatmaps.create_relative_performance_heatmap(
                    sample_returns_data, baseline_returns, interactive=False
                )

                assert result is mock_fig

    def test_statistical_significance_overlay(
        self, sample_returns_data, sample_statistical_results
    ):
        """Test statistical significance overlay on heatmaps."""
        heatmaps = PerformanceHeatmaps()

        # First create aggregated data
        aggregated_data = heatmaps._aggregate_returns(sample_returns_data, "monthly")

        with patch("src.evaluation.reporting.heatmaps.HAS_MATPLOTLIB", True):
            with patch("matplotlib.pyplot.subplots") as mock_subplots:
                mock_fig = Mock()
                mock_axes = [Mock(), Mock(), Mock()]
                mock_subplots.return_value = (mock_fig, mock_axes)

                result = heatmaps.add_statistical_significance_overlay(
                    aggregated_data, sample_statistical_results
                )

                assert result is mock_fig


class TestOperationalEfficiencyAnalysis:
    """Test operational efficiency analysis functionality."""

    def test_turnover_analysis_chart(self, sample_turnover_data, sample_performance_metrics):
        """Test turnover analysis chart creation."""
        operational = OperationalEfficiencyAnalysis()

        with patch("src.evaluation.reporting.operational_analysis.HAS_MATPLOTLIB", True):
            with patch("matplotlib.pyplot.subplots") as mock_subplots:
                mock_fig = Mock()
                mock_axes = np.array([[Mock(), Mock()], [Mock(), Mock()]])
                mock_subplots.return_value = (mock_fig, mock_axes)

                result = operational.create_turnover_analysis_chart(
                    sample_turnover_data, sample_performance_metrics, interactive=False
                )

                assert result is mock_fig

    def test_transaction_cost_visualization(self, sample_returns_data, sample_turnover_data):
        """Test transaction cost impact visualization."""
        operational = OperationalEfficiencyAnalysis()

        with patch("src.evaluation.reporting.operational_analysis.HAS_MATPLOTLIB", True):
            with patch("matplotlib.pyplot.subplots") as mock_subplots:
                mock_fig = Mock()
                mock_axes = np.array([[Mock(), Mock()], [Mock(), Mock()]])
                mock_subplots.return_value = (mock_fig, mock_axes)

                result = operational.create_transaction_cost_impact_visualization(
                    sample_returns_data, sample_turnover_data, interactive=False
                )

                assert result is mock_fig

    def test_constraint_compliance_dashboard(self):
        """Test constraint compliance dashboard creation."""
        operational = OperationalEfficiencyAnalysis()

        # Create sample constraint data
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        constraint_data = {}

        for approach in ["HRP", "LSTM", "GAT"]:
            constraint_data[approach] = {
                "weight_constraint": pd.Series(np.random.uniform(0, 0.08, len(dates)), index=dates),
                "sector_constraint": pd.Series(np.random.uniform(0, 0.12, len(dates)), index=dates),
            }

        with patch("src.evaluation.reporting.operational_analysis.HAS_MATPLOTLIB", True):
            with patch("matplotlib.pyplot.subplots") as mock_subplots:
                mock_fig = Mock()
                mock_axes = np.array([[Mock(), Mock()], [Mock(), Mock()]])
                mock_subplots.return_value = (mock_fig, mock_axes)

                result = operational.create_constraint_compliance_dashboard(
                    constraint_data, interactive=False
                )

                assert result is mock_fig


class TestMarketRegimeAnalysis:
    """Test market regime analysis functionality."""

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for regime detection."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=252, freq="D")

        # Create market data with regime-like patterns
        returns = []
        for i in range(len(dates)):
            if i < 84:  # First third - bull market
                returns.append(np.random.normal(0.001, 0.015))
            elif i < 168:  # Middle third - bear market
                returns.append(np.random.normal(-0.0005, 0.025))
            else:  # Last third - sideways market
                returns.append(np.random.normal(0.0002, 0.010))

        return pd.Series(returns, index=dates)

    def test_regime_detection_threshold(self, sample_market_data):
        """Test threshold-based regime detection."""
        regime_analysis = MarketRegimeAnalysis()

        regimes = regime_analysis.detect_market_regimes(sample_market_data, method="threshold")

        assert isinstance(regimes, pd.Series)
        assert len(regimes) == len(sample_market_data)
        assert all(
            regime in ["bull", "bear", "sideways", "neutral"]
            for regime in regimes.dropna().unique()
        )

    def test_regime_detection_kmeans(self, sample_market_data):
        """Test K-means based regime detection."""
        regime_analysis = MarketRegimeAnalysis()

        regimes = regime_analysis.detect_market_regimes(
            sample_market_data, method="kmeans", n_regimes=3
        )

        assert isinstance(regimes, pd.Series)
        assert len(regimes) == len(sample_market_data)
        unique_regimes = regimes.dropna().unique()
        assert len(unique_regimes) <= 3  # Should have at most 3 regimes

    def test_regime_performance_table(self, sample_returns_data, sample_market_data):
        """Test regime-specific performance table creation."""
        regime_analysis = MarketRegimeAnalysis()

        # Detect regimes
        regime_labels = regime_analysis.detect_market_regimes(sample_market_data)

        # Create performance tables
        performance_tables = regime_analysis.create_regime_specific_performance_table(
            sample_returns_data, regime_labels
        )

        assert isinstance(performance_tables, dict)
        for regime, table in performance_tables.items():
            assert isinstance(table, pd.DataFrame)
            assert len(table) <= len(sample_returns_data)  # At most one row per approach

    def test_regime_transition_analysis(self, sample_returns_data, sample_market_data):
        """Test regime transition analysis."""
        regime_analysis = MarketRegimeAnalysis()

        # Detect regimes
        regime_labels = regime_analysis.detect_market_regimes(sample_market_data)

        with patch("src.evaluation.reporting.regime_analysis.HAS_MATPLOTLIB", True):
            with patch("matplotlib.pyplot.subplots") as mock_subplots:
                mock_fig = Mock()
                mock_axes = np.array([[Mock(), Mock()], [Mock(), Mock()]])
                mock_subplots.return_value = (mock_fig, mock_axes)

                result = regime_analysis.create_regime_transition_analysis(
                    sample_returns_data, regime_labels, interactive=False
                )

                assert result is mock_fig

    def test_regime_statistical_testing(self, sample_returns_data, sample_market_data):
        """Test regime-based statistical significance testing."""
        regime_analysis = MarketRegimeAnalysis()

        # Detect regimes
        regime_labels = regime_analysis.detect_market_regimes(sample_market_data)

        # Create baseline returns
        baseline_returns = {"benchmark": sample_returns_data["HRP"] * 0.8}

        statistical_results = regime_analysis.add_regime_based_statistical_testing(
            sample_returns_data, regime_labels, baseline_returns
        )

        assert isinstance(statistical_results, dict)
        for approach, results in statistical_results.items():
            assert isinstance(results, dict)
            for regime_key, stats in results.items():
                assert regime_key.startswith("regime_")
                assert isinstance(stats, dict)
                if "p_value_vs_zero" in stats:
                    assert isinstance(stats["p_value_vs_zero"], (int, float))


class TestIntegrationScenarios:
    """Test integration between different visualization components."""

    def test_end_to_end_visualization_pipeline(
        self,
        sample_returns_data,
        sample_performance_metrics,
        sample_turnover_data,
        sample_statistical_results,
    ):
        """Test complete visualization pipeline integration."""
        # Initialize all visualization components
        tables = PerformanceComparisonTables()
        charts = TimeSeriesCharts()
        risk_return = RiskReturnAnalysis()
        heatmaps = PerformanceHeatmaps()
        operational = OperationalEfficiencyAnalysis()
        regime_analysis = MarketRegimeAnalysis()

        # Test table creation
        performance_table = tables.create_performance_ranking_table(
            sample_performance_metrics, sample_statistical_results
        )
        assert isinstance(performance_table, pd.DataFrame)

        # Test charts creation (mocked)
        with patch("src.evaluation.reporting.charts.HAS_MATPLOTLIB", True):
            with patch("matplotlib.pyplot.subplots"):
                charts_result = charts.plot_cumulative_returns(
                    sample_returns_data, interactive=False
                )
                assert charts_result is not None

        # Test risk-return analysis
        with patch("src.evaluation.reporting.risk_return.HAS_MATPLOTLIB", True):
            with patch("matplotlib.pyplot.subplots"):
                risk_return_result = risk_return.plot_risk_return_scatter(
                    sample_performance_metrics, interactive=False
                )
                assert risk_return_result is not None

        # Test operational analysis
        with patch("src.evaluation.reporting.operational_analysis.HAS_MATPLOTLIB", True):
            with patch("matplotlib.pyplot.subplots"):
                operational_result = operational.create_turnover_analysis_chart(
                    sample_turnover_data, sample_performance_metrics, interactive=False
                )
                assert operational_result is not None

        # Verify all components work together
        assert len(performance_table) == len(sample_performance_metrics)

    def test_configuration_consistency(self):
        """Test that all configurations are consistent and work together."""
        # Test that all config classes can be instantiated
        table_config = TableConfig()
        chart_config = ChartConfig()
        risk_return_config = RiskReturnConfig()
        heatmap_config = HeatmapConfig()
        operational_config = OperationalConfig()
        regime_config = RegimeAnalysisConfig()

        # Test that all components accept their configurations
        tables = PerformanceComparisonTables(table_config)
        charts = TimeSeriesCharts(chart_config)
        risk_return = RiskReturnAnalysis(risk_return_config)
        heatmaps = PerformanceHeatmaps(heatmap_config)
        operational = OperationalEfficiencyAnalysis(operational_config)
        regime_analysis = MarketRegimeAnalysis(regime_config)

        # Verify configurations are properly stored
        assert tables.config is table_config
        assert charts.config is chart_config
        assert risk_return.config is risk_return_config
        assert heatmaps.config is heatmap_config
        assert operational.config is operational_config
        assert regime_analysis.config is regime_config

    def test_error_handling(self, sample_returns_data):
        """Test error handling across visualization components."""
        # Test empty data handling
        empty_data = {}

        tables = PerformanceComparisonTables()
        with pytest.raises((ValueError, KeyError)) or warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tables.create_performance_ranking_table(empty_data)

        # Test mismatched data handling
        mismatched_data = {"approach1": sample_returns_data["HRP"]}
        mismatched_performance = {"approach2": {"sharpe_ratio": 1.0}}

        # Should handle gracefully or raise appropriate error
        try:
            result = tables.create_performance_ranking_table(mismatched_performance)
            assert isinstance(result, pd.DataFrame)
        except (ValueError, KeyError):
            pass  # Expected for mismatched data

    def test_memory_efficiency(self, sample_returns_data):
        """Test that visualization components handle large datasets efficiently."""
        # Create larger dataset
        large_dates = pd.date_range("2020-01-01", periods=2520, freq="D")  # 10x larger
        large_returns_data = {}

        np.random.seed(42)
        for approach in sample_returns_data.keys():
            large_returns_data[approach] = pd.Series(
                np.random.normal(0.0005, 0.012, len(large_dates)), index=large_dates
            )

        # Test that components can handle larger datasets
        charts = TimeSeriesCharts()

        # Mock the plotting to avoid actual visualization
        with patch("src.evaluation.reporting.charts.HAS_MATPLOTLIB", True):
            with patch("matplotlib.pyplot.subplots") as mock_subplots:
                mock_fig = Mock()
                mock_ax = Mock()
                mock_subplots.return_value = (mock_fig, mock_ax)

                result = charts.plot_cumulative_returns(large_returns_data, interactive=False)

                assert result is mock_fig
                # Verify the method completed without memory errors
                mock_subplots.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
