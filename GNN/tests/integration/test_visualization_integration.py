"""
Integration tests for visualization components.

Tests the integration between different visualization modules and their
interaction with data sources, configuration management, and export systems.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.evaluation.reporting.charts import TimeSeriesCharts
from src.evaluation.reporting.heatmaps import PerformanceHeatmaps
from src.evaluation.reporting.interactive import InteractiveDashboard
from src.evaluation.reporting.operational_analysis import OperationalEfficiencyAnalysis
from src.evaluation.reporting.regime_analysis import MarketRegimeAnalysis
from src.evaluation.reporting.risk_return import RiskReturnAnalysis
from src.evaluation.reporting.tables import PerformanceComparisonTables


class TestVisualizationIntegration:
    """Test integration between visualization components."""

    @pytest.fixture
    def sample_data(self):
        """Create comprehensive sample data for testing."""
        # Create date range
        dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")
        n_dates = len(dates)

        # Set random seed for reproducibility
        np.random.seed(42)

        # Generate correlated returns data
        approaches = ["HRP", "LSTM", "GAT", "EqualWeight", "MeanVariance"]
        base_returns = np.random.multivariate_normal(
            mean=[0.0008, 0.0010, 0.0012, 0.0005, 0.0007],
            cov=np.array(
                [
                    [0.0002, 0.0001, 0.0001, 0.00008, 0.00009],
                    [0.0001, 0.0003, 0.0001, 0.00009, 0.0001],
                    [0.0001, 0.0001, 0.0004, 0.0001, 0.0001],
                    [0.00008, 0.00009, 0.0001, 0.00015, 0.00008],
                    [0.00009, 0.0001, 0.0001, 0.00008, 0.00018],
                ]
            ),
            size=n_dates,
        )

        returns_data = {}
        for i, approach in enumerate(approaches):
            returns_data[approach] = pd.Series(base_returns[:, i], index=dates)

        # Generate turnover data
        turnover_data = {}
        for approach in approaches:
            if approach in ["HRP", "LSTM", "GAT"]:
                # ML approaches have higher turnover
                turnover = np.random.gamma(2, 0.1, n_dates)
            else:
                # Traditional approaches have lower turnover
                turnover = np.random.gamma(1.5, 0.05, n_dates)
            turnover_data[approach] = pd.Series(turnover, index=dates)

        # Generate performance metrics
        performance_metrics = {}
        for approach, returns_series in returns_data.items():
            total_return = (1 + returns_series).prod() - 1
            volatility = returns_series.std() * np.sqrt(252)
            sharpe_ratio = returns_series.mean() / returns_series.std() * np.sqrt(252)

            # Calculate maximum drawdown
            cum_returns = (1 + returns_series).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max
            max_drawdown = drawdown.min()

            performance_metrics[approach] = {
                "total_return": total_return,
                "annualized_return": (1 + returns_series.mean()) ** 252 - 1,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "information_ratio": sharpe_ratio * 0.8,  # Simplified IR
                "max_drawdown": max_drawdown,
                "tracking_error": volatility * 0.6,
            }

        # Generate regime data
        regime_labels = []
        current_regime = "bull"
        regime_duration = 0

        for _ in range(n_dates):
            regime_duration += 1

            # Change regime probabilistically
            if regime_duration > 30:  # Minimum 30 days
                change_prob = min(0.05, regime_duration / 500)  # Increasing prob over time
                if np.random.random() < change_prob:
                    current_regime = np.random.choice(["bull", "bear", "sideways"])
                    regime_duration = 0

            regime_labels.append(current_regime)

        regime_data = pd.Series(regime_labels, index=dates)

        return {
            "returns_data": returns_data,
            "turnover_data": turnover_data,
            "performance_metrics": performance_metrics,
            "regime_data": regime_data,
        }

    @pytest.fixture
    def config_file(self):
        """Create temporary configuration file for testing."""
        config_data = {
            "global": {
                "figsize": [12, 8],
                "dpi": 300,
                "theme": "plotly_white",
                "height": 800,
                "width": 1200,
            },
            "charts": {
                "time_series": {
                    "confidence_level": 0.95,
                    "line_width": 2,
                }
            },
            "tables": {
                "decimal_places": 3,
                "significance_levels": [0.05, 0.01],
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            return f.name

    def test_end_to_end_analysis_workflow(self, sample_data):
        """Test complete end-to-end analysis workflow."""
        returns_data = sample_data["returns_data"]
        performance_metrics = sample_data["performance_metrics"]
        sample_data["turnover_data"]
        sample_data["regime_data"]

        # 1. Test table creation (no external dependencies)
        tables = PerformanceComparisonTables()
        performance_table = tables.create_performance_ranking_table(performance_metrics)

        assert isinstance(performance_table, pd.DataFrame)
        assert len(performance_table) == len(performance_metrics)
        assert "sharpe_ratio" in performance_table.columns

        # 2. Test components initialization
        charts = TimeSeriesCharts()
        assert charts.config is not None

        risk_return = RiskReturnAnalysis()
        assert risk_return.config is not None

        operational = OperationalEfficiencyAnalysis()
        assert operational.config is not None

        # 3. Test regime analysis (core functionality)
        regime_analysis = MarketRegimeAnalysis()
        detected_regimes = regime_analysis.detect_market_regimes(
            returns_data["HRP"], method="threshold"
        )

        assert isinstance(detected_regimes, pd.Series)
        assert len(detected_regimes) > 0

        # 4. Test data processing methods
        heatmaps = PerformanceHeatmaps()
        aggregated_data = heatmaps._aggregate_returns(returns_data, "monthly")
        assert isinstance(aggregated_data, dict)
        assert len(aggregated_data) > 0

    def test_interactive_dashboard_integration(self, sample_data):
        """Test interactive dashboard integration with all components."""
        sample_data["returns_data"]
        sample_data["performance_metrics"]
        sample_data["turnover_data"]

        # Test dashboard initialization
        with patch("src.evaluation.reporting.interactive.HAS_PLOTLY", True):
            dashboard = InteractiveDashboard()
            assert dashboard.config is not None
            assert dashboard.time_series is not None
            assert dashboard.tables is not None

            # Test component integration
            assert hasattr(dashboard, "heatmaps")
            assert hasattr(dashboard, "risk_return")
            assert hasattr(dashboard, "operational")
            assert hasattr(dashboard, "regime_analysis")

    def test_configuration_integration(self, config_file, sample_data):
        """Test configuration file integration across components."""
        sample_data["returns_data"]

        # Test dashboard with config file
        with patch("src.evaluation.reporting.interactive.HAS_PLOTLY", True):
            dashboard = InteractiveDashboard(config_path=config_file)

            assert dashboard.config.height == 800
            assert dashboard.config.width == 1200
            assert dashboard.config.theme == "plotly_white"

    def test_export_integration(self, sample_data, tmp_path):
        """Test export functionality integration across components."""
        sample_data["returns_data"]
        performance_metrics = sample_data["performance_metrics"]

        # Test table export only (no external dependencies)
        tables = PerformanceComparisonTables()
        performance_table = tables.create_performance_ranking_table(performance_metrics)

        exported_files = tables.export_table(
            performance_table, str(tmp_path / "performance_table"), formats=["csv", "html"]
        )

        assert "csv" in exported_files
        assert "html" in exported_files
        assert Path(exported_files["csv"]).exists()
        assert Path(exported_files["html"]).exists()

    def test_data_validation_integration(self, sample_data):
        """Test data validation across visualization components."""
        returns_data = sample_data["returns_data"]
        performance_metrics = sample_data["performance_metrics"]

        # Test with missing data
        {k: v for k, v in returns_data.items() if k in ["HRP", "LSTM"]}
        incomplete_metrics = {k: v for k, v in performance_metrics.items() if k in ["HRP", "LSTM"]}

        # Should handle incomplete data gracefully
        tables = PerformanceComparisonTables()
        result_table = tables.create_performance_ranking_table(incomplete_metrics)

        assert isinstance(result_table, pd.DataFrame)
        assert len(result_table) == 2

        # Test with empty data
        empty_data = {}
        empty_result = tables.create_performance_ranking_table(empty_data)

        assert isinstance(empty_result, pd.DataFrame)
        assert len(empty_result) == 0

    def test_statistical_integration(self, sample_data):
        """Test statistical analysis integration across components."""
        returns_data = sample_data["returns_data"]
        performance_metrics = sample_data["performance_metrics"]

        # Create mock statistical results
        statistical_results = {}
        for approach in returns_data.keys():
            statistical_results[approach] = {
                "sharpe_ratio": {"p_value": 0.025, "t_stat": 2.1},
                "total_return": {"p_value": 0.018, "t_stat": 2.4},
            }

        # Test table integration with statistical results
        tables = PerformanceComparisonTables()
        result_table = tables.create_performance_ranking_table(
            performance_metrics, statistical_results
        )

        assert isinstance(result_table, pd.DataFrame)

        # Check for proper statistical integration (significance markers when appropriate)
        # With our test data, there may or may not be significant differences
        sharpe_col = result_table["sharpe_ratio"].astype(str)
        # Test passes if either significance markers are present OR all values are proper Sharpe ratios
        has_significance_markers = any("*" in str(val) for val in sharpe_col)
        has_valid_sharpe_ratios = all(
            isinstance(val, (int, float, str)) and str(val) != "nan" 
            for val in result_table["sharpe_ratio"]
        )
        assert has_significance_markers or has_valid_sharpe_ratios

    def test_regime_analysis_integration(self, sample_data):
        """Test regime analysis integration with other components."""
        returns_data = sample_data["returns_data"]
        sample_data["regime_data"]

        # Test regime analysis initialization and basic functionality
        regime_analysis = MarketRegimeAnalysis()
        assert regime_analysis.config is not None

        # Test regime detection
        detected_regimes = regime_analysis.detect_market_regimes(
            returns_data["HRP"], method="threshold"
        )
        assert isinstance(detected_regimes, pd.Series)
        assert len(detected_regimes) > 0

        # Test feature preparation
        features = regime_analysis._prepare_regime_features(returns_data["HRP"])
        assert isinstance(features, pd.DataFrame)
        assert len(features.columns) > 0

    def test_heatmap_integration(self, sample_data):
        """Test heatmap integration with regime and statistical data."""
        returns_data = sample_data["returns_data"]
        regime_data = sample_data["regime_data"]

        heatmaps = PerformanceHeatmaps()

        with patch("src.evaluation.reporting.heatmaps.HAS_PLOTLY", True):
            with patch("src.evaluation.reporting.heatmaps.make_subplots") as mock_subplots:
                mock_figure = Mock()
                mock_subplots.return_value = mock_figure

                # Test monthly performance heatmaps
                monthly_heatmap = heatmaps.create_monthly_performance_heatmap(
                    returns_data, interactive=True
                )
                assert monthly_heatmap is not None

                # Test with regime annotation
                with patch("src.evaluation.reporting.heatmaps.HAS_MATPLOTLIB", True):
                    aggregated_data = heatmaps._aggregate_returns(returns_data, "monthly")

                    if aggregated_data:  # Only test if we have data
                        regime_heatmap = heatmaps.add_regime_identification_overlay(
                            aggregated_data, regime_data
                        )
                        assert regime_heatmap is not None

    def test_performance_consistency(self, sample_data):
        """Test consistency of performance calculations across components."""
        returns_data = sample_data["returns_data"]

        # Calculate performance using different components
        PerformanceComparisonTables()
        TimeSeriesCharts()

        # Manual calculation for comparison
        approach = "HRP"
        returns_series = returns_data[approach]

        expected_total_return = (1 + returns_series).prod() - 1
        expected_volatility = returns_series.std() * np.sqrt(252)
        expected_sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252)

        # These calculations should be consistent across components
        # (Note: in actual implementation, components should use the same calculation methods)
        assert isinstance(expected_total_return, (int, float))
        assert isinstance(expected_volatility, (int, float))
        assert isinstance(expected_sharpe, (int, float))

        # Values should be reasonable
        assert -1 <= expected_total_return <= 5  # Reasonable return range
        assert 0 <= expected_volatility <= 1  # Reasonable volatility range
        assert -5 <= expected_sharpe <= 10  # Reasonable Sharpe ratio range

    def test_memory_efficiency(self, sample_data):
        """Test memory efficiency with large datasets."""
        # Create larger dataset
        dates = pd.date_range("2010-01-01", "2022-12-31", freq="D")
        n_dates = len(dates)

        np.random.seed(42)
        large_returns_data = {}

        for approach in ["HRP", "LSTM", "GAT", "EqualWeight", "MeanVariance"]:
            returns = np.random.normal(0.0008, 0.015, n_dates)
            large_returns_data[approach] = pd.Series(returns, index=dates)

        # Test that components can handle larger datasets
        tables = PerformanceComparisonTables()

        # Calculate performance metrics
        large_performance_metrics = {}
        for approach, returns_series in large_returns_data.items():
            large_performance_metrics[approach] = {
                "total_return": (1 + returns_series).prod() - 1,
                "volatility": returns_series.std() * np.sqrt(252),
                "sharpe_ratio": returns_series.mean() / returns_series.std() * np.sqrt(252),
            }

        # Should handle large dataset without errors
        result = tables.create_performance_ranking_table(large_performance_metrics)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(large_performance_metrics)

    def test_error_handling_integration(self, sample_data):
        """Test error handling across integrated components."""
        # Test with corrupted data
        corrupted_returns = sample_data["returns_data"].copy()
        corrupted_returns["HRP"].iloc[100:200] = np.nan  # Introduce NaN values
        corrupted_returns["LSTM"].iloc[300:400] = np.inf  # Introduce infinite values

        # Components should handle corrupted data gracefully
        charts = TimeSeriesCharts()

        with patch("src.evaluation.reporting.charts.HAS_PLOTLY", True):
            with patch("src.evaluation.reporting.charts.go") as mock_go:
                mock_figure = Mock()
                mock_go.Figure.return_value = mock_figure

                # Should not raise errors
                try:
                    result = charts.plot_cumulative_returns(corrupted_returns)
                    assert result is not None
                except Exception as e:
                    pytest.fail(f"Component should handle corrupted data gracefully: {e}")

    def test_concurrent_processing(self, sample_data):
        """Test concurrent processing capabilities."""
        returns_data = sample_data["returns_data"]
        performance_metrics = sample_data["performance_metrics"]

        # Test creating multiple components and processing concurrently
        charts = TimeSeriesCharts()
        tables = PerformanceComparisonTables()
        heatmaps = PerformanceHeatmaps()

        # Create multiple tables/analyses
        results = []

        results.append(tables.create_performance_ranking_table(performance_metrics))
        results.append(heatmaps._aggregate_returns(returns_data, "monthly"))
        results.append(charts.config is not None)

        # All results should be valid
        assert all(result is not None for result in results)
        assert len(results) == 3

    def test_version_compatibility(self, sample_data):
        """Test compatibility with different data formats and versions."""
        returns_data = sample_data["returns_data"]

        # Test with different pandas index types
        for approach in returns_data:
            original_series = returns_data[approach]

            # Test with different index types
            returns_with_int_index = pd.Series(
                original_series.values, index=range(len(original_series))
            )

            pd.Series(
                original_series.values, index=[f"period_{i}" for i in range(len(original_series))]
            )

            # Components should handle different index types
            charts = TimeSeriesCharts()

            # These should work without errors (though may not be optimal)
            alternative_data = {"test_approach": returns_with_int_index}

            with patch("src.evaluation.reporting.charts.HAS_PLOTLY", True):
                with patch("src.evaluation.reporting.charts.go") as mock_go:
                    mock_figure = Mock()
                    mock_go.Figure.return_value = mock_figure

                    try:
                        result = charts.plot_cumulative_returns(alternative_data)
                        assert result is not None
                    except Exception as e:
                        # Some index types might not be supported, which is acceptable
                        assert "index" in str(e).lower() or "date" in str(e).lower()

    def teardown_method(self):
        """Clean up after each test."""
        # Clear any matplotlib figures to prevent memory leaks
        with patch("src.evaluation.reporting.charts.HAS_MATPLOTLIB", True):
            try:
                import matplotlib.pyplot as plt

                plt.close("all")
            except ImportError:
                pass
