"""
Unit tests for time series charts module.

Tests the comprehensive time series visualization framework including
cumulative returns, drawdown analysis, rolling metrics, and interactive features.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch

from src.evaluation.reporting.charts import ChartConfig, TimeSeriesCharts


class TestChartConfig:
    """Test ChartConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ChartConfig()
        
        assert config.figsize == (12, 8)
        assert config.dpi == 300
        assert config.style == "whitegrid"
        assert config.color_palette == "husl"
        assert config.save_format == "png"
        assert config.interactive is True
        assert config.confidence_level == 0.95

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ChartConfig(
            figsize=(10, 6),
            dpi=150,
            style="darkgrid",
            color_palette="Set2",
            save_format="pdf",
            interactive=False,
            confidence_level=0.99
        )
        
        assert config.figsize == (10, 6)
        assert config.dpi == 150
        assert config.style == "darkgrid"
        assert config.color_palette == "Set2"
        assert config.save_format == "pdf"
        assert config.interactive is False
        assert config.confidence_level == 0.99


class TestTimeSeriesCharts:
    """Test TimeSeriesCharts class."""

    @pytest.fixture
    def sample_returns_data(self):
        """Create sample returns data for testing."""
        dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")
        np.random.seed(42)
        
        returns_data = {
            "HRP": pd.Series(
                np.random.normal(0.0008, 0.015, len(dates)), index=dates
            ),
            "LSTM": pd.Series(
                np.random.normal(0.0010, 0.018, len(dates)), index=dates
            ),
            "GAT": pd.Series(
                np.random.normal(0.0012, 0.020, len(dates)), index=dates
            ),
        }
        
        return returns_data

    @pytest.fixture
    def sample_confidence_intervals(self):
        """Create sample confidence intervals for testing."""
        dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")
        np.random.seed(42)
        
        confidence_intervals = {
            "HRP": {
                "lower": pd.Series(
                    np.random.normal(0.0006, 0.012, len(dates)), index=dates
                ),
                "upper": pd.Series(
                    np.random.normal(0.0010, 0.018, len(dates)), index=dates
                ),
            },
        }
        
        return confidence_intervals

    @pytest.fixture
    def charts(self):
        """Create TimeSeriesCharts instance for testing."""
        config = ChartConfig()
        return TimeSeriesCharts(config)

    def test_initialization(self):
        """Test TimeSeriesCharts initialization."""
        charts = TimeSeriesCharts()
        assert charts.config is not None
        assert charts.config.figsize == (12, 8)

    def test_initialization_with_config(self):
        """Test TimeSeriesCharts initialization with custom config."""
        config = ChartConfig(figsize=(10, 6), dpi=150)
        charts = TimeSeriesCharts(config)
        
        assert charts.config.figsize == (10, 6)
        assert charts.config.dpi == 150

    @patch("src.evaluation.reporting.charts.HAS_PLOTLY", True)
    @patch("src.evaluation.reporting.charts.go")
    def test_plot_cumulative_returns_interactive(self, mock_go, charts, sample_returns_data):
        """Test interactive cumulative returns plotting."""
        mock_figure = Mock()
        mock_go.Figure.return_value = mock_figure
        
        result = charts.plot_cumulative_returns(
            sample_returns_data, interactive=True
        )
        
        assert result == mock_figure
        mock_go.Figure.assert_called_once()

    @patch("src.evaluation.reporting.charts.HAS_MATPLOTLIB", True)
    @patch("src.evaluation.reporting.charts.plt")
    def test_plot_cumulative_returns_static(self, mock_plt, charts, sample_returns_data):
        """Test static cumulative returns plotting."""
        mock_figure = Mock()
        mock_plt.subplots.return_value = (mock_figure, Mock())
        
        result = charts.plot_cumulative_returns(
            sample_returns_data, interactive=False
        )
        
        assert result == mock_figure
        mock_plt.subplots.assert_called_once()

    def test_plot_cumulative_returns_with_confidence_intervals(
        self, charts, sample_returns_data, sample_confidence_intervals
    ):
        """Test cumulative returns plotting with confidence intervals."""
        with patch("src.evaluation.reporting.charts.HAS_PLOTLY", True):
            with patch("src.evaluation.reporting.charts.go") as mock_go:
                mock_figure = Mock()
                mock_go.Figure.return_value = mock_figure
                
                charts.plot_cumulative_returns(
                    sample_returns_data,
                    confidence_intervals=sample_confidence_intervals,
                    interactive=True
                )
                
                # Verify confidence intervals were processed
                assert mock_go.Figure.return_value.add_trace.call_count >= len(sample_returns_data)

    def test_plot_cumulative_returns_with_benchmark(
        self, charts, sample_returns_data
    ):
        """Test cumulative returns plotting with benchmark."""
        dates = sample_returns_data["HRP"].index
        benchmark_returns = pd.Series(
            np.random.normal(0.0005, 0.012, len(dates)), index=dates
        )
        
        with patch("src.evaluation.reporting.charts.HAS_PLOTLY", True):
            with patch("src.evaluation.reporting.charts.go") as mock_go:
                mock_figure = Mock()
                mock_go.Figure.return_value = mock_figure
                
                charts.plot_cumulative_returns(
                    sample_returns_data,
                    benchmark_returns=benchmark_returns,
                    interactive=True
                )
                
                # Verify benchmark was added
                assert mock_go.Figure.return_value.add_trace.call_count >= len(sample_returns_data) + 1

    @patch("src.evaluation.reporting.charts.HAS_PLOTLY", True)
    @patch("src.evaluation.reporting.charts.go")
    def test_plot_drawdown_analysis_interactive(self, mock_go, charts, sample_returns_data):
        """Test interactive drawdown analysis plotting."""
        mock_figure = Mock()
        mock_make_subplots = Mock(return_value=mock_figure)
        
        with patch("src.evaluation.reporting.charts.make_subplots", mock_make_subplots):
            result = charts.plot_drawdown_analysis(sample_returns_data, interactive=True)
            
            assert result == mock_figure
            mock_make_subplots.assert_called_once()

    @patch("src.evaluation.reporting.charts.HAS_MATPLOTLIB", True)
    @patch("src.evaluation.reporting.charts.plt")
    def test_plot_drawdown_analysis_static(self, mock_plt, charts, sample_returns_data):
        """Test static drawdown analysis plotting."""
        mock_figure = Mock()
        mock_axes = [Mock(), Mock()]
        mock_plt.subplots.return_value = (mock_figure, mock_axes)
        
        result = charts.plot_drawdown_analysis(sample_returns_data, interactive=False)
        
        assert result == mock_figure
        mock_plt.subplots.assert_called_once()

    def test_plot_rolling_performance_metrics(self, charts):
        """Test rolling performance metrics plotting."""
        # Create sample rolling metrics data
        dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")
        np.random.seed(42)
        
        rolling_metrics_data = {
            "HRP": {
                "sharpe_ratio": pd.Series(np.random.normal(1.2, 0.3, len(dates)), index=dates),
                "information_ratio": pd.Series(np.random.normal(0.8, 0.2, len(dates)), index=dates),
                "max_drawdown": pd.Series(np.random.normal(-0.1, 0.05, len(dates)), index=dates),
                "volatility": pd.Series(np.random.normal(0.15, 0.03, len(dates)), index=dates),
            },
        }
        
        with patch("src.evaluation.reporting.charts.HAS_PLOTLY", True):
            with patch("src.evaluation.reporting.charts.make_subplots") as mock_make_subplots:
                mock_figure = Mock()
                mock_make_subplots.return_value = mock_figure
                
                result = charts.plot_rolling_performance_metrics(
                    rolling_metrics_data, interactive=True
                )
                
                assert result == mock_figure
                mock_make_subplots.assert_called_once()

    def test_plot_rolling_performance_metrics_with_custom_metrics(self, charts):
        """Test rolling performance metrics plotting with custom metrics."""
        dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")
        np.random.seed(42)
        
        rolling_metrics_data = {
            "HRP": {
                "custom_metric_1": pd.Series(np.random.normal(0.1, 0.02, len(dates)), index=dates),
                "custom_metric_2": pd.Series(np.random.normal(0.2, 0.05, len(dates)), index=dates),
            },
        }
        
        custom_metrics = ["custom_metric_1", "custom_metric_2"]
        
        with patch("src.evaluation.reporting.charts.HAS_PLOTLY", True):
            with patch("src.evaluation.reporting.charts.make_subplots") as mock_make_subplots:
                mock_figure = Mock()
                mock_make_subplots.return_value = mock_figure
                
                result = charts.plot_rolling_performance_metrics(
                    rolling_metrics_data,
                    metrics_to_plot=custom_metrics,
                    interactive=True
                )
                
                assert result == mock_figure
                # Verify custom metrics were used
                call_args = mock_make_subplots.call_args
                assert call_args[1]["rows"] == len(custom_metrics)

    def test_create_performance_dashboard(self, charts, sample_returns_data):
        """Test performance dashboard creation."""
        with patch("src.evaluation.reporting.charts.HAS_PLOTLY", True):
            with patch("src.evaluation.reporting.charts.make_subplots") as mock_make_subplots:
                with patch("src.evaluation.reporting.charts.PerformanceAnalytics") as mock_perf:
                    mock_figure = Mock()
                    mock_make_subplots.return_value = mock_figure
                    mock_perf.return_value.sharpe_ratio.return_value = 1.2
                    mock_perf.return_value.maximum_drawdown.return_value = (-0.1, None)
                    
                    result = charts.create_performance_dashboard(sample_returns_data)
                    
                    assert result == mock_figure
                    mock_make_subplots.assert_called_once()

    def test_export_chart(self, charts):
        """Test chart export functionality."""
        with patch("src.evaluation.reporting.charts.HAS_PLOTLY", True):
            mock_figure = Mock()
            mock_figure.write_image = Mock()
            mock_figure.write_html = Mock()
            
            # Test export
            exported_files = charts.export_chart(
                mock_figure, "test_chart", ["png", "html"]
            )
            
            assert "png" in exported_files
            assert "html" in exported_files
            mock_figure.write_image.assert_called_once()
            mock_figure.write_html.assert_called_once()

    def test_export_chart_matplotlib(self, charts):
        """Test chart export functionality for matplotlib figures."""
        with patch("src.evaluation.reporting.charts.HAS_MATPLOTLIB", True):
            mock_figure = Mock()
            mock_figure.savefig = Mock()
            
            # Test export
            exported_files = charts.export_chart(
                mock_figure, "test_chart", ["png", "pdf"]
            )
            
            assert "png" in exported_files
            assert "pdf" in exported_files
            assert mock_figure.savefig.call_count == 2

    def test_plot_no_libraries_available(self):
        """Test behavior when no plotting libraries are available."""
        with patch("src.evaluation.reporting.charts.HAS_PLOTLY", False):
            with patch("src.evaluation.reporting.charts.HAS_MATPLOTLIB", False):
                charts = TimeSeriesCharts()
                
                with pytest.raises(ImportError, match="Neither Plotly nor Matplotlib available"):
                    charts.plot_cumulative_returns({"test": pd.Series([1, 2, 3])})

    def test_save_path_handling(self, charts, sample_returns_data, tmp_path):
        """Test save path handling in plot methods."""
        save_path = tmp_path / "test_chart.png"
        
        with patch("src.evaluation.reporting.charts.HAS_PLOTLY", True):
            with patch("src.evaluation.reporting.charts.go") as mock_go:
                mock_figure = Mock()
                mock_go.Figure.return_value = mock_figure
                
                charts.plot_cumulative_returns(
                    sample_returns_data,
                    save_path=str(save_path),
                    interactive=True
                )
                
                # Verify save method was called
                mock_figure.write_html.assert_called_once()

    def test_empty_data_handling(self, charts):
        """Test handling of empty data."""
        empty_data = {}
        
        with patch("src.evaluation.reporting.charts.HAS_PLOTLY", True):
            with patch("src.evaluation.reporting.charts.go") as mock_go:
                mock_figure = Mock()
                mock_go.Figure.return_value = mock_figure
                
                # Should not raise an error
                result = charts.plot_cumulative_returns(empty_data, interactive=True)
                assert result == mock_figure

    def test_single_approach_data(self, charts):
        """Test plotting with single approach data."""
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        single_data = {
            "HRP": pd.Series(np.random.normal(0.001, 0.015, len(dates)), index=dates)
        }
        
        with patch("src.evaluation.reporting.charts.HAS_PLOTLY", True):
            with patch("src.evaluation.reporting.charts.go") as mock_go:
                mock_figure = Mock()
                mock_go.Figure.return_value = mock_figure
                
                result = charts.plot_cumulative_returns(single_data, interactive=True)
                assert result == mock_figure

    def test_missing_data_in_series(self, charts):
        """Test handling of missing data in time series."""
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        data_with_nan = {
            "HRP": pd.Series(np.random.normal(0.001, 0.015, len(dates)), index=dates)
        }
        # Introduce some NaN values
        data_with_nan["HRP"].iloc[10:20] = np.nan
        
        with patch("src.evaluation.reporting.charts.HAS_PLOTLY", True):
            with patch("src.evaluation.reporting.charts.go") as mock_go:
                mock_figure = Mock()
                mock_go.Figure.return_value = mock_figure
                
                # Should handle NaN values gracefully
                result = charts.plot_cumulative_returns(data_with_nan, interactive=True)
                assert result == mock_figure

    def test_different_date_ranges(self, charts):
        """Test plotting with different date ranges across approaches."""
        dates1 = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        dates2 = pd.date_range("2020-06-01", "2021-05-31", freq="D")
        
        mixed_data = {
            "HRP": pd.Series(np.random.normal(0.001, 0.015, len(dates1)), index=dates1),
            "LSTM": pd.Series(np.random.normal(0.001, 0.018, len(dates2)), index=dates2),
        }
        
        with patch("src.evaluation.reporting.charts.HAS_PLOTLY", True):
            with patch("src.evaluation.reporting.charts.go") as mock_go:
                mock_figure = Mock()
                mock_go.Figure.return_value = mock_figure
                
                result = charts.plot_cumulative_returns(mixed_data, interactive=True)
                assert result == mock_figure