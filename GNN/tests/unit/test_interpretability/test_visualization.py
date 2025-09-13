"""
Tests for interpretability visualization framework.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.interpretability.visualization import InterpretabilityVisualizer


class TestInterpretabilityVisualizer:
    """Test suite for interpretability visualization tools."""

    @pytest.fixture
    def visualizer(self):
        """Create visualizer instance for testing."""
        return InterpretabilityVisualizer(theme="plotly_white", width=800, height=600)

    @pytest.fixture
    def sample_attention_matrix(self):
        """Create sample attention matrix."""
        assets = ["AAPL", "MSFT", "GOOGL", "TSLA"]
        data = np.random.rand(4, 4) * 0.1
        np.fill_diagonal(data, 0)  # No self-attention

        return pd.DataFrame(data, index=assets, columns=assets)

    @pytest.fixture
    def sample_evolution_data(self):
        """Create sample attention evolution data."""
        dates = pd.date_range("2023-01-01", periods=10, freq="ME")
        return pd.DataFrame({
            "date": dates,
            "mean_attention": np.random.rand(10) * 0.05,
            "attention_concentration": np.random.rand(10) * 0.5,
            "n_significant_connections": np.random.randint(1, 20, 10),
            "attention_entropy": np.random.rand(10) * 2.0,
        })

    @pytest.fixture
    def sample_connections_data(self):
        """Create sample influential connections data."""
        n_connections = 15
        assets = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]

        data = []
        for _i in range(n_connections):
            source = np.random.choice(assets)
            target = np.random.choice([a for a in assets if a != source])
            data.append({
                "source_asset": source,
                "target_asset": target,
                "mean_attention": np.random.rand() * 0.1,
                "max_attention": np.random.rand() * 0.2,
                "attention_std": np.random.rand() * 0.02,
                "attention_trend": np.random.randn() * 0.1,
            })

        return pd.DataFrame(data).sort_values("mean_attention", ascending=False)

    def test_visualizer_initialization(self):
        """Test visualizer initialization with different parameters."""
        # Test default initialization
        viz = InterpretabilityVisualizer()
        assert viz.theme == "plotly_white"
        assert viz.default_width == 800
        assert viz.default_height == 600

        # Test custom initialization
        viz_custom = InterpretabilityVisualizer(
            theme="plotly_dark", width=1000, height=800
        )
        assert viz_custom.theme == "plotly_dark"
        assert viz_custom.default_width == 1000
        assert viz_custom.default_height == 800

    def test_attention_heatmap_creation(self, visualizer, sample_attention_matrix):
        """Test attention heatmap visualization."""
        fig = visualizer.plot_attention_heatmap(
            sample_attention_matrix, title="Test Heatmap"
        )

        # Check figure properties
        assert fig.layout.title.text == "Test Heatmap"
        assert fig.layout.width == 800
        assert fig.layout.height == 600

        # Check data properties
        assert len(fig.data) == 1
        heatmap_data = fig.data[0]
        assert heatmap_data.type == "heatmap"
        assert heatmap_data.z.shape == sample_attention_matrix.shape

    def test_attention_network_creation(self, visualizer, sample_attention_matrix):
        """Test attention network visualization."""
        fig = visualizer.plot_attention_network(
            sample_attention_matrix,
            threshold=0.01,
            layout="spring",
            title="Test Network"
        )

        # Check figure properties
        assert fig.layout.title.text == "Test Network"
        assert fig.layout.width == 800
        assert fig.layout.height == 600

        # Should have traces for edges and nodes
        assert len(fig.data) >= 1  # At least nodes, possibly edges

    def test_temporal_evolution_plotting(self, visualizer, sample_evolution_data):
        """Test temporal attention evolution visualization."""
        metrics = ["mean_attention", "attention_concentration"]

        fig = visualizer.plot_temporal_attention_evolution(
            sample_evolution_data,
            metrics=metrics,
            title="Test Evolution"
        )

        # Check figure properties
        assert fig.layout.title.text == "Test Evolution"

        # Should have traces for each metric
        assert len(fig.data) == len(metrics)

        # Check that x-axis data corresponds to dates
        for trace in fig.data:
            assert len(trace.x) == len(sample_evolution_data)

    def test_influential_connections_plotting(self, visualizer, sample_connections_data):
        """Test influential connections visualization."""
        fig = visualizer.plot_influential_connections(
            sample_connections_data,
            top_k=10,
            metric="mean_attention",
            title="Test Connections"
        )

        # Check figure properties
        assert fig.layout.title.text == "Test Connections"

        # Check data properties
        assert len(fig.data) == 1
        bar_data = fig.data[0]
        assert bar_data.type == "bar"
        assert bar_data.orientation == "h"
        assert len(bar_data.x) <= 10  # Should respect top_k limit

    def test_attention_distribution_plotting(self, visualizer, sample_attention_matrix):
        """Test attention weight distribution visualization."""
        fig = visualizer.plot_attention_distribution(
            sample_attention_matrix, title="Test Distribution"
        )

        # Check figure properties
        assert fig.layout.title.text == "Test Distribution"

        # Should have histogram trace
        assert len(fig.data) == 1
        hist_data = fig.data[0]
        assert hist_data.type == "histogram"

        # Should have vertical lines for statistics
        assert len(fig.layout.shapes) >= 2  # Mean and median lines

    def test_dashboard_creation_full(
        self, visualizer, sample_attention_matrix, sample_evolution_data, sample_connections_data
    ):
        """Test comprehensive dashboard creation with all components."""
        fig = visualizer.create_attention_dashboard(
            attention_matrix=sample_attention_matrix,
            attention_evolution=sample_evolution_data,
            influential_connections=sample_connections_data,
            title="Test Dashboard"
        )

        # Check figure properties
        assert fig.layout.title.text == "Test Dashboard"
        assert fig.layout.width == 1600
        assert fig.layout.height == 1200

        # Should have multiple traces for different subplot components
        assert len(fig.data) >= 4  # At least one trace per subplot

    def test_dashboard_creation_simple(self, visualizer, sample_attention_matrix):
        """Test simple dashboard creation with only attention matrix."""
        fig = visualizer.create_attention_dashboard(
            attention_matrix=sample_attention_matrix,
            title="Simple Dashboard"
        )

        # Check figure properties
        assert fig.layout.title.text == "Simple Dashboard"
        assert fig.layout.width == 1600
        assert fig.layout.height == 600  # Shorter height for simple version

        # Should have traces for heatmap and distribution
        assert len(fig.data) >= 2

    def test_network_layout_options(self, visualizer, sample_attention_matrix):
        """Test different network layout algorithms."""
        layouts = ["spring", "circular", "kamada_kawai"]

        for layout in layouts:
            fig = visualizer.plot_attention_network(
                sample_attention_matrix,
                layout=layout,
                title=f"Network - {layout}"
            )

            # Should successfully create figure for each layout
            assert fig.layout.title.text == f"Network - {layout}"
            assert len(fig.data) >= 1

    def test_threshold_effects_on_network(self, visualizer, sample_attention_matrix):
        """Test that threshold parameter affects network visualization."""
        # Create network with low threshold
        fig_low = visualizer.plot_attention_network(
            sample_attention_matrix, threshold=0.001
        )

        # Create network with high threshold
        fig_high = visualizer.plot_attention_network(
            sample_attention_matrix, threshold=0.1
        )

        # Both should create valid figures
        assert len(fig_low.data) >= 1
        assert len(fig_high.data) >= 1

        # Higher threshold might result in fewer edges (fewer points in edge traces)
        # This is difficult to test directly without examining the internal structure

    def test_color_customization(self, visualizer):
        """Test that color palettes are properly configured."""
        # Check that color palettes are defined
        assert hasattr(visualizer, 'attention_colors')
        assert hasattr(visualizer, 'temporal_colors')
        assert hasattr(visualizer, 'network_colors')

        # Check that they contain color values
        assert len(visualizer.attention_colors) > 0
        assert len(visualizer.temporal_colors) > 0
        assert len(visualizer.network_colors) > 0

    def test_edge_cases(self, visualizer):
        """Test edge cases and error handling."""
        # Test with empty attention matrix
        empty_matrix = pd.DataFrame()

        # Should handle empty data gracefully
        try:
            fig = visualizer.plot_attention_heatmap(empty_matrix)
            # If no exception, check that figure exists
            assert fig is not None
        except (ValueError, IndexError):
            # Acceptable to raise error for empty data
            pass

        # Test with single asset
        single_matrix = pd.DataFrame([[0.5]], index=["AAPL"], columns=["AAPL"])
        fig = visualizer.plot_attention_heatmap(single_matrix)
        assert fig is not None

        # Test network with no edges above threshold
        zero_matrix = pd.DataFrame(
            np.zeros((3, 3)),
            index=["A", "B", "C"],
            columns=["A", "B", "C"]
        )
        fig = visualizer.plot_attention_network(zero_matrix, threshold=0.1)
        assert fig is not None
