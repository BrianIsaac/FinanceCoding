"""
Unit tests for GAT graph construction pipeline.

Tests graph building, dynamic universe handling, caching mechanisms,
and edge construction methods.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.gat.graph_builder import (
    GraphBuildConfig,
    GraphCache,
    _build_ensemble_graph,
    _compute_adaptive_knn_k,
    _prune_edges_by_weight,
    build_graph_from_returns,
    build_period_graph,
)


class TestGraphBuildConfig:
    """Test graph building configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GraphBuildConfig()

        assert config.lookback_days == 252
        assert config.filter_method == "mst"
        assert config.use_edge_attr is True
        assert config.enable_caching is True
        assert config.adaptive_knn is True

    def test_enhanced_parameters(self):
        """Test enhanced configuration parameters."""
        config = GraphBuildConfig(
            adaptive_knn=False,
            edge_pruning_threshold=0.05,
            multi_graph_methods=["knn", "mst"],
            ensemble_weights=[0.6, 0.4],
        )

        assert config.adaptive_knn is False
        assert config.edge_pruning_threshold == 0.05
        assert config.multi_graph_methods == ["knn", "mst"]
        assert config.ensemble_weights == [0.6, 0.4]


class TestAdaptiveParameters:
    """Test adaptive parameter computation."""

    def test_compute_adaptive_knn_k(self):
        """Test adaptive k-NN parameter computation."""
        config = GraphBuildConfig(adaptive_knn=True, min_knn_k=5, max_knn_k=50)

        # Small universe
        k_small = _compute_adaptive_knn_k(10, config)
        assert k_small >= config.min_knn_k
        assert k_small <= min(9, config.max_knn_k)  # Can't exceed n-1

        # Large universe
        k_large = _compute_adaptive_knn_k(400, config)
        assert k_large >= config.min_knn_k
        assert k_large <= config.max_knn_k
        assert k_large > k_small  # Should scale with universe size

    def test_adaptive_knn_disabled(self):
        """Test when adaptive k-NN is disabled."""
        config = GraphBuildConfig(adaptive_knn=False, knn_k=15)

        k = _compute_adaptive_knn_k(100, config)
        assert k == config.knn_k

    def test_edge_pruning(self):
        """Test edge pruning by weight."""
        config = GraphBuildConfig(edge_pruning_threshold=0.1)

        # Sample edges and attributes
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        edge_attrs = np.array(
            [
                [0.15, 0.15, 1.0],  # Strong edge - keep
                [0.05, 0.05, 1.0],  # Weak edge - prune
                [0.12, 0.12, 1.0],  # Medium edge - keep
                [0.08, 0.08, 1.0],  # Weak edge - prune
            ]
        )

        pruned_edges, pruned_attrs = _prune_edges_by_weight(edges, edge_attrs, config)

        # Should keep only strong edges
        assert len(pruned_edges) == 2
        assert all(pruned_attrs[:, 1] >= 0.1)  # All kept edges above threshold


class TestGraphConstruction:
    """Test graph construction methods."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data."""
        dates = pd.date_range("2020-01-01", "2021-12-31", freq="D")
        tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "META"]

        # Generate correlated returns
        np.random.seed(42)
        base_returns = np.random.normal(0.001, 0.02, size=(len(dates), 1))
        correlations = np.array([0.8, 0.6, 0.4, 0.3, 0.2])

        returns_data = []
        for _i, corr in enumerate(correlations):
            noise = np.random.normal(0, 0.01, size=(len(dates), 1))
            asset_returns = corr * base_returns + (1 - corr) * noise
            returns_data.append(asset_returns.flatten())

        returns_matrix = np.column_stack(returns_data)

        return pd.DataFrame(returns_matrix, index=dates, columns=tickers)

    def test_basic_graph_construction(self, sample_returns):
        """Test basic graph construction with MST method."""
        config = GraphBuildConfig(filter_method="mst", lookback_days=100)
        tickers = list(sample_returns.columns)

        # Use last 150 days to ensure sufficient data
        returns_window = sample_returns.tail(150)
        features_matrix = None
        ts = sample_returns.index[-1]

        graph_data = build_graph_from_returns(returns_window, features_matrix, tickers, ts, config)

        # Check graph structure
        assert graph_data.x.shape[0] == len(tickers)  # Nodes
        assert graph_data.x.shape[1] == 1  # Dummy features
        assert graph_data.edge_index.shape[0] == 2  # Source, target indices
        assert graph_data.edge_attr is not None
        assert graph_data.edge_attr.shape[1] == 3  # [rho, |rho|, sign]
        assert hasattr(graph_data, "tickers")
        assert graph_data.tickers == tickers

    def test_knn_graph_construction(self, sample_returns):
        """Test k-NN graph construction."""
        config = GraphBuildConfig(filter_method="knn", knn_k=3)
        tickers = list(sample_returns.columns)

        returns_window = sample_returns.tail(150)
        graph_data = build_graph_from_returns(
            returns_window, None, tickers, sample_returns.index[-1], config
        )

        # k-NN should create more edges than MST
        n_edges = graph_data.edge_index.shape[1]
        assert n_edges > 0
        # For undirected k-NN, expect roughly k * n_nodes edges

    def test_ensemble_graph_construction(self, sample_returns):
        """Test ensemble graph construction."""
        config = GraphBuildConfig(
            multi_graph_methods=["mst", "knn"], ensemble_weights=[0.6, 0.4], knn_k=3
        )
        tickers = list(sample_returns.columns)

        returns_window = sample_returns.tail(150)
        graph_data = build_graph_from_returns(
            returns_window, None, tickers, sample_returns.index[-1], config
        )

        # Ensemble should combine edges from multiple methods
        assert graph_data.edge_index.shape[1] > 0
        assert graph_data.edge_attr is not None

    def test_custom_features(self, sample_returns):
        """Test graph construction with custom node features."""
        config = GraphBuildConfig(filter_method="mst")
        tickers = list(sample_returns.columns)

        # Create custom features matrix
        features_matrix = np.random.rand(len(tickers), 5)

        returns_window = sample_returns.tail(150)
        graph_data = build_graph_from_returns(
            returns_window, features_matrix, tickers, sample_returns.index[-1], config
        )

        # Should use custom features
        assert graph_data.x.shape == (len(tickers), 5)
        assert not torch.allclose(graph_data.x, torch.ones_like(graph_data.x))

    def test_empty_edges_handling(self):
        """Test handling of graphs with no edges."""
        config = GraphBuildConfig(threshold_abs_corr=0.99, filter_method="threshold")

        # Create uncorrelated data (should result in no edges above threshold)
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        tickers = ["A", "B", "C"]

        np.random.seed(42)
        returns_data = np.random.normal(0, 0.01, size=(len(dates), len(tickers)))
        returns_df = pd.DataFrame(returns_data, index=dates, columns=tickers)

        graph_data = build_graph_from_returns(returns_df, None, tickers, dates[-1], config)

        # Should handle empty edge list gracefully
        assert graph_data.edge_index.shape == (2, 0)
        assert graph_data.edge_attr is None or graph_data.edge_attr.shape[0] == 0


class TestPeriodGraphBuilder:
    """Test period-based graph building."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data."""
        dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")
        tickers = ["AAPL", "MSFT", "GOOGL"]

        np.random.seed(42)
        returns_data = np.random.normal(0.001, 0.02, size=(len(dates), len(tickers)))

        return pd.DataFrame(returns_data, index=dates, columns=tickers)

    def test_build_period_graph(self, sample_returns):
        """Test period graph building."""
        config = GraphBuildConfig(lookback_days=100)
        tickers = list(sample_returns.columns)
        period_end = pd.Timestamp("2022-06-01")

        graph_data = build_period_graph(sample_returns, period_end, tickers, None, config)

        assert hasattr(graph_data, "ts")
        assert graph_data.ts.date() == period_end.date()

    def test_insufficient_history(self, sample_returns):
        """Test handling of insufficient historical data."""
        config = GraphBuildConfig(lookback_days=500)  # More than available
        tickers = list(sample_returns.columns)
        period_end = pd.Timestamp("2019-12-01")  # Before data starts (2020-01-01)

        with pytest.raises(ValueError, match="Not enough history before period_end"):
            build_period_graph(sample_returns, period_end, tickers, None, config)


class TestGraphCaching:
    """Test graph caching functionality."""

    @pytest.fixture(autouse=True)
    def setup_temp_dir(self):
        """Set up temporary directory for caching tests."""
        self.temp_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cache_initialization(self):
        """Test cache initialization."""
        cache_dir = Path(self.temp_dir) / "test_cache"
        cache = GraphCache(str(cache_dir), ttl_days=7)

        assert cache.cache_dir == str(cache_dir)
        assert cache.ttl_days == 7
        assert cache_dir.exists()

    def test_cache_key_generation(self):
        """Test cache key generation."""
        cache = GraphCache(self.temp_dir)
        config = GraphBuildConfig(lookback_days=100, filter_method="mst")

        tickers = ["AAPL", "MSFT"]
        ts = pd.Timestamp("2022-01-01")
        returns_hash = "test_hash"

        key1 = cache._get_cache_key(tickers, ts, config, returns_hash)
        key2 = cache._get_cache_key(tickers, ts, config, returns_hash)

        # Same inputs should produce same key
        assert key1 == key2
        assert isinstance(key1, str)
        assert len(key1) == 32  # MD5 hash length

    def test_cache_operations(self):
        """Test cache store and retrieve operations."""
        cache = GraphCache(self.temp_dir)
        config = GraphBuildConfig(enable_caching=True)

        # Create sample data
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        tickers = ["AAPL", "MSFT"]
        returns_data = pd.DataFrame(
            np.random.rand(len(dates), len(tickers)), index=dates, columns=tickers
        )

        ts = pd.Timestamp("2020-06-01")

        # Create a mock graph data object
        from torch_geometric.data import Data

        graph_data = Data(
            x=torch.randn(2, 5),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            edge_attr=torch.randn(1, 3),
        )
        graph_data.tickers = tickers
        graph_data.ts = ts

        # Test cache miss
        cached_graph = cache.get_cached_graph(returns_data, tickers, ts, config)
        assert cached_graph is None

        # Cache the graph
        cache.cache_graph(graph_data, returns_data, tickers, ts, config)

        # Test cache hit
        cached_graph = cache.get_cached_graph(returns_data, tickers, ts, config)
        assert cached_graph is not None
        assert torch.equal(cached_graph.x, graph_data.x)

    def test_cache_with_disabled_caching(self):
        """Test cache behavior when caching is disabled."""
        cache = GraphCache(self.temp_dir)
        config = GraphBuildConfig(enable_caching=False)

        # Create sample data
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        tickers = ["AAPL", "MSFT"]
        returns_data = pd.DataFrame(
            np.random.rand(len(dates), len(tickers)), index=dates, columns=tickers
        )

        # Should always return None when caching disabled
        cached_graph = cache.get_cached_graph(
            returns_data, tickers, pd.Timestamp("2020-06-01"), config
        )
        assert cached_graph is None


class TestEnsembleGraphBuilder:
    """Test ensemble graph building functionality."""

    def test_ensemble_graph_equal_weights(self):
        """Test ensemble graph with equal weights."""
        # Create sample correlation matrix
        n_assets = 5
        np.random.seed(42)
        C = np.random.rand(n_assets, n_assets)
        C = (C + C.T) / 2  # Make symmetric
        np.fill_diagonal(C, 1.0)

        config = GraphBuildConfig(knn_k=2)
        methods = ["mst", "knn"]

        edges, attrs = _build_ensemble_graph(C, methods, None, config)

        assert len(edges) > 0
        assert len(attrs) == len(edges)
        assert attrs.shape[1] == 3  # [rho, |rho|, sign]

    def test_ensemble_graph_weighted(self):
        """Test ensemble graph with custom weights."""
        C = np.array(
            [[1.0, 0.8, 0.3, 0.1], [0.8, 1.0, 0.4, 0.2], [0.3, 0.4, 1.0, 0.6], [0.1, 0.2, 0.6, 1.0]]
        )

        config = GraphBuildConfig(knn_k=2)
        methods = ["mst", "knn"]
        weights = [0.3, 0.7]  # Favor k-NN

        edges, attrs = _build_ensemble_graph(C, methods, weights, config)

        assert len(edges) > 0
        assert len(attrs) == len(edges)

    def test_ensemble_invalid_weights(self):
        """Test ensemble graph with mismatched weights."""
        C = np.eye(3)
        config = GraphBuildConfig()

        methods = ["mst", "knn"]
        invalid_weights = [0.5]  # Wrong length

        with pytest.raises(ValueError, match="Number of weights must match"):
            _build_ensemble_graph(C, methods, invalid_weights, config)


if __name__ == "__main__":
    pytest.main([__file__])
