"""
Tests for GAT attention weight extraction and analysis.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from src.evaluation.interpretability.gat_explainer import (
    AttentionAnalysisConfig,
    GATExplainer,
)
from src.models.gat.gat_model import GATPortfolio


class TestGATExplainer:
    """Test suite for GAT explainer functionality."""

    @pytest.fixture
    def sample_gat_model(self):
        """Create sample GAT model for testing."""
        model = GATPortfolio(
            in_dim=10,
            hidden_dim=32,
            heads=4,
            num_layers=2,
            dropout=0.1,
            use_edge_attr=True,
            head="direct",
            activation="sparsemax",
        )
        model.eval()
        return model

    @pytest.fixture
    def sample_graph_data(self):
        """Create sample graph data for testing."""
        n_nodes = 5
        x = torch.randn(n_nodes, 10)

        # Create simple edge connectivity
        edge_index = torch.tensor([
            [0, 0, 1, 1, 2, 2, 3, 3, 4],
            [1, 2, 2, 3, 3, 4, 4, 0, 0],
        ], dtype=torch.long)

        edge_attr = torch.randn(edge_index.size(1), 3)
        mask_valid = torch.ones(n_nodes, dtype=torch.bool)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), mask_valid

    @pytest.fixture
    def attention_config(self):
        """Create attention analysis configuration."""
        return AttentionAnalysisConfig(
            aggregation_method="mean",
            temporal_windows=6,
            attention_threshold=0.01,
            normalize_weights=True,
        )

    def test_gat_explainer_initialization(self, sample_gat_model, attention_config):
        """Test GAT explainer initialization."""
        explainer = GATExplainer(sample_gat_model, attention_config)

        assert explainer.model is sample_gat_model
        assert explainer.config is attention_config
        assert explainer.model.training is False  # Should be in eval mode
        assert len(explainer.attention_weights) == 0
        assert len(explainer.layer_outputs) == 0

    def test_attention_weight_extraction(self, sample_gat_model, sample_graph_data, attention_config):
        """Test attention weight extraction from GAT model."""
        graph_data, mask_valid = sample_graph_data
        explainer = GATExplainer(sample_gat_model, attention_config)

        # Extract attention weights
        attention_weights = explainer.extract_attention_weights(
            x=graph_data.x,
            edge_index=graph_data.edge_index,
            mask_valid=mask_valid,
            edge_attr=graph_data.edge_attr,
        )

        # Check that we have attention weights
        assert isinstance(attention_weights, dict)
        assert len(attention_weights) > 0

        # Check layer-specific weights
        layer_keys = [k for k in attention_weights.keys() if k.startswith("layer_")]
        assert len(layer_keys) >= 1  # Should have at least one layer

        for layer_key in layer_keys:
            weights = attention_weights[layer_key]
            assert isinstance(weights, torch.Tensor)
            assert weights.dim() == 2  # [n_edges, n_heads or 1]
            assert weights.size(0) == graph_data.edge_index.size(1)  # Match number of edges

    def test_attention_matrix_building(self, sample_gat_model, sample_graph_data, attention_config):
        """Test attention matrix construction."""
        graph_data, mask_valid = sample_graph_data
        explainer = GATExplainer(sample_gat_model, attention_config)

        # Create dummy attention weights
        n_edges = graph_data.edge_index.size(1)
        attention_weights = torch.randn(n_edges, 1).abs()  # Positive weights

        asset_names = [f"ASSET_{i}" for i in range(graph_data.x.size(0))]

        # Build attention matrix
        attention_matrix = explainer.build_attention_matrix(
            attention_weights=attention_weights,
            edge_index=graph_data.edge_index,
            n_nodes=graph_data.x.size(0),
            asset_names=asset_names,
        )

        # Check matrix properties
        assert isinstance(attention_matrix, pd.DataFrame)
        assert attention_matrix.shape == (graph_data.x.size(0), graph_data.x.size(0))
        assert list(attention_matrix.index) == asset_names
        assert list(attention_matrix.columns) == asset_names
        assert (attention_matrix.values >= 0).all()  # Non-negative weights

    def test_top_attention_pairs(self, sample_gat_model, attention_config):
        """Test extraction of top attention pairs."""
        explainer = GATExplainer(sample_gat_model, attention_config)

        # Create sample attention matrix
        asset_names = ["AAPL", "MSFT", "GOOGL", "TSLA"]
        n_assets = len(asset_names)
        attention_data = np.random.rand(n_assets, n_assets) * 0.1
        np.fill_diagonal(attention_data, 0)  # No self-attention

        attention_matrix = pd.DataFrame(
            attention_data,
            index=asset_names,
            columns=asset_names,
        )

        # Get top attention pairs
        top_pairs = explainer.get_top_attention_pairs(attention_matrix, top_k=5)

        # Check results
        assert isinstance(top_pairs, pd.DataFrame)
        if not top_pairs.empty:
            assert len(top_pairs) <= 5
            assert all(col in top_pairs.columns for col in [
                "source_asset", "target_asset", "attention_weight", "relationship_type"
            ])
            # Check that pairs are sorted by attention weight
            assert top_pairs["attention_weight"].is_monotonic_decreasing

    def test_attention_aggregation_methods(self, sample_gat_model, sample_graph_data):
        """Test different attention weight aggregation methods."""
        graph_data, mask_valid = sample_graph_data

        methods = ["mean", "max", "sum"]

        for method in methods:
            config = AttentionAnalysisConfig(aggregation_method=method)
            explainer = GATExplainer(sample_gat_model, config)

            # Create dummy layer weights
            n_edges = graph_data.edge_index.size(1)
            layer_weights = [
                torch.randn(n_edges, 4).abs(),  # 4 attention heads
                torch.randn(n_edges, 4).abs(),
            ]

            # Test aggregation
            aggregated = explainer._aggregate_attention_weights(
                layer_weights, graph_data.edge_index, mask_valid
            )

            assert isinstance(aggregated, torch.Tensor)
            assert aggregated.shape == (n_edges, 1)
            assert (aggregated >= 0).all()  # Non-negative weights

    def test_attention_entropy_calculation(self, sample_gat_model, attention_config):
        """Test attention entropy calculation."""
        explainer = GATExplainer(sample_gat_model, attention_config)

        # Test with uniform distribution (high entropy)
        uniform_matrix = np.ones((4, 4)) * 0.25
        np.fill_diagonal(uniform_matrix, 0)
        uniform_entropy = explainer._calculate_attention_entropy(uniform_matrix)

        # Test with concentrated distribution (low entropy)
        concentrated_matrix = np.zeros((4, 4))
        concentrated_matrix[0, 1] = 1.0
        concentrated_entropy = explainer._calculate_attention_entropy(concentrated_matrix)

        # Uniform distribution should have higher entropy
        assert uniform_entropy > concentrated_entropy
        assert uniform_entropy >= 0
        assert concentrated_entropy >= 0

    def test_temporal_attention_analysis(self, sample_gat_model, attention_config):
        """Test temporal attention evolution analysis."""
        explainer = GATExplainer(sample_gat_model, attention_config)

        # Create sample attention matrices over time
        n_assets = 4
        asset_names = [f"ASSET_{i}" for i in range(n_assets)]
        dates = pd.date_range("2023-01-01", periods=5, freq="ME")

        attention_matrices = []
        for _ in dates:
            matrix_data = np.random.rand(n_assets, n_assets) * 0.1
            np.fill_diagonal(matrix_data, 0)
            attention_matrices.append(pd.DataFrame(
                matrix_data, index=asset_names, columns=asset_names
            ))

        # Analyze evolution
        evolution_df = explainer._analyze_attention_evolution(attention_matrices, dates)

        assert isinstance(evolution_df, pd.DataFrame)
        assert len(evolution_df) == len(dates)
        assert "date" in evolution_df.columns
        assert "mean_attention" in evolution_df.columns
        assert "attention_concentration" in evolution_df.columns
        assert "attention_entropy" in evolution_df.columns

    def test_influential_connections_analysis(self, sample_gat_model, attention_config):
        """Test influential connections identification."""
        explainer = GATExplainer(sample_gat_model, attention_config)

        # Create sample attention matrices over time
        n_assets = 3
        asset_names = ["AAPL", "MSFT", "GOOGL"]

        attention_matrices = []
        for _ in range(4):
            matrix_data = np.random.rand(n_assets, n_assets) * 0.1
            np.fill_diagonal(matrix_data, 0)
            attention_matrices.append(pd.DataFrame(
                matrix_data, index=asset_names, columns=asset_names
            ))

        # Identify influential connections
        influential_df = explainer._identify_influential_connections(
            attention_matrices, asset_names
        )

        assert isinstance(influential_df, pd.DataFrame)
        assert len(influential_df) <= n_assets * (n_assets - 1)  # Max possible connections
        assert all(col in influential_df.columns for col in [
            "source_asset", "target_asset", "mean_attention", "attention_std"
        ])
        # Check that connections are sorted by mean attention
        assert influential_df["mean_attention"].is_monotonic_decreasing

    def test_config_parameter_effects(self, sample_gat_model, sample_graph_data):
        """Test that configuration parameters affect results appropriately."""
        graph_data, mask_valid = sample_graph_data

        # Test threshold effect
        low_threshold_config = AttentionAnalysisConfig(attention_threshold=0.001)
        high_threshold_config = AttentionAnalysisConfig(attention_threshold=0.1)

        explainer_low = GATExplainer(sample_gat_model, low_threshold_config)
        explainer_high = GATExplainer(sample_gat_model, high_threshold_config)

        # Create test attention matrix
        asset_names = ["A", "B", "C"]
        attention_data = np.array([
            [0.0, 0.05, 0.02],
            [0.03, 0.0, 0.08],
            [0.01, 0.04, 0.0],
        ])
        attention_matrix = pd.DataFrame(attention_data, index=asset_names, columns=asset_names)

        # Get top pairs with different thresholds
        pairs_low = explainer_low.get_top_attention_pairs(attention_matrix)
        pairs_high = explainer_high.get_top_attention_pairs(attention_matrix)

        # Higher threshold should result in fewer pairs
        assert len(pairs_high) <= len(pairs_low)

    def test_edge_cases(self, sample_gat_model, attention_config):
        """Test edge cases and error handling."""
        explainer = GATExplainer(sample_gat_model, attention_config)

        # Test with empty attention matrix
        empty_matrix = pd.DataFrame()
        top_pairs = explainer.get_top_attention_pairs(empty_matrix)
        assert len(top_pairs) == 0

        # Test with single asset
        single_asset_matrix = pd.DataFrame([[0.0]], index=["AAPL"], columns=["AAPL"])
        single_pairs = explainer.get_top_attention_pairs(single_asset_matrix)
        # Should be empty since we exclude self-attention by default
        assert len(single_pairs) == 0
