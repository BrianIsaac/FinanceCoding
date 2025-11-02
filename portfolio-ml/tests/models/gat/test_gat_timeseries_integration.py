"""Simplified integration tests for GAT with time-series features.

Focuses on core functionality without requiring full training pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.gat.model import GATModelConfig
from src.models.gat.gat_model import GATPortfolio


@pytest.fixture
def sample_timeseries_data():
    """Create sample time-series data for testing."""
    num_assets = 20
    seq_length = 60
    num_features = 1
    return torch.randn(num_assets, seq_length, num_features)


class TestGATWithTimeseriesConfig:
    """Test GAT instantiation and forward pass with time-series configuration."""

    def test_paper_preset_configuration(self):
        """Test paper preset creates time-series configuration."""
        config = GATModelConfig(preset="paper_reproduction")

        assert config.node_feature_type == "timeseries"
        assert config.timeseries_length == 756
        assert config.temporal_encoder_type == "lstm"
        assert config.use_temporal_encoder is True

    def test_enhanced_preset_static(self):
        """Test enhanced preset still uses static features."""
        config = GATModelConfig(preset="enhanced")

        assert config.node_feature_type == "static"

    def test_timeseries_gat_instantiation(self):
        """Test GAT can be instantiated with time-series configuration."""
        config = GATModelConfig(
            node_feature_type="timeseries",
            timeseries_length=60,
            temporal_encoder_type="conv1d",
        )

        # Create GAT model
        gat = GATPortfolio(
            in_dim=1,
            hidden_dim=config.hidden_dim,
            heads=config.num_attention_heads,
            num_layers=config.num_layers,
            use_temporal_encoder=config.use_temporal_encoder,
            temporal_encoder_type=config.temporal_encoder_type,
            timeseries_length=config.timeseries_length,
        )

        assert gat.temporal_encoder is not None
        assert hasattr(gat, "temporal_encoder")

    def test_all_encoder_types_work(self, sample_timeseries_data):
        """Test all three encoder types work with GAT."""
        for encoder_type in ["lstm", "conv1d", "transformer"]:
            config = GATModelConfig(
                node_feature_type="timeseries",
                timeseries_length=60,
                temporal_encoder_type=encoder_type,
            )

            gat = GATPortfolio(
                in_dim=1,
                hidden_dim=config.hidden_dim,
                heads=config.num_attention_heads,
                num_layers=config.num_layers,
                use_temporal_encoder=config.use_temporal_encoder,
                temporal_encoder_type=config.temporal_encoder_type,
                timeseries_length=config.timeseries_length,
            )

            # Test temporal encoder directly
            with torch.no_grad():
                encoded = gat.temporal_encoder(sample_timeseries_data)

            assert encoded.shape == (20, config.hidden_dim), f"{encoder_type} failed"
            assert torch.isfinite(encoded).all(), f"{encoder_type}: NaN/Inf"

    def test_different_timeseries_lengths(self):
        """Test GAT works with different time-series lengths."""
        for ts_length in [10, 30, 60, 120]:
            config = GATModelConfig(
                node_feature_type="timeseries",
                timeseries_length=ts_length,
                temporal_encoder_type="conv1d",
            )

            gat = GATPortfolio(
                in_dim=1,
                hidden_dim=config.hidden_dim,
                heads=config.num_attention_heads,
                num_layers=config.num_layers,
                use_temporal_encoder=config.use_temporal_encoder,
                temporal_encoder_type=config.temporal_encoder_type,
                timeseries_length=config.timeseries_length,
            )

            # Test with appropriate sequence length
            x = torch.randn(10, ts_length, 1)
            with torch.no_grad():
                encoded = gat.temporal_encoder(x)

            assert encoded.shape == (10, config.hidden_dim)
            assert torch.isfinite(encoded).all()

    def test_backward_compatibility_static_config(self):
        """Test static feature configuration still works."""
        config = GATModelConfig(node_feature_type="static")

        # Should not have temporal encoder
        assert config.node_feature_type == "static"

        gat = GATPortfolio(
            in_dim=config.hidden_dim,  # Static features are already embedded
            hidden_dim=config.hidden_dim,
            heads=config.num_attention_heads,
            num_layers=config.num_layers,
            use_temporal_encoder=False,
        )

        assert gat.temporal_encoder is None

    def test_multiple_features(self):
        """Test configuration with multiple time-series features."""
        config = GATModelConfig(
            node_feature_type="timeseries",
            timeseries_length=60,
            timeseries_features=["volatility", "returns", "momentum"],
            temporal_encoder_type="conv1d",
        )

        assert len(config.timeseries_features) == 3
        assert "volatility" in config.timeseries_features
        assert "returns" in config.timeseries_features
        assert "momentum" in config.timeseries_features

        # Test that multi-feature data works
        x = torch.randn(10, 60, 3)  # 3 features
        gat = GATPortfolio(
            in_dim=3,  # 3 input features
            hidden_dim=config.hidden_dim,
            heads=config.num_attention_heads,
            num_layers=config.num_layers,
            use_temporal_encoder=config.use_temporal_encoder,
            temporal_encoder_type=config.temporal_encoder_type,
            timeseries_length=config.timeseries_length,
        )

        with torch.no_grad():
            encoded = gat.temporal_encoder(x)

        assert encoded.shape == (10, config.hidden_dim)
        assert torch.isfinite(encoded).all()


class TestGradientFlow:
    """Test gradients flow correctly through time-series GAT."""

    def test_gradients_through_temporal_encoder(self, sample_timeseries_data):
        """Test gradients flow through temporal encoder."""
        config = GATModelConfig(
            node_feature_type="timeseries",
            timeseries_length=60,
            temporal_encoder_type="conv1d",
        )

        gat = GATPortfolio(
            in_dim=1,
            hidden_dim=config.hidden_dim,
            heads=config.num_attention_heads,
            num_layers=config.num_layers,
            use_temporal_encoder=config.use_temporal_encoder,
            temporal_encoder_type=config.temporal_encoder_type,
            timeseries_length=config.timeseries_length,
        )

        x = sample_timeseries_data.clone().requires_grad_(True)
        encoded = gat.temporal_encoder(x)
        loss = encoded.sum()
        loss.backward()

        assert x.grad is not None, "No gradients"
        assert torch.isfinite(x.grad).all(), "Gradients contain NaN/Inf"


class TestEdgeCases:
    """Test edge cases and potential failure modes."""

    def test_single_asset(self):
        """Test with single asset (batch size 1)."""
        config = GATModelConfig(
            node_feature_type="timeseries",
            timeseries_length=60,
            temporal_encoder_type="conv1d",
        )

        gat = GATPortfolio(
            in_dim=1,
            hidden_dim=config.hidden_dim,
            heads=config.num_attention_heads,
            num_layers=config.num_layers,
            use_temporal_encoder=config.use_temporal_encoder,
            temporal_encoder_type=config.temporal_encoder_type,
            timeseries_length=config.timeseries_length,
        )

        x = torch.randn(1, 60, 1)  # Single asset
        with torch.no_grad():
            encoded = gat.temporal_encoder(x)

        assert encoded.shape == (1, config.hidden_dim)
        assert torch.isfinite(encoded).all()

    def test_short_sequence(self):
        """Test with very short sequences."""
        config = GATModelConfig(
            node_feature_type="timeseries",
            timeseries_length=5,  # Very short
            temporal_encoder_type="conv1d",
        )

        gat = GATPortfolio(
            in_dim=1,
            hidden_dim=config.hidden_dim,
            heads=config.num_attention_heads,
            num_layers=config.num_layers,
            use_temporal_encoder=config.use_temporal_encoder,
            temporal_encoder_type=config.temporal_encoder_type,
            timeseries_length=config.timeseries_length,
        )

        x = torch.randn(10, 5, 1)
        with torch.no_grad():
            encoded = gat.temporal_encoder(x)

        assert encoded.shape == (10, config.hidden_dim)
        assert torch.isfinite(encoded).all()

    def test_large_batch(self):
        """Test with large batch size."""
        config = GATModelConfig(
            node_feature_type="timeseries",
            timeseries_length=60,
            temporal_encoder_type="conv1d",
        )

        gat = GATPortfolio(
            in_dim=1,
            hidden_dim=config.hidden_dim,
            heads=config.num_attention_heads,
            num_layers=config.num_layers,
            use_temporal_encoder=config.use_temporal_encoder,
            temporal_encoder_type=config.temporal_encoder_type,
            timeseries_length=config.timeseries_length,
        )

        x = torch.randn(200, 60, 1)  # 200 assets
        with torch.no_grad():
            encoded = gat.temporal_encoder(x)

        assert encoded.shape == (200, config.hidden_dim)
        assert torch.isfinite(encoded).all()
