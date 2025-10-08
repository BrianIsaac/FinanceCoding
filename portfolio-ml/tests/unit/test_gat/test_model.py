"""
Unit tests for GAT portfolio model implementation.

Tests the core GAT model architecture, portfolio optimization components,
and integration with the PortfolioModel interface.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.base.portfolio_model import PortfolioConstraints
from src.models.gat.gat_model import GATPortfolio, HeadCfg
from src.models.gat.model import GATModelConfig, GATPortfolioModel, SharpeRatioLoss


class TestGATModelConfig:
    """Test GAT model configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GATModelConfig()

        assert config.input_features == 10
        assert config.hidden_dim == 64
        assert config.num_layers == 3
        assert config.num_attention_heads == 8
        assert config.dropout == 0.3
        assert config.learning_rate == 0.001
        assert config.use_mixed_precision is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = GATModelConfig(
            hidden_dim=128, num_layers=4, learning_rate=0.0005, use_mixed_precision=False
        )

        assert config.hidden_dim == 128
        assert config.num_layers == 4
        assert config.learning_rate == 0.0005
        assert config.use_mixed_precision is False


class TestSharpeRatioLoss:
    """Test Sharpe ratio loss function."""

    def test_sharpe_ratio_calculation(self):
        """Test basic Sharpe ratio calculation."""
        loss_fn = SharpeRatioLoss(risk_free_rate=0.0, constraint_penalty=0.0)

        # Create sample data
        weights = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float32)  # [1, 3]
        returns = torch.tensor([[0.1, 0.05, -0.02]], dtype=torch.float32)  # [1, 3]

        loss = loss_fn(weights, returns)

        # Should be negative Sharpe ratio (loss for maximization)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar

    def test_constraint_penalties(self):
        """Test constraint penalty calculations."""
        loss_fn = SharpeRatioLoss(constraint_penalty=1.0)

        # Weights that don't sum to 1 and include negative values
        weights = torch.tensor([[0.6, 0.3, -0.1]], dtype=torch.float32)
        returns = torch.tensor([[0.1, 0.05, 0.02]], dtype=torch.float32)

        loss = loss_fn(weights, returns)

        # Should include penalty for constraint violations
        assert loss.item() > 0  # Penalties should make loss positive

    def test_time_series_returns(self):
        """Test with time series returns data."""
        loss_fn = SharpeRatioLoss(lookback_window=5)

        # Time series format: [batch_size, time_steps, n_assets]
        weights = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float32)  # [1, 3]
        returns = torch.tensor(
            [
                [
                    [0.1, 0.05, 0.02],
                    [0.02, 0.03, 0.01],
                    [-0.01, 0.02, 0.03],
                    [0.03, -0.01, 0.02],
                    [0.01, 0.04, -0.01],
                ]
            ],
            dtype=torch.float32,
        )  # [1, 5, 3]

        loss = loss_fn(weights, returns)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0


class TestGATPortfolioModel:
    """Test GAT portfolio model implementation."""

    @pytest.fixture
    def constraints(self):
        """Create test portfolio constraints."""
        return PortfolioConstraints(
            long_only=True,
            max_position_weight=0.5,  # 50% max per position allows feasible portfolios
            transaction_cost_bps=10.0,
        )

    @pytest.fixture
    def config(self):
        """Create test model configuration."""
        return GATModelConfig(input_features=5, hidden_dim=32, num_layers=2, max_epochs=10)

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data."""
        dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")
        tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "META"]

        # Generate synthetic returns
        np.random.seed(42)
        returns_data = np.random.normal(0.001, 0.02, size=(len(dates), len(tickers)))

        return pd.DataFrame(returns_data, index=dates, columns=tickers)

    def test_model_initialization(self, constraints, config):
        """Test model initialization."""
        model = GATPortfolioModel(constraints, config)

        assert model.constraints == constraints
        assert model.config == config
        assert model.device.type in ["cpu", "cuda"]
        assert model.is_fitted is False

    def test_get_model_info(self, constraints, config):
        """Test model info retrieval."""
        model = GATPortfolioModel(constraints, config)
        info = model.get_model_info()

        assert info["model_type"] == "GAT"
        assert "architecture" in info
        assert "constraints" in info
        assert info["is_fitted"] is False

    def test_predict_weights_unfitted(self, constraints, config):
        """Test that predict_weights raises error when model is not fitted."""
        model = GATPortfolioModel(constraints, config)
        universe = ["AAPL", "MSFT", "GOOGL"]
        date = pd.Timestamp("2022-01-01")

        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict_weights(date, universe)

    def test_feature_preparation(self, constraints, config, sample_returns):
        """Test feature preparation from returns data."""
        model = GATPortfolioModel(constraints, config)
        universe = ["AAPL", "MSFT", "GOOGL"]

        features = model._prepare_features(sample_returns, universe)

        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(universe)
        assert features.shape[1] == 10  # Expected number of features
        assert not np.isnan(features).any()

    def test_model_building(self, constraints, config):
        """Test GAT model building."""
        model = GATPortfolioModel(constraints, config)

        gat_model = model._build_model(input_dim=10)

        assert isinstance(gat_model, GATPortfolio)
        assert gat_model.training is True  # Should be in training mode initially

    def test_weight_validation(self, constraints, config):
        """Test portfolio weight validation."""
        model = GATPortfolioModel(constraints, config)
        universe = ["AAPL", "MSFT", "GOOGL"]

        # Test with invalid weights (negative, not summing to 1)
        invalid_weights = pd.Series([-0.1, 0.6, 0.3], index=universe)
        validated = model.validate_weights(invalid_weights)

        # Should be long-only and sum to 1
        assert all(validated >= 0)
        assert abs(validated.sum() - 1.0) < 1e-6

    @patch("src.models.gat.model.build_period_graph")
    def test_fit_insufficient_data(self, mock_build_graph, constraints, config, sample_returns):
        """Test fit method with insufficient data."""
        model = GATPortfolioModel(constraints, config)
        universe = ["AAPL", "MSFT"]

        # Use very short period
        start_date = pd.Timestamp("2022-12-01")
        end_date = pd.Timestamp("2022-12-31")

        with pytest.raises(ValueError, match="Insufficient training data"):
            model.fit(sample_returns, universe, (start_date, end_date))


class TestGATArchitecture:
    """Test core GAT architecture components."""

    def test_gat_portfolio_forward_pass(self):
        """Test GAT model forward pass."""
        model = GATPortfolio(in_dim=10, hidden_dim=32, num_layers=2, heads=4, head="direct")

        # Create sample input
        n_assets = 5
        x = torch.randn(n_assets, 10)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        mask_valid = torch.ones(n_assets, dtype=torch.bool)
        edge_attr = torch.randn(5, 3)

        # Forward pass
        weights, memory, reg_loss = model(x, edge_index, mask_valid, edge_attr)

        # Check outputs
        assert weights.shape == (n_assets,)
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-4)
        assert all(weights >= 0)  # Long-only constraint

    def test_gat_portfolio_markowitz_head(self):
        """Test GAT model with Markowitz head."""
        model = GATPortfolio(in_dim=10, hidden_dim=32, num_layers=2, heads=4, head="markowitz")

        # Create sample input
        n_assets = 5
        x = torch.randn(n_assets, 10)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        mask_valid = torch.ones(n_assets, dtype=torch.bool)

        # Forward pass
        mu_hat, memory, reg_loss = model(x, edge_index, mask_valid)

        # Check outputs
        assert mu_hat.shape == (n_assets,)
        assert isinstance(mu_hat, torch.Tensor)

    def test_memory_mechanism(self):
        """Test temporal memory mechanism."""
        model = GATPortfolio(in_dim=10, hidden_dim=32, num_layers=2, mem_hidden=16)

        n_assets = 5
        x = torch.randn(n_assets, 10)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        mask_valid = torch.ones(n_assets, dtype=torch.bool)
        prev_mem = torch.randn(n_assets, 16)

        # Forward pass with memory
        weights, new_memory, reg_loss = model(x, edge_index, mask_valid, prev_mem=prev_mem)

        assert new_memory.shape == (n_assets, 16)
        assert not torch.equal(prev_mem, new_memory)  # Memory should be updated


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""

    def test_empty_universe_handling(self):
        """Test handling of empty universe."""
        constraints = PortfolioConstraints()
        config = GATModelConfig()
        model = GATPortfolioModel(constraints, config)

        # Empty universe should raise appropriate error
        with pytest.raises((ValueError, IndexError)):
            model.predict_weights(pd.Timestamp("2022-01-01"), [])

    def test_single_asset_universe(self):
        """Test handling of single-asset universe."""
        constraints = PortfolioConstraints(max_position_weight=1.0)  # Allow 100% in single asset
        config = GATModelConfig()
        model = GATPortfolioModel(constraints, config)

        # Single asset should get 100% weight
        universe = ["AAPL"]
        raw_weights = pd.Series([0.8], index=universe)  # Less than 1
        validated = model.validate_weights(raw_weights)

        assert validated.sum() == pytest.approx(1.0)
        assert validated["AAPL"] == pytest.approx(1.0)

    def test_gpu_memory_constraints(self):
        """Test GPU memory constraint handling."""
        config = GATModelConfig(max_vram_gb=1.0)  # Very low limit
        constraints = PortfolioConstraints()

        model = GATPortfolioModel(constraints, config)

        # Should initialize without error even with low memory limit
        assert model.config.max_vram_gb == 1.0

    def test_different_activation_functions(self):
        """Test different activation functions for portfolio weights."""
        for activation in ["softmax", "sparsemax"]:
            config = GATModelConfig()
            config.head_config = HeadCfg(mode="direct", activation=activation)

            model = GATPortfolio(in_dim=10, hidden_dim=32, head="direct", activation=activation)

            n_assets = 5
            x = torch.randn(n_assets, 10)
            edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
            mask_valid = torch.ones(n_assets, dtype=torch.bool)

            weights, _, _ = model(x, edge_index, mask_valid)

            # Weights should be valid regardless of activation
            assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-4)
            assert all(weights >= 0)


if __name__ == "__main__":
    pytest.main([__file__])
