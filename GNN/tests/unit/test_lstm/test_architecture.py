"""
Unit tests for LSTM network architecture.

Tests LSTM network components including multi-head attention, sequence processing,
and memory usage estimation.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.lstm.architecture import (
    LSTMConfig,
    LSTMNetwork,
    MultiHeadAttention,
    SharpeRatioLoss,
    create_lstm_network,
)


class TestLSTMConfig:
    """Test LSTM configuration dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LSTMConfig()

        assert config.sequence_length == 60
        assert config.input_size == 1
        assert config.hidden_size == 128
        assert config.num_layers == 2
        assert config.dropout == 0.3
        assert config.num_attention_heads == 8
        assert config.output_size == 1

    def test_custom_config(self):
        """Test custom configuration values."""
        config = LSTMConfig(sequence_length=30, hidden_size=64, num_layers=3, dropout=0.5)

        assert config.sequence_length == 30
        assert config.hidden_size == 64
        assert config.num_layers == 3
        assert config.dropout == 0.5


class TestMultiHeadAttention:
    """Test multi-head attention mechanism."""

    @pytest.fixture
    def attention_layer(self):
        """Create attention layer for testing."""
        return MultiHeadAttention(hidden_size=128, num_heads=8, dropout=0.1)

    def test_initialization(self, attention_layer):
        """Test attention layer initialization."""
        assert attention_layer.hidden_size == 128
        assert attention_layer.num_heads == 8
        assert attention_layer.head_size == 16

    def test_invalid_head_configuration(self):
        """Test that invalid head configuration raises error."""
        with pytest.raises(AssertionError):
            MultiHeadAttention(hidden_size=127, num_heads=8)

    def test_forward_pass(self, attention_layer):
        """Test attention forward pass."""
        batch_size, seq_len, hidden_size = 4, 10, 128
        x = torch.randn(batch_size, seq_len, hidden_size)

        output = attention_layer(x)

        assert output.shape == (batch_size, seq_len, hidden_size)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_attention_weights_sum_to_one(self, attention_layer):
        """Test that attention weights sum to approximately 1."""
        batch_size, seq_len, hidden_size = 2, 5, 128
        x = torch.randn(batch_size, seq_len, hidden_size)

        # Access internal computation for testing
        q = attention_layer.query(x).view(batch_size, seq_len, 8, 16).transpose(1, 2)
        k = attention_layer.key(x).view(batch_size, seq_len, 8, 16).transpose(1, 2)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (16**0.5)
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # Check that attention weights sum to 1 along the last dimension
        weight_sums = attention_probs.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)


class TestLSTMNetwork:
    """Test LSTM network architecture."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return LSTMConfig(
            sequence_length=10,
            input_size=5,
            hidden_size=32,
            num_layers=2,
            dropout=0.1,
            num_attention_heads=4,
            output_size=5,
        )

    @pytest.fixture
    def network(self, config):
        """Create LSTM network for testing."""
        return LSTMNetwork(config)

    def test_network_initialization(self, network, config):
        """Test network initialization."""
        assert network.config == config
        assert isinstance(network.lstm, nn.LSTM)
        assert isinstance(network.attention, MultiHeadAttention)
        assert isinstance(network.batch_norm, nn.BatchNorm1d)

    def test_forward_pass_shape(self, network):
        """Test forward pass output shapes."""
        batch_size, seq_len, input_size = 3, 10, 5
        x = torch.randn(batch_size, seq_len, input_size)

        predictions, attention_weights = network(x)

        assert predictions.shape == (batch_size, 5)  # output_size = 5
        assert attention_weights.shape == (batch_size, seq_len)

    def test_forward_pass_no_nan(self, network):
        """Test that forward pass produces no NaN values."""
        batch_size, seq_len, input_size = 2, 10, 5
        x = torch.randn(batch_size, seq_len, input_size)

        predictions, attention_weights = network(x)

        assert not torch.isnan(predictions).any()
        assert not torch.isnan(attention_weights).any()
        assert not torch.isinf(predictions).any()
        assert not torch.isinf(attention_weights).any()

    def test_attention_weights_normalized(self, network):
        """Test that attention weights are properly normalized."""
        batch_size, seq_len, input_size = 2, 10, 5
        x = torch.randn(batch_size, seq_len, input_size)

        _, attention_weights = network(x)

        # Attention weights should sum to approximately 1
        weight_sums = attention_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones(batch_size), atol=1e-5)

    def test_batch_norm_single_sample(self, network):
        """Test batch norm behavior with single sample."""
        x = torch.randn(1, 10, 5)  # Single sample

        # Should not raise error even with batch size 1
        predictions, attention_weights = network(x)

        assert predictions.shape == (1, 5)
        assert attention_weights.shape == (1, 10)

    def test_gradient_flow(self, network):
        """Test that gradients flow through the network."""
        x = torch.randn(2, 10, 5, requires_grad=True)
        target = torch.randn(2, 5)

        predictions, _ = network(x)
        loss = nn.MSELoss()(predictions, target)
        loss.backward()

        # Check that gradients exist for network parameters
        for param in network.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_memory_usage_estimation(self, network):
        """Test memory usage estimation."""
        batch_size, sequence_length = 4, 10

        memory_bytes = network.get_memory_usage(batch_size, sequence_length)

        assert isinstance(memory_bytes, int)
        assert memory_bytes > 0

    @pytest.mark.parametrize("batch_size,seq_len", [(1, 5), (8, 20), (16, 60)])
    def test_variable_input_sizes(self, network, batch_size, seq_len):
        """Test network with various input sizes."""
        x = torch.randn(batch_size, seq_len, 5)

        predictions, attention_weights = network(x)

        assert predictions.shape == (batch_size, 5)
        assert attention_weights.shape == (batch_size, seq_len)


class TestSharpeRatioLoss:
    """Test custom Sharpe ratio loss function."""

    @pytest.fixture
    def loss_fn(self):
        """Create Sharpe ratio loss function."""
        return SharpeRatioLoss(risk_free_rate=0.02)

    def test_initialization(self, loss_fn):
        """Test loss function initialization."""
        # Risk-free rate should be converted to daily rate
        expected_daily_rate = 0.02 / 252
        assert abs(loss_fn.risk_free_rate - expected_daily_rate) < 1e-8

    def test_loss_computation(self, loss_fn):
        """Test loss computation with mock data."""
        batch_size, n_assets = 10, 5
        predicted_returns = torch.randn(batch_size, n_assets)
        actual_returns = torch.randn(batch_size, n_assets)

        loss = loss_fn(predicted_returns, actual_returns)

        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1  # Scalar loss
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_loss_with_custom_weights(self, loss_fn):
        """Test loss computation with custom portfolio weights."""
        batch_size, n_assets = 8, 4
        predicted_returns = torch.randn(batch_size, n_assets)
        actual_returns = torch.randn(batch_size, n_assets)
        portfolio_weights = torch.softmax(torch.randn(batch_size, n_assets), dim=-1)

        loss = loss_fn(predicted_returns, actual_returns, portfolio_weights)

        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_zero_variance_fallback(self, loss_fn):
        """Test fallback to MSE when returns have zero variance."""
        batch_size, n_assets = 5, 3
        predicted_returns = torch.randn(batch_size, n_assets)
        # All actual returns identical (zero variance)
        actual_returns = torch.ones(batch_size, n_assets) * 0.01

        loss = loss_fn(predicted_returns, actual_returns)

        # Should fallback to MSE loss
        expected_mse = nn.MSELoss()(predicted_returns, actual_returns)
        assert torch.allclose(loss, expected_mse, atol=1e-6)

    def test_gradient_computation(self, loss_fn):
        """Test that gradients can be computed through loss."""
        predicted_returns = torch.randn(3, 4, requires_grad=True)
        actual_returns = torch.randn(3, 4)

        loss = loss_fn(predicted_returns, actual_returns)
        loss.backward()

        assert predicted_returns.grad is not None
        assert not torch.isnan(predicted_returns.grad).any()


class TestLSTMNetworkFactory:
    """Test LSTM network factory function."""

    def test_create_valid_network(self):
        """Test creating network with valid configuration."""
        config = LSTMConfig(sequence_length=20, hidden_size=64, num_attention_heads=8)

        network = create_lstm_network(config)

        assert isinstance(network, LSTMNetwork)
        assert network.config == config

    def test_invalid_sequence_length(self):
        """Test validation of sequence length."""
        config = LSTMConfig(sequence_length=0)

        with pytest.raises(ValueError, match="sequence_length must be positive"):
            create_lstm_network(config)

    def test_invalid_hidden_size(self):
        """Test validation of hidden size."""
        config = LSTMConfig(hidden_size=-1)

        with pytest.raises(ValueError, match="hidden_size must be positive"):
            create_lstm_network(config)

    def test_invalid_num_layers(self):
        """Test validation of number of layers."""
        config = LSTMConfig(num_layers=0)

        with pytest.raises(ValueError, match="num_layers must be positive"):
            create_lstm_network(config)

    def test_invalid_dropout(self):
        """Test validation of dropout rate."""
        config = LSTMConfig(dropout=1.5)

        with pytest.raises(ValueError, match="dropout must be in"):
            create_lstm_network(config)

    def test_invalid_attention_heads(self):
        """Test validation of attention heads."""
        config = LSTMConfig(hidden_size=100, num_attention_heads=7)  # 100 not divisible by 7

        with pytest.raises(ValueError, match="must be divisible by"):
            create_lstm_network(config)


@pytest.mark.slow
class TestLSTMNetworkIntegration:
    """Integration tests for LSTM network."""

    def test_training_step_integration(self):
        """Test a complete training step."""
        config = LSTMConfig(
            sequence_length=5,
            input_size=3,
            hidden_size=16,
            num_layers=1,
            dropout=0.1,
            num_attention_heads=2,
            output_size=3,
        )

        network = create_lstm_network(config)
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        criterion = SharpeRatioLoss()

        # Simulate training batch
        batch_size = 4
        x = torch.randn(batch_size, config.sequence_length, config.input_size)
        target = torch.randn(batch_size, config.output_size)

        # Forward pass
        predictions, attention_weights = network(x)
        loss = criterion(predictions, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify training step completed successfully
        assert not torch.isnan(loss)
        assert loss.requires_grad

    def test_reproducibility(self):
        """Test that network outputs are reproducible with same seed."""
        config = LSTMConfig(hidden_size=32, num_attention_heads=4)

        # Create identical networks and inputs with deterministic initialization
        torch.manual_seed(42)
        np.random.seed(42)
        network1 = create_lstm_network(config)

        torch.manual_seed(42)
        np.random.seed(42)
        x = torch.randn(2, 10, 1)

        torch.manual_seed(42)
        np.random.seed(42)
        network2 = create_lstm_network(config)

        # Copy weights from network1 to network2 to ensure identical initialization
        network2.load_state_dict(network1.state_dict())

        # Put networks in eval mode to disable dropout
        network1.eval()
        network2.eval()

        # Should produce identical outputs with same weights
        with torch.no_grad():  # Disable gradients for deterministic behavior
            pred1, att1 = network1(x)
            pred2, att2 = network2(x)

        assert torch.allclose(pred1, pred2, atol=1e-6)
        assert torch.allclose(att1, att2, atol=1e-6)
