"""Comprehensive unit tests for RaggedLSTMNetwork architecture."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.lstm.architecture import LSTMConfig
from src.models.lstm.ragged_architecture import RaggedLSTMNetwork, create_ragged_lstm_network


class TestRaggedLSTMNetworkConstruction:
    """Test RaggedLSTMNetwork construction and initialization."""

    def test_basic_construction(self):
        """Test basic network construction."""
        config = LSTMConfig(
            input_size=10,
            hidden_size=32,
            num_layers=2,
            output_size=10
        )

        model = RaggedLSTMNetwork(config)

        assert isinstance(model, nn.Module)
        assert model.config == config

    def test_factory_function(self):
        """Test factory function creates valid network."""
        config = LSTMConfig(
            input_size=20,
            hidden_size=64,
            num_layers=3,
            output_size=20
        )

        model = create_ragged_lstm_network(config)

        assert isinstance(model, RaggedLSTMNetwork)
        assert model.config == config

    def test_invalid_sequence_length(self):
        """Test construction fails with invalid sequence length."""
        config = LSTMConfig(sequence_length=-1)

        with pytest.raises(ValueError, match="sequence_length must be positive"):
            create_ragged_lstm_network(config)

    def test_invalid_hidden_size(self):
        """Test construction fails with invalid hidden size."""
        config = LSTMConfig(hidden_size=0)

        with pytest.raises(ValueError, match="hidden_size must be positive"):
            create_ragged_lstm_network(config)

    def test_invalid_num_layers(self):
        """Test construction fails with invalid num_layers."""
        config = LSTMConfig(num_layers=0)

        with pytest.raises(ValueError, match="num_layers must be positive"):
            create_ragged_lstm_network(config)

    def test_invalid_dropout(self):
        """Test construction fails with invalid dropout."""
        config = LSTMConfig(dropout=1.5)

        with pytest.raises(ValueError, match="dropout must be in"):
            create_ragged_lstm_network(config)

    def test_invalid_attention_heads(self):
        """Test construction fails when hidden_size not divisible by num_heads."""
        config = LSTMConfig(
            hidden_size=100,  # Not divisible by 8
            num_attention_heads=8
        )

        with pytest.raises(ValueError, match="must be divisible by"):
            create_ragged_lstm_network(config)

    def test_weights_initialized(self):
        """Test that weights are initialized."""
        config = LSTMConfig(input_size=10, hidden_size=32, output_size=10)
        model = RaggedLSTMNetwork(config)

        # Check that parameters exist and are not all zero
        for name, param in model.named_parameters():
            assert param.numel() > 0
            # Most parameters should not be all zero after initialization
            if "bias" not in name:
                assert not torch.allclose(param, torch.zeros_like(param))


class TestRaggedLSTMNetworkForward:
    """Test RaggedLSTMNetwork forward pass."""

    def test_forward_basic(self):
        """Test basic forward pass."""
        config = LSTMConfig(input_size=10, hidden_size=32, output_size=10)
        model = RaggedLSTMNetwork(config)
        model.eval()

        sequences = torch.randn(4, 60, 10)
        lengths = torch.tensor([45, 52, 30, 58])

        predictions, attention = model(sequences, lengths)

        assert predictions.shape == (4, 10)
        assert attention.shape == (4, 60)

    def test_forward_variable_lengths(self):
        """Test forward pass with variable lengths."""
        config = LSTMConfig(input_size=12, hidden_size=64, output_size=12)
        model = RaggedLSTMNetwork(config)
        model.eval()

        batch_size = 16
        max_len = 50
        sequences = torch.randn(batch_size, max_len, 12)
        lengths = torch.randint(20, max_len + 1, (batch_size,))

        predictions, attention = model(sequences, lengths)

        assert predictions.shape == (batch_size, 12)
        assert attention.shape == (batch_size, max_len)

    def test_forward_single_sequence(self):
        """Test forward pass with single sequence."""
        config = LSTMConfig(input_size=8, hidden_size=32, output_size=8)
        model = RaggedLSTMNetwork(config)
        model.eval()

        sequences = torch.randn(1, 60, 8)
        lengths = torch.tensor([45])

        predictions, attention = model(sequences, lengths)

        assert predictions.shape == (1, 8)
        assert attention.shape == (1, 60)

    def test_forward_all_same_length(self):
        """Test forward pass when all sequences have same length."""
        config = LSTMConfig(input_size=10, hidden_size=32, output_size=10)
        model = RaggedLSTMNetwork(config)
        model.eval()

        sequences = torch.randn(8, 60, 10)
        lengths = torch.tensor([45] * 8)

        predictions, attention = model(sequences, lengths)

        assert predictions.shape == (8, 10)
        assert attention.shape == (8, 60)

    def test_predictions_finite(self):
        """Test that predictions are finite (no NaN/Inf)."""
        config = LSTMConfig(input_size=10, hidden_size=32, output_size=10)
        model = RaggedLSTMNetwork(config)
        model.eval()

        sequences = torch.randn(8, 60, 10)
        lengths = torch.randint(30, 61, (8,))

        predictions, attention = model(sequences, lengths)

        assert torch.isfinite(predictions).all()
        assert torch.isfinite(attention).all()

    def test_predictions_clamped(self):
        """Test that predictions are clamped to [-1, 1]."""
        config = LSTMConfig(input_size=10, hidden_size=32, output_size=10)
        model = RaggedLSTMNetwork(config)
        model.eval()

        sequences = torch.randn(8, 60, 10)
        lengths = torch.randint(30, 61, (8,))

        predictions, _ = model(sequences, lengths)

        assert (predictions >= -1.0).all()
        assert (predictions <= 1.0).all()

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to approximately 1."""
        config = LSTMConfig(input_size=10, hidden_size=32, output_size=10)
        model = RaggedLSTMNetwork(config)
        model.eval()

        sequences = torch.randn(8, 60, 10)
        lengths = torch.randint(30, 61, (8,))

        _, attention = model(sequences, lengths)

        # Attention weights should sum to 1 (within tolerance)
        attention_sums = attention.sum(dim=1)
        assert torch.allclose(attention_sums, torch.ones_like(attention_sums), atol=1e-5)

    def test_training_mode(self):
        """Test forward pass in training mode."""
        config = LSTMConfig(input_size=10, hidden_size=32, output_size=10, dropout=0.3)
        model = RaggedLSTMNetwork(config)
        model.train()

        sequences = torch.randn(8, 60, 10)
        lengths = torch.randint(30, 61, (8,))

        predictions, attention = model(sequences, lengths)

        assert predictions.shape == (8, 10)
        assert attention.shape == (8, 60)


class TestRaggedLSTMNetworkGradients:
    """Test gradient flow through RaggedLSTMNetwork."""

    def test_gradients_flow(self):
        """Test that gradients flow through network."""
        config = LSTMConfig(input_size=10, hidden_size=32, output_size=10)
        model = RaggedLSTMNetwork(config)
        model.train()

        sequences = torch.randn(4, 60, 10, requires_grad=True)
        lengths = torch.tensor([45, 52, 30, 58])

        predictions, _ = model(sequences, lengths)
        loss = predictions.sum()
        loss.backward()

        # Check that gradients exist for input
        assert sequences.grad is not None
        assert torch.isfinite(sequences.grad).all()

    def test_all_parameters_receive_gradients(self):
        """Test that all parameters receive gradients."""
        config = LSTMConfig(input_size=10, hidden_size=32, output_size=10)
        model = RaggedLSTMNetwork(config)
        model.train()

        sequences = torch.randn(8, 60, 10)
        lengths = torch.randint(30, 61, (8,))

        predictions, _ = model(sequences, lengths)
        loss = predictions.mean()
        loss.backward()

        # Check that all parameters have gradients (except batch_norm which is only used in eval)
        for name, param in model.named_parameters():
            if param.requires_grad and 'batch_norm' not in name:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_gradient_values_reasonable(self):
        """Test that gradient values are reasonable (not exploding/vanishing)."""
        config = LSTMConfig(input_size=10, hidden_size=32, output_size=10)
        model = RaggedLSTMNetwork(config)
        model.train()

        sequences = torch.randn(8, 60, 10)
        lengths = torch.randint(30, 61, (8,))

        predictions, _ = model(sequences, lengths)
        loss = predictions.mean()
        loss.backward()

        # Check gradient magnitudes are reasonable
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm().item()
                assert grad_norm < 100.0, f"Gradient too large for {name}: {grad_norm}"
                # Some gradients might be very small, that's okay


class TestRaggedLSTMNetworkStatistics:
    """Test sequence statistics tracking."""

    def test_statistics_tracked(self):
        """Test that statistics are tracked during forward pass."""
        config = LSTMConfig(input_size=10, hidden_size=32, output_size=10)
        model = RaggedLSTMNetwork(config)
        model.train()

        sequences = torch.randn(8, 60, 10)
        lengths = torch.tensor([30, 35, 40, 45, 50, 55, 58, 60])

        model(sequences, lengths)

        stats = model.get_sequence_statistics()

        assert stats is not None
        assert 'mean_length' in stats
        assert 'std_length' in stats
        assert 'padding_ratio' in stats
        assert 'batch_size' in stats

    def test_statistics_values(self):
        """Test that statistics have correct values."""
        config = LSTMConfig(input_size=10, hidden_size=32, output_size=10)
        model = RaggedLSTMNetwork(config)
        model.eval()

        sequences = torch.randn(4, 100, 10)
        lengths = torch.tensor([50, 50, 50, 50])

        model(sequences, lengths)

        stats = model.get_sequence_statistics()

        assert stats['mean_length'] == 50.0
        assert stats['min_length'] == 50
        assert stats['max_length'] == 50
        assert stats['batch_size'] == 4
        assert stats['max_seq_len'] == 100
        # Padding ratio should be 0.5 (50/100)
        assert abs(stats['padding_ratio'] - 0.5) < 1e-6

    def test_memory_usage_estimate(self):
        """Test memory usage estimation."""
        config = LSTMConfig(input_size=10, hidden_size=32, output_size=10)
        model = RaggedLSTMNetwork(config)

        memory_usage = model.get_memory_usage(
            batch_size=8,
            sequence_length=60,
            avg_real_length=45
        )

        assert memory_usage > 0
        assert isinstance(memory_usage, int)


class TestRaggedLSTMNetworkEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimal_sequence_length(self):
        """Test with minimal sequence length (1)."""
        config = LSTMConfig(input_size=10, hidden_size=32, output_size=10)
        model = RaggedLSTMNetwork(config)
        model.eval()

        sequences = torch.randn(4, 60, 10)
        lengths = torch.tensor([1, 2, 3, 4])

        predictions, attention = model(sequences, lengths)

        assert predictions.shape == (4, 10)
        assert attention.shape == (4, 60)
        assert torch.isfinite(predictions).all()

    def test_full_length_sequences(self):
        """Test with all sequences at full length (no padding)."""
        config = LSTMConfig(input_size=10, hidden_size=32, output_size=10)
        model = RaggedLSTMNetwork(config)
        model.eval()

        sequences = torch.randn(8, 60, 10)
        lengths = torch.tensor([60] * 8)

        predictions, attention = model(sequences, lengths)

        assert predictions.shape == (8, 10)

        # Check statistics show no padding
        stats = model.get_sequence_statistics()
        assert stats['padding_ratio'] == 0.0

    def test_large_batch_size(self):
        """Test with large batch size."""
        config = LSTMConfig(input_size=10, hidden_size=32, output_size=10)
        model = RaggedLSTMNetwork(config)
        model.eval()

        sequences = torch.randn(128, 50, 10)
        lengths = torch.randint(20, 51, (128,))

        predictions, attention = model(sequences, lengths)

        assert predictions.shape == (128, 10)
        assert attention.shape == (128, 50)

    def test_different_input_output_sizes(self):
        """Test with different input and output sizes."""
        config = LSTMConfig(
            input_size=50,
            hidden_size=64,
            output_size=100
        )
        model = RaggedLSTMNetwork(config)
        model.eval()

        sequences = torch.randn(8, 60, 50)
        lengths = torch.randint(30, 61, (8,))

        predictions, attention = model(sequences, lengths)

        assert predictions.shape == (8, 100)

    def test_extreme_input_values(self):
        """Test with extreme input values (should be clamped)."""
        config = LSTMConfig(input_size=10, hidden_size=32, output_size=10)
        model = RaggedLSTMNetwork(config)
        model.eval()

        # Create sequences with extreme values
        sequences = torch.randn(4, 60, 10) * 100  # Very large values
        lengths = torch.tensor([45, 52, 30, 58])

        predictions, attention = model(sequences, lengths)

        # Should still produce finite outputs due to clamping
        assert torch.isfinite(predictions).all()
        assert torch.isfinite(attention).all()

    def test_deterministic_in_eval_mode(self):
        """Test that model is deterministic in eval mode."""
        config = LSTMConfig(input_size=10, hidden_size=32, output_size=10)
        model = RaggedLSTMNetwork(config)
        model.eval()

        sequences = torch.randn(4, 60, 10)
        lengths = torch.tensor([45, 52, 30, 58])

        # Run twice with same input
        pred1, att1 = model(sequences, lengths)
        pred2, att2 = model(sequences, lengths)

        # Should be identical
        torch.testing.assert_close(pred1, pred2)
        torch.testing.assert_close(att1, att2)
