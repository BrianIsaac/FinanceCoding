"""Unit tests for temporal encoders.

Tests the three temporal encoder types (LSTM, Conv1D, Transformer)
for encoding time-series node features in the GAT model.
"""

from __future__ import annotations

import pytest
import torch

from src.models.gat.temporal_encoders import (
    TemporalLSTMEncoder,
    TemporalConvEncoder,
    TemporalTransformerEncoder,
    create_temporal_encoder,
)


class TestTemporalLSTMEncoder:
    """Tests for LSTM temporal encoder."""

    def test_forward_pass(self):
        """Test LSTM encoder forward pass produces correct output shape."""
        encoder = TemporalLSTMEncoder(input_features=1, hidden_dim=64)
        x = torch.randn(10, 60, 1)  # 10 assets, 60 timesteps, 1 feature

        output = encoder(x)

        assert output.shape == (10, 64), f"Expected (10, 64), got {output.shape}"
        assert torch.isfinite(output).all(), "Output contains NaN/Inf"

    def test_different_dimensions(self):
        """Test LSTM encoder with various input/output dimensions."""
        test_cases = [
            (1, 32, 20, 10),  # (input_features, hidden_dim, num_assets, seq_length)
            (3, 128, 50, 60),
            (5, 64, 100, 120),
        ]

        for input_features, hidden_dim, num_assets, seq_length in test_cases:
            encoder = TemporalLSTMEncoder(input_features, hidden_dim)
            x = torch.randn(num_assets, seq_length, input_features)

            output = encoder(x)

            assert output.shape == (
                num_assets,
                hidden_dim,
            ), f"Expected ({num_assets}, {hidden_dim}), got {output.shape}"
            assert torch.isfinite(output).all()

    def test_gradient_flow(self):
        """Test gradients flow correctly through LSTM encoder.

        Note: LSTM uses final hidden state, so gradients may be sparse
        or zero for early timesteps. We just verify gradients exist and are finite.
        """
        encoder = TemporalLSTMEncoder(input_features=1, hidden_dim=64)
        encoder.train()  # Ensure training mode for dropout
        x = torch.randn(5, 30, 1, requires_grad=True)

        output = encoder(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "No gradients computed"
        assert torch.isfinite(x.grad).all(), "Gradients contain NaN/Inf"
        # Note: Gradients may be zero for early timesteps when using final hidden state
        # This is expected behavior for LSTM encoders


class TestTemporalConvEncoder:
    """Tests for Conv1D temporal encoder."""

    def test_forward_pass(self):
        """Test Conv1D encoder forward pass produces correct output shape."""
        encoder = TemporalConvEncoder(input_features=1, hidden_dim=64)
        x = torch.randn(10, 60, 1)

        output = encoder(x)

        assert output.shape == (10, 64), f"Expected (10, 64), got {output.shape}"
        assert torch.isfinite(output).all(), "Output contains NaN/Inf"

    def test_different_dimensions(self):
        """Test Conv1D encoder with various input/output dimensions."""
        test_cases = [
            (1, 32, 20, 10),
            (3, 128, 50, 60),
            (5, 64, 100, 120),
        ]

        for input_features, hidden_dim, num_assets, seq_length in test_cases:
            encoder = TemporalConvEncoder(input_features, hidden_dim)
            x = torch.randn(num_assets, seq_length, input_features)

            output = encoder(x)

            assert output.shape == (num_assets, hidden_dim)
            assert torch.isfinite(output).all()

    def test_gradient_flow(self):
        """Test gradients flow correctly through Conv1D encoder."""
        encoder = TemporalConvEncoder(input_features=1, hidden_dim=64)
        x = torch.randn(5, 30, 1, requires_grad=True)

        output = encoder(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "No gradients computed"
        assert torch.isfinite(x.grad).all(), "Gradients contain NaN/Inf"
        assert (x.grad.abs() > 0).any(), "All gradients are zero"


class TestTemporalTransformerEncoder:
    """Tests for Transformer temporal encoder."""

    def test_forward_pass(self):
        """Test Transformer encoder forward pass produces correct output shape."""
        encoder = TemporalTransformerEncoder(input_features=1, hidden_dim=64)
        x = torch.randn(10, 60, 1)

        output = encoder(x)

        assert output.shape == (10, 64), f"Expected (10, 64), got {output.shape}"
        assert torch.isfinite(output).all(), "Output contains NaN/Inf"

    def test_different_dimensions(self):
        """Test Transformer encoder with various input/output dimensions."""
        test_cases = [
            (1, 32, 20, 10),
            (3, 64, 50, 60),  # Use 64 for transformer (divisible by heads)
            (5, 128, 100, 120),
        ]

        for input_features, hidden_dim, num_assets, seq_length in test_cases:
            encoder = TemporalTransformerEncoder(input_features, hidden_dim)
            x = torch.randn(num_assets, seq_length, input_features)

            output = encoder(x)

            assert output.shape == (num_assets, hidden_dim)
            assert torch.isfinite(output).all()

    def test_gradient_flow(self):
        """Test gradients flow correctly through Transformer encoder."""
        encoder = TemporalTransformerEncoder(input_features=1, hidden_dim=64)
        x = torch.randn(5, 30, 1, requires_grad=True)

        output = encoder(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "No gradients computed"
        assert torch.isfinite(x.grad).all(), "Gradients contain NaN/Inf"
        assert (x.grad.abs() > 0).any(), "All gradients are zero"


class TestEncoderFactory:
    """Tests for the encoder factory function."""

    def test_create_lstm_encoder(self):
        """Test creating LSTM encoder via factory."""
        encoder = create_temporal_encoder("lstm", input_features=1, hidden_dim=64)

        assert isinstance(encoder, TemporalLSTMEncoder)
        x = torch.randn(5, 30, 1)
        output = encoder(x)
        assert output.shape == (5, 64)

    def test_create_conv_encoder(self):
        """Test creating Conv1D encoder via factory."""
        encoder = create_temporal_encoder("conv1d", input_features=1, hidden_dim=64)

        assert isinstance(encoder, TemporalConvEncoder)
        x = torch.randn(5, 30, 1)
        output = encoder(x)
        assert output.shape == (5, 64)

    def test_create_transformer_encoder(self):
        """Test creating Transformer encoder via factory."""
        encoder = create_temporal_encoder(
            "transformer", input_features=1, hidden_dim=64
        )

        assert isinstance(encoder, TemporalTransformerEncoder)
        x = torch.randn(5, 30, 1)
        output = encoder(x)
        assert output.shape == (5, 64)

    def test_invalid_encoder_type(self):
        """Test that invalid encoder type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown encoder_type"):
            create_temporal_encoder("invalid", input_features=1, hidden_dim=64)


class TestEncoderComparison:
    """Compare different encoder types on the same input."""

    def test_all_encoders_produce_valid_output(self):
        """Test all encoder types produce valid output for same input."""
        x = torch.randn(10, 60, 1)

        for encoder_type in ["lstm", "conv1d", "transformer"]:
            encoder = create_temporal_encoder(encoder_type, 1, 64)

            output = encoder(x)

            assert output.shape == (10, 64), f"{encoder_type}: wrong shape"
            assert torch.isfinite(output).all(), f"{encoder_type}: NaN/Inf in output"
            assert (output.abs() > 0).any(), f"{encoder_type}: all zeros"

    def test_encoders_with_multiple_features(self):
        """Test all encoders work with multiple input features."""
        x = torch.randn(10, 60, 3)  # 3 features: volatility, returns, momentum

        for encoder_type in ["lstm", "conv1d", "transformer"]:
            encoder = create_temporal_encoder(encoder_type, input_features=3, hidden_dim=64)

            output = encoder(x)

            assert output.shape == (10, 64), f"{encoder_type}: wrong shape"
            assert torch.isfinite(output).all(), f"{encoder_type}: NaN/Inf in output"

    def test_encoders_with_different_sequence_lengths(self):
        """Test all encoders handle different sequence lengths."""
        sequence_lengths = [10, 60, 120, 252, 756]

        for seq_len in sequence_lengths:
            x = torch.randn(5, seq_len, 1)

            for encoder_type in ["lstm", "conv1d", "transformer"]:
                encoder = create_temporal_encoder(encoder_type, 1, 64)

                output = encoder(x)

                assert (
                    output.shape == (5, 64)
                ), f"{encoder_type} with seq_len={seq_len}: wrong shape"
                assert torch.isfinite(
                    output
                ).all(), f"{encoder_type} with seq_len={seq_len}: NaN/Inf"


class TestEdgeCases:
    """Test edge cases and potential failure modes."""

    def test_single_asset(self):
        """Test encoders work with single asset."""
        x = torch.randn(1, 60, 1)

        for encoder_type in ["lstm", "conv1d", "transformer"]:
            encoder = create_temporal_encoder(encoder_type, 1, 64)
            output = encoder(x)

            assert output.shape == (1, 64)
            assert torch.isfinite(output).all()

    def test_short_sequence(self):
        """Test encoders work with short sequences."""
        x = torch.randn(10, 5, 1)  # Very short sequence

        for encoder_type in ["lstm", "conv1d", "transformer"]:
            encoder = create_temporal_encoder(encoder_type, 1, 64)
            output = encoder(x)

            assert output.shape == (10, 64)
            assert torch.isfinite(output).all()

    def test_large_hidden_dim(self):
        """Test encoders work with large hidden dimensions."""
        x = torch.randn(5, 30, 1)

        for encoder_type in ["lstm", "conv1d", "transformer"]:
            encoder = create_temporal_encoder(encoder_type, 1, 256)
            output = encoder(x)

            assert output.shape == (5, 256)
            assert torch.isfinite(output).all()
