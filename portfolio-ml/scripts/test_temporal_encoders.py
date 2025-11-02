"""Test temporal encoders with forward and backward passes."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models.gat.temporal_encoders import (
    TemporalLSTMEncoder,
    TemporalConvEncoder,
    TemporalTransformerEncoder,
    create_temporal_encoder,
)


def test_encoder(encoder_name: str, encoder: torch.nn.Module):
    """Test a temporal encoder with forward and backward passes.

    Args:
        encoder_name: Name of the encoder for display
        encoder: The encoder module to test
    """
    print(f"\nTesting {encoder_name}:")
    print(f"  Input features: {encoder.input_features}")
    print(f"  Hidden dim: {encoder.hidden_dim}")

    # Create dummy time-series data
    num_assets = 10
    time_steps = 60
    features = 1
    x = torch.randn(num_assets, time_steps, features)

    # Forward pass
    encoded = encoder(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {encoded.shape}")
    print(f"  Output mean: {encoded.mean().item():.4f}")
    print(f"  Output std: {encoded.std().item():.4f}")

    # Verify output shape
    assert encoded.shape == (num_assets, encoder.hidden_dim), (
        f"Wrong output shape: expected ({num_assets}, {encoder.hidden_dim}), "
        f"got {encoded.shape}"
    )

    # Backward pass (verify gradients flow)
    loss = encoded.mean()
    loss.backward()

    # Check that parameters have gradients
    has_gradients = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in encoder.parameters()
    )

    print(f"  Gradients computed: {has_gradients}")

    assert has_gradients, "No gradients computed!"
    print(f"  ✅ {encoder_name} passed all tests")


def main():
    """Run all temporal encoder tests."""
    print("=" * 60)
    print("TEMPORAL ENCODER TESTS")
    print("=" * 60)

    hidden_dim = 64

    # Test LSTM encoder
    lstm_encoder = TemporalLSTMEncoder(input_features=1, hidden_dim=hidden_dim)
    test_encoder("TemporalLSTMEncoder", lstm_encoder)

    # Test Conv1D encoder
    conv_encoder = TemporalConvEncoder(input_features=1, hidden_dim=hidden_dim)
    test_encoder("TemporalConvEncoder", conv_encoder)

    # Test Transformer encoder
    transformer_encoder = TemporalTransformerEncoder(
        input_features=1, hidden_dim=hidden_dim
    )
    test_encoder("TemporalTransformerEncoder", transformer_encoder)

    # Test factory function
    print("\nTesting factory function:")
    for encoder_type in ["lstm", "conv1d", "transformer"]:
        encoder = create_temporal_encoder(encoder_type, input_features=1, hidden_dim=64)
        print(f"  ✅ Created {encoder_type} encoder via factory")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
