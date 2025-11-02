"""Temporal encoding modules for time-series node features in GAT."""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TemporalLSTMEncoder(nn.Module):
    """LSTM-based temporal encoder for time-series node features.

    Encodes variable-length time-series features into fixed-size embeddings
    suitable for graph attention networks.

    Recommended for:
    - Moderate sequence lengths (60-200 timesteps)
    - When capturing long-term dependencies is critical
    - When training time is not the primary concern
    """

    def __init__(
        self,
        input_features: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        """Initialise LSTM temporal encoder.

        Args:
            input_features: Number of features per timestep
            hidden_dim: Hidden dimension for LSTM and output
            num_layers: Number of LSTM layers
            dropout: Dropout rate (applied if num_layers > 1)
        """
        super().__init__()

        self.input_features = input_features
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode time-series features.

        Args:
            x: Time-series features [num_assets, time_steps, features]

        Returns:
            Encoded features [num_assets, hidden_dim]
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use final hidden state
        final_state = h_n[-1]  # [num_assets, hidden_dim]

        # Project and normalise
        encoded = self.projection(final_state)
        encoded = self.layer_norm(encoded)

        return encoded


class TemporalConvEncoder(nn.Module):
    """1D Convolutional encoder for time-series node features.

    Fastest option with lowest memory usage. Good for local temporal patterns.

    Recommended for:
    - Short to moderate sequences (30-100 timesteps)
    - When training speed is critical
    - When memory is constrained
    - Initial experiments
    """

    def __init__(
        self,
        input_features: int,
        hidden_dim: int,
        kernel_size: int = 5,
        num_layers: int = 2,
    ):
        """Initialise Conv1D temporal encoder.

        Args:
            input_features: Number of features per timestep
            hidden_dim: Hidden dimension and output size
            kernel_size: Convolutional kernel size
            num_layers: Number of convolutional layers
        """
        super().__init__()

        self.input_features = input_features
        self.hidden_dim = hidden_dim

        # Build convolutional layers
        layers = []
        in_channels = input_features

        for _ in range(num_layers):
            layers.extend([
                nn.Conv1d(
                    in_channels,
                    hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
            ])
            in_channels = hidden_dim

        # Global pooling
        layers.append(nn.AdaptiveAvgPool1d(1))

        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode time-series features.

        Args:
            x: Time-series features [num_assets, time_steps, features]

        Returns:
            Encoded features [num_assets, hidden_dim]
        """
        # Transpose for Conv1d: [num_assets, features, time_steps]
        x = x.transpose(1, 2)

        # Apply convolutions
        encoded = self.conv_layers(x)  # [num_assets, hidden_dim, 1]

        # Remove time dimension
        encoded = encoded.squeeze(-1)  # [num_assets, hidden_dim]

        return encoded


class TemporalTransformerEncoder(nn.Module):
    """Transformer-based temporal encoder for time-series node features.

    Best for capturing global temporal dependencies with attention.
    Higher memory usage but parallel processing.

    Recommended for:
    - Long sequences (200+ timesteps)
    - When global context is important
    - When GPU memory is abundant
    - Production systems after Conv1D/LSTM validation
    """

    def __init__(
        self,
        input_features: int,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """Initialise Transformer temporal encoder.

        Args:
            input_features: Number of features per timestep
            hidden_dim: Hidden dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            num_layers: Number of transformer encoder layers
            dropout: Dropout rate
        """
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.input_features = input_features
        self.hidden_dim = hidden_dim

        # Input embedding
        self.embedding = nn.Linear(input_features, hidden_dim)

        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, hidden_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode time-series features.

        Args:
            x: Time-series features [num_assets, time_steps, features]

        Returns:
            Encoded features [num_assets, hidden_dim]
        """
        num_assets, time_steps, _ = x.shape

        # Embed input
        x = self.embedding(x)  # [num_assets, time_steps, hidden_dim]

        # Add positional encoding
        x = x + self.pos_encoder[:, :time_steps, :]

        # Transformer encoding
        encoded = self.transformer(x)  # [num_assets, time_steps, hidden_dim]

        # Pool over time dimension (mean pooling)
        pooled = encoded.mean(dim=1)  # [num_assets, hidden_dim]

        # Output projection
        output = self.output_proj(pooled)

        return output


def create_temporal_encoder(
    encoder_type: Literal["lstm", "conv1d", "transformer"],
    input_features: int,
    hidden_dim: int,
    **kwargs,
) -> nn.Module:
    """Factory function to create temporal encoder.

    Args:
        encoder_type: Type of encoder ("lstm", "conv1d", "transformer")
        input_features: Number of features per timestep
        hidden_dim: Hidden dimension for encoder output
        **kwargs: Additional encoder-specific arguments

    Returns:
        Initialised temporal encoder module

    Raises:
        ValueError: If encoder_type is invalid

    Examples:
        >>> # Create LSTM encoder
        >>> encoder = create_temporal_encoder("lstm", input_features=1, hidden_dim=64)

        >>> # Create Conv1D encoder with custom kernel
        >>> encoder = create_temporal_encoder(
        ...     "conv1d", input_features=1, hidden_dim=64, kernel_size=7
        ... )
    """
    if encoder_type == "lstm":
        return TemporalLSTMEncoder(input_features, hidden_dim, **kwargs)
    elif encoder_type == "conv1d":
        return TemporalConvEncoder(input_features, hidden_dim, **kwargs)
    elif encoder_type == "transformer":
        return TemporalTransformerEncoder(input_features, hidden_dim, **kwargs)
    else:
        raise ValueError(
            f"Unknown encoder_type: {encoder_type}. "
            f"Must be one of: lstm, conv1d, transformer"
        )
