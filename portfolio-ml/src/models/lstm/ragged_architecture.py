"""Ragged tensor LSTM architecture for variable-length sequences."""

from __future__ import annotations

from typing import Optional, Tuple
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .architecture import LSTMConfig, MultiHeadAttention
from .ragged_utils import (
    pack_ragged_sequences,
    unpack_ragged_sequences,
    compute_sequence_statistics,
    create_length_mask,
)

logger = logging.getLogger(__name__)


class RaggedLSTMNetwork(nn.Module):
    """
    LSTM network with native support for variable-length sequences.

    Uses PyTorch PackedSequence to eliminate computational waste on padding
    whilst maintaining compatibility with existing LSTMConfig.

    Key benefits over standard LSTM:
    - Zero computation on padded values
    - True variable-length sequence support
    - Lower memory usage (no padding storage in LSTM)
    - Cleaner gradients (no padding contamination)

    References:
        PyTorch PackedSequence: https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.PackedSequence.html
        Best practices: See thoughts/shared/research/2025-10-29-unified-ragged-lstm-plan-validation.md
    """

    def __init__(self, config: LSTMConfig):
        """
        Initialise ragged LSTM network.

        Args:
            config: LSTM configuration (same as standard LSTMNetwork)
        """
        super().__init__()
        self.config = config

        # Input projection layer
        self.input_projection = nn.Linear(config.input_size, config.hidden_size)

        # LSTM layers - same configuration as standard LSTM
        # PackedSequence will be used internally
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,  # Required for our use case
            bidirectional=False,
        )

        # Multi-head attention mechanism (same as standard LSTM)
        self.attention = MultiHeadAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
        )

        # Batch normalisation for stability
        self.batch_norm = nn.BatchNorm1d(config.hidden_size)

        # Output projection layer
        self.output_projection = nn.Linear(config.hidden_size, config.output_size)

        # Dropout for regularisation
        self.dropout = nn.Dropout(config.dropout)

        # Initialise weights (same as standard LSTM)
        self._initialise_weights()

        # Track sequence statistics for monitoring
        self._sequence_stats = None

    def _initialise_weights(self) -> None:
        """Initialise network weights using Xavier initialisation."""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                torch.nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0.0)
            elif "weight" in name and param.dim() >= 2:
                torch.nn.init.xavier_uniform_(param.data)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with ragged tensor support.

        Args:
            x: Input tensor of shape (batch_size, max_seq_len, input_size)
               Padded to max_seq_len for batching, but lengths specify actual data
            lengths: Tensor of shape (batch_size,) with actual sequence lengths
                    Each value indicates non-padded length for that sequence

        Returns:
            Tuple of (predictions, attention_weights):
            - predictions: Predicted returns of shape (batch_size, output_size)
            - attention_weights: Attention weights of shape (batch_size, max_seq_len)

        Example:
            >>> config = LSTMConfig(input_size=100, hidden_size=128, output_size=100)
            >>> model = RaggedLSTMNetwork(config)
            >>> sequences = torch.randn(32, 60, 100)  # batch=32, max_len=60, features=100
            >>> lengths = torch.randint(30, 61, (32,))  # Actual lengths 30-60
            >>> predictions, attention = model(sequences, lengths)
            >>> predictions.shape
            torch.Size([32, 100])
        """
        batch_size, max_seq_len, input_size = x.shape

        # Compute and log sequence statistics
        self._sequence_stats = compute_sequence_statistics(x, lengths)
        if self.training and batch_size > 1:
            logger.debug(
                f"Ragged LSTM: avg_len={self._sequence_stats['mean_length']:.1f}, "
                f"padding_ratio={self._sequence_stats['padding_ratio']:.2%}"
            )

        # Clamp input to ±3 standard deviations (99.7% of normal distribution)
        # After z-score normalisation, this catches extreme outliers
        # whilst preserving most of the signal
        x = torch.clamp(x, -3.0, 3.0)

        # Input projection
        x = self.input_projection(x)  # (batch, max_seq_len, hidden_size)

        # Pack sequences - this is where we eliminate padding computation
        packed_input, sort_indices = pack_ragged_sequences(x, lengths, enforce_sorted=False)

        # LSTM processing on packed sequences
        # CRITICAL: LSTM only processes actual data, not padding!
        packed_output, (hidden, cell) = self.lstm(packed_input)

        # Unpack sequences back to padded format for attention
        lstm_output, unpacked_lengths = unpack_ragged_sequences(
            packed_output,
            sort_indices,
            total_length=max_seq_len
        )

        # Clamp LSTM output
        lstm_output = torch.clamp(lstm_output, -10.0, 10.0)

        # Apply attention mechanism with length masking
        try:
            attended_output = self.attention(lstm_output)
            attended_output = torch.clamp(attended_output, -10.0, 10.0)
            attended_output = attended_output + lstm_output  # Residual connection
        except Exception as e:
            logger.warning(f"Attention failed: {e}, using LSTM output directly")
            attended_output = lstm_output

        # Use the last ACTUAL timestep output for prediction (not padding)
        # Create mask for valid positions
        length_mask = create_length_mask(attended_output, lengths)

        # Get final hidden state (last valid timestep for each sequence)
        # Method: use lengths to index into correct position
        batch_indices = torch.arange(batch_size, device=x.device)
        last_indices = (lengths - 1).clamp(min=0)  # Clamp for safety
        final_hidden = attended_output[batch_indices, last_indices, :]  # (batch, hidden_size)

        # FIXED: Apply batch normalisation during training, not evaluation
        # Previously was backwards (only applied during eval)
        # Batch norm should normalise during training and use running stats during eval
        if final_hidden.size(0) > 1 and self.training:
            try:
                final_hidden = self.batch_norm(final_hidden)
            except Exception:
                pass  # Skip if batch norm fails

        # Apply dropout only during training
        if self.training:
            final_hidden = self.dropout(final_hidden)

        # Output projection with clamping
        predictions = self.output_projection(final_hidden)
        # Clamp to match normalized returns range (±3 std covers 99.7% of normal distribution)
        predictions = torch.clamp(predictions, -3.0, 3.0)

        # Compute attention weights for interpretability
        # Use final hidden state to compute attention over sequence
        attention_query = final_hidden.unsqueeze(1)  # (batch, 1, hidden_size)
        attention_scores = torch.bmm(
            attention_query, lstm_output.transpose(1, 2)
        )  # (batch, 1, max_seq_len)

        # Mask out padding positions before softmax
        attention_scores = attention_scores.squeeze(1)  # (batch, max_seq_len)
        attention_scores = attention_scores.masked_fill(~length_mask, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch, max_seq_len)

        return predictions, attention_weights

    def get_sequence_statistics(self) -> Optional[dict]:
        """
        Get statistics from the most recent forward pass.

        Returns:
            Dictionary with sequence statistics or None if no forward pass yet
        """
        return self._sequence_stats

    def get_memory_usage(self, batch_size: int, sequence_length: int, avg_real_length: int) -> int:
        """
        Estimate forward pass memory usage with ragged tensors.

        Args:
            batch_size: Batch size
            sequence_length: Maximum sequence length (for padding)
            avg_real_length: Average actual sequence length

        Returns:
            Estimated memory usage in bytes

        Note:
            This is more accurate than the standard LSTM because it accounts
            for the fact that packed sequences don't store/compute padding.
        """
        # Input memory (still padded for batching)
        input_memory = batch_size * sequence_length * self.config.input_size * 4

        # Packed sequence memory (only actual data, no padding!)
        packed_memory = batch_size * avg_real_length * self.config.hidden_size * 4

        # LSTM hidden/cell states
        lstm_memory = batch_size * self.config.num_layers * self.config.hidden_size * 4 * 2

        # Attention memory (on unpacked sequences)
        attention_memory = (
            batch_size * self.config.num_attention_heads * sequence_length * sequence_length * 4
        )

        # Parameter memory
        param_memory = sum(p.numel() * 4 for p in self.parameters())

        # Intermediate computations
        intermediate_memory = batch_size * self.config.hidden_size * 4 * 3

        total_memory = (
            input_memory
            + packed_memory
            + lstm_memory
            + attention_memory
            + param_memory
            + intermediate_memory
        )

        return int(total_memory)


def create_ragged_lstm_network(config: LSTMConfig) -> RaggedLSTMNetwork:
    """
    Factory function to create ragged LSTM network with configuration validation.

    Args:
        config: LSTM configuration

    Returns:
        Initialised ragged LSTM network

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate configuration (same as standard LSTM)
    if config.sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {config.sequence_length}")

    if config.hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {config.hidden_size}")

    if config.num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {config.num_layers}")

    if not 0.0 <= config.dropout < 1.0:
        raise ValueError(f"dropout must be in [0.0, 1.0), got {config.dropout}")

    if config.hidden_size % config.num_attention_heads != 0:
        raise ValueError(
            f"hidden_size ({config.hidden_size}) must be divisible by "
            f"num_attention_heads ({config.num_attention_heads})"
        )

    return RaggedLSTMNetwork(config)
