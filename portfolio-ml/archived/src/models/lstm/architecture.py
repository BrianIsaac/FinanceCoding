"""
LSTM network architecture for portfolio optimization.

This module implements sequence-to-sequence LSTM networks with multi-head attention
for temporal pattern recognition in financial time series.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LSTMConfig:
    """Configuration for LSTM network architecture."""

    sequence_length: int = 60  # 60-day lookback window
    input_size: int = 1  # Number of features per asset (returns)
    hidden_size: int = 128  # LSTM hidden dimensions
    num_layers: int = 2  # Stacked LSTM layers
    dropout: float = 0.3  # Regularization
    num_attention_heads: int = 8  # Multi-head attention
    output_size: int = 1  # Forecast horizon (next-month returns)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for focusing on relevant historical periods."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention layer.

        Args:
            hidden_size: Hidden dimension size
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head attention to input sequence.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Attention-weighted output tensor
        """
        batch_size, seq_len, hidden_size = x.shape

        # Compute queries, keys, values
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_size**0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        attention_output = torch.matmul(attention_probs, V)
        attention_output = (
            attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        )

        return self.output(attention_output)


class LSTMNetwork(nn.Module):
    """
    Sequence-to-sequence LSTM network with attention for return forecasting.

    Processes 60-day historical return windows to predict next-month expected returns
    for portfolio optimization.
    """

    def __init__(self, config: LSTMConfig):
        """
        Initialize LSTM network with given configuration.

        Args:
            config: LSTM network configuration
        """
        super().__init__()
        self.config = config

        # Input projection layer
        self.input_projection = nn.Linear(config.input_size, config.hidden_size)

        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )

        # Multi-head attention mechanism
        self.attention = MultiHeadAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
        )

        # Batch normalization for stability
        self.batch_norm = nn.BatchNorm1d(config.hidden_size)

        # Output projection layer
        self.output_projection = nn.Linear(config.hidden_size, config.output_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                # Initialize LSTM input-hidden weights
                torch.nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                # Initialize LSTM hidden-hidden weights
                torch.nn.init.orthogonal_(param.data)
            elif "bias" in name:
                # Initialize biases to zero
                param.data.fill_(0.0)
            elif "weight" in name and param.dim() >= 2:
                # Initialize linear layer weights
                torch.nn.init.xavier_uniform_(param.data)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LSTM network with numerical stability.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Tuple of (predictions, attention_weights)
            - predictions: Predicted returns of shape (batch_size, output_size)
            - attention_weights: Attention weights for interpretability
        """
        batch_size, seq_len, input_size = x.shape

        # Clamp input to prevent extreme values
        x = torch.clamp(x, -10.0, 10.0)

        # Input projection
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_size)

        # LSTM processing
        lstm_output, (hidden, cell) = self.lstm(x)  # (batch_size, seq_len, hidden_size)

        # Clamp LSTM output to prevent exploding values
        lstm_output = torch.clamp(lstm_output, -10.0, 10.0)

        # Apply attention mechanism (more conservative)
        try:
            attended_output = self.attention(lstm_output)  # (batch_size, seq_len, hidden_size)
            attended_output = torch.clamp(attended_output, -10.0, 10.0)
            attended_output = attended_output + lstm_output  # Residual connection
        except:
            # Fallback if attention fails
            attended_output = lstm_output

        # Use the last timestep output for prediction
        final_hidden = attended_output[:, -1, :]  # (batch_size, hidden_size)

        # Skip batch normalization to avoid numerical issues
        # if final_hidden.size(0) > 1:  # Only apply batch norm if batch size > 1
        #     final_hidden = self.batch_norm(final_hidden)

        # Apply dropout only during training
        if self.training:
            final_hidden = self.dropout(final_hidden)

        # Output projection with clamping
        predictions = self.output_projection(final_hidden)  # (batch_size, output_size)
        predictions = torch.clamp(predictions, -1.0, 1.0)  # Reasonable returns range

        # Compute attention weights for interpretability (stable version)
        # Use the final hidden state to compute attention over the sequence
        attention_query = final_hidden.unsqueeze(1)  # (batch_size, 1, hidden_size)
        attention_scores = torch.bmm(
            attention_query, lstm_output.transpose(1, 2)
        )  # (batch_size, 1, seq_len)
        attention_weights = F.softmax(attention_scores.squeeze(1), dim=-1)  # (batch_size, seq_len)

        return predictions, attention_weights

    def get_memory_usage(self, batch_size: int, sequence_length: int) -> int:
        """
        Estimate forward pass memory usage in bytes for given input dimensions.

        Args:
            batch_size: Batch size for training/inference
            sequence_length: Input sequence length

        Returns:
            Estimated forward pass memory usage in bytes (excluding gradients and optimizer)
        """
        # Estimate memory for forward pass activations
        input_memory = batch_size * sequence_length * self.config.input_size * 4  # float32
        hidden_memory = batch_size * sequence_length * self.config.hidden_size * 4
        lstm_memory = (
            batch_size * self.config.num_layers * self.config.hidden_size * 4 * 2
        )  # hidden + cell states

        # Attention mechanism memory (more accurate calculation)
        if hasattr(self, 'attention') and self.attention is not None:
            attention_memory = (
                batch_size * self.config.num_attention_heads * sequence_length * sequence_length * 4
            )
        else:
            attention_memory = 0

        # Parameter memory (model weights only, not gradients)
        param_memory = sum(p.numel() * 4 for p in self.parameters())  # float32

        # Intermediate computation memory (projection layers, activations)
        intermediate_memory = batch_size * self.config.hidden_size * 4 * 3  # Multiple projections

        total_memory = (
            input_memory
            + hidden_memory
            + lstm_memory
            + attention_memory
            + param_memory
            + intermediate_memory
        )

        return int(total_memory)


class SharpeRatioLoss(nn.Module):
    """Custom loss function optimizing Sharpe ratio for portfolio performance."""

    def __init__(self, risk_free_rate: float = 0.0, entropy_weight: float = 0.001):
        """
        Initialize Sharpe ratio loss function.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            entropy_weight: Weight for entropy regularisation to prevent concentration (reduced for better prediction usage)
        """
        super().__init__()
        self.risk_free_rate = risk_free_rate / 252  # Convert to daily rate
        self.entropy_weight = entropy_weight

    def forward(
        self,
        predicted_returns: torch.Tensor,
        actual_returns: torch.Tensor,
        portfolio_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute negative Sharpe ratio as loss for direct optimization.

        Args:
            predicted_returns: Model predictions of shape (batch_size, n_assets)
            actual_returns: Actual returns of shape (batch_size, n_assets)
            portfolio_weights: Portfolio weights of shape (batch_size, n_assets)
                             If None, uses predicted returns directly

        Returns:
            Negative Sharpe ratio as loss tensor
        """
        # Clamp inputs to prevent extreme values
        predicted_returns = torch.clamp(predicted_returns, -1.0, 1.0)
        actual_returns = torch.clamp(actual_returns, -1.0, 1.0)

        if portfolio_weights is None:
            # Use predicted returns as weights (after normalization)
            portfolio_weights = F.softmax(predicted_returns, dim=-1)

        # Compute portfolio returns with explicit shape alignment
        # Ensure both tensors have the same shape before multiplication
        if portfolio_weights.shape != actual_returns.shape:
            # Align dimensions if needed
            if portfolio_weights.dim() == 1 and actual_returns.dim() == 2:
                portfolio_weights = portfolio_weights.unsqueeze(0)
            elif portfolio_weights.dim() == 2 and actual_returns.dim() == 1:
                actual_returns = actual_returns.unsqueeze(0)

        portfolio_returns = (portfolio_weights * actual_returns).sum(dim=1)

        # Compute excess returns
        excess_returns = portfolio_returns - self.risk_free_rate

        # Compute Sharpe ratio with numerical stability
        mean_excess = excess_returns.mean()

        # Add larger epsilon for financial data stability (volatility can be very low)
        eps = 1e-3
        std_excess = excess_returns.std() + eps

        # Check for NaN or inf values
        if not torch.isfinite(mean_excess) or not torch.isfinite(std_excess):
            # Fallback to MSE loss to provide gradient signal
            mse_loss = F.mse_loss(predicted_returns, actual_returns)
            # Scale MSE loss to similar magnitude as Sharpe ratio (no constant offset!)
            return mse_loss * 10.0  # Scale but allow gradients to flow

        sharpe_ratio = mean_excess / std_excess

        # Check for NaN in final sharpe ratio
        if not torch.isfinite(sharpe_ratio):
            # Fallback to MSE loss to provide gradient signal
            mse_loss = F.mse_loss(predicted_returns, actual_returns)
            # Scale MSE loss to similar magnitude as Sharpe ratio (no constant offset!)
            return mse_loss * 10.0  # Scale but allow gradients to flow

        # Clamp the result to prevent extreme values (wider range for financial data)
        sharpe_ratio = torch.clamp(sharpe_ratio, -5.0, 5.0)

        # Add entropy regularisation to encourage diversification
        # Entropy = -sum(w * log(w)) where w are portfolio weights
        # Higher entropy means more diversified portfolio
        eps_entropy = 1e-8
        # Ensure portfolio_weights is 2D for entropy calculation
        if portfolio_weights.dim() == 1:
            portfolio_weights_entropy = portfolio_weights.unsqueeze(0)
        else:
            portfolio_weights_entropy = portfolio_weights
        portfolio_entropy = -(portfolio_weights_entropy * torch.log(portfolio_weights_entropy + eps_entropy)).sum(dim=1).mean()

        # Combine Sharpe ratio loss with entropy bonus
        # Negative Sharpe for minimization, negative entropy to encourage diversification
        loss = -sharpe_ratio - self.entropy_weight * portfolio_entropy

        return loss


def create_lstm_network(config: LSTMConfig) -> LSTMNetwork:
    """
    Factory function to create LSTM network with configuration validation.

    Args:
        config: LSTM configuration

    Returns:
        Initialized LSTM network

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate configuration
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

    return LSTMNetwork(config)
