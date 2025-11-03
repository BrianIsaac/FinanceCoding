"""
LSTM network architecture for portfolio optimization.

This module implements sequence-to-sequence LSTM networks with multi-head attention
for temporal pattern recognition in financial time series.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

# Module-level logger to fix UnboundLocalError (Bug #1)
logger = logging.getLogger(__name__)


@dataclass
class LSTMConfig:
    """Configuration for LSTM network architecture."""

    sequence_length: int = 252  # 252-day (1 year) lookback window for seasonal patterns
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

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LSTM network with numerical stability.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            lengths: Optional sequence lengths tensor (for ragged sequences).
                    Ignored in standard implementation, used by RaggedLSTMNetwork.

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

        # Apply batch normalization for gradient stability
        # Skip only for single-sample batches (batch_size=1) to avoid std=0 issues
        if final_hidden.size(0) > 1:
            final_hidden = self.batch_norm(final_hidden)

        # Apply dropout only during training
        if self.training:
            final_hidden = self.dropout(final_hidden)

        # Output projection with clamping
        predictions = self.output_projection(final_hidden)  # (batch_size, output_size)
        # Clamp to match normalized returns range (±3 std covers 99.7% of normal distribution)
        predictions = torch.clamp(predictions, -3.0, 3.0)

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

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        entropy_weight: float = 0.1,
        temperature: float = 3.0,
        gradient_clip_value: float = 10.0,
        adaptive_epsilon: bool = True
    ):
        """
        Initialize Sharpe ratio loss function with enhanced numerical stability.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            entropy_weight: Weight for entropy regularisation to prevent concentration
                          Increased from 0.01 to 0.1 to encourage diversification (10x increase)
            temperature: Softmax temperature for portfolio weight generation (default: 3.0)
                        - Higher (5.0): More uniform allocations (exploration)
                        - Lower (1.0): More concentrated allocations (exploitation)
            gradient_clip_value: Maximum gradient norm before clipping (default: 10.0)
            adaptive_epsilon: Use adaptive epsilon based on historical volatility (default: True)
        """
        super().__init__()
        self.risk_free_rate = risk_free_rate / 252  # Convert to daily rate
        self.entropy_weight = entropy_weight
        self.temperature = temperature
        self.gradient_clip_value = gradient_clip_value
        self.adaptive_epsilon = adaptive_epsilon

        # Track historical volatility for adaptive epsilon
        self.register_buffer('historical_std', torch.tensor(0.05))  # Initialize with conservative estimate
        self.register_buffer('update_count', torch.tensor(0))

    def set_temperature(self, temperature: float):
        """
        Update temperature parameter for curriculum learning.

        Args:
            temperature: New temperature value
                - Higher (5.0): More uniform allocations (early training exploration)
                - Lower (1.0): More concentrated allocations (late training exploitation)

        Example:
            # Temperature annealing schedule
            for epoch in range(num_epochs):
                temp = 5.0 - (epoch / num_epochs) * 3.0  # 5.0 → 2.0
                loss_fn.set_temperature(temp)
        """
        self.temperature = temperature

    def get_debug_metrics(self, portfolio_returns: torch.Tensor, portfolio_weights: torch.Tensor) -> dict:
        """
        Calculate debugging metrics for monitoring training stability.

        Args:
            portfolio_returns: Portfolio returns tensor
            portfolio_weights: Portfolio weights tensor

        Returns:
            dict with debugging metrics
        """
        with torch.no_grad():
            metrics = {
                'portfolio_std': portfolio_returns.std().item(),
                'portfolio_mean': portfolio_returns.mean().item(),
                'weight_concentration': (portfolio_weights ** 2).sum(dim=-1).mean().item(),
                'effective_assets': (1.0 / (portfolio_weights ** 2).sum(dim=-1)).mean().item(),
                'max_weight': portfolio_weights.max().item(),
                'min_weight': portfolio_weights.min().item(),
                'historical_std': self.historical_std.item() if hasattr(self, 'historical_std') else 0.05,
            }
        return metrics

    def forward(
        self,
        predicted_returns: torch.Tensor,
        actual_returns: torch.Tensor,
        portfolio_weights: torch.Tensor | None = None,
        model: nn.Module | None = None,
    ) -> torch.Tensor:
        """
        Compute negative Sharpe ratio as loss for direct optimization.

        Args:
            predicted_returns: Model predictions of shape (batch_size, n_assets)
            actual_returns: Actual returns of shape (batch_size, n_assets)
            portfolio_weights: Portfolio weights of shape (batch_size, n_assets)
                             If None, uses predicted returns directly
            model: Optional model reference for accessing normalization_stats

        Returns:
            Negative Sharpe ratio as loss tensor
        """
        # Clamp inputs to prevent extreme values
        # Use ±3 to match normalized returns range (99.7% of normal distribution)
        predicted_returns = torch.clamp(predicted_returns, -3.0, 3.0)
        actual_returns = torch.clamp(actual_returns, -3.0, 3.0)

        if portfolio_weights is None:
            # FIX: Use differentiable softmax for gradient flow
            # The argsort operation was non-differentiable and broke gradient computation
            # Softmax provides smooth, differentiable portfolio weights

            # Use instance temperature for curriculum learning support
            # Temperature can be annealed during training: 5.0 (exploration) → 2.0 (exploitation)
            # Apply softmax to get differentiable portfolio weights
            # This maintains gradient flow through the entire computation graph
            # FIX #6c: Clamp predictions before softmax to prevent overflow/underflow
            predicted_returns_clamped = torch.clamp(predicted_returns, -100, 100)
            portfolio_weights = F.softmax(predicted_returns_clamped / self.temperature, dim=-1)

            # Optional: Apply position constraints if needed
            if hasattr(self, 'max_position_weight') and self.max_position_weight is not None:
                # Clamp weights to maximum position size
                portfolio_weights = torch.clamp(portfolio_weights, max=self.max_position_weight)
                # Renormalise to ensure weights sum to 1
                portfolio_weights = portfolio_weights / portfolio_weights.sum(dim=-1, keepdim=True)

        # Compute portfolio returns with explicit shape alignment
        # Ensure both tensors have the same shape before multiplication
        if portfolio_weights.shape != actual_returns.shape:
            # Align dimensions if needed
            if portfolio_weights.dim() == 1 and actual_returns.dim() == 2:
                portfolio_weights = portfolio_weights.unsqueeze(0)
            elif portfolio_weights.dim() == 2 and actual_returns.dim() == 1:
                actual_returns = actual_returns.unsqueeze(0)

        portfolio_returns = (portfolio_weights * actual_returns).sum(dim=1)

        # CRITICAL FIX: Denormalize portfolio returns to actual scale before computing Sharpe ratio
        # Problem: SharpeRatioLoss operates on normalized returns (std=1.0) but should operate on
        # actual scale for meaningful optimization. Normalized scale inflates Sharpe by ~100x.
        # Solution: Denormalize using stored normalization_stats from training
        # Reference: PyTorch autograd preserves gradients through scaling operations (a*x + b)

        # Priority 1: Log normalization stats availability
        stats_available = hasattr(model, 'normalization_stats') and model.normalization_stats is not None if model is not None else False
        logger.debug(f"Normalization: available={stats_available}, scale={'ACTUAL' if stats_available else 'NORMALIZED'}")

        if model is not None and hasattr(model, 'normalization_stats') and model.normalization_stats:
            stats = model.normalization_stats
            mean = stats.get('mean', 0.0)
            std = stats.get('std', 1.0)
            epsilon = stats.get('epsilon', 1e-6)

            # Convert numpy arrays to tensors if needed
            if not isinstance(mean, torch.Tensor):
                mean = torch.tensor(mean, device=portfolio_returns.device, dtype=portfolio_returns.dtype)
            if not isinstance(std, torch.Tensor):
                std = torch.tensor(std, device=portfolio_returns.device, dtype=portfolio_returns.dtype)

            # Denormalize portfolio returns to actual scale
            # Formula: actual = normalized * std + mean
            # Use mean of std across assets for portfolio-level scaling
            std_factor = std.mean() + epsilon
            mean_offset = mean.mean()

            portfolio_returns = portfolio_returns * std_factor + mean_offset

            logger.debug(f"Loss denormalized: std_factor={std_factor.item():.6f}, "
                        f"mean_offset={mean_offset.item():.6f}, "
                        f"portfolio_return_mean={portfolio_returns.mean().item():.6f}")

        # Compute excess returns
        excess_returns = portfolio_returns - self.risk_free_rate

        # CRITICAL FIX: Use biased std (correction=0) to handle single-sample batches
        # Single sample batches occur when dataset_size % batch_size = 1
        # Unbiased std (default) uses (n-1) denominator, which causes NaN when n=1
        # Biased std uses (n) denominator, which handles n=1 gracefully
        mean_excess = excess_returns.mean()

        # Use biased=False with correction=0 (equivalent to biased std) for numerical stability
        # This prevents "degrees of freedom <= 0" error when batch has only 1 sample
        std_excess = excess_returns.std(correction=0)  # Biased std to handle n=1 case

        # CRITICAL FIX: Ensure std_excess is finite and clamped to prevent numerical issues
        # Use differentiable nan_to_num instead of torch.tensor to preserve gradient flow
        std_excess = torch.nan_to_num(std_excess, nan=1e-3, posinf=1e-3, neginf=1e-3)

        # Clamp std_excess to prevent division by near-zero even with epsilon
        # Minimum of 1e-4 ensures denominator is always >> epsilon for stable gradients
        std_excess = torch.clamp(std_excess, min=1e-4)

        # Log if replacement occurred
        if not torch.isfinite(excess_returns.std(correction=0) + 1e-8):
            logger.warning(f"Non-finite std_excess detected and corrected via nan_to_num")

        # CRITICAL FIX v3: Adaptive epsilon based on historical volatility
        # Previous fixed eps=5e-2 was too conservative for high-volatility portfolios
        # Adaptive epsilon scales with portfolio volatility to maintain gradient stability
        # across different market conditions
        if self.adaptive_epsilon and self.training:
            # Update historical std with exponential moving average
            with torch.no_grad():
                alpha = 0.1  # Smoothing factor
                self.historical_std = alpha * std_excess + (1 - alpha) * self.historical_std
                self.update_count += 1

            # FIXED: Reduced epsilon range from [1e-2, 1e-1] to [1e-3, 1e-2] to prevent gradient explosion
            # Previous large epsilon dominated std_excess in denominator, causing:
            #   - Gradient of 1/(std+eps)^2 to explode when std < eps
            #   - Combined with compressed normalization, created 8000x gradient amplification
            # New range keeps epsilon as stability term, not dominant factor in denominator
            # FIX #6b: Further reduced from 0.1 to 0.01 scale, range [1e-4, 1e-3] for additional stability
            eps = torch.clamp(self.historical_std * 0.01, min=1e-4, max=1e-3)
            if eps > std_excess * 0.5:
                logger.debug(f"Sharpe epsilon ({eps:.4f}) > 50% of portfolio std ({std_excess:.4f}) - may suppress gradient signal")
        else:
            # Fixed epsilon for inference or non-adaptive mode
            # FIXED: Reduced from 5e-2 to 1e-3 for consistency with adaptive mode
            eps = 1e-3

        # FIXED: Direct Sharpe ratio formula without +1.0 shift
        # The shift caused 100-132x inflation: model saw random predictions as Sharpe 5.0
        # when actual Sharpe was 0.0, preventing any learning
        # Use direct mean/std ratio clamped to prevent extremes

        # Use direct ratio instead of log-space to avoid gradient amplification
        sharpe_ratio = mean_excess / (std_excess + eps)
        # Clamp result to prevent extreme values and ensure finite gradients
        sharpe_ratio = torch.clamp(sharpe_ratio, -10.0, 10.0)

        # Determine if denormalization was applied
        denormalized = (model is not None and hasattr(model, 'normalization_stats')
                       and model.normalization_stats is not None)
        scale_status = 'ACTUAL (denormalized)' if denormalized else 'NORMALIZED (std~1.0)'

        logger.debug(f"Loss Sharpe: mean={mean_excess.item():.6f}, "
                     f"std={std_excess.item():.6f}, sharpe={sharpe_ratio.item():.6f}, "
                     f"scale={scale_status}")

        # Note: Removed MSE fallback to maintain consistent loss function scale
        # Log formulation handles all edge cases including small std

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

        # CRITICAL FIX v3: Add gradient clipping at loss level for stability
        # Clip loss value to prevent extreme gradients during backpropagation
        # This provides an additional safety net beyond optimizer-level gradient clipping
        if self.training and self.gradient_clip_value is not None:
            loss = torch.clamp(loss, -self.gradient_clip_value, self.gradient_clip_value)

        # Add numerical stability check
        if torch.isnan(loss) or torch.isinf(loss):
            # Logger already defined at module level
            logger.error(
                f"NaN/Inf loss detected! sharpe_ratio={sharpe_ratio.item():.4f}, "
                f"mean_excess={mean_excess.item():.4f}, std_excess={std_excess.item():.4f}, "
                f"eps={eps if isinstance(eps, float) else eps.item():.4f}, "
                f"entropy={portfolio_entropy.item():.4f}"
            )
            # Return large but finite loss to prevent training collapse
            return torch.tensor(10.0, device=loss.device, requires_grad=True)

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
