"""Portfolio loss functions for GAT training.

Simple, proven losses that maintain healthy gradient flow.
NO complex logarithmic Sharpe (causes vanishing gradients).

Implements:
- Portfolio Return Loss: Maximise returns, penalise volatility
- Negative Sharpe: Direct Sharpe ratio optimization (with proper scaling)
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PortfolioReturnLoss(nn.Module):
    """Simple loss that maximises returns while penalising volatility.

    Loss = -mean_return + lambda * volatility

    Benefits:
    - Clean, simple formulation
    - Maintains healthy gradients (no log operations)
    - Direct optimization of risk-return tradeoff
    """

    def __init__(
        self,
        vol_penalty: float = 0.5,
        risk_free_rate: float = 0.0,
    ):
        """Initialize portfolio return loss.

        Args:
            vol_penalty: Weight for volatility penalty (default 0.5)
            risk_free_rate: Annual risk-free rate (converted to daily)
        """
        super().__init__()
        self.vol_penalty = vol_penalty
        self.risk_free_rate = risk_free_rate / 252  # Convert annual to daily

    def forward(
        self,
        weights: torch.Tensor,
        returns: torch.Tensor,
        **kwargs  # Absorb unused arguments for compatibility
    ) -> torch.Tensor:
        """Compute portfolio loss.

        Args:
            weights: Portfolio weights [batch_size, n_assets] or [n_assets]
            returns: Asset returns [batch_size, time_steps, n_assets] or [time_steps, n_assets]

        Returns:
            Loss value (scalar)
        """
        # Handle dimension mismatches
        if weights.dim() == 1:
            weights = weights.unsqueeze(0)  # [1, n_assets]

        if returns.dim() == 2:
            # [time_steps, n_assets] → [1, time_steps, n_assets]
            returns = returns.unsqueeze(0)

        # Compute portfolio returns
        # [batch, 1, n_assets] × [batch, time, n_assets] → [batch, time]
        portfolio_returns = (weights.unsqueeze(1) * returns).sum(dim=2)

        # Check for NaN in portfolio returns (defensive check)
        if torch.isnan(portfolio_returns).any():
            logger.warning("NaN detected in portfolio_returns, returning large penalty loss")
            # Return penalty loss connected to computational graph (maintains gradients)
            return weights.sum() * 0.0 + 1000.0  # Large penalty while preserving grad_fn

        # Mean return (to maximise)
        mean_return = portfolio_returns.mean()

        # Volatility (to minimise) - add small epsilon to prevent division issues
        volatility = portfolio_returns.std() + 1e-8

        # Check for NaN in loss components
        if torch.isnan(mean_return) or torch.isnan(volatility):
            logger.warning(f"NaN in loss components: mean_return={mean_return}, volatility={volatility}")
            # Return penalty loss connected to computational graph (maintains gradients)
            return weights.sum() * 0.0 + 1000.0  # Large penalty while preserving grad_fn

        # Combined loss
        loss = -mean_return + self.vol_penalty * volatility

        return loss


class NegativeSharpeLoss(nn.Module):
    """Negative Sharpe ratio loss with proper scaling.

    Uses direct Sharpe computation with scaling to maintain gradients.
    Simpler than logarithmic formulation, avoids vanishing gradients.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.03,
        scaling_factor: float = 100.0,
    ):
        """Initialize negative Sharpe loss.

        Args:
            risk_free_rate: Annual risk-free rate
            scaling_factor: Scale factor to prevent vanishing gradients (default 100)
        """
        super().__init__()
        self.risk_free_rate = risk_free_rate / 252  # Convert annual to daily
        self.scaling_factor = scaling_factor

    def forward(
        self,
        weights: torch.Tensor,
        returns: torch.Tensor,
        **kwargs  # Absorb unused arguments for compatibility
    ) -> torch.Tensor:
        """Compute negative Sharpe ratio loss.

        Args:
            weights: Portfolio weights [batch_size, n_assets] or [n_assets]
            returns: Asset returns [batch_size, time_steps, n_assets] or [time_steps, n_assets]

        Returns:
            Loss value (scalar)
        """
        # Handle dimension mismatches
        if weights.dim() == 1:
            weights = weights.unsqueeze(0)

        if returns.dim() == 2:
            returns = returns.unsqueeze(0)

        # Compute portfolio returns
        portfolio_returns = (weights.unsqueeze(1) * returns).sum(dim=2)

        # Check for NaN in portfolio returns (defensive check)
        if torch.isnan(portfolio_returns).any():
            logger.warning("NaN detected in portfolio_returns, returning large penalty loss")
            # Return penalty loss connected to computational graph (maintains gradients)
            return weights.sum() * 0.0 + 1000.0  # Large penalty while preserving grad_fn

        # Excess returns
        excess_returns = portfolio_returns - self.risk_free_rate

        # Sharpe ratio
        mean_excess = excess_returns.mean()
        std_excess = excess_returns.std() + 1e-6  # Prevent division by zero

        # Check for NaN in Sharpe components
        if torch.isnan(mean_excess) or torch.isnan(std_excess):
            logger.warning(f"NaN in Sharpe components: mean_excess={mean_excess}, std_excess={std_excess}")
            # Return penalty loss connected to computational graph (maintains gradients)
            return weights.sum() * 0.0 + 1000.0  # Large penalty while preserving grad_fn

        sharpe = mean_excess / std_excess

        # Negative Sharpe (we minimize loss, so negate Sharpe to maximize it)
        # Scale to prevent vanishing gradients
        loss = -sharpe * self.scaling_factor

        return loss


def get_loss_function(
    loss_type: str = "return",
    **kwargs
) -> nn.Module:
    """Factory function to create loss functions.

    Args:
        loss_type: Type of loss ("return" or "sharpe")
        **kwargs: Additional arguments passed to loss constructor

    Returns:
        Loss function module
    """
    if loss_type == "return":
        return PortfolioReturnLoss(**kwargs)
    elif loss_type == "sharpe":
        return NegativeSharpeLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose 'return' or 'sharpe'.")
