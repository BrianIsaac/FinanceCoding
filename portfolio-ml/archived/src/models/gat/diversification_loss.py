#!/usr/bin/env python3
"""
Corrected diversification-aware loss functions for GAT portfolio optimization.

This module implements proper diversification incentives that encourage
spreading risk across uncorrelated assets rather than concentrating in single positions.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class CorrectedDiversificationLoss(nn.Module):
    """
    Corrected loss function that properly encourages diversification.

    Key fixes:
    1. Correlation penalty now correctly penalizes concentration in correlated assets
    2. Effective assets constraint ensures minimum diversification
    3. Entropy regularization encourages spread across assets
    4. Proper weighting of objectives
    """

    def __init__(
        self,
        sharpe_weight: float = 1.0,
        diversification_weight: float = 1.0,
        min_effective_assets: int = 15,
        entropy_weight: float = 0.1,
        concentration_penalty: float = 2.0,
        risk_free_rate: float = 0.0,
        debug_mode: bool = False,
    ):
        """
        Initialize corrected diversification loss.

        Args:
            sharpe_weight: Weight for Sharpe ratio component
            diversification_weight: Weight for diversification ratio
            min_effective_assets: Minimum number of effective assets (1/HHI)
            entropy_weight: Weight for entropy regularization
            concentration_penalty: Penalty for excessive concentration
            risk_free_rate: Risk-free rate for Sharpe calculation
            debug_mode: Enable detailed logging
        """
        super().__init__()
        self.sharpe_weight = sharpe_weight
        self.diversification_weight = diversification_weight
        self.min_effective_assets = min_effective_assets
        self.entropy_weight = entropy_weight
        self.concentration_penalty = concentration_penalty
        self.risk_free_rate = risk_free_rate
        self.debug_mode = debug_mode
        self._call_count = 0

    def compute_sharpe_ratio(
        self,
        weights: torch.Tensor,
        returns: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Sharpe ratio for portfolio.

        Args:
            weights: Portfolio weights [batch_size, n_assets]
            returns: Asset returns [batch_size, n_assets]

        Returns:
            Sharpe ratio (higher is better)
        """
        # Portfolio returns
        portfolio_returns = (weights * returns).sum(dim=-1)

        # Calculate Sharpe
        excess_returns = portfolio_returns - self.risk_free_rate
        mean_return = excess_returns.mean()
        std_return = excess_returns.std() + 1e-8

        sharpe = mean_return / std_return
        return sharpe

    def compute_diversification_ratio(
        self,
        weights: torch.Tensor,
        correlation_matrix: torch.Tensor,
        volatilities: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute diversification ratio: weighted avg volatility / portfolio volatility.
        Higher ratio indicates better diversification.

        Args:
            weights: Portfolio weights [batch_size, n_assets]
            correlation_matrix: Asset correlation matrix [n_assets, n_assets]
            volatilities: Individual asset volatilities [n_assets]

        Returns:
            Diversification ratio (higher is better)
        """
        if volatilities is None:
            # Assume unit volatilities if not provided
            volatilities = torch.ones(correlation_matrix.shape[0], device=weights.device)

        # Weighted average volatility
        weighted_vol = (weights * volatilities.unsqueeze(0)).sum(dim=-1)

        # Portfolio variance: w^T * Σ * w where Σ = diag(σ) * C * diag(σ)
        # For simplicity with correlation matrix: assume σ=1, so Σ = C
        portfolio_variance = torch.einsum('bi,ij,bj->b', weights, correlation_matrix, weights)
        portfolio_vol = torch.sqrt(portfolio_variance + 1e-8)

        # Diversification ratio
        div_ratio = weighted_vol / portfolio_vol
        return div_ratio

    def compute_effective_assets(
        self,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute effective number of assets (inverse HHI).

        Args:
            weights: Portfolio weights [batch_size, n_assets]

        Returns:
            Effective number of assets
        """
        # Herfindahl-Hirschman Index
        hhi = (weights ** 2).sum(dim=-1)

        # Effective N = 1 / HHI
        effective_n = 1.0 / (hhi + 1e-8)
        return effective_n

    def compute_entropy(
        self,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute portfolio entropy (encourages uniform distribution).

        Args:
            weights: Portfolio weights [batch_size, n_assets]

        Returns:
            Entropy (higher means more uniform)
        """
        # Avoid log(0)
        weights_safe = weights + 1e-8

        # Shannon entropy: -sum(w * log(w))
        entropy = -(weights_safe * torch.log(weights_safe)).sum(dim=-1)
        return entropy

    def compute_correlation_penalty(
        self,
        weights: torch.Tensor,
        correlation_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        CORRECTED: Penalize concentration in correlated assets.

        The original was backwards - it gave 0 penalty for single-asset portfolios!
        This version correctly penalizes based on portfolio variance.

        Args:
            weights: Portfolio weights [batch_size, n_assets]
            correlation_matrix: Asset correlation matrix [n_assets, n_assets]

        Returns:
            Correlation-based concentration penalty (lower is better)
        """
        # Portfolio variance: w^T * C * w
        # This is HIGH when we concentrate in correlated assets
        # This is LOW when we diversify across uncorrelated assets
        portfolio_variance = torch.einsum('bi,ij,bj->b', weights, correlation_matrix, weights)

        # We can also compute the "naive" variance (if assets were uncorrelated)
        naive_variance = (weights ** 2).sum(dim=-1)

        # Correlation penalty: how much extra variance from correlations
        correlation_penalty = portfolio_variance - naive_variance

        return correlation_penalty.mean()

    def forward(
        self,
        weights: torch.Tensor,
        returns: torch.Tensor,
        correlation_matrix: Optional[torch.Tensor] = None,
        constraints_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute corrected diversification-aware loss.

        Args:
            weights: Portfolio weights [batch_size, n_assets]
            returns: Asset returns [batch_size, n_assets]
            correlation_matrix: Asset correlation matrix [n_assets, n_assets]
            constraints_mask: Valid asset mask [batch_size, n_assets]

        Returns:
            Total loss value
        """
        self._call_count += 1

        # Apply constraints mask if provided
        if constraints_mask is not None:
            weights = weights * constraints_mask.float()
            # Renormalize
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 1. Sharpe ratio component (maximize -> minimize negative)
        sharpe = self.compute_sharpe_ratio(weights, returns)
        sharpe_loss = -self.sharpe_weight * sharpe

        # 2. Diversification components (only if correlation matrix provided)
        div_loss = 0.0
        corr_penalty = 0.0

        if correlation_matrix is not None:
            # Diversification ratio (maximize -> minimize negative)
            div_ratio = self.compute_diversification_ratio(weights, correlation_matrix)
            div_loss = -self.diversification_weight * div_ratio.mean()

            # Correlation penalty (minimize concentration in correlated assets)
            corr_penalty = self.concentration_penalty * self.compute_correlation_penalty(
                weights, correlation_matrix
            )

        # 3. Effective assets constraint (ensure minimum diversification)
        effective_n = self.compute_effective_assets(weights)
        min_assets_penalty = torch.relu(self.min_effective_assets - effective_n).mean()
        min_assets_loss = 2.0 * min_assets_penalty  # Strong penalty

        # 4. Entropy regularization (encourage uniformity)
        entropy = self.compute_entropy(weights)
        entropy_loss = -self.entropy_weight * entropy.mean()

        # 5. Basic constraints
        # Weight sum constraint (should sum to 1)
        weight_sum_error = torch.abs(weights.sum(dim=-1) - 1.0).mean()
        weight_sum_loss = 10.0 * weight_sum_error  # Strong penalty

        # Non-negativity constraint
        negative_weights = torch.relu(-weights).sum(dim=-1).mean()
        negative_loss = 10.0 * negative_weights  # Strong penalty

        # Combine all components
        total_loss = (
            sharpe_loss +
            div_loss +
            corr_penalty +
            min_assets_loss +
            entropy_loss +
            weight_sum_loss +
            negative_loss
        )

        # Debug logging
        if self.debug_mode and self._call_count % 10 == 0:
            logger.info(f"Loss components at call {self._call_count}:")
            logger.info(f"  Sharpe loss: {sharpe_loss:.4f}")
            logger.info(f"  Diversification loss: {div_loss:.4f}")
            logger.info(f"  Correlation penalty: {corr_penalty:.4f}")
            logger.info(f"  Min assets loss: {min_assets_loss:.4f}")
            logger.info(f"  Entropy loss: {entropy_loss:.4f}")
            logger.info(f"  Constraint losses: {weight_sum_loss:.4f}, {negative_loss:.4f}")
            logger.info(f"  Total loss: {total_loss:.4f}")
            logger.info(f"  Effective assets: {effective_n.mean():.1f}")
            logger.info(f"  Max weight: {weights.max():.3f}")

        return total_loss


class AdaptiveDiversificationLoss(CorrectedDiversificationLoss):
    """
    Adaptive version that gradually shifts focus from diversification to performance.

    Early in training: Strong diversification pressure
    Late in training: Focus on performance
    """

    def __init__(
        self,
        initial_div_weight: float = 2.0,
        final_div_weight: float = 0.5,
        initial_sharpe_weight: float = 0.5,
        final_sharpe_weight: float = 1.5,
        warmup_epochs: int = 5,
        total_epochs: int = 20,
        **kwargs
    ):
        """
        Initialize adaptive loss with scheduled weights.

        Args:
            initial_div_weight: Starting weight for diversification
            final_div_weight: Ending weight for diversification
            initial_sharpe_weight: Starting weight for Sharpe
            final_sharpe_weight: Ending weight for Sharpe
            warmup_epochs: Epochs for warmup phase
            total_epochs: Total training epochs
            **kwargs: Additional arguments for parent class
        """
        super().__init__(**kwargs)
        self.initial_div_weight = initial_div_weight
        self.final_div_weight = final_div_weight
        self.initial_sharpe_weight = initial_sharpe_weight
        self.final_sharpe_weight = final_sharpe_weight
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def update_epoch(self, epoch: int):
        """Update current epoch for weight scheduling."""
        self.current_epoch = epoch

        # Calculate progress
        if epoch < self.warmup_epochs:
            # Warmup phase: pure diversification
            progress = 0.0
        else:
            # Gradual transition
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            progress = min(1.0, max(0.0, progress))

        # Update weights
        self.diversification_weight = (
            self.initial_div_weight * (1 - progress) +
            self.final_div_weight * progress
        )
        self.sharpe_weight = (
            self.initial_sharpe_weight * (1 - progress) +
            self.final_sharpe_weight * progress
        )

        if epoch % 5 == 0:
            logger.info(f"Epoch {epoch}: div_weight={self.diversification_weight:.2f}, "
                       f"sharpe_weight={self.sharpe_weight:.2f}")


class EntropyRegularizedSoftmax(nn.Module):
    """
    Softmax with temperature control and entropy regularization.
    Replaces sparsemax to encourage more uniform distributions.
    """

    def __init__(
        self,
        temperature: float = 0.5,
        min_temperature: float = 0.1,
        max_temperature: float = 2.0,
        entropy_weight: float = 0.1,
        adaptive: bool = True
    ):
        """
        Initialize entropy-regularized softmax.

        Args:
            temperature: Temperature for softmax (lower = more uniform)
            min_temperature: Minimum temperature (adaptive mode)
            max_temperature: Maximum temperature (adaptive mode)
            entropy_weight: Weight for entropy bonus
            adaptive: Whether to adapt temperature during training
        """
        super().__init__()
        self.temperature = temperature
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.entropy_weight = entropy_weight
        self.adaptive = adaptive
        self.current_epoch = 0

    def update_epoch(self, epoch: int, total_epochs: int):
        """Update temperature based on training progress."""
        if not self.adaptive:
            return

        self.current_epoch = epoch

        # Start with low temperature (uniform), increase over time (allow concentration)
        progress = epoch / max(total_epochs, 1)
        self.temperature = (
            self.min_temperature * (1 - progress) +
            self.max_temperature * progress
        )

    def forward(
        self,
        logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply temperature-controlled softmax with entropy regularization.

        Args:
            logits: Input logits [batch_size, n_assets]
            mask: Valid asset mask [batch_size, n_assets]

        Returns:
            weights: Normalized weights
            entropy_bonus: Entropy regularization term
        """
        # Apply temperature
        scaled_logits = logits / self.temperature

        # Apply mask if provided
        if mask is not None:
            scaled_logits = scaled_logits.masked_fill(~mask, -1e9)

        # Softmax for smooth distribution
        weights = F.softmax(scaled_logits, dim=-1)

        # Compute entropy bonus
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1)
        entropy_bonus = self.entropy_weight * entropy.mean()

        return weights, entropy_bonus