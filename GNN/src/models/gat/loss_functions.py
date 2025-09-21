#!/usr/bin/env python3
"""
Enhanced loss functions for GAT-based portfolio optimization.

This module provides sophisticated loss functions for direct portfolio optimization
including Sharpe ratio optimization, Markowitz mean-variance optimization,
and constraint-aware loss functions.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    pass  # type: ignore

__all__ = [
    "sharpe_loss",  # Keep original function for backward compatibility
    "SharpeRatioLoss",
    "CorrelationAwareSharpeRatioLoss",  # New correlation-aware loss for GAT
    "MarkownitzLayer",
    "ConstraintAwareLoss",
    "CombinedPortfolioLoss",
    "turnover_penalty_indexed",  # Keep original for compatibility
    "entropy_penalty",  # Keep original for compatibility
    "GradientFlowMonitor",  # New debugging utility
    "create_debug_loss_function",  # Factory for debugging losses
]


# Original functions for backward compatibility
def sharpe_loss(returns: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    -(mean / std) over a vector of period returns.
    returns: shape (..., T) or (T,). We aggregate over the last dim.
    """
    if returns.ndim == 0:
        returns = returns.unsqueeze(0)
    mu = returns.mean(dim=-1)
    sd = returns.std(dim=-1, unbiased=False).clamp_min(eps)
    return -(mu / sd).mean()


def turnover_penalty_indexed(
    weights: list[torch.Tensor],
    tickers_list: list[list[str]],
    tc_decimal: float,
) -> torch.Tensor:
    """
    L1 turnover between consecutive weight vectors whose names (tickers) may differ.
    """
    if len(weights) <= 1:
        dev = weights[0].device if weights else "cpu"
        return torch.tensor(0.0, device=dev)

    dev = weights[0].device
    total = torch.tensor(0.0, device=dev)

    for prev_w, curr_w, prev_names, curr_names in zip(
        weights[:-1], weights[1:], tickers_list[:-1], tickers_list[1:]
    ):
        if prev_w.numel() == 0 and curr_w.numel() == 0:
            continue

        prev_idx = {n: i for i, n in enumerate(prev_names)}
        curr_idx = {n: i for i, n in enumerate(curr_names)}

        acc = torch.tensor(0.0, device=dev)
        for n, i in curr_idx.items():
            w_curr = curr_w[i]
            if n in prev_idx:
                w_prev = prev_w[prev_idx[n]]
                acc = acc + (w_curr - w_prev).abs()
            else:
                acc = acc + w_curr.abs()

        for n, j in prev_idx.items():
            if n not in curr_idx:
                acc = acc + prev_w[j].abs()

        total = total + acc * tc_decimal

    return total


def entropy_penalty(weights: list[torch.Tensor], coef: float, eps: float = 1e-12) -> torch.Tensor:
    """
    Sum of w * log(w) across provided weight vectors (negative for simplex weights).
    """
    if coef <= 0.0 or not weights:
        dev = weights[0].device if weights else "cpu"
        return torch.tensor(0.0, device=dev)
    acc = torch.tensor(0.0, device=weights[0].device)
    for w in weights:
        w_ = w.clamp_min(eps)
        acc = acc + (w_ * w_.log()).sum()
    return coef * acc


# Enhanced portfolio optimization classes
class SharpeRatioLoss(nn.Module):
    """Enhanced Sharpe ratio loss function for direct portfolio optimization with debugging."""

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        constraint_penalty: float = 1.0,
        lookback_window: int = 252,
        debug_mode: bool = False,
        min_loss_threshold: float = 1e-6,
    ):
        """
        Initialize Sharpe ratio loss with enhanced debugging.

        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            constraint_penalty: Weight for constraint violation penalty
            lookback_window: Window for return statistics calculation
            debug_mode: Enable detailed debugging output
            min_loss_threshold: Minimum loss value to prevent zero gradients
        """
        super().__init__()
        self.risk_free_rate = risk_free_rate
        self.constraint_penalty = constraint_penalty
        self.lookback_window = lookback_window
        self.debug_mode = debug_mode
        self.min_loss_threshold = min_loss_threshold
        self._call_count = 0

    def forward(
        self,
        portfolio_weights: torch.Tensor,
        returns: torch.Tensor,
        constraints_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute negative Sharpe ratio as loss with comprehensive debugging.

        Args:
            portfolio_weights: Portfolio weights [batch_size, n_assets]
            returns: Asset returns [batch_size, n_assets] or [batch_size, time_steps, n_assets]
            constraints_mask: Mask for valid assets [batch_size, n_assets]

        Returns:
            Loss value (negative Sharpe ratio + constraint penalties)
        """
        self._call_count += 1

        # Enhanced input validation with debugging
        import logging
        logger = logging.getLogger(__name__)

        if self.debug_mode and self._call_count <= 5:
            logger.debug(f"SharpeRatioLoss Call #{self._call_count}:")
            logger.debug(f"  Weights shape: {portfolio_weights.shape}, sum: {portfolio_weights.sum():.6f}")
            logger.debug(f"  Returns shape: {returns.shape}, mean: {returns.mean():.6f}")
            logger.debug(f"  Weights grad: {portfolio_weights.requires_grad}")

        # Check for problematic input values
        if torch.isnan(portfolio_weights).any() or torch.isinf(portfolio_weights).any():
            if self.debug_mode:
                logger.warning("NaN/Inf detected in portfolio weights")
            portfolio_weights = torch.nan_to_num(portfolio_weights, nan=0.0, posinf=1.0, neginf=0.0)

        if torch.isnan(returns).any() or torch.isinf(returns).any():
            nan_count = torch.isnan(returns).sum().item()
            inf_count = torch.isinf(returns).sum().item()
            if self.debug_mode and self._call_count <= 10:  # Limit debug spam
                logger.warning(f"Data quality issue: {nan_count} NaN, {inf_count} Inf values in returns tensor")

            # More conservative replacement to avoid distorting loss function
            returns = torch.nan_to_num(returns, nan=0.0, posinf=0.05, neginf=-0.05)

            # If too many bad values, return a penalty loss
            total_values = returns.numel()
            bad_ratio = (nan_count + inf_count) / total_values
            if bad_ratio > 0.1:  # More than 10% bad data
                if self.debug_mode:
                    logger.warning(f"High bad data ratio: {bad_ratio:.2%}, applying penalty loss")
                # Use penalty that depends on weights to maintain gradients
                weight_penalty = torch.sum(portfolio_weights ** 2)  # L2 penalty
                concentration_penalty = torch.max(portfolio_weights) - torch.min(portfolio_weights)  # Concentration penalty
                return 1.0 + 0.1 * weight_penalty + 0.1 * concentration_penalty

        # Handle different return tensor shapes
        if returns.dim() == 3:  # [batch_size, time_steps, n_assets]
            lookback = min(self.lookback_window, returns.size(1))
            returns = returns[:, -lookback:, :]
            portfolio_returns = torch.sum(
                portfolio_weights.unsqueeze(1) * returns, dim=-1
            )  # [batch_size, time_steps]
        else:  # [batch_size, n_assets] - single period
            if constraints_mask is not None:
                portfolio_weights = portfolio_weights * constraints_mask.float()
                returns = returns * constraints_mask.float()
            portfolio_returns = torch.sum(portfolio_weights * returns, dim=-1)  # [batch_size]

        # Calculate excess returns with enhanced numerical stability
        excess_returns = portfolio_returns - self.risk_free_rate

        # Debug portfolio return statistics
        if self.debug_mode and self._call_count <= 5:
            logger.debug(f"  Portfolio returns: mean={portfolio_returns.mean():.6f}, std={portfolio_returns.std():.6f}")
            logger.debug(f"  Excess returns: mean={excess_returns.mean():.6f}, std={excess_returns.std():.6f}")

        # Enhanced Sharpe ratio calculation with robust error handling
        if excess_returns.dim() == 2:  # Time series case
            mean_excess = excess_returns.mean(dim=1)
            std_excess = excess_returns.std(dim=1, unbiased=False)

            # Improved handling of edge cases
            if excess_returns.shape[1] == 1:
                std_proxy = torch.abs(excess_returns.squeeze(1)) + 1e-4
                std_excess = torch.maximum(std_excess, std_proxy)
            else:
                std_excess = torch.clamp(std_excess, min=1e-4)

        else:  # Single period case
            mean_excess = excess_returns
            if excess_returns.numel() > 1:
                std_excess = torch.std(excess_returns) + 1e-4
            else:
                # Use a reasonable default volatility estimate
                std_excess = torch.tensor(0.01, device=excess_returns.device)

        # Robust Sharpe ratio calculation with gradient-friendly clamping
        eps = 1e-6
        mean_excess_stable = torch.clamp(mean_excess, min=-10.0, max=10.0)
        std_excess_stable = torch.clamp(std_excess, min=eps, max=10.0)

        sharpe_ratio = mean_excess_stable / std_excess_stable
        sharpe_ratio = torch.clamp(sharpe_ratio, min=-5.0, max=5.0)

        # Enhanced constraint penalties with proper gradient flow
        weight_sum_error = torch.abs(portfolio_weights.sum(dim=-1) - 1.0)
        weight_sum_penalty = torch.mean(weight_sum_error)

        negative_weights = torch.relu(-portfolio_weights)
        negative_weight_penalty = torch.mean(negative_weights)

        # Concentration penalty (Herfindahl-Hirschman Index)
        weight_squares = portfolio_weights ** 2
        concentration_penalty = torch.mean(torch.sum(weight_squares, dim=-1))

        # Debug constraint components
        if self.debug_mode and self._call_count <= 5:
            logger.debug(f"  Weight sum penalty: {weight_sum_penalty:.6f}")
            logger.debug(f"  Negative weight penalty: {negative_weight_penalty:.6f}")
            logger.debug(f"  Concentration penalty: {concentration_penalty:.6f}")

        # Compute loss components with enhanced stability
        sharpe_component = -sharpe_ratio.mean()
        constraint_component = (
            weight_sum_penalty +
            negative_weight_penalty +
            0.1 * concentration_penalty
        )

        # Scale constraint penalty appropriately
        scaled_constraint_penalty = self.constraint_penalty * constraint_component

        # Combine components
        total_loss = sharpe_component + scaled_constraint_penalty

        # Ensure minimum loss magnitude for gradient flow
        if torch.abs(total_loss) < self.min_loss_threshold:
            sign = torch.sign(total_loss) if total_loss != 0 else torch.tensor(1.0, device=total_loss.device)
            total_loss = sign * self.min_loss_threshold

        # Enhanced error handling and fallback
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            if self.debug_mode:
                logger.warning(f"Invalid loss detected: sharpe={sharpe_component:.6f}, constraints={scaled_constraint_penalty:.6f}")

            # Fallback loss that ensures gradient flow
            fallback_loss = (
                1.0 +  # Base loss
                10.0 * weight_sum_penalty +  # Strong weight sum constraint
                10.0 * negative_weight_penalty +  # Strong non-negativity constraint
                concentration_penalty  # Mild diversification constraint
            )
            total_loss = fallback_loss

        # Final debugging output
        if self.debug_mode and self._call_count <= 5:
            logger.debug(f"  Final loss: {total_loss:.6f} (sharpe: {sharpe_component:.6f}, constraints: {scaled_constraint_penalty:.6f})")
            logger.debug(f"  Loss requires_grad: {total_loss.requires_grad}")

        return total_loss


class CorrelationAwareSharpeRatioLoss(SharpeRatioLoss):
    """
    Enhanced Sharpe ratio loss with correlation penalty for GAT models.

    This loss function penalizes allocating to correlated assets, encouraging
    the GAT to diversify across uncorrelated clusters rather than concentrating
    within correlated groups.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        constraint_penalty: float = 1.0,
        correlation_penalty: float = 0.5,
        cluster_penalty: float = 0.3,
        lookback_window: int = 252,
        debug_mode: bool = False,
        min_loss_threshold: float = 1e-6,
    ):
        """
        Initialize correlation-aware Sharpe ratio loss.

        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            constraint_penalty: Weight for constraint violation penalty
            correlation_penalty: Weight for correlation-based allocation penalty
            cluster_penalty: Weight for within-cluster concentration penalty
            lookback_window: Window for return statistics calculation
            debug_mode: Enable detailed debugging output
            min_loss_threshold: Minimum loss value to prevent zero gradients
        """
        super().__init__(
            risk_free_rate=risk_free_rate,
            constraint_penalty=constraint_penalty,
            lookback_window=lookback_window,
            debug_mode=debug_mode,
            min_loss_threshold=min_loss_threshold
        )
        self.correlation_penalty = correlation_penalty
        self.cluster_penalty = cluster_penalty

    def compute_correlation_penalty(
        self,
        weights: torch.Tensor,
        correlation_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute penalty for allocating to correlated assets.

        Args:
            weights: Portfolio weights [batch_size, n_assets] or [n_assets]
            correlation_matrix: Asset correlation matrix [n_assets, n_assets]

        Returns:
            Correlation penalty scalar
        """
        if weights.dim() == 1:
            weights = weights.unsqueeze(0)

        # Compute weighted correlation (how much we're allocating to correlated pairs)
        # w^T @ C @ w gives the portfolio correlation
        batch_size = weights.shape[0]
        n_assets = weights.shape[1]

        # Ensure correlation matrix is the right size
        if correlation_matrix.shape[0] != n_assets:
            # If correlation matrix doesn't match, return zero penalty (data mismatch)
            return torch.tensor(0.0, device=weights.device)

        # Compute quadratic form: w^T @ C @ w for each batch
        weighted_corr = torch.zeros(batch_size, device=weights.device)
        for i in range(batch_size):
            w = weights[i]
            # Use absolute correlation to penalize both positive and negative correlation
            abs_corr = torch.abs(correlation_matrix)
            # Zero out diagonal (don't penalize self-correlation)
            abs_corr = abs_corr - torch.diag(torch.diag(abs_corr))
            # Compute weighted correlation
            weighted_corr[i] = w @ abs_corr @ w

        return weighted_corr.mean()

    def compute_cluster_penalty(
        self,
        weights: torch.Tensor,
        correlation_matrix: torch.Tensor,
        n_clusters: int = 10
    ) -> torch.Tensor:
        """
        Compute penalty for concentrating allocation within correlation clusters.

        Args:
            weights: Portfolio weights [batch_size, n_assets] or [n_assets]
            correlation_matrix: Asset correlation matrix [n_assets, n_assets]
            n_clusters: Number of clusters to identify

        Returns:
            Cluster concentration penalty
        """
        if weights.dim() == 1:
            weights = weights.unsqueeze(0)

        batch_size = weights.shape[0]
        n_assets = weights.shape[1]

        # Simple clustering based on correlation threshold
        high_corr_threshold = 0.7

        # Find highly correlated pairs
        high_corr_mask = torch.abs(correlation_matrix) > high_corr_threshold

        # For each asset, find its cluster (assets it's highly correlated with)
        cluster_penalties = []
        for b in range(batch_size):
            w = weights[b]
            max_cluster_weight = 0.0

            # Check each asset's cluster
            for i in range(n_assets):
                if w[i] > 0.01:  # Only consider assets with meaningful weight
                    # Find assets highly correlated with asset i
                    cluster_mask = high_corr_mask[i]
                    cluster_weight = (w * cluster_mask.float()).sum()
                    max_cluster_weight = max(max_cluster_weight, cluster_weight)

            cluster_penalties.append(max_cluster_weight)

        return torch.stack(cluster_penalties).mean()

    def forward(
        self,
        portfolio_weights: torch.Tensor,
        returns: torch.Tensor,
        constraints_mask: torch.Tensor | None = None,
        correlation_matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute correlation-aware Sharpe ratio loss.

        Args:
            portfolio_weights: Portfolio weights [batch_size, n_assets]
            returns: Asset returns [batch_size, n_assets] or [batch_size, time_steps, n_assets]
            constraints_mask: Mask for valid assets [batch_size, n_assets]
            correlation_matrix: Asset correlation matrix [n_assets, n_assets]

        Returns:
            Total loss including Sharpe ratio, constraints, and correlation penalties
        """
        # Get base Sharpe ratio loss
        base_loss = super().forward(portfolio_weights, returns, constraints_mask)

        # Add correlation penalties if correlation matrix is provided
        if correlation_matrix is not None and self.correlation_penalty > 0:
            corr_penalty = self.compute_correlation_penalty(portfolio_weights, correlation_matrix)
            cluster_penalty = self.compute_cluster_penalty(portfolio_weights, correlation_matrix)

            total_loss = (
                base_loss +
                self.correlation_penalty * corr_penalty +
                self.cluster_penalty * cluster_penalty
            )

            if self.debug_mode:
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"  Correlation penalty: {corr_penalty:.6f}")
                logger.debug(f"  Cluster penalty: {cluster_penalty:.6f}")
                logger.debug(f"  Total loss: {total_loss:.6f}")

            return total_loss

        return base_loss


class MarkownitzLayer(nn.Module):
    """Markowitz mean-variance optimization layer for risk-return balance."""

    def __init__(
        self,
        risk_aversion: float = 1.0,
        transaction_cost: float = 0.001,
        regularization: float = 1e-4,
    ):
        """
        Initialize Markowitz optimization layer.

        Args:
            risk_aversion: Risk aversion parameter (higher = more risk-averse)
            transaction_cost: Transaction cost per trade
            regularization: Regularization parameter for covariance matrix
        """
        super().__init__()
        self.risk_aversion = risk_aversion
        self.transaction_cost = transaction_cost
        self.regularization = regularization

    def forward(
        self,
        expected_returns: torch.Tensor,  # [batch_size, n_assets]
        covariance_matrix: torch.Tensor,  # [batch_size, n_assets, n_assets] or [n_assets, n_assets]
        constraints_mask: torch.Tensor | None = None,  # [batch_size, n_assets]
        previous_weights: torch.Tensor | None = None,  # [batch_size, n_assets]
    ) -> torch.Tensor:
        """
        Compute optimal portfolio weights using mean-variance optimization.

        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Asset return covariance matrix
            constraints_mask: Valid asset mask
            previous_weights: Previous portfolio weights for turnover calculation

        Returns:
            Optimal portfolio weights
        """
        batch_size, n_assets = expected_returns.shape

        # Ensure covariance matrix has the right shape
        if covariance_matrix.dim() == 2:
            covariance_matrix = covariance_matrix.unsqueeze(0).expand(batch_size, -1, -1)

        # Add regularization to covariance matrix for numerical stability
        reg_cov = covariance_matrix + self.regularization * torch.eye(
            n_assets, device=covariance_matrix.device
        ).unsqueeze(0)

        # Apply constraints mask to returns
        if constraints_mask is not None:
            expected_returns = expected_returns * constraints_mask.float()

        # Markowitz optimization: w* = (1/γ) * Σ^(-1) * μ
        # where γ is risk aversion, Σ is covariance, μ is expected returns
        try:
            # Compute precision matrix (inverse of covariance)
            precision_matrix = torch.linalg.pinv(reg_cov)

            # Optimal weights (before normalization)
            raw_weights = (
                torch.bmm(precision_matrix, expected_returns.unsqueeze(-1)).squeeze(-1)
                / self.risk_aversion
            )

            # Apply transaction costs if previous weights provided
            if previous_weights is not None:
                turnover = torch.abs(raw_weights - previous_weights)
                transaction_cost_penalty = self.transaction_cost * turnover
                raw_weights = raw_weights - transaction_cost_penalty

            # Apply constraints
            if constraints_mask is not None:
                raw_weights = raw_weights * constraints_mask.float()

            # Ensure non-negative weights (long-only constraint)
            raw_weights = torch.relu(raw_weights)

            # Normalize to sum to 1
            weight_sums = raw_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            optimal_weights = raw_weights / weight_sums

            return optimal_weights

        except torch.linalg.LinAlgError:
            # Fallback to equal weights if optimization fails
            if constraints_mask is not None:
                n_valid = constraints_mask.float().sum(dim=-1, keepdim=True)
                equal_weights = constraints_mask.float() / n_valid.clamp(min=1)
            else:
                equal_weights = torch.ones_like(expected_returns) / n_assets

            return equal_weights


class ConstraintAwareLoss(nn.Module):
    """Loss function that enforces portfolio constraints."""

    def __init__(
        self,
        weight_sum_penalty: float = 10.0,
        negative_weight_penalty: float = 10.0,
        concentration_penalty: float = 1.0,
        turnover_penalty: float = 0.1,
    ):
        """
        Initialize constraint-aware loss.

        Args:
            weight_sum_penalty: Penalty for weights not summing to 1
            negative_weight_penalty: Penalty for negative weights
            concentration_penalty: Penalty for concentrated portfolios
            turnover_penalty: Penalty for high turnover
        """
        super().__init__()
        self.weight_sum_penalty = weight_sum_penalty
        self.negative_weight_penalty = negative_weight_penalty
        self.concentration_penalty = concentration_penalty
        self.turnover_penalty = turnover_penalty

    def forward(
        self,
        portfolio_weights: torch.Tensor,
        previous_weights: torch.Tensor | None = None,
        max_weight: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute constraint violation penalties.

        Args:
            portfolio_weights: Current portfolio weights
            previous_weights: Previous weights for turnover calculation
            max_weight: Maximum weight per asset

        Returns:
            Total constraint penalty
        """
        penalty = torch.tensor(0.0, device=portfolio_weights.device)

        # Weight sum constraint (should sum to 1)
        weight_sum_error = torch.abs(portfolio_weights.sum(dim=-1) - 1.0).mean()
        penalty += self.weight_sum_penalty * weight_sum_error

        # Long-only constraint (no negative weights)
        negative_weights = torch.relu(-portfolio_weights).sum(dim=-1).mean()
        penalty += self.negative_weight_penalty * negative_weights

        # Concentration penalty (Herfindahl-Hirschman Index)
        hhi = (portfolio_weights**2).sum(dim=-1).mean()
        penalty += self.concentration_penalty * hhi

        # Maximum weight constraint
        max_weight_violations = torch.relu(portfolio_weights - max_weight).sum(dim=-1).mean()
        penalty += self.weight_sum_penalty * max_weight_violations

        # Turnover penalty
        if previous_weights is not None:
            turnover = torch.abs(portfolio_weights - previous_weights).sum(dim=-1).mean()
            penalty += self.turnover_penalty * turnover

        return penalty


class CombinedPortfolioLoss(nn.Module):
    """Combined loss function integrating multiple portfolio objectives."""

    def __init__(
        self,
        sharpe_weight: float = 1.0,
        constraint_weight: float = 1.0,
        markowitz_weight: float = 0.5,
        risk_aversion: float = 1.0,
    ):
        """
        Initialize combined portfolio loss.

        Args:
            sharpe_weight: Weight for Sharpe ratio component
            constraint_weight: Weight for constraint penalties
            markowitz_weight: Weight for Markowitz component
            risk_aversion: Risk aversion for Markowitz optimization
        """
        super().__init__()
        self.sharpe_weight = sharpe_weight
        self.constraint_weight = constraint_weight
        self.markowitz_weight = markowitz_weight

        self.sharpe_loss = SharpeRatioLoss()
        self.constraint_loss = ConstraintAwareLoss()
        self.markowitz_layer = MarkownitzLayer(risk_aversion=risk_aversion)

    def forward(
        self,
        portfolio_weights: torch.Tensor,
        returns: torch.Tensor,
        expected_returns: torch.Tensor,
        covariance_matrix: torch.Tensor,
        constraints_mask: torch.Tensor | None = None,
        previous_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined portfolio loss with multiple components.

        Args:
            portfolio_weights: Portfolio weights
            returns: Historical returns
            expected_returns: Expected returns
            covariance_matrix: Return covariance matrix
            constraints_mask: Valid asset mask
            previous_weights: Previous weights

        Returns:
            Dictionary with loss components and total loss
        """
        # Sharpe ratio component
        sharpe_component = self.sharpe_loss(portfolio_weights, returns, constraints_mask)

        # Constraint component
        constraint_component = self.constraint_loss(portfolio_weights, previous_weights)

        # Markowitz component (distance from optimal)
        if self.markowitz_weight > 0:
            optimal_weights = self.markowitz_layer(
                expected_returns, covariance_matrix, constraints_mask, previous_weights
            )
            markowitz_component = F.mse_loss(portfolio_weights, optimal_weights)
        else:
            markowitz_component = torch.tensor(0.0, device=portfolio_weights.device)

        # Combined loss
        total_loss = (
            self.sharpe_weight * sharpe_component
            + self.constraint_weight * constraint_component
            + self.markowitz_weight * markowitz_component
        )

        return {
            "total_loss": total_loss,
            "sharpe_component": sharpe_component,
            "constraint_component": constraint_component,
            "markowitz_component": markowitz_component,
        }


class GradientFlowMonitor:
    """Monitor gradient flow and loss dynamics during training."""

    def __init__(self, log_frequency: int = 10):
        """
        Initialize gradient flow monitor.

        Args:
            log_frequency: Log every N calls
        """
        self.log_frequency = log_frequency
        self.call_count = 0
        self.gradient_stats = []
        self.loss_history = []

    def check_gradient_flow(self, model: nn.Module, loss: torch.Tensor) -> dict[str, float]:
        """
        Analyse gradient flow through model parameters.

        Args:
            model: PyTorch model to analyse
            loss: Current loss value

        Returns:
            Dictionary with gradient statistics
        """
        self.call_count += 1

        # Store loss
        loss_value = loss.item() if isinstance(loss, torch.Tensor) else loss
        self.loss_history.append(loss_value)

        # Collect gradient statistics
        grad_stats = {
            "total_grad_norm": 0.0,
            "max_grad": 0.0,
            "min_grad": 0.0,
            "zero_grad_params": 0,
            "total_params": 0,
            "loss_value": loss_value,
        }

        total_norm = 0.0
        max_grad = float('-inf')
        min_grad = float('inf')
        zero_count = 0
        total_count = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                # Calculate parameter-wise gradient norm
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2

                # Track min/max gradients
                param_max = param.grad.data.max().item()
                param_min = param.grad.data.min().item()
                max_grad = max(max_grad, param_max)
                min_grad = min(min_grad, param_min)

                # Count zero gradients
                zero_mask = (torch.abs(param.grad.data) < 1e-8)
                zero_count += zero_mask.sum().item()
                total_count += param.grad.data.numel()

            else:
                # Parameter has no gradient
                zero_count += param.data.numel()
                total_count += param.data.numel()

        grad_stats["total_grad_norm"] = (total_norm ** 0.5) if total_norm > 0 else 0.0
        grad_stats["max_grad"] = max_grad if max_grad != float('-inf') else 0.0
        grad_stats["min_grad"] = min_grad if min_grad != float('inf') else 0.0
        grad_stats["zero_grad_params"] = zero_count
        grad_stats["total_params"] = total_count
        grad_stats["zero_grad_fraction"] = zero_count / max(total_count, 1)

        self.gradient_stats.append(grad_stats)

        # Log periodically
        if self.call_count % self.log_frequency == 0:
            self._log_gradient_summary()

        return grad_stats

    def _log_gradient_summary(self) -> None:
        """Log summary of recent gradient statistics."""
        import logging
        logger = logging.getLogger(__name__)

        if not self.gradient_stats:
            return

        recent_stats = self.gradient_stats[-self.log_frequency:]

        avg_loss = sum(s["loss_value"] for s in recent_stats) / len(recent_stats)
        avg_grad_norm = sum(s["total_grad_norm"] for s in recent_stats) / len(recent_stats)
        avg_zero_fraction = sum(s["zero_grad_fraction"] for s in recent_stats) / len(recent_stats)

        logger.info(f"Gradient Flow Summary (last {len(recent_stats)} steps):")
        logger.info(f"  Average Loss: {avg_loss:.6f}")
        logger.info(f"  Average Gradient Norm: {avg_grad_norm:.6f}")
        logger.info(f"  Average Zero Gradient Fraction: {avg_zero_fraction:.3f}")

        # Check for potential issues
        if avg_grad_norm < 1e-6:
            logger.warning("Gradient norm is very small - potential vanishing gradients")
        elif avg_grad_norm > 100:
            logger.warning("Gradient norm is very large - potential exploding gradients")

        if avg_zero_fraction > 0.9:
            logger.warning("High fraction of zero gradients - potential dead neurons")

        if avg_loss < 1e-6:
            logger.warning("Loss is very small - potential loss computation issues")

    def get_gradient_summary(self) -> dict[str, any]:
        """Get comprehensive gradient flow summary."""
        if not self.gradient_stats:
            return {}

        # Calculate summary statistics
        losses = [s["loss_value"] for s in self.gradient_stats]
        grad_norms = [s["total_grad_norm"] for s in self.gradient_stats]
        zero_fractions = [s["zero_grad_fraction"] for s in self.gradient_stats]

        summary = {
            "total_steps": len(self.gradient_stats),
            "loss_stats": {
                "mean": sum(losses) / len(losses),
                "min": min(losses),
                "max": max(losses),
                "std": (sum((l - sum(losses)/len(losses))**2 for l in losses) / len(losses))**0.5,
            },
            "gradient_norm_stats": {
                "mean": sum(grad_norms) / len(grad_norms),
                "min": min(grad_norms),
                "max": max(grad_norms),
                "std": (sum((g - sum(grad_norms)/len(grad_norms))**2 for g in grad_norms) / len(grad_norms))**0.5,
            },
            "zero_gradient_stats": {
                "mean_fraction": sum(zero_fractions) / len(zero_fractions),
                "min_fraction": min(zero_fractions),
                "max_fraction": max(zero_fractions),
            },
        }

        return summary


def create_debug_loss_function(
    loss_type: str = "sharpe",
    debug_mode: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss functions with debugging enabled.

    Args:
        loss_type: Type of loss function ("sharpe", "combined", "constraint")
        debug_mode: Enable debugging output
        **kwargs: Additional arguments for loss function

    Returns:
        Configured loss function with debugging
    """
    if loss_type == "sharpe":
        return SharpeRatioLoss(debug_mode=debug_mode, **kwargs)
    elif loss_type == "combined":
        return CombinedPortfolioLoss(**kwargs)
    elif loss_type == "constraint":
        return ConstraintAwareLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def diagnose_loss_issues(
    model: nn.Module,
    loss_fn: nn.Module,
    sample_input: tuple[torch.Tensor, ...],
    num_steps: int = 5
) -> dict[str, any]:
    """
    Diagnose potential issues with loss function and gradient flow.

    Args:
        model: Model to diagnose
        loss_fn: Loss function to test
        sample_input: Sample input tuple (weights, returns, etc.)
        num_steps: Number of forward/backward steps to test

    Returns:
        Diagnostic report
    """
    import logging
    logger = logging.getLogger(__name__)

    monitor = GradientFlowMonitor(log_frequency=1)
    report = {
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "loss_values": [],
        "gradient_issues": [],
    }

    logger.info(f"Diagnosing loss function over {num_steps} steps...")

    for step in range(num_steps):
        # Forward pass
        model.zero_grad()

        # Compute loss
        loss = loss_fn(*sample_input)
        loss_value = loss.item()
        report["loss_values"].append(loss_value)

        # Backward pass
        try:
            loss.backward()
        except Exception as e:
            report["gradient_issues"].append(f"Step {step}: Backward pass failed - {e}")
            continue

        # Check gradients
        grad_stats = monitor.check_gradient_flow(model, loss)

        # Check for issues
        if loss_value == 0.0:
            report["gradient_issues"].append(f"Step {step}: Zero loss detected")
        if torch.isnan(loss) or torch.isinf(loss):
            report["gradient_issues"].append(f"Step {step}: NaN/Inf loss detected")
        if grad_stats["total_grad_norm"] < 1e-8:
            report["gradient_issues"].append(f"Step {step}: Vanishing gradients (norm={grad_stats['total_grad_norm']:.2e})")
        if grad_stats["zero_grad_fraction"] > 0.95:
            report["gradient_issues"].append(f"Step {step}: High zero gradient fraction ({grad_stats['zero_grad_fraction']:.2f})")

    # Add summary statistics
    report["gradient_summary"] = monitor.get_gradient_summary()

    logger.info(f"Diagnosis complete. Found {len(report['gradient_issues'])} potential issues.")

    return report
