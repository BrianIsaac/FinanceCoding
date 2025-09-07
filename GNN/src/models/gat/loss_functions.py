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
    "MarkownitzLayer",
    "ConstraintAwareLoss",
    "CombinedPortfolioLoss",
    "turnover_penalty_indexed",  # Keep original for compatibility
    "entropy_penalty",  # Keep original for compatibility
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
    """Enhanced Sharpe ratio loss function for direct portfolio optimization."""

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        constraint_penalty: float = 1.0,
        lookback_window: int = 252,
    ):
        """
        Initialize Sharpe ratio loss.

        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            constraint_penalty: Weight for constraint violation penalty
            lookback_window: Window for return statistics calculation
        """
        super().__init__()
        self.risk_free_rate = risk_free_rate
        self.constraint_penalty = constraint_penalty
        self.lookback_window = lookback_window

    def forward(
        self,
        portfolio_weights: torch.Tensor,
        returns: torch.Tensor,
        constraints_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute negative Sharpe ratio as loss for direct optimization.

        Args:
            portfolio_weights: Portfolio weights [batch_size, n_assets]
            returns: Asset returns [batch_size, n_assets] or [batch_size, time_steps, n_assets]
            constraints_mask: Mask for valid assets [batch_size, n_assets]

        Returns:
            Loss value (negative Sharpe ratio + constraint penalties)
        """
        # Handle different return tensor shapes
        if returns.dim() == 3:  # [batch_size, time_steps, n_assets]
            # Use the most recent time steps up to lookback window
            lookback = min(self.lookback_window, returns.size(1))
            returns = returns[:, -lookback:, :]

            # Calculate portfolio returns over time
            portfolio_returns = torch.sum(
                portfolio_weights.unsqueeze(1) * returns, dim=-1
            )  # [batch_size, time_steps]
        else:  # [batch_size, n_assets] - single period
            # Apply constraints mask if provided
            if constraints_mask is not None:
                portfolio_weights = portfolio_weights * constraints_mask.float()
                returns = returns * constraints_mask.float()

            # Calculate portfolio returns
            portfolio_returns = torch.sum(portfolio_weights * returns, dim=-1)  # [batch_size]

        # Calculate excess returns
        excess_returns = portfolio_returns - self.risk_free_rate

        # Sharpe ratio calculation with numerical stability
        if excess_returns.dim() == 2:  # Time series case
            mean_excess = excess_returns.mean(dim=1)  # Mean over time
            std_excess = excess_returns.std(dim=1, unbiased=False) + 1e-8
            # Handle case where we only have one time step
            if excess_returns.shape[1] == 1:
                std_excess = torch.ones_like(mean_excess) * 1e-6  # Fallback small std
        else:  # Single period case
            mean_excess = excess_returns.mean()
            # For single period, we can't compute meaningful Sharpe, use normalized approach
            std_excess = torch.tensor(1.0)  # Normalize to avoid huge ratios

        sharpe_ratio = mean_excess / std_excess

        # Constraint penalties
        weight_sum_penalty = torch.mean(torch.abs(portfolio_weights.sum(dim=-1) - 1.0))
        negative_weight_penalty = torch.mean(torch.relu(-portfolio_weights))

        total_loss = -sharpe_ratio.mean() + self.constraint_penalty * (
            weight_sum_penalty + negative_weight_penalty
        )

        return total_loss


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
