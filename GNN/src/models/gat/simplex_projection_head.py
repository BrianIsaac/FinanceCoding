#!/usr/bin/env python3
"""
Simplex Projection Head for GAT Portfolio Model.

This module implements the missing component from the original proposal:
a learned transformation that converts GAT's relationship-aware representations
into portfolio weights on the probability simplex.

The key insight: Attention scores show relationships between assets,
but portfolio weights require a separate learned transformation that
considers these relationships to produce diversified allocations.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SimplexProjectionHead(nn.Module):
    """
    Transforms GAT node embeddings into portfolio weights on the simplex.

    This is the critical missing component from the original implementation.
    Instead of using attention scores directly as weights, this layer:
    1. Takes node representations from GAT (relationship-aware embeddings)
    2. Transforms them through learned layers (allocation decision)
    3. Projects onto the simplex (valid portfolio weights)

    Architecture based on the proposal's pipeline diagram:
    GAT with attention → Simplex projection long only → Top K sparsity
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        temperature: float = 1.0,
        use_batch_norm: bool = True,
        activation: str = "relu",
    ):
        """
        Initialize the simplex projection head.

        Args:
            input_dim: Dimension of GAT node embeddings
            hidden_dim: Hidden dimension for transformation layers
            num_layers: Number of transformation layers before projection
            dropout: Dropout rate for regularization
            temperature: Temperature for final softmax (lower = more concentrated)
            use_batch_norm: Whether to use batch normalization
            activation: Activation function ('relu', 'gelu', 'tanh')
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.temperature = temperature

        # Build transformation network
        layers = []
        current_dim = input_dim

        for i in range(num_layers - 1):
            # Linear transformation
            layers.append(nn.Linear(current_dim, hidden_dim))

            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")

            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            current_dim = hidden_dim

        # Final layer to allocation scores (no activation)
        # Output dimension should be 1 per asset for allocation score
        layers.append(nn.Linear(current_dim, 1))

        self.allocation_transform = nn.Sequential(*layers)

        # Learnable temperature parameter (optional)
        self.learnable_temperature = nn.Parameter(torch.ones(1) * temperature)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier/He initialization."""
        for module in self.allocation_transform.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_scores: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform node embeddings to portfolio weights.

        This is the key transformation that was missing:
        - Input: GAT node embeddings (relationship-aware representations)
        - Output: Portfolio weights (capital allocation decisions)

        Args:
            node_embeddings: Node representations from GAT [batch_size, n_assets, embedding_dim]
                            or [n_assets, embedding_dim]
            mask: Boolean mask for valid assets [batch_size, n_assets] or [n_assets]
            return_scores: Whether to return allocation scores before softmax

        Returns:
            portfolio_weights: Weights on the simplex [batch_size, n_assets] or [n_assets]
            allocation_scores: (Optional) Scores before projection
        """
        # Handle both batched and single sample inputs
        original_shape = node_embeddings.shape
        if node_embeddings.dim() == 2:
            # Single sample: [n_assets, embedding_dim]
            node_embeddings = node_embeddings.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, n_assets, embedding_dim = node_embeddings.shape

        # Reshape for transformation: [batch_size * n_assets, embedding_dim]
        embeddings_flat = node_embeddings.view(-1, embedding_dim)

        # Transform embeddings to allocation scores
        # This is the learned mapping from relationships to allocations
        allocation_scores_flat = self.allocation_transform(embeddings_flat)  # [batch_size * n_assets, 1]

        # Reshape back: [batch_size, n_assets]
        allocation_scores = allocation_scores_flat.view(batch_size, n_assets)

        # Apply temperature scaling
        # Lower temperature = more concentrated allocations
        # Higher temperature = more uniform allocations
        scaled_scores = allocation_scores / self.learnable_temperature

        # Apply mask if provided
        if mask is not None:
            # Set masked positions to very negative value
            scaled_scores = scaled_scores.masked_fill(~mask, -1e9)

        # Project onto simplex using softmax
        # This ensures: sum(weights) = 1 and all weights >= 0
        portfolio_weights = F.softmax(scaled_scores, dim=-1)

        # Handle numerical issues
        portfolio_weights = torch.where(
            torch.isnan(portfolio_weights),
            torch.zeros_like(portfolio_weights),
            portfolio_weights
        )

        # Renormalize if needed
        weight_sums = portfolio_weights.sum(dim=-1, keepdim=True)
        portfolio_weights = portfolio_weights / weight_sums.clamp(min=1e-8)

        # Squeeze if needed
        if squeeze_output:
            portfolio_weights = portfolio_weights.squeeze(0)
            allocation_scores = allocation_scores.squeeze(0)

        if return_scores:
            return portfolio_weights, allocation_scores
        return portfolio_weights


class RelationAwareAllocationHead(SimplexProjectionHead):
    """
    Enhanced version that explicitly considers relationships in allocation.

    This version takes both node embeddings and attention weights,
    using the attention to inform diversification decisions.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        attention_dim: int = 8,
        **kwargs
    ):
        """
        Initialize relation-aware allocation head.

        Args:
            input_dim: Dimension of GAT node embeddings
            hidden_dim: Hidden dimension for transformation layers
            num_layers: Number of transformation layers
            attention_dim: Expected number of attention heads from GAT
            **kwargs: Additional arguments for parent class
        """
        super().__init__(input_dim, hidden_dim, num_layers, **kwargs)

        self.attention_dim = attention_dim

        # Additional layer to process attention information
        self.attention_processor = nn.Sequential(
            nn.Linear(attention_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Combine embeddings with attention information
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Output single allocation score per asset
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_scores: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform embeddings to weights considering attention patterns.

        Args:
            node_embeddings: Node representations from GAT
            attention_weights: Attention weights from GAT [batch_size, n_heads, n_assets, n_assets]
            mask: Boolean mask for valid assets
            return_scores: Whether to return allocation scores

        Returns:
            portfolio_weights: Weights on the simplex
            allocation_scores: (Optional) Scores before projection
        """
        # Handle dimensions
        if node_embeddings.dim() == 2:
            node_embeddings = node_embeddings.unsqueeze(0)
            if attention_weights is not None:
                attention_weights = attention_weights.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, n_assets, embedding_dim = node_embeddings.shape

        # If attention weights provided, use them to inform allocation
        if attention_weights is not None:
            # Average attention across heads: [batch_size, n_assets, n_assets]
            avg_attention = attention_weights.mean(dim=1)

            # For each asset, get its average attention to others
            # This tells us how "connected" each asset is
            attention_summary = avg_attention.mean(dim=-1)  # [batch_size, n_assets]

            # Process attention information
            attention_features = self.attention_processor(
                attention_summary.unsqueeze(-1).expand(-1, -1, self.attention_dim)
            )

            # Transform embeddings
            embeddings_flat = node_embeddings.view(-1, embedding_dim)
            transformed_embeddings = self.allocation_transform(embeddings_flat)
            transformed_embeddings = transformed_embeddings.view(batch_size, n_assets, -1)

            # Combine with attention information
            # Key insight: High attention = high correlation = reduce allocation
            combined = torch.cat([transformed_embeddings, attention_features], dim=-1)
            allocation_scores = self.fusion_layer(combined).squeeze(-1)
        else:
            # Fallback to standard transformation
            embeddings_flat = node_embeddings.view(-1, embedding_dim)
            allocation_scores_flat = self.allocation_transform(embeddings_flat)
            allocation_scores = allocation_scores_flat.view(batch_size, n_assets)

        # Apply temperature and project to simplex
        scaled_scores = allocation_scores / self.learnable_temperature

        if mask is not None:
            scaled_scores = scaled_scores.masked_fill(~mask, -1e9)

        portfolio_weights = F.softmax(scaled_scores, dim=-1)

        # Ensure valid weights
        portfolio_weights = torch.where(
            torch.isnan(portfolio_weights),
            torch.zeros_like(portfolio_weights),
            portfolio_weights
        )

        weight_sums = portfolio_weights.sum(dim=-1, keepdim=True)
        portfolio_weights = portfolio_weights / weight_sums.clamp(min=1e-8)

        if squeeze_output:
            portfolio_weights = portfolio_weights.squeeze(0)
            allocation_scores = allocation_scores.squeeze(0)

        if return_scores:
            return portfolio_weights, allocation_scores
        return portfolio_weights


class DiversificationAwareProjectionHead(SimplexProjectionHead):
    """
    Projection head that explicitly encourages diversification.

    This version includes built-in mechanisms to prevent concentration
    in highly correlated assets identified by GAT attention.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        min_effective_assets: int = 15,
        diversification_strength: float = 1.0,
        **kwargs
    ):
        """
        Initialize diversification-aware projection head.

        Args:
            input_dim: Dimension of GAT node embeddings
            hidden_dim: Hidden dimension for transformation layers
            num_layers: Number of transformation layers
            min_effective_assets: Minimum number of effective assets (1/HHI)
            diversification_strength: How strongly to encourage diversification
            **kwargs: Additional arguments for parent class
        """
        super().__init__(input_dim, hidden_dim, num_layers, **kwargs)

        self.min_effective_assets = min_effective_assets
        self.diversification_strength = diversification_strength

        # Additional layer for diversification scores
        self.diversification_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output between 0 and 1
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        correlation_matrix: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_scores: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform embeddings to diversified portfolio weights.

        Args:
            node_embeddings: Node representations from GAT
            correlation_matrix: Correlation matrix between assets
            mask: Boolean mask for valid assets
            return_scores: Whether to return allocation scores

        Returns:
            portfolio_weights: Diversified weights on the simplex
            allocation_scores: (Optional) Scores before projection
        """
        # Handle dimensions
        if node_embeddings.dim() == 2:
            node_embeddings = node_embeddings.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, n_assets, embedding_dim = node_embeddings.shape

        # Get base allocation scores
        embeddings_flat = node_embeddings.view(-1, embedding_dim)
        allocation_scores_flat = self.allocation_transform(embeddings_flat)
        allocation_scores = allocation_scores_flat.view(batch_size, n_assets)

        # Compute diversification scores for each asset
        # Assets with high diversification scores are preferred
        div_scores_flat = self.diversification_layer(embeddings_flat)
        div_scores = div_scores_flat.view(batch_size, n_assets)

        # If correlation matrix provided, penalize correlated allocations
        if correlation_matrix is not None:
            # For each asset, compute its average correlation with others
            avg_correlation = correlation_matrix.abs().mean(dim=-1)

            # Reduce allocation scores for highly correlated assets
            # (1 - avg_correlation) gives higher weight to less correlated assets
            correlation_penalty = 1.0 - avg_correlation.unsqueeze(0)

            # Apply correlation-based adjustment
            allocation_scores = allocation_scores * (
                1.0 + self.diversification_strength * correlation_penalty
            )

        # Combine allocation scores with diversification preference
        final_scores = allocation_scores + self.diversification_strength * div_scores

        # Apply temperature and project to simplex
        scaled_scores = final_scores / self.learnable_temperature

        if mask is not None:
            scaled_scores = scaled_scores.masked_fill(~mask, -1e9)

        portfolio_weights = F.softmax(scaled_scores, dim=-1)

        # Post-process to ensure minimum diversification
        # If effective assets < min, apply entropy regularization
        effective_assets = 1.0 / (portfolio_weights ** 2).sum(dim=-1)

        if effective_assets.mean() < self.min_effective_assets:
            # Add small uniform component to increase diversification
            uniform_weights = torch.ones_like(portfolio_weights) / n_assets
            alpha = 0.1  # Blend factor
            portfolio_weights = (1 - alpha) * portfolio_weights + alpha * uniform_weights

        # Ensure valid weights
        weight_sums = portfolio_weights.sum(dim=-1, keepdim=True)
        portfolio_weights = portfolio_weights / weight_sums.clamp(min=1e-8)

        if squeeze_output:
            portfolio_weights = portfolio_weights.squeeze(0)
            allocation_scores = allocation_scores.squeeze(0)

        if return_scores:
            return portfolio_weights, allocation_scores
        return portfolio_weights