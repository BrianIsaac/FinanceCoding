#!/usr/bin/env python3
"""
GAT-based portfolio model.

Key features added/kept vs. your previous version:
- Clean GAT/GATv2 backbone with residual connections, dropout, and LayerNorm.
- Optional use of edge attributes (expects 1-d edge weight; otherwise we average to 1-d).
- Lightweight temporal memory via a GRU cell (node-wise), controlled from cfg.temporal.*
- Two heads:
    * "direct": produces a valid weight vector via masked softmax/sparsemax (sum=1, nonnegative).
    * "markowitz": produces per-asset expected-return scores µ̂; allocation done by Markowitz layer outside.
- Mask-aware activations so invalid nodes never receive weight/score.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

try:
    from torch_geometric.nn import GATConv, GATv2Conv  # type: ignore
except Exception:  # pragma: no cover
    GATConv = GATv2Conv = None  # type: ignore


# ---------------------------- utils ----------------------------


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Numerically stable softmax over only True entries of mask (boolean). Zeros elsewhere.
    """
    # Clamp input logits to prevent extreme values
    logits = torch.clamp(logits, min=-50.0, max=50.0)

    # Use a more conservative negative value
    very_neg = -100.0  # Stable value instead of dtype min
    z = torch.where(mask, logits, torch.full_like(logits, very_neg))

    # Numerically stable softmax with max subtraction
    z_max = z.max(dim=dim, keepdim=True)[0]
    z_stable = z - z_max

    # Apply softmax with numerical stability
    exp_z = torch.exp(torch.clamp(z_stable, min=-50.0, max=50.0))
    exp_z = torch.where(mask, exp_z, torch.zeros_like(exp_z))

    # Normalize with numerical stability
    s = exp_z.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    p = exp_z / s

    # Final validation - ensure no NaN or Inf values
    p = torch.where(torch.isfinite(p), p, torch.zeros_like(p))

    return p


def sparsemax(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Numerically stable sparsemax implementation.
    """
    # Clamp input tensor to prevent extreme values
    tensor = torch.clamp(tensor, min=-100.0, max=100.0)

    tensor = tensor.transpose(dim, -1)
    input_flat = tensor.reshape(-1, tensor.size(-1))

    # Numerically stable sorting and computation
    zs = torch.sort(input_flat, descending=True, dim=-1).values
    zs = torch.clamp(zs, min=-100.0, max=100.0)  # Additional stability

    range_ = torch.arange(1, zs.size(-1) + 1, device=tensor.device, dtype=tensor.dtype).view(1, -1)
    cssv = zs.cumsum(dim=-1) - 1

    # Prevent division by zero and extreme values
    range_safe = range_.clamp(min=1e-12)
    cond = zs - cssv / range_safe > 0
    k = cond.sum(dim=-1).clamp(min=1)

    # Numerically stable tau computation
    tau = cssv.gather(1, (k - 1).unsqueeze(1)).squeeze(1) / k.clamp(min=1e-12)
    tau = torch.clamp(tau, min=-100.0, max=100.0)

    output_flat = torch.clamp(input_flat - tau.unsqueeze(1), min=0.0, max=1.0)
    output = output_flat.view_as(tensor)
    output = output.transpose(dim, -1)

    # Ensure proper normalization with numerical stability
    s = output.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    output = output / s

    # Final validation - ensure no NaN or Inf values
    output = torch.where(torch.isfinite(output), output, torch.zeros_like(output))

    return output


def masked_sparsemax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Numerically stable sparsemax over only True entries of mask (boolean). Zeros elsewhere.
    """
    # Clamp input logits to prevent extreme values
    logits = torch.clamp(logits, min=-50.0, max=50.0)

    # Use a more conservative negative value
    very_neg = -100.0
    z = torch.where(mask, logits, torch.full_like(logits, very_neg))

    # Apply sparsemax (now numerically stable)
    p = sparsemax(z, dim=dim)

    # Ensure zeros where mask is False
    p = torch.where(mask, p, torch.zeros_like(p))

    # Renormalize to ensure sum = 1 for valid entries
    s = p.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    output = p / s

    # Final validation
    return torch.where(torch.isfinite(output), output, torch.zeros_like(output))


# ---------------------------- portfolio-specific layers ----------------------------


class PortfolioConstraintLayer(nn.Module):
    """Portfolio constraint enforcement layer for GAT models."""

    def __init__(self, temperature: float = 1.0, min_weight: float = 1e-6):
        """
        Initialize portfolio constraint layer.

        Args:
            temperature: Temperature parameter for softmax/sparsemax
            min_weight: Minimum weight threshold
        """
        super().__init__()
        self.temperature = temperature
        self.min_weight = min_weight

    def forward(
        self, logits: torch.Tensor, mask: torch.Tensor, activation: str = "sparsemax"
    ) -> torch.Tensor:
        """
        Apply portfolio constraints to raw logits.

        Args:
            logits: Raw portfolio weight logits [batch_size, n_assets]
            mask: Valid asset mask [batch_size, n_assets]
            activation: Constraint activation ("softmax" or "sparsemax")

        Returns:
            Constrained portfolio weights
        """
        # Clamp input logits to prevent extreme values before processing
        logits = torch.clamp(logits, min=-50.0, max=50.0)

        # Scale by temperature with bounds checking
        temperature = max(self.temperature, 1e-6)  # Prevent division by zero
        scaled_logits = logits / temperature
        scaled_logits = torch.clamp(scaled_logits, min=-50.0, max=50.0)

        # Apply constraint-aware activation
        if activation.lower() == "sparsemax":
            weights = masked_sparsemax(scaled_logits, mask.bool(), dim=-1)
        else:
            weights = masked_softmax(scaled_logits, mask.bool(), dim=-1)

        # Validate weights before threshold application
        weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))

        # Apply minimum weight threshold
        weights = torch.where(weights < self.min_weight, torch.zeros_like(weights), weights)

        # Renormalize to ensure sum = 1 with numerical stability
        weight_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        weights = weights / weight_sum

        # Final validation and clamping
        weights = torch.clamp(weights, min=0.0, max=1.0)
        weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))

        # Ensure weights sum to approximately 1.0
        final_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        weights = weights / final_sum

        return weights


class PortfolioRegularization(nn.Module):
    """Portfolio-specific regularization terms."""

    def __init__(
        self,
        concentration_penalty: float = 0.1,
        turnover_penalty: float = 0.05,
        diversification_reward: float = 0.02,
    ):
        """
        Initialize portfolio regularization.

        Args:
            concentration_penalty: Penalty for concentrated portfolios
            turnover_penalty: Penalty for high turnover
            diversification_reward: Reward for diversified portfolios
        """
        super().__init__()
        self.concentration_penalty = concentration_penalty
        self.turnover_penalty = turnover_penalty
        self.diversification_reward = diversification_reward

    def forward(
        self, weights: torch.Tensor, prev_weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute portfolio regularization loss.

        Args:
            weights: Current portfolio weights [batch_size, n_assets]
            prev_weights: Previous portfolio weights for turnover calculation

        Returns:
            Regularization loss
        """
        reg_loss = torch.tensor(0.0, device=weights.device)

        # Concentration penalty (negative entropy)
        if self.concentration_penalty > 0:
            weights_safe = weights.clamp(min=1e-12)
            entropy = -(weights_safe * torch.log(weights_safe)).sum(dim=-1).mean()
            reg_loss += self.concentration_penalty * (-entropy)

        # Turnover penalty
        if self.turnover_penalty > 0 and prev_weights is not None:
            turnover = torch.abs(weights - prev_weights).sum(dim=-1).mean()
            reg_loss += self.turnover_penalty * turnover

        # Diversification reward (Herfindahl-Hirschman Index penalty)
        if self.diversification_reward > 0:
            hhi = (weights**2).sum(dim=-1).mean()
            reg_loss += self.diversification_reward * hhi

        return reg_loss


# ---------------------------- blocks ----------------------------


class GATBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.2,
        use_gatv2: bool = True,
        use_edge_attr: bool = True,
        edge_dim: int = 1,
        residual: bool = True,
    ) -> None:
        super().__init__()
        Conv = GATv2Conv if (use_gatv2 and GATv2Conv is not None) else GATConv
        if Conv is None:
            raise RuntimeError("torch_geometric is required for GAT/GATv2.")
        self.use_edge_attr = use_edge_attr
        self.edge_dim = edge_dim if use_edge_attr else None  # type: ignore

        self.conv = Conv(
            in_dim,
            out_dim,
            heads=heads,
            dropout=dropout,
            add_self_loops=True,
            edge_dim=(edge_dim if use_edge_attr else None),
            concat=True,
            bias=True,
        )
        self.proj = nn.Linear(out_dim * heads, out_dim, bias=True)
        self.ln = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual and (in_dim == out_dim)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None
    ) -> torch.Tensor:
        if self.use_edge_attr and edge_attr is not None:
            # Enhanced edge attribute processing for portfolio applications
            if edge_attr.dim() == 1:
                ea = edge_attr.view(-1, 1)
            elif self.edge_dim and edge_attr.size(-1) == self.edge_dim:
                # Use all edge features if they match expected dimension
                ea = edge_attr
            elif edge_attr.size(-1) > self.edge_dim:
                # For 3D edge attributes [rho, |rho|, sign], use all features
                if self.edge_dim == 3 and edge_attr.size(-1) == 3:
                    ea = edge_attr
                else:
                    # Compress via weighted mean (give more weight to correlation strength)
                    if edge_attr.size(-1) >= 2:
                        weights = torch.tensor([0.5, 0.3, 0.2], device=edge_attr.device)[
                            : edge_attr.size(-1)
                        ]
                        weights = weights / weights.sum()
                        ea = (edge_attr * weights).sum(dim=-1, keepdim=True)
                    else:
                        ea = edge_attr.mean(dim=-1, keepdim=True)
            else:
                ea = edge_attr
            h = self.conv(x, edge_index, ea)
        else:
            h = self.conv(x, edge_index)

        h = self.proj(h)

        if self.residual:
            h = h + x

        h = F.gelu(h)  # Using GELU for better gradient flow
        h = self.ln(h)
        h = self.dropout(h)
        return h


# ---------------------------- model ----------------------------


@dataclass
class HeadCfg:
    mode: str = "markowitz"  # "markowitz" | "direct"
    activation: str = "sparsemax"  # used only if mode == "direct": "softmax" | "sparsemax"


class GATPortfolio(nn.Module):
    """
    CORRECTED GAT model for portfolio optimization with proper two-stage architecture.

    Key fix: Implements the missing simplex projection head from the proposal.
    GAT layers create relationship-aware embeddings, then a separate projection
    head transforms these to portfolio weights.

    Returns:
        if head == "direct":  (weights[N], new_mem[N, mem_dim], regularization_loss)
        if head == "markowitz": (mu_hat[N], new_mem[N, mem_dim], regularization_loss)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        residual: bool = True,
        use_gatv2: bool = True,
        use_edge_attr: bool = True,
        head: str = "markowitz",
        activation: str = "sparsemax",
        mem_hidden: int | None = None,
        constraint_aware: bool = True,
        portfolio_temperature: float = 1.0,
        graph_type: str = "default",  # For selecting appropriate projection head
    ) -> None:
        super().__init__()
        self.graph_type = graph_type
        self.use_edge_attr = use_edge_attr
        self.head_mode = head
        self.head_activation = activation if head == "direct" else "none"
        self.constraint_aware = constraint_aware
        self.portfolio_temperature = portfolio_temperature

        # Backbone with enhanced architecture
        layers = []
        d_in = in_dim
        for li in range(num_layers):
            # Progressive hidden dimension reduction for better feature compression
            layer_hidden = hidden_dim if li < num_layers - 1 else max(hidden_dim // 2, 32)

            block = GATBlock(
                in_dim=d_in,
                out_dim=layer_hidden,
                heads=heads,
                dropout=dropout,
                use_gatv2=use_gatv2,
                use_edge_attr=use_edge_attr,
                edge_dim=3,  # Enhanced for [rho, |rho|, sign] edge attributes
                residual=residual,
            )
            layers.append(block)
            d_in = layer_hidden
        self.gnn = nn.ModuleList(layers)

        self.head_in_dim = d_in  # Updated to use final layer dimension
        self.heads = heads

        # Enhanced readout with portfolio-specific layers
        if constraint_aware and head == "direct":
            # Portfolio-aware readout for direct weight prediction
            self.readout = nn.Sequential(
                nn.Linear(self.head_in_dim, self.head_in_dim * 2),
                nn.BatchNorm1d(self.head_in_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.head_in_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),  # Reduced dropout for final layer
                nn.Linear(hidden_dim, 1),
            )
        else:
            # Standard readout for Markowitz head
            self.readout = nn.Sequential(
                nn.Linear(self.head_in_dim, self.head_in_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.head_in_dim, 1),
            )

        # Portfolio constraint layer for direct weight prediction
        if constraint_aware and head == "direct":
            self.constraint_layer = PortfolioConstraintLayer(temperature=portfolio_temperature)
        else:
            self.constraint_layer = None

        # Node-wise temporal memory (if enabled)
        self._mem_dim = int(mem_hidden) if mem_hidden is not None else 0
        if self._mem_dim > 0:
            self.mem_gru = nn.GRUCell(self.head_in_dim, self._mem_dim)
            # fuse memory back into features before head
            self.mem_fuse = nn.Linear(self.head_in_dim + self._mem_dim, self.head_in_dim)
        else:
            self.mem_gru = None
            self.mem_fuse = None

        # Portfolio-specific regularization
        self.portfolio_reg = PortfolioRegularization() if constraint_aware else None

        # CRITICAL FIX: Add proper simplex projection head
        # This is the missing component from the proposal!
        if head == "direct":
            # Import at top of module if not already there
            try:
                from .simplex_projection_head import (
                    SimplexProjectionHead,
                    RelationAwareAllocationHead,
                    DiversificationAwareProjectionHead
                )
            except ImportError:
                logger.warning("Simplex projection heads not found, using fallback")
                SimplexProjectionHead = None

            if SimplexProjectionHead is not None:
                # Select projection head based on graph type
                if graph_type.lower() in ['tmfg', 'knn']:
                    # These graphs tend to concentrate - use stronger diversification
                    self.simplex_projection_head = DiversificationAwareProjectionHead(
                        input_dim=self.head_in_dim,
                        hidden_dim=hidden_dim,
                        num_layers=3,
                        min_effective_assets=20,
                        diversification_strength=2.0,
                        dropout=dropout,
                        temperature=portfolio_temperature,
                    )
                    logger.info(f"Using DiversificationAwareProjectionHead for {graph_type}")
                elif graph_type.lower() == 'mst':
                    # MST benefits from relation-aware allocation
                    self.simplex_projection_head = RelationAwareAllocationHead(
                        input_dim=self.head_in_dim,
                        hidden_dim=hidden_dim,
                        num_layers=3,
                        attention_dim=heads,
                        dropout=dropout,
                        temperature=portfolio_temperature,
                    )
                    logger.info(f"Using RelationAwareAllocationHead for {graph_type}")
                else:
                    # Default simplex projection
                    self.simplex_projection_head = SimplexProjectionHead(
                        input_dim=self.head_in_dim,
                        hidden_dim=hidden_dim,
                        num_layers=2,
                        dropout=dropout,
                        temperature=portfolio_temperature,
                    )
                    logger.info("Using standard SimplexProjectionHead")
            else:
                self.simplex_projection_head = None
                logger.warning("Falling back to direct attention weights (incorrect but compatible)")
        else:
            self.simplex_projection_head = None

        self.reset_parameters()

    # ----------------- properties -----------------
    @property
    def mem_dim(self) -> int:
        return self._mem_dim

    # ----------------- nn API -----------------
    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Kaiming initialization for better gradient flow with GELU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)  # Small positive bias to prevent dead neurons

    def forward(
        self,
        x: torch.Tensor,  # [N, F]
        edge_index: torch.Tensor,  # [2, E]
        mask_valid: torch.Tensor,  # [N] bool
        edge_attr: torch.Tensor | None = None,  # [E, d_e] optional
        prev_mem: torch.Tensor | None = None,  # [N, mem_dim] optional
        prev_weights: torch.Tensor | None = None,  # [N] for regularization
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # Backbone
        h = x
        for block in self.gnn:
            h = block(h, edge_index, edge_attr if self.use_edge_attr else None)

        # Memory update (node-wise)
        if self._mem_dim > 0 and self.mem_gru is not None:
            if prev_mem is None:
                prev_mem = torch.zeros(h.size(0), self._mem_dim, device=h.device, dtype=h.dtype)
            m_new = self.mem_gru(h, prev_mem)
            h = torch.cat([h, m_new], dim=-1)
            h = F.gelu(self.mem_fuse(h))
        else:
            m_new = torch.zeros(h.size(0), 0, device=h.device, dtype=h.dtype)

        # Head with enhanced processing
        if h.dim() == 2 and h.size(0) > 1:
            # Apply batch normalization if we have multiple samples
            h = F.layer_norm(h, h.shape[-1:])

        scores = self.readout(h).squeeze(-1)  # [N]
        reg_loss = None

        if self.head_mode == "direct":
            # CRITICAL FIX: Use simplex projection head instead of direct attention weights
            if hasattr(self, 'simplex_projection_head') and self.simplex_projection_head is not None:
                # This is the correct two-stage process:
                # 1. GAT creates node embeddings (h)
                # 2. Simplex projection transforms embeddings to weights

                # Get correlation matrix if using DiversificationAwareProjectionHead
                correlation_matrix = None
                if hasattr(self.simplex_projection_head, 'min_effective_assets'):
                    # For diversification-aware head, we might want to pass correlation
                    # This would need to be passed in from the model wrapper
                    correlation_matrix = None  # Would be passed from caller if available

                # Apply projection head to get portfolio weights
                # Different projection heads expect different parameters
                if hasattr(self.simplex_projection_head, 'min_effective_assets'):
                    # DiversificationAwareProjectionHead expects correlation_matrix
                    w = self.simplex_projection_head(
                        h,
                        correlation_matrix=correlation_matrix,
                        mask=mask_valid
                    )
                elif hasattr(self.simplex_projection_head, 'attention_processor'):
                    # RelationAwareAllocationHead expects attention_weights
                    # For now, pass None as we don't track attention weights separately
                    w = self.simplex_projection_head(
                        h,
                        attention_weights=None,
                        mask=mask_valid
                    )
                else:
                    # SimplexProjectionHead only needs embeddings and mask
                    w = self.simplex_projection_head(
                        h,
                        mask=mask_valid
                    )

                # Compute portfolio regularization if enabled
                if self.portfolio_reg is not None:
                    if prev_weights is not None:
                        reg_loss = self.portfolio_reg(
                            w.unsqueeze(0) if w.dim() == 1 else w,
                            prev_weights.unsqueeze(0) if prev_weights.dim() == 1 else prev_weights,
                        )
                    else:
                        reg_loss = self.portfolio_reg(w.unsqueeze(0) if w.dim() == 1 else w)

                logger.debug(f"Using simplex projection head: max weight={w.max():.3f}, effective assets={1.0/((w**2).sum()):.1f}")

            elif self.constraint_layer is not None:
                # Fallback to constraint layer if no projection head
                if scores.dim() == 1:
                    scores = scores.unsqueeze(0)
                    mask_valid = mask_valid.unsqueeze(0)
                    unsqueeze_output = True
                else:
                    unsqueeze_output = False

                w = self.constraint_layer(scores, mask_valid, self.head_activation)

                if unsqueeze_output:
                    w = w.squeeze(0)

                # Compute portfolio regularization if enabled
                if self.portfolio_reg is not None:
                    if prev_weights is not None:
                        reg_loss = self.portfolio_reg(
                            w.unsqueeze(0) if w.dim() == 1 else w,
                            prev_weights.unsqueeze(0) if prev_weights.dim() == 1 else prev_weights,
                        )
                    else:
                        reg_loss = self.portfolio_reg(w.unsqueeze(0) if w.dim() == 1 else w)

                logger.warning("Using constraint layer fallback (not ideal)")

            else:
                # Final fallback to original incorrect implementation
                if self.head_activation.lower() == "softmax":
                    w = masked_softmax(scores, mask_valid.bool(), dim=-1)
                else:
                    w = masked_sparsemax(scores, mask_valid.bool(), dim=-1)

                logger.warning("Using direct attention scores as weights (INCORRECT - proposal mismatch!)")

            return w, m_new, reg_loss

        # Enhanced markowitz head with additional processing
        mu_hat = torch.where(mask_valid.bool(), scores, torch.zeros_like(scores))

        # Apply additional normalization for Markowitz scores
        if self.constraint_aware:
            # Center the expected returns around zero for better optimization
            valid_mu = mu_hat[mask_valid.bool()]
            if len(valid_mu) > 0:
                mu_mean = valid_mu.mean()
                mu_hat = torch.where(mask_valid.bool(), mu_hat - mu_mean, torch.zeros_like(mu_hat))

        return mu_hat, m_new, reg_loss


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
