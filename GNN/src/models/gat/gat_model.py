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

import torch
from torch import nn
from torch.nn import functional as F

try:
    from torch_geometric.nn import GATConv, GATv2Conv  # type: ignore
except Exception:  # pragma: no cover
    GATConv = GATv2Conv = None  # type: ignore


# ---------------------------- utils ----------------------------


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Softmax over only True entries of mask (boolean). Zeros elsewhere.
    """
    very_neg = torch.finfo(logits.dtype).min / 4.0
    z = torch.where(mask, logits, torch.full_like(logits, very_neg))
    p = F.softmax(z, dim=dim)
    p = torch.where(mask, p, torch.zeros_like(p))
    s = p.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    return p / s


def sparsemax(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # after
    tensor = tensor.transpose(dim, -1)
    input_flat = tensor.reshape(-1, tensor.size(-1))
    zs = torch.sort(input_flat, descending=True, dim=-1).values
    range_ = torch.arange(1, zs.size(-1) + 1, device=tensor.device, dtype=tensor.dtype).view(1, -1)
    cssv = zs.cumsum(dim=-1) - 1
    cond = zs - cssv / range_ > 0
    k = cond.sum(dim=-1).clamp(min=1)
    tau = cssv.gather(1, (k - 1).unsqueeze(1)).squeeze(1) / k
    output_flat = torch.clamp(input_flat - tau.unsqueeze(1), min=0.0)
    output = output_flat.view_as(tensor)
    output = output.transpose(dim, -1)
    s = output.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    return output / s


def masked_sparsemax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    very_neg = torch.finfo(logits.dtype).min / 4.0
    z = torch.where(mask, logits, torch.full_like(logits, very_neg))
    p = sparsemax(z, dim=dim)
    return torch.where(mask, p, torch.zeros_like(p))


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
        # Scale by temperature
        scaled_logits = logits / self.temperature

        # Apply constraint-aware activation
        if activation.lower() == "sparsemax":
            weights = masked_sparsemax(scaled_logits, mask.bool(), dim=-1)
        else:
            weights = masked_softmax(scaled_logits, mask.bool(), dim=-1)

        # Apply minimum weight threshold
        weights = torch.where(weights < self.min_weight, torch.zeros_like(weights), weights)

        # Renormalize to ensure sum = 1
        weight_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        weights = weights / weight_sum

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

        h = F.elu(h)  # Using ELU for better gradient flow
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
    Enhanced GAT model for portfolio optimization with portfolio-specific optimizations.

    Returns:
        if head == "direct":  (weights[N], new_mem[N, mem_dim])
        if head == "markowitz": (mu_hat[N], new_mem[N, mem_dim])
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
    ) -> None:
        super().__init__()
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
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(self.head_in_dim * 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.5),  # Reduced dropout for final layer
                nn.Linear(hidden_dim, 1),
            )
        else:
            # Standard readout for Markowitz head
            self.readout = nn.Sequential(
                nn.Linear(self.head_in_dim, self.head_in_dim),
                nn.ReLU(inplace=True),
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

        self.reset_parameters()

    # ----------------- properties -----------------
    @property
    def mem_dim(self) -> int:
        return self._mem_dim

    # ----------------- nn API -----------------
    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
            h = F.relu(self.mem_fuse(h))
        else:
            m_new = torch.zeros(h.size(0), 0, device=h.device, dtype=h.dtype)

        # Head with enhanced processing
        if h.dim() == 2 and h.size(0) > 1:
            # Apply batch normalization if we have multiple samples
            h = F.layer_norm(h, h.shape[-1:])

        scores = self.readout(h).squeeze(-1)  # [N]
        reg_loss = None

        if self.head_mode == "direct":
            # Enhanced direct weight prediction with constraint layer
            if self.constraint_layer is not None:
                # Use constraint-aware layer for better weight prediction
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
            else:
                # Fallback to original implementation
                if self.head_activation.lower() == "softmax":
                    w = masked_softmax(scores, mask_valid.bool(), dim=-1)
                else:
                    w = masked_sparsemax(scores, mask_valid.bool(), dim=-1)

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
