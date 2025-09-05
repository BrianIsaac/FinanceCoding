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
from typing import Optional, Tuple

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


def sparsemax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # after
    input = input.transpose(dim, -1)
    input_flat = input.reshape(-1, input.size(-1))
    zs = torch.sort(input_flat, descending=True, dim=-1).values
    range_ = torch.arange(1, zs.size(-1) + 1, device=input.device, dtype=input.dtype).view(1, -1)
    cssv = zs.cumsum(dim=-1) - 1
    cond = zs - cssv / range_ > 0
    k = cond.sum(dim=-1).clamp(min=1)
    tau = cssv.gather(1, (k - 1).unsqueeze(1)).squeeze(1) / k
    output_flat = torch.clamp(input_flat - tau.unsqueeze(1), min=0.0)
    output = output_flat.view_as(input)
    output = output.transpose(dim, -1)
    s = output.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    return output / s


def masked_sparsemax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    very_neg = torch.finfo(logits.dtype).min / 4.0
    z = torch.where(mask, logits, torch.full_like(logits, very_neg))
    p = sparsemax(z, dim=dim)
    return torch.where(mask, p, torch.zeros_like(p))


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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor]) -> torch.Tensor:
        if self.use_edge_attr and edge_attr is not None:
            # Be robust: if edge_attr has more than 1 feature, compress to 1 via mean.
            if edge_attr.dim() == 1:
                ea = edge_attr.view(-1, 1)
            elif edge_attr.size(-1) != 1:
                ea = edge_attr.mean(dim=-1, keepdim=True)
            else:
                ea = edge_attr
            h = self.conv(x, edge_index, ea)
        else:
            h = self.conv(x, edge_index)
        h = self.proj(h)
        if self.residual:
            h = h + x
        h = F.elu(h)
        h = self.ln(h)
        h = self.dropout(h)
        return h


# ---------------------------- model ----------------------------

@dataclass
class HeadCfg:
    mode: str = "markowitz"          # "markowitz" | "direct"
    activation: str = "sparsemax"    # used only if mode == "direct": "softmax" | "sparsemax"


class GATPortfolio(nn.Module):
    """
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
        mem_hidden: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.use_edge_attr = use_edge_attr
        self.head_mode = head
        self.head_activation = activation if head == "direct" else "none"

        # Backbone
        layers = []
        d_in = in_dim
        for li in range(num_layers):
            block = GATBlock(
                in_dim=d_in,
                out_dim=hidden_dim,
                heads=heads,
                dropout=dropout,
                use_gatv2=use_gatv2,
                use_edge_attr=use_edge_attr,
                edge_dim=1,   # we standardize to 1-d edge weight
                residual=residual,
            )
            layers.append(block)
            d_in = hidden_dim
        self.gnn = nn.ModuleList(layers)

        self.head_in_dim = hidden_dim
        self.heads = heads

        # Readout: simple MLP on top of the last block output
        self.readout = nn.Sequential(
            nn.Linear(self.head_in_dim, self.head_in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.head_in_dim, 1),
        )

        # Node-wise temporal memory (if enabled)
        self._mem_dim = int(mem_hidden) if mem_hidden is not None else 0
        if self._mem_dim > 0:
            self.mem_gru = nn.GRUCell(self.head_in_dim, self._mem_dim)
            # fuse memory back into features before head
            self.mem_fuse = nn.Linear(self.head_in_dim + self._mem_dim, self.head_in_dim)
        else:
            self.mem_gru = None
            self.mem_fuse = None

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
        x: torch.Tensor,                     # [N, F]
        edge_index: torch.Tensor,            # [2, E]
        mask_valid: torch.Tensor,            # [N] bool
        edge_attr: Optional[torch.Tensor] = None,   # [E, d_e] optional
        prev_mem: Optional[torch.Tensor] = None,    # [N, mem_dim] optional
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        # Head
        scores = self.readout(h).squeeze(-1)  # [N]

        if self.head_mode == "direct":
            # Nonnegative weights that sum to 1 over valid nodes
            if self.head_activation.lower() == "softmax":
                w = masked_softmax(scores, mask_valid.bool(), dim=-1)
            else:
                w = masked_sparsemax(scores, mask_valid.bool(), dim=-1)
            return w, m_new

        # markowitz head: return µ̂ (scores). Invalid nodes -> 0
        mu_hat = torch.where(mask_valid.bool(), scores, torch.zeros_like(scores))
        return mu_hat, m_new


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
