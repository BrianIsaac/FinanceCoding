#!/usr/bin/env python3
from __future__ import annotations
from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv

def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Softmax over valid entries only."""
    m = mask.to(dtype=logits.dtype)
    z = logits + (m - 1.0) * 1e9
    w = torch.softmax(z, dim=-1)
    s = (w * m).sum().clamp_min(eps)
    return (w * m) / s

class GATPortfolio(nn.Module):
    """GAT encoder with optional temporal GRU memory and two output modes.

    Modes:
      - 'direct'    : score -> masked softmax weights (long-only, sum=1)
      - 'markowitz' : score -> μ̂ (per-asset expected return), weights decided outside
                      by a differentiable Markowitz layer.

    Edge weights:
      If use_edge_attr=True and edge_attr is provided with shape (E, edge_dim),
      they are fed to the attention via 'edge_dim' projections.

    Temporal memory:
      If prev_mem (N, Dm) is provided, a GRUCell updates per-node state and the
      head reads from the new memory.
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
        head: str = "direct",            # 'direct' | 'markowitz'
        mem_hidden: Optional[int] = None # None => equals last GAT dim
    ) -> None:
        super().__init__()
        assert num_layers >= 1
        self.dropout = dropout
        self.residual = residual
        self.use_edge_attr = use_edge_attr
        self.head_mode = head

        Conv = GATv2Conv if use_gatv2 else GATConv
        edge_dim = 1 if use_edge_attr else None

        layers = []
        last_dim = in_dim
        for _ in range(num_layers):
            conv = Conv(
                in_channels=last_dim,
                out_channels=hidden_dim,
                heads=heads,
                concat=True,
                dropout=dropout,
                add_self_loops=True,
                edge_dim=edge_dim,   # <— include edge attributes in attention
            )
            layers.append(conv)
            last_dim = hidden_dim * heads
        self.convs = nn.ModuleList(layers)

        # Temporal memory (optional): size = last_dim by default
        self.mem_dim = mem_hidden or last_dim
        self.use_memory = True
        if self.mem_dim != last_dim:
            self.to_mem = nn.Linear(last_dim, self.mem_dim)
        else:
            self.to_mem = nn.Identity()
        self.gru = nn.GRUCell(self.mem_dim, self.mem_dim)

        # Heads
        if self.head_mode == "direct":
            self.scorer = nn.Linear(self.mem_dim, 1)         # scores -> masked softmax
        elif self.head_mode == "markowitz":
            self.mu_head = nn.Linear(self.mem_dim, 1)        # μ̂ per node
        else:
            raise ValueError("head must be 'direct' or 'markowitz'")

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode node features through GAT stack (consuming edge_attr if present)."""
        h = x
        for conv in self.convs:
            if self.use_edge_attr and edge_attr is not None:
                h_new = conv(h, edge_index, edge_attr)
            else:
                h_new = conv(h, edge_index)
            h_new = F.elu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            if self.residual and h_new.shape == h.shape:
                h = h + h_new
            else:
                h = h_new
        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        mask_valid: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        prev_mem: Optional[torch.Tensor] = None,   # (N, mem_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            (out, new_mem)
            - if head='direct'   : out = weights (N,)
            - if head='markowitz': out = mu_hat (N,)
        """
        h = self.encode(x, edge_index, edge_attr)
        h = self.to_mem(h)
        if prev_mem is not None:
            new_mem = self.gru(h, prev_mem)    # temporal update
        else:
            new_mem = h

        if self.head_mode == "direct":
            scores = self.scorer(new_mem).squeeze(-1)
            w = masked_softmax(scores, mask_valid)
            return w, new_mem
        else:
            mu_hat = self.mu_head(new_mem).squeeze(-1)
            # mask invalids to zero μ̂ (won't receive weight downstream anyway)
            mu_hat = mu_hat * mask_valid.to(mu_hat.dtype)
            return mu_hat, new_mem
