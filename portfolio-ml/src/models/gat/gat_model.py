"""Clean GAT portfolio model - research-proven architecture.

Based on Korangi et al. (2024) and FinGAT (2021):
- 2-layer GAT: 8 heads → 1 head
- Hidden dimension: 16 (proven effective)
- Batch normalization for stability
- LeakyReLU activation
- Dropout 0.3
- Reduce-weight allocation (w = s²/Σs²)
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv
except ImportError:
    raise ImportError(
        "PyTorch Geometric is required. Install with: "
        "pip install torch-geometric"
    )

from .allocation import ReduceWeightAllocator

logger = logging.getLogger(__name__)


class GATPortfolio(nn.Module):
    """Graph Attention Network for portfolio optimization.

    Clean, simple architecture proven in research papers.
    """

    def __init__(
        self,
        n_features: int = 11,
        hidden_dim: int = 16,
        n_heads_layer1: int = 8,
        n_heads_layer2: int = 1,
        dropout: float = 0.3,
        max_weight: float | None = 0.15,
    ):
        """Initialize GAT portfolio model.

        Args:
            n_features: Number of input node features
            hidden_dim: Hidden dimension size
            n_heads_layer1: Number of attention heads in first layer
            n_heads_layer2: Number of attention heads in second layer
            dropout: Dropout probability
            max_weight: Maximum weight per asset (e.g., 0.15 for 15%)
        """
        super().__init__()

        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # First GAT layer: n_features → hidden_dim * n_heads
        self.gat1 = GATConv(
            in_channels=n_features,
            out_channels=hidden_dim,
            heads=n_heads_layer1,
            dropout=dropout,
            concat=True  # Concatenate multi-head outputs
        )

        # Layer normalization after first layer (better than BatchNorm for single-sample training)
        self.ln1 = nn.LayerNorm(hidden_dim * n_heads_layer1)

        # Second GAT layer: hidden_dim * n_heads → hidden_dim
        self.gat2 = GATConv(
            in_channels=hidden_dim * n_heads_layer1,
            out_channels=hidden_dim,
            heads=n_heads_layer2,
            dropout=dropout,
            concat=False  # Average multi-head outputs (only 1 head anyway)
        )

        # Layer normalization after second layer (better than BatchNorm for single-sample training)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Allocation head: Reduce-weight mechanism
        self.allocator = ReduceWeightAllocator(
            input_dim=hidden_dim,
            max_weight=max_weight
        )

        # Initialize weights with Xavier (proven for financial data)
        self.apply(self._init_weights)

        logger.info(
            f"Initialized GAT: {n_features} features → {hidden_dim}D "
            f"({n_heads_layer1} heads → {n_heads_layer2} head), "
            f"dropout={dropout}, max_weight={max_weight}"
        )

    def _init_weights(self, module):
        """Initialize weights with Xavier normal (prevents mode collapse)."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass through GAT.

        Args:
            x: Node features [n_nodes, n_features]
            edge_index: Graph edge index [2, n_edges]
            mask: Optional mask for valid assets [n_nodes]

        Returns:
            Portfolio weights [n_nodes]
        """
        # Check input for NaN
        if torch.isnan(x).any():
            logger.error(f"NaN detected in input features: {torch.isnan(x).sum().item()} NaN values")

        # First GAT layer
        x = self.gat1(x, edge_index)
        if torch.isnan(x).any():
            logger.error(f"NaN detected after GAT1: {torch.isnan(x).sum().item()} NaN values")

        x = self.ln1(x)
        if torch.isnan(x).any():
            logger.error(f"NaN detected after LayerNorm1: {torch.isnan(x).sum().item()} NaN values")

        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second GAT layer
        x = self.gat2(x, edge_index)
        if torch.isnan(x).any():
            logger.error(f"NaN detected after GAT2: {torch.isnan(x).sum().item()} NaN values")

        x = self.ln2(x)
        if torch.isnan(x).any():
            logger.error(f"NaN detected after LayerNorm2: {torch.isnan(x).sum().item()} NaN values")

        x = F.leaky_relu(x)

        # Convert to portfolio weights
        weights = self.allocator(x, mask=mask)
        if torch.isnan(weights).any():
            logger.error(f"NaN detected in final weights: {torch.isnan(weights).sum().item()} NaN values")

        return weights

    def get_attention_weights(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get attention weights for visualization/analysis.

        Args:
            x: Node features [n_nodes, n_features]
            edge_index: Graph edge index [2, n_edges]

        Returns:
            Tuple of (layer1_attention, layer2_attention)
        """
        # Forward through first layer with return_attention_weights
        _, (edge_index_1, alpha_1) = self.gat1(
            x, edge_index, return_attention_weights=True
        )

        x = self.ln1(self.gat1(x, edge_index))
        x = F.leaky_relu(x)

        # Forward through second layer with return_attention_weights
        _, (edge_index_2, alpha_2) = self.gat2(
            x, edge_index, return_attention_weights=True
        )

        return alpha_1, alpha_2

    def count_parameters(self) -> dict[str, int]:
        """Count trainable parameters by component.

        Returns:
            Dictionary of parameter counts
        """
        counts = {
            "gat1": sum(p.numel() for p in self.gat1.parameters() if p.requires_grad),
            "gat2": sum(p.numel() for p in self.gat2.parameters() if p.requires_grad),
            "allocator": sum(p.numel() for p in self.allocator.parameters() if p.requires_grad),
            "total": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
        return counts
