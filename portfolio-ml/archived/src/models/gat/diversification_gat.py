"""
Diversification-aware GAT model for portfolio optimization.

This module implements a Graph Attention Network that properly handles asset correlations
by selecting representative assets from correlation clusters rather than allocating to
all correlated assets.
"""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from sklearn.cluster import SpectralClustering
from torch_geometric.nn import GATv2Conv

logger = logging.getLogger(__name__)


class CorrelationAwareGraphBuilder:
    """
    Builds graphs with correlation-aware edge weights that encourage diversification.
    """

    def __init__(self, correlation_threshold: float = 0.7):
        """
        Initialize correlation-aware graph builder.

        Args:
            correlation_threshold: Threshold above which assets are considered highly correlated
        """
        self.correlation_threshold = correlation_threshold

    def build_diversification_graph(
        self,
        correlation_matrix: np.ndarray,
        method: str = "negative_edges"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build graph that encourages diversification.

        Args:
            correlation_matrix: Asset correlation matrix
            method: Graph construction method

        Returns:
            edge_index: Graph edges [2, num_edges]
            edge_weights: Edge weights (negative for correlated assets)
        """
        n_assets = correlation_matrix.shape[0]

        if method == "negative_edges":
            # Create edges for all pairs
            edge_list = []
            edge_weights = []

            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    corr = correlation_matrix[i, j]

                    # Add edge if correlation is significant
                    if abs(corr) > 0.1:
                        edge_list.append([i, j])
                        edge_list.append([j, i])  # Undirected

                        # Negative weight for high correlation (penalize selecting both)
                        # Positive weight for negative correlation (good to have both)
                        if corr > self.correlation_threshold:
                            weight = -abs(corr)  # Strong penalty for selecting both
                        elif corr < -0.3:
                            weight = 0.5  # Mild reward for hedging pairs
                        else:
                            weight = 0.0  # Neutral

                        edge_weights.extend([weight, weight])

            edge_index = np.array(edge_list).T if edge_list else np.zeros((2, 0))
            edge_weights = np.array(edge_weights) if edge_weights else np.zeros(0)

        elif method == "cluster_representative":
            # Build edges only between cluster representatives
            clusters = self._identify_correlation_clusters(correlation_matrix)
            edge_list = []
            edge_weights = []

            # Connect representatives of different clusters
            unique_clusters = np.unique(clusters)
            for c1 in unique_clusters:
                for c2 in unique_clusters:
                    if c1 < c2:
                        # Find representative (highest avg correlation within cluster)
                        c1_mask = clusters == c1
                        c2_mask = clusters == c2

                        # Get representatives
                        c1_nodes = np.where(c1_mask)[0]
                        c2_nodes = np.where(c2_mask)[0]

                        # Connect representatives
                        for n1 in c1_nodes[:1]:  # Just the representative
                            for n2 in c2_nodes[:1]:
                                edge_list.append([n1, n2])
                                edge_list.append([n2, n1])

                                # Weight based on cluster correlation
                                cluster_corr = np.mean(correlation_matrix[c1_mask][:, c2_mask])
                                weight = 1.0 - abs(cluster_corr)  # Higher weight for uncorrelated clusters
                                edge_weights.extend([weight, weight])

            edge_index = np.array(edge_list).T if edge_list else np.zeros((2, 0))
            edge_weights = np.array(edge_weights) if edge_weights else np.zeros(0)

        else:
            raise ValueError(f"Unknown method: {method}")

        return edge_index, edge_weights

    def _identify_correlation_clusters(
        self,
        correlation_matrix: np.ndarray,
        n_clusters: Optional[int] = None
    ) -> np.ndarray:
        """
        Identify correlation-based asset clusters.

        Args:
            correlation_matrix: Asset correlation matrix
            n_clusters: Number of clusters (auto-detect if None)

        Returns:
            Cluster labels for each asset
        """
        n_assets = correlation_matrix.shape[0]

        if n_clusters is None:
            # Auto-detect optimal number of clusters
            n_clusters = max(5, min(20, n_assets // 10))

        # Convert correlation to affinity matrix
        affinity_matrix = (correlation_matrix + 1) / 2  # Scale to [0, 1]

        # Spectral clustering
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42
        )

        clusters = clustering.fit_predict(affinity_matrix)
        return clusters


class DiversificationGAT(nn.Module):
    """
    GAT model that enforces diversification through correlation-aware selection.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        correlation_penalty: float = 0.5,
        cluster_selection: bool = True
    ):
        """
        Initialize Diversification-aware GAT.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (1 for portfolio score)
            num_heads: Number of attention heads
            num_layers: Number of GAT layers
            dropout: Dropout rate
            correlation_penalty: Weight for correlation penalty in loss
            cluster_selection: Whether to use cluster-based selection
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.correlation_penalty = correlation_penalty
        self.cluster_selection = cluster_selection

        # GAT layers
        self.gat_layers = nn.ModuleList()

        # First layer
        self.gat_layers.append(
            GATv2Conv(
                input_dim,
                hidden_dim,
                heads=num_heads,
                dropout=dropout,
                edge_dim=3,  # Match original GAT edge dimension
                concat=True
            )
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATv2Conv(
                    hidden_dim * num_heads,
                    hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=3,  # Match original GAT edge dimension
                    concat=True
                )
            )

        # Output layer
        self.gat_layers.append(
            GATv2Conv(
                hidden_dim * num_heads,
                output_dim,
                heads=1,
                dropout=dropout,
                edge_dim=3,  # Match original GAT edge dimension
                concat=False
            )
        )

        # Additional layers for processing
        self.batch_norm = nn.BatchNorm1d(hidden_dim * num_heads)
        self.dropout = nn.Dropout(dropout)

        # Correlation-aware components
        self.correlation_processor = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.graph_builder = CorrelationAwareGraphBuilder()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        correlation_matrix: Optional[torch.Tensor] = None,
        clusters: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with correlation-aware processing.

        Args:
            x: Node features [n_nodes, input_dim]
            edge_index: Graph edges [2, n_edges]
            edge_attr: Edge attributes [n_edges, edge_dim] or None
            correlation_matrix: Asset correlation matrix [n_nodes, n_nodes]
            clusters: Pre-computed cluster labels [n_nodes]

        Returns:
            portfolio_weights: Final portfolio weights [n_nodes]
            gat_scores: Raw GAT output scores [n_nodes]
        """
        # Standard GAT processing
        h = x

        # Handle edge attributes
        edge_attr_to_use = edge_attr
        if edge_attr is not None and edge_attr.dim() == 1:
            # If 1D edge weights provided, expand to 2D
            edge_attr_to_use = edge_attr.unsqueeze(-1)

        for i, layer in enumerate(self.gat_layers[:-1]):
            h = layer(h, edge_index, edge_attr_to_use)
            h = F.elu(h)
            h = self.dropout(h)

        # Final layer
        gat_scores = self.gat_layers[-1](h, edge_index, edge_attr_to_use)
        gat_scores = gat_scores.squeeze(-1)

        # Apply correlation-aware selection
        if self.cluster_selection and correlation_matrix is not None:
            portfolio_weights = self._cluster_based_selection(
                gat_scores, correlation_matrix, clusters
            )
        else:
            # Standard softmax (but with correlation penalty in loss)
            portfolio_weights = F.softmax(gat_scores, dim=0)

        return portfolio_weights, gat_scores

    def _cluster_based_selection(
        self,
        gat_scores: torch.Tensor,
        correlation_matrix: torch.Tensor,
        clusters: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Select best asset from each correlation cluster.

        Args:
            gat_scores: GAT output scores [n_nodes]
            correlation_matrix: Correlation matrix [n_nodes, n_nodes]
            clusters: Pre-computed clusters (compute if None)

        Returns:
            Portfolio weights with cluster-based selection
        """
        n_assets = gat_scores.shape[0]
        device = gat_scores.device

        # Compute clusters if not provided
        if clusters is None:
            corr_numpy = correlation_matrix.detach().cpu().numpy()
            clusters_numpy = self.graph_builder._identify_correlation_clusters(corr_numpy)
            clusters = torch.tensor(clusters_numpy, device=device)

        # Initialize weights
        weights = torch.zeros_like(gat_scores)

        # For each cluster, select best asset
        unique_clusters = torch.unique(clusters)
        selected_indices = []

        for cluster_id in unique_clusters:
            cluster_mask = clusters == cluster_id
            cluster_scores = gat_scores[cluster_mask]

            if len(cluster_scores) > 0:
                # Select best in cluster
                best_in_cluster_idx = cluster_scores.argmax()
                global_idx = torch.where(cluster_mask)[0][best_in_cluster_idx]
                selected_indices.append(global_idx)

        # Allocate weights only to selected assets
        if selected_indices:
            selected_indices = torch.stack(selected_indices)
            selected_scores = gat_scores[selected_indices]

            # Apply softmax only to selected assets
            selected_weights = F.softmax(selected_scores, dim=0)
            weights[selected_indices] = selected_weights
        else:
            # Fallback to equal weights if no selection
            weights = torch.ones_like(gat_scores) / n_assets

        return weights

    def compute_correlation_penalty(
        self,
        weights: torch.Tensor,
        correlation_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute penalty for allocating to correlated assets.

        Args:
            weights: Portfolio weights [n_assets]
            correlation_matrix: Correlation matrix [n_assets, n_assets]

        Returns:
            Correlation penalty scalar
        """
        # Weighted correlation exposure
        weighted_corr = weights @ correlation_matrix @ weights

        # Penalty increases with correlation concentration
        # Subtract diagonal (self-correlation) to focus on cross-correlations
        n_assets = weights.shape[0]
        identity_contribution = weights.pow(2).sum()

        correlation_penalty = weighted_corr - identity_contribution

        return correlation_penalty * self.correlation_penalty


class DiversificationLoss(nn.Module):
    """
    Loss function that combines Sharpe ratio with diversification penalties.
    """

    def __init__(
        self,
        sharpe_weight: float = 0.4,
        correlation_weight: float = 0.3,
        concentration_weight: float = 0.3
    ):
        """
        Initialize diversification loss.

        Args:
            sharpe_weight: Weight for Sharpe ratio component
            correlation_weight: Weight for correlation penalty
            concentration_weight: Weight for concentration penalty
        """
        super().__init__()
        self.sharpe_weight = sharpe_weight
        self.correlation_weight = correlation_weight
        self.concentration_weight = concentration_weight

    def forward(
        self,
        weights: torch.Tensor,
        returns: torch.Tensor,
        correlation_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            weights: Portfolio weights [batch_size, n_assets] or [n_assets]
            returns: Asset returns [batch_size, n_assets] or [n_periods, n_assets]
            correlation_matrix: Asset correlations [n_assets, n_assets]

        Returns:
            Combined loss scalar
        """
        # Ensure proper dimensions
        if weights.dim() == 1:
            weights = weights.unsqueeze(0)
        if returns.dim() == 1:
            returns = returns.unsqueeze(0)

        # Sharpe ratio loss (negative for maximization)
        portfolio_returns = (weights * returns).sum(dim=-1)
        sharpe = portfolio_returns.mean() / (portfolio_returns.std() + 1e-8)
        sharpe_loss = -sharpe * self.sharpe_weight

        # Correlation penalty
        correlation_loss = torch.tensor(0.0, device=weights.device)
        if correlation_matrix is not None and self.correlation_weight > 0:
            weighted_corr = torch.matmul(
                torch.matmul(weights, correlation_matrix),
                weights.T
            ).diagonal().mean()
            correlation_loss = weighted_corr * self.correlation_weight

        # Concentration penalty (HHI)
        concentration = (weights ** 2).sum(dim=-1).mean()
        concentration_loss = concentration * self.concentration_weight

        total_loss = sharpe_loss + correlation_loss + concentration_loss

        return total_loss