"""
GAT Attention Weight Visualization Framework.

This module provides interpretability tools for Graph Attention Network models,
including attention weight extraction, aggregation, and visualization capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ...models.gat.gat_model import GATPortfolio
from ...models.gat.graph_builder import GraphBuildConfig, build_period_graph


@dataclass
class AttentionAnalysisConfig:
    """Configuration for attention weight analysis."""

    aggregation_method: str = "mean"  # "mean", "max", "sum"
    temporal_windows: int = 12  # Number of historical periods to analyze
    attention_threshold: float = 0.01  # Minimum attention weight to consider significant
    normalize_weights: bool = True
    include_self_attention: bool = False


class GATExplainer:
    """
    GAT attention weight extractor and analyzer.

    Provides tools for extracting, aggregating, and analyzing attention weights
    from trained GAT models to understand which asset relationships drive
    allocation decisions.
    """

    def __init__(self, model: GATPortfolio, config: AttentionAnalysisConfig | None = None):
        """
        Initialize GAT explainer.

        Args:
            model: Trained GAT model to analyze
            config: Configuration for attention analysis
        """
        self.model = model
        self.config = config or AttentionAnalysisConfig()
        self.model.eval()  # Ensure model is in evaluation mode

        # Store attention weights during forward pass
        self.attention_weights: list[torch.Tensor] = []
        self.layer_outputs: list[torch.Tensor] = []

        # Register hooks to capture attention weights
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward hooks to capture attention weights from GAT layers."""
        def attention_hook(layer_idx: int):
            def hook_fn(module: Any, input: Any, output: Any) -> None:
                # Extract attention weights from GATConv/GATv2Conv layers
                if hasattr(module, 'conv') and hasattr(module.conv, '_alpha'):
                    # Attention weights from torch_geometric GAT layers
                    alpha = module.conv._alpha
                    if alpha is not None:
                        self.attention_weights.append(alpha.detach().cpu())
                        self.layer_outputs.append(output.detach().cpu())
            return hook_fn

        # Register hooks for each GAT layer
        for i, layer in enumerate(self.model.gnn):
            layer.register_forward_hook(attention_hook(i))

    def extract_attention_weights(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        mask_valid: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Extract attention weights from GAT model.

        Args:
            x: Node features [n_nodes, n_features]
            edge_index: Edge connectivity [2, n_edges]
            mask_valid: Valid node mask [n_nodes]
            edge_attr: Optional edge attributes [n_edges, edge_dim]

        Returns:
            Dictionary containing attention weights per layer and aggregated weights
        """
        # Clear previous attention weights
        self.attention_weights.clear()
        self.layer_outputs.clear()

        # Forward pass to capture attention weights
        with torch.no_grad():
            _ = self.model(x, edge_index, mask_valid, edge_attr)

        if not self.attention_weights:
            raise ValueError("No attention weights captured. Check model architecture.")

        # Process attention weights
        processed_weights = {}

        for layer_idx, attn_weights in enumerate(self.attention_weights):
            # Reshape attention weights to [n_edges, n_heads]
            if attn_weights.dim() == 1:
                attn_weights = attn_weights.view(-1, 1)
            elif attn_weights.dim() == 2 and attn_weights.size(1) == 1:
                # Single head case
                pass
            else:
                # Multi-head case: reshape appropriately
                n_edges = edge_index.size(1)
                n_heads = self.model.heads
                if attn_weights.numel() == n_edges * n_heads:
                    attn_weights = attn_weights.view(n_edges, n_heads)
                else:
                    # Fallback: take first n_edges elements
                    attn_weights = attn_weights[:n_edges].view(n_edges, 1)

            processed_weights[f"layer_{layer_idx}"] = attn_weights

        # Aggregate attention weights across layers
        if len(processed_weights) > 0:
            aggregated = self._aggregate_attention_weights(
                list(processed_weights.values()),
                edge_index,
                mask_valid
            )
            processed_weights["aggregated"] = aggregated

        return processed_weights

    def _aggregate_attention_weights(
        self,
        layer_weights: list[torch.Tensor],
        edge_index: torch.Tensor,
        mask_valid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate attention weights across layers and heads.

        Args:
            layer_weights: List of attention weight tensors per layer
            edge_index: Edge connectivity matrix
            mask_valid: Valid node mask

        Returns:
            Aggregated attention weights [n_edges, 1]
        """
        if not layer_weights:
            return torch.zeros(edge_index.size(1), 1)

        # Ensure all tensors have the same shape
        n_edges = edge_index.size(1)
        processed_weights = []

        for weights in layer_weights:
            if weights.size(0) != n_edges:
                # Pad or truncate to match edge count
                if weights.size(0) > n_edges:
                    weights = weights[:n_edges]
                else:
                    padding = torch.zeros(n_edges - weights.size(0), weights.size(1))
                    weights = torch.cat([weights, padding], dim=0)
            processed_weights.append(weights)

        # Stack and aggregate
        stacked = torch.stack(processed_weights, dim=0)  # [n_layers, n_edges, n_heads]

        # Aggregate across heads within each layer
        if self.config.aggregation_method == "mean":
            layer_aggregated = stacked.mean(dim=2)  # [n_layers, n_edges]
        elif self.config.aggregation_method == "max":
            layer_aggregated = stacked.max(dim=2)[0]
        elif self.config.aggregation_method == "sum":
            layer_aggregated = stacked.sum(dim=2)
        else:
            layer_aggregated = stacked.mean(dim=2)

        # Aggregate across layers
        if self.config.aggregation_method == "mean":
            final_weights = layer_aggregated.mean(dim=0)  # [n_edges]
        elif self.config.aggregation_method == "max":
            final_weights = layer_aggregated.max(dim=0)[0]
        elif self.config.aggregation_method == "sum":
            final_weights = layer_aggregated.sum(dim=0)
        else:
            final_weights = layer_aggregated.mean(dim=0)

        # Normalize if requested
        if self.config.normalize_weights:
            final_weights = F.softmax(final_weights, dim=0)

        return final_weights.unsqueeze(-1)

    def build_attention_matrix(
        self,
        attention_weights: torch.Tensor,
        edge_index: torch.Tensor,
        n_nodes: int,
        asset_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Build attention matrix from edge attention weights.

        Args:
            attention_weights: Attention weights for edges [n_edges, 1]
            edge_index: Edge connectivity [2, n_edges]
            n_nodes: Number of nodes in the graph
            asset_names: Optional asset names for the matrix index/columns

        Returns:
            Attention matrix as pandas DataFrame
        """
        # Initialize attention matrix
        attention_matrix = torch.zeros(n_nodes, n_nodes)

        # Fill matrix with attention weights
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        weights = attention_weights.squeeze(-1)

        for i, (src, tgt) in enumerate(zip(source_nodes, target_nodes)):
            if not self.config.include_self_attention and src == tgt:
                continue
            attention_matrix[src, tgt] = weights[i]

        # Convert to pandas DataFrame
        if asset_names is not None:
            if len(asset_names) != n_nodes:
                asset_names = [f"Asset_{i}" for i in range(n_nodes)]
        else:
            asset_names = [f"Asset_{i}" for i in range(n_nodes)]

        df = pd.DataFrame(
            attention_matrix.numpy(),
            index=asset_names,
            columns=asset_names
        )

        return df

    def analyze_temporal_attention_evolution(
        self,
        returns: pd.DataFrame,
        universe: list[str],
        dates: list[pd.Timestamp],
        graph_config: GraphBuildConfig,
        features_matrix: np.ndarray | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Analyze attention weight evolution over time.

        Args:
            returns: Historical returns DataFrame
            universe: List of asset tickers
            dates: List of analysis dates
            graph_config: Graph construction configuration
            features_matrix: Pre-computed features matrix

        Returns:
            Dictionary containing temporal attention analysis results
        """
        temporal_analysis = {}
        attention_matrices = []

        for date in dates:
            try:
                # Build graph for this period
                graph_data = build_period_graph(
                    returns_daily=returns,
                    period_end=date,
                    tickers=universe,
                    features_matrix=features_matrix,
                    cfg=graph_config,
                )

                # Extract attention weights
                attention_weights = self.extract_attention_weights(
                    x=graph_data.x,
                    edge_index=graph_data.edge_index,
                    mask_valid=torch.ones(len(universe), dtype=torch.bool),
                    edge_attr=graph_data.edge_attr,
                )

                # Build attention matrix
                if "aggregated" in attention_weights:
                    attn_matrix = self.build_attention_matrix(
                        attention_weights["aggregated"],
                        graph_data.edge_index,
                        len(universe),
                        universe,
                    )
                    attention_matrices.append(attn_matrix)

            except Exception:
                # Skip problematic dates
                continue

        if attention_matrices:
            # Create temporal evolution analysis
            temporal_analysis["attention_evolution"] = self._analyze_attention_evolution(
                attention_matrices, dates[:len(attention_matrices)]
            )

            # Identify most influential connections over time
            temporal_analysis["influential_connections"] = self._identify_influential_connections(
                attention_matrices, universe
            )

        return temporal_analysis

    def _analyze_attention_evolution(
        self,
        attention_matrices: list[pd.DataFrame],
        dates: list[pd.Timestamp],
    ) -> pd.DataFrame:
        """
        Analyze how attention patterns evolve over time.

        Args:
            attention_matrices: List of attention matrices over time
            dates: Corresponding dates

        Returns:
            DataFrame with temporal attention statistics
        """
        evolution_stats = []

        for i, (matrix, date) in enumerate(zip(attention_matrices, dates)):
            # Remove diagonal (self-attention) if requested
            if not self.config.include_self_attention:
                matrix = matrix.copy()
                np.fill_diagonal(matrix.values, 0)

            # Calculate statistics
            stats = {
                "date": date,
                "mean_attention": matrix.values.mean(),
                "max_attention": matrix.values.max(),
                "attention_concentration": (matrix.values ** 2).sum(),
                "n_significant_connections": (matrix.values > self.config.attention_threshold).sum(),
                "attention_entropy": self._calculate_attention_entropy(matrix.values),
            }

            # Compare with previous period if available
            if i > 0:
                prev_matrix = attention_matrices[i-1]
                diff_matrix = matrix - prev_matrix
                stats.update({
                    "attention_change_mean": diff_matrix.values.mean(),
                    "attention_change_std": diff_matrix.values.std(),
                    "attention_stability": 1 - (diff_matrix.values ** 2).sum() / (matrix.values ** 2).sum(),
                })

            evolution_stats.append(stats)

        return pd.DataFrame(evolution_stats)

    def _identify_influential_connections(
        self,
        attention_matrices: list[pd.DataFrame],
        asset_names: list[str],
    ) -> pd.DataFrame:
        """
        Identify the most influential asset connections over time.

        Args:
            attention_matrices: List of attention matrices
            asset_names: Asset names

        Returns:
            DataFrame with influential connection statistics
        """
        # Stack all matrices to analyze connections
        stacked_matrices = np.stack([matrix.values for matrix in attention_matrices])

        influential_connections = []

        for i, asset1 in enumerate(asset_names):
            for j, asset2 in enumerate(asset_names):
                if i == j and not self.config.include_self_attention:
                    continue

                # Analyze this connection over time
                connection_weights = stacked_matrices[:, i, j]

                connection_stats = {
                    "source_asset": asset1,
                    "target_asset": asset2,
                    "mean_attention": connection_weights.mean(),
                    "max_attention": connection_weights.max(),
                    "attention_std": connection_weights.std(),
                    "attention_trend": np.corrcoef(np.arange(len(connection_weights)), connection_weights)[0, 1],
                    "significant_periods": (connection_weights > self.config.attention_threshold).sum(),
                }

                influential_connections.append(connection_stats)

        df = pd.DataFrame(influential_connections)
        return df.sort_values("mean_attention", ascending=False)

    def _calculate_attention_entropy(self, attention_matrix: np.ndarray) -> float:
        """
        Calculate attention entropy to measure concentration.

        Args:
            attention_matrix: Attention weight matrix

        Returns:
            Entropy value (higher = more distributed attention)
        """
        # Flatten and normalize weights
        weights = attention_matrix.flatten()
        weights = weights[weights > 0]  # Remove zero weights

        if len(weights) == 0:
            return 0.0

        # Normalize to probabilities
        probs = weights / weights.sum()

        # Calculate entropy
        entropy = -(probs * np.log(probs + 1e-12)).sum()
        return float(entropy)

    def get_top_attention_pairs(
        self,
        attention_matrix: pd.DataFrame,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """
        Get top-k asset pairs by attention weight.

        Args:
            attention_matrix: Attention weight matrix
            top_k: Number of top pairs to return

        Returns:
            DataFrame with top attention pairs
        """
        # Create list of all pairs with their attention weights
        pairs = []

        for i, asset1 in enumerate(attention_matrix.index):
            for j, asset2 in enumerate(attention_matrix.columns):
                if i == j and not self.config.include_self_attention:
                    continue

                weight = attention_matrix.iloc[i, j]
                if weight > self.config.attention_threshold:
                    pairs.append({
                        "source_asset": asset1,
                        "target_asset": asset2,
                        "attention_weight": weight,
                        "relationship_type": self._classify_relationship(asset1, asset2, weight),
                    })

        # Sort by attention weight and return top-k
        df = pd.DataFrame(pairs)
        if not df.empty:
            df = df.sort_values("attention_weight", ascending=False).head(top_k)

        return df

    def _classify_relationship(self, asset1: str, asset2: str, weight: float) -> str:
        """
        Classify the type of relationship between two assets.

        Args:
            asset1: First asset
            asset2: Second asset
            weight: Attention weight

        Returns:
            Relationship classification
        """
        # Simple classification based on attention weight magnitude
        if weight > 0.1:
            return "Strong"
        elif weight > 0.05:
            return "Moderate"
        elif weight > 0.01:
            return "Weak"
        else:
            return "Minimal"
