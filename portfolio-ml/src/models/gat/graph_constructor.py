"""Graph construction for GAT portfolio models.

Implements three proven graph filtering methods:
- MST (Minimum Spanning Tree): Connects assets with minimal total distance
- kNN (k-Nearest Neighbors): Each asset connects to k most correlated assets
- TMFG (Triangulated Maximally Filtered Graph): Planar graph, O(3n) edges
"""

import logging
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import spearmanr
from sklearn.neighbors import kneighbors_graph

logger = logging.getLogger(__name__)


def build_graph(
    returns: pd.DataFrame,
    universe: list[str],
    method: Literal["mst", "knn", "tmfg"] = "mst",
    lookback_days: int = 63,
    knn_k: int = 5,
) -> torch.Tensor:
    """Build graph adjacency and edge_index for GAT.

    Args:
        returns: Historical returns DataFrame [time, assets]
        universe: List of asset tickers
        method: Graph construction method ("mst", "knn", "tmfg")
        lookback_days: Correlation lookback window
        knn_k: Number of neighbors for kNN method

    Returns:
        edge_index: PyTorch Geometric edge index [2, num_edges]
    """
    # Get returns for universe assets (handle missing)
    available_tickers = [t for t in universe if t in returns.columns]
    returns_subset = returns[available_tickers].iloc[-lookback_days:]

    # Compute correlation matrix
    if method == "mst":
        # Spearman correlation for stability (proven in research)
        corr_matrix, _ = spearmanr(returns_subset.values, nan_policy='omit')
        if corr_matrix.ndim == 0:  # Handle single asset case
            corr_matrix = np.array([[1.0]])
    else:
        # Pearson correlation for kNN/TMFG
        corr_matrix = returns_subset.corr().values

    # Handle NaN
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Ensure symmetric and fill diagonal
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    np.fill_diagonal(corr_matrix, 1.0)

    # Build graph based on method
    if method == "mst":
        adjacency = _build_mst(corr_matrix)
    elif method == "knn":
        adjacency = _build_knn(corr_matrix, k=knn_k)
    elif method == "tmfg":
        adjacency = _build_tmfg(corr_matrix)
    else:
        raise ValueError(f"Unknown graph method: {method}")

    # Convert adjacency to edge_index for PyTorch Geometric
    edge_index = _adjacency_to_edge_index(adjacency)

    logger.info(
        f"Built {method.upper()} graph: {len(available_tickers)} nodes, "
        f"{edge_index.shape[1]} edges (lookback={lookback_days} days)"
    )

    return edge_index


def _build_mst(corr_matrix: np.ndarray) -> np.ndarray:
    """Build Minimum Spanning Tree from correlation matrix.

    Args:
        corr_matrix: Correlation matrix [n_assets, n_assets]

    Returns:
        Adjacency matrix [n_assets, n_assets]
    """
    # Convert correlation to distance
    distance = 1 - np.abs(corr_matrix)

    # Compute MST
    mst = minimum_spanning_tree(distance).toarray()

    # Make symmetric (MST is directed, we want undirected)
    adjacency = mst + mst.T

    # Binarize (set all edges to 1)
    adjacency = (adjacency > 0).astype(float)

    return adjacency


def _build_knn(corr_matrix: np.ndarray, k: int = 5) -> np.ndarray:
    """Build k-Nearest Neighbors graph from correlation matrix.

    Args:
        corr_matrix: Correlation matrix [n_assets, n_assets]
        k: Number of nearest neighbors

    Returns:
        Adjacency matrix [n_assets, n_assets]
    """
    n_assets = corr_matrix.shape[0]
    k_safe = min(k, n_assets - 1)  # Can't have more neighbors than assets

    # Build kNN graph using sklearn
    # Higher correlation = closer neighbors
    knn_graph = kneighbors_graph(
        corr_matrix,
        n_neighbors=k_safe,
        mode='connectivity',
        include_self=False
    ).toarray()

    # Make symmetric
    adjacency = np.maximum(knn_graph, knn_graph.T)

    return adjacency


def _build_tmfg(corr_matrix: np.ndarray) -> np.ndarray:
    """Build Triangulated Maximally Filtered Graph.

    Simplified TMFG: Builds a planar graph with ~3n edges.
    For production, use the fast-tmfg package if available.

    Args:
        corr_matrix: Correlation matrix [n_assets, n_assets]

    Returns:
        Adjacency matrix [n_assets, n_assets]
    """
    try:
        # Try to use fast-tmfg if available
        from fast_tmfg import TMFG

        # Distance correlation would be ideal but requires dcor package
        # For now, use correlation directly
        tmfg = TMFG()
        _, _, adjacency = tmfg.fit_transform(
            weights=corr_matrix ** 2,  # Squared correlation as weights
            output='unweighted_sparse_W_matrix'
        )
        return adjacency

    except ImportError:
        # Fallback: Use MST as approximation
        logger.warning(
            "fast-tmfg package not available, falling back to MST. "
            "Install with: pip install fast-tmfg"
        )
        return _build_mst(corr_matrix)


def _adjacency_to_edge_index(adjacency: np.ndarray) -> torch.Tensor:
    """Convert adjacency matrix to PyTorch Geometric edge_index format.

    Args:
        adjacency: Adjacency matrix [n_nodes, n_nodes]

    Returns:
        edge_index: Edge index [2, num_edges]
    """
    # Get edge coordinates
    edge_list = np.array(np.nonzero(adjacency))

    # Convert to PyTorch tensor
    edge_index = torch.tensor(edge_list, dtype=torch.long)

    return edge_index
