"""
HRP clustering implementation using correlation distance matrices.

This module implements hierarchical clustering for portfolio construction using
correlation-based distance metrics and configurable linkage methods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


@dataclass
class ClusteringConfig:
    """Configuration for HRP clustering algorithm."""

    linkage_method: str = "single"
    min_observations: int = 252
    correlation_method: str = "pearson"
    min_correlation_threshold: float | None = None
    enable_quasi_diagonalization: bool = False
    memory_efficient: bool = True
    chunk_size: int = 100


class HRPClustering:
    """
    Hierarchical clustering implementation for HRP portfolio construction.

    Implements correlation distance-based hierarchical clustering with
    configurable linkage methods for asset hierarchy construction.
    """

    def __init__(self, config: ClusteringConfig | None = None):
        """
        Initialize HRP clustering engine.

        Args:
            config: Clustering configuration parameters
        """
        self.config = config or ClusteringConfig()
        self._linkage_matrix: np.ndarray | None = None
        self._distance_matrix: np.ndarray | None = None

    def build_correlation_distance(
        self, returns: pd.DataFrame, method: str | None = None
    ) -> np.ndarray:
        """
        Convert correlation matrix to distance metric using (1 - correlation)/2.

        Args:
            returns: Historical returns DataFrame with datetime index and asset columns
            method: Correlation method override (pearson, spearman, kendall)

        Returns:
            Distance matrix as 2D numpy array

        Raises:
            ValueError: If returns data is insufficient
        """
        correlation_method = method or self.config.correlation_method

        # Validate input data
        if returns.empty or returns.isna().all().all():
            raise ValueError("Returns data is empty")

        if len(returns) < self.config.min_observations:
            raise ValueError(
                f"Insufficient observations: {len(returns)} < {self.config.min_observations}"
            )

        # Calculate correlation matrix
        correlation_matrix = returns.corr(method=correlation_method)

        # Handle NaN values in correlation matrix
        correlation_matrix = correlation_matrix.fillna(0.0)

        # Apply minimum correlation threshold if specified
        if self.config.min_correlation_threshold is not None:
            mask = np.abs(correlation_matrix) < self.config.min_correlation_threshold
            correlation_matrix = correlation_matrix.where(~mask, 0.0)

        # Convert to distance metric: (1 - correlation) / 2
        distance_matrix = (1.0 - correlation_matrix.values) / 2.0

        # Ensure distance matrix is symmetric and positive semi-definite
        distance_matrix = np.maximum(distance_matrix, distance_matrix.T)
        np.fill_diagonal(distance_matrix, 0.0)

        # Clamp distances to valid range [0, 1]
        distance_matrix = np.clip(distance_matrix, 0.0, 1.0)

        self._distance_matrix = distance_matrix
        return distance_matrix

    def hierarchical_clustering(
        self, distance_matrix: np.ndarray | None = None, linkage_method: str | None = None
    ) -> np.ndarray:
        """
        Build asset hierarchy using correlation distances.

        Args:
            distance_matrix: Precomputed distance matrix (uses cached if None)
            linkage_method: Linkage method override (single, complete, average, ward)

        Returns:
            Linkage matrix for hierarchical clustering

        Raises:
            ValueError: If no distance matrix is available
        """
        # Use provided distance matrix or cached version
        if distance_matrix is None:
            if self._distance_matrix is None:
                raise ValueError(
                    "No distance matrix available. Call build_correlation_distance first."
                )
            distance_matrix = self._distance_matrix

        # Get linkage method
        method = linkage_method or self.config.linkage_method

        # Validate linkage method
        valid_methods = ["single", "complete", "average", "ward", "centroid", "median"]
        if method not in valid_methods:
            raise ValueError(f"Invalid linkage method: {method}. Must be one of {valid_methods}")

        # Convert distance matrix to condensed form for scipy
        try:
            condensed_distances = squareform(distance_matrix, checks=False)
        except ValueError as e:
            raise ValueError(f"Invalid distance matrix format: {e}")

        # Perform hierarchical clustering
        linkage_matrix = linkage(condensed_distances, method=method)

        self._linkage_matrix = linkage_matrix
        return linkage_matrix

    def build_cluster_tree(
        self, asset_names: list[str], linkage_matrix: np.ndarray | None = None
    ) -> dict[str, any]:
        """
        Create cluster tree structure for recursive bisection algorithm.

        Args:
            asset_names: List of asset identifiers
            linkage_matrix: Precomputed linkage matrix (uses cached if None)

        Returns:
            Dictionary representing cluster tree structure
        """
        # Use provided linkage matrix or cached version
        if linkage_matrix is None:
            if self._linkage_matrix is None:
                raise ValueError("No linkage matrix available. Call hierarchical_clustering first.")
            linkage_matrix = self._linkage_matrix

        n_assets = len(asset_names)

        if linkage_matrix.shape[0] != n_assets - 1:
            raise ValueError(
                f"Linkage matrix shape {linkage_matrix.shape} incompatible with {n_assets} assets"
            )

        # Build cluster tree recursively
        def _build_node(node_id: int) -> dict:
            """Recursively build cluster tree node."""
            if node_id < n_assets:
                # Leaf node (single asset)
                return {
                    "id": node_id,
                    "type": "leaf",
                    "asset": asset_names[node_id],
                    "assets": [asset_names[node_id]],
                    "distance": 0.0,
                }
            else:
                # Internal node (cluster)
                cluster_idx = node_id - n_assets
                left_id = int(linkage_matrix[cluster_idx, 0])
                right_id = int(linkage_matrix[cluster_idx, 1])
                distance = linkage_matrix[cluster_idx, 2]

                left_node = _build_node(left_id)
                right_node = _build_node(right_id)

                return {
                    "id": node_id,
                    "type": "cluster",
                    "left": left_node,
                    "right": right_node,
                    "assets": left_node["assets"] + right_node["assets"],
                    "distance": distance,
                }

        # Root node is the last cluster
        root_id = n_assets + linkage_matrix.shape[0] - 1
        return _build_node(root_id)

    def get_cluster_assets(self, cluster_tree: dict, target_assets: list[str]) -> list[str]:
        """
        Extract assets from cluster tree that match target asset list.

        Args:
            cluster_tree: Cluster tree dictionary
            target_assets: List of assets to filter for

        Returns:
            Filtered list of assets present in both tree and target list
        """
        tree_assets = cluster_tree.get("assets", [])
        return [asset for asset in tree_assets if asset in target_assets]

    def validate_correlation_matrix(self, correlation_matrix: pd.DataFrame) -> tuple[bool, str]:
        """
        Validate correlation matrix for clustering suitability.

        Args:
            correlation_matrix: Correlation matrix to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if matrix is square
            if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
                return False, "Correlation matrix must be square"

            # Check diagonal elements
            diagonal = np.diag(correlation_matrix.values)
            if not np.allclose(diagonal, 1.0, atol=1e-6):
                return False, "Correlation matrix diagonal must be 1.0"

            # Check if matrix is symmetric
            if not np.allclose(correlation_matrix.values, correlation_matrix.values.T, atol=1e-6):
                return False, "Correlation matrix must be symmetric"

            # Check for valid correlation range [-1, 1]
            values = correlation_matrix.values
            if np.any(values < -1.0) or np.any(values > 1.0):
                return False, "Correlation values must be in range [-1, 1]"

            # Check for excessive NaN values
            nan_ratio = correlation_matrix.isna().sum().sum() / correlation_matrix.size
            if nan_ratio > 0.5:
                return False, "Too many NaN values in correlation matrix"

            return True, "Valid correlation matrix"

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def quasi_diagonalize_correlation(
        self, correlation_matrix: pd.DataFrame, linkage_matrix: np.ndarray
    ) -> pd.DataFrame:
        """
        Apply quasi-diagonalization to improve clustering stability.

        Reorders correlation matrix based on hierarchical clustering results
        to create a quasi-diagonal structure that enhances numerical stability.

        Args:
            correlation_matrix: Original correlation matrix
            linkage_matrix: Hierarchical clustering linkage matrix

        Returns:
            Quasi-diagonalized correlation matrix with reordered assets
        """
        if not self.config.enable_quasi_diagonalization:
            return correlation_matrix

        try:
            from scipy.cluster.hierarchy import leaves_list

            # Get the leaf order from hierarchical clustering
            leaf_order = leaves_list(linkage_matrix)

            # Reorder correlation matrix based on clustering
            asset_names = correlation_matrix.index.tolist()
            reordered_assets = [asset_names[i] for i in leaf_order]

            # Create reordered correlation matrix
            quasi_diag_matrix = correlation_matrix.loc[reordered_assets, reordered_assets]

            return quasi_diag_matrix

        except Exception:
            # Return original matrix if quasi-diagonalization fails
            return correlation_matrix

    def build_correlation_memory_efficient(
        self, returns: pd.DataFrame, method: str | None = None
    ) -> pd.DataFrame:
        """
        Memory-efficient correlation matrix calculation for large universes.

        Uses chunked processing to handle large correlation matrices without
        excessive memory usage.

        Args:
            returns: Historical returns DataFrame
            method: Correlation method override

        Returns:
            Correlation matrix as pandas DataFrame
        """
        if not self.config.memory_efficient or len(returns.columns) <= self.config.chunk_size:
            # Use standard correlation calculation for small datasets
            return returns.corr(method=method or self.config.correlation_method)

        correlation_method = method or self.config.correlation_method
        n_assets = len(returns.columns)
        asset_names = returns.columns.tolist()

        # Initialize correlation matrix
        correlation_matrix = pd.DataFrame(np.eye(n_assets), index=asset_names, columns=asset_names)

        # Process in chunks to manage memory
        chunk_size = self.config.chunk_size

        for i in range(0, n_assets, chunk_size):
            end_i = min(i + chunk_size, n_assets)
            chunk_i_assets = asset_names[i:end_i]
            chunk_i_returns = returns[chunk_i_assets]

            for j in range(i, n_assets, chunk_size):
                end_j = min(j + chunk_size, n_assets)
                chunk_j_assets = asset_names[j:end_j]
                chunk_j_returns = returns[chunk_j_assets]

                # Calculate correlation for this chunk pair
                if i == j:
                    # Diagonal chunk - calculate correlation within chunk
                    chunk_corr = chunk_i_returns.corr(method=correlation_method)
                    correlation_matrix.loc[chunk_i_assets, chunk_i_assets] = chunk_corr
                else:
                    # Off-diagonal chunk - calculate cross-correlation
                    combined_returns = pd.concat([chunk_i_returns, chunk_j_returns], axis=1)
                    combined_corr = combined_returns.corr(method=correlation_method)

                    # Extract cross-correlation blocks
                    cross_corr_ij = combined_corr.loc[chunk_i_assets, chunk_j_assets]
                    cross_corr_ji = combined_corr.loc[chunk_j_assets, chunk_i_assets]

                    # Fill symmetric positions
                    correlation_matrix.loc[chunk_i_assets, chunk_j_assets] = cross_corr_ij
                    correlation_matrix.loc[chunk_j_assets, chunk_i_assets] = cross_corr_ji

        return correlation_matrix
