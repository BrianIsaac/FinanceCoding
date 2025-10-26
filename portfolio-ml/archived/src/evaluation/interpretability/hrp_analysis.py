"""
HRP Clustering Analysis and Visualization Framework.

This module provides interpretability tools for Hierarchical Risk Parity models,
including clustering analysis, dendrogram visualization, and allocation logic
explanation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster

from ...models.hrp.model import HRPModel


@dataclass
class HRPAnalysisConfig:
    """Configuration for HRP analysis and visualization."""

    dendrogram_levels: int = 10  # Maximum levels to show in dendrogram
    cluster_threshold: float = 0.5  # Threshold for flat clustering
    sector_mapping: dict[str, str] | None = None  # Asset to sector mapping
    show_allocation_splits: bool = True  # Show allocation at each split
    correlation_threshold: float = 0.3  # Minimum correlation to show in heatmap


class HRPAnalyzer:
    """
    HRP clustering and allocation analyzer.

    Provides tools for analyzing hierarchical clustering structure,
    correlation patterns, and allocation logic in HRP models.
    """

    def __init__(self, model: HRPModel, config: HRPAnalysisConfig | None = None):
        """
        Initialize HRP analyzer.

        Args:
            model: Trained HRP portfolio model
            config: Analysis configuration
        """
        self.model = model
        self.config = config or HRPAnalysisConfig()

        if not model.is_fitted:
            raise ValueError("HRP model must be fitted before analysis")

    def analyze_clustering_structure(
        self,
        returns: pd.DataFrame,
        universe: list[str],
    ) -> dict[str, Any]:
        """
        Analyze HRP clustering structure and hierarchy.

        Args:
            returns: Historical returns DataFrame
            universe: Asset universe for analysis

        Returns:
            Dictionary containing clustering analysis results
        """
        # Build correlation distance matrix
        clustering_engine = self.model.clustering_engine
        distance_matrix = clustering_engine.build_correlation_distance(returns[universe])

        # Perform hierarchical clustering
        linkage_matrix = clustering_engine.hierarchical_clustering(distance_matrix)

        # Create dendrogram data
        dendrogram_data = dendrogram(
            linkage_matrix,
            labels=universe,
            no_plot=True,
            truncate_mode='level',
            p=self.config.dendrogram_levels
        )

        # Perform flat clustering at threshold
        cluster_labels = fcluster(
            linkage_matrix,
            t=self.config.cluster_threshold,
            criterion='distance'
        )

        # Analyze cluster composition
        cluster_analysis = self._analyze_cluster_composition(
            universe, cluster_labels, dendrogram_data
        )

        # Calculate clustering quality metrics
        quality_metrics = self._calculate_clustering_quality(
            distance_matrix, cluster_labels
        )

        return {
            "distance_matrix": distance_matrix,
            "linkage_matrix": linkage_matrix,
            "dendrogram_data": dendrogram_data,
            "cluster_labels": cluster_labels,
            "cluster_analysis": cluster_analysis,
            "quality_metrics": quality_metrics,
            "asset_names": universe,
        }

    def analyze_correlation_patterns(
        self,
        returns: pd.DataFrame,
        universe: list[str],
    ) -> dict[str, Any]:
        """
        Analyze correlation patterns driving clustering decisions.

        Args:
            returns: Historical returns DataFrame
            universe: Asset universe

        Returns:
            Correlation analysis results
        """
        # Calculate correlation matrix
        correlation_matrix = returns[universe].corr()

        # Identify high correlation pairs
        high_corr_pairs = self._find_high_correlation_pairs(
            correlation_matrix, threshold=self.config.correlation_threshold
        )

        # Analyze correlation blocks
        correlation_blocks = self._identify_correlation_blocks(correlation_matrix)

        # Calculate correlation statistics
        corr_stats = self._calculate_correlation_statistics(correlation_matrix)

        return {
            "correlation_matrix": correlation_matrix,
            "high_correlation_pairs": high_corr_pairs,
            "correlation_blocks": correlation_blocks,
            "correlation_statistics": corr_stats,
        }

    def analyze_allocation_logic(
        self,
        returns: pd.DataFrame,
        universe: list[str],
        target_date: pd.Timestamp,
    ) -> dict[str, Any]:
        """
        Analyze HRP allocation logic and recursive bisection.

        Args:
            returns: Historical returns DataFrame
            universe: Asset universe
            target_date: Date for allocation analysis

        Returns:
            Allocation logic analysis results
        """
        # Get portfolio weights for target date
        portfolio_weights = self.model.predict_weights(target_date, universe)

        # Build clustering structure
        clustering_results = self.analyze_clustering_structure(returns, universe)

        # Trace allocation decisions through hierarchy
        allocation_tree = self._trace_allocation_decisions(
            clustering_results["linkage_matrix"],
            clustering_results["dendrogram_data"],
            portfolio_weights,
            universe,
        )

        # Analyze allocation concentration
        allocation_analysis = self._analyze_allocation_concentration(portfolio_weights)

        # Compare to equal-weight baseline
        baseline_comparison = self._compare_to_baseline(portfolio_weights)

        return {
            "portfolio_weights": portfolio_weights,
            "allocation_tree": allocation_tree,
            "allocation_analysis": allocation_analysis,
            "baseline_comparison": baseline_comparison,
            "clustering_rationale": clustering_results,
        }

    def analyze_sector_alignment(
        self,
        universe: list[str],
        cluster_labels: np.ndarray,
    ) -> dict[str, Any]:
        """
        Analyze alignment between clusters and fundamental sectors.

        Args:
            universe: Asset universe
            cluster_labels: Cluster assignments from hierarchical clustering

        Returns:
            Sector alignment analysis
        """
        if self.config.sector_mapping is None:
            # Create dummy sector mapping for demonstration
            sector_mapping = self._create_dummy_sector_mapping(universe)
        else:
            sector_mapping = self.config.sector_mapping

        # Analyze cluster-sector alignment
        alignment_analysis = self._analyze_cluster_sector_alignment(
            universe, cluster_labels, sector_mapping
        )

        # Calculate alignment metrics
        alignment_metrics = self._calculate_alignment_metrics(
            cluster_labels, sector_mapping, universe
        )

        return {
            "sector_mapping": sector_mapping,
            "alignment_analysis": alignment_analysis,
            "alignment_metrics": alignment_metrics,
        }

    def _analyze_cluster_composition(
        self,
        asset_names: list[str],
        cluster_labels: np.ndarray,
        dendrogram_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Analyze composition of each cluster."""
        cluster_composition = {}

        # Group assets by cluster
        unique_clusters = np.unique(cluster_labels)

        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_assets = [asset_names[i] for i in range(len(asset_names)) if cluster_mask[i]]

            cluster_composition[f"cluster_{cluster_id}"] = {
                "assets": cluster_assets,
                "size": len(cluster_assets),
                "percentage": len(cluster_assets) / len(asset_names) * 100,
            }

        return {
            "clusters": cluster_composition,
            "n_clusters": len(unique_clusters),
            "dendrogram_structure": dendrogram_data,
        }

    def _calculate_clustering_quality(
        self,
        distance_matrix: np.ndarray,
        cluster_labels: np.ndarray,
    ) -> dict[str, float]:
        """Calculate clustering quality metrics."""
        from sklearn.metrics import calinski_harabasz_score, silhouette_score

        # Convert distance matrix to feature matrix for sklearn metrics
        try:
            # Use MDS to convert distances to coordinates
            from sklearn.manifold import MDS

            mds = MDS(n_components=min(10, len(distance_matrix)),
                     dissimilarity='precomputed', random_state=42)
            coordinates = mds.fit_transform(distance_matrix)

            # Calculate clustering quality metrics
            silhouette = silhouette_score(coordinates, cluster_labels)
            calinski_harabasz = calinski_harabasz_score(coordinates, cluster_labels)

            # Calculate intra-cluster distances
            intra_cluster_dist = self._calculate_intra_cluster_distances(
                distance_matrix, cluster_labels
            )

            return {
                "silhouette_score": float(silhouette),
                "calinski_harabasz_score": float(calinski_harabasz),
                "avg_intra_cluster_distance": float(intra_cluster_dist),
                "n_clusters": len(np.unique(cluster_labels)),
            }

        except ImportError:
            # Fallback without sklearn
            intra_cluster_dist = self._calculate_intra_cluster_distances(
                distance_matrix, cluster_labels
            )

            return {
                "avg_intra_cluster_distance": float(intra_cluster_dist),
                "n_clusters": len(np.unique(cluster_labels)),
            }

    def _calculate_intra_cluster_distances(
        self,
        distance_matrix: np.ndarray,
        cluster_labels: np.ndarray,
    ) -> float:
        """Calculate average intra-cluster distances."""
        total_distance = 0.0
        total_pairs = 0

        unique_clusters = np.unique(cluster_labels)

        for cluster_id in unique_clusters:
            cluster_indices = np.where(cluster_labels == cluster_id)[0]

            if len(cluster_indices) > 1:
                # Calculate pairwise distances within cluster
                for i in range(len(cluster_indices)):
                    for j in range(i + 1, len(cluster_indices)):
                        idx_i, idx_j = cluster_indices[i], cluster_indices[j]
                        total_distance += distance_matrix[idx_i, idx_j]
                        total_pairs += 1

        return total_distance / total_pairs if total_pairs > 0 else 0.0

    def _find_high_correlation_pairs(
        self,
        correlation_matrix: pd.DataFrame,
        threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Find pairs of assets with high correlation."""
        high_corr_pairs = []

        for i in range(len(correlation_matrix)):
            for j in range(i + 1, len(correlation_matrix)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) >= threshold:
                    high_corr_pairs.append({
                        "asset1": correlation_matrix.index[i],
                        "asset2": correlation_matrix.index[j],
                        "correlation": float(corr),
                        "abs_correlation": float(abs(corr)),
                    })

        # Sort by absolute correlation
        high_corr_pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)

        return high_corr_pairs

    def _identify_correlation_blocks(
        self,
        correlation_matrix: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Identify blocks of highly correlated assets."""
        # Simple block identification based on correlation threshold
        blocks = []
        visited = set()

        for asset in correlation_matrix.index:
            if asset in visited:
                continue

            # Find all assets highly correlated with this one
            high_corr_assets = [asset]
            corr_values = []

            for other_asset in correlation_matrix.index:
                if other_asset != asset and other_asset not in visited:
                    corr = abs(correlation_matrix.loc[asset, other_asset])
                    if corr >= self.config.correlation_threshold:
                        high_corr_assets.append(other_asset)
                        corr_values.append(corr)

            if len(high_corr_assets) > 1:
                blocks.append({
                    "assets": high_corr_assets,
                    "size": len(high_corr_assets),
                    "avg_correlation": np.mean(corr_values) if corr_values else 0.0,
                    "min_correlation": min(corr_values) if corr_values else 0.0,
                    "max_correlation": max(corr_values) if corr_values else 0.0,
                })

                # Mark assets as visited
                visited.update(high_corr_assets)

        return blocks

    def _calculate_correlation_statistics(
        self,
        correlation_matrix: pd.DataFrame,
    ) -> dict[str, float]:
        """Calculate correlation matrix statistics."""
        # Remove diagonal elements
        corr_values = correlation_matrix.values
        mask = ~np.eye(corr_values.shape[0], dtype=bool)
        off_diagonal = corr_values[mask]

        return {
            "mean_correlation": float(np.mean(off_diagonal)),
            "std_correlation": float(np.std(off_diagonal)),
            "median_correlation": float(np.median(off_diagonal)),
            "min_correlation": float(np.min(off_diagonal)),
            "max_correlation": float(np.max(off_diagonal)),
            "mean_abs_correlation": float(np.mean(np.abs(off_diagonal))),
        }

    def _trace_allocation_decisions(
        self,
        linkage_matrix: np.ndarray,
        dendrogram_data: dict[str, Any],
        portfolio_weights: pd.Series,
        asset_names: list[str],
    ) -> dict[str, Any]:
        """Trace allocation decisions through the hierarchy."""
        # Build allocation tree from dendrogram structure
        allocation_tree = {
            "root": {
                "total_weight": 1.0,
                "assets": asset_names,
                "individual_weights": portfolio_weights.to_dict(),
            }
        }

        # Add hierarchical splits (simplified implementation)
        # In practice, this would trace through the actual HRP recursive bisection
        n_assets = len(asset_names)

        for i in range(len(linkage_matrix)):
            left_idx = int(linkage_matrix[i, 0])
            right_idx = int(linkage_matrix[i, 1])

            # Get assets in each cluster
            if left_idx < n_assets:
                left_assets = [asset_names[left_idx]]
            else:
                # This would be a more complex recursive lookup in practice
                left_assets = []

            if right_idx < n_assets:
                right_assets = [asset_names[right_idx]]
            else:
                right_assets = []

            if left_assets and right_assets:
                left_weight = sum(portfolio_weights.get(asset, 0) for asset in left_assets)
                right_weight = sum(portfolio_weights.get(asset, 0) for asset in right_assets)

                allocation_tree[f"split_{i}"] = {
                    "left_cluster": {
                        "assets": left_assets,
                        "total_weight": left_weight,
                    },
                    "right_cluster": {
                        "assets": right_assets,
                        "total_weight": right_weight,
                    },
                    "split_ratio": left_weight / (left_weight + right_weight) if (left_weight + right_weight) > 0 else 0.5,
                }

        return allocation_tree

    def _analyze_allocation_concentration(
        self,
        portfolio_weights: pd.Series,
    ) -> dict[str, Any]:
        """Analyze concentration of portfolio allocations."""
        weights_array = portfolio_weights.values

        # Calculate concentration metrics
        herfindahl_index = (weights_array ** 2).sum()
        effective_n_stocks = 1.0 / herfindahl_index if herfindahl_index > 0 else 0

        # Gini coefficient
        n = len(weights_array)
        sorted_weights = np.sort(weights_array)
        cumsum_weights = np.cumsum(sorted_weights)
        gini = (n + 1 - 2 * np.sum(cumsum_weights) / cumsum_weights[-1]) / n if cumsum_weights[-1] > 0 else 0

        # Top concentrations
        top_positions = portfolio_weights.nlargest(5)

        return {
            "herfindahl_hirschman_index": float(herfindahl_index),
            "effective_n_stocks": float(effective_n_stocks),
            "gini_coefficient": float(gini),
            "top_5_concentration": float(top_positions.sum()),
            "top_positions": top_positions.to_dict(),
            "max_weight": float(portfolio_weights.max()),
            "min_weight": float(portfolio_weights.min()),
            "weight_std": float(portfolio_weights.std()),
        }

    def _compare_to_baseline(
        self,
        portfolio_weights: pd.Series,
    ) -> dict[str, Any]:
        """Compare HRP allocation to equal-weight baseline."""
        n_assets = len(portfolio_weights)
        equal_weight = 1.0 / n_assets

        # Calculate differences
        weight_differences = portfolio_weights - equal_weight

        # Active share (sum of absolute differences / 2)
        active_share = weight_differences.abs().sum() / 2.0

        # Tracking error components
        overweights = weight_differences[weight_differences > 0]
        underweights = weight_differences[weight_differences < 0]

        return {
            "equal_weight_baseline": equal_weight,
            "active_share": float(active_share),
            "max_overweight": float(overweights.max()) if len(overweights) > 0 else 0.0,
            "max_underweight": float(abs(underweights.min())) if len(underweights) > 0 else 0.0,
            "n_overweight": len(overweights),
            "n_underweight": len(underweights),
            "overweight_concentration": float(overweights.sum()) if len(overweights) > 0 else 0.0,
        }

    def _create_dummy_sector_mapping(self, universe: list[str]) -> dict[str, str]:
        """Create dummy sector mapping for assets."""
        # Simple heuristic-based sector assignment
        sectors = {
            "Technology": ["AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "NVDA"],
            "Finance": ["JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA"],
            "Healthcare": ["JNJ", "PFE", "UNH", "CVX", "ABBV", "TMO", "DHR", "ABT"],
            "Consumer": ["PG", "KO", "PEP", "WMT", "HD", "MCD", "NKE", "SBUX"],
            "Industrial": ["BA", "CAT", "GE", "MMM", "UTX", "HON", "LMT", "RTX"],
        }

        # Reverse mapping
        asset_to_sector = {}
        for sector, assets in sectors.items():
            for asset in assets:
                if asset in universe:
                    asset_to_sector[asset] = sector

        # Assign remaining assets to "Other"
        for asset in universe:
            if asset not in asset_to_sector:
                asset_to_sector[asset] = "Other"

        return asset_to_sector

    def _analyze_cluster_sector_alignment(
        self,
        universe: list[str],
        cluster_labels: np.ndarray,
        sector_mapping: dict[str, str],
    ) -> dict[str, Any]:
        """Analyze alignment between clusters and sectors."""
        alignment_results = {}

        unique_clusters = np.unique(cluster_labels)

        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_assets = [universe[i] for i in range(len(universe)) if cluster_mask[i]]

            # Count sectors in this cluster
            sector_counts = {}
            for asset in cluster_assets:
                sector = sector_mapping.get(asset, "Unknown")
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

            # Find dominant sector
            dominant_sector = max(sector_counts, key=sector_counts.get) if sector_counts else "Unknown"
            sector_purity = sector_counts.get(dominant_sector, 0) / len(cluster_assets)

            alignment_results[f"cluster_{cluster_id}"] = {
                "assets": cluster_assets,
                "sector_counts": sector_counts,
                "dominant_sector": dominant_sector,
                "sector_purity": sector_purity,
                "size": len(cluster_assets),
            }

        return alignment_results

    def _calculate_alignment_metrics(
        self,
        cluster_labels: np.ndarray,
        sector_mapping: dict[str, str],
        universe: list[str],
    ) -> dict[str, float]:
        """Calculate overall alignment metrics."""
        # Calculate adjusted rand index or similar metrics
        # Simplified implementation

        total_assets = len(universe)
        correctly_clustered = 0

        unique_clusters = np.unique(cluster_labels)

        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_assets = [universe[i] for i in range(len(universe)) if cluster_mask[i]]

            if len(cluster_assets) > 1:
                # Check if assets in cluster are from same sector
                sectors = [sector_mapping.get(asset, "Unknown") for asset in cluster_assets]
                dominant_sector = max(set(sectors), key=sectors.count)
                same_sector_count = sectors.count(dominant_sector)
                correctly_clustered += same_sector_count

        alignment_score = correctly_clustered / total_assets if total_assets > 0 else 0.0

        return {
            "alignment_score": alignment_score,
            "correctly_clustered_assets": correctly_clustered,
            "total_assets": total_assets,
        }
