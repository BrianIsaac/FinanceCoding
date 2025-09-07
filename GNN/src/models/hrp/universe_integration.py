"""
HRP integration utilities for dynamic universe management.

This module provides utilities to integrate HRP clustering with UniverseBuilder
for handling time-varying asset membership in portfolio construction.
"""

from __future__ import annotations

import pandas as pd

from src.data.processors.universe_builder import UniverseBuilder
from src.models.hrp.clustering import ClusteringConfig, HRPClustering


class HRPUniverseIntegration:
    """
    Integration layer for HRP clustering with dynamic universe management.

    Handles time-varying asset membership by combining HRP clustering
    with UniverseBuilder for dynamic S&P index constituents.
    """

    def __init__(
        self,
        universe_builder: UniverseBuilder,
        clustering_config: ClusteringConfig | None = None,
    ):
        """
        Initialize HRP universe integration.

        Args:
            universe_builder: UniverseBuilder instance for dynamic membership
            clustering_config: Configuration for HRP clustering
        """
        self.universe_builder = universe_builder
        self.clustering_config = clustering_config or ClusteringConfig()
        self.hrp_clustering = HRPClustering(self.clustering_config)

        # Cache for universe snapshots
        self._universe_cache: pd.DataFrame | None = None

    def get_universe_assets_for_date(
        self, date: pd.Timestamp, universe_calendar: pd.DataFrame | None = None
    ) -> list[str]:
        """
        Get active universe assets for a specific date.

        Args:
            date: Date for which to retrieve universe assets
            universe_calendar: Optional preloaded universe calendar

        Returns:
            List of asset tickers active on the specified date
        """
        if universe_calendar is None and self._universe_cache is not None:
            universe_calendar = self._universe_cache
        elif universe_calendar is None:
            raise ValueError("No universe calendar provided or cached")

        # Filter universe calendar for the specific date
        date_mask = universe_calendar["date"] <= date
        if date_mask.any():
            # Get the latest universe snapshot on or before the target date
            latest_date = universe_calendar[date_mask]["date"].max()
            assets = (
                universe_calendar[universe_calendar["date"] == latest_date]["ticker"]
                .unique()
                .tolist()
            )
            return sorted(assets)
        else:
            return []

    def align_returns_with_universe(
        self,
        returns: pd.DataFrame,
        date: pd.Timestamp,
        universe_calendar: pd.DataFrame | None = None,
        lookback_days: int = 756,
    ) -> pd.DataFrame:
        """
        Align returns data with universe membership for clustering.

        Args:
            returns: Historical returns DataFrame
            date: Target date for universe membership
            universe_calendar: Optional preloaded universe calendar
            lookback_days: Number of lookback days for clustering

        Returns:
            Returns DataFrame aligned with universe membership and sufficient history
        """
        # Get universe assets for the target date
        universe_assets = self.get_universe_assets_for_date(date, universe_calendar)

        # Calculate lookback window
        end_date = date
        start_date = date - pd.Timedelta(days=lookback_days)

        # Filter returns for the lookback period
        time_mask = (returns.index >= start_date) & (returns.index < end_date)
        period_returns = returns[time_mask]

        # Filter returns for universe assets (only those present in both)
        available_assets = [asset for asset in universe_assets if asset in period_returns.columns]
        aligned_returns = period_returns[available_assets]

        # Remove assets with insufficient data
        min_obs = self.clustering_config.min_observations
        sufficient_data_mask = aligned_returns.count() >= min_obs
        final_assets = sufficient_data_mask[sufficient_data_mask].index.tolist()

        return aligned_returns[final_assets]

    def validate_universe_clustering_feasibility(
        self,
        returns: pd.DataFrame,
        date: pd.Timestamp,
        universe_calendar: pd.DataFrame | None = None,
        min_assets_for_clustering: int = 10,
    ) -> tuple[bool, str, dict]:
        """
        Validate whether clustering is feasible for given universe and date.

        Args:
            returns: Historical returns DataFrame
            date: Target date for clustering
            universe_calendar: Optional preloaded universe calendar
            min_assets_for_clustering: Minimum number of assets required

        Returns:
            Tuple of (is_feasible, error_message, validation_metrics)
        """
        try:
            # Get aligned returns
            aligned_returns = self.align_returns_with_universe(returns, date, universe_calendar)

            n_assets = len(aligned_returns.columns)
            n_observations = len(aligned_returns)

            # Check minimum asset count
            if n_assets < min_assets_for_clustering:
                return (
                    False,
                    f"Insufficient assets: {n_assets} < {min_assets_for_clustering}",
                    {"n_assets": n_assets, "n_observations": n_observations},
                )

            # Check minimum observations
            if n_observations < self.clustering_config.min_observations:
                return (
                    False,
                    f"Insufficient observations: {n_observations} < {self.clustering_config.min_observations}",
                    {"n_assets": n_assets, "n_observations": n_observations},
                )

            # Check for excessive missing data
            missing_ratio = aligned_returns.isna().sum().sum() / aligned_returns.size
            if missing_ratio > 0.3:
                return (
                    False,
                    f"Too much missing data: {missing_ratio:.2%}",
                    {
                        "n_assets": n_assets,
                        "n_observations": n_observations,
                        "missing_ratio": missing_ratio,
                    },
                )

            # Calculate additional metrics
            metrics = {
                "n_assets": n_assets,
                "n_observations": n_observations,
                "missing_ratio": missing_ratio,
                "lookback_days": (aligned_returns.index.max() - aligned_returns.index.min()).days,
                "universe_coverage": len(
                    [
                        a
                        for a in self.get_universe_assets_for_date(date, universe_calendar)
                        if a in returns.columns
                    ]
                )
                / len(self.get_universe_assets_for_date(date, universe_calendar)),
            }

            return True, "Clustering feasible", metrics

        except Exception as e:
            return False, f"Validation error: {str(e)}", {}

    def build_universe_aware_clusters(
        self,
        returns: pd.DataFrame,
        date: pd.Timestamp,
        universe_calendar: pd.DataFrame | None = None,
    ) -> dict:
        """
        Build HRP clusters considering universe membership constraints.

        Args:
            returns: Historical returns DataFrame
            date: Target date for clustering
            universe_calendar: Optional preloaded universe calendar

        Returns:
            Dictionary containing cluster tree and metadata

        Raises:
            ValueError: If clustering is not feasible
        """
        # Validate feasibility
        is_feasible, error_msg, metrics = self.validate_universe_clustering_feasibility(
            returns, date, universe_calendar
        )

        if not is_feasible:
            raise ValueError(f"Clustering not feasible: {error_msg}")

        # Get aligned returns
        aligned_returns = self.align_returns_with_universe(returns, date, universe_calendar)

        # Build correlation distance matrix
        distance_matrix = self.hrp_clustering.build_correlation_distance(aligned_returns)

        # Perform hierarchical clustering
        linkage_matrix = self.hrp_clustering.hierarchical_clustering(distance_matrix)

        # Build cluster tree
        asset_names = aligned_returns.columns.tolist()
        cluster_tree = self.hrp_clustering.build_cluster_tree(asset_names, linkage_matrix)

        return {
            "cluster_tree": cluster_tree,
            "distance_matrix": distance_matrix,
            "linkage_matrix": linkage_matrix,
            "asset_names": asset_names,
            "clustering_date": date,
            "validation_metrics": metrics,
            "config": self.clustering_config,
        }

    def cache_universe_calendar(self, universe_calendar: pd.DataFrame) -> None:
        """Cache universe calendar for repeated use."""
        self._universe_cache = universe_calendar.copy()

    def clear_universe_cache(self) -> None:
        """Clear cached universe calendar."""
        self._universe_cache = None
