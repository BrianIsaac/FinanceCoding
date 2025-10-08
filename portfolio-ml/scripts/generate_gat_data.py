#!/usr/bin/env python3
"""
GAT Data Generation Pipeline.

Generates graph snapshots and labels for GAT model training from existing returns data.
Creates the required data/graphs/ and data/labels/ directories for GAT training.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.gat.graph_builder import GraphBuildConfig, build_period_graph

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GATDataGenerator:
    """
    Generate graphs and labels for GAT training from returns data.
    """

    def __init__(
        self,
        returns_path: str | Path = "data/final_new_pipeline/returns_daily_final.parquet",
        output_base: str | Path = "data",
        lookback_days: int = 252,
        rebalance_freq: str = "monthly",
    ):
        """
        Initialize GAT data generator.

        Args:
            returns_path: Path to daily returns parquet file
            output_base: Base directory for data/graphs and data/labels
            lookback_days: Lookback period for graph construction
            rebalance_freq: Rebalancing frequency (monthly, weekly)
        """
        self.returns_path = Path(returns_path)
        self.output_base = Path(output_base)
        self.lookback_days = lookback_days
        self.rebalance_freq = rebalance_freq

        # Create output directories
        self.graphs_dir = self.output_base / "graphs"
        self.labels_dir = self.output_base / "labels"
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        # Load returns data
        self.returns_data = self._load_returns_data()

        # Select universe (limit for memory efficiency)
        self.universe = self._select_universe()

        logger.info("GAT data generator initialized")
        logger.info(f"Returns data shape: {self.returns_data.shape}")
        logger.info(f"Universe size: {len(self.universe)} assets")
        logger.info(f"Date range: {self.returns_data.index[0]} to {self.returns_data.index[-1]}")

    def _load_returns_data(self) -> pd.DataFrame:
        """Load and validate returns data."""
        if not self.returns_path.exists():
            raise FileNotFoundError(f"Returns data not found at {self.returns_path}")

        returns = pd.read_parquet(self.returns_path)
        logger.info(f"Loaded returns data: {returns.shape}")

        # Ensure datetime index
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns.index = pd.to_datetime(returns.index)

        # Sort by date
        returns = returns.sort_index()

        # Fill NaN with 0 (conservative)
        returns = returns.fillna(0.0)

        return returns

    def _select_universe(self, max_assets: int = 200) -> list[str]:
        """
        Select universe based on data completeness and trading activity.

        Args:
            max_assets: Maximum number of assets to include

        Returns:
            List of selected tickers
        """
        # Calculate data completeness
        completeness = self.returns_data.count() / len(self.returns_data)

        # Calculate trading activity (non-zero returns)
        activity = (self.returns_data != 0).sum() / len(self.returns_data)

        # Calculate volatility (proxy for liquidity)
        volatility = self.returns_data.std()

        # Combined score (completeness * activity * volatility)
        scores = completeness * activity * volatility
        scores = scores.fillna(0)

        # Select top assets
        selected = scores.nlargest(max_assets).index.tolist()

        logger.info(
            f"Selected {len(selected)} assets from {len(self.returns_data.columns)} available"
        )
        logger.info(f"Average completeness: {completeness[selected].mean():.2%}")
        logger.info(f"Average activity: {activity[selected].mean():.2%}")

        return selected

    def _generate_rebalance_dates(self) -> list[pd.Timestamp]:
        """Generate rebalancing dates based on frequency."""
        start_date = self.returns_data.index[self.lookback_days]  # Ensure enough lookback
        end_date = self.returns_data.index[-2]  # Leave one day for labels

        if self.rebalance_freq == "monthly":
            # End of each month
            dates = pd.date_range(start_date, end_date, freq="M")
        elif self.rebalance_freq == "weekly":
            # End of each week (Friday)
            dates = pd.date_range(start_date, end_date, freq="W-FRI")
        else:
            raise ValueError(f"Unsupported rebalance frequency: {self.rebalance_freq}")

        # Filter to actual trading days
        valid_dates = []
        for date in dates:
            # Find the closest trading day (on or before the target date)
            mask = self.returns_data.index <= date
            if mask.any():
                actual_date = self.returns_data.index[mask][-1]
                valid_dates.append(actual_date)

        return valid_dates

    def _generate_labels(self, date: pd.Timestamp, next_period_days: int = 21) -> pd.DataFrame:
        """
        Generate labels (next-period returns) for given date.

        Args:
            date: Current rebalance date
            next_period_days: Forward-looking period for returns (21 = ~1 month)

        Returns:
            DataFrame with r_next column
        """
        try:
            # Find date indices
            date_idx = self.returns_data.index.get_loc(date)

            # Calculate forward returns
            end_idx = min(date_idx + next_period_days, len(self.returns_data) - 1)

            if end_idx <= date_idx:
                # Not enough forward data
                r_next = pd.Series(0.0, index=self.universe)
            else:
                # Cumulative returns over the next period
                future_returns = self.returns_data.iloc[date_idx + 1 : end_idx + 1][self.universe]
                r_next = (1 + future_returns).prod() - 1  # Compound returns
                r_next = r_next.fillna(0.0)

            # Create labels DataFrame
            labels = pd.DataFrame({"r_next": r_next})

            return labels

        except Exception as e:
            logger.error(f"Error generating labels for {date}: {e}")
            # Return zero labels as fallback
            return pd.DataFrame({"r_next": pd.Series(0.0, index=self.universe)})

    def _generate_graph(self, date: pd.Timestamp, method: str = "mst") -> Any:
        """
        Generate graph for given date using specified method.

        Args:
            date: Rebalance date
            method: Graph construction method (mst, tmfg, knn, threshold)

        Returns:
            torch_geometric Data object
        """
        try:
            # Create graph build configuration
            if method.startswith("knn"):
                # Extract k value from method name (e.g., "knn_10")
                k = int(method.split("_")[-1]) if "_" in method else 10
                cfg = GraphBuildConfig(
                    lookback_days=self.lookback_days,
                    filter_method="knn",
                    knn_k=k,
                    use_edge_attr=True,
                    enable_caching=True,
                )
            else:
                cfg = GraphBuildConfig(
                    lookback_days=self.lookback_days,
                    filter_method=method,
                    use_edge_attr=True,
                    enable_caching=True,
                )

            # Build graph
            graph = build_period_graph(
                returns_daily=self.returns_data,
                period_end=date,
                tickers=self.universe,
                features_matrix=None,  # Use default constant features
                cfg=cfg,
            )

            return graph

        except Exception as e:
            logger.error(f"Error generating {method} graph for {date}: {e}")
            return None

    def generate_single_method_data(
        self, method: str = "mst", max_periods: int | None = None
    ) -> dict[str, Any]:
        """
        Generate graphs and labels for a single graph construction method.

        Args:
            method: Graph construction method
            max_periods: Maximum number of periods to generate (None for all)

        Returns:
            Dictionary with generation statistics
        """
        logger.info(f"Starting {method} data generation")

        rebalance_dates = self._generate_rebalance_dates()
        if max_periods:
            rebalance_dates = rebalance_dates[:max_periods]

        stats = {
            "method": method,
            "total_periods": len(rebalance_dates),
            "successful_graphs": 0,
            "successful_labels": 0,
            "failed_periods": [],
        }

        for i, date in enumerate(rebalance_dates):
            logger.info(
                f"Processing {method} - Period {i + 1}/{len(rebalance_dates)}: {date.strftime('%Y-%m-%d')}"
            )

            try:
                # Generate graph
                graph = self._generate_graph(date, method)
                if graph is not None:
                    # Save graph
                    graph_path = self.graphs_dir / f"graph_{date.strftime('%Y-%m-%d')}.pt"
                    torch.save(graph, graph_path)
                    stats["successful_graphs"] += 1
                    logger.debug(f"Saved graph to {graph_path}")
                else:
                    stats["failed_periods"].append(f"{date}: graph generation failed")
                    continue

                # Generate labels
                labels = self._generate_labels(date)
                labels_path = self.labels_dir / f"labels_{date.strftime('%Y-%m-%d')}.parquet"
                labels.to_parquet(labels_path)
                stats["successful_labels"] += 1
                logger.debug(f"Saved labels to {labels_path}")

            except Exception as e:
                error_msg = f"{date}: {str(e)}"
                stats["failed_periods"].append(error_msg)
                logger.error(f"Failed to process period {date}: {e}")

        logger.info(f"Completed {method} data generation:")
        logger.info(f"  Successful graphs: {stats['successful_graphs']}/{stats['total_periods']}")
        logger.info(f"  Successful labels: {stats['successful_labels']}/{stats['total_periods']}")
        logger.info(f"  Failed periods: {len(stats['failed_periods'])}")

        return stats

    def generate_all_methods_data(
        self, methods: list[str] | None = None, max_periods: int | None = None
    ) -> dict[str, Any]:
        """
        Generate data for multiple graph construction methods.

        Args:
            methods: List of methods to generate (None for default set)
            max_periods: Maximum periods per method

        Returns:
            Combined statistics
        """
        if methods is None:
            methods = ["mst", "tmfg", "knn_5", "knn_10", "knn_15", "threshold"]

        logger.info(f"Starting comprehensive data generation for methods: {methods}")

        all_stats = {}
        for method in methods:
            try:
                stats = self.generate_single_method_data(method, max_periods)
                all_stats[method] = stats
            except Exception as e:
                logger.error(f"Failed to generate data for method {method}: {e}")
                all_stats[method] = {"error": str(e)}

        # Summary
        total_graphs = sum(s.get("successful_graphs", 0) for s in all_stats.values())
        total_labels = sum(s.get("successful_labels", 0) for s in all_stats.values())

        logger.info("=" * 50)
        logger.info("COMPREHENSIVE DATA GENERATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total graphs generated: {total_graphs}")
        logger.info(f"Total labels generated: {total_labels}")
        logger.info(f"Graph files location: {self.graphs_dir}")
        logger.info(f"Label files location: {self.labels_dir}")

        return all_stats


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate GAT training data")
    parser.add_argument("--method", type=str, default="mst", help="Graph construction method")
    parser.add_argument("--all-methods", action="store_true", help="Generate data for all methods")
    parser.add_argument("--max-periods", type=int, default=None, help="Maximum periods to generate")
    parser.add_argument(
        "--lookback-days", type=int, default=252, help="Lookback days for graph construction"
    )
    parser.add_argument(
        "--rebalance-freq",
        type=str,
        default="monthly",
        choices=["monthly", "weekly"],
        help="Rebalancing frequency",
    )

    args = parser.parse_args()

    # Initialize generator
    generator = GATDataGenerator(
        lookback_days=args.lookback_days, rebalance_freq=args.rebalance_freq
    )

    # Generate data
    if args.all_methods:
        generator.generate_all_methods_data(max_periods=args.max_periods)
    else:
        generator.generate_single_method_data(args.method, args.max_periods)


if __name__ == "__main__":
    main()
