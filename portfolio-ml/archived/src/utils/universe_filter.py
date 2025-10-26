"""
Universe filtering utilities for model training with time-varying membership.

This module provides utilities to correctly filter assets based on actual
index membership during training periods, eliminating survivorship bias.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class UniverseFilter:
    """
    Utility for filtering assets based on time-varying universe membership.

    Ensures models are trained only on assets that were actually in the index
    during the specific training period, eliminating survivorship bias.
    """

    def __init__(self, universe_membership_path: str | Path = "data/processed/universe_membership_clean.csv"):
        """
        Initialize universe filter.

        Args:
            universe_membership_path: Path to universe membership CSV file
        """
        self.membership_path = Path(universe_membership_path)
        self._membership_data: pd.DataFrame | None = None

    def _load_membership_data(self) -> pd.DataFrame:
        """Load and cache universe membership data."""
        if self._membership_data is None:
            if not self.membership_path.exists():
                raise FileNotFoundError(f"Universe membership file not found: {self.membership_path}")

            self._membership_data = pd.read_csv(self.membership_path)
            self._membership_data["start"] = pd.to_datetime(self._membership_data["start"])
            self._membership_data["end"] = pd.to_datetime(self._membership_data["end"])

        return self._membership_data

    def get_universe_for_period(
        self,
        start_date: pd.Timestamp | str,
        end_date: pd.Timestamp | str,
        max_assets: int | None = None
    ) -> list[str]:
        """
        Get assets that were in the universe during the specified period.

        Args:
            start_date: Start of training period
            end_date: End of training period
            max_assets: Maximum number of assets to return (for memory constraints)

        Returns:
            List of asset tickers that were active during the period
        """
        membership_df = self._load_membership_data()

        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)

        # Find assets that were active during any part of the training period
        active_mask = (
            (membership_df["start"] <= end_ts) &
            (
                (membership_df["end"].isna()) |
                (membership_df["end"] >= start_ts)
            )
        )

        active_assets = membership_df[active_mask]["ticker"].unique().tolist()

        if max_assets and len(active_assets) > max_assets:
            # Sort by membership duration (longer membership = more stable)
            asset_durations = {}
            for asset in active_assets:
                asset_records = membership_df[membership_df["ticker"] == asset]
                total_duration = 0
                for _, record in asset_records.iterrows():
                    record_start = max(record["start"], start_ts)
                    record_end = min(record["end"] if pd.notna(record["end"]) else end_ts, end_ts)
                    if record_end > record_start:
                        total_duration += (record_end - record_start).days
                asset_durations[asset] = total_duration

            # Select assets with longest membership duration
            sorted_assets = sorted(asset_durations.items(), key=lambda x: x[1], reverse=True)
            active_assets = [asset for asset, _ in sorted_assets[:max_assets]]

        return active_assets

    def get_universe_for_date(self, date: pd.Timestamp | str) -> list[str]:
        """
        Get assets that were in the universe on a specific date.

        Args:
            date: Specific date to check

        Returns:
            List of asset tickers active on that date
        """
        membership_df = self._load_membership_data()
        date_ts = pd.to_datetime(date)

        active_mask = (
            (membership_df["start"] <= date_ts) &
            (
                (membership_df["end"].isna()) |
                (membership_df["end"] > date_ts)
            )
        )

        return membership_df[active_mask]["ticker"].unique().tolist()

    def filter_returns_data(
        self,
        returns_data: pd.DataFrame,
        start_date: pd.Timestamp | str,
        end_date: pd.Timestamp | str,
        max_assets: int | None = None
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Filter returns data to only include assets active during the period.

        Args:
            returns_data: Full returns DataFrame
            start_date: Start of training period
            end_date: End of training period
            max_assets: Maximum number of assets to return

        Returns:
            Tuple of (filtered_returns, universe_list)
        """
        universe = self.get_universe_for_period(start_date, end_date, max_assets)

        # Filter to assets that exist in both universe and returns data
        available_universe = [asset for asset in universe if asset in returns_data.columns]

        if not available_universe:
            raise ValueError(f"No universe assets found in returns data columns")

        filtered_returns = returns_data[available_universe].copy()

        return filtered_returns, available_universe

    def get_membership_stats(self, start_date: str, end_date: str) -> dict[str, Any]:
        """Get statistics about universe membership during a period."""
        membership_df = self._load_membership_data()

        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)

        # Assets active during period
        active_assets = self.get_universe_for_period(start_ts, end_ts)

        # Assets that joined during period
        joined_mask = (
            (membership_df["start"] >= start_ts) &
            (membership_df["start"] <= end_ts)
        )
        joined_assets = membership_df[joined_mask]["ticker"].unique()

        # Assets that left during period
        left_mask = (
            (membership_df["end"] >= start_ts) &
            (membership_df["end"] <= end_ts) &
            membership_df["end"].notna()
        )
        left_assets = membership_df[left_mask]["ticker"].unique()

        return {
            "period": f"{start_date} to {end_date}",
            "total_active_assets": len(active_assets),
            "assets_joined": len(joined_assets),
            "assets_left": len(left_assets),
            "turnover_rate": (len(joined_assets) + len(left_assets)) / len(active_assets) if active_assets else 0,
            "active_assets": active_assets,
            "joined_assets": joined_assets.tolist(),
            "left_assets": left_assets.tolist(),
        }