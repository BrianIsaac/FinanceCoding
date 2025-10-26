#!/usr/bin/env python3
"""
Execute Universe Builder to create dynamic membership calendar.
This script implements Subtask 1.2 from Story 5.1.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.config.data import UniverseConfig
from src.data.processors.universe_builder import UniverseBuilder


def main():
    """Execute universe construction for S&P MidCap 400."""

    # Configure universe parameters from Story 5.1 specs
    universe_config = UniverseConfig(
        universe_type="midcap400",
        min_market_cap=None,
        min_avg_volume=None,
        exclude_sectors=None
    )

    # Initialize universe builder
    builder = UniverseBuilder(
        universe_config=universe_config,
        output_dir="data/processed"
    )

    # Define evaluation period from Story 5.1
    start_date = "2016-01-01"
    end_date = "2024-12-31"


    # Subtask 1.2: Build membership intervals
    membership_df = builder.build_membership_intervals(start_date, end_date)


    # Save membership intervals
    membership_path = "data/processed/universe_membership_intervals.csv"
    membership_df.to_csv(membership_path, index=False)

    # Subtask 1.2: Create monthly snapshots
    snapshots_df = builder.create_monthly_snapshots(membership_df, start_date, end_date)


    # Save monthly snapshots
    snapshots_path = "data/processed/universe_snapshots_monthly.csv"
    snapshots_df.to_csv(snapshots_path, index=False)

    # Quick validation



if __name__ == "__main__":
    main()
