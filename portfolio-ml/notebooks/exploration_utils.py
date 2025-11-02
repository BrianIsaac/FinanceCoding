"""Utilities for membership-aware data exploration and quality profiling.

This module provides reusable functions for exploring financial time series data
with proper respect for universe membership periods. ALL analysis functions ensure
that statistics are calculated only during active membership periods.

===== CRITICAL PRINCIPLE =====
Every function that analyses price/volume data MUST use membership information to
avoid incorrectly treating expected empty cells as "missing data".

DO NOT calculate statistics based on the date range in the raw parquet files.
The data was collected to follow membership activity, so ALL analysis must
respect when tickers were actually active members of the universe.

Use get_membership_mask() and filter_to_membership_periods() to ensure this.
===============================
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def load_raw_data(
    raw_data_dir: Path = Path("data/final_new_pipeline/raw"),
    membership_path: Path = Path("data/processed/universe_calendar_midcap400.parquet"),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw prices, volumes, source mapping, and universe membership.

    Args:
        raw_data_dir: Directory containing raw parquet files
        membership_path: Path to universe calendar parquet

    Returns:
        Tuple of (prices_df, volumes_df, source_map_df, membership_df)
    """
    prices = pd.read_parquet(raw_data_dir / "prices_raw.parquet")
    volumes = pd.read_parquet(raw_data_dir / "volumes_raw.parquet")
    source_map = pd.read_parquet(raw_data_dir / "source_mapping.parquet")
    membership = pd.read_parquet(membership_path)

    # Ensure datetime indices
    prices.index = pd.to_datetime(prices.index)
    volumes.index = pd.to_datetime(volumes.index)
    membership['date'] = pd.to_datetime(membership['date'])

    return prices, volumes, source_map, membership


def get_membership_mask(
    prices: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    """Create a boolean mask indicating which cells should have data based on membership.

    This is the FOUNDATION for all membership-aware analysis. A cell is True if that
    ticker was an active member on that date according to the membership calendar.

    Args:
        prices: Prices DataFrame with DatetimeIndex and ticker columns
        membership: Universe calendar with 'date' and 'ticker' columns

    Returns:
        Boolean DataFrame with same shape as prices, True where ticker was active
    """
    # Create empty mask
    mask = pd.DataFrame(False, index=prices.index, columns=prices.columns)

    # For each monthly membership snapshot, mark active tickers
    for date in membership['date'].unique():
        active_tickers = membership[membership['date'] == date]['ticker'].values

        # Find the date range this snapshot covers (until next snapshot or end of data)
        next_dates = membership['date'].unique()
        next_dates = next_dates[next_dates > date]
        end_date = next_dates.min() if len(next_dates) > 0 else prices.index.max()

        # Mark all dates from this snapshot to next as active for these tickers
        date_mask = (prices.index >= date) & (prices.index < end_date)
        mask.loc[date_mask, active_tickers] = True

    return mask


def filter_to_membership_periods(
    df: pd.DataFrame,
    membership_mask: pd.DataFrame,
) -> pd.DataFrame:
    """Filter DataFrame to only include data during membership periods.

    Sets all cells outside membership periods to NaN. This ensures that any
    subsequent analysis (mean, std, quantiles, etc.) ONLY uses data during
    active membership.

    Args:
        df: Prices or volumes DataFrame
        membership_mask: Boolean mask from get_membership_mask()

    Returns:
        Filtered DataFrame with NaN outside membership periods
    """
    return df.where(membership_mask)


def profile_membership_aware_coverage(
    df: pd.DataFrame,
    membership_mask: pd.DataFrame,
) -> Dict[str, Union[float, int]]:
    """Calculate data coverage respecting membership periods (CORRECT analysis).

    This function ONLY looks at cells where tickers were active members.
    Empty cells outside membership are IGNORED (they're expected, not missing).

    Args:
        df: Prices or volumes DataFrame
        membership_mask: Boolean mask from get_membership_mask()

    Returns:
        Dictionary with membership-aware coverage statistics
    """
    # Only consider cells where ticker was active
    expected_cells = membership_mask.sum().sum()
    actual_cells = (membership_mask & ~df.isna()).sum().sum()

    return {
        "expected_cells": expected_cells,
        "actual_cells": actual_cells,
        "coverage_pct": (actual_cells / expected_cells * 100) if expected_cells > 0 else 0,
        "true_missing_cells": expected_cells - actual_cells,
        "true_missing_pct": ((expected_cells - actual_cells) / expected_cells * 100)
                            if expected_cells > 0 else 0,
    }


def detect_gaps_during_membership(
    df: pd.DataFrame,
    membership_mask: pd.DataFrame,
    min_gap_days: int = 5,
) -> pd.DataFrame:
    """Detect data gaps during active membership periods (TRUE missing data).

    Only gaps during membership periods are considered missing data.

    Args:
        df: Prices or volumes DataFrame
        membership_mask: Boolean mask from get_membership_mask()
        min_gap_days: Minimum consecutive days to count as a gap

    Returns:
        DataFrame with gap information (ticker, gap_start, gap_end, gap_days)
    """
    gaps = []

    for ticker in df.columns:
        if ticker not in membership_mask.columns:
            continue

        # Only look at data during membership
        active_dates = membership_mask[membership_mask[ticker]].index
        if len(active_dates) == 0:
            continue

        ticker_data = df.loc[active_dates, ticker]

        # Find gaps
        is_missing = ticker_data.isna()
        if not is_missing.any():
            continue

        gap_starts = is_missing & ~is_missing.shift(1, fill_value=False)
        gap_ends = is_missing & ~is_missing.shift(-1, fill_value=False)

        start_dates = ticker_data[gap_starts].index
        end_dates = ticker_data[gap_ends].index

        for start, end in zip(start_dates, end_dates):
            gap_days = (end - start).days + 1
            if gap_days >= min_gap_days:
                gaps.append({
                    'ticker': ticker,
                    'gap_start': start,
                    'gap_end': end,
                    'gap_days': gap_days,
                })

    return pd.DataFrame(gaps)


def analyse_membership_calendar(membership: pd.DataFrame) -> Dict[str, any]:
    """Analyse the universe membership calendar itself.

    Provides comprehensive EDA on the membership parquet structure.

    Args:
        membership: Universe calendar DataFrame

    Returns:
        Dictionary with membership statistics and insights
    """
    # Snapshot-level analysis
    snapshots_analysis = {
        "total_snapshots": membership['date'].nunique(),
        "date_range": f"{membership['date'].min()} to {membership['date'].max()}",
        "avg_constituents_per_snapshot": membership.groupby('date').size().mean(),
        "median_constituents_per_snapshot": membership.groupby('date').size().median(),
        "min_constituents": membership.groupby('date').size().min(),
        "max_constituents": membership.groupby('date').size().max(),
    }

    # Ticker-level analysis
    ticker_counts = membership.groupby('ticker').size()
    tickers_analysis = {
        "total_unique_tickers": membership['ticker'].nunique(),
        "avg_snapshots_per_ticker": ticker_counts.mean(),
        "median_snapshots_per_ticker": ticker_counts.median(),
        "min_snapshots": ticker_counts.min(),
        "max_snapshots": ticker_counts.max(),
    }

    # Start verification analysis
    verification_analysis = {
        "total_verified_starts": membership['start_verified'].sum(),
        "total_unverified_starts": (~membership['start_verified']).sum(),
        "verification_rate_pct": (membership['start_verified'].sum() / len(membership)) * 100,
    }

    # Metadata analysis
    metadata_analysis = {
        "index_names": membership['index_name'].unique().tolist(),
        "universe_types": membership['universe_type'].unique().tolist(),
        "rebalance_frequencies": membership['rebalance_frequency'].unique().tolist(),
        "data_sources": membership['data_source'].unique().tolist(),
        "algorithms": membership['algorithm'].unique().tolist(),
    }

    return {
        "snapshots": snapshots_analysis,
        "tickers": tickers_analysis,
        "verification": verification_analysis,
        "metadata": metadata_analysis,
    }


def analyse_source_attribution(source_map: pd.DataFrame) -> pd.DataFrame:
    """Analyse which data sources provided data for tickers.

    Args:
        source_map: Source mapping DataFrame with 'ticker' and 'source' columns

    Returns:
        DataFrame with source counts and percentages
    """
    counts = source_map['source'].value_counts()

    result = pd.DataFrame({
        "count": counts,
        "percentage": (counts / len(source_map)) * 100,
    })

    return result.sort_values("count", ascending=False)
