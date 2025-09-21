"""
Membership-aware data cleaning utilities.

This module provides functions to clean financial data while respecting
asset membership periods in the investment universe.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def load_dynamic_universe(universe_path: str) -> pd.DataFrame:
    """
    Load and process dynamic universe membership data.

    Args:
        universe_path: Path to universe membership CSV file

    Returns:
        DataFrame with MultiIndex (date, ticker) showing membership status
    """
    universe_df = pd.read_csv(universe_path)

    # Convert dates
    universe_df['start'] = pd.to_datetime(universe_df['start'])
    universe_df['end'] = pd.to_datetime(universe_df['end'])

    return universe_df


def get_universe_at_date(universe_df: pd.DataFrame, date: pd.Timestamp) -> list[str]:
    """
    Get the list of assets in the universe at a specific date.

    Args:
        universe_df: Universe membership DataFrame
        date: Date to check membership

    Returns:
        List of ticker symbols in universe at that date
    """
    mask = (universe_df['start'] <= date) & (universe_df['end'] >= date)
    return universe_df[mask]['ticker'].unique().tolist()


def create_membership_mask(
    returns_data: pd.DataFrame,
    universe_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a boolean mask indicating when each asset is in the universe.

    Args:
        returns_data: Returns DataFrame with DatetimeIndex and tickers as columns
        universe_df: Universe membership DataFrame

    Returns:
        Boolean DataFrame with same shape as returns_data
    """
    membership_mask = pd.DataFrame(
        False,
        index=returns_data.index,
        columns=returns_data.columns
    )

    for _, row in universe_df.iterrows():
        ticker = row['ticker']
        if ticker in membership_mask.columns:
            # Mark periods when asset is in universe
            mask_dates = (membership_mask.index >= row['start']) & \
                        (membership_mask.index <= row['end'])
            membership_mask.loc[mask_dates, ticker] = True

    return membership_mask


def clean_returns_with_membership(
    returns_data: pd.DataFrame,
    universe_df: pd.DataFrame,
    max_daily_return: float = 2.0,  # 200% daily return
    min_daily_return: float = -0.8,  # -80% daily return
    z_score_threshold: float = 8.0,
    cross_sectional_threshold: float = 10.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean returns data while respecting membership periods.

    Only cleans data during periods when assets are actually in the universe.
    Outside membership periods, NaN values are expected and preserved.

    Args:
        returns_data: Raw returns DataFrame
        universe_df: Universe membership DataFrame
        max_daily_return: Maximum acceptable daily return
        min_daily_return: Minimum acceptable daily return
        z_score_threshold: Z-score threshold for statistical outliers
        cross_sectional_threshold: Cross-sectional z-score threshold

    Returns:
        Tuple of (cleaned_returns, membership_mask)
    """
    logger.info("Starting membership-aware data cleaning...")

    # Create membership mask
    membership_mask = create_membership_mask(returns_data, universe_df)

    # Work with a copy
    cleaned_returns = returns_data.copy()

    # Step 1: Remove obvious data errors ONLY during membership periods
    obvious_errors = ((cleaned_returns > max_daily_return) |
                     (cleaned_returns < min_daily_return)) & membership_mask
    cleaned_returns[obvious_errors] = np.nan
    logger.info(f"Removed {obvious_errors.sum().sum()} obvious data errors during membership periods")

    # Step 2: Statistical outlier detection (only for membership periods)
    # Calculate rolling statistics only for periods in universe
    masked_returns = cleaned_returns.where(membership_mask)

    rolling_std = masked_returns.rolling(window=252, min_periods=60).std()
    rolling_mean = masked_returns.rolling(window=252, min_periods=60).mean()
    z_scores = (masked_returns - rolling_mean).abs() / (rolling_std + 1e-8)

    # Flag outliers only during membership
    statistical_outliers = (z_scores > z_score_threshold) & membership_mask
    cleaned_returns[statistical_outliers] = np.nan
    logger.info(f"Removed {statistical_outliers.sum().sum()} statistical outliers during membership periods")

    # Step 3: Cross-sectional outlier detection
    # Only compare assets that are in universe on the same day
    for date in cleaned_returns.index:
        # Get assets in universe at this date
        date_universe = get_universe_at_date(universe_df, date)
        if len(date_universe) > 1:
            # Calculate cross-sectional statistics only for in-universe assets
            date_returns = cleaned_returns.loc[date, date_universe]
            valid_returns = date_returns.dropna()

            if len(valid_returns) > 1:
                median_return = valid_returns.median()
                mad = (valid_returns - median_return).abs().median()

                if mad > 0:
                    z_scores = (valid_returns - median_return).abs() / (mad * 1.4826 + 1e-8)
                    outliers = z_scores[z_scores > cross_sectional_threshold].index
                    cleaned_returns.loc[date, outliers] = np.nan

    # Step 4: Fill NaN values ONLY during membership periods
    # Use forward fill with limit only within membership periods
    for ticker in cleaned_returns.columns:
        if ticker in universe_df['ticker'].values:
            # Get membership periods for this ticker
            ticker_memberships = universe_df[universe_df['ticker'] == ticker]

            for _, membership in ticker_memberships.iterrows():
                # Only fill during membership period
                period_mask = (cleaned_returns.index >= membership['start']) & \
                             (cleaned_returns.index <= membership['end'])

                if period_mask.any():
                    # Forward fill with limit
                    period_data = cleaned_returns.loc[period_mask, ticker]
                    period_data = period_data.ffill(limit=5)
                    # Backward fill for start of membership
                    period_data = period_data.bfill(limit=5)
                    # Fill remaining NaNs with 0 (no return)
                    period_data = period_data.fillna(0.0)
                    cleaned_returns.loc[period_mask, ticker] = period_data

    # Outside membership periods, set to NaN (not 0)
    cleaned_returns[~membership_mask] = np.nan

    logger.info(f"Cleaned returns shape: {cleaned_returns.shape}")
    logger.info(f"Assets with membership data: {membership_mask.any().sum()}")

    return cleaned_returns, membership_mask


def get_rolling_universe(
    universe_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    rebalance_frequency: str = 'MS'
) -> dict[pd.Timestamp, list[str]]:
    """
    Get the dynamic universe for each rebalancing date.

    Args:
        universe_df: Universe membership DataFrame
        start_date: Backtest start date
        end_date: Backtest end date
        rebalance_frequency: Pandas frequency string for rebalancing

    Returns:
        Dictionary mapping rebalance dates to list of tickers in universe
    """
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=rebalance_frequency)

    universe_schedule = {}
    for date in rebalance_dates:
        universe_schedule[date] = get_universe_at_date(universe_df, date)

    return universe_schedule