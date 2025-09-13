#!/usr/bin/env python3
"""
Create sample dataset for testing and development.

This script generates synthetic financial data that mimics S&P MidCap 400
characteristics for testing the ML models when real data is not available.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def generate_sample_universe(n_assets: int = 400) -> pd.DataFrame:
    """
    Generate sample universe membership data.

    Args:
        n_assets: Number of assets to generate

    Returns:
        DataFrame with universe membership over time
    """
    # Create date range (5 years of business days)
    dates = pd.date_range("2019-01-01", "2024-01-01", freq="B")

    # Generate asset tickers
    tickers = [f"TICKER_{i:03d}" for i in range(n_assets)]

    # Generate membership (most assets are always in, some drop out occasionally)
    membership = np.ones((len(dates), n_assets), dtype=bool)

    # Simulate some assets dropping out and rejoining
    for i in range(n_assets):
        if np.random.random() < 0.1:  # 10% chance of having membership gaps
            # Create some gaps in membership
            gap_starts = np.random.choice(len(dates), size=np.random.randint(1, 4), replace=False)
            for gap_start in gap_starts:
                gap_length = np.random.randint(20, 100)  # 1-5 month gaps
                gap_end = min(gap_start + gap_length, len(dates))
                membership[gap_start:gap_end, i] = False

    universe_df = pd.DataFrame(membership, index=dates, columns=tickers)
    return universe_df


def generate_sample_returns(universe_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate realistic sample returns and prices.

    Args:
        universe_df: Universe membership DataFrame

    Returns:
        Tuple of (prices, returns) DataFrames
    """
    dates = universe_df.index
    tickers = universe_df.columns
    n_dates = len(dates)
    n_assets = len(tickers)

    # Generate factor loadings for realistic correlation structure
    n_factors = 10
    factor_loadings = np.random.normal(0, 0.3, (n_assets, n_factors))

    # Generate factor returns (common market drivers)
    factor_returns = np.random.normal(0, 0.015, (n_dates, n_factors))

    # Generate idiosyncratic returns
    idiosyncratic_vol = np.random.uniform(0.01, 0.04, n_assets)  # 1-4% daily vol
    idiosyncratic_returns = np.random.normal(0, 1, (n_dates, n_assets)) * idiosyncratic_vol

    # Combine factor and idiosyncratic components
    factor_component = factor_returns @ factor_loadings.T
    total_returns = factor_component + idiosyncratic_returns

    # Add small positive drift (equity risk premium)
    drift = np.random.normal(0.0002, 0.0001, n_assets)  # ~5% annual drift
    total_returns += drift

    # Apply universe membership (NaN for periods when asset not in universe)
    returns = pd.DataFrame(total_returns, index=dates, columns=tickers)
    returns = returns.where(universe_df, np.nan)

    # Generate prices from returns
    initial_prices = np.random.uniform(20, 200, n_assets)  # Random starting prices
    log_prices = np.log(initial_prices) + np.cumsum(np.log(1 + returns.fillna(0)), axis=0)
    prices = pd.DataFrame(np.exp(log_prices), index=dates, columns=tickers)

    # Apply universe membership to prices as well
    prices = prices.where(universe_df, np.nan)

    return prices, returns


def create_sample_data():
    """Create sample dataset for testing."""
    logger = setup_logging()
    logger.info("Creating sample dataset for testing...")

    # Create output directories
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    (data_dir / "sample").mkdir(exist_ok=True)

    # Generate sample data
    logger.info("Generating sample universe (400 assets)...")
    universe_df = generate_sample_universe(n_assets=400)

    logger.info("Generating sample returns and prices...")
    prices, returns = generate_sample_returns(universe_df)

    # Save sample data
    sample_dir = data_dir / "sample"

    universe_path = sample_dir / "universe_membership.parquet"
    prices_path = sample_dir / "prices_daily.parquet"
    returns_path = sample_dir / "returns_daily.parquet"

    universe_df.to_parquet(universe_path)
    prices.to_parquet(prices_path)
    returns.to_parquet(returns_path)

    logger.info(f"Sample universe saved to {universe_path}")
    logger.info(f"Sample prices saved to {prices_path}")
    logger.info(f"Sample returns saved to {returns_path}")

    # Data quality summary
    logger.info("Sample data summary:")
    logger.info(f"  Universe shape: {universe_df.shape}")
    logger.info(f"  Prices shape: {prices.shape}")
    logger.info(f"  Returns shape: {returns.shape}")
    logger.info(f"  Date range: {prices.index.min()} to {prices.index.max()}")
    logger.info(f"  Average assets per day: {universe_df.sum(axis=1).mean():.1f}")
    logger.info(f"  Non-null return observations: {returns.count().sum()}")

    # Calculate some basic statistics
    daily_returns = returns.stack().dropna()
    logger.info("  Daily return statistics:")
    logger.info(f"    Mean: {daily_returns.mean():.6f}")
    logger.info(f"    Std: {daily_returns.std():.6f}")
    logger.info(f"    Skew: {daily_returns.skew():.3f}")
    logger.info(f"    Kurtosis: {daily_returns.kurtosis():.3f}")

    logger.info("Sample dataset creation completed successfully!")


if __name__ == "__main__":
    create_sample_data()
