#!/usr/bin/env python3
"""
Download and prepare data for portfolio optimization models.

This script downloads S&P MidCap 400 data from multiple sources and prepares
it for model training and backtesting.
"""

import logging
from pathlib import Path

from src.config.base import load_config
from src.data.collectors.stooq import StooqCollector
from src.data.collectors.yfinance import YFinanceCollector
from src.data.processors.gap_filling import GapFillProcessor
from src.data.processors.universe_builder import UniverseBuilder


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def download_data():
    """Download and prepare portfolio data."""
    logger = setup_logging()
    logger.info("Starting data download and preparation")

    # Load configuration
    config = load_config("configs/data/midcap400.yaml")

    # Create output directories
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    (data_dir / "raw").mkdir(exist_ok=True)
    (data_dir / "processed").mkdir(exist_ok=True)

    # Get S&P MidCap 400 universe
    logger.info("Building S&P MidCap 400 universe...")
    universe_builder = UniverseBuilder(config)
    universe_data = universe_builder.build_universe(
        index_name="sp_midcap_400",
        start_date=config.start_date,
        end_date=config.end_date
    )

    # Save universe membership
    universe_path = data_dir / "processed" / "universe_membership.parquet"
    universe_data.to_parquet(universe_path)
    logger.info(f"Universe membership saved to {universe_path}")

    # Get unique tickers from universe
    tickers = universe_data.columns.tolist()
    logger.info(f"Found {len(tickers)} tickers in universe")

    # Download price data from multiple sources
    logger.info("Downloading price data from Stooq...")
    stooq_collector = StooqCollector(config.collector_config)
    stooq_data = stooq_collector.collect_data(
        tickers=tickers,
        start_date=config.start_date,
        end_date=config.end_date
    )

    logger.info("Downloading additional data from Yahoo Finance...")
    yfinance_collector = YFinanceCollector(config.collector_config)
    yfinance_data = yfinance_collector.collect_data(
        tickers=tickers,
        start_date=config.start_date,
        end_date=config.end_date
    )

    # Fill gaps using multiple sources
    logger.info("Processing and gap-filling data...")
    gap_filler = GapFillProcessor(config.processing_config)

    # Combine and fill gaps for prices
    if stooq_data is not None and yfinance_data is not None:
        combined_prices = gap_filler.fill_gaps(
            primary_data=stooq_data,
            fallback_data=yfinance_data,
            method="forward_fill"
        )
    elif stooq_data is not None:
        combined_prices = stooq_data
    elif yfinance_data is not None:
        combined_prices = yfinance_data
    else:
        raise RuntimeError("No price data could be downloaded")

    # Calculate returns
    logger.info("Calculating returns...")
    returns = combined_prices.pct_change().dropna()

    # Save processed data
    prices_path = data_dir / "processed" / "prices_daily.parquet"
    returns_path = data_dir / "processed" / "returns_daily.parquet"

    combined_prices.to_parquet(prices_path)
    returns.to_parquet(returns_path)

    logger.info(f"Prices saved to {prices_path}")
    logger.info(f"Returns saved to {returns_path}")

    # Data quality summary
    logger.info("Data quality summary:")
    logger.info(f"  Price data shape: {combined_prices.shape}")
    logger.info(f"  Returns data shape: {returns.shape}")
    logger.info(f"  Date range: {combined_prices.index.min()} to {combined_prices.index.max()}")
    logger.info(f"  Missing values in prices: {combined_prices.isnull().sum().sum()}")
    logger.info(f"  Missing values in returns: {returns.isnull().sum().sum()}")

    logger.info("Data download and preparation completed successfully!")


if __name__ == "__main__":
    download_data()
