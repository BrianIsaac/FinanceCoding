#!/usr/bin/env python3
"""
Complete NEW Pipeline Implementation - Full Universe Collection
Using Stories 1.2 and 1.3 modular implementation to collect all 822 historical tickers.
"""

import logging
import sys
import time
from pathlib import Path

import pandas as pd

# Configure logging with proper directory structure
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "pipeline_execution.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.config.data import CollectorConfig, UniverseConfig, ValidationConfig  # noqa: E402
from src.data.collectors.stooq import StooqCollector  # noqa: E402
from src.data.collectors.yfinance import YFinanceCollector  # noqa: E402
from src.data.processors.data_quality_validator import DataQualityValidator  # noqa: E402
from src.data.processors.gap_filling import GapFiller  # noqa: E402
from src.data.processors.universe_builder import UniverseBuilder  # noqa: E402


def _build_membership_intervals_from_calendar(
    universe_calendar: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build ticker-specific membership intervals from universe calendar snapshots.

    Args:
        universe_calendar: DataFrame with columns [date, ticker, index_name, ...]

    Returns:
        DataFrame with columns [ticker, start, end] where:
        - start: earliest date ticker appears in calendar
        - end: latest date ticker appears, or None if still active

    Example:
        >>> calendar = pd.DataFrame({
        ...     'date': ['2020-01-01', '2020-02-01', '2020-01-01'],
        ...     'ticker': ['AAPL', 'AAPL', 'MSFT']
        ... })
        >>> intervals = _build_membership_intervals_from_calendar(calendar)
        >>> intervals['ticker'].tolist()
        ['AAPL', 'MSFT']
    """
    intervals = []

    for ticker in universe_calendar["ticker"].unique():
        ticker_dates = universe_calendar[
            universe_calendar["ticker"] == ticker
        ]["date"].sort_values()

        # Find date range from monthly snapshots
        start_date = ticker_dates.min()
        end_date = ticker_dates.max()

        # Check if ticker is active in most recent snapshot
        most_recent = universe_calendar["date"].max()
        is_active = end_date >= most_recent

        intervals.append({
            "ticker": ticker,
            "start": start_date.strftime("%Y-%m-%d"),
            "end": None if is_active else end_date.strftime("%Y-%m-%d"),
        })

    return pd.DataFrame(intervals)


def collect_single_ticker_approach(
    collector,
    membership_df: pd.DataFrame,
    global_start: str = "2010-01-01",
    global_end: str = "2024-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Collect data using per-ticker membership periods.

    Args:
        collector: StooqCollector or YFinanceCollector instance
        membership_df: DataFrame with columns [ticker, start, end]
        global_start: Absolute earliest date to collect (fallback for tickers without start)
        global_end: Absolute latest date to collect (fallback for active tickers)

    Returns:
        Tuple of (prices_df, volumes_df, collected_tickers)
    """
    collector_name = collector.config.source_name

    logger.info(f"=== Collecting data via {collector_name} with per-ticker periods ===")

    # Prepare ticker periods list
    ticker_periods = []
    for _, row in membership_df.iterrows():
        ticker_periods.append({
            "ticker": row["ticker"],
            "start": row["start"] if pd.notna(row["start"]) else global_start,
            "end": row["end"] if pd.notna(row["end"]) else global_end,
        })

    logger.info(f"Prepared {len(ticker_periods)} ticker-period combinations")

    # Call appropriate collector method
    if collector_name == "stooq":
        ohlcv_data = collector.collect_ohlcv_data(ticker_periods)  # Refactored method
        prices_df = ohlcv_data["close"]
        volumes_df = ohlcv_data["volume"]
    elif collector_name == "yfinance":
        prices_df, volumes_df = collector.download_with_ticker_periods(ticker_periods)  # New method
    else:
        raise ValueError(f"Unknown collector: {collector_name}")

    collected_tickers = prices_df.columns.tolist() if not prices_df.empty else []

    logger.info(f"Collection complete: {len(collected_tickers)} tickers collected")
    if not prices_df.empty:
        logger.info(f"Date range: {prices_df.index.min()} to {prices_df.index.max()}")

    return prices_df, volumes_df, collected_tickers


def main():
    """Execute our complete NEW pipeline implementation with full universe."""

    logger.info("=" * 80)
    logger.info("STARTING COMPLETE NEW PIPELINE IMPLEMENTATION")
    logger.info("=" * 80)

    start_time = time.time()

    # Configuration setup
    logger.info("Setting up pipeline configuration...")
    universe_config = UniverseConfig(
        universe_type="midcap400", min_market_cap=None, min_avg_volume=None, exclude_sectors=None
    )

    stooq_config = CollectorConfig(
        source_name="stooq", rate_limit=10.0, timeout=15, retry_attempts=3, retry_delay=1.0
    )

    yfinance_config = CollectorConfig(
        source_name="yfinance", rate_limit=5.0, timeout=10, retry_attempts=3, retry_delay=1.0
    )

    validation_config = ValidationConfig(
        missing_data_threshold=0.10,
        price_change_threshold=0.50,
        volume_threshold=1000,
        validate_business_days=True,
        fill_method="forward",
        generate_reports=True,
        report_output_dir="logs/validation_reports",
    )

    # Initialize processors
    logger.info("Initializing data processors...")
    universe_builder = UniverseBuilder(universe_config, "data/processed")
    stooq_collector = StooqCollector(stooq_config)
    yfinance_collector = YFinanceCollector(yfinance_config)
    gap_filler = GapFiller(validation_config)
    quality_validator = DataQualityValidator(validation_config)

    # Get the full historical universe
    logger.info("Loading universe composition...")
    try:
        # Load existing universe calendar
        universe_calendar = pd.read_parquet("data/processed/universe_calendar_midcap400.parquet")

        # Build membership intervals from calendar snapshots
        membership_df = _build_membership_intervals_from_calendar(universe_calendar)

        logger.info(f"Loaded universe: {len(membership_df)} ticker-period combinations")
        logger.info(f"Unique tickers: {membership_df['ticker'].nunique()}")
    except Exception:
        logger.info("Building universe membership intervals from scratch...")
        membership_df = universe_builder.build_membership_intervals("2016-01-01", "2024-12-31")
        universe_calendar = universe_builder.create_monthly_snapshots(
            membership_df, "2016-01-01", "2024-12-31"
        )
        # Save for future use
        universe_calendar.to_parquet("data/processed/universe_calendar_midcap400.parquet", index=False)
        logger.info(f"Built universe: {len(membership_df)} ticker-period combinations")

    # Extract ticker list for backward compatibility (will be replaced in later phases)
    all_tickers = sorted(membership_df["ticker"].unique())

    # Smart source selection: Test Stooq availability first
    logger.info("=" * 60)
    logger.info("PHASE 1: DATA SOURCE SELECTION")
    logger.info("=" * 60)

    logger.info("Testing Stooq API availability...")
    stooq_available = stooq_collector.check_rate_limit_status()

    if stooq_available:
        logger.info("✅ Stooq API available - proceeding with Stooq as primary source")
        logger.info("=" * 60)
        logger.info("PHASE 2: STOOQ DATA COLLECTION")
        logger.info("=" * 60)

        stooq_prices, stooq_volumes, collected_stooq = collect_single_ticker_approach(
            stooq_collector, membership_df
        )

        # Save Stooq results
        if not stooq_prices.empty:
            output_dir = Path("data/stooq_new")
            output_dir.mkdir(exist_ok=True)

            stooq_prices.to_parquet(output_dir / "prices.parquet", compression="gzip")
            stooq_volumes.to_parquet(output_dir / "volume.parquet", compression="gzip")
            logger.info(f"Saved Stooq data to {output_dir}: {stooq_prices.shape[1]} tickers")

        # Identify missing tickers and get them from YFinance
        missing_tickers = list(set(all_tickers) - set(collected_stooq))
        logger.info(f"Missing from Stooq: {len(missing_tickers)} tickers")

        if missing_tickers:
            logger.info("=" * 60)
            logger.info("PHASE 3: YFINANCE AUGMENTATION FOR MISSING TICKERS")
            logger.info("=" * 60)

            # Filter membership_df to missing tickers only
            missing_membership = membership_df[membership_df["ticker"].isin(missing_tickers)]

            yf_prices, yf_volumes, collected_yf = collect_single_ticker_approach(
                yfinance_collector, missing_membership
            )

            # Merge Stooq and YFinance data
            if not stooq_prices.empty and not yf_prices.empty:
                logger.info("Merging Stooq and YFinance data using splice-fill methodology...")
                combined_prices, combined_volumes = yfinance_collector.merge_with_primary_source(
                    stooq_prices, stooq_volumes, yf_prices, yf_volumes
                )
                logger.info(f"Merged data shape: {combined_prices.shape}")
            elif not stooq_prices.empty:
                logger.info("Using Stooq data only")
                combined_prices = stooq_prices
                combined_volumes = stooq_volumes
            elif not yf_prices.empty:
                logger.info("Using YFinance data only")
                combined_prices = yf_prices
                combined_volumes = yf_volumes
            else:
                logger.error("No data collected from either source!")
                return False
        else:
            logger.info("Using Stooq data only (all tickers collected successfully)")
            combined_prices = stooq_prices
            combined_volumes = stooq_volumes
    else:
        logger.warning("❌ Stooq API rate limited - switching to YFinance as primary source")
        logger.info("=" * 60)
        logger.info("PHASE 2: YFINANCE DATA COLLECTION (PRIMARY SOURCE)")
        logger.info("=" * 60)

        yf_prices, yf_volumes, collected_yf = collect_single_ticker_approach(
            yfinance_collector, membership_df
        )

        if not yf_prices.empty:
            combined_prices = yf_prices
            combined_volumes = yf_volumes
            logger.info(f"YFinance data shape: {combined_prices.shape}")
            logger.info(f"Collected {len(collected_yf)} tickers successfully from YFinance")

            # Save YFinance results
            output_dir = Path("data/yfinance_primary")
            output_dir.mkdir(exist_ok=True)

            combined_prices.to_parquet(output_dir / "prices.parquet", compression="gzip")
            combined_volumes.to_parquet(output_dir / "volume.parquet", compression="gzip")
            logger.info(f"Saved YFinance data to {output_dir}")
        else:
            logger.error("No data collected from YFinance!")
            return False

    # Apply our enhanced gap-filling
    logger.info("=" * 60)
    logger.info("PHASE 4: GAP FILLING AND DATA CLEANING")
    logger.info("=" * 60)

    filled_prices = combined_prices.copy()
    fill_stats = {}

    for ticker in combined_prices.columns:
        if ticker in combined_volumes.columns:
            original_na = combined_prices[ticker].isna().sum()

            # Forward fill with volume validation
            filled_prices[ticker] = gap_filler.forward_fill(
                combined_prices[ticker],
                volume_series=combined_volumes[ticker],
                min_volume=validation_config.volume_threshold,
                limit=10,
            )

            # Forward fill for remaining gaps (avoid temporal leakage)
            filled_prices[ticker] = gap_filler.forward_fill(filled_prices[ticker], limit=10)

            final_na = filled_prices[ticker].isna().sum()
            fill_stats[ticker] = {
                "original_na": original_na,
                "final_na": final_na,
                "filled": original_na - final_na,
            }

    total_filled = sum(stats["filled"] for stats in fill_stats.values())
    logger.info(f"Gap filling completed: {total_filled} total gaps filled across all tickers")

    # Quality validation
    logger.info("=" * 60)
    logger.info("PHASE 5: DATA QUALITY VALIDATION")
    logger.info("=" * 60)

    data_dict = {"prices": filled_prices, "volume": combined_volumes}

    logger.info("Running comprehensive data quality validation...")
    validation_results = quality_validator.validate_complete_dataset(
        data_dict=data_dict, universe_tickers=all_tickers, generate_report=True
    )

    quality_score = validation_results.get("overall_quality_score", 0)
    logger.info(f"Data quality validation completed. Overall score: {quality_score:.3f}")

    # Generate returns
    logger.info("=" * 60)
    logger.info("PHASE 6: FINAL DATA PROCESSING AND SAVING")
    logger.info("=" * 60)

    logger.info("Generating daily returns data...")
    returns_daily = filled_prices.pct_change()

    # Save final datasets
    output_dir = Path("data/final_new_pipeline")
    output_dir.mkdir(exist_ok=True)

    logger.info(f"Saving final datasets to {output_dir}...")
    filled_prices.to_parquet(output_dir / "prices_final.parquet", compression="gzip")
    combined_volumes.to_parquet(output_dir / "volume_final.parquet", compression="gzip")
    returns_daily.to_parquet(output_dir / "returns_daily_final.parquet", compression="gzip")
    logger.info("All datasets saved successfully")

    # Calculate final metrics
    final_coverage = (filled_prices.notna().sum() / len(filled_prices)) * 100

    summary = {
        "total_tickers": len(filled_prices.columns),
        "target_universe_size": len(all_tickers),
        "universe_coverage_pct": (len(filled_prices.columns) / len(all_tickers)) * 100,
        "date_range": f"{filled_prices.index.min()} to {filled_prices.index.max()}",
        "average_coverage": final_coverage.mean(),
        "tickers_95pct_coverage": (final_coverage >= 95).sum(),
        "total_gaps_filled": total_filled,
        "quality_score": validation_results.get("overall_quality_score", 0),
        "stooq_tickers": len(collected_stooq) if "collected_stooq" in locals() else 0,
        "yfinance_tickers": len(collected_yf) if "collected_yf" in locals() else 0,
    }

    # Compare to original merged dataset
    try:
        old_merged = pd.read_parquet("data/merged/prices.parquet")

        if len(filled_prices.columns) >= len(old_merged.columns):
            pass
        else:
            coverage_improvement = summary["average_coverage"] - 68.7  # Original was 68.7%

    except Exception:
        pass

    # Save summary
    import json

    with open(output_dir / "new_pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Pipeline summary saved to new_pipeline_summary.json")

    # Calculate and log execution time
    execution_time = time.time() - start_time
    execution_hours = execution_time / 3600

    logger.info("=" * 80)
    logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Total execution time: {execution_hours:.2f} hours ({execution_time:.1f} seconds)")
    logger.info(f"Final universe coverage: {summary['universe_coverage_pct']:.1f}%")
    logger.info(f"Data quality score: {summary['quality_score']:.3f}")
    logger.info(f"Average ticker coverage: {summary['average_coverage']:.1f}%")
    logger.info(f"Tickers with >95% coverage: {summary['tickers_95pct_coverage']}")
    logger.info(f"Total gaps filled: {summary['total_gaps_filled']}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
