#!/usr/bin/env python3
"""
Data Collection Pipeline with Waterfall Fallback Strategy

Collects S&P MidCap 400 historical data using a waterfall approach across four
data sources (Stooq → Tiingo → Polygon → YFinance), where each source fills
only the gaps left by previous sources. Respects per-ticker membership periods
to avoid collecting unnecessary historical data.

Architecture:
1. Load universe membership intervals from parquet
2. Attempt collection from sources in priority order
3. Track collected tickers and pass only missing tickers to next source
4. Combine data from all sources
5. Apply gap filling with volume validation
6. Validate data quality and generate reports
7. Save final datasets (prices, volumes, returns)
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
from src.data.collectors.polygon import PolygonCollector  # noqa: E402
from src.data.collectors.stooq import StooqCollector  # noqa: E402
from src.data.collectors.tiingo import TiingoCollector  # noqa: E402
from src.data.collectors.wikipedia import WikipediaCollector  # noqa: E402
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
        ticker_dates = universe_calendar[universe_calendar["ticker"] == ticker][
            "date"
        ].sort_values()

        # Find date range from monthly snapshots
        start_date = ticker_dates.min()
        end_date = ticker_dates.max()

        # Check if ticker is active in most recent snapshot
        most_recent = universe_calendar["date"].max()
        is_active = end_date >= most_recent

        intervals.append(
            {
                "ticker": ticker,
                "start": start_date.strftime("%Y-%m-%d"),
                "end": None if is_active else end_date.strftime("%Y-%m-%d"),
            }
        )

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
        ticker_periods.append(
            {
                "ticker": row["ticker"],
                "start": row["start"] if pd.notna(row["start"]) else global_start,
                "end": row["end"] if pd.notna(row["end"]) else global_end,
            }
        )

    logger.info(f"Prepared {len(ticker_periods)} ticker-period combinations")

    # Call appropriate collector method
    if collector_name in ["stooq", "tiingo", "polygon"]:
        ohlcv_data = collector.collect_ohlcv_data(ticker_periods)
        prices_df = ohlcv_data["close"]
        volumes_df = ohlcv_data["volume"]
    elif collector_name == "yfinance":
        prices_df, volumes_df = collector.download_with_ticker_periods(ticker_periods)
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
    logger.info("=" * 60)
    logger.info("PHASE 1: CONFIGURATION AND UNIVERSE LOADING")
    logger.info("=" * 60)
    logger.info("Setting up pipeline configuration...")
    universe_config = UniverseConfig(
        universe_type="midcap400", min_market_cap=None, min_avg_volume=None, exclude_sectors=None
    )

    stooq_config = CollectorConfig(
        source_name="stooq", rate_limit=10.0, timeout=15, retry_attempts=3, retry_delay=1.0
    )

    tiingo_config = CollectorConfig(
        source_name="tiingo", rate_limit=72.0, timeout=15, retry_attempts=3, retry_delay=1.0
    )

    polygon_config = CollectorConfig(
        source_name="polygon", rate_limit=12.0, timeout=15, retry_attempts=3, retry_delay=1.0
    )

    yfinance_config = CollectorConfig(
        source_name="yfinance", rate_limit=5.0, timeout=10, retry_attempts=3, retry_delay=1.0
    )

    wiki_config = CollectorConfig(
        source_name="wikipedia",
        rate_limit=1.0,  # Conservative: 1 request per second
        timeout=30,  # Longer timeout for HTML parsing
        retry_attempts=3,
        retry_delay=2.0,
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
    tiingo_collector = TiingoCollector(tiingo_config)
    polygon_collector = PolygonCollector(polygon_config)
    yfinance_collector = YFinanceCollector(yfinance_config)
    wiki_collector = WikipediaCollector(wiki_config)
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
        universe_calendar.to_parquet(
            "data/processed/universe_calendar_midcap400.parquet", index=False
        )
        logger.info(f"Built universe: {len(membership_df)} ticker-period combinations")

    # Extract ticker list for backward compatibility (will be replaced in later phases)
    all_tickers = sorted(membership_df["ticker"].unique())

    # New unified implementation
    logger.info("=" * 60)
    logger.info("PHASE 2: WATERFALL DATA COLLECTION")
    logger.info("=" * 60)
    logger.info(f"Target universe: {len(all_tickers)} tickers")
    logger.info("Collection strategy: Stooq → Tiingo → Polygon → YFinance (waterfall)")
    logger.info("")

    # Define collection sources in priority order
    sources = [
        ("Stooq", stooq_collector),
        ("Tiingo", tiingo_collector),
        ("Polygon", polygon_collector),
        ("YFinance", yfinance_collector),
        ("Wikipedia", wiki_collector),
    ]

    # Track collection results across all sources
    all_prices_list = []
    all_volumes_list = []
    collected_tickers = set()
    collection_stats = {}
    source_collection_time = {}

    # Start with full universe
    remaining_tickers = set(all_tickers)
    initial_ticker_count = len(all_tickers)

    for source_idx, (source_name, collector) in enumerate(sources, 1):
        logger.info("=" * 60)
        logger.info(f"SOURCE {source_idx}/4: {source_name.upper()}")
        logger.info("=" * 60)

        # Skip if no tickers remaining
        if not remaining_tickers:
            logger.info(f"✓ All {initial_ticker_count} tickers already collected")
            logger.info(f"  Skipping {source_name}")
            collection_stats[source_name] = 0
            continue

        # Skip Polygon if no API key
        if source_name == "Polygon" and not collector.api_key:
            logger.info("⊘ Polygon API key not available - skipping")
            collection_stats[source_name] = 0
            continue

        # Filter membership to remaining tickers only
        remaining_membership = membership_df[membership_df["ticker"].isin(remaining_tickers)]

        logger.info(f"Attempting to collect {len(remaining_tickers)} remaining tickers")
        logger.info(
            f"  ({len(remaining_tickers)}/{initial_ticker_count} = "
            f"{len(remaining_tickers) / initial_ticker_count * 100:.1f}% of universe)"
        )

        # Track collection time
        source_start_time = time.time()

        try:
            # Collect data for remaining tickers
            prices, volumes, collected = collect_single_ticker_approach(
                collector, remaining_membership
            )

            source_elapsed = time.time() - source_start_time
            source_collection_time[source_name] = source_elapsed

            if not prices.empty:
                # Log successful collection details
                logger.info(f"✓ {source_name} collection successful:")
                logger.info(f"  Tickers collected: {len(collected)}")
                logger.info(
                    f"  Coverage: {len(collected)}/{len(remaining_tickers)} "
                    f"({len(collected) / len(remaining_tickers) * 100:.1f}% of remaining)"
                )
                logger.info(f"  Data shape: {prices.shape[1]} tickers × {prices.shape[0]} dates")
                logger.info(f"  Date range: {prices.index.min()} to {prices.index.max()}")
                logger.info(f"  Non-null values: {prices.notna().sum().sum():,}")
                logger.info(f"  Collection time: {source_elapsed:.1f}s")

                # Log sample tickers
                sample_tickers = sorted(collected)[:5]
                logger.info(f"  Sample tickers: {', '.join(sample_tickers)}")
                if len(collected) > 5:
                    logger.info(f"  ... and {len(collected) - 5} more")

                all_prices_list.append(prices)
                all_volumes_list.append(volumes)
                collected_tickers.update(collected)
                remaining_tickers -= set(collected)
                collection_stats[source_name] = len(collected)

                # Log updated status
                total_collected = len(collected_tickers)
                logger.info(
                    f"  Progress: {total_collected}/{initial_ticker_count} "
                    f"({total_collected / initial_ticker_count * 100:.1f}%) total collected"
                )
                logger.info(f"  Remaining: {len(remaining_tickers)} tickers")

                if remaining_tickers and len(remaining_tickers) <= 10:
                    logger.info(f"  Still missing: {', '.join(sorted(remaining_tickers))}")

            else:
                collection_stats[source_name] = 0
                logger.warning(f"✗ {source_name} returned empty DataFrame")
                logger.warning(f"  All {len(remaining_tickers)} tickers failed")
                logger.warning("  Moving to next source...")

        except ValueError as e:
            # Handle API key errors specifically
            collection_stats[source_name] = 0
            source_collection_time[source_name] = time.time() - source_start_time
            logger.error(f"✗ {source_name} configuration error:")
            logger.error(f"  Error: {e}")
            logger.error("  This usually means an API key is missing or invalid")
            logger.info("  Moving to next source...")
            continue

        except Exception as e:
            collection_stats[source_name] = 0
            source_collection_time[source_name] = time.time() - source_start_time
            logger.error(f"✗ {source_name} collection failed:")
            logger.error(f"  Error type: {type(e).__name__}")
            logger.error(f"  Error message: {e}")
            logger.info("  Moving to next source...")
            continue

        logger.info("")  # Blank line for readability

    # Verify we collected something
    if not all_prices_list:
        logger.error("=" * 60)
        logger.error("CRITICAL ERROR: NO DATA COLLECTED")
        logger.error("=" * 60)
        logger.error("All data sources failed to collect any tickers")
        logger.error("Possible causes:")
        logger.error("  - All API keys missing or invalid")
        logger.error("  - All sources rate-limited simultaneously")
        logger.error("  - Network connectivity issues")
        logger.error("  - Invalid ticker symbols in universe")
        return False

    # Combine all sources
    logger.info("=" * 60)
    logger.info("PHASE 3: COMBINING DATA FROM ALL SOURCES")
    logger.info("=" * 60)

    combined_prices = pd.concat(all_prices_list, axis=1)
    combined_volumes = pd.concat(all_volumes_list, axis=1)

    logger.info("Combined dataset statistics:")
    logger.info(
        f"  Final shape: {combined_prices.shape[1]} tickers × {combined_prices.shape[0]} dates"
    )
    logger.info(f"  Date range: {combined_prices.index.min()} to {combined_prices.index.max()}")
    logger.info(f"  Total trading days: {len(combined_prices)}")
    logger.info(f"  Non-null values: {combined_prices.notna().sum().sum():,}")
    logger.info("")

    logger.info("Collection breakdown by source:")
    total_collection_time = sum(source_collection_time.values())
    for source_name, count in collection_stats.items():
        pct = (count / len(all_tickers)) * 100 if all_tickers else 0
        elapsed = source_collection_time.get(source_name, 0)
        time_pct = (elapsed / total_collection_time * 100) if total_collection_time > 0 else 0
        logger.info(
            f"  {source_name:.<12} {count:>4} tickers ({pct:>5.1f}%) | "
            f"{elapsed:>6.1f}s ({time_pct:>5.1f}%)"
        )

    logger.info("")
    logger.info(
        f"Total collection time: {total_collection_time:.1f}s "
        f"({total_collection_time / 60:.1f} minutes)"
    )
    logger.info(
        f"Universe coverage: {len(collected_tickers)}/{len(all_tickers)} "
        f"({len(collected_tickers) / len(all_tickers) * 100:.1f}%)"
    )

    if remaining_tickers:
        logger.warning("")
        logger.warning(f"⚠ Failed to collect {len(remaining_tickers)} tickers from any source:")
        failed_list = sorted(remaining_tickers)
        for i in range(0, len(failed_list), 10):
            batch = failed_list[i : i + 10]
            logger.warning(f"  {', '.join(batch)}")
        logger.warning("")
        logger.warning("These tickers may be:")
        logger.warning("  - Delisted or inactive during collection period")
        logger.warning("  - Invalid ticker symbols")
        logger.warning("  - Not available in any data source")

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

    # Calculate and log per-ticker coverage statistics
    logger.info("=" * 60)
    logger.info("DATA COVERAGE ANALYSIS")
    logger.info("=" * 60)

    # Calculate coverage per ticker (before gap filling)
    pre_fill_coverage = (combined_prices.notna().sum() / len(combined_prices)) * 100
    # Calculate coverage per ticker (after gap filling)
    post_fill_coverage = (filled_prices.notna().sum() / len(filled_prices)) * 100

    logger.info("Coverage distribution (after gap filling):")
    coverage_bins = [0, 50, 70, 80, 90, 95, 100]
    for i in range(len(coverage_bins) - 1):
        low, high = coverage_bins[i], coverage_bins[i + 1]
        count = ((post_fill_coverage >= low) & (post_fill_coverage < high)).sum()
        if i == len(coverage_bins) - 2:  # Last bin includes 100%
            count = (post_fill_coverage >= low).sum()
        pct = (count / len(post_fill_coverage)) * 100 if len(post_fill_coverage) > 0 else 0
        logger.info(f"  {low:>3}%-{high:>3}%: {count:>4} tickers ({pct:>5.1f}%)")

    # Identify low-coverage tickers
    low_coverage_threshold = 70
    low_coverage_tickers = post_fill_coverage[post_fill_coverage < low_coverage_threshold]
    if not low_coverage_tickers.empty:
        logger.warning("")
        logger.warning(
            f"⚠ Tickers with <{low_coverage_threshold}% coverage "
            f"({len(low_coverage_tickers)} tickers):"
        )
        for ticker in sorted(low_coverage_tickers.index)[:20]:
            cov = post_fill_coverage[ticker]
            logger.warning(f"  {ticker:.<8} {cov:>5.1f}%")
        if len(low_coverage_tickers) > 20:
            logger.warning(f"  ... and {len(low_coverage_tickers) - 20} more")

    # Overall statistics
    logger.info("")
    logger.info("Overall coverage statistics:")
    logger.info(f"  Mean coverage: {post_fill_coverage.mean():.2f}%")
    logger.info(f"  Median coverage: {post_fill_coverage.median():.2f}%")
    logger.info(f"  Min coverage: {post_fill_coverage.min():.2f}%")
    logger.info(f"  Max coverage: {post_fill_coverage.max():.2f}%")
    logger.info(f"  Std deviation: {post_fill_coverage.std():.2f}%")

    # Gap filling impact
    logger.info("")
    logger.info("Gap filling impact:")
    logger.info(f"  Average coverage before: {pre_fill_coverage.mean():.2f}%")
    logger.info(f"  Average coverage after: {post_fill_coverage.mean():.2f}%")
    logger.info(
        f"  Improvement: +{post_fill_coverage.mean() - pre_fill_coverage.mean():.2f} percentage points"
    )
    logger.info(f"  Total gaps filled: {total_filled:,}")

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
        "universe_coverage_pct": (len(collected_tickers) / len(all_tickers)) * 100,
        "date_range": f"{filled_prices.index.min()} to {filled_prices.index.max()}",
        "average_coverage": final_coverage.mean(),
        "tickers_95pct_coverage": (final_coverage >= 95).sum(),
        "total_gaps_filled": total_filled,
        "quality_score": validation_results.get("overall_quality_score", 0),
        "stooq_tickers": collection_stats.get("Stooq", 0),
        "tiingo_tickers": collection_stats.get("Tiingo", 0),
        "polygon_tickers": collection_stats.get("Polygon", 0),
        "yfinance_tickers": collection_stats.get("YFinance", 0),
    }

    # Compare to original merged dataset
    try:
        old_merged = pd.read_parquet("data/merged/prices.parquet")
        _ = len(old_merged.columns)  # Check we can read it
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
    logger.info("")

    logger.info("DATA COLLECTION SUMMARY BY SOURCE:")
    logger.info(f"  Target universe size: {summary['target_universe_size']} tickers")
    logger.info("")
    for source in ["Stooq", "Tiingo", "Polygon", "YFinance"]:
        source_key = f"{source.lower()}_tickers"
        count = summary.get(source_key, 0)
        pct = (
            (count / summary["target_universe_size"] * 100)
            if summary["target_universe_size"] > 0
            else 0
        )
        if count > 0:
            logger.info(f"  ✓ {source:.<12} {count:>4} tickers ({pct:>5.1f}%)")
        else:
            logger.info(f"  ✗ {source:.<12} {count:>4} tickers ({pct:>5.1f}%)")
    logger.info("")
    logger.info(
        f"  Total coverage: {summary['total_tickers']} tickers "
        f"({summary['universe_coverage_pct']:.1f}%)"
    )

    logger.info("")
    logger.info("DATA QUALITY METRICS:")
    logger.info(f"  Quality score: {summary['quality_score']:.3f}")
    logger.info(f"  Average ticker coverage: {summary['average_coverage']:.1f}%")
    logger.info(
        f"  Tickers with ≥95% coverage: {summary['tickers_95pct_coverage']} "
        f"({summary['tickers_95pct_coverage'] / summary['total_tickers'] * 100:.1f}%)"
    )
    logger.info(f"  Total gaps filled: {summary['total_gaps_filled']:,}")

    logger.info("")
    logger.info("DATA COMPLETENESS:")
    # Calculate additional completeness metrics
    if summary["total_tickers"] > 0 and summary["target_universe_size"] > 0:
        missing_tickers = summary["target_universe_size"] - summary["total_tickers"]
        if missing_tickers > 0:
            logger.warning(
                f"  ⚠ Missing {missing_tickers} tickers from target universe "
                f"({missing_tickers / summary['target_universe_size'] * 100:.1f}%)"
            )
        else:
            logger.info(f"  ✓ Complete universe coverage ({summary['total_tickers']} tickers)")

        # Calculate data density
        total_possible_values = summary["total_tickers"] * len(filled_prices)
        total_actual_values = filled_prices.notna().sum().sum()
        data_density = (
            (total_actual_values / total_possible_values * 100) if total_possible_values > 0 else 0
        )
        logger.info(
            f"  Data density: {data_density:.2f}% "
            f"({total_actual_values:,} / {total_possible_values:,} cells)"
        )

    logger.info("")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("  - prices_final.parquet")
    logger.info("  - volume_final.parquet")
    logger.info("  - returns_daily_final.parquet")
    logger.info("  - new_pipeline_summary.json")
    logger.info("=" * 80)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
