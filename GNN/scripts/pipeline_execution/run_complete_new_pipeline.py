#!/usr/bin/env python3
"""
Complete NEW Pipeline Implementation - Full Universe Collection
Using Stories 1.2 and 1.3 modular implementation to collect all 822 historical tickers.
"""

import sys
import time
from pathlib import Path

import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.config.data import CollectorConfig, UniverseConfig, ValidationConfig
from src.data.collectors.stooq import StooqCollector
from src.data.collectors.yfinance import YFinanceCollector
from src.data.processors.data_quality_validator import DataQualityValidator
from src.data.processors.gap_filling import GapFiller
from src.data.processors.universe_builder import UniverseBuilder


def collect_in_batches(collector, tickers, batch_size=50, start_date="2010-01-01", end_date="2024-12-31"):
    """Collect data in batches to avoid rate limiting."""

    all_prices = []
    all_volumes = []
    collected_tickers = []

    total_batches = (len(tickers) + batch_size - 1) // batch_size

    for i, start_idx in enumerate(range(0, len(tickers), batch_size)):
        batch = tickers[start_idx:start_idx + batch_size]

        try:
            # Use the correct collector methods based on type
            if hasattr(collector, 'collect_batch_data'):
                # YFinanceCollector method
                prices, volumes = collector.collect_batch_data(
                    batch, start_date=start_date, end_date=end_date
                )
            else:
                # StooqCollector method - use collect_prices_and_volume
                prices, volumes = collector.collect_prices_and_volume(
                    batch, start_date=start_date, end_date=end_date
                )

            if not prices.empty:
                all_prices.append(prices)
                all_volumes.append(volumes)
                collected_tickers.extend(prices.columns.tolist())
            else:
                pass

        except Exception:
            # Continue with next batch
            continue

        # Rate limiting delay between batches
        if i < total_batches - 1:  # Don't delay after last batch
            time.sleep(3)

    # Combine all collected data
    if all_prices:
        combined_prices = pd.concat(all_prices, axis=1)
        combined_volumes = pd.concat(all_volumes, axis=1)

        # Remove duplicates (in case of overlap)
        combined_prices = combined_prices.loc[:, ~combined_prices.columns.duplicated()]
        combined_volumes = combined_volumes.loc[:, ~combined_volumes.columns.duplicated()]

        return combined_prices, combined_volumes, collected_tickers
    else:
        return pd.DataFrame(), pd.DataFrame(), []


def main():
    """Execute our complete NEW pipeline implementation with full universe."""


    # Configuration setup
    universe_config = UniverseConfig(
        universe_type="midcap400",
        min_market_cap=None,
        min_avg_volume=None,
        exclude_sectors=None
    )

    stooq_config = CollectorConfig(
        source_name="stooq",
        rate_limit=10.0,
        timeout=15,
        retry_attempts=3,
        retry_delay=1.0
    )

    yfinance_config = CollectorConfig(
        source_name="yfinance",
        rate_limit=5.0,
        timeout=10,
        retry_attempts=3,
        retry_delay=1.0
    )

    validation_config = ValidationConfig(
        missing_data_threshold=0.10,
        price_change_threshold=0.50,
        volume_threshold=1000,
        validate_business_days=True,
        fill_method="forward",
        generate_reports=True,
        report_output_dir="data/validation_reports"
    )

    # Initialize processors
    universe_builder = UniverseBuilder(universe_config, "data/processed")
    stooq_collector = StooqCollector(stooq_config)
    yfinance_collector = YFinanceCollector(yfinance_config)
    gap_filler = GapFiller(validation_config)
    quality_validator = DataQualityValidator(validation_config)


    # Get the full historical universe
    try:
        universe_snapshots = pd.read_csv("data/processed/universe_snapshots_monthly.csv")
        universe_snapshots['date'] = pd.to_datetime(universe_snapshots['date'])
        all_tickers = sorted(universe_snapshots['ticker'].unique())
    except Exception:
        membership_df = universe_builder.build_membership_intervals("2016-01-01", "2024-12-31")
        universe_snapshots = universe_builder.create_monthly_snapshots(
            membership_df, "2016-01-01", "2024-12-31"
        )
        all_tickers = sorted(membership_df['ticker'].unique())
        universe_snapshots.to_csv("data/processed/universe_snapshots_monthly.csv", index=False)


    # Collect ALL historical tickers using our modular StooqCollector

    stooq_prices, stooq_volumes, collected_stooq = collect_in_batches(
        stooq_collector,
        all_tickers,
        batch_size=30,  # Smaller batches for better success rate
        start_date="2010-01-01",
        end_date="2024-12-31"
    )


    # Save Stooq results
    if not stooq_prices.empty:
        output_dir = Path("data/stooq_new")
        output_dir.mkdir(exist_ok=True)

        stooq_prices.to_parquet(output_dir / "prices.parquet", compression="gzip")
        stooq_volumes.to_parquet(output_dir / "volume.parquet", compression="gzip")


    # Identify missing tickers and get them from YFinance
    missing_tickers = list(set(all_tickers) - set(collected_stooq))

    if missing_tickers:

        yf_prices, yf_volumes, collected_yf = collect_in_batches(
            yfinance_collector,
            missing_tickers,
            batch_size=20,  # Even smaller for YFinance
            start_date="2010-01-01",
            end_date="2024-12-31"
        )


        # Merge Stooq and YFinance data
        if not stooq_prices.empty and not yf_prices.empty:

            # Use our YFinance collector's merge method
            combined_prices, combined_volumes = yfinance_collector.merge_with_primary_source(
                stooq_prices, stooq_volumes, yf_prices, yf_volumes
            )


        elif not stooq_prices.empty:
            combined_prices = stooq_prices
            combined_volumes = stooq_volumes
        elif not yf_prices.empty:
            combined_prices = yf_prices
            combined_volumes = yf_volumes
        else:
            return False
    else:
        combined_prices = stooq_prices
        combined_volumes = stooq_volumes


    # Apply our enhanced gap-filling

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
                limit=10
            )

            # Backward fill for remaining gaps
            filled_prices[ticker] = gap_filler.backward_fill(
                filled_prices[ticker],
                limit=10
            )

            final_na = filled_prices[ticker].isna().sum()
            fill_stats[ticker] = {
                'original_na': original_na,
                'final_na': final_na,
                'filled': original_na - final_na
            }

    total_filled = sum(stats['filled'] for stats in fill_stats.values())

    # Quality validation

    data_dict = {
        'prices': filled_prices,
        'volume': combined_volumes
    }

    validation_results = quality_validator.validate_complete_dataset(
        data_dict=data_dict,
        universe_tickers=all_tickers,
        generate_report=True
    )



    # Generate returns
    returns_daily = filled_prices.pct_change()

    # Save final datasets
    output_dir = Path("data/final_new_pipeline")
    output_dir.mkdir(exist_ok=True)

    filled_prices.to_parquet(output_dir / "prices_final.parquet", compression="gzip")
    combined_volumes.to_parquet(output_dir / "volume_final.parquet", compression="gzip")
    returns_daily.to_parquet(output_dir / "returns_daily_final.parquet", compression="gzip")

    # Calculate final metrics
    final_coverage = (filled_prices.notna().sum() / len(filled_prices)) * 100

    summary = {
        'total_tickers': len(filled_prices.columns),
        'target_universe_size': len(all_tickers),
        'universe_coverage_pct': (len(filled_prices.columns) / len(all_tickers)) * 100,
        'date_range': f"{filled_prices.index.min()} to {filled_prices.index.max()}",
        'average_coverage': final_coverage.mean(),
        'tickers_95pct_coverage': (final_coverage >= 95).sum(),
        'total_gaps_filled': total_filled,
        'quality_score': validation_results.get('overall_quality_score', 0),
        'stooq_tickers': len(collected_stooq),
        'yfinance_tickers': len(collected_yf) if 'collected_yf' in locals() else 0
    }


    # Compare to original merged dataset
    try:
        old_merged = pd.read_parquet("data/merged/prices.parquet")

        if len(filled_prices.columns) >= len(old_merged.columns):
            pass
        else:
            coverage_improvement = summary['average_coverage'] - 68.7  # Original was 68.7%

    except Exception:
        pass

    # Save summary
    import json
    with open(output_dir / "new_pipeline_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)


    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
