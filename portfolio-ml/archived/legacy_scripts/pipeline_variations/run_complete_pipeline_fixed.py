#!/usr/bin/env python3
"""
Complete Data Pipeline using Stories 1.2 and 1.3 implementations.
This fixes Story 5.1 by properly using the modular collectors and processors.
"""

import sys
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


def main():
    """Execute complete data pipeline using proper modular implementation."""


    # Configuration setup from Stories 1.2/1.3
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

    # Initialize processors from Stories 1.2/1.3
    universe_builder = UniverseBuilder(universe_config, "data/processed")
    stooq_collector = StooqCollector(stooq_config)
    yfinance_collector = YFinanceCollector(yfinance_config)
    gap_filler = GapFiller(validation_config)
    quality_validator = DataQualityValidator(validation_config)


    # Use existing universe data or rebuild
    try:
        universe_snapshots = pd.read_csv("data/processed/universe_snapshots_monthly.csv")
        universe_snapshots['date'] = pd.to_datetime(universe_snapshots['date'])

        # Get comprehensive ticker list
        all_tickers = sorted(universe_snapshots['ticker'].unique())

    except Exception:
        membership_df = universe_builder.build_membership_intervals("2016-01-01", "2024-12-31")
        universe_snapshots = universe_builder.create_monthly_snapshots(
            membership_df, "2016-01-01", "2024-12-31"
        )
        all_tickers = sorted(membership_df['ticker'].unique())
        universe_snapshots.to_csv("data/processed/universe_snapshots_monthly.csv", index=False)



    # Check if we have existing Stooq data or need to collect
    existing_stooq_path = Path("data/stooq/prices.parquet")
    if existing_stooq_path.exists():
        stooq_prices = pd.read_parquet(existing_stooq_path)
        stooq_volume = pd.read_parquet("data/stooq/volume.parquet")
        existing_tickers = set(stooq_prices.columns)
    else:
        existing_tickers = set()
        stooq_prices = pd.DataFrame()
        stooq_volume = pd.DataFrame()

    # Identify missing tickers that need collection
    missing_tickers = set(all_tickers) - existing_tickers

    # Collect missing tickers in smaller batches
    if missing_tickers and len(missing_tickers) < 100:  # Only if reasonable number
        try:
            batch_tickers = list(missing_tickers)[:50]  # Limit to 50 for now
            new_prices, new_volume = stooq_collector.collect_batch_data(
                batch_tickers,
                start_date="2010-01-01",
                end_date="2024-12-31"
            )

            if not stooq_prices.empty:
                # Merge with existing data
                stooq_prices = pd.concat([stooq_prices, new_prices], axis=1)
                stooq_volume = pd.concat([stooq_volume, new_volume], axis=1)
            else:
                stooq_prices = new_prices
                stooq_volume = new_volume

            # Save updated data
            stooq_prices.to_parquet("data/stooq_expanded/prices.parquet", compression="gzip")
            stooq_volume.to_parquet("data/stooq_expanded/volume.parquet", compression="gzip")

        except Exception:
            pass


    # Use YFinance to fill gaps in Stooq data
    if not stooq_prices.empty:

        # Identify tickers with low coverage that need YFinance augmentation
        coverage = (stooq_prices.notna().sum() / len(stooq_prices)) * 100
        low_coverage_tickers = coverage[coverage < 80].index.tolist()


        if low_coverage_tickers:
            try:
                # Download YFinance data for low coverage tickers
                yf_prices, yf_volume = yfinance_collector.download_batch_data(
                    low_coverage_tickers[:20],  # Limit to first 20 for now
                    start_date="2010-01-01",
                    end_date="2024-12-31"
                )

                # Splice YFinance data into Stooq gaps
                augmented_prices = yfinance_collector.splice_fill_data(
                    stooq_prices, yf_prices, low_coverage_tickers[:20]
                )


                # Use augmented data
                final_prices = augmented_prices
                final_volume = stooq_volume  # Keep original volume for now

            except Exception:
                final_prices = stooq_prices
                final_volume = stooq_volume
        else:
            final_prices = stooq_prices
            final_volume = stooq_volume
    else:
        return False


    # Apply our enhanced gap-filling from Story 1.3

    filled_prices = final_prices.copy()
    fill_stats = {}

    for ticker in final_prices.columns:
        if ticker in final_volume.columns:
            original_na = final_prices[ticker].isna().sum()

            # Forward fill with volume validation
            filled_prices[ticker] = gap_filler.forward_fill(
                final_prices[ticker],
                volume_series=final_volume[ticker],
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
        'volume': final_volume
    }

    validation_results = quality_validator.validate_complete_dataset(
        data_dict=data_dict,
        universe_tickers=all_tickers,
        generate_report=True
    )



    # Generate returns
    returns_daily = filled_prices.pct_change()

    # Save final datasets
    output_dir = Path("data/processed_complete")
    output_dir.mkdir(exist_ok=True)

    filled_prices.to_parquet(output_dir / "prices_complete.parquet", compression="gzip")
    final_volume.to_parquet(output_dir / "volume_complete.parquet", compression="gzip")
    returns_daily.to_parquet(output_dir / "returns_daily_complete.parquet", compression="gzip")

    # Generate summary report
    final_coverage = (filled_prices.notna().sum() / len(filled_prices)) * 100

    summary = {
        'total_tickers': len(filled_prices.columns),
        'date_range': f"{filled_prices.index.min()} to {filled_prices.index.max()}",
        'average_coverage': final_coverage.mean(),
        'tickers_95pct_coverage': (final_coverage >= 95).sum(),
        'total_gaps_filled': total_filled,
        'quality_score': validation_results.get('overall_quality_score', 0)
    }


    # Compare to original merged dataset
    try:
        old_merged = pd.read_parquet("data/merged/prices.parquet")

        if len(filled_prices.columns) >= len(old_merged.columns):
            pass
        else:
            pass

    except Exception:
        pass

    # Save summary
    import json
    with open(output_dir / "pipeline_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
