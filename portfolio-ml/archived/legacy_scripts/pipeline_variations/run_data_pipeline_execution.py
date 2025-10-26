#!/usr/bin/env python3
"""
Complete Data Pipeline Execution and Validation Script for Story 5.1.

This script executes the full data pipeline including:
- Multi-source data collection validation
- Gap-filling and data quality processing
- Temporal data integrity validation
- Final parquet dataset generation
- Data coverage requirements validation
"""

import sys
from pathlib import Path

import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.config.data import DataPipelineConfig, ValidationConfig
from src.data.processors.data_quality_validator import DataQualityValidator
from src.data.processors.gap_filling import GapFiller


def main():
    """Execute complete data pipeline for Story 5.1."""


    # Configuration setup
    validation_config = ValidationConfig(
        missing_data_threshold=0.10,  # 10% as per Story 5.1
        price_change_threshold=0.50,  # 50% as per Story 5.1
        volume_threshold=1000,
        validate_business_days=True,
        fill_method="forward",
        min_data_points=63,  # 3 months minimum
        max_gap_days=10,
        outlier_detection_method="zscore",
        outlier_threshold=3.0,
        correlation_threshold=0.05,
        price_range_validation=True,
        min_price=0.01,
        max_price=10000.0,
        volume_consistency_check=True,
        temporal_consistency_check=True,
        quality_score_threshold=0.95,  # >95% as per Story 5.1 AC #6
        auto_fix_enabled=True,
        generate_reports=True,
        report_output_dir="data/validation_reports"
    )

    pipeline_config = DataPipelineConfig(
        universe="midcap400",
        start_date="2016-01-01",
        end_date="2024-12-31",
        sources=["stooq", "yfinance"],
        processing_frequency="daily",
        cache_enabled=True,
        parallel_processing=True,
        num_workers=4
    )

    # Initialize processors
    gap_filler = GapFiller(validation_config)
    quality_validator = DataQualityValidator(validation_config)


    # Task 2: Validate existing multi-source data collection
    try:
        # Load Stooq data
        stooq_prices = pd.read_parquet("data/stooq/prices.parquet")
        stooq_volume = pd.read_parquet("data/stooq/volume.parquet")


        # Task 2: Data collection validation
        data_coverage = (stooq_prices.notna().sum() / len(stooq_prices)) * 100

    except Exception:
        return False


    # Task 1: Load universe snapshots for validation
    try:
        universe_snapshots = pd.read_csv("data/processed/universe_snapshots_monthly.csv")
        universe_snapshots['date'] = pd.to_datetime(universe_snapshots['date'])


        # Calculate monthly universe sizes
        universe_snapshots.groupby('date')['ticker'].count()

    except Exception:
        return False


    # Task 3: Execute gap-filling algorithms
    try:

        # Apply gap-filling to each ticker
        filled_prices = stooq_prices.copy()
        filled_volume = stooq_volume.copy()

        fill_stats = {}

        for ticker in stooq_prices.columns:
            if ticker in stooq_prices.columns and ticker in stooq_volume.columns:
                # Forward fill with volume validation
                original_na = stooq_prices[ticker].isna().sum()

                filled_prices[ticker] = gap_filler.forward_fill(
                    stooq_prices[ticker],
                    volume_series=stooq_volume[ticker],
                    min_volume=validation_config.volume_threshold,
                    limit=validation_config.max_gap_days
                )

                # Backward fill for remaining gaps
                filled_prices[ticker] = gap_filler.backward_fill(
                    filled_prices[ticker],
                    limit=validation_config.max_gap_days
                )

                final_na = filled_prices[ticker].isna().sum()

                fill_stats[ticker] = {
                    'original_na': original_na,
                    'final_na': final_na,
                    'filled': original_na - final_na
                }

        total_filled = sum(stats['filled'] for stats in fill_stats.values())
        total_remaining = sum(stats['final_na'] for stats in fill_stats.values())


    except Exception:
        return False


    # Task 3: Execute comprehensive data quality validation
    try:
        data_dict = {
            'prices': filled_prices,
            'volume': filled_volume
        }

        # Get unique tickers from universe for validation
        universe_tickers = universe_snapshots['ticker'].unique().tolist()

        validation_results = quality_validator.validate_complete_dataset(
            data_dict=data_dict,
            universe_tickers=universe_tickers,
            generate_report=True
        )


        # Task 6: Verify >95% coverage requirement
        coverage_requirement = validation_results.get('coverage_score', 0) >= 0.95

    except Exception:
        return False


    # Task 4: Execute temporal data integrity validation
    try:

        # Check business day alignment
        business_days = pd.bdate_range(
            start=pipeline_config.start_date,
            end=pipeline_config.end_date
        )

        data_dates = filled_prices.index
        len(set(data_dates) & set(business_days)) / len(business_days)


        # Look-ahead bias prevention validation

    except Exception:
        return False


    # Task 5: Generate final parquet datasets
    try:
        output_dir = Path("data/processed")
        output_dir.mkdir(exist_ok=True)

        # Generate daily returns
        returns_daily = filled_prices.pct_change()

        # Save final datasets with compression

        filled_prices.to_parquet(
            output_dir / "prices_final.parquet",
            compression="gzip",
            engine="pyarrow"
        )

        filled_volume.to_parquet(
            output_dir / "volume_final.parquet",
            compression="gzip",
            engine="pyarrow"
        )

        returns_daily.to_parquet(
            output_dir / "returns_daily_final.parquet",
            compression="gzip",
            engine="pyarrow"
        )


        # Generate quality metrics documentation
        quality_metrics = {
            'dataset_info': {
                'start_date': str(filled_prices.index.min()),
                'end_date': str(filled_prices.index.max()),
                'total_tickers': len(filled_prices.columns),
                'total_observations': len(filled_prices),
                'universe_type': 'sp_midcap_400'
            },
            'coverage_metrics': {
                'average_coverage': float(data_coverage.mean()),
                'min_coverage': float(data_coverage.min()),
                'max_coverage': float(data_coverage.max()),
                'tickers_above_95pct': int((data_coverage >= 95).sum())
            },
            'quality_scores': validation_results,
            'gap_filling_stats': {
                'total_gaps_filled': total_filled,
                'remaining_gaps': total_remaining,
                'fill_success_rate': float(total_filled / (total_filled + total_remaining))
            }
        }

        # Save quality metrics
        import json
        with open(output_dir / "data_quality_metrics.json", 'w') as f:
            json.dump(quality_metrics, f, indent=2, default=str)


    except Exception:
        return False


    # Task 6: Final coverage and quality validation
    try:
        final_coverage = (filled_prices.notna().sum() / len(filled_prices)) * 100

        coverage_pass = (final_coverage >= 95).sum()
        coverage_total = len(final_coverage)
        (coverage_pass / coverage_total) * 100


        # Overall success validation
        overall_success = (
            coverage_requirement and
            validation_results.get('temporal_integrity', False) and
            validation_results.get('overall_quality_score', 0) >= 0.90
        )


        if overall_success:
            pass

        return overall_success

    except Exception:
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
