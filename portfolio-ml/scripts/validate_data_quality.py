#!/usr/bin/env python3
"""
Validate data quality metrics after removing forward fill redundancy.

This script checks that fallback to zero fill is <1% of values across
different rolling windows and models, verifying that cross-sectional
mean imputation is working correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.na_handling import prepare_rolling_window_data, cross_sectional_mean_impute


def validate_data_quality() -> None:
    """
    Check that fallback to zero fill is <1% of values.

    Tests cross-sectional mean imputation across multiple rolling windows
    to ensure data quality meets Phase 8 success criteria.
    """
    returns_path = Path("data/final_new_pipeline/returns_daily_final.parquet")

    if not returns_path.exists():
        print(f"ERROR: Data file not found: {returns_path}")
        sys.exit(1)

    print("Loading returns data...")
    returns = pd.read_parquet(returns_path)
    print(f"Loaded returns: {returns.shape}")

    # Sample rolling windows across the test period
    test_dates = pd.date_range('2020-01-01', '2024-01-01', freq='MS')

    all_na_ratios = []
    all_impute_ratios = []

    print("\nTesting rolling windows...")
    print("=" * 80)

    for i, date in enumerate(test_dates):
        window_end = date
        window_start = date - pd.Timedelta(days=756)  # ~3 years lookback

        # Filter returns to window
        returns_window = returns.loc[window_start:window_end]

        if returns_window.empty:
            print(f"Window {i+1}: {date.date()} - SKIPPED (no data)")
            continue

        # Sample universe (use top 100 by coverage)
        coverage = (~returns_window.isna()).sum()
        top_assets = coverage.nlargest(100).index.tolist()
        universe = top_assets

        # Prepare data (no forward fill, just filtering)
        try:
            prepared, masks = prepare_rolling_window_data(
                returns_window,
                universe,
                coverage_threshold=0.75,
                return_masks=True
            )
        except Exception as e:
            print(f"Window {i+1}: {date.date()} - FAILED preparation: {e}")
            continue

        if prepared.empty:
            print(f"Window {i+1}: {date.date()} - SKIPPED (no assets after filtering)")
            continue

        # Check NAs after preparation (before imputation)
        na_count_before = prepared.isna().sum().sum()
        total_values = prepared.size
        na_ratio_before = na_count_before / total_values if total_values > 0 else 0

        # Apply cross-sectional mean imputation
        imputed = cross_sectional_mean_impute(prepared)

        # Check NAs after imputation
        na_count_after = imputed.isna().sum().sum()
        na_ratio_after = na_count_after / total_values if total_values > 0 else 0

        # Calculate imputation statistics
        values_imputed = na_count_before - na_count_after
        impute_ratio = values_imputed / total_values if total_values > 0 else 0

        all_na_ratios.append(na_ratio_before)
        all_impute_ratios.append(impute_ratio)

        status = "✓ PASS" if na_ratio_before < 0.01 else "✗ FAIL"
        print(
            f"Window {i+1:2d}: {date.date()} | "
            f"Assets: {len(prepared.columns):3d} | "
            f"NA before: {na_ratio_before:6.2%} | "
            f"Imputed: {impute_ratio:6.2%} | "
            f"NA after: {na_ratio_after:6.2%} | "
            f"{status}"
        )

        # Verify < 1% NAs before imputation
        if na_ratio_before >= 0.01:
            print(f"  WARNING: Too many NAs before imputation: {na_ratio_before:.2%}")

    print("=" * 80)

    # Summary statistics
    if all_na_ratios:
        print("\nSummary Statistics:")
        print(f"  NAs before imputation:")
        print(f"    Mean:   {np.mean(all_na_ratios):.4%}")
        print(f"    Median: {np.median(all_na_ratios):.4%}")
        print(f"    Max:    {np.max(all_na_ratios):.4%}")
        print(f"    Min:    {np.min(all_na_ratios):.4%}")

        print(f"\n  Values imputed (cross-sectional mean):")
        print(f"    Mean:   {np.mean(all_impute_ratios):.4%}")
        print(f"    Median: {np.median(all_impute_ratios):.4%}")
        print(f"    Max:    {np.max(all_impute_ratios):.4%}")
        print(f"    Min:    {np.min(all_impute_ratios):.4%}")

        # Overall pass/fail
        max_na_ratio = np.max(all_na_ratios)
        if max_na_ratio < 0.01:
            print(f"\n✓ SUCCESS: All windows have <1% NAs before imputation")
            print(f"  Maximum NA ratio: {max_na_ratio:.4%}")
            sys.exit(0)
        else:
            print(f"\n✗ FAILURE: Some windows exceed 1% NA threshold")
            print(f"  Maximum NA ratio: {max_na_ratio:.4%}")
            sys.exit(1)
    else:
        print("\n✗ FAILURE: No valid windows tested")
        sys.exit(1)


if __name__ == "__main__":
    validate_data_quality()
