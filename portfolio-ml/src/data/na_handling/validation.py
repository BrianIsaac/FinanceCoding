"""Data quality validation for NA handling."""

from __future__ import annotations

from typing import Dict, Any
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def calculate_data_quality_metrics(
    returns: pd.DataFrame,
    universe: list[str],
    masks: dict[str, pd.Series],
    date: pd.Timestamp | None = None,
    verify_membership_aware: bool = True,
) -> Dict[str, Any]:
    """
    Calculate comprehensive data quality metrics with membership-aware coverage.

    CRITICAL: The universe parameter MUST be membership-filtered (i.e., only assets
    in the index at the given date), NOT the static all-time universe. Using static
    universe artificially deflates coverage by 25-30 percentage points.

    Example:
        - WRONG: universe = all 621 assets ever in S&P MidCap 400
          Results in coverage = 313/621 = 50.4% (artificially low)
        - CORRECT: universe = 400 assets in index on 2019-01-01
          Results in coverage = 313/400 = 78.3% (actual quality)

    Args:
        returns: Prepared returns DataFrame
        universe: Membership-filtered target universe (assets in index at date)
        masks: Validity masks from prepare_rolling_window_data()
        date: Optional date for verification logging (should match universe date)
        verify_membership_aware: If True and len(universe) > 500, log warning about
            potentially using static universe instead of membership-filtered

    Returns:
        Dictionary containing:
        - requested_assets: Number of assets in target universe (membership-filtered)
        - available_assets: Number found in data
        - valid_assets: Number meeting quality criteria
        - coverage_ratio: Fraction of membership-filtered universe that's valid
        - na_count: Total NaN values in prepared data
        - na_ratio: Fraction of values that are NaN
        - is_membership_aware: Boolean indicating if universe size suggests membership filtering

    Raises:
        Warning: If universe size > 500 and verify_membership_aware=True, suggesting
            static universe is being used instead of membership-filtered
    """
    universe_size = len(universe)

    # Verify universe is membership-filtered (not static 621)
    is_membership_aware = universe_size <= 500  # S&P MidCap 400 rarely exceeds 450

    if not is_membership_aware and date:
        logger.warning(
            f"ASSET ALIGNMENT ISSUE at {date.date()}: "
            f"universe_size={universe_size} suggests static all-time universe, "
            f"expected_membership_filtered_approx_400 for S&P MidCap 400, "
            f"this_causes_coverage_deflation_25_to_30_pct"
        )

    if verify_membership_aware and not is_membership_aware:
        logger.warning(
            f"Universe size {universe_size} suggests static all-time universe instead of "
            f"membership-filtered universe. Coverage ratio will be artificially low. "
            f"Expected size ~400 for S&P MidCap 400."
        )
        if date:
            logger.warning(f"  Date: {date.date()}")
            logger.warning(
                f"  Ensure universe is filtered using get_universe_at_date(universe_df, date)"
            )

    valid_count = masks['valid'].sum()
    coverage_ratio = valid_count / universe_size if universe_size > 0 else 0.0

    # Log coverage calculation for transparency
    if date:
        logger.debug(
            f"Data quality at {date.date()}: {valid_count}/{universe_size} assets valid "
            f"({coverage_ratio:.1%} coverage, {'membership-aware' if is_membership_aware else 'STATIC UNIVERSE'})"
        )

    metrics = {
        'requested_assets': universe_size,
        'available_assets': returns.shape[1],
        'valid_assets': valid_count,
        'coverage_ratio': coverage_ratio,
        'na_count': returns.isna().sum().sum(),
        'na_ratio': returns.isna().sum().sum() / returns.size if returns.size > 0 else 0.0,
        'is_membership_aware': is_membership_aware,
    }

    return metrics


def validate_prepared_data(
    returns: pd.DataFrame,
    min_assets: int = 2,
    max_na_ratio: float = 0.05,
) -> tuple[bool, str]:
    """
    Validate prepared returns data meets minimum quality standards.

    Args:
        returns: Prepared returns DataFrame
        min_assets: Minimum number of assets required
        max_na_ratio: Maximum allowable NaN ratio

    Returns:
        Tuple of (is_valid, error_message)

    Examples:
        >>> is_valid, msg = validate_prepared_data(returns, min_assets=10)
        >>> if not is_valid:
        ...     raise ValueError(msg)
    """
    if returns.shape[1] < min_assets:
        return False, f"Too few assets: {returns.shape[1]} < {min_assets}"

    na_ratio = returns.isna().sum().sum() / returns.size if returns.size > 0 else 0.0

    if na_ratio > max_na_ratio:
        return False, f"Too many NaN values: {na_ratio:.1%} > {max_na_ratio:.1%}"

    zero_var_count = (returns.std() <= 1e-8).sum()
    if zero_var_count > 0:
        return False, f"{zero_var_count} assets have zero variance"

    return True, "Data quality acceptable"
