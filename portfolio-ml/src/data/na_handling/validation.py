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
) -> Dict[str, Any]:
    """
    Calculate comprehensive data quality metrics.

    Args:
        returns: Prepared returns DataFrame
        universe: Original target universe
        masks: Validity masks from prepare_rolling_window_data()

    Returns:
        Dictionary containing:
        - requested_assets: Number of assets in target universe
        - available_assets: Number found in data
        - valid_assets: Number meeting quality criteria
        - coverage_ratio: Fraction of universe that's valid
        - na_count: Total NaN values in prepared data
        - na_ratio: Fraction of values that are NaN
    """
    metrics = {
        'requested_assets': len(universe),
        'available_assets': returns.shape[1],
        'valid_assets': masks['valid'].sum(),
        'coverage_ratio': masks['valid'].sum() / len(universe),
        'na_count': returns.isna().sum().sum(),
        'na_ratio': returns.isna().sum().sum() / returns.size if returns.size > 0 else 0.0,
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
