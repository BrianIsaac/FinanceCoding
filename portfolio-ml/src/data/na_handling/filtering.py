"""Column-wise filtering strategies for missing data handling."""

from __future__ import annotations

from typing import Tuple
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def filter_all_na_columns(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are entirely NA.

    Args:
        returns: Returns DataFrame with potential all-NA columns

    Returns:
        DataFrame with all-NA columns removed

    Examples:
        >>> data = pd.DataFrame({'A': [1, 2, 3], 'B': [np.nan, np.nan, np.nan], 'C': [4, 5, 6]})
        >>> filtered = filter_all_na_columns(data)
        >>> list(filtered.columns)
        ['A', 'C']
    """
    filtered = returns.dropna(axis=1, how='all')

    logger.debug(f"filter_all_na_columns: shape change {returns.shape} -> {filtered.shape}")

    dropped_count = len(returns.columns) - len(filtered.columns)
    if dropped_count > 0:
        logger.info(f"Dropped {dropped_count} all-NA columns")

    return filtered


def filter_by_coverage_threshold(
    returns: pd.DataFrame,
    coverage_threshold: float = 0.75,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Filter assets based on time-series data coverage.

    Calculates per-asset non-null ratio and keeps only assets meeting
    the coverage threshold. This prevents using assets with insufficient
    historical data that would bias statistical estimates.

    Args:
        returns: Returns DataFrame with datetime index
        coverage_threshold: Minimum ratio of non-null observations (0-1)

    Returns:
        Tuple of (filtered_returns, coverage_mask) where coverage_mask
        is a boolean Series indicating which assets meet threshold

    Examples:
        >>> # Asset A: 252/252 valid, B: 200/252 valid, C: 150/252 valid
        >>> filtered, mask = filter_by_coverage_threshold(returns, coverage_threshold=0.75)
        >>> # Returns A and B (both > 75%), excludes C
    """
    asset_coverage = returns.count() / len(returns)

    coverage_mask = asset_coverage >= coverage_threshold

    threshold_value = coverage_threshold
    passing_count = coverage_mask.sum()
    failing_count = (~coverage_mask).sum()
    logger.debug(
        f"Coverage threshold filter: threshold={threshold_value:.1%}, "
        f"passing_assets={passing_count}, failing_assets={failing_count}, "
        f"coverage_stats: min={asset_coverage.min():.1%}, "
        f"max={asset_coverage.max():.1%}, median={asset_coverage.median():.1%}"
    )

    sufficient_assets = coverage_mask[coverage_mask].index.tolist()

    if len(sufficient_assets) == 0:
        raise ValueError(
            f"No assets meet coverage threshold {coverage_threshold:.0%}. "
            f"Best coverage: {asset_coverage.max():.1%}"
        )

    dropped_count = len(returns.columns) - len(sufficient_assets)
    if dropped_count > 0:
        logger.info(
            f"Coverage filtering: kept {len(sufficient_assets)}/{len(returns.columns)} "
            f"assets (threshold: {coverage_threshold:.0%})"
        )

    filtered = returns[sufficient_assets]

    logger.info(
        f"Coverage filter: shape {returns.shape} -> {filtered.shape}, "
        f"mean_coverage={asset_coverage.mean():.1%}, "
        f"min_coverage={asset_coverage[coverage_mask].min():.1%}, "
        f"max_coverage={asset_coverage.max():.1%}"
    )

    return filtered, coverage_mask


def filter_zero_variance_assets(
    returns: pd.DataFrame,
    variance_threshold: float = 1e-5,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Remove assets with zero or near-zero variance.

    Zero-variance assets cause singularity issues in covariance matrices
    and provide no useful information for portfolio optimisation. The threshold
    of 1e-5 matches LSTM training expectations to ensure consistency across
    data filtering and model training stages.

    Args:
        returns: Returns DataFrame
        variance_threshold: Minimum standard deviation threshold (default: 1e-5)

    Returns:
        Tuple of (filtered_returns, variance_mask) where variance_mask
        is a boolean Series indicating assets with sufficient variance

    Examples:
        >>> # Asset A: std=0.02, B: std=1e-10, C: std=0.015
        >>> filtered, mask = filter_zero_variance_assets(returns)
        >>> # Returns A and C, excludes B (near-zero variance)
    """
    asset_std = returns.std()
    variance_mask = asset_std > variance_threshold

    valid_assets = variance_mask[variance_mask].index.tolist()

    if len(valid_assets) == 0:
        raise ValueError(
            f"No assets have variance > {variance_threshold}. "
            f"Max std: {asset_std.max():.2e}"
        )

    removed_count = len(returns.columns) - len(valid_assets)
    if removed_count > 0:
        logger.info(
            f"Variance filtering: removed {removed_count} zero-variance assets "
            f"(threshold: {variance_threshold:.2e})"
        )

    filtered = returns[valid_assets]

    logger.info(
        f"Variance filter: shape {returns.shape} -> {filtered.shape}, "
        f"mean_std={asset_std[variance_mask].mean():.4e}, "
        f"min_std={asset_std[variance_mask].min():.4e}, "
        f"max_std={asset_std.max():.4e}"
    )

    return filtered, variance_mask


def prepare_rolling_window_data(
    returns_window: pd.DataFrame,
    universe: list[str],
    coverage_threshold: float = 0.75,
    variance_threshold: float = 1e-5,
    return_masks: bool = True,
) -> Tuple[pd.DataFrame, dict[str, pd.Series]]:
    """
    Unified data preparation pipeline for rolling windows.

    Implements filtering strategy without forward fill (now handled at collection):
    1. Filter to available universe assets
    2. Drop all-NA columns
    3. Filter by coverage threshold
    4. Filter zero-variance assets

    Args:
        returns_window: Returns data for rolling window (already gap-filled at collection)
        universe: Target universe of assets
        coverage_threshold: Minimum non-null ratio (model-specific)
        variance_threshold: Minimum standard deviation (default: 1e-5, matches LSTM training)
        return_masks: Whether to return validity masks

    Returns:
        Tuple of (prepared_returns, masks_dict) where masks_dict contains:
        - 'coverage': Boolean Series of assets meeting coverage threshold
        - 'variance': Boolean Series of assets with sufficient variance
        - 'valid': Combined boolean Series (coverage AND variance)

    Raises:
        ValueError: If no assets meet quality criteria

    Examples:
        >>> data, masks = prepare_rolling_window_data(
        ...     returns_window,
        ...     universe=['AAPL', 'MSFT', 'GOOGL'],
        ...     coverage_threshold=0.75,
        ... )
        >>> valid_assets = masks['valid'][masks['valid']].index.tolist()
    """
    available_assets = [a for a in universe if a in returns_window.columns]

    if len(available_assets) == 0:
        raise ValueError("No assets from universe found in returns data")

    logger.debug(
        f"Universe coverage: {len(available_assets)}/{len(universe)} assets available"
    )

    filtered = returns_window[available_assets].copy()

    filtered = filter_all_na_columns(filtered)

    filtered, coverage_mask = filter_by_coverage_threshold(
        filtered, coverage_threshold
    )

    filtered, variance_mask = filter_zero_variance_assets(
        filtered, variance_threshold
    )

    valid_mask = pd.Series(False, index=universe)
    valid_assets = filtered.columns.tolist()
    valid_mask[valid_assets] = True

    coverage_mask_aligned = pd.Series(False, index=universe)
    coverage_mask_aligned[coverage_mask.index] = coverage_mask

    variance_mask_aligned = pd.Series(False, index=universe)
    variance_mask_aligned[variance_mask.index] = variance_mask

    masks = {
        'coverage': coverage_mask_aligned,
        'variance': variance_mask_aligned,
        'valid': valid_mask,
    }

    logger.info(
        f"Prepared {len(filtered.columns)}/{len(universe)} assets "
        f"({len(filtered.columns)/len(universe):.0%} of universe)"
    )

    if return_masks:
        return filtered, masks
    else:
        return filtered, {}
