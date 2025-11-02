"""Imputation strategies for missing return data."""

from __future__ import annotations

from typing import Optional
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def cross_sectional_mean_impute(
    returns: pd.DataFrame,
    membership_mask: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Impute missing values with cross-sectional mean at each date.

    Based on Bryzgalova et al. (2024, Journal of Financial Economics) findings
    that simple cross-sectional mean imputation often outperforms sophisticated
    methods for portfolio optimisation due to block missingness patterns and
    low cross-sectional correlations in financial data.

    Args:
        returns: Returns DataFrame with NaN values
        membership_mask: Optional boolean mask of valid membership periods
            If provided, only computes cross-sectional mean over active assets

    Note on membership_mask sourcing:
        In rolling backtest context, membership_mask should be derived from
        universe_df using create_membership_mask() utility function from
        src.utils.membership_aware_cleaning module.

        Example usage in model context:
            >>> from src.utils.membership_aware_cleaning import create_membership_mask
            >>> membership_mask = create_membership_mask(returns, universe_df)
            >>> imputed = cross_sectional_mean_impute(returns, membership_mask)

    Returns:
        Imputed returns DataFrame with NaN replaced by cross-sectional means

    References:
        Bryzgalova et al. (2024): "Missing Financial Data"
        Journal of Financial Economics, https://arxiv.org/html/2207.13071v6

    Examples:
        >>> # Date 1: A=0.01, B=NaN, C=0.03 → B filled with 0.02 (mean of A,C)
        >>> imputed = cross_sectional_mean_impute(returns)
    """
    imputed = returns.copy()

    total_imputed = 0

    for date in returns.index:
        date_returns = returns.loc[date]

        if membership_mask is not None and date in membership_mask.index:
            active_mask = membership_mask.loc[date]
            active_returns = date_returns[active_mask]
        else:
            active_returns = date_returns

        observed_returns = active_returns.dropna()

        if len(observed_returns) == 0:
            logger.warning(f"No valid returns at date {date}")
            continue

        cs_mean = observed_returns.mean()

        missing_mask = date_returns.isna()

        if missing_mask.any():
            imputed.loc[date, missing_mask] = cs_mean
            total_imputed += missing_mask.sum()

    if total_imputed > 0:
        total_values = returns.size
        impute_ratio = total_imputed / total_values
        logger.info(
            f"Imputed {total_imputed}/{total_values} values ({impute_ratio:.1%}) "
            f"using cross-sectional mean"
        )

    return imputed


def impute_with_fallback(
    returns: pd.DataFrame,
    primary_method: str = "cross_sectional_mean",
    membership_mask: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Impute missing values with fallback strategy.

    Primary: Cross-sectional mean imputation (academic best practice)
    Fallback: Zero fill (conservative, avoids introducing false signals)

    Args:
        returns: Returns DataFrame with NaN values
        primary_method: Primary imputation method ('cross_sectional_mean', 'forward_fill', 'zero')
        membership_mask: Optional membership mask for cross-sectional mean

    Returns:
        Fully imputed returns DataFrame with no NaN values

    Raises:
        ValueError: If unknown imputation method specified
    """
    if primary_method == "cross_sectional_mean":
        imputed = cross_sectional_mean_impute(returns, membership_mask)
    elif primary_method == "forward_fill":
        imputed = returns.ffill()
    elif primary_method == "zero":
        imputed = returns.fillna(0.0)
    else:
        raise ValueError(f"Unknown imputation method: {primary_method}")

    if imputed.isna().any().any():
        remaining_na = imputed.isna().sum().sum()
        logger.warning(
            f"After {primary_method}, {remaining_na} NaN remain - filling with 0.0"
        )
        imputed = imputed.fillna(0.0)

    return imputed
