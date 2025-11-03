"""Utilities for membership-aware data exploration and quality profiling.

This module provides reusable functions for exploring financial time series data
with proper respect for universe membership periods. ALL analysis functions ensure
that statistics are calculated only during active membership periods.

===== CRITICAL PRINCIPLE =====
Every function that analyses price/volume data MUST use membership information to
avoid incorrectly treating expected empty cells as "missing data".

DO NOT calculate statistics based on the date range in the raw parquet files.
The data was collected to follow membership activity, so ALL analysis must
respect when tickers were actually active members of the universe.

Use get_membership_mask() and filter_to_membership_periods() to ensure this.
===============================
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf, adfuller


def load_raw_data(
    raw_data_dir: Path = Path("data/final_new_pipeline/raw"),
    membership_path: Path = Path("data/processed/universe_calendar_midcap400.parquet"),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw prices, volumes, source mapping, and universe membership.

    Args:
        raw_data_dir: Directory containing raw parquet files
        membership_path: Path to universe calendar parquet

    Returns:
        Tuple of (prices_df, volumes_df, source_map_df, membership_df)
    """
    prices = pd.read_parquet(raw_data_dir / "prices_raw.parquet")
    volumes = pd.read_parquet(raw_data_dir / "volumes_raw.parquet")
    source_map = pd.read_parquet(raw_data_dir / "source_mapping.parquet")
    membership = pd.read_parquet(membership_path)

    # Ensure datetime indices
    prices.index = pd.to_datetime(prices.index)
    volumes.index = pd.to_datetime(volumes.index)
    membership['date'] = pd.to_datetime(membership['date'])

    return prices, volumes, source_map, membership


def get_membership_mask(
    prices: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    """Create a boolean mask indicating which cells should have data based on membership.

    This is the FOUNDATION for all membership-aware analysis. A cell is True if that
    ticker was an active member on that date according to the membership calendar.

    Args:
        prices: Prices DataFrame with DatetimeIndex and ticker columns
        membership: Universe calendar with 'date' and 'ticker' columns

    Returns:
        Boolean DataFrame with same shape as prices, True where ticker was active
    """
    # Create empty mask
    mask = pd.DataFrame(False, index=prices.index, columns=prices.columns)

    # For each monthly membership snapshot, mark active tickers
    for date in membership['date'].unique():
        active_tickers = membership[membership['date'] == date]['ticker'].values

        # Find the date range this snapshot covers (until next snapshot or end of data)
        next_dates = membership['date'].unique()
        next_dates = next_dates[next_dates > date]
        end_date = next_dates.min() if len(next_dates) > 0 else prices.index.max()

        # Mark all dates from this snapshot to next as active for these tickers
        date_mask = (prices.index >= date) & (prices.index < end_date)
        mask.loc[date_mask, active_tickers] = True

    return mask


def filter_to_membership_periods(
    df: pd.DataFrame,
    membership_mask: pd.DataFrame,
) -> pd.DataFrame:
    """Filter DataFrame to only include data during membership periods.

    Sets all cells outside membership periods to NaN. This ensures that any
    subsequent analysis (mean, std, quantiles, etc.) ONLY uses data during
    active membership.

    Args:
        df: Prices or volumes DataFrame
        membership_mask: Boolean mask from get_membership_mask()

    Returns:
        Filtered DataFrame with NaN outside membership periods
    """
    return df.where(membership_mask)


def profile_membership_aware_coverage(
    df: pd.DataFrame,
    membership_mask: pd.DataFrame,
) -> Dict[str, Union[float, int]]:
    """Calculate data coverage respecting membership periods (CORRECT analysis).

    This function ONLY looks at cells where tickers were active members.
    Empty cells outside membership are IGNORED (they're expected, not missing).

    Args:
        df: Prices or volumes DataFrame
        membership_mask: Boolean mask from get_membership_mask()

    Returns:
        Dictionary with membership-aware coverage statistics
    """
    # Only consider cells where ticker was active
    expected_cells = membership_mask.sum().sum()
    actual_cells = (membership_mask & ~df.isna()).sum().sum()

    return {
        "expected_cells": expected_cells,
        "actual_cells": actual_cells,
        "coverage_pct": (actual_cells / expected_cells * 100) if expected_cells > 0 else 0,
        "true_missing_cells": expected_cells - actual_cells,
        "true_missing_pct": ((expected_cells - actual_cells) / expected_cells * 100)
                            if expected_cells > 0 else 0,
    }


def detect_gaps_during_membership(
    df: pd.DataFrame,
    membership_mask: pd.DataFrame,
    min_gap_days: int = 5,
) -> pd.DataFrame:
    """Detect data gaps during active membership periods (TRUE missing data).

    Only gaps during membership periods are considered missing data.

    Args:
        df: Prices or volumes DataFrame
        membership_mask: Boolean mask from get_membership_mask()
        min_gap_days: Minimum consecutive days to count as a gap

    Returns:
        DataFrame with gap information (ticker, gap_start, gap_end, gap_days)
    """
    gaps = []

    for ticker in df.columns:
        if ticker not in membership_mask.columns:
            continue

        # Only look at data during membership
        active_dates = membership_mask[membership_mask[ticker]].index
        if len(active_dates) == 0:
            continue

        ticker_data = df.loc[active_dates, ticker]

        # Find gaps
        is_missing = ticker_data.isna()
        if not is_missing.any():
            continue

        gap_starts = is_missing & ~is_missing.shift(1, fill_value=False)
        gap_ends = is_missing & ~is_missing.shift(-1, fill_value=False)

        start_dates = ticker_data[gap_starts].index
        end_dates = ticker_data[gap_ends].index

        for start, end in zip(start_dates, end_dates):
            gap_days = (end - start).days + 1
            if gap_days >= min_gap_days:
                gaps.append({
                    'ticker': ticker,
                    'gap_start': start,
                    'gap_end': end,
                    'gap_days': gap_days,
                })

    return pd.DataFrame(gaps)


def analyse_membership_calendar(membership: pd.DataFrame) -> Dict[str, any]:
    """Analyse the universe membership calendar itself.

    Provides comprehensive EDA on the membership parquet structure.

    Args:
        membership: Universe calendar DataFrame

    Returns:
        Dictionary with membership statistics and insights
    """
    # Snapshot-level analysis
    snapshots_analysis = {
        "total_snapshots": membership['date'].nunique(),
        "date_range": f"{membership['date'].min()} to {membership['date'].max()}",
        "avg_constituents_per_snapshot": membership.groupby('date').size().mean(),
        "median_constituents_per_snapshot": membership.groupby('date').size().median(),
        "min_constituents": membership.groupby('date').size().min(),
        "max_constituents": membership.groupby('date').size().max(),
    }

    # Ticker-level analysis
    ticker_counts = membership.groupby('ticker').size()
    tickers_analysis = {
        "total_unique_tickers": membership['ticker'].nunique(),
        "avg_snapshots_per_ticker": ticker_counts.mean(),
        "median_snapshots_per_ticker": ticker_counts.median(),
        "min_snapshots": ticker_counts.min(),
        "max_snapshots": ticker_counts.max(),
    }

    # Start verification analysis
    verification_analysis = {
        "total_verified_starts": membership['start_verified'].sum(),
        "total_unverified_starts": (~membership['start_verified']).sum(),
        "verification_rate_pct": (membership['start_verified'].sum() / len(membership)) * 100,
    }

    # Metadata analysis
    metadata_analysis = {
        "index_names": membership['index_name'].unique().tolist(),
        "universe_types": membership['universe_type'].unique().tolist(),
        "rebalance_frequencies": membership['rebalance_frequency'].unique().tolist(),
        "data_sources": membership['data_source'].unique().tolist(),
        "algorithms": membership['algorithm'].unique().tolist(),
    }

    return {
        "snapshots": snapshots_analysis,
        "tickers": tickers_analysis,
        "verification": verification_analysis,
        "metadata": metadata_analysis,
    }


def analyse_source_attribution(source_map: pd.DataFrame) -> pd.DataFrame:
    """Analyse which data sources provided data for tickers.

    Args:
        source_map: Source mapping DataFrame with 'ticker' and 'source' columns

    Returns:
        DataFrame with source counts and percentages
    """
    counts = source_map['source'].value_counts()

    result = pd.DataFrame({
        "count": counts,
        "percentage": (counts / len(source_map)) * 100,
    })

    return result.sort_values("count", ascending=False)


# ============================================================================
# SECTION 8: TEMPORAL PROPERTIES AND PREDICTIVE CHARACTERISTICS
# ============================================================================


def compute_returns(prices: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
    """Compute returns from prices DataFrame.

    Args:
        prices: Prices DataFrame (should be membership-filtered)
        method: 'log' for log returns, 'simple' for arithmetic returns

    Returns:
        Returns DataFrame with same shape as prices
    """
    if method == 'log':
        returns = np.log(prices / prices.shift(1))
    elif method == 'simple':
        returns = prices.pct_change()
    else:
        raise ValueError(f"Unknown method: {method}. Use 'log' or 'simple'")

    return returns


def test_stationarity_adf(series: pd.Series) -> Dict[str, float]:
    """Run Augmented Dickey-Fuller test for stationarity.

    Args:
        series: Time series to test (must have sufficient non-NaN observations)

    Returns:
        Dictionary with test statistics and p-value
    """
    clean_series = series.dropna()

    if len(clean_series) < 20:
        return {
            'adf_statistic': np.nan,
            'p_value': np.nan,
            'critical_1pct': np.nan,
            'critical_5pct': np.nan,
            'is_stationary_5pct': False,
        }

    try:
        result = adfuller(clean_series, maxlag=None, autolag='AIC')

        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_1pct': result[4]['1%'],
            'critical_5pct': result[4]['5%'],
            'is_stationary_5pct': result[1] < 0.05,
        }
    except Exception:
        return {
            'adf_statistic': np.nan,
            'p_value': np.nan,
            'critical_1pct': np.nan,
            'critical_5pct': np.nan,
            'is_stationary_5pct': False,
        }


def analyse_returns_stationarity(returns: pd.DataFrame) -> pd.DataFrame:
    """Analyse stationarity across all tickers using ADF test.

    Args:
        returns: Returns DataFrame (membership-filtered)

    Returns:
        DataFrame with stationarity test results per ticker
    """
    results = []

    for ticker in returns.columns:
        ticker_returns = returns[ticker]
        adf_results = test_stationarity_adf(ticker_returns)
        adf_results['ticker'] = ticker
        adf_results['n_obs'] = ticker_returns.notna().sum()
        results.append(adf_results)

    return pd.DataFrame(results)


def compute_autocorrelation_profile(
    series: pd.Series,
    nlags: int = 20,
) -> Dict[str, np.ndarray]:
    """Compute ACF and PACF for a time series.

    Args:
        series: Time series
        nlags: Number of lags to compute

    Returns:
        Dictionary with ACF and PACF arrays
    """
    clean_series = series.dropna()

    if len(clean_series) < nlags + 10:
        return {
            'acf': np.full(nlags + 1, np.nan),
            'pacf': np.full(nlags + 1, np.nan),
        }

    try:
        acf_values = acf(clean_series, nlags=nlags, fft=False)
        pacf_values = pacf(clean_series, nlags=nlags, method='ywm')

        return {
            'acf': acf_values,
            'pacf': pacf_values,
        }
    except Exception:
        return {
            'acf': np.full(nlags + 1, np.nan),
            'pacf': np.full(nlags + 1, np.nan),
        }


def analyse_cross_sectional_autocorrelation(
    returns: pd.DataFrame,
    nlags: int = 20,
) -> Dict[str, np.ndarray]:
    """Analyse autocorrelation aggregated across all tickers.

    Args:
        returns: Returns DataFrame (membership-filtered)
        nlags: Number of lags to compute

    Returns:
        Dictionary with mean ACF/PACF and confidence intervals
    """
    all_acf = []
    all_pacf = []

    for ticker in returns.columns:
        result = compute_autocorrelation_profile(returns[ticker], nlags=nlags)
        if not np.isnan(result['acf'][0]):
            all_acf.append(result['acf'])
            all_pacf.append(result['pacf'])

    if len(all_acf) == 0:
        return {
            'mean_acf': np.full(nlags + 1, np.nan),
            'mean_pacf': np.full(nlags + 1, np.nan),
            'std_acf': np.full(nlags + 1, np.nan),
            'std_pacf': np.full(nlags + 1, np.nan),
        }

    all_acf = np.array(all_acf)
    all_pacf = np.array(all_pacf)

    return {
        'mean_acf': np.nanmean(all_acf, axis=0),
        'mean_pacf': np.nanmean(all_pacf, axis=0),
        'std_acf': np.nanstd(all_acf, axis=0),
        'std_pacf': np.nanstd(all_pacf, axis=0),
        'median_acf': np.nanmedian(all_acf, axis=0),
        'median_pacf': np.nanmedian(all_pacf, axis=0),
    }


def compute_variance_ratio(series: pd.Series, q: int) -> float:
    """Compute variance ratio test statistic.

    VR(q) = Var(r_t + ... + r_{t+q-1}) / (q * Var(r_t))
    Under random walk, VR(q) = 1

    Args:
        series: Returns time series
        q: Holding period (number of periods)

    Returns:
        Variance ratio statistic
    """
    clean_series = series.dropna()

    if len(clean_series) < q * 3:
        return np.nan

    # Variance of 1-period returns
    var_1 = np.var(clean_series, ddof=1)

    if var_1 == 0:
        return np.nan

    # Variance of q-period returns
    q_period_returns = clean_series.rolling(window=q).sum().dropna()
    var_q = np.var(q_period_returns, ddof=1)

    # Variance ratio
    vr = var_q / (q * var_1)

    return vr


def analyse_variance_ratios(
    returns: pd.DataFrame,
    periods: List[int] = [2, 5, 10, 20],
) -> pd.DataFrame:
    """Compute variance ratios across all tickers.

    Args:
        returns: Returns DataFrame (membership-filtered)
        periods: List of holding periods to test

    Returns:
        DataFrame with variance ratios per ticker and period
    """
    results = []

    for ticker in returns.columns:
        ticker_result = {'ticker': ticker}
        for q in periods:
            vr = compute_variance_ratio(returns[ticker], q)
            ticker_result[f'VR_{q}'] = vr
        results.append(ticker_result)

    return pd.DataFrame(results)


def compute_rolling_correlation_matrix(
    returns: pd.DataFrame,
    window: int = 252,
    min_periods: int = 126,
) -> List[pd.DataFrame]:
    """Compute rolling correlation matrices.

    Args:
        returns: Returns DataFrame (membership-filtered)
        window: Rolling window size
        min_periods: Minimum periods for correlation calculation

    Returns:
        List of correlation matrices, one per time point
    """
    correlation_matrices = []

    for i in range(min_periods, len(returns)):
        window_returns = returns.iloc[max(0, i - window):i]

        # Only compute if we have enough data
        if len(window_returns) >= min_periods:
            corr_matrix = window_returns.corr(min_periods=min_periods)
            correlation_matrices.append({
                'date': returns.index[i],
                'correlation_matrix': corr_matrix,
            })

    return correlation_matrices


def analyse_correlation_dynamics(
    returns: pd.DataFrame,
    window: int = 252,
) -> pd.DataFrame:
    """Analyse dynamics of average pairwise correlation over time.

    Args:
        returns: Returns DataFrame (membership-filtered)
        window: Rolling window size for correlation computation

    Returns:
        DataFrame with time series of correlation statistics
    """
    correlation_stats = []

    for i in range(window, len(returns)):
        window_returns = returns.iloc[i - window:i]

        # Compute correlation matrix
        corr_matrix = window_returns.corr(min_periods=int(window * 0.5))

        if corr_matrix.shape[0] > 1:
            # Extract upper triangle (exclude diagonal)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            correlations = corr_matrix.values[mask]

            # Remove NaN values
            correlations = correlations[~np.isnan(correlations)]

            if len(correlations) > 0:
                correlation_stats.append({
                    'date': returns.index[i],
                    'mean_correlation': np.mean(correlations),
                    'median_correlation': np.median(correlations),
                    'std_correlation': np.std(correlations),
                    'min_correlation': np.min(correlations),
                    'max_correlation': np.max(correlations),
                    'n_pairs': len(correlations),
                })

    return pd.DataFrame(correlation_stats).set_index('date')


def test_arch_effects(series: pd.Series, lags: int = 5) -> Dict[str, float]:
    """Test for ARCH effects (volatility clustering) using Ljung-Box test on squared returns.

    Args:
        series: Returns time series
        lags: Number of lags for Ljung-Box test

    Returns:
        Dictionary with test statistics
    """
    clean_series = series.dropna()

    if len(clean_series) < lags * 3:
        return {
            'lb_statistic': np.nan,
            'p_value': np.nan,
            'has_arch_effects': False,
        }

    # Compute squared returns
    squared_returns = clean_series ** 2

    # Simple Ljung-Box approximation
    n = len(squared_returns)
    acf_squared = acf(squared_returns, nlags=lags, fft=False)

    # Ljung-Box statistic
    lb_stat = n * (n + 2) * np.sum(acf_squared[1:] ** 2 / (n - np.arange(1, lags + 1)))

    # Chi-square test with lags degrees of freedom
    p_value = 1 - stats.chi2.cdf(lb_stat, lags)

    return {
        'lb_statistic': lb_stat,
        'p_value': p_value,
        'has_arch_effects': p_value < 0.05,
    }


def analyse_arch_effects_cross_section(returns: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    """Test for ARCH effects across all tickers.

    Args:
        returns: Returns DataFrame (membership-filtered)
        lags: Number of lags for test

    Returns:
        DataFrame with ARCH test results per ticker
    """
    results = []

    for ticker in returns.columns:
        arch_result = test_arch_effects(returns[ticker], lags=lags)
        arch_result['ticker'] = ticker
        results.append(arch_result)

    return pd.DataFrame(results)
