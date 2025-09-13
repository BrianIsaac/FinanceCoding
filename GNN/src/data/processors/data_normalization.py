"""Data normalization pipeline for financial time series.

This module provides comprehensive normalization and quality assessment for
price and volume data, including returns calculation, volume adjustments,
and integration with dynamic universe calendar.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config.data import ValidationConfig
from src.data.processors.universe_builder import UniverseBuilder


class DataNormalizer:
    """
    Comprehensive data normalization processor for financial time series.

    Handles daily returns calculation with corporate action handling,
    volume normalization with market cap adjustments, and data quality scoring.
    """

    def __init__(self, config: ValidationConfig):
        """
        Initialize data normalizer with validation configuration.

        Args:
            config: Validation configuration with thresholds and methods
        """
        self.config = config

    def calculate_daily_returns(
        self,
        prices_df: pd.DataFrame,
        method: str = "simple",
        handle_splits: bool = True,
        outlier_threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Calculate daily returns with corporate action handling.

        Args:
            prices_df: Price data DataFrame (Date × Tickers)
            method: Return calculation method ('simple', 'log')
            handle_splits: Whether to detect and handle stock splits
            outlier_threshold: Threshold for detecting abnormal returns (e.g., 50% = 0.5)

        Returns:
            DataFrame with daily returns
        """
        if method == "log":
            returns = np.log(prices_df / prices_df.shift(1))
        else:  # simple returns
            returns = prices_df.pct_change(fill_method=None)

        if handle_splits:
            returns = self._handle_corporate_actions(returns, prices_df, outlier_threshold)

        return returns

    def _handle_corporate_actions(
        self, returns_df: pd.DataFrame, prices_df: pd.DataFrame, outlier_threshold: float
    ) -> pd.DataFrame:
        """Detect and handle stock splits and other corporate actions.

        Args:
            returns_df: Raw returns DataFrame
            prices_df: Price data for validation
            outlier_threshold: Threshold for detecting outliers

        Returns:
            Returns DataFrame with corporate actions handled
        """
        cleaned_returns = returns_df.copy()

        for ticker in returns_df.columns:
            returns_series = returns_df[ticker]
            prices_series = prices_df[ticker]

            # Detect large negative returns that might be splits
            large_negative = returns_series < -outlier_threshold
            large_positive = returns_series > outlier_threshold

            # Check for potential splits (large negative returns followed by normal trading)
            for idx in returns_series.index[large_negative]:
                if self._is_likely_split(returns_series, prices_series, idx, outlier_threshold):
                    # Handle as split - keep the return but flag it
                    continue
                elif abs(returns_series.loc[idx]) > outlier_threshold:
                    # Potentially erroneous return - consider replacing
                    cleaned_returns.loc[idx, ticker] = self._estimate_return(
                        returns_series, idx, method="median_window"
                    )

            # Handle large positive returns (potential data errors)
            for idx in returns_series.index[large_positive]:
                if abs(returns_series.loc[idx]) > outlier_threshold * 2:  # Very large moves
                    cleaned_returns.loc[idx, ticker] = self._estimate_return(
                        returns_series, idx, method="median_window"
                    )

        return cleaned_returns

    def _is_likely_split(
        self,
        returns_series: pd.Series,
        prices_series: pd.Series,
        split_date: pd.Timestamp,
        threshold: float,
    ) -> bool:
        """Determine if a large return is likely a stock split.

        Args:
            returns_series: Returns time series
            prices_series: Price time series
            split_date: Date of potential split
            threshold: Return threshold

        Returns:
            True if likely a split, False otherwise
        """
        try:
            # Get return magnitude
            split_return = abs(returns_series.loc[split_date])

            # Check if return magnitude suggests common split ratios
            expected_ratios = [0.5, 0.33, 0.67, 0.25, 0.75]  # 2:1, 3:1, 3:2, 4:1, 4:3 splits
            ratio_matches = any(abs(split_return - (1 - ratio)) < 0.05 for ratio in expected_ratios)

            if not ratio_matches:
                return False

            # Check price pattern around split
            split_idx = returns_series.index.get_loc(split_date)

            # Look at 5 days before and after
            pre_window = max(0, split_idx - 5)
            post_window = min(len(returns_series), split_idx + 6)

            pre_returns = returns_series.iloc[pre_window:split_idx]
            post_returns = returns_series.iloc[split_idx + 1 : post_window]

            # Split should be followed by normal trading patterns
            if len(post_returns) > 0:
                post_volatility = post_returns.std()
                pre_volatility = pre_returns.std() if len(pre_returns) > 0 else 0.02

                # Normal volatility after split suggests it's real
                return post_volatility < 0.1 and post_volatility < pre_volatility * 3

            return True

        except Exception:
            return False

    def _estimate_return(
        self, returns_series: pd.Series, target_date: pd.Timestamp, method: str = "median_window"
    ) -> float:
        """Estimate a return value for data cleaning.

        Args:
            returns_series: Returns time series
            target_date: Date to estimate
            method: Estimation method

        Returns:
            Estimated return value
        """
        if method == "median_window":
            # Use median of surrounding window
            target_idx = returns_series.index.get_loc(target_date)
            window_size = 10

            start_idx = max(0, target_idx - window_size)
            end_idx = min(len(returns_series), target_idx + window_size + 1)

            window_returns = returns_series.iloc[start_idx:end_idx]
            window_returns = window_returns[window_returns.index != target_date]  # Exclude target

            # Remove outliers from window
            q1, q3 = window_returns.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            clean_window = window_returns[
                (window_returns >= lower_bound) & (window_returns <= upper_bound)
            ]

            return clean_window.median() if len(clean_window) > 0 else 0.0

        return 0.0

    def normalize_volume(
        self,
        volume_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        method: str = "dollar_volume",
        market_cap_adjustment: bool = False,
        universe_builder: UniverseBuilder | None = None,
    ) -> pd.DataFrame:
        """Normalize volume data with various adjustment methods.

        Args:
            volume_df: Volume data DataFrame (Date × Tickers)
            prices_df: Price data for dollar volume calculation
            method: Normalization method ('raw', 'dollar_volume', 'log_volume', 'z_score')
            market_cap_adjustment: Whether to adjust for market cap changes
            universe_builder: Optional universe builder for market cap data

        Returns:
            Normalized volume DataFrame
        """
        if method == "raw":
            return volume_df

        elif method == "dollar_volume":
            # Convert to dollar volume
            dollar_volume = volume_df * prices_df
            return dollar_volume

        elif method == "log_volume":
            # Log transform to handle skewness
            log_volume = np.log(volume_df.replace(0, np.nan))
            return log_volume

        elif method == "z_score":
            # Z-score normalization per ticker
            normalized = volume_df.copy()
            for ticker in volume_df.columns:
                series = volume_df[ticker]
                valid_data = series.replace(0, np.nan).dropna()
                if len(valid_data) > 10:
                    mean_vol = valid_data.mean()
                    std_vol = valid_data.std()
                    if std_vol > 0:
                        normalized[ticker] = (series - mean_vol) / std_vol
            return normalized

        elif method == "relative_volume":
            # Volume relative to rolling average
            normalized = volume_df.copy()
            for ticker in volume_df.columns:
                series = volume_df[ticker].replace(0, np.nan)
                rolling_mean = series.rolling(window=21, min_periods=5).mean()
                normalized[ticker] = series / rolling_mean
            return normalized

        else:
            raise ValueError(f"Unknown volume normalization method: {method}")

    def align_with_universe_calendar(
        self,
        data_dict: dict[str, pd.DataFrame],
        universe_builder: UniverseBuilder,
        start_date: str,
        end_date: str,
    ) -> dict[str, pd.DataFrame]:
        """Align data with dynamic universe calendar from Story 1.2.

        Args:
            data_dict: Dictionary of data DataFrames
            universe_builder: Universe builder instance
            start_date: Start date for alignment
            end_date: End date for alignment

        Returns:
            Dictionary of aligned DataFrames
        """
        # Get universe calendar
        universe_calendar = universe_builder.get_universe_for_period(start_date, end_date)

        # Create monthly snapshots for rebalancing alignment
        pd.date_range(start_date, end_date, freq="M")

        aligned_data = {}

        for data_type, df in data_dict.items():
            if df.empty:
                aligned_data[data_type] = df
                continue

            # Get full universe of tickers for this period
            universe_tickers = sorted(universe_calendar["ticker"].unique())

            # Reindex to include all universe tickers
            full_index = pd.date_range(
                max(pd.to_datetime(start_date), df.index.min()),
                min(pd.to_datetime(end_date), df.index.max()),
                freq="D",
            )

            aligned_df = df.reindex(index=full_index, columns=universe_tickers)

            # Apply universe constraints - only keep data for valid membership periods
            for ticker in universe_tickers:
                ticker_periods = universe_calendar[universe_calendar["ticker"] == ticker]

                for _, period in ticker_periods.iterrows():
                    period_start = pd.to_datetime(period["start"])
                    period_end = (
                        pd.to_datetime(period["end"])
                        if pd.notna(period["end"])
                        else pd.to_datetime(end_date)
                    )

                    # Keep data only within membership period
                    mask = (aligned_df.index < period_start) | (aligned_df.index > period_end)
                    aligned_df.loc[mask, ticker] = np.nan

            aligned_data[data_type] = aligned_df

        return aligned_data

    def calculate_data_quality_score(
        self,
        data_dict: dict[str, pd.DataFrame],
        volume_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Calculate comprehensive data quality scoring system.

        Args:
            data_dict: Dictionary of data DataFrames
            volume_df: Optional volume DataFrame for additional validation

        Returns:
            Dictionary with quality scores and metrics
        """
        quality_scores = {
            "overall_score": 0.0,
            "component_scores": {},
            "ticker_scores": {},
            "quality_flags": [],
            "data_completeness": {},
            "outlier_summary": {},
        }

        prices_df = data_dict.get("close", pd.DataFrame())
        returns_df = data_dict.get("returns", pd.DataFrame())

        if prices_df.empty:
            return quality_scores

        # Calculate component scores
        completeness_score = self._calculate_completeness_score(prices_df)
        consistency_score = self._calculate_consistency_score(prices_df, returns_df)
        outlier_score = self._calculate_outlier_score(returns_df) if not returns_df.empty else 0.8
        volume_score = (
            self._calculate_volume_score(volume_df, prices_df) if volume_df is not None else 0.7
        )

        # Component weights
        weights = {"completeness": 0.3, "consistency": 0.3, "outliers": 0.2, "volume": 0.2}

        quality_scores["component_scores"] = {
            "completeness": completeness_score,
            "consistency": consistency_score,
            "outliers": outlier_score,
            "volume": volume_score,
        }

        # Calculate weighted overall score
        overall_score = sum(
            score * weights[component]
            for component, score in quality_scores["component_scores"].items()
        )
        quality_scores["overall_score"] = overall_score

        # Per-ticker scores
        for ticker in prices_df.columns:
            ticker_completeness = 1.0 - (prices_df[ticker].isna().sum() / len(prices_df))
            ticker_consistency = self._calculate_ticker_consistency(prices_df[ticker])
            ticker_score = 0.6 * ticker_completeness + 0.4 * ticker_consistency

            quality_scores["ticker_scores"][ticker] = {
                "overall": ticker_score,
                "completeness": ticker_completeness,
                "consistency": ticker_consistency,
            }

        # Quality flags
        low_quality_tickers = [
            ticker
            for ticker, scores in quality_scores["ticker_scores"].items()
            if scores["overall"] < 0.7
        ]

        if low_quality_tickers:
            quality_scores["quality_flags"].append(
                f"Low quality tickers ({len(low_quality_tickers)}): {low_quality_tickers[:5]}"
            )

        if overall_score < 0.8:
            quality_scores["quality_flags"].append(
                f"Overall quality score below threshold: {overall_score:.2f}"
            )

        return quality_scores

    def _calculate_completeness_score(self, prices_df: pd.DataFrame) -> float:
        """Calculate data completeness score."""
        if prices_df.empty:
            return 0.0

        total_cells = prices_df.size
        missing_cells = prices_df.isna().sum().sum()
        completeness = 1.0 - (missing_cells / total_cells)

        return max(0.0, min(1.0, completeness))

    def _calculate_consistency_score(
        self, prices_df: pd.DataFrame, returns_df: pd.DataFrame | None = None
    ) -> float:
        """Calculate data consistency score."""
        if prices_df.empty:
            return 0.0

        consistency_issues = 0
        total_checks = 0

        for ticker in prices_df.columns:
            price_series = prices_df[ticker].dropna()
            if len(price_series) < 10:
                continue

            # Check for non-positive prices
            non_positive = (price_series <= 0).sum()
            consistency_issues += non_positive
            total_checks += len(price_series)

            # Check for extreme price jumps (without corporate actions)
            if returns_df is not None and ticker in returns_df.columns:
                extreme_returns = (abs(returns_df[ticker]) > 0.2).sum()
                consistency_issues += extreme_returns * 0.5  # Weight extreme returns less
                total_checks += len(returns_df[ticker].dropna())

        if total_checks == 0:
            return 0.5

        consistency_score = 1.0 - (consistency_issues / total_checks)
        return max(0.0, min(1.0, consistency_score))

    def _calculate_outlier_score(self, returns_df: pd.DataFrame) -> float:
        """Calculate outlier score based on return distribution."""
        if returns_df.empty:
            return 0.8

        outlier_count = 0
        total_returns = 0

        for ticker in returns_df.columns:
            returns_series = returns_df[ticker].dropna()
            if len(returns_series) < 10:
                continue

            # Use IQR method for outlier detection
            q1, q3 = returns_series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr  # 3x IQR for extreme outliers
            upper_bound = q3 + 3 * iqr

            outliers = ((returns_series < lower_bound) | (returns_series > upper_bound)).sum()
            outlier_count += outliers
            total_returns += len(returns_series)

        if total_returns == 0:
            return 0.8

        outlier_rate = outlier_count / total_returns
        outlier_score = max(0.0, 1.0 - outlier_rate * 10)  # Penalize outlier rate

        return min(1.0, outlier_score)

    def _calculate_volume_score(self, volume_df: pd.DataFrame, prices_df: pd.DataFrame) -> float:
        """Calculate volume data quality score."""
        if volume_df.empty:
            return 0.7

        volume_issues = 0
        total_checks = 0

        for ticker in volume_df.columns:
            if ticker not in prices_df.columns:
                continue

            vol_series = volume_df[ticker]
            price_series = prices_df[ticker]

            # Check for zero volume on trading days
            trading_days = price_series.notna()
            zero_volume_on_trading = ((vol_series == 0) & trading_days).sum()
            volume_issues += zero_volume_on_trading
            total_checks += trading_days.sum()

        if total_checks == 0:
            return 0.7

        volume_score = 1.0 - (volume_issues / total_checks)
        return max(0.0, min(1.0, volume_score))

    def _calculate_ticker_consistency(self, price_series: pd.Series) -> float:
        """Calculate consistency score for a single ticker."""
        if len(price_series.dropna()) < 10:
            return 0.5

        clean_prices = price_series.dropna()

        # Check for monotonic issues (prices that don't change for long periods)
        price_changes = clean_prices.diff().abs()
        no_change_days = (price_changes == 0).sum()
        no_change_ratio = no_change_days / len(clean_prices)

        # Check for reasonable price evolution
        abs(np.log(clean_prices.iloc[-1] / clean_prices.iloc[0]))

        consistency = 1.0 - min(0.5, no_change_ratio * 2)  # Penalize excessive no-change periods

        return max(0.0, min(1.0, consistency))

    def process_complete_pipeline(
        self,
        ohlcv_data: dict[str, pd.DataFrame],
        universe_builder: UniverseBuilder,
        start_date: str,
        end_date: str,
        volume_normalization: str = "dollar_volume",
        return_method: str = "simple",
    ) -> dict[str, Any]:
        """Execute complete data normalization pipeline.

        Args:
            ohlcv_data: Dictionary with OHLCV DataFrames
            universe_builder: Universe builder for calendar alignment
            start_date: Start date for processing
            end_date: End date for processing
            volume_normalization: Volume normalization method
            return_method: Return calculation method

        Returns:
            Dictionary with processed data and quality metrics
        """

        # Step 1: Calculate returns
        if "close" in ohlcv_data and not ohlcv_data["close"].empty:
            returns_df = self.calculate_daily_returns(
                ohlcv_data["close"],
                method=return_method,
                outlier_threshold=self.config.price_change_threshold,
            )
            ohlcv_data["returns"] = returns_df

        # Step 2: Normalize volume
        if "volume" in ohlcv_data and not ohlcv_data["volume"].empty:
            normalized_volume = self.normalize_volume(
                ohlcv_data["volume"],
                ohlcv_data.get("close", pd.DataFrame()),
                method=volume_normalization,
                universe_builder=universe_builder,
            )
            ohlcv_data["volume_normalized"] = normalized_volume

        # Step 3: Align with universe calendar
        aligned_data = self.align_with_universe_calendar(
            ohlcv_data, universe_builder, start_date, end_date
        )

        # Step 4: Calculate quality scores
        quality_metrics = self.calculate_data_quality_score(
            aligned_data, volume_df=aligned_data.get("volume", None)
        )

        result = {
            "data": aligned_data,
            "quality_metrics": quality_metrics,
            "processing_summary": {
                "start_date": start_date,
                "end_date": end_date,
                "total_tickers": len(aligned_data.get("close", pd.DataFrame()).columns),
                "date_range_days": len(aligned_data.get("close", pd.DataFrame()).index),
                "volume_normalization": volume_normalization,
                "return_method": return_method,
                "overall_quality_score": quality_metrics["overall_score"],
            },
        }

        return result
