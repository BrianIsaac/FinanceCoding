"""Gap-filling algorithms for financial time series data.

This module provides comprehensive gap-filling strategies for price and volume data,
including forward/backward fill, interpolation methods, and volume validation.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import interpolate

from src.config.data import ValidationConfig


class GapFiller:
    """
    Advanced gap-filling processor for financial time series data.

    Provides multiple interpolation methods with volume validation and
    configurable quality thresholds for robust data completion.
    """

    def __init__(self, config: ValidationConfig):
        """
        Initialize gap filler with validation configuration.

        Args:
            config: Validation configuration with thresholds and methods
        """
        self.config = config

    def forward_fill(
        self,
        series: pd.Series,
        limit: int | None = None,
        volume_series: pd.Series | None = None,
        min_volume: int = 1000,
    ) -> pd.Series:
        """Forward fill with optional volume validation.

        Args:
            series: Price series to fill
            limit: Maximum number of consecutive fills
            volume_series: Optional volume series for validation
            min_volume: Minimum volume threshold for valid fills

        Returns:
            Forward-filled series
        """
        filled = series.fillna(method="ffill", limit=limit)

        if volume_series is not None:
            # Only keep fills where volume meets minimum threshold
            volume_valid = volume_series >= min_volume
            # Revert fills where volume is too low
            mask = series.isna() & filled.notna() & ~volume_valid
            filled[mask] = np.nan

        return filled

    def backward_fill(
        self,
        series: pd.Series,
        limit: int | None = None,
        volume_series: pd.Series | None = None,
        min_volume: int = 1000,
    ) -> pd.Series:
        """Backward fill with optional volume validation.

        Args:
            series: Price series to fill
            limit: Maximum number of consecutive fills
            volume_series: Optional volume series for validation
            min_volume: Minimum volume threshold for valid fills

        Returns:
            Backward-filled series
        """
        filled = series.fillna(method="bfill", limit=limit)

        if volume_series is not None:
            volume_valid = volume_series >= min_volume
            mask = series.isna() & filled.notna() & ~volume_valid
            filled[mask] = np.nan

        return filled

    def linear_interpolate(
        self,
        series: pd.Series,
        max_gap_days: int = 5,
        volume_series: pd.Series | None = None,
        min_volume: int = 1000,
    ) -> pd.Series:
        """Linear interpolation with gap size and volume constraints.

        Args:
            series: Price series to interpolate
            max_gap_days: Maximum gap size in days for interpolation
            volume_series: Optional volume series for validation
            min_volume: Minimum volume threshold

        Returns:
            Interpolated series
        """
        filled = series.copy()

        # Identify gaps and their sizes
        is_na = series.isna()
        gap_starts = is_na & ~is_na.shift(1).fillna(False)
        gap_ends = ~is_na & is_na.shift(1).fillna(False)

        # Find gap ranges
        gap_ranges = []
        start_idx = None

        for idx in series.index:
            if gap_starts.loc[idx]:
                start_idx = idx
            elif gap_ends.loc[idx] and start_idx is not None:
                gap_days = (idx - start_idx).days
                if gap_days <= max_gap_days:
                    gap_ranges.append((start_idx, idx, gap_days))
                start_idx = None

        # Interpolate small gaps only
        for start_date, end_date, gap_days in gap_ranges:
            gap_mask = (series.index >= start_date) & (series.index < end_date)

            # Get boundary values
            before_val = (
                series.loc[series.index < start_date].iloc[-1]
                if len(series.loc[series.index < start_date]) > 0
                else None
            )
            after_val = (
                series.loc[series.index >= end_date].iloc[0]
                if len(series.loc[series.index >= end_date]) > 0
                else None
            )

            if before_val is not None and after_val is not None:
                # Linear interpolation
                gap_indices = series.index[gap_mask]
                start_timestamp = series.index[series.index < start_date][-1]
                end_timestamp = series.index[series.index >= end_date][0]

                total_days = (end_timestamp - start_timestamp).days
                for gap_idx in gap_indices:
                    days_from_start = (gap_idx - start_timestamp).days
                    progress = days_from_start / total_days
                    interpolated_value = before_val + progress * (after_val - before_val)

                    # Volume validation
                    if volume_series is not None:
                        if volume_series.loc[gap_idx] >= min_volume:
                            filled.loc[gap_idx] = interpolated_value
                    else:
                        filled.loc[gap_idx] = interpolated_value

        return filled

    def spline_interpolate(
        self,
        series: pd.Series,
        max_gap_days: int = 3,
        spline_order: int = 2,
        volume_series: pd.Series | None = None,
        min_volume: int = 1000,
    ) -> pd.Series:
        """Spline interpolation for smooth gap filling.

        Args:
            series: Price series to interpolate
            max_gap_days: Maximum gap size for interpolation
            spline_order: Order of spline (1=linear, 2=quadratic, 3=cubic)
            volume_series: Optional volume series for validation
            min_volume: Minimum volume threshold

        Returns:
            Spline-interpolated series
        """
        filled = series.copy()

        if series.isna().all():
            return filled

        # Get non-NaN data for interpolation
        valid_data = series.dropna()
        if len(valid_data) < spline_order + 1:
            # Fall back to linear interpolation
            return self.linear_interpolate(series, max_gap_days, volume_series, min_volume)

        # Convert dates to numeric for spline
        numeric_index = np.array([(idx - series.index[0]).days for idx in valid_data.index])

        try:
            # Create spline function
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spline = interpolate.UnivariateSpline(
                    numeric_index,
                    valid_data.values,
                    k=min(spline_order, len(valid_data) - 1),
                    s=0,  # Interpolation (not smoothing)
                )

            # Fill gaps
            is_na = series.isna()
            gap_starts = is_na & ~is_na.shift(1).fillna(False)
            gap_ends = ~is_na & is_na.shift(1).fillna(False)

            start_idx = None
            for idx in series.index:
                if gap_starts.loc[idx]:
                    start_idx = idx
                elif gap_ends.loc[idx] and start_idx is not None:
                    gap_days = (idx - start_idx).days
                    if gap_days <= max_gap_days:
                        # Interpolate this gap
                        gap_mask = (series.index >= start_idx) & (series.index < idx)
                        gap_indices = series.index[gap_mask]

                        for gap_idx in gap_indices:
                            numeric_pos = (gap_idx - series.index[0]).days
                            interpolated_value = float(spline(numeric_pos))

                            # Volume validation
                            if volume_series is not None:
                                if volume_series.loc[gap_idx] >= min_volume:
                                    filled.loc[gap_idx] = interpolated_value
                            else:
                                filled.loc[gap_idx] = interpolated_value
                    start_idx = None

        except Exception:
            # Fall back to linear interpolation on error
            return self.linear_interpolate(series, max_gap_days, volume_series, min_volume)

        return filled

    def hybrid_fill(
        self,
        series: pd.Series,
        volume_series: pd.Series | None = None,
        small_gap_days: int = 3,
        medium_gap_days: int = 7,
        max_gap_days: int = 14,
    ) -> pd.Series:
        """Hybrid gap-filling strategy using multiple methods based on gap size.

        Args:
            series: Price series to fill
            volume_series: Optional volume series for validation
            small_gap_days: Threshold for small gaps (use spline)
            medium_gap_days: Threshold for medium gaps (use linear)
            max_gap_days: Maximum gap size to fill

        Returns:
            Filled series using hybrid approach
        """
        filled = series.copy()
        volume_threshold = self.config.volume_threshold

        # Identify gaps and categorize by size
        is_na = series.isna()
        gap_starts = is_na & ~is_na.shift(1).fillna(False)
        gap_ends = ~is_na & is_na.shift(1).fillna(False)

        gap_info = []
        start_idx = None

        for idx in series.index:
            if gap_starts.loc[idx]:
                start_idx = idx
            elif gap_ends.loc[idx] and start_idx is not None:
                gap_days = (idx - start_idx).days
                gap_info.append(
                    {
                        "start": start_idx,
                        "end": idx,
                        "days": gap_days,
                        "method": self._choose_fill_method(
                            gap_days, small_gap_days, medium_gap_days, max_gap_days
                        ),
                    }
                )
                start_idx = None

        # Apply different methods based on gap size
        for gap in gap_info:
            if gap["method"] == "skip":
                continue

            gap_mask = (series.index >= gap["start"]) & (series.index < gap["end"])

            if gap["method"] == "spline":
                filled_gap = self.spline_interpolate(
                    filled,
                    gap["days"],
                    volume_series=volume_series,
                    min_volume=volume_threshold,
                )
            elif gap["method"] == "linear":
                filled_gap = self.linear_interpolate(
                    filled,
                    gap["days"],
                    volume_series=volume_series,
                    min_volume=volume_threshold,
                )
            elif gap["method"] == "forward":
                filled_gap = self.forward_fill(
                    filled,
                    limit=gap["days"],
                    volume_series=volume_series,
                    min_volume=volume_threshold,
                )
            else:  # backward
                filled_gap = self.backward_fill(
                    filled,
                    limit=gap["days"],
                    volume_series=volume_series,
                    min_volume=volume_threshold,
                )

            # Update only the gap portion
            filled.loc[gap_mask] = filled_gap.loc[gap_mask]

        return filled

    def _choose_fill_method(
        self, gap_days: int, small_gap_days: int, medium_gap_days: int, max_gap_days: int
    ) -> str:
        """Choose appropriate fill method based on gap size.

        Args:
            gap_days: Size of gap in days
            small_gap_days: Small gap threshold
            medium_gap_days: Medium gap threshold
            max_gap_days: Maximum fillable gap

        Returns:
            Fill method name
        """
        if gap_days > max_gap_days:
            return "skip"
        elif gap_days <= small_gap_days:
            return "spline"
        elif gap_days <= medium_gap_days:
            return "linear"
        else:
            return self.config.fill_method  # Use configured method

    def process_dataframe(
        self,
        prices_df: pd.DataFrame,
        volume_df: pd.DataFrame | None = None,
        method: str = "hybrid",
    ) -> pd.DataFrame:
        """Process entire DataFrame with gap filling.

        Args:
            prices_df: Price data DataFrame (Date × Tickers)
            volume_df: Optional volume DataFrame for validation
            method: Fill method ('hybrid', 'forward', 'backward', 'linear', 'spline')

        Returns:
            DataFrame with gaps filled
        """
        filled_df = prices_df.copy()

        for ticker in prices_df.columns:
            price_series = prices_df[ticker]
            volume_series = volume_df[ticker] if volume_df is not None else None

            if method == "hybrid":
                filled_series = self.hybrid_fill(price_series, volume_series)
            elif method == "forward":
                filled_series = self.forward_fill(
                    price_series,
                    volume_series=volume_series,
                    min_volume=self.config.volume_threshold,
                )
            elif method == "backward":
                filled_series = self.backward_fill(
                    price_series,
                    volume_series=volume_series,
                    min_volume=self.config.volume_threshold,
                )
            elif method == "linear":
                filled_series = self.linear_interpolate(
                    price_series,
                    volume_series=volume_series,
                    min_volume=self.config.volume_threshold,
                )
            elif method == "spline":
                filled_series = self.spline_interpolate(
                    price_series,
                    volume_series=volume_series,
                    min_volume=self.config.volume_threshold,
                )
            else:
                raise ValueError(f"Unknown fill method: {method}")

            filled_df[ticker] = filled_series

        return filled_df

    def validate_fill_quality(
        self,
        original_df: pd.DataFrame,
        filled_df: pd.DataFrame,
        volume_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Validate quality of gap filling results.

        Args:
            original_df: Original data with gaps
            filled_df: Data after gap filling
            volume_df: Optional volume data

        Returns:
            Dictionary with quality metrics
        """
        validation_results = {
            "total_cells": original_df.size,
            "original_missing": original_df.isna().sum().sum(),
            "filled_missing": filled_df.isna().sum().sum(),
            "cells_filled": 0,
            "fill_rate": 0.0,
            "ticker_quality": {},
            "quality_flags": [],
        }

        cells_filled = validation_results["original_missing"] - validation_results["filled_missing"]
        validation_results["cells_filled"] = int(cells_filled)
        validation_results["fill_rate"] = float(
            cells_filled / validation_results["original_missing"]
            if validation_results["original_missing"] > 0
            else 0.0
        )

        # Per-ticker quality assessment
        for ticker in original_df.columns:
            orig_missing = original_df[ticker].isna().sum()
            filled_missing = filled_df[ticker].isna().sum()
            ticker_filled = orig_missing - filled_missing

            ticker_quality = {
                "original_missing": int(orig_missing),
                "filled_missing": int(filled_missing),
                "cells_filled": int(ticker_filled),
                "fill_rate": float(ticker_filled / orig_missing if orig_missing > 0 else 0.0),
                "data_coverage": float(filled_df[ticker].notna().sum() / len(filled_df)),
            }

            # Quality flags
            if ticker_quality["data_coverage"] < self.config.missing_data_threshold:
                validation_results["quality_flags"].append(
                    f"Low coverage for {ticker}: {ticker_quality['data_coverage']:.2%}"
                )

            validation_results["ticker_quality"][ticker] = ticker_quality

        return validation_results
