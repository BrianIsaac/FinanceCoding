"""
Missing data detection and handling for rolling backtest execution.

This module provides sophisticated missing data handling capabilities
for portfolio models during retraining, including gap detection,
intelligent filling strategies, and data quality validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MissingDataStrategy(Enum):
    """Strategies for handling missing data."""

    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    LINEAR_INTERPOLATE = "linear_interpolate"
    SPLINE_INTERPOLATE = "spline_interpolate"
    CROSS_SECTIONAL_MEDIAN = "cross_sectional_median"
    DROP_ASSETS = "drop_assets"
    DROP_PERIODS = "drop_periods"
    ZERO_FILL = "zero_fill"


@dataclass
class DataQualityMetrics:
    """Metrics describing data quality."""

    total_observations: int = 0
    missing_observations: int = 0
    missing_ratio: float = 0.0
    consecutive_missing_max: int = 0
    assets_with_missing: int = 0
    complete_assets: int = 0
    complete_periods: int = 0
    data_gaps: list[tuple[pd.Timestamp, pd.Timestamp]] = field(default_factory=list)


@dataclass
class MissingDataConfig:
    """Configuration for missing data handling."""

    # Detection parameters
    max_missing_ratio: float = 0.1  # 10% max missing data
    max_consecutive_missing: int = 5  # Max 5 consecutive missing values
    min_data_coverage: float = 0.8  # Min 80% data coverage per asset

    # Handling strategies
    primary_strategy: MissingDataStrategy = MissingDataStrategy.FORWARD_FILL
    fallback_strategy: MissingDataStrategy = MissingDataStrategy.DROP_ASSETS
    forward_fill_limit: int = 3
    backward_fill_limit: int = 2

    # Quality control
    validate_after_filling: bool = True
    require_monotonic_dates: bool = True
    handle_extreme_values: bool = True
    extreme_value_threshold: float = 5.0  # Z-score threshold


class MissingDataHandler:
    """
    Comprehensive missing data detection and handling system.

    This handler provides intelligent strategies for dealing with missing
    data in financial time series during model retraining, including:
    - Gap detection and analysis
    - Multiple filling strategies with fallbacks
    - Data quality validation and metrics
    - Asset filtering based on data completeness
    """

    def __init__(self, config: MissingDataConfig):
        """Initialize missing data handler."""
        self.config = config
        self._quality_history: list[DataQualityMetrics] = []

    def analyze_data_quality(self, data: pd.DataFrame) -> DataQualityMetrics:
        """
        Comprehensive analysis of data quality and missing patterns.

        Args:
            data: Input data with potential missing values

        Returns:
            Detailed data quality metrics
        """
        metrics = DataQualityMetrics()

        # Basic statistics
        metrics.total_observations = data.size
        metrics.missing_observations = data.isna().sum().sum()
        metrics.missing_ratio = (
            metrics.missing_observations / metrics.total_observations
            if metrics.total_observations > 0
            else 0.0
        )

        # Per-asset analysis
        asset_missing = data.isna().sum()
        metrics.assets_with_missing = (asset_missing > 0).sum()
        metrics.complete_assets = (asset_missing == 0).sum()

        # Per-period analysis
        period_missing = data.isna().sum(axis=1)
        metrics.complete_periods = (period_missing == 0).sum()

        # Consecutive missing analysis
        max_consecutive = 0
        for col in data.columns:
            series = data[col]
            current_consecutive = 0

            for value in series:
                if pd.isna(value):
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0

        metrics.consecutive_missing_max = max_consecutive

        # Data gaps detection
        if isinstance(data.index, pd.DatetimeIndex):
            metrics.data_gaps = self._detect_temporal_gaps(data.index)

        logger.debug(
            f"Data quality: {metrics.missing_ratio:.3f} missing ratio, "
            f"{metrics.complete_assets} complete assets"
        )

        return metrics

    def handle_missing_data(
        self,
        data: pd.DataFrame,
        universe: list[str] | None = None,
    ) -> tuple[pd.DataFrame, DataQualityMetrics]:
        """
        Handle missing data using configured strategies.

        Args:
            data: Input data with missing values
            universe: Optional asset universe to filter

        Returns:
            Tuple of (cleaned_data, quality_metrics)
        """
        logger.info("Starting missing data handling")

        # Initial quality analysis
        initial_metrics = self.analyze_data_quality(data)
        self._quality_history.append(initial_metrics)

        # Filter to universe if specified
        working_data = data.copy()
        if universe:
            available_assets = [asset for asset in universe if asset in data.columns]
            working_data = working_data[available_assets]
            logger.debug(f"Filtered to universe: {len(available_assets)} assets")

        # Validate data before processing
        if self.config.require_monotonic_dates and isinstance(working_data.index, pd.DatetimeIndex):
            if not working_data.index.is_monotonic_increasing:
                working_data = working_data.sort_index()
                logger.warning("Data index was not monotonic - sorted by date")

        # Apply primary strategy
        try:
            cleaned_data = self._apply_missing_strategy(working_data, self.config.primary_strategy)
        except Exception as e:
            logger.warning(f"Primary strategy {self.config.primary_strategy} failed: {e}")
            cleaned_data = self._apply_missing_strategy(working_data, self.config.fallback_strategy)

        # Handle extreme values if configured
        if self.config.handle_extreme_values:
            cleaned_data = self._handle_extreme_values(cleaned_data)

        # Final validation
        if self.config.validate_after_filling:
            cleaned_data = self._validate_cleaned_data(cleaned_data)

        # Final quality analysis
        final_metrics = self.analyze_data_quality(cleaned_data)

        logger.info(
            f"Missing data handling complete: "
            f"{initial_metrics.missing_ratio:.3f} -> "
            f"{final_metrics.missing_ratio:.3f} missing ratio"
        )

        return cleaned_data, final_metrics

    def _apply_missing_strategy(
        self,
        data: pd.DataFrame,
        strategy: MissingDataStrategy,
    ) -> pd.DataFrame:
        """Apply specific missing data strategy."""

        if strategy == MissingDataStrategy.FORWARD_FILL:
            return data.ffill(limit=self.config.forward_fill_limit)

        elif strategy == MissingDataStrategy.BACKWARD_FILL:
            return data.bfill(limit=self.config.backward_fill_limit)

        elif strategy == MissingDataStrategy.LINEAR_INTERPOLATE:
            return data.interpolate(method="linear", limit=self.config.forward_fill_limit)

        elif strategy == MissingDataStrategy.SPLINE_INTERPOLATE:
            return data.interpolate(method="spline", order=2, limit=self.config.forward_fill_limit)

        elif strategy == MissingDataStrategy.CROSS_SECTIONAL_MEDIAN:
            # Fill with cross-sectional median for each time period
            filled_data = data.copy()
            for idx in data.index:
                row_median = data.loc[idx].median()
                filled_data.loc[idx] = data.loc[idx].fillna(row_median)
            return filled_data

        elif strategy == MissingDataStrategy.DROP_ASSETS:
            # Drop assets with too much missing data
            asset_completeness = data.notna().mean()
            keep_assets = asset_completeness[
                asset_completeness >= self.config.min_data_coverage
            ].index
            return data[keep_assets].dropna()

        elif strategy == MissingDataStrategy.DROP_PERIODS:
            # Drop time periods with too much missing data
            return data.dropna(axis=0, how="any")

        elif strategy == MissingDataStrategy.ZERO_FILL:
            return data.fillna(0.0)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _handle_extreme_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle extreme values in the data."""

        cleaned_data = data.copy()

        for col in data.columns:
            series = data[col]
            if series.dtype in ["float64", "float32", "int64", "int32"]:
                # Calculate z-score
                mean_val = series.mean()
                std_val = series.std()

                if std_val > 0:
                    z_scores = abs((series - mean_val) / std_val)
                    extreme_mask = z_scores > self.config.extreme_value_threshold

                    if extreme_mask.any():
                        # Replace extreme values with median
                        median_val = series.median()
                        cleaned_data.loc[extreme_mask, col] = median_val
                        logger.debug(f"Replaced {extreme_mask.sum()} extreme values in {col}")

        return cleaned_data

    def _validate_cleaned_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate cleaned data and apply final filters."""

        # Remove assets with remaining missing data above threshold
        asset_missing_ratios = data.isna().mean()
        valid_assets = asset_missing_ratios[
            asset_missing_ratios <= self.config.max_missing_ratio
        ].index

        if len(valid_assets) < len(data.columns):
            logger.warning(
                f"Removing {len(data.columns) - len(valid_assets)} assets "
                f"due to excessive missing data"
            )

        validated_data = data[valid_assets]

        # Check for consecutive missing values
        for col in validated_data.columns:
            series = validated_data[col]
            consecutive_count = 0
            max_consecutive = 0

            for value in series:
                if pd.isna(value):
                    consecutive_count += 1
                    max_consecutive = max(max_consecutive, consecutive_count)
                else:
                    consecutive_count = 0

            if max_consecutive > self.config.max_consecutive_missing:
                logger.warning(f"Asset {col} has {max_consecutive} consecutive missing values")

        return validated_data

    def _detect_temporal_gaps(
        self, date_index: pd.DatetimeIndex
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Detect gaps in temporal data."""

        if len(date_index) < 2:
            return []

        gaps = []
        expected_freq = pd.infer_freq(date_index)

        if expected_freq:
            # Generate expected date range
            expected_dates = pd.date_range(
                start=date_index[0], end=date_index[-1], freq=expected_freq
            )

            # Find missing dates
            missing_dates = expected_dates.difference(date_index)

            if len(missing_dates) > 0:
                # Group consecutive missing dates into gaps
                missing_dates = missing_dates.sort_values()
                gap_start = missing_dates[0]
                gap_end = missing_dates[0]

                for i in range(1, len(missing_dates)):
                    if (missing_dates[i] - gap_end).days <= 1:
                        gap_end = missing_dates[i]
                    else:
                        gaps.append((gap_start, gap_end))
                        gap_start = missing_dates[i]
                        gap_end = missing_dates[i]

                gaps.append((gap_start, gap_end))

        return gaps

    def get_data_quality_summary(self) -> dict[str, Any]:
        """Get summary of data quality across all processed datasets."""

        if not self._quality_history:
            return {"message": "No data quality history available"}

        recent_metrics = self._quality_history[-10:]  # Last 10 datasets

        return {
            "datasets_processed": len(self._quality_history),
            "avg_missing_ratio": np.mean([m.missing_ratio for m in recent_metrics]),
            "avg_complete_assets": np.mean([m.complete_assets for m in recent_metrics]),
            "max_consecutive_missing": max([m.consecutive_missing_max for m in recent_metrics]),
            "total_gaps_detected": sum([len(m.data_gaps) for m in recent_metrics]),
            "data_quality_trend": (
                "improving"
                if len(recent_metrics) > 1
                and recent_metrics[-1].missing_ratio < recent_metrics[0].missing_ratio
                else "stable"
            ),
        }

    def recommend_strategy(self, data: pd.DataFrame) -> MissingDataStrategy:
        """Recommend optimal missing data strategy based on data characteristics."""

        metrics = self.analyze_data_quality(data)

        # High missing ratio - drop problematic assets
        if metrics.missing_ratio > 0.2:
            return MissingDataStrategy.DROP_ASSETS

        # Long consecutive missing sequences - interpolation might not work well
        if metrics.consecutive_missing_max > 10:
            return MissingDataStrategy.DROP_PERIODS

        # Moderate missing with short gaps - interpolation works well
        if metrics.missing_ratio < 0.05 and metrics.consecutive_missing_max <= 3:
            return MissingDataStrategy.LINEAR_INTERPOLATE

        # Default to forward fill for financial data
        return MissingDataStrategy.FORWARD_FILL

    def create_data_quality_report(
        self,
        data: pd.DataFrame,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """Create comprehensive data quality report."""

        metrics = self.analyze_data_quality(data)

        # Per-asset analysis
        asset_analysis = {}
        for col in data.columns:
            series = data[col]
            asset_analysis[col] = {
                "missing_count": series.isna().sum(),
                "missing_ratio": series.isna().mean(),
                "first_valid": series.first_valid_index(),
                "last_valid": series.last_valid_index(),
                "data_range_days": (
                    (series.last_valid_index() - series.first_valid_index()).days
                    if series.first_valid_index() and series.last_valid_index()
                    else 0
                ),
            }

        # Time period analysis
        period_analysis = {}
        for idx in data.index:
            row = data.loc[idx]
            period_analysis[idx.isoformat()] = {
                "missing_count": row.isna().sum(),
                "missing_ratio": row.isna().mean(),
                "available_assets": row.notna().sum(),
            }

        report = {
            "summary": {
                "total_observations": metrics.total_observations,
                "missing_observations": metrics.missing_observations,
                "missing_ratio": metrics.missing_ratio,
                "complete_assets": metrics.complete_assets,
                "assets_with_missing": metrics.assets_with_missing,
                "max_consecutive_missing": metrics.consecutive_missing_max,
                "temporal_gaps": len(metrics.data_gaps),
            },
            "asset_analysis": asset_analysis,
            "period_analysis": period_analysis,
            "data_gaps": [
                {"start": gap[0].isoformat(), "end": gap[1].isoformat()}
                for gap in metrics.data_gaps
            ],
            "recommendations": {
                "recommended_strategy": self.recommend_strategy(data).value,
                "quality_acceptable": metrics.missing_ratio <= self.config.max_missing_ratio,
                "require_asset_filtering": metrics.assets_with_missing > len(data.columns) * 0.5,
            },
        }

        if output_path:
            import json

            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Data quality report saved to {output_path}")

        return report
