"""
Enhanced rolling window generator for portfolio backtesting.

This module provides a comprehensive rolling window generation system with
strict temporal validation, data integrity monitoring, and walk-forward
analysis capabilities specifically designed for portfolio backtesting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.validation.rolling_validation import (
    RollSplit,
    ValidationPeriod,
)

logger = logging.getLogger(__name__)


@dataclass
class WindowGenerationConfig:
    """Configuration for advanced rolling window generation."""

    # Basic temporal parameters
    training_months: int = 36
    validation_months: int = 12
    test_months: int = 12
    step_months: int = 12

    # Data quality requirements
    min_training_samples: int = 252
    min_validation_samples: int = 20
    min_test_samples: int = 20
    max_gap_days: int = 5

    # Window generation options
    require_full_periods: bool = True
    allow_partial_windows: bool = False
    min_assets_per_window: int = 10

    # Advanced features
    enable_adaptive_stepping: bool = False
    adaptive_step_factor: float = 0.8
    enable_overlap_detection: bool = True


class RollingWindowGenerator:
    """
    Advanced rolling window generator with comprehensive validation.

    This generator creates rolling train/validation/test splits with:
    - Strict 36/12/12 month protocol
    - Temporal integrity validation
    - Data quality assessment
    - Adaptive stepping options
    - Memory-efficient processing
    """

    def __init__(self, config: WindowGenerationConfig):
        """Initialize rolling window generator."""
        self.config = config
        self._generation_stats: dict[str, Any] = {}

    def generate_rolling_windows(
        self,
        data_timestamps: list[pd.Timestamp],
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> list[RollSplit]:
        """
        Generate rolling windows with comprehensive validation.

        Args:
            data_timestamps: Available data timestamps
            start_date: Optional start date override
            end_date: Optional end date override

        Returns:
            List of validated rolling splits
        """

        logger.info("Starting enhanced rolling window generation")

        # Validate and prepare timestamps
        validated_timestamps = self._validate_timestamps(data_timestamps)

        if not validated_timestamps:
            raise ValueError("No valid timestamps after validation")

        # Determine effective date range
        effective_start = start_date or validated_timestamps[0]
        effective_end = end_date or validated_timestamps[-1]

        # Generate candidate windows
        candidate_windows = self._generate_candidate_windows(
            validated_timestamps, effective_start, effective_end
        )

        # Validate and filter windows
        valid_windows = self._validate_and_filter_windows(candidate_windows, validated_timestamps)

        # Apply post-processing
        final_windows = self._post_process_windows(valid_windows, validated_timestamps)

        # Generate statistics
        self._generation_stats = self._compute_generation_stats(final_windows, validated_timestamps)

        logger.info(
            f"Generated {len(final_windows)} valid windows from "
            f"{len(candidate_windows)} candidates"
        )

        return final_windows

    def _validate_timestamps(self, timestamps: list[pd.Timestamp]) -> list[pd.Timestamp]:
        """Validate and clean input timestamps."""

        # Remove duplicates and sort
        unique_timestamps = sorted(set(timestamps))

        # Check for minimum data requirements
        if len(unique_timestamps) < self.config.min_training_samples:
            raise ValueError(
                f"Insufficient timestamps: {len(unique_timestamps)} < "
                f"{self.config.min_training_samples}"
            )

        # Detect large gaps
        if len(unique_timestamps) > 1:
            gaps = [
                (unique_timestamps[i + 1] - unique_timestamps[i]).days
                for i in range(len(unique_timestamps) - 1)
            ]
            max_gap = max(gaps)

            if max_gap > self.config.max_gap_days * 5:  # Allow larger gaps in timestamp list
                logger.warning(f"Large data gap detected: {max_gap} days")

        return unique_timestamps

    def _generate_candidate_windows(
        self,
        timestamps: list[pd.Timestamp],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> list[RollSplit]:
        """Generate candidate rolling windows."""

        candidate_windows = []
        current_start = self._align_to_month_start(start_date)

        step_size = self.config.step_months

        while True:
            # Calculate window boundaries - Fix: Training should be BACKWARD looking
            # current_start is the prediction date, so training uses past data
            train_start = self._subtract_months(current_start, self.config.training_months)
            train_end = current_start
            val_start = train_end  # Validation starts after training (at prediction date)
            val_end = self._add_months(val_start, self.config.validation_months)
            test_start = val_end   # Test starts after validation
            test_end = self._add_months(test_start, self.config.test_months)

            # Check if window exceeds available data
            if test_end > end_date:
                break

            # Create validation periods
            try:
                train_period = ValidationPeriod(current_start, train_end)
                val_period = ValidationPeriod(val_start, val_end)
                test_period = ValidationPeriod(test_start, test_end)

                # Create split
                split = RollSplit(train_period, val_period, test_period)
                candidate_windows.append(split)

            except ValueError as e:
                logger.warning(f"Invalid window at {current_start}: {e}")

            # Adaptive stepping
            if self.config.enable_adaptive_stepping:
                step_size = self._calculate_adaptive_step(candidate_windows, timestamps)

            # Advance to next window
            current_start = self._add_months(current_start, step_size)

            # Prevent infinite loops
            if current_start >= end_date:
                break

        return candidate_windows

    def _validate_and_filter_windows(
        self,
        candidate_windows: list[RollSplit],
        timestamps: list[pd.Timestamp],
    ) -> list[RollSplit]:
        """Validate and filter candidate windows."""

        valid_windows = []
        timestamp_set = set(timestamps)

        for window in candidate_windows:
            # Check data availability
            if not self._check_data_availability(window, timestamp_set):
                logger.debug(
                    f"Insufficient data for window starting {window.train_period.start_date}"
                )
                continue

            # Check temporal integrity
            if not self._check_temporal_integrity(window):
                logger.warning(
                    f"Temporal integrity violation for window starting {window.train_period.start_date}"
                )
                continue

            # Check data quality
            if not self._check_data_quality(window, timestamps):
                logger.debug(
                    f"Data quality issues for window starting {window.train_period.start_date}"
                )
                continue

            valid_windows.append(window)

        return valid_windows

    def _check_data_availability(self, window: RollSplit, timestamp_set: set[pd.Timestamp]) -> bool:
        """Check if window has sufficient data availability."""

        # Count samples in each period
        train_samples = sum(1 for ts in timestamp_set if window.train_period.contains_date(ts))

        val_samples = sum(1 for ts in timestamp_set if window.validation_period.contains_date(ts))

        test_samples = sum(1 for ts in timestamp_set if window.test_period.contains_date(ts))

        # Check minimums
        sufficient_train = train_samples >= self.config.min_training_samples
        sufficient_val = val_samples >= self.config.min_validation_samples
        sufficient_test = test_samples >= self.config.min_test_samples

        return sufficient_train and sufficient_val and sufficient_test

    def _check_temporal_integrity(self, window: RollSplit) -> bool:
        """Check temporal integrity of window."""

        # Ensure no period overlaps
        if window.train_period.end_date > window.validation_period.start_date:
            return False

        if window.validation_period.end_date > window.test_period.start_date:
            return False

        # Ensure proper ordering
        if (
            window.train_period.start_date >= window.validation_period.start_date
            or window.validation_period.start_date >= window.test_period.start_date
        ):
            return False

        return True

    def _check_data_quality(self, window: RollSplit, timestamps: list[pd.Timestamp]) -> bool:
        """Check data quality within window periods."""

        # Get timestamps for each period
        train_timestamps = [ts for ts in timestamps if window.train_period.contains_date(ts)]

        val_timestamps = [ts for ts in timestamps if window.validation_period.contains_date(ts)]

        test_timestamps = [ts for ts in timestamps if window.test_period.contains_date(ts)]

        # Check for excessive gaps in each period
        for period_timestamps in [train_timestamps, val_timestamps, test_timestamps]:
            if len(period_timestamps) > 1:
                gaps = [
                    (period_timestamps[i + 1] - period_timestamps[i]).days
                    for i in range(len(period_timestamps) - 1)
                ]
                max_gap = max(gaps) if gaps else 0

                if max_gap > self.config.max_gap_days:
                    return False

        return True

    def _post_process_windows(
        self,
        windows: list[RollSplit],
        timestamps: list[pd.Timestamp],
    ) -> list[RollSplit]:
        """Apply post-processing to validated windows."""

        if not self.config.enable_overlap_detection:
            return windows

        # Detect and resolve overlaps if necessary
        non_overlapping = []

        for window in windows:
            # Check for significant overlap with previous windows
            has_overlap = False

            for existing_window in non_overlapping:
                if self._detect_window_overlap(window, existing_window):
                    has_overlap = True
                    break

            if not has_overlap:
                non_overlapping.append(window)

        return non_overlapping

    def _detect_window_overlap(self, window1: RollSplit, window2: RollSplit) -> bool:
        """Detect if two windows have problematic overlap."""

        # Check training period overlap
        train1_start, train1_end = window1.train_period.start_date, window1.train_period.end_date
        train2_start, train2_end = window2.train_period.start_date, window2.train_period.end_date

        # Calculate overlap
        overlap_start = max(train1_start, train2_start)
        overlap_end = min(train1_end, train2_end)

        if overlap_start < overlap_end:
            overlap_days = (overlap_end - overlap_start).days
            train1_days = (train1_end - train1_start).days

            # Consider overlap problematic if > 90% of training period
            overlap_ratio = overlap_days / train1_days
            return overlap_ratio > 0.9

        return False

    def _calculate_adaptive_step(
        self,
        existing_windows: list[RollSplit],
        timestamps: list[pd.Timestamp],
    ) -> int:
        """Calculate adaptive step size based on data availability."""

        # Simple adaptive logic - can be enhanced
        base_step = self.config.step_months

        if len(existing_windows) > 0:
            last_window = existing_windows[-1]

            # Check data density in last window
            train_timestamps = [
                ts for ts in timestamps if last_window.train_period.contains_date(ts)
            ]

            expected_days = last_window.train_period.duration_days
            actual_days = len(train_timestamps)
            density_ratio = actual_days / expected_days

            # Adjust step size based on data density
            if density_ratio < 0.7:  # Sparse data
                adjusted_step = int(base_step * self.config.adaptive_step_factor)
                return max(1, adjusted_step)  # Minimum 1 month

        return base_step

    def _compute_generation_stats(
        self,
        windows: list[RollSplit],
        timestamps: list[pd.Timestamp],
    ) -> dict[str, Any]:
        """Compute statistics about window generation."""

        if not windows:
            return {"error": "No windows generated"}

        # Basic statistics
        total_period = windows[-1].test_period.end_date - windows[0].train_period.start_date

        # Data utilization
        total_timestamps = len(timestamps)
        utilized_timestamps = set()

        for window in windows:
            window_timestamps = [
                ts
                for ts in timestamps
                if (
                    window.train_period.contains_date(ts)
                    or window.validation_period.contains_date(ts)
                    or window.test_period.contains_date(ts)
                )
            ]
            utilized_timestamps.update(window_timestamps)

        utilization_rate = len(utilized_timestamps) / total_timestamps

        # Period statistics
        avg_training_days = np.mean([w.train_period.duration_days for w in windows])
        avg_step_days = (
            np.mean(
                [
                    (
                        windows[i + 1].train_period.start_date - windows[i].train_period.start_date
                    ).days
                    for i in range(len(windows) - 1)
                ]
            )
            if len(windows) > 1
            else 0
        )

        return {
            "total_windows": len(windows),
            "total_period_days": total_period.days,
            "data_utilization_rate": round(utilization_rate, 3),
            "avg_training_period_days": round(avg_training_days, 1),
            "avg_step_size_days": round(avg_step_days, 1),
            "first_window_start": windows[0].train_period.start_date.isoformat(),
            "last_window_end": windows[-1].test_period.end_date.isoformat(),
        }

    def get_generation_stats(self) -> dict[str, Any]:
        """Get statistics from last window generation."""
        return self._generation_stats.copy()

    @staticmethod
    def _add_months(timestamp: pd.Timestamp, months: int) -> pd.Timestamp:
        """Add months to timestamp with proper handling."""
        return (timestamp + pd.DateOffset(months=months)).normalize()

    @staticmethod
    def _subtract_months(timestamp: pd.Timestamp, months: int) -> pd.Timestamp:
        """Subtract months from timestamp with proper handling."""
        return (timestamp - pd.DateOffset(months=months)).normalize()

    @staticmethod
    def _align_to_month_start(timestamp: pd.Timestamp) -> pd.Timestamp:
        """Align timestamp to start of month."""
        return timestamp.replace(day=1).normalize()


class WalkForwardAnalyzer:
    """
    Analyzes walk-forward progression and temporal integrity.

    This class provides comprehensive analysis of walk-forward testing
    including progression validation, temporal coverage analysis,
    and bias detection.
    """

    def __init__(self):
        """Initialize walk-forward analyzer."""
        self._analysis_cache: dict[str, Any] = {}

    def analyze_walk_forward_progression(
        self,
        windows: list[RollSplit],
        timestamps: list[pd.Timestamp],
        expected_step_months: int = 12,
    ) -> dict[str, Any]:
        """
        Analyze walk-forward progression for integrity and consistency.

        Args:
            windows: List of rolling windows
            timestamps: Available data timestamps
            expected_step_months: Expected step size in months

        Returns:
            Comprehensive analysis results
        """

        if len(windows) < 2:
            return {"error": "Insufficient windows for progression analysis"}

        analysis = {
            "progression_validation": self._validate_step_progression(
                windows, expected_step_months
            ),
            "temporal_coverage": self._analyze_temporal_coverage(windows, timestamps),
            "bias_detection": self._detect_potential_bias(windows, timestamps),
            "consistency_metrics": self._calculate_consistency_metrics(windows),
        }

        self._analysis_cache = analysis
        return analysis

    def _validate_step_progression(
        self, windows: list[RollSplit], expected_step_months: int
    ) -> dict[str, Any]:
        """Validate that step progression follows expected pattern."""

        actual_steps = []
        for i in range(1, len(windows)):
            step_days = (
                windows[i].train_period.start_date - windows[i - 1].train_period.start_date
            ).days
            actual_steps.append(step_days)

        expected_step_days = expected_step_months * 30.44  # Average days per month

        # Statistical validation
        mean_step = np.mean(actual_steps)
        std_step = np.std(actual_steps)

        # Allow reasonable tolerance for calendar variations
        tolerance_days = 15
        valid_progression = abs(mean_step - expected_step_days) <= tolerance_days

        return {
            "valid_progression": valid_progression,
            "expected_step_days": round(expected_step_days, 1),
            "actual_mean_step_days": round(mean_step, 1),
            "step_std_days": round(std_step, 2),
            "step_consistency": round(1.0 - (std_step / mean_step), 3) if mean_step > 0 else 0.0,
            "all_steps": actual_steps,
        }

    def _analyze_temporal_coverage(
        self, windows: list[RollSplit], timestamps: list[pd.Timestamp]
    ) -> dict[str, Any]:
        """Analyze temporal coverage of walk-forward windows."""

        if not timestamps:
            return {"error": "No timestamps provided"}

        total_range = max(timestamps) - min(timestamps)

        # Calculate coverage by each window type
        train_coverage = sum(w.train_period.duration_days for w in windows)
        val_coverage = sum(w.validation_period.duration_days for w in windows)
        test_coverage = sum(w.test_period.duration_days for w in windows)

        # Unique coverage (accounting for walk-forward overlap)
        unique_start = windows[0].train_period.start_date
        unique_end = windows[-1].test_period.end_date
        unique_coverage_days = (unique_end - unique_start).days

        return {
            "total_data_range_days": total_range.days,
            "unique_coverage_days": unique_coverage_days,
            "coverage_efficiency": round(unique_coverage_days / total_range.days, 3),
            "train_period_days": train_coverage,
            "validation_period_days": val_coverage,
            "test_period_days": test_coverage,
            "coverage_start": unique_start.isoformat(),
            "coverage_end": unique_end.isoformat(),
        }

    def _detect_potential_bias(
        self, windows: list[RollSplit], timestamps: list[pd.Timestamp]
    ) -> dict[str, Any]:
        """Detect potential sources of bias in walk-forward setup."""

        bias_checks = {
            "look_ahead_bias": self._check_look_ahead_bias(windows),
            "data_snooping_bias": self._check_data_snooping_bias(windows, timestamps),
            "survivorship_bias": self._check_survivorship_bias(windows, timestamps),
        }

        overall_clean = all(not check["detected"] for check in bias_checks.values())

        return {
            "overall_bias_free": overall_clean,
            "bias_checks": bias_checks,
        }

    def _check_look_ahead_bias(self, windows: list[RollSplit]) -> dict[str, Any]:
        """Check for look-ahead bias in window setup."""

        violations = []

        for i, window in enumerate(windows):
            # Check period separation
            if window.train_period.end_date > window.validation_period.start_date:
                violations.append(f"Window {i}: Training overlaps validation")

            if window.validation_period.end_date > window.test_period.start_date:
                violations.append(f"Window {i}: Validation overlaps test")

        return {
            "detected": len(violations) > 0,
            "violations": violations,
            "clean_windows": len(windows) - len(violations),
        }

    def _check_data_snooping_bias(
        self, windows: list[RollSplit], timestamps: list[pd.Timestamp]
    ) -> dict[str, Any]:
        """Check for potential data snooping bias."""

        # Simple check for excessive window overlap
        high_overlap_count = 0

        for i in range(1, len(windows)):
            current_train = windows[i].train_period
            prev_train = windows[i - 1].train_period

            # Calculate training period overlap
            overlap_start = max(current_train.start_date, prev_train.start_date)
            overlap_end = min(current_train.end_date, prev_train.end_date)

            if overlap_start < overlap_end:
                overlap_days = (overlap_end - overlap_start).days
                total_days = current_train.duration_days
                overlap_ratio = overlap_days / total_days

                # Flag if overlap > 95% (too similar)
                if overlap_ratio > 0.95:
                    high_overlap_count += 1

        return {
            "detected": high_overlap_count > len(windows) * 0.8,  # 80% threshold
            "high_overlap_windows": high_overlap_count,
            "total_windows": len(windows),
            "overlap_ratio": round(high_overlap_count / len(windows), 3),
        }

    def _check_survivorship_bias(
        self, windows: list[RollSplit], timestamps: list[pd.Timestamp]
    ) -> dict[str, Any]:
        """Check for potential survivorship bias."""

        # Check for consistent data availability across all windows
        data_availability = []

        timestamp_set = set(timestamps)

        for window in windows:
            train_data = sum(1 for ts in timestamp_set if window.train_period.contains_date(ts))

            expected_train_days = window.train_period.duration_days
            availability_ratio = train_data / expected_train_days
            data_availability.append(availability_ratio)

        # Check for suspiciously high availability (potential survivorship bias)
        avg_availability = np.mean(data_availability)
        min_availability = min(data_availability)

        # Flag if availability is too high (> 95% might indicate survivorship bias)
        suspicious_availability = avg_availability > 0.95 and min_availability > 0.90

        return {
            "detected": suspicious_availability,
            "avg_data_availability": round(avg_availability, 3),
            "min_data_availability": round(min_availability, 3),
            "availability_variance": round(np.var(data_availability), 4),
            "note": "High availability might indicate survivorship bias",
        }

    def _calculate_consistency_metrics(self, windows: list[RollSplit]) -> dict[str, Any]:
        """Calculate consistency metrics across windows."""

        # Period duration consistency
        train_durations = [w.train_period.duration_days for w in windows]
        val_durations = [w.validation_period.duration_days for w in windows]
        test_durations = [w.test_period.duration_days for w in windows]

        return {
            "train_period_consistency": {
                "mean_days": round(np.mean(train_durations), 1),
                "std_days": round(np.std(train_durations), 2),
                "cv": round(np.std(train_durations) / np.mean(train_durations), 4),
            },
            "validation_period_consistency": {
                "mean_days": round(np.mean(val_durations), 1),
                "std_days": round(np.std(val_durations), 2),
                "cv": round(np.std(val_durations) / np.mean(val_durations), 4),
            },
            "test_period_consistency": {
                "mean_days": round(np.mean(test_durations), 1),
                "std_days": round(np.std(test_durations), 2),
                "cv": round(np.std(test_durations) / np.mean(test_durations), 4),
            },
        }
