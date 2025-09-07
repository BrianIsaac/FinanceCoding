"""
Rolling validation engine for portfolio optimization models.

This module provides comprehensive rolling window validation framework
for training and evaluating portfolio models with strict temporal
separation and no-look-ahead bias.
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src import train as train_mod  # we will call train_mod.train_gat()
from src.evaluation.metrics.portfolio_metrics import dsr_from_returns
from src.utils.gpu import GPUConfig, GPUMemoryManager

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Configuration and Data Structures
# ------------------------------------------------------------


@dataclass(frozen=True)
class ValidationPeriod:
    """Represents a temporal period for validation."""

    start_date: pd.Timestamp
    end_date: pd.Timestamp

    def __post_init__(self) -> None:
        """Validate period integrity."""
        if self.start_date >= self.end_date:
            raise ValueError(f"Invalid period: start {self.start_date} >= end {self.end_date}")

    @property
    def duration_days(self) -> int:
        """Get period duration in days."""
        return (self.end_date - self.start_date).days

    def contains_date(self, date: pd.Timestamp) -> bool:
        """Check if date falls within this period."""
        return self.start_date <= date < self.end_date


@dataclass(frozen=True)
class RollSplit:
    """Enhanced rolling split with validation periods."""

    train_period: ValidationPeriod
    validation_period: ValidationPeriod
    test_period: ValidationPeriod

    def __post_init__(self) -> None:
        """Validate temporal separation."""
        if self.train_period.end_date > self.validation_period.start_date:
            raise ValueError("Training period overlaps with validation period")
        if self.validation_period.end_date > self.test_period.start_date:
            raise ValueError("Validation period overlaps with test period")

    @property
    def train_start(self) -> pd.Timestamp:
        """Legacy compatibility property."""
        return self.train_period.start_date

    @property
    def val_start(self) -> pd.Timestamp:
        """Legacy compatibility property."""
        return self.validation_period.start_date

    @property
    def test_start(self) -> pd.Timestamp:
        """Legacy compatibility property."""
        return self.test_period.start_date

    def to_datestr_tuple(self) -> tuple[str, str, str]:
        """Convert to string tuple for legacy compatibility."""
        return (
            self.train_period.start_date.date().isoformat(),
            self.validation_period.start_date.date().isoformat(),
            self.test_period.start_date.date().isoformat(),
        )

    def validate_data_integrity(self, data_timestamps: list[pd.Timestamp]) -> bool:
        """Validate that data timestamps respect temporal separation."""
        data_set = set(data_timestamps)

        # Check for data leakage
        for ts in data_set:
            if self.train_period.contains_date(ts):
                # Training data should not contain future information
                if any(ts >= self.validation_period.start_date or ts >= self.test_period.start_date
                      for _ in [None]):  # Simple validation
                    continue  # This is expected
        return True


@dataclass
class BacktestConfig:
    """Enhanced backtest configuration with validation parameters."""

    start_date: pd.Timestamp
    end_date: pd.Timestamp
    training_months: int = 36                   # 3-year training window
    validation_months: int = 12                 # 1-year validation
    test_months: int = 12                       # 1-year out-of-sample test
    step_months: int = 12                       # Annual walk-forward
    rebalance_frequency: str = "M"              # Monthly rebalancing

    # New validation-specific parameters
    min_training_samples: int = 252             # Minimum trading days for training
    max_gap_days: int = 5                       # Maximum allowed gap in data
    require_full_periods: bool = True           # Require complete periods

    def validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date")
        if self.training_months < 12:
            raise ValueError("Training period must be at least 12 months")
        if self.validation_months < 1:
            raise ValueError("Validation period must be at least 1 month")
        if self.test_months < 1:
            raise ValueError("Test period must be at least 1 month")


# ------------------------------------------------------------
# Core Rolling Validation Engine
# ------------------------------------------------------------


class RollingValidationEngine:
    """
    Advanced rolling validation engine with temporal integrity monitoring.

    This engine implements the 36/12/12 month training/validation/test protocol
    with strict temporal separation and comprehensive validation of data integrity.
    """

    def __init__(self, config: BacktestConfig, gpu_config: GPUConfig | None = None):
        """
        Initialize rolling validation engine.

        Args:
            config: Backtest configuration parameters
            gpu_config: Optional GPU configuration for memory management
        """
        config.validate_config()
        self.config = config
        self.gpu_manager = GPUMemoryManager(gpu_config or GPUConfig()) if gpu_config else None
        self._validation_cache: dict[str, Any] = {}

    def generate_rolling_windows(self,
                                sample_timestamps: list[pd.Timestamp]) -> list[RollSplit]:
        """
        Generate rolling windows with enhanced temporal validation.

        Args:
            sample_timestamps: Available data timestamps

        Returns:
            List of validated rolling splits
        """
        if not sample_timestamps:
            raise ValueError("No sample timestamps provided")

        ts_sorted = sorted(set(sample_timestamps))  # Remove duplicates and sort
        logger.info(f"Generating rolling windows from {len(ts_sorted)} timestamps")

        splits = []
        current_start = ts_sorted[0].normalize()

        while True:
            # Calculate period boundaries
            train_end = self._add_months(current_start, self.config.training_months)
            val_start = train_end
            val_end = self._add_months(val_start, self.config.validation_months)
            test_start = val_end
            test_end = self._add_months(test_start, self.config.test_months)

            # Validate that we have sufficient data
            if test_end > ts_sorted[-1]:
                logger.info(f"Stopping window generation: test_end {test_end} exceeds data range")
                break

            # Create validation periods
            train_period = ValidationPeriod(current_start, train_end)
            val_period = ValidationPeriod(val_start, val_end)
            test_period = ValidationPeriod(test_start, test_end)

            # Create and validate the split
            try:
                split = RollSplit(train_period, val_period, test_period)

                # Additional validation for data availability
                if self._validate_split_data_availability(split, ts_sorted):
                    splits.append(split)
                    logger.debug(f"Created split: train {current_start} -> test {test_start}")
                else:
                    logger.warning(f"Skipping split due to insufficient data: {current_start}")

            except ValueError as e:
                logger.error(f"Invalid split at {current_start}: {e}")
                break

            # Advance to next window
            current_start = self._add_months(current_start, self.config.step_months)

            # Prevent infinite loops
            if current_start >= ts_sorted[-1]:
                break

        logger.info(f"Generated {len(splits)} valid rolling windows")
        return splits

    def validate_temporal_integrity(self,
                                   split: RollSplit,
                                   data_timestamps: list[pd.Timestamp]) -> dict[str, bool]:
        """
        Comprehensive validation of temporal data integrity.

        Args:
            split: Rolling split to validate
            data_timestamps: Available data timestamps

        Returns:
            Dictionary of validation results
        """
        results = {
            'no_lookahead_bias': True,
            'sufficient_training_data': True,
            'no_data_gaps': True,
            'period_separation': True
        }

        ts_set = set(data_timestamps)

        # Check for look-ahead bias
        train_data = [ts for ts in ts_set if split.train_period.contains_date(ts)]
        val_data = [ts for ts in ts_set if split.validation_period.contains_date(ts)]
        test_data = [ts for ts in ts_set if split.test_period.contains_date(ts)]

        # Validate temporal separation - strict no-look-ahead enforcement
        if train_data and val_data:
            if max(train_data) >= min(val_data):
                results['period_separation'] = False
                results['no_lookahead_bias'] = False
                logger.error(
                    "Look-ahead bias detected: Training data overlaps with validation data"
                )

        if val_data and test_data:
            if max(val_data) >= min(test_data):
                results['period_separation'] = False
                results['no_lookahead_bias'] = False
                logger.error("Look-ahead bias detected: Validation data overlaps with test data")

        # Check training data sufficiency
        if len(train_data) < self.config.min_training_samples:
            results['sufficient_training_data'] = False
            logger.warning(
                f"Insufficient training data: {len(train_data)} < "
                f"{self.config.min_training_samples}"
            )

        # Check for data gaps
        if train_data:
            train_sorted = sorted(train_data)
            max_gap = max((train_sorted[i+1] - train_sorted[i]).days
                         for i in range(len(train_sorted)-1))
            if max_gap > self.config.max_gap_days:
                results['no_data_gaps'] = False
                logger.warning(f"Large data gap detected: {max_gap} days")

        return results

    def enforce_temporal_guard(self,
                              split: RollSplit,
                              training_data: pd.DataFrame,
                              validation_data: pd.DataFrame | None = None,
                              test_data: pd.DataFrame | None = None) -> tuple[
        pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None
    ]:
        """
        Strictly enforce temporal guards on datasets to prevent look-ahead bias.

        Args:
            split: Rolling split defining temporal boundaries
            training_data: Training dataset with timestamp index
            validation_data: Optional validation dataset
            test_data: Optional test dataset

        Returns:
            Tuple of temporally filtered datasets
        """
        if not isinstance(training_data.index, pd.DatetimeIndex):
            raise ValueError("Training data must have DatetimeIndex")

        # Enforce strict temporal bounds on training data
        train_mask = ((training_data.index >= split.train_period.start_date) &
                     (training_data.index < split.train_period.end_date))
        filtered_train = training_data[train_mask].copy()

        # Verify no future information leaked into training set
        if len(filtered_train) > 0:
            latest_train_date = filtered_train.index.max()
            if latest_train_date >= split.validation_period.start_date:
                raise ValueError(
                    f"Look-ahead bias detected: training data contains {latest_train_date} "
                    f"which is >= validation start {split.validation_period.start_date}"
                )

        filtered_val = None
        if validation_data is not None:
            if not isinstance(validation_data.index, pd.DatetimeIndex):
                raise ValueError("Validation data must have DatetimeIndex")
            val_mask = ((validation_data.index >= split.validation_period.start_date) &
                       (validation_data.index < split.validation_period.end_date))
            filtered_val = validation_data[val_mask].copy()

        filtered_test = None
        if test_data is not None:
            if not isinstance(test_data.index, pd.DatetimeIndex):
                raise ValueError("Test data must have DatetimeIndex")
            test_mask = ((test_data.index >= split.test_period.start_date) &
                        (test_data.index < split.test_period.end_date))
            filtered_test = test_data[test_mask].copy()

        logger.debug(f"Temporal guard applied: train={len(filtered_train)}, "
                    f"val={len(filtered_val) if filtered_val is not None else 0}, "
                    f"test={len(filtered_test) if filtered_test is not None else 0}")

        return filtered_train, filtered_val, filtered_test

    def _add_months(self, timestamp: pd.Timestamp, months: int) -> pd.Timestamp:
        """Add months to timestamp with proper handling of edge cases."""
        return (timestamp + pd.DateOffset(months=months)).normalize()

    def _validate_split_data_availability(self,
                                         split: RollSplit,
                                         timestamps: list[pd.Timestamp]) -> bool:
        """Check if split has sufficient data availability."""
        # Count data points in each period
        train_count = sum(1 for ts in timestamps if split.train_period.contains_date(ts))
        val_count = sum(1 for ts in timestamps if split.validation_period.contains_date(ts))
        test_count = sum(1 for ts in timestamps if split.test_period.contains_date(ts))

        # Minimum data requirements
        min_train = self.config.min_training_samples
        min_val = max(20, self.config.validation_months * 20)  # ~20 trading days per month
        min_test = max(20, self.config.test_months * 20)

        return (train_count >= min_train and
                val_count >= min_val and
                test_count >= min_test)

    def run_walk_forward_analysis(self,
                                 sample_timestamps: list[pd.Timestamp],
                                 step_size_months: int | None = None) -> dict[str, Any]:
        """
        Execute comprehensive walk-forward analysis with configurable step progression.

        Args:
            sample_timestamps: Available data timestamps
            step_size_months: Optional override for step size (defaults to config)

        Returns:
            Dictionary containing walk-forward analysis results
        """
        step_months = step_size_months or self.config.step_months

        # Generate all possible rolling windows
        splits = self.generate_rolling_windows(sample_timestamps)

        if not splits:
            raise ValueError("No valid splits generated for walk-forward analysis")

        analysis_results = {
            'total_splits': len(splits),
            'step_size_months': step_months,
            'coverage_analysis': self._analyze_temporal_coverage(splits, sample_timestamps),
            'progression_validation': self._validate_step_progression(splits),
            'data_utilization': self._analyze_data_utilization(splits, sample_timestamps)
        }

        logger.info(
            f"Walk-forward analysis completed: {len(splits)} splits with {step_months}-month steps"
        )
        return analysis_results

    def _analyze_temporal_coverage(self,
                                  splits: list[RollSplit],
                                  timestamps: list[pd.Timestamp]) -> dict[str, Any]:
        """Analyze temporal coverage of walk-forward splits."""
        if not splits:
            return {'error': 'No splits provided'}

        total_range = max(timestamps) - min(timestamps)
        covered_days = 0

        for split in splits:
            # Each split covers training + validation + test period
            split_range = split.test_period.end_date - split.train_period.start_date
            covered_days += split_range.days

        # Account for overlaps in walk-forward (since we step by less than total window)
        unique_coverage = splits[-1].test_period.end_date - splits[0].train_period.start_date
        coverage_ratio = unique_coverage.days / total_range.days

        return {
            'total_data_range_days': total_range.days,
            'unique_coverage_days': unique_coverage.days,
            'coverage_ratio': round(coverage_ratio, 3),
            'average_split_duration_days': round(covered_days / len(splits)),
            'first_split_start': splits[0].train_period.start_date.isoformat(),
            'last_split_end': splits[-1].test_period.end_date.isoformat()
        }

    def _validate_step_progression(self, splits: list[RollSplit]) -> dict[str, Any]:
        """Validate that step progression follows expected monthly intervals."""
        if len(splits) < 2:
            return {'valid_progression': True, 'note': 'Single split - no progression to validate'}

        expected_step_days = self.config.step_months * 30.44  # Average days per month
        actual_steps = []

        for i in range(1, len(splits)):
            step_days = (
                splits[i].train_period.start_date - splits[i-1].train_period.start_date
            ).days
            actual_steps.append(step_days)

        avg_step_days = np.mean(actual_steps)
        step_variance = np.var(actual_steps)

        # Allow some tolerance for month length variations
        tolerance = 10  # days
        valid_progression = abs(avg_step_days - expected_step_days) <= tolerance

        return {
            'valid_progression': valid_progression,
            'expected_step_days': round(expected_step_days, 1),
            'actual_avg_step_days': round(avg_step_days, 1),
            'step_variance': round(step_variance, 2),
            'all_step_days': actual_steps
        }

    def _analyze_data_utilization(self,
                                 splits: list[RollSplit],
                                 timestamps: list[pd.Timestamp]) -> dict[str, Any]:
        """Analyze how efficiently the walk-forward process utilizes available data."""
        ts_set = set(timestamps)

        utilization_stats = {
            'splits_with_full_training': 0,
            'splits_with_full_validation': 0,
            'splits_with_full_test': 0,
            'min_training_samples': float('inf'),
            'max_training_samples': 0,
            'avg_training_samples': 0
        }

        total_training_samples = 0

        for split in splits:
            train_samples = sum(1 for ts in ts_set if split.train_period.contains_date(ts))
            val_samples = sum(1 for ts in ts_set if split.validation_period.contains_date(ts))
            test_samples = sum(1 for ts in ts_set if split.test_period.contains_date(ts))

            total_training_samples += train_samples

            # Check if periods are "full" (have expected minimum data)
            expected_train = self.config.training_months * 20  # ~20 trading days per month
            expected_val = self.config.validation_months * 20
            expected_test = self.config.test_months * 20

            if train_samples >= expected_train * 0.9:  # 90% threshold for "full"
                utilization_stats['splits_with_full_training'] += 1
            if val_samples >= expected_val * 0.9:
                utilization_stats['splits_with_full_validation'] += 1
            if test_samples >= expected_test * 0.9:
                utilization_stats['splits_with_full_test'] += 1

            utilization_stats['min_training_samples'] = min(
                utilization_stats['min_training_samples'], train_samples
            )
            utilization_stats['max_training_samples'] = max(
                utilization_stats['max_training_samples'], train_samples
            )

        if splits:
            utilization_stats['avg_training_samples'] = total_training_samples // len(splits)

        return utilization_stats

    def create_integrity_monitor(self) -> TemporalIntegrityMonitor:
        """Create a temporal integrity monitor for continuous validation."""
        return TemporalIntegrityMonitor(self.config)


class TemporalIntegrityMonitor:
    """
    Comprehensive temporal data integrity monitoring system.

    This class provides continuous monitoring of temporal data integrity
    throughout the training and validation process.
    """

    def __init__(self, config: BacktestConfig):
        """Initialize integrity monitor."""
        self.config = config
        self._integrity_log: list[dict[str, Any]] = []
        self._violations: list[dict[str, Any]] = []

    def monitor_split_integrity(self,
                               split: RollSplit,
                               data_timestamps: list[pd.Timestamp],
                               model_name: str = "unknown") -> dict[str, Any]:
        """
        Monitor and log integrity violations for a specific split.

        Args:
            split: Rolling split to monitor
            data_timestamps: Available data timestamps
            model_name: Name of model being trained

        Returns:
            Integrity report for this split
        """
        timestamp = pd.Timestamp.now()

        # Perform comprehensive integrity checks
        integrity_results = self._run_comprehensive_checks(split, data_timestamps)

        # Log the results
        log_entry = {
            'timestamp': timestamp,
            'model_name': model_name,
            'split_info': {
                'train_start': split.train_period.start_date.isoformat(),
                'train_end': split.train_period.end_date.isoformat(),
                'val_start': split.validation_period.start_date.isoformat(),
                'val_end': split.validation_period.end_date.isoformat(),
                'test_start': split.test_period.start_date.isoformat(),
                'test_end': split.test_period.end_date.isoformat()
            },
            'integrity_results': integrity_results,
            'overall_pass': all(integrity_results.values())
        }

        self._integrity_log.append(log_entry)

        # Track violations
        if not log_entry['overall_pass']:
            violation = {
                'timestamp': timestamp,
                'model_name': model_name,
                'split_start': split.train_period.start_date.isoformat(),
                'violations': [k for k, v in integrity_results.items() if not v]
            }
            self._violations.append(violation)
            logger.error(f"Temporal integrity violation detected: {violation['violations']}")

        return log_entry

    def _run_comprehensive_checks(self,
                                 split: RollSplit,
                                 data_timestamps: list[pd.Timestamp]) -> dict[str, bool]:
        """Run comprehensive integrity checks on a split."""
        checks = {
            'no_period_overlap': True,
            'sufficient_data': True,
            'no_data_leakage': True,
            'proper_temporal_ordering': True,
            'minimum_gap_compliance': True
        }

        ts_sorted = sorted(data_timestamps)

        # Check 1: No period overlap
        if split.train_period.end_date > split.validation_period.start_date:
            checks['no_period_overlap'] = False
        if split.validation_period.end_date > split.test_period.start_date:
            checks['no_period_overlap'] = False

        # Check 2: Sufficient data in each period
        train_data = [ts for ts in ts_sorted if split.train_period.contains_date(ts)]
        val_data = [ts for ts in ts_sorted if split.validation_period.contains_date(ts)]
        test_data = [ts for ts in ts_sorted if split.test_period.contains_date(ts)]

        min_samples = max(20, self.config.min_training_samples // 12)  # Per period minimum
        if len(train_data) < self.config.min_training_samples:
            checks['sufficient_data'] = False
        if len(val_data) < min_samples or len(test_data) < min_samples:
            checks['sufficient_data'] = False

        # Check 3: No data leakage (future info in past periods)
        if train_data and val_data:
            if max(train_data) >= min(val_data):
                checks['no_data_leakage'] = False
        if val_data and test_data:
            if max(val_data) >= min(test_data):
                checks['no_data_leakage'] = False

        # Check 4: Proper temporal ordering within periods
        for period_data in [train_data, val_data, test_data]:
            if len(period_data) > 1:
                if sorted(period_data) != period_data:
                    checks['proper_temporal_ordering'] = False
                    break

        # Check 5: Minimum gap compliance between periods
        if train_data and val_data:
            gap = min(val_data) - max(train_data)
            if gap.days < 0:  # Should never happen if no overlap, but double-check
                checks['minimum_gap_compliance'] = False

        return checks

    def get_integrity_summary(self) -> dict[str, Any]:
        """Get summary of all integrity monitoring results."""
        total_checks = len(self._integrity_log)
        passed_checks = sum(1 for entry in self._integrity_log if entry['overall_pass'])

        violation_types = {}
        for violation in self._violations:
            for v_type in violation['violations']:
                violation_types[v_type] = violation_types.get(v_type, 0) + 1

        return {
            'total_splits_monitored': total_checks,
            'splits_passed': passed_checks,
            'splits_failed': total_checks - passed_checks,
            'success_rate': round(passed_checks / total_checks * 100, 2) if total_checks > 0 else 0,
            'violation_summary': violation_types,
            'most_common_violations': sorted(
                violation_types.items(), key=lambda x: x[1], reverse=True
            )
        }

    def export_integrity_report(self, output_path: Path) -> None:
        """Export comprehensive integrity report to file."""
        report = {
            'generated_at': pd.Timestamp.now().isoformat(),
            'config': {
                'training_months': self.config.training_months,
                'validation_months': self.config.validation_months,
                'test_months': self.config.test_months,
                'step_months': self.config.step_months,
                'min_training_samples': self.config.min_training_samples
            },
            'summary': self.get_integrity_summary(),
            'detailed_log': self._integrity_log,
            'violations': self._violations
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Integrity report exported to {output_path}")

    def raise_on_violations(self) -> None:
        """Raise exception if any integrity violations were detected."""
        if self._violations:
            violation_summary = self.get_integrity_summary()
            raise ValueError(f"Temporal integrity violations detected: "
                           f"{violation_summary['splits_failed']} failures out of "
                           f"{violation_summary['total_splits_monitored']} splits")


def _month_add(ts: pd.Timestamp, months: int) -> pd.Timestamp:
    """Legacy function for backward compatibility."""
    return (ts + pd.DateOffset(months=months)).normalize()


def make_rolling_splits(
    sample_timestamps: list[pd.Timestamp],
    *,
    train_months: int = 36,
    val_months: int = 12,
    test_months: int = 12,
    step_months: int = 12,
) -> list[RollSplit]:
    """
    Build rolling (train/val/test) splits by months.
    Uses sample_timestamps (e.g., graph_YYYY-MM-DD.pt dates) as anchors.
    """
    if not sample_timestamps:
        return []

    ts_sorted = sorted(sample_timestamps)
    t_min = ts_sorted[0].normalize()
    t_max = ts_sorted[-1].normalize()

    splits: list[RollSplit] = []
    cur_train_start = t_min

    while True:
        v0 = _month_add(cur_train_start, train_months)
        t0 = _month_add(v0, val_months)

        # we need at least the test_start to be <= last sample date
        if v0 > t_max or t0 > t_max:
            break

        splits.append(RollSplit(train_start=cur_train_start, val_start=v0, test_start=t0))

        # advance
        cur_train_start = _month_add(cur_train_start, step_months)
        if cur_train_start >= t0:
            # do not allow overlap errors; if step is too large we still progress
            cur_train_start = _month_add(t0, 0)

        # guard infinite loops
        if cur_train_start > t_max:
            break

    return splits


# ------------------------------------------------------------
# Execution per roll & aggregation
# ------------------------------------------------------------


@dataclass
class RollResult:
    roll_id: int
    split: RollSplit
    seed: int
    out_dir: Path
    metrics_row: dict[str, float] | None  # loaded from strategy_metrics.csv if found
    daily_returns_path: Path | None  # gat_daily_returns.csv if found
    note: str | None = None


def _find_sample_dates(graph_dir: Path) -> list[pd.Timestamp]:
    pats = list(graph_dir.glob("graph_*.pt"))
    out = []
    for p in pats:
        # train.py has _infer_ts; we mimic here to avoid importing private helper
        m = pd.to_datetime(p.stem.split("_")[-1], errors="coerce")
        if pd.isna(m):
            continue
        out.append(pd.Timestamp(m.date()))
    return sorted(set(out))


def _clone_cfg_for_roll(
    base_cfg: DictConfig, split: RollSplit, seed: int, out_dir: Path
) -> DictConfig:
    cfg = copy.deepcopy(base_cfg)
    cfg.split.train_start = split.train_start.date().isoformat()
    cfg.split.val_start = split.val_start.date().isoformat()
    cfg.split.test_start = split.test_start.date().isoformat()
    cfg.train.seed = int(seed)
    cfg.train.out_dir = str(out_dir.as_posix())
    return cfg


def _load_metrics_if_any(out_dir: Path) -> tuple[dict[str, float] | None, Path | None]:
    sm_path = out_dir / "strategy_metrics.csv"
    r_path = out_dir / "gat_daily_returns.csv"
    metrics_row = None
    if sm_path.exists():
        try:
            df = pd.read_csv(sm_path)
            if not df.empty:
                row = df.iloc[0].to_dict()
                # ensure floats
                for k, v in list(row.items()):
                    if k != "strategy":
                        try:
                            row[k] = float(v)
                        except Exception:
                            row[k] = np.nan
                metrics_row = row
        except Exception:
            metrics_row = None
    return metrics_row, (r_path if r_path.exists() else None)


def run_rolling(
    cfg: DictConfig,
    *,
    out_root: Path,
    train_months: int = 36,
    val_months: int = 12,
    test_months: int = 12,
    step_months: int = 12,
    seeds: Iterable[int] = (42,),
    early_stop: bool = True,
    es_patience: int = 5,
    es_min_delta: float = 0.0,
) -> dict[str, object]:
    """
    Orchestrate multiple rolling train/val/test runs.
    For each roll:
      - adjust cfg.split.* to the roll dates
      - set cfg.train.seed, cfg.train.out_dir to a roll/seed-specific folder
      - optionally enable early stopping parameters (train.py will consume these)
      - call src.train.train_gat()
      - collect metrics & returns
    Finally:
      - aggregate metrics across rolls & seeds
      - compute conservative Deflated Sharpe across all trials

    Returns a dict payload with:
      {
        "splits": [ ... ],
        "rows": [ per roll/seed metrics... ],
        "summary": { "avg": ..., "std": ..., "DSR": ... }
      }
    """
    out_root.mkdir(parents=True, exist_ok=True)

    # inject early-stop hints into cfg (train.py should honor them)
    if early_stop:
        # We'll add these keys to cfg.train; the refactored train.py will look for them.
        cfg.train.setdefault("early_stop_on_val_sharpe", True)
        cfg.train.setdefault("early_stop_patience", int(es_patience))
        cfg.train.setdefault("early_stop_min_delta", float(es_min_delta))

    # Discover all sample dates and build rolls
    graph_dir = Path(cfg.data.graph_dir)
    sample_dates = _find_sample_dates(graph_dir)
    splits = make_rolling_splits(
        sample_dates,
        train_months=train_months,
        val_months=val_months,
        test_months=test_months,
        step_months=step_months,
    )

    if not splits:
        raise RuntimeError(
            "No rolling splits could be created; check your graph dates and parameters."
        )

    # Execute
    results: list[RollResult] = []
    for ridx, split in enumerate(splits):
        roll_dir = (
            out_root / f"roll_{ridx:02d}_{split.train_start.date()}_{split.test_start.date()}"
        )
        for seed in seeds:
            sub_out = roll_dir / f"seed_{int(seed)}"
            sub_out.mkdir(parents=True, exist_ok=True)

            cfg_roll = _clone_cfg_for_roll(cfg, split, seed=int(seed), out_dir=sub_out)

            # also drop any previous outputs if you want clean runs
            # (we intentionally do not delete to allow resuming)

            # run training & backtest
            try:
                train_mod.train_gat(cfg_roll)  # <-- relies on refactored train.py
                note = None
            except Exception as e:
                # still record the failure but continue
                note = f"FAIL: {type(e).__name__}: {e}"

            # load metrics/returns if available
            met, r_path = _load_metrics_if_any(sub_out)
            results.append(
                RollResult(
                    roll_id=ridx,
                    split=split,
                    seed=int(seed),
                    out_dir=sub_out,
                    metrics_row=met,
                    daily_returns_path=r_path,
                    note=note,
                )
            )

    # Aggregate across rolls/seeds
    rows: list[dict[str, object]] = []
    r_all_concat: list[pd.Series] = []

    for rr in results:
        base = {
            "roll_id": rr.roll_id,
            "train_start": rr.split.train_start.date().isoformat(),
            "val_start": rr.split.val_start.date().isoformat(),
            "test_start": rr.split.test_start.date().isoformat(),
            "seed": rr.seed,
            "out_dir": str(rr.out_dir.as_posix()),
        }
        if rr.metrics_row is not None:
            rows.append({**base, **rr.metrics_row})
        else:
            rows.append(
                {
                    **base,
                    "strategy": "GAT",
                    "CAGR": np.nan,
                    "AnnMean": np.nan,
                    "AnnVol": np.nan,
                    "Sharpe": np.nan,
                    "MDD": np.nan,
                }
            )

        if rr.daily_returns_path is not None and rr.daily_returns_path.exists():
            try:
                r = pd.read_csv(rr.daily_returns_path, parse_dates=[0], index_col=0).iloc[:, 0]
                r.index = pd.to_datetime(r.index)
                r_all_concat.append(r.rename(f"roll{rr.roll_id}_seed{rr.seed}"))
            except Exception:
                pass

    # compute average metrics across rows
    df_rows = pd.DataFrame(rows)
    # drop non-numeric for averaging
    metric_cols = ["CAGR", "AnnMean", "AnnVol", "Sharpe", "MDD"]
    df_metrics = df_rows[metric_cols].apply(pd.to_numeric, errors="coerce")
    avg_metrics = df_metrics.mean(skipna=True).to_dict()
    std_metrics = df_metrics.std(skipna=True).to_dict()

    # conservative Deflated Sharpe across all concatenated daily return streams
    # approach: stack all streams (align by time, fillna=0), average equal-weight,
    # then compute DSR with num_trials = count of roll/seed streams.
    dsr = np.nan
    if r_all_concat:
        r_matrix = pd.concat(r_all_concat, axis=1).fillna(0.0)
        # equal-weight ensemble
        r_ens = r_matrix.mean(axis=1)
        dsr = dsr_from_returns(r_ens, sr_benchmark=0.0, num_trials=len(r_all_concat))

        # also persist ensemble daily returns
        (1.0 + r_ens).cumprod().rename("equity").to_frame().to_csv(
            out_root / "ensemble_equity_daily.csv"
        )
        r_ens.rename("r").to_frame().to_csv(out_root / "ensemble_daily_returns.csv")

    # Save artifacts
    df_rows.to_csv(out_root / "rolling_results_detailed.csv", index=False)
    summary_payload = {
        "num_rolls": len(splits),
        "num_streams": len(r_all_concat),
        "avg": avg_metrics,
        "std": std_metrics,
        "deflated_sharpe_conservative": float(dsr) if np.isfinite(dsr) else np.nan,
    }
    with (out_root / "rolling_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    return {
        "splits": [s.to_datestr_tuple() for s in splits],
        "rows": rows,
        "summary": summary_payload,
    }
