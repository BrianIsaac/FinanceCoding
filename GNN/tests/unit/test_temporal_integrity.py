"""
Unit tests for temporal integrity validation system.

Tests the temporal integrity validator and continuous monitoring
to ensure no look-ahead bias and proper temporal data handling.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.evaluation.validation.rolling_validation import RollSplit, ValidationPeriod
from src.evaluation.validation.temporal_integrity import (
    ContinuousIntegrityMonitor,
    IntegrityCheckResult,
    IntegrityViolation,
    TemporalIntegrityValidator,
)


class TestTemporalIntegrityValidator:
    """Test temporal integrity validator functionality."""

    @pytest.fixture
    def validator(self) -> TemporalIntegrityValidator:
        """Temporal integrity validator instance."""
        return TemporalIntegrityValidator(strict_mode=False)

    @pytest.fixture
    def strict_validator(self) -> TemporalIntegrityValidator:
        """Strict mode validator instance."""
        return TemporalIntegrityValidator(strict_mode=True)

    @pytest.fixture
    def valid_split(self) -> RollSplit:
        """Valid rolling split for testing."""
        train_period = ValidationPeriod(pd.Timestamp("2020-01-01"), pd.Timestamp("2023-01-01"))
        val_period = ValidationPeriod(pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01"))
        test_period = ValidationPeriod(pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01"))

        return RollSplit(train_period, val_period, test_period)

    @pytest.fixture
    def invalid_split(self) -> RollSplit:
        """Invalid rolling split with overlapping periods."""
        train_period = ValidationPeriod(
            pd.Timestamp("2020-01-01"), pd.Timestamp("2023-06-01")  # Overlaps with validation!
        )
        val_period = ValidationPeriod(pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01"))
        test_period = ValidationPeriod(pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01"))

        return RollSplit(train_period, val_period, test_period)

    @pytest.fixture
    def sample_timestamps(self) -> list[pd.Timestamp]:
        """Sample timestamps for testing."""
        return pd.date_range(start="2020-01-01", end="2025-12-31", freq="D").tolist()

    def test_initialization(self):
        """Test validator initialization."""
        validator = TemporalIntegrityValidator(strict_mode=True)
        assert validator.strict_mode is True
        assert validator.violations_log == []
        assert validator.check_history == []

    def test_valid_split_passes_validation(
        self,
        validator: TemporalIntegrityValidator,
        valid_split: RollSplit,
        sample_timestamps: list[pd.Timestamp],
    ):
        """Test that valid split passes all integrity checks."""
        result = validator.validate_split_integrity(valid_split, sample_timestamps, "test_model")

        assert isinstance(result, IntegrityCheckResult)
        assert result.passed is True
        assert result.check_name == "split_integrity_test_model"
        assert len([v for v in result.violations if v.severity == "critical"]) == 0

    def test_invalid_split_fails_validation(
        self,
        validator: TemporalIntegrityValidator,
        invalid_split: RollSplit,
        sample_timestamps: list[pd.Timestamp],
    ):
        """Test that invalid split fails integrity checks."""
        result = validator.validate_split_integrity(invalid_split, sample_timestamps, "test_model")

        assert result.passed is False
        assert len(result.violations) > 0

        # Should detect period overlap violation
        overlap_violations = [v for v in result.violations if v.violation_type == "period_overlap"]
        assert len(overlap_violations) > 0

    def test_strict_mode_raises_exception(
        self,
        strict_validator: TemporalIntegrityValidator,
        invalid_split: RollSplit,
        sample_timestamps: list[pd.Timestamp],
    ):
        """Test that strict mode raises exceptions on critical violations."""
        with pytest.raises(ValueError, match="Critical temporal integrity violations"):
            strict_validator.validate_split_integrity(
                invalid_split, sample_timestamps, "test_model"
            )

    def test_period_separation_check(self, validator: TemporalIntegrityValidator):
        """Test period separation checking."""
        # Create split with overlapping periods
        overlapping_split = RollSplit(
            ValidationPeriod(pd.Timestamp("2020-01-01"), pd.Timestamp("2023-06-01")),
            ValidationPeriod(pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")),  # Overlap!
            ValidationPeriod(pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")),
        )

        result = validator._check_period_separation(overlapping_split)

        assert result.passed is False
        assert len(result.violations) > 0
        assert result.violations[0].violation_type == "period_overlap"
        assert result.violations[0].severity == "critical"

    def test_temporal_ordering_check(self, validator: TemporalIntegrityValidator):
        """Test temporal ordering checking."""
        # Create split with wrong temporal order
        wrong_order_split = RollSplit(
            ValidationPeriod(pd.Timestamp("2020-01-01"), pd.Timestamp("2023-01-01")),
            ValidationPeriod(
                pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")
            ),  # Wrong order!
            ValidationPeriod(pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")),
        )

        result = validator._check_temporal_ordering(wrong_order_split)

        assert result.passed is False
        assert len(result.violations) > 0
        assert result.violations[0].violation_type == "temporal_ordering"

    def test_data_leakage_detection(
        self, validator: TemporalIntegrityValidator, valid_split: RollSplit
    ):
        """Test data leakage detection."""
        # Create timestamps that would cause leakage
        leaky_timestamps = [
            pd.Timestamp("2020-06-01"),  # Training period
            pd.Timestamp("2023-06-01"),  # Should be in validation but appears in training range
            pd.Timestamp("2024-06-01"),  # Test period
        ]

        result = validator._check_data_leakage(valid_split, leaky_timestamps)

        # Should pass because timestamps are properly separated
        assert result.passed is True

    def test_future_information_access_detection(self, validator: TemporalIntegrityValidator):
        """Test future information access detection."""
        split = RollSplit(
            ValidationPeriod(pd.Timestamp("2020-01-01"), pd.Timestamp("2023-01-01")),
            ValidationPeriod(pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")),
            ValidationPeriod(pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")),
        )

        # Create problematic timestamps
        future_timestamps = [
            pd.Timestamp("2022-01-01"),  # Training - OK
            pd.Timestamp("2023-06-01"),  # Validation - OK
            pd.Timestamp("2024-06-01"),  # Test - OK
        ]

        result = validator._check_future_information_access(split, future_timestamps)

        # Should pass as timestamps are appropriate for their periods
        assert result.passed is True

    def test_data_sufficiency_check(
        self, validator: TemporalIntegrityValidator, valid_split: RollSplit
    ):
        """Test data sufficiency checking."""
        # Create insufficient timestamps
        sparse_timestamps = [
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-01-15"),  # Only 2 timestamps in 3-year training period
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2024-01-01"),
        ]

        result = validator._check_data_sufficiency(
            valid_split,
            sparse_timestamps,
            min_train_samples=10,
            min_val_samples=1,
            min_test_samples=1,
        )

        # Should detect insufficient training data
        insufficient_violations = [
            v
            for v in result.violations
            if v.violation_type == "insufficient_data" and "training" in v.description
        ]
        assert len(insufficient_violations) > 0

    def test_boundary_check(self, validator: TemporalIntegrityValidator, valid_split: RollSplit):
        """Test period boundary checking."""
        # Create timestamps that don't cover the full split range
        limited_timestamps = pd.date_range(
            start="2021-01-01", end="2024-06-01", freq="D"  # After split start  # Before split end
        ).tolist()

        result = validator._check_period_boundaries(valid_split, limited_timestamps)

        # Should warn about boundary mismatches
        boundary_violations = [
            v for v in result.violations if v.violation_type == "boundary_mismatch"
        ]
        assert len(boundary_violations) > 0

    def test_violation_summary(
        self,
        validator: TemporalIntegrityValidator,
        invalid_split: RollSplit,
        sample_timestamps: list[pd.Timestamp],
    ):
        """Test violation summary generation."""
        # Generate some violations
        validator.validate_split_integrity(invalid_split, sample_timestamps)

        summary = validator.get_violation_summary()

        assert isinstance(summary, dict)
        assert "total_violations" in summary
        assert "critical_violations" in summary
        assert "violations_by_type" in summary
        assert summary["total_violations"] > 0

    def test_integrity_report_export(
        self,
        validator: TemporalIntegrityValidator,
        invalid_split: RollSplit,
        sample_timestamps: list[pd.Timestamp],
    ):
        """Test integrity report export."""
        # Generate violations
        validator.validate_split_integrity(invalid_split, sample_timestamps)

        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "integrity_report.json"
            validator.export_integrity_report(output_path)

            assert output_path.exists()

            # Validate report content
            with open(output_path) as f:
                report = json.load(f)

            assert "generated_at" in report
            assert "summary" in report
            assert "detailed_violations" in report
            assert len(report["detailed_violations"]) > 0

    def test_clear_history(
        self,
        validator: TemporalIntegrityValidator,
        invalid_split: RollSplit,
        sample_timestamps: list[pd.Timestamp],
    ):
        """Test clearing violation history."""
        # Generate violations
        validator.validate_split_integrity(invalid_split, sample_timestamps)

        assert len(validator.violations_log) > 0
        assert len(validator.check_history) > 0

        validator.clear_history()

        assert len(validator.violations_log) == 0
        assert len(validator.check_history) == 0

    def test_raise_on_critical_violations(
        self,
        validator: TemporalIntegrityValidator,
        invalid_split: RollSplit,
        sample_timestamps: list[pd.Timestamp],
    ):
        """Test raising exception on critical violations."""
        # Generate critical violations
        validator.validate_split_integrity(invalid_split, sample_timestamps)

        with pytest.raises(ValueError, match="Critical temporal integrity violations"):
            validator.raise_on_critical_violations()


class TestContinuousIntegrityMonitor:
    """Test continuous integrity monitoring functionality."""

    @pytest.fixture
    def validator(self) -> TemporalIntegrityValidator:
        """Validator for monitor."""
        return TemporalIntegrityValidator(strict_mode=False)

    @pytest.fixture
    def monitor(self, validator: TemporalIntegrityValidator) -> ContinuousIntegrityMonitor:
        """Continuous integrity monitor instance."""
        return ContinuousIntegrityMonitor(validator, alert_threshold=3)

    def test_initialization(self, validator: TemporalIntegrityValidator):
        """Test monitor initialization."""
        monitor = ContinuousIntegrityMonitor(validator, alert_threshold=5)
        assert monitor.validator == validator
        assert monitor.alert_threshold == 5
        assert monitor.monitoring_active is False
        assert monitor.violation_count == 0

    def test_start_stop_monitoring(self, monitor: ContinuousIntegrityMonitor):
        """Test starting and stopping monitoring."""
        assert monitor.monitoring_active is False

        monitor.start_monitoring()
        assert monitor.monitoring_active is True
        assert monitor.violation_count == 0

        monitor.stop_monitoring()
        assert monitor.monitoring_active is False

    def test_data_access_monitoring(self, monitor: ContinuousIntegrityMonitor):
        """Test data access monitoring."""
        monitor.start_monitoring()

        # Valid access (past data)
        valid_access = monitor.monitor_data_access(
            access_timestamp=pd.Timestamp("2023-01-01"),
            data_timestamp=pd.Timestamp("2022-01-01"),
            operation="training",
        )
        assert valid_access is True

        # Invalid access (future data)
        invalid_access = monitor.monitor_data_access(
            access_timestamp=pd.Timestamp("2022-01-01"),
            data_timestamp=pd.Timestamp("2023-01-01"),
            operation="training",
        )
        assert invalid_access is False
        assert monitor.violation_count == 1

    def test_model_training_monitoring(self, monitor: ContinuousIntegrityMonitor):
        """Test model training monitoring."""
        monitor.start_monitoring()

        # Valid training data
        training_timestamps = pd.date_range(start="2020-01-01", end="2022-12-31", freq="D").tolist()

        valid_training = monitor.monitor_model_training(
            training_start=pd.Timestamp("2020-01-01"),
            training_end=pd.Timestamp("2023-01-01"),
            data_timestamps=training_timestamps,
            model_name="test_model",
        )
        assert valid_training is True

        # Invalid training data (future data included)
        future_training_timestamps = pd.date_range(
            start="2020-01-01", end="2024-12-31", freq="D"  # Beyond training end
        ).tolist()

        invalid_training = monitor.monitor_model_training(
            training_start=pd.Timestamp("2020-01-01"),
            training_end=pd.Timestamp("2023-01-01"),
            data_timestamps=future_training_timestamps,
            model_name="test_model",
        )
        assert invalid_training is False
        assert monitor.violation_count == 1

    def test_prediction_monitoring(self, monitor: ContinuousIntegrityMonitor):
        """Test prediction generation monitoring."""
        monitor.start_monitoring()

        # Valid prediction timing
        valid_prediction = monitor.monitor_prediction_generation(
            prediction_time=pd.Timestamp("2023-01-01"),
            data_cutoff=pd.Timestamp("2022-12-31"),
            model_name="test_model",
        )
        assert valid_prediction is True

        # Invalid prediction timing
        invalid_prediction = monitor.monitor_prediction_generation(
            prediction_time=pd.Timestamp("2022-01-01"),
            data_cutoff=pd.Timestamp("2023-01-01"),
            model_name="test_model",
        )
        assert invalid_prediction is False
        assert monitor.violation_count == 1

    def test_alert_threshold_handling(self, monitor: ContinuousIntegrityMonitor):
        """Test alert threshold handling."""
        monitor.start_monitoring()

        # Mock alert handler
        alert_handler = Mock()
        monitor.add_alert_handler(alert_handler)

        # Generate violations up to threshold
        for i in range(monitor.alert_threshold):
            monitor.monitor_data_access(
                access_timestamp=pd.Timestamp("2022-01-01"),
                data_timestamp=pd.Timestamp(f"2023-0{i+1}-01"),
                operation="test",
            )

        # Alert should be triggered
        assert alert_handler.called
        assert monitor.violation_count == monitor.alert_threshold

    def test_strict_mode_exception_handling(self, validator: TemporalIntegrityValidator):
        """Test strict mode exception handling."""
        strict_validator = TemporalIntegrityValidator(strict_mode=True)
        monitor = ContinuousIntegrityMonitor(strict_validator, alert_threshold=1)
        monitor.start_monitoring()

        # Should raise exception on critical violation
        with pytest.raises(ValueError, match="Critical temporal integrity violation"):
            monitor.monitor_data_access(
                access_timestamp=pd.Timestamp("2022-01-01"),
                data_timestamp=pd.Timestamp("2023-01-01"),
                operation="training",
            )

    def test_monitoring_stats(self, monitor: ContinuousIntegrityMonitor):
        """Test monitoring statistics."""
        stats = monitor.get_monitoring_stats()

        assert isinstance(stats, dict)
        assert "monitoring_active" in stats
        assert "total_violations" in stats
        assert "alert_threshold" in stats

        monitor.start_monitoring()
        updated_stats = monitor.get_monitoring_stats()
        assert updated_stats["monitoring_active"] is True

    def test_inactive_monitoring_bypass(self, monitor: ContinuousIntegrityMonitor):
        """Test that monitoring is bypassed when inactive."""
        # Monitor is not started
        assert monitor.monitoring_active is False

        # Should return True (pass) when monitoring is inactive
        result = monitor.monitor_data_access(
            access_timestamp=pd.Timestamp("2022-01-01"),
            data_timestamp=pd.Timestamp("2023-01-01"),  # Future data - would normally fail
            operation="test",
        )

        assert result is True
        assert monitor.violation_count == 0
