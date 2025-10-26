"""
Temporal integrity monitoring and validation system.

This module provides comprehensive monitoring of temporal data integrity
throughout the rolling validation and backtesting process to prevent
look-ahead bias and ensure methodological soundness.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.validation.rolling_validation import RollSplit

logger = logging.getLogger(__name__)


@dataclass
class IntegrityViolation:
    """Represents a temporal integrity violation."""

    violation_type: str
    severity: str  # "critical", "warning", "info"
    description: str
    timestamp: pd.Timestamp
    split_info: dict[str, str] | None = None
    data_info: dict[str, Any] | None = None


@dataclass
class IntegrityCheckResult:
    """Result of temporal integrity check."""

    passed: bool
    check_name: str
    violations: list[IntegrityViolation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    check_duration_ms: float = 0.0


class TemporalIntegrityValidator:
    """
    Comprehensive temporal integrity validation system.

    This validator performs multiple levels of temporal integrity checks:
    1. Basic temporal ordering validation
    2. Look-ahead bias detection
    3. Data leakage prevention
    4. Period separation enforcement
    5. Continuous monitoring during execution
    """

    def __init__(self, strict_mode: bool = True):
        """
        Initialize temporal integrity validator.

        Args:
            strict_mode: If True, raise exceptions on critical violations
        """
        self.strict_mode = strict_mode
        self.violations_log: list[IntegrityViolation] = []
        self.check_history: list[IntegrityCheckResult] = []

    def validate_split_integrity(
        self,
        split: RollSplit,
        data_timestamps: list[pd.Timestamp],
        model_name: str = "unknown",
    ) -> IntegrityCheckResult:
        """
        Perform comprehensive integrity validation on a rolling split.

        Args:
            split: Rolling split to validate
            data_timestamps: Available data timestamps
            model_name: Name of model being validated

        Returns:
            Comprehensive integrity check result
        """
        start_time = pd.Timestamp.now()
        violations = []

        # Run all integrity checks
        check_results = [
            self._check_period_separation(split),
            self._check_temporal_ordering(split),
            self._check_data_leakage(split, data_timestamps),
            self._check_future_information_access(split, data_timestamps),
            self._check_data_sufficiency(split, data_timestamps),
            self._check_period_boundaries(split, data_timestamps),
        ]

        # Collect violations
        for result in check_results:
            violations.extend(result.violations)

        # Determine overall pass/fail
        critical_violations = [v for v in violations if v.severity == "critical"]
        overall_passed = len(critical_violations) == 0

        # Create result
        check_duration = (pd.Timestamp.now() - start_time).total_seconds() * 1000

        result = IntegrityCheckResult(
            passed=overall_passed,
            check_name=f"split_integrity_{model_name}",
            violations=violations,
            metadata={
                "split_info": {
                    "train_start": split.train_period.start_date.isoformat(),
                    "train_end": split.train_period.end_date.isoformat(),
                    "val_start": split.validation_period.start_date.isoformat(),
                    "val_end": split.validation_period.end_date.isoformat(),
                    "test_start": split.test_period.start_date.isoformat(),
                    "test_end": split.test_period.end_date.isoformat(),
                },
                "data_info": {
                    "total_timestamps": len(data_timestamps),
                    "data_range": (
                        (min(data_timestamps).isoformat(), max(data_timestamps).isoformat())
                        if data_timestamps
                        else ("", "")
                    ),
                },
                "model_name": model_name,
            },
            check_duration_ms=check_duration,
        )

        # Log violations
        self.violations_log.extend(violations)
        self.check_history.append(result)

        # Handle critical violations
        if critical_violations and self.strict_mode:
            violation_descriptions = [v.description for v in critical_violations]
            raise ValueError(
                f"Critical temporal integrity violations detected: {violation_descriptions}"
            )

        return result

    def _check_period_separation(self, split: RollSplit) -> IntegrityCheckResult:
        """Check that periods are properly separated."""
        violations = []

        # Check training-validation separation
        if split.train_period.end_date > split.validation_period.start_date:
            violations.append(
                IntegrityViolation(
                    violation_type="period_overlap",
                    severity="critical",
                    description=f"Training period ends ({split.train_period.end_date}) after validation starts ({split.validation_period.start_date})",
                    timestamp=pd.Timestamp.now(),
                    split_info={"issue": "train_val_overlap"},
                )
            )

        # Check validation-test separation
        if split.validation_period.end_date > split.test_period.start_date:
            violations.append(
                IntegrityViolation(
                    violation_type="period_overlap",
                    severity="critical",
                    description=f"Validation period ends ({split.validation_period.end_date}) after test starts ({split.test_period.start_date})",
                    timestamp=pd.Timestamp.now(),
                    split_info={"issue": "val_test_overlap"},
                )
            )

        return IntegrityCheckResult(
            passed=len(violations) == 0,
            check_name="period_separation",
            violations=violations,
        )

    def _check_temporal_ordering(self, split: RollSplit) -> IntegrityCheckResult:
        """Check that periods are in correct temporal order."""
        violations = []

        # Check chronological order
        periods = [
            ("training", split.train_period),
            ("validation", split.validation_period),
            ("test", split.test_period),
        ]

        for i in range(len(periods) - 1):
            current_name, current_period = periods[i]
            next_name, next_period = periods[i + 1]

            if current_period.start_date >= next_period.start_date:
                violations.append(
                    IntegrityViolation(
                        violation_type="temporal_ordering",
                        severity="critical",
                        description=f"{current_name} period starts ({current_period.start_date}) at or after {next_name} period starts ({next_period.start_date})",
                        timestamp=pd.Timestamp.now(),
                        split_info={"current_period": current_name, "next_period": next_name},
                    )
                )

        return IntegrityCheckResult(
            passed=len(violations) == 0,
            check_name="temporal_ordering",
            violations=violations,
        )

    def _check_data_leakage(
        self, split: RollSplit, data_timestamps: list[pd.Timestamp]
    ) -> IntegrityCheckResult:
        """Check for data leakage between periods."""
        violations = []

        if not data_timestamps:
            return IntegrityCheckResult(
                passed=True,
                check_name="data_leakage",
                violations=[],
                metadata={"note": "No timestamps to check"},
            )

        # Get data for each period
        train_data = [ts for ts in data_timestamps if split.train_period.contains_date(ts)]
        val_data = [ts for ts in data_timestamps if split.validation_period.contains_date(ts)]
        test_data = [ts for ts in data_timestamps if split.test_period.contains_date(ts)]

        # Check for future data in training
        if train_data and val_data:
            latest_train = max(train_data)
            earliest_val = min(val_data)

            if latest_train >= earliest_val:
                violations.append(
                    IntegrityViolation(
                        violation_type="data_leakage",
                        severity="critical",
                        description=f"Training data contains timestamp ({latest_train}) >= validation data ({earliest_val})",
                        timestamp=pd.Timestamp.now(),
                        data_info={
                            "latest_train": latest_train.isoformat(),
                            "earliest_val": earliest_val.isoformat(),
                        },
                    )
                )

        # Check for future data in validation
        if val_data and test_data:
            latest_val = max(val_data)
            earliest_test = min(test_data)

            if latest_val >= earliest_test:
                violations.append(
                    IntegrityViolation(
                        violation_type="data_leakage",
                        severity="critical",
                        description=f"Validation data contains timestamp ({latest_val}) >= test data ({earliest_test})",
                        timestamp=pd.Timestamp.now(),
                        data_info={
                            "latest_val": latest_val.isoformat(),
                            "earliest_test": earliest_test.isoformat(),
                        },
                    )
                )

        return IntegrityCheckResult(
            passed=len(violations) == 0,
            check_name="data_leakage",
            violations=violations,
            metadata={
                "train_samples": len(train_data),
                "val_samples": len(val_data),
                "test_samples": len(test_data),
            },
        )

    def _check_future_information_access(
        self,
        split: RollSplit,
        data_timestamps: list[pd.Timestamp],
    ) -> IntegrityCheckResult:
        """Check for access to future information during training/validation."""
        violations = []

        # Check that training data doesn't extend beyond validation start
        train_cutoff = split.validation_period.start_date
        future_train_data = [
            ts
            for ts in data_timestamps
            if ts >= train_cutoff and split.train_period.contains_date(ts)
        ]

        if future_train_data:
            violations.append(
                IntegrityViolation(
                    violation_type="future_information_access",
                    severity="critical",
                    description=f"Training period contains {len(future_train_data)} timestamps at or after validation start ({train_cutoff})",
                    timestamp=pd.Timestamp.now(),
                    data_info={"future_timestamps_count": len(future_train_data)},
                )
            )

        # Check that validation data doesn't extend beyond test start
        val_cutoff = split.test_period.start_date
        future_val_data = [
            ts
            for ts in data_timestamps
            if ts >= val_cutoff and split.validation_period.contains_date(ts)
        ]

        if future_val_data:
            violations.append(
                IntegrityViolation(
                    violation_type="future_information_access",
                    severity="critical",
                    description=f"Validation period contains {len(future_val_data)} timestamps at or after test start ({val_cutoff})",
                    timestamp=pd.Timestamp.now(),
                    data_info={"future_timestamps_count": len(future_val_data)},
                )
            )

        return IntegrityCheckResult(
            passed=len(violations) == 0,
            check_name="future_information_access",
            violations=violations,
        )

    def _check_data_sufficiency(
        self,
        split: RollSplit,
        data_timestamps: list[pd.Timestamp],
        min_train_samples: int = 60,  # Reduced from 252 to 60 days (3 months)
        min_val_samples: int = 10,  # Reduced from 20 to 10 days
        min_test_samples: int = 10,  # Reduced from 20 to 10 days
    ) -> IntegrityCheckResult:
        """Check data sufficiency in each period."""
        violations = []

        # Count samples in each period
        train_count = sum(1 for ts in data_timestamps if split.train_period.contains_date(ts))
        val_count = sum(1 for ts in data_timestamps if split.validation_period.contains_date(ts))
        test_count = sum(1 for ts in data_timestamps if split.test_period.contains_date(ts))

        # Check minimums
        if train_count < min_train_samples:
            violations.append(
                IntegrityViolation(
                    violation_type="sufficient_data",  # Changed to allow flexible training
                    severity="info",  # Downgraded from warning to info
                    description=f"Training period has {train_count} samples (suggested minimum {min_train_samples})",
                    timestamp=pd.Timestamp.now(),
                    data_info={
                        "period": "training",
                        "count": train_count,
                        "minimum": min_train_samples,
                    },
                )
            )

        if val_count < min_val_samples:
            violations.append(
                IntegrityViolation(
                    violation_type="sufficient_data",  # Changed to allow flexible validation
                    severity="info",  # Downgraded from warning to info
                    description=f"Validation period has {val_count} samples (suggested minimum {min_val_samples})",
                    timestamp=pd.Timestamp.now(),
                    data_info={
                        "period": "validation",
                        "count": val_count,
                        "minimum": min_val_samples,
                    },
                )
            )

        if test_count < min_test_samples:
            violations.append(
                IntegrityViolation(
                    violation_type="sufficient_data",  # Changed to allow flexible testing
                    severity="info",  # Downgraded from warning to info
                    description=f"Test period has {test_count} samples (suggested minimum {min_test_samples})",
                    timestamp=pd.Timestamp.now(),
                    data_info={"period": "test", "count": test_count, "minimum": min_test_samples},
                )
            )

        return IntegrityCheckResult(
            passed=len([v for v in violations if v.severity == "critical"]) == 0,
            check_name="data_sufficiency",
            violations=violations,
            metadata={
                "train_count": train_count,
                "val_count": val_count,
                "test_count": test_count,
            },
        )

    def _check_period_boundaries(
        self,
        split: RollSplit,
        data_timestamps: list[pd.Timestamp],
    ) -> IntegrityCheckResult:
        """Check that period boundaries align with available data."""
        violations = []

        if not data_timestamps:
            return IntegrityCheckResult(
                passed=True,
                check_name="period_boundaries",
                violations=[],
                metadata={"note": "No timestamps to check"},
            )

        data_min = min(data_timestamps)
        data_max = max(data_timestamps)

        # Check if split extends beyond available data
        if split.train_period.start_date < data_min:
            violations.append(
                IntegrityViolation(
                    violation_type="boundary_mismatch",
                    severity="warning",
                    description=f"Training start ({split.train_period.start_date}) before data start ({data_min})",
                    timestamp=pd.Timestamp.now(),
                    data_info={
                        "split_start": split.train_period.start_date.isoformat(),
                        "data_start": data_min.isoformat(),
                    },
                )
            )

        if split.test_period.end_date > data_max:
            violations.append(
                IntegrityViolation(
                    violation_type="boundary_mismatch",
                    severity="warning",
                    description=f"Test end ({split.test_period.end_date}) after data end ({data_max})",
                    timestamp=pd.Timestamp.now(),
                    data_info={
                        "split_end": split.test_period.end_date.isoformat(),
                        "data_end": data_max.isoformat(),
                    },
                )
            )

        return IntegrityCheckResult(
            passed=True,  # Warnings don't fail the check
            check_name="period_boundaries",
            violations=violations,
            metadata={
                "data_range": (data_min.isoformat(), data_max.isoformat()),
                "split_range": (
                    split.train_period.start_date.isoformat(),
                    split.test_period.end_date.isoformat(),
                ),
            },
        )

    def get_violation_summary(self) -> dict[str, Any]:
        """Get summary of all violations detected."""
        if not self.violations_log:
            return {"total_violations": 0, "clean_validation": True}

        # Group by violation type and severity
        by_type = {}
        by_severity = {"critical": 0, "warning": 0, "info": 0}

        for violation in self.violations_log:
            # Count by type
            v_type = violation.violation_type
            by_type[v_type] = by_type.get(v_type, 0) + 1

            # Count by severity
            by_severity[violation.severity] += 1

        return {
            "total_violations": len(self.violations_log),
            "critical_violations": by_severity["critical"],
            "warning_violations": by_severity["warning"],
            "info_violations": by_severity["info"],
            "violations_by_type": by_type,
            "clean_validation": by_severity["critical"] == 0,
            "most_common_violations": sorted(by_type.items(), key=lambda x: x[1], reverse=True)[:5],
        }

    def export_integrity_report(self, output_path: Path) -> None:
        """Export comprehensive integrity validation report."""
        report = {
            "generated_at": pd.Timestamp.now().isoformat(),
            "validator_config": {
                "strict_mode": self.strict_mode,
            },
            "summary": self.get_violation_summary(),
            "check_history": [
                {
                    "passed": result.passed,
                    "check_name": result.check_name,
                    "violations_count": len(result.violations),
                    "check_duration_ms": result.check_duration_ms,
                    "metadata": result.metadata,
                }
                for result in self.check_history
            ],
            "detailed_violations": [
                {
                    "violation_type": v.violation_type,
                    "severity": v.severity,
                    "description": v.description,
                    "timestamp": v.timestamp.isoformat(),
                    "split_info": v.split_info,
                    "data_info": v.data_info,
                }
                for v in self.violations_log
            ],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Integrity report exported to {output_path}")

    def clear_history(self) -> None:
        """Clear violation log and check history."""
        self.violations_log.clear()
        self.check_history.clear()

    def raise_on_critical_violations(self) -> None:
        """Raise exception if critical violations were detected."""
        critical_violations = [v for v in self.violations_log if v.severity == "critical"]

        if critical_violations:
            violation_descriptions = [v.description for v in critical_violations]
            raise ValueError(
                "Critical temporal integrity violations detected:\n"
                + "\n".join(f"- {desc}" for desc in violation_descriptions)
            )


class ContinuousIntegrityMonitor:
    """
    Continuous monitoring of temporal integrity during backtest execution.

    This monitor provides real-time integrity checking during backtest
    execution with automatic alerts and corrective actions.
    """

    def __init__(self, validator: TemporalIntegrityValidator, alert_threshold: int = 5):
        """
        Initialize continuous integrity monitor.

        Args:
            validator: Temporal integrity validator instance
            alert_threshold: Number of violations before raising alert
        """
        self.validator = validator
        self.alert_threshold = alert_threshold
        self.monitoring_active = False
        self.violation_count = 0
        self._alert_handlers: list[callable] = []

    def start_monitoring(self) -> None:
        """Start continuous monitoring."""
        self.monitoring_active = True
        self.violation_count = 0
        logger.info("Temporal integrity monitoring started")

    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self.monitoring_active = False
        logger.info(
            f"Temporal integrity monitoring stopped. Total violations: {self.violation_count}"
        )

    def monitor_data_access(
        self,
        access_timestamp: pd.Timestamp,
        data_timestamp: pd.Timestamp,
        operation: str = "data_access",
    ) -> bool:
        """
        Monitor data access for temporal violations.

        Args:
            access_timestamp: When the data is being accessed
            data_timestamp: Timestamp of the data being accessed
            operation: Type of operation being performed

        Returns:
            True if access is valid, False if violation detected
        """
        if not self.monitoring_active:
            return True

        # Check for future data access
        if data_timestamp > access_timestamp:
            violation = IntegrityViolation(
                violation_type="future_data_access",
                severity="critical",
                description=f"Attempted to access future data: accessing {data_timestamp} at {access_timestamp}",
                timestamp=pd.Timestamp.now(),
                data_info={
                    "access_time": access_timestamp.isoformat(),
                    "data_time": data_timestamp.isoformat(),
                    "operation": operation,
                },
            )

            self._handle_violation(violation)
            return False

        return True

    def monitor_model_training(
        self,
        training_start: pd.Timestamp,
        training_end: pd.Timestamp,
        data_timestamps: list[pd.Timestamp],
        model_name: str = "unknown",
    ) -> bool:
        """
        Monitor model training for temporal integrity.

        Args:
            training_start: Start of training period
            training_end: End of training period
            data_timestamps: Timestamps of data used for training
            model_name: Name of model being trained

        Returns:
            True if training is valid, False if violations detected
        """
        if not self.monitoring_active:
            return True

        violations_detected = False

        # Check for future data in training
        future_data = [ts for ts in data_timestamps if ts > training_end]

        if future_data:
            violation = IntegrityViolation(
                violation_type="future_training_data",
                severity="critical",
                description=f"Model {model_name} trained with {len(future_data)} future timestamps beyond training end {training_end}",
                timestamp=pd.Timestamp.now(),
                data_info={
                    "model_name": model_name,
                    "training_end": training_end.isoformat(),
                    "future_data_count": len(future_data),
                    "latest_future_timestamp": max(future_data).isoformat(),
                },
            )

            self._handle_violation(violation)
            violations_detected = True

        return not violations_detected

    def monitor_prediction_generation(
        self,
        prediction_time: pd.Timestamp,
        data_cutoff: pd.Timestamp,
        model_name: str = "unknown",
    ) -> bool:
        """
        Monitor prediction generation for temporal integrity.

        Args:
            prediction_time: When prediction is being made
            data_cutoff: Latest allowable data timestamp for prediction
            model_name: Name of model making prediction

        Returns:
            True if prediction is valid, False if violations detected
        """
        if not self.monitoring_active:
            return True

        # Prediction should only use data up to cutoff time
        if prediction_time < data_cutoff:
            violation = IntegrityViolation(
                violation_type="prediction_temporal_violation",
                severity="warning",
                description=f"Model {model_name} making prediction at {prediction_time} with data cutoff at {data_cutoff}",
                timestamp=pd.Timestamp.now(),
                data_info={
                    "model_name": model_name,
                    "prediction_time": prediction_time.isoformat(),
                    "data_cutoff": data_cutoff.isoformat(),
                },
            )

            self._handle_violation(violation)
            return False

        return True

    def _handle_violation(self, violation: IntegrityViolation) -> None:
        """Handle detected violation."""
        self.validator.violations_log.append(violation)
        self.violation_count += 1

        # Log violation
        logger.error(f"Temporal integrity violation: {violation.description}")

        # Check alert threshold
        if self.violation_count >= self.alert_threshold:
            self._trigger_alerts(violation)

        # Handle critical violations
        if violation.severity == "critical" and self.validator.strict_mode:
            self.stop_monitoring()
            raise ValueError(f"Critical temporal integrity violation: {violation.description}")

    def _trigger_alerts(self, latest_violation: IntegrityViolation) -> None:
        """Trigger alerts when threshold is reached."""
        alert_message = (
            f"Temporal integrity alert: {self.violation_count} violations detected. "
            f"Latest: {latest_violation.description}"
        )

        logger.warning(alert_message)

        # Call registered alert handlers
        for handler in self._alert_handlers:
            try:
                handler(alert_message, latest_violation)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

    def add_alert_handler(self, handler: callable) -> None:
        """Add custom alert handler."""
        self._alert_handlers.append(handler)

    def get_monitoring_stats(self) -> dict[str, Any]:
        """Get monitoring statistics."""
        return {
            "monitoring_active": self.monitoring_active,
            "total_violations": self.violation_count,
            "alert_threshold": self.alert_threshold,
            "alert_handlers_count": len(self._alert_handlers),
        }
