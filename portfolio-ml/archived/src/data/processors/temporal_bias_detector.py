"""
Automated look-ahead bias detection system for data pipeline.

This module provides comprehensive automated detection of temporal integrity
violations specifically during data collection, processing, and storage phases.
Integrates with the existing temporal integrity system to provide early detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.validation.temporal_integrity import (
    ContinuousIntegrityMonitor,
    IntegrityViolation,
    TemporalIntegrityValidator,
)

logger = logging.getLogger(__name__)


@dataclass
class BiasDetectionResult:
    """Result of look-ahead bias detection."""

    clean: bool
    violations_detected: int
    critical_violations: int
    pipeline_stage: str
    detection_timestamp: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    details: dict[str, Any] = field(default_factory=dict)


class DataPipelineBiasDetector:
    """
    Automated look-ahead bias detection for data pipeline operations.

    This detector monitors data collection, gap-filling, and universe construction
    processes to ensure temporal integrity is maintained throughout the pipeline.
    """

    def __init__(
        self,
        strict_mode: bool = True,
        auto_report: bool = True,
        report_dir: str = "data/quality_reports",
    ):
        """
        Initialize bias detector.

        Args:
            strict_mode: If True, raise exceptions on critical violations
            auto_report: If True, automatically generate violation reports
            report_dir: Directory for quality reports
        """
        self.strict_mode = strict_mode
        self.auto_report = auto_report
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # Initialize temporal integrity components
        self.validator = TemporalIntegrityValidator(strict_mode=strict_mode)
        self.monitor = ContinuousIntegrityMonitor(self.validator, alert_threshold=3)

        # Detection history
        self.detection_history: list[BiasDetectionResult] = []

    def validate_data_collection_integrity(
        self,
        collected_data: pd.DataFrame,
        collection_date: pd.Timestamp,
        universe_calendar: pd.DataFrame = None,
        source_name: str = "unknown",
    ) -> BiasDetectionResult:
        """
        Validate temporal integrity of collected market data.

        Args:
            collected_data: DataFrame with market data (must have datetime index)
            collection_date: Date when data collection was performed
            universe_calendar: Optional universe membership calendar
            source_name: Name of data source (Stooq, Yahoo Finance, etc.)

        Returns:
            Bias detection result with violations summary
        """
        violations = []
        pipeline_stage = f"data_collection_{source_name}"

        # Validate data timestamps don't exceed collection date
        if hasattr(collected_data, "index") and isinstance(collected_data.index, pd.DatetimeIndex):
            future_data = collected_data.index[collected_data.index > collection_date]

            if len(future_data) > 0:
                violation = IntegrityViolation(
                    violation_type="future_data_collection",
                    severity="critical",
                    description=f"Collected {len(future_data)} data points with timestamps after collection date {collection_date}",
                    timestamp=pd.Timestamp.now(),
                    data_info={
                        "source": source_name,
                        "collection_date": collection_date.isoformat(),
                        "future_data_count": len(future_data),
                        "latest_future_timestamp": future_data.max().isoformat(),
                    },
                )
                violations.append(violation)

        # Validate universe membership alignment if provided
        if universe_calendar is not None:
            universe_violations = self._check_universe_temporal_alignment(
                collected_data, universe_calendar, collection_date
            )
            violations.extend(universe_violations)

        # Log violations with monitor
        for violation in violations:
            self.validator.violations_log.append(violation)

        # Create result
        critical_count = sum(1 for v in violations if v.severity == "critical")
        result = BiasDetectionResult(
            clean=critical_count == 0,
            violations_detected=len(violations),
            critical_violations=critical_count,
            pipeline_stage=pipeline_stage,
            details={
                "source_name": source_name,
                "collection_date": collection_date.isoformat(),
                "data_shape": collected_data.shape if hasattr(collected_data, "shape") else "unknown",
                "violations": [
                    {"type": v.violation_type, "severity": v.severity, "description": v.description}
                    for v in violations
                ],
            },
        )

        self.detection_history.append(result)

        if self.auto_report and violations:
            self._generate_violation_report(result, violations)

        if critical_count > 0 and self.strict_mode:
            raise ValueError(
                f"Critical temporal integrity violations in {pipeline_stage}: {critical_count} violations"
            )

        return result

    def validate_gap_filling_integrity(
        self,
        original_data: pd.DataFrame,
        filled_data: pd.DataFrame,
        fill_date: pd.Timestamp,
        fill_method: str = "forward_fill",
    ) -> BiasDetectionResult:
        """
        Validate temporal integrity of gap-filling operations.

        Args:
            original_data: Original data with gaps
            filled_data: Data after gap filling
            fill_date: Date when gap filling was performed
            fill_method: Method used for gap filling

        Returns:
            Bias detection result with violations summary
        """
        violations = []
        pipeline_stage = f"gap_filling_{fill_method}"

        # Check for future information in gap filling
        if hasattr(filled_data, "index") and isinstance(filled_data.index, pd.DatetimeIndex):
            # Identify filled gaps
            original_mask = pd.isna(original_data) if hasattr(original_data, "isna") else pd.Series([])
            filled_mask = pd.notna(filled_data) if hasattr(filled_data, "notna") else pd.Series([])

            if len(original_mask) > 0 and len(filled_mask) > 0:
                gap_filled_mask = original_mask & filled_mask

                # Check if gap filling used future information
                for timestamp in filled_data.index[gap_filled_mask]:
                    if timestamp > fill_date:
                        violation = IntegrityViolation(
                            violation_type="future_information_gap_filling",
                            severity="critical",
                            description=f"Gap filling at {timestamp} performed after fill date {fill_date}",
                            timestamp=pd.Timestamp.now(),
                            data_info={
                                "fill_method": fill_method,
                                "fill_date": fill_date.isoformat(),
                                "gap_timestamp": timestamp.isoformat(),
                            },
                        )
                        violations.append(violation)

        # Log violations
        for violation in violations:
            self.validator.violations_log.append(violation)

        # Create result
        critical_count = sum(1 for v in violations if v.severity == "critical")
        result = BiasDetectionResult(
            clean=critical_count == 0,
            violations_detected=len(violations),
            critical_violations=critical_count,
            pipeline_stage=pipeline_stage,
            details={
                "fill_method": fill_method,
                "fill_date": fill_date.isoformat(),
                "original_shape": getattr(original_data, "shape", "unknown"),
                "filled_shape": getattr(filled_data, "shape", "unknown"),
            },
        )

        self.detection_history.append(result)
        return result

    def validate_universe_construction_integrity(
        self,
        membership_data: pd.DataFrame,
        construction_date: pd.Timestamp,
        universe_type: str = "midcap400",
    ) -> BiasDetectionResult:
        """
        Validate temporal integrity of universe construction.

        Args:
            membership_data: DataFrame with universe membership intervals
            construction_date: Date when universe was constructed
            universe_type: Type of universe being constructed

        Returns:
            Bias detection result with violations summary
        """
        violations = []
        pipeline_stage = f"universe_construction_{universe_type}"

        # Check for future membership information
        if "start" in membership_data.columns:
            future_starts = membership_data["start"] > construction_date
            if future_starts.any():
                violation = IntegrityViolation(
                    violation_type="future_membership_information",
                    severity="critical",
                    description=f"Universe construction used {future_starts.sum()} future membership start dates",
                    timestamp=pd.Timestamp.now(),
                    data_info={
                        "universe_type": universe_type,
                        "construction_date": construction_date.isoformat(),
                        "future_starts_count": future_starts.sum(),
                    },
                )
                violations.append(violation)

        # Validate membership transition logic
        transition_violations = self._validate_membership_transitions(membership_data, construction_date)
        violations.extend(transition_violations)

        # Log violations
        for violation in violations:
            self.validator.violations_log.append(violation)

        # Create result
        critical_count = sum(1 for v in violations if v.severity == "critical")
        result = BiasDetectionResult(
            clean=critical_count == 0,
            violations_detected=len(violations),
            critical_violations=critical_count,
            pipeline_stage=pipeline_stage,
            details={
                "universe_type": universe_type,
                "construction_date": construction_date.isoformat(),
                "membership_shape": getattr(membership_data, "shape", "unknown"),
            },
        )

        self.detection_history.append(result)
        return result

    def validate_parquet_generation_integrity(
        self,
        source_data: pd.DataFrame,
        generated_parquet_path: Path,
        generation_date: pd.Timestamp,
        data_type: str = "prices",
    ) -> BiasDetectionResult:
        """
        Validate temporal integrity of parquet dataset generation.

        Args:
            source_data: Source data used for parquet generation
            generated_parquet_path: Path to generated parquet file
            generation_date: Date when parquet was generated
            data_type: Type of data (prices, returns, volume)

        Returns:
            Bias detection result with violations summary
        """
        violations = []
        pipeline_stage = f"parquet_generation_{data_type}"

        # Read generated parquet for validation
        if generated_parquet_path.exists():
            try:
                generated_data = pd.read_parquet(generated_parquet_path)

                # Check for future timestamps in generated data
                if hasattr(generated_data, "index") and isinstance(
                    generated_data.index, pd.DatetimeIndex
                ):
                    future_timestamps = generated_data.index[generated_data.index > generation_date]

                    if len(future_timestamps) > 0:
                        violation = IntegrityViolation(
                            violation_type="future_timestamps_in_parquet",
                            severity="critical",
                            description=f"Generated parquet contains {len(future_timestamps)} future timestamps",
                            timestamp=pd.Timestamp.now(),
                            data_info={
                                "data_type": data_type,
                                "generation_date": generation_date.isoformat(),
                                "future_timestamps_count": len(future_timestamps),
                                "parquet_path": str(generated_parquet_path),
                            },
                        )
                        violations.append(violation)

            except Exception as e:
                violation = IntegrityViolation(
                    violation_type="parquet_validation_error",
                    severity="warning",
                    description=f"Could not validate generated parquet: {str(e)}",
                    timestamp=pd.Timestamp.now(),
                    data_info={"parquet_path": str(generated_parquet_path), "error": str(e)},
                )
                violations.append(violation)

        # Log violations
        for violation in violations:
            self.validator.violations_log.append(violation)

        # Create result
        critical_count = sum(1 for v in violations if v.severity == "critical")
        result = BiasDetectionResult(
            clean=critical_count == 0,
            violations_detected=len(violations),
            critical_violations=critical_count,
            pipeline_stage=pipeline_stage,
            details={
                "data_type": data_type,
                "generation_date": generation_date.isoformat(),
                "parquet_path": str(generated_parquet_path),
                "source_shape": getattr(source_data, "shape", "unknown"),
            },
        )

        self.detection_history.append(result)
        return result

    def _check_universe_temporal_alignment(
        self,
        market_data: pd.DataFrame,
        universe_calendar: pd.DataFrame,
        reference_date: pd.Timestamp,
    ) -> list[IntegrityViolation]:
        """Check temporal alignment between market data and universe membership."""
        violations = []

        if "start" in universe_calendar.columns and "end" in universe_calendar.columns:
            # Check for universe membership changes that occur after reference date
            future_changes = (
                (universe_calendar["start"] > reference_date)
                | (universe_calendar["end"] > reference_date)
            )

            if future_changes.any():
                violation = IntegrityViolation(
                    violation_type="universe_future_information",
                    severity="warning",
                    description=f"Universe calendar contains {future_changes.sum()} changes after reference date",
                    timestamp=pd.Timestamp.now(),
                    data_info={
                        "reference_date": reference_date.isoformat(),
                        "future_changes_count": future_changes.sum(),
                    },
                )
                violations.append(violation)

        return violations

    def _validate_membership_transitions(
        self, membership_data: pd.DataFrame, construction_date: pd.Timestamp
    ) -> list[IntegrityViolation]:
        """Validate logical consistency of membership transitions."""
        violations = []

        if "start" in membership_data.columns and "end" in membership_data.columns:
            # Check for overlapping membership periods for same ticker
            if "ticker" in membership_data.columns:
                for ticker in membership_data["ticker"].unique():
                    ticker_data = membership_data[membership_data["ticker"] == ticker].sort_values("start")

                    for i in range(len(ticker_data) - 1):
                        current_end = ticker_data.iloc[i]["end"]
                        next_start = ticker_data.iloc[i + 1]["start"]

                        if pd.notna(current_end) and current_end > next_start:
                            violation = IntegrityViolation(
                                violation_type="membership_overlap",
                                severity="warning",
                                description=f"Ticker {ticker} has overlapping membership periods",
                                timestamp=pd.Timestamp.now(),
                                data_info={
                                    "ticker": ticker,
                                    "current_end": current_end.isoformat(),
                                    "next_start": next_start.isoformat(),
                                },
                            )
                            violations.append(violation)

        return violations

    def _generate_violation_report(
        self, result: BiasDetectionResult, violations: list[IntegrityViolation]
    ) -> None:
        """Generate automated violation report."""
        report_path = (
            self.report_dir / f"bias_violations_{result.pipeline_stage}_{result.detection_timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        )

        report_data = {
            "detection_result": {
                "clean": result.clean,
                "violations_detected": result.violations_detected,
                "critical_violations": result.critical_violations,
                "pipeline_stage": result.pipeline_stage,
                "detection_timestamp": result.detection_timestamp.isoformat(),
                "details": result.details,
            },
            "violations": [
                {
                    "type": v.violation_type,
                    "severity": v.severity,
                    "description": v.description,
                    "timestamp": v.timestamp.isoformat(),
                    "split_info": v.split_info,
                    "data_info": v.data_info,
                }
                for v in violations
            ],
        }

        import json

        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        logger.info(f"Bias violation report generated: {report_path}")

    def get_pipeline_integrity_summary(self) -> dict[str, Any]:
        """Get comprehensive summary of pipeline integrity status."""
        if not self.detection_history:
            return {"status": "no_checks_performed", "total_checks": 0}

        total_checks = len(self.detection_history)
        clean_checks = sum(1 for r in self.detection_history if r.clean)
        total_violations = sum(r.violations_detected for r in self.detection_history)
        critical_violations = sum(r.critical_violations for r in self.detection_history)

        # Group by pipeline stage
        by_stage = {}
        for result in self.detection_history:
            stage = result.pipeline_stage
            if stage not in by_stage:
                by_stage[stage] = {"checks": 0, "violations": 0, "critical": 0}

            by_stage[stage]["checks"] += 1
            by_stage[stage]["violations"] += result.violations_detected
            by_stage[stage]["critical"] += result.critical_violations

        return {
            "status": "clean" if critical_violations == 0 else "violations_detected",
            "total_checks": total_checks,
            "clean_checks": clean_checks,
            "total_violations": total_violations,
            "critical_violations": critical_violations,
            "success_rate": clean_checks / total_checks if total_checks > 0 else 0,
            "by_pipeline_stage": by_stage,
            "latest_check": self.detection_history[-1].detection_timestamp.isoformat(),
        }

    def export_comprehensive_report(self, output_path: Path) -> None:
        """Export comprehensive bias detection report."""
        summary = self.get_pipeline_integrity_summary()
        validator_summary = self.validator.get_violation_summary()

        comprehensive_report = {
            "generated_at": pd.Timestamp.now().isoformat(),
            "detector_config": {
                "strict_mode": self.strict_mode,
                "auto_report": self.auto_report,
                "report_dir": str(self.report_dir),
            },
            "pipeline_summary": summary,
            "validator_summary": validator_summary,
            "detection_history": [
                {
                    "clean": r.clean,
                    "violations_detected": r.violations_detected,
                    "critical_violations": r.critical_violations,
                    "pipeline_stage": r.pipeline_stage,
                    "detection_timestamp": r.detection_timestamp.isoformat(),
                    "details": r.details,
                }
                for r in self.detection_history
            ],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        with open(output_path, "w") as f:
            json.dump(comprehensive_report, f, indent=2, default=str)

        logger.info(f"Comprehensive bias detection report exported: {output_path}")

    def clear_history(self) -> None:
        """Clear detection history and validator history."""
        self.detection_history.clear()
        self.validator.clear_history()
