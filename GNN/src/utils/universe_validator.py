"""
Universe Compatibility Validator.

This module provides centralised validation of universe compatibility across
training and inference phases to prevent dimension mismatches and ensure
consistent model behaviour.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class UniverseValidationResult:
    """Results of universe validation."""
    is_valid: bool
    training_universe: list[str]
    inference_universe: list[str]
    compatible_assets: list[str]
    missing_from_training: list[str]
    missing_from_inference: list[str]
    recommendations: list[str]
    severity: str  # 'info', 'warning', 'error'


class UniverseCompatibilityValidator:
    """
    Centralized validator for universe compatibility across training and inference.

    Ensures that models trained on one universe can properly handle inference
    requests for different universes.
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize the validator.

        Args:
            strict_mode: If True, requires exact universe matches
        """
        self.strict_mode = strict_mode
        self.validation_history: list[UniverseValidationResult] = []

    def validate_training_universe(
        self,
        universe: list[str],
        returns_data: pd.DataFrame,
        min_coverage: float = 0.8
    ) -> UniverseValidationResult:
        """
        Validate a training universe against available data.

        Args:
            universe: Proposed training universe
            returns_data: Available returns data
            min_coverage: Minimum data coverage required

        Returns:
            Validation result with recommendations
        """
        available_assets = list(returns_data.columns)
        compatible_assets = [asset for asset in universe if asset in available_assets]
        missing_assets = [asset for asset in universe if asset not in available_assets]

        coverage_ratio = len(compatible_assets) / len(universe) if universe else 0

        recommendations = []
        severity = "info"

        if coverage_ratio < min_coverage:
            severity = "error"
            recommendations.append(
                f"Universe coverage {coverage_ratio:.1%} below minimum {min_coverage:.1%}"
            )
            recommendations.append(
                f"Consider removing assets: {missing_assets[:5]}{'...' if len(missing_assets) > 5 else ''}"
            )
        elif missing_assets:
            severity = "warning"
            recommendations.append(
                f"{len(missing_assets)} assets missing from returns data"
            )

        if len(compatible_assets) < 50:
            severity = "warning"
            recommendations.append(
                f"Small universe size ({len(compatible_assets)}) may impact ML model performance"
            )
        elif len(compatible_assets) > 600:
            recommendations.append(
                f"Large universe size ({len(compatible_assets)}) - ensure sufficient GPU memory"
            )

        result = UniverseValidationResult(
            is_valid=coverage_ratio >= min_coverage,
            training_universe=universe,
            inference_universe=universe,  # Same for training validation
            compatible_assets=compatible_assets,
            missing_from_training=[],  # N/A for training validation
            missing_from_inference=missing_assets,
            recommendations=recommendations,
            severity=severity
        )

        self.validation_history.append(result)
        return result

    def validate_inference_compatibility(
        self,
        training_universe: list[str],
        inference_universe: list[str],
        model_type: str = "unknown"
    ) -> UniverseValidationResult:
        """
        Validate compatibility between training and inference universes.

        Args:
            training_universe: Universe used during model training
            inference_universe: Universe requested for inference
            model_type: Type of model for specific recommendations

        Returns:
            Validation result with handling recommendations
        """
        training_set = set(training_universe)
        inference_set = set(inference_universe)

        compatible_assets = list(training_set & inference_set)
        missing_from_training = list(inference_set - training_set)
        missing_from_inference = list(training_set - inference_set)

        coverage_ratio = len(compatible_assets) / len(inference_universe) if inference_universe else 0

        recommendations = []
        severity = "info"

        if self.strict_mode and training_universe != inference_universe:
            severity = "error"
            recommendations.append("Strict mode: inference universe must exactly match training universe")
        elif coverage_ratio < 0.5:
            severity = "error"
            recommendations.append(
                f"Poor universe compatibility {coverage_ratio:.1%} - model may perform poorly"
            )
        elif coverage_ratio < 0.8:
            severity = "warning"
            recommendations.append(
                f"Limited universe compatibility {coverage_ratio:.1%}"
            )

        # Model-specific recommendations
        if model_type.lower() == "lstm":
            if missing_from_training:
                recommendations.append(
                    f"LSTM: {len(missing_from_training)} new assets will receive zero weights"
                )
            if len(compatible_assets) != len(training_universe):
                recommendations.append(
                    "LSTM: Consider retraining with updated universe for optimal performance"
                )
        elif model_type.lower() == "gat":
            if missing_from_training:
                recommendations.append(
                    f"GAT: {len(missing_from_training)} new assets need feature imputation"
                )
        elif model_type.lower() == "hrp":
            if len(compatible_assets) < 20:
                recommendations.append(
                    f"HRP: Small compatible universe ({len(compatible_assets)}) may impact clustering"
                )

        # Asset expansion/contraction recommendations
        if missing_from_training:
            recommendations.append(
                f"New assets requiring handling: {missing_from_training[:3]}{'...' if len(missing_from_training) > 3 else ''}"
            )
        if missing_from_inference:
            recommendations.append(
                f"Trained assets not in inference: {missing_from_inference[:3]}{'...' if len(missing_from_inference) > 3 else ''}"
            )

        result = UniverseValidationResult(
            is_valid=coverage_ratio >= 0.5 and (not self.strict_mode or training_universe == inference_universe),
            training_universe=training_universe,
            inference_universe=inference_universe,
            compatible_assets=compatible_assets,
            missing_from_training=missing_from_training,
            missing_from_inference=missing_from_inference,
            recommendations=recommendations,
            severity=severity
        )

        self.validation_history.append(result)
        return result

    def validate_model_dimensions(
        self,
        model_config: dict[str, Any],
        actual_universe_size: int,
        model_type: str
    ) -> UniverseValidationResult:
        """
        Validate model configuration against actual universe dimensions.

        Args:
            model_config: Model configuration dictionary
            actual_universe_size: Actual number of assets in universe
            model_type: Type of model for validation rules

        Returns:
            Validation result with dimension recommendations
        """
        recommendations = []
        severity = "info"
        is_valid = True

        if model_type.lower() == "lstm":
            input_size = model_config.get("input_size", model_config.get("max_input_size", 500))

            if actual_universe_size > input_size:
                severity = "error"
                is_valid = False
                recommendations.append(
                    f"CRITICAL: Universe size {actual_universe_size} > LSTM input size {input_size} - this will truncate assets!"
                )
                recommendations.append(
                    f"Increase max_input_size to {actual_universe_size + 50} or implement proper asset selection logic"
                )
            elif actual_universe_size < input_size * 0.5:
                severity = "warning"
                recommendations.append(
                    f"Universe size {actual_universe_size} much smaller than LSTM capacity {input_size}"
                )
                recommendations.append(
                    "Consider reducing model size or expanding universe"
                )
        elif model_type.lower() == "gat":
            expected_features = model_config.get("input_features", 10)
            # GAT dimension validation would need feature matrix inspection

        # Memory validation
        max_memory_gb = model_config.get("max_memory_gb", 11.0)
        estimated_memory = self._estimate_memory_usage(actual_universe_size, model_type, model_config)

        if estimated_memory > max_memory_gb:
            severity = "error"
            is_valid = False
            recommendations.append(
                f"Estimated memory {estimated_memory:.1f}GB > limit {max_memory_gb:.1f}GB"
            )
            recommendations.append(
                "Reduce universe size or batch size"
            )

        result = UniverseValidationResult(
            is_valid=is_valid,
            training_universe=[],
            inference_universe=[],
            compatible_assets=[],
            missing_from_training=[],
            missing_from_inference=[],
            recommendations=recommendations,
            severity=severity
        )

        return result

    def _estimate_memory_usage(
        self,
        universe_size: int,
        model_type: str,
        config: dict[str, Any]
    ) -> float:
        """Estimate memory usage for given model configuration."""
        batch_size = config.get("batch_size", 32)

        if model_type.lower() == "lstm":
            sequence_length = config.get("sequence_length", 60)
            hidden_size = config.get("hidden_size", 128)

            # Rough estimation: batch_size * sequence_length * universe_size * 4 bytes * safety_factor
            memory_gb = (batch_size * sequence_length * universe_size * 4 * 3) / (1024**3)
            return memory_gb
        elif model_type.lower() == "gat":
            # GAT memory usage depends on graph structure
            avg_edges_per_node = config.get("knn_k", 8)
            memory_gb = (universe_size * avg_edges_per_node * 4 * 2) / (1024**3)
            return memory_gb
        else:
            # Conservative estimate for unknown models
            return 1.0

    def log_validation_summary(self, result: UniverseValidationResult, context: str = "") -> None:
        """Log validation results with appropriate severity."""
        prefix = f"{context}: " if context else ""

        if result.severity == "error":
            logger.error(f"{prefix}Universe validation failed")
            for rec in result.recommendations:
                logger.error(f"  - {rec}")
        elif result.severity == "warning":
            logger.warning(f"{prefix}Universe validation issues detected")
            for rec in result.recommendations:
                logger.warning(f"  - {rec}")
        else:
            logger.info(f"{prefix}Universe validation passed")
            if result.recommendations:
                for rec in result.recommendations:
                    logger.info(f"  - {rec}")

    def get_validation_report(self) -> dict[str, Any]:
        """Generate comprehensive validation report."""
        if not self.validation_history:
            return {"status": "no_validations_performed"}

        errors = [r for r in self.validation_history if r.severity == "error"]
        warnings = [r for r in self.validation_history if r.severity == "warning"]

        return {
            "total_validations": len(self.validation_history),
            "errors": len(errors),
            "warnings": len(warnings),
            "success_rate": (len(self.validation_history) - len(errors)) / len(self.validation_history),
            "recent_issues": [r.recommendations for r in self.validation_history[-5:] if r.recommendations],
            "strict_mode": self.strict_mode
        }