"""
Enhanced Model Metadata Generation for Training-Backtest Compatibility.

This module provides comprehensive metadata generation utilities that ensure
models contain all necessary information for seamless backtesting integration.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class MetadataGenerator:
    """Enhanced metadata generator for model training compatibility."""

    def __init__(self, config_path: str | Path | None = None):
        """
        Initialise metadata generator.

        Args:
            config_path: Path to shared configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self.config = self._load_config() if self.config_path else {}

    def _load_config(self) -> dict[str, Any]:
        """Load shared configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {self.config_path}: {e}")
            return {}

    def generate_enhanced_metadata(
        self,
        model_type: str,
        model_config: dict[str, Any],
        trained_universe: list[str],
        training_period: dict[str, str],
        training_data: pd.DataFrame | None = None,
        training_history: dict[str, list] | None = None,
        custom_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Generate comprehensive metadata for model compatibility.

        Args:
            model_type: Type of model (LSTM, GAT, HRP)
            model_config: Model-specific configuration
            trained_universe: List of assets model was trained on
            training_period: Start and end dates of training
            training_data: Original training data for hashing
            training_history: Training metrics history
            custom_fields: Additional model-specific metadata

        Returns:
            Complete metadata dictionary
        """
        metadata = {
            # Core model information
            "model_type": model_type,
            "model_version": self._generate_version_hash(model_config, training_period),
            "trained_universe": sorted(trained_universe),  # Ensure consistent ordering
            "universe_size": len(trained_universe),
            "training_timestamp": datetime.now().isoformat(),

            # Training configuration
            "training_period": training_period,
            "model_config": model_config,
            "constraints": self.config.get("portfolio_constraints", {}),

            # Data integrity
            "training_data_hash": self._hash_training_data(training_data) if training_data is not None else None,
            "model_config_hash": self._hash_dict(model_config),
            "universe_hash": self._hash_list(trained_universe),

            # Training quality metrics
            "training_history": training_history or {},
            "training_quality": self._assess_training_quality(training_history) if training_history else {},

            # Compatibility information
            "compatibility": {
                "backtest_compatible": True,
                "dimension_requirements": self._get_dimension_requirements(model_type, model_config),
                "universe_requirements": self._get_universe_requirements(model_type),
                "preprocessing_requirements": self._get_preprocessing_requirements(),
            },

            # Additional fields
            **(custom_fields or {}),
        }

        # Add model-specific metadata
        if model_type == "LSTM":
            metadata["network_dimensions"] = self._get_lstm_dimensions(model_config)
        elif model_type == "GAT":
            metadata["graph_config"] = self._get_gat_config(model_config)
        elif model_type == "HRP":
            metadata["hierarchical_config"] = self._get_hrp_config(model_config)

        return metadata

    def _generate_version_hash(self, model_config: dict[str, Any], training_period: dict[str, str]) -> str:
        """Generate version hash based on configuration and training period."""
        version_data = {
            "config": model_config,
            "period": training_period,
            "timestamp": datetime.now().strftime("%Y%m%d"),
        }
        return hashlib.sha256(json.dumps(version_data, sort_keys=True).encode()).hexdigest()[:12]

    def _hash_training_data(self, data: pd.DataFrame) -> str:
        """Generate hash of training data for integrity checking."""
        try:
            # Use data shape, column names, and sample of values
            hash_data = {
                "shape": data.shape,
                "columns": sorted(data.columns.tolist()),
                "dtypes": data.dtypes.to_dict(),
                "sample_hash": hashlib.md5(data.head(100).to_string().encode()).hexdigest(),
            }
            return hashlib.sha256(json.dumps(hash_data, sort_keys=True).encode()).hexdigest()[:16]
        except Exception as e:
            logger.warning(f"Failed to hash training data: {e}")
            return "hash_failed"

    def _hash_dict(self, data: dict[str, Any]) -> str:
        """Generate hash of dictionary data."""
        return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]

    def _hash_list(self, data: list[str]) -> str:
        """Generate hash of list data."""
        return hashlib.md5(json.dumps(sorted(data)).encode()).hexdigest()[:12]

    def _assess_training_quality(self, training_history: dict[str, list]) -> dict[str, Any]:
        """Assess training quality from history metrics."""
        quality = {}

        if "val_loss" in training_history and training_history["val_loss"]:
            val_losses = training_history["val_loss"]
            quality["final_val_loss"] = val_losses[-1]
            quality["best_val_loss"] = min(val_losses)
            quality["training_epochs"] = len(val_losses)

            # Check for convergence
            if len(val_losses) > 10:
                recent_variance = np.var(val_losses[-10:])
                quality["convergence_variance"] = float(recent_variance)
                quality["converged"] = recent_variance < 1e-6

            # Check for overfitting
            if "train_loss" in training_history and len(training_history["train_loss"]) == len(val_losses):
                train_losses = training_history["train_loss"]
                final_gap = abs(train_losses[-1] - val_losses[-1])
                quality["overfitting_gap"] = float(final_gap)
                quality["overfitting_risk"] = "high" if final_gap > 0.1 else "low"

        return quality

    def _get_dimension_requirements(self, model_type: str, model_config: dict[str, Any]) -> dict[str, Any]:
        """Get dimension requirements for model compatibility."""
        if model_type == "LSTM":
            return {
                "input_size": model_config.get("standard_input_size", 600),
                "sequence_length": model_config.get("sequence_length", 60),
                "flexible_universe": False,  # LSTM requires fixed dimensions
            }
        elif model_type == "GAT":
            return {
                "min_universe_size": 10,
                "max_universe_size": 1000,
                "flexible_universe": True,  # GAT can handle variable graph sizes
            }
        elif model_type == "HRP":
            return {
                "min_universe_size": 2,
                "max_universe_size": None,
                "flexible_universe": True,  # HRP is fully flexible
            }
        return {}

    def _get_universe_requirements(self, model_type: str) -> dict[str, Any]:
        """Get universe requirements for model compatibility."""
        return {
            "requires_exact_match": model_type == "LSTM",
            "supports_subset": model_type in ["GAT", "HRP"],
            "supports_expansion": model_type == "HRP",
        }

    def _get_preprocessing_requirements(self) -> dict[str, Any]:
        """Get preprocessing requirements from config."""
        data_settings = self.config.get("data_settings", {})
        return {
            "extreme_return_threshold": data_settings.get("extreme_return_threshold", 10),
            "max_forward_fill": data_settings.get("max_forward_fill", 5),
            "min_data_coverage": data_settings.get("min_data_coverage", 0.8),
            "lookback_days": data_settings.get("lookback_days", 252),
        }

    def _get_lstm_dimensions(self, model_config: dict[str, Any]) -> dict[str, Any]:
        """Get LSTM-specific dimension information."""
        return {
            "input_size": model_config.get("standard_input_size", 600),
            "output_size": model_config.get("standard_input_size", 600),
            "hidden_size": model_config.get("hidden_size", 128),
            "num_layers": model_config.get("num_layers", 2),
            "sequence_length": model_config.get("sequence_length", 60),
        }

    def _get_gat_config(self, model_config: dict[str, Any]) -> dict[str, Any]:
        """Get GAT-specific configuration information."""
        return {
            "input_features": model_config.get("input_features", 10),
            "hidden_dim": model_config.get("hidden_dim", 64),
            "num_layers": model_config.get("num_layers", 3),
            "num_attention_heads": model_config.get("num_attention_heads", 8),
            "lookback_days": model_config.get("lookback_days", 252),
        }

    def _get_hrp_config(self, model_config: dict[str, Any]) -> dict[str, Any]:
        """Get HRP-specific configuration information."""
        return {
            "lookback_days": model_config.get("lookback_days", 756),
            "min_observations": model_config.get("min_observations", 252),
            "correlation_method": model_config.get("correlation_method", "pearson"),
            "rebalance_frequency": model_config.get("rebalance_frequency", "monthly"),
        }

    def save_metadata(self, metadata: dict[str, Any], output_path: str | Path) -> None:
        """Save metadata to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Enhanced metadata saved to {output_path}")

    def validate_compatibility(
        self,
        metadata: dict[str, Any],
        target_universe: list[str],
        target_period: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """
        Validate model compatibility with target scenario.

        Args:
            metadata: Model metadata to validate
            target_universe: Target universe for backtesting
            target_period: Target period for backtesting

        Returns:
            Compatibility report
        """
        report = {
            "compatible": True,
            "warnings": [],
            "errors": [],
            "recommendations": [],
        }

        # Check universe compatibility
        trained_universe = set(metadata.get("trained_universe", []))
        target_universe_set = set(target_universe)

        overlap = trained_universe.intersection(target_universe_set)
        coverage = len(overlap) / len(target_universe_set) if target_universe_set else 0

        if coverage < 0.5:
            report["errors"].append(f"Low universe coverage: {coverage:.1%}")
            report["compatible"] = False
        elif coverage < 0.8:
            report["warnings"].append(f"Moderate universe coverage: {coverage:.1%}")

        # Check dimension requirements
        dim_req = metadata.get("compatibility", {}).get("dimension_requirements", {})
        if not dim_req.get("flexible_universe", True):
            if len(target_universe) != dim_req.get("input_size", 0):
                report["errors"].append(
                    f"Dimension mismatch: model requires {dim_req.get('input_size')} assets, got {len(target_universe)}"
                )
                report["compatible"] = False

        # Check training quality
        training_quality = metadata.get("training_quality", {})
        if not training_quality.get("converged", True):
            report["warnings"].append("Model may not have converged during training")

        if training_quality.get("overfitting_risk") == "high":
            report["warnings"].append("Model shows signs of overfitting")

        return report


def create_default_generator(config_path: str | None = None) -> MetadataGenerator:
    """Create metadata generator with default configuration."""
    if config_path is None:
        config_path = "configs/model_config.yaml"

    return MetadataGenerator(config_path)