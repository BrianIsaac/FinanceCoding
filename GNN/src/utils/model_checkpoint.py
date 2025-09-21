"""
Standardised Model Checkpoint Manager.

This module provides unified checkpointing and metadata management across all
model types (HRP, LSTM, GAT) to ensure consistency and compatibility.
"""

from __future__ import annotations

import json
import logging
import pickle
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ModelCheckpointMetadata:
    """Standardised metadata for all model checkpoints."""

    # Core model information
    model_type: str
    model_version: str
    model_name: str

    # Training configuration
    training_config: dict[str, Any]
    model_config: dict[str, Any]
    constraints_config: dict[str, Any]

    # Universe information
    trained_universe: list[str]
    universe_size: int
    universe_source: str

    # Training details
    training_period_start: str
    training_period_end: str
    training_timestamp: str
    training_duration_seconds: float

    # Performance metrics (optional)
    training_metrics: dict[str, Any] = field(default_factory=dict)
    validation_metrics: dict[str, Any] = field(default_factory=dict)

    # Technical details
    checkpoint_format: str = "pytorch"  # "pytorch", "pickle", "custom"
    framework_version: str = ""
    python_version: str = ""

    # Compatibility information
    input_features: dict[str, Any] = field(default_factory=dict)
    output_features: dict[str, Any] = field(default_factory=dict)
    universe_validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialisation."""
        return {
            "model_type": self.model_type,
            "model_version": self.model_version,
            "model_name": self.model_name,
            "training_config": self.training_config,
            "model_config": self.model_config,
            "constraints_config": self.constraints_config,
            "trained_universe": self.trained_universe,
            "universe_size": self.universe_size,
            "universe_source": self.universe_source,
            "training_period_start": self.training_period_start,
            "training_period_end": self.training_period_end,
            "training_timestamp": self.training_timestamp,
            "training_duration_seconds": self.training_duration_seconds,
            "training_metrics": self.training_metrics,
            "validation_metrics": self.validation_metrics,
            "checkpoint_format": self.checkpoint_format,
            "framework_version": self.framework_version,
            "python_version": self.python_version,
            "input_features": self.input_features,
            "output_features": self.output_features,
            "universe_validation": self.universe_validation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelCheckpointMetadata:
        """Create from dictionary."""
        return cls(**data)


class ModelCheckpointManager:
    """
    Unified checkpoint manager for all model types.

    Provides consistent checkpointing, loading, and metadata management
    across HRP, LSTM, and GAT models.
    """

    def __init__(self, base_output_dir: str | Path):
        """
        Initialise checkpoint manager.

        Args:
            base_output_dir: Base directory for all model checkpoints
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

    def save_model_checkpoint(
        self,
        model: Any,
        metadata: ModelCheckpointMetadata,
        checkpoint_name: str | None = None
    ) -> Path:
        """
        Save model checkpoint with standardised metadata.

        Args:
            model: The trained model instance
            metadata: Standardised metadata
            checkpoint_name: Optional custom checkpoint name

        Returns:
            Path to saved checkpoint directory
        """
        # Create checkpoint directory structure
        model_dir = self.base_output_dir / metadata.model_type.lower()
        if checkpoint_name:
            checkpoint_dir = model_dir / checkpoint_name
        else:
            checkpoint_dir = model_dir / f"{metadata.model_name}_{metadata.training_timestamp[:10]}"

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model based on type
        model_path = self._save_model_file(model, metadata, checkpoint_dir)

        # Save metadata
        metadata_path = checkpoint_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2, default=str)

        # Create checkpoint info file
        checkpoint_info = {
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "checkpoint_dir": str(checkpoint_dir),
            "model_type": metadata.model_type,
            "model_name": metadata.model_name,
            "created_at": metadata.training_timestamp,
        }

        info_path = checkpoint_dir / "checkpoint_info.json"
        with open(info_path, "w") as f:
            json.dump(checkpoint_info, f, indent=2)

        logger.info(f"Saved {metadata.model_type} model checkpoint to {checkpoint_dir}")
        return checkpoint_dir

    def _save_model_file(
        self,
        model: Any,
        metadata: ModelCheckpointMetadata,
        checkpoint_dir: Path
    ) -> Path:
        """Save the actual model file based on model type."""
        model_type = metadata.model_type.lower()

        if model_type == "hrp":
            # HRP models use pickle
            model_path = checkpoint_dir / "model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            metadata.checkpoint_format = "pickle"

        elif model_type in ["lstm", "gat"]:
            # PyTorch models use .pt format
            model_path = checkpoint_dir / "model.pt"
            if hasattr(model, "save_model"):
                model.save_model(str(model_path))
            else:
                # Fallback for direct PyTorch models
                import torch
                torch.save(model, model_path)
            metadata.checkpoint_format = "pytorch"

        else:
            # Generic fallback
            model_path = checkpoint_dir / "model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            metadata.checkpoint_format = "pickle"

        return model_path

    def load_model_checkpoint(self, checkpoint_path: str | Path) -> tuple[Any, ModelCheckpointMetadata]:
        """
        Load model and metadata from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            Tuple of (model, metadata)
        """
        checkpoint_dir = Path(checkpoint_path)

        # Load metadata
        metadata_path = checkpoint_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path) as f:
            metadata_dict = json.load(f)

        metadata = ModelCheckpointMetadata.from_dict(metadata_dict)

        # Load model based on format
        model = self._load_model_file(checkpoint_dir, metadata)

        logger.info(f"Loaded {metadata.model_type} model from {checkpoint_dir}")
        return model, metadata

    def _load_model_file(self, checkpoint_dir: Path, metadata: ModelCheckpointMetadata) -> Any:
        """Load the actual model file based on checkpoint format."""
        if metadata.checkpoint_format == "pickle":
            model_path = checkpoint_dir / "model.pkl"
            with open(model_path, "rb") as f:
                return pickle.load(f)

        elif metadata.checkpoint_format == "pytorch":
            model_path = checkpoint_dir / "model.pt"
            import torch
            return torch.load(model_path, map_location="cpu")

        else:
            raise ValueError(f"Unknown checkpoint format: {metadata.checkpoint_format}")

    def list_checkpoints(self, model_type: str | None = None) -> list[dict[str, Any]]:
        """
        List available checkpoints.

        Args:
            model_type: Optional filter by model type

        Returns:
            List of checkpoint information
        """
        checkpoints = []

        for model_dir in self.base_output_dir.iterdir():
            if not model_dir.is_dir():
                continue

            if model_type and model_dir.name != model_type.lower():
                continue

            for checkpoint_dir in model_dir.iterdir():
                if not checkpoint_dir.is_dir():
                    continue

                info_path = checkpoint_dir / "checkpoint_info.json"
                if info_path.exists():
                    with open(info_path) as f:
                        checkpoint_info = json.load(f)
                    checkpoints.append(checkpoint_info)

        # Sort by creation time
        checkpoints.sort(key=lambda x: x["created_at"], reverse=True)
        return checkpoints

    def validate_checkpoint(self, checkpoint_path: str | Path) -> dict[str, Any]:
        """
        Validate checkpoint integrity.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            Validation results
        """
        checkpoint_dir = Path(checkpoint_path)
        validation_results = {
            "is_valid": True,
            "issues": [],
            "metadata_exists": False,
            "model_exists": False,
            "checkpoint_info_exists": False,
        }

        # Check metadata
        metadata_path = checkpoint_dir / "metadata.json"
        if metadata_path.exists():
            validation_results["metadata_exists"] = True
            try:
                with open(metadata_path) as f:
                    metadata_dict = json.load(f)
                ModelCheckpointMetadata.from_dict(metadata_dict)
            except Exception as e:
                validation_results["is_valid"] = False
                validation_results["issues"].append(f"Invalid metadata: {e}")
        else:
            validation_results["is_valid"] = False
            validation_results["issues"].append("Metadata file missing")

        # Check checkpoint info
        info_path = checkpoint_dir / "checkpoint_info.json"
        validation_results["checkpoint_info_exists"] = info_path.exists()

        # Check model file
        model_files = list(checkpoint_dir.glob("model.*"))
        if model_files:
            validation_results["model_exists"] = True
        else:
            validation_results["is_valid"] = False
            validation_results["issues"].append("Model file missing")

        return validation_results

    def cleanup_old_checkpoints(
        self,
        model_type: str,
        keep_latest: int = 5
    ) -> list[Path]:
        """
        Clean up old checkpoints, keeping only the latest ones.

        Args:
            model_type: Model type to clean up
            keep_latest: Number of latest checkpoints to keep

        Returns:
            List of removed checkpoint paths
        """
        checkpoints = self.list_checkpoints(model_type)

        if len(checkpoints) <= keep_latest:
            return []

        # Remove oldest checkpoints
        removed_paths = []
        for checkpoint in checkpoints[keep_latest:]:
            checkpoint_dir = Path(checkpoint["checkpoint_dir"])
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)
                removed_paths.append(checkpoint_dir)
                logger.info(f"Removed old checkpoint: {checkpoint_dir}")

        return removed_paths