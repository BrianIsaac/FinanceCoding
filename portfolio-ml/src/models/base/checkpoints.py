"""
Model checkpoint management system for portfolio optimization models.

This module provides comprehensive model serialization, versioning, and persistence
functionality to enable consistent backtesting and model reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for model checkpoints."""

    model_name: str
    model_type: str  # 'hrp', 'lstm', 'gat'
    version: str
    created_at: pd.Timestamp

    # Model configuration
    hyperparameters: dict[str, Any]
    training_config: dict[str, Any]

    # Training metrics
    training_metrics: dict[str, float] = field(default_factory=dict)
    validation_metrics: dict[str, float] = field(default_factory=dict)

    # Data information
    training_period: tuple[str, str] | None = None  # (start_date, end_date)
    validation_period: tuple[str, str] | None = None
    data_hash: str | None = None  # Hash of training data for reproducibility

    # Technical metadata
    model_hash: str | None = None  # Hash of model parameters
    file_size_mb: float | None = None
    pytorch_version: str | None = None
    python_version: str | None = None

    # Additional info
    tags: list[str] = field(default_factory=list)
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "hyperparameters": self.hyperparameters,
            "training_config": self.training_config,
            "training_metrics": self.training_metrics,
            "validation_metrics": self.validation_metrics,
            "training_period": self.training_period,
            "validation_period": self.validation_period,
            "data_hash": self.data_hash,
            "model_hash": self.model_hash,
            "file_size_mb": self.file_size_mb,
            "pytorch_version": self.pytorch_version,
            "python_version": self.python_version,
            "tags": self.tags,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointMetadata:
        """Create metadata from dictionary."""
        data = data.copy()
        data["created_at"] = pd.Timestamp(data["created_at"])
        return cls(**data)


class ModelCheckpointManager:
    """
    Comprehensive model checkpoint management system.

    This manager provides versioned model serialization, metadata tracking,
    and integrity validation for consistent backtesting and reproducibility.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        enable_versioning: bool = True,
        max_versions_per_model: int = 10,
        compression_level: int = 6,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            enable_versioning: Whether to enable automatic versioning
            max_versions_per_model: Maximum versions to keep per model
            compression_level: Compression level for model files (0-9)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.enable_versioning = enable_versioning
        self.max_versions_per_model = max_versions_per_model
        self.compression_level = compression_level

        # Create directory structure
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.checkpoint_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)

        # Model registry
        self._model_registry: dict[str, list[str]] = {}
        self._load_registry()

    def save_checkpoint(
        self,
        model: Any,
        model_name: str,
        model_type: str,
        hyperparameters: dict[str, Any],
        training_config: dict[str, Any],
        training_metrics: dict[str, float] | None = None,
        validation_metrics: dict[str, float] | None = None,
        training_period: tuple[str, str] | None = None,
        validation_period: tuple[str, str] | None = None,
        data_hash: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        version: str | None = None,
    ) -> str:
        """
        Save model checkpoint with comprehensive metadata.

        Args:
            model: Model to save
            model_name: Name of the model
            model_type: Type of model ('hrp', 'lstm', 'gat')
            hyperparameters: Model hyperparameters
            training_config: Training configuration
            training_metrics: Training performance metrics
            validation_metrics: Validation performance metrics
            training_period: Training period (start_date, end_date)
            validation_period: Validation period (start_date, end_date)
            data_hash: Hash of training data
            tags: Optional tags for categorization
            notes: Optional notes about the checkpoint
            version: Specific version string (auto-generated if None)

        Returns:
            Checkpoint version string
        """
        # Generate version if not provided
        if version is None:
            version = self._generate_version(model_name)

        # Create checkpoint directory
        checkpoint_path = self._get_checkpoint_path(model_name, version)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Calculate model hash for integrity checking
        model_hash = self._calculate_model_hash(model)

        # Create metadata
        metadata = CheckpointMetadata(
            model_name=model_name,
            model_type=model_type,
            version=version,
            created_at=pd.Timestamp.now(),
            hyperparameters=hyperparameters,
            training_config=training_config,
            training_metrics=training_metrics or {},
            validation_metrics=validation_metrics or {},
            training_period=training_period,
            validation_period=validation_period,
            data_hash=data_hash,
            model_hash=model_hash,
            pytorch_version=torch.__version__,
            python_version=self._get_python_version(),
            tags=tags or [],
            notes=notes,
        )

        # Save model
        model_path = checkpoint_path / "model.pkl"
        self._save_model(model, model_path)

        # Update file size in metadata
        metadata.file_size_mb = model_path.stat().st_size / (1024 * 1024)

        # Save metadata
        metadata_path = checkpoint_path / "metadata.json"
        self._save_metadata(metadata, metadata_path)

        # Save hyperparameters and config separately for easy access
        self._save_hyperparameters(hyperparameters, checkpoint_path / "hyperparameters.yaml")
        self._save_config(training_config, checkpoint_path / "training_config.yaml")

        # Update registry
        self._update_registry(model_name, version)

        # Cleanup old versions if necessary
        if self.enable_versioning:
            self._cleanup_old_versions(model_name)

        logger.info(f"Saved checkpoint: {model_name} v{version} ({metadata.file_size_mb:.2f}MB)")
        return version

    def load_checkpoint(
        self, model_name: str, version: str | None = None, validate_integrity: bool = True
    ) -> tuple[Any, CheckpointMetadata]:
        """
        Load model checkpoint with metadata.

        Args:
            model_name: Name of the model to load
            version: Specific version to load (latest if None)
            validate_integrity: Whether to validate checkpoint integrity

        Returns:
            Tuple of (model, metadata)
        """
        # Get version to load
        if version is None:
            version = self.get_latest_version(model_name)
            if version is None:
                raise ValueError(f"No checkpoints found for model: {model_name}")

        checkpoint_path = self._get_checkpoint_path(model_name, version)
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {model_name} v{version}")

        # Load metadata
        metadata_path = checkpoint_path / "metadata.json"
        metadata = self._load_metadata(metadata_path)

        # Load model
        model_path = checkpoint_path / "model.pkl"
        model = self._load_model(model_path)

        # Validate integrity if requested
        if validate_integrity:
            self._validate_checkpoint_integrity(model, metadata, checkpoint_path)

        logger.info(f"Loaded checkpoint: {model_name} v{version}")
        return model, metadata

    def list_checkpoints(self, model_name: str | None = None) -> pd.DataFrame:
        """
        List available checkpoints.

        Args:
            model_name: Optional model name filter

        Returns:
            DataFrame with checkpoint information
        """
        checkpoints = []

        models_to_check = [model_name] if model_name else self._model_registry.keys()

        for model in models_to_check:
            if model not in self._model_registry:
                continue

            for version in self._model_registry[model]:
                try:
                    metadata_path = self._get_checkpoint_path(model, version) / "metadata.json"
                    if metadata_path.exists():
                        metadata = self._load_metadata(metadata_path)

                        checkpoint_info = {
                            "model_name": metadata.model_name,
                            "model_type": metadata.model_type,
                            "version": metadata.version,
                            "created_at": metadata.created_at,
                            "file_size_mb": metadata.file_size_mb,
                            "training_sharpe": metadata.training_metrics.get("sharpe_ratio"),
                            "validation_sharpe": metadata.validation_metrics.get("sharpe_ratio"),
                            "tags": ", ".join(metadata.tags),
                            "notes": metadata.notes,
                        }
                        checkpoints.append(checkpoint_info)

                except Exception as e:
                    logger.warning(f"Error reading checkpoint {model} v{version}: {e}")

        if not checkpoints:
            return pd.DataFrame()

        df = pd.DataFrame(checkpoints)
        return df.sort_values(["model_name", "created_at"], ascending=[True, False])

    def delete_checkpoint(self, model_name: str, version: str) -> bool:
        """
        Delete a specific checkpoint.

        Args:
            model_name: Name of the model
            version: Version to delete

        Returns:
            Whether deletion was successful
        """
        checkpoint_path = self._get_checkpoint_path(model_name, version)

        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {model_name} v{version}")
            return False

        try:
            shutil.rmtree(checkpoint_path)

            # Update registry
            if model_name in self._model_registry:
                if version in self._model_registry[model_name]:
                    self._model_registry[model_name].remove(version)
                    self._save_registry()

            logger.info(f"Deleted checkpoint: {model_name} v{version}")
            return True

        except Exception as e:
            logger.error(f"Error deleting checkpoint {model_name} v{version}: {e}")
            return False

    def get_latest_version(self, model_name: str) -> str | None:
        """Get latest version of a model."""
        if model_name not in self._model_registry:
            return None

        versions = self._model_registry[model_name]
        if not versions:
            return None

        # Sort versions and return the latest
        return max(versions, key=self._version_sort_key)

    def get_checkpoint_metadata(self, model_name: str, version: str) -> CheckpointMetadata:
        """Get metadata for a specific checkpoint."""
        metadata_path = self._get_checkpoint_path(model_name, version) / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Checkpoint not found: {model_name} v{version}")

        return self._load_metadata(metadata_path)

    def export_checkpoint(self, model_name: str, version: str, export_path: str | Path) -> None:
        """
        Export checkpoint to external location.

        Args:
            model_name: Name of the model
            version: Version to export
            export_path: Path to export to
        """
        source_path = self._get_checkpoint_path(model_name, version)
        export_path = Path(export_path)

        if not source_path.exists():
            raise ValueError(f"Checkpoint not found: {model_name} v{version}")

        # Create export directory
        export_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy checkpoint directory
        if export_path.exists():
            shutil.rmtree(export_path)

        shutil.copytree(source_path, export_path)
        logger.info(f"Exported checkpoint to: {export_path}")

    def _get_checkpoint_path(self, model_name: str, version: str) -> Path:
        """Get path for a specific checkpoint."""
        return self.checkpoint_dir / model_name / version

    def _generate_version(self, model_name: str) -> str:
        """Generate version string for a model."""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        if not self.enable_versioning:
            return "latest"

        # Check if this timestamp already exists
        base_version = f"v{timestamp}"
        counter = 1
        version = base_version

        while model_name in self._model_registry and version in self._model_registry[model_name]:
            version = f"{base_version}_{counter:02d}"
            counter += 1

        return version

    def _calculate_model_hash(self, model: Any) -> str:
        """Calculate hash of model parameters."""
        if hasattr(model, "state_dict"):
            # PyTorch model
            state_dict = model.state_dict()
            model_bytes = pickle.dumps({k: v.cpu().numpy() for k, v in state_dict.items()})
        else:
            # Other model types
            model_bytes = pickle.dumps(model)

        return hashlib.md5(model_bytes).hexdigest()

    def _save_model(self, model: Any, model_path: Path) -> None:
        """Save model to file."""
        try:
            if hasattr(model, "state_dict"):
                # PyTorch model - save state dict
                torch.save(model.state_dict(), model_path)
            else:
                # Other model types - use pickle
                with open(model_path, "wb") as f:
                    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error(f"Error saving model to {model_path}: {e}")
            raise

    def _load_model(self, model_path: Path) -> Any:
        """Load model from file."""
        try:
            # Try PyTorch format first
            try:
                return torch.load(model_path, map_location="cpu")
            except Exception:
                # Fall back to pickle
                with open(model_path, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise

    def _save_metadata(self, metadata: CheckpointMetadata, metadata_path: Path) -> None:
        """Save metadata to file."""
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2, default=str)

    def _load_metadata(self, metadata_path: Path) -> CheckpointMetadata:
        """Load metadata from file."""
        with open(metadata_path) as f:
            data = json.load(f)
        return CheckpointMetadata.from_dict(data)

    def _save_hyperparameters(self, hyperparameters: dict[str, Any], path: Path) -> None:
        """Save hyperparameters to YAML file."""
        OmegaConf.save(hyperparameters, path)

    def _save_config(self, config: dict[str, Any], path: Path) -> None:
        """Save configuration to YAML file."""
        OmegaConf.save(config, path)

    def _validate_checkpoint_integrity(
        self, model: Any, metadata: CheckpointMetadata, checkpoint_path: Path
    ) -> None:
        """Validate checkpoint integrity."""
        # Verify model hash
        current_hash = self._calculate_model_hash(model)
        if metadata.model_hash and current_hash != metadata.model_hash:
            raise ValueError("Model integrity check failed: hash mismatch")

        # Verify required files exist
        required_files = [
            "metadata.json",
            "model.pkl",
            "hyperparameters.yaml",
            "training_config.yaml",
        ]
        for file_name in required_files:
            file_path = checkpoint_path / file_name
            if not file_path.exists():
                raise ValueError(f"Checkpoint incomplete: missing {file_name}")

        logger.debug("Checkpoint integrity validated successfully")

    def _update_registry(self, model_name: str, version: str) -> None:
        """Update model registry."""
        if model_name not in self._model_registry:
            self._model_registry[model_name] = []

        if version not in self._model_registry[model_name]:
            self._model_registry[model_name].append(version)

        self._save_registry()

    def _cleanup_old_versions(self, model_name: str) -> None:
        """Clean up old versions beyond the limit."""
        if model_name not in self._model_registry:
            return

        versions = self._model_registry[model_name]
        if len(versions) <= self.max_versions_per_model:
            return

        # Sort versions and keep only the most recent ones
        sorted_versions = sorted(versions, key=self._version_sort_key, reverse=True)
        versions_to_delete = sorted_versions[self.max_versions_per_model :]

        for version in versions_to_delete:
            try:
                self.delete_checkpoint(model_name, version)
                logger.info(f"Cleaned up old version: {model_name} v{version}")
            except Exception as e:
                logger.warning(f"Error cleaning up {model_name} v{version}: {e}")

    def _version_sort_key(self, version: str) -> str:
        """Generate sort key for version strings."""
        # Handle 'latest' version
        if version == "latest":
            return "9999999999_999999"

        # Extract timestamp for sorting
        if version.startswith("v"):
            version = version[1:]  # Remove 'v' prefix

        return version

    def _load_registry(self) -> None:
        """Load model registry from disk."""
        registry_path = self.checkpoint_dir / "registry.json"

        if registry_path.exists():
            try:
                with open(registry_path) as f:
                    self._model_registry = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading registry: {e}")
                self._model_registry = {}
        else:
            self._model_registry = {}

    def _save_registry(self) -> None:
        """Save model registry to disk."""
        registry_path = self.checkpoint_dir / "registry.json"

        try:
            with open(registry_path, "w") as f:
                json.dump(self._model_registry, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving registry: {e}")

    def _get_python_version(self) -> str:
        """Get current Python version."""
        import sys

        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


class ModelStatePreserver:
    """
    Model state preservation system for consistent backtesting.

    This class ensures that models can be exactly reproduced for
    backtesting purposes, preserving all relevant state including
    random seeds, data preprocessing parameters, and model weights.
    """

    def __init__(self, checkpoint_manager: ModelCheckpointManager):
        """
        Initialize model state preserver.

        Args:
            checkpoint_manager: Checkpoint manager instance
        """
        self.checkpoint_manager = checkpoint_manager
        self._preserved_states: dict[str, dict[str, Any]] = {}

    def preserve_training_state(
        self,
        model_name: str,
        model: Any,
        optimizer_state: dict[str, Any] | None = None,
        scheduler_state: dict[str, Any] | None = None,
        random_states: dict[str, Any] | None = None,
        data_preprocessing_params: dict[str, Any] | None = None,
        feature_transformations: dict[str, Any] | None = None,
        training_step: int = 0,
        epoch: int = 0,
    ) -> str:
        """
        Preserve complete training state for reproducible backtesting.

        Args:
            model_name: Name of the model
            model: The trained model
            optimizer_state: State of the optimizer
            scheduler_state: State of the learning rate scheduler
            random_states: Random number generator states
            data_preprocessing_params: Data preprocessing parameters
            feature_transformations: Feature transformation parameters
            training_step: Current training step
            epoch: Current epoch

        Returns:
            State preservation ID
        """
        # Generate unique state ID
        state_id = f"{model_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

        # Collect comprehensive state information
        complete_state = {
            "model_state": self._extract_model_state(model),
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
            "random_states": random_states or self._capture_random_states(),
            "data_preprocessing_params": data_preprocessing_params,
            "feature_transformations": feature_transformations,
            "training_metadata": {
                "training_step": training_step,
                "epoch": epoch,
                "preservation_timestamp": pd.Timestamp.now().isoformat(),
                "model_name": model_name,
            },
            "environment_info": self._capture_environment_info(),
        }

        # Store state
        self._preserved_states[state_id] = complete_state

        # Save to persistent storage
        self._save_preserved_state(state_id, complete_state)

        logger.info(f"Preserved training state: {state_id}")
        return state_id

    def restore_training_state(
        self,
        state_id: str,
        model: Any,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        restore_random_states: bool = True,
    ) -> dict[str, Any]:
        """
        Restore complete training state for consistent backtesting.

        Args:
            state_id: State preservation ID
            model: Model to restore state to
            optimizer: Optimizer to restore state to
            scheduler: Scheduler to restore state to
            restore_random_states: Whether to restore random states

        Returns:
            Dictionary with restoration information
        """
        # Load preserved state
        if state_id not in self._preserved_states:
            self._load_preserved_state(state_id)

        if state_id not in self._preserved_states:
            raise ValueError(f"Preserved state not found: {state_id}")

        state = self._preserved_states[state_id]
        restoration_info = {}

        # Restore model state
        self._restore_model_state(model, state["model_state"])
        restoration_info["model_restored"] = True

        # Restore optimizer state
        if optimizer and state["optimizer_state"]:
            try:
                optimizer.load_state_dict(state["optimizer_state"])
                restoration_info["optimizer_restored"] = True
            except Exception as e:
                logger.warning(f"Could not restore optimizer state: {e}")
                restoration_info["optimizer_restored"] = False

        # Restore scheduler state
        if scheduler and state["scheduler_state"]:
            try:
                scheduler.load_state_dict(state["scheduler_state"])
                restoration_info["scheduler_restored"] = True
            except Exception as e:
                logger.warning(f"Could not restore scheduler state: {e}")
                restoration_info["scheduler_restored"] = False

        # Restore random states
        if restore_random_states and state["random_states"]:
            self._restore_random_states(state["random_states"])
            restoration_info["random_states_restored"] = True

        # Return metadata for verification
        restoration_info.update(
            {
                "state_id": state_id,
                "original_timestamp": state["training_metadata"]["preservation_timestamp"],
                "restoration_timestamp": pd.Timestamp.now().isoformat(),
                "training_step": state["training_metadata"]["training_step"],
                "epoch": state["training_metadata"]["epoch"],
                "data_preprocessing_params": state["data_preprocessing_params"],
                "feature_transformations": state["feature_transformations"],
            }
        )

        logger.info(f"Restored training state: {state_id}")
        return restoration_info

    def create_backtest_snapshot(
        self,
        model_name: str,
        model: Any,
        validation_period: tuple[str, str],
        training_data_hash: str,
        performance_metrics: dict[str, float],
        hyperparameters: dict[str, Any],
        preprocessing_pipeline: Any | None = None,
    ) -> str:
        """
        Create a complete snapshot for backtesting reproducibility.

        Args:
            model_name: Name of the model
            model: Trained model
            validation_period: Validation period (start, end)
            training_data_hash: Hash of training data
            performance_metrics: Model performance metrics
            hyperparameters: Model hyperparameters
            preprocessing_pipeline: Data preprocessing pipeline

        Returns:
            Snapshot ID
        """
        snapshot_id = f"backtest_{model_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

        # Create comprehensive snapshot
        snapshot = {
            "model_state": self._extract_model_state(model),
            "validation_period": validation_period,
            "training_data_hash": training_data_hash,
            "performance_metrics": performance_metrics,
            "hyperparameters": hyperparameters,
            "preprocessing_pipeline": self._serialize_preprocessing_pipeline(
                preprocessing_pipeline
            ),
            "random_states": self._capture_random_states(),
            "environment_info": self._capture_environment_info(),
            "snapshot_metadata": {
                "snapshot_id": snapshot_id,
                "model_name": model_name,
                "created_at": pd.Timestamp.now().isoformat(),
                "snapshot_type": "backtest",
            },
        }

        # Store snapshot
        self._preserved_states[snapshot_id] = snapshot
        self._save_preserved_state(snapshot_id, snapshot)

        logger.info(f"Created backtest snapshot: {snapshot_id}")
        return snapshot_id

    def validate_backtest_consistency(
        self, snapshot_id: str, current_model: Any, current_data_hash: str, tolerance: float = 1e-6
    ) -> dict[str, Any]:
        """
        Validate consistency between current state and backtest snapshot.

        Args:
            snapshot_id: Snapshot to validate against
            current_model: Current model to compare
            current_data_hash: Hash of current data
            tolerance: Numerical tolerance for comparisons

        Returns:
            Validation results
        """
        if snapshot_id not in self._preserved_states:
            self._load_preserved_state(snapshot_id)

        if snapshot_id not in self._preserved_states:
            raise ValueError(f"Snapshot not found: {snapshot_id}")

        snapshot = self._preserved_states[snapshot_id]
        validation_results = {
            "snapshot_id": snapshot_id,
            "validation_timestamp": pd.Timestamp.now().isoformat(),
            "consistent": True,
            "issues": [],
        }

        # Validate model state consistency
        model_consistent = self._validate_model_state_consistency(
            current_model, snapshot["model_state"], tolerance
        )
        validation_results["model_consistent"] = model_consistent
        if not model_consistent:
            validation_results["consistent"] = False
            validation_results["issues"].append("Model state mismatch")

        # Validate data consistency
        data_consistent = current_data_hash == snapshot["training_data_hash"]
        validation_results["data_consistent"] = data_consistent
        if not data_consistent:
            validation_results["consistent"] = False
            validation_results["issues"].append("Training data mismatch")

        # Validate environment consistency
        env_consistent = self._validate_environment_consistency(snapshot["environment_info"])
        validation_results["environment_consistent"] = env_consistent
        if not env_consistent:
            validation_results["consistent"] = False
            validation_results["issues"].append("Environment differences detected")

        return validation_results

    def _extract_model_state(self, model: Any) -> dict[str, Any]:
        """Extract complete model state."""
        if hasattr(model, "state_dict"):
            # PyTorch model
            state_dict = model.state_dict()
            return {
                "type": "pytorch",
                "state_dict": {k: v.cpu().numpy() for k, v in state_dict.items()},
                "model_class": model.__class__.__name__,
            }
        else:
            # Scikit-learn or other model
            return {"type": "sklearn", "model": model, "model_class": model.__class__.__name__}

    def _restore_model_state(self, model: Any, model_state: dict[str, Any]) -> None:
        """Restore model state."""
        if model_state["type"] == "pytorch" and hasattr(model, "load_state_dict"):
            # Convert numpy arrays back to tensors
            state_dict = {k: torch.from_numpy(v) for k, v in model_state["state_dict"].items()}
            model.load_state_dict(state_dict)
        elif model_state["type"] == "sklearn":
            # For sklearn models, we would need to replace the entire model
            # This is a limitation of the current approach
            logger.warning("Cannot restore sklearn model state in-place")

    def _capture_random_states(self) -> dict[str, Any]:
        """Capture all random number generator states."""
        import random

        random_states = {"python_random": random.getstate(), "numpy_random": np.random.get_state()}

        if torch.cuda.is_available():
            random_states["torch_cuda_random"] = torch.cuda.get_rng_state_all()

        if hasattr(torch, "get_rng_state"):
            random_states["torch_random"] = torch.get_rng_state()

        return random_states

    def _restore_random_states(self, random_states: dict[str, Any]) -> None:
        """Restore random number generator states."""
        import random

        if "python_random" in random_states:
            random.setstate(random_states["python_random"])

        if "numpy_random" in random_states:
            np.random.set_state(random_states["numpy_random"])

        if "torch_random" in random_states:
            torch.set_rng_state(random_states["torch_random"])

        if "torch_cuda_random" in random_states and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(random_states["torch_cuda_random"])

    def _capture_environment_info(self) -> dict[str, Any]:
        """Capture environment information."""
        import platform
        import sys

        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "pytorch_version": torch.__version__ if "torch" in sys.modules else None,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "cuda_available": torch.cuda.is_available() if "torch" in sys.modules else False,
            "cuda_version": (
                torch.version.cuda if "torch" in sys.modules and torch.cuda.is_available() else None
            ),
        }

    def _serialize_preprocessing_pipeline(self, pipeline: Any) -> dict[str, Any] | None:
        """Serialize preprocessing pipeline."""
        if pipeline is None:
            return None

        try:
            return {"pipeline": pickle.dumps(pipeline).hex(), "type": pipeline.__class__.__name__}
        except Exception as e:
            logger.warning(f"Could not serialize preprocessing pipeline: {e}")
            return None

    def _validate_model_state_consistency(
        self, current_model: Any, preserved_state: dict[str, Any], tolerance: float
    ) -> bool:
        """Validate model state consistency."""
        try:
            current_state = self._extract_model_state(current_model)

            if current_state["type"] != preserved_state["type"]:
                return False

            if current_state["type"] == "pytorch":
                # Compare state dicts numerically
                current_dict = current_state["state_dict"]
                preserved_dict = preserved_state["state_dict"]

                if set(current_dict.keys()) != set(preserved_dict.keys()):
                    return False

                for key in current_dict.keys():
                    if not np.allclose(current_dict[key], preserved_dict[key], atol=tolerance):
                        return False

            return True

        except Exception as e:
            logger.warning(f"Error validating model state consistency: {e}")
            return False

    def _validate_environment_consistency(self, preserved_env: dict[str, Any]) -> bool:
        """Validate environment consistency."""
        current_env = self._capture_environment_info()

        # Check critical environment components
        critical_components = ["python_version", "pytorch_version", "numpy_version"]

        for component in critical_components:
            if current_env.get(component) != preserved_env.get(component):
                logger.warning(f"Environment mismatch: {component}")
                return False

        return True

    def _save_preserved_state(self, state_id: str, state: dict[str, Any]) -> None:
        """Save preserved state to disk."""
        state_path = self.checkpoint_manager.checkpoint_dir / "preserved_states" / f"{state_id}.pkl"
        state_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(state_path, "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error(f"Error saving preserved state {state_id}: {e}")
            raise

    def _load_preserved_state(self, state_id: str) -> None:
        """Load preserved state from disk."""
        state_path = self.checkpoint_manager.checkpoint_dir / "preserved_states" / f"{state_id}.pkl"

        if not state_path.exists():
            return

        try:
            with open(state_path, "rb") as f:
                state = pickle.load(f)
            self._preserved_states[state_id] = state
        except Exception as e:
            logger.error(f"Error loading preserved state {state_id}: {e}")
            raise
