"""
Model retraining engine for rolling backtest execution.

This module provides automated model retraining capabilities including
dynamic universe management, missing data handling, and model state
persistence across retraining cycles.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.validation.rolling_validation import RollSplit
from src.models.base import PortfolioModel
from src.utils.gpu import GPUConfig, GPUMemoryManager

logger = logging.getLogger(__name__)


@dataclass
class RetrainingConfig:
    """Configuration for model retraining engine."""

    # Data handling
    min_training_samples: int = 252
    max_missing_ratio: float = 0.1  # Max 10% missing data allowed
    forward_fill_limit: int = 5  # Max 5 days forward fill
    min_assets_for_training: int = 10

    # Model persistence
    enable_checkpointing: bool = True
    checkpoint_dir: Path | None = None
    keep_checkpoint_history: int = 5

    # Performance optimization
    enable_gpu_acceleration: bool = False
    gpu_memory_fraction: float = 0.8
    batch_training: bool = False
    max_concurrent_models: int = 1

    # Universe management
    universe_stability_threshold: float = 0.8  # Min 80% overlap between periods
    handle_universe_changes: bool = True
    rebalance_on_universe_change: bool = True


@dataclass
class UniverseChangeEvent:
    """Represents a universe membership change event."""

    date: pd.Timestamp
    added_assets: set[str] = field(default_factory=set)
    removed_assets: set[str] = field(default_factory=set)
    stability_ratio: float = 0.0
    reason: str = ""


@dataclass
class RetrainingResult:
    """Results from model retraining operation."""

    success: bool
    training_time_seconds: float
    model_checkpoint_path: Path | None = None
    training_samples: int = 0
    training_assets: int = 0
    validation_metrics: dict[str, float] = field(default_factory=dict)
    universe_changes: list[UniverseChangeEvent] = field(default_factory=list)
    memory_stats: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None


class ModelRetrainingEngine:
    """
    Automated model retraining engine for rolling backtest execution.

    This engine handles:
    - Model retraining on rolling windows with proper data validation
    - Dynamic universe membership changes for S&P MidCap 400
    - Missing data detection and handling during retraining
    - Model state persistence and checkpoint management
    - GPU memory management for intensive training operations
    """

    def __init__(self, config: RetrainingConfig):
        """
        Initialize model retraining engine.

        Args:
            config: Retraining configuration parameters
        """
        self.config = config

        # Initialize checkpoint directory
        if config.enable_checkpointing and config.checkpoint_dir:
            config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize GPU management
        self.gpu_manager = None
        if config.enable_gpu_acceleration:
            gpu_config = GPUConfig(memory_fraction=config.gpu_memory_fraction)
            self.gpu_manager = GPUMemoryManager(gpu_config)

        # State tracking
        self.retraining_history: dict[str, list[RetrainingResult]] = {}
        self.universe_history: list[UniverseChangeEvent] = []
        self.checkpoint_registry: dict[str, list[Path]] = {}

    def retrain_model(
        self,
        model: PortfolioModel,
        model_name: str,
        split: RollSplit,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame | None = None,
        universe_data: pd.DataFrame | None = None,
        previous_universe: list[str] | None = None,
    ) -> RetrainingResult:
        """
        Retrain model on current rolling window with comprehensive validation.

        Args:
            model: Portfolio model to retrain
            model_name: Model identifier
            split: Rolling split information
            train_data: Training data for current window
            val_data: Optional validation data
            universe_data: Optional dynamic universe data
            previous_universe: Previous training universe for comparison

        Returns:
            Comprehensive retraining results
        """
        logger.info(
            f"Starting retraining for {model_name} on period {split.train_period.start_date}"
        )

        start_time = pd.Timestamp.now()
        result = RetrainingResult(success=False, training_time_seconds=0.0)

        try:
            # Step 1: Handle universe changes
            current_universe, universe_changes = self._handle_universe_management(
                train_data, universe_data, previous_universe, split.train_period.start_date
            )
            result.universe_changes = universe_changes

            # Step 2: Clean and validate training data
            cleaned_data, data_stats = self._clean_and_validate_data(
                train_data, current_universe, split
            )
            result.training_samples = data_stats["samples"]
            result.training_assets = data_stats["assets"]

            # Step 3: Save checkpoint before retraining
            if self.config.enable_checkpointing:
                checkpoint_path = self._save_model_checkpoint(model, model_name, split)
                result.model_checkpoint_path = checkpoint_path

            # Step 4: Clear GPU memory if available
            if self.gpu_manager:
                self.gpu_manager.clear_cache()

            # Step 5: Execute model training
            training_success = self._execute_model_training(
                model, cleaned_data, current_universe, split
            )

            if not training_success:
                result.error_message = "Model training failed"
                return result

            # Step 6: Validate on validation data if available
            if val_data is not None and not val_data.empty:
                result.validation_metrics = self._validate_trained_model(
                    model, val_data, current_universe, split
                )

            # Step 7: Collect memory statistics
            if self.gpu_manager:
                result.memory_stats = self.gpu_manager.get_memory_stats()

            # Step 8: Update tracking
            self._update_retraining_history(model_name, result)

            result.success = True
            result.training_time_seconds = (pd.Timestamp.now() - start_time).total_seconds()

            logger.info(
                f"Successfully retrained {model_name} in {result.training_time_seconds:.2f}s"
            )

        except Exception as e:
            result.error_message = str(e)
            result.training_time_seconds = (pd.Timestamp.now() - start_time).total_seconds()
            logger.error(f"Retraining failed for {model_name}: {e}")

        return result

    def _handle_universe_management(
        self,
        train_data: pd.DataFrame,
        universe_data: pd.DataFrame | None,
        previous_universe: list[str] | None,
        current_date: pd.Timestamp,
    ) -> tuple[list[str], list[UniverseChangeEvent]]:
        """Handle dynamic universe membership changes."""

        universe_changes = []

        # Determine current universe
        if universe_data is not None:
            # Use dynamic universe if available
            universe_mask = (
                universe_data.loc[current_date]
                if current_date in universe_data.index
                else universe_data.iloc[-1]
            )
            current_universe = universe_mask[universe_mask].index.tolist()
        else:
            # Use assets available in training data
            current_universe = train_data.columns.tolist()

        # Detect universe changes
        if previous_universe is not None and self.config.handle_universe_changes:
            change_event = self._detect_universe_changes(
                previous_universe, current_universe, current_date
            )
            if change_event.added_assets or change_event.removed_assets:
                universe_changes.append(change_event)
                self.universe_history.append(change_event)
                logger.info(
                    f"Universe change detected: +{len(change_event.added_assets)}, "
                    f"-{len(change_event.removed_assets)} assets"
                )

        # Filter to assets with sufficient data
        universe_with_data = [
            asset
            for asset in current_universe
            if asset in train_data.columns and not train_data[asset].isna().all()
        ]

        if len(universe_with_data) < self.config.min_assets_for_training:
            logger.warning(
                f"Insufficient assets with data: {len(universe_with_data)} < "
                f"{self.config.min_assets_for_training}"
            )

        return universe_with_data, universe_changes

    def _detect_universe_changes(
        self,
        previous_universe: list[str],
        current_universe: list[str],
        date: pd.Timestamp,
    ) -> UniverseChangeEvent:
        """Detect changes in universe membership."""

        prev_set = set(previous_universe)
        curr_set = set(current_universe)

        added = curr_set - prev_set
        removed = prev_set - curr_set
        common = prev_set & curr_set

        # Calculate stability ratio
        stability_ratio = (
            len(common) / max(len(prev_set), len(curr_set)) if prev_set or curr_set else 1.0
        )

        # Determine reason for change
        reason = ""
        if added and removed:
            reason = f"Rotation: {len(removed)} out, {len(added)} in"
        elif added:
            reason = f"Expansion: {len(added)} assets added"
        elif removed:
            reason = f"Contraction: {len(removed)} assets removed"

        return UniverseChangeEvent(
            date=date,
            added_assets=added,
            removed_assets=removed,
            stability_ratio=stability_ratio,
            reason=reason,
        )

    def _clean_and_validate_data(
        self,
        train_data: pd.DataFrame,
        universe: list[str],
        split: RollSplit,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Clean and validate training data."""

        # Filter to universe assets
        available_assets = [asset for asset in universe if asset in train_data.columns]
        filtered_data = train_data[available_assets].copy()

        # Check data sufficiency
        if len(filtered_data) < self.config.min_training_samples:
            raise ValueError(
                f"Insufficient training samples: {len(filtered_data)} < "
                f"{self.config.min_training_samples}"
            )

        # Handle missing values
        missing_ratio = filtered_data.isna().sum().sum() / (
            filtered_data.shape[0] * filtered_data.shape[1]
        )
        if missing_ratio > self.config.max_missing_ratio:
            logger.warning(
                f"High missing data ratio: {missing_ratio:.3f} > {self.config.max_missing_ratio}"
            )

        # Forward fill limited missing values
        cleaned_data = filtered_data.ffill(limit=self.config.forward_fill_limit)

        # Drop assets with excessive missing data
        asset_completeness = cleaned_data.notna().mean()
        complete_assets = asset_completeness[
            asset_completeness > (1 - self.config.max_missing_ratio)
        ].index.tolist()

        if len(complete_assets) < self.config.min_assets_for_training:
            logger.warning(
                f"Few assets after cleaning: {len(complete_assets)} < "
                f"{self.config.min_assets_for_training}"
            )

        final_data = cleaned_data[complete_assets].dropna()

        data_stats = {
            "samples": len(final_data),
            "assets": len(final_data.columns),
            "original_samples": len(train_data),
            "original_assets": len(available_assets),
            "missing_ratio": missing_ratio,
        }

        logger.debug(
            f"Data cleaning: {data_stats['original_samples']}x{data_stats['original_assets']} -> "
            f"{data_stats['samples']}x{data_stats['assets']}"
        )

        return final_data, data_stats

    def _save_model_checkpoint(
        self,
        model: PortfolioModel,
        model_name: str,
        split: RollSplit,
    ) -> Path | None:
        """Save model checkpoint before retraining."""

        if not self.config.checkpoint_dir:
            return None

        checkpoint_filename = f"{model_name}_{split.train_period.start_date.strftime('%Y%m%d')}.pkl"
        checkpoint_path = self.config.checkpoint_dir / checkpoint_filename

        try:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(
                    {
                        "model": model,
                        "split_info": {
                            "train_start": split.train_period.start_date,
                            "train_end": split.train_period.end_date,
                            "val_start": split.validation_period.start_date,
                            "test_start": split.test_period.start_date,
                        },
                        "timestamp": pd.Timestamp.now(),
                    },
                    f,
                )

            # Update checkpoint registry
            if model_name not in self.checkpoint_registry:
                self.checkpoint_registry[model_name] = []

            self.checkpoint_registry[model_name].append(checkpoint_path)

            # Maintain checkpoint history limit
            if len(self.checkpoint_registry[model_name]) > self.config.keep_checkpoint_history:
                old_checkpoint = self.checkpoint_registry[model_name].pop(0)
                if old_checkpoint.exists():
                    old_checkpoint.unlink()

            logger.debug(f"Model checkpoint saved: {checkpoint_path}")
            return checkpoint_path

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return None

    def _execute_model_training(
        self,
        model: PortfolioModel,
        train_data: pd.DataFrame,
        universe: list[str],
        split: RollSplit,
    ) -> bool:
        """Execute model training with error handling."""

        try:
            # Execute model fitting
            model.fit(
                returns=train_data,
                universe=universe,
                fit_period=(split.train_period.start_date, split.train_period.end_date),
            )
            return True

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False

    def _validate_trained_model(
        self,
        model: PortfolioModel,
        val_data: pd.DataFrame,
        universe: list[str],
        split: RollSplit,
    ) -> dict[str, float]:
        """Validate trained model on validation data."""

        try:
            # Generate validation predictions
            val_start = split.validation_period.start_date
            val_weights = model.predict_weights(date=val_start, universe=universe)

            if val_weights.empty:
                return {"validation_error": 1.0}

            # Calculate validation performance
            val_period_data = val_data.loc[:, val_weights.index.intersection(val_data.columns)]

            if val_period_data.empty:
                return {"validation_error": 1.0}

            # Calculate portfolio returns
            aligned_weights = val_weights.reindex(val_period_data.columns, fill_value=0)
            portfolio_returns = (val_period_data * aligned_weights).sum(axis=1)

            # Calculate basic metrics
            metrics = {}
            if len(portfolio_returns) > 1:
                metrics["validation_return"] = portfolio_returns.mean() * 252  # Annualized
                metrics["validation_volatility"] = portfolio_returns.std() * np.sqrt(252)
                metrics["validation_sharpe"] = (
                    metrics["validation_return"] / metrics["validation_volatility"]
                    if metrics["validation_volatility"] > 0
                    else 0.0
                )
                metrics["validation_samples"] = len(portfolio_returns)

            return metrics

        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return {"validation_error": 1.0}

    def _update_retraining_history(self, model_name: str, result: RetrainingResult) -> None:
        """Update retraining history for model."""

        if model_name not in self.retraining_history:
            self.retraining_history[model_name] = []

        self.retraining_history[model_name].append(result)

        # Limit history size
        max_history = 50  # Keep last 50 retraining results
        if len(self.retraining_history[model_name]) > max_history:
            self.retraining_history[model_name] = self.retraining_history[model_name][-max_history:]

    def get_retraining_stats(self, model_name: str | None = None) -> dict[str, Any]:
        """Get retraining statistics for model(s)."""

        if model_name:
            if model_name not in self.retraining_history:
                return {"error": f"No history for model {model_name}"}

            history = self.retraining_history[model_name]
            successful_retrains = [r for r in history if r.success]

            return {
                "model_name": model_name,
                "total_retrains": len(history),
                "successful_retrains": len(successful_retrains),
                "success_rate": len(successful_retrains) / len(history) if history else 0.0,
                "avg_training_time": (
                    np.mean([r.training_time_seconds for r in successful_retrains])
                    if successful_retrains
                    else 0.0
                ),
                "avg_training_samples": (
                    np.mean([r.training_samples for r in successful_retrains])
                    if successful_retrains
                    else 0
                ),
                "recent_universe_changes": (
                    len(list(self.universe_history[-10:])) if self.universe_history else 0
                ),
            }
        else:
            # Aggregate stats across all models
            all_retrains = []
            for history in self.retraining_history.values():
                all_retrains.extend(history)

            successful_retrains = [r for r in all_retrains if r.success]

            return {
                "total_models": len(self.retraining_history),
                "total_retrains": len(all_retrains),
                "successful_retrains": len(successful_retrains),
                "overall_success_rate": (
                    len(successful_retrains) / len(all_retrains) if all_retrains else 0.0
                ),
                "total_universe_changes": len(self.universe_history),
                "checkpoints_saved": sum(len(paths) for paths in self.checkpoint_registry.values()),
            }

    def load_model_checkpoint(
        self, model_name: str, checkpoint_date: pd.Timestamp | None = None
    ) -> PortfolioModel | None:
        """Load model from checkpoint."""

        if not self.config.enable_checkpointing or model_name not in self.checkpoint_registry:
            return None

        checkpoints = self.checkpoint_registry[model_name]
        if not checkpoints:
            return None

        # Select checkpoint
        if checkpoint_date:
            # Find checkpoint closest to requested date
            checkpoint_path = None
            for path in checkpoints:
                # Extract date from filename
                filename = path.stem
                try:
                    date_str = filename.split("_")[-1]
                    path_date = pd.to_datetime(date_str, format="%Y%m%d")
                    if path_date <= checkpoint_date:
                        checkpoint_path = path
                except:
                    continue
        else:
            # Use most recent checkpoint
            checkpoint_path = checkpoints[-1]

        if not checkpoint_path or not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path, "rb") as f:
                checkpoint_data = pickle.load(f)
                return checkpoint_data["model"]
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def cleanup_old_checkpoints(self, keep_days: int = 30) -> int:
        """Clean up old checkpoints."""

        if not self.config.checkpoint_dir:
            return 0

        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=keep_days)
        cleaned_count = 0

        for _model_name, checkpoints in self.checkpoint_registry.items():
            for checkpoint_path in checkpoints.copy():
                try:
                    # Extract date from filename
                    filename = checkpoint_path.stem
                    date_str = filename.split("_")[-1]
                    checkpoint_date = pd.to_datetime(date_str, format="%Y%m%d")

                    if checkpoint_date < cutoff_date and checkpoint_path.exists():
                        checkpoint_path.unlink()
                        checkpoints.remove(checkpoint_path)
                        cleaned_count += 1

                except Exception as e:
                    logger.warning(f"Error cleaning checkpoint {checkpoint_path}: {e}")

        logger.info(f"Cleaned {cleaned_count} old checkpoints")
        return cleaned_count


class UniverseManager:
    """
    Manages dynamic universe membership for S&P MidCap 400 and other indices.

    Handles:
    - Universe composition tracking over time
    - Change detection and validation
    - Impact assessment of universe changes
    - Historical universe reconstruction
    """

    def __init__(self):
        """Initialize universe manager."""
        self.universe_history: dict[pd.Timestamp, set[str]] = {}
        self.change_log: list[UniverseChangeEvent] = []

    def update_universe(
        self,
        date: pd.Timestamp,
        universe_assets: list[str],
        detect_changes: bool = True,
    ) -> UniverseChangeEvent | None:
        """Update universe for given date and detect changes."""

        current_universe = set(universe_assets)
        self.universe_history[date] = current_universe

        if not detect_changes:
            return None

        # Find previous universe
        previous_dates = [d for d in self.universe_history.keys() if d < date]
        if not previous_dates:
            return None

        previous_date = max(previous_dates)
        previous_universe = self.universe_history[previous_date]

        # Detect changes
        added = current_universe - previous_universe
        removed = previous_universe - current_universe

        if added or removed:
            change_event = UniverseChangeEvent(
                date=date,
                added_assets=added,
                removed_assets=removed,
                stability_ratio=len(current_universe & previous_universe)
                / len(current_universe | previous_universe),
            )
            self.change_log.append(change_event)
            return change_event

        return None

    def get_universe_at_date(
        self,
        date: pd.Timestamp,
        method: str = "nearest",
    ) -> set[str] | None:
        """Get universe composition at specific date."""

        if date in self.universe_history:
            return self.universe_history[date]

        if method == "nearest":
            # Find nearest date
            available_dates = list(self.universe_history.keys())
            if not available_dates:
                return None

            nearest_date = min(available_dates, key=lambda d: abs((d - date).days))
            return self.universe_history[nearest_date]

        return None

    def analyze_universe_stability(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> dict[str, Any]:
        """Analyze universe stability over time period."""

        period_changes = [
            change for change in self.change_log if start_date <= change.date <= end_date
        ]

        if not period_changes:
            return {"stable_period": True, "changes": 0}

        total_additions = sum(len(change.added_assets) for change in period_changes)
        total_removals = sum(len(change.removed_assets) for change in period_changes)
        avg_stability = np.mean([change.stability_ratio for change in period_changes])

        return {
            "stable_period": len(period_changes) <= 2 and avg_stability > 0.9,
            "changes": len(period_changes),
            "total_additions": total_additions,
            "total_removals": total_removals,
            "avg_stability_ratio": avg_stability,
            "change_frequency": len(period_changes)
            / ((end_date - start_date).days / 30),  # Changes per month
        }
