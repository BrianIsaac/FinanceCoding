#!/usr/bin/env python3
"""
Unified Model Training Orchestrator with Enhanced Compatibility.

This improved version includes:
- Shared configuration management
- Universe compatibility tracking
- Enhanced metadata persistence
- Improved error handling and fallbacks

Key improvements:
- Uses shared model_config.yaml for consistency
- Saves comprehensive metadata for backtest compatibility
- Implements universe management for dynamic asset handling
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.base.portfolio_model import PortfolioConstraints
from src.models.gat.model import GATModelConfig, GATPortfolioModel
from src.models.hrp.model import HRPConfig, HRPModel
from src.models.lstm.model import LSTMModelConfig, LSTMPortfolioModel
from src.models.model_registry import ModelRegistry
from src.utils.error_handling import EnhancedErrorHandler, error_handler
from src.utils.gpu import GPUConfig, GPUMemoryManager
from src.utils.model_checkpoint import ModelCheckpointManager, ModelCheckpointMetadata
from src.utils.universe_filter import UniverseFilter
from src.utils.universe_manager import UniverseManager, create_default_manager
from src.utils.universe_validator import UniverseCompatibilityValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/train_all_models.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Ensure directories exist
Path("logs").mkdir(exist_ok=True)
Path("outputs/training").mkdir(parents=True, exist_ok=True)


class UnifiedModelTrainer:
    """
    Enhanced model trainer with improved compatibility and metadata tracking.

    Key improvements:
    - Uses shared configuration for consistency
    - Tracks universe compatibility metadata
    - Implements robust error handling
    - Saves comprehensive model state for backtesting
    """

    def __init__(
        self,
        config_path: str | Path = "configs/model_config.yaml",
        output_dir: str | Path = "outputs/training",
        quick_test: bool = False,
    ):
        """
        Initialise improved trainer with shared configuration.

        Args:
            config_path: Path to shared configuration file
            output_dir: Directory for training outputs
            quick_test: Enable quick test mode
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quick_test = quick_test

        # Initialize enhanced error handling
        self.error_handler = EnhancedErrorHandler(
            component_name="UnifiedModelTrainer",
            log_file=self.output_dir / "training_errors.log",
            enable_detailed_logging=True
        )

        # Load shared configuration with error handling
        try:
            self.config = self._load_shared_config(config_path)
        except Exception as e:
            error_context = self.error_handler.handle_error(
                error=e,
                operation="load_configuration",
                context_data={"config_path": str(config_path)},
                critical=True
            )
            raise RuntimeError(f"Configuration loading failed: {error_context.user_guidance}") from e

        # Initialise universe manager
        self.universe_manager = create_default_manager()

        # Initialise GPU management
        gpu_config = GPUConfig(max_memory_gb=self.config["compute_settings"]["gpu_memory_limit_gb"])
        self.gpu_manager = GPUMemoryManager(config=gpu_config)

        # Initialise universe validator
        self.universe_validator = UniverseCompatibilityValidator(strict_mode=False)

        # Initialise checkpoint manager
        self.checkpoint_manager = ModelCheckpointManager(self.output_dir)

        # Training results storage
        self.training_results: dict[str, Any] = {}

    def generate_training_report(self) -> dict[str, Any]:
        """Generate comprehensive training report including error analysis."""
        report = {
            "training_summary": {
                "timestamp": pd.Timestamp.now().isoformat(),
                "total_time": 0.0,  # Will be filled by caller
                "configuration": str(self.config.get("model_config", "Unknown")),
                "output_dir": str(self.output_dir)
            },
            "model_results": self.training_results,
            "error_analysis": self.error_handler.get_error_summary(),
            "universe_validation": {}  # Will be filled if validation is performed
        }

        # Save detailed error report
        self.error_handler.save_error_report(self.output_dir / "error_report.json")

        return report

    def _load_shared_config(self, config_path: str | Path) -> dict:
        """Load shared configuration from YAML."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded shared configuration from {config_path}")
        return config

    def _create_portfolio_constraints(self) -> PortfolioConstraints:
        """Create standardised portfolio constraints from config."""
        constraints_config = self.config["portfolio_constraints"]
        return PortfolioConstraints(
            long_only=constraints_config["long_only"],
            max_position_weight=constraints_config["max_position_weight"],
            max_monthly_turnover=constraints_config["max_monthly_turnover"],
            min_weight_threshold=constraints_config["min_weight_threshold"],
            top_k_positions=constraints_config["top_k_positions"],
            transaction_cost_bps=constraints_config["transaction_cost_bps"],
        )

    def _create_standardised_metadata(
        self,
        model_type: str,
        model_name: str,
        universe: list[str],
        fit_period: tuple,
        training_start_time: float,
        model_config: dict[str, Any],
        constraints: PortfolioConstraints,
        **additional_metadata
    ) -> ModelCheckpointMetadata:
        """Create standardised metadata for model checkpoints."""
        training_timestamp = pd.Timestamp.now().isoformat()

        return ModelCheckpointMetadata(
            model_type=model_type,
            model_version="2.0",  # Updated version with standardised checkpointing
            model_name=model_name,
            training_config={
                "train_start_date": "2016-01-01",
                "train_end_date": self.config["training_settings"]["train_end_date"],
                "min_data_coverage": self.config["data_settings"]["min_data_coverage"],
                "quick_test": self.quick_test,
            },
            model_config=model_config,
            constraints_config=constraints.__dict__,
            trained_universe=universe,
            universe_size=len(universe),
            universe_source="dynamic_membership_filtered",
            training_period_start=fit_period[0].isoformat(),
            training_period_end=fit_period[1].isoformat(),
            training_timestamp=training_timestamp,
            training_duration_seconds=time.time() - training_start_time,
            framework_version=torch.__version__,
            python_version=platform.python_version(),
            universe_validation=self.universe_validator.get_validation_report(),
            **additional_metadata
        )

    def _validate_lstm_dimensions(self, universe_size: int, lstm_config: dict) -> tuple[int, dict]:
        """
        Validate and adjust LSTM input dimensions based on universe size.

        Args:
            universe_size: Actual number of assets in universe
            lstm_config: LSTM configuration dictionary

        Returns:
            Tuple of (optimal_input_size, dimension_info)
        """
        min_size = lstm_config.get("min_input_size", 50)
        max_size = lstm_config.get("max_input_size", 500)
        use_dynamic = lstm_config.get("use_dynamic_input_size", True)

        dimension_info = {
            "universe_size": universe_size,
            "min_size": min_size,
            "max_size": max_size,
            "use_dynamic": use_dynamic,
            "padding_needed": 0,
            "truncation_needed": 0,
        }

        if not use_dynamic:
            # Use fixed minimum size
            optimal_size = min_size
            if universe_size < min_size:
                dimension_info["padding_needed"] = min_size - universe_size
                logger.warning(f"Universe size {universe_size} < min_size {min_size}, padding will be needed")
            elif universe_size > min_size:
                dimension_info["truncation_needed"] = universe_size - min_size
                logger.warning(f"Universe size {universe_size} > fixed_size {min_size}, truncation will be applied")
        else:
            # Dynamic sizing with constraints
            if universe_size < min_size:
                optimal_size = min_size
                dimension_info["padding_needed"] = min_size - universe_size
                logger.warning(f"Universe size {universe_size} < min_size {min_size}, using min_size with padding")
            elif universe_size > max_size:
                optimal_size = max_size
                dimension_info["truncation_needed"] = universe_size - max_size
                logger.warning(f"Universe size {universe_size} > max_size {max_size}, using max_size with truncation")
            else:
                optimal_size = universe_size
                logger.info(f"Using dynamic input size: {optimal_size} (matches universe size)")

        dimension_info["optimal_size"] = optimal_size
        return optimal_size, dimension_info

    def _load_returns_data(self) -> tuple[pd.DataFrame, list[str]]:
        """
        Load returns data and determine training universe.

        Returns:
            Tuple of (returns DataFrame, universe list)
        """
        returns_path = Path("data/final_new_pipeline/returns_daily_final.parquet")
        if not returns_path.exists():
            raise FileNotFoundError(f"Returns data not found: {returns_path}")

        returns_data = pd.read_parquet(returns_path)
        logger.info(f"Loaded returns data: {returns_data.shape}")

        # Define training period from config
        start_date = pd.Timestamp("2016-01-01")
        end_date = pd.Timestamp(self.config["training_settings"]["train_end_date"])

        # Use universe manager to get appropriate training universe
        universe = self.universe_manager.get_training_universe(
            returns_data,
            start_date,
            end_date,
            min_coverage=self.config["data_settings"]["min_data_coverage"],
        )

        logger.info(f"Training universe: {len(universe)} assets from {start_date} to {end_date}")

        # Comprehensive universe size validation
        expected_size = self.config.get("universe_settings", {}).get("expected_universe_size", 500)
        min_size = self.config.get("universe_settings", {}).get("min_assets", 50)
        max_size = self.config.get("universe_settings", {}).get("max_assets", 600)

        # Size validation with detailed feedback
        if len(universe) < min_size:
            raise ValueError(f"Universe size ({len(universe)}) below minimum required ({min_size}) for robust ML training")

        if len(universe) > max_size:
            logger.warning(f"Universe size ({len(universe)}) exceeds maximum recommended ({max_size}) - may cause memory issues")

        # Performance and statistical significance warnings
        if len(universe) < expected_size * 0.7:
            logger.warning(f"Universe size ({len(universe)}) significantly below expected ({expected_size})")
            logger.warning("This may impact model performance and statistical significance of results")

        if len(universe) > expected_size * 1.3:
            logger.warning(f"Universe size ({len(universe)}) significantly above expected ({expected_size})")
            logger.warning("Ensure sufficient computational resources and consider dimensionality reduction")

        # Performance impact warnings
        if len(universe) < 150:
            logger.warning(f"Small universe size ({len(universe)}) may impact ML model performance")
            logger.warning("Consider increasing min_data_coverage threshold or expanding universe definition")
        elif len(universe) > 550:
            logger.warning(f"Large universe size ({len(universe)}) detected")
            logger.warning("GPU memory usage will be high - monitoring recommended")

        # Log universe statistics for debugging
        logger.info(f"Universe validation: size={len(universe)}, min={min_size}, max={max_size}, expected={expected_size}")
        logger.info(f"Universe coverage: {len(universe)/expected_size:.1%} of expected size")

        # Filter returns to training period
        mask = (returns_data.index >= start_date) & (returns_data.index <= end_date)
        training_returns = returns_data.loc[mask, universe].copy()

        return training_returns, universe

    def train_hrp_models(self) -> dict[str, Any]:
        """Train HRP models with improved metadata tracking."""
        logger.info("Starting HRP model training with enhanced metadata")
        start_time = time.time()

        # Load data
        returns_data, universe = self._load_returns_data()
        constraints = self._create_portfolio_constraints()

        # Validate training universe
        validation_result = self.universe_validator.validate_training_universe(
            universe=universe,
            returns_data=returns_data,
            min_coverage=0.8
        )
        self.universe_validator.log_validation_summary(validation_result, "HRP Training")

        if not validation_result.is_valid:
            logger.error("HRP training universe validation failed")
            raise ValueError(f"Universe validation failed: {validation_result.recommendations}")

        hrp_results = {}
        hrp_config = self.config["model_configs"]["hrp"]

        # Train single HRP configuration for simplicity
        config_name = "hrp_standard"
        logger.info(f"Training HRP configuration: {config_name}")

        try:
            training_start_time = time.time()

            # Create HRP model
            model_config = HRPConfig(
                lookback_days=hrp_config["lookback_days"],
                min_observations=hrp_config["min_observations"],
                correlation_method=hrp_config["correlation_method"],
                rebalance_frequency=hrp_config["rebalance_frequency"],
            )

            model = HRPModel(constraints, model_config)

            # Fit model
            fit_period = (returns_data.index[0], returns_data.index[-1])
            model.fit(returns_data, universe, fit_period)

            # Create standardised metadata
            metadata = self._create_standardised_metadata(
                model_type="HRP",
                model_name=config_name,
                universe=universe,
                fit_period=fit_period,
                training_start_time=training_start_time,
                model_config={
                    "lookback_days": model_config.lookback_days,
                    "min_observations": model_config.min_observations,
                    "correlation_method": model_config.correlation_method,
                    "rebalance_frequency": model_config.rebalance_frequency,
                },
                constraints=constraints,
            )

            # Save model using standardised checkpoint manager
            checkpoint_dir = self.checkpoint_manager.save_model_checkpoint(
                model=model,
                metadata=metadata,
                checkpoint_name=config_name
            )

            hrp_results[config_name] = {
                "status": "success",
                "checkpoint_dir": str(checkpoint_dir),
                "training_time": time.time() - training_start_time,
                "universe_size": len(universe),
            }

            logger.info(f"HRP model trained successfully: {config_name}")

        except Exception as e:
            error_context = self.error_handler.handle_error(
                error=e,
                operation="train_hrp_model",
                context_data={
                    "config_name": config_name,
                    "universe_size": len(universe) if 'universe' in locals() else 0,
                    "training_start": start_date.isoformat() if 'start_date' in locals() else None,
                    "training_end": end_date.isoformat() if 'end_date' in locals() else None,
                    "fit_period": str(fit_period) if 'fit_period' in locals() else None
                },
                recovery_strategy="memory_cleanup"
            )

            hrp_results[config_name] = {
                "status": "failed",
                "error": str(e),
                "error_id": error_context.error_id,
                "user_guidance": error_context.user_guidance,
                "recovery_attempted": error_context.recovery_attempted,
                "recovery_successful": error_context.recovery_successful
            }

        return hrp_results

    def train_lstm_models(self) -> dict[str, Any]:
        """Train LSTM models with improved dimension handling."""
        logger.info("Starting LSTM model training with dimension management")
        start_time = time.time()

        # Load data
        returns_data, universe = self._load_returns_data()
        constraints = self._create_portfolio_constraints()

        # Validate training universe
        validation_result = self.universe_validator.validate_training_universe(
            universe=universe,
            returns_data=returns_data,
            min_coverage=0.8
        )
        self.universe_validator.log_validation_summary(validation_result, "LSTM Training")

        if not validation_result.is_valid:
            logger.error("LSTM training universe validation failed")
            raise ValueError(f"Universe validation failed: {validation_result.recommendations}")

        lstm_results = {}
        lstm_config = self.config["model_configs"]["lstm"]

        config_name = "lstm_standard"
        logger.info(f"Training LSTM configuration: {config_name}")

        try:
            # Create LSTM model configuration
            model_config = LSTMModelConfig()
            model_config.lstm_config.sequence_length = lstm_config["sequence_length"]
            model_config.lstm_config.hidden_size = lstm_config["hidden_size"]
            model_config.lstm_config.num_layers = lstm_config["num_layers"]
            model_config.lstm_config.dropout = lstm_config["dropout"]

            # Use dynamic input sizing based on actual universe
            actual_universe_size = len(universe)

            # Validate and determine optimal dimensions
            optimal_size, dimension_info = self._validate_lstm_dimensions(actual_universe_size, lstm_config)

            logger.info(f"LSTM dimension validation: universe={actual_universe_size}, optimal={optimal_size}")
            if dimension_info["padding_needed"] > 0:
                logger.info(f"Will apply zero-padding for {dimension_info['padding_needed']} missing assets")
            if dimension_info["truncation_needed"] > 0:
                logger.info(f"Will truncate to top {optimal_size} most liquid assets")

            # Smart batch size adjustment for optimal GPU utilisation
            base_batch_size = lstm_config["batch_size"]

            # Calculate memory-aware batch size with aggressive optimisation for better GPU utilisation
            if optimal_size > 600:
                # Very large universes: reduce batch size significantly
                batch_size = max(16, int(base_batch_size * 0.5))
                logger.info(f"Reduced batch size to {batch_size} for very large universe ({optimal_size} assets)")
            elif optimal_size > 500:
                # Large universes (500-600): this is our current case with 553 assets
                # Based on log analysis showing 10% GPU utilisation, we need much higher batch size
                batch_size = min(384, max(128, int(base_batch_size * 3.0)))  # Aggressive increase
                logger.info(f"Significantly increased batch size to {batch_size} for large universe ({optimal_size} assets) to improve GPU utilisation")
            elif optimal_size > 400:
                # Medium-large universes: moderate increase
                batch_size = min(256, max(96, int(base_batch_size * 2.0)))
                logger.info(f"Increased batch size to {batch_size} for medium-large universe ({optimal_size} assets)")
            elif optimal_size < 150:
                # Small universes: increase batch size for better GPU utilisation
                batch_size = min(128, int(base_batch_size * 2.0))
                logger.info(f"Increased batch size to {batch_size} for small universe ({optimal_size} assets)")
            elif optimal_size < 300:
                # Medium universes: moderate increase
                batch_size = min(96, int(base_batch_size * 1.5))
                logger.info(f"Moderately increased batch size to {batch_size} for medium universe ({optimal_size} assets)")
            else:
                # Standard size universes: still increase for better utilisation
                batch_size = min(128, max(64, int(base_batch_size * 1.5)))
                logger.info(f"Increased batch size to {batch_size} for standard universe size {optimal_size}")

            # Apply GPU memory constraints with corrected memory estimation
            gpu_memory_gb = self.config["compute_settings"]["gpu_memory_limit_gb"]

            # Corrected memory estimation for LSTM training
            # Consider: model parameters, gradients, activations, optimizer states
            model_params_gb = optimal_size * lstm_config["hidden_size"] * 4 * 4 / (1024**3)  # LSTM weights (input, forget, cell, output gates)
            activation_memory_gb = batch_size * lstm_config["sequence_length"] * optimal_size * 4 / (1024**3)  # Forward pass activations
            gradient_memory_gb = model_params_gb  # Gradients same size as parameters
            optimizer_memory_gb = model_params_gb * 2  # Adam optimizer states (momentum + variance)

            total_memory_per_batch_gb = model_params_gb + activation_memory_gb + gradient_memory_gb + optimizer_memory_gb

            # Calculate maximum safe batch size using 80% of GPU memory
            target_memory_gb = gpu_memory_gb * 0.80
            max_safe_batch = max(16, int(target_memory_gb / (total_memory_per_batch_gb / batch_size)))

            # If current batch size exceeds safe limit, reduce it
            if batch_size > max_safe_batch:
                batch_size = max_safe_batch
                logger.warning(f"Batch size reduced to {batch_size} due to GPU memory constraints")

            # For 553 assets with 10% utilisation, we can be much more aggressive
            if batch_size < 256 and optimal_size > 500:
                batch_size = min(512, batch_size * 4)  # Quadruple batch size for large universes
                logger.info(f"Aggressively increased batch size to {batch_size} for better GPU utilisation with {optimal_size} assets")

            # Log final batch size decision with realistic utilisation estimate
            final_memory_gb = total_memory_per_batch_gb * (batch_size / 64)  # Scale from base estimate
            utilisation_pct = (final_memory_gb / gpu_memory_gb) * 100
            logger.info(f"Final LSTM batch size: {batch_size} (estimated {utilisation_pct:.1f}% GPU memory utilisation: {final_memory_gb:.2f}GB/{gpu_memory_gb}GB)")

            model_config.lstm_config.input_size = optimal_size
            model_config.lstm_config.output_size = optimal_size

            # CRITICAL: Update training configuration with optimised batch size
            model_config.training_config.batch_size = batch_size
            logger.info(f"Updated training config with optimised batch size: {batch_size}")

            # Validate model dimensions
            dimension_validation = self.universe_validator.validate_model_dimensions(
                model_config=lstm_config,
                actual_universe_size=optimal_size,
                model_type="lstm"
            )
            self.universe_validator.log_validation_summary(dimension_validation, "LSTM Dimensions")

            if not dimension_validation.is_valid:
                logger.error("LSTM model dimension validation failed")
                raise ValueError(f"Model dimension validation failed: {dimension_validation.recommendations}")

            # Create model
            model = LSTMPortfolioModel(constraints, model_config)

            # Fit model
            fit_period = (returns_data.index[0], returns_data.index[-1])
            checkpoint_dir = self.output_dir / "lstm" / config_name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            model.fit(returns_data, universe, fit_period, checkpoint_dir)

            # Prepare comprehensive metadata
            metadata = {
                "model_type": "LSTM",
                "model_version": "1.1",  # Updated version with universe fixes
                "trained_universe": universe,
                "universe_size": len(universe),
                "universe_source": "dynamic_membership_filtered",
                "data_coverage_threshold": self.config["data_settings"]["min_data_coverage"],
                "network_dimensions": {
                    "input_size": model_config.lstm_config.input_size,
                    "output_size": model_config.lstm_config.output_size,
                    "hidden_size": model_config.lstm_config.hidden_size,
                    "sequence_length": model_config.lstm_config.sequence_length,
                    "batch_size": batch_size,
                },
                "dimension_info": dimension_info,
                "training_period": {
                    "start": fit_period[0].isoformat(),
                    "end": fit_period[1].isoformat(),
                },
                "training_config": {
                    "epochs": lstm_config["epochs"],
                    "patience": lstm_config["patience"],
                    "learning_rate": lstm_config["learning_rate"],
                },
                "constraints": constraints.__dict__,
                "training_timestamp": pd.Timestamp.now().isoformat(),
            }

            # Save model with metadata
            checkpoint_path = checkpoint_dir / "model_best.pt"
            model.save_model(checkpoint_path)

            # Save metadata separately
            metadata_path = checkpoint_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            lstm_results[config_name] = {
                "status": "success",
                "checkpoint_path": str(checkpoint_path),
                "metadata_path": str(metadata_path),
                "training_time": time.time() - start_time,
                "universe_size": len(universe),
                "network_dims": f"{optimal_size}x{lstm_config['hidden_size']}",
            }

            logger.info(f"LSTM model trained successfully: {config_name}")

        except Exception as e:
            logger.error(f"Failed to train LSTM model: {e}", exc_info=True)

            # Detailed error information for debugging
            error_details = {
                "status": "failed",
                "error": str(e),
                "error_type": type(e).__name__,
                "config_name": config_name,
                "universe_size": len(universe) if 'universe' in locals() else 0,
                "timestamp": pd.Timestamp.now().isoformat(),
            }

            # Add dimension info if available
            if 'dimension_info' in locals():
                error_details["dimension_info"] = dimension_info

            # Add specific recommendations based on error type
            if "NameError" in str(type(e)):
                error_details["recommendation"] = "Check variable names and configuration keys"
            elif "dimension" in str(e).lower() or "size" in str(e).lower():
                error_details["recommendation"] = "Check input/output dimension compatibility"
            elif "memory" in str(e).lower() or "cuda" in str(e).lower():
                error_details["recommendation"] = "Reduce batch size or sequence length"
            else:
                error_details["recommendation"] = "Check model configuration and data format"

            lstm_results[config_name] = error_details

        return lstm_results

    def train_gat_models(self) -> dict[str, Any]:
        """Train GAT models with improved universe tracking."""
        logger.info("Starting GAT model training with universe alignment")
        start_time = time.time()

        # Load data
        returns_data, universe = self._load_returns_data()
        constraints = self._create_portfolio_constraints()

        # Validate training universe
        validation_result = self.universe_validator.validate_training_universe(
            universe=universe,
            returns_data=returns_data,
            min_coverage=0.8
        )
        self.universe_validator.log_validation_summary(validation_result, "GAT Training")

        if not validation_result.is_valid:
            logger.error("GAT training universe validation failed")
            raise ValueError(f"Universe validation failed: {validation_result.recommendations}")

        gat_results = {}
        gat_config = self.config["model_configs"]["gat"]

        config_name = "gat_standard"
        logger.info(f"Training GAT configuration: {config_name}")

        try:
            # Create GAT model configuration
            model_config = GATModelConfig()
            model_config.input_features = gat_config["input_features"]
            model_config.hidden_dim = gat_config["hidden_dim"]
            model_config.num_layers = gat_config["num_layers"]
            model_config.num_attention_heads = gat_config["num_attention_heads"]
            model_config.dropout = gat_config["dropout"]
            model_config.learning_rate = gat_config["learning_rate"]
            model_config.batch_size = gat_config["batch_size"]
            model_config.max_epochs = 10 if self.quick_test else 100

            # Validate model dimensions
            dimension_validation = self.universe_validator.validate_model_dimensions(
                model_config=gat_config,
                actual_universe_size=len(universe),
                model_type="gat"
            )
            self.universe_validator.log_validation_summary(dimension_validation, "GAT Dimensions")

            if not dimension_validation.is_valid:
                logger.error("GAT model dimension validation failed")
                raise ValueError(f"Model dimension validation failed: {dimension_validation.recommendations}")

            # Create model
            model = GATPortfolioModel(constraints, model_config)

            # Fit model
            fit_period = (returns_data.index[0], returns_data.index[-1])
            model.fit(returns_data, universe, fit_period)

            # Prepare comprehensive metadata
            metadata = {
                "model_type": "GAT",
                "model_version": "1.0",
                "trained_universe": universe,
                "universe_size": len(universe),
                "graph_config": {
                    "lookback_days": gat_config["lookback_days"],
                    "input_features": gat_config["input_features"],
                },
                "training_period": {
                    "start": fit_period[0].isoformat(),
                    "end": fit_period[1].isoformat(),
                },
                "constraints": constraints.__dict__,
                "training_timestamp": pd.Timestamp.now().isoformat(),
            }

            # Save model with metadata
            checkpoint_dir = self.output_dir / "gat" / config_name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / "model_best.pt"

            model.save_model(str(checkpoint_path))

            # Save metadata separately
            metadata_path = checkpoint_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            gat_results[config_name] = {
                "status": "success",
                "checkpoint_path": str(checkpoint_path),
                "metadata_path": str(metadata_path),
                "training_time": time.time() - start_time,
                "universe_size": len(universe),
            }

            logger.info(f"GAT model trained successfully: {config_name}")

        except Exception as e:
            logger.error(f"Failed to train GAT model: {e}", exc_info=True)

            # Enhanced error information for debugging
            error_details = {
                "status": "failed",
                "error": str(e),
                "error_type": type(e).__name__,
                "config_name": config_name,
                "universe_size": len(universe) if 'universe' in locals() else 0,
                "timestamp": pd.Timestamp.now().isoformat(),
            }

            # Add specific recommendations based on error type
            if "tensor" in str(e).lower() and "size" in str(e).lower():
                error_details["recommendation"] = "Check tensor dimension compatibility between nodes and edges"
                error_details["likely_cause"] = "GAT model tensor dimension mismatch"
            elif "memory" in str(e).lower() or "cuda" in str(e).lower():
                error_details["recommendation"] = "Reduce batch size or universe size"
                error_details["likely_cause"] = "GPU memory overflow"
            elif "graph" in str(e).lower():
                error_details["recommendation"] = "Check graph construction and edge connectivity"
                error_details["likely_cause"] = "Graph structure issues"
            else:
                error_details["recommendation"] = "Check GAT model configuration and data format"
                error_details["likely_cause"] = "Configuration or data compatibility issue"

            gat_results[config_name] = error_details

        return gat_results

    def run_training_pipeline(self, models: list[str] | None = None) -> dict[str, Any]:
        """
        Execute complete training pipeline.

        Args:
            models: List of models to train (default: all)

        Returns:
            Training results dictionary
        """
        logger.info("=" * 80)
        logger.info("STARTING UNIFIED MODEL TRAINING PIPELINE")
        logger.info("=" * 80)

        start_time = time.time()

        if models is None:
            models = ["hrp", "lstm", "gat"]

        all_results = {}

        # Train each model type
        if "hrp" in models:
            logger.info("Training HRP models...")
            all_results["hrp"] = self.train_hrp_models()

        if "lstm" in models:
            logger.info("Training LSTM models...")
            all_results["lstm"] = self.train_lstm_models()

        if "gat" in models:
            logger.info("Training GAT models...")
            all_results["gat"] = self.train_gat_models()

        total_time = time.time() - start_time

        # Validate training results before generating report
        training_success = self._validate_training_results(all_results)

        # Generate training report
        self._generate_training_report(all_results, total_time)

        # Generate universe validation report
        validation_report = self.universe_validator.get_validation_report()
        logger.info("=" * 50)
        logger.info("UNIVERSE VALIDATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total validations: {validation_report['total_validations']}")
        logger.info(f"Errors: {validation_report['errors']}")
        logger.info(f"Warnings: {validation_report['warnings']}")
        logger.info(f"Success rate: {validation_report['success_rate']:.1%}")
        if validation_report['recent_issues']:
            logger.info("Recent issues:")
            for issues in validation_report['recent_issues']:
                for issue in issues:
                    logger.info(f"  - {issue}")

        logger.info("=" * 80)
        logger.info(f"Training pipeline completed in {total_time:.2f}s")
        logger.info("=" * 80)

        return all_results

    def _validate_training_results(self, results: dict[str, Any]) -> bool:
        """
        Validate training results and provide comprehensive feedback.

        Args:
            results: Dictionary containing training results for all models

        Returns:
            True if training was successful enough to proceed, False otherwise
        """
        logger.info("=" * 50)
        logger.info("TRAINING RESULTS VALIDATION")
        logger.info("=" * 50)

        total_models = 0
        successful_models = 0
        failed_models = []
        critical_failures = []

        for model_type, model_results in results.items():
            for model_name, result in model_results.items():
                total_models += 1
                if result.get("status") == "success":
                    successful_models += 1
                    logger.info(f"✓ {model_type.upper()} {model_name}: SUCCESS")
                else:
                    failed_models.append(f"{model_type.upper()} {model_name}")
                    error_msg = result.get("error", "Unknown error")
                    logger.error(f"✗ {model_type.upper()} {model_name}: FAILED - {error_msg}")

                    # Check for critical failures that should stop pipeline
                    if "tensor" in error_msg.lower() and "dimension" in error_msg.lower():
                        critical_failures.append(f"{model_type.upper()} {model_name}: Tensor dimension mismatch")
                    elif "memory" in error_msg.lower() and "cuda" in error_msg.lower():
                        critical_failures.append(f"{model_type.upper()} {model_name}: GPU memory issue")

        # Summary statistics
        success_rate = successful_models / total_models if total_models > 0 else 0
        logger.info(f"Success rate: {successful_models}/{total_models} ({success_rate:.1%})")

        # Determine if training was successful enough to proceed
        pipeline_success = True

        if success_rate < 0.5:
            logger.error("CRITICAL: Less than 50% of models trained successfully")
            pipeline_success = False

        if critical_failures:
            logger.error("CRITICAL: Critical failures detected:")
            for failure in critical_failures:
                logger.error(f"  - {failure}")
            pipeline_success = False

        # Check if at least one model from each type succeeded (if attempted)
        for model_type in results:
            type_success = any(
                result.get("status") == "success"
                for result in results[model_type].values()
            )
            if not type_success:
                logger.warning(f"WARNING: No successful {model_type.upper()} models")
                if model_type in ["hrp", "lstm"]:  # Critical models
                    logger.error(f"CRITICAL: {model_type.upper()} is required for backtesting")
                    pipeline_success = False

        if pipeline_success:
            logger.info("✓ Training validation PASSED - pipeline can proceed")
        else:
            logger.error("✗ Training validation FAILED - review errors before proceeding")
            logger.error("Consider fixing critical issues or running with reduced model set")

        return pipeline_success

    def _generate_training_report(self, results: dict[str, Any], total_time: float) -> None:
        """Generate comprehensive training report."""
        report = {
            "training_summary": {
                "timestamp": pd.Timestamp.now().isoformat(),
                "total_time": total_time,
                "configuration": "configs/model_config.yaml",
                "output_dir": str(self.output_dir),
            },
            "model_results": results,
            "success_summary": {},
            "universe_validation": self.universe_validator.get_validation_report(),
        }

        # Calculate success rates
        for model_type, model_results in results.items():
            success_count = sum(1 for r in model_results.values() if r.get("status") == "success")
            total_count = len(model_results)
            report["success_summary"][model_type] = {
                "successful": success_count,
                "total": total_count,
                "success_rate": success_count / total_count if total_count > 0 else 0,
            }

        # Save report
        report_path = self.output_dir / "training_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Training report saved to {report_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Unified Model Training Orchestrator with Enhanced Compatibility"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to shared configuration file",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="hrp,lstm,gat",
        help="Comma-separated list of models to train",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/training",
        help="Output directory for training results",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Enable quick test mode",
    )

    args = parser.parse_args()

    # Parse models list
    models_to_train = [m.strip().lower() for m in args.models.split(",")]

    try:
        # Initialise trainer
        trainer = UnifiedModelTrainer(
            config_path=args.config,
            output_dir=args.output_dir,
            quick_test=args.quick_test,
        )

        # Run training
        trainer.run_training_pipeline(models=models_to_train)

        logger.info("Training completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())