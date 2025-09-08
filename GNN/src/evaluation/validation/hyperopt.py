"""
Hyperparameter optimization framework for portfolio optimization models.

This module provides comprehensive hyperparameter optimization using validation-based
approaches with temporal constraints and integration with Optuna for systematic tuning.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import CmaEsSampler, TPESampler

from src.evaluation.validation.rolling_validation import (
    RollingValidationEngine,
    RollSplit,
)
from src.utils.gpu import GPUConfig, GPUMemoryManager

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterSpace:
    """Define hyperparameter search space for optimization."""

    name: str
    param_type: str  # 'float', 'int', 'categorical'
    low: float | None = None
    high: float | None = None
    choices: list[Any] | None = None
    log_scale: bool = False

    def __post_init__(self) -> None:
        """Validate hyperparameter space definition."""
        if self.param_type in ["float", "int"] and (self.low is None or self.high is None):
            raise ValueError(f"Numeric parameter {self.name} requires low and high bounds")
        if self.param_type == "categorical" and not self.choices:
            raise ValueError(f"Categorical parameter {self.name} requires choices")


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""

    # Optuna settings
    n_trials: int = 100
    n_startup_trials: int = 10
    n_warmup_steps: int = 5
    timeout_seconds: int | None = None

    # Validation settings
    validation_metric: str = "sharpe_ratio"
    maximize_metric: bool = True
    early_stopping_patience: int = 20
    min_trials_for_pruning: int = 5

    # Resource constraints
    max_memory_gb: float = 11.0
    enable_gpu_optimization: bool = True
    parallel_trials: int = 1

    # Reproducibility
    random_seed: int = 42

    def validate_config(self) -> None:
        """Validate optimization configuration."""
        if self.n_trials <= 0:
            raise ValueError("Number of trials must be positive")
        if self.validation_metric not in ["sharpe_ratio", "return", "volatility", "max_drawdown"]:
            raise ValueError(f"Invalid validation metric: {self.validation_metric}")


class ValidationBasedOptimizer:
    """
    Systematic hyperparameter optimizer with temporal validation constraints.

    This optimizer uses rolling validation to evaluate hyperparameters while
    respecting temporal data integrity and providing robust optimization.
    """

    def __init__(
        self,
        validation_engine: RollingValidationEngine,
        config: OptimizationConfig,
        gpu_config: GPUConfig | None = None,
    ):
        """
        Initialize validation-based optimizer.

        Args:
            validation_engine: Rolling validation engine
            config: Optimization configuration
            gpu_config: Optional GPU configuration
        """
        config.validate_config()
        self.validation_engine = validation_engine
        self.config = config
        self.gpu_manager = GPUMemoryManager(gpu_config or GPUConfig()) if gpu_config else None

        # Optimization state
        self.study: optuna.Study | None = None
        self.hyperparameter_spaces: dict[str, HyperparameterSpace] = {}
        self.optimization_history: list[dict[str, Any]] = []
        self.best_params: dict[str, Any] | None = None
        self.best_score: float | None = None

    def define_hyperparameter_space(self, spaces: list[HyperparameterSpace]) -> None:
        """
        Define hyperparameter search spaces.

        Args:
            spaces: List of hyperparameter spaces to optimize
        """
        self.hyperparameter_spaces = {space.name: space for space in spaces}
        logger.info(f"Defined hyperparameter spaces: {list(self.hyperparameter_spaces.keys())}")

    def create_study(
        self, study_name: str, storage: str | None = None, load_if_exists: bool = True
    ) -> optuna.Study:
        """
        Create or load Optuna study for optimization.

        Args:
            study_name: Name of the optimization study
            storage: Optional storage URL for persistence
            load_if_exists: Whether to load existing study if it exists

        Returns:
            Optuna study object
        """
        # Configure sampler based on problem characteristics
        if len(self.hyperparameter_spaces) <= 5:
            sampler = TPESampler(
                n_startup_trials=self.config.n_startup_trials,
                n_warmup_steps=self.config.n_warmup_steps,
                seed=self.config.random_seed,
            )
        else:
            # Use CMA-ES for higher dimensional problems
            sampler = CmaEsSampler(seed=self.config.random_seed)

        # Configure pruner for early stopping of unpromising trials
        pruner = MedianPruner(
            n_startup_trials=self.config.min_trials_for_pruning,
            n_warmup_steps=self.config.n_warmup_steps,
        )

        direction = "maximize" if self.config.maximize_metric else "minimize"

        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=load_if_exists,
        )

        logger.info(f"Created study '{study_name}' with direction '{direction}'")
        return self.study

    def _suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, Any]:
        """
        Suggest hyperparameters for a trial based on defined spaces.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested hyperparameters
        """
        suggested_params = {}

        for name, space in self.hyperparameter_spaces.items():
            if space.param_type == "float":
                if space.log_scale:
                    suggested_params[name] = trial.suggest_float(
                        name, space.low, space.high, log=True
                    )
                else:
                    suggested_params[name] = trial.suggest_float(name, space.low, space.high)
            elif space.param_type == "int":
                if space.log_scale:
                    suggested_params[name] = trial.suggest_int(
                        name, int(space.low), int(space.high), log=True
                    )
                else:
                    suggested_params[name] = trial.suggest_int(
                        name, int(space.low), int(space.high)
                    )
            elif space.param_type == "categorical":
                suggested_params[name] = trial.suggest_categorical(name, space.choices)

        return suggested_params

    def _evaluate_hyperparameters(
        self,
        params: dict[str, Any],
        model_factory: Callable,
        data_timestamps: list[pd.Timestamp],
        trial: optuna.Trial | None = None,
    ) -> float:
        """
        Evaluate hyperparameters using rolling validation.

        Args:
            params: Hyperparameters to evaluate
            model_factory: Function that creates model with given parameters
            data_timestamps: Available data timestamps
            trial: Optional Optuna trial for pruning

        Returns:
            Validation score
        """
        # Generate rolling splits
        splits = self.validation_engine.generate_rolling_windows(data_timestamps)

        if not splits:
            raise ValueError("No valid splits generated for hyperparameter evaluation")

        split_scores = []

        for split_idx, split in enumerate(splits):
            try:
                # Create model with current hyperparameters
                model = model_factory(**params)

                # GPU memory optimization if enabled
                if self.gpu_manager and self.config.enable_gpu_optimization:
                    # Monitor memory usage during evaluation
                    memory_stats = self.gpu_manager.get_memory_stats()
                    logger.debug(f"Memory before split {split_idx}: {memory_stats}")

                # Validate temporal integrity
                integrity_results = self.validation_engine.validate_temporal_integrity(
                    split, data_timestamps
                )

                if not all(integrity_results.values()):
                    logger.warning(f"Temporal integrity violation in split {split_idx}")
                    continue

                # Evaluate model on this split
                split_score = self._evaluate_single_split(model, split, data_timestamps)
                split_scores.append(split_score)

                # Intermediate reporting for pruning
                if trial is not None and len(split_scores) > self.config.min_trials_for_pruning:
                    intermediate_score = np.mean(split_scores)
                    trial.report(intermediate_score, split_idx)

                    # Check if trial should be pruned
                    if trial.should_prune():
                        logger.info(f"Trial pruned at split {split_idx}")
                        raise optuna.TrialPruned()

            except Exception as e:
                logger.error(f"Error evaluating split {split_idx}: {e}")
                # Continue with other splits rather than failing completely
                continue

        if not split_scores:
            raise ValueError("No valid split scores obtained")

        # Aggregate scores across splits
        final_score = self._aggregate_split_scores(split_scores)

        logger.debug(f"Hyperparameters {params} achieved score {final_score}")
        return final_score

    def _evaluate_single_split(
        self, model: Any, split: RollSplit, data_timestamps: list[pd.Timestamp]
    ) -> float:
        """
        Evaluate model on a single validation split.

        Args:
            model: Model to evaluate
            split: Rolling split for validation
            data_timestamps: Available data timestamps

        Returns:
            Performance score for this split
        """
        # This is a placeholder implementation
        # In practice, this would:
        # 1. Train model on training period
        # 2. Validate on validation period
        # 3. Return the validation metric (Sharpe ratio, etc.)

        # For now, return a mock score based on hyperparameters
        # Real implementation would integrate with actual model training

        # Extract validation period data
        val_data = [ts for ts in data_timestamps if split.validation_period.contains_date(ts)]

        if len(val_data) < 20:  # Minimum data requirement
            return float("-inf") if self.config.maximize_metric else float("inf")

        # Mock evaluation - replace with actual model evaluation
        mock_score = np.random.normal(0.1, 0.05)  # Mock Sharpe ratio

        return mock_score

    def _aggregate_split_scores(self, scores: list[float]) -> float:
        """
        Aggregate scores across multiple validation splits.

        Args:
            scores: List of scores from different splits

        Returns:
            Aggregated score
        """
        if not scores:
            return float("-inf") if self.config.maximize_metric else float("inf")

        # Use mean as default aggregation, but could be configurable
        return float(np.mean(scores))

    def optimize(
        self,
        model_factory: Callable,
        data_timestamps: list[pd.Timestamp],
        study_name: str = "hyperopt_study",
    ) -> dict[str, Any]:
        """
        Execute hyperparameter optimization.

        Args:
            model_factory: Function that creates model with given parameters
            data_timestamps: Available data timestamps
            study_name: Name for the optimization study

        Returns:
            Optimization results
        """
        if not self.hyperparameter_spaces:
            raise ValueError("No hyperparameter spaces defined")

        # Create study if not exists
        if self.study is None:
            self.create_study(study_name)

        def objective(trial: optuna.Trial) -> float:
            """Objective function for Optuna optimization."""
            params = self._suggest_hyperparameters(trial)

            try:
                score = self._evaluate_hyperparameters(
                    params, model_factory, data_timestamps, trial
                )

                # Record trial in history
                self.optimization_history.append(
                    {
                        "trial_number": trial.number,
                        "params": params,
                        "score": score,
                        "timestamp": pd.Timestamp.now(),
                        "status": "completed",
                    }
                )

                return score

            except optuna.TrialPruned:
                # Record pruned trial
                self.optimization_history.append(
                    {
                        "trial_number": trial.number,
                        "params": params,
                        "score": None,
                        "timestamp": pd.Timestamp.now(),
                        "status": "pruned",
                    }
                )
                raise
            except Exception as e:
                # Record failed trial
                self.optimization_history.append(
                    {
                        "trial_number": trial.number,
                        "params": params,
                        "score": None,
                        "timestamp": pd.Timestamp.now(),
                        "status": "failed",
                        "error": str(e),
                    }
                )
                raise

        # Run optimization
        logger.info(f"Starting optimization with {self.config.n_trials} trials")

        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_seconds,
            n_jobs=self.config.parallel_trials,
        )

        # Store best results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        logger.info(f"Optimization completed. Best score: {self.best_score}")
        logger.info(f"Best parameters: {self.best_params}")

        return self.get_optimization_results()

    def get_optimization_results(self) -> dict[str, Any]:
        """Get comprehensive optimization results."""
        if self.study is None:
            return {"status": "No optimization run yet"}

        trials_df = self.study.trials_dataframe()

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": len(self.study.trials),
            "n_completed_trials": len(
                [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            ),
            "n_pruned_trials": len(
                [t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]
            ),
            "n_failed_trials": len(
                [t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL]
            ),
            "optimization_history": self.optimization_history,
            "hyperparameter_importance": self._get_hyperparameter_importance(),
            "trials_dataframe": trials_df.to_dict("records") if not trials_df.empty else [],
        }

    def _get_hyperparameter_importance(self) -> dict[str, float]:
        """Calculate hyperparameter importance scores."""
        if self.study is None or len(self.study.trials) < 10:
            return {}

        try:
            importance = optuna.importance.get_param_importances(self.study)
            return {k: float(v) for k, v in importance.items()}
        except Exception as e:
            logger.warning(f"Could not calculate parameter importance: {e}")
            return {}

    def export_optimization_report(self, output_path: Path) -> None:
        """Export comprehensive optimization report."""
        results = self.get_optimization_results()

        report = {
            "generated_at": pd.Timestamp.now().isoformat(),
            "configuration": {
                "n_trials": self.config.n_trials,
                "validation_metric": self.config.validation_metric,
                "maximize_metric": self.config.maximize_metric,
                "random_seed": self.config.random_seed,
            },
            "hyperparameter_spaces": {
                name: {
                    "type": space.param_type,
                    "low": space.low,
                    "high": space.high,
                    "choices": space.choices,
                    "log_scale": space.log_scale,
                }
                for name, space in self.hyperparameter_spaces.items()
            },
            "results": results,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Optimization report exported to {output_path}")


# Model-specific hyperparameter space definitions
class ModelHyperparameterSpaces:
    """
    Pre-defined hyperparameter spaces for different portfolio optimization models.

    This class provides model-specific hyperparameter spaces based on best practices
    and domain knowledge for HRP, LSTM, and GAT models.
    """

    @staticmethod
    def get_hrp_hyperparameter_space() -> list[HyperparameterSpace]:
        """
        Get hyperparameter space for Hierarchical Risk Parity (HRP) model.

        Returns:
            List of hyperparameter spaces for HRP model
        """
        return [
            HyperparameterSpace(
                name="linkage_method",
                param_type="categorical",
                choices=["single", "complete", "average", "ward"],
            ),
            HyperparameterSpace(
                name="distance_metric",
                param_type="categorical",
                choices=["euclidean", "manhattan", "cosine", "correlation"],
            ),
            HyperparameterSpace(name="lookback_window", param_type="int", low=60, high=252),
            HyperparameterSpace(name="min_periods_ratio", param_type="float", low=0.5, high=1.0),
            HyperparameterSpace(
                name="correlation_threshold", param_type="float", low=0.1, high=0.9
            ),
            HyperparameterSpace(
                name="rebalance_frequency", param_type="categorical", choices=["D", "W", "M", "Q"]
            ),
            HyperparameterSpace(
                name="risk_budget_method",
                param_type="categorical",
                choices=["equal_risk", "volatility_weighted", "inverse_variance"],
            ),
        ]

    @staticmethod
    def get_lstm_hyperparameter_space() -> list[HyperparameterSpace]:
        """
        Get hyperparameter space for LSTM temporal pattern recognition model.

        Returns:
            List of hyperparameter spaces for LSTM model
        """
        return [
            HyperparameterSpace(name="hidden_size", param_type="int", low=32, high=512),
            HyperparameterSpace(name="num_layers", param_type="int", low=1, high=4),
            HyperparameterSpace(name="dropout_rate", param_type="float", low=0.0, high=0.5),
            HyperparameterSpace(name="sequence_length", param_type="int", low=10, high=60),
            HyperparameterSpace(
                name="learning_rate", param_type="float", low=1e-5, high=1e-2, log_scale=True
            ),
            HyperparameterSpace(
                name="batch_size", param_type="categorical", choices=[8, 16, 32, 64, 128]
            ),
            HyperparameterSpace(
                name="weight_decay", param_type="float", low=1e-6, high=1e-2, log_scale=True
            ),
            HyperparameterSpace(name="gradient_clipping", param_type="float", low=0.1, high=2.0),
            HyperparameterSpace(
                name="optimizer_type",
                param_type="categorical",
                choices=["adam", "adamw", "rmsprop", "sgd"],
            ),
            HyperparameterSpace(name="scheduler_patience", param_type="int", low=5, high=25),
            HyperparameterSpace(name="early_stopping_patience", param_type="int", low=10, high=50),
        ]

    @staticmethod
    def get_gat_hyperparameter_space() -> list[HyperparameterSpace]:
        """
        Get hyperparameter space for Graph Attention Network (GAT) model.

        Returns:
            List of hyperparameter spaces for GAT model
        """
        return [
            HyperparameterSpace(name="hidden_dim", param_type="int", low=32, high=256),
            HyperparameterSpace(name="num_heads", param_type="categorical", choices=[1, 2, 4, 8]),
            HyperparameterSpace(name="num_layers", param_type="int", low=1, high=4),
            HyperparameterSpace(name="dropout_rate", param_type="float", low=0.0, high=0.6),
            HyperparameterSpace(name="attention_dropout", param_type="float", low=0.0, high=0.3),
            HyperparameterSpace(
                name="learning_rate", param_type="float", low=1e-5, high=1e-2, log_scale=True
            ),
            HyperparameterSpace(
                name="weight_decay", param_type="float", low=1e-6, high=1e-2, log_scale=True
            ),
            HyperparameterSpace(
                name="graph_construction_method",
                param_type="categorical",
                choices=["correlation", "mutual_information", "distance", "sector"],
            ),
            HyperparameterSpace(name="edge_threshold", param_type="float", low=0.1, high=0.8),
            HyperparameterSpace(
                name="aggregation_method",
                param_type="categorical",
                choices=["mean", "max", "sum", "attention"],
            ),
            HyperparameterSpace(
                name="residual_connections", param_type="categorical", choices=[True, False]
            ),
            HyperparameterSpace(name="layer_norm", param_type="categorical", choices=[True, False]),
            HyperparameterSpace(
                name="batch_size",
                param_type="categorical",
                choices=[4, 8, 16, 32],  # Smaller for GAT due to memory requirements
            ),
            HyperparameterSpace(
                name="gradient_accumulation_steps", param_type="categorical", choices=[2, 4, 8, 16]
            ),
        ]

    @staticmethod
    def get_unified_constraint_hyperparameter_space() -> list[HyperparameterSpace]:
        """
        Get hyperparameter space for unified constraint system parameters.

        Returns:
            List of hyperparameter spaces for constraint parameters
        """
        return [
            HyperparameterSpace(name="max_weight", param_type="float", low=0.05, high=0.5),
            HyperparameterSpace(name="min_weight", param_type="float", low=0.0, high=0.02),
            HyperparameterSpace(name="max_sector_weight", param_type="float", low=0.15, high=0.8),
            HyperparameterSpace(name="turnover_penalty", param_type="float", low=0.0, high=0.1),
            HyperparameterSpace(
                name="transaction_cost_bps", param_type="float", low=1.0, high=20.0
            ),
            HyperparameterSpace(
                name="risk_aversion", param_type="float", low=0.1, high=10.0, log_scale=True
            ),
            HyperparameterSpace(name="leverage_limit", param_type="float", low=1.0, high=2.0),
            HyperparameterSpace(
                name="diversification_threshold", param_type="int", low=10, high=100
            ),
        ]

    @staticmethod
    def get_composite_model_space(model_types: list[str]) -> list[HyperparameterSpace]:
        """
        Get combined hyperparameter space for multiple model types.

        Args:
            model_types: List of model types to include ('hrp', 'lstm', 'gat')

        Returns:
            Combined hyperparameter spaces with model-specific prefixes
        """
        combined_spaces = []

        model_space_map = {
            "hrp": ModelHyperparameterSpaces.get_hrp_hyperparameter_space,
            "lstm": ModelHyperparameterSpaces.get_lstm_hyperparameter_space,
            "gat": ModelHyperparameterSpaces.get_gat_hyperparameter_space,
        }

        for model_type in model_types:
            if model_type not in model_space_map:
                logger.warning(f"Unknown model type: {model_type}")
                continue

            model_spaces = model_space_map[model_type]()

            # Add model prefix to parameter names to avoid conflicts
            for space in model_spaces:
                prefixed_space = HyperparameterSpace(
                    name=f"{model_type}_{space.name}",
                    param_type=space.param_type,
                    low=space.low,
                    high=space.high,
                    choices=space.choices,
                    log_scale=space.log_scale,
                )
                combined_spaces.append(prefixed_space)

        # Add unified constraint parameters (shared across models)
        combined_spaces.extend(
            ModelHyperparameterSpaces.get_unified_constraint_hyperparameter_space()
        )

        return combined_spaces

    @staticmethod
    def get_reduced_space(model_type: str, max_params: int = 10) -> list[HyperparameterSpace]:
        """
        Get reduced hyperparameter space for faster optimization.

        Args:
            model_type: Type of model ('hrp', 'lstm', 'gat')
            max_params: Maximum number of parameters to include

        Returns:
            Reduced hyperparameter space with most important parameters
        """
        full_spaces = {
            "hrp": ModelHyperparameterSpaces.get_hrp_hyperparameter_space(),
            "lstm": ModelHyperparameterSpaces.get_lstm_hyperparameter_space(),
            "gat": ModelHyperparameterSpaces.get_gat_hyperparameter_space(),
        }

        if model_type not in full_spaces:
            raise ValueError(f"Unknown model type: {model_type}")

        # Priority order for parameter importance (model-specific)
        priority_params = {
            "hrp": [
                "linkage_method",
                "lookback_window",
                "distance_metric",
                "correlation_threshold",
                "rebalance_frequency",
            ],
            "lstm": [
                "hidden_size",
                "num_layers",
                "learning_rate",
                "sequence_length",
                "dropout_rate",
                "batch_size",
                "weight_decay",
            ],
            "gat": [
                "hidden_dim",
                "num_heads",
                "learning_rate",
                "num_layers",
                "dropout_rate",
                "batch_size",
                "edge_threshold",
            ],
        }

        full_space_dict = {space.name: space for space in full_spaces[model_type]}
        priority_list = priority_params[model_type]

        reduced_spaces = []
        for param_name in priority_list[:max_params]:
            if param_name in full_space_dict:
                reduced_spaces.append(full_space_dict[param_name])

        return reduced_spaces


# Convenience functions for quick setup
def create_hrp_optimizer(
    validation_engine: RollingValidationEngine, n_trials: int = 50, reduced_space: bool = False
) -> ValidationBasedOptimizer:
    """Create HRP model optimizer with default settings."""
    config = OptimizationConfig(n_trials=n_trials, validation_metric="sharpe_ratio")
    optimizer = ValidationBasedOptimizer(validation_engine, config)

    if reduced_space:
        spaces = ModelHyperparameterSpaces.get_reduced_space("hrp")
    else:
        spaces = ModelHyperparameterSpaces.get_hrp_hyperparameter_space()

    optimizer.define_hyperparameter_space(spaces)
    return optimizer


def create_lstm_optimizer(
    validation_engine: RollingValidationEngine, n_trials: int = 100, reduced_space: bool = False
) -> ValidationBasedOptimizer:
    """Create LSTM model optimizer with default settings."""
    config = OptimizationConfig(n_trials=n_trials, validation_metric="sharpe_ratio")
    optimizer = ValidationBasedOptimizer(validation_engine, config)

    if reduced_space:
        spaces = ModelHyperparameterSpaces.get_reduced_space("lstm")
    else:
        spaces = ModelHyperparameterSpaces.get_lstm_hyperparameter_space()

    optimizer.define_hyperparameter_space(spaces)
    return optimizer


def create_gat_optimizer(
    validation_engine: RollingValidationEngine, n_trials: int = 80, reduced_space: bool = False
) -> ValidationBasedOptimizer:
    """Create GAT model optimizer with default settings."""
    config = OptimizationConfig(n_trials=n_trials, validation_metric="sharpe_ratio")
    optimizer = ValidationBasedOptimizer(validation_engine, config)

    if reduced_space:
        spaces = ModelHyperparameterSpaces.get_reduced_space("gat")
    else:
        spaces = ModelHyperparameterSpaces.get_gat_hyperparameter_space()

    optimizer.define_hyperparameter_space(spaces)
    return optimizer


class ValidationPerformanceTracker:
    """
    Comprehensive validation performance tracking with early stopping logic.

    This class tracks validation performance across trials and implements
    sophisticated early stopping strategies for hyperparameter optimization.
    """

    def __init__(
        self,
        metric_name: str = "sharpe_ratio",
        maximize_metric: bool = True,
        patience: int = 20,
        min_delta: float = 0.001,
        restore_best_weights: bool = True,
    ):
        """
        Initialize performance tracker.

        Args:
            metric_name: Name of metric to track
            maximize_metric: Whether to maximize the metric
            patience: Number of trials without improvement before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights on early stop
        """
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        # Tracking state
        self.trial_history: list[dict[str, Any]] = []
        self.best_score: float | None = None
        self.best_trial: int | None = None
        self.best_params: dict[str, Any] | None = None
        self.trials_without_improvement = 0
        self.should_stop = False

    def update(
        self,
        trial_number: int,
        score: float,
        params: dict[str, Any],
        additional_metrics: dict[str, float] | None = None,
    ) -> bool:
        """
        Update tracker with new trial results.

        Args:
            trial_number: Current trial number
            score: Primary metric score
            params: Hyperparameters for this trial
            additional_metrics: Optional additional metrics

        Returns:
            Whether early stopping should be triggered
        """
        # Record trial
        trial_record = {
            "trial": trial_number,
            "score": score,
            "params": params,
            "timestamp": pd.Timestamp.now(),
            "additional_metrics": additional_metrics or {},
        }
        self.trial_history.append(trial_record)

        # Check for improvement
        is_improvement = self._check_improvement(score)

        if is_improvement:
            self.best_score = score
            self.best_trial = trial_number
            self.best_params = params.copy()
            self.trials_without_improvement = 0
            logger.info(f"New best score: {score:.6f} at trial {trial_number}")
        else:
            self.trials_without_improvement += 1

        # Check early stopping condition
        if self.trials_without_improvement >= self.patience:
            self.should_stop = True
            logger.info(f"Early stopping triggered after {trial_number} trials")
            logger.info(f"Best score: {self.best_score:.6f} at trial {self.best_trial}")

        return self.should_stop

    def _check_improvement(self, score: float) -> bool:
        """Check if current score is an improvement."""
        if self.best_score is None:
            return True

        if self.maximize_metric:
            return score > (self.best_score + self.min_delta)
        else:
            return score < (self.best_score - self.min_delta)

    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.trial_history:
            return {"status": "No trials recorded"}

        scores = [trial["score"] for trial in self.trial_history]

        return {
            "best_score": self.best_score,
            "best_trial": self.best_trial,
            "best_params": self.best_params,
            "total_trials": len(self.trial_history),
            "trials_without_improvement": self.trials_without_improvement,
            "early_stopped": self.should_stop,
            "score_statistics": {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "median": float(np.median(scores)),
            },
            "improvement_rate": self._calculate_improvement_rate(),
            "convergence_analysis": self._analyze_convergence(),
        }

    def _calculate_improvement_rate(self) -> float:
        """Calculate rate of improvement over trials."""
        if len(self.trial_history) < 2:
            return 0.0

        scores = [trial["score"] for trial in self.trial_history]
        improvements = 0

        for i in range(1, len(scores)):
            if self.maximize_metric:
                if scores[i] > scores[i - 1]:
                    improvements += 1
            else:
                if scores[i] < scores[i - 1]:
                    improvements += 1

        return improvements / (len(scores) - 1)

    def _analyze_convergence(self) -> dict[str, Any]:
        """Analyze convergence patterns in the optimization."""
        if len(self.trial_history) < 10:
            return {"status": "Insufficient data for convergence analysis"}

        scores = [trial["score"] for trial in self.trial_history]

        # Calculate rolling best scores
        rolling_best = []
        current_best = scores[0]

        for score in scores:
            if self.maximize_metric:
                current_best = max(current_best, score)
            else:
                current_best = min(current_best, score)
            rolling_best.append(current_best)

        # Calculate improvement periods
        last_improvement_idx = 0
        for i in range(len(rolling_best) - 1, -1, -1):
            if i > 0 and rolling_best[i] != rolling_best[i - 1]:
                last_improvement_idx = i
                break

        stagnation_period = len(rolling_best) - last_improvement_idx - 1

        return {
            "stagnation_period": stagnation_period,
            "stagnation_ratio": stagnation_period / len(scores),
            "last_improvement_trial": last_improvement_idx,
            "convergence_trend": (
                "converging" if stagnation_period > self.patience // 2 else "exploring"
            ),
        }

    def plot_performance_history(self, output_path: Path | None = None) -> None:
        """Plot performance history (placeholder - would use matplotlib)."""
        # This would create performance plots if matplotlib is available
        # For now, just log the performance trend
        if not self.trial_history:
            logger.warning("No trial history to plot")
            return

        scores = [trial["score"] for trial in self.trial_history]
        trials = [trial["trial"] for trial in self.trial_history]

        logger.info(f"Performance trend over {len(trials)} trials:")
        logger.info(f"Starting score: {scores[0]:.6f}")
        logger.info(f"Final score: {scores[-1]:.6f}")
        logger.info(f"Best score: {self.best_score:.6f} at trial {self.best_trial}")

        # Save performance data to CSV if path provided
        if output_path:
            df = pd.DataFrame(self.trial_history)
            df.to_csv(output_path.with_suffix(".csv"), index=False)
            logger.info(f"Performance history saved to {output_path.with_suffix('.csv')}")


class EarlyStoppingCallback:
    """
    Early stopping callback for integration with training loops.

    This callback can be integrated with model training to implement
    early stopping based on validation performance.
    """

    def __init__(
        self,
        monitor: str = "val_sharpe",
        patience: int = 10,
        min_delta: float = 0.0001,
        mode: str = "max",
        restore_best_weights: bool = True,
        verbose: int = 1,
    ):
        """
        Initialize early stopping callback.

        Args:
            monitor: Metric to monitor
            patience: Number of epochs without improvement before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' or 'min' mode for metric
            restore_best_weights: Whether to restore best weights
            verbose: Verbosity level
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        if mode == "max":
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            self.monitor_op = np.less
            self.best = np.Inf

        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

    def __call__(self, epoch: int, logs: dict[str, float]) -> bool:
        """
        Check early stopping condition.

        Args:
            epoch: Current epoch number
            logs: Dictionary of metrics

        Returns:
            Whether to stop training
        """
        current = logs.get(self.monitor)
        if current is None:
            logger.warning(
                f"Early stopping conditioned on metric '{self.monitor}' "
                f"which is not available. Available metrics are: {list(logs.keys())}"
            )
            return False

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                # In practice, this would save model weights
                self.best_weights = epoch  # Placeholder
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.verbose > 0:
                    logger.info(f"Early stopping at epoch {epoch}")
                    logger.info(f"Restoring model weights from epoch {self.best_weights}")
                return True

        return False

    def on_train_end(self) -> dict[str, Any]:
        """Return summary of early stopping behavior."""
        return {
            "stopped_early": self.stopped_epoch > 0,
            "stopped_epoch": self.stopped_epoch,
            "best_value": self.best,
            "patience_used": self.wait,
            "patience_limit": self.patience,
        }
