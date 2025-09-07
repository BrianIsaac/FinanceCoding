"""
Rolling backtest engine with strict no-look-ahead validation.

This module provides a comprehensive rolling backtest engine that integrates
rolling window generation, temporal validation, model retraining, and backtest
execution with strict enforcement of temporal data integrity.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics.returns import PerformanceAnalytics
from src.evaluation.validation.rolling_validation import (
    BacktestConfig,
    RollingValidationEngine,
    RollSplit,
)
from src.models.base import PortfolioModel
from src.utils.gpu import GPUConfig, GPUMemoryManager

logger = logging.getLogger(__name__)


@dataclass
class RollingBacktestConfig:
    """Enhanced configuration for rolling backtest engine."""

    # Temporal configuration (from BacktestConfig)
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    training_months: int = 36
    validation_months: int = 12
    test_months: int = 12
    step_months: int = 12
    rebalance_frequency: str = "M"

    # Data integrity parameters
    min_training_samples: int = 252
    max_gap_days: int = 5
    require_full_periods: bool = True

    # Backtest execution parameters
    initial_capital: float = 1000000.0
    transaction_cost_bps: float = 10.0
    benchmark_ticker: str = "SPY"

    # Memory management
    gpu_config: GPUConfig | None = None
    batch_size: int = 32
    enable_memory_monitoring: bool = True

    # Output configuration
    output_dir: Path | None = None
    save_intermediate_results: bool = True
    enable_progress_tracking: bool = True

    def to_backtest_config(self) -> BacktestConfig:
        """Convert to BacktestConfig for rolling validation engine."""
        return BacktestConfig(
            start_date=self.start_date,
            end_date=self.end_date,
            training_months=self.training_months,
            validation_months=self.validation_months,
            test_months=self.test_months,
            step_months=self.step_months,
            rebalance_frequency=self.rebalance_frequency,
            min_training_samples=self.min_training_samples,
            max_gap_days=self.max_gap_days,
            require_full_periods=self.require_full_periods,
        )


@dataclass
class ModelRetrainingState:
    """Track model retraining state across rolling windows."""

    model_checkpoints: dict[str, Any] = field(default_factory=dict)
    training_history: list[dict[str, Any]] = field(default_factory=list)
    performance_metrics: list[dict[str, float]] = field(default_factory=list)
    last_training_date: pd.Timestamp | None = None
    retraining_count: int = 0


@dataclass
class RollingBacktestResults:
    """Comprehensive results from rolling backtest execution."""

    splits: list[RollSplit]
    portfolio_returns: dict[str, pd.Series]
    portfolio_weights: dict[str, pd.DataFrame]
    transaction_costs: dict[str, pd.Series]
    performance_metrics: dict[str, dict[str, float]]
    model_performance: dict[str, list[dict[str, Any]]]
    temporal_integrity_report: dict[str, Any]
    memory_usage_stats: dict[str, Any]
    execution_summary: dict[str, Any]


class RollingBacktestEngine:
    """
    Comprehensive rolling backtest engine with temporal integrity enforcement.

    This engine implements the complete rolling backtest workflow:
    1. Generate rolling windows with 36/12/12 month protocol
    2. Maintain strict temporal separation to prevent look-ahead bias
    3. Execute model retraining on each window
    4. Run backtest execution with transaction costs
    5. Track performance metrics and memory usage
    6. Provide comprehensive validation and reporting
    """

    def __init__(self, config: RollingBacktestConfig):
        """
        Initialize rolling backtest engine.

        Args:
            config: Rolling backtest configuration
        """
        self.config = config
        self.performance_analytics = PerformanceAnalytics()

        # Initialize rolling validation engine
        backtest_config = config.to_backtest_config()
        self.rolling_engine = RollingValidationEngine(
            backtest_config,
            config.gpu_config
        )

        # Initialize temporal integrity monitor
        self.integrity_monitor = self.rolling_engine.create_integrity_monitor()

        # Initialize memory management
        self.gpu_manager = (
            GPUMemoryManager(config.gpu_config)
            if config.gpu_config else None
        )

        # Initialize output directory
        if config.output_dir:
            config.output_dir.mkdir(parents=True, exist_ok=True)

        # State tracking
        self.model_states: dict[str, ModelRetrainingState] = {}
        self._execution_log: list[dict[str, Any]] = []

    def run_rolling_backtest(
        self,
        models: dict[str, PortfolioModel],
        data: dict[str, pd.DataFrame],
        universe_data: pd.DataFrame | None = None,
    ) -> RollingBacktestResults:
        """
        Execute complete rolling backtest across all models and time periods.

        Args:
            models: Dictionary of portfolio models to backtest
            data: Dictionary containing returns and other market data
            universe_data: Optional dynamic universe membership data

        Returns:
            Comprehensive rolling backtest results
        """
        logger.info("Starting rolling backtest execution")

        # Validate inputs
        self._validate_inputs(models, data)

        # Generate rolling windows
        sample_timestamps = list(data["returns"].index)
        splits = self.rolling_engine.generate_rolling_windows(sample_timestamps)

        if not splits:
            raise ValueError("No valid rolling windows could be generated")

        logger.info(f"Generated {len(splits)} rolling windows")

        # Initialize result containers
        results = RollingBacktestResults(
            splits=splits,
            portfolio_returns={},
            portfolio_weights={},
            transaction_costs={},
            performance_metrics={},
            model_performance={},
            temporal_integrity_report={},
            memory_usage_stats={},
            execution_summary={}
        )

        # Execute backtest for each model
        for model_name, model in models.items():
            logger.info(f"Starting backtest for model: {model_name}")

            # Initialize model state tracking
            self.model_states[model_name] = ModelRetrainingState()

            # Execute rolling backtest for this model
            model_results = self._execute_model_backtest(
                model_name, model, data, universe_data, splits
            )

            # Store results
            results.portfolio_returns[model_name] = model_results["returns"]
            results.portfolio_weights[model_name] = model_results["weights"]
            results.transaction_costs[model_name] = model_results["costs"]
            results.performance_metrics[model_name] = model_results["metrics"]
            results.model_performance[model_name] = model_results["model_stats"]

        # Generate integrity report
        results.temporal_integrity_report = self.integrity_monitor.get_integrity_summary()

        # Collect memory usage statistics
        if self.gpu_manager:
            results.memory_usage_stats = self.gpu_manager.get_memory_stats()

        # Generate execution summary
        results.execution_summary = self._generate_execution_summary(results)

        # Save results if configured
        if self.config.output_dir and self.config.save_intermediate_results:
            self._save_results(results)

        logger.info("Rolling backtest execution completed")
        return results

    def _execute_model_backtest(
        self,
        model_name: str,
        model: PortfolioModel,
        data: dict[str, pd.DataFrame],
        universe_data: pd.DataFrame | None,
        splits: list[RollSplit],
    ) -> dict[str, Any]:
        """Execute rolling backtest for a single model."""

        returns_list = []
        weights_list = []
        costs_list = []
        model_stats = []

        for i, split in enumerate(splits):
            logger.info(
                f"Processing window {i+1}/{len(splits)} for {model_name}: "
                f"{split.train_period.start_date} - {split.test_period.end_date}"
            )

            # Monitor temporal integrity
            integrity_check = self.integrity_monitor.monitor_split_integrity(
                split, list(data["returns"].index), model_name
            )

            if not integrity_check["overall_pass"]:
                logger.error(f"Temporal integrity check failed for split {i}")
                continue

            # Extract and validate data for this split
            try:
                split_data = self._prepare_split_data(split, data, universe_data)

                # Execute model retraining
                training_results = self._retrain_model_on_split(
                    model, split, split_data, model_name
                )

                # Execute backtest on test period
                backtest_results = self._execute_split_backtest(
                    model, split, split_data, training_results
                )

                # Store results
                if backtest_results["returns"] is not None:
                    returns_list.append(backtest_results["returns"])
                    weights_list.append(backtest_results["weights"])
                    costs_list.append(backtest_results["costs"])
                    model_stats.append({
                        "split_index": i,
                        "train_start": split.train_period.start_date,
                        "test_start": split.test_period.start_date,
                        **training_results,
                        **backtest_results["metrics"]
                    })

            except Exception as e:
                logger.error(f"Error processing split {i} for {model_name}: {e}")
                continue

        # Combine results across splits
        combined_returns = pd.concat(returns_list) if returns_list else pd.Series(dtype=float)
        combined_weights = pd.concat(weights_list) if weights_list else pd.DataFrame()
        combined_costs = pd.concat(costs_list) if costs_list else pd.Series(dtype=float)

        # Calculate performance metrics
        if not combined_returns.empty:
            performance_metrics = self.performance_analytics.calculate_portfolio_metrics(
                combined_returns
            )
        else:
            performance_metrics = {}

        return {
            "returns": combined_returns,
            "weights": combined_weights,
            "costs": combined_costs,
            "metrics": performance_metrics,
            "model_stats": model_stats,
        }

    def _prepare_split_data(
        self,
        split: RollSplit,
        data: dict[str, pd.DataFrame],
        universe_data: pd.DataFrame | None,
    ) -> dict[str, pd.DataFrame]:
        """Prepare and validate data for a specific split with temporal guards."""

        returns_data = data["returns"]

        # Apply strict temporal guards
        train_data, val_data, test_data = self.rolling_engine.enforce_temporal_guard(
            split, returns_data, returns_data, returns_data
        )

        # Prepare universe for each period
        universe_train = (
            universe_data.loc[
                split.train_period.start_date:split.train_period.end_date
            ]
            if universe_data is not None
            else None
        )

        universe_test = (
            universe_data.loc[
                split.test_period.start_date:split.test_period.end_date
            ]
            if universe_data is not None
            else None
        )

        return {
            "train_returns": train_data,
            "val_returns": val_data,
            "test_returns": test_data,
            "train_universe": universe_train,
            "test_universe": universe_test,
        }

    def _retrain_model_on_split(
        self,
        model: PortfolioModel,
        split: RollSplit,
        split_data: dict[str, pd.DataFrame],
        model_name: str,
    ) -> dict[str, Any]:
        """Retrain model on training data for current split."""

        train_returns = split_data["train_returns"]
        val_returns = split_data["val_returns"]
        train_universe = (
            split_data["train_universe"].columns.tolist()
            if split_data["train_universe"] is not None
            else train_returns.columns.tolist()
        )

        # Record training start
        training_start = pd.Timestamp.now()

        try:
            # Fit model on training data
            model.fit(
                returns=train_returns,
                universe=train_universe,
                fit_period=(split.train_period.start_date, split.train_period.end_date),
            )

            # Validate on validation period if available
            validation_metrics = {}
            if val_returns is not None and not val_returns.empty:
                # Generate validation predictions
                val_weights = model.predict_weights(
                    date=split.validation_period.start_date,
                    universe=train_universe,
                )

                # Calculate validation performance
                if not val_weights.empty:
                    val_period_returns = val_returns.loc[:, val_weights.index]
                    portfolio_returns = (val_period_returns * val_weights).sum(axis=1)

                    validation_metrics = self.performance_analytics.calculate_portfolio_metrics(
                        portfolio_returns
                    )

            training_duration = (pd.Timestamp.now() - training_start).total_seconds()

            # Update model state
            model_state = self.model_states[model_name]
            model_state.last_training_date = split.train_period.end_date
            model_state.retraining_count += 1
            model_state.performance_metrics.append(validation_metrics)

            return {
                "training_success": True,
                "training_duration_seconds": training_duration,
                "validation_metrics": validation_metrics,
                "training_samples": len(train_returns),
                "training_period_days": split.train_period.duration_days,
            }

        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            return {
                "training_success": False,
                "error": str(e),
                "training_duration_seconds": (pd.Timestamp.now() - training_start).total_seconds(),
            }

    def _execute_split_backtest(
        self,
        model: PortfolioModel,
        split: RollSplit,
        split_data: dict[str, pd.DataFrame],
        training_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute backtest on test period for current split."""

        if not training_results.get("training_success", False):
            return {
                "returns": None,
                "weights": None,
                "costs": None,
                "metrics": {},
            }

        test_returns = split_data["test_returns"]
        test_universe = (
            split_data["test_universe"].columns.tolist()
            if split_data["test_universe"] is not None
            else test_returns.columns.tolist()
        )

        # Generate rebalancing dates for test period
        rebalance_dates = pd.date_range(
            start=split.test_period.start_date,
            end=split.test_period.end_date,
            freq="MS",  # Month start
        )
        rebalance_dates = [d for d in rebalance_dates if d in test_returns.index]

        if not rebalance_dates:
            logger.warning("No rebalancing dates in test period")
            return {
                "returns": pd.Series(dtype=float),
                "weights": pd.DataFrame(),
                "costs": pd.Series(dtype=float),
                "metrics": {},
            }

        # Execute rebalancing strategy
        weights_list = []
        returns_list = []
        costs_list = []
        previous_weights = None

        for i, rebalance_date in enumerate(rebalance_dates):
            try:
                # Generate portfolio weights
                current_weights = model.predict_weights(
                    date=rebalance_date,
                    universe=test_universe,
                )

                if current_weights.empty:
                    logger.warning(f"Empty weights for {rebalance_date}")
                    continue

                # Calculate transaction costs
                costs = 0.0
                if previous_weights is not None:
                    # Simple transaction cost calculation (can be enhanced)
                    weight_changes = abs(current_weights - previous_weights.reindex(current_weights.index, fill_value=0)).sum()
                    costs = weight_changes * self.config.transaction_cost_bps / 10000.0

                # Calculate returns for holding period
                if i < len(rebalance_dates) - 1:
                    next_date = rebalance_dates[i + 1]
                    period_returns = test_returns.loc[rebalance_date:next_date, current_weights.index]
                    if not period_returns.empty:
                        daily_returns = (period_returns * current_weights).sum(axis=1).iloc[1:]
                        returns_list.extend(daily_returns.tolist())
                else:
                    # Last period - hold until end of test period
                    period_returns = test_returns.loc[rebalance_date:, current_weights.index]
                    if not period_returns.empty:
                        daily_returns = (period_returns * current_weights).sum(axis=1).iloc[1:]
                        returns_list.extend(daily_returns.tolist())

                weights_list.append(current_weights)
                costs_list.append(costs)
                previous_weights = current_weights

            except Exception as e:
                logger.error(f"Error during rebalancing on {rebalance_date}: {e}")
                continue

        # Combine results
        weights_df = pd.DataFrame(weights_list, index=rebalance_dates[:len(weights_list)]) if weights_list else pd.DataFrame()
        returns_series = pd.Series(returns_list, name="portfolio_returns") if returns_list else pd.Series(dtype=float)
        costs_series = pd.Series(costs_list, index=rebalance_dates[:len(costs_list)], name="transaction_costs") if costs_list else pd.Series(dtype=float)

        # Calculate period metrics
        period_metrics = {}
        if not returns_series.empty:
            period_metrics = self.performance_analytics.calculate_portfolio_metrics(returns_series)

        return {
            "returns": returns_series,
            "weights": weights_df,
            "costs": costs_series,
            "metrics": period_metrics,
        }

    def _validate_inputs(
        self,
        models: dict[str, PortfolioModel],
        data: dict[str, pd.DataFrame]
    ) -> None:
        """Validate inputs for rolling backtest."""

        if not models:
            raise ValueError("No models provided for backtesting")

        if "returns" not in data:
            raise ValueError("Returns data is required")

        returns_data = data["returns"]
        if not isinstance(returns_data.index, pd.DatetimeIndex):
            raise ValueError("Returns data must have DatetimeIndex")

        # Validate date range
        data_start = returns_data.index.min()
        data_end = returns_data.index.max()

        if self.config.start_date < data_start:
            logger.warning(f"Backtest start date {self.config.start_date} is before data start {data_start}")

        if self.config.end_date > data_end:
            logger.warning(f"Backtest end date {self.config.end_date} is after data end {data_end}")

    def _generate_execution_summary(self, results: RollingBacktestResults) -> dict[str, Any]:
        """Generate execution summary statistics."""

        summary = {
            "total_splits": len(results.splits),
            "models_tested": len(results.portfolio_returns),
            "execution_timestamp": pd.Timestamp.now().isoformat(),
            "config_summary": {
                "training_months": self.config.training_months,
                "validation_months": self.config.validation_months,
                "test_months": self.config.test_months,
                "step_months": self.config.step_months,
            }
        }

        # Add per-model summary statistics
        model_summaries = {}
        for model_name, returns in results.portfolio_returns.items():
            if not returns.empty:
                metrics = results.performance_metrics.get(model_name, {})
                model_summaries[model_name] = {
                    "total_return_days": len(returns),
                    "sharpe_ratio": metrics.get("sharpe_ratio", np.nan),
                    "total_return": (1 + returns).prod() - 1,
                    "annualized_return": metrics.get("annualized_return", np.nan),
                    "annualized_volatility": metrics.get("annualized_volatility", np.nan),
                }

        summary["model_performance_summary"] = model_summaries

        return summary

    def _save_results(self, results: RollingBacktestResults) -> None:
        """Save backtest results to disk."""

        if not self.config.output_dir:
            return

        output_dir = self.config.output_dir

        # Save portfolio returns
        for model_name, returns in results.portfolio_returns.items():
            if not returns.empty:
                returns.to_csv(output_dir / f"{model_name}_returns.csv")

        # Save portfolio weights
        for model_name, weights in results.portfolio_weights.items():
            if not weights.empty:
                weights.to_csv(output_dir / f"{model_name}_weights.csv")

        # Save performance metrics
        metrics_df = pd.DataFrame(results.performance_metrics).T
        metrics_df.to_csv(output_dir / "performance_metrics.csv")

        # Save execution summary
        import json
        with open(output_dir / "execution_summary.json", "w") as f:
            json.dump(results.execution_summary, f, indent=2, default=str)

        # Save integrity report
        if self.config.output_dir:
            self.integrity_monitor.export_integrity_report(
                output_dir / "temporal_integrity_report.json"
            )

        logger.info(f"Results saved to {output_dir}")


class ModelRetrainingEngine:
    """
    Handles automated model retraining across rolling windows.

    This class manages the retraining process including:
    - Model state persistence across retraining cycles
    - Dynamic universe membership handling
    - Missing data detection and handling during retraining
    - Memory management during intensive retraining operations
    """

    def __init__(
        self,
        gpu_config: GPUConfig | None = None,
        checkpoint_dir: Path | None = None
    ):
        """Initialize model retraining engine."""
        self.gpu_manager = GPUMemoryManager(gpu_config) if gpu_config else None
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def retrain_model(
        self,
        model: PortfolioModel,
        train_data: pd.DataFrame,
        universe: list[str],
        split_info: RollSplit,
        model_name: str,
    ) -> dict[str, Any]:
        """
        Retrain model with comprehensive state management.

        Args:
            model: Portfolio model to retrain
            train_data: Training data for current window
            universe: Asset universe for training
            split_info: Rolling split information
            model_name: Name identifier for model

        Returns:
            Dictionary with retraining results and metrics
        """

        # Handle missing data
        cleaned_data = self._handle_missing_data(train_data, universe)

        # Save checkpoint before retraining
        if self.checkpoint_dir:
            self._save_model_checkpoint(model, split_info, model_name)

        # Execute retraining with memory management
        training_results = self._execute_retraining(
            model, cleaned_data, universe, split_info
        )

        return training_results

    def _handle_missing_data(
        self,
        data: pd.DataFrame,
        universe: list[str]
    ) -> pd.DataFrame:
        """Handle missing data during retraining."""

        # Filter to universe assets
        available_assets = [asset for asset in universe if asset in data.columns]
        filtered_data = data[available_assets]

        # Handle missing values
        # Forward fill for up to 5 days, then drop
        cleaned_data = filtered_data.fillna(method='ffill', limit=5).dropna()

        logger.info(
            f"Data cleaning: {len(universe)} universe assets, "
            f"{len(available_assets)} available, "
            f"{len(cleaned_data.columns)} after cleaning"
        )

        return cleaned_data

    def _save_model_checkpoint(
        self,
        model: PortfolioModel,
        split_info: RollSplit,
        model_name: str
    ) -> None:
        """Save model checkpoint before retraining."""

        if not self.checkpoint_dir:
            return

        checkpoint_path = (
            self.checkpoint_dir /
            f"{model_name}_{split_info.train_period.start_date.strftime('%Y%m%d')}.pkl"
        )

        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(model, f)
            logger.debug(f"Model checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def _execute_retraining(
        self,
        model: PortfolioModel,
        train_data: pd.DataFrame,
        universe: list[str],
        split_info: RollSplit,
    ) -> dict[str, Any]:
        """Execute model retraining with memory management."""

        start_time = pd.Timestamp.now()

        try:
            # Clear GPU memory if available
            if self.gpu_manager:
                self.gpu_manager.clear_cache()

            # Execute training
            model.fit(
                returns=train_data,
                universe=universe,
                fit_period=(split_info.train_period.start_date, split_info.train_period.end_date),
            )

            training_time = (pd.Timestamp.now() - start_time).total_seconds()

            return {
                "success": True,
                "training_time_seconds": training_time,
                "training_samples": len(train_data),
                "assets_trained": len(train_data.columns),
                "memory_stats": self.gpu_manager.get_memory_stats() if self.gpu_manager else {},
            }

        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "training_time_seconds": (pd.Timestamp.now() - start_time).total_seconds(),
            }
