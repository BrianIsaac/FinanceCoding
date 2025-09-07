"""
Experiment orchestration framework for portfolio optimization models.

This module provides a unified framework for running experiments
with different models, configurations, and evaluation metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from omegaconf import OmegaConf

from src.config.data import DataConfig
from src.data.loaders.portfolio_data import PortfolioDataLoader
from src.evaluation.backtest.engine import BacktestConfig, BacktestEngine
from src.models.base.portfolio_model import PortfolioConstraints
from src.models.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    name: str
    description: str = ""

    # Data configuration
    data_config: dict[str, Any] = field(default_factory=dict)

    # Model configurations
    models: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Evaluation configuration
    evaluation: dict[str, Any] = field(default_factory=dict)

    # Output configuration
    output_dir: Path = Path("outputs/experiments")
    save_results: bool = True

    @classmethod
    def from_yaml(cls, config_path: Path) -> ExperimentConfig:
        """
        Load experiment configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            ExperimentConfig instance
        """
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        return cls(
            name=config_dict.get("experiment_name", "unnamed"),
            description=config_dict.get("description", ""),
            data_config=config_dict.get("data", {}),
            models=config_dict.get("models", {}),
            evaluation=config_dict.get("evaluation", {}),
            output_dir=Path(
                config_dict.get("project", {}).get("output_dir", "outputs/experiments")
            ),
        )


@dataclass
class ExperimentResults:
    """Results from a model experiment."""

    model_name: str
    variant_name: str
    backtest_results: dict[str, Any]
    performance_metrics: dict[str, float]
    execution_time: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "variant_name": self.variant_name,
            "backtest_results": self.backtest_results,
            "performance_metrics": self.performance_metrics,
            "execution_time": self.execution_time,
        }


class ExperimentRunner:
    """
    Orchestration framework for running portfolio optimization experiments.

    Handles model loading, data preparation, backtesting execution,
    and results compilation across multiple models and variants.
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment runner.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.results: list[ExperimentResults] = []

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize data loader
        self.data_loader = self._initialize_data_loader()

    def _initialize_data_loader(self) -> PortfolioDataLoader:
        """Initialize data loader from configuration."""
        data_config_path = Path(
            self.config.data_config.get("config_file", "configs/data/default.yaml")
        )

        if data_config_path.exists():
            data_config = DataConfig.from_yaml(data_config_path)
        else:
            logger.warning(f"Data config file {data_config_path} not found, using defaults")
            data_config = DataConfig()

        return PortfolioDataLoader(data_config)

    def run_experiment(self) -> dict[str, Any]:
        """
        Execute complete experiment across all models and variants.

        Returns:
            Dictionary containing all experiment results
        """
        logger.info(f"Starting experiment: {self.config.name}")

        # Load data
        returns_data, universe_data = self._load_experiment_data()

        if returns_data.empty:
            logger.error("No data loaded, cannot run experiment")
            return {"error": "No data available"}

        # Run each model configuration
        for model_name, model_config in self.config.models.items():
            if not model_config.get("enabled", True):
                logger.info(f"Skipping disabled model: {model_name}")
                continue

            self._run_model_variants(model_name, model_config, returns_data, universe_data)

        # Compile and save results
        experiment_summary = self._compile_results()

        if self.config.save_results:
            self._save_results(experiment_summary)

        logger.info(f"Experiment completed: {self.config.name}")
        return experiment_summary

    def _load_experiment_data(self) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Load returns and universe data for the experiment."""
        try:
            # Load returns data
            returns_data = self.data_loader.load_returns(
                start_date=self.config.data_config.get("train_start", "2016-01-01"),
                end_date=self.config.data_config.get("test_end", "2024-12-31"),
            )

            # Load universe data if available
            try:
                universe_data = self.data_loader.load_universe_calendar()
            except Exception as e:
                logger.warning(f"Could not load universe data: {e}")
                universe_data = None

            logger.info(
                f"Loaded data: {len(returns_data)} days, {len(returns_data.columns)} assets"
            )
            return returns_data, universe_data

        except Exception as e:
            logger.error(f"Error loading experiment data: {e}")
            return pd.DataFrame(), None

    def _run_model_variants(
        self,
        model_name: str,
        model_config: dict[str, Any],
        returns_data: pd.DataFrame,
        universe_data: pd.DataFrame | None,
    ) -> None:
        """Run all variants for a specific model."""
        variants = model_config.get(
            "variants", [{"name": f"{model_name}_default", "overrides": {}}]
        )

        for variant in variants:
            self._run_single_model(model_name, variant, model_config, returns_data, universe_data)

    def _run_single_model(
        self,
        model_name: str,
        variant: dict[str, Any],
        model_config: dict[str, Any],
        returns_data: pd.DataFrame,
        universe_data: pd.DataFrame | None,
    ) -> None:
        """Run backtest for a single model variant."""
        variant_name = variant.get("name", f"{model_name}_default")

        logger.info(f"Running {model_name} variant: {variant_name}")

        try:
            import time

            start_time = time.time()

            # Create model instance
            model = self._create_model_instance(model_name, model_config, variant)

            # Create backtest engine
            backtest_engine = self._create_backtest_engine()

            # Run backtest
            backtest_results = backtest_engine.run_backtest(model, returns_data, universe_data)

            execution_time = time.time() - start_time

            # Store results
            result = ExperimentResults(
                model_name=model_name,
                variant_name=variant_name,
                backtest_results=backtest_results,
                performance_metrics=backtest_results.get("performance_metrics", {}),
                execution_time=execution_time,
            )

            self.results.append(result)
            logger.info(f"Completed {variant_name} in {execution_time:.2f}s")

        except Exception as e:
            logger.error(f"Error running {variant_name}: {e}", exc_info=True)

    def _create_model_instance(
        self, model_name: str, model_config: dict[str, Any], variant: dict[str, Any]
    ):
        """Create model instance with configuration."""
        # Load base model configuration
        config_file = model_config.get("config_file")
        if config_file and Path(config_file).exists():
            with open(config_file) as f:
                base_config = yaml.safe_load(f)
        else:
            base_config = {}

        # Apply variant overrides
        overrides = variant.get("overrides", {})
        config = OmegaConf.merge(OmegaConf.create(base_config), OmegaConf.create(overrides))

        # Create constraints
        constraint_config = config.get("constraints", {})
        constraints = PortfolioConstraints(
            long_only=constraint_config.get("long_only", True),
            top_k_positions=constraint_config.get("top_k_positions"),
            max_position_weight=constraint_config.get("max_position_weight", 0.10),
            max_monthly_turnover=constraint_config.get("max_monthly_turnover", 0.20),
            transaction_cost_bps=constraint_config.get("transaction_cost_bps", 10.0),
        )

        # Create model using registry
        if model_name == "lstm":
            from src.models.lstm.model import LSTMModelConfig

            # Create LSTM-specific configuration
            lstm_config = LSTMModelConfig()

            # Apply configuration overrides
            if "architecture" in config:
                arch_config = config["architecture"]
                lstm_config.lstm_config.hidden_size = arch_config.get(
                    "hidden_size", lstm_config.lstm_config.hidden_size
                )
                lstm_config.lstm_config.num_layers = arch_config.get(
                    "num_layers", lstm_config.lstm_config.num_layers
                )
                lstm_config.lstm_config.dropout = arch_config.get(
                    "dropout", lstm_config.lstm_config.dropout
                )

            if "training" in config:
                train_config = config["training"]
                lstm_config.training_config.learning_rate = train_config.get(
                    "learning_rate", lstm_config.training_config.learning_rate
                )
                lstm_config.training_config.epochs = train_config.get(
                    "max_epochs", lstm_config.training_config.epochs
                )
                lstm_config.training_config.batch_size = train_config.get(
                    "batch_size", lstm_config.training_config.batch_size
                )

            return ModelRegistry.create_model(
                model_name, constraints=constraints, config=lstm_config
            )
        else:
            return ModelRegistry.create_model(model_name, constraints=constraints)

    def _create_backtest_engine(self) -> BacktestEngine:
        """Create backtest engine from configuration."""
        backtest_config = self.config.evaluation.get("backtest", {})

        config = BacktestConfig(
            start_date=pd.to_datetime(
                self.config.data_config.get("test_start", "2023-01-01")
            ).date(),
            end_date=pd.to_datetime(self.config.data_config.get("test_end", "2024-12-31")).date(),
            rebalance_frequency=backtest_config.get("rebalancing_frequency", "M"),
            initial_capital=backtest_config.get("initial_capital", 1000000.0),
            transaction_cost_bps=backtest_config.get("transaction_costs", 10.0)
            * 10000,  # Convert to bps
        )

        return BacktestEngine(config)

    def _compile_results(self) -> dict[str, Any]:
        """Compile experiment results into summary."""
        if not self.results:
            return {"error": "No results to compile"}

        summary = {
            "experiment_name": self.config.name,
            "description": self.config.description,
            "total_models": len(self.results),
            "models": {},
        }

        # Group results by model
        for result in self.results:
            model_name = result.model_name
            if model_name not in summary["models"]:
                summary["models"][model_name] = {
                    "variants": [],
                    "best_sharpe": float("-inf"),
                    "best_variant": None,
                }

            variant_summary = {
                "name": result.variant_name,
                "performance_metrics": result.performance_metrics,
                "execution_time": result.execution_time,
            }

            summary["models"][model_name]["variants"].append(variant_summary)

            # Track best performing variant
            sharpe = result.performance_metrics.get("sharpe_ratio", float("-inf"))
            if sharpe > summary["models"][model_name]["best_sharpe"]:
                summary["models"][model_name]["best_sharpe"] = sharpe
                summary["models"][model_name]["best_variant"] = result.variant_name

        return summary

    def _save_results(self, summary: dict[str, Any]) -> None:
        """Save experiment results to files."""
        # Save summary
        summary_path = self.config.output_dir / f"{self.config.name}_summary.yaml"
        with open(summary_path, "w") as f:
            yaml.dump(summary, f, default_flow_style=False)

        # Save detailed results
        detailed_results = [result.to_dict() for result in self.results]
        detailed_path = self.config.output_dir / f"{self.config.name}_detailed.yaml"
        with open(detailed_path, "w") as f:
            yaml.dump(detailed_results, f, default_flow_style=False)

        logger.info(f"Results saved to {self.config.output_dir}")


def run_experiment_from_config(config_path: Path) -> dict[str, Any]:
    """
    Run experiment from configuration file.

    Args:
        config_path: Path to experiment configuration YAML

    Returns:
        Experiment results summary
    """
    config = ExperimentConfig.from_yaml(config_path)
    runner = ExperimentRunner(config)
    return runner.run_experiment()
