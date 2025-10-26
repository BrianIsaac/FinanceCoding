"""Core sensitivity analysis engine with parameter grid generation."""

import logging
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from src.evaluation.backtest.rolling_engine import RollingBacktestEngine
from src.evaluation.metrics.returns import PerformanceAnalytics
from src.evaluation.validation.significance import StatisticalValidation

logger = logging.getLogger(__name__)


@dataclass
class ParameterGrid:
    """Parameter grid definition for sensitivity analysis."""

    model_type: str
    parameters: dict[str, list[Any]]
    constraints: dict[str, Any] = field(default_factory=dict)

    def generate_combinations(self) -> list[dict[str, Any]]:
        """Generate all parameter combinations from the grid.

        Returns:
            List of parameter dictionaries for testing
        """
        param_names = list(self.parameters.keys())
        param_values = list(self.parameters.values())

        combinations = []
        for combo in product(*param_values):
            param_dict = dict(zip(param_names, combo))
            param_dict.update(self.constraints)
            combinations.append(param_dict)

        return combinations

    def validate_parameters(self, params: dict[str, Any]) -> bool:
        """Validate parameter combination against constraints.

        Args:
            params: Parameter dictionary to validate

        Returns:
            True if parameters are valid
        """
        # Basic validation - can be extended with model-specific rules
        for key, value in params.items():
            if key in self.parameters:
                if value not in self.parameters[key]:
                    return False
        return True


@dataclass
class SensitivityResult:
    """Results from sensitivity analysis."""

    model_type: str
    parameter_combination: dict[str, Any]
    performance_metrics: dict[str, float]
    backtest_results: pd.DataFrame
    execution_time: float
    memory_usage: float
    error: Optional[str] = None


class SensitivityAnalysisEngine:
    """Comprehensive sensitivity analysis engine for parameter testing."""

    def __init__(
        self,
        backtest_engine: RollingBacktestEngine,
        performance_analytics: PerformanceAnalytics,
        statistical_validator: StatisticalValidation,
        max_workers: int = 4,
        memory_limit_gb: int = 12,
    ):
        """Initialize sensitivity analysis engine.

        Args:
            backtest_engine: Rolling backtest engine for model execution
            performance_analytics: Performance metrics calculator
            statistical_validator: Statistical significance testing
            max_workers: Maximum parallel workers for parameter testing
            memory_limit_gb: GPU memory limit for model configuration cycling
        """
        self.backtest_engine = backtest_engine
        self.performance_analytics = performance_analytics
        self.statistical_validator = statistical_validator
        self.max_workers = max_workers
        self.memory_limit_gb = memory_limit_gb

        self.results: list[SensitivityResult] = []
        self.parameter_grids: dict[str, ParameterGrid] = {}

    def register_parameter_grid(self, model_type: str, grid: ParameterGrid) -> None:
        """Register parameter grid for a model type.

        Args:
            model_type: Model identifier (hrp, lstm, gat)
            grid: Parameter grid configuration
        """
        self.parameter_grids[model_type] = grid
        logger.info(
            f"Registered parameter grid for {model_type} with {len(grid.generate_combinations())} combinations"
        )

    def run_sensitivity_analysis(
        self,
        model_types: list[str],
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
        progress_callback: Optional[Callable] = None,
    ) -> dict[str, list[SensitivityResult]]:
        """Execute sensitivity analysis across model types and parameter combinations.

        Args:
            model_types: List of model types to analyze
            data: Market data for backtesting
            start_date: Analysis start date
            end_date: Analysis end date
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping model types to sensitivity results
        """
        logger.info(f"Starting sensitivity analysis for models: {model_types}")

        all_results = {}
        total_combinations = sum(
            len(self.parameter_grids[model].generate_combinations())
            for model in model_types
            if model in self.parameter_grids
        )

        completed = 0

        for model_type in model_types:
            if model_type not in self.parameter_grids:
                logger.warning(f"No parameter grid registered for {model_type}")
                continue

            grid = self.parameter_grids[model_type]
            combinations = grid.generate_combinations()

            logger.info(f"Testing {len(combinations)} parameter combinations for {model_type}")

            model_results = []

            # Execute parameter combinations with memory management
            for _i, params in enumerate(combinations):
                try:
                    result = self._execute_parameter_combination(
                        model_type, params, data, start_date, end_date
                    )
                    model_results.append(result)

                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total_combinations)

                except Exception as e:
                    logger.error(f"Error testing {model_type} with params {params}: {e}")
                    error_result = SensitivityResult(
                        model_type=model_type,
                        parameter_combination=params,
                        performance_metrics={},
                        backtest_results=pd.DataFrame(),
                        execution_time=0.0,
                        memory_usage=0.0,
                        error=str(e),
                    )
                    model_results.append(error_result)
                    completed += 1

            all_results[model_type] = model_results

        self.results = [result for results in all_results.values() for result in results]
        logger.info(f"Sensitivity analysis completed. Total results: {len(self.results)}")

        return all_results

    def _execute_parameter_combination(
        self,
        model_type: str,
        parameters: dict[str, Any],
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> SensitivityResult:
        """Execute single parameter combination.

        Args:
            model_type: Model type identifier
            parameters: Parameter configuration
            data: Market data
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            Sensitivity analysis result
        """
        import os
        import time

        import psutil

        start_time = time.time()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        try:
            # Configure model with parameters
            model_config = self._create_model_config(model_type, parameters)

            # Execute rolling backtest
            backtest_results = self.backtest_engine.run_rolling_backtest(
                model_config=model_config, data=data, start_date=start_date, end_date=end_date
            )

            # Calculate performance metrics
            performance_metrics = self.performance_analytics.calculate_portfolio_metrics(
                backtest_results
            )

            execution_time = time.time() - start_time
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = final_memory - initial_memory

            return SensitivityResult(
                model_type=model_type,
                parameter_combination=parameters,
                performance_metrics=performance_metrics,
                backtest_results=backtest_results,
                execution_time=execution_time,
                memory_usage=memory_usage,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed to execute {model_type} with params {parameters}: {e}")

            return SensitivityResult(
                model_type=model_type,
                parameter_combination=parameters,
                performance_metrics={},
                backtest_results=pd.DataFrame(),
                execution_time=execution_time,
                memory_usage=0.0,
                error=str(e),
            )

    def _create_model_config(self, model_type: str, parameters: dict[str, Any]) -> dict[str, Any]:
        """Create model configuration from parameters.

        Args:
            model_type: Model type identifier
            parameters: Parameter dictionary

        Returns:
            Model configuration dictionary
        """
        base_config = {"model_type": model_type, "parameters": parameters.copy()}

        return base_config

    def analyze_parameter_sensitivity(self, metric: str = "sharpe_ratio") -> pd.DataFrame:
        """Analyze parameter sensitivity for a specific metric.

        Args:
            metric: Performance metric to analyze

        Returns:
            DataFrame with parameter sensitivity analysis
        """
        if not self.results:
            raise ValueError("No sensitivity analysis results available")

        sensitivity_data = []

        for result in self.results:
            if result.error is None and metric in result.performance_metrics:
                row = result.parameter_combination.copy()
                row["model_type"] = result.model_type
                row[metric] = result.performance_metrics[metric]
                row["execution_time"] = result.execution_time
                row["memory_usage"] = result.memory_usage
                sensitivity_data.append(row)

        if not sensitivity_data:
            raise ValueError(f"No valid results found for metric {metric}")

        return pd.DataFrame(sensitivity_data)

    def calculate_parameter_stability(
        self, metric: str = "sharpe_ratio"
    ) -> dict[str, dict[str, float]]:
        """Calculate parameter stability scores across configurations.

        Args:
            metric: Performance metric for stability analysis

        Returns:
            Dictionary mapping model types to parameter stability scores
        """
        stability_scores = {}

        for model_type in self.parameter_grids.keys():
            model_results = [
                r for r in self.results if r.model_type == model_type and r.error is None
            ]

            if not model_results:
                continue

            # Get all parameter names for this model
            param_names = list(self.parameter_grids[model_type].parameters.keys())

            model_stability = {}

            for param_name in param_names:
                # Group results by parameter value
                param_groups = {}
                for result in model_results:
                    param_value = result.parameter_combination.get(param_name)
                    if param_value not in param_groups:
                        param_groups[param_value] = []
                    param_groups[param_value].append(result.performance_metrics.get(metric, 0.0))

                # Calculate stability as inverse of coefficient of variation
                param_values = []
                for group_values in param_groups.values():
                    if group_values:
                        param_values.extend(group_values)

                if param_values:
                    std_dev = np.std(param_values)
                    mean_val = np.mean(param_values)
                    cv = std_dev / abs(mean_val) if mean_val != 0 else float("inf")
                    stability = 1.0 / (1.0 + cv)  # Higher stability = lower variation
                    model_stability[param_name] = stability

            stability_scores[model_type] = model_stability

        return stability_scores

    def get_best_parameters(
        self, metric: str = "sharpe_ratio", top_k: int = 5
    ) -> dict[str, list[dict[str, Any]]]:
        """Get best parameter combinations for each model type.

        Args:
            metric: Performance metric for ranking
            top_k: Number of top combinations to return

        Returns:
            Dictionary mapping model types to top parameter combinations
        """
        best_params = {}

        for model_type in self.parameter_grids.keys():
            model_results = [
                r for r in self.results if r.model_type == model_type and r.error is None
            ]

            # Sort by metric (descending)
            model_results.sort(
                key=lambda x: x.performance_metrics.get(metric, -float("inf")), reverse=True
            )

            top_results = model_results[:top_k]
            best_params[model_type] = [
                {
                    "parameters": result.parameter_combination,
                    "performance": {metric: result.performance_metrics.get(metric, 0.0)},
                    "execution_time": result.execution_time,
                }
                for result in top_results
            ]

        return best_params

    def export_results(self, filepath: str) -> None:
        """Export sensitivity analysis results to file.

        Args:
            filepath: Path to save results
        """
        results_data = []

        for result in self.results:
            row = {
                "model_type": result.model_type,
                "execution_time": result.execution_time,
                "memory_usage": result.memory_usage,
                "error": result.error,
            }
            row.update(result.parameter_combination)
            row.update(result.performance_metrics)
            results_data.append(row)

        df = pd.DataFrame(results_data)

        if filepath.endswith(".csv"):
            df.to_csv(filepath, index=False)
        elif filepath.endswith(".parquet"):
            df.to_parquet(filepath, index=False)
        else:
            raise ValueError("Unsupported file format. Use .csv or .parquet")

        logger.info(f"Sensitivity analysis results exported to {filepath}")
