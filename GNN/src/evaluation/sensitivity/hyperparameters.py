"""Model-specific hyperparameter testing framework."""

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.evaluation.validation.bootstrap import BootstrapMethodology
from src.evaluation.validation.corrections import MultipleComparisonCorrections
from src.evaluation.validation.significance import StatisticalValidation

from .engine import ParameterGrid, SensitivityAnalysisEngine

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterTestResult:
    """Results from hyperparameter sensitivity testing."""

    model_type: str
    parameter_name: str
    parameter_values: list[Any]
    metric_values: list[float]
    statistical_significance: dict[str, float]
    effect_size: float
    optimal_value: Any
    confidence_interval: tuple


class HyperparameterTester:
    """Comprehensive hyperparameter sensitivity testing."""

    def __init__(
        self,
        sensitivity_engine: SensitivityAnalysisEngine,
        statistical_validator: StatisticalValidation,
        bootstrap_methodology: Optional[BootstrapMethodology] = None,
        multiple_comparison_corrector: Optional[MultipleComparisonCorrections] = None,
    ):
        """Initialize hyperparameter tester.

        Args:
            sensitivity_engine: Core sensitivity analysis engine
            statistical_validator: Statistical significance testing
            bootstrap_methodology: Bootstrap framework for confidence intervals
            multiple_comparison_corrector: Multiple comparison corrections
        """
        self.sensitivity_engine = sensitivity_engine
        self.statistical_validator = statistical_validator
        self.bootstrap_methodology = bootstrap_methodology or BootstrapMethodology(n_bootstrap=1000)
        self.multiple_comparison_corrector = (
            multiple_comparison_corrector or MultipleComparisonCorrections()
        )
        self.test_results: dict[str, list[HyperparameterTestResult]] = {}

    def create_hrp_parameter_grid(self) -> ParameterGrid:
        """Create HRP hyperparameter grid based on technical specifications.

        Returns:
            Parameter grid for HRP model testing
        """
        hrp_parameters = {
            "lookback_days": [252, 504, 756, 1008],  # 1, 2, 3, 4 years
            "linkage_method": ["single", "complete", "average", "ward"],
            "distance_metric": ["correlation", "angular", "absolute_correlation"],
            "min_weight": [0.01, 0.02, 0.05],  # Minimum position weight
            "max_weight": [0.15, 0.20, 0.25],  # Maximum position weight
        }

        constraints = {"rebalance_frequency": "monthly", "transaction_cost_bps": 10.0}

        return ParameterGrid(model_type="hrp", parameters=hrp_parameters, constraints=constraints)

    def create_lstm_parameter_grid(self) -> ParameterGrid:
        """Create LSTM hyperparameter grid based on technical specifications.

        Returns:
            Parameter grid for LSTM model testing
        """
        lstm_parameters = {
            "sequence_length": [30, 45, 60, 90],  # Days of historical data
            "hidden_size": [64, 128, 256],  # Hidden layer dimensions
            "num_layers": [1, 2, 3],  # LSTM layers
            "dropout": [0.1, 0.3, 0.5],  # Dropout rate
            "learning_rate": [0.0001, 0.001, 0.01],  # Optimizer learning rate
            "batch_size": [32, 64, 128],  # Training batch size
            "epochs": [50, 100, 200],  # Training epochs
        }

        constraints = {
            "rebalance_frequency": "monthly",
            "transaction_cost_bps": 10.0,
            "optimizer": "adam",
        }

        return ParameterGrid(model_type="lstm", parameters=lstm_parameters, constraints=constraints)

    def create_gat_parameter_grid(self) -> ParameterGrid:
        """Create GAT hyperparameter grid based on technical specifications.

        Returns:
            Parameter grid for GAT model testing
        """
        gat_parameters = {
            "attention_heads": [2, 4, 8],  # Number of attention heads
            "hidden_dim": [64, 128, 256],  # Hidden layer dimensions
            "dropout": [0.1, 0.3, 0.5],  # Dropout rate
            "learning_rate": [0.0001, 0.001, 0.01],  # Optimizer learning rate
            "graph_construction": ["k_nn", "mst", "tmfg"],  # Graph construction method
            "k_neighbors": [5, 10, 15, 20],  # For k-NN graph construction
            "edge_threshold": [0.1, 0.2, 0.3],  # Edge weight threshold
            "num_layers": [2, 3, 4],  # GAT layers
        }

        constraints = {
            "rebalance_frequency": "monthly",
            "transaction_cost_bps": 10.0,
            "optimizer": "adam",
        }

        return ParameterGrid(model_type="gat", parameters=gat_parameters, constraints=constraints)

    def test_hyperparameter_sensitivity(
        self,
        model_type: str,
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
        target_metric: str = "sharpe_ratio",
        significance_level: float = 0.05,
    ) -> list[HyperparameterTestResult]:
        """Test hyperparameter sensitivity for a model type.

        Args:
            model_type: Model identifier (hrp, lstm, gat)
            data: Market data for backtesting
            start_date: Analysis start date
            end_date: Analysis end date
            target_metric: Performance metric for analysis
            significance_level: Statistical significance threshold

        Returns:
            List of hyperparameter test results
        """
        logger.info(f"Starting hyperparameter sensitivity analysis for {model_type}")

        # Get appropriate parameter grid
        if model_type == "hrp":
            parameter_grid = self.create_hrp_parameter_grid()
        elif model_type == "lstm":
            parameter_grid = self.create_lstm_parameter_grid()
        elif model_type == "gat":
            parameter_grid = self.create_gat_parameter_grid()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Register parameter grid and run sensitivity analysis
        self.sensitivity_engine.register_parameter_grid(model_type, parameter_grid)

        sensitivity_results = self.sensitivity_engine.run_sensitivity_analysis(
            model_types=[model_type], data=data, start_date=start_date, end_date=end_date
        )

        # Analyze each hyperparameter
        hyperparameter_results = []

        for param_name in parameter_grid.parameters.keys():
            result = self._analyze_single_hyperparameter(
                model_type=model_type,
                parameter_name=param_name,
                sensitivity_results=sensitivity_results[model_type],
                target_metric=target_metric,
                significance_level=significance_level,
            )
            hyperparameter_results.append(result)

        self.test_results[model_type] = hyperparameter_results

        logger.info(
            f"Completed hyperparameter analysis for {model_type}. "
            f"Analyzed {len(hyperparameter_results)} parameters"
        )

        return hyperparameter_results

    def _analyze_single_hyperparameter(
        self,
        model_type: str,
        parameter_name: str,
        sensitivity_results: list,
        target_metric: str,
        significance_level: float,
    ) -> HyperparameterTestResult:
        """Analyze sensitivity of a single hyperparameter.

        Args:
            model_type: Model identifier
            parameter_name: Name of parameter to analyze
            sensitivity_results: Results from sensitivity analysis
            target_metric: Performance metric for analysis
            significance_level: Statistical significance threshold

        Returns:
            Hyperparameter test result
        """
        # Extract parameter values and corresponding metrics
        param_data = {}

        for result in sensitivity_results:
            if result.error is None and target_metric in result.performance_metrics:
                param_value = result.parameter_combination.get(parameter_name)
                metric_value = result.performance_metrics[target_metric]

                if param_value not in param_data:
                    param_data[param_value] = []
                param_data[param_value].append(metric_value)

        if len(param_data) < 2:
            logger.warning(f"Insufficient data for parameter {parameter_name} analysis")
            return HyperparameterTestResult(
                model_type=model_type,
                parameter_name=parameter_name,
                parameter_values=list(param_data.keys()),
                metric_values=[],
                statistical_significance={},
                effect_size=0.0,
                optimal_value=None,
                confidence_interval=(0.0, 0.0),
            )

        # Calculate statistical significance
        parameter_values = list(param_data.keys())
        metric_groups = [param_data[val] for val in parameter_values]

        # Perform ANOVA or Kruskal-Wallis test
        if len(metric_groups) > 2:
            stat_result = self._kruskal_wallis_test(metric_groups)
        else:
            stat_result = self._mannwhitney_u_test(metric_groups[0], metric_groups[1])

        # Calculate effect size (eta-squared for ANOVA, r for Mann-Whitney)
        effect_size = self._calculate_effect_size(metric_groups, stat_result)

        # Find optimal parameter value
        param_means = {val: np.mean(param_data[val]) for val in parameter_values}
        optimal_value = max(param_means.keys(), key=lambda x: param_means[x])

        # Calculate confidence interval for optimal value using bootstrap
        optimal_metrics = param_data[optimal_value]
        confidence_interval = self._bootstrap_confidence_interval(
            optimal_metrics, confidence_level=1 - significance_level
        )

        metric_values = [param_means[val] for val in parameter_values]

        return HyperparameterTestResult(
            model_type=model_type,
            parameter_name=parameter_name,
            parameter_values=parameter_values,
            metric_values=metric_values,
            statistical_significance=stat_result,
            effect_size=effect_size,
            optimal_value=optimal_value,
            confidence_interval=confidence_interval,
        )

    def _calculate_effect_size(
        self, metric_groups: list[list[float]], stat_result: dict[str, float]
    ) -> float:
        """Calculate effect size for parameter differences.

        Args:
            metric_groups: Groups of metric values for each parameter value
            stat_result: Statistical test results

        Returns:
            Effect size measure
        """
        # Flatten all values
        all_values = [val for group in metric_groups for val in group]
        total_n = len(all_values)

        if total_n < 3:
            return 0.0

        # Calculate eta-squared (proportion of variance explained)
        group_means = [np.mean(group) for group in metric_groups]
        grand_mean = np.mean(all_values)

        # Between-group sum of squares
        ss_between = sum(
            len(group) * (mean - grand_mean) ** 2 for group, mean in zip(metric_groups, group_means)
        )

        # Total sum of squares
        ss_total = sum((val - grand_mean) ** 2 for val in all_values)

        if ss_total == 0:
            return 0.0

        eta_squared = ss_between / ss_total
        return eta_squared

    def compare_hyperparameter_importance(
        self, model_type: str, metric: str = "sharpe_ratio"
    ) -> pd.DataFrame:
        """Compare relative importance of hyperparameters.

        Args:
            model_type: Model identifier
            metric: Performance metric for comparison

        Returns:
            DataFrame with hyperparameter importance ranking
        """
        if model_type not in self.test_results:
            raise ValueError(f"No hyperparameter results for {model_type}")

        results = self.test_results[model_type]

        importance_data = []
        for result in results:
            importance_data.append(
                {
                    "parameter": result.parameter_name,
                    "effect_size": result.effect_size,
                    "p_value": result.statistical_significance.get("p_value", 1.0),
                    "optimal_value": result.optimal_value,
                    "metric_range": (
                        max(result.metric_values) - min(result.metric_values)
                        if result.metric_values
                        else 0
                    ),
                    "significance": (
                        "Yes"
                        if result.statistical_significance.get("p_value", 1.0) < 0.05
                        else "No"
                    ),
                }
            )

        importance_df = pd.DataFrame(importance_data)

        # Sort by effect size (descending)
        importance_df = importance_df.sort_values("effect_size", ascending=False)

        return importance_df

    def get_optimal_hyperparameters(
        self, model_type: str, significance_threshold: float = 0.05
    ) -> dict[str, Any]:
        """Get optimal hyperparameters based on sensitivity analysis.

        Args:
            model_type: Model identifier
            significance_threshold: Only include significant parameters

        Returns:
            Dictionary of optimal hyperparameters
        """
        if model_type not in self.test_results:
            raise ValueError(f"No hyperparameter results for {model_type}")

        results = self.test_results[model_type]
        optimal_params = {}

        for result in results:
            p_value = result.statistical_significance.get("p_value", 1.0)

            if p_value <= significance_threshold and result.optimal_value is not None:
                optimal_params[result.parameter_name] = result.optimal_value

        logger.info(f"Found {len(optimal_params)} significant optimal parameters for {model_type}")

        return optimal_params

    def create_hyperparameter_sensitivity_report(self, model_type: str, output_path: str) -> None:
        """Create comprehensive hyperparameter sensitivity report.

        Args:
            model_type: Model identifier
            output_path: Path to save the report
        """
        if model_type not in self.test_results:
            raise ValueError(f"No hyperparameter results for {model_type}")

        results = self.test_results[model_type]

        # Create summary statistics
        report_sections = []

        report_sections.append(
            f"# Hyperparameter Sensitivity Analysis Report: {model_type.upper()}"
        )
        report_sections.append("")
        report_sections.append("## Summary")
        report_sections.append(f"- Total parameters analyzed: {len(results)}")

        significant_params = [
            r for r in results if r.statistical_significance.get("p_value", 1.0) < 0.05
        ]
        report_sections.append(f"- Significant parameters (p < 0.05): {len(significant_params)}")

        if significant_params:
            avg_effect_size = np.mean([r.effect_size for r in significant_params])
            report_sections.append(f"- Average effect size: {avg_effect_size:.4f}")

        # Parameter details
        report_sections.append("")
        report_sections.append("## Parameter Analysis")

        for result in sorted(results, key=lambda x: x.effect_size, reverse=True):
            p_value = result.statistical_significance.get("p_value", 1.0)
            significance = (
                "***"
                if p_value < 0.001
                else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            )

            report_sections.append("")
            report_sections.append(f"### {result.parameter_name} {significance}")
            report_sections.append(f"- Effect size: {result.effect_size:.4f}")
            report_sections.append(f"- P-value: {p_value:.6f}")
            report_sections.append(f"- Optimal value: {result.optimal_value}")

            if result.metric_values:
                report_sections.append(
                    f"- Performance range: {min(result.metric_values):.4f} - {max(result.metric_values):.4f}"
                )

            if result.confidence_interval:
                report_sections.append(
                    f"- 95% CI: ({result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f})"
                )

        # Optimal configuration
        optimal_params = self.get_optimal_hyperparameters(model_type)
        if optimal_params:
            report_sections.append("")
            report_sections.append("## Recommended Configuration")
            for param, value in optimal_params.items():
                report_sections.append(f"- {param}: {value}")

        # Save report
        with open(output_path, "w") as f:
            f.write("\n".join(report_sections))

        logger.info(f"Hyperparameter sensitivity report saved to {output_path}")

    def _kruskal_wallis_test(self, groups: list[list[float]]) -> dict[str, float]:
        """Perform Kruskal-Wallis test for multiple groups.

        Args:
            groups: List of metric value groups for each parameter value

        Returns:
            Dictionary with test results
        """
        from scipy.stats import kruskal

        # Filter out empty groups
        valid_groups = [group for group in groups if len(group) > 0]

        if len(valid_groups) < 2:
            return {"statistic": 0.0, "p_value": 1.0}

        try:
            statistic, p_value = kruskal(*valid_groups)
            return {"statistic": statistic, "p_value": p_value}
        except Exception as e:
            logger.warning(f"Kruskal-Wallis test failed: {e}")
            return {"statistic": 0.0, "p_value": 1.0}

    def _mannwhitney_u_test(self, group1: list[float], group2: list[float]) -> dict[str, float]:
        """Perform Mann-Whitney U test for two groups.

        Args:
            group1: First group of metric values
            group2: Second group of metric values

        Returns:
            Dictionary with test results
        """
        from scipy.stats import mannwhitneyu

        if len(group1) == 0 or len(group2) == 0:
            return {"statistic": 0.0, "p_value": 1.0}

        try:
            statistic, p_value = mannwhitneyu(group1, group2, alternative="two-sided")
            return {"statistic": statistic, "p_value": p_value}
        except Exception as e:
            logger.warning(f"Mann-Whitney U test failed: {e}")
            return {"statistic": 0.0, "p_value": 1.0}

    def _bootstrap_confidence_interval(
        self, data: list[float], confidence_level: float = 0.95
    ) -> tuple:
        """Calculate bootstrap confidence interval.

        Args:
            data: Data for confidence interval calculation
            confidence_level: Confidence level

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if len(data) == 0:
            return (0.0, 0.0)

        try:
            # Simple percentile bootstrap
            n_bootstrap = 1000
            bootstrap_means = []

            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))

            alpha = 1 - confidence_level
            lower_percentile = 100 * (alpha / 2)
            upper_percentile = 100 * (1 - alpha / 2)

            lower_bound = np.percentile(bootstrap_means, lower_percentile)
            upper_bound = np.percentile(bootstrap_means, upper_percentile)

            return (lower_bound, upper_bound)

        except Exception as e:
            logger.warning(f"Bootstrap confidence interval calculation failed: {e}")
            return (0.0, 0.0)

    def test_multiple_hyperparameters_with_correction(
        self,
        model_type: str,
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
        target_metric: str = "sharpe_ratio",
        significance_level: float = 0.05,
        correction_method: str = "bonferroni",
    ) -> list[HyperparameterTestResult]:
        """Test multiple hyperparameters with multiple comparison correction.

        Args:
            model_type: Model identifier
            data: Market data
            start_date: Analysis start date
            end_date: Analysis end date
            target_metric: Performance metric
            significance_level: Statistical significance threshold
            correction_method: Multiple comparison correction method

        Returns:
            List of hyperparameter test results with corrected p-values
        """
        # First run standard hyperparameter testing
        results = self.test_hyperparameter_sensitivity(
            model_type, data, start_date, end_date, target_metric, significance_level
        )

        # Extract p-values for correction
        p_values = [result.statistical_significance.get("p_value", 1.0) for result in results]

        # Apply multiple comparison correction
        if correction_method == "bonferroni":
            correction_result = self.multiple_comparison_corrector.bonferroni_correction(
                p_values, alpha=significance_level
            )
        else:
            logger.warning(f"Unsupported correction method: {correction_method}")
            return results

        # Update results with corrected p-values
        corrected_p_values = correction_result["corrected_p_values"]
        rejected = correction_result["rejected"]

        for i, result in enumerate(results):
            if i < len(corrected_p_values):
                result.statistical_significance["corrected_p_value"] = corrected_p_values[i]
                result.statistical_significance["significant_after_correction"] = rejected[i]

        logger.info(
            f"Applied {correction_method} correction. "
            f"Significant parameters: {sum(rejected)}/{len(results)}"
        )

        return results
