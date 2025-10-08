"""Transaction cost sensitivity analysis framework."""

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics.returns import PerformanceAnalytics
from src.evaluation.validation.significance import StatisticalValidation

from .engine import SensitivityAnalysisEngine, SensitivityResult

logger = logging.getLogger(__name__)


@dataclass
class TransactionCostScenario:
    """Transaction cost scenario definition."""

    name: str
    cost_bps: float
    description: str


@dataclass
class CostImpactResult:
    """Results from transaction cost impact analysis."""

    model_type: str
    cost_scenario: TransactionCostScenario
    original_metrics: dict[str, float]
    cost_adjusted_metrics: dict[str, float]
    performance_impact: dict[str, float]
    ranking_change: int
    statistical_significance: dict[str, float]


class TransactionCostAnalyzer:
    """Comprehensive transaction cost sensitivity testing framework."""

    def __init__(
        self,
        sensitivity_engine: SensitivityAnalysisEngine,
        performance_analytics: PerformanceAnalytics,
        statistical_validator: StatisticalValidation,
    ):
        """Initialize transaction cost analyzer.

        Args:
            sensitivity_engine: Core sensitivity analysis engine
            performance_analytics: Performance metrics calculator
            statistical_validator: Statistical significance testing
        """
        self.sensitivity_engine = sensitivity_engine
        self.performance_analytics = performance_analytics
        self.statistical_validator = statistical_validator

        # Define standard cost scenarios
        self.cost_scenarios = [
            TransactionCostScenario(
                name="aggressive",
                cost_bps=5.0,
                description="Aggressive trading with low transaction costs",
            ),
            TransactionCostScenario(
                name="baseline", cost_bps=10.0, description="Baseline transaction costs"
            ),
            TransactionCostScenario(
                name="conservative",
                cost_bps=20.0,
                description="Conservative trading with higher transaction costs",
            ),
        ]

        self.cost_impact_results: dict[str, list[CostImpactResult]] = {}

    def analyze_transaction_cost_impact(
        self,
        model_types: list[str],
        base_results: dict[str, list[SensitivityResult]],
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
        target_metric: str = "sharpe_ratio",
    ) -> dict[str, list[CostImpactResult]]:
        """Analyze transaction cost impact across all models and scenarios.

        Args:
            model_types: List of model types to analyze
            base_results: Baseline sensitivity results without cost adjustments
            data: Market data for backtesting
            start_date: Analysis start date
            end_date: Analysis end date
            target_metric: Performance metric for analysis

        Returns:
            Dictionary mapping model types to cost impact results
        """
        logger.info(f"Starting transaction cost impact analysis for models: {model_types}")

        all_cost_results = {}

        for model_type in model_types:
            logger.info(f"Analyzing transaction cost impact for {model_type}")

            if model_type not in base_results:
                logger.warning(f"No base results available for {model_type}")
                continue

            model_cost_results = []

            # Get best performing configuration for this model
            best_config = self._get_best_configuration(base_results[model_type], target_metric)

            if not best_config:
                logger.warning(f"No valid configuration found for {model_type}")
                continue

            # Analyze each cost scenario
            for scenario in self.cost_scenarios:
                cost_result = self._analyze_cost_scenario(
                    model_type=model_type,
                    base_config=best_config,
                    cost_scenario=scenario,
                    data=data,
                    start_date=start_date,
                    end_date=end_date,
                    target_metric=target_metric,
                )

                if cost_result:
                    model_cost_results.append(cost_result)

            all_cost_results[model_type] = model_cost_results

        # Calculate ranking changes across models and scenarios
        self._calculate_ranking_stability(all_cost_results, target_metric)

        self.cost_impact_results = all_cost_results

        logger.info(
            f"Transaction cost impact analysis completed for {len(all_cost_results)} models"
        )

        return all_cost_results

    def _get_best_configuration(
        self, model_results: list[SensitivityResult], target_metric: str
    ) -> Optional[SensitivityResult]:
        """Get best performing configuration for a model.

        Args:
            model_results: List of sensitivity results for the model
            target_metric: Performance metric for ranking

        Returns:
            Best performing configuration or None
        """
        valid_results = [
            r for r in model_results if r.error is None and target_metric in r.performance_metrics
        ]

        if not valid_results:
            return None

        # Sort by target metric (descending)
        valid_results.sort(key=lambda x: x.performance_metrics[target_metric], reverse=True)

        return valid_results[0]

    def _analyze_cost_scenario(
        self,
        model_type: str,
        base_config: SensitivityResult,
        cost_scenario: TransactionCostScenario,
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
        target_metric: str,
    ) -> Optional[CostImpactResult]:
        """Analyze impact of specific cost scenario.

        Args:
            model_type: Model identifier
            base_config: Base configuration result
            cost_scenario: Transaction cost scenario
            data: Market data
            start_date: Analysis start date
            end_date: Analysis end date
            target_metric: Performance metric for analysis

        Returns:
            Cost impact result or None if analysis fails
        """
        try:
            # Create modified configuration with new transaction costs
            modified_params = base_config.parameter_combination.copy()
            modified_params["transaction_cost_bps"] = cost_scenario.cost_bps

            # Execute backtest with modified costs
            cost_adjusted_result = self.sensitivity_engine._execute_parameter_combination(
                model_type=model_type,
                parameters=modified_params,
                data=data,
                start_date=start_date,
                end_date=end_date,
            )

            if cost_adjusted_result.error:
                logger.error(
                    f"Cost scenario {cost_scenario.name} failed for {model_type}: {cost_adjusted_result.error}"
                )
                return None

            # Calculate performance impact
            performance_impact = self._calculate_performance_impact(
                base_config.performance_metrics, cost_adjusted_result.performance_metrics
            )

            # Statistical significance test
            if (
                len(base_config.backtest_results) > 0
                and len(cost_adjusted_result.backtest_results) > 0
            ):
                base_returns = base_config.backtest_results.get("returns", pd.Series())
                cost_returns = cost_adjusted_result.backtest_results.get("returns", pd.Series())

                if len(base_returns) > 0 and len(cost_returns) > 0:
                    stat_result = self.statistical_validator.sharpe_ratio_test(
                        base_returns, cost_returns
                    )
                else:
                    stat_result = {"p_value": 1.0, "is_significant": False}
            else:
                stat_result = {"p_value": 1.0, "is_significant": False}

            return CostImpactResult(
                model_type=model_type,
                cost_scenario=cost_scenario,
                original_metrics=base_config.performance_metrics,
                cost_adjusted_metrics=cost_adjusted_result.performance_metrics,
                performance_impact=performance_impact,
                ranking_change=0,  # Will be calculated later
                statistical_significance=stat_result,
            )

        except Exception as e:
            logger.error(
                f"Error analyzing cost scenario {cost_scenario.name} for {model_type}: {e}"
            )
            return None

    def _calculate_performance_impact(
        self, original_metrics: dict[str, float], cost_adjusted_metrics: dict[str, float]
    ) -> dict[str, float]:
        """Calculate performance impact due to transaction costs.

        Args:
            original_metrics: Original performance metrics
            cost_adjusted_metrics: Cost-adjusted performance metrics

        Returns:
            Dictionary of performance impacts
        """
        impact = {}

        for metric_name in original_metrics.keys():
            if metric_name in cost_adjusted_metrics:
                original_value = original_metrics[metric_name]
                adjusted_value = cost_adjusted_metrics[metric_name]

                # Calculate absolute and relative impact
                absolute_impact = adjusted_value - original_value
                relative_impact = (
                    (absolute_impact / abs(original_value)) * 100 if original_value != 0 else 0
                )

                impact[f"{metric_name}_absolute"] = absolute_impact
                impact[f"{metric_name}_relative"] = relative_impact

        return impact

    def _calculate_ranking_stability(
        self, cost_results: dict[str, list[CostImpactResult]], target_metric: str
    ) -> None:
        """Calculate ranking changes across cost scenarios.

        Args:
            cost_results: Cost impact results for all models
            target_metric: Performance metric for ranking
        """
        # Create baseline rankings (using baseline cost scenario)
        baseline_rankings = {}

        for model_type, results in cost_results.items():
            baseline_result = next((r for r in results if r.cost_scenario.name == "baseline"), None)
            if baseline_result:
                baseline_rankings[model_type] = baseline_result.cost_adjusted_metrics.get(
                    target_metric, 0.0
                )

        # Sort by performance to get ranking
        sorted_baseline = sorted(baseline_rankings.items(), key=lambda x: x[1], reverse=True)
        baseline_rank_map = {model: idx for idx, (model, _) in enumerate(sorted_baseline)}

        # Calculate ranking changes for each scenario
        for scenario in self.cost_scenarios:
            if scenario.name == "baseline":
                continue

            scenario_rankings = {}
            for model_type, results in cost_results.items():
                scenario_result = next(
                    (r for r in results if r.cost_scenario.name == scenario.name), None
                )
                if scenario_result:
                    scenario_rankings[model_type] = scenario_result.cost_adjusted_metrics.get(
                        target_metric, 0.0
                    )

            # Sort by performance to get new ranking
            sorted_scenario = sorted(scenario_rankings.items(), key=lambda x: x[1], reverse=True)
            scenario_rank_map = {model: idx for idx, (model, _) in enumerate(sorted_scenario)}

            # Update ranking change in results
            for model_type, results in cost_results.items():
                for result in results:
                    if result.cost_scenario.name == scenario.name:
                        baseline_rank = baseline_rank_map.get(model_type, 0)
                        new_rank = scenario_rank_map.get(model_type, 0)
                        result.ranking_change = new_rank - baseline_rank

    def analyze_ranking_stability(self, target_metric: str = "sharpe_ratio") -> pd.DataFrame:
        """Analyze ranking stability across cost scenarios.

        Args:
            target_metric: Performance metric for ranking analysis

        Returns:
            DataFrame with ranking stability analysis
        """
        if not self.cost_impact_results:
            raise ValueError(
                "No cost impact results available. Run analyze_transaction_cost_impact first."
            )

        stability_data = []

        for model_type, results in self.cost_impact_results.items():
            model_data = {
                "model_type": model_type,
                "max_ranking_change": 0,
                "avg_performance_impact": 0.0,
                "significant_impacts": 0,
            }

            performance_impacts = []
            max_ranking_change = 0
            significant_count = 0

            for result in results:
                # Track maximum ranking change
                max_ranking_change = max(max_ranking_change, abs(result.ranking_change))

                # Track performance impacts
                if f"{target_metric}_relative" in result.performance_impact:
                    impact = result.performance_impact[f"{target_metric}_relative"]
                    performance_impacts.append(abs(impact))

                # Count significant impacts
                if result.statistical_significance.get("is_significant", False):
                    significant_count += 1

                # Add scenario-specific data
                scenario_data = model_data.copy()
                scenario_data.update(
                    {
                        "cost_scenario": result.cost_scenario.name,
                        "cost_bps": result.cost_scenario.cost_bps,
                        "ranking_change": result.ranking_change,
                        "performance_impact_pct": result.performance_impact.get(
                            f"{target_metric}_relative", 0.0
                        ),
                        "is_significant": result.statistical_significance.get(
                            "is_significant", False
                        ),
                        "p_value": result.statistical_significance.get("p_value", 1.0),
                    }
                )
                stability_data.append(scenario_data)

            # Update summary statistics
            model_data["max_ranking_change"] = max_ranking_change
            model_data["avg_performance_impact"] = (
                np.mean(performance_impacts) if performance_impacts else 0.0
            )
            model_data["significant_impacts"] = significant_count

        return pd.DataFrame(stability_data)

    def get_cost_sensitivity_summary(self, target_metric: str = "sharpe_ratio") -> dict[str, Any]:
        """Get summary of cost sensitivity analysis.

        Args:
            target_metric: Performance metric for summary

        Returns:
            Dictionary with cost sensitivity summary
        """
        if not self.cost_impact_results:
            raise ValueError("No cost impact results available.")

        summary = {
            "models_analyzed": len(self.cost_impact_results),
            "cost_scenarios": len(self.cost_scenarios),
            "most_sensitive_model": None,
            "least_sensitive_model": None,
            "avg_impact_by_scenario": {},
            "ranking_stability_score": 0.0,
        }

        # Calculate model sensitivity scores
        model_sensitivity = {}

        for model_type, results in self.cost_impact_results.items():
            impacts = []
            ranking_changes = []

            for result in results:
                if f"{target_metric}_relative" in result.performance_impact:
                    impacts.append(abs(result.performance_impact[f"{target_metric}_relative"]))
                ranking_changes.append(abs(result.ranking_change))

            # Sensitivity score combines performance impact and ranking stability
            avg_impact = np.mean(impacts) if impacts else 0.0
            avg_ranking_change = np.mean(ranking_changes) if ranking_changes else 0.0

            sensitivity_score = avg_impact + (
                avg_ranking_change * 10
            )  # Weight ranking changes more
            model_sensitivity[model_type] = sensitivity_score

        if model_sensitivity:
            summary["most_sensitive_model"] = max(
                model_sensitivity.keys(), key=lambda x: model_sensitivity[x]
            )
            summary["least_sensitive_model"] = min(
                model_sensitivity.keys(), key=lambda x: model_sensitivity[x]
            )

        # Calculate average impact by scenario
        scenario_impacts = {scenario.name: [] for scenario in self.cost_scenarios}

        for results in self.cost_impact_results.values():
            for result in results:
                if f"{target_metric}_relative" in result.performance_impact:
                    impact = abs(result.performance_impact[f"{target_metric}_relative"])
                    scenario_impacts[result.cost_scenario.name].append(impact)

        for scenario_name, impacts in scenario_impacts.items():
            summary["avg_impact_by_scenario"][scenario_name] = np.mean(impacts) if impacts else 0.0

        # Calculate overall ranking stability score (lower is more stable)
        all_ranking_changes = []
        for results in self.cost_impact_results.values():
            for result in results:
                all_ranking_changes.append(abs(result.ranking_change))

        summary["ranking_stability_score"] = (
            np.mean(all_ranking_changes) if all_ranking_changes else 0.0
        )

        return summary

    def create_cost_impact_visualization_data(
        self, target_metric: str = "sharpe_ratio"
    ) -> dict[str, pd.DataFrame]:
        """Create data for cost impact visualizations.

        Args:
            target_metric: Performance metric for visualization

        Returns:
            Dictionary containing DataFrames for different visualizations
        """
        if not self.cost_impact_results:
            raise ValueError("No cost impact results available.")

        visualization_data = {}

        # 1. Performance Impact Heatmap Data
        heatmap_data = []
        for model_type, results in self.cost_impact_results.items():
            for result in results:
                heatmap_data.append(
                    {
                        "model": model_type,
                        "cost_scenario": result.cost_scenario.name,
                        "cost_bps": result.cost_scenario.cost_bps,
                        "performance_impact": result.performance_impact.get(
                            f"{target_metric}_relative", 0.0
                        ),
                        "is_significant": result.statistical_significance.get(
                            "is_significant", False
                        ),
                    }
                )

        visualization_data["heatmap"] = pd.DataFrame(heatmap_data)

        # 2. Ranking Stability Data
        ranking_data = []
        for model_type, results in self.cost_impact_results.items():
            for result in results:
                ranking_data.append(
                    {
                        "model": model_type,
                        "cost_scenario": result.cost_scenario.name,
                        "cost_bps": result.cost_scenario.cost_bps,
                        "ranking_change": result.ranking_change,
                        "original_performance": result.original_metrics.get(target_metric, 0.0),
                        "adjusted_performance": result.cost_adjusted_metrics.get(
                            target_metric, 0.0
                        ),
                    }
                )

        visualization_data["ranking"] = pd.DataFrame(ranking_data)

        # 3. Cost Sensitivity Distribution Data
        sensitivity_data = []
        for model_type, results in self.cost_impact_results.items():
            impacts = [
                result.performance_impact.get(f"{target_metric}_relative", 0.0)
                for result in results
            ]
            sensitivity_data.append(
                {
                    "model": model_type,
                    "mean_impact": np.mean([abs(x) for x in impacts]),
                    "std_impact": np.std([abs(x) for x in impacts]),
                    "max_impact": max([abs(x) for x in impacts]) if impacts else 0.0,
                    "min_impact": min([abs(x) for x in impacts]) if impacts else 0.0,
                }
            )

        visualization_data["sensitivity"] = pd.DataFrame(sensitivity_data)

        return visualization_data

    def export_cost_analysis_results(self, filepath: str) -> None:
        """Export cost analysis results to file.

        Args:
            filepath: Path to save results
        """
        if not self.cost_impact_results:
            raise ValueError("No cost impact results available.")

        export_data = []

        for model_type, results in self.cost_impact_results.items():
            for result in results:
                row = {
                    "model_type": model_type,
                    "cost_scenario": result.cost_scenario.name,
                    "cost_bps": result.cost_scenario.cost_bps,
                    "ranking_change": result.ranking_change,
                    "is_significant": result.statistical_significance.get("is_significant", False),
                    "p_value": result.statistical_significance.get("p_value", 1.0),
                }

                # Add original metrics
                for metric, value in result.original_metrics.items():
                    row[f"original_{metric}"] = value

                # Add cost-adjusted metrics
                for metric, value in result.cost_adjusted_metrics.items():
                    row[f"adjusted_{metric}"] = value

                # Add performance impacts
                for metric, value in result.performance_impact.items():
                    row[f"impact_{metric}"] = value

                export_data.append(row)

        df = pd.DataFrame(export_data)

        if filepath.endswith(".csv"):
            df.to_csv(filepath, index=False)
        elif filepath.endswith(".parquet"):
            df.to_parquet(filepath, index=False)
        else:
            raise ValueError("Unsupported file format. Use .csv or .parquet")

        logger.info(f"Transaction cost analysis results exported to {filepath}")
