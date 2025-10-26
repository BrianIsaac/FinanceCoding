"""Constraint violation analysis framework."""

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
class ConstraintViolation:
    """Constraint violation record."""

    constraint_type: str
    violation_magnitude: float
    violation_frequency: float
    violation_dates: list[str]
    impact_on_performance: float
    penalty_applied: float


@dataclass
class ConstraintAnalysisResult:
    """Results from constraint violation analysis."""

    model_type: str
    parameter_combination: dict[str, Any]
    constraint_violations: list[ConstraintViolation]
    total_violation_frequency: float
    avg_violation_magnitude: float
    performance_with_enforcement: dict[str, float]
    performance_without_enforcement: dict[str, float]
    enforcement_impact: dict[str, float]
    robustness_score: float
    statistical_significance: dict[str, float]


class ConstraintAnalyzer:
    """Constraint violation monitoring and impact analysis framework."""

    def __init__(
        self,
        sensitivity_engine: SensitivityAnalysisEngine,
        performance_analytics: PerformanceAnalytics,
        statistical_validator: StatisticalValidation,
    ):
        """Initialize constraint analyzer.

        Args:
            sensitivity_engine: Core sensitivity analysis engine
            performance_analytics: Performance metrics calculator
            statistical_validator: Statistical significance testing
        """
        self.sensitivity_engine = sensitivity_engine
        self.performance_analytics = performance_analytics
        self.statistical_validator = statistical_validator

        # Define constraint types to monitor
        self.constraint_types = {
            "position_weight": {
                "max_weight": 0.10,
                "min_weight": 0.01,
                "description": "Individual position weight limits",
            },
            "turnover": {
                "max_monthly_turnover": 0.20,
                "description": "Monthly portfolio turnover limit",
            },
            "sector_concentration": {
                "max_sector_weight": 0.25,
                "description": "Maximum sector concentration",
            },
            "long_only": {"min_weight": 0.0, "description": "No short positions allowed"},
            "top_k_positions": {"max_positions": 100, "description": "Maximum number of positions"},
        }

        self.constraint_analysis_results: dict[str, list[ConstraintAnalysisResult]] = {}

    def analyze_constraint_violations(
        self,
        model_types: list[str],
        sensitivity_results: dict[str, list[SensitivityResult]],
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
        constraint_enforcement_levels: list[str] = None,
    ) -> dict[str, list[ConstraintAnalysisResult]]:
        """Analyze constraint violations across models and configurations.

        Args:
            model_types: List of model types to analyze
            sensitivity_results: Sensitivity analysis results
            data: Market data for backtesting
            start_date: Analysis start date
            end_date: Analysis end date
            constraint_enforcement_levels: List of enforcement levels to test

        Returns:
            Dictionary mapping model types to constraint analysis results
        """
        logger.info(f"Starting constraint violation analysis for models: {model_types}")

        if constraint_enforcement_levels is None:
            constraint_enforcement_levels = ["strict", "moderate", "relaxed"]

        all_constraint_results = {}

        for model_type in model_types:
            logger.info(f"Analyzing constraint violations for {model_type}")

            if model_type not in sensitivity_results:
                logger.warning(f"No sensitivity results available for {model_type}")
                continue

            model_constraint_results = []

            # Analyze top performing configurations
            top_configs = self._get_top_configurations(sensitivity_results[model_type], top_k=5)

            for config in top_configs:
                for enforcement_level in constraint_enforcement_levels:
                    constraint_result = self._analyze_configuration_constraints(
                        model_type=model_type,
                        config=config,
                        enforcement_level=enforcement_level,
                        data=data,
                        start_date=start_date,
                        end_date=end_date,
                    )

                    if constraint_result:
                        model_constraint_results.append(constraint_result)

            all_constraint_results[model_type] = model_constraint_results

        self.constraint_analysis_results = all_constraint_results

        logger.info(
            f"Constraint violation analysis completed for {len(all_constraint_results)} models"
        )

        return all_constraint_results

    def _get_top_configurations(
        self, sensitivity_results: list[SensitivityResult], top_k: int = 5
    ) -> list[SensitivityResult]:
        """Get top performing configurations for constraint analysis.

        Args:
            sensitivity_results: List of sensitivity results
            top_k: Number of top configurations to return

        Returns:
            List of top performing configurations
        """
        valid_results = [
            r
            for r in sensitivity_results
            if r.error is None and "sharpe_ratio" in r.performance_metrics
        ]

        if not valid_results:
            return []

        # Sort by Sharpe ratio (descending)
        valid_results.sort(key=lambda x: x.performance_metrics["sharpe_ratio"], reverse=True)

        return valid_results[:top_k]

    def _analyze_configuration_constraints(
        self,
        model_type: str,
        config: SensitivityResult,
        enforcement_level: str,
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> Optional[ConstraintAnalysisResult]:
        """Analyze constraint violations for a specific configuration.

        Args:
            model_type: Model identifier
            config: Configuration to analyze
            enforcement_level: Constraint enforcement level
            data: Market data
            start_date: Analysis start date
            end_date: Analysis end date

        Returns:
            Constraint analysis result or None if analysis fails
        """
        try:
            # Create constraint enforcement configuration
            enforcement_params = self._create_enforcement_parameters(enforcement_level)

            # Run backtest with strict constraint enforcement
            strict_params = config.parameter_combination.copy()
            strict_params.update(enforcement_params["strict"])

            strict_result = self.sensitivity_engine._execute_parameter_combination(
                model_type=model_type,
                parameters=strict_params,
                data=data,
                start_date=start_date,
                end_date=end_date,
            )

            # Run backtest with relaxed constraints
            relaxed_params = config.parameter_combination.copy()
            relaxed_params.update(enforcement_params["relaxed"])

            relaxed_result = self.sensitivity_engine._execute_parameter_combination(
                model_type=model_type,
                parameters=relaxed_params,
                data=data,
                start_date=start_date,
                end_date=end_date,
            )

            if strict_result.error or relaxed_result.error:
                logger.error(f"Constraint analysis failed for {model_type}")
                return None

            # Analyze constraint violations
            violations = self._detect_constraint_violations(
                strict_result.backtest_results, enforcement_params["strict"]
            )

            # Calculate enforcement impact
            enforcement_impact = self._calculate_enforcement_impact(
                strict_result.performance_metrics, relaxed_result.performance_metrics
            )

            # Calculate robustness score
            robustness_score = self._calculate_robustness_score(violations, enforcement_impact)

            # Statistical significance test
            stat_result = self._test_enforcement_significance(
                strict_result.backtest_results, relaxed_result.backtest_results
            )

            return ConstraintAnalysisResult(
                model_type=model_type,
                parameter_combination=config.parameter_combination,
                constraint_violations=violations,
                total_violation_frequency=(
                    sum(v.violation_frequency for v in violations) / len(violations)
                    if violations
                    else 0.0
                ),
                avg_violation_magnitude=(
                    sum(v.violation_magnitude for v in violations) / len(violations)
                    if violations
                    else 0.0
                ),
                performance_with_enforcement=strict_result.performance_metrics,
                performance_without_enforcement=relaxed_result.performance_metrics,
                enforcement_impact=enforcement_impact,
                robustness_score=robustness_score,
                statistical_significance=stat_result,
            )

        except Exception as e:
            logger.error(f"Error in constraint analysis for {model_type}: {e}")
            return None

    def _create_enforcement_parameters(self, enforcement_level: str) -> dict[str, dict[str, Any]]:
        """Create constraint enforcement parameters.

        Args:
            enforcement_level: Level of constraint enforcement

        Returns:
            Dictionary with strict and relaxed constraint parameters
        """
        enforcement_params = {
            "strict": {
                "max_position_weight": 0.05,
                "max_monthly_turnover": 0.15,
                "max_sector_weight": 0.20,
                "enforce_long_only": True,
                "constraint_penalty_multiplier": 2.0,
            },
            "relaxed": {
                "max_position_weight": 0.15,
                "max_monthly_turnover": 0.30,
                "max_sector_weight": 0.40,
                "enforce_long_only": False,
                "constraint_penalty_multiplier": 0.5,
            },
        }

        if enforcement_level == "moderate":
            # Moderate enforcement is between strict and relaxed
            enforcement_params["strict"]["max_position_weight"] = 0.08
            enforcement_params["strict"]["max_monthly_turnover"] = 0.20
            enforcement_params["strict"]["constraint_penalty_multiplier"] = 1.5

        return enforcement_params

    def _detect_constraint_violations(
        self, backtest_results: pd.DataFrame, constraint_params: dict[str, Any]
    ) -> list[ConstraintViolation]:
        """Detect constraint violations in backtest results.

        Args:
            backtest_results: Backtest results to analyze
            constraint_params: Constraint parameters to check against

        Returns:
            List of constraint violations detected
        """
        violations = []

        try:
            # Position weight violations
            if "weights" in backtest_results.columns:
                weight_violations = self._detect_weight_violations(
                    backtest_results["weights"], constraint_params.get("max_position_weight", 0.10)
                )
                violations.extend(weight_violations)

            # Turnover violations
            if "turnover" in backtest_results.columns:
                turnover_violations = self._detect_turnover_violations(
                    backtest_results["turnover"],
                    constraint_params.get("max_monthly_turnover", 0.20),
                )
                violations.extend(turnover_violations)

            # Sector concentration violations
            if "sector_weights" in backtest_results.columns:
                sector_violations = self._detect_sector_violations(
                    backtest_results["sector_weights"],
                    constraint_params.get("max_sector_weight", 0.25),
                )
                violations.extend(sector_violations)

        except Exception as e:
            logger.warning(f"Error detecting constraint violations: {e}")

        return violations

    def _detect_weight_violations(
        self, weights_series: pd.Series, max_weight: float
    ) -> list[ConstraintViolation]:
        """Detect position weight constraint violations.

        Args:
            weights_series: Series of position weights over time
            max_weight: Maximum allowed position weight

        Returns:
            List of weight violations
        """
        violations = []
        violation_dates = []
        violation_magnitudes = []

        try:
            for date, weights in weights_series.items():
                if pd.notna(weights) and hasattr(weights, "__len__"):
                    weights_array = np.array(weights) if isinstance(weights, list) else weights
                    max_position_weight = np.max(weights_array) if len(weights_array) > 0 else 0.0

                    if max_position_weight > max_weight:
                        violation_dates.append(str(date))
                        violation_magnitudes.append(max_position_weight - max_weight)

            if violation_dates:
                violations.append(
                    ConstraintViolation(
                        constraint_type="position_weight",
                        violation_magnitude=np.mean(violation_magnitudes),
                        violation_frequency=len(violation_dates) / len(weights_series),
                        violation_dates=violation_dates,
                        impact_on_performance=0.0,  # Will be calculated separately
                        penalty_applied=0.0,
                    )
                )

        except Exception as e:
            logger.warning(f"Error detecting weight violations: {e}")

        return violations

    def _detect_turnover_violations(
        self, turnover_series: pd.Series, max_turnover: float
    ) -> list[ConstraintViolation]:
        """Detect turnover constraint violations.

        Args:
            turnover_series: Series of portfolio turnover over time
            max_turnover: Maximum allowed monthly turnover

        Returns:
            List of turnover violations
        """
        violations = []

        try:
            turnover_violations = turnover_series[turnover_series > max_turnover]

            if len(turnover_violations) > 0:
                violations.append(
                    ConstraintViolation(
                        constraint_type="turnover",
                        violation_magnitude=float(turnover_violations.mean() - max_turnover),
                        violation_frequency=len(turnover_violations) / len(turnover_series),
                        violation_dates=[str(d) for d in turnover_violations.index],
                        impact_on_performance=0.0,
                        penalty_applied=0.0,
                    )
                )

        except Exception as e:
            logger.warning(f"Error detecting turnover violations: {e}")

        return violations

    def _detect_sector_violations(
        self, sector_weights_series: pd.Series, max_sector_weight: float
    ) -> list[ConstraintViolation]:
        """Detect sector concentration constraint violations.

        Args:
            sector_weights_series: Series of sector weights over time
            max_sector_weight: Maximum allowed sector weight

        Returns:
            List of sector violations
        """
        violations = []
        violation_dates = []
        violation_magnitudes = []

        try:
            for date, sector_weights in sector_weights_series.items():
                if pd.notna(sector_weights) and isinstance(sector_weights, dict):
                    max_sector = max(sector_weights.values()) if sector_weights else 0.0

                    if max_sector > max_sector_weight:
                        violation_dates.append(str(date))
                        violation_magnitudes.append(max_sector - max_sector_weight)

            if violation_dates:
                violations.append(
                    ConstraintViolation(
                        constraint_type="sector_concentration",
                        violation_magnitude=np.mean(violation_magnitudes),
                        violation_frequency=len(violation_dates) / len(sector_weights_series),
                        violation_dates=violation_dates,
                        impact_on_performance=0.0,
                        penalty_applied=0.0,
                    )
                )

        except Exception as e:
            logger.warning(f"Error detecting sector violations: {e}")

        return violations

    def _calculate_enforcement_impact(
        self, enforced_metrics: dict[str, float], relaxed_metrics: dict[str, float]
    ) -> dict[str, float]:
        """Calculate performance impact of constraint enforcement.

        Args:
            enforced_metrics: Performance metrics with strict enforcement
            relaxed_metrics: Performance metrics with relaxed constraints

        Returns:
            Dictionary of enforcement impacts
        """
        impact = {}

        for metric_name in enforced_metrics.keys():
            if metric_name in relaxed_metrics:
                enforced_value = enforced_metrics[metric_name]
                relaxed_value = relaxed_metrics[metric_name]

                # Calculate absolute and relative impact
                absolute_impact = enforced_value - relaxed_value
                relative_impact = (
                    (absolute_impact / abs(relaxed_value)) * 100 if relaxed_value != 0 else 0.0
                )

                impact[f"{metric_name}_absolute"] = absolute_impact
                impact[f"{metric_name}_relative"] = relative_impact

        return impact

    def _calculate_robustness_score(
        self, violations: list[ConstraintViolation], enforcement_impact: dict[str, float]
    ) -> float:
        """Calculate robustness score based on violations and impact.

        Args:
            violations: List of constraint violations
            enforcement_impact: Performance impact of enforcement

        Returns:
            Robustness score (higher = more robust)
        """
        if not violations:
            return 1.0  # Perfect robustness if no violations

        # Calculate violation severity
        avg_frequency = sum(v.violation_frequency for v in violations) / len(violations)
        avg_magnitude = sum(v.violation_magnitude for v in violations) / len(violations)

        # Calculate performance stability
        sharpe_impact = abs(enforcement_impact.get("sharpe_ratio_relative", 0.0)) / 100.0

        # Robustness score combines low violation frequency/magnitude with stable performance
        violation_penalty = (avg_frequency * 0.5) + (avg_magnitude * 0.3)
        performance_penalty = sharpe_impact * 0.2

        robustness_score = 1.0 - min(violation_penalty + performance_penalty, 1.0)

        return max(robustness_score, 0.0)

    def _test_enforcement_significance(
        self, enforced_results: pd.DataFrame, relaxed_results: pd.DataFrame
    ) -> dict[str, float]:
        """Test statistical significance of constraint enforcement impact.

        Args:
            enforced_results: Results with strict enforcement
            relaxed_results: Results with relaxed constraints

        Returns:
            Dictionary with statistical test results
        """
        try:
            enforced_returns = (
                enforced_results.get("returns", pd.Series())
                if isinstance(enforced_results, pd.DataFrame)
                else pd.Series()
            )
            relaxed_returns = (
                relaxed_results.get("returns", pd.Series())
                if isinstance(relaxed_results, pd.DataFrame)
                else pd.Series()
            )

            if len(enforced_returns) > 0 and len(relaxed_returns) > 0:
                return self.statistical_validator.sharpe_ratio_test(
                    enforced_returns, relaxed_returns
                )
            else:
                return {"p_value": 1.0, "is_significant": False, "test_statistic": 0.0}

        except Exception as e:
            logger.warning(f"Error in enforcement significance test: {e}")
            return {"p_value": 1.0, "is_significant": False, "test_statistic": 0.0}

    def get_constraint_optimization_recommendations(self) -> dict[str, dict[str, Any]]:
        """Generate constraint optimization recommendations with robustness analysis.

        Returns:
            Dictionary mapping model types to constraint recommendations
        """
        if not self.constraint_analysis_results:
            raise ValueError("No constraint analysis results available.")

        recommendations = {}

        for model_type, results in self.constraint_analysis_results.items():
            if not results:
                continue

            # Find configuration with best robustness score
            best_result = max(results, key=lambda x: x.robustness_score)

            # Calculate aggregate statistics
            avg_violation_freq = np.mean([r.total_violation_frequency for r in results])
            np.mean([r.robustness_score for r in results])

            # Determine constraint sensitivity
            performance_impacts = [
                r.enforcement_impact.get("sharpe_ratio_relative", 0.0) for r in results
            ]
            avg_performance_impact = np.mean([abs(x) for x in performance_impacts])

            # Generate recommendations
            if avg_violation_freq > 0.2:
                constraint_recommendation = "Relax constraints - frequent violations detected"
            elif avg_performance_impact > 10.0:
                constraint_recommendation = "Moderate enforcement - high performance impact"
            else:
                constraint_recommendation = "Maintain current constraints - good balance"

            recommendations[model_type] = {
                "optimal_robustness_score": best_result.robustness_score,
                "avg_violation_frequency": avg_violation_freq,
                "avg_performance_impact": avg_performance_impact,
                "constraint_recommendation": constraint_recommendation,
                "most_violated_constraint": self._identify_most_violated_constraint(results),
                "enforcement_sensitivity": "High" if avg_performance_impact > 5.0 else "Low",
                "optimal_parameters": best_result.parameter_combination,
                "confidence_level": 1 - best_result.statistical_significance.get("p_value", 1.0),
            }

        return recommendations

    def _identify_most_violated_constraint(self, results: list[ConstraintAnalysisResult]) -> str:
        """Identify the most frequently violated constraint type.

        Args:
            results: List of constraint analysis results

        Returns:
            Most violated constraint type
        """
        constraint_frequencies = {}

        for result in results:
            for violation in result.constraint_violations:
                constraint_type = violation.constraint_type
                if constraint_type not in constraint_frequencies:
                    constraint_frequencies[constraint_type] = []
                constraint_frequencies[constraint_type].append(violation.violation_frequency)

        if not constraint_frequencies:
            return "none"

        # Calculate average frequency for each constraint type
        avg_frequencies = {
            constraint_type: np.mean(frequencies)
            for constraint_type, frequencies in constraint_frequencies.items()
        }

        return max(avg_frequencies.keys(), key=lambda x: avg_frequencies[x])

    def create_constraint_analysis_visualization_data(self) -> dict[str, pd.DataFrame]:
        """Create data for constraint analysis visualizations.

        Returns:
            Dictionary containing DataFrames for different visualizations
        """
        if not self.constraint_analysis_results:
            raise ValueError("No constraint analysis results available.")

        visualization_data = {}

        # 1. Violation Frequency Heatmap
        violation_data = []
        for model_type, results in self.constraint_analysis_results.items():
            for result in results:
                for violation in result.constraint_violations:
                    violation_data.append(
                        {
                            "model": model_type,
                            "constraint_type": violation.constraint_type,
                            "violation_frequency": violation.violation_frequency,
                            "violation_magnitude": violation.violation_magnitude,
                            "robustness_score": result.robustness_score,
                        }
                    )

        visualization_data["violations"] = pd.DataFrame(violation_data)

        # 2. Enforcement Impact Analysis
        impact_data = []
        for model_type, results in self.constraint_analysis_results.items():
            for result in results:
                impact_data.append(
                    {
                        "model": model_type,
                        "robustness_score": result.robustness_score,
                        "total_violation_frequency": result.total_violation_frequency,
                        "sharpe_impact": result.enforcement_impact.get(
                            "sharpe_ratio_relative", 0.0
                        ),
                        "is_significant": result.statistical_significance.get(
                            "is_significant", False
                        ),
                        "performance_with_enforcement": result.performance_with_enforcement.get(
                            "sharpe_ratio", 0.0
                        ),
                        "performance_without_enforcement": result.performance_without_enforcement.get(
                            "sharpe_ratio", 0.0
                        ),
                    }
                )

        visualization_data["enforcement_impact"] = pd.DataFrame(impact_data)

        # 3. Robustness Ranking
        robustness_data = []
        for model_type, results in self.constraint_analysis_results.items():
            if results:
                best_robustness = max(r.robustness_score for r in results)
                avg_robustness = np.mean([r.robustness_score for r in results])

                robustness_data.append(
                    {
                        "model": model_type,
                        "best_robustness": best_robustness,
                        "avg_robustness": avg_robustness,
                        "robustness_range": best_robustness
                        - min(r.robustness_score for r in results),
                    }
                )

        visualization_data["robustness"] = pd.DataFrame(robustness_data)

        return visualization_data

    def export_constraint_analysis_results(self, filepath: str) -> None:
        """Export constraint analysis results to file.

        Args:
            filepath: Path to save results
        """
        if not self.constraint_analysis_results:
            raise ValueError("No constraint analysis results available.")

        export_data = []

        for model_type, results in self.constraint_analysis_results.items():
            for result in results:
                base_row = {
                    "model_type": model_type,
                    "robustness_score": result.robustness_score,
                    "total_violation_frequency": result.total_violation_frequency,
                    "avg_violation_magnitude": result.avg_violation_magnitude,
                    "is_significant": result.statistical_significance.get("is_significant", False),
                    "p_value": result.statistical_significance.get("p_value", 1.0),
                }

                # Add parameter combination
                for param, value in result.parameter_combination.items():
                    base_row[f"param_{param}"] = value

                # Add performance metrics
                for metric, value in result.performance_with_enforcement.items():
                    base_row[f"enforced_{metric}"] = value

                for metric, value in result.performance_without_enforcement.items():
                    base_row[f"relaxed_{metric}"] = value

                # Add enforcement impacts
                for metric, value in result.enforcement_impact.items():
                    base_row[f"impact_{metric}"] = value

                # Add violation details
                for i, violation in enumerate(result.constraint_violations):
                    violation_row = base_row.copy()
                    violation_row.update(
                        {
                            f"violation_{i}_type": violation.constraint_type,
                            f"violation_{i}_frequency": violation.violation_frequency,
                            f"violation_{i}_magnitude": violation.violation_magnitude,
                            f"violation_{i}_count": len(violation.violation_dates),
                        }
                    )
                    export_data.append(violation_row)

                # If no violations, still add the base row
                if not result.constraint_violations:
                    export_data.append(base_row)

        df = pd.DataFrame(export_data)

        if filepath.endswith(".csv"):
            df.to_csv(filepath, index=False)
        elif filepath.endswith(".parquet"):
            df.to_parquet(filepath, index=False)
        else:
            raise ValueError("Unsupported file format. Use .csv or .parquet")

        logger.info(f"Constraint analysis results exported to {filepath}")
