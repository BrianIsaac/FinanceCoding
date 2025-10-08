"""Portfolio size sensitivity analysis framework."""

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
class PortfolioSizeConfig:
    """Portfolio size configuration."""

    top_k: int
    description: str
    expected_diversification: str


@dataclass
class PortfolioSizeResult:
    """Results from portfolio size sensitivity analysis."""

    model_type: str
    size_config: PortfolioSizeConfig
    performance_metrics: dict[str, float]
    diversification_metrics: dict[str, float]
    concentration_metrics: dict[str, float]
    optimal_configuration: bool
    statistical_significance: dict[str, float]


class PortfolioSizeAnalyzer:
    """Top-k portfolio size sensitivity testing framework."""

    def __init__(
        self,
        sensitivity_engine: SensitivityAnalysisEngine,
        performance_analytics: PerformanceAnalytics,
        statistical_validator: StatisticalValidation,
    ):
        """Initialize portfolio size analyzer.

        Args:
            sensitivity_engine: Core sensitivity analysis engine
            performance_analytics: Performance metrics calculator
            statistical_validator: Statistical significance testing
        """
        self.sensitivity_engine = sensitivity_engine
        self.performance_analytics = performance_analytics
        self.statistical_validator = statistical_validator

        # Define portfolio size configurations as per story requirements
        self.size_configs = [
            PortfolioSizeConfig(
                top_k=20,
                description="Concentrated portfolio - high alpha potential",
                expected_diversification="Low",
            ),
            PortfolioSizeConfig(
                top_k=30,
                description="Moderately concentrated - balanced approach",
                expected_diversification="Medium-Low",
            ),
            PortfolioSizeConfig(
                top_k=50,
                description="Balanced portfolio - diversification vs concentration",
                expected_diversification="Medium",
            ),
            PortfolioSizeConfig(
                top_k=75,
                description="Well-diversified - reduced idiosyncratic risk",
                expected_diversification="Medium-High",
            ),
            PortfolioSizeConfig(
                top_k=100,
                description="Highly diversified - maximum risk reduction",
                expected_diversification="High",
            ),
        ]

        self.size_analysis_results: dict[str, list[PortfolioSizeResult]] = {}

    def analyze_portfolio_size_sensitivity(
        self,
        model_types: list[str],
        base_results: dict[str, list[SensitivityResult]],
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
        target_metric: str = "sharpe_ratio",
    ) -> dict[str, list[PortfolioSizeResult]]:
        """Analyze portfolio size sensitivity across all models.

        Args:
            model_types: List of model types to analyze
            base_results: Base sensitivity results for parameter selection
            data: Market data for backtesting
            start_date: Analysis start date
            end_date: Analysis end date
            target_metric: Performance metric for analysis

        Returns:
            Dictionary mapping model types to portfolio size results
        """
        logger.info(f"Starting portfolio size sensitivity analysis for models: {model_types}")

        all_size_results = {}

        for model_type in model_types:
            logger.info(f"Analyzing portfolio size sensitivity for {model_type}")

            if model_type not in base_results:
                logger.warning(f"No base results available for {model_type}")
                continue

            # Get optimal hyperparameter configuration
            optimal_config = self._get_optimal_configuration(
                base_results[model_type], target_metric
            )

            if not optimal_config:
                logger.warning(f"No valid configuration found for {model_type}")
                continue

            model_size_results = []

            # Test each portfolio size configuration
            for size_config in self.size_configs:
                size_result = self._analyze_portfolio_size(
                    model_type=model_type,
                    base_config=optimal_config,
                    size_config=size_config,
                    data=data,
                    start_date=start_date,
                    end_date=end_date,
                    target_metric=target_metric,
                )

                if size_result:
                    model_size_results.append(size_result)

            # Identify optimal portfolio size for this model
            self._identify_optimal_size(model_size_results, target_metric)

            all_size_results[model_type] = model_size_results

        self.size_analysis_results = all_size_results

        logger.info(
            f"Portfolio size sensitivity analysis completed for {len(all_size_results)} models"
        )

        return all_size_results

    def _get_optimal_configuration(
        self, model_results: list[SensitivityResult], target_metric: str
    ) -> Optional[SensitivityResult]:
        """Get optimal hyperparameter configuration for a model.

        Args:
            model_results: List of sensitivity results for the model
            target_metric: Performance metric for ranking

        Returns:
            Optimal configuration or None
        """
        valid_results = [
            r for r in model_results if r.error is None and target_metric in r.performance_metrics
        ]

        if not valid_results:
            return None

        # Sort by target metric (descending)
        valid_results.sort(key=lambda x: x.performance_metrics[target_metric], reverse=True)

        return valid_results[0]

    def _analyze_portfolio_size(
        self,
        model_type: str,
        base_config: SensitivityResult,
        size_config: PortfolioSizeConfig,
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
        target_metric: str,
    ) -> Optional[PortfolioSizeResult]:
        """Analyze impact of specific portfolio size configuration.

        Args:
            model_type: Model identifier
            base_config: Base parameter configuration
            size_config: Portfolio size configuration
            data: Market data
            start_date: Analysis start date
            end_date: Analysis end date
            target_metric: Performance metric for analysis

        Returns:
            Portfolio size result or None if analysis fails
        """
        try:
            # Create modified configuration with new portfolio size
            modified_params = base_config.parameter_combination.copy()
            modified_params["top_k_positions"] = size_config.top_k

            # Execute backtest with modified portfolio size
            size_result = self.sensitivity_engine._execute_parameter_combination(
                model_type=model_type,
                parameters=modified_params,
                data=data,
                start_date=start_date,
                end_date=end_date,
            )

            if size_result.error:
                logger.error(
                    f"Portfolio size {size_config.top_k} failed for {model_type}: {size_result.error}"
                )
                return None

            # Calculate diversification and concentration metrics
            diversification_metrics = self._calculate_diversification_metrics(
                size_result.backtest_results, size_config.top_k
            )

            concentration_metrics = self._calculate_concentration_metrics(
                size_result.backtest_results, size_config.top_k
            )

            # Statistical significance test vs base configuration
            stat_result = self._test_size_significance(
                base_config.backtest_results, size_result.backtest_results
            )

            return PortfolioSizeResult(
                model_type=model_type,
                size_config=size_config,
                performance_metrics=size_result.performance_metrics,
                diversification_metrics=diversification_metrics,
                concentration_metrics=concentration_metrics,
                optimal_configuration=False,  # Will be determined later
                statistical_significance=stat_result,
            )

        except Exception as e:
            logger.error(
                f"Error analyzing portfolio size {size_config.top_k} for {model_type}: {e}"
            )
            return None

    def _calculate_diversification_metrics(
        self, backtest_results: pd.DataFrame, portfolio_size: int
    ) -> dict[str, float]:
        """Calculate diversification metrics for portfolio.

        Args:
            backtest_results: Backtest results with position weights
            portfolio_size: Number of positions in portfolio

        Returns:
            Dictionary of diversification metrics
        """
        diversification_metrics = {
            "effective_n_stocks": portfolio_size,
            "diversification_ratio": 0.0,
            "concentration_hhi": 0.0,
            "weight_entropy": 0.0,
            "max_weight": 0.0,
            "weight_std": 0.0,
        }

        try:
            # Extract position weights if available
            if "weights" in backtest_results.columns:
                backtest_results["weights"]

                # Calculate average diversification metrics across time
                hhi_values = []
                entropy_values = []
                max_weights = []
                weight_stds = []

                for _, row in backtest_results.iterrows():
                    if pd.notna(row["weights"]) and hasattr(row["weights"], "__len__"):
                        weights = (
                            np.array(row["weights"])
                            if isinstance(row["weights"], list)
                            else row["weights"]
                        )
                        weights = weights[weights > 0]  # Only positive weights

                        if len(weights) > 0:
                            # Herfindahl-Hirschman Index (lower = more diversified)
                            hhi = np.sum(weights**2)
                            hhi_values.append(hhi)

                            # Weight entropy (higher = more diversified)
                            entropy = -np.sum(weights * np.log(weights + 1e-10))
                            entropy_values.append(entropy)

                            # Maximum weight
                            max_weights.append(np.max(weights))

                            # Weight standard deviation
                            weight_stds.append(np.std(weights))

                if hhi_values:
                    diversification_metrics["concentration_hhi"] = np.mean(hhi_values)
                    diversification_metrics["weight_entropy"] = np.mean(entropy_values)
                    diversification_metrics["max_weight"] = np.mean(max_weights)
                    diversification_metrics["weight_std"] = np.mean(weight_stds)

                    # Effective number of stocks (1/HHI)
                    diversification_metrics["effective_n_stocks"] = (
                        1.0 / np.mean(hhi_values) if np.mean(hhi_values) > 0 else portfolio_size
                    )

                    # Diversification ratio (normalized entropy)
                    max_entropy = np.log(portfolio_size)
                    diversification_metrics["diversification_ratio"] = (
                        np.mean(entropy_values) / max_entropy if max_entropy > 0 else 0.0
                    )

        except Exception as e:
            logger.warning(f"Error calculating diversification metrics: {e}")

        return diversification_metrics

    def _calculate_concentration_metrics(
        self, backtest_results: pd.DataFrame, portfolio_size: int
    ) -> dict[str, float]:
        """Calculate concentration risk metrics for portfolio.

        Args:
            backtest_results: Backtest results
            portfolio_size: Number of positions in portfolio

        Returns:
            Dictionary of concentration metrics
        """
        concentration_metrics = {
            "active_share": 0.0,
            "tracking_error": 0.0,
            "idiosyncratic_risk": 0.0,
            "systematic_risk": 0.0,
            "concentration_score": 0.0,
        }

        try:
            # Calculate concentration score based on portfolio size
            # Smaller portfolios have higher concentration
            max_size = 100  # Maximum size from our configurations
            concentration_metrics["concentration_score"] = 1.0 - (portfolio_size / max_size)

            # Estimate idiosyncratic vs systematic risk based on portfolio size
            # More positions = less idiosyncratic risk
            diversification_benefit = min(portfolio_size / 30.0, 1.0)  # Asymptotic benefit
            concentration_metrics["idiosyncratic_risk"] = 1.0 - diversification_benefit
            concentration_metrics["systematic_risk"] = diversification_benefit

            # Calculate tracking error if benchmark returns are available
            if (
                "benchmark_returns" in backtest_results.columns
                and "returns" in backtest_results.columns
            ):
                portfolio_returns = backtest_results["returns"].dropna()
                benchmark_returns = backtest_results["benchmark_returns"].dropna()

                if len(portfolio_returns) > 0 and len(benchmark_returns) > 0:
                    # Align series
                    aligned_data = pd.DataFrame(
                        {"portfolio": portfolio_returns, "benchmark": benchmark_returns}
                    ).dropna()

                    if len(aligned_data) > 1:
                        excess_returns = aligned_data["portfolio"] - aligned_data["benchmark"]
                        concentration_metrics["tracking_error"] = excess_returns.std() * np.sqrt(
                            252
                        )  # Annualized
                        concentration_metrics["active_share"] = abs(excess_returns).mean()

        except Exception as e:
            logger.warning(f"Error calculating concentration metrics: {e}")

        return concentration_metrics

    def _test_size_significance(
        self, base_results: pd.DataFrame, size_results: pd.DataFrame
    ) -> dict[str, float]:
        """Test statistical significance of portfolio size impact.

        Args:
            base_results: Base configuration backtest results
            size_results: Size-modified backtest results

        Returns:
            Dictionary with statistical test results
        """
        try:
            base_returns = (
                base_results.get("returns", pd.Series())
                if isinstance(base_results, pd.DataFrame)
                else pd.Series()
            )
            size_returns = (
                size_results.get("returns", pd.Series())
                if isinstance(size_results, pd.DataFrame)
                else pd.Series()
            )

            if len(base_returns) > 0 and len(size_returns) > 0:
                return self.statistical_validator.sharpe_ratio_test(base_returns, size_returns)
            else:
                return {"p_value": 1.0, "is_significant": False, "test_statistic": 0.0}

        except Exception as e:
            logger.warning(f"Error in size significance test: {e}")
            return {"p_value": 1.0, "is_significant": False, "test_statistic": 0.0}

    def _identify_optimal_size(
        self, size_results: list[PortfolioSizeResult], target_metric: str
    ) -> None:
        """Identify optimal portfolio size for the model.

        Args:
            size_results: List of portfolio size results
            target_metric: Performance metric for optimization
        """
        if not size_results:
            return

        # Sort by target metric (descending)
        size_results.sort(
            key=lambda x: x.performance_metrics.get(target_metric, -float("inf")), reverse=True
        )

        # Mark the best performing size as optimal
        size_results[0].optimal_configuration = True

    def analyze_diversification_tradeoffs(
        self, target_metric: str = "sharpe_ratio"
    ) -> pd.DataFrame:
        """Analyze diversification vs concentration trade-offs.

        Args:
            target_metric: Performance metric for analysis

        Returns:
            DataFrame with trade-off analysis
        """
        if not self.size_analysis_results:
            raise ValueError(
                "No portfolio size results available. Run analyze_portfolio_size_sensitivity first."
            )

        tradeoff_data = []

        for model_type, results in self.size_analysis_results.items():
            for result in results:
                tradeoff_data.append(
                    {
                        "model_type": model_type,
                        "portfolio_size": result.size_config.top_k,
                        "description": result.size_config.description,
                        "expected_diversification": result.size_config.expected_diversification,
                        "performance": result.performance_metrics.get(target_metric, 0.0),
                        "effective_n_stocks": result.diversification_metrics.get(
                            "effective_n_stocks", 0
                        ),
                        "diversification_ratio": result.diversification_metrics.get(
                            "diversification_ratio", 0.0
                        ),
                        "concentration_hhi": result.diversification_metrics.get(
                            "concentration_hhi", 0.0
                        ),
                        "idiosyncratic_risk": result.concentration_metrics.get(
                            "idiosyncratic_risk", 0.0
                        ),
                        "systematic_risk": result.concentration_metrics.get("systematic_risk", 0.0),
                        "tracking_error": result.concentration_metrics.get("tracking_error", 0.0),
                        "is_optimal": result.optimal_configuration,
                        "is_significant": result.statistical_significance.get(
                            "is_significant", False
                        ),
                        "p_value": result.statistical_significance.get("p_value", 1.0),
                    }
                )

        return pd.DataFrame(tradeoff_data)

    def get_optimal_portfolio_size_recommendations(self) -> dict[str, dict[str, Any]]:
        """Get optimal portfolio size recommendations with statistical validation.

        Returns:
            Dictionary mapping model types to optimal size recommendations
        """
        if not self.size_analysis_results:
            raise ValueError("No portfolio size results available.")

        recommendations = {}

        for model_type, results in self.size_analysis_results.items():
            optimal_result = next((r for r in results if r.optimal_configuration), None)

            if optimal_result:
                # Calculate efficiency metrics
                all_performances = [r.performance_metrics.get("sharpe_ratio", 0.0) for r in results]
                performance_range = max(all_performances) - min(all_performances)

                recommendations[model_type] = {
                    "optimal_size": optimal_result.size_config.top_k,
                    "description": optimal_result.size_config.description,
                    "performance": optimal_result.performance_metrics.get("sharpe_ratio", 0.0),
                    "diversification_ratio": optimal_result.diversification_metrics.get(
                        "diversification_ratio", 0.0
                    ),
                    "concentration_hhi": optimal_result.diversification_metrics.get(
                        "concentration_hhi", 0.0
                    ),
                    "effective_stocks": optimal_result.diversification_metrics.get(
                        "effective_n_stocks", 0
                    ),
                    "idiosyncratic_risk": optimal_result.concentration_metrics.get(
                        "idiosyncratic_risk", 0.0
                    ),
                    "is_statistically_significant": optimal_result.statistical_significance.get(
                        "is_significant", False
                    ),
                    "confidence_level": 1
                    - optimal_result.statistical_significance.get("p_value", 1.0),
                    "performance_sensitivity": performance_range,
                    "recommendation_strength": (
                        "Strong"
                        if optimal_result.statistical_significance.get("is_significant", False)
                        else "Moderate"
                    ),
                }

        return recommendations

    def create_portfolio_size_visualization_data(
        self, target_metric: str = "sharpe_ratio"
    ) -> dict[str, pd.DataFrame]:
        """Create data for portfolio size visualizations.

        Args:
            target_metric: Performance metric for visualization

        Returns:
            Dictionary containing DataFrames for different visualizations
        """
        if not self.size_analysis_results:
            raise ValueError("No portfolio size results available.")

        visualization_data = {}

        # 1. Performance vs Size Curve Data
        performance_data = []
        for model_type, results in self.size_analysis_results.items():
            for result in results:
                performance_data.append(
                    {
                        "model": model_type,
                        "portfolio_size": result.size_config.top_k,
                        "performance": result.performance_metrics.get(target_metric, 0.0),
                        "diversification_ratio": result.diversification_metrics.get(
                            "diversification_ratio", 0.0
                        ),
                        "concentration_hhi": result.diversification_metrics.get(
                            "concentration_hhi", 0.0
                        ),
                        "is_optimal": result.optimal_configuration,
                        "is_significant": result.statistical_significance.get(
                            "is_significant", False
                        ),
                    }
                )

        visualization_data["performance_curve"] = pd.DataFrame(performance_data)

        # 2. Diversification Efficiency Data
        efficiency_data = []
        for model_type, results in self.size_analysis_results.items():
            for result in results:
                # Calculate efficiency as performance per unit of diversification
                diversification = result.diversification_metrics.get("diversification_ratio", 0.01)
                performance = result.performance_metrics.get(target_metric, 0.0)
                efficiency = performance / diversification if diversification > 0 else 0.0

                efficiency_data.append(
                    {
                        "model": model_type,
                        "portfolio_size": result.size_config.top_k,
                        "efficiency": efficiency,
                        "performance": performance,
                        "diversification": diversification,
                        "effective_stocks": result.diversification_metrics.get(
                            "effective_n_stocks", 0
                        ),
                        "idiosyncratic_risk": result.concentration_metrics.get(
                            "idiosyncratic_risk", 0.0
                        ),
                    }
                )

        visualization_data["efficiency"] = pd.DataFrame(efficiency_data)

        # 3. Risk Decomposition Data
        risk_data = []
        for model_type, results in self.size_analysis_results.items():
            for result in results:
                risk_data.append(
                    {
                        "model": model_type,
                        "portfolio_size": result.size_config.top_k,
                        "idiosyncratic_risk": result.concentration_metrics.get(
                            "idiosyncratic_risk", 0.0
                        ),
                        "systematic_risk": result.concentration_metrics.get("systematic_risk", 0.0),
                        "tracking_error": result.concentration_metrics.get("tracking_error", 0.0),
                        "total_risk": result.concentration_metrics.get("idiosyncratic_risk", 0.0)
                        + result.concentration_metrics.get("systematic_risk", 0.0),
                    }
                )

        visualization_data["risk_decomposition"] = pd.DataFrame(risk_data)

        return visualization_data

    def export_portfolio_size_results(self, filepath: str) -> None:
        """Export portfolio size analysis results to file.

        Args:
            filepath: Path to save results
        """
        if not self.size_analysis_results:
            raise ValueError("No portfolio size results available.")

        export_data = []

        for model_type, results in self.size_analysis_results.items():
            for result in results:
                row = {
                    "model_type": model_type,
                    "portfolio_size": result.size_config.top_k,
                    "description": result.size_config.description,
                    "expected_diversification": result.size_config.expected_diversification,
                    "is_optimal": result.optimal_configuration,
                    "is_significant": result.statistical_significance.get("is_significant", False),
                    "p_value": result.statistical_significance.get("p_value", 1.0),
                }

                # Add performance metrics
                for metric, value in result.performance_metrics.items():
                    row[f"performance_{metric}"] = value

                # Add diversification metrics
                for metric, value in result.diversification_metrics.items():
                    row[f"diversification_{metric}"] = value

                # Add concentration metrics
                for metric, value in result.concentration_metrics.items():
                    row[f"concentration_{metric}"] = value

                export_data.append(row)

        df = pd.DataFrame(export_data)

        if filepath.endswith(".csv"):
            df.to_csv(filepath, index=False)
        elif filepath.endswith(".parquet"):
            df.to_parquet(filepath, index=False)
        else:
            raise ValueError("Unsupported file format. Use .csv or .parquet")

        logger.info(f"Portfolio size analysis results exported to {filepath}")
