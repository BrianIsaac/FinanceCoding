"""
Academic performance metrics with uncertainty quantification.

This module implements performance metrics with confidence intervals,
statistical significance testing, and uncertainty bounds for academic research.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import bootstrap

logger = logging.getLogger(__name__)


@dataclass
class AcademicPerformanceReport:
    """Performance report with academic uncertainty quantification."""

    metrics: dict[str, float]
    confidence_intervals: dict[str, Tuple[float, float]]
    significance_tests: dict[str, dict[str, Any]]
    uncertainty_bounds: dict[str, Tuple[float, float]]
    academic_caveats: list[str]
    confidence_level: float
    methodology_used: str


class AcademicPerformanceMetrics:
    """
    Calculate performance metrics with uncertainty quantification.

    This class provides academically rigorous performance metrics with
    confidence intervals, significance testing, and uncertainty bounds.
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        bootstrap_iterations: int = 10000,
        multiple_testing_correction: str = "bonferroni",
    ):
        """
        Initialise academic performance metrics calculator.

        Args:
            confidence_level: Confidence level for intervals (e.g., 0.95 for 95%)
            bootstrap_iterations: Number of bootstrap iterations
            multiple_testing_correction: Method for multiple testing correction
        """
        self.confidence_level = confidence_level
        self.bootstrap_iterations = bootstrap_iterations
        self.multiple_testing_correction = multiple_testing_correction

    def calculate_with_uncertainty(
        self,
        returns: pd.Series | np.ndarray,
        confidence_scores: Optional[np.ndarray] = None,
        benchmark_returns: Optional[pd.Series | np.ndarray] = None,
    ) -> AcademicPerformanceReport:
        """
        Calculate performance metrics with uncertainty quantification.

        Args:
            returns: Portfolio returns
            confidence_scores: Optional confidence scores for each period
            benchmark_returns: Optional benchmark returns for relative metrics

        Returns:
            AcademicPerformanceReport with metrics and uncertainty
        """
        returns = np.asarray(returns)
        confidence_scores = confidence_scores if confidence_scores is not None else np.ones(len(returns))

        # Calculate base metrics
        metrics = self._calculate_base_metrics(returns, benchmark_returns)

        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(returns, metrics)

        # Perform significance tests
        significance_tests = self._perform_significance_tests(returns, benchmark_returns)

        # Calculate uncertainty bounds
        uncertainty_bounds = self._calculate_uncertainty_bounds(returns, confidence_scores)

        # Generate academic caveats
        caveats = self._generate_performance_caveats(
            returns, confidence_scores, metrics, significance_tests
        )

        # Determine methodology based on average confidence
        avg_confidence = np.mean(confidence_scores)
        methodology = self._determine_methodology(avg_confidence)

        return AcademicPerformanceReport(
            metrics=metrics,
            confidence_intervals=confidence_intervals,
            significance_tests=significance_tests,
            uncertainty_bounds=uncertainty_bounds,
            academic_caveats=caveats,
            confidence_level=self.confidence_level,
            methodology_used=methodology,
        )

    def _calculate_base_metrics(
        self,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """Calculate base performance metrics."""
        metrics = {}

        # Basic statistics
        metrics["mean_return"] = np.mean(returns)
        metrics["volatility"] = np.std(returns)
        metrics["skewness"] = stats.skew(returns)
        metrics["kurtosis"] = stats.kurtosis(returns)

        # Risk-adjusted metrics
        metrics["sharpe_ratio"] = self._calculate_sharpe_ratio(returns)
        metrics["sortino_ratio"] = self._calculate_sortino_ratio(returns)
        metrics["calmar_ratio"] = self._calculate_calmar_ratio(returns)

        # Downside risk metrics
        metrics["max_drawdown"] = self._calculate_max_drawdown(returns)
        metrics["var_95"] = np.percentile(returns, 5)
        metrics["cvar_95"] = np.mean(returns[returns <= metrics["var_95"]])

        # Relative metrics if benchmark provided
        if benchmark_returns is not None:
            benchmark_returns = np.asarray(benchmark_returns)
            excess_returns = returns - benchmark_returns[:len(returns)]
            metrics["tracking_error"] = np.std(excess_returns)
            metrics["information_ratio"] = (
                np.mean(excess_returns) / metrics["tracking_error"]
                if metrics["tracking_error"] > 0
                else 0
            )

        return metrics

    def _calculate_confidence_intervals(
        self,
        returns: np.ndarray,
        metrics: dict[str, float],
    ) -> dict[str, Tuple[float, float]]:
        """Calculate bootstrap confidence intervals for metrics."""
        confidence_intervals = {}

        # Define metric calculation functions for bootstrap
        def sharpe_ratio(x):
            return np.mean(x) / np.std(x) if np.std(x) > 0 else 0

        def sortino_ratio(x):
            downside_returns = x[x < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
            return np.mean(x) / downside_std if downside_std > 0 else 0

        # Bootstrap confidence intervals for key metrics
        for metric_name, metric_func in [
            ("sharpe_ratio", sharpe_ratio),
            ("mean_return", np.mean),
            ("volatility", np.std),
            ("sortino_ratio", sortino_ratio),
        ]:
            try:
                # Use scipy.stats.bootstrap for confidence intervals
                result = bootstrap(
                    (returns,),
                    metric_func,
                    n_resamples=self.bootstrap_iterations,
                    confidence_level=self.confidence_level,
                    method="percentile",
                )
                confidence_intervals[metric_name] = (
                    result.confidence_interval.low,
                    result.confidence_interval.high,
                )
            except Exception as e:
                logger.warning(f"Failed to calculate CI for {metric_name}: {e}")
                # Fallback to normal approximation
                se = self._calculate_standard_error(returns, metric_name)
                z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
                confidence_intervals[metric_name] = (
                    metrics[metric_name] - z_score * se,
                    metrics[metric_name] + z_score * se,
                )

        return confidence_intervals

    def _perform_significance_tests(
        self,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
    ) -> dict[str, dict[str, Any]]:
        """Perform statistical significance tests."""
        tests = {}

        # Test if mean return is significantly different from zero
        t_stat, p_value = stats.ttest_1samp(returns, 0)
        tests["mean_return_significance"] = {
            "test": "one-sample t-test",
            "null_hypothesis": "mean return = 0",
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < (1 - self.confidence_level),
        }

        # Test if Sharpe ratio is significantly different from zero
        sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        # Asymptotic test for Sharpe ratio
        n = len(returns)
        se_sharpe = np.sqrt((1 + 0.5 * sharpe**2) / n)
        z_stat = sharpe / se_sharpe
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
        tests["sharpe_ratio_significance"] = {
            "test": "asymptotic z-test",
            "null_hypothesis": "Sharpe ratio = 0",
            "z_statistic": z_stat,
            "p_value": p_value,
            "significant": p_value < (1 - self.confidence_level),
        }

        # Test against benchmark if provided
        if benchmark_returns is not None:
            benchmark_returns = np.asarray(benchmark_returns)[:len(returns)]
            t_stat, p_value = stats.ttest_rel(returns, benchmark_returns)
            tests["vs_benchmark_significance"] = {
                "test": "paired t-test",
                "null_hypothesis": "returns = benchmark returns",
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < (1 - self.confidence_level),
            }

        # Apply multiple testing correction
        if self.multiple_testing_correction and len(tests) > 1:
            self._apply_multiple_testing_correction(tests)

        return tests

    def _calculate_uncertainty_bounds(
        self,
        returns: np.ndarray,
        confidence_scores: np.ndarray,
    ) -> dict[str, Tuple[float, float]]:
        """Calculate uncertainty bounds based on confidence scores."""
        bounds = {}

        # Weight returns by confidence
        weights = confidence_scores / np.sum(confidence_scores)
        weighted_mean = np.sum(returns * weights)
        weighted_std = np.sqrt(np.sum(weights * (returns - weighted_mean) ** 2))

        # Calculate bounds for different metrics
        z_score = stats.norm.ppf((1 + self.confidence_level) / 2)

        # Weighted confidence bounds
        bounds["weighted_mean_return"] = (
            weighted_mean - z_score * weighted_std / np.sqrt(len(returns)),
            weighted_mean + z_score * weighted_std / np.sqrt(len(returns)),
        )

        # Uncertainty due to low confidence periods
        low_confidence_mask = confidence_scores < 0.7
        if np.any(low_confidence_mask):
            # Calculate bounds excluding low confidence periods
            high_conf_returns = returns[~low_confidence_mask]
            if len(high_conf_returns) > 0:
                bounds["high_confidence_mean"] = (
                    np.mean(high_conf_returns) - z_score * np.std(high_conf_returns) / np.sqrt(len(high_conf_returns)),
                    np.mean(high_conf_returns) + z_score * np.std(high_conf_returns) / np.sqrt(len(high_conf_returns)),
                )

        return bounds

    def _generate_performance_caveats(
        self,
        returns: np.ndarray,
        confidence_scores: np.ndarray,
        metrics: dict[str, float],
        significance_tests: dict[str, dict[str, Any]],
    ) -> list[str]:
        """Generate academic caveats based on analysis."""
        caveats = []

        # Check sample size
        n = len(returns)
        if n < 30:
            caveats.append(f"Small sample size (n={n}) limits statistical power")
        elif n < 100:
            caveats.append(f"Moderate sample size (n={n}) may affect precision")

        # Check confidence scores
        avg_confidence = np.mean(confidence_scores)
        if avg_confidence < 0.7:
            caveats.append(f"Low average confidence ({avg_confidence:.2f}) increases uncertainty")

        low_conf_pct = np.mean(confidence_scores < 0.6) * 100
        if low_conf_pct > 20:
            caveats.append(f"{low_conf_pct:.1f}% of periods have low confidence")

        # Check statistical significance
        non_significant = [
            test_name
            for test_name, test_result in significance_tests.items()
            if not test_result.get("significant", False)
        ]
        if non_significant:
            caveats.append(f"Non-significant results for: {', '.join(non_significant)}")

        # Check distribution properties
        if abs(metrics.get("skewness", 0)) > 1:
            caveats.append(f"Returns show significant skewness ({metrics['skewness']:.2f})")

        if metrics.get("kurtosis", 0) > 3:
            caveats.append(f"Returns show excess kurtosis ({metrics['kurtosis']:.2f})")

        # Check risk metrics
        if metrics.get("max_drawdown", 0) < -0.2:
            caveats.append(f"Large drawdown observed ({metrics['max_drawdown']:.1%})")

        return caveats

    def _determine_methodology(self, avg_confidence: float) -> str:
        """Determine methodology description based on confidence."""
        if avg_confidence >= 0.9:
            return "Standard academic methods with parametric tests"
        elif avg_confidence >= 0.7:
            return "Robust methods with bootstrap confidence intervals"
        elif avg_confidence >= 0.5:
            return "Conservative estimates with non-parametric tests"
        else:
            return "Exploratory analysis with high uncertainty"

    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0

    def _calculate_sortino_ratio(self, returns: np.ndarray, target_return: float = 0) -> float:
        """Calculate Sortino ratio."""
        excess_returns = returns - target_return
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        return np.mean(excess_returns) / downside_std if downside_std > 0 else 0

    def _calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """Calculate Calmar ratio."""
        max_dd = self._calculate_max_drawdown(returns)
        return np.mean(returns) / abs(max_dd) if max_dd != 0 else 0

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        return np.min(drawdowns) if len(drawdowns) > 0 else 0

    def _calculate_standard_error(self, returns: np.ndarray, metric_name: str) -> float:
        """Calculate standard error for a metric."""
        n = len(returns)
        if metric_name == "mean_return":
            return np.std(returns) / np.sqrt(n)
        elif metric_name == "sharpe_ratio":
            sharpe = self._calculate_sharpe_ratio(returns)
            # Asymptotic standard error for Sharpe ratio
            return np.sqrt((1 + 0.5 * sharpe**2) / n)
        else:
            # Default to standard error of mean
            return np.std(returns) / np.sqrt(n)

    def _apply_multiple_testing_correction(
        self,
        tests: dict[str, dict[str, Any]],
    ) -> None:
        """Apply multiple testing correction to p-values."""
        if self.multiple_testing_correction == "bonferroni":
            n_tests = len(tests)
            for test_result in tests.values():
                test_result["p_value_corrected"] = min(test_result["p_value"] * n_tests, 1.0)
                test_result["significant_corrected"] = test_result["p_value_corrected"] < (1 - self.confidence_level)


def create_academic_performance_calculator() -> AcademicPerformanceMetrics:
    """Factory function to create default academic performance calculator."""
    return AcademicPerformanceMetrics()