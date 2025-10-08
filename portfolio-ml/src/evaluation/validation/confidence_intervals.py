"""Comprehensive confidence intervals framework for portfolio performance metrics.

Implements multiple methods for calculating confidence intervals including bootstrap,
asymptotic, and specialized financial metrics approaches with time-varying analysis
and visualization capabilities.
"""

import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, t

from .bootstrap import BootstrapMethodology


@dataclass
class ConfidenceIntervalResult:
    """Container for confidence interval results."""

    lower_bound: float
    upper_bound: float
    point_estimate: float
    confidence_level: float
    method: str
    additional_info: dict


class ComprehensiveConfidenceIntervals:
    """Framework for comprehensive confidence interval analysis."""

    def __init__(
        self,
        confidence_levels: list[float] = None,
        bootstrap_samples: int = 1000,
        random_state: Optional[int] = None,
    ):
        """Initialize confidence intervals framework.

        Args:
            confidence_levels: List of confidence levels to calculate
            bootstrap_samples: Number of bootstrap samples
            random_state: Random seed for reproducibility
        """
        self.confidence_levels = confidence_levels or [0.90, 0.95, 0.99]
        self.bootstrap = BootstrapMethodology(
            n_bootstrap=bootstrap_samples, random_state=random_state
        )

    def bootstrap_confidence_intervals_multi_level(
        self,
        performance_metric: Callable,
        returns: Union[pd.Series, np.ndarray],
        methods: list[str] = None,
    ) -> dict[str, dict[float, ConfidenceIntervalResult]]:
        """Bootstrap confidence intervals for multiple confidence levels and methods.

        Args:
            performance_metric: Function that calculates performance metric
            returns: Return series
            methods: Bootstrap methods to use

        Returns:
            Dictionary mapping methods to confidence level results
        """
        if methods is None:
            methods = ["percentile", "bias_corrected", "bca"]

        results = {}

        for method in methods:
            method_results = {}

            for confidence_level in self.confidence_levels:
                try:
                    lower, upper, bootstrap_info = self.bootstrap.bootstrap_confidence_intervals(
                        performance_metric, returns, confidence_level, method
                    )

                    method_results[confidence_level] = ConfidenceIntervalResult(
                        lower_bound=lower,
                        upper_bound=upper,
                        point_estimate=bootstrap_info["original_statistic"],
                        confidence_level=confidence_level,
                        method=f"bootstrap_{method}",
                        additional_info=bootstrap_info,
                    )

                except Exception as e:
                    warnings.warn(
                        f"Bootstrap CI calculation failed for {method} at {confidence_level}: {e}",
                        stacklevel=2,
                    )
                    method_results[confidence_level] = ConfidenceIntervalResult(
                        lower_bound=np.nan,
                        upper_bound=np.nan,
                        point_estimate=np.nan,
                        confidence_level=confidence_level,
                        method=f"bootstrap_{method}",
                        additional_info={"error": str(e)},
                    )

            results[method] = method_results

        return results

    def asymptotic_confidence_intervals_delta_method(
        self,
        performance_metric: Callable,
        returns: Union[pd.Series, np.ndarray],
        gradient_func: Optional[Callable] = None,
    ) -> dict[float, ConfidenceIntervalResult]:
        """Asymptotic confidence intervals using delta method for complex metrics.

        Args:
            performance_metric: Performance metric function
            returns: Return series
            gradient_func: Function to compute gradient (if None, uses numerical gradient)

        Returns:
            Dictionary mapping confidence levels to CI results
        """
        returns_array = np.asarray(returns)
        n = len(returns_array)

        # Calculate point estimate
        point_estimate = performance_metric(returns_array)

        # Calculate or estimate gradient
        if gradient_func is not None:
            gradient = gradient_func(returns_array)
        else:
            gradient = self._numerical_gradient(performance_metric, returns_array)

        # Calculate sample moments
        sample_mean = np.mean(returns_array)
        sample_var = np.var(returns_array, ddof=1)
        np.array([sample_mean, sample_var])

        # Asymptotic variance using delta method
        moment_cov_matrix = self._calculate_moment_covariance_matrix(returns_array)
        asymptotic_variance = gradient.T @ moment_cov_matrix @ gradient / n

        if asymptotic_variance < 0:
            warnings.warn("Negative asymptotic variance in delta method", stacklevel=2)
            asymptotic_variance = 0

        asymptotic_se = np.sqrt(asymptotic_variance)

        # Calculate confidence intervals for all levels
        results = {}

        for confidence_level in self.confidence_levels:
            alpha = 1 - confidence_level
            z_critical = norm.ppf(1 - alpha / 2)

            margin_of_error = z_critical * asymptotic_se
            lower_bound = point_estimate - margin_of_error
            upper_bound = point_estimate + margin_of_error

            results[confidence_level] = ConfidenceIntervalResult(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                point_estimate=point_estimate,
                confidence_level=confidence_level,
                method="asymptotic_delta_method",
                additional_info={
                    "asymptotic_variance": asymptotic_variance,
                    "standard_error": asymptotic_se,
                    "gradient": gradient,
                    "sample_size": n,
                },
            )

        return results

    def time_varying_confidence_intervals(
        self,
        returns: Union[pd.Series, np.ndarray],
        performance_metric: Callable,
        window_size: int = 252,
        step_size: int = 63,
        method: str = "bootstrap",
    ) -> pd.DataFrame:
        """Time-varying confidence interval analysis for rolling performance.

        Args:
            returns: Return series
            performance_metric: Performance metric function
            window_size: Rolling window size
            step_size: Step size between windows
            method: CI calculation method

        Returns:
            DataFrame with time-varying confidence intervals
        """
        returns_series = pd.Series(returns) if not isinstance(returns, pd.Series) else returns
        results = []

        for start_idx in range(0, len(returns_series) - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window_returns = returns_series.iloc[start_idx:end_idx]
            window_date = (
                returns_series.index[end_idx - 1]
                if hasattr(returns_series, "index")
                else end_idx - 1
            )

            try:
                if method == "bootstrap":
                    # Use primary confidence level for time-varying analysis
                    primary_level = (
                        self.confidence_levels[1]
                        if len(self.confidence_levels) > 1
                        else self.confidence_levels[0]
                    )

                    lower, upper, bootstrap_info = self.bootstrap.bootstrap_confidence_intervals(
                        performance_metric, window_returns, primary_level
                    )

                    result_row = {
                        "date": window_date,
                        "point_estimate": bootstrap_info["original_statistic"],
                        "lower_bound": lower,
                        "upper_bound": upper,
                        "confidence_level": primary_level,
                        "method": method,
                        "window_size": window_size,
                        "bootstrap_std": bootstrap_info.get("bootstrap_std", np.nan),
                    }

                elif method == "asymptotic":
                    ci_results = self._asymptotic_ci_simple(performance_metric, window_returns)
                    primary_level = (
                        self.confidence_levels[1]
                        if len(self.confidence_levels) > 1
                        else self.confidence_levels[0]
                    )

                    result_row = {
                        "date": window_date,
                        "point_estimate": ci_results[primary_level].point_estimate,
                        "lower_bound": ci_results[primary_level].lower_bound,
                        "upper_bound": ci_results[primary_level].upper_bound,
                        "confidence_level": primary_level,
                        "method": method,
                        "window_size": window_size,
                        "standard_error": ci_results[primary_level].additional_info.get(
                            "standard_error", np.nan
                        ),
                    }

                results.append(result_row)

            except Exception as e:
                warnings.warn(
                    f"Time-varying CI calculation failed for window ending {window_date}: {e}",
                    stacklevel=2,
                )
                # Add NaN row to maintain time series structure
                result_row = {
                    "date": window_date,
                    "point_estimate": np.nan,
                    "lower_bound": np.nan,
                    "upper_bound": np.nan,
                    "confidence_level": self.confidence_levels[0],
                    "method": method,
                    "window_size": window_size,
                }
                results.append(result_row)

        return pd.DataFrame(results)

    def confidence_interval_visualization_framework(
        self, ci_results: Union[dict, pd.DataFrame], performance_name: str = "Performance Metric"
    ) -> dict[str, Union[pd.DataFrame, dict]]:
        """Prepare confidence interval data for visualization and interpretation.

        Args:
            ci_results: CI results from other methods
            performance_name: Name of performance metric for labeling

        Returns:
            Dictionary containing visualization data and interpretation
        """
        visualization_data = {}

        if isinstance(ci_results, pd.DataFrame):
            # Time-varying CI data
            visualization_data["time_series"] = {
                "dates": ci_results["date"].tolist(),
                "point_estimates": ci_results["point_estimate"].tolist(),
                "lower_bounds": ci_results["lower_bound"].tolist(),
                "upper_bounds": ci_results["upper_bound"].tolist(),
                "confidence_level": (
                    ci_results["confidence_level"].iloc[0] if not ci_results.empty else np.nan
                ),
            }

            # Calculate summary statistics
            valid_data = ci_results.dropna()
            if not valid_data.empty:
                visualization_data["summary_stats"] = {
                    "mean_point_estimate": valid_data["point_estimate"].mean(),
                    "mean_ci_width": (valid_data["upper_bound"] - valid_data["lower_bound"]).mean(),
                    "stability_measure": (
                        valid_data["point_estimate"].std()
                        / abs(valid_data["point_estimate"].mean())
                        if valid_data["point_estimate"].mean() != 0
                        else np.inf
                    ),
                    "periods_analyzed": len(valid_data),
                }

        elif isinstance(ci_results, dict):
            # Multi-level or multi-method CI data
            if isinstance(list(ci_results.values())[0], dict):
                # Multi-level results
                visualization_data["multi_level"] = {}

                for method, level_results in ci_results.items():
                    method_data = {
                        "confidence_levels": [],
                        "lower_bounds": [],
                        "upper_bounds": [],
                        "ci_widths": [],
                        "point_estimate": None,
                    }

                    for level, ci_result in level_results.items():
                        if not np.isnan(ci_result.lower_bound):
                            method_data["confidence_levels"].append(level)
                            method_data["lower_bounds"].append(ci_result.lower_bound)
                            method_data["upper_bounds"].append(ci_result.upper_bound)
                            method_data["ci_widths"].append(
                                ci_result.upper_bound - ci_result.lower_bound
                            )

                            if method_data["point_estimate"] is None:
                                method_data["point_estimate"] = ci_result.point_estimate

                    visualization_data["multi_level"][method] = method_data

        # Add interpretation guidelines
        visualization_data["interpretation"] = self._generate_ci_interpretation(
            ci_results, performance_name
        )

        return visualization_data

    def multi_metric_confidence_intervals(
        self,
        returns: Union[pd.Series, np.ndarray],
        metrics: dict[str, Callable],
        method: str = "bootstrap",
    ) -> dict[str, dict[float, ConfidenceIntervalResult]]:
        """Calculate confidence intervals for multiple performance metrics.

        Args:
            returns: Return series
            metrics: Dictionary mapping metric names to metric functions
            method: CI calculation method

        Returns:
            Dictionary mapping metric names to CI results
        """
        results = {}

        for metric_name, metric_func in metrics.items():
            try:
                if method == "bootstrap":
                    metric_results = {}

                    for confidence_level in self.confidence_levels:
                        lower, upper, bootstrap_info = (
                            self.bootstrap.bootstrap_confidence_intervals(
                                metric_func, returns, confidence_level
                            )
                        )

                        metric_results[confidence_level] = ConfidenceIntervalResult(
                            lower_bound=lower,
                            upper_bound=upper,
                            point_estimate=bootstrap_info["original_statistic"],
                            confidence_level=confidence_level,
                            method=f"bootstrap_{method}",
                            additional_info=bootstrap_info,
                        )

                    results[metric_name] = metric_results

                elif method == "asymptotic":
                    results[metric_name] = self._asymptotic_ci_simple(metric_func, returns)

            except Exception as e:
                warnings.warn(f"CI calculation failed for metric {metric_name}: {e}", stacklevel=2)
                results[metric_name] = {
                    level: ConfidenceIntervalResult(
                        lower_bound=np.nan,
                        upper_bound=np.nan,
                        point_estimate=np.nan,
                        confidence_level=level,
                        method=method,
                        additional_info={"error": str(e)},
                    )
                    for level in self.confidence_levels
                }

        return results

    # Helper methods
    def _numerical_gradient(
        self, performance_metric: Callable, returns: np.ndarray, h: float = 1e-6
    ) -> np.ndarray:
        """Calculate numerical gradient for delta method."""
        len(returns)
        gradient = np.zeros(2)  # Gradient w.r.t. mean and variance

        # Original value
        original_value = performance_metric(returns)

        # Gradient w.r.t. mean
        perturbed_returns_mean = returns + h
        perturbed_value_mean = performance_metric(perturbed_returns_mean)
        gradient[0] = (perturbed_value_mean - original_value) / h

        # Gradient w.r.t. variance (approximate using standard deviation perturbation)
        returns_centered = returns - np.mean(returns)
        perturbed_returns_var = returns_centered * (1 + h) + np.mean(returns)
        perturbed_value_var = performance_metric(perturbed_returns_var)
        gradient[1] = (perturbed_value_var - original_value) / h

        return gradient

    def _calculate_moment_covariance_matrix(self, returns: np.ndarray) -> np.ndarray:
        """Calculate covariance matrix of sample moments."""
        n = len(returns)

        # Sample moments
        sample_mean = np.mean(returns)
        sample_var = np.var(returns, ddof=1)

        # Second moments for covariance calculations
        np.mean(returns**2)
        third_moment = np.mean((returns - sample_mean) ** 3)
        fourth_moment = np.mean((returns - sample_mean) ** 4)

        # Covariance matrix elements
        var_mean = sample_var
        var_var = (fourth_moment - sample_var**2) / n
        cov_mean_var = third_moment / n

        return np.array([[var_mean, cov_mean_var], [cov_mean_var, var_var]])

    def _asymptotic_ci_simple(
        self, performance_metric: Callable, returns: Union[pd.Series, np.ndarray]
    ) -> dict[float, ConfidenceIntervalResult]:
        """Simple asymptotic CI calculation for common metrics."""
        returns_array = np.asarray(returns)
        n = len(returns_array)

        point_estimate = performance_metric(returns_array)

        # Simple standard error approximation (works well for Sharpe ratio and similar metrics)
        # Using jackknife estimation
        jackknife_estimates = []

        for i in range(n):
            jackknife_sample = np.concatenate([returns_array[:i], returns_array[i + 1 :]])
            try:
                jackknife_est = performance_metric(jackknife_sample)
                jackknife_estimates.append(jackknife_est)
            except (ValueError, ZeroDivisionError):
                continue

        if len(jackknife_estimates) > 0:
            jackknife_mean = np.mean(jackknife_estimates)
            jackknife_var = (
                np.sum((np.array(jackknife_estimates) - jackknife_mean) ** 2) * (n - 1) / n
            )
            standard_error = np.sqrt(jackknife_var)
        else:
            # Fallback to simple approximation
            standard_error = np.std(returns_array) / np.sqrt(n)

        results = {}

        for confidence_level in self.confidence_levels:
            alpha = 1 - confidence_level
            t_critical = t.ppf(1 - alpha / 2, n - 1)

            margin_of_error = t_critical * standard_error
            lower_bound = point_estimate - margin_of_error
            upper_bound = point_estimate + margin_of_error

            results[confidence_level] = ConfidenceIntervalResult(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                point_estimate=point_estimate,
                confidence_level=confidence_level,
                method="asymptotic_jackknife",
                additional_info={
                    "standard_error": standard_error,
                    "degrees_of_freedom": n - 1,
                    "jackknife_estimates": len(jackknife_estimates),
                },
            )

        return results

    def _generate_ci_interpretation(
        self, ci_results: Union[dict, pd.DataFrame], performance_name: str
    ) -> dict[str, str]:
        """Generate interpretation guidelines for confidence intervals."""
        interpretation = {
            "metric_name": performance_name,
            "general_guidance": f"Confidence intervals provide a range of plausible values for the true {performance_name}. "
            "Wider intervals indicate greater uncertainty in the estimate.",
        }

        if isinstance(ci_results, pd.DataFrame) and not ci_results.empty:
            # Time-varying interpretation
            valid_data = ci_results.dropna()
            if not valid_data.empty:
                avg_width = (valid_data["upper_bound"] - valid_data["lower_bound"]).mean()
                stability = valid_data["point_estimate"].std()

                interpretation["time_varying"] = (
                    f"Average confidence interval width: {avg_width:.4f}. "
                    f"Performance stability (standard deviation): {stability:.4f}. "
                    "Narrower intervals and lower stability indicate more consistent performance."
                )

        elif isinstance(ci_results, dict):
            # Multi-level or multi-method interpretation
            interpretation["multi_level"] = (
                "Higher confidence levels produce wider intervals. "
                "Compare different methods (percentile, bias-corrected, BCa) to assess robustness. "
                "BCa intervals are generally preferred when available."
            )

        return interpretation


# Specialized confidence interval functions for common financial metrics
def sharpe_ratio_ci_analytical(
    returns: np.ndarray, confidence_level: float = 0.95
) -> tuple[float, float]:
    """Analytical confidence interval for Sharpe ratio using Miller-Gehr method."""
    n = len(returns)
    if n < 2:
        return np.nan, np.nan

    sharpe = np.mean(returns) / np.std(returns, ddof=1) if np.std(returns, ddof=1) > 0 else 0

    # Analytical variance formula for Sharpe ratio
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns, excess=True)

    sharpe_variance = (1 + 0.5 * sharpe**2 - skew * sharpe + (kurt - 3) * sharpe**2 / 4) / n
    sharpe_se = np.sqrt(sharpe_variance)

    alpha = 1 - confidence_level
    z_critical = norm.ppf(1 - alpha / 2)

    lower = sharpe - z_critical * sharpe_se
    upper = sharpe + z_critical * sharpe_se

    return lower, upper


def information_ratio_ci(
    active_returns: np.ndarray, confidence_level: float = 0.95
) -> tuple[float, float]:
    """Confidence interval for Information Ratio (IR)."""
    if len(active_returns) < 2:
        return np.nan, np.nan

    ir = (
        np.mean(active_returns) / np.std(active_returns, ddof=1)
        if np.std(active_returns, ddof=1) > 0
        else 0
    )
    n = len(active_returns)

    # Using similar variance formula as Sharpe ratio
    skew = stats.skew(active_returns)
    kurt = stats.kurtosis(active_returns, excess=True)

    ir_variance = (1 + 0.5 * ir**2 - skew * ir + (kurt - 3) * ir**2 / 4) / n
    ir_se = np.sqrt(ir_variance)

    alpha = 1 - confidence_level
    z_critical = norm.ppf(1 - alpha / 2)

    lower = ir - z_critical * ir_se
    upper = ir + z_critical * ir_se

    return lower, upper
