"""Bootstrap methodology for non-parametric statistical testing and confidence intervals.

Implements comprehensive bootstrap procedures for portfolio performance evaluation,
including confidence intervals for performance metrics and significance testing.
"""

import multiprocessing as mp
import warnings
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats


class BootstrapMethodology:
    """Bootstrap framework for non-parametric statistical inference in portfolio evaluation."""

    def __init__(
        self, n_bootstrap: int = 1000, random_state: Optional[int] = None, n_jobs: int = -1
    ):
        """Initialize bootstrap framework.

        Args:
            n_bootstrap: Number of bootstrap samples
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all CPUs)
        """
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()

        if random_state is not None:
            np.random.seed(random_state)

    def bootstrap_confidence_intervals(
        self,
        performance_metric: Callable,
        returns: Union[pd.Series, np.ndarray],
        confidence_level: float = 0.95,
        method: str = "percentile",
    ) -> tuple[float, float, dict]:
        """Bootstrap confidence intervals for performance metrics.

        Args:
            performance_metric: Function that calculates performance metric from returns
            returns: Return series
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
            method: Bootstrap CI method ('percentile', 'bias_corrected', 'bca')

        Returns:
            Tuple of (lower_bound, upper_bound, bootstrap_stats)
        """
        returns_array = np.asarray(returns)
        n = len(returns_array)

        # Original statistic
        original_stat = performance_metric(returns_array)

        # Generate bootstrap samples
        bootstrap_stats = []

        for _i in range(self.n_bootstrap):
            # Bootstrap sample with replacement
            boot_indices = np.random.choice(n, size=n, replace=True)
            boot_returns = returns_array[boot_indices]

            try:
                boot_stat = performance_metric(boot_returns)
                bootstrap_stats.append(boot_stat)
            except (ValueError, ZeroDivisionError):
                # Handle cases where metric calculation fails
                bootstrap_stats.append(np.nan)

        bootstrap_stats = np.array(bootstrap_stats)
        valid_stats = bootstrap_stats[~np.isnan(bootstrap_stats)]

        if len(valid_stats) < self.n_bootstrap * 0.8:
            warnings.warn("High proportion of invalid bootstrap samples", stacklevel=2)

        alpha = 1 - confidence_level

        if method == "percentile":
            lower_bound = np.percentile(valid_stats, 100 * alpha / 2)
            upper_bound = np.percentile(valid_stats, 100 * (1 - alpha / 2))

        elif method == "bias_corrected":
            # Bias-corrected percentile method
            bias_correction = stats.norm.ppf(
                (np.sum(valid_stats < original_stat)) / len(valid_stats)
            )

            alpha_lower = stats.norm.cdf(2 * bias_correction + stats.norm.ppf(alpha / 2))
            alpha_upper = stats.norm.cdf(2 * bias_correction + stats.norm.ppf(1 - alpha / 2))

            lower_bound = np.percentile(valid_stats, 100 * alpha_lower)
            upper_bound = np.percentile(valid_stats, 100 * alpha_upper)

        elif method == "bca":
            # Bias-corrected and accelerated (BCa) method
            bias_correction = stats.norm.ppf(
                (np.sum(valid_stats < original_stat)) / len(valid_stats)
            )

            # Acceleration constant using jackknife
            acceleration = self._calculate_acceleration(performance_metric, returns_array)

            # Adjusted percentiles
            z_alpha_2 = stats.norm.ppf(alpha / 2)
            z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)

            alpha_lower = stats.norm.cdf(
                bias_correction
                + (bias_correction + z_alpha_2) / (1 - acceleration * (bias_correction + z_alpha_2))
            )
            alpha_upper = stats.norm.cdf(
                bias_correction
                + (bias_correction + z_1_alpha_2)
                / (1 - acceleration * (bias_correction + z_1_alpha_2))
            )

            lower_bound = np.percentile(valid_stats, 100 * alpha_lower)
            upper_bound = np.percentile(valid_stats, 100 * alpha_upper)

        else:
            raise ValueError(f"Unknown bootstrap method: {method}")

        bootstrap_info = {
            "original_statistic": original_stat,
            "bootstrap_mean": np.mean(valid_stats),
            "bootstrap_std": np.std(valid_stats),
            "bootstrap_samples": len(valid_stats),
            "method": method,
            "confidence_level": confidence_level,
        }

        return lower_bound, upper_bound, bootstrap_info

    def _calculate_acceleration(self, performance_metric: Callable, returns: np.ndarray) -> float:
        """Calculate acceleration constant for BCa method using jackknife.

        Args:
            performance_metric: Performance metric function
            returns: Return series

        Returns:
            Acceleration constant
        """
        n = len(returns)
        jackknife_stats = []

        for i in range(n):
            # Leave-one-out sample
            jackknife_returns = np.concatenate([returns[:i], returns[i + 1 :]])
            try:
                jack_stat = performance_metric(jackknife_returns)
                jackknife_stats.append(jack_stat)
            except (ValueError, ZeroDivisionError):
                jackknife_stats.append(np.nan)

        jackknife_stats = np.array(jackknife_stats)
        valid_jack_stats = jackknife_stats[~np.isnan(jackknife_stats)]

        if len(valid_jack_stats) == 0:
            return 0.0

        jack_mean = np.mean(valid_jack_stats)

        # Acceleration calculation
        numerator = np.sum((jack_mean - valid_jack_stats) ** 3)
        denominator = 6 * (np.sum((jack_mean - valid_jack_stats) ** 2)) ** (3 / 2)

        acceleration = numerator / denominator if denominator != 0 else 0.0

        return acceleration

    def bootstrap_significance_test(
        self,
        returns_a: Union[pd.Series, np.ndarray],
        returns_b: Union[pd.Series, np.ndarray],
        performance_metric: Callable,
        alternative: str = "two-sided",
    ) -> dict[str, float]:
        """Bootstrap significance test for performance metric differences.

        Args:
            returns_a: Return series for portfolio A
            returns_b: Return series for portfolio B
            performance_metric: Performance metric function
            alternative: Test alternative ('two-sided', 'greater', 'less')

        Returns:
            Dictionary containing test results
        """
        ret_a = np.asarray(returns_a)
        ret_b = np.asarray(returns_b)

        if len(ret_a) != len(ret_b):
            raise ValueError("Return series must have equal length")

        # Original statistics
        original_stat_a = performance_metric(ret_a)
        original_stat_b = performance_metric(ret_b)
        original_diff = original_stat_a - original_stat_b

        # Combined dataset for null hypothesis (no difference)
        combined_returns = np.concatenate([ret_a, ret_b])
        n_a, n_b = len(ret_a), len(ret_b)

        # Bootstrap samples under null hypothesis
        bootstrap_diffs = []

        for _i in range(self.n_bootstrap):
            # Resample combined dataset
            boot_combined = np.random.choice(
                combined_returns, size=len(combined_returns), replace=True
            )

            # Split into two groups of original sizes
            boot_a = boot_combined[:n_a]
            boot_b = boot_combined[n_a : n_a + n_b]

            try:
                boot_stat_a = performance_metric(boot_a)
                boot_stat_b = performance_metric(boot_b)
                boot_diff = boot_stat_a - boot_stat_b
                bootstrap_diffs.append(boot_diff)
            except (ValueError, ZeroDivisionError):
                bootstrap_diffs.append(np.nan)

        bootstrap_diffs = np.array(bootstrap_diffs)
        valid_diffs = bootstrap_diffs[~np.isnan(bootstrap_diffs)]

        # Calculate p-value based on alternative hypothesis
        if alternative == "two-sided":
            p_value = np.mean(np.abs(valid_diffs) >= np.abs(original_diff))
        elif alternative == "greater":
            p_value = np.mean(valid_diffs >= original_diff)
        elif alternative == "less":
            p_value = np.mean(valid_diffs <= original_diff)
        else:
            raise ValueError(f"Unknown alternative: {alternative}")

        return {
            "original_diff": original_diff,
            "bootstrap_p_value": p_value,
            "bootstrap_mean_diff": np.mean(valid_diffs),
            "bootstrap_std_diff": np.std(valid_diffs),
            "valid_samples": len(valid_diffs),
            "alternative": alternative,
        }

    def paired_bootstrap_test(
        self,
        returns_a: Union[pd.Series, np.ndarray],
        returns_b: Union[pd.Series, np.ndarray],
        performance_metric: Callable,
    ) -> dict[str, float]:
        """Paired bootstrap test maintaining temporal structure of returns.

        Args:
            returns_a: Return series for portfolio A
            returns_b: Return series for portfolio B
            performance_metric: Performance metric function

        Returns:
            Dictionary containing paired bootstrap test results
        """
        ret_a = np.asarray(returns_a)
        ret_b = np.asarray(returns_b)

        if len(ret_a) != len(ret_b):
            raise ValueError("Return series must have equal length")

        n = len(ret_a)

        # Original difference
        original_stat_a = performance_metric(ret_a)
        original_stat_b = performance_metric(ret_b)
        original_diff = original_stat_a - original_stat_b

        # Paired differences
        paired_diffs = ret_a - ret_b

        # Bootstrap resampling of paired differences
        bootstrap_diffs = []

        for _i in range(self.n_bootstrap):
            # Resample paired differences with replacement
            boot_indices = np.random.choice(n, size=n, replace=True)
            boot_paired_diffs = paired_diffs[boot_indices]

            # Reconstruct return series under null (no difference)
            boot_a_null = ret_a[boot_indices]
            boot_b_null = boot_a_null - boot_paired_diffs + np.mean(paired_diffs)

            try:
                boot_stat_a = performance_metric(boot_a_null)
                boot_stat_b = performance_metric(boot_b_null)
                boot_diff = boot_stat_a - boot_stat_b
                bootstrap_diffs.append(boot_diff)
            except (ValueError, ZeroDivisionError):
                bootstrap_diffs.append(np.nan)

        bootstrap_diffs = np.array(bootstrap_diffs)
        valid_diffs = bootstrap_diffs[~np.isnan(bootstrap_diffs)]

        # Two-sided p-value
        p_value = np.mean(np.abs(valid_diffs) >= np.abs(original_diff))

        return {
            "original_diff": original_diff,
            "paired_bootstrap_p_value": p_value,
            "bootstrap_mean_diff": np.mean(valid_diffs),
            "bootstrap_std_diff": np.std(valid_diffs),
            "valid_samples": len(valid_diffs),
        }


class MultiMetricBootstrap:
    """Bootstrap framework for multiple performance metrics simultaneously."""

    def __init__(self, bootstrap_framework: BootstrapMethodology):
        """Initialize with bootstrap framework.

        Args:
            bootstrap_framework: BootstrapMethodology instance
        """
        self.bootstrap = bootstrap_framework

    def multi_metric_confidence_intervals(
        self,
        returns: Union[pd.Series, np.ndarray],
        metrics: dict[str, Callable],
        confidence_level: float = 0.95,
    ) -> pd.DataFrame:
        """Calculate confidence intervals for multiple metrics.

        Args:
            returns: Return series
            metrics: Dictionary mapping metric names to metric functions
            confidence_level: Confidence level

        Returns:
            DataFrame with confidence intervals for each metric
        """
        results = []

        for metric_name, metric_func in metrics.items():
            lower_bound, upper_bound, bootstrap_info = (
                self.bootstrap.bootstrap_confidence_intervals(
                    metric_func, returns, confidence_level
                )
            )

            result = {
                "metric": metric_name,
                "original_value": bootstrap_info["original_statistic"],
                "bootstrap_mean": bootstrap_info["bootstrap_mean"],
                "bootstrap_std": bootstrap_info["bootstrap_std"],
                "ci_lower": lower_bound,
                "ci_upper": upper_bound,
                "confidence_level": confidence_level,
            }

            results.append(result)

        return pd.DataFrame(results)

    def metric_comparison_matrix(
        self, returns_dict: dict[str, Union[pd.Series, np.ndarray]], metrics: dict[str, Callable]
    ) -> dict[str, pd.DataFrame]:
        """Create comparison matrices for multiple metrics across portfolios.

        Args:
            returns_dict: Dictionary mapping portfolio names to return series
            metrics: Dictionary mapping metric names to metric functions

        Returns:
            Dictionary mapping metric names to comparison DataFrames
        """
        results = {}

        for metric_name, metric_func in metrics.items():
            portfolio_names = list(returns_dict.keys())
            comparison_matrix = []

            for portfolio_a in portfolio_names:
                row = []
                for portfolio_b in portfolio_names:
                    if portfolio_a == portfolio_b:
                        row.append({"p_value": np.nan, "diff": 0.0})
                    else:
                        test_result = self.bootstrap.bootstrap_significance_test(
                            returns_dict[portfolio_a], returns_dict[portfolio_b], metric_func
                        )
                        row.append(
                            {
                                "p_value": test_result["bootstrap_p_value"],
                                "diff": test_result["original_diff"],
                            }
                        )

                comparison_matrix.append(row)

            # Convert to DataFrame format
            p_value_matrix = pd.DataFrame(
                [[cell["p_value"] for cell in row] for row in comparison_matrix],
                index=portfolio_names,
                columns=portfolio_names,
            )

            diff_matrix = pd.DataFrame(
                [[cell["diff"] for cell in row] for row in comparison_matrix],
                index=portfolio_names,
                columns=portfolio_names,
            )

            results[metric_name] = {"p_values": p_value_matrix, "differences": diff_matrix}

        return results


# Standard performance metric functions for bootstrap testing
def sharpe_ratio(returns: np.ndarray) -> float:
    """Calculate Sharpe ratio."""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    return np.mean(returns) / np.std(returns)


def sortino_ratio(returns: np.ndarray, target_return: float = 0.0) -> float:
    """Calculate Sortino ratio."""
    excess_returns = returns - target_return
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return np.inf if np.mean(excess_returns) > 0 else 0.0

    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    if downside_deviation == 0:
        return np.inf if np.mean(excess_returns) > 0 else 0.0

    return np.mean(excess_returns) / downside_deviation


def maximum_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown."""
    if len(returns) == 0:
        return 0.0

    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max

    return abs(np.min(drawdowns))


def calmar_ratio(returns: np.ndarray) -> float:
    """Calculate Calmar ratio."""
    mdd = maximum_drawdown(returns)
    if mdd == 0:
        return np.inf if np.mean(returns) > 0 else 0.0

    annual_return = np.mean(returns) * 252  # Assuming daily returns
    return annual_return / mdd
