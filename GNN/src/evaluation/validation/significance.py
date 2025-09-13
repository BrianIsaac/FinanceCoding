"""Statistical significance testing framework for portfolio performance evaluation.

Implements comprehensive statistical tests for evaluating performance differences,
including Sharpe ratio significance testing using the Jobson-Korkie test with
Memmel correction and pairwise comparison frameworks.
"""

import warnings
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm


class StatisticalValidation:
    """Statistical validation framework for portfolio performance analysis."""

    @staticmethod
    def sharpe_ratio_test(
        returns_a: Union[pd.Series, np.ndarray],
        returns_b: Union[pd.Series, np.ndarray],
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """Test statistical significance of Sharpe ratio differences using Jobson-Korkie test with Memmel correction.

        Args:
            returns_a: Return series for portfolio A
            returns_b: Return series for portfolio B
            alpha: Significance level (default: 0.05)

        Returns:
            Dictionary containing test statistics, p-value, and significance results
        """
        # Convert to numpy arrays for calculation
        ret_a = np.asarray(returns_a)
        ret_b = np.asarray(returns_b)

        if len(ret_a) != len(ret_b):
            raise ValueError("Return series must have equal length")

        n = len(ret_a)

        # Handle edge cases
        if n <= 1:
            raise ValueError(
                "Insufficient data for statistical testing - need at least 2 observations"
            )

        # Calculate basic statistics
        mean_a, mean_b = np.mean(ret_a), np.mean(ret_b)
        std_a, std_b = np.std(ret_a, ddof=1), np.std(ret_b, ddof=1)

        # Handle zero volatility case
        if std_a == 0 or std_b == 0:
            warnings.warn("Zero volatility detected in returns series", stacklevel=2)
            return {
                "test_statistic": np.nan,
                "p_value": np.nan,
                "is_significant": False,
                "confidence_level": 1 - alpha,
                "sharpe_a": float(mean_a / std_a if std_a > 0 else np.nan),
                "sharpe_b": float(mean_b / std_b if std_b > 0 else np.nan),
                "sharpe_diff": np.nan,
            }

        # Calculate Sharpe ratios
        sharpe_a = mean_a / std_a
        sharpe_b = mean_b / std_b
        sharpe_diff = sharpe_a - sharpe_b

        # Calculate correlation between return series
        corr = np.corrcoef(ret_a, ret_b)[0, 1]

        # Jobson-Korkie test statistic with Memmel correction
        # Variance of Sharpe ratio difference
        var_sharpe_diff = (1 / n) * (
            2
            - 2 * corr * (sharpe_a * sharpe_b)
            + 0.5 * sharpe_a**2 * (1 - corr**2)
            + 0.5 * sharpe_b**2 * (1 - corr**2)
        )

        # Memmel correction for small sample bias (avoid division by zero)
        if n > 1:
            memmel_correction = 1 - (1 / (4 * (n - 1))) * (
                sharpe_a**2 + sharpe_b**2 - 2 * corr * sharpe_a * sharpe_b
            )
        else:
            memmel_correction = 1.0

        var_sharpe_diff_corrected = var_sharpe_diff * memmel_correction

        # Test statistic
        if var_sharpe_diff_corrected <= 0:
            warnings.warn("Zero volatility detected", stacklevel=2)
            test_stat = np.nan
        else:
            test_stat = sharpe_diff / np.sqrt(var_sharpe_diff_corrected)

        # Two-tailed p-value
        if np.isnan(test_stat):
            p_value = np.nan
        else:
            p_value = 2 * (1 - norm.cdf(abs(test_stat)))

        return {
            "test_statistic": test_stat,
            "p_value": p_value,
            "is_significant": bool(p_value < alpha),
            "confidence_level": 1 - alpha,
            "sharpe_a": float(sharpe_a),
            "sharpe_b": float(sharpe_b),
            "sharpe_diff": float(sharpe_diff),
            "correlation": corr,
            "sample_size": n,
            "method": "Jobson-Korkie with Memmel correction",
        }

    @staticmethod
    def asymptotic_sharpe_test(
        returns_a: Union[pd.Series, np.ndarray],
        returns_b: Union[pd.Series, np.ndarray],
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """Asymptotic statistical test for Sharpe ratio differences with proper variance calculations.

        Args:
            returns_a: Return series for portfolio A
            returns_b: Return series for portfolio B
            alpha: Significance level

        Returns:
            Dictionary containing asymptotic test results
        """
        ret_a = np.asarray(returns_a)
        ret_b = np.asarray(returns_b)
        n = len(ret_a)

        # Calculate moments
        mean_a, mean_b = np.mean(ret_a), np.mean(ret_b)
        var_a, var_b = np.var(ret_a, ddof=1), np.var(ret_b, ddof=1)
        std_a, std_b = np.sqrt(var_a), np.sqrt(var_b)

        if std_a == 0 or std_b == 0:
            warnings.warn("Zero volatility in asymptotic test", stacklevel=2)
            return {"test_statistic": np.nan, "p_value": np.nan, "is_significant": False}

        # Calculate higher moments for asymptotic variance
        skew_a = stats.skew(ret_a)
        skew_b = stats.skew(ret_b)
        kurt_a = stats.kurtosis(ret_a)
        kurt_b = stats.kurtosis(ret_b)

        # Sharpe ratios
        sharpe_a = mean_a / std_a
        sharpe_b = mean_b / std_b

        # Cross-moments
        cov_ab = np.cov(ret_a, ret_b)[0, 1]
        cov_ab / (std_a * std_b) if (std_a * std_b) > 0 else 0

        # Asymptotic variance using delta method
        # Partial derivatives
        d_sharpe_a_mean = 1 / std_a
        d_sharpe_a_var = -mean_a / (2 * var_a ** (3 / 2))
        d_sharpe_b_mean = -1 / std_b
        d_sharpe_b_var = mean_b / (2 * var_b ** (3 / 2))

        # Variance-covariance matrix elements
        var_mean_a = var_a / n
        var_mean_b = var_b / n
        var_var_a = (kurt_a + 2) * var_a**2 / n
        var_var_b = (kurt_b + 2) * var_b**2 / n
        cov_mean_var_a = skew_a * var_a ** (3 / 2) / n
        cov_mean_var_b = skew_b * var_b ** (3 / 2) / n

        # Asymptotic variance of Sharpe difference
        asymp_var = (
            d_sharpe_a_mean**2 * var_mean_a
            + d_sharpe_a_var**2 * var_var_a
            + d_sharpe_b_mean**2 * var_mean_b
            + d_sharpe_b_var**2 * var_var_b
            + 2 * d_sharpe_a_mean * d_sharpe_a_var * cov_mean_var_a
            + 2 * d_sharpe_b_mean * d_sharpe_b_var * cov_mean_var_b
        )

        # Test statistic
        sharpe_diff = sharpe_a - sharpe_b
        test_stat = sharpe_diff / np.sqrt(asymp_var) if asymp_var > 0 else 0
        p_value = 2 * (1 - norm.cdf(abs(test_stat)))

        return {
            "test_statistic": test_stat,
            "p_value": p_value,
            "is_significant": p_value < alpha,
            "asymptotic_variance": asymp_var,
            "sharpe_diff": sharpe_diff,
            "method": "Asymptotic delta method",
        }

    @staticmethod
    def pairwise_comparison_framework(
        returns_dict: dict[str, Union[pd.Series, np.ndarray]],
        baseline_keys: Optional[list[str]] = None,
        alpha: float = 0.05,
        method: str = "jobson_korkie",
    ) -> pd.DataFrame:
        """Framework for pairwise comparisons of multiple portfolios against baselines.

        Args:
            returns_dict: Dictionary mapping portfolio names to return series
            baseline_keys: List of baseline portfolio keys (if None, compares all vs all)
            alpha: Significance level
            method: Statistical test method ('jobson_korkie' or 'asymptotic')

        Returns:
            DataFrame containing pairwise comparison results
        """
        results = []

        if baseline_keys is None:
            # All vs all comparison
            portfolios = list(returns_dict.keys())
            comparisons = [(i, j) for i in portfolios for j in portfolios if i != j]
        else:
            # ML approaches vs baselines
            ml_portfolios = [k for k in returns_dict.keys() if k not in baseline_keys]
            comparisons = [(ml, baseline) for ml in ml_portfolios for baseline in baseline_keys]

        test_func = (
            StatisticalValidation.sharpe_ratio_test
            if method == "jobson_korkie"
            else StatisticalValidation.asymptotic_sharpe_test
        )

        for portfolio_a, portfolio_b in comparisons:
            returns_a = returns_dict[portfolio_a]
            returns_b = returns_dict[portfolio_b]

            test_result = test_func(returns_a, returns_b, alpha)

            result_row = {
                "portfolio_a": portfolio_a,
                "portfolio_b": portfolio_b,
                "sharpe_a": test_result.get("sharpe_a", np.nan),
                "sharpe_b": test_result.get("sharpe_b", np.nan),
                "sharpe_diff": test_result.get("sharpe_diff", np.nan),
                "test_statistic": test_result["test_statistic"],
                "p_value": test_result["p_value"],
                "is_significant": test_result["is_significant"],
                "method": method,
                "sample_size": len(returns_a),
            }

            results.append(result_row)

        return pd.DataFrame(results)


class PerformanceSignificanceTest:
    """Extended statistical testing framework for comprehensive performance evaluation."""

    def __init__(self, alpha: float = 0.05):
        """Initialize with significance level.

        Args:
            alpha: Type I error rate (significance level)
        """
        self.alpha = alpha

    def comprehensive_comparison(
        self, returns_dict: dict[str, pd.Series], metrics: Optional[list[str]] = None
    ) -> dict[str, pd.DataFrame]:
        """Perform comprehensive statistical comparison across multiple performance metrics.

        Args:
            returns_dict: Dictionary of return series for each portfolio
            metrics: List of metrics to test ('sharpe', 'sortino', 'calmar', etc.)

        Returns:
            Dictionary mapping metric names to comparison DataFrames
        """
        if metrics is None:
            metrics = ["sharpe"]

        results = {}

        for metric in metrics:
            if metric == "sharpe":
                results[metric] = StatisticalValidation.pairwise_comparison_framework(
                    returns_dict, alpha=self.alpha
                )
            # Additional metrics can be added here

        return results

    def rolling_significance_analysis(
        self, returns_dict: dict[str, pd.Series], window_size: int = 252, step_size: int = 63
    ) -> pd.DataFrame:
        """Analyze significance of performance differences across rolling windows.

        Args:
            returns_dict: Dictionary of return series
            window_size: Rolling window size in periods
            step_size: Step size between windows

        Returns:
            DataFrame with rolling significance analysis results
        """
        list(returns_dict.keys())
        results = []

        # Get aligned series
        aligned_returns = pd.DataFrame(returns_dict).dropna()

        for start_idx in range(0, len(aligned_returns) - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window_returns = aligned_returns.iloc[start_idx:end_idx]

            window_dict = {col: window_returns[col] for col in window_returns.columns}

            # Perform pairwise comparisons for this window
            window_results = StatisticalValidation.pairwise_comparison_framework(
                window_dict, alpha=self.alpha
            )

            window_results["window_start"] = aligned_returns.index[start_idx]
            window_results["window_end"] = aligned_returns.index[end_idx - 1]

            results.append(window_results)

        if not results:
            # Return empty DataFrame with expected columns when no windows can be created
            return pd.DataFrame(columns=["window_start", "window_end"])

        return pd.concat(results, ignore_index=True)
