"""Rolling window consistency analysis for portfolio performance evaluation.

Implements comprehensive analysis of performance stability and consistency across
time periods, including rolling Sharpe ratio stability, performance persistence,
and regime-specific validation with temporal stability metrics.
"""

import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr


@dataclass
class ConsistencyMetrics:
    """Container for consistency analysis results."""

    stability_score: float
    persistence_score: float
    regime_consistency: dict[str, float]
    temporal_stability: dict[str, float]
    rolling_statistics: pd.DataFrame


class RollingConsistencyAnalyzer:
    """Framework for analyzing performance consistency across rolling windows."""

    def __init__(self, window_size: int = 252, step_size: int = 63, min_periods: int = 126):
        """Initialize rolling consistency analyzer.

        Args:
            window_size: Rolling window size in periods (default: 252 for 1 year daily)
            step_size: Step size between windows (default: 63 for quarterly)
            min_periods: Minimum periods required for calculations
        """
        self.window_size = window_size
        self.step_size = step_size
        self.min_periods = min_periods

    def rolling_sharpe_stability_test(
        self,
        returns: Union[pd.Series, np.ndarray],
        benchmark_returns: Optional[Union[pd.Series, np.ndarray]] = None,
        confidence_level: float = 0.95,
    ) -> dict[str, Union[pd.DataFrame, float, dict]]:
        """Test rolling Sharpe ratio stability across evaluation windows.

        Args:
            returns: Return series to analyze
            benchmark_returns: Optional benchmark returns for relative analysis
            confidence_level: Confidence level for stability bounds

        Returns:
            Dictionary containing rolling stability analysis results
        """
        returns_series = pd.Series(returns) if not isinstance(returns, pd.Series) else returns

        # Calculate rolling Sharpe ratios
        rolling_sharpe = self._calculate_rolling_sharpe(returns_series)

        if len(rolling_sharpe) < 2:
            warnings.warn("Insufficient rolling windows for stability analysis", stacklevel=2)
            return {
                "rolling_sharpe_ratios": rolling_sharpe,
                "stability_score": np.nan,
                "stability_pvalue": np.nan,
                "consistency_metrics": {},
            }

        # Stability tests
        stability_results = self._analyze_sharpe_stability(rolling_sharpe, confidence_level)

        # Benchmark comparison if provided
        benchmark_results = {}
        if benchmark_returns is not None:
            benchmark_series = (
                pd.Series(benchmark_returns)
                if not isinstance(benchmark_returns, pd.Series)
                else benchmark_returns
            )
            benchmark_sharpe = self._calculate_rolling_sharpe(benchmark_series)

            if len(benchmark_sharpe) == len(rolling_sharpe):
                benchmark_results = self._compare_rolling_sharpe(rolling_sharpe, benchmark_sharpe)

        return {
            "rolling_sharpe_ratios": rolling_sharpe,
            "stability_score": stability_results["stability_score"],
            "stability_pvalue": stability_results["stability_pvalue"],
            "consistency_metrics": stability_results["consistency_metrics"],
            "benchmark_comparison": benchmark_results,
            "analysis_parameters": {
                "window_size": self.window_size,
                "step_size": self.step_size,
                "confidence_level": confidence_level,
            },
        }

    def performance_persistence_analysis(
        self,
        returns_dict: dict[str, Union[pd.Series, np.ndarray]],
        performance_metric: Callable = None,
    ) -> dict[str, Union[pd.DataFrame, float]]:
        """Analyze performance persistence using rank correlation tests.

        Args:
            returns_dict: Dictionary mapping strategy names to return series
            performance_metric: Function to calculate performance metric (default: Sharpe ratio)

        Returns:
            Dictionary containing persistence analysis results
        """
        if performance_metric is None:
            performance_metric = self._sharpe_ratio

        # Calculate rolling performance for each strategy
        rolling_performance = {}
        for strategy_name, returns in returns_dict.items():
            returns_series = pd.Series(returns) if not isinstance(returns, pd.Series) else returns
            rolling_perf = self._calculate_rolling_performance(returns_series, performance_metric)
            rolling_performance[strategy_name] = rolling_perf

        # Create combined DataFrame
        performance_df = pd.DataFrame(rolling_performance)
        performance_df = performance_df.dropna()

        if len(performance_df) < 2:
            warnings.warn("Insufficient data for persistence analysis", stacklevel=2)
            return {"persistence_results": pd.DataFrame(), "summary_statistics": {}}

        # Calculate rank correlations between consecutive periods
        persistence_results = self._calculate_persistence_metrics(performance_df)

        # Calculate summary statistics
        summary_stats = self._summarize_persistence_results(persistence_results)

        return {
            "rolling_performance": performance_df,
            "persistence_results": persistence_results,
            "summary_statistics": summary_stats,
        }

    def regime_specific_consistency_validation(
        self,
        returns: Union[pd.Series, np.ndarray],
        regime_indicator: Optional[Union[pd.Series, np.ndarray]] = None,
        regime_detection_method: str = "volatility_based",
    ) -> dict[str, Union[pd.DataFrame, dict]]:
        """Validate performance consistency across different market regimes.

        Args:
            returns: Return series to analyze
            regime_indicator: Optional pre-defined regime labels
            regime_detection_method: Method for automatic regime detection

        Returns:
            Dictionary containing regime-specific consistency analysis
        """
        returns_series = pd.Series(returns) if not isinstance(returns, pd.Series) else returns

        # Determine market regimes
        if regime_indicator is not None:
            regimes = pd.Series(regime_indicator)
        else:
            regimes = self._detect_market_regimes(returns_series, regime_detection_method)

        # Align returns and regimes
        aligned_data = pd.DataFrame({"returns": returns_series, "regime": regimes}).dropna()

        # Calculate performance metrics by regime
        regime_performance = self._calculate_regime_performance(aligned_data)

        # Test consistency across regimes
        consistency_tests = self._test_regime_consistency(aligned_data)

        # Transition analysis
        transition_analysis = self._analyze_regime_transitions(aligned_data)

        return {
            "regime_performance": regime_performance,
            "consistency_tests": consistency_tests,
            "transition_analysis": transition_analysis,
            "regime_summary": regimes.value_counts().to_dict(),
        }

    def temporal_stability_framework(
        self, returns: Union[pd.Series, np.ndarray], metrics: list[str] = None
    ) -> dict[str, Union[pd.DataFrame, dict]]:
        """Build comprehensive temporal stability metrics and visualization framework.

        Args:
            returns: Return series to analyze
            metrics: List of stability metrics to calculate

        Returns:
            Dictionary containing temporal stability analysis and visualization data
        """
        if metrics is None:
            metrics = ["sharpe_ratio", "volatility", "skewness", "maximum_drawdown"]

        returns_series = pd.Series(returns) if not isinstance(returns, pd.Series) else returns

        # Calculate rolling metrics
        rolling_metrics = self._calculate_rolling_metrics(returns_series, metrics)

        # Stability analysis for each metric
        stability_analysis = {}
        for metric in metrics:
            if metric in rolling_metrics.columns:
                metric_series = rolling_metrics[metric].dropna()
                if len(metric_series) >= 2:
                    stability_analysis[metric] = self._analyze_metric_stability(metric_series)

        # Cross-metric correlation analysis
        correlation_analysis = self._analyze_metric_correlations(rolling_metrics)

        # Trend analysis
        trend_analysis = self._analyze_temporal_trends(rolling_metrics)

        # Visualization data preparation
        visualization_data = self._prepare_visualization_data(rolling_metrics, stability_analysis)

        return {
            "rolling_metrics": rolling_metrics,
            "stability_analysis": stability_analysis,
            "correlation_analysis": correlation_analysis,
            "trend_analysis": trend_analysis,
            "visualization_data": visualization_data,
        }

    # Helper methods
    def _calculate_rolling_sharpe(self, returns: pd.Series) -> pd.Series:
        """Calculate rolling Sharpe ratios."""
        rolling_mean = returns.rolling(window=self.window_size, min_periods=self.min_periods).mean()
        rolling_std = returns.rolling(window=self.window_size, min_periods=self.min_periods).std()

        rolling_sharpe = rolling_mean / rolling_std
        return rolling_sharpe.dropna()

    def _calculate_rolling_performance(
        self, returns: pd.Series, performance_metric: Callable
    ) -> pd.Series:
        """Calculate rolling performance using specified metric."""
        rolling_performance = []
        rolling_dates = []

        for i in range(self.window_size - 1, len(returns), self.step_size):
            window_start = max(0, i - self.window_size + 1)
            window_returns = returns.iloc[window_start : i + 1]

            if len(window_returns) >= self.min_periods:
                try:
                    perf = performance_metric(window_returns.values)
                    rolling_performance.append(perf)
                    rolling_dates.append(returns.index[i] if hasattr(returns, "index") else i)
                except (ValueError, ZeroDivisionError):
                    rolling_performance.append(np.nan)
                    rolling_dates.append(returns.index[i] if hasattr(returns, "index") else i)

        return pd.Series(rolling_performance, index=rolling_dates)

    def _analyze_sharpe_stability(self, rolling_sharpe: pd.Series, confidence_level: float) -> dict:
        """Analyze stability of rolling Sharpe ratios."""
        # Remove NaN values
        valid_sharpe = rolling_sharpe.dropna()

        if len(valid_sharpe) < 2:
            return {
                "stability_score": np.nan,
                "stability_pvalue": np.nan,
                "consistency_metrics": {},
            }

        # Calculate stability metrics
        mean_sharpe = np.mean(valid_sharpe)
        std_sharpe = np.std(valid_sharpe, ddof=1)
        cv_sharpe = std_sharpe / abs(mean_sharpe) if mean_sharpe != 0 else np.inf

        # Stability score (inverse of coefficient of variation)
        stability_score = 1 / (1 + cv_sharpe) if not np.isinf(cv_sharpe) else 0

        # Statistical test for stability (test if variance is "small")
        # Using chi-square test for variance
        alpha = 1 - confidence_level
        chi2_stat = (len(valid_sharpe) - 1) * std_sharpe**2
        stats.chi2.ppf(1 - alpha / 2, len(valid_sharpe) - 1)
        stability_pvalue = 1 - stats.chi2.cdf(chi2_stat, len(valid_sharpe) - 1)

        # Additional consistency metrics
        consistency_metrics = {
            "mean_sharpe": mean_sharpe,
            "std_sharpe": std_sharpe,
            "coefficient_of_variation": cv_sharpe,
            "min_sharpe": np.min(valid_sharpe),
            "max_sharpe": np.max(valid_sharpe),
            "sharpe_range": np.max(valid_sharpe) - np.min(valid_sharpe),
            "percentage_positive": np.sum(valid_sharpe > 0) / len(valid_sharpe) * 100,
            "trend_correlation": self._calculate_trend_correlation(valid_sharpe),
        }

        return {
            "stability_score": stability_score,
            "stability_pvalue": stability_pvalue,
            "consistency_metrics": consistency_metrics,
        }

    def _compare_rolling_sharpe(self, sharpe_a: pd.Series, sharpe_b: pd.Series) -> dict:
        """Compare rolling Sharpe ratios between two series."""
        # Align series
        aligned = pd.DataFrame({"a": sharpe_a, "b": sharpe_b}).dropna()

        if len(aligned) < 2:
            return {"correlation": np.nan, "relative_stability": np.nan}

        # Calculate correlation
        correlation, corr_pvalue = pearsonr(aligned["a"], aligned["b"])

        # Relative stability (ratio of coefficients of variation)
        cv_a = (
            np.std(aligned["a"]) / abs(np.mean(aligned["a"]))
            if np.mean(aligned["a"]) != 0
            else np.inf
        )
        cv_b = (
            np.std(aligned["b"]) / abs(np.mean(aligned["b"]))
            if np.mean(aligned["b"]) != 0
            else np.inf
        )

        relative_stability = cv_b / cv_a if not np.isinf(cv_a) and cv_a != 0 else np.nan

        return {
            "correlation": correlation,
            "correlation_pvalue": corr_pvalue,
            "relative_stability": relative_stability,
            "outperformance_percentage": np.sum(aligned["a"] > aligned["b"]) / len(aligned) * 100,
        }

    def _calculate_persistence_metrics(self, performance_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate persistence metrics using rank correlations."""
        results = []

        for i in range(len(performance_df) - 1):
            current_period = performance_df.iloc[i]
            next_period = performance_df.iloc[i + 1]

            # Remove NaN values
            valid_data = pd.DataFrame({"current": current_period, "next": next_period}).dropna()

            if len(valid_data) >= 3:  # Need at least 3 observations for meaningful correlation
                # Spearman rank correlation
                spearman_corr, spearman_p = spearmanr(valid_data["current"], valid_data["next"])

                # Pearson correlation
                pearson_corr, pearson_p = pearsonr(valid_data["current"], valid_data["next"])

                results.append(
                    {
                        "period": i,
                        "spearman_correlation": spearman_corr,
                        "spearman_pvalue": spearman_p,
                        "pearson_correlation": pearson_corr,
                        "pearson_pvalue": pearson_p,
                        "n_observations": len(valid_data),
                    }
                )

        return pd.DataFrame(results)

    def _summarize_persistence_results(self, persistence_results: pd.DataFrame) -> dict:
        """Summarize persistence analysis results."""
        if persistence_results.empty:
            return {}

        summary = {
            "average_spearman_correlation": persistence_results["spearman_correlation"].mean(),
            "average_pearson_correlation": persistence_results["pearson_correlation"].mean(),
            "significant_spearman_periods": np.sum(persistence_results["spearman_pvalue"] < 0.05),
            "significant_pearson_periods": np.sum(persistence_results["pearson_pvalue"] < 0.05),
            "total_periods": len(persistence_results),
            "persistence_strength": persistence_results["spearman_correlation"].mean(),
        }

        # Classify persistence strength
        avg_corr = summary["persistence_strength"]
        if avg_corr > 0.7:
            summary["persistence_category"] = "High"
        elif avg_corr > 0.4:
            summary["persistence_category"] = "Moderate"
        elif avg_corr > 0.1:
            summary["persistence_category"] = "Low"
        else:
            summary["persistence_category"] = "None"

        return summary

    def _detect_market_regimes(self, returns: pd.Series, method: str) -> pd.Series:
        """Detect market regimes automatically."""
        if method == "volatility_based":
            # Use rolling volatility to define regimes
            rolling_vol = returns.rolling(
                window=63, min_periods=30
            ).std()  # Quarterly rolling volatility
            vol_median = rolling_vol.median()

            regimes = pd.Series(index=returns.index, dtype="object")
            regimes[rolling_vol <= vol_median] = "Low Volatility"
            regimes[rolling_vol > vol_median] = "High Volatility"

        elif method == "return_based":
            # Use rolling returns to define regimes
            rolling_returns = returns.rolling(window=63, min_periods=30).mean()

            regimes = pd.Series(index=returns.index, dtype="object")
            regimes[rolling_returns > 0] = "Bull Market"
            regimes[rolling_returns <= 0] = "Bear Market"

        else:
            warnings.warn(f"Unknown regime detection method: {method}", stacklevel=2)
            regimes = pd.Series(["Unknown"] * len(returns), index=returns.index)

        return regimes

    def _calculate_regime_performance(self, aligned_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate performance metrics by regime."""
        regime_stats = []

        for regime in aligned_data["regime"].unique():
            regime_returns = aligned_data[aligned_data["regime"] == regime]["returns"]

            if len(regime_returns) > 0:
                stats_dict = {
                    "regime": regime,
                    "mean_return": regime_returns.mean(),
                    "volatility": regime_returns.std(),
                    "sharpe_ratio": (
                        regime_returns.mean() / regime_returns.std()
                        if regime_returns.std() > 0
                        else 0
                    ),
                    "skewness": stats.skew(regime_returns),
                    "kurtosis": stats.kurtosis(regime_returns),
                    "max_drawdown": self._calculate_max_drawdown(regime_returns),
                    "n_observations": len(regime_returns),
                }
                regime_stats.append(stats_dict)

        return pd.DataFrame(regime_stats)

    def _test_regime_consistency(self, aligned_data: pd.DataFrame) -> dict:
        """Test for consistency across market regimes."""
        regimes = aligned_data["regime"].unique()

        if len(regimes) < 2:
            return {"regime_equality_test": {"statistic": np.nan, "pvalue": np.nan}}

        # Group returns by regime
        regime_groups = [
            aligned_data[aligned_data["regime"] == regime]["returns"].values for regime in regimes
        ]

        # ANOVA test for equality of means across regimes
        try:
            f_stat, f_pvalue = stats.f_oneway(*regime_groups)
            anova_result = {"f_statistic": f_stat, "f_pvalue": f_pvalue}
        except ValueError:
            anova_result = {"f_statistic": np.nan, "f_pvalue": np.nan}

        # Levene's test for equality of variances
        try:
            levene_stat, levene_pvalue = stats.levene(*regime_groups)
            levene_result = {"levene_statistic": levene_stat, "levene_pvalue": levene_pvalue}
        except ValueError:
            levene_result = {"levene_statistic": np.nan, "levene_pvalue": np.nan}

        return {
            "regime_equality_test": anova_result,
            "variance_equality_test": levene_result,
            "n_regimes": len(regimes),
        }

    def _analyze_regime_transitions(self, aligned_data: pd.DataFrame) -> dict:
        """Analyze regime transition effects on performance."""
        regime_series = aligned_data["regime"]
        returns_series = aligned_data["returns"]

        # Identify regime transitions
        regime_changes = regime_series != regime_series.shift(1)
        transition_points = regime_changes[regime_changes].index

        if len(transition_points) == 0:
            return {"n_transitions": 0, "transition_effects": {}}

        # Analyze returns around transitions
        transition_effects = []
        window = 10  # Look at +/- 10 periods around transitions

        for transition_point in transition_points[window:-window]:  # Avoid edge effects
            pre_transition = returns_series.loc[transition_point - window : transition_point - 1]
            post_transition = returns_series.loc[transition_point : transition_point + window - 1]

            if len(pre_transition) == window and len(post_transition) == window:
                # t-test for difference in means
                t_stat, t_pvalue = stats.ttest_ind(pre_transition, post_transition)

                transition_effects.append(
                    {
                        "transition_date": transition_point,
                        "pre_mean": pre_transition.mean(),
                        "post_mean": post_transition.mean(),
                        "mean_difference": post_transition.mean() - pre_transition.mean(),
                        "t_statistic": t_stat,
                        "t_pvalue": t_pvalue,
                    }
                )

        return {
            "n_transitions": len(transition_points),
            "transition_effects": pd.DataFrame(transition_effects),
            "average_transition_impact": (
                np.mean([effect["mean_difference"] for effect in transition_effects])
                if transition_effects
                else np.nan
            ),
        }

    def _calculate_rolling_metrics(self, returns: pd.Series, metrics: list[str]) -> pd.DataFrame:
        """Calculate rolling metrics for temporal stability analysis."""
        rolling_data = {}

        for metric in metrics:
            if metric == "sharpe_ratio":
                rolling_data[metric] = self._calculate_rolling_sharpe(returns)
            elif metric == "volatility":
                rolling_data[metric] = returns.rolling(
                    window=self.window_size, min_periods=self.min_periods
                ).std()
            elif metric == "skewness":
                rolling_data[metric] = returns.rolling(
                    window=self.window_size, min_periods=self.min_periods
                ).skew()
            elif metric == "maximum_drawdown":
                rolling_data[metric] = self._calculate_rolling_max_drawdown(returns)
            # Additional metrics can be added here

        return pd.DataFrame(rolling_data)

    def _analyze_metric_stability(self, metric_series: pd.Series) -> dict:
        """Analyze stability of a specific metric over time."""
        # Basic statistics
        mean_val = metric_series.mean()
        std_val = metric_series.std()
        cv_val = std_val / abs(mean_val) if mean_val != 0 else np.inf

        # Trend analysis
        trend_correlation = self._calculate_trend_correlation(metric_series)

        # Change point detection (simple version)
        change_points = self._detect_change_points(metric_series)

        return {
            "mean": mean_val,
            "std": std_val,
            "coefficient_of_variation": cv_val,
            "min_value": metric_series.min(),
            "max_value": metric_series.max(),
            "trend_correlation": trend_correlation,
            "n_change_points": len(change_points),
            "stability_score": 1 / (1 + cv_val) if not np.isinf(cv_val) else 0,
        }

    def _analyze_metric_correlations(self, rolling_metrics: pd.DataFrame) -> pd.DataFrame:
        """Analyze correlations between different rolling metrics."""
        return rolling_metrics.corr()

    def _analyze_temporal_trends(self, rolling_metrics: pd.DataFrame) -> dict:
        """Analyze temporal trends in rolling metrics."""
        trends = {}

        for metric in rolling_metrics.columns:
            metric_series = rolling_metrics[metric].dropna()
            if len(metric_series) >= 2:
                trend_corr = self._calculate_trend_correlation(metric_series)

                # Simple linear regression for trend
                x = np.arange(len(metric_series))
                y = metric_series.values
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                trends[metric] = {
                    "trend_correlation": trend_corr,
                    "slope": slope,
                    "r_squared": r_value**2,
                    "p_value": p_value,
                    "trend_direction": "increasing" if slope > 0 else "decreasing",
                }

        return trends

    def _prepare_visualization_data(
        self, rolling_metrics: pd.DataFrame, stability_analysis: dict
    ) -> dict:
        """Prepare data for visualization."""
        viz_data = {
            "time_series_data": rolling_metrics.to_dict("records"),
            "stability_scores": {
                metric: analysis["stability_score"]
                for metric, analysis in stability_analysis.items()
            },
            "summary_statistics": {
                metric: {
                    "mean": rolling_metrics[metric].mean(),
                    "std": rolling_metrics[metric].std(),
                    "min": rolling_metrics[metric].min(),
                    "max": rolling_metrics[metric].max(),
                }
                for metric in rolling_metrics.columns
            },
        }

        return viz_data

    # Utility methods
    def _sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns)

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())

    def _calculate_rolling_max_drawdown(self, returns: pd.Series) -> pd.Series:
        """Calculate rolling maximum drawdown."""
        rolling_mdd = []
        rolling_dates = []

        for i in range(self.window_size - 1, len(returns)):
            window_returns = returns.iloc[i - self.window_size + 1 : i + 1]
            mdd = self._calculate_max_drawdown(window_returns)
            rolling_mdd.append(mdd)
            rolling_dates.append(returns.index[i] if hasattr(returns, "index") else i)

        return pd.Series(rolling_mdd, index=rolling_dates)

    def _calculate_trend_correlation(self, series: pd.Series) -> float:
        """Calculate correlation with time trend."""
        if len(series) < 2:
            return np.nan

        time_index = np.arange(len(series))
        correlation, _ = pearsonr(time_index, series.values)
        return correlation

    def _detect_change_points(self, series: pd.Series, threshold: float = 1.5) -> list[int]:
        """Simple change point detection using rolling statistics."""
        if len(series) < 10:
            return []

        # Calculate rolling mean and std
        rolling_mean = series.rolling(window=5, center=True).mean()
        rolling_std = series.rolling(window=5, center=True).std()

        # Detect points where values deviate significantly from rolling statistics
        deviations = np.abs((series - rolling_mean) / rolling_std)
        change_points = []

        for i in range(len(deviations)):
            if deviations.iloc[i] > threshold:
                change_points.append(i)

        return change_points
