"""
Rolling performance analysis framework for time-varying metrics.

This module provides comprehensive rolling performance analysis with configurable
window sizes, market regime detection, and performance stability analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .returns import ReturnAnalyzer, ReturnMetricsConfig
from .risk import RiskAnalytics, RiskMetricsConfig


@dataclass
class RollingAnalysisConfig:
    """Configuration for rolling performance analysis."""

    default_windows: list[int] = None  # Window sizes in trading days
    min_window_size: int = 63  # Minimum 3 months
    max_window_size: int = 756  # Maximum 3 years
    step_size: int = 21  # Monthly steps
    confidence_level: float = 0.95
    regime_detection_lookback: int = 252  # 1 year for regime detection

    def __post_init__(self):
        """Set default window sizes if not provided."""
        if self.default_windows is None:
            self.default_windows = [63, 126, 252, 504, 756]  # 3M, 6M, 1Y, 2Y, 3Y


class RollingPerformanceAnalyzer:
    """
    Comprehensive rolling performance analysis framework.

    Implements rolling metrics calculation with configurable window sizes,
    time-varying performance analysis, market regime detection, and
    performance stability analysis across time periods.
    """

    def __init__(self, config: RollingAnalysisConfig = None):
        """
        Initialize rolling performance analyzer.

        Args:
            config: Configuration for rolling analysis
        """
        self.config = config or RollingAnalysisConfig()
        self.return_analyzer = ReturnAnalyzer(ReturnMetricsConfig())
        self.risk_analyzer = RiskAnalytics(RiskMetricsConfig())

    def calculate_rolling_metrics(
        self,
        returns: pd.Series,
        window_size: int = 252,
        step_size: int = None,
        metrics_to_calculate: list[str] = None,
    ) -> pd.DataFrame:
        """
        Implement rolling metrics calculation with configurable window sizes.

        Args:
            returns: Daily returns series
            window_size: Rolling window size in trading days
            step_size: Step size for rolling calculation (default: window_size // 4)
            metrics_to_calculate: List of metrics to calculate (default: all)

        Returns:
            DataFrame with rolling metrics over time
        """
        if step_size is None:
            step_size = max(1, window_size // 4)

        if metrics_to_calculate is None:
            metrics_to_calculate = [
                "annualized_return",
                "annualized_volatility",
                "sharpe_ratio",
                "max_drawdown",
                "calmar_ratio",
                "sortino_ratio",
                "skewness",
                "kurtosis",
                "var_95",
                "cvar_95",
            ]

        if len(returns) < window_size:
            return pd.DataFrame()

        # Prepare results container
        results = []
        dates = []

        # Calculate rolling metrics
        for start_idx in range(0, len(returns) - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window_returns = returns.iloc[start_idx:end_idx]
            window_end_date = returns.index[end_idx - 1]

            if len(window_returns) == window_size:  # Ensure full window
                # Calculate metrics for this window
                basic_metrics = self.return_analyzer.calculate_basic_metrics(window_returns)
                risk_metrics = self.risk_analyzer.calculate_comprehensive_risk_metrics(
                    window_returns
                )

                # Extract requested metrics
                window_result = {"date": window_end_date}

                for metric in metrics_to_calculate:
                    if metric in basic_metrics:
                        window_result[metric] = basic_metrics[metric]
                    elif metric == "sortino_ratio":
                        window_result[metric] = self.risk_analyzer.calculate_sortino_ratio(
                            window_returns
                        )
                    elif "var_" in metric or "cvar_" in metric:
                        window_result[metric] = risk_metrics.get("var_metrics", {}).get(metric, 0.0)
                    else:
                        # Try to get from risk metrics
                        window_result[metric] = risk_metrics.get(metric, 0.0)

                results.append(window_result)
                dates.append(window_end_date)

        if not results:
            return pd.DataFrame()

        # Create DataFrame
        rolling_df = pd.DataFrame(results)
        rolling_df.set_index("date", inplace=True)

        return rolling_df

    def calculate_time_varying_performance(
        self, returns: pd.Series, benchmark_returns: pd.Series = None
    ) -> dict[str, Any]:
        """
        Add time-varying performance analysis and trend identification.

        Args:
            returns: Portfolio returns series
            benchmark_returns: Optional benchmark returns for comparison

        Returns:
            Dictionary containing time-varying performance analysis
        """
        results = {}

        # Calculate rolling metrics for multiple window sizes
        for window_size in self.config.default_windows:
            if len(returns) >= window_size:
                rolling_metrics = self.calculate_rolling_metrics(returns, window_size)

                if not rolling_metrics.empty:
                    # Calculate trends for key metrics
                    window_name = f"{window_size}d"
                    results[window_name] = {
                        "metrics": rolling_metrics,
                        "trends": self._calculate_metric_trends(rolling_metrics),
                        "stability": self._calculate_stability_metrics(rolling_metrics),
                    }

        # Cross-window analysis
        if len(results) > 1:
            results["cross_window_analysis"] = self._analyze_cross_window_patterns(results)

        # Benchmark comparison if available
        if benchmark_returns is not None:
            results["relative_performance"] = self._analyze_relative_performance(
                returns, benchmark_returns
            )

        return results

    def detect_market_regimes(
        self,
        returns: pd.Series,
        regime_indicators: dict[str, pd.Series] = None,
        method: str = "volatility_based",
    ) -> pd.DataFrame:
        """
        Create market regime detection and performance attribution by regime.

        Args:
            returns: Portfolio returns series
            regime_indicators: Optional external regime indicators
            method: Regime detection method ("volatility_based", "return_based", "external")

        Returns:
            DataFrame with regime classifications and performance by regime
        """
        if method == "volatility_based":
            regimes = self._detect_volatility_regimes(returns)
        elif method == "return_based":
            regimes = self._detect_return_regimes(returns)
        elif method == "external" and regime_indicators:
            regimes = self._use_external_regimes(regime_indicators)
        else:
            # Default to volatility-based
            regimes = self._detect_volatility_regimes(returns)

        # Calculate performance by regime
        regime_performance = self._calculate_regime_performance(returns, regimes)

        # Combine results
        result_df = pd.DataFrame(
            {
                "returns": returns,
                "regime": regimes,
            }
        )

        result_df = result_df.dropna()

        return result_df, regime_performance

    def calculate_performance_stability(
        self, returns: pd.Series, benchmark_returns: pd.Series = None
    ) -> dict[str, float]:
        """
        Implement performance stability analysis across time periods.

        Args:
            returns: Portfolio returns series
            benchmark_returns: Optional benchmark returns

        Returns:
            Dictionary containing stability metrics
        """
        stability_metrics = {}

        # Calculate rolling metrics for stability analysis
        rolling_sharpe = self._calculate_rolling_sharpe(returns, 252)
        rolling_volatility = returns.rolling(252).std() * np.sqrt(252)
        rolling_returns = returns.rolling(252).mean() * 252

        # Stability of key metrics
        if not rolling_sharpe.empty:
            stability_metrics["sharpe_stability"] = (
                1 / rolling_sharpe.std() if rolling_sharpe.std() > 0 else 0.0
            )
            stability_metrics["sharpe_trend_slope"] = self._calculate_trend_slope(rolling_sharpe)

        if not rolling_volatility.empty:
            stability_metrics["volatility_stability"] = (
                1 / rolling_volatility.std() if rolling_volatility.std() > 0 else 0.0
            )
            stability_metrics["volatility_trend_slope"] = self._calculate_trend_slope(
                rolling_volatility
            )

        if not rolling_returns.empty:
            stability_metrics["return_stability"] = (
                1 / rolling_returns.std() if rolling_returns.std() > 0 else 0.0
            )
            stability_metrics["return_trend_slope"] = self._calculate_trend_slope(rolling_returns)

        # Consistency metrics
        if len(returns) > 252:
            # Annual consistency (positive returns each year)
            annual_returns = returns.resample("Y").apply(lambda x: (1 + x).prod() - 1)
            stability_metrics["annual_consistency"] = (annual_returns > 0).mean()

            # Monthly consistency
            monthly_returns = returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
            stability_metrics["monthly_win_rate"] = (monthly_returns > 0).mean()

        # Relative stability if benchmark provided
        if benchmark_returns is not None:
            excess_returns = returns - benchmark_returns
            rolling_excess_returns = excess_returns.rolling(252).mean() * 252

            if not rolling_excess_returns.empty:
                stability_metrics["excess_return_stability"] = (
                    1 / rolling_excess_returns.std() if rolling_excess_returns.std() > 0 else 0.0
                )
                stability_metrics["outperformance_consistency"] = (
                    rolling_excess_returns > 0
                ).mean()

        return stability_metrics

    def generate_rolling_analysis_report(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series = None,
        regime_indicators: dict[str, pd.Series] = None,
    ) -> dict[str, Any]:
        """
        Generate comprehensive rolling performance analysis report.

        Args:
            returns: Portfolio returns series
            benchmark_returns: Optional benchmark returns
            regime_indicators: Optional regime indicator data

        Returns:
            Comprehensive rolling analysis report
        """
        report = {}

        # Time-varying performance analysis
        report["time_varying_performance"] = self.calculate_time_varying_performance(
            returns, benchmark_returns
        )

        # Market regime analysis
        regime_df, regime_performance = self.detect_market_regimes(returns, regime_indicators)
        report["market_regimes"] = {
            "regime_classification": regime_df,
            "regime_performance": regime_performance,
        }

        # Stability analysis
        report["stability_analysis"] = self.calculate_performance_stability(
            returns, benchmark_returns
        )

        # Summary insights
        report["summary_insights"] = self._generate_summary_insights(report)

        return report

    def _calculate_rolling_sharpe(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        risk_free_daily = 0.02 / 252  # 2% annual risk-free rate
        excess_returns = returns - risk_free_daily
        rolling_mean = excess_returns.rolling(window).mean()
        rolling_std = excess_returns.rolling(window).std()
        rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)
        return rolling_sharpe.dropna()

    def _calculate_metric_trends(self, rolling_metrics: pd.DataFrame) -> dict[str, float]:
        """Calculate trends in rolling metrics."""
        trends = {}

        for column in rolling_metrics.columns:
            if rolling_metrics[column].notna().any():
                trend_slope = self._calculate_trend_slope(rolling_metrics[column])
                trends[f"{column}_trend"] = trend_slope

        return trends

    def _calculate_trend_slope(self, series: pd.Series) -> float:
        """Calculate trend slope using linear regression."""
        clean_series = series.dropna()
        if len(clean_series) < 2:
            return 0.0

        x = np.arange(len(clean_series))
        slope, _, _, _, _ = stats.linregress(x, clean_series.values)
        return slope

    def _calculate_stability_metrics(self, rolling_metrics: pd.DataFrame) -> dict[str, float]:
        """Calculate stability metrics for rolling performance."""
        stability = {}

        for column in rolling_metrics.columns:
            series = rolling_metrics[column].dropna()
            if len(series) > 1:
                # Coefficient of variation as stability measure
                stability[f"{column}_stability"] = (
                    1 / (series.std() / abs(series.mean())) if series.mean() != 0 else 0.0
                )

                # Consistency (percentage of positive trend periods)
                if "return" in column.lower():
                    stability[f"{column}_consistency"] = (series > 0).mean()

        return stability

    def _analyze_cross_window_patterns(self, window_results: dict[str, Any]) -> dict[str, Any]:
        """Analyze patterns across different window sizes."""
        analysis = {}

        # Compare stability across window sizes
        stability_comparison = {}
        for window_name, window_data in window_results.items():
            if isinstance(window_data, dict) and "stability" in window_data:
                stability_comparison[window_name] = window_data["stability"]

        analysis["stability_by_window"] = stability_comparison

        # Find most stable window size
        if stability_comparison:
            # Use Sharpe stability as primary metric
            sharpe_stabilities = {
                window: data.get("sharpe_ratio_stability", 0.0)
                for window, data in stability_comparison.items()
            }
            most_stable_window = max(sharpe_stabilities.keys(), key=sharpe_stabilities.get)
            analysis["most_stable_window"] = most_stable_window

        return analysis

    def _analyze_relative_performance(
        self, returns: pd.Series, benchmark_returns: pd.Series
    ) -> dict[str, Any]:
        """Analyze performance relative to benchmark across time."""
        # Align returns
        aligned_data = pd.DataFrame({"portfolio": returns, "benchmark": benchmark_returns}).dropna()

        if aligned_data.empty:
            return {}

        excess_returns = aligned_data["portfolio"] - aligned_data["benchmark"]

        # Rolling relative performance
        rolling_excess = excess_returns.rolling(252).mean() * 252
        rolling_tracking_error = excess_returns.rolling(252).std() * np.sqrt(252)
        rolling_info_ratio = rolling_excess / rolling_tracking_error

        return {
            "excess_returns": excess_returns,
            "rolling_excess_returns": rolling_excess.dropna(),
            "rolling_tracking_error": rolling_tracking_error.dropna(),
            "rolling_information_ratio": rolling_info_ratio.dropna(),
            "outperformance_rate": (excess_returns > 0).mean(),
        }

    def _detect_volatility_regimes(
        self, returns: pd.Series, threshold_percentile: float = 75
    ) -> pd.Series:
        """Detect market regimes based on volatility."""
        # Calculate rolling volatility
        rolling_vol = returns.rolling(self.config.regime_detection_lookback).std() * np.sqrt(252)

        # Define regime threshold
        vol_threshold = rolling_vol.quantile(threshold_percentile / 100)

        # Classify regimes
        regimes = pd.Series(index=returns.index, dtype="object")
        regimes[rolling_vol <= vol_threshold] = "Low Volatility"
        regimes[rolling_vol > vol_threshold] = "High Volatility"
        regimes = regimes.fillna("Unknown")

        return regimes

    def _detect_return_regimes(self, returns: pd.Series) -> pd.Series:
        """Detect market regimes based on return patterns."""
        # Calculate rolling returns
        rolling_returns = returns.rolling(self.config.regime_detection_lookback).mean() * 252

        # Define regimes based on return quantiles
        regimes = pd.Series(index=returns.index, dtype="object")
        regimes[rolling_returns <= rolling_returns.quantile(0.33)] = "Bear Market"
        regimes[rolling_returns > rolling_returns.quantile(0.67)] = "Bull Market"
        regimes[
            (rolling_returns > rolling_returns.quantile(0.33))
            & (rolling_returns <= rolling_returns.quantile(0.67))
        ] = "Neutral Market"
        regimes = regimes.fillna("Unknown")

        return regimes

    def _use_external_regimes(self, regime_indicators: dict[str, pd.Series]) -> pd.Series:
        """Use external regime indicators."""
        # For now, use the first indicator provided
        first_indicator = list(regime_indicators.values())[0]
        return first_indicator

    def _calculate_regime_performance(
        self, returns: pd.Series, regimes: pd.Series
    ) -> dict[str, dict[str, float]]:
        """Calculate performance metrics by market regime."""
        regime_performance = {}

        for regime in regimes.unique():
            if regime != "Unknown":
                regime_returns = returns[regimes == regime]

                if len(regime_returns) > 0:
                    # Basic performance metrics
                    basic_metrics = self.return_analyzer.calculate_basic_metrics(regime_returns)
                    risk_metrics = self.risk_analyzer.calculate_comprehensive_risk_metrics(
                        regime_returns
                    )

                    regime_performance[regime] = {
                        "num_observations": len(regime_returns),
                        "frequency": len(regime_returns) / len(returns),
                        **basic_metrics,
                        "win_rate": (regime_returns > 0).mean(),
                        "avg_positive_return": (
                            regime_returns[regime_returns > 0].mean()
                            if (regime_returns > 0).any()
                            else 0.0
                        ),
                        "avg_negative_return": (
                            regime_returns[regime_returns < 0].mean()
                            if (regime_returns < 0).any()
                            else 0.0
                        ),
                        "downside_deviation": risk_metrics.get("downside_deviation", 0.0),
                    }

        return regime_performance

    def _generate_summary_insights(self, report: dict[str, Any]) -> dict[str, str]:
        """Generate summary insights from rolling analysis."""
        insights = {}

        # Stability insights
        if "stability_analysis" in report:
            stability = report["stability_analysis"]

            # Overall stability assessment
            sharpe_stability = stability.get("sharpe_stability", 0.0)
            if sharpe_stability > 2.0:
                insights["stability_assessment"] = "High performance stability across time periods"
            elif sharpe_stability > 1.0:
                insights["stability_assessment"] = "Moderate performance stability"
            else:
                insights["stability_assessment"] = "Low performance stability - high variability"

            # Consistency assessment
            monthly_win_rate = stability.get("monthly_win_rate", 0.5)
            if monthly_win_rate > 0.6:
                insights["consistency_assessment"] = "Strong monthly consistency"
            elif monthly_win_rate > 0.5:
                insights["consistency_assessment"] = "Moderate monthly consistency"
            else:
                insights["consistency_assessment"] = "Low monthly consistency"

        # Regime performance insights
        if "market_regimes" in report and "regime_performance" in report["market_regimes"]:
            regime_perf = report["market_regimes"]["regime_performance"]

            # Find best performing regime
            if regime_perf:
                best_regime = max(
                    regime_perf.keys(), key=lambda x: regime_perf[x].get("annualized_return", 0.0)
                )
                insights["best_regime"] = f"Best performance in {best_regime} conditions"

        return insights
