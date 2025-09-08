"""
Comprehensive benchmark comparison framework for portfolio evaluation.

This module provides comprehensive benchmark comparison including
equal-weight baselines, mean-variance optimization benchmarks,
S&P MidCap 400 index tracking, and multi-benchmark ranking analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .attribution import AttributionConfig, PerformanceAttributionAnalyzer
from .returns import ReturnAnalyzer, ReturnMetricsConfig
from .risk import RiskAnalytics, RiskMetricsConfig


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark comparison analysis."""

    risk_free_rate: float = 0.02
    trading_days_per_year: int = 252
    rebalance_frequency: str = "M"  # Monthly rebalancing for constructed benchmarks
    min_weight: float = 0.01  # Minimum 1% weight for equal-weight benchmark
    max_weight: float = 0.20  # Maximum 20% weight for mean-variance benchmark
    target_volatility: float = 0.12  # 12% target volatility for risk-parity benchmarks


class BenchmarkComparator:
    """
    Comprehensive benchmark comparison framework.

    Integrates equal-weight baseline comparison, mean-variance optimization
    benchmarks, S&P MidCap 400 index tracking, and multi-benchmark performance
    ranking and analysis.
    """

    def __init__(self, config: BenchmarkConfig = None):
        """
        Initialize benchmark comparator.

        Args:
            config: Configuration for benchmark comparison
        """
        self.config = config or BenchmarkConfig()
        self.return_analyzer = ReturnAnalyzer(
            ReturnMetricsConfig(
                risk_free_rate=self.config.risk_free_rate,
                trading_days_per_year=self.config.trading_days_per_year,
            )
        )
        self.risk_analyzer = RiskAnalytics(
            RiskMetricsConfig(
                risk_free_rate=self.config.risk_free_rate,
                trading_days_per_year=self.config.trading_days_per_year,
            )
        )
        self.attribution_analyzer = PerformanceAttributionAnalyzer(AttributionConfig())

    def create_equal_weight_benchmark(
        self, asset_universe: pd.DataFrame, asset_returns: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Integrate equal-weight baseline comparison with existing framework.

        Args:
            asset_universe: DataFrame indicating which assets are available at each date
            asset_returns: Asset returns data

        Returns:
            Dictionary containing equal-weight benchmark data and performance
        """
        if asset_universe.empty or asset_returns.empty:
            return {"error": "Missing required data for equal-weight benchmark"}

        # Align data
        common_dates = asset_universe.index.intersection(asset_returns.index)
        if common_dates.empty:
            return {"error": "No common dates between universe and returns data"}

        universe_aligned = asset_universe.loc[common_dates]
        returns_aligned = asset_returns.loc[common_dates]

        # Calculate equal-weight portfolio
        equal_weight_returns = []
        equal_weight_weights = []

        for date in common_dates:
            # Get available assets for this date
            available_assets = universe_aligned.loc[date]
            if isinstance(available_assets, pd.Series):
                available_assets = available_assets[available_assets == True]
            else:
                available_assets = available_assets.dropna()

            if len(available_assets) == 0:
                continue

            # Calculate equal weights
            num_assets = len(available_assets)
            equal_weight = 1.0 / num_assets

            # Get returns for available assets
            date_returns = returns_aligned.loc[date, available_assets.index]
            date_returns = date_returns.dropna()

            if len(date_returns) > 0:
                # Calculate portfolio return
                portfolio_return = (date_returns * equal_weight).sum()
                equal_weight_returns.append(portfolio_return)

                # Store weights
                weights_dict = dict.fromkeys(date_returns.index, equal_weight)
                equal_weight_weights.append(weights_dict)

        if not equal_weight_returns:
            return {"error": "No valid equal-weight returns calculated"}

        # Create return series
        equal_weight_series = pd.Series(
            equal_weight_returns, index=common_dates[: len(equal_weight_returns)]
        )

        # Calculate performance metrics
        performance_metrics = self.return_analyzer.calculate_basic_metrics(equal_weight_series)
        risk_metrics = self.risk_analyzer.calculate_comprehensive_risk_metrics(equal_weight_series)

        return {
            "returns": equal_weight_series,
            "weights": equal_weight_weights,
            "performance_metrics": performance_metrics,
            "risk_metrics": risk_metrics,
            "num_assets_avg": np.mean([len(w) for w in equal_weight_weights]),
            "rebalancing_dates": common_dates[: len(equal_weight_returns)],
        }

    def create_mean_variance_benchmark(
        self, asset_returns: pd.DataFrame, lookback_window: int = 252, rebalance_frequency: int = 21
    ) -> dict[str, Any]:
        """
        Implement mean-variance optimization benchmark.

        Args:
            asset_returns: Asset returns data
            lookback_window: Lookback window for optimization (trading days)
            rebalance_frequency: Rebalancing frequency (trading days)

        Returns:
            Dictionary containing mean-variance benchmark data and performance
        """
        if asset_returns.empty or len(asset_returns) < lookback_window:
            return {"error": "Insufficient data for mean-variance optimization"}

        mv_returns = []
        mv_weights = []
        rebalance_dates = []

        # Iterate through rebalancing dates
        for i in range(lookback_window, len(asset_returns), rebalance_frequency):
            # Get lookback data
            start_idx = i - lookback_window
            lookback_data = asset_returns.iloc[start_idx:i].dropna(axis=1)

            if len(lookback_data.columns) < 2:
                continue

            # Calculate expected returns and covariance
            expected_returns = lookback_data.mean() * self.config.trading_days_per_year
            cov_matrix = lookback_data.cov() * self.config.trading_days_per_year

            # Optimize portfolio
            optimal_weights = self._optimize_mean_variance_portfolio(expected_returns, cov_matrix)

            if optimal_weights is not None:
                # Calculate returns for the next period
                next_period_start = i
                next_period_end = min(i + rebalance_frequency, len(asset_returns))

                for j in range(next_period_start, next_period_end):
                    if j < len(asset_returns):
                        date_returns = asset_returns.iloc[j]
                        aligned_returns = date_returns.reindex(optimal_weights.index, fill_value=0)
                        portfolio_return = (optimal_weights * aligned_returns).sum()

                        mv_returns.append(portfolio_return)
                        mv_weights.append(optimal_weights.to_dict())
                        rebalance_dates.append(asset_returns.index[j])

        if not mv_returns:
            return {"error": "No valid mean-variance returns calculated"}

        # Create return series
        mv_series = pd.Series(mv_returns, index=rebalance_dates)

        # Calculate performance metrics
        performance_metrics = self.return_analyzer.calculate_basic_metrics(mv_series)
        risk_metrics = self.risk_analyzer.calculate_comprehensive_risk_metrics(mv_series)

        return {
            "returns": mv_series,
            "weights": mv_weights,
            "performance_metrics": performance_metrics,
            "risk_metrics": risk_metrics,
            "optimization_params": {
                "lookback_window": lookback_window,
                "rebalance_frequency": rebalance_frequency,
                "min_weight": self.config.min_weight,
                "max_weight": self.config.max_weight,
            },
        }

    def integrate_sp_midcap_tracking(
        self, sp_midcap_returns: pd.Series, portfolio_returns: pd.Series
    ) -> dict[str, Any]:
        """
        Create S&P MidCap 400 index tracking and comparison.

        Args:
            sp_midcap_returns: S&P MidCap 400 index returns
            portfolio_returns: Portfolio returns to compare

        Returns:
            Dictionary containing S&P MidCap 400 comparison metrics
        """
        if sp_midcap_returns.empty or portfolio_returns.empty:
            return {"error": "Missing required return data"}

        # Align data
        aligned_data = pd.DataFrame(
            {"portfolio": portfolio_returns, "sp_midcap": sp_midcap_returns}
        ).dropna()

        if aligned_data.empty:
            return {"error": "No overlapping data between portfolio and S&P MidCap 400"}

        portfolio_aligned = aligned_data["portfolio"]
        benchmark_aligned = aligned_data["sp_midcap"]

        # Calculate tracking metrics
        tracking_error = self.risk_analyzer.calculate_tracking_error(
            portfolio_aligned, benchmark_aligned
        )
        information_ratio = self.risk_analyzer.calculate_information_ratio(
            portfolio_aligned, benchmark_aligned
        )

        # Performance comparison
        portfolio_metrics = self.return_analyzer.calculate_basic_metrics(portfolio_aligned)
        benchmark_metrics = self.return_analyzer.calculate_basic_metrics(benchmark_aligned)

        # Attribution analysis
        attribution_results = self.attribution_analyzer.decompose_alpha_beta(
            portfolio_aligned, benchmark_aligned
        )

        # Rolling comparison
        rolling_comparison = self._calculate_rolling_comparison(
            portfolio_aligned, benchmark_aligned
        )

        return {
            "tracking_metrics": {
                "tracking_error": tracking_error,
                "information_ratio": information_ratio,
                "correlation": portfolio_aligned.corr(benchmark_aligned),
            },
            "performance_comparison": {
                "portfolio": portfolio_metrics,
                "sp_midcap_400": benchmark_metrics,
                "excess_return": portfolio_metrics["annualized_return"]
                - benchmark_metrics["annualized_return"],
                "excess_volatility": portfolio_metrics["annualized_volatility"]
                - benchmark_metrics["annualized_volatility"],
            },
            "attribution": attribution_results,
            "rolling_comparison": rolling_comparison,
            "aligned_returns": aligned_data,
        }

    def create_multi_benchmark_ranking(
        self, portfolio_returns: pd.Series, benchmarks: dict[str, pd.Series]
    ) -> dict[str, Any]:
        """
        Add multi-benchmark performance ranking and analysis.

        Args:
            portfolio_returns: Portfolio returns to rank
            benchmarks: Dictionary of benchmark name -> return series

        Returns:
            Dictionary containing multi-benchmark ranking and analysis
        """
        if not benchmarks:
            return {"error": "No benchmarks provided for comparison"}

        # Combine all return series
        all_returns = {"Portfolio": portfolio_returns}
        all_returns.update(benchmarks)

        # Align all data
        combined_data = pd.DataFrame(all_returns).dropna()

        if combined_data.empty:
            return {"error": "No common data across portfolio and benchmarks"}

        # Calculate performance metrics for all
        performance_comparison = {}
        risk_comparison = {}

        for name, returns in combined_data.items():
            perf_metrics = self.return_analyzer.calculate_basic_metrics(returns)
            risk_metrics = self.risk_analyzer.calculate_comprehensive_risk_metrics(returns)

            performance_comparison[name] = perf_metrics
            risk_comparison[name] = risk_metrics

        # Create rankings
        rankings = self._create_performance_rankings(performance_comparison)

        # Risk-adjusted rankings
        risk_adjusted_rankings = self._create_risk_adjusted_rankings(performance_comparison)

        # Pairwise comparisons
        pairwise_comparisons = self._calculate_pairwise_comparisons(combined_data)

        # Statistical significance tests
        significance_tests = self._perform_significance_tests(combined_data)

        return {
            "performance_metrics": performance_comparison,
            "risk_metrics": risk_comparison,
            "rankings": rankings,
            "risk_adjusted_rankings": risk_adjusted_rankings,
            "pairwise_comparisons": pairwise_comparisons,
            "significance_tests": significance_tests,
            "summary": self._generate_ranking_summary(rankings, risk_adjusted_rankings),
        }

    def generate_benchmark_report(
        self,
        portfolio_returns: pd.Series,
        asset_returns: pd.DataFrame = None,
        sp_midcap_returns: pd.Series = None,
        custom_benchmarks: dict[str, pd.Series] = None,
    ) -> dict[str, Any]:
        """
        Generate comprehensive benchmark comparison report.

        Args:
            portfolio_returns: Portfolio returns to analyze
            asset_returns: Asset returns for constructing benchmarks
            sp_midcap_returns: S&P MidCap 400 returns
            custom_benchmarks: Additional custom benchmarks

        Returns:
            Comprehensive benchmark comparison report
        """
        report = {}
        benchmarks_created = {}

        # Create equal-weight benchmark if asset data available
        if asset_returns is not None:
            # Create a simple universe (all assets available all the time for simplicity)
            asset_universe = asset_returns.notna()
            equal_weight_result = self.create_equal_weight_benchmark(asset_universe, asset_returns)

            if "error" not in equal_weight_result:
                report["equal_weight_benchmark"] = equal_weight_result
                benchmarks_created["Equal Weight"] = equal_weight_result["returns"]

            # Create mean-variance benchmark
            mv_result = self.create_mean_variance_benchmark(asset_returns)
            if "error" not in mv_result:
                report["mean_variance_benchmark"] = mv_result
                benchmarks_created["Mean Variance"] = mv_result["returns"]

        # S&P MidCap 400 comparison
        if sp_midcap_returns is not None:
            sp_comparison = self.integrate_sp_midcap_tracking(sp_midcap_returns, portfolio_returns)
            if "error" not in sp_comparison:
                report["sp_midcap_comparison"] = sp_comparison
                benchmarks_created["S&P MidCap 400"] = sp_midcap_returns

        # Add custom benchmarks
        if custom_benchmarks:
            benchmarks_created.update(custom_benchmarks)

        # Multi-benchmark ranking
        if benchmarks_created:
            ranking_analysis = self.create_multi_benchmark_ranking(
                portfolio_returns, benchmarks_created
            )
            if "error" not in ranking_analysis:
                report["multi_benchmark_analysis"] = ranking_analysis

        # Summary insights
        report["benchmark_summary"] = self._generate_benchmark_summary(report)

        return report

    def _optimize_mean_variance_portfolio(
        self, expected_returns: pd.Series, cov_matrix: pd.DataFrame
    ) -> pd.Series | None:
        """Optimize mean-variance portfolio using quadratic programming."""
        try:
            n_assets = len(expected_returns)

            # Objective function (negative Sharpe ratio)
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                portfolio_std = np.sqrt(portfolio_variance)
                sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_std
                return -sharpe_ratio  # Minimize negative Sharpe ratio

            # Constraints
            constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]  # Weights sum to 1

            # Bounds (min and max weights)
            bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_assets)]

            # Initial guess (equal weights)
            x0 = np.ones(n_assets) / n_assets

            # Optimize
            result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)

            if result.success:
                return pd.Series(result.x, index=expected_returns.index)
            else:
                return None

        except Exception:
            return None

    def _calculate_rolling_comparison(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series, window: int = 252
    ) -> dict[str, pd.Series]:
        """Calculate rolling comparison metrics."""
        if len(portfolio_returns) < window:
            return {}

        # Rolling excess returns
        excess_returns = portfolio_returns - benchmark_returns
        rolling_excess = excess_returns.rolling(window).mean() * self.config.trading_days_per_year

        # Rolling tracking error
        rolling_tracking_error = excess_returns.rolling(window).std() * np.sqrt(
            self.config.trading_days_per_year
        )

        # Rolling information ratio
        rolling_ir = rolling_excess / rolling_tracking_error

        # Rolling correlation
        rolling_corr = portfolio_returns.rolling(window).corr(benchmark_returns)

        return {
            "rolling_excess_returns": rolling_excess.dropna(),
            "rolling_tracking_error": rolling_tracking_error.dropna(),
            "rolling_information_ratio": rolling_ir.dropna(),
            "rolling_correlation": rolling_corr.dropna(),
        }

    def _create_performance_rankings(self, performance_data: dict[str, dict]) -> dict[str, dict]:
        """Create performance rankings across multiple metrics."""
        metrics_to_rank = [
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "calmar_ratio",
        ]

        rankings = {}

        for metric in metrics_to_rank:
            metric_values = {}
            for name, data in performance_data.items():
                if metric in data:
                    metric_values[name] = data[metric]

            if metric_values:
                # Sort based on whether higher or lower is better
                reverse_sort = metric not in ["annualized_volatility", "max_drawdown"]
                sorted_items = sorted(
                    metric_values.items(), key=lambda x: x[1], reverse=reverse_sort
                )

                rankings[metric] = {
                    "ranked_list": [
                        (name, value, rank + 1) for rank, (name, value) in enumerate(sorted_items)
                    ],
                    "portfolio_rank": next(
                        (
                            rank + 1
                            for rank, (name, _) in enumerate(sorted_items)
                            if name == "Portfolio"
                        ),
                        None,
                    ),
                }

        return rankings

    def _create_risk_adjusted_rankings(self, performance_data: dict[str, dict]) -> dict[str, Any]:
        """Create risk-adjusted performance rankings."""
        risk_adjusted_metrics = ["sharpe_ratio", "calmar_ratio", "sortino_ratio"]

        rankings = {}

        for metric in risk_adjusted_metrics:
            metric_values = {}
            for name, data in performance_data.items():
                if metric in data:
                    metric_values[name] = data[metric]

            if metric_values:
                sorted_items = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
                rankings[metric] = {
                    "ranked_list": [
                        (name, value, rank + 1) for rank, (name, value) in enumerate(sorted_items)
                    ],
                    "portfolio_rank": next(
                        (
                            rank + 1
                            for rank, (name, _) in enumerate(sorted_items)
                            if name == "Portfolio"
                        ),
                        None,
                    ),
                }

        # Overall risk-adjusted score (average of normalized ranks)
        if rankings:
            overall_scores = {}
            num_metrics = len(rankings)

            for name in performance_data.keys():
                total_score = 0
                valid_metrics = 0

                for metric_data in rankings.values():
                    portfolio_rank = next(
                        (rank for n, _, rank in metric_data["ranked_list"] if n == name), None
                    )
                    if portfolio_rank:
                        # Normalize rank (1 = best, convert to score where higher = better)
                        normalized_score = (len(performance_data) - portfolio_rank + 1) / len(
                            performance_data
                        )
                        total_score += normalized_score
                        valid_metrics += 1

                if valid_metrics > 0:
                    overall_scores[name] = total_score / valid_metrics

            if overall_scores:
                sorted_overall = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
                rankings["overall_risk_adjusted"] = {
                    "ranked_list": [
                        (name, score, rank + 1) for rank, (name, score) in enumerate(sorted_overall)
                    ],
                    "portfolio_rank": next(
                        (
                            rank + 1
                            for rank, (name, _) in enumerate(sorted_overall)
                            if name == "Portfolio"
                        ),
                        None,
                    ),
                }

        return rankings

    def _calculate_pairwise_comparisons(self, combined_data: pd.DataFrame) -> dict[str, dict]:
        """Calculate pairwise comparisons between portfolio and each benchmark."""
        comparisons = {}
        portfolio_returns = combined_data["Portfolio"]

        for benchmark_name in combined_data.columns:
            if benchmark_name != "Portfolio":
                benchmark_returns = combined_data[benchmark_name]

                # Excess returns
                excess_returns = portfolio_returns - benchmark_returns

                # Win rate
                win_rate = (excess_returns > 0).mean()

                # Average excess return
                avg_excess = excess_returns.mean() * self.config.trading_days_per_year

                # Volatility of excess returns
                excess_volatility = excess_returns.std() * np.sqrt(
                    self.config.trading_days_per_year
                )

                comparisons[benchmark_name] = {
                    "win_rate": win_rate,
                    "avg_excess_return": avg_excess,
                    "excess_volatility": excess_volatility,
                    "information_ratio": (
                        avg_excess / excess_volatility if excess_volatility > 0 else 0.0
                    ),
                }

        return comparisons

    def _perform_significance_tests(self, combined_data: pd.DataFrame) -> dict[str, dict]:
        """Perform statistical significance tests for performance differences."""
        from scipy.stats import ttest_rel

        significance_tests = {}
        portfolio_returns = combined_data["Portfolio"]

        for benchmark_name in combined_data.columns:
            if benchmark_name != "Portfolio":
                benchmark_returns = combined_data[benchmark_name]

                # Paired t-test for return difference
                t_stat, p_value = ttest_rel(portfolio_returns, benchmark_returns)

                significance_tests[benchmark_name] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant_at_5pct": p_value < 0.05,
                    "significant_at_1pct": p_value < 0.01,
                    "outperforms": t_stat > 0,
                }

        return significance_tests

    def _generate_ranking_summary(
        self, rankings: dict, risk_adjusted_rankings: dict
    ) -> dict[str, str]:
        """Generate summary of ranking results."""
        summary = {}

        # Overall performance rank
        if "overall_risk_adjusted" in risk_adjusted_rankings:
            overall_rank = risk_adjusted_rankings["overall_risk_adjusted"]["portfolio_rank"]
            total_strategies = len(risk_adjusted_rankings["overall_risk_adjusted"]["ranked_list"])

            if overall_rank == 1:
                summary["overall_performance"] = (
                    f"Best performing strategy (1 of {total_strategies})"
                )
            elif overall_rank <= total_strategies // 3:
                summary["overall_performance"] = (
                    f"Top-tier performance ({overall_rank} of {total_strategies})"
                )
            elif overall_rank <= 2 * total_strategies // 3:
                summary["overall_performance"] = (
                    f"Mid-tier performance ({overall_rank} of {total_strategies})"
                )
            else:
                summary["overall_performance"] = (
                    f"Lower-tier performance ({overall_rank} of {total_strategies})"
                )

        # Sharpe ratio rank
        if "sharpe_ratio" in rankings:
            sharpe_rank = rankings["sharpe_ratio"]["portfolio_rank"]
            summary["risk_adjusted_rank"] = f"Sharpe ratio rank: {sharpe_rank}"

        return summary

    def _generate_benchmark_summary(self, report: dict[str, Any]) -> dict[str, str]:
        """Generate overall benchmark comparison summary."""
        summary = {}

        # S&P MidCap 400 performance
        if "sp_midcap_comparison" in report:
            sp_data = report["sp_midcap_comparison"]
            excess_return = sp_data["performance_comparison"]["excess_return"]

            if excess_return > 0.02:  # 2% outperformance
                summary["sp_midcap_performance"] = (
                    f"Strong outperformance vs S&P MidCap 400: {excess_return:.2%}"
                )
            elif excess_return > 0:
                summary["sp_midcap_performance"] = (
                    f"Modest outperformance vs S&P MidCap 400: {excess_return:.2%}"
                )
            else:
                summary["sp_midcap_performance"] = (
                    f"Underperformance vs S&P MidCap 400: {excess_return:.2%}"
                )

        # Multi-benchmark ranking
        if "multi_benchmark_analysis" in report:
            rankings = report["multi_benchmark_analysis"]["risk_adjusted_rankings"]
            if "overall_risk_adjusted" in rankings:
                overall_rank = rankings["overall_risk_adjusted"]["portfolio_rank"]
                summary["overall_benchmark_rank"] = f"Overall benchmark rank: {overall_rank}"

        return summary
