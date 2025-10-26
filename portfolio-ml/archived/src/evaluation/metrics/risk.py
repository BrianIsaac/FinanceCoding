"""
Advanced risk analytics module for portfolio evaluation.

This module provides comprehensive risk metrics including tracking error,
Information Ratio, downside deviation, and win rate analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class RiskMetricsConfig:
    """Configuration for risk metrics calculation."""

    risk_free_rate: float = 0.02
    trading_days_per_year: int = 252
    confidence_levels: list[float] = None

    def __post_init__(self):
        """Set default confidence levels if not provided."""
        if self.confidence_levels is None:
            self.confidence_levels = [0.90, 0.95, 0.99]


class RiskAnalytics:
    """
    Comprehensive risk analytics for portfolio returns.

    Provides advanced risk metrics including tracking error, Information Ratio,
    downside deviation, and multi-horizon win rate analysis.
    """

    def __init__(self, config: RiskMetricsConfig = None):
        """
        Initialize risk analytics.

        Args:
            config: Configuration for risk metrics calculation
        """
        self.config = config or RiskMetricsConfig()

    def calculate_tracking_error(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ) -> float:
        """
        Calculate annualized tracking error vs benchmark.

        Args:
            portfolio_returns: Portfolio daily returns
            benchmark_returns: Benchmark daily returns

        Returns:
            Annualized tracking error
        """
        if len(portfolio_returns) != len(benchmark_returns):
            aligned_data = pd.DataFrame(
                {"portfolio": portfolio_returns, "benchmark": benchmark_returns}
            ).dropna()

            if aligned_data.empty:
                return 0.0

            portfolio_returns = aligned_data["portfolio"]
            benchmark_returns = aligned_data["benchmark"]

        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(self.config.trading_days_per_year)

        return tracking_error

    def calculate_information_ratio(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ) -> float:
        """
        Calculate Information Ratio with proper active return analysis.

        Args:
            portfolio_returns: Portfolio daily returns
            benchmark_returns: Benchmark daily returns

        Returns:
            Information Ratio (active return / tracking error)
        """
        if len(portfolio_returns) != len(benchmark_returns):
            aligned_data = pd.DataFrame(
                {"portfolio": portfolio_returns, "benchmark": benchmark_returns}
            ).dropna()

            if aligned_data.empty:
                return 0.0

            portfolio_returns = aligned_data["portfolio"]
            benchmark_returns = aligned_data["benchmark"]

        excess_returns = portfolio_returns - benchmark_returns
        active_return = excess_returns.mean() * self.config.trading_days_per_year
        tracking_error = excess_returns.std() * np.sqrt(self.config.trading_days_per_year)

        if tracking_error == 0:
            return 0.0

        return active_return / tracking_error

    def calculate_win_rate_analysis(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series = None
    ) -> dict[str, float]:
        """
        Calculate win rate analysis for monthly, quarterly, and annual periods.

        Args:
            portfolio_returns: Portfolio daily returns
            benchmark_returns: Optional benchmark returns for relative win rate

        Returns:
            Dictionary containing win rates for different periods
        """
        # Convert daily returns to different periods
        portfolio_monthly = self._resample_returns(portfolio_returns, "M")
        portfolio_quarterly = self._resample_returns(portfolio_returns, "Q")
        portfolio_annual = self._resample_returns(portfolio_returns, "A")

        results = {}

        # Absolute win rates (positive returns)
        results["monthly_win_rate"] = (portfolio_monthly > 0).mean()
        results["quarterly_win_rate"] = (portfolio_quarterly > 0).mean()
        results["annual_win_rate"] = (portfolio_annual > 0).mean()

        # Relative win rates if benchmark provided
        if benchmark_returns is not None:
            benchmark_monthly = self._resample_returns(benchmark_returns, "M")
            benchmark_quarterly = self._resample_returns(benchmark_returns, "Q")
            benchmark_annual = self._resample_returns(benchmark_returns, "A")

            # Align periods
            monthly_excess = self._align_returns(portfolio_monthly, benchmark_monthly)
            quarterly_excess = self._align_returns(portfolio_quarterly, benchmark_quarterly)
            annual_excess = self._align_returns(portfolio_annual, benchmark_annual)

            if not monthly_excess.empty:
                results["monthly_outperform_rate"] = (monthly_excess > 0).mean()
            if not quarterly_excess.empty:
                results["quarterly_outperform_rate"] = (quarterly_excess > 0).mean()
            if not annual_excess.empty:
                results["annual_outperform_rate"] = (annual_excess > 0).mean()

        return results

    def calculate_downside_deviation(
        self, returns: pd.Series, minimum_acceptable_return: float = 0.0
    ) -> float:
        """
        Calculate downside deviation and semi-variance risk measures.

        Args:
            returns: Daily returns series
            minimum_acceptable_return: MAR for downside calculation (default 0)

        Returns:
            Annualized downside deviation
        """
        # Convert MAR to daily if it's annual
        daily_mar = minimum_acceptable_return / self.config.trading_days_per_year

        # Calculate downside returns
        downside_returns = returns - daily_mar
        downside_returns = downside_returns[downside_returns < 0]

        if len(downside_returns) == 0:
            return 0.0

        # Calculate downside deviation
        downside_deviation = np.sqrt((downside_returns**2).mean())

        # Annualize
        return downside_deviation * np.sqrt(self.config.trading_days_per_year)

    def calculate_sortino_ratio(
        self, returns: pd.Series, minimum_acceptable_return: float = 0.0
    ) -> float:
        """
        Calculate Sortino ratio using downside deviation.

        Args:
            returns: Daily returns series
            minimum_acceptable_return: MAR for calculation (annual rate)

        Returns:
            Annualized Sortino ratio
        """
        # Calculate excess return over MAR
        daily_mar = minimum_acceptable_return / self.config.trading_days_per_year
        excess_return = returns.mean() - daily_mar
        annualized_excess = excess_return * self.config.trading_days_per_year

        # Calculate downside deviation
        downside_dev = self.calculate_downside_deviation(returns, minimum_acceptable_return)

        if downside_dev == 0:
            return 0.0

        return annualized_excess / downside_dev

    def calculate_comprehensive_risk_metrics(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series = None
    ) -> dict[str, Any]:
        """
        Calculate comprehensive risk metrics for portfolio.

        Args:
            portfolio_returns: Portfolio daily returns
            benchmark_returns: Optional benchmark returns

        Returns:
            Dictionary containing all risk metrics
        """
        metrics = {}

        # Downside risk metrics
        metrics["downside_deviation"] = self.calculate_downside_deviation(portfolio_returns)
        metrics["sortino_ratio"] = self.calculate_sortino_ratio(portfolio_returns)

        # Value at Risk for multiple confidence levels
        var_metrics = {}
        for conf_level in self.config.confidence_levels:
            var_value = portfolio_returns.quantile(1 - conf_level)
            cvar_value = portfolio_returns[portfolio_returns <= var_value].mean()
            var_metrics[f"var_{int(conf_level*100)}"] = var_value
            var_metrics[f"cvar_{int(conf_level*100)}"] = cvar_value

        metrics["var_metrics"] = var_metrics

        # Win rate analysis
        win_rates = self.calculate_win_rate_analysis(portfolio_returns, benchmark_returns)
        metrics["win_rates"] = win_rates

        # Benchmark-relative metrics if available
        if benchmark_returns is not None:
            metrics["tracking_error"] = self.calculate_tracking_error(
                portfolio_returns, benchmark_returns
            )
            metrics["information_ratio"] = self.calculate_information_ratio(
                portfolio_returns, benchmark_returns
            )

            # Beta calculation
            aligned_data = pd.DataFrame(
                {"portfolio": portfolio_returns, "benchmark": benchmark_returns}
            ).dropna()

            if len(aligned_data) > 1:
                covariance = aligned_data["portfolio"].cov(aligned_data["benchmark"])
                benchmark_variance = aligned_data["benchmark"].var()
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0.0
                metrics["beta"] = beta

                # Correlation
                correlation = aligned_data["portfolio"].corr(aligned_data["benchmark"])
                metrics["correlation"] = correlation

        return metrics

    def _resample_returns(self, returns: pd.Series, freq: str) -> pd.Series:
        """Resample daily returns to specified frequency."""
        if not isinstance(returns.index, pd.DatetimeIndex):
            # If no datetime index, create one assuming daily frequency
            returns = returns.copy()
            returns.index = pd.date_range(start="2020-01-01", periods=len(returns), freq="D")

        # Calculate period returns (compound daily returns)
        return (1 + returns).resample(freq).prod() - 1

    def _align_returns(self, returns1: pd.Series, returns2: pd.Series) -> pd.Series:
        """Align two return series and calculate excess returns."""
        aligned_data = pd.DataFrame({"ret1": returns1, "ret2": returns2}).dropna()

        if aligned_data.empty:
            return pd.Series(dtype=float)

        return aligned_data["ret1"] - aligned_data["ret2"]
