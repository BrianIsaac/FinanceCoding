"""
Return-based performance metrics for portfolio evaluation.

This module provides comprehensive return analytics including
risk-adjusted returns, drawdown analysis, and benchmark comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ReturnMetricsConfig:
    """Configuration for return metrics calculation."""

    risk_free_rate: float = 0.02  # 2% annual
    trading_days_per_year: int = 252
    confidence_level: float = 0.95
    benchmark_ticker: str = "SPY"


class ReturnAnalyzer:
    """
    Analyzer for return-based portfolio performance metrics.

    Provides comprehensive performance analytics including
    risk-adjusted returns, volatility measures, and tail risk metrics.
    """

    def __init__(self, config: ReturnMetricsConfig):
        """
        Initialize return analyzer.

        Args:
            config: Configuration for metrics calculation
        """
        self.config = config

    def calculate_basic_metrics(
        self, returns: pd.Series, benchmark_returns: pd.Series | None = None
    ) -> dict[str, float]:
        """
        Calculate basic return and risk metrics.

        Args:
            returns: Portfolio returns time series
            benchmark_returns: Benchmark returns for comparison

        Returns:
            Dictionary containing annualized return, volatility,
            Sharpe ratio, and other basic metrics
        """
        if returns.empty:
            return self._empty_metrics()

        # Calculate basic statistics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (self.config.trading_days_per_year / len(returns)) - 1

        # Volatility (annualized)
        annualized_volatility = returns.std() * np.sqrt(self.config.trading_days_per_year)

        # Sharpe ratio
        excess_returns = returns - (self.config.risk_free_rate / self.config.trading_days_per_year)
        sharpe_ratio = (
            np.sqrt(self.config.trading_days_per_year) * excess_returns.mean() / excess_returns.std()
            if excess_returns.std() > 0 else 0.0
        )

        # Maximum drawdown
        max_dd, dd_start, dd_end = self.calculate_maximum_drawdown(returns)

        # Calmar ratio (annual return / max drawdown)
        calmar_ratio = annualized_return / abs(max_dd) if max_dd != 0 else 0.0

        # Higher moments
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # Value at Risk and Conditional VaR
        var_95 = returns.quantile(1 - self.config.confidence_level)
        tail_returns = returns[returns <= var_95]
        cvar_95 = tail_returns.mean() if len(tail_returns) > 0 else var_95

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar_ratio,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "var_95": var_95,
            "cvar_95": cvar_95,
        }

    def calculate_risk_metrics(self, returns: pd.Series) -> dict[str, float]:
        """
        Calculate risk-specific metrics.

        Args:
            returns: Portfolio returns time series

        Returns:
            Dictionary containing VaR, CVaR, maximum drawdown,
            and other risk metrics

        Note:
            This is a stub implementation. Risk metrics calculation
            will be implemented in future stories.
        """
        # Stub implementation
        return {
            "value_at_risk": 0.0,
            "conditional_var": 0.0,
            "maximum_drawdown": 0.0,
            "drawdown_duration": 0.0,
            "downside_deviation": 0.0,
            "sortino_ratio": 0.0,
        }

    def calculate_benchmark_comparison(
        self, returns: pd.Series, benchmark_returns: pd.Series
    ) -> dict[str, float]:
        """
        Calculate benchmark comparison metrics.

        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns

        Returns:
            Dictionary containing alpha, beta, information ratio,
            and tracking error

        Note:
            This is a stub implementation. Benchmark comparison
            will be implemented in future stories.
        """
        # Stub implementation
        return {
            "alpha": 0.0,
            "beta": 0.0,
            "information_ratio": 0.0,
            "tracking_error": 0.0,
            "correlation": 0.0,
        }

    def calculate_maximum_drawdown(self, returns: pd.Series) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
        """
        Calculate maximum drawdown with start and end dates.

        Args:
            returns: Portfolio returns time series

        Returns:
            Tuple of (max_drawdown, peak_date, trough_date)
        """
        if returns.empty:
            return 0.0, None, None

        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()

        # Calculate running maximum (peaks)
        running_max = cumulative.expanding().max()

        # Calculate drawdown series
        drawdown = (cumulative - running_max) / running_max

        # Find maximum drawdown
        max_drawdown = drawdown.min()

        if pd.isna(max_drawdown) or max_drawdown == 0:
            return 0.0, None, None

        # Find the date of maximum drawdown (trough)
        max_dd_date = drawdown.idxmin()

        # Find the start of the drawdown period (peak before trough)
        peak_date = running_max.loc[:max_dd_date].idxmax()

        return max_drawdown, peak_date, max_dd_date

    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float | None = None) -> float:
        """
        Calculate Sharpe ratio with proper annualization.

        Args:
            returns: Portfolio returns time series
            risk_free_rate: Risk-free rate (uses config default if None)

        Returns:
            Annualized Sharpe ratio
        """
        if returns.empty or returns.std() == 0:
            return 0.0

        rf_rate = risk_free_rate if risk_free_rate is not None else self.config.risk_free_rate
        daily_rf_rate = rf_rate / self.config.trading_days_per_year

        excess_returns = returns - daily_rf_rate

        return np.sqrt(self.config.trading_days_per_year) * excess_returns.mean() / excess_returns.std()

    def _empty_metrics(self) -> dict[str, float]:
        """Return empty metrics dictionary for edge cases."""
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "var_95": 0.0,
            "cvar_95": 0.0,
        }
