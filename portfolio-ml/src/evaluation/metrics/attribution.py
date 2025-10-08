"""
Performance attribution analysis framework for portfolio evaluation.

This module provides comprehensive performance attribution including
factor-based attribution, alpha vs beta decomposition, sector analysis,
and risk factor exposure tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


@dataclass
class AttributionConfig:
    """Configuration for performance attribution analysis."""

    attribution_window: int = 252  # 1 year window for attribution analysis
    min_observations: int = 63  # Minimum 3 months of data
    confidence_level: float = 0.95
    risk_factors: list[str] = None
    sector_mapping: dict[str, str] = None

    def __post_init__(self):
        """Set default risk factors if not provided."""
        if self.risk_factors is None:
            self.risk_factors = ["market", "size", "value", "momentum", "quality", "volatility"]


class PerformanceAttributionAnalyzer:
    """
    Comprehensive performance attribution analysis system.

    Provides factor-based attribution analysis, alpha vs beta decomposition,
    sector and style attribution, and risk factor exposure tracking.
    """

    def __init__(self, config: AttributionConfig = None):
        """
        Initialize performance attribution analyzer.

        Args:
            config: Configuration for attribution analysis
        """
        self.config = config or AttributionConfig()

    def calculate_factor_based_attribution(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        benchmark_returns: pd.Series = None,
    ) -> dict[str, Any]:
        """
        Implement factor-based attribution analysis framework.

        Args:
            portfolio_returns: Portfolio returns time series
            factor_returns: DataFrame with factor returns (columns = factors)
            benchmark_returns: Optional benchmark returns for relative attribution

        Returns:
            Dictionary containing factor attribution results
        """
        # Align data
        aligned_data = pd.DataFrame({"portfolio": portfolio_returns})

        if benchmark_returns is not None:
            aligned_data["benchmark"] = benchmark_returns
            aligned_data["excess_returns"] = aligned_data["portfolio"] - aligned_data["benchmark"]
            dependent_var = "excess_returns"
        else:
            dependent_var = "portfolio"

        # Add factor returns to aligned data
        for factor in factor_returns.columns:
            if factor in aligned_data.columns:
                continue
            aligned_data[factor] = factor_returns[factor]

        # Remove missing data
        aligned_data = aligned_data.dropna()

        if len(aligned_data) < self.config.min_observations:
            return {"error": "Insufficient data for factor attribution analysis"}

        # Prepare regression data
        y = aligned_data[dependent_var].values
        X = aligned_data[factor_returns.columns].values
        factor_names = factor_returns.columns.tolist()

        # Perform factor regression
        reg_model = LinearRegression(fit_intercept=True)
        reg_model.fit(X, y)

        # Calculate attribution results
        factor_exposures = dict(zip(factor_names, reg_model.coef_))
        alpha = reg_model.intercept_
        r_squared = reg_model.score(X, y)

        # Calculate factor contributions to returns
        factor_contributions = {}
        total_factor_contribution = 0.0

        for i, factor_name in enumerate(factor_names):
            factor_mean_return = aligned_data[factor_name].mean()
            factor_contribution = reg_model.coef_[i] * factor_mean_return
            factor_contributions[factor_name] = {
                "exposure": reg_model.coef_[i],
                "mean_return": factor_mean_return,
                "contribution": factor_contribution,
                "contribution_annualized": factor_contribution * 252,
            }
            total_factor_contribution += factor_contribution

        # Calculate residual statistics
        predicted_returns = reg_model.predict(X)
        residuals = y - predicted_returns

        attribution_results = {
            "factor_exposures": factor_exposures,
            "factor_contributions": factor_contributions,
            "alpha": alpha,
            "alpha_annualized": alpha * 252,
            "r_squared": r_squared,
            "total_factor_contribution": total_factor_contribution,
            "total_factor_contribution_annualized": total_factor_contribution * 252,
            "residual_volatility": np.std(residuals) * np.sqrt(252),
            "information_ratio": (
                (alpha / np.std(residuals)) * np.sqrt(252) if np.std(residuals) > 0 else 0.0
            ),
            "tracking_error": np.std(residuals) * np.sqrt(252),
        }

        # Statistical significance tests
        attribution_results["significance_tests"] = self._calculate_statistical_significance(
            y, X, reg_model, factor_names, aligned_data.index
        )

        return attribution_results

    def decompose_alpha_beta(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        market_returns: pd.Series = None,
    ) -> dict[str, float]:
        """
        Add alpha vs beta decomposition for excess returns.

        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            market_returns: Market returns (if different from benchmark)

        Returns:
            Dictionary containing alpha and beta decomposition
        """
        if market_returns is None:
            market_returns = benchmark_returns

        # Align data
        aligned_data = pd.DataFrame(
            {
                "portfolio": portfolio_returns,
                "benchmark": benchmark_returns,
                "market": market_returns,
            }
        ).dropna()

        if len(aligned_data) < self.config.min_observations:
            return {"error": "Insufficient data for alpha/beta analysis"}

        # Calculate excess returns
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        excess_portfolio = aligned_data["portfolio"] - risk_free_rate
        excess_market = aligned_data["market"] - risk_free_rate
        excess_benchmark = aligned_data["benchmark"] - risk_free_rate

        # Market beta (CAPM)
        market_beta = np.cov(excess_portfolio, excess_market)[0, 1] / np.var(excess_market)
        market_alpha = excess_portfolio.mean() - market_beta * excess_market.mean()

        # Benchmark beta (for tracking)
        benchmark_beta = np.cov(excess_portfolio, excess_benchmark)[0, 1] / np.var(excess_benchmark)
        benchmark_alpha = excess_portfolio.mean() - benchmark_beta * excess_benchmark.mean()

        # Jensen's alpha
        jensen_alpha = market_alpha

        # Decomposition of total return
        total_return = aligned_data["portfolio"].mean() * 252
        risk_free_contribution = risk_free_rate * 252
        market_beta_contribution = (
            market_beta * (aligned_data["market"].mean() - risk_free_rate) * 252
        )
        alpha_contribution = market_alpha * 252

        # Information ratio components
        active_return = (aligned_data["portfolio"] - aligned_data["benchmark"]).mean() * 252
        tracking_error = (aligned_data["portfolio"] - aligned_data["benchmark"]).std() * np.sqrt(
            252
        )
        information_ratio = active_return / tracking_error if tracking_error > 0 else 0.0

        return {
            "market_beta": market_beta,
            "market_alpha": market_alpha,
            "market_alpha_annualized": market_alpha * 252,
            "benchmark_beta": benchmark_beta,
            "benchmark_alpha": benchmark_alpha,
            "benchmark_alpha_annualized": benchmark_alpha * 252,
            "jensen_alpha": jensen_alpha,
            "jensen_alpha_annualized": jensen_alpha * 252,
            "return_decomposition": {
                "total_return": total_return,
                "risk_free_contribution": risk_free_contribution,
                "market_beta_contribution": market_beta_contribution,
                "alpha_contribution": alpha_contribution,
            },
            "active_return": active_return,
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
            "correlation_with_market": np.corrcoef(
                aligned_data["portfolio"], aligned_data["market"]
            )[0, 1],
            "correlation_with_benchmark": np.corrcoef(
                aligned_data["portfolio"], aligned_data["benchmark"]
            )[0, 1],
        }

    def calculate_sector_style_attribution(
        self,
        portfolio_weights: pd.DataFrame,
        sector_returns: pd.DataFrame,
        style_factors: pd.DataFrame,
        benchmark_weights: pd.DataFrame = None,
    ) -> dict[str, Any]:
        """
        Create sector and style attribution analysis.

        Args:
            portfolio_weights: Portfolio weights over time
            sector_returns: Sector returns data
            style_factors: Style factor exposures/returns
            benchmark_weights: Optional benchmark weights for relative attribution

        Returns:
            Dictionary containing sector and style attribution
        """
        attribution_results = {}

        # Sector attribution
        if not sector_returns.empty:
            sector_attribution = self._calculate_sector_attribution(
                portfolio_weights, sector_returns, benchmark_weights
            )
            attribution_results["sector_attribution"] = sector_attribution

        # Style attribution
        if not style_factors.empty:
            style_attribution = self._calculate_style_attribution(
                portfolio_weights, style_factors, benchmark_weights
            )
            attribution_results["style_attribution"] = style_attribution

        # Combined attribution summary
        if (
            "sector_attribution" in attribution_results
            and "style_attribution" in attribution_results
        ):
            attribution_results["attribution_summary"] = self._combine_attribution_results(
                attribution_results["sector_attribution"], attribution_results["style_attribution"]
            )

        return attribution_results

    def track_risk_factor_exposures(
        self, portfolio_returns: pd.Series, factor_returns: pd.DataFrame, window_size: int = None
    ) -> pd.DataFrame:
        """
        Implement risk factor exposure tracking and analysis.

        Args:
            portfolio_returns: Portfolio returns time series
            factor_returns: Factor returns data
            window_size: Rolling window size (default: attribution_window)

        Returns:
            DataFrame with rolling factor exposures over time
        """
        if window_size is None:
            window_size = self.config.attribution_window

        if len(portfolio_returns) < window_size:
            return pd.DataFrame()

        # Prepare results storage
        exposure_results = []
        dates = []

        # Calculate rolling factor exposures
        for i in range(window_size, len(portfolio_returns)):
            end_date = portfolio_returns.index[i]
            start_idx = i - window_size

            # Get window data
            window_portfolio = portfolio_returns.iloc[start_idx:i]
            window_factors = factor_returns.iloc[start_idx:i]

            # Align data
            aligned_data = pd.DataFrame({"portfolio": window_portfolio})
            for factor in window_factors.columns:
                aligned_data[factor] = window_factors[factor]
            aligned_data = aligned_data.dropna()

            if len(aligned_data) >= self.config.min_observations:
                # Perform regression
                y = aligned_data["portfolio"].values
                X = aligned_data[window_factors.columns].values

                reg_model = LinearRegression(fit_intercept=True)
                reg_model.fit(X, y)

                # Store results
                exposure_record = {"date": end_date}
                for j, factor_name in enumerate(window_factors.columns):
                    exposure_record[f"{factor_name}_exposure"] = reg_model.coef_[j]

                exposure_record["alpha"] = reg_model.intercept_
                exposure_record["r_squared"] = reg_model.score(X, y)

                exposure_results.append(exposure_record)
                dates.append(end_date)

        if not exposure_results:
            return pd.DataFrame()

        # Create DataFrame
        exposures_df = pd.DataFrame(exposure_results)
        exposures_df.set_index("date", inplace=True)

        return exposures_df

    def generate_attribution_report(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        factor_returns: pd.DataFrame,
        portfolio_weights: pd.DataFrame = None,
        sector_returns: pd.DataFrame = None,
        style_factors: pd.DataFrame = None,
    ) -> dict[str, Any]:
        """
        Generate comprehensive performance attribution report.

        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            factor_returns: Factor returns data
            portfolio_weights: Optional portfolio weights
            sector_returns: Optional sector returns
            style_factors: Optional style factors

        Returns:
            Comprehensive attribution analysis report
        """
        report = {}

        # Factor-based attribution
        report["factor_attribution"] = self.calculate_factor_based_attribution(
            portfolio_returns, factor_returns, benchmark_returns
        )

        # Alpha/Beta decomposition
        report["alpha_beta_decomposition"] = self.decompose_alpha_beta(
            portfolio_returns, benchmark_returns
        )

        # Risk factor exposure tracking
        report["factor_exposures"] = self.track_risk_factor_exposures(
            portfolio_returns, factor_returns
        )

        # Sector and style attribution (if data available)
        if portfolio_weights is not None and (
            sector_returns is not None or style_factors is not None
        ):
            report["sector_style_attribution"] = self.calculate_sector_style_attribution(
                portfolio_weights,
                sector_returns if sector_returns is not None else pd.DataFrame(),
                style_factors if style_factors is not None else pd.DataFrame(),
            )

        # Attribution summary
        report["attribution_summary"] = self._generate_attribution_summary(report)

        return report

    def _calculate_sector_attribution(
        self,
        portfolio_weights: pd.DataFrame,
        sector_returns: pd.DataFrame,
        benchmark_weights: pd.DataFrame = None,
    ) -> dict[str, Any]:
        """Calculate sector-level attribution."""
        # This is a simplified implementation
        # In practice, this would require detailed sector mapping

        sector_attribution = {}

        # Calculate average sector exposures
        if not portfolio_weights.empty:
            avg_weights = portfolio_weights.mean()
            sector_attribution["avg_sector_weights"] = avg_weights.to_dict()

        # Calculate sector contributions (simplified)
        if not sector_returns.empty:
            sector_contributions = {}
            for sector in sector_returns.columns:
                if sector in portfolio_weights.columns:
                    # Weight * return contribution
                    avg_weight = portfolio_weights[sector].mean()
                    avg_return = sector_returns[sector].mean() * 252
                    sector_contributions[sector] = avg_weight * avg_return

            sector_attribution["sector_contributions"] = sector_contributions

        return sector_attribution

    def _calculate_style_attribution(
        self,
        portfolio_weights: pd.DataFrame,
        style_factors: pd.DataFrame,
        benchmark_weights: pd.DataFrame = None,
    ) -> dict[str, Any]:
        """Calculate style-based attribution."""
        style_attribution = {}

        # Calculate style exposures over time
        if not portfolio_weights.empty and not style_factors.empty:
            # This would typically involve mapping portfolio holdings to style characteristics
            # For now, provide a placeholder structure
            style_attribution["style_exposures"] = {
                "value": 0.0,
                "growth": 0.0,
                "quality": 0.0,
                "momentum": 0.0,
            }

            style_attribution["style_contributions"] = {
                "value": 0.0,
                "growth": 0.0,
                "quality": 0.0,
                "momentum": 0.0,
            }

        return style_attribution

    def _combine_attribution_results(
        self, sector_attribution: dict[str, Any], style_attribution: dict[str, Any]
    ) -> dict[str, Any]:
        """Combine sector and style attribution results."""
        combined = {
            "total_sector_contribution": sum(
                sector_attribution.get("sector_contributions", {}).values()
            ),
            "total_style_contribution": sum(
                style_attribution.get("style_contributions", {}).values()
            ),
            "attribution_breakdown": {
                "sector_effects": sector_attribution.get("sector_contributions", {}),
                "style_effects": style_attribution.get("style_contributions", {}),
            },
        }

        return combined

    def _calculate_statistical_significance(
        self,
        y: np.ndarray,
        X: np.ndarray,
        reg_model: LinearRegression,
        factor_names: list[str],
        dates: pd.Index,
    ) -> dict[str, Any]:
        """Calculate statistical significance of attribution results."""
        # Calculate standard errors and t-statistics
        n = len(y)
        k = X.shape[1]

        # Residuals and residual sum of squares
        y_pred = reg_model.predict(X)
        residuals = y - y_pred
        rss = np.sum(residuals**2)
        mse = rss / (n - k - 1)

        # Covariance matrix of coefficients
        try:
            X_with_intercept = np.column_stack([np.ones(n), X])
            cov_matrix = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            std_errors = np.sqrt(np.diag(cov_matrix))

            # t-statistics
            coefficients = np.concatenate([[reg_model.intercept_], reg_model.coef_])
            t_stats = coefficients / std_errors

            # p-values
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))

            significance_results = {
                "alpha_t_stat": t_stats[0],
                "alpha_p_value": p_values[0],
                "alpha_significant": p_values[0] < (1 - self.config.confidence_level),
                "factor_significance": {},
            }

            for i, factor_name in enumerate(factor_names):
                significance_results["factor_significance"][factor_name] = {
                    "t_statistic": t_stats[i + 1],
                    "p_value": p_values[i + 1],
                    "significant": p_values[i + 1] < (1 - self.config.confidence_level),
                    "coefficient": reg_model.coef_[i],
                }

            return significance_results

        except np.linalg.LinAlgError:
            # Fallback if matrix inversion fails
            return {
                "alpha_t_stat": 0.0,
                "alpha_p_value": 1.0,
                "alpha_significant": False,
                "factor_significance": {factor: {"significant": False} for factor in factor_names},
            }

    def _generate_attribution_summary(self, report: dict[str, Any]) -> dict[str, str]:
        """Generate summary insights from attribution analysis."""
        summary = {}

        # Alpha assessment
        if "alpha_beta_decomposition" in report:
            alpha_data = report["alpha_beta_decomposition"]
            market_alpha = alpha_data.get("market_alpha_annualized", 0.0)

            if market_alpha > 0.02:  # 2% threshold
                summary["alpha_assessment"] = f"Strong positive alpha: {market_alpha:.2%} annually"
            elif market_alpha > 0:
                summary["alpha_assessment"] = f"Positive alpha: {market_alpha:.2%} annually"
            else:
                summary["alpha_assessment"] = f"Negative alpha: {market_alpha:.2%} annually"

        # Factor attribution assessment
        if "factor_attribution" in report and "error" not in report["factor_attribution"]:
            factor_data = report["factor_attribution"]
            r_squared = factor_data.get("r_squared", 0.0)

            if r_squared > 0.8:
                summary["factor_explanation"] = "High factor model explanatory power"
            elif r_squared > 0.5:
                summary["factor_explanation"] = "Moderate factor model explanatory power"
            else:
                summary["factor_explanation"] = "Low factor model explanatory power"

        # Beta assessment
        if "alpha_beta_decomposition" in report:
            beta_data = report["alpha_beta_decomposition"]
            market_beta = beta_data.get("market_beta", 1.0)

            if market_beta > 1.2:
                summary["beta_assessment"] = "High beta - amplifies market movements"
            elif market_beta < 0.8:
                summary["beta_assessment"] = "Low beta - dampens market movements"
            else:
                summary["beta_assessment"] = "Market-like beta exposure"

        return summary
