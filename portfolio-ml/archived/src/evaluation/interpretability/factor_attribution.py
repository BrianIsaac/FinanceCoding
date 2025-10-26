"""
Risk Factor Attribution and Traditional Finance Integration.

This module provides tools for mapping model decisions to traditional risk factors
and connecting ML allocations to factor loadings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

SKLEARN_AVAILABLE = True


@dataclass
class FactorAttributionConfig:
    """Configuration for factor attribution analysis."""

    factor_models: list[str] = None
    lookback_window: int = 252
    min_observations: int = 60
    significance_level: float = 0.05
    max_factors: int = 10
    pca_variance_threshold: float = 0.95
    rolling_window: int = 63
    risk_free_rate: float = 0.02

    def __post_init__(self):
        if self.factor_models is None:
            self.factor_models = ["fama_french_3", "momentum", "quality", "size", "value"]


@runtime_checkable
class FactorDataProvider(Protocol):
    """Protocol for factor data providers."""

    def get_factor_returns(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """Get factor returns for the specified period."""
        ...

    def get_factor_loadings(self, assets: list[str], date: pd.Timestamp) -> pd.DataFrame:
        """Get factor loadings for assets at a specific date."""
        ...


class FactorAttributor:
    """
    Risk factor attribution analyzer.

    Maps model decisions to traditional risk factors and provides
    factor loading analysis for ML-generated portfolios.
    """

    def __init__(
        self,
        config: FactorAttributionConfig | None = None,
        factor_data_provider: FactorDataProvider | None = None,
    ):
        """
        Initialize factor attributor.

        Args:
            config: Factor attribution configuration
            factor_data_provider: Provider for factor data
        """
        self.config = config or FactorAttributionConfig()
        self.factor_data_provider = factor_data_provider
        self._cached_factor_data = {}

    def analyze_factor_exposure(
        self,
        portfolio_weights: pd.Series,
        returns: pd.DataFrame,
        benchmark_weights: pd.Series | None = None,
        factor_returns: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """
        Analyze factor exposure of portfolio.

        Args:
            portfolio_weights: Portfolio weights by asset
            returns: Asset returns data
            benchmark_weights: Benchmark weights for comparison
            factor_returns: Factor returns data

        Returns:
            Factor exposure analysis results
        """
        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(portfolio_weights, returns)

        if factor_returns is None:
            factor_returns = self._generate_synthetic_factors(returns)

        # Perform factor regression
        factor_loadings = self._estimate_factor_loadings(portfolio_returns, factor_returns)

        # Calculate factor contributions
        factor_contributions = self._calculate_factor_contributions(
            factor_loadings, factor_returns, portfolio_returns
        )

        # Analyze active exposures vs benchmark
        active_exposures = {}
        if benchmark_weights is not None:
            benchmark_returns = self._calculate_portfolio_returns(benchmark_weights, returns)
            benchmark_loadings = self._estimate_factor_loadings(benchmark_returns, factor_returns)
            active_exposures = self._calculate_active_exposures(factor_loadings, benchmark_loadings)

        # Risk attribution
        risk_attribution = self._calculate_risk_attribution(factor_loadings, factor_returns)

        # Factor timing analysis
        timing_analysis = self._analyze_factor_timing(portfolio_weights, returns, factor_returns)

        return {
            "factor_loadings": factor_loadings,
            "factor_contributions": factor_contributions,
            "active_exposures": active_exposures,
            "risk_attribution": risk_attribution,
            "timing_analysis": timing_analysis,
            "portfolio_returns": portfolio_returns,
            "factor_returns": factor_returns,
            "summary_statistics": self._calculate_summary_statistics(
                factor_loadings, factor_contributions, risk_attribution
            ),
        }

    def _calculate_portfolio_returns(
        self,
        weights: pd.Series,
        returns: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate portfolio returns from weights and asset returns.

        Args:
            weights: Portfolio weights
            returns: Asset returns

        Returns:
            Portfolio returns time series
        """
        # Align weights with returns columns
        common_assets = weights.index.intersection(returns.columns)
        if len(common_assets) == 0:
            raise ValueError("No common assets between weights and returns")

        aligned_weights = weights[common_assets]
        aligned_returns = returns[common_assets]

        # Normalize weights to sum to 1
        aligned_weights = aligned_weights / aligned_weights.sum()

        # Calculate portfolio returns
        portfolio_returns = (aligned_returns * aligned_weights).sum(axis=1)

        return portfolio_returns

    def _generate_synthetic_factors(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic factor returns using PCA and style factors.

        Args:
            returns: Asset returns data

        Returns:
            Synthetic factor returns
        """
        # Market factor (equal-weighted market return)
        market_factor = returns.mean(axis=1)

        # Add some noise to prevent singular matrices
        noise_scale = market_factor.std() * 0.1 if market_factor.std() > 0 else 0.001

        # Size factor (small minus big)
        vol_scores = returns.std()
        if len(vol_scores) >= 3:
            n_third = max(1, len(vol_scores) // 3)
            small_cap = returns[vol_scores.nlargest(n_third).index].mean(axis=1)
            large_cap = returns[vol_scores.nsmallest(n_third).index].mean(axis=1)
            size_factor = small_cap - large_cap
        else:
            # Fallback for small universes
            size_factor = market_factor * 0.5 + np.random.normal(0, noise_scale, len(market_factor))

        # Momentum factor
        momentum_window = min(21, len(returns) - 1)  # Use shorter window for small datasets
        momentum_scores = returns.rolling(momentum_window).mean().iloc[-1]
        if len(momentum_scores.dropna()) >= 3:
            n_third = max(1, len(momentum_scores.dropna()) // 3)
            valid_scores = momentum_scores.dropna()
            winners = returns[valid_scores.nlargest(n_third).index].mean(axis=1)
            losers = returns[valid_scores.nsmallest(n_third).index].mean(axis=1)
            momentum_factor = winners - losers
        else:
            # Fallback for small universes
            momentum_factor = (
                market_factor * 0.3 + np.random.normal(0, noise_scale, len(market_factor))
            )

        # Quality factor (low volatility minus high volatility)
        if len(vol_scores) >= 3:
            n_third = max(1, len(vol_scores) // 3)
            low_vol = returns[vol_scores.nsmallest(n_third).index].mean(axis=1)
            high_vol = returns[vol_scores.nlargest(n_third).index].mean(axis=1)
            quality_factor = low_vol - high_vol
        else:
            # Fallback for small universes
            quality_factor = (
                -market_factor * 0.2 + np.random.normal(0, noise_scale, len(market_factor))
            )

        # Combine all factors
        factor_returns = pd.DataFrame({
            "Market": market_factor,
            "Size": size_factor,
            "Momentum": momentum_factor,
            "Quality": quality_factor,
        })

        # Fill any NaN values and add small random noise to prevent singularity
        factor_returns = factor_returns.fillna(0)

        # Add small amount of independent noise to each factor to prevent perfect correlation
        for col in factor_returns.columns:
            factor_returns[col] += np.random.normal(0, noise_scale * 0.1, len(factor_returns))

        return factor_returns

    def _estimate_factor_loadings(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Estimate factor loadings using linear regression.

        Args:
            portfolio_returns: Portfolio returns
            factor_returns: Factor returns

        Returns:
            Factor loading estimation results
        """
        # Align data
        common_dates = portfolio_returns.index.intersection(factor_returns.index)
        if len(common_dates) < self.config.min_observations:
            raise ValueError(
                f"Insufficient observations: {len(common_dates)} < {self.config.min_observations}"
            )

        y = portfolio_returns[common_dates].values
        x = factor_returns.loc[common_dates].values

        # Add intercept (alpha)
        x_with_intercept = np.column_stack([np.ones(len(x)), x])

        # Perform regression
        try:
            if SKLEARN_AVAILABLE:
                reg = LinearRegression(fit_intercept=False)
                reg.fit(x_with_intercept, y)

                coefficients = reg.coef_
                alpha = coefficients[0]
                betas = coefficients[1:]

                # Calculate R-squared
                y_pred = reg.predict(x_with_intercept)
                r_squared = reg.score(x_with_intercept, y)

                # Calculate residuals
                residuals = y - y_pred

            else:
                # Fallback using numpy
                coefficients = np.linalg.lstsq(x_with_intercept, y, rcond=None)[0]
                alpha = coefficients[0]
                betas = coefficients[1:]

                y_pred = x_with_intercept @ coefficients
                residuals = y - y_pred
                r_squared = 1 - np.var(residuals) / np.var(y)

            # Calculate t-statistics and p-values
            residual_var = np.var(residuals)
            x_inv = np.linalg.inv(x_with_intercept.T @ x_with_intercept)
            se = np.sqrt(np.diag(x_inv) * residual_var)
            t_stats = coefficients / se
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), len(y) - len(coefficients)))

            # Create results dictionary
            factor_names = ["Alpha"] + list(factor_returns.columns)
            loadings = {
                "coefficients": dict(zip(factor_names, coefficients)),
                "t_statistics": dict(zip(factor_names, t_stats)),
                "p_values": dict(zip(factor_names, p_values)),
                "standard_errors": dict(zip(factor_names, se)),
                "alpha": float(alpha),
                "betas": dict(zip(factor_returns.columns, betas)),
                "r_squared": float(r_squared),
                "residuals": pd.Series(residuals, index=common_dates),
                "fitted_values": pd.Series(y_pred, index=common_dates),
            }

            # Add significance flags
            loadings["significant_factors"] = [
                name for name, p_val in loadings["p_values"].items()
                if p_val < self.config.significance_level
            ]

            return loadings

        except np.linalg.LinAlgError:
            # Handle singular matrix
            return {
                "coefficients": {},
                "error": "Singular matrix - insufficient variation in factors",
                "alpha": 0.0,
                "betas": {},
                "r_squared": 0.0,
            }

    def _calculate_factor_contributions(
        self,
        factor_loadings: dict[str, Any],
        factor_returns: pd.DataFrame,
        portfolio_returns: pd.Series,
    ) -> dict[str, Any]:
        """
        Calculate factor contributions to portfolio returns.

        Args:
            factor_loadings: Factor loading results
            factor_returns: Factor returns
            portfolio_returns: Portfolio returns

        Returns:
            Factor contribution analysis
        """
        if "betas" not in factor_loadings or not factor_loadings["betas"]:
            return {"contributions": {}, "total_explained": 0.0}

        # Calculate contributions for each factor
        contributions = {}
        factor_contribution_series = {}

        common_dates = portfolio_returns.index.intersection(factor_returns.index)

        for factor_name, beta in factor_loadings["betas"].items():
            if factor_name in factor_returns.columns:
                factor_contrib = beta * factor_returns[factor_name].loc[common_dates]
                contributions[factor_name] = {
                    "total_contribution": float(factor_contrib.sum()),
                    "mean_contribution": float(factor_contrib.mean()),
                    "volatility_contribution": float(factor_contrib.std()),
                    "beta": float(beta),
                }
                factor_contribution_series[factor_name] = factor_contrib

        # Calculate total explained return
        total_explained = sum(contrib["total_contribution"] for contrib in contributions.values())

        # Calculate unexplained return (alpha + residuals)
        alpha_contribution = factor_loadings.get("alpha", 0.0) * len(common_dates)
        residual_contribution = factor_loadings.get("residuals", pd.Series()).sum()
        unexplained = alpha_contribution + residual_contribution

        return {
            "contributions": contributions,
            "contribution_series": factor_contribution_series,
            "total_explained": float(total_explained),
            "alpha_contribution": float(alpha_contribution),
            "residual_contribution": float(residual_contribution),
            "unexplained_return": float(unexplained),
            "explanation_ratio": float(
                total_explained / (total_explained + abs(unexplained) + 1e-8)
            ),
        }

    def _calculate_active_exposures(
        self,
        portfolio_loadings: dict[str, Any],
        benchmark_loadings: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Calculate active factor exposures vs benchmark.

        Args:
            portfolio_loadings: Portfolio factor loadings
            benchmark_loadings: Benchmark factor loadings

        Returns:
            Active exposure analysis
        """
        portfolio_betas = portfolio_loadings.get("betas", {})
        benchmark_betas = benchmark_loadings.get("betas", {})

        active_betas = {}
        for factor in set(portfolio_betas.keys()) | set(benchmark_betas.keys()):
            portfolio_beta = portfolio_betas.get(factor, 0.0)
            benchmark_beta = benchmark_betas.get(factor, 0.0)
            active_betas[factor] = portfolio_beta - benchmark_beta

        # Calculate tracking error components
        tracking_error_components = {}
        for factor, active_beta in active_betas.items():
            tracking_error_components[factor] = abs(active_beta)

        return {
            "active_betas": active_betas,
            "tracking_error_components": tracking_error_components,
            "total_active_risk": sum(abs(beta) for beta in active_betas.values()),
            "largest_active_exposures": sorted(
                active_betas.items(), key=lambda x: abs(x[1]), reverse=True
            )[:5],
        }

    def _calculate_risk_attribution(
        self,
        factor_loadings: dict[str, Any],
        factor_returns: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Calculate risk attribution to factors.

        Args:
            factor_loadings: Factor loadings
            factor_returns: Factor returns

        Returns:
            Risk attribution analysis
        """
        betas = factor_loadings.get("betas", {})

        if not betas:
            return {"factor_risks": {}, "total_systematic_risk": 0.0}

        # Calculate factor variances
        factor_vars = factor_returns.var()

        # Calculate factor risk contributions
        factor_risks = {}
        for factor, beta in betas.items():
            if factor in factor_vars.index:
                factor_risk = (beta ** 2) * factor_vars[factor]
                factor_risks[factor] = {
                    "variance_contribution": float(factor_risk),
                    "volatility_contribution": float(np.sqrt(factor_risk)),
                    "beta": float(beta),
                    "factor_volatility": float(np.sqrt(factor_vars[factor])),
                }

        # Calculate total systematic risk
        systematic_variance = sum(risk["variance_contribution"] for risk in factor_risks.values())

        # Idiosyncratic risk from residuals
        residuals = factor_loadings.get("residuals", pd.Series())
        idiosyncratic_variance = residuals.var() if len(residuals) > 0 else 0.0

        return {
            "factor_risks": factor_risks,
            "systematic_variance": float(systematic_variance),
            "systematic_volatility": float(np.sqrt(systematic_variance)),
            "idiosyncratic_variance": float(idiosyncratic_variance),
            "idiosyncratic_volatility": float(np.sqrt(idiosyncratic_variance)),
            "total_variance": float(systematic_variance + idiosyncratic_variance),
            "systematic_ratio": float(
                systematic_variance / (systematic_variance + idiosyncratic_variance + 1e-8)
            ),
        }

    def _analyze_factor_timing(
        self,
        portfolio_weights: pd.Series,
        returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Analyze factor timing in portfolio decisions.

        Args:
            portfolio_weights: Portfolio weights
            returns: Asset returns
            factor_returns: Factor returns

        Returns:
            Factor timing analysis
        """
        # Calculate correlation between portfolio returns and factor returns
        portfolio_returns = self._calculate_portfolio_returns(portfolio_weights, returns)

        correlations = {}
        for factor in factor_returns.columns:
            common_dates = portfolio_returns.index.intersection(factor_returns.index)
            if len(common_dates) > 30:
                corr = portfolio_returns[common_dates].corr(factor_returns[factor][common_dates])
                correlations[factor] = float(corr) if not np.isnan(corr) else 0.0

        # Identify timing patterns (simplified)
        timing_signals = {}
        for factor, corr in correlations.items():
            if abs(corr) > 0.3:  # Threshold for significant timing
                timing_signals[factor] = {
                    "correlation": corr,
                    "timing_strength": "Strong" if abs(corr) > 0.5 else "Moderate",
                    "direction": "Positive" if corr > 0 else "Negative",
                }

        return {
            "factor_correlations": correlations,
            "timing_signals": timing_signals,
            "strongest_timing_factor": (
                max(correlations.items(), key=lambda x: abs(x[1]))[0]
                if correlations else None
            ),
        }

    def _calculate_summary_statistics(
        self,
        factor_loadings: dict[str, Any],
        factor_contributions: dict[str, Any],
        risk_attribution: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Calculate summary statistics for factor analysis.

        Args:
            factor_loadings: Factor loadings
            factor_contributions: Factor contributions
            risk_attribution: Risk attribution

        Returns:
            Summary statistics
        """
        summary = {
            "model_quality": {
                "r_squared": factor_loadings.get("r_squared", 0.0),
                "significant_factors": len(factor_loadings.get("significant_factors", [])),
                "explanation_ratio": factor_contributions.get("explanation_ratio", 0.0),
            },
            "risk_decomposition": {
                "systematic_ratio": risk_attribution.get("systematic_ratio", 0.0),
                "systematic_volatility": risk_attribution.get("systematic_volatility", 0.0),
                "idiosyncratic_volatility": risk_attribution.get("idiosyncratic_volatility", 0.0),
            },
        }

        # Top factor contributions
        contributions = factor_contributions.get("contributions", {})
        if contributions:
            top_factors = sorted(
                contributions.items(),
                key=lambda x: abs(x[1]["total_contribution"]),
                reverse=True
            )[:3]
            summary["top_factor_contributors"] = [
                {"factor": factor, "contribution": data["total_contribution"]}
                for factor, data in top_factors
            ]

        return summary
