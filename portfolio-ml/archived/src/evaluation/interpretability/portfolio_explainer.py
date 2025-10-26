"""
Portfolio Allocation Explanation Engine.

This module provides tools for creating interpretable explanations of
portfolio allocations and investment decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PortfolioExplanationConfig:
    """Configuration for portfolio explanation generation."""

    max_contributors: int = 10
    explanation_format: str = "technical"  # "technical" or "plain_english"
    significance_threshold: float = 0.01  # Minimum weight to consider significant
    benchmark_weights: dict[str, float] | None = None  # Benchmark for comparison
    include_risk_metrics: bool = True
    include_performance_attribution: bool = True


class PortfolioExplainer:
    """
    Portfolio allocation explanation generator.

    Creates interpretable explanations linking model outputs to
    specific investment rationales and allocation decisions.
    """

    def __init__(self, config: PortfolioExplanationConfig | None = None):
        """
        Initialize portfolio explainer.

        Args:
            config: Portfolio explanation configuration
        """
        self.config = config or PortfolioExplanationConfig()

    def explain_allocation(
        self,
        portfolio_weights: pd.Series,
        expected_returns: pd.Series | None = None,
        risk_data: dict[str, Any] | None = None,
        market_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Generate comprehensive explanation for portfolio allocation.

        Args:
            portfolio_weights: Portfolio weights by asset
            expected_returns: Expected returns for assets (optional)
            risk_data: Risk metrics and covariance data (optional)
            market_context: Market context and sector information (optional)

        Returns:
            Portfolio allocation explanations
        """
        explanations = {}

        # 1. Basic allocation analysis
        explanations["allocation_summary"] = self._analyze_allocation_summary(portfolio_weights)

        # 2. Top contributors and detractors
        explanations["key_positions"] = self._identify_key_positions(portfolio_weights)

        # 3. Allocation rationale based on returns
        if expected_returns is not None:
            explanations["return_rationale"] = self._explain_return_rationale(
                portfolio_weights, expected_returns
            )

        # 4. Risk-based explanations
        if risk_data is not None and self.config.include_risk_metrics:
            explanations["risk_rationale"] = self._explain_risk_rationale(
                portfolio_weights, risk_data
            )

        # 5. Benchmark comparison
        explanations["benchmark_comparison"] = self._compare_to_benchmark(portfolio_weights)

        # 6. Sector/style analysis
        if market_context is not None:
            explanations["market_context"] = self._analyze_market_context(
                portfolio_weights, market_context
            )

        # 7. Investment themes
        explanations["investment_themes"] = self._identify_investment_themes(portfolio_weights)

        # 8. Plain English summary
        if self.config.explanation_format == "plain_english":
            explanations["plain_english_summary"] = self._generate_plain_english_summary(
                explanations
            )

        return explanations

    def build_allocation_decision_tree(
        self,
        portfolio_weights: pd.Series,
        feature_data: dict[str, pd.Series] | None = None,
    ) -> dict[str, Any]:
        """
        Build decision tree explaining allocation logic.

        Args:
            portfolio_weights: Portfolio weights
            feature_data: Asset features driving decisions (optional)

        Returns:
            Decision tree structure explaining allocations
        """
        # Create simplified decision tree structure
        decision_tree = {
            "root": {
                "question": "Portfolio Allocation Decision",
                "total_weight": 1.0,
                "n_assets": len(portfolio_weights),
            }
        }

        # Categorize assets by weight ranges
        high_weight_assets = portfolio_weights[portfolio_weights >= 0.1]
        medium_weight_assets = portfolio_weights[
            (portfolio_weights >= 0.05) & (portfolio_weights < 0.1)
        ]
        low_weight_assets = portfolio_weights[
            (portfolio_weights >= self.config.significance_threshold) & (portfolio_weights < 0.05)
        ]

        # Build tree branches
        if len(high_weight_assets) > 0:
            decision_tree["high_conviction"] = {
                "question": f"High conviction positions (≥10%): {len(high_weight_assets)} assets",
                "assets": high_weight_assets.to_dict(),
                "total_weight": high_weight_assets.sum(),
                "rationale": "Core portfolio positions with high confidence",
            }

        if len(medium_weight_assets) > 0:
            decision_tree["medium_conviction"] = {
                "question": f"Medium positions (5-10%): {len(medium_weight_assets)} assets",
                "assets": medium_weight_assets.to_dict(),
                "total_weight": medium_weight_assets.sum(),
                "rationale": "Diversification and opportunity positions",
            }

        if len(low_weight_assets) > 0:
            decision_tree["low_weight"] = {
                "question": f"Small positions (<5%): {len(low_weight_assets)} assets",
                "assets": low_weight_assets.to_dict(),
                "total_weight": low_weight_assets.sum(),
                "rationale": "Risk management and tactical allocations",
            }

        return decision_tree

    def analyze_allocation_changes(
        self,
        current_weights: pd.Series,
        previous_weights: pd.Series,
        returns_data: pd.Series | None = None,
    ) -> dict[str, Any]:
        """
        Analyze and explain changes in portfolio allocation.

        Args:
            current_weights: Current portfolio weights
            previous_weights: Previous portfolio weights
            returns_data: Recent returns data (optional)

        Returns:
            Analysis of allocation changes
        """
        # Calculate weight changes
        all_assets = current_weights.index.union(previous_weights.index)
        current_full = current_weights.reindex(all_assets, fill_value=0.0)
        previous_full = previous_weights.reindex(all_assets, fill_value=0.0)

        weight_changes = current_full - previous_full

        # Categorize changes
        increased_positions = weight_changes[weight_changes > 0.01].sort_values(ascending=False)
        decreased_positions = weight_changes[weight_changes < -0.01].sort_values()
        new_positions = current_full[(current_full > 0) & (previous_full == 0)]
        closed_positions = previous_full[(previous_full > 0) & (current_full == 0)]

        # Calculate turnover
        turnover = weight_changes.abs().sum() / 2.0

        # Generate explanations
        change_analysis = {
            "turnover": float(turnover),
            "n_changes": (weight_changes.abs() > 0.001).sum(),
            "increased_positions": increased_positions.to_dict(),
            "decreased_positions": decreased_positions.to_dict(),
            "new_positions": new_positions.to_dict(),
            "closed_positions": closed_positions.to_dict(),
        }

        # Add performance attribution if returns available
        if returns_data is not None:
            change_analysis["performance_impact"] = self._analyze_change_performance_impact(
                weight_changes, returns_data
            )

        # Generate change explanations
        change_analysis["change_rationale"] = self._generate_change_rationale(
            increased_positions, decreased_positions, new_positions, closed_positions
        )

        return change_analysis

    def create_client_report(
        self,
        portfolio_weights: pd.Series,
        explanation_data: dict[str, Any],
        performance_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create client-ready portfolio explanation report.

        Args:
            portfolio_weights: Portfolio weights
            explanation_data: Detailed explanation data
            performance_data: Performance metrics (optional)

        Returns:
            Client-ready report with explanations
        """
        report = {
            "executive_summary": self._create_executive_summary(portfolio_weights, explanation_data),
            "key_holdings": self._format_key_holdings(explanation_data.get("key_positions", {})),
            "investment_strategy": self._explain_investment_strategy(explanation_data),
            "risk_profile": self._summarize_risk_profile(explanation_data),
        }

        if performance_data:
            report["performance_summary"] = self._format_performance_summary(performance_data)

        return report

    def _analyze_allocation_summary(self, portfolio_weights: pd.Series) -> dict[str, Any]:
        """Analyze basic portfolio allocation characteristics."""
        weights = portfolio_weights.values

        summary = {
            "n_positions": len(portfolio_weights),
            "n_significant_positions": (portfolio_weights >= self.config.significance_threshold).sum(),
            "max_weight": float(portfolio_weights.max()),
            "min_weight": float(portfolio_weights.min()),
            "top_10_concentration": float(portfolio_weights.nlargest(10).sum()),
            "herfindahl_index": float((weights ** 2).sum()),
            "effective_n_stocks": float(1.0 / (weights ** 2).sum()) if (weights ** 2).sum() > 0 else 0,
            "weight_dispersion": float(portfolio_weights.std()),
        }

        # Categorize concentration
        if summary["herfindahl_index"] > 0.2:
            summary["concentration_level"] = "High"
        elif summary["herfindahl_index"] > 0.1:
            summary["concentration_level"] = "Medium"
        else:
            summary["concentration_level"] = "Low"

        return summary

    def _identify_key_positions(self, portfolio_weights: pd.Series) -> dict[str, Any]:
        """Identify key portfolio positions and their characteristics."""
        # Sort by weight
        sorted_weights = portfolio_weights.sort_values(ascending=False)

        # Identify top contributors
        top_contributors = sorted_weights.head(self.config.max_contributors)

        # Identify significant positions
        significant_positions = sorted_weights[
            sorted_weights >= self.config.significance_threshold
        ]

        # Calculate contribution percentages
        contribution_pcts = (top_contributors / top_contributors.sum() * 100).round(1)

        return {
            "top_contributors": top_contributors.to_dict(),
            "contribution_percentages": contribution_pcts.to_dict(),
            "significant_positions": significant_positions.to_dict(),
            "largest_position": {
                "asset": sorted_weights.index[0],
                "weight": float(sorted_weights.iloc[0]),
                "percentage_of_portfolio": float(sorted_weights.iloc[0] * 100),
            },
            "top_5_total_weight": float(sorted_weights.head(5).sum()),
            "position_tiers": {
                "large_positions": sorted_weights[sorted_weights >= 0.05].to_dict(),
                "medium_positions": sorted_weights[
                    (sorted_weights >= 0.02) & (sorted_weights < 0.05)
                ].to_dict(),
                "small_positions": sorted_weights[
                    (sorted_weights >= 0.01) & (sorted_weights < 0.02)
                ].to_dict(),
            },
        }

    def _explain_return_rationale(
        self, portfolio_weights: pd.Series, expected_returns: pd.Series
    ) -> dict[str, Any]:
        """Explain allocation rationale based on expected returns."""
        # Calculate weighted expected return
        portfolio_return = (portfolio_weights * expected_returns).sum()

        # Find highest return assets
        high_return_assets = expected_returns.sort_values(ascending=False).head(10)

        # Analyze return-weight relationship
        return_weight_corr = portfolio_weights.corr(expected_returns)

        # Identify potential return tilts
        return_tilts = []
        for asset in portfolio_weights.index:
            if asset in expected_returns.index:
                return_rank = (expected_returns >= expected_returns[asset]).sum()
                weight_rank = (portfolio_weights >= portfolio_weights[asset]).sum()

                if weight_rank < return_rank - 2:  # Overweighted relative to return rank
                    return_tilts.append({
                        "asset": asset,
                        "weight": float(portfolio_weights[asset]),
                        "expected_return": float(expected_returns[asset]),
                        "return_rank": int(return_rank),
                        "weight_rank": int(weight_rank),
                        "tilt": "overweight",
                    })
                elif weight_rank > return_rank + 2:  # Underweighted
                    return_tilts.append({
                        "asset": asset,
                        "weight": float(portfolio_weights[asset]),
                        "expected_return": float(expected_returns[asset]),
                        "return_rank": int(return_rank),
                        "weight_rank": int(weight_rank),
                        "tilt": "underweight",
                    })

        return {
            "portfolio_expected_return": float(portfolio_return),
            "return_weight_correlation": float(return_weight_corr),
            "high_return_assets": high_return_assets.to_dict(),
            "return_tilts": return_tilts,
            "return_strategy": (
                "Return-focused" if return_weight_corr > 0.3 else
                "Risk-focused" if return_weight_corr < -0.1 else
                "Balanced"
            ),
        }

    def _explain_risk_rationale(
        self, portfolio_weights: pd.Series, risk_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Explain allocation rationale based on risk characteristics."""
        risk_explanation = {}

        if "volatility" in risk_data:
            volatilities = risk_data["volatility"]
            portfolio_vol = np.sqrt(
                (portfolio_weights ** 2 * volatilities ** 2).sum()
            )  # Simplified calculation

            risk_explanation["portfolio_volatility"] = float(portfolio_vol)
            risk_explanation["volatility_weighted_avg"] = float(
                (portfolio_weights * volatilities).sum()
            )

        if "covariance_matrix" in risk_data:
            cov_matrix = risk_data["covariance_matrix"]
            portfolio_variance = np.dot(
                portfolio_weights.values, np.dot(cov_matrix.values, portfolio_weights.values)
            )
            risk_explanation["portfolio_risk"] = float(np.sqrt(portfolio_variance))

        return risk_explanation

    def _compare_to_benchmark(self, portfolio_weights: pd.Series) -> dict[str, Any]:
        """Compare portfolio to benchmark (equal weight by default)."""
        if self.config.benchmark_weights:
            benchmark = pd.Series(self.config.benchmark_weights)
        else:
            # Use equal weight as benchmark
            benchmark = pd.Series(1.0 / len(portfolio_weights), index=portfolio_weights.index)

        # Align indices
        all_assets = portfolio_weights.index.union(benchmark.index)
        portfolio_full = portfolio_weights.reindex(all_assets, fill_value=0.0)
        benchmark_full = benchmark.reindex(all_assets, fill_value=0.0)

        # Calculate active weights
        active_weights = portfolio_full - benchmark_full

        # Active share
        active_share = active_weights.abs().sum() / 2.0

        # Top over/underweights
        overweights = active_weights[active_weights > 0.01].sort_values(ascending=False)
        underweights = active_weights[active_weights < -0.01].sort_values()

        return {
            "active_share": float(active_share),
            "overweights": overweights.to_dict(),
            "underweights": underweights.to_dict(),
            "benchmark_type": "Equal Weight" if not self.config.benchmark_weights else "Custom",
            "tracking_error_contributors": {
                "top_overweight": overweights.index[0] if len(overweights) > 0 else None,
                "top_underweight": underweights.index[0] if len(underweights) > 0 else None,
            },
        }

    def _analyze_market_context(
        self, portfolio_weights: pd.Series, market_context: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze portfolio in market context."""
        context_analysis = {}

        if "sectors" in market_context:
            sector_mapping = market_context["sectors"]
            sector_weights = {}

            for asset, weight in portfolio_weights.items():
                sector = sector_mapping.get(asset, "Other")
                sector_weights[sector] = sector_weights.get(sector, 0) + weight

            context_analysis["sector_allocation"] = sector_weights
            context_analysis["dominant_sector"] = max(sector_weights, key=sector_weights.get)

        return context_analysis

    def _identify_investment_themes(self, portfolio_weights: pd.Series) -> dict[str, Any]:
        """Identify investment themes from portfolio composition."""
        # Simple thematic analysis based on asset names/tickers
        themes = {
            "Growth": 0.0,
            "Value": 0.0,
            "Technology": 0.0,
            "Defensive": 0.0,
        }

        # Heuristic-based theme assignment
        for asset, weight in portfolio_weights.items():
            asset_upper = asset.upper()

            # Technology theme
            if any(ticker in asset_upper for ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]):
                themes["Technology"] += weight
                themes["Growth"] += weight

            # Defensive theme
            elif any(ticker in asset_upper for ticker in ["JNJ", "PG", "KO", "WMT"]):
                themes["Defensive"] += weight
                themes["Value"] += weight

            # Financial theme
            elif any(ticker in asset_upper for ticker in ["JPM", "BAC", "WFC"]):
                themes["Value"] += weight

        # Identify dominant themes
        dominant_theme = max(themes, key=themes.get)

        return {
            "theme_weights": themes,
            "dominant_theme": dominant_theme,
            "theme_concentration": themes[dominant_theme],
            "theme_diversification": len([t for t in themes.values() if t > 0.05]),
        }

    def _generate_plain_english_summary(self, explanations: dict[str, Any]) -> str:
        """Generate plain English summary of portfolio allocation."""
        summary_parts = []

        # Basic allocation info
        if "allocation_summary" in explanations:
            alloc = explanations["allocation_summary"]
            summary_parts.append(
                f"This portfolio holds {alloc['n_positions']} positions with "
                f"{alloc['concentration_level'].lower()} concentration."
            )

        # Key positions
        if "key_positions" in explanations:
            key_pos = explanations["key_positions"]
            if "largest_position" in key_pos:
                largest = key_pos["largest_position"]
                summary_parts.append(
                    f"The largest holding is {largest['asset']} at "
                    f"{largest['percentage_of_portfolio']:.1f}% of the portfolio."
                )

        # Investment themes
        if "investment_themes" in explanations:
            themes = explanations["investment_themes"]
            dominant = themes["dominant_theme"]
            concentration = themes["theme_concentration"]
            summary_parts.append(
                f"The portfolio has a {dominant.lower()} tilt with "
                f"{concentration:.1%} allocation to this theme."
            )

        # Benchmark comparison
        if "benchmark_comparison" in explanations:
            benchmark = explanations["benchmark_comparison"]
            active_share = benchmark["active_share"]
            summary_parts.append(
                f"The strategy shows {active_share:.1%} active share relative to the benchmark."
            )

        return " ".join(summary_parts)

    def _generate_change_rationale(
        self,
        increased_positions: pd.Series,
        decreased_positions: pd.Series,
        new_positions: pd.Series,
        closed_positions: pd.Series,
    ) -> list[str]:
        """Generate rationale for allocation changes."""
        rationales = []

        if len(increased_positions) > 0:
            top_increase = increased_positions.index[0]
            rationales.append(
                f"Increased position in {top_increase} by "
                f"{increased_positions.iloc[0]:.2%} - likely due to improved outlook"
            )

        if len(decreased_positions) > 0:
            top_decrease = decreased_positions.index[0]
            rationales.append(
                f"Reduced position in {top_decrease} by "
                f"{abs(decreased_positions.iloc[0]):.2%} - risk management or profit taking"
            )

        if len(new_positions) > 0:
            rationales.append(
                f"Added {len(new_positions)} new position(s): "
                f"{', '.join(new_positions.index[:3])} - portfolio expansion"
            )

        if len(closed_positions) > 0:
            rationales.append(
                f"Closed {len(closed_positions)} position(s): "
                f"{', '.join(closed_positions.index[:3])} - portfolio optimization"
            )

        return rationales

    def _analyze_change_performance_impact(
        self, weight_changes: pd.Series, returns_data: pd.Series
    ) -> dict[str, float]:
        """Analyze performance impact of allocation changes."""
        # Calculate hypothetical impact
        performance_impact = (weight_changes * returns_data).sum()

        return {
            "total_impact": float(performance_impact),
            "positive_contributors": float(
                (weight_changes[weight_changes > 0] * returns_data).sum()
            ),
            "negative_contributors": float(
                (weight_changes[weight_changes < 0] * returns_data).sum()
            ),
        }

    def _create_executive_summary(
        self, portfolio_weights: pd.Series, explanation_data: dict[str, Any]
    ) -> dict[str, str]:
        """Create executive summary for client report."""
        return {
            "portfolio_overview": f"Portfolio with {len(portfolio_weights)} positions",
            "investment_approach": "Quantitative model-driven allocation",
            "key_characteristics": "Systematic risk management with active allocation",
        }

    def _format_key_holdings(self, key_positions: dict[str, Any]) -> dict[str, Any]:
        """Format key holdings for client presentation."""
        if not key_positions:
            return {}

        return {
            "top_holdings": key_positions.get("top_contributors", {}),
            "position_sizing": "Positions sized based on expected risk-adjusted returns",
        }

    def _explain_investment_strategy(self, explanation_data: dict[str, Any]) -> str:
        """Explain overall investment strategy."""
        return "Systematic portfolio construction using quantitative models"

    def _summarize_risk_profile(self, explanation_data: dict[str, Any]) -> str:
        """Summarize portfolio risk profile."""
        return "Diversified risk profile with systematic risk management"

    def _format_performance_summary(self, performance_data: dict[str, Any]) -> dict[str, Any]:
        """Format performance summary for client report."""
        return {
            "summary": "Performance driven by systematic allocation decisions",
            "attribution": performance_data,
        }
