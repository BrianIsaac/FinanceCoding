"""
Strategic recommendation engine for ML-enhanced portfolio construction deployment.

This module provides comprehensive strategic recommendations including decision trees,
scenario analysis, market regime integration, and implementation roadmaps for
optimal approach selection based on institutional constraints and objectives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Plotting libraries can be added when visualization features are implemented
HAS_MATPLOTLIB = False
HAS_PLOTLY = False

from src.evaluation.reporting.regime_analysis import MarketRegimeAnalysis


@dataclass
class StrategyConfig:
    """Configuration for strategic recommendation engine."""

    # Institutional constraint weights
    performance_weight: float = 0.40
    feasibility_weight: float = 0.25
    risk_weight: float = 0.20
    cost_weight: float = 0.15

    # Risk tolerance levels
    risk_tolerance_levels: dict[str, float] = None

    # Implementation priority scoring
    time_to_value_weight: float = 0.30
    strategic_impact_weight: float = 0.40
    resource_efficiency_weight: float = 0.30

    def __post_init__(self):
        if self.risk_tolerance_levels is None:
            self.risk_tolerance_levels = {
                "conservative": 0.15,  # Max 15% drawdown tolerance
                "moderate": 0.25,  # Max 25% drawdown tolerance
                "aggressive": 0.40,  # Max 40% drawdown tolerance
            }


class StrategyRecommendationEngine:
    """
    Strategic recommendation engine for institutional ML deployment decisions.

    Provides comprehensive decision frameworks, scenario analysis, market regime
    integration, and implementation roadmaps tailored to institutional constraints.
    """

    def __init__(self, config: StrategyConfig = None):
        """
        Initialize strategic recommendation engine.

        Args:
            config: Strategy recommendation configuration
        """
        self.config = config or StrategyConfig()
        self.regime_analyzer = MarketRegimeAnalysis()

    def create_decision_tree_framework(
        self,
        performance_results: pd.DataFrame,
        feasibility_results: pd.DataFrame,
        institutional_constraints: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Create decision tree framework for optimal approach selection.

        Args:
            performance_results: Comprehensive performance analysis results
            feasibility_results: Implementation feasibility assessment results
            institutional_constraints: Institution-specific constraints and preferences

        Returns:
            Dictionary containing decision tree framework and recommendations
        """
        # Extract institutional parameters
        risk_tolerance = institutional_constraints.get("risk_tolerance", "moderate")
        aum_size = institutional_constraints.get("aum_millions", 500)
        institutional_constraints.get("regulatory_complexity", "standard")
        computational_budget = institutional_constraints.get("computational_budget", "medium")
        institutional_constraints.get("preferred_timeline_months", 6)

        # Calculate institutional suitability scores for each model
        suitability_scores = []

        for model in performance_results["Model"].unique():
            perf_row = performance_results[performance_results["Model"] == model].iloc[0]
            feas_row = feasibility_results[feasibility_results["Model"] == model].iloc[0]

            # Calculate component scores (0-100)
            performance_score = self._calculate_performance_score(perf_row, risk_tolerance)
            feasibility_score = self._calculate_feasibility_score(feas_row, computational_budget)
            risk_score = self._calculate_risk_score(perf_row, risk_tolerance)
            cost_score = self._calculate_cost_score(feas_row, aum_size)

            # Weighted overall score
            overall_score = (
                performance_score * self.config.performance_weight
                + feasibility_score * self.config.feasibility_weight
                + risk_score * self.config.risk_weight
                + cost_score * self.config.cost_weight
            )

            suitability_scores.append(
                {
                    "Model": model,
                    "Performance Score": performance_score,
                    "Feasibility Score": feasibility_score,
                    "Risk Score": risk_score,
                    "Cost Score": cost_score,
                    "Overall Suitability": overall_score,
                    "Primary Recommendation": self._generate_recommendation_category(
                        performance_score, feasibility_score, risk_score, cost_score
                    ),
                }
            )

        suitability_df = pd.DataFrame(suitability_scores)
        suitability_df = suitability_df.sort_values("Overall Suitability", ascending=False)

        # Generate decision tree logic
        decision_tree = self._create_decision_logic(suitability_df, institutional_constraints)

        return {
            "suitability_scores": suitability_df,
            "decision_tree": decision_tree,
            "top_recommendation": suitability_df.iloc[0]["Model"],
            "institutional_profile": self._create_institutional_profile(institutional_constraints),
            "implementation_priority": self._prioritize_implementation(
                suitability_df, institutional_constraints
            ),
        }

    def perform_scenario_analysis(
        self,
        performance_results: pd.DataFrame,
        market_regime_analysis: dict[str, Any],
        scenario_specifications: dict[str, dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Perform scenario analysis for different market conditions and objectives.

        Args:
            performance_results: Performance analysis results
            market_regime_analysis: Market regime performance analysis
            scenario_specifications: Custom scenario parameters

        Returns:
            DataFrame with scenario analysis results
        """
        if scenario_specifications is None:
            scenario_specifications = {
                "Bull Market High Growth": {
                    "market_regime": "bull",
                    "risk_appetite": "high",
                    "return_target": 0.25,
                },
                "Bear Market Capital Preservation": {
                    "market_regime": "bear",
                    "risk_appetite": "low",
                    "return_target": 0.05,
                },
                "Volatile Market Stability": {
                    "market_regime": "volatile",
                    "risk_appetite": "medium",
                    "return_target": 0.12,
                },
                "Rising Rate Environment": {
                    "market_regime": "rising_rates",
                    "risk_appetite": "medium",
                    "return_target": 0.15,
                },
                "Low Volatility Grinding": {
                    "market_regime": "low_vol",
                    "risk_appetite": "low",
                    "return_target": 0.08,
                },
            }

        scenario_results = []

        for scenario_name, scenario_params in scenario_specifications.items():
            for model in performance_results["Model"].unique():
                model_performance = performance_results[performance_results["Model"] == model].iloc[
                    0
                ]

                # Adjust performance based on scenario
                scenario_performance = self._adjust_performance_for_scenario(
                    model_performance, scenario_params, market_regime_analysis.get(model, {})
                )

                scenario_results.append(
                    {
                        "Scenario": scenario_name,
                        "Model": model,
                        "Expected Sharpe Ratio": scenario_performance["adjusted_sharpe"],
                        "Expected Annual Return (%)": scenario_performance["adjusted_return"] * 100,
                        "Expected Max Drawdown (%)": scenario_performance["adjusted_drawdown"]
                        * 100,
                        "Regime Suitability": scenario_performance["regime_suitability"],
                        "Risk-Adjusted Score": scenario_performance["risk_adjusted_score"],
                        "Scenario Rank": 0,  # Will be calculated after all models processed
                    }
                )

        scenario_df = pd.DataFrame(scenario_results)

        # Calculate rankings within each scenario
        for scenario in scenario_df["Scenario"].unique():
            scenario_mask = scenario_df["Scenario"] == scenario
            scenario_df.loc[scenario_mask, "Scenario Rank"] = scenario_df.loc[
                scenario_mask, "Risk-Adjusted Score"
            ].rank(ascending=False)

        return scenario_df.sort_values(["Scenario", "Scenario Rank"])

    def integrate_market_regime_analysis(
        self,
        performance_results: pd.DataFrame,
        regime_performance_data: dict[str, dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Integrate market regime performance analysis into strategic recommendations.

        Args:
            performance_results: Base performance results
            regime_performance_data: Regime-specific performance data

        Returns:
            DataFrame with regime-integrated analysis
        """
        regime_integrated_results = []

        market_regimes = ["bull", "bear", "sideways", "volatile", "low_volatility"]

        for model in performance_results["Model"].unique():
            model_data = {"Model": model}

            # Base performance metrics
            base_perf = performance_results[performance_results["Model"] == model].iloc[0]
            model_data["Overall Sharpe"] = base_perf.get("Sharpe Ratio", 0)

            # Regime-specific performance
            regime_data = regime_performance_data.get(model, {})

            for regime in market_regimes:
                regime_metrics = regime_data.get(regime, {})
                model_data[f"{regime.title()} Sharpe"] = regime_metrics.get("sharpe_ratio", 0)
                model_data[f"{regime.title()} Return (%)"] = (
                    regime_metrics.get("annual_return", 0) * 100
                )
                model_data[f"{regime.title()} Drawdown (%)"] = (
                    abs(regime_metrics.get("max_drawdown", 0)) * 100
                )

            # Calculate regime consistency metrics
            regime_sharpes = [
                regime_data.get(regime, {}).get("sharpe_ratio", 0) for regime in market_regimes
            ]
            model_data["Regime Consistency"] = (
                1 - (np.std(regime_sharpes) / np.mean(regime_sharpes))
                if np.mean(regime_sharpes) > 0
                else 0
            )
            model_data["Best Regime"] = (
                market_regimes[np.argmax(regime_sharpes)] if regime_sharpes else "Unknown"
            )
            model_data["Worst Regime"] = (
                market_regimes[np.argmin(regime_sharpes)] if regime_sharpes else "Unknown"
            )

            # Regime adaptation score
            positive_regimes = sum(1 for sharpe in regime_sharpes if sharpe > 0)
            model_data["Regime Adaptation Score"] = positive_regimes / len(market_regimes)

            regime_integrated_results.append(model_data)

        regime_df = pd.DataFrame(regime_integrated_results)

        # Add regime-based rankings
        regime_df["Consistency Rank"] = regime_df["Regime Consistency"].rank(ascending=False)
        regime_df["Adaptation Rank"] = regime_df["Regime Adaptation Score"].rank(ascending=False)

        return regime_df.sort_values("Regime Consistency", ascending=False)

    def generate_implementation_roadmap(
        self,
        recommended_approaches: list[str],
        feasibility_results: pd.DataFrame,
        institutional_constraints: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Generate detailed implementation roadmap with prioritized rollout sequence.

        Args:
            recommended_approaches: List of recommended model approaches
            feasibility_results: Implementation feasibility assessment
            institutional_constraints: Institutional constraints and preferences

        Returns:
            Dictionary containing implementation roadmap and timeline
        """
        roadmap_phases = []

        # Sort approaches by implementation priority
        priority_scores = []
        for approach in recommended_approaches:
            feas_row = feasibility_results[feasibility_results["Model"] == approach].iloc[0]

            # Calculate priority score based on time-to-value, impact, and efficiency
            time_to_value = 1 / float(feas_row.get("Total Timeline (months)", "6").replace(",", ""))
            strategic_impact = float(feas_row.get("Overall Complexity (0-100)", "50")) / 100
            resource_efficiency = 1 / float(
                feas_row.get("Risk-Adjusted TCO ($)", "100000").replace(",", "")
            )

            priority_score = (
                time_to_value * self.config.time_to_value_weight
                + strategic_impact * self.config.strategic_impact_weight
                + resource_efficiency * self.config.resource_efficiency_weight
            )

            priority_scores.append((approach, priority_score))

        # Sort by priority score
        priority_scores.sort(key=lambda x: x[1], reverse=True)
        prioritized_approaches = [approach for approach, _ in priority_scores]

        # Create phased implementation plan
        cumulative_months = 0
        cumulative_cost = 0

        for i, approach in enumerate(prioritized_approaches):
            feas_row = feasibility_results[feasibility_results["Model"] == approach].iloc[0]

            timeline_months = float(feas_row.get("Total Timeline (months)", "6"))
            implementation_cost = float(
                feas_row.get("Total Implementation ($)", "100000").replace(",", "")
            )

            # Phase planning
            phase_number = i + 1
            phase_start = cumulative_months
            phase_end = cumulative_months + timeline_months

            # Parallel implementation considerations
            if i > 0 and timeline_months < 4:  # Can overlap short implementations
                overlap_months = min(2, timeline_months * 0.5)
                phase_start = max(0, phase_start - overlap_months)
                phase_end = phase_start + timeline_months

            cumulative_months = phase_end
            cumulative_cost += implementation_cost

            roadmap_phases.append(
                {
                    "Phase": phase_number,
                    "Approach": approach,
                    "Start Month": f"{phase_start:.1f}",
                    "End Month": f"{phase_end:.1f}",
                    "Duration (months)": f"{timeline_months:.1f}",
                    "Phase Cost ($)": f"{implementation_cost:,.0f}",
                    "Cumulative Cost ($)": f"{cumulative_cost:,.0f}",
                    "Key Milestones": self._generate_phase_milestones(approach, timeline_months),
                    "Success Criteria": self._generate_success_criteria(approach),
                    "Risk Mitigation": self._generate_risk_mitigation(approach),
                }
            )

        # Create monitoring and evaluation framework
        monitoring_framework = {
            "Performance Metrics": [
                "Rolling Sharpe ratio vs baseline",
                "Risk-adjusted returns",
                "Maximum drawdown monitoring",
                "Transaction cost analysis",
                "Model stability indicators",
            ],
            "Review Frequency": "Monthly performance review, Quarterly strategy review",
            "Escalation Triggers": [
                "Sharpe ratio decline > 0.3 for 2+ months",
                "Drawdown exceeding 1.5x historical maximum",
                "Model prediction accuracy decline > 10%",
                "Transaction costs exceeding budget by > 20%",
            ],
            "Success Thresholds": [
                "Sustained Sharpe ratio improvement > 0.2",
                "Risk-adjusted returns exceeding baseline by > 15%",
                "Operational stability > 99.5% uptime",
                "Total implementation cost within 110% of budget",
            ],
        }

        return {
            "implementation_phases": pd.DataFrame(roadmap_phases),
            "total_timeline_months": cumulative_months,
            "total_cost": cumulative_cost,
            "monitoring_framework": monitoring_framework,
            "rollout_strategy": self._create_rollout_strategy(
                prioritized_approaches, institutional_constraints
            ),
            "contingency_planning": self._create_contingency_plans(prioritized_approaches),
        }

    def _calculate_performance_score(
        self, performance_row: pd.Series, risk_tolerance: str
    ) -> float:
        """Calculate performance suitability score based on risk tolerance."""
        sharpe_ratio = float(str(performance_row.get("Sharpe Ratio", "0")).split()[0])
        annual_return = float(str(performance_row.get("Annual Return (%)", "0")).replace("%", ""))

        # Risk tolerance adjustments
        risk_multipliers = {"conservative": 0.8, "moderate": 1.0, "aggressive": 1.2}
        multiplier = risk_multipliers.get(risk_tolerance, 1.0)

        # Performance score (0-100)
        performance_score = min(100, (sharpe_ratio / 2.0) * 50 + (annual_return / 25.0) * 50)
        return performance_score * multiplier

    def _calculate_feasibility_score(
        self, feasibility_row: pd.Series, computational_budget: str
    ) -> float:
        """Calculate feasibility score based on computational constraints."""
        complexity = float(str(feasibility_row.get("Overall Complexity (0-100)", "50")).split()[0])
        timeline = float(str(feasibility_row.get("Total Timeline (months)", "6")).split()[0])

        # Budget adjustments
        budget_multipliers = {"low": 0.7, "medium": 1.0, "high": 1.3}
        multiplier = budget_multipliers.get(computational_budget, 1.0)

        # Feasibility score (0-100, lower complexity/timeline is better)
        feasibility_score = max(0, 100 - complexity - (timeline * 5))
        return min(100, feasibility_score * multiplier)

    def _calculate_risk_score(self, performance_row: pd.Series, risk_tolerance: str) -> float:
        """Calculate risk suitability score."""
        max_drawdown = float(str(performance_row.get("Max Drawdown (%)", "10")).replace("%", ""))
        volatility = float(str(performance_row.get("Volatility (%)", "15")).replace("%", ""))

        # Risk tolerance thresholds
        drawdown_threshold = self.config.risk_tolerance_levels.get(risk_tolerance, 0.25) * 100

        # Risk score (0-100, lower risk metrics are better for risk-averse)
        if risk_tolerance == "conservative":
            risk_score = max(
                0, 100 - (max_drawdown / drawdown_threshold) * 60 - (volatility / 20) * 40
            )
        elif risk_tolerance == "aggressive":
            risk_score = min(
                100, 100 - (max_drawdown / drawdown_threshold) * 20 - (volatility / 25) * 20
            )
        else:  # moderate
            risk_score = max(
                0, 100 - (max_drawdown / drawdown_threshold) * 40 - (volatility / 20) * 30
            )

        return risk_score

    def _calculate_cost_score(self, feasibility_row: pd.Series, aum_millions: float) -> float:
        """Calculate cost effectiveness score based on AUM."""
        tco_str = str(feasibility_row.get("Risk-Adjusted TCO ($)", "100000")).replace(",", "")
        tco = float(tco_str)

        # Cost as percentage of AUM
        cost_percentage = (tco / (aum_millions * 1000000)) * 100

        # Cost score (0-100, lower cost percentage is better)
        cost_score = max(0, 100 - (cost_percentage * 10))
        return min(100, cost_score)

    def _generate_recommendation_category(
        self, perf: float, feas: float, risk: float, cost: float
    ) -> str:
        """Generate recommendation category based on scores."""
        if perf > 80 and feas > 70:
            return "Strongly Recommended"
        elif perf > 60 and feas > 60 and risk > 60:
            return "Recommended"
        elif perf > 40 and feas > 50:
            return "Consider with Caution"
        else:
            return "Not Recommended"

    def _create_decision_logic(
        self, suitability_df: pd.DataFrame, constraints: dict[str, Any]
    ) -> dict[str, str]:
        """Create decision tree logic based on suitability scores."""
        top_model = suitability_df.iloc[0]

        decision_logic = {
            "Primary Decision": f"Deploy {top_model['Model']} as primary approach",
            "Rationale": f"Highest suitability score ({top_model['Overall Suitability']:.1f}/100)",
            "Secondary Options": ", ".join(suitability_df.head(3)["Model"].tolist()[1:]),
            "Key Decision Factors": self._identify_key_factors(suitability_df),
        }

        return decision_logic

    def _identify_key_factors(self, suitability_df: pd.DataFrame) -> list[str]:
        """Identify key decision factors from suitability analysis."""
        factors = []

        # Performance differentiation
        perf_range = (
            suitability_df["Performance Score"].max() - suitability_df["Performance Score"].min()
        )
        if perf_range > 20:
            factors.append("Significant performance differences between approaches")

        # Feasibility constraints
        feas_min = suitability_df["Feasibility Score"].min()
        if feas_min < 50:
            factors.append("Implementation feasibility is a key constraint")

        # Risk considerations
        risk_range = suitability_df["Risk Score"].max() - suitability_df["Risk Score"].min()
        if risk_range > 25:
            factors.append("Risk profile varies significantly between approaches")

        return factors

    def _create_institutional_profile(self, constraints: dict[str, Any]) -> dict[str, str]:
        """Create institutional profile summary."""
        return {
            "Risk Profile": constraints.get("risk_tolerance", "moderate").title(),
            "AUM Category": self._categorize_aum(constraints.get("aum_millions", 500)),
            "Computational Resources": constraints.get("computational_budget", "medium").title(),
            "Regulatory Environment": constraints.get("regulatory_complexity", "standard").title(),
            "Implementation Urgency": self._categorize_timeline(
                constraints.get("preferred_timeline_months", 6)
            ),
        }

    def _categorize_aum(self, aum_millions: float) -> str:
        """Categorize AUM size."""
        if aum_millions < 100:
            return "Small ($<100M)"
        elif aum_millions < 1000:
            return "Medium ($100M-1B)"
        else:
            return "Large ($>1B)"

    def _categorize_timeline(self, months: int) -> str:
        """Categorize implementation timeline urgency."""
        if months < 4:
            return "Urgent (<4 months)"
        elif months < 8:
            return "Standard (4-8 months)"
        else:
            return "Flexible (>8 months)"

    def _prioritize_implementation(
        self, suitability_df: pd.DataFrame, constraints: dict[str, Any]
    ) -> list[str]:
        """Prioritize implementation order."""
        return suitability_df.head(3)["Model"].tolist()

    def _adjust_performance_for_scenario(
        self,
        base_performance: pd.Series,
        scenario_params: dict[str, Any],
        regime_data: dict[str, Any],
    ) -> dict[str, float]:
        """Adjust performance metrics for specific scenario."""
        base_sharpe = float(str(base_performance.get("Sharpe Ratio", "1.0")).split()[0])
        base_return = (
            float(str(base_performance.get("Annual Return (%)", "10")).replace("%", "")) / 100
        )
        base_drawdown = (
            float(str(base_performance.get("Max Drawdown (%)", "15")).replace("%", "")) / 100
        )

        # Regime adjustments
        regime = scenario_params.get("market_regime", "normal")
        regime_multipliers = regime_data.get(
            regime, {"sharpe": 1.0, "return": 1.0, "drawdown": 1.0}
        )

        # Risk appetite adjustments
        risk_appetite = scenario_params.get("risk_appetite", "medium")
        risk_adjustments = {"low": 0.8, "medium": 1.0, "high": 1.2}
        risk_mult = risk_adjustments.get(risk_appetite, 1.0)

        adjusted_sharpe = base_sharpe * regime_multipliers.get("sharpe", 1.0) * risk_mult
        adjusted_return = base_return * regime_multipliers.get("return", 1.0) * risk_mult
        adjusted_drawdown = base_drawdown * regime_multipliers.get("drawdown", 1.0)

        # Calculate scenario-specific metrics
        regime_suitability = self._calculate_regime_suitability(regime, base_performance)
        risk_adjusted_score = (adjusted_sharpe * 0.6 + (adjusted_return / 0.25) * 0.4) * (
            1 - adjusted_drawdown
        )

        return {
            "adjusted_sharpe": adjusted_sharpe,
            "adjusted_return": adjusted_return,
            "adjusted_drawdown": adjusted_drawdown,
            "regime_suitability": regime_suitability,
            "risk_adjusted_score": risk_adjusted_score,
        }

    def _calculate_regime_suitability(self, regime: str, performance: pd.Series) -> str:
        """Calculate suitability for specific market regime."""
        # Simplified regime suitability logic
        regime_preferences = {
            "bull": "High return strategies preferred",
            "bear": "Defensive strategies preferred",
            "volatile": "Adaptive strategies preferred",
            "rising_rates": "Duration-neutral strategies preferred",
            "low_vol": "Momentum strategies preferred",
        }
        return regime_preferences.get(regime, "Standard suitability")

    def _generate_phase_milestones(self, approach: str, duration_months: float) -> list[str]:
        """Generate phase-specific milestones."""
        milestones = [
            f"Month {duration_months * 0.25:.1f}: Data pipeline complete",
            f"Month {duration_months * 0.5:.1f}: Model development complete",
            f"Month {duration_months * 0.75:.1f}: Testing and validation complete",
            f"Month {duration_months:.1f}: Production deployment complete",
        ]
        return milestones

    def _generate_success_criteria(self, approach: str) -> list[str]:
        """Generate approach-specific success criteria."""
        return [
            "Model performance meets or exceeds backtesting projections",
            "System uptime > 99.5%",
            "Implementation timeline within 110% of plan",
            "Budget variance < 15%",
            "Risk metrics within acceptable parameters",
        ]

    def _generate_risk_mitigation(self, approach: str) -> list[str]:
        """Generate risk mitigation strategies."""
        return [
            "Parallel development tracks for critical components",
            "Regular model performance monitoring and alerts",
            "Fallback to baseline strategy if performance degrades",
            "Staged rollout with gradual capital allocation",
            "Technical and business stakeholder review gates",
        ]

    def _create_rollout_strategy(
        self, approaches: list[str], constraints: dict[str, Any]
    ) -> dict[str, str]:
        """Create detailed rollout strategy."""
        return {
            "Initial Deployment": f"Start with {approaches[0]} at 25% target allocation",
            "Scale-up Timeline": "Increase to 50% allocation after 3 months of stable performance",
            "Full Deployment": "Reach 100% target allocation after 6 months",
            "Parallel Development": f"Begin {approaches[1]} development during {approaches[0]} stabilization",
            "Performance Gates": "Each scale-up requires monthly Sharpe ratio > baseline + 0.1",
        }

    def _create_contingency_plans(self, approaches: list[str]) -> dict[str, str]:
        """Create contingency planning framework."""
        return {
            "Performance Degradation": f"Revert to baseline strategy, accelerate {approaches[1]} development",
            "Technical Failure": "Automated fallback to equal-weight baseline within 24 hours",
            "Resource Constraints": f"Reduce scope to {approaches[0]} only, extend timeline",
            "Regulatory Changes": "Pause deployment, conduct compliance review, adjust approach",
            "Market Regime Shift": "Activate regime-specific model parameters, monitor performance",
        }
