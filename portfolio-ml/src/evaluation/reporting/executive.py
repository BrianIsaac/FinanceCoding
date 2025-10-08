"""
Executive summary report generation for comprehensive performance analysis.

This module creates management-ready executive summaries with key performance rankings,
operational feasibility scoring, and strategic recommendations based on comprehensive
model evaluation results.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("Matplotlib/Seaborn not available. Static plotting disabled.", stacklevel=2)

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    warnings.warn("Plotly not available. Interactive plotting disabled.", stacklevel=2)


@dataclass
class ExecutiveConfig:
    """Configuration for executive summary generation."""

    figsize: tuple[int, int] = (12, 8)
    dpi: int = 300
    style: str = "whitegrid"
    color_palette: str = "Set2"
    significance_threshold: float = 0.05
    top_n_models: int = 5
    decimal_places: int = 3


class ExecutiveSummaryGenerator:
    """
    Executive summary generator for comprehensive model evaluation.

    Creates management-ready summaries with performance rankings, feasibility assessment,
    and strategic recommendations for institutional deployment.
    """

    def __init__(self, config: ExecutiveConfig = None):
        """
        Initialize executive summary generator.

        Args:
            config: Executive summary configuration
        """
        self.config = config or ExecutiveConfig()
        self._setup_plotting()

    def _setup_plotting(self) -> None:
        """Setup plotting configurations."""
        if HAS_MATPLOTLIB:
            plt.style.use("seaborn-v0_8-whitegrid")
            sns.set_palette(self.config.color_palette)

    def generate_executive_dashboard(
        self,
        performance_results: dict[str, pd.DataFrame],
        operational_metrics: dict[str, dict[str, Any]],
        statistical_results: dict[str, Any],
        output_dir: Path = None,
    ) -> dict[str, Any]:
        """
        Generate executive dashboard with key performance rankings.

        Args:
            performance_results: Dictionary containing performance metrics by model
            operational_metrics: Operational efficiency metrics by model
            statistical_results: Statistical significance testing results
            output_dir: Optional output directory for saving dashboard

        Returns:
            Dictionary containing dashboard data and visualizations
        """
        # Calculate key performance rankings
        rankings = self._calculate_performance_rankings(performance_results, statistical_results)

        # Calculate operational feasibility scores
        feasibility_scores = self._calculate_feasibility_scores(operational_metrics)

        # Create executive dashboard visualization
        dashboard_fig = self._create_executive_dashboard_plot(rankings, feasibility_scores)

        # Generate summary statistics
        summary_stats = self._generate_summary_statistics(
            rankings, feasibility_scores, statistical_results
        )

        dashboard_data = {
            "rankings": rankings,
            "feasibility_scores": feasibility_scores,
            "summary_statistics": summary_stats,
            "dashboard_figure": dashboard_fig,
        }

        if output_dir:
            self._save_executive_dashboard(dashboard_data, output_dir)

        return dashboard_data

    def _calculate_performance_rankings(
        self, performance_results: dict[str, pd.DataFrame], statistical_results: dict[str, Any]
    ) -> pd.DataFrame:
        """
        Calculate comprehensive performance rankings.

        Args:
            performance_results: Performance metrics by model
            statistical_results: Statistical significance results

        Returns:
            DataFrame with performance rankings and metrics
        """
        ranking_data = []

        for model_name, results in performance_results.items():
            if results.empty:
                continue

            # Extract key metrics
            sharpe_ratio = results["sharpe_ratio"].mean()
            information_ratio = results["information_ratio"].mean()
            cagr = results["annual_return"].mean()
            max_drawdown = results["max_drawdown"].mean()
            volatility = results["volatility"].mean()

            # Calculate statistical significance indicators
            is_significant = self._check_statistical_significance(model_name, statistical_results)

            ranking_data.append(
                {
                    "Model": model_name,
                    "Sharpe Ratio": sharpe_ratio,
                    "Information Ratio": information_ratio,
                    "CAGR (%)": cagr * 100,
                    "Max Drawdown (%)": abs(max_drawdown) * 100,
                    "Volatility (%)": volatility * 100,
                    "Statistically Significant": is_significant,
                }
            )

        rankings_df = pd.DataFrame(ranking_data)

        # Calculate composite score (weighted combination of metrics)
        if not rankings_df.empty:
            rankings_df["Composite Score"] = (
                0.4 * rankings_df["Sharpe Ratio"]
                + 0.3 * rankings_df["Information Ratio"]
                + 0.2 * (rankings_df["CAGR (%)"] / 100)
                - 0.1 * (rankings_df["Max Drawdown (%)"] / 100)
            )

            # Rank by composite score
            rankings_df["Overall Rank"] = rankings_df["Composite Score"].rank(ascending=False)
            rankings_df = rankings_df.sort_values("Overall Rank")

        return rankings_df

    def _calculate_feasibility_scores(
        self, operational_metrics: dict[str, dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Calculate operational feasibility scores.

        Args:
            operational_metrics: Operational metrics by model

        Returns:
            DataFrame with feasibility scores and components
        """
        feasibility_data = []

        for model_name, metrics in operational_metrics.items():
            # Computational requirements (lower is better)
            gpu_memory = metrics.get("gpu_memory_gb", 8.0)
            training_time = metrics.get("training_time_hours", 2.0)
            inference_cost = metrics.get("inference_cost_per_prediction", 0.01)

            # Normalize computational scores (0-100, higher is better)
            gpu_score = max(0, 100 - (gpu_memory / 12.0) * 100)  # 12GB max
            time_score = max(0, 100 - (training_time / 24.0) * 100)  # 24 hours max
            cost_score = max(0, 100 - (inference_cost / 0.1) * 100)  # $0.10 max

            # Operational complexity (lower is better)
            turnover = metrics.get("average_turnover", 0.2)
            transaction_costs = metrics.get("transaction_cost_bps", 10.0)
            monitoring_complexity = metrics.get("monitoring_score", 50)  # 0-100 scale

            # Normalize operational scores (0-100, higher is better)
            turnover_score = max(0, 100 - (turnover * 200))  # 50% max
            cost_impact_score = max(0, 100 - (transaction_costs / 20.0) * 100)  # 20 bps max
            simplicity_score = 100 - monitoring_complexity

            # Calculate overall feasibility score
            feasibility_score = (
                0.25 * gpu_score
                + 0.20 * time_score
                + 0.15 * cost_score
                + 0.20 * turnover_score
                + 0.15 * cost_impact_score
                + 0.05 * simplicity_score
            )

            feasibility_data.append(
                {
                    "Model": model_name,
                    "Feasibility Score": feasibility_score,
                    "GPU Memory Score": gpu_score,
                    "Training Time Score": time_score,
                    "Inference Cost Score": cost_score,
                    "Turnover Score": turnover_score,
                    "Transaction Cost Score": cost_impact_score,
                    "Simplicity Score": simplicity_score,
                    "GPU Memory (GB)": gpu_memory,
                    "Training Time (hours)": training_time,
                    "Turnover (%)": turnover * 100,
                    "Transaction Costs (bps)": transaction_costs,
                }
            )

        feasibility_df = pd.DataFrame(feasibility_data)

        if not feasibility_df.empty:
            feasibility_df["Feasibility Rank"] = feasibility_df["Feasibility Score"].rank(
                ascending=False
            )
            feasibility_df = feasibility_df.sort_values("Feasibility Rank")

        return feasibility_df

    def _check_statistical_significance(
        self, model_name: str, statistical_results: dict[str, Any]
    ) -> bool:
        """
        Check if model performance is statistically significant.

        Args:
            model_name: Name of the model
            statistical_results: Statistical test results

        Returns:
            True if model is statistically significant vs baseline
        """
        if not statistical_results:
            return False

        model_tests = statistical_results.get(model_name, {})
        pvalue = model_tests.get("pvalue_vs_baseline", 1.0)

        return pvalue < self.config.significance_threshold

    def _create_executive_dashboard_plot(
        self, rankings: pd.DataFrame, feasibility_scores: pd.DataFrame
    ) -> go.Figure:
        """
        Create executive dashboard visualization.

        Args:
            rankings: Performance rankings DataFrame
            feasibility_scores: Feasibility scores DataFrame

        Returns:
            Plotly figure for executive dashboard
        """
        if not HAS_PLOTLY or rankings.empty or feasibility_scores.empty:
            return None

        # Merge data for plotting
        merged_data = rankings.merge(
            feasibility_scores[["Model", "Feasibility Score"]], on="Model", how="inner"
        )

        # Create scatter plot
        fig = go.Figure()

        # Add scatter points for each model
        for _idx, row in merged_data.iterrows():
            color = "red" if row["Statistically Significant"] else "blue"
            symbol = "star" if row["Statistically Significant"] else "circle"

            fig.add_trace(
                go.Scatter(
                    x=[row["Feasibility Score"]],
                    y=[row["Sharpe Ratio"]],
                    mode="markers+text",
                    name=row["Model"],
                    text=[row["Model"]],
                    textposition="top center",
                    marker={"size": 15, "color": color, "symbol": symbol},
                    hovertemplate=f"<b>{row['Model']}</b><br>"
                    + f"Sharpe Ratio: {row['Sharpe Ratio']:.3f}<br>"
                    + f"Feasibility Score: {row['Feasibility Score']:.1f}<br>"
                    + f"CAGR: {row['CAGR (%)']:.1f}%<br>"
                    + f"Max Drawdown: {row['Max Drawdown (%)']:.1f}%<br>"
                    + f"Significant: {row['Statistically Significant']}<br>"
                    + "<extra></extra>",
                )
            )

        # Add quadrant lines
        x_median = merged_data["Feasibility Score"].median()
        y_median = merged_data["Sharpe Ratio"].median()

        fig.add_hline(y=y_median, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=x_median, line_dash="dash", line_color="gray", opacity=0.5)

        # Add quadrant labels
        fig.add_annotation(
            x=x_median + 10,
            y=merged_data["Sharpe Ratio"].max() * 0.95,
            text="High Performance<br>High Feasibility",
            showarrow=False,
        )
        fig.add_annotation(
            x=x_median - 10,
            y=merged_data["Sharpe Ratio"].max() * 0.95,
            text="High Performance<br>Low Feasibility",
            showarrow=False,
        )

        fig.update_layout(
            title="Executive Dashboard: Performance vs Implementation Feasibility",
            xaxis_title="Implementation Feasibility Score",
            yaxis_title="Sharpe Ratio",
            height=600,
            showlegend=False,
            template="plotly_white",
        )

        return fig

    def _generate_summary_statistics(
        self,
        rankings: pd.DataFrame,
        feasibility_scores: pd.DataFrame,
        statistical_results: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Generate executive summary statistics.

        Args:
            rankings: Performance rankings
            feasibility_scores: Feasibility scores
            statistical_results: Statistical test results

        Returns:
            Dictionary with summary statistics
        """
        if rankings.empty:
            return {}

        # Top performing models
        top_models = rankings.head(self.config.top_n_models)

        # Statistical significance summary
        significant_models = rankings[rankings["Statistically Significant"]]["Model"].tolist()

        # Best combination of performance and feasibility
        merged_data = rankings.merge(
            feasibility_scores[["Model", "Feasibility Score"]], on="Model", how="inner"
        )
        if not merged_data.empty:
            merged_data["Combined Score"] = (
                merged_data["Composite Score"] * 0.7 + merged_data["Feasibility Score"] / 100 * 0.3
            )
            recommended_model = merged_data.loc[merged_data["Combined Score"].idxmax(), "Model"]
        else:
            recommended_model = top_models.iloc[0]["Model"] if not top_models.empty else "N/A"

        summary = {
            "total_models_evaluated": len(rankings),
            "statistically_significant_models": len(significant_models),
            "top_performer": top_models.iloc[0]["Model"] if not top_models.empty else "N/A",
            "best_sharpe_ratio": (
                top_models.iloc[0]["Sharpe Ratio"] if not top_models.empty else 0.0
            ),
            "recommended_model": recommended_model,
            "significant_models": significant_models,
            "average_sharpe_improvement": rankings["Sharpe Ratio"].mean()
            - 1.0,  # Assuming baseline of 1.0
            "models_above_baseline": len(rankings[rankings["Sharpe Ratio"] > 1.0]),
        }

        return summary

    def generate_recommendation_matrix(
        self,
        rankings: pd.DataFrame,
        feasibility_scores: pd.DataFrame,
        institutional_constraints: dict[str, Any] = None,
    ) -> pd.DataFrame:
        """
        Generate recommendation matrix linking approaches to institutional use cases.

        Args:
            rankings: Performance rankings
            feasibility_scores: Feasibility scores
            institutional_constraints: Institutional constraint specifications

        Returns:
            DataFrame with recommendation matrix
        """
        if rankings.empty or feasibility_scores.empty:
            return pd.DataFrame()

        # Default institutional constraints
        if institutional_constraints is None:
            institutional_constraints = {
                "risk_tolerance": "medium",  # low, medium, high
                "computational_budget": "medium",  # low, medium, high
                "implementation_timeline": "medium",  # short, medium, long
                "regulatory_requirements": "standard",  # light, standard, strict
            }

        recommendations = []

        # Merge performance and feasibility data
        merged_data = rankings.merge(
            feasibility_scores[["Model", "Feasibility Score"]], on="Model", how="inner"
        )

        for _, row in merged_data.iterrows():
            model = row["Model"]

            # Calculate suitability scores for different use cases
            use_cases = {
                "Conservative Institution": self._score_conservative_suitability(row),
                "Aggressive Hedge Fund": self._score_aggressive_suitability(row),
                "Large Asset Manager": self._score_large_manager_suitability(row),
                "Boutique Fund": self._score_boutique_suitability(row),
                "Family Office": self._score_family_office_suitability(row),
            }

            for use_case, score in use_cases.items():
                recommendations.append(
                    {
                        "Model": model,
                        "Use Case": use_case,
                        "Suitability Score": score,
                        "Primary Strength": self._identify_primary_strength(row),
                        "Key Consideration": self._identify_key_consideration(row),
                    }
                )

        recommendation_df = pd.DataFrame(recommendations)
        if not recommendation_df.empty:
            recommendation_df = recommendation_df.sort_values(
                ["Use Case", "Suitability Score"], ascending=[True, False]
            )

        return recommendation_df

    def _score_conservative_suitability(self, model_data: pd.Series) -> float:
        """Score model suitability for conservative institutions."""
        score = (
            0.3 * min(model_data["Sharpe Ratio"] / 2.0, 1.0)  # Cap at 2.0 Sharpe
            + 0.25 * (1 - model_data["Max Drawdown (%)"] / 20.0)  # Penalize high drawdowns
            + 0.25 * (model_data["Feasibility Score"] / 100)
            + 0.2 * (1 if model_data["Statistically Significant"] else 0.5)
        )
        return max(0, min(score, 1.0)) * 100

    def _score_aggressive_suitability(self, model_data: pd.Series) -> float:
        """Score model suitability for aggressive hedge funds."""
        score = (
            0.4 * min(model_data["Sharpe Ratio"] / 3.0, 1.0)  # Higher Sharpe preference
            + 0.3 * min(model_data["CAGR (%)"] / 30.0, 1.0)  # High return preference
            + 0.2 * (1 if model_data["Statistically Significant"] else 0.7)
            + 0.1 * (model_data["Feasibility Score"] / 100)  # Less concerned with feasibility
        )
        return max(0, min(score, 1.0)) * 100

    def _score_large_manager_suitability(self, model_data: pd.Series) -> float:
        """Score model suitability for large asset managers."""
        score = (
            0.25 * min(model_data["Sharpe Ratio"] / 2.0, 1.0)
            + 0.35 * (model_data["Feasibility Score"] / 100)  # High feasibility importance
            + 0.25 * (1 if model_data["Statistically Significant"] else 0.3)
            + 0.15 * (1 - model_data["Max Drawdown (%)"] / 25.0)
        )
        return max(0, min(score, 1.0)) * 100

    def _score_boutique_suitability(self, model_data: pd.Series) -> float:
        """Score model suitability for boutique funds."""
        score = (
            0.35 * min(model_data["Sharpe Ratio"] / 2.5, 1.0)
            + 0.25 * (1 - (100 - model_data["Feasibility Score"]) / 100)  # Prefer simple models
            + 0.25 * (1 if model_data["Statistically Significant"] else 0.5)
            + 0.15 * min(model_data["CAGR (%)"] / 25.0, 1.0)
        )
        return max(0, min(score, 1.0)) * 100

    def _score_family_office_suitability(self, model_data: pd.Series) -> float:
        """Score model suitability for family offices."""
        score = (
            0.3 * min(model_data["Sharpe Ratio"] / 1.8, 1.0)  # Moderate Sharpe preference
            + 0.3 * (1 - model_data["Max Drawdown (%)"] / 15.0)  # Low drawdown preference
            + 0.25 * (model_data["Feasibility Score"] / 100)
            + 0.15 * (1 if model_data["Statistically Significant"] else 0.6)
        )
        return max(0, min(score, 1.0)) * 100

    def _identify_primary_strength(self, model_data: pd.Series) -> str:
        """Identify the primary strength of a model."""
        if model_data["Sharpe Ratio"] > 2.0:
            return "Exceptional Risk-Adjusted Returns"
        elif model_data["CAGR (%)"] > 25:
            return "High Absolute Returns"
        elif model_data["Max Drawdown (%)"] < 10:
            return "Low Downside Risk"
        elif model_data["Feasibility Score"] > 80:
            return "High Implementation Feasibility"
        else:
            return "Balanced Performance"

    def _identify_key_consideration(self, model_data: pd.Series) -> str:
        """Identify key implementation consideration for a model."""
        if model_data["Feasibility Score"] < 50:
            return "High Implementation Complexity"
        elif model_data["Max Drawdown (%)"] > 20:
            return "Significant Drawdown Risk"
        elif not model_data["Statistically Significant"]:
            return "Statistical Significance Uncertain"
        elif model_data["Volatility (%)"] > 20:
            return "High Volatility"
        else:
            return "Monitor Model Stability"

    def create_management_summary(
        self,
        rankings: pd.DataFrame,
        feasibility_scores: pd.DataFrame,
        statistical_results: dict[str, Any],
        output_path: Path = None,
    ) -> str:
        """
        Create one-page management summary with key findings and recommendations.

        Args:
            rankings: Performance rankings
            feasibility_scores: Feasibility scores
            statistical_results: Statistical test results
            output_path: Optional path to save summary

        Returns:
            Formatted string containing management summary
        """
        if rankings.empty:
            return "No performance data available for summary generation."

        # Generate summary statistics
        summary_stats = self._generate_summary_statistics(
            rankings, feasibility_scores, statistical_results
        )

        # Create formatted summary
        summary_text = f"""
EXECUTIVE SUMMARY: ML-Enhanced Portfolio Construction Analysis

PERFORMANCE OVERVIEW
===================
• Total Models Evaluated: {summary_stats['total_models_evaluated']}
• Statistically Significant Approaches: {summary_stats['statistically_significant_models']}
• Top Performing Model: {summary_stats['top_performer']}
• Best Sharpe Ratio Achieved: {summary_stats['best_sharpe_ratio']:.3f}

KEY FINDINGS
============
• {summary_stats['models_above_baseline']} of {summary_stats['total_models_evaluated']} models outperformed baseline
• Average Sharpe ratio improvement: {summary_stats['average_sharpe_improvement']:.3f}
• Statistically significant models: {', '.join(summary_stats['significant_models'])}

STRATEGIC RECOMMENDATION
=======================
Recommended Primary Deployment: {summary_stats['recommended_model']}

This recommendation balances superior risk-adjusted returns with implementation
feasibility, providing the optimal combination for institutional deployment.

IMPLEMENTATION PRIORITIES
========================
1. IMMEDIATE: Implement {summary_stats['recommended_model']} for core allocation
2. PILOT: Test {summary_stats['top_performer']} with limited allocation
3. RESEARCH: Continue monitoring {len(summary_stats['significant_models'])} significant approaches

RISK CONSIDERATIONS
==================
• All recommendations based on {len(rankings)} model rigorous statistical testing
• {summary_stats['statistically_significant_models']} models show statistical significance vs baseline
• Continuous monitoring required for model performance degradation
• Implementation timeline: 3-6 months for primary recommendation

NEXT STEPS
==========
1. Approve recommended approach for pilot implementation
2. Allocate development resources for integration
3. Establish monitoring framework for ongoing evaluation
4. Plan gradual rollout with risk management oversight

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        """

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(summary_text)

        return summary_text

    def _save_executive_dashboard(self, dashboard_data: dict[str, Any], output_dir: Path) -> None:
        """
        Save executive dashboard components to files.

        Args:
            dashboard_data: Dashboard data and visualizations
            output_dir: Output directory for saving files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save rankings table
        rankings_path = output_dir / "executive_rankings.csv"
        dashboard_data["rankings"].to_csv(rankings_path, index=False)

        # Save feasibility scores
        feasibility_path = output_dir / "feasibility_scores.csv"
        dashboard_data["feasibility_scores"].to_csv(feasibility_path, index=False)

        # Save dashboard plot
        if dashboard_data["dashboard_figure"] and HAS_PLOTLY:
            plot_path = output_dir / "executive_dashboard.html"
            dashboard_data["dashboard_figure"].write_html(plot_path)

        # Save summary statistics
        summary_path = output_dir / "executive_summary_stats.json"
        import json

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(dashboard_data["summary_statistics"], f, indent=2)
