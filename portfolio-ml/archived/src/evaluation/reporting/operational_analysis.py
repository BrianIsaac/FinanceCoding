"""
Operational efficiency analysis visualization framework.

This module provides comprehensive operational analysis including turnover analysis,
transaction cost impact, constraint compliance monitoring, and implementation
shortfall analysis with realistic trading costs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("Matplotlib/Seaborn not available. Static plotting disabled.", stacklevel=2)

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    warnings.warn("Plotly not available. Interactive plotting disabled.", stacklevel=2)


@dataclass
class OperationalConfig:
    """Configuration for operational analysis visualizations."""

    figsize: tuple[int, int] = (12, 8)
    dpi: int = 300
    style: str = "whitegrid"
    color_palette: str = "husl"
    transaction_cost_basis_points: float = 5.0  # Default transaction cost in bps
    compliance_threshold: float = 0.05  # 5% threshold for constraint violations


class OperationalEfficiencyAnalysis:
    """
    Operational efficiency analysis visualization framework.

    Provides comprehensive operational analysis including turnover, transaction costs,
    constraint compliance, and implementation shortfall analysis.
    """

    def __init__(self, config: OperationalConfig = None):
        """
        Initialize operational efficiency analysis framework.

        Args:
            config: Configuration for operational analysis behavior
        """
        self.config = config or OperationalConfig()

        if HAS_MATPLOTLIB:
            sns.set_style(self.config.style)
            plt.rcParams["figure.dpi"] = self.config.dpi

    def create_turnover_analysis_chart(
        self,
        turnover_data: dict[str, pd.Series],
        performance_data: dict[str, dict[str, float]] | None = None,
        save_path: str | Path | None = None,
        interactive: bool = True,
    ) -> plt.Figure | go.Figure:
        """
        Build turnover analysis charts comparing all approaches.

        Args:
            turnover_data: Dictionary mapping approach names to turnover time series
            performance_data: Optional performance metrics for correlation analysis
            save_path: Path to save the chart
            interactive: Whether to create interactive plot

        Returns:
            Matplotlib Figure or Plotly Figure
        """
        if interactive and HAS_PLOTLY:
            return self._create_turnover_chart_interactive(
                turnover_data, performance_data, save_path
            )
        elif HAS_MATPLOTLIB:
            return self._create_turnover_chart_static(turnover_data, performance_data, save_path)
        else:
            raise ImportError("Neither Plotly nor Matplotlib available for plotting")

    def _create_turnover_chart_interactive(
        self,
        turnover_data: dict[str, pd.Series],
        performance_data: dict[str, dict[str, float]] | None,
        save_path: str | Path | None,
    ) -> go.Figure:
        """Create interactive turnover analysis chart using Plotly."""
        # Create subplots: time series and distribution
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Turnover Time Series",
                "Turnover Distribution",
                "Turnover vs Performance",
                "Rolling Statistics",
            ),
            specs=[[{"colspan": 2}, None], [{"type": "histogram"}, {"type": "scatter"}]],
            vertical_spacing=0.1,
        )

        colors = px.colors.qualitative.Set2

        # 1. Time series plot (top, full width)
        for idx, (approach, turnover_series) in enumerate(turnover_data.items()):
            fig.add_trace(
                go.Scatter(
                    x=turnover_series.index,
                    y=turnover_series.values,
                    mode="lines+markers",
                    name=approach,
                    line={"color": colors[idx % len(colors)], "width": 2},
                    marker={"size": 4},
                    hovertemplate=f"<b>{approach}</b><br>"
                    + "Date: %{x}<br>"
                    + "Turnover: %{y:.2%}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # 2. Distribution plot (bottom left)
        for idx, (approach, turnover_series) in enumerate(turnover_data.items()):
            fig.add_trace(
                go.Histogram(
                    x=turnover_series.values,
                    name=f"{approach} Distribution",
                    opacity=0.7,
                    histnorm="probability density",
                    showlegend=False,
                    marker_color=colors[idx % len(colors)],
                ),
                row=2,
                col=1,
            )

        # 3. Turnover vs Performance scatter (bottom right)
        if performance_data:
            for idx, approach in enumerate(turnover_data.keys()):
                if approach in performance_data:
                    avg_turnover = turnover_data[approach].mean()
                    sharpe_ratio = performance_data[approach].get("sharpe_ratio", 0)

                    fig.add_trace(
                        go.Scatter(
                            x=[avg_turnover],
                            y=[sharpe_ratio],
                            mode="markers+text",
                            text=[approach],
                            textposition="top center",
                            marker={"size": 12, "color": colors[idx % len(colors)]},
                            name=f"{approach} Efficiency",
                            showlegend=False,
                            hovertemplate=f"<b>{approach}</b><br>"
                            + "Avg Turnover: %{x:.2%}<br>"
                            + "Sharpe Ratio: %{y:.3f}<extra></extra>",
                        ),
                        row=2,
                        col=2,
                    )

        # Update layout
        fig.update_layout(title="Turnover Analysis Dashboard", height=800)

        # Update axes
        fig.update_yaxes(title_text="Turnover", tickformat=".1%", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Turnover", tickformat=".1%", row=2, col=1)
        fig.update_yaxes(title_text="Density", row=2, col=1)

        if performance_data:
            fig.update_xaxes(title_text="Average Turnover", tickformat=".1%", row=2, col=2)
            fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=2)

        if save_path:
            fig.write_html(str(save_path).replace(".png", ".html"))

        return fig

    def _create_turnover_chart_static(
        self,
        turnover_data: dict[str, pd.Series],
        performance_data: dict[str, dict[str, float]] | None,
        save_path: str | Path | None,
    ) -> plt.Figure:
        """Create static turnover analysis chart using Matplotlib."""
        fig, axes = plt.subplots(2, 2, figsize=self.config.figsize)
        fig.suptitle("Turnover Analysis Dashboard", fontsize=16, fontweight="bold")

        colors = sns.color_palette(self.config.color_palette, len(turnover_data))

        # 1. Time series plot (top left)
        ax1 = axes[0, 0]
        for idx, (approach, turnover_series) in enumerate(turnover_data.items()):
            ax1.plot(
                turnover_series.index,
                turnover_series.values,
                label=approach,
                color=colors[idx],
                linewidth=2,
                marker="o",
                markersize=3,
            )

        ax1.set_title("Turnover Time Series")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Turnover")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

        # 2. Distribution plot (top right)
        ax2 = axes[0, 1]
        turnover_values_list = [
            turnover_series.values for turnover_series in turnover_data.values()
        ]
        labels = list(turnover_data.keys())

        ax2.hist(turnover_values_list, bins=20, alpha=0.7, label=labels, color=colors, density=True)
        ax2.set_title("Turnover Distribution")
        ax2.set_xlabel("Turnover")
        ax2.set_ylabel("Density")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

        # 3. Summary statistics (bottom left)
        ax3 = axes[1, 0]
        summary_stats = []
        approaches = []

        for approach, turnover_series in turnover_data.items():
            approaches.append(approach)
            summary_stats.append(
                [
                    turnover_series.mean(),
                    turnover_series.std(),
                    turnover_series.median(),
                    turnover_series.max(),
                ]
            )

        summary_df = pd.DataFrame(
            summary_stats, index=approaches, columns=["Mean", "Std", "Median", "Max"]
        )

        # Create grouped bar chart
        x = np.arange(len(approaches))
        width = 0.2

        for i, col in enumerate(summary_df.columns):
            ax3.bar(x + i * width, summary_df[col], width, label=col, alpha=0.8)

        ax3.set_title("Turnover Summary Statistics")
        ax3.set_xlabel("Approach")
        ax3.set_ylabel("Turnover")
        ax3.set_xticks(x + width * 1.5)
        ax3.set_xticklabels(approaches, rotation=45, ha="right")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

        # 4. Turnover vs Performance (bottom right)
        ax4 = axes[1, 1]
        if performance_data:
            turnover_means = []
            sharpe_ratios = []
            labels = []

            for approach in turnover_data.keys():
                if approach in performance_data:
                    turnover_means.append(turnover_data[approach].mean())
                    sharpe_ratios.append(performance_data[approach].get("sharpe_ratio", 0))
                    labels.append(approach)

            ax4.scatter(
                turnover_means,
                sharpe_ratios,
                c=colors[: len(labels)],
                s=100,
                alpha=0.7,
                edgecolors="black",
            )

            for i, label in enumerate(labels):
                ax4.annotate(
                    label,
                    (turnover_means[i], sharpe_ratios[i]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9,
                )

            ax4.set_title("Turnover vs Performance Efficiency")
            ax4.set_xlabel("Average Turnover")
            ax4.set_ylabel("Sharpe Ratio")
            ax4.grid(True, alpha=0.3)
            ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
        else:
            ax4.text(
                0.5,
                0.5,
                "Performance data not available",
                ha="center",
                va="center",
                transform=ax4.transAxes,
            )
            ax4.set_title("Turnover vs Performance Efficiency")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")

        return fig

    def create_transaction_cost_impact_visualization(
        self,
        returns_data: dict[str, pd.Series],
        turnover_data: dict[str, pd.Series],
        cost_basis_points: float | None = None,
        save_path: str | Path | None = None,
        interactive: bool = True,
    ) -> plt.Figure | go.Figure:
        """
        Create transaction cost impact visualization framework.

        Args:
            returns_data: Dictionary mapping approach names to return series
            turnover_data: Dictionary mapping approach names to turnover series
            cost_basis_points: Transaction cost in basis points (default from config)
            save_path: Path to save the visualization
            interactive: Whether to create interactive plot

        Returns:
            Matplotlib Figure or Plotly Figure
        """
        cost_bps = cost_basis_points or self.config.transaction_cost_basis_points
        cost_rate = cost_bps / 10000.0  # Convert basis points to decimal

        # Calculate returns before and after transaction costs
        results = {}
        for approach in returns_data.keys():
            if approach in turnover_data:
                gross_returns = returns_data[approach]
                turnover_series = turnover_data[approach]

                # Align series
                aligned_returns, aligned_turnover = gross_returns.align(
                    turnover_series, join="inner"
                )

                # Calculate transaction costs
                transaction_costs = aligned_turnover * cost_rate
                net_returns = aligned_returns - transaction_costs

                results[approach] = {
                    "gross_returns": aligned_returns,
                    "net_returns": net_returns,
                    "transaction_costs": transaction_costs,
                    "turnover": aligned_turnover,
                }

        if interactive and HAS_PLOTLY:
            return self._create_transaction_cost_interactive(results, cost_bps, save_path)
        elif HAS_MATPLOTLIB:
            return self._create_transaction_cost_static(results, cost_bps, save_path)
        else:
            raise ImportError("Neither Plotly nor Matplotlib available for plotting")

    def _create_transaction_cost_interactive(
        self,
        results: dict[str, dict[str, pd.Series]],
        cost_bps: float,
        save_path: str | Path | None,
    ) -> go.Figure:
        """Create interactive transaction cost impact visualization."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Cumulative Returns: Gross vs Net",
                "Transaction Cost Impact",
                "Cost Distribution by Approach",
                "Performance Impact Summary",
            ),
            specs=[[{"colspan": 2}, None], [{"type": "histogram"}, {"type": "bar"}]],
            vertical_spacing=0.12,
        )

        colors = px.colors.qualitative.Set2

        # 1. Cumulative returns comparison (top, full width)
        for idx, (approach, data) in enumerate(results.items()):
            gross_cum = (1 + data["gross_returns"]).cumprod()
            net_cum = (1 + data["net_returns"]).cumprod()

            color = colors[idx % len(colors)]

            # Gross returns
            fig.add_trace(
                go.Scatter(
                    x=gross_cum.index,
                    y=gross_cum.values,
                    mode="lines",
                    name=f"{approach} (Gross)",
                    line={"color": color, "width": 2, "dash": "solid"},
                    hovertemplate=f"<b>{approach} Gross</b><br>"
                    + "Date: %{x}<br>"
                    + "Cumulative Return: %{y:.2%}<extra></extra>",
                ),
                row=1,
                col=1,
            )

            # Net returns
            fig.add_trace(
                go.Scatter(
                    x=net_cum.index,
                    y=net_cum.values,
                    mode="lines",
                    name=f"{approach} (Net)",
                    line={"color": color, "width": 2, "dash": "dash"},
                    hovertemplate=f"<b>{approach} Net</b><br>"
                    + "Date: %{x}<br>"
                    + "Cumulative Return: %{y:.2%}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # 2. Transaction cost distribution (bottom left)
        for idx, (approach, data) in enumerate(results.items()):
            fig.add_trace(
                go.Histogram(
                    x=data["transaction_costs"].values * 10000,  # Convert to basis points
                    name=f"{approach} Costs",
                    opacity=0.7,
                    histnorm="probability density",
                    showlegend=False,
                    marker_color=colors[idx % len(colors)],
                ),
                row=2,
                col=1,
            )

        # 3. Performance impact summary (bottom right)
        approaches = []
        gross_sharpe = []
        net_sharpe = []
        cost_impact = []

        for approach, data in results.items():
            approaches.append(approach)

            # Calculate Sharpe ratios
            gross_sr = data["gross_returns"].mean() / data["gross_returns"].std() * np.sqrt(252)
            net_sr = data["net_returns"].mean() / data["net_returns"].std() * np.sqrt(252)

            gross_sharpe.append(gross_sr)
            net_sharpe.append(net_sr)
            cost_impact.append((gross_sr - net_sr) / gross_sr if gross_sr != 0 else 0)

        fig.add_trace(
            go.Bar(
                x=approaches,
                y=gross_sharpe,
                name="Gross Sharpe",
                marker_color="lightblue",
                opacity=0.7,
            ),
            row=2,
            col=2,
        )

        fig.add_trace(
            go.Bar(
                x=approaches, y=net_sharpe, name="Net Sharpe", marker_color="darkblue", opacity=0.7
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(title=f"Transaction Cost Impact Analysis ({cost_bps} bps)", height=800)

        # Update axes
        fig.update_yaxes(title_text="Cumulative Return", tickformat=".1%", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Transaction Cost (bps)", row=2, col=1)
        fig.update_yaxes(title_text="Density", row=2, col=1)
        fig.update_xaxes(title_text="Approach", row=2, col=2)
        fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=2)

        if save_path:
            fig.write_html(str(save_path).replace(".png", ".html"))

        return fig

    def _create_transaction_cost_static(
        self,
        results: dict[str, dict[str, pd.Series]],
        cost_bps: float,
        save_path: str | Path | None,
    ) -> plt.Figure:
        """Create static transaction cost impact visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Transaction Cost Impact Analysis ({cost_bps} bps)", fontsize=16, fontweight="bold"
        )

        colors = sns.color_palette(self.config.color_palette, len(results))

        # 1. Cumulative returns comparison (top left)
        ax1 = axes[0, 0]
        for idx, (approach, data) in enumerate(results.items()):
            gross_cum = (1 + data["gross_returns"]).cumprod()
            net_cum = (1 + data["net_returns"]).cumprod()

            ax1.plot(
                gross_cum.index,
                gross_cum.values,
                label=f"{approach} (Gross)",
                color=colors[idx],
                linewidth=2,
                linestyle="-",
            )
            ax1.plot(
                net_cum.index,
                net_cum.values,
                label=f"{approach} (Net)",
                color=colors[idx],
                linewidth=2,
                linestyle="--",
            )

        ax1.set_title("Cumulative Returns: Gross vs Net")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Cumulative Return")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

        # 2. Transaction cost distribution (top right)
        ax2 = axes[0, 1]
        cost_data = []
        labels = []

        for approach, data in results.items():
            cost_data.append(data["transaction_costs"].values * 10000)  # Convert to bps
            labels.append(approach)

        ax2.hist(cost_data, bins=20, alpha=0.7, label=labels, color=colors, density=True)
        ax2.set_title("Transaction Cost Distribution")
        ax2.set_xlabel("Transaction Cost (bps)")
        ax2.set_ylabel("Density")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Performance impact comparison (bottom left)
        ax3 = axes[1, 0]
        approaches = []
        gross_returns_annual = []
        net_returns_annual = []

        for approach, data in results.items():
            approaches.append(approach)
            gross_returns_annual.append((1 + data["gross_returns"].mean()) ** 252 - 1)
            net_returns_annual.append((1 + data["net_returns"].mean()) ** 252 - 1)

        x = np.arange(len(approaches))
        width = 0.35

        ax3.bar(x - width / 2, gross_returns_annual, width, label="Gross Return", alpha=0.8)
        ax3.bar(x + width / 2, net_returns_annual, width, label="Net Return", alpha=0.8)

        ax3.set_title("Annualized Return Impact")
        ax3.set_xlabel("Approach")
        ax3.set_ylabel("Annualized Return")
        ax3.set_xticks(x)
        ax3.set_xticklabels(approaches, rotation=45, ha="right")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

        # 4. Sharpe ratio impact (bottom right)
        ax4 = axes[1, 1]
        gross_sharpe = []
        net_sharpe = []

        for _approach, data in results.items():
            gross_sr = data["gross_returns"].mean() / data["gross_returns"].std() * np.sqrt(252)
            net_sr = data["net_returns"].mean() / data["net_returns"].std() * np.sqrt(252)
            gross_sharpe.append(gross_sr)
            net_sharpe.append(net_sr)

        ax4.bar(x - width / 2, gross_sharpe, width, label="Gross Sharpe", alpha=0.8)
        ax4.bar(x + width / 2, net_sharpe, width, label="Net Sharpe", alpha=0.8)

        ax4.set_title("Sharpe Ratio Impact")
        ax4.set_xlabel("Approach")
        ax4.set_ylabel("Sharpe Ratio")
        ax4.set_xticks(x)
        ax4.set_xticklabels(approaches, rotation=45, ha="right")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")

        return fig

    def create_constraint_compliance_dashboard(
        self,
        constraint_data: dict[str, dict[str, pd.Series]],
        violation_threshold: float | None = None,
        save_path: str | Path | None = None,
        interactive: bool = True,
    ) -> plt.Figure | go.Figure:
        """
        Implement constraint compliance monitoring dashboards.

        Args:
            constraint_data: Dictionary mapping approach names to constraint violation series
            violation_threshold: Threshold for significant violations (default from config)
            save_path: Path to save the dashboard
            interactive: Whether to create interactive plot

        Returns:
            Matplotlib Figure or Plotly Figure
        """
        threshold = violation_threshold or self.config.compliance_threshold

        if interactive and HAS_PLOTLY:
            return self._create_compliance_dashboard_interactive(
                constraint_data, threshold, save_path
            )
        elif HAS_MATPLOTLIB:
            return self._create_compliance_dashboard_static(constraint_data, threshold, save_path)
        else:
            raise ImportError("Neither Plotly nor Matplotlib available for plotting")

    def _create_compliance_dashboard_interactive(
        self,
        constraint_data: dict[str, dict[str, pd.Series]],
        threshold: float,
        save_path: str | Path | None,
    ) -> go.Figure:
        """Create interactive constraint compliance dashboard using Plotly."""
        # Analyze constraint types
        all_constraint_types = set()
        for approach_data in constraint_data.values():
            all_constraint_types.update(approach_data.keys())

        constraint_types = sorted(all_constraint_types)
        n_constraints = len(constraint_types)

        # Create subplots for each constraint type
        fig = make_subplots(
            rows=n_constraints,
            cols=2,
            subplot_titles=[f"{constraint} Violations" for constraint in constraint_types]
            + [f"{constraint} Summary" for constraint in constraint_types],
            vertical_spacing=0.05,
            horizontal_spacing=0.1,
        )

        colors = px.colors.qualitative.Set2

        for constraint_idx, constraint_type in enumerate(constraint_types):
            # Time series plot (left column)
            for approach_idx, (approach, approach_data) in enumerate(constraint_data.items()):
                if constraint_type in approach_data:
                    violation_series = approach_data[constraint_type]

                    fig.add_trace(
                        go.Scatter(
                            x=violation_series.index,
                            y=violation_series.values,
                            mode="lines+markers",
                            name=f"{approach}" if constraint_idx == 0 else None,
                            showlegend=(constraint_idx == 0),
                            line={"color": colors[approach_idx % len(colors)], "width": 2},
                            marker={"size": 4},
                            hovertemplate=f"<b>{approach}</b><br>"
                            + f"{constraint_type} Violation: %{{y:.2%}}<br>"
                            + "Date: %{x}<extra></extra>",
                        ),
                        row=constraint_idx + 1,
                        col=1,
                    )

                    # Add threshold line
                    fig.add_hline(
                        y=threshold,
                        line_dash="dash",
                        line_color="red",
                        row=constraint_idx + 1,
                        col=1,
                    )

            # Summary bar chart (right column)
            approaches = []
            violation_rates = []
            max_violations = []

            for approach, approach_data in constraint_data.items():
                if constraint_type in approach_data:
                    violation_series = approach_data[constraint_type]
                    violation_rate = (violation_series > threshold).mean()
                    max_violation = violation_series.max()

                    approaches.append(approach)
                    violation_rates.append(violation_rate)
                    max_violations.append(max_violation)

            # Violation rate bars
            fig.add_trace(
                go.Bar(
                    x=approaches,
                    y=violation_rates,
                    name=f"{constraint_type} Rate",
                    marker_color="red",
                    opacity=0.7,
                    showlegend=False,
                    hovertemplate="<b>%{x}</b><br>" + "Violation Rate: %{y:.1%}<extra></extra>",
                ),
                row=constraint_idx + 1,
                col=2,
            )

        # Update layout
        fig.update_layout(
            title="Constraint Compliance Monitoring Dashboard", height=300 * n_constraints
        )

        if save_path:
            fig.write_html(str(save_path).replace(".png", ".html"))

        return fig

    def _create_compliance_dashboard_static(
        self,
        constraint_data: dict[str, dict[str, pd.Series]],
        threshold: float,
        save_path: str | Path | None,
    ) -> plt.Figure:
        """Create static constraint compliance dashboard using Matplotlib."""
        # Analyze constraint types
        all_constraint_types = set()
        for approach_data in constraint_data.values():
            all_constraint_types.update(approach_data.keys())

        constraint_types = sorted(all_constraint_types)
        n_constraints = len(constraint_types)

        fig, axes = plt.subplots(n_constraints, 2, figsize=(14, 4 * n_constraints))
        if n_constraints == 1:
            axes = axes.reshape(1, -1)

        colors = sns.color_palette(self.config.color_palette, len(constraint_data))

        for constraint_idx, constraint_type in enumerate(constraint_types):
            ax_ts = axes[constraint_idx, 0]  # Time series
            ax_summary = axes[constraint_idx, 1]  # Summary

            # Time series plot
            for approach_idx, (approach, approach_data) in enumerate(constraint_data.items()):
                if constraint_type in approach_data:
                    violation_series = approach_data[constraint_type]

                    ax_ts.plot(
                        violation_series.index,
                        violation_series.values,
                        label=approach if constraint_idx == 0 else None,
                        color=colors[approach_idx],
                        linewidth=2,
                        marker="o",
                        markersize=3,
                    )

            # Add threshold line
            ax_ts.axhline(
                y=threshold,
                color="red",
                linestyle="--",
                alpha=0.8,
                label="Threshold" if constraint_idx == 0 else None,
            )

            ax_ts.set_title(f"{constraint_type} Violations Over Time")
            ax_ts.set_xlabel("Date")
            ax_ts.set_ylabel("Violation Level")
            if constraint_idx == 0:
                ax_ts.legend()
            ax_ts.grid(True, alpha=0.3)
            ax_ts.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

            # Summary statistics
            approaches = []
            violation_rates = []
            max_violations = []

            for approach, approach_data in constraint_data.items():
                if constraint_type in approach_data:
                    violation_series = approach_data[constraint_type]
                    violation_rate = (violation_series > threshold).mean()
                    max_violation = violation_series.max()

                    approaches.append(approach)
                    violation_rates.append(violation_rate)
                    max_violations.append(max_violation)

            x = np.arange(len(approaches))
            width = 0.35

            ax_summary.bar(
                x - width / 2,
                violation_rates,
                width,
                label="Violation Rate",
                color="red",
                alpha=0.7,
            )
            ax_summary.bar(
                x + width / 2,
                max_violations,
                width,
                label="Max Violation",
                color="darkred",
                alpha=0.7,
            )

            ax_summary.set_title(f"{constraint_type} Compliance Summary")
            ax_summary.set_xlabel("Approach")
            ax_summary.set_ylabel("Violation Metric")
            ax_summary.set_xticks(x)
            ax_summary.set_xticklabels(approaches, rotation=45, ha="right")
            if constraint_idx == 0:
                ax_summary.legend()
            ax_summary.grid(True, alpha=0.3)
            ax_summary.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

        fig.suptitle(
            f"Constraint Compliance Monitoring Dashboard (Threshold: {threshold:.1%})",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")

        return fig

    def create_implementation_shortfall_analysis(
        self,
        expected_returns: dict[str, pd.Series],
        actual_returns: dict[str, pd.Series],
        turnover_data: dict[str, pd.Series],
        save_path: str | Path | None = None,
        interactive: bool = True,
    ) -> plt.Figure | go.Figure:
        """
        Add implementation shortfall analysis with realistic trading costs.

        Args:
            expected_returns: Dictionary mapping approach names to expected return series
            actual_returns: Dictionary mapping approach names to actual return series
            turnover_data: Dictionary mapping approach names to turnover series
            save_path: Path to save the analysis
            interactive: Whether to create interactive plot

        Returns:
            Matplotlib Figure or Plotly Figure
        """
        # Calculate implementation shortfall
        shortfall_data = {}

        for approach in expected_returns.keys():
            if approach in actual_returns and approach in turnover_data:
                expected = expected_returns[approach]
                actual = actual_returns[approach]
                turnover = turnover_data[approach]

                # Align series
                aligned_expected, aligned_actual = expected.align(actual, join="inner")
                _, aligned_turnover = aligned_expected.align(turnover, join="inner")

                # Calculate shortfall components
                shortfall = aligned_expected - aligned_actual

                shortfall_data[approach] = {
                    "expected_returns": aligned_expected,
                    "actual_returns": aligned_actual,
                    "shortfall": shortfall,
                    "turnover": aligned_turnover,
                }

        if interactive and HAS_PLOTLY:
            return self._create_shortfall_interactive(shortfall_data, save_path)
        elif HAS_MATPLOTLIB:
            return self._create_shortfall_static(shortfall_data, save_path)
        else:
            raise ImportError("Neither Plotly nor Matplotlib available for plotting")

    def _create_shortfall_interactive(
        self, shortfall_data: dict[str, dict[str, pd.Series]], save_path: str | Path | None
    ) -> go.Figure:
        """Create interactive implementation shortfall analysis."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Implementation Shortfall Over Time",
                "Shortfall vs Turnover",
                "Cumulative Impact",
                "Summary Statistics",
            ),
            vertical_spacing=0.1,
        )

        colors = px.colors.qualitative.Set2

        # 1. Shortfall time series (top left)
        for idx, (approach, data) in enumerate(shortfall_data.items()):
            fig.add_trace(
                go.Scatter(
                    x=data["shortfall"].index,
                    y=data["shortfall"].values * 10000,  # Convert to basis points
                    mode="lines+markers",
                    name=f"{approach}",
                    line={"color": colors[idx % len(colors)], "width": 2},
                    marker={"size": 4},
                    hovertemplate=f"<b>{approach}</b><br>"
                    + "Date: %{x}<br>"
                    + "Shortfall: %{y:.1f} bps<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # 2. Shortfall vs Turnover scatter (top right)
        for idx, (approach, data) in enumerate(shortfall_data.items()):
            fig.add_trace(
                go.Scatter(
                    x=data["turnover"].values,
                    y=data["shortfall"].values * 10000,
                    mode="markers",
                    name=f"{approach} Correlation",
                    marker={"color": colors[idx % len(colors)], "size": 6, "opacity": 0.6},
                    showlegend=False,
                    hovertemplate=f"<b>{approach}</b><br>"
                    + "Turnover: %{x:.2%}<br>"
                    + "Shortfall: %{y:.1f} bps<extra></extra>",
                ),
                row=1,
                col=2,
            )

        # 3. Cumulative impact (bottom left)
        for idx, (approach, data) in enumerate(shortfall_data.items()):
            cumulative_shortfall = data["shortfall"].cumsum()

            fig.add_trace(
                go.Scatter(
                    x=cumulative_shortfall.index,
                    y=cumulative_shortfall.values * 10000,
                    mode="lines",
                    name=f"{approach} Cumulative",
                    line={"color": colors[idx % len(colors)], "width": 2},
                    showlegend=False,
                    hovertemplate=f"<b>{approach}</b><br>"
                    + "Date: %{x}<br>"
                    + "Cumulative Shortfall: %{y:.1f} bps<extra></extra>",
                ),
                row=2,
                col=1,
            )

        # 4. Summary statistics table (bottom right)
        approaches = []
        mean_shortfall = []
        std_shortfall = []

        for approach, data in shortfall_data.items():
            approaches.append(approach)
            mean_shortfall.append(data["shortfall"].mean() * 10000)
            std_shortfall.append(data["shortfall"].std() * 10000)

        fig.add_trace(
            go.Table(
                header={
                    "values": ["Approach", "Mean Shortfall (bps)", "Std Shortfall (bps)"],
                    "fill_color": "lightblue",
                    "align": "left",
                },
                cells={
                    "values": [
                        approaches,
                        [f"{val:.2f}" for val in mean_shortfall],
                        [f"{val:.2f}" for val in std_shortfall],
                    ],
                    "fill_color": "white",
                    "align": "left",
                },
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(title="Implementation Shortfall Analysis", height=800)

        # Update axes
        fig.update_yaxes(title_text="Shortfall (bps)", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Turnover", tickformat=".1%", row=1, col=2)
        fig.update_yaxes(title_text="Shortfall (bps)", row=1, col=2)
        fig.update_yaxes(title_text="Cumulative Shortfall (bps)", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)

        if save_path:
            fig.write_html(str(save_path).replace(".png", ".html"))

        return fig

    def _create_shortfall_static(
        self, shortfall_data: dict[str, dict[str, pd.Series]], save_path: str | Path | None
    ) -> plt.Figure:
        """Create static implementation shortfall analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Implementation Shortfall Analysis", fontsize=16, fontweight="bold")

        colors = sns.color_palette(self.config.color_palette, len(shortfall_data))

        # 1. Shortfall time series (top left)
        ax1 = axes[0, 0]
        for idx, (approach, data) in enumerate(shortfall_data.items()):
            ax1.plot(
                data["shortfall"].index,
                data["shortfall"].values * 10000,
                label=approach,
                color=colors[idx],
                linewidth=2,
                marker="o",
                markersize=3,
            )

        ax1.set_title("Implementation Shortfall Over Time")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Shortfall (bps)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color="black", linestyle="-", alpha=0.5)

        # 2. Shortfall vs Turnover scatter (top right)
        ax2 = axes[0, 1]
        for idx, (approach, data) in enumerate(shortfall_data.items()):
            ax2.scatter(
                data["turnover"].values,
                data["shortfall"].values * 10000,
                label=approach,
                color=colors[idx],
                alpha=0.6,
                s=30,
            )

        ax2.set_title("Shortfall vs Turnover")
        ax2.set_xlabel("Turnover")
        ax2.set_ylabel("Shortfall (bps)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

        # 3. Cumulative impact (bottom left)
        ax3 = axes[1, 0]
        for idx, (approach, data) in enumerate(shortfall_data.items()):
            cumulative_shortfall = data["shortfall"].cumsum()
            ax3.plot(
                cumulative_shortfall.index,
                cumulative_shortfall.values * 10000,
                label=approach,
                color=colors[idx],
                linewidth=2,
            )

        ax3.set_title("Cumulative Implementation Shortfall")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Cumulative Shortfall (bps)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color="black", linestyle="-", alpha=0.5)

        # 4. Summary statistics (bottom right)
        ax4 = axes[1, 1]
        approaches = []
        mean_shortfall = []
        std_shortfall = []

        for approach, data in shortfall_data.items():
            approaches.append(approach)
            mean_shortfall.append(data["shortfall"].mean() * 10000)
            std_shortfall.append(data["shortfall"].std() * 10000)

        x = np.arange(len(approaches))
        width = 0.35

        ax4.bar(x - width / 2, mean_shortfall, width, label="Mean Shortfall", alpha=0.8)
        ax4.bar(x + width / 2, std_shortfall, width, label="Std Shortfall", alpha=0.8)

        ax4.set_title("Shortfall Summary Statistics")
        ax4.set_xlabel("Approach")
        ax4.set_ylabel("Shortfall (bps)")
        ax4.set_xticks(x)
        ax4.set_xticklabels(approaches, rotation=45, ha="right")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color="black", linestyle="-", alpha=0.5)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")

        return fig
