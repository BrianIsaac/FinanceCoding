"""
Time series visualization framework for portfolio performance analysis.

This module provides comprehensive time series plots including cumulative returns,
drawdown analysis, rolling performance metrics, and interactive visualizations.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

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

from src.evaluation.metrics.returns import PerformanceAnalytics
from src.evaluation.validation.bootstrap import BootstrapMethodology


@dataclass
class ChartConfig:
    """Configuration for chart generation."""

    figsize: tuple[int, int] = (12, 8)
    dpi: int = 300
    style: str = "whitegrid"  # seaborn style
    color_palette: str = "husl"
    save_format: str = "png"
    interactive: bool = True
    confidence_level: float = 0.95


class TimeSeriesCharts:
    """
    Time series visualization framework for performance analysis.

    Provides comprehensive charting capabilities including cumulative returns,
    drawdown analysis, rolling metrics, and interactive visualizations.
    """

    def __init__(self, config: ChartConfig = None):
        """
        Initialize time series charts framework.

        Args:
            config: Configuration for chart appearance and behavior
        """
        self.config = config or ChartConfig()

        if HAS_MATPLOTLIB:
            sns.set_style(self.config.style)
            plt.rcParams["figure.dpi"] = self.config.dpi

        if HAS_PLOTLY and self.config.interactive:
            # Set default plotly template
            import plotly.io as pio

            pio.templates.default = "plotly_white"

        if "BootstrapMethodology" in globals():
            self.bootstrap_methods = BootstrapMethodology()
        else:
            self.bootstrap_methods = None

    def plot_cumulative_returns(
        self,
        returns_data: dict[str, pd.Series],
        confidence_intervals: dict[str, dict[str, pd.Series]] | None = None,
        benchmark_returns: pd.Series | None = None,
        save_path: str | Path | None = None,
        interactive: bool | None = None,
    ) -> plt.Figure | go.Figure:
        """
        Create cumulative returns plot with optional confidence intervals.

        Args:
            returns_data: Dictionary mapping approach names to return series
            confidence_intervals: Optional confidence intervals for each approach
            benchmark_returns: Optional benchmark returns for comparison
            save_path: Path to save the chart
            interactive: Override config for interactivity

        Returns:
            Matplotlib Figure or Plotly Figure
        """
        use_interactive = interactive if interactive is not None else self.config.interactive

        if use_interactive and HAS_PLOTLY:
            return self._plot_cumulative_returns_interactive(
                returns_data, confidence_intervals, benchmark_returns, save_path
            )
        elif HAS_MATPLOTLIB:
            return self._plot_cumulative_returns_static(
                returns_data, confidence_intervals, benchmark_returns, save_path
            )
        else:
            raise ImportError("Neither Plotly nor Matplotlib available for plotting")

    def _plot_cumulative_returns_interactive(
        self,
        returns_data: dict[str, pd.Series],
        confidence_intervals: dict[str, dict[str, pd.Series]] | None,
        benchmark_returns: pd.Series | None,
        save_path: str | Path | None,
    ) -> go.Figure:
        """Create interactive cumulative returns plot using Plotly."""
        fig = go.Figure()

        # Color palette
        colors = px.colors.qualitative.Set2
        color_idx = 0

        # Plot each approach
        for approach_name, returns_series in returns_data.items():
            # Calculate cumulative returns
            cum_returns = (1 + returns_series).cumprod()

            # Main line
            fig.add_trace(
                go.Scatter(
                    x=cum_returns.index,
                    y=cum_returns.values,
                    mode="lines",
                    name=approach_name,
                    line={"color": colors[color_idx % len(colors)], "width": 2},
                    hovertemplate=f"<b>{approach_name}</b><br>"
                    + "Date: %{x}<br>"
                    + "Cumulative Return: %{y:.2%}<extra></extra>",
                )
            )

            # Add confidence intervals if available
            if confidence_intervals and approach_name in confidence_intervals:
                ci_data = confidence_intervals[approach_name]
                if "lower" in ci_data and "upper" in ci_data:
                    lower_ci = (1 + ci_data["lower"]).cumprod()
                    upper_ci = (1 + ci_data["upper"]).cumprod()

                    # Add confidence interval as filled area
                    fig.add_trace(
                        go.Scatter(
                            x=list(lower_ci.index) + list(upper_ci.index[::-1]),
                            y=list(lower_ci.values) + list(upper_ci.values[::-1]),
                            fill=(
                                "tonexty"
                                if approach_name != list(returns_data.keys())[0]
                                else "tozeroy"
                            ),
                            fillcolor=f"rgba({colors[color_idx % len(colors)][4:-1]}, 0.2)",
                            line={"color": "rgba(255,255,255,0)"},
                            showlegend=False,
                            name=f"{approach_name} CI",
                            hoverinfo="skip",
                        )
                    )

            color_idx += 1

        # Add benchmark if provided
        if benchmark_returns is not None:
            benchmark_cum = (1 + benchmark_returns).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=benchmark_cum.index,
                    y=benchmark_cum.values,
                    mode="lines",
                    name="Benchmark",
                    line={"color": "black", "width": 2, "dash": "dash"},
                    hovertemplate="<b>Benchmark</b><br>"
                    + "Date: %{x}<br>"
                    + "Cumulative Return: %{y:.2%}<extra></extra>",
                )
            )

        # Update layout
        fig.update_layout(
            title="Cumulative Returns Comparison",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            yaxis_tickformat=".1%",
            hovermode="x unified",
            legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
            height=600,
        )

        # Add range selector
        fig.update_layout(
            xaxis={
                "rangeselector": {
                    "buttons": [
                        {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
                        {"count": 2, "label": "2Y", "step": "year", "stepmode": "backward"},
                        {"count": 5, "label": "5Y", "step": "year", "stepmode": "backward"},
                        {"step": "all"},
                    ]
                },
                "rangeslider": {"visible": True},
                "type": "date",
            }
        )

        if save_path:
            fig.write_html(str(save_path).replace(".png", ".html"))

        return fig

    def _plot_cumulative_returns_static(
        self,
        returns_data: dict[str, pd.Series],
        confidence_intervals: dict[str, dict[str, pd.Series]] | None,
        benchmark_returns: pd.Series | None,
        save_path: str | Path | None,
    ) -> plt.Figure:
        """Create static cumulative returns plot using Matplotlib."""
        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Color palette
        colors = sns.color_palette(self.config.color_palette, len(returns_data))

        # Plot each approach
        for idx, (approach_name, returns_series) in enumerate(returns_data.items()):
            # Calculate cumulative returns
            cum_returns = (1 + returns_series).cumprod()

            # Main line
            ax.plot(
                cum_returns.index,
                cum_returns.values,
                label=approach_name,
                color=colors[idx],
                linewidth=2,
            )

            # Add confidence intervals if available
            if confidence_intervals and approach_name in confidence_intervals:
                ci_data = confidence_intervals[approach_name]
                if "lower" in ci_data and "upper" in ci_data:
                    lower_ci = (1 + ci_data["lower"]).cumprod()
                    upper_ci = (1 + ci_data["upper"]).cumprod()

                    ax.fill_between(
                        lower_ci.index,
                        lower_ci.values,
                        upper_ci.values,
                        alpha=0.2,
                        color=colors[idx],
                    )

        # Add benchmark if provided
        if benchmark_returns is not None:
            benchmark_cum = (1 + benchmark_returns).cumprod()
            ax.plot(
                benchmark_cum.index,
                benchmark_cum.values,
                label="Benchmark",
                color="black",
                linewidth=2,
                linestyle="--",
            )

        # Formatting
        ax.set_title("Cumulative Returns Comparison", fontsize=16, fontweight="bold")
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Cumulative Return", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")

        return fig

    def plot_drawdown_analysis(
        self,
        returns_data: dict[str, pd.Series],
        save_path: str | Path | None = None,
        interactive: bool | None = None,
    ) -> plt.Figure | go.Figure:
        """
        Create unified drawdown analysis charts showing peak-to-trough periods.

        Args:
            returns_data: Dictionary mapping approach names to return series
            save_path: Path to save the chart
            interactive: Override config for interactivity

        Returns:
            Matplotlib Figure or Plotly Figure
        """
        use_interactive = interactive if interactive is not None else self.config.interactive

        if use_interactive and HAS_PLOTLY:
            return self._plot_drawdown_interactive(returns_data, save_path)
        elif HAS_MATPLOTLIB:
            return self._plot_drawdown_static(returns_data, save_path)
        else:
            raise ImportError("Neither Plotly nor Matplotlib available for plotting")

    def _plot_drawdown_interactive(
        self, returns_data: dict[str, pd.Series], save_path: str | Path | None
    ) -> go.Figure:
        """Create interactive drawdown analysis using Plotly."""
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=("Cumulative Returns", "Drawdown Analysis"),
            vertical_spacing=0.1,
        )

        colors = px.colors.qualitative.Set2

        for idx, (approach_name, returns_series) in enumerate(returns_data.items()):
            # Calculate cumulative returns and drawdowns
            cum_returns = (1 + returns_series).cumprod()
            running_max = cum_returns.expanding(min_periods=1).max()
            drawdown = (cum_returns - running_max) / running_max

            color = colors[idx % len(colors)]

            # Cumulative returns subplot
            fig.add_trace(
                go.Scatter(
                    x=cum_returns.index,
                    y=cum_returns.values,
                    mode="lines",
                    name=approach_name,
                    line={"color": color, "width": 2},
                    hovertemplate=(
                        f"<b>{approach_name}</b><br>Date: %{{x}}<br>"
                        f"Return: %{{y:.2%}}<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

            # Drawdown subplot
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    mode="lines",
                    name=f"{approach_name} DD",
                    line={"color": color, "width": 2},
                    fill="tonexty" if idx == 0 else "tozeroy",
                    fillcolor=f"rgba({color[4:-1]}, 0.3)",
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{approach_name}</b><br>Date: %{{x}}<br>"
                        f"Drawdown: %{{y:.2%}}<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )

        # Update layout
        fig.update_layout(
            title="Performance and Drawdown Analysis", height=800, hovermode="x unified"
        )

        # Update y-axes
        fig.update_yaxes(title_text="Cumulative Return", tickformat=".1%", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown", tickformat=".1%", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)

        if save_path:
            fig.write_html(str(save_path).replace(".png", ".html"))

        return fig

    def _plot_drawdown_static(
        self, returns_data: dict[str, pd.Series], save_path: str | Path | None
    ) -> plt.Figure:
        """Create static drawdown analysis using Matplotlib."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config.figsize, sharex=True)

        colors = sns.color_palette(self.config.color_palette, len(returns_data))

        for idx, (approach_name, returns_series) in enumerate(returns_data.items()):
            # Calculate cumulative returns and drawdowns
            cum_returns = (1 + returns_series).cumprod()
            running_max = cum_returns.expanding(min_periods=1).max()
            drawdown = (cum_returns - running_max) / running_max

            color = colors[idx]

            # Cumulative returns
            ax1.plot(
                cum_returns.index, cum_returns.values, label=approach_name, color=color, linewidth=2
            )

            # Drawdown
            ax2.fill_between(
                drawdown.index, drawdown.values, 0, alpha=0.3, color=color, label=f"{approach_name}"
            )
            ax2.plot(drawdown.index, drawdown.values, color=color, linewidth=1)

        # Formatting
        ax1.set_title("Performance and Drawdown Analysis", fontsize=16, fontweight="bold")
        ax1.set_ylabel("Cumulative Return", fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

        ax2.set_xlabel("Date", fontsize=12)
        ax2.set_ylabel("Drawdown", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")

        return fig

    def plot_rolling_performance_metrics(
        self,
        rolling_metrics_data: dict[str, dict[str, pd.Series]],
        metrics_to_plot: list[str] = None,
        save_path: str | Path | None = None,
        interactive: bool | None = None,
    ) -> plt.Figure | go.Figure:
        """
        Create rolling performance metrics visualization over time.

        Args:
            rolling_metrics_data: Dictionary mapping approach names to metrics series
            metrics_to_plot: List of metric names to plot (defaults to key metrics)
            save_path: Path to save the chart
            interactive: Override config for interactivity

        Returns:
            Matplotlib Figure or Plotly Figure
        """
        if metrics_to_plot is None:
            metrics_to_plot = ["sharpe_ratio", "information_ratio", "max_drawdown", "volatility"]

        use_interactive = interactive if interactive is not None else self.config.interactive

        if use_interactive and HAS_PLOTLY:
            return self._plot_rolling_metrics_interactive(
                rolling_metrics_data, metrics_to_plot, save_path
            )
        elif HAS_MATPLOTLIB:
            return self._plot_rolling_metrics_static(
                rolling_metrics_data, metrics_to_plot, save_path
            )
        else:
            raise ImportError("Neither Plotly nor Matplotlib available for plotting")

    def _plot_rolling_metrics_interactive(
        self,
        rolling_metrics_data: dict[str, dict[str, pd.Series]],
        metrics_to_plot: list[str],
        save_path: str | Path | None,
    ) -> go.Figure:
        """Create interactive rolling metrics plot using Plotly."""
        n_metrics = len(metrics_to_plot)
        fig = make_subplots(
            rows=n_metrics,
            cols=1,
            shared_xaxes=True,
            subplot_titles=metrics_to_plot,
            vertical_spacing=0.05,
        )

        colors = px.colors.qualitative.Set2

        for metric_idx, metric_name in enumerate(metrics_to_plot):
            for approach_idx, (approach_name, metrics_dict) in enumerate(
                rolling_metrics_data.items()
            ):
                if metric_name in metrics_dict:
                    metric_series = metrics_dict[metric_name]

                    fig.add_trace(
                        go.Scatter(
                            x=metric_series.index,
                            y=metric_series.values,
                            mode="lines",
                            name=f"{approach_name}" if metric_idx == 0 else None,
                            showlegend=(metric_idx == 0),
                            line={"color": colors[approach_idx % len(colors)], "width": 2},
                            hovertemplate=f"<b>{approach_name}</b><br>"
                            + f"{metric_name}: %{{y:.3f}}<br>"
                            + "Date: %{x}<extra></extra>",
                        ),
                        row=metric_idx + 1,
                        col=1,
                    )

        # Update layout
        fig.update_layout(
            title="Rolling Performance Metrics", height=200 * n_metrics, hovermode="x unified"
        )

        fig.update_xaxes(title_text="Date", row=n_metrics, col=1)

        if save_path:
            fig.write_html(str(save_path).replace(".png", ".html"))

        return fig

    def _plot_rolling_metrics_static(
        self,
        rolling_metrics_data: dict[str, dict[str, pd.Series]],
        metrics_to_plot: list[str],
        save_path: str | Path | None,
    ) -> plt.Figure:
        """Create static rolling metrics plot using Matplotlib."""
        n_metrics = len(metrics_to_plot)
        figsize = (self.config.figsize[0], self.config.figsize[1] * n_metrics / 2)
        fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)

        if n_metrics == 1:
            axes = [axes]

        colors = sns.color_palette(self.config.color_palette, len(rolling_metrics_data))

        for metric_idx, metric_name in enumerate(metrics_to_plot):
            ax = axes[metric_idx]

            for approach_idx, (approach_name, metrics_dict) in enumerate(
                rolling_metrics_data.items()
            ):
                if metric_name in metrics_dict:
                    metric_series = metrics_dict[metric_name]

                    ax.plot(
                        metric_series.index,
                        metric_series.values,
                        label=approach_name if metric_idx == 0 else None,
                        color=colors[approach_idx],
                        linewidth=2,
                    )

            ax.set_ylabel(metric_name.replace("_", " ").title(), fontsize=10)
            ax.grid(True, alpha=0.3)

            if metric_idx == 0:
                ax.legend()

        axes[-1].set_xlabel("Date", fontsize=12)
        fig.suptitle("Rolling Performance Metrics", fontsize=16, fontweight="bold")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")

        return fig

    def create_performance_dashboard(
        self,
        returns_data: dict[str, pd.Series],
        rolling_metrics_data: dict[str, dict[str, pd.Series]] | None = None,
        save_path: str | Path | None = None,
    ) -> go.Figure:
        """
        Create comprehensive performance dashboard with multiple charts.

        Args:
            returns_data: Dictionary mapping approach names to return series
            rolling_metrics_data: Optional rolling metrics data
            save_path: Path to save the dashboard

        Returns:
            Plotly Figure with dashboard
        """
        if not HAS_PLOTLY:
            raise ImportError("Plotly required for interactive dashboard")

        # Create subplot layout
        specs = [
            [{"colspan": 2}, None],  # Cumulative returns (full width)
            [{"rowspan": 2}, {"type": "table"}],  # Drawdown + summary table
            [None, {"type": "bar"}],  # Performance bars
        ]

        fig = make_subplots(
            rows=3,
            cols=2,
            specs=specs,
            subplot_titles=(
                "Cumulative Returns",
                "",
                "Drawdown Analysis",
                "Performance Summary",
                "",
                "Key Metrics",
            ),
            vertical_spacing=0.08,
            horizontal_spacing=0.1,
        )

        colors = px.colors.qualitative.Set2

        # 1. Cumulative Returns (top, full width)
        for idx, (approach_name, returns_series) in enumerate(returns_data.items()):
            cum_returns = (1 + returns_series).cumprod()

            fig.add_trace(
                go.Scatter(
                    x=cum_returns.index,
                    y=cum_returns.values,
                    mode="lines",
                    name=approach_name,
                    line={"color": colors[idx % len(colors)], "width": 2},
                ),
                row=1,
                col=1,
            )

        # 2. Drawdown Analysis (middle left)
        for idx, (approach_name, returns_series) in enumerate(returns_data.items()):
            cum_returns = (1 + returns_series).cumprod()
            running_max = cum_returns.expanding(min_periods=1).max()
            drawdown = (cum_returns - running_max) / running_max

            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    mode="lines",
                    name=f"{approach_name} DD",
                    line={"color": colors[idx % len(colors)], "width": 2},
                    showlegend=False,
                    fill="tonexty" if idx > 0 else "tozeroy",
                    fillcolor=f"rgba({colors[idx % len(colors)][4:-1]}, 0.3)",
                ),
                row=2,
                col=1,
            )

        # 3. Performance Summary Table (middle right)
        summary_data = []
        for approach_name, returns_series in returns_data.items():
            perf_analytics = PerformanceAnalytics()
            sharpe = perf_analytics.sharpe_ratio(returns_series)
            total_return = (1 + returns_series).prod() - 1
            max_dd = perf_analytics.maximum_drawdown(returns_series)[0]

            summary_data.append(
                [approach_name, f"{sharpe:.3f}", f"{total_return:.2%}", f"{max_dd:.2%}"]
            )

        fig.add_trace(
            go.Table(
                header={
                    "values": ["Approach", "Sharpe Ratio", "Total Return", "Max Drawdown"],
                    "fill_color": "lightblue",
                    "align": "left",
                },
                cells={"values": list(zip(*summary_data)), "fill_color": "white", "align": "left"},
            ),
            row=2,
            col=2,
        )

        # 4. Key Metrics Bar Chart (bottom right)
        if rolling_metrics_data:
            approaches = list(rolling_metrics_data.keys())
            sharpe_values = []

            for approach in approaches:
                if "sharpe_ratio" in rolling_metrics_data[approach]:
                    sharpe_values.append(rolling_metrics_data[approach]["sharpe_ratio"].iloc[-1])
                else:
                    # Calculate from returns if rolling not available
                    returns_series = returns_data.get(approach, pd.Series())
                    if not returns_series.empty:
                        sharpe_values.append(PerformanceAnalytics.sharpe_ratio(returns_series))
                    else:
                        sharpe_values.append(0)

            fig.add_trace(
                go.Bar(
                    x=approaches,
                    y=sharpe_values,
                    name="Sharpe Ratio",
                    marker_color=colors[: len(approaches)],
                    showlegend=False,
                ),
                row=3,
                col=2,
            )

        # Update layout
        fig.update_layout(title="Portfolio Performance Dashboard", height=900, showlegend=True)

        # Update axes
        fig.update_yaxes(title_text="Cumulative Return", tickformat=".1%", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown", tickformat=".1%", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=3, col=2)

        if save_path:
            fig.write_html(str(save_path))

        return fig

    def export_chart(
        self, figure: plt.Figure | go.Figure, filename: str, formats: list[str] = None
    ) -> dict[str, str]:
        """
        Export chart in multiple formats.

        Args:
            figure: Matplotlib or Plotly figure to export
            filename: Base filename for export
            formats: Export formats (png, pdf, html, svg)

        Returns:
            Dictionary mapping format names to file paths
        """
        if formats is None:
            formats = ["png", "html"] if HAS_PLOTLY else ["png"]

        exported_files = {}

        for format_type in formats:
            filepath = f"{filename}.{format_type}"

            try:
                if isinstance(figure, go.Figure):
                    # Plotly figure
                    if format_type in ["png", "pdf", "svg"]:
                        figure.write_image(filepath)
                        exported_files[format_type] = filepath
                    elif format_type == "html":
                        figure.write_html(filepath)
                        exported_files[format_type] = filepath

                elif HAS_MATPLOTLIB and hasattr(figure, "savefig"):
                    # Matplotlib figure
                    if format_type in ["png", "pdf", "svg"]:
                        figure.savefig(filepath, dpi=self.config.dpi, bbox_inches="tight")
                        exported_files[format_type] = filepath

            except Exception as e:
                warnings.warn(f"Failed to export {format_type}: {e}", stacklevel=2)

        return exported_files
