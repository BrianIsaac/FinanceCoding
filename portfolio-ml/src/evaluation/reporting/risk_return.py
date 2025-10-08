"""
Risk-return analysis visualization framework.

This module provides comprehensive risk-return visualizations including efficient frontier
scatter plots, confidence ellipses, regime-specific positioning, and animated evolution
over rolling windows.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import EllipticEnvelope

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Ellipse

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

from src.evaluation.validation.bootstrap import BootstrapMethodology


@dataclass
class RiskReturnConfig:
    """Configuration for risk-return visualizations."""

    figsize: tuple[int, int] = (10, 8)
    dpi: int = 300
    style: str = "whitegrid"
    color_palette: str = "husl"
    confidence_level: float = 0.95
    sharpe_isolines: list[float] = None
    efficient_frontier_points: int = 100

    def __post_init__(self):
        if self.sharpe_isolines is None:
            self.sharpe_isolines = [0.5, 1.0, 1.5, 2.0]


class RiskReturnAnalysis:
    """
    Risk-return analysis visualization framework.

    Provides comprehensive risk-return analysis including efficient frontier plots,
    confidence ellipses, regime analysis, and animated rolling window analysis.
    """

    def __init__(self, config: RiskReturnConfig = None):
        """
        Initialize risk-return analysis framework.

        Args:
            config: Configuration for visualization behavior
        """
        self.config = config or RiskReturnConfig()

        if HAS_MATPLOTLIB:
            sns.set_style(self.config.style)
            plt.rcParams["figure.dpi"] = self.config.dpi

        if "BootstrapMethodology" in globals():
            self.bootstrap_methods = BootstrapMethodology()
        else:
            self.bootstrap_methods = None

    def plot_risk_return_scatter(
        self,
        performance_data: dict[str, dict[str, float]],
        confidence_ellipses: dict[str, dict[str, float]] | None = None,
        benchmark_data: dict[str, float] | None = None,
        save_path: str | Path | None = None,
        interactive: bool = True,
    ) -> plt.Figure | go.Figure:
        """
        Create risk-return scatter plot with Sharpe ratio isolines.

        Args:
            performance_data: Dictionary mapping approach names to risk-return metrics
            confidence_ellipses: Optional confidence ellipse parameters
            benchmark_data: Optional benchmark risk-return data
            save_path: Path to save the chart
            interactive: Whether to create interactive plot

        Returns:
            Matplotlib Figure or Plotly Figure
        """
        if interactive and HAS_PLOTLY:
            return self._plot_risk_return_interactive(
                performance_data, confidence_ellipses, benchmark_data, save_path
            )
        elif HAS_MATPLOTLIB:
            return self._plot_risk_return_static(
                performance_data, confidence_ellipses, benchmark_data, save_path
            )
        else:
            raise ImportError("Neither Plotly nor Matplotlib available for plotting")

    def _plot_risk_return_interactive(
        self,
        performance_data: dict[str, dict[str, float]],
        confidence_ellipses: dict[str, dict[str, float]] | None,
        benchmark_data: dict[str, float] | None,
        save_path: str | Path | None,
    ) -> go.Figure:
        """Create interactive risk-return scatter plot using Plotly."""
        fig = go.Figure()

        # Extract risk and return data
        approaches = []
        returns = []
        volatilities = []
        sharpe_ratios = []

        for approach, metrics in performance_data.items():
            approaches.append(approach)
            returns.append(metrics.get("annualized_return", 0))
            volatilities.append(metrics.get("volatility", 0))
            sharpe_ratios.append(metrics.get("sharpe_ratio", 0))

        # Create color scale based on Sharpe ratios
        colors = px.colors.qualitative.Set2

        # Main scatter plot
        fig.add_trace(
            go.Scatter(
                x=volatilities,
                y=returns,
                mode="markers+text",
                text=approaches,
                textposition="top center",
                marker={
                    "size": 12,
                    "color": sharpe_ratios,
                    "colorscale": "viridis",
                    "colorbar": {"title": "Sharpe Ratio"},
                    "line": {"width": 2, "color": "DarkSlateGrey"},
                },
                name="Approaches",
                hovertemplate="<b>%{text}</b><br>"
                + "Return: %{y:.2%}<br>"
                + "Volatility: %{x:.2%}<br>"
                + "Sharpe: %{marker.color:.3f}<extra></extra>",
            )
        )

        # Add benchmark if provided
        if benchmark_data:
            fig.add_trace(
                go.Scatter(
                    x=[benchmark_data.get("volatility", 0)],
                    y=[benchmark_data.get("annualized_return", 0)],
                    mode="markers+text",
                    text=["Benchmark"],
                    textposition="top center",
                    marker={
                        "size": 15,
                        "color": "red",
                        "symbol": "diamond",
                        "line": {"width": 2, "color": "darkred"},
                    },
                    name="Benchmark",
                    hovertemplate="<b>Benchmark</b><br>"
                    + "Return: %{y:.2%}<br>"
                    + "Volatility: %{x:.2%}<extra></extra>",
                )
            )

        # Add Sharpe ratio isolines
        if volatilities and returns:
            vol_range = np.linspace(0, max(volatilities) * 1.2, 100)

            for sharpe_level in self.config.sharpe_isolines:
                isoline_returns = sharpe_level * vol_range

                fig.add_trace(
                    go.Scatter(
                        x=vol_range,
                        y=isoline_returns,
                        mode="lines",
                        line={"dash": "dash", "width": 1, "color": "gray"},
                        name=f"Sharpe = {sharpe_level}",
                        showlegend=True,
                        hovertemplate=f"Sharpe Ratio = {sharpe_level}<extra></extra>",
                    )
                )

        # Add confidence ellipses if provided
        if confidence_ellipses:
            self._add_confidence_ellipses_plotly(fig, confidence_ellipses, approaches, colors)

        # Update layout
        fig.update_layout(
            title="Risk-Return Analysis with Efficient Frontier",
            xaxis_title="Volatility (Annualized)",
            yaxis_title="Return (Annualized)",
            xaxis_tickformat=".1%",
            yaxis_tickformat=".1%",
            hovermode="closest",
            height=600,
            width=800,
        )

        if save_path:
            fig.write_html(str(save_path).replace(".png", ".html"))

        return fig

    def _plot_risk_return_static(
        self,
        performance_data: dict[str, dict[str, float]],
        confidence_ellipses: dict[str, dict[str, float]] | None,
        benchmark_data: dict[str, float] | None,
        save_path: str | Path | None,
    ) -> plt.Figure:
        """Create static risk-return scatter plot using Matplotlib."""
        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Extract data
        approaches = []
        returns = []
        volatilities = []
        sharpe_ratios = []

        for approach, metrics in performance_data.items():
            approaches.append(approach)
            returns.append(metrics.get("annualized_return", 0))
            volatilities.append(metrics.get("volatility", 0))
            sharpe_ratios.append(metrics.get("sharpe_ratio", 0))

        # Create scatter plot with color coding by Sharpe ratio
        scatter = ax.scatter(
            volatilities,
            returns,
            c=sharpe_ratios,
            cmap="viridis",
            s=100,
            alpha=0.7,
            edgecolors="black",
        )

        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label("Sharpe Ratio")

        # Add labels for each point
        for i, approach in enumerate(approaches):
            ax.annotate(
                approach,
                (volatilities[i], returns[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

        # Add benchmark if provided
        if benchmark_data:
            ax.scatter(
                benchmark_data.get("volatility", 0),
                benchmark_data.get("annualized_return", 0),
                c="red",
                s=150,
                marker="D",
                label="Benchmark",
                edgecolors="darkred",
                linewidth=2,
            )
            ax.legend()

        # Add Sharpe ratio isolines
        if volatilities and returns:
            vol_range = np.linspace(0, max(volatilities) * 1.2, 100)

            for sharpe_level in self.config.sharpe_isolines:
                isoline_returns = sharpe_level * vol_range
                ax.plot(
                    vol_range,
                    isoline_returns,
                    "--",
                    alpha=0.5,
                    color="gray",
                    label=(
                        f"Sharpe = {sharpe_level}"
                        if sharpe_level == self.config.sharpe_isolines[0]
                        else ""
                    ),
                )

        # Add confidence ellipses if provided
        if confidence_ellipses:
            self._add_confidence_ellipses_matplotlib(ax, confidence_ellipses, approaches)

        # Formatting
        ax.set_xlabel("Volatility (Annualized)", fontsize=12)
        ax.set_ylabel("Return (Annualized)", fontsize=12)
        ax.set_title("Risk-Return Analysis with Efficient Frontier", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Format axes as percentages
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")

        return fig

    def create_regime_specific_analysis(
        self,
        regime_performance_data: dict[str, dict[str, dict[str, float]]],
        regime_names: list[str] = None,
        save_path: str | Path | None = None,
        interactive: bool = True,
    ) -> plt.Figure | go.Figure:
        """
        Create regime-specific risk-return positioning analysis.

        Args:
            regime_performance_data: Dictionary mapping regimes to approach performance data
            regime_names: Optional list of regime names for display
            save_path: Path to save the chart
            interactive: Whether to create interactive plot

        Returns:
            Matplotlib Figure or Plotly Figure
        """
        if regime_names is None:
            regime_names = list(regime_performance_data.keys())

        if interactive and HAS_PLOTLY:
            return self._plot_regime_analysis_interactive(
                regime_performance_data, regime_names, save_path
            )
        elif HAS_MATPLOTLIB:
            return self._plot_regime_analysis_static(
                regime_performance_data, regime_names, save_path
            )
        else:
            raise ImportError("Neither Plotly nor Matplotlib available for plotting")

    def _plot_regime_analysis_interactive(
        self,
        regime_performance_data: dict[str, dict[str, dict[str, float]]],
        regime_names: list[str],
        save_path: str | Path | None,
    ) -> go.Figure:
        """Create interactive regime-specific analysis using Plotly."""
        n_regimes = len(regime_names)
        cols = 2
        rows = (n_regimes + 1) // 2

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[f"{regime} Market" for regime in regime_names],
            horizontal_spacing=0.1,
            vertical_spacing=0.1,
        )

        for idx, regime in enumerate(regime_names):
            row = idx // cols + 1
            col = idx % cols + 1

            performance_data = regime_performance_data.get(regime, {})

            approaches = []
            returns = []
            volatilities = []
            sharpe_ratios = []

            for approach, metrics in performance_data.items():
                approaches.append(approach)
                returns.append(metrics.get("annualized_return", 0))
                volatilities.append(metrics.get("volatility", 0))
                sharpe_ratios.append(metrics.get("sharpe_ratio", 0))

            # Add scatter plot for this regime
            fig.add_trace(
                go.Scatter(
                    x=volatilities,
                    y=returns,
                    mode="markers+text",
                    text=approaches,
                    textposition="top center",
                    marker={
                        "size": 10,
                        "color": sharpe_ratios,
                        "colorscale": "viridis",
                        "showscale": (idx == 0),  # Only show colorbar for first subplot
                        "colorbar": {"title": "Sharpe Ratio"} if idx == 0 else None,
                        "line": {"width": 1, "color": "DarkSlateGrey"},
                    },
                    name=f"{regime}",
                    showlegend=False,
                    hovertemplate=f"<b>%{{text}}</b> ({regime})<br>"
                    + "Return: %{y:.2%}<br>"
                    + "Volatility: %{x:.2%}<br>"
                    + "Sharpe: %{marker.color:.3f}<extra></extra>",
                ),
                row=row,
                col=col,
            )

            # Add Sharpe isolines for each regime
            if volatilities and returns:
                vol_max = max(volatilities) * 1.1 if volatilities else 0.2
                vol_range = np.linspace(0, vol_max, 50)

                for sharpe_level in [0.5, 1.0, 1.5]:
                    isoline_returns = sharpe_level * vol_range

                    fig.add_trace(
                        go.Scatter(
                            x=vol_range,
                            y=isoline_returns,
                            mode="lines",
                            line={"dash": "dash", "width": 1, "color": "gray"},
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row,
                        col=col,
                    )

        # Update layout
        fig.update_layout(
            title="Risk-Return Analysis by Market Regime", height=300 * rows, hovermode="closest"
        )

        # Update axes
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                if (i - 1) * cols + j <= len(regime_names):
                    fig.update_xaxes(title_text="Volatility", tickformat=".1%", row=i, col=j)
                    fig.update_yaxes(title_text="Return", tickformat=".1%", row=i, col=j)

        if save_path:
            fig.write_html(str(save_path).replace(".png", ".html"))

        return fig

    def _plot_regime_analysis_static(
        self,
        regime_performance_data: dict[str, dict[str, dict[str, float]]],
        regime_names: list[str],
        save_path: str | Path | None,
    ) -> plt.Figure:
        """Create static regime-specific analysis using Matplotlib."""
        n_regimes = len(regime_names)
        cols = 2
        rows = (n_regimes + 1) // 2

        figsize = (self.config.figsize[0] * 2, self.config.figsize[1] * rows)
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, regime in enumerate(regime_names):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]

            performance_data = regime_performance_data.get(regime, {})

            approaches = []
            returns = []
            volatilities = []
            sharpe_ratios = []

            for approach, metrics in performance_data.items():
                approaches.append(approach)
                returns.append(metrics.get("annualized_return", 0))
                volatilities.append(metrics.get("volatility", 0))
                sharpe_ratios.append(metrics.get("sharpe_ratio", 0))

            # Create scatter plot
            if volatilities and returns:
                scatter = ax.scatter(
                    volatilities,
                    returns,
                    c=sharpe_ratios,
                    cmap="viridis",
                    s=80,
                    alpha=0.7,
                    edgecolors="black",
                )

                # Add labels
                for i, approach in enumerate(approaches):
                    ax.annotate(
                        approach,
                        (volatilities[i], returns[i]),
                        xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=8,
                    )

                # Add Sharpe isolines
                vol_max = max(volatilities) * 1.1
                vol_range = np.linspace(0, vol_max, 50)

                for sharpe_level in [0.5, 1.0, 1.5]:
                    isoline_returns = sharpe_level * vol_range
                    ax.plot(vol_range, isoline_returns, "--", alpha=0.4, color="gray")

            ax.set_title(f"{regime} Market", fontweight="bold")
            ax.set_xlabel("Volatility")
            ax.set_ylabel("Return")
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

        # Hide unused subplots
        for idx in range(len(regime_names), rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].set_visible(False)

        # Add colorbar
        if regime_names:
            cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.8)
            cbar.set_label("Sharpe Ratio")

        plt.suptitle("Risk-Return Analysis by Market Regime", fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")

        return fig

    def create_animated_risk_return_evolution(
        self,
        rolling_performance_data: dict[str, list[dict[str, float]]],
        time_periods: list[str],
        save_path: str | Path | None = None,
    ) -> go.Figure:
        """
        Create animated risk-return evolution over rolling windows.

        Args:
            rolling_performance_data: Dictionary mapping approaches to time series
                                       of performance data
            time_periods: List of time period labels
            save_path: Path to save the animation

        Returns:
            Plotly Figure with animation
        """
        if not HAS_PLOTLY:
            raise ImportError("Plotly required for animated visualizations")

        # Prepare data for animation
        frames = []

        for period_idx, period in enumerate(time_periods):
            frame_data = []

            for approach, performance_series in rolling_performance_data.items():
                if period_idx < len(performance_series):
                    metrics = performance_series[period_idx]
                    frame_data.append(
                        {
                            "approach": approach,
                            "return": metrics.get("annualized_return", 0),
                            "volatility": metrics.get("volatility", 0),
                            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                            "period": period,
                        }
                    )

            # Create frame
            if frame_data:
                frame_df = pd.DataFrame(frame_data)

                frame = go.Frame(
                    data=[
                        go.Scatter(
                            x=frame_df["volatility"],
                            y=frame_df["return"],
                            mode="markers+text",
                            text=frame_df["approach"],
                            textposition="top center",
                            marker={
                                "size": 12,
                                "color": frame_df["sharpe_ratio"],
                                "colorscale": "viridis",
                                "colorbar": {"title": "Sharpe Ratio"},
                                "line": {"width": 2, "color": "DarkSlateGrey"},
                                "cmin": min(
                                    [
                                        min(series, key=lambda x: x.get("sharpe_ratio", 0))[
                                            "sharpe_ratio"
                                        ]
                                        for series in rolling_performance_data.values()
                                    ]
                                ),
                                "cmax": max(
                                    [
                                        max(series, key=lambda x: x.get("sharpe_ratio", 0))[
                                            "sharpe_ratio"
                                        ]
                                        for series in rolling_performance_data.values()
                                    ]
                                ),
                            },
                            hovertemplate="<b>%{text}</b><br>"
                            + "Return: %{y:.2%}<br>"
                            + "Volatility: %{x:.2%}<br>"
                            + "Sharpe: %{marker.color:.3f}<extra></extra>",
                        )
                    ],
                    name=period,
                )
                frames.append(frame)

        # Create initial plot (first frame data)
        initial_data = frames[0].data if frames else []

        fig = go.Figure(data=initial_data, frames=frames)

        # Add Sharpe ratio isolines (static)
        vol_range = np.linspace(0, 0.3, 100)  # Adjust range as needed

        for sharpe_level in self.config.sharpe_isolines:
            isoline_returns = sharpe_level * vol_range
            fig.add_trace(
                go.Scatter(
                    x=vol_range,
                    y=isoline_returns,
                    mode="lines",
                    line={"dash": "dash", "width": 1, "color": "gray"},
                    name=f"Sharpe = {sharpe_level}",
                    hoverinfo="skip",
                )
            )

        # Add animation controls
        fig.update_layout(
            title="Risk-Return Evolution Over Time",
            xaxis_title="Volatility (Annualized)",
            yaxis_title="Return (Annualized)",
            xaxis_tickformat=".1%",
            yaxis_tickformat=".1%",
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 1000, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 300},
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                }
            ],
            sliders=[
                {
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {
                        "font": {"size": 20},
                        "prefix": "Period:",
                        "visible": True,
                        "xanchor": "right",
                    },
                    "transition": {"duration": 300, "easing": "cubic-in-out"},
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [
                                [period],
                                {
                                    "frame": {"duration": 300, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 300},
                                },
                            ],
                            "label": period,
                            "method": "animate",
                        }
                        for period in time_periods
                    ],
                }
            ],
        )

        if save_path:
            fig.write_html(str(save_path))

        return fig

    def _add_confidence_ellipses_plotly(
        self,
        fig: go.Figure,
        confidence_ellipses: dict[str, dict[str, float]],
        approaches: list[str],
        colors: list[str],
    ) -> None:
        """Add confidence ellipses to Plotly figure."""
        for idx, approach in enumerate(approaches):
            if approach in confidence_ellipses:
                ellipse_params = confidence_ellipses[approach]

                # Create ellipse points
                center_x = ellipse_params.get("center_x", 0)
                center_y = ellipse_params.get("center_y", 0)
                width = ellipse_params.get("width", 0.01)
                height = ellipse_params.get("height", 0.01)
                angle = ellipse_params.get("angle", 0)

                # Generate ellipse points
                theta = np.linspace(0, 2 * np.pi, 100)
                ellipse_x = (
                    width / 2 * np.cos(theta) * np.cos(angle)
                    - height / 2 * np.sin(theta) * np.sin(angle)
                    + center_x
                )
                ellipse_y = (
                    width / 2 * np.cos(theta) * np.sin(angle)
                    + height / 2 * np.sin(theta) * np.cos(angle)
                    + center_y
                )

                fig.add_trace(
                    go.Scatter(
                        x=ellipse_x,
                        y=ellipse_y,
                        mode="lines",
                        line={"color": colors[idx % len(colors)], "dash": "dot"},
                        name=f"{approach} CI",
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    def _add_confidence_ellipses_matplotlib(
        self, ax: plt.Axes, confidence_ellipses: dict[str, dict[str, float]], approaches: list[str]
    ) -> None:
        """Add confidence ellipses to Matplotlib axes."""
        colors = sns.color_palette(self.config.color_palette, len(approaches))

        for idx, approach in enumerate(approaches):
            if approach in confidence_ellipses:
                ellipse_params = confidence_ellipses[approach]

                center_x = ellipse_params.get("center_x", 0)
                center_y = ellipse_params.get("center_y", 0)
                width = ellipse_params.get("width", 0.01)
                height = ellipse_params.get("height", 0.01)
                angle = ellipse_params.get("angle", 0)

                ellipse = Ellipse(
                    (center_x, center_y),
                    width,
                    height,
                    angle=np.degrees(angle),
                    fill=False,
                    color=colors[idx],
                    linestyle="--",
                    alpha=0.7,
                )

                ax.add_patch(ellipse)

    def calculate_confidence_ellipses(
        self, returns_data: dict[str, pd.Series], confidence_level: float = None
    ) -> dict[str, dict[str, float]]:
        """
        Calculate confidence ellipses for risk-return estimates using bootstrap methods.

        Args:
            returns_data: Dictionary mapping approach names to return series
            confidence_level: Confidence level for ellipses

        Returns:
            Dictionary mapping approach names to ellipse parameters
        """
        confidence_level = confidence_level or self.config.confidence_level
        ellipses = {}

        for approach, returns_series in returns_data.items():
            if len(returns_series) < 10:  # Need sufficient data
                continue

            # Calculate return and volatility estimates with bootstrap
            n_bootstrap = 1000
            bootstrap_returns = []
            bootstrap_volatilities = []

            for _ in range(n_bootstrap):
                # Bootstrap sample
                sample = returns_series.sample(n=len(returns_series), replace=True)

                # Calculate metrics
                annualized_return = (1 + sample.mean()) ** 252 - 1
                volatility = sample.std() * np.sqrt(252)

                bootstrap_returns.append(annualized_return)
                bootstrap_volatilities.append(volatility)

            # Fit ellipse to bootstrap distribution
            data_points = np.column_stack([bootstrap_volatilities, bootstrap_returns])

            try:
                # Use robust covariance estimation
                robust_cov = EllipticEnvelope(contamination=0.1).fit(data_points)
                center = robust_cov.location_
                covariance = robust_cov.covariance_

                # Calculate ellipse parameters
                eigenvals, eigenvecs = np.linalg.eigh(covariance)

                # Sort eigenvalues and eigenvectors
                order = eigenvals.argsort()[::-1]
                eigenvals = eigenvals[order]
                eigenvecs = eigenvecs[:, order]

                # Calculate ellipse dimensions
                chi2_val = stats.chi2.ppf(confidence_level, 2)
                width = 2 * np.sqrt(chi2_val * eigenvals[0])
                height = 2 * np.sqrt(chi2_val * eigenvals[1])

                # Calculate angle
                angle = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])

                ellipses[approach] = {
                    "center_x": center[0],
                    "center_y": center[1],
                    "width": width,
                    "height": height,
                    "angle": angle,
                }

            except Exception as e:
                warnings.warn(
                    f"Failed to calculate confidence ellipse for {approach}: {e}", stacklevel=2
                )
                continue

        return ellipses
