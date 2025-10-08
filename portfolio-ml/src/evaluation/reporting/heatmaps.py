"""
Performance heat map framework for portfolio analysis.

This module provides comprehensive heat map visualizations including monthly performance,
relative performance vs baselines, statistical significance overlays, and regime
identification annotations.
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
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    warnings.warn("Plotly not available. Interactive plotting disabled.", stacklevel=2)


@dataclass
class HeatmapConfig:
    """Configuration for heat map visualizations."""

    figsize: tuple[int, int] = (12, 8)
    dpi: int = 300
    style: str = "white"
    colormap: str = "RdBu_r"  # Red-Blue diverging colormap
    center_value: float = 0.0  # Center point for diverging colormap
    annot_fontsize: int = 8
    title_fontsize: int = 14
    significance_symbols: bool = True


class PerformanceHeatmaps:
    """
    Performance heat map framework for portfolio analysis.

    Provides comprehensive heat map visualizations including monthly performance,
    relative comparisons, statistical significance overlays, and regime annotations.
    """

    def __init__(self, config: HeatmapConfig = None):
        """
        Initialize performance heat maps framework.

        Args:
            config: Configuration for heat map appearance and behavior
        """
        self.config = config or HeatmapConfig()

        if HAS_MATPLOTLIB:
            sns.set_style(self.config.style)
            plt.rcParams["figure.dpi"] = self.config.dpi

    def create_monthly_performance_heatmap(
        self,
        returns_data: dict[str, pd.Series],
        aggregation_method: str = "monthly",
        save_path: str | Path | None = None,
        interactive: bool = True,
    ) -> plt.Figure | go.Figure:
        """
        Generate monthly performance heat maps for all approaches.

        Args:
            returns_data: Dictionary mapping approach names to return series
            aggregation_method: Aggregation method ('monthly', 'quarterly')
            save_path: Path to save the heat map
            interactive: Whether to create interactive plot

        Returns:
            Matplotlib Figure or Plotly Figure
        """
        # Aggregate returns by time period
        aggregated_data = self._aggregate_returns(returns_data, aggregation_method)

        if interactive and HAS_PLOTLY:
            return self._create_monthly_heatmap_interactive(
                aggregated_data, save_path, aggregation_method
            )
        elif HAS_MATPLOTLIB:
            return self._create_monthly_heatmap_static(
                aggregated_data, save_path, aggregation_method
            )
        else:
            raise ImportError("Neither Plotly nor Matplotlib available for plotting")

    def _aggregate_returns(
        self, returns_data: dict[str, pd.Series], method: str
    ) -> dict[str, pd.DataFrame]:
        """Aggregate returns by specified time period."""
        aggregated = {}

        for approach, returns_series in returns_data.items():
            if method == "monthly":
                # Group by year-month and calculate monthly returns
                monthly_returns = returns_series.groupby(
                    [returns_series.index.year, returns_series.index.month]
                ).apply(lambda x: (1 + x).prod() - 1)

                # Convert to DataFrame with year-month structure
                pivot_data = []
                for (year, month), return_val in monthly_returns.items():
                    pivot_data.append({"Year": year, "Month": month, "Return": return_val})

                df = pd.DataFrame(pivot_data)
                if not df.empty:
                    pivot_df = df.pivot(index="Year", columns="Month", values="Return")

                    # Rename columns to month names
                    month_names = [
                        "Jan",
                        "Feb",
                        "Mar",
                        "Apr",
                        "May",
                        "Jun",
                        "Jul",
                        "Aug",
                        "Sep",
                        "Oct",
                        "Nov",
                        "Dec",
                    ]
                    pivot_df.columns = [month_names[i - 1] for i in pivot_df.columns]

                    aggregated[approach] = pivot_df

            elif method == "quarterly":
                # Group by year-quarter
                quarterly_returns = returns_series.groupby(
                    [returns_series.index.year, returns_series.index.quarter]
                ).apply(lambda x: (1 + x).prod() - 1)

                pivot_data = []
                for (year, quarter), return_val in quarterly_returns.items():
                    pivot_data.append(
                        {"Year": year, "Quarter": f"Q{quarter}", "Return": return_val}
                    )

                df = pd.DataFrame(pivot_data)
                if not df.empty:
                    pivot_df = df.pivot(index="Year", columns="Quarter", values="Return")
                    aggregated[approach] = pivot_df

        return aggregated

    def _create_monthly_heatmap_interactive(
        self,
        aggregated_data: dict[str, pd.DataFrame],
        save_path: str | Path | None,
        period_type: str,
    ) -> go.Figure:
        """Create interactive monthly performance heat map using Plotly."""
        n_approaches = len(aggregated_data)

        if n_approaches == 0:
            raise ValueError("No data available for heat map creation")

        # Create subplots
        approaches = list(aggregated_data.keys())
        fig = make_subplots(
            rows=n_approaches, cols=1, subplot_titles=approaches, vertical_spacing=0.05
        )

        # Determine global color scale
        all_values = []
        for df in aggregated_data.values():
            all_values.extend(df.values.flatten())
        all_values = [v for v in all_values if not pd.isna(v)]

        if all_values:
            vmin, vmax = min(all_values), max(all_values)
            # Make symmetric around zero
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max
        else:
            vmin, vmax = -0.1, 0.1

        # Create heat map for each approach
        for idx, (approach, df) in enumerate(aggregated_data.items()):
            if df.empty:
                continue

            # Prepare data for heatmap
            z_values = df.values
            x_labels = list(df.columns)
            y_labels = list(df.index)

            # Create heatmap
            fig.add_trace(
                go.Heatmap(
                    z=z_values,
                    x=x_labels,
                    y=y_labels,
                    colorscale="RdBu",
                    zmid=0,
                    zmin=vmin,
                    zmax=vmax,
                    showscale=(idx == 0),  # Only show colorbar for first subplot
                    colorbar={"title": "Return", "tickformat": ".2%"} if idx == 0 else None,
                    hovertemplate="<b>%{y} %{x}</b><br>Return: %{z:.2%}<extra></extra>",
                    name=approach,
                ),
                row=idx + 1,
                col=1,
            )

        # Update layout
        period_label = "Monthly" if period_type == "monthly" else "Quarterly"
        fig.update_layout(
            title=f"{period_label} Performance Heat Maps",
            height=200 * n_approaches,
        )

        if save_path:
            fig.write_html(str(save_path).replace(".png", ".html"))

        return fig

    def _create_monthly_heatmap_static(
        self,
        aggregated_data: dict[str, pd.DataFrame],
        save_path: str | Path | None,
        period_type: str,
    ) -> plt.Figure:
        """Create static monthly performance heat map using Matplotlib."""
        n_approaches = len(aggregated_data)

        if n_approaches == 0:
            raise ValueError("No data available for heat map creation")

        # Create subplots
        figsize = (self.config.figsize[0], 3 * n_approaches)
        fig, axes = plt.subplots(n_approaches, 1, figsize=figsize)
        if n_approaches == 1:
            axes = [axes]

        # Determine global color scale
        all_values = []
        for df in aggregated_data.values():
            all_values.extend(df.values.flatten())
        all_values = [v for v in all_values if not pd.isna(v)]

        if all_values:
            abs_max = max(abs(min(all_values)), abs(max(all_values)))
            vmin, vmax = -abs_max, abs_max
        else:
            vmin, vmax = -0.1, 0.1

        # Create heat map for each approach
        for idx, (approach, df) in enumerate(aggregated_data.items()):
            ax = axes[idx]

            if df.empty:
                ax.text(
                    0.5, 0.5, "No Data Available", ha="center", va="center", transform=ax.transAxes
                )
                ax.set_title(approach)
                continue

            # Create heatmap
            sns.heatmap(
                df,
                ax=ax,
                cmap=self.config.colormap,
                center=self.config.center_value,
                vmin=vmin,
                vmax=vmax,
                annot=True,
                fmt=".1%",
                annot_kws={"size": self.config.annot_fontsize},
                cbar=(idx == 0),  # Only show colorbar for first subplot
                cbar_kws={"format": "%.1%%"} if idx == 0 else None,
            )

            ax.set_title(approach, fontsize=self.config.title_fontsize, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("Year" if idx == n_approaches - 1 else "")

        period_label = "Monthly" if period_type == "monthly" else "Quarterly"
        fig.suptitle(f"{period_label} Performance Heat Maps", fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")

        return fig

    def create_relative_performance_heatmap(
        self,
        returns_data: dict[str, pd.Series],
        baseline_returns: dict[str, pd.Series],
        save_path: str | Path | None = None,
        interactive: bool = True,
    ) -> plt.Figure | go.Figure:
        """
        Create relative performance heat maps (ML approaches vs baselines).

        Args:
            returns_data: Dictionary mapping ML approach names to return series
            baseline_returns: Dictionary mapping baseline names to return series
            save_path: Path to save the heat map
            interactive: Whether to create interactive plot

        Returns:
            Matplotlib Figure or Plotly Figure
        """
        # Calculate relative performance
        relative_performance = {}

        for ml_approach, ml_returns in returns_data.items():
            approach_relatives = {}

            for baseline_name, baseline_returns_series in baseline_returns.items():
                # Align time series
                aligned_ml, aligned_baseline = ml_returns.align(
                    baseline_returns_series, join="inner"
                )

                if len(aligned_ml) > 0:
                    # Calculate excess returns
                    excess_returns = aligned_ml - aligned_baseline
                    approach_relatives[f"vs_{baseline_name}"] = excess_returns

            if approach_relatives:
                relative_performance[ml_approach] = approach_relatives

        # Aggregate relative performance monthly
        aggregated_relative = {}
        for approach, baseline_dict in relative_performance.items():
            approach_aggregated = {}

            for baseline_name, excess_series in baseline_dict.items():
                monthly_excess = excess_series.groupby(
                    [excess_series.index.year, excess_series.index.month]
                ).sum()

                # Convert to DataFrame
                pivot_data = []
                for (year, month), excess_val in monthly_excess.items():
                    pivot_data.append({"Year": year, "Month": month, "Excess": excess_val})

                if pivot_data:
                    df = pd.DataFrame(pivot_data)
                    pivot_df = df.pivot(index="Year", columns="Month", values="Excess")

                    # Rename columns to month names
                    month_names = [
                        "Jan",
                        "Feb",
                        "Mar",
                        "Apr",
                        "May",
                        "Jun",
                        "Jul",
                        "Aug",
                        "Sep",
                        "Oct",
                        "Nov",
                        "Dec",
                    ]
                    pivot_df.columns = [month_names[i - 1] for i in pivot_df.columns if i <= 12]

                    approach_aggregated[baseline_name] = pivot_df

            if approach_aggregated:
                aggregated_relative[approach] = approach_aggregated

        if interactive and HAS_PLOTLY:
            return self._create_relative_heatmap_interactive(aggregated_relative, save_path)
        elif HAS_MATPLOTLIB:
            return self._create_relative_heatmap_static(aggregated_relative, save_path)
        else:
            raise ImportError("Neither Plotly nor Matplotlib available for plotting")

    def _create_relative_heatmap_interactive(
        self, aggregated_relative: dict[str, dict[str, pd.DataFrame]], save_path: str | Path | None
    ) -> go.Figure:
        """Create interactive relative performance heat map using Plotly."""
        # Count total subplots needed
        total_plots = sum(len(baseline_dict) for baseline_dict in aggregated_relative.values())

        if total_plots == 0:
            raise ValueError("No relative performance data available")

        # Create subplot structure
        approaches = list(aggregated_relative.keys())
        max_baselines = max(len(baseline_dict) for baseline_dict in aggregated_relative.values())

        fig = make_subplots(
            rows=len(approaches),
            cols=max_baselines,
            subplot_titles=[],  # We'll add titles manually
            vertical_spacing=0.08,
            horizontal_spacing=0.05,
        )

        # Determine global color scale for excess returns
        all_excess_values = []
        for approach_dict in aggregated_relative.values():
            for df in approach_dict.values():
                all_excess_values.extend(df.values.flatten())
        all_excess_values = [v for v in all_excess_values if not pd.isna(v)]

        if all_excess_values:
            abs_max = max(abs(min(all_excess_values)), abs(max(all_excess_values)))
            vmin, vmax = -abs_max, abs_max
        else:
            vmin, vmax = -0.05, 0.05

        # Create heat maps
        for row_idx, (approach, baseline_dict) in enumerate(aggregated_relative.items()):
            for col_idx, (baseline_name, df) in enumerate(baseline_dict.items()):
                if df.empty:
                    continue

                # Add heatmap
                fig.add_trace(
                    go.Heatmap(
                        z=df.values,
                        x=list(df.columns),
                        y=list(df.index),
                        colorscale="RdBu",
                        zmid=0,
                        zmin=vmin,
                        zmax=vmax,
                        showscale=(row_idx == 0 and col_idx == 0),
                        colorbar=(
                            {"title": "Excess Return", "tickformat": ".2%"}
                            if (row_idx == 0 and col_idx == 0)
                            else None
                        ),
                        hovertemplate=f"<b>{approach} vs {baseline_name}</b><br>"
                        + "%{y} %{x}<br>Excess Return: %{z:.2%}<extra></extra>",
                    ),
                    row=row_idx + 1,
                    col=col_idx + 1,
                )

        # Update layout
        fig.update_layout(
            title="Relative Performance Heat Maps (ML vs Baselines)", height=250 * len(approaches)
        )

        if save_path:
            fig.write_html(str(save_path).replace(".png", ".html"))

        return fig

    def _create_relative_heatmap_static(
        self, aggregated_relative: dict[str, dict[str, pd.DataFrame]], save_path: str | Path | None
    ) -> plt.Figure:
        """Create static relative performance heat map using Matplotlib."""
        # Count total subplots needed
        approaches = list(aggregated_relative.keys())
        max_baselines = max(len(baseline_dict) for baseline_dict in aggregated_relative.values())

        # Create subplot grid
        fig, axes = plt.subplots(
            len(approaches), max_baselines, figsize=(4 * max_baselines, 3 * len(approaches))
        )

        if len(approaches) == 1 and max_baselines == 1:
            axes = [[axes]]
        elif len(approaches) == 1:
            axes = [axes]
        elif max_baselines == 1:
            axes = [[ax] for ax in axes]

        # Determine global color scale
        all_excess_values = []
        for approach_dict in aggregated_relative.values():
            for df in approach_dict.values():
                all_excess_values.extend(df.values.flatten())
        all_excess_values = [v for v in all_excess_values if not pd.isna(v)]

        if all_excess_values:
            abs_max = max(abs(min(all_excess_values)), abs(max(all_excess_values)))
            vmin, vmax = -abs_max, abs_max
        else:
            vmin, vmax = -0.05, 0.05

        # Create heat maps
        for row_idx, (approach, baseline_dict) in enumerate(aggregated_relative.items()):
            for col_idx, (baseline_name, df) in enumerate(baseline_dict.items()):
                ax = axes[row_idx][col_idx]

                if df.empty:
                    ax.text(
                        0.5,
                        0.5,
                        "No Data Available",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(f"{approach} vs {baseline_name}")
                    continue

                # Create heatmap
                sns.heatmap(
                    df,
                    ax=ax,
                    cmap="RdBu_r",
                    center=0,
                    vmin=vmin,
                    vmax=vmax,
                    annot=True,
                    fmt=".1%",
                    annot_kws={"size": 6},
                    cbar=(row_idx == 0 and col_idx == 0),
                    cbar_kws={"format": "%.1%%"} if (row_idx == 0 and col_idx == 0) else None,
                )

                ax.set_title(f"{approach} vs {baseline_name}", fontsize=10)
                ax.set_xlabel("")
                ax.set_ylabel("Year" if col_idx == 0 else "")

            # Hide unused subplots
            for col_idx in range(len(baseline_dict), max_baselines):
                axes[row_idx][col_idx].set_visible(False)

        fig.suptitle(
            "Relative Performance Heat Maps (ML vs Baselines)", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")

        return fig

    def add_statistical_significance_overlay(
        self,
        performance_heatmap_data: dict[str, pd.DataFrame],
        statistical_results: dict[str, dict[str, float]],
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """
        Add statistical significance overlays on heat maps.

        Args:
            performance_heatmap_data: Heat map data from monthly performance analysis
            statistical_results: Statistical significance results by approach and period
            save_path: Path to save the annotated heat map

        Returns:
            Matplotlib Figure with significance overlays
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib required for significance overlays")

        n_approaches = len(performance_heatmap_data)
        figsize = (self.config.figsize[0], 3 * n_approaches)
        fig, axes = plt.subplots(n_approaches, 1, figsize=figsize)

        if n_approaches == 1:
            axes = [axes]

        # Determine global color scale
        all_values = []
        for df in performance_heatmap_data.values():
            all_values.extend(df.values.flatten())
        all_values = [v for v in all_values if not pd.isna(v)]

        if all_values:
            abs_max = max(abs(min(all_values)), abs(max(all_values)))
            vmin, vmax = -abs_max, abs_max
        else:
            vmin, vmax = -0.1, 0.1

        for idx, (approach, df) in enumerate(performance_heatmap_data.items()):
            ax = axes[idx]

            if df.empty:
                continue

            # Create base heatmap
            sns.heatmap(
                df,
                ax=ax,
                cmap=self.config.colormap,
                center=self.config.center_value,
                vmin=vmin,
                vmax=vmax,
                annot=False,  # We'll add custom annotations
                cbar=(idx == 0),
                cbar_kws={"format": "%.1%%"} if idx == 0 else None,
            )

            # Add custom annotations with significance symbols
            for i in range(len(df.index)):
                for j in range(len(df.columns)):
                    value = df.iloc[i, j]
                    if pd.notna(value):
                        # Get significance for this period
                        year = df.index[i]
                        month = df.columns[j]
                        period_key = f"{year}_{month}"

                        # Check if we have statistical results for this approach and period
                        significance_symbol = ""
                        if (
                            approach in statistical_results
                            and period_key in statistical_results[approach]
                        ):
                            p_value = statistical_results[approach][period_key]
                            significance_symbol = self._get_significance_symbol(p_value)

                        # Combine value and significance symbol
                        display_text = f"{value:.1%}{significance_symbol}"

                        # Choose text color based on background
                        text_color = "white" if abs(value) > abs_max * 0.5 else "black"

                        ax.text(
                            j + 0.5,
                            i + 0.5,
                            display_text,
                            ha="center",
                            va="center",
                            color=text_color,
                            fontsize=self.config.annot_fontsize,
                            fontweight="bold" if significance_symbol else "normal",
                        )

            ax.set_title(
                f"{approach} (with significance levels)",
                fontsize=self.config.title_fontsize,
                fontweight="bold",
            )
            ax.set_xlabel("")
            ax.set_ylabel("Year" if idx == n_approaches - 1 else "")

        # Add legend for significance symbols
        legend_text = "Significance levels: * p<0.05, ** p<0.01, *** p<0.001"
        fig.text(0.5, 0.02, legend_text, ha="center", fontsize=10, style="italic")

        fig.suptitle(
            "Monthly Performance with Statistical Significance",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")

        return fig

    def add_regime_identification_overlay(
        self,
        performance_heatmap_data: dict[str, pd.DataFrame],
        regime_data: pd.Series,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """
        Implement regime identification and annotation on heat maps.

        Args:
            performance_heatmap_data: Heat map data from monthly performance analysis
            regime_data: Series with regime labels indexed by date
            save_path: Path to save the annotated heat map

        Returns:
            Matplotlib Figure with regime annotations
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib required for regime overlays")

        n_approaches = len(performance_heatmap_data)
        figsize = (self.config.figsize[0], 3.5 * n_approaches)
        fig, axes = plt.subplots(n_approaches, 1, figsize=figsize)

        if n_approaches == 1:
            axes = [axes]

        # Define regime colors
        regime_colors = {"bull": "green", "bear": "red", "sideways": "orange", "neutral": "gray"}

        # Determine global color scale
        all_values = []
        for df in performance_heatmap_data.values():
            all_values.extend(df.values.flatten())
        all_values = [v for v in all_values if not pd.isna(v)]

        if all_values:
            abs_max = max(abs(min(all_values)), abs(max(all_values)))
            vmin, vmax = -abs_max, abs_max
        else:
            vmin, vmax = -0.1, 0.1

        for idx, (approach, df) in enumerate(performance_heatmap_data.items()):
            ax = axes[idx]

            if df.empty:
                continue

            # Create base heatmap
            sns.heatmap(
                df,
                ax=ax,
                cmap=self.config.colormap,
                center=self.config.center_value,
                vmin=vmin,
                vmax=vmax,
                annot=True,
                fmt=".1%",
                annot_kws={"size": self.config.annot_fontsize},
                cbar=(idx == 0),
                cbar_kws={"format": "%.1%%"} if idx == 0 else None,
            )

            # Add regime borders/indicators
            for i, year in enumerate(df.index):
                for j, month_name in enumerate(df.columns):
                    # Convert month name back to number
                    month_names = [
                        "Jan",
                        "Feb",
                        "Mar",
                        "Apr",
                        "May",
                        "Jun",
                        "Jul",
                        "Aug",
                        "Sep",
                        "Oct",
                        "Nov",
                        "Dec",
                    ]
                    month = month_names.index(month_name) + 1 if month_name in month_names else 1

                    # Find regime for this year-month
                    period_date = pd.Timestamp(year=year, month=month, day=15)  # Mid-month

                    # Find closest regime data
                    if not regime_data.empty:
                        closest_idx = regime_data.index.get_indexer(
                            [period_date], method="nearest"
                        )[0]
                        if closest_idx >= 0 and closest_idx < len(regime_data):
                            regime = regime_data.iloc[closest_idx]

                            # Add colored border based on regime
                            color = regime_colors.get(regime.lower(), "gray")

                            # Add border rectangle
                            from matplotlib.patches import Rectangle

                            rect = Rectangle(
                                (j, i), 1, 1, linewidth=3, edgecolor=color, facecolor="none"
                            )
                            ax.add_patch(rect)

            ax.set_title(
                f"{approach} (with market regimes)",
                fontsize=self.config.title_fontsize,
                fontweight="bold",
            )
            ax.set_xlabel("")
            ax.set_ylabel("Year" if idx == n_approaches - 1 else "")

        # Create regime legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color=color, lw=3, label=f"{regime.capitalize()} Market")
            for regime, color in regime_colors.items()
        ]

        fig.legend(handles=legend_elements, loc="center", bbox_to_anchor=(0.5, 0.02), ncol=4)

        fig.suptitle(
            "Monthly Performance with Market Regime Identification",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)  # Make room for legend

        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")

        return fig

    def _get_significance_symbol(self, p_value: float) -> str:
        """Get significance symbol based on p-value."""
        if p_value <= 0.001:
            return "***"
        elif p_value <= 0.01:
            return "**"
        elif p_value <= 0.05:
            return "*"
        else:
            return ""

    def export_heatmap(
        self, figure: plt.Figure | go.Figure, filename: str, formats: list[str] = None
    ) -> dict[str, str]:
        """
        Export heat map in multiple formats.

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
