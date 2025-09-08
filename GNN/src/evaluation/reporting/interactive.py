"""
Interactive dashboard framework for comprehensive portfolio performance analysis.

This module provides a unified dashboard framework that integrates all visualization
components into interactive, web-based dashboards with real-time updates and
comprehensive analysis capabilities.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    warnings.warn("Plotly not available. Interactive dashboards disabled.", stacklevel=2)

from src.evaluation.reporting.charts import TimeSeriesCharts
from src.evaluation.reporting.heatmaps import PerformanceHeatmaps
from src.evaluation.reporting.operational_analysis import OperationalEfficiencyAnalysis
from src.evaluation.reporting.regime_analysis import MarketRegimeAnalysis
from src.evaluation.reporting.risk_return import RiskReturnAnalysis
from src.evaluation.reporting.tables import PerformanceComparisonTables


@dataclass
class DashboardConfig:
    """Configuration for interactive dashboards."""

    height: int = 1000
    width: int = 1400
    theme: str = "plotly_white"
    update_frequency: int = 5000
    export_formats: list[str] = None
    responsive: bool = True
    
    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ["html", "png", "pdf"]


class InteractiveDashboard:
    """
    Interactive dashboard framework for comprehensive portfolio performance analysis.
    
    Provides unified dashboards integrating all visualization components with
    real-time updates, interactive controls, and comprehensive analysis capabilities.
    """

    def __init__(
        self,
        config: DashboardConfig = None,
        config_path: str | Path | None = None,
    ):
        """
        Initialize interactive dashboard framework.

        Args:
            config: Dashboard configuration object
            config_path: Path to dashboard configuration YAML file
        """
        if config_path:
            self.config = self._load_config(config_path)
        else:
            self.config = config or DashboardConfig()
            
        if not HAS_PLOTLY:
            raise ImportError("Plotly required for interactive dashboards")
            
        # Initialize visualization components
        self.time_series = TimeSeriesCharts()
        self.tables = PerformanceComparisonTables()
        self.heatmaps = PerformanceHeatmaps()
        self.risk_return = RiskReturnAnalysis()
        self.operational = OperationalEfficiencyAnalysis()
        self.regime_analysis = MarketRegimeAnalysis()

    def _load_config(self, config_path: str | Path) -> DashboardConfig:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
            
        # Extract relevant configuration
        global_config = config_data.get("global", {})
        
        return DashboardConfig(
            height=global_config.get("height", 1000),
            width=global_config.get("width", 1400),
            theme=global_config.get("theme", "plotly_white"),
            update_frequency=global_config.get("update_frequency", 5000),
            export_formats=global_config.get("export_formats", ["html", "png", "pdf"]),
            responsive=global_config.get("responsive", True),
        )

    def create_main_performance_dashboard(
        self,
        returns_data: dict[str, pd.Series],
        performance_metrics: dict[str, dict[str, float]],
        rolling_metrics: dict[str, dict[str, pd.Series]] | None = None,
        benchmark_data: dict[str, pd.Series] | None = None,
        save_path: str | Path | None = None,
    ) -> go.Figure:
        """
        Create comprehensive main performance dashboard.

        Args:
            returns_data: Dictionary mapping approach names to return series
            performance_metrics: Performance metrics by approach
            rolling_metrics: Optional rolling performance metrics
            benchmark_data: Optional benchmark returns for comparison
            save_path: Path to save the dashboard

        Returns:
            Plotly Figure with comprehensive dashboard
        """
        # Create subplot structure
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Cumulative Returns Comparison",
                "",
                "Drawdown Analysis", 
                "Key Performance Metrics",
                "Risk-Return Analysis",
                "Rolling Sharpe Ratio",
            ),
            specs=[
                [{"colspan": 2}, None],  # Cumulative returns (full width)
                [{}, {"type": "table"}],  # Drawdown + metrics table
                [{}, {}],  # Risk-return + rolling metrics
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1,
        )

        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]

        # 1. Cumulative Returns (top, full width)
        for idx, (approach, returns_series) in enumerate(returns_data.items()):
            cum_returns = (1 + returns_series).cumprod()
            
            fig.add_trace(
                go.Scatter(
                    x=cum_returns.index,
                    y=cum_returns.values,
                    mode="lines",
                    name=approach,
                    line={"color": colors[idx % len(colors)], "width": 2},
                    hovertemplate=f"<b>{approach}</b><br>"
                    + "Date: %{x}<br>"
                    + "Cumulative Return: %{y:.2%}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # Add benchmark if provided
        if benchmark_data:
            for benchmark_name, benchmark_returns in benchmark_data.items():
                benchmark_cum = (1 + benchmark_returns).cumprod()
                fig.add_trace(
                    go.Scatter(
                        x=benchmark_cum.index,
                        y=benchmark_cum.values,
                        mode="lines",
                        name=f"{benchmark_name} (Benchmark)",
                        line={"color": "black", "width": 2, "dash": "dash"},
                        hovertemplate=f"<b>{benchmark_name}</b><br>"
                        + "Date: %{x}<br>"
                        + "Cumulative Return: %{y:.2%}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

        # 2. Drawdown Analysis (middle left)
        for idx, (approach, returns_series) in enumerate(returns_data.items()):
            cum_returns = (1 + returns_series).cumprod()
            running_max = cum_returns.expanding(min_periods=1).max()
            drawdown = (cum_returns - running_max) / running_max

            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    mode="lines",
                    name=f"{approach} DD",
                    line={"color": colors[idx % len(colors)], "width": 2},
                    fill="tonexty" if idx > 0 else "tozeroy",
                    fillcolor=f"rgba({colors[idx % len(colors)][1:]}, 0.3)"
                    if colors[idx % len(colors)].startswith("#")
                    else f"rgba(128,128,128,0.3)",
                    showlegend=False,
                    hovertemplate=f"<b>{approach}</b><br>"
                    + "Date: %{x}<br>"
                    + "Drawdown: %{y:.2%}<extra></extra>",
                ),
                row=2,
                col=1,
            )

        # 3. Performance Metrics Table (middle right)
        if performance_metrics:
            table_data = []
            approaches = list(performance_metrics.keys())
            metrics = ["sharpe_ratio", "total_return", "max_drawdown", "volatility"]
            
            for metric in metrics:
                row_data = [metric.replace("_", " ").title()]
                for approach in approaches:
                    value = performance_metrics[approach].get(metric, 0)
                    if metric in ["total_return", "max_drawdown", "volatility"]:
                        row_data.append(f"{value:.2%}")
                    else:
                        row_data.append(f"{value:.3f}")
                table_data.append(row_data)

            fig.add_trace(
                go.Table(
                    header={
                        "values": ["Metric"] + approaches,
                        "fill_color": "lightblue",
                        "align": "left",
                        "font": {"size": 10},
                    },
                    cells={
                        "values": list(zip(*table_data)),
                        "fill_color": "white",
                        "align": "left",
                        "font": {"size": 9},
                    },
                ),
                row=2,
                col=2,
            )

        # 4. Risk-Return Scatter (bottom left)
        for idx, approach in enumerate(returns_data.keys()):
            if approach in performance_metrics:
                metrics = performance_metrics[approach]
                volatility = metrics.get("volatility", 0)
                ann_return = metrics.get("annualized_return", metrics.get("total_return", 0))
                sharpe = metrics.get("sharpe_ratio", 0)

                fig.add_trace(
                    go.Scatter(
                        x=[volatility],
                        y=[ann_return],
                        mode="markers+text",
                        text=[approach],
                        textposition="top center",
                        marker={
                            "size": 15,
                            "color": sharpe,
                            "colorscale": "viridis",
                            "showscale": idx == 0,
                            "colorbar": {"title": "Sharpe Ratio", "x": 0.48} if idx == 0 else None,
                            "line": {"width": 2, "color": "DarkSlateGrey"},
                        },
                        name=f"{approach} Risk-Return",
                        showlegend=False,
                        hovertemplate=f"<b>{approach}</b><br>"
                        + "Return: %{y:.2%}<br>"
                        + "Volatility: %{x:.2%}<br>"
                        + f"Sharpe: {sharpe:.3f}<extra></extra>",
                    ),
                    row=3,
                    col=1,
                )

        # 5. Rolling Sharpe Ratio (bottom right)
        if rolling_metrics:
            for idx, (approach, metrics_dict) in enumerate(rolling_metrics.items()):
                if "sharpe_ratio" in metrics_dict:
                    sharpe_series = metrics_dict["sharpe_ratio"]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=sharpe_series.index,
                            y=sharpe_series.values,
                            mode="lines",
                            name=f"{approach} Rolling Sharpe",
                            line={"color": colors[idx % len(colors)], "width": 2},
                            showlegend=False,
                            hovertemplate=f"<b>{approach}</b><br>"
                            + "Date: %{x}<br>"
                            + "Rolling Sharpe: %{y:.3f}<extra></extra>",
                        ),
                        row=3,
                        col=2,
                    )

        # Update layout
        fig.update_layout(
            title="Portfolio Performance Dashboard",
            height=self.config.height,
            width=self.config.width,
            template=self.config.theme,
            showlegend=True,
            hovermode="x unified",
        )

        # Update axes
        fig.update_yaxes(title_text="Cumulative Return", tickformat=".1%", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=1)
        
        fig.update_yaxes(title_text="Drawdown", tickformat=".1%", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        fig.update_yaxes(title_text="Return", tickformat=".1%", row=3, col=1)
        fig.update_xaxes(title_text="Volatility", tickformat=".1%", row=3, col=1)
        
        fig.update_yaxes(title_text="Rolling Sharpe", row=3, col=2)
        fig.update_xaxes(title_text="Date", row=3, col=2)

        # Add range selector and slider
        fig.update_xaxes(
            rangeselector={
                "buttons": [
                    {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
                    {"count": 2, "label": "2Y", "step": "year", "stepmode": "backward"},
                    {"count": 5, "label": "5Y", "step": "year", "stepmode": "backward"},
                    {"step": "all"},
                ]
            },
            rangeslider={"visible": True},
            row=1,
            col=1,
        )

        if save_path:
            fig.write_html(str(save_path))

        return fig

    def create_risk_analysis_dashboard(
        self,
        returns_data: dict[str, pd.Series],
        performance_metrics: dict[str, dict[str, float]],
        regime_data: pd.Series | None = None,
        save_path: str | Path | None = None,
    ) -> go.Figure:
        """
        Create comprehensive risk analysis dashboard.

        Args:
            returns_data: Dictionary mapping approach names to return series
            performance_metrics: Performance metrics by approach
            regime_data: Optional regime classification data
            save_path: Path to save the dashboard

        Returns:
            Plotly Figure with risk analysis dashboard
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Value at Risk Analysis",
                "Return Correlation Matrix",
                "Regime-Specific Performance", 
                "Tail Risk Metrics",
            ),
            specs=[
                [{}, {"type": "heatmap"}],
                [{}, {"type": "bar"}],
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
        )

        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"
        ]

        # 1. VaR Analysis (top left)
        var_levels = [0.95, 0.99, 0.999]
        for idx, (approach, returns_series) in enumerate(returns_data.items()):
            var_values = []
            for level in var_levels:
                var = returns_series.quantile(1 - level)
                var_values.append(var * 100)  # Convert to percentage points
                
            fig.add_trace(
                go.Bar(
                    x=[f"{level:.1%}" for level in var_levels],
                    y=var_values,
                    name=f"{approach} VaR",
                    marker_color=colors[idx % len(colors)],
                    opacity=0.8,
                ),
                row=1,
                col=1,
            )

        # 2. Correlation Matrix (top right)
        returns_df = pd.DataFrame(returns_data)
        corr_matrix = returns_df.corr()
        
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale="RdBu",
                zmid=0,
                showscale=True,
                colorbar={"title": "Correlation"},
                hovertemplate="<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # 3. Regime-Specific Performance (bottom left)
        if regime_data is not None:
            # Calculate regime-specific metrics
            unique_regimes = regime_data.dropna().unique()
            
            for idx, regime in enumerate(unique_regimes[:3]):  # Limit to 3 regimes
                regime_returns = []
                approach_names = []
                
                for approach, returns_series in returns_data.items():
                    regime_mask = regime_data == regime
                    regime_performance = returns_series[regime_mask]
                    
                    if len(regime_performance) > 10:
                        avg_return = regime_performance.mean() * 252  # Annualized
                        regime_returns.append(avg_return * 100)  # Convert to percentage
                        approach_names.append(approach)
                
                if regime_returns:
                    fig.add_trace(
                        go.Bar(
                            x=approach_names,
                            y=regime_returns,
                            name=f"{regime.capitalize()} Regime",
                            marker_color=colors[idx % len(colors)],
                            opacity=0.8,
                        ),
                        row=2,
                        col=1,
                    )

        # 4. Tail Risk Metrics (bottom right)
        approaches = []
        skewness_values = []
        kurtosis_values = []
        
        for approach, returns_series in returns_data.items():
            approaches.append(approach)
            skewness_values.append(returns_series.skew())
            kurtosis_values.append(returns_series.kurtosis())
            
        fig.add_trace(
            go.Bar(
                x=approaches,
                y=skewness_values,
                name="Skewness",
                marker_color="lightcoral",
                opacity=0.8,
            ),
            row=2,
            col=2,
        )
        
        fig.add_trace(
            go.Bar(
                x=approaches,
                y=kurtosis_values,
                name="Excess Kurtosis",
                marker_color="darkred",
                opacity=0.8,
                yaxis="y2",
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title="Risk Analysis Dashboard",
            height=self.config.height,
            width=self.config.width,
            template=self.config.theme,
            showlegend=True,
        )

        # Update axes
        fig.update_yaxes(title_text="VaR (%)", row=1, col=1)
        fig.update_xaxes(title_text="Confidence Level", row=1, col=1)
        
        if regime_data is not None:
            fig.update_yaxes(title_text="Annualized Return (%)", row=2, col=1)
            fig.update_xaxes(title_text="Approach", row=2, col=1)
        
        fig.update_yaxes(title_text="Skewness", row=2, col=2)
        fig.update_yaxes(title_text="Excess Kurtosis", secondary_y=True, row=2, col=2)
        fig.update_xaxes(title_text="Approach", row=2, col=2)

        if save_path:
            fig.write_html(str(save_path))

        return fig

    def create_operational_dashboard(
        self,
        returns_data: dict[str, pd.Series],
        turnover_data: dict[str, pd.Series],
        performance_metrics: dict[str, dict[str, float]],
        constraint_data: dict[str, dict[str, pd.Series]] | None = None,
        save_path: str | Path | None = None,
    ) -> go.Figure:
        """
        Create comprehensive operational efficiency dashboard.

        Args:
            returns_data: Dictionary mapping approach names to return series
            turnover_data: Dictionary mapping approach names to turnover series
            performance_metrics: Performance metrics by approach
            constraint_data: Optional constraint violation data
            save_path: Path to save the dashboard

        Returns:
            Plotly Figure with operational dashboard
        """
        fig = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=(
                "Portfolio Turnover",
                "Transaction Cost Impact",
                "Implementation Shortfall",
                "Constraint Violations",
                "Operational Efficiency",
                "Cost Attribution",
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.06,
        )

        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"
        ]

        # 1. Portfolio Turnover (top left)
        for idx, (approach, turnover_series) in enumerate(turnover_data.items()):
            fig.add_trace(
                go.Scatter(
                    x=turnover_series.index,
                    y=turnover_series.values,
                    mode="lines+markers",
                    name=f"{approach}",
                    line={"color": colors[idx % len(colors)], "width": 2},
                    marker={"size": 4},
                ),
                row=1,
                col=1,
            )

        # 2. Transaction Cost Impact (top middle)
        cost_bps = 5.0  # 5 basis points default
        approaches = []
        gross_sharpe = []
        net_sharpe = []
        
        for approach in returns_data.keys():
            if approach in turnover_data and approach in performance_metrics:
                approaches.append(approach)
                gross_sr = performance_metrics[approach].get("sharpe_ratio", 0)
                
                # Estimate net Sharpe after transaction costs
                avg_turnover = turnover_data[approach].mean()
                cost_drag = avg_turnover * (cost_bps / 10000) * 12  # Annualized
                returns_series = returns_data[approach]
                net_returns_mean = returns_series.mean() - cost_drag / 252
                net_sr = net_returns_mean / returns_series.std() * (252 ** 0.5)
                
                gross_sharpe.append(gross_sr)
                net_sharpe.append(net_sr)
                
        fig.add_trace(
            go.Bar(
                x=approaches,
                y=gross_sharpe,
                name="Gross Sharpe",
                marker_color="lightblue",
                opacity=0.8,
            ),
            row=1,
            col=2,
        )
        
        fig.add_trace(
            go.Bar(
                x=approaches,
                y=net_sharpe,
                name="Net Sharpe",
                marker_color="darkblue",
                opacity=0.8,
            ),
            row=1,
            col=2,
        )

        # 3. Implementation Shortfall (top right)
        # Simplified shortfall analysis
        for idx, approach in enumerate(approaches):
            # Estimate shortfall as function of turnover and volatility
            if approach in turnover_data and approach in returns_data:
                turnover_series = turnover_data[approach]
                returns_series = returns_data[approach]
                volatility = returns_series.std() * (252 ** 0.5)
                
                # Simple shortfall estimate: turnover * volatility * cost factor
                shortfall_series = turnover_series * volatility * 0.01  # 1% of vol per turnover
                
                fig.add_trace(
                    go.Scatter(
                        x=shortfall_series.index,
                        y=shortfall_series.values * 10000,  # Convert to basis points
                        mode="lines",
                        name=f"{approach} Shortfall",
                        line={"color": colors[idx % len(colors)], "width": 2},
                        showlegend=False,
                    ),
                    row=1,
                    col=3,
                )

        # 4. Constraint Violations (bottom left)
        if constraint_data:
            violation_rates = {}
            for approach, constraints in constraint_data.items():
                total_violations = 0
                total_periods = 0
                
                for constraint_name, violation_series in constraints.items():
                    violations = (violation_series > 0.05).sum()  # 5% threshold
                    total_violations += violations
                    total_periods += len(violation_series)
                    
                if total_periods > 0:
                    violation_rates[approach] = total_violations / total_periods
                    
            if violation_rates:
                fig.add_trace(
                    go.Bar(
                        x=list(violation_rates.keys()),
                        y=list(violation_rates.values()),
                        name="Violation Rate",
                        marker_color="red",
                        opacity=0.8,
                    ),
                    row=2,
                    col=1,
                )

        # 5. Operational Efficiency Scatter (bottom middle)
        efficiency_x = []
        efficiency_y = []
        efficiency_labels = []
        
        for approach in approaches:
            if approach in turnover_data and approach in performance_metrics:
                avg_turnover = turnover_data[approach].mean()
                sharpe_ratio = performance_metrics[approach].get("sharpe_ratio", 0)
                
                efficiency_x.append(avg_turnover)
                efficiency_y.append(sharpe_ratio)
                efficiency_labels.append(approach)
                
        fig.add_trace(
            go.Scatter(
                x=efficiency_x,
                y=efficiency_y,
                mode="markers+text",
                text=efficiency_labels,
                textposition="top center",
                marker={"size": 12, "opacity": 0.8},
                name="Efficiency",
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        # 6. Cost Attribution (bottom right)
        if approaches:
            cost_components = ["Trading", "Market Impact", "Timing", "Other"]
            cost_values = [40, 30, 20, 10]  # Example percentages
            
            fig.add_trace(
                go.Bar(
                    x=cost_components,
                    y=cost_values,
                    name="Cost Attribution",
                    marker_color=colors[:len(cost_components)],
                    showlegend=False,
                ),
                row=2,
                col=3,
            )

        # Update layout
        fig.update_layout(
            title="Operational Efficiency Dashboard",
            height=self.config.height,
            width=self.config.width,
            template=self.config.theme,
            showlegend=True,
        )

        # Update axes titles
        fig.update_yaxes(title_text="Turnover", tickformat=".1%", row=1, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=2)
        fig.update_yaxes(title_text="Shortfall (bps)", row=1, col=3)
        
        if constraint_data:
            fig.update_yaxes(title_text="Violation Rate", tickformat=".1%", row=2, col=1)
            
        fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=2)
        fig.update_xaxes(title_text="Turnover", tickformat=".1%", row=2, col=2)
        fig.update_yaxes(title_text="Cost (%)", row=2, col=3)

        if save_path:
            fig.write_html(str(save_path))

        return fig

    def export_dashboard(
        self,
        figure: go.Figure,
        filename: str,
        formats: list[str] = None,
    ) -> dict[str, str]:
        """
        Export dashboard in multiple formats.

        Args:
            figure: Plotly figure to export
            filename: Base filename for export
            formats: Export formats (defaults to config)

        Returns:
            Dictionary mapping format names to file paths
        """
        export_formats = formats or self.config.export_formats
        exported_files = {}

        for format_type in export_formats:
            filepath = f"{filename}.{format_type}"

            try:
                if format_type == "html":
                    figure.write_html(filepath, include_plotlyjs="cdn")
                    exported_files["html"] = filepath
                elif format_type in ["png", "pdf", "svg"]:
                    figure.write_image(filepath, width=self.config.width, height=self.config.height)
                    exported_files[format_type] = filepath
            except Exception as e:
                warnings.warn(f"Failed to export {format_type}: {e}", stacklevel=2)

        return exported_files

    def create_summary_report(
        self,
        returns_data: dict[str, pd.Series],
        performance_metrics: dict[str, dict[str, float]],
        statistical_results: dict[str, dict[str, Any]] | None = None,
        save_path: str | Path | None = None,
    ) -> str:
        """
        Create comprehensive summary report combining all analyses.

        Args:
            returns_data: Dictionary mapping approach names to return series
            performance_metrics: Performance metrics by approach
            statistical_results: Optional statistical significance results
            save_path: Path to save the report

        Returns:
            HTML string with complete report
        """
        # Create individual dashboards
        main_dashboard = self.create_main_performance_dashboard(
            returns_data, performance_metrics
        )
        
        # Create summary tables
        performance_table = self.tables.create_performance_ranking_table(
            performance_metrics, statistical_results
        )

        # Generate HTML report
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Portfolio Performance Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin: 30px 0; }}
                .dashboard {{ margin: 20px 0; }}
                h1, h2 {{ color: #333; }}
                .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>Portfolio Performance Analysis Report</h1>
            
            <div class="section summary">
                <h2>Executive Summary</h2>
                <p>This report presents a comprehensive analysis of portfolio performance across multiple approaches, 
                including traditional and machine learning-based strategies.</p>
                
                <h3>Key Findings:</h3>
                <ul>
                    <li>Analysis period: {returns_data[list(returns_data.keys())[0]].index[0].strftime('%Y-%m-%d')} 
                        to {returns_data[list(returns_data.keys())[0]].index[-1].strftime('%Y-%m-%d')}</li>
                    <li>Number of approaches analyzed: {len(returns_data)}</li>
                    <li>Total observations: {len(returns_data[list(returns_data.keys())[0]])}</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Performance Dashboard</h2>
                <div class="dashboard">
                    {main_dashboard.to_html(include_plotlyjs=False, div_id="main-dashboard")}
                </div>
            </div>
            
            <div class="section">
                <h2>Performance Rankings</h2>
                {performance_table.to_html(classes='table table-striped', table_id='performance-table')}
            </div>
            
            <div class="section">
                <h2>Statistical Analysis</h2>
                {self._create_statistical_summary(statistical_results) if statistical_results else '<p>No statistical analysis available.</p>'}
            </div>
            
            <div class="section">
                <h2>Risk Analysis</h2>
                {self._create_risk_summary(returns_data, performance_metrics)}
            </div>
            
            <div class="section">
                <h2>Conclusions and Recommendations</h2>
                {self._create_conclusions(performance_metrics)}
            </div>
            
        </body>
        </html>
        """

        if save_path:
            with open(save_path, "w") as f:
                f.write(html_report)

        return html_report

    def _create_statistical_summary(self, statistical_results: dict[str, dict[str, Any]]) -> str:
        """Create statistical analysis summary HTML."""
        summary_html = "<h3>Statistical Significance Results</h3><ul>"
        
        for approach, results in statistical_results.items():
            significant_tests = 0
            total_tests = len(results)
            
            for test_name, test_data in results.items():
                if isinstance(test_data, dict) and test_data.get("p_value", 1.0) < 0.05:
                    significant_tests += 1
                    
            summary_html += f"<li><strong>{approach}</strong>: {significant_tests}/{total_tests} tests significant at 5% level</li>"
            
        summary_html += "</ul>"
        return summary_html

    def _create_risk_summary(
        self, returns_data: dict[str, pd.Series], performance_metrics: dict[str, dict[str, float]]
    ) -> str:
        """Create risk analysis summary HTML."""
        summary_html = "<h3>Risk Characteristics</h3><ul>"
        
        for approach, returns_series in returns_data.items():
            vol = returns_series.std() * (252 ** 0.5)
            max_dd = performance_metrics[approach].get("max_drawdown", 0)
            var_95 = returns_series.quantile(0.05)
            
            summary_html += f"""
            <li><strong>{approach}</strong>:
                <ul>
                    <li>Volatility: {vol:.1%}</li>
                    <li>Maximum Drawdown: {max_dd:.1%}</li>
                    <li>95% VaR: {var_95:.2%}</li>
                </ul>
            </li>
            """
            
        summary_html += "</ul>"
        return summary_html

    def _create_conclusions(self, performance_metrics: dict[str, dict[str, float]]) -> str:
        """Create conclusions and recommendations HTML."""
        # Find best performing approach by Sharpe ratio
        best_sharpe = max(
            performance_metrics.items(),
            key=lambda x: x[1].get("sharpe_ratio", 0)
        )
        
        # Find lowest risk approach by volatility
        lowest_vol = min(
            performance_metrics.items(),
            key=lambda x: x[1].get("volatility", float("inf"))
        )

        conclusions_html = f"""
        <h3>Key Insights</h3>
        <ul>
            <li><strong>Best Risk-Adjusted Performance</strong>: {best_sharpe[0]} 
                (Sharpe Ratio: {best_sharpe[1].get('sharpe_ratio', 0):.3f})</li>
            <li><strong>Lowest Risk Approach</strong>: {lowest_vol[0]} 
                (Volatility: {lowest_vol[1].get('volatility', 0):.1%})</li>
        </ul>
        
        <h3>Recommendations</h3>
        <ul>
            <li>Consider diversification across multiple approaches to reduce concentration risk</li>
            <li>Monitor regime changes and adjust allocations accordingly</li>
            <li>Regular rebalancing to maintain target risk levels</li>
            <li>Implement transaction cost controls to preserve net performance</li>
        </ul>
        """
        
        return conclusions_html