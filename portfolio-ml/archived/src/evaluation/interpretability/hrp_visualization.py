"""
HRP-specific visualization tools.

This module provides specialized visualization methods for HRP clustering analysis,
dendrogram plots, and allocation tree displays.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class HRPVisualizer:
    """
    Specialized visualizations for HRP analysis.
    """

    def __init__(self, theme: str = "plotly_white", width: int = 800, height: int = 600):
        """
        Initialize HRP visualizer.

        Args:
            theme: Plotly theme to use
            width: Default figure width
            height: Default figure height
        """
        self.theme = theme
        self.default_width = width
        self.default_height = height

    def plot_hrp_dendrogram(
        self,
        dendrogram_data: dict[str, Any],
        asset_names: list[str],
        title: str = "HRP Hierarchical Clustering Dendrogram",
        width: int | None = None,
        height: int | None = None,
    ) -> go.Figure:
        """
        Plot HRP dendrogram showing hierarchical clustering structure.

        Args:
            dendrogram_data: Dendrogram data from scipy
            asset_names: Asset names for labeling
            title: Plot title
            width: Figure width (optional)
            height: Figure height (optional)

        Returns:
            Plotly figure object
        """
        width = width or max(800, len(asset_names) * 15)
        height = height or self.default_height

        # Extract dendrogram components
        icoord = dendrogram_data['icoord']
        dcoord = dendrogram_data['dcoord']
        labels = dendrogram_data['ivl']

        fig = go.Figure()

        # Add dendrogram lines
        for _i, (x, y) in enumerate(zip(icoord, dcoord)):
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                line={"color": 'black', "width": 2},
                showlegend=False,
                hoverinfo='skip',
            ))

        # Add asset labels
        x_labels = np.arange(5, len(labels) * 10 + 5, 10)

        fig.add_trace(go.Scatter(
            x=x_labels,
            y=[0] * len(labels),
            mode='markers+text',
            text=labels,
            textposition="bottom center",
            marker={"size": 8, "color": 'blue'},
            showlegend=False,
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Assets",
            yaxis_title="Distance",
            template=self.theme,
            width=width,
            height=height,
            xaxis={"showticklabels": False},
        )

        return fig

    def plot_correlation_heatmap(
        self,
        correlation_matrix: pd.DataFrame,
        title: str = "Asset Correlation Matrix",
        cluster_order: list[str] | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> go.Figure:
        """
        Plot correlation matrix heatmap.

        Args:
            correlation_matrix: Asset correlation matrix
            title: Plot title
            cluster_order: Optional asset ordering from clustering
            width: Figure width (optional)
            height: Figure height (optional)

        Returns:
            Plotly figure object
        """
        width = width or self.default_width
        height = height or self.default_height

        # Reorder matrix if cluster order provided
        if cluster_order:
            valid_assets = [asset for asset in cluster_order if asset in correlation_matrix.index]
            if valid_assets:
                correlation_matrix = correlation_matrix.loc[valid_assets, valid_assets]

        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu_r',
            zmid=0,
            showscale=True,
            colorbar={"title": "Correlation"},
            text=np.round(correlation_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 8},
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Assets",
            yaxis_title="Assets",
            template=self.theme,
            width=width,
            height=height,
        )

        return fig

    def plot_allocation_tree(
        self,
        allocation_tree: dict[str, Any],
        title: str = "HRP Allocation Tree",
        width: int | None = None,
        height: int | None = None,
    ) -> go.Figure:
        """
        Plot HRP allocation tree.

        Args:
            allocation_tree: Allocation tree structure
            title: Plot title
            width: Figure width (optional)
            height: Figure height (optional)

        Returns:
            Plotly figure object
        """
        width = width or self.default_width
        height = height or self.default_height

        fig = go.Figure()

        # Plot root allocation
        if "root" in allocation_tree:
            root_data = allocation_tree["root"]
            assets = root_data["assets"]
            weights = [root_data["individual_weights"].get(asset, 0) for asset in assets]

            fig.add_trace(go.Bar(
                x=assets,
                y=weights,
                name="Portfolio Weights",
                marker_color='skyblue',
                text=[f"{w:.3f}" for w in weights],
                textposition='auto',
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Assets",
            yaxis_title="Portfolio Weight",
            template=self.theme,
            width=width,
            height=height,
            showlegend=False,
        )

        return fig

    def plot_cluster_composition(
        self,
        cluster_analysis: dict[str, Any],
        sector_alignment: dict[str, Any] | None = None,
        title: str = "HRP Cluster Composition",
        width: int | None = None,
        height: int | None = None,
    ) -> go.Figure:
        """
        Plot cluster composition analysis.

        Args:
            cluster_analysis: Cluster composition data
            sector_alignment: Optional sector alignment data
            title: Plot title
            width: Figure width (optional)
            height: Figure height (optional)

        Returns:
            Plotly figure object
        """
        width = width or self.default_width
        height = height or self.default_height

        clusters = cluster_analysis.get("clusters", {})

        if not clusters:
            return go.Figure().update_layout(title=f"{title} (No Data)")

        # Create pie chart for cluster sizes
        cluster_names = list(clusters.keys())
        cluster_sizes = [clusters[name]["size"] for name in cluster_names]

        fig = go.Figure(data=go.Pie(
            labels=[f"Cluster {name.split('_')[1]}" for name in cluster_names],
            values=cluster_sizes,
            textinfo='label+percent+value',
        ))

        fig.update_layout(
            title=title,
            template=self.theme,
            width=width,
            height=height,
        )

        return fig

    def create_hrp_dashboard(
        self,
        clustering_results: dict[str, Any],
        correlation_analysis: dict[str, Any],
        allocation_analysis: dict[str, Any],
        title: str = "HRP Analysis Dashboard",
    ) -> go.Figure:
        """
        Create comprehensive HRP dashboard.

        Args:
            clustering_results: Clustering analysis results
            correlation_analysis: Correlation analysis results
            allocation_analysis: Allocation analysis results
            title: Dashboard title

        Returns:
            Plotly dashboard figure
        """
        # Create 2x2 subplot layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Asset Clustering",
                "Correlation Matrix",
                "Portfolio Allocation",
                "Quality Metrics"
            ],
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}],
            ],
            horizontal_spacing=0.1,
            vertical_spacing=0.15,
        )

        # 1. Cluster visualization (simplified)
        if "asset_names" in clustering_results:
            asset_names = clustering_results["asset_names"]
            cluster_labels = clustering_results.get("cluster_labels", [])

            colors = px.colors.qualitative.Set3

            if len(cluster_labels) == len(asset_names):
                node_colors = [colors[label % len(colors)] for label in cluster_labels]
            else:
                node_colors = ['blue'] * len(asset_names)

            fig.add_trace(go.Scatter(
                x=list(range(len(asset_names))),
                y=[1] * len(asset_names),
                mode='markers+text',
                text=asset_names,
                textposition="top center",
                marker={"size": 12, "color": node_colors},
                name='Assets',
            ), row=1, col=1)

        # 2. Correlation heatmap
        if "correlation_matrix" in correlation_analysis:
            corr_matrix = correlation_analysis["correlation_matrix"]

            fig.add_trace(go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu_r',
                zmid=0,
                showscale=False,
            ), row=1, col=2)

        # 3. Portfolio weights
        if "portfolio_weights" in allocation_analysis:
            weights = allocation_analysis["portfolio_weights"]

            fig.add_trace(go.Bar(
                x=weights.index,
                y=weights.values,
                marker_color='lightgreen',
                text=[f"{w:.3f}" for w in weights.values],
                textposition='auto',
            ), row=2, col=1)

        # 4. Quality metrics
        if "quality_metrics" in clustering_results:
            metrics = clustering_results["quality_metrics"]

            metric_names = list(metrics.keys())[:4]  # Show top 4 metrics
            metric_values = [metrics[name] for name in metric_names]

            fig.add_trace(go.Bar(
                x=metric_names,
                y=metric_values,
                marker_color='orange',
            ), row=2, col=2)

        fig.update_layout(
            title=title,
            template=self.theme,
            width=1600,
            height=1200,
            showlegend=False,
        )

        # Update axis labels
        fig.update_xaxes(title_text="Asset Index", row=1, col=1)
        fig.update_xaxes(title_text="Assets", row=1, col=2)
        fig.update_xaxes(title_text="Assets", row=2, col=1)
        fig.update_xaxes(title_text="Metrics", row=2, col=2)

        fig.update_yaxes(title_text="Cluster", row=1, col=1)
        fig.update_yaxes(title_text="Assets", row=1, col=2)
        fig.update_yaxes(title_text="Weight", row=2, col=1)
        fig.update_yaxes(title_text="Value", row=2, col=2)

        return fig
