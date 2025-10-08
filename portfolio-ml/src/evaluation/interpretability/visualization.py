"""
Interpretability visualization framework.

This module provides visualization tools for model interpretability analysis,
including attention heatmaps, network graphs, temporal evolution charts,
and interactive dashboards.
"""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class InterpretabilityVisualizer:
    """
    Visualization tools for model interpretability analysis.

    Provides methods for creating interactive visualizations of attention weights,
    temporal patterns, portfolio explanations, and other interpretability metrics.
    """

    def __init__(self, theme: str = "plotly_white", width: int = 800, height: int = 600):
        """
        Initialize interpretability visualizer.

        Args:
            theme: Plotly theme to use for visualizations
            width: Default figure width
            height: Default figure height
        """
        self.theme = theme
        self.default_width = width
        self.default_height = height

        # Color palettes for different visualization types
        self.attention_colors = px.colors.sequential.Viridis
        self.temporal_colors = px.colors.qualitative.Set3
        self.network_colors = px.colors.qualitative.Plotly

    def plot_attention_heatmap(
        self,
        attention_matrix: pd.DataFrame,
        title: str = "GAT Attention Weights",
        width: int | None = None,
        height: int | None = None,
        show_values: bool = True,
        cluster_assets: bool = False,
    ) -> go.Figure:
        """
        Create heatmap visualization of attention weights.

        Args:
            attention_matrix: Attention weight matrix
            title: Plot title
            width: Figure width (optional)
            height: Figure height (optional)
            show_values: Whether to show values in cells
            cluster_assets: Whether to cluster assets by attention similarity

        Returns:
            Plotly figure object
        """
        width = width or self.default_width
        height = height or self.default_height

        # Apply clustering if requested
        if cluster_assets and len(attention_matrix) > 2:
            try:
                from scipy.cluster.hierarchy import dendrogram, linkage
                from scipy.spatial.distance import squareform

                # Create distance matrix from attention weights
                attention_dist = 1 - attention_matrix.fillna(0)
                condensed_dist = squareform(attention_dist, checks=False)

                # Perform hierarchical clustering
                linkage_matrix = linkage(condensed_dist, method='ward')
                dendro = dendrogram(linkage_matrix, no_plot=True)
                cluster_order = dendro['leaves']

                # Reorder matrix based on clustering
                reordered_assets = [attention_matrix.index[i] for i in cluster_order]
                attention_matrix = attention_matrix.loc[reordered_assets, reordered_assets]

            except ImportError:
                # Fall back to original order if scipy not available
                pass

        fig = go.Figure(data=go.Heatmap(
            z=attention_matrix.values,
            x=attention_matrix.columns,
            y=attention_matrix.index,
            colorscale="Viridis",
            showscale=True,
            text=attention_matrix.values.round(4) if show_values else None,
            texttemplate="%{text}" if show_values else None,
            textfont={"size": 10},
            colorbar={"title": "Attention Weight"},
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Target Asset",
            yaxis_title="Source Asset",
            template=self.theme,
            width=width,
            height=height,
        )

        return fig

    def plot_attention_network(
        self,
        attention_matrix: pd.DataFrame,
        threshold: float = 0.01,
        layout: str = "spring",
        title: str = "GAT Attention Network",
        width: int | None = None,
        height: int | None = None,
    ) -> go.Figure:
        """
        Create network visualization of attention weights.

        Args:
            attention_matrix: Attention weight matrix
            threshold: Minimum attention weight to show edge
            layout: Network layout algorithm ("spring", "circular", "kamada_kawai")
            title: Plot title
            width: Figure width (optional)
            height: Figure height (optional)

        Returns:
            Plotly figure object
        """
        width = width or self.default_width
        height = height or self.default_height

        # Create NetworkX graph
        G = nx.from_pandas_adjacency(attention_matrix, create_using=nx.DiGraph)

        # Remove edges below threshold
        edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < threshold]
        G.remove_edges_from(edges_to_remove)

        # Calculate node positions using specified layout
        if layout == "spring":
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)

        # Extract node and edge information
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = list(G.nodes())

        edge_x = []
        edge_y = []
        edge_info = []

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            weight = G[edge[0]][edge[1]]['weight']
            edge_info.append(f"{edge[0]} → {edge[1]}: {weight:.4f}")

        # Create the figure
        fig = go.Figure()

        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line={"width": 0.5, "color": '#888'},
            hoverinfo='none',
            mode='lines'
        ))

        # Add nodes with enhanced interactivity
        node_info = []
        for node in G.nodes():
            degree = G.degree(node)
            in_degree = G.in_degree(node) if hasattr(G, 'in_degree') else degree
            out_degree = G.out_degree(node) if hasattr(G, 'out_degree') else degree

            # Calculate weighted degree
            weighted_degree = sum(G[node][neighbor]['weight'] for neighbor in G.neighbors(node))

            info = (
                f"Asset: {node}<br>"
                f"Connections: {degree}<br>"
                f"Incoming: {in_degree}<br>"
                f"Outgoing: {out_degree}<br>"
                f"Weighted Degree: {weighted_degree:.4f}"
            )
            node_info.append(info)

        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            hovertext=node_info,
            marker={
                "showscale": True,
                "colorscale": 'Viridis',
                "reversescale": True,
                "color": [],
                "size": [max(10, min(25, G.degree(node) * 3)) for node in G.nodes()],  # Size by degree
                "colorbar": {
                    "thickness": 15,
                    "title": "Node Degree",
                    "xanchor": "left",
                    "titleside": "right"
                },
                "line_width": 2,
                "opacity": 0.8,
            }
        ))

        # Color nodes by degree centrality
        node_degrees = [G.degree(node) for node in G.nodes()]
        fig.data[-1].marker.color = node_degrees

        fig.update_layout(
            title=title,
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin={"b": 20, "l": 5, "r": 5, "t": 40},
            annotations=[
                {
                    "text": f"Attention threshold: {threshold}",
                    "showarrow": False,
                    "xref": "paper", "yref": "paper",
                    "x": 0.005, "y": -0.002,
                    "xanchor": 'left', "yanchor": 'bottom',
                    "font": {"color": 'grey', "size": 12}
                }
            ],
            xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            template=self.theme,
            width=width,
            height=height,
        )

        return fig

    def plot_temporal_attention_evolution(
        self,
        attention_evolution: pd.DataFrame,
        metrics: list[str] | None = None,
        title: str = "Attention Evolution Over Time",
        width: int | None = None,
        height: int | None = None,
    ) -> go.Figure:
        """
        Plot temporal evolution of attention metrics.

        Args:
            attention_evolution: DataFrame with temporal attention statistics
            metrics: List of metrics to plot (if None, plots key metrics)
            title: Plot title
            width: Figure width (optional)
            height: Figure height (optional)

        Returns:
            Plotly figure object
        """
        width = width or self.default_width
        height = height or max(400, self.default_height // 2)

        if metrics is None:
            metrics = ["mean_attention", "attention_concentration", "n_significant_connections"]

        # Create subplots for each metric
        n_metrics = len(metrics)
        fig = make_subplots(
            rows=n_metrics, cols=1,
            shared_xaxes=True,
            subplot_titles=metrics,
            vertical_spacing=0.08,
        )

        colors = itertools.cycle(self.temporal_colors)

        for i, metric in enumerate(metrics):
            if metric in attention_evolution.columns:
                color = next(colors)
                fig.add_trace(
                    go.Scatter(
                        x=attention_evolution['date'],
                        y=attention_evolution[metric],
                        mode='lines+markers',
                        name=metric,
                        line={"color": color},
                        marker={"color": color, "size": 4},
                    ),
                    row=i+1, col=1
                )

        fig.update_layout(
            title=title,
            template=self.theme,
            width=width,
            height=height * n_metrics,
            showlegend=False,
        )

        fig.update_xaxes(title_text="Date", row=n_metrics, col=1)

        return fig

    def plot_attention_evolution_heatmap(
        self,
        temporal_attention_data: dict[str, pd.DataFrame],
        connection_pair: tuple[str, str] | None = None,
        title: str = "Attention Evolution Over Time",
        width: int | None = None,
        height: int | None = None,
    ) -> go.Figure:
        """
        Plot attention evolution as a time-series heatmap.

        Args:
            temporal_attention_data: Dictionary mapping dates to attention matrices
            connection_pair: Specific connection pair to highlight (optional)
            title: Plot title
            width: Figure width (optional)
            height: Figure height (optional)

        Returns:
            Plotly figure object showing attention evolution over time
        """
        width = width or self.default_width
        height = height or max(400, self.default_height // 2)

        # Convert temporal data to time series format
        dates = sorted(temporal_attention_data.keys())

        if connection_pair:
            # Show evolution for specific connection pair
            source, target = connection_pair
            values = []

            for date in dates:
                matrix = temporal_attention_data[date]
                if source in matrix.index and target in matrix.columns:
                    values.append(matrix.loc[source, target])
                else:
                    values.append(0.0)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                mode='lines+markers',
                name=f"{source} → {target}",
                line={"width": 2},
                marker={"size": 6},
            ))

            fig.update_layout(
                title=f"{title}: {source} → {target}",
                xaxis_title="Date",
                yaxis_title="Attention Weight",
                template=self.theme,
                width=width,
                height=height,
            )
        else:
            # Show overall attention statistics evolution
            evolution_stats = []

            for date in dates:
                matrix = temporal_attention_data[date]
                np.fill_diagonal(matrix.values, 0)  # Remove self-attention

                stats = {
                    'date': date,
                    'mean_attention': matrix.values.mean(),
                    'max_attention': matrix.values.max(),
                    'attention_std': matrix.values.std(),
                }
                evolution_stats.append(stats)

            df = pd.DataFrame(evolution_stats)

            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                subplot_titles=["Mean Attention", "Max Attention", "Attention Std Dev"],
                vertical_spacing=0.1,
            )

            # Add traces for each metric
            metrics = ['mean_attention', 'max_attention', 'attention_std']
            colors = ['blue', 'red', 'green']

            for i, (metric, color) in enumerate(zip(metrics, colors)):
                fig.add_trace(
                    go.Scatter(
                        x=df['date'],
                        y=df[metric],
                        mode='lines+markers',
                        name=metric.replace('_', ' ').title(),
                        line={"color": color, "width": 2},
                        marker={"color": color, "size": 4},
                    ),
                    row=i+1, col=1
                )

            fig.update_layout(
                title=title,
                template=self.theme,
                width=width,
                height=height * 3,
                showlegend=False,
            )

            fig.update_xaxes(title_text="Date", row=3, col=1)

        return fig

    def plot_influential_connections(
        self,
        influential_connections: pd.DataFrame,
        top_k: int = 15,
        metric: str = "mean_attention",
        title: str = "Most Influential Asset Connections",
        width: int | None = None,
        height: int | None = None,
    ) -> go.Figure:
        """
        Plot most influential asset connections.

        Args:
            influential_connections: DataFrame with connection statistics
            top_k: Number of top connections to show
            metric: Metric to use for ranking
            title: Plot title
            width: Figure width (optional)
            height: Figure height (optional)

        Returns:
            Plotly figure object
        """
        width = width or self.default_width
        height = height or self.default_height

        # Get top connections
        top_connections = influential_connections.head(top_k).copy()

        # Create connection labels
        top_connections['connection_label'] = (
            top_connections['source_asset'] + ' → ' + top_connections['target_asset']
        )

        # Create horizontal bar chart
        fig = go.Figure(data=go.Bar(
            x=top_connections[metric],
            y=top_connections['connection_label'],
            orientation='h',
            marker={
                "color": top_connections[metric],
                "colorscale": 'Viridis',
                "showscale": True,
                "colorbar": {"title": metric.replace('_', ' ').title()},
            },
            text=top_connections[metric].round(4),
            textposition='auto',
        ))

        fig.update_layout(
            title=title,
            xaxis_title=metric.replace('_', ' ').title(),
            yaxis_title="Asset Connection",
            template=self.theme,
            width=width,
            height=height,
            yaxis={"autorange": "reversed"},
        )

        return fig

    def plot_attention_distribution(
        self,
        attention_matrix: pd.DataFrame,
        title: str = "Attention Weight Distribution",
        width: int | None = None,
        height: int | None = None,
    ) -> go.Figure:
        """
        Plot distribution of attention weights.

        Args:
            attention_matrix: Attention weight matrix
            title: Plot title
            width: Figure width (optional)
            height: Figure height (optional)

        Returns:
            Plotly figure object
        """
        width = width or self.default_width
        height = height or max(400, self.default_height // 2)

        # Flatten attention weights (excluding diagonal if self-attention is excluded)
        weights = attention_matrix.values.flatten()
        weights = weights[weights > 0]  # Remove zero weights

        fig = go.Figure()

        # Add histogram
        fig.add_trace(go.Histogram(
            x=weights,
            nbinsx=50,
            name="Attention Weights",
            marker={"color": 'skyblue', "opacity": 0.7},
        ))

        # Add statistics annotations
        mean_weight = weights.mean()
        median_weight = np.median(weights)

        fig.add_vline(
            x=mean_weight,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_weight:.4f}",
        )

        fig.add_vline(
            x=median_weight,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Median: {median_weight:.4f}",
        )

        fig.update_layout(
            title=title,
            xaxis_title="Attention Weight",
            yaxis_title="Frequency",
            template=self.theme,
            width=width,
            height=height,
            showlegend=False,
        )

        return fig

    def create_attention_dashboard(
        self,
        attention_matrix: pd.DataFrame,
        attention_evolution: pd.DataFrame | None = None,
        influential_connections: pd.DataFrame | None = None,
        title: str = "GAT Attention Analysis Dashboard",
    ) -> go.Figure:
        """
        Create comprehensive dashboard for attention analysis.

        Args:
            attention_matrix: Current attention weight matrix
            attention_evolution: Temporal attention evolution data
            influential_connections: Influential connection statistics
            title: Dashboard title

        Returns:
            Plotly dashboard figure
        """
        # Create subplots layout
        if attention_evolution is not None and influential_connections is not None:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    "Attention Heatmap",
                    "Attention Network",
                    "Temporal Evolution",
                    "Top Connections"
                ],
                specs=[
                    [{"type": "xy"}, {"type": "xy"}],
                    [{"type": "xy"}, {"type": "xy"}],
                ],
                horizontal_spacing=0.1,
                vertical_spacing=0.15,
            )
        else:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=["Attention Heatmap", "Attention Distribution"],
                horizontal_spacing=0.15,
            )

        # Add attention heatmap
        heatmap_fig = self.plot_attention_heatmap(attention_matrix, title="", show_values=False)
        for trace in heatmap_fig.data:
            fig.add_trace(trace, row=1, col=1)

        # Add attention distribution
        dist_fig = self.plot_attention_distribution(attention_matrix, title="")
        for trace in dist_fig.data:
            if attention_evolution is not None and influential_connections is not None:
                # Network plot for full dashboard
                network_fig = self.plot_attention_network(attention_matrix, title="")
                for trace in network_fig.data:
                    fig.add_trace(trace, row=1, col=2)
            else:
                # Distribution plot for simple dashboard
                fig.add_trace(trace, row=1, col=2)

        # Add temporal evolution if available
        if attention_evolution is not None and len(fig._grid_ref) >= 3:
            evolution_fig = self.plot_temporal_attention_evolution(attention_evolution, title="")
            for trace in evolution_fig.data:
                fig.add_trace(trace, row=2, col=1)

        # Add top connections if available
        if influential_connections is not None and len(fig._grid_ref) >= 4:
            connections_fig = self.plot_influential_connections(
                influential_connections, top_k=10, title=""
            )
            for trace in connections_fig.data:
                fig.add_trace(trace, row=2, col=2)

        fig.update_layout(
            title=title,
            template=self.theme,
            width=1600,
            height=1200 if attention_evolution is not None else 600,
            showlegend=False,
        )

        return fig

    def plot_temporal_importance_heatmap(
        self,
        temporal_heatmap_data: pd.DataFrame,
        asset: str | None = None,
        title: str = "LSTM Temporal Importance",
        width: int | None = None,
        height: int | None = None,
    ) -> go.Figure:
        """
        Plot temporal importance heatmap for LSTM predictions.

        Args:
            temporal_heatmap_data: DataFrame with temporal importance data
            asset: Specific asset to plot (if None, aggregates across assets)
            title: Plot title
            width: Figure width (optional)
            height: Figure height (optional)

        Returns:
            Plotly figure object
        """
        width = width or self.default_width
        height = height or self.default_height

        # Filter data for specific asset if requested
        if asset:
            plot_data = temporal_heatmap_data[temporal_heatmap_data['asset'] == asset].copy()
            title = f"{title}: {asset}"
        else:
            # Aggregate across all assets
            plot_data = (
                temporal_heatmap_data
                .groupby(['prediction_date', 'historical_date', 'days_back'])
                .agg({'temporal_importance': 'mean'})
                .reset_index()
            )

        if plot_data.empty:
            return go.Figure().update_layout(title=f"{title} (No Data)")

        # Create pivot table for heatmap
        heatmap_matrix = plot_data.pivot_table(
            index='days_back',
            columns='prediction_date',
            values='temporal_importance',
            fill_value=0
        )

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_matrix.values,
            x=heatmap_matrix.columns,
            y=heatmap_matrix.index,
            colorscale='RdYlBu_r',
            showscale=True,
            colorbar={"title": "Temporal Importance"},
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Prediction Date",
            yaxis_title="Days Back",
            template=self.theme,
            width=width,
            height=height,
        )

        return fig
