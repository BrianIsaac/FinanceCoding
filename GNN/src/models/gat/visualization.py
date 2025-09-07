"""
Attention Weight Visualization and Interpretation for GAT Portfolio Models.

This module provides utilities for extracting, visualizing, and interpreting 
attention weights from GAT models to understand asset relationships and 
portfolio attribution patterns.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import torch

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    warnings.warn("Matplotlib/Seaborn not available. Visualization features disabled.")

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    warnings.warn("NetworkX not available. Graph visualization features disabled.")

from .gat_model import GATPortfolio

__all__ = [
    "AttentionExtractor",
    "AttentionVisualizer",
    "PortfolioAttribution",
    "InteractiveVizDashboard",
]


class AttentionExtractor:
    """Extract and process attention weights from GAT models."""

    def __init__(self, model: GATPortfolio):
        """
        Initialize attention extractor.
        
        Args:
            model: Trained GAT portfolio model
        """
        self.model = model
        self.attention_weights: dict[str, torch.Tensor] = {}
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []

    def register_attention_hooks(self) -> None:
        """Register hooks to capture attention weights during forward pass."""
        def attention_hook(name: str):
            def hook_fn(module, input, output):
                # GAT attention weights are typically returned as part of the output
                # or stored in module attributes during forward pass
                if hasattr(module, 'alpha') and module.alpha is not None:
                    self.attention_weights[name] = module.alpha.detach().cpu()
                elif hasattr(module, '_alpha_src') and module._alpha_src is not None:
                    # For PyTorch Geometric GAT implementations
                    alpha_src = module._alpha_src.detach().cpu()
                    alpha_dst = module._alpha_dst.detach().cpu() if hasattr(module, '_alpha_dst') else alpha_src
                    self.attention_weights[f"{name}_src"] = alpha_src
                    self.attention_weights[f"{name}_dst"] = alpha_dst
            return hook_fn

        # Register hooks for each GAT layer
        for i, layer in enumerate(self.model.gnn):
            if hasattr(layer, 'conv'):
                hook_handle = layer.conv.register_forward_hook(attention_hook(f"layer_{i}"))
                self.hooks.append(hook_handle)

    def remove_hooks(self) -> None:
        """Remove all registered attention hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def extract_attention(
        self,
        graph_data: Any,
        node_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Extract attention weights for a graph.
        
        Args:
            graph_data: Graph data object with x, edge_index, edge_attr
            node_mask: Optional mask for valid nodes
            
        Returns:
            Dictionary of attention weights by layer
        """
        self.model.eval()
        self.attention_weights.clear()

        # Register hooks
        self.register_attention_hooks()

        try:
            # Forward pass to trigger hooks
            x = graph_data.x
            edge_index = graph_data.edge_index
            edge_attr = graph_data.edge_attr if hasattr(graph_data, 'edge_attr') else None

            if node_mask is None:
                node_mask = torch.ones(x.size(0), dtype=torch.bool)

            with torch.no_grad():
                _ = self.model(x, edge_index, node_mask, edge_attr)

            return self.attention_weights.copy()

        finally:
            self.remove_hooks()

    def compute_attention_statistics(
        self,
        attention_weights: dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        tickers: list[str]
    ) -> pd.DataFrame:
        """
        Compute attention weight statistics for analysis.
        
        Args:
            attention_weights: Attention weights from extract_attention
            edge_index: Graph edge indices
            tickers: Asset ticker symbols
            
        Returns:
            DataFrame with attention statistics per edge
        """
        stats_list = []

        for layer_name, weights in attention_weights.items():
            if weights.dim() == 1:  # Edge-level attention
                n_edges = min(len(weights), edge_index.size(1))

                for i in range(n_edges):
                    src_idx, dst_idx = edge_index[0, i].item(), edge_index[1, i].item()
                    src_ticker = tickers[src_idx] if src_idx < len(tickers) else f"node_{src_idx}"
                    dst_ticker = tickers[dst_idx] if dst_idx < len(tickers) else f"node_{dst_idx}"

                    stats_list.append({
                        'layer': layer_name,
                        'source': src_ticker,
                        'target': dst_ticker,
                        'attention_weight': weights[i].item(),
                        'edge_index': i
                    })

        return pd.DataFrame(stats_list)


class AttentionVisualizer:
    """Create visualizations of attention weights and patterns."""

    def __init__(self, figsize: tuple[int, int] = (12, 8)):
        """
        Initialize attention visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        if not HAS_PLOTTING:
            raise ImportError("Matplotlib/Seaborn required for visualization")

        self.figsize = figsize
        sns.set_style("whitegrid")

    def plot_attention_heatmap(
        self,
        attention_stats: pd.DataFrame,
        layer: str | None = None,
        top_k: int = 50,
        save_path: Path | str | None = None
    ) -> plt.Figure:
        """
        Create attention weight heatmap.
        
        Args:
            attention_stats: Attention statistics DataFrame
            layer: Specific layer to plot (None for all layers)
            top_k: Number of top attention weights to show
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Filter by layer if specified
        if layer is not None:
            data = attention_stats[attention_stats['layer'] == layer].copy()
        else:
            data = attention_stats.copy()

        # Get top-k attention weights
        data = data.nlargest(top_k, 'attention_weight')

        # Create pivot table for heatmap
        heatmap_data = data.pivot_table(
            index='source', columns='target',
            values='attention_weight', fill_value=0
        )

        # Create heatmap
        fig, ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(
            heatmap_data, annot=True, cmap='viridis',
            ax=ax, fmt='.3f', cbar_kws={'label': 'Attention Weight'}
        )

        ax.set_title(f'Attention Weights Heatmap - {layer or "All Layers"}')
        ax.set_xlabel('Target Assets')
        ax.set_ylabel('Source Assets')

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_attention_distribution(
        self,
        attention_stats: pd.DataFrame,
        save_path: Path | str | None = None
    ) -> plt.Figure:
        """
        Plot attention weight distributions by layer.
        
        Args:
            attention_stats: Attention statistics DataFrame
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 1, figsize=self.figsize)

        # Distribution by layer
        sns.boxplot(
            data=attention_stats, x='layer', y='attention_weight', ax=axes[0]
        )
        axes[0].set_title('Attention Weight Distribution by Layer')
        axes[0].tick_params(axis='x', rotation=45)

        # Overall histogram
        axes[1].hist(
            attention_stats['attention_weight'], bins=50, alpha=0.7, color='skyblue'
        )
        axes[1].set_title('Overall Attention Weight Distribution')
        axes[1].set_xlabel('Attention Weight')
        axes[1].set_ylabel('Frequency')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_network_graph(
        self,
        attention_stats: pd.DataFrame,
        layer: str | None = None,
        threshold: float = 0.1,
        save_path: Path | str | None = None
    ) -> plt.Figure:
        """
        Create network graph visualization of attention weights.
        
        Args:
            attention_stats: Attention statistics DataFrame
            layer: Specific layer to visualize
            threshold: Minimum attention weight to show edge
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if not HAS_NETWORKX:
            raise ImportError("NetworkX required for network graph visualization")

        # Filter data
        if layer is not None:
            data = attention_stats[attention_stats['layer'] == layer].copy()
        else:
            data = attention_stats.copy()

        data = data[data['attention_weight'] >= threshold]

        # Create network graph
        G = nx.DiGraph()

        # Add edges with attention weights
        for _, row in data.iterrows():
            G.add_edge(
                row['source'], row['target'],
                weight=row['attention_weight']
            )

        # Create layout
        pos = nx.spring_layout(G, k=1, iterations=50)

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, node_color='lightblue', node_size=500, ax=ax
        )

        # Draw edges with varying thickness based on attention weight
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1

        nx.draw_networkx_edges(
            G, pos, width=[w / max_weight * 3 for w in weights],
            alpha=0.6, edge_color='gray', ax=ax
        )

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

        ax.set_title(f'Attention Network Graph - {layer or "All Layers"}')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


class PortfolioAttribution:
    """Analyze portfolio performance attribution based on attention patterns."""

    def __init__(self, model: GATPortfolio):
        """
        Initialize portfolio attribution analyzer.
        
        Args:
            model: Trained GAT portfolio model
        """
        self.model = model
        self.extractor = AttentionExtractor(model)

    def compute_attention_attribution(
        self,
        graph_data: Any,
        portfolio_weights: torch.Tensor,
        returns: torch.Tensor,
        tickers: list[str]
    ) -> pd.DataFrame:
        """
        Compute portfolio attribution based on attention patterns.
        
        Args:
            graph_data: Graph data for the period
            portfolio_weights: Portfolio weights
            returns: Asset returns for the period
            tickers: Asset ticker symbols
            
        Returns:
            DataFrame with attribution analysis
        """
        # Extract attention weights
        attention_weights = self.extractor.extract_attention(graph_data)
        attention_stats = self.extractor.compute_attention_statistics(
            attention_weights, graph_data.edge_index, tickers
        )

        # Compute portfolio returns and contributions
        portfolio_return = (portfolio_weights * returns).sum()
        individual_contributions = portfolio_weights * returns

        # Aggregate attention by asset
        attention_by_asset = attention_stats.groupby('source')['attention_weight'].agg([
            'mean', 'std', 'count', 'sum'
        ]).reset_index()
        attention_by_asset.columns = ['ticker', 'avg_attention', 'std_attention', 'edge_count', 'total_attention']

        # Create attribution DataFrame
        attribution_data = []
        for i, ticker in enumerate(tickers):
            if i < len(portfolio_weights):
                weight = portfolio_weights[i].item()
                return_contrib = individual_contributions[i].item()

                # Get attention statistics for this asset
                asset_attention = attention_by_asset[attention_by_asset['ticker'] == ticker]

                if len(asset_attention) > 0:
                    avg_attention = asset_attention['avg_attention'].iloc[0]
                    edge_count = asset_attention['edge_count'].iloc[0]
                else:
                    avg_attention = 0.0
                    edge_count = 0

                attribution_data.append({
                    'ticker': ticker,
                    'portfolio_weight': weight,
                    'return_contribution': return_contrib,
                    'avg_attention': avg_attention,
                    'edge_count': edge_count,
                    'attention_weighted_contrib': return_contrib * avg_attention
                })

        attribution_df = pd.DataFrame(attribution_data)

        # Add rankings
        attribution_df['weight_rank'] = attribution_df['portfolio_weight'].rank(ascending=False)
        attribution_df['attention_rank'] = attribution_df['avg_attention'].rank(ascending=False)
        attribution_df['contrib_rank'] = attribution_df['return_contribution'].rank(ascending=False)

        return attribution_df

    def analyze_regime_attention(
        self,
        graph_data_list: list[Any],
        returns_list: list[torch.Tensor],
        tickers: list[str],
        regime_labels: list[str] | None = None
    ) -> dict[str, pd.DataFrame]:
        """
        Analyze attention patterns across different market regimes.
        
        Args:
            graph_data_list: List of graph data for different periods
            returns_list: List of return tensors
            tickers: Asset ticker symbols
            regime_labels: Labels for each regime/period
            
        Returns:
            Dictionary mapping regime labels to attention statistics
        """
        if regime_labels is None:
            regime_labels = [f"period_{i}" for i in range(len(graph_data_list))]

        regime_attention = {}

        for graph_data, returns, regime in zip(graph_data_list, returns_list, regime_labels):
            attention_weights = self.extractor.extract_attention(graph_data)
            attention_stats = self.extractor.compute_attention_statistics(
                attention_weights, graph_data.edge_index, tickers
            )

            # Add regime information
            attention_stats['regime'] = regime
            attention_stats['market_return'] = returns.mean().item()
            attention_stats['market_volatility'] = returns.std().item()

            regime_attention[regime] = attention_stats

        return regime_attention


class InteractiveVizDashboard:
    """Create interactive dashboard for attention weight exploration."""

    def __init__(self):
        """Initialize interactive visualization dashboard."""
        self.has_plotly = self._check_plotly()

    def _check_plotly(self) -> bool:
        """Check if Plotly is available for interactive plots."""
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            return True
        except ImportError:
            warnings.warn("Plotly not available. Interactive features disabled.")
            return False

    def create_interactive_heatmap(
        self,
        attention_stats: pd.DataFrame,
        layer: str | None = None
    ) -> Any:
        """
        Create interactive attention heatmap using Plotly.
        
        Args:
            attention_stats: Attention statistics DataFrame
            layer: Specific layer to plot
            
        Returns:
            Plotly figure object
        """
        if not self.has_plotly:
            raise ImportError("Plotly required for interactive visualizations")

        import plotly.graph_objects as go

        # Filter by layer if specified
        if layer is not None:
            data = attention_stats[attention_stats['layer'] == layer].copy()
        else:
            data = attention_stats.copy()

        # Create pivot table
        heatmap_data = data.pivot_table(
            index='source', columns='target',
            values='attention_weight', fill_value=0
        )

        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='Viridis',
            colorbar=dict(title='Attention Weight')
        ))

        fig.update_layout(
            title=f'Interactive Attention Heatmap - {layer or "All Layers"}',
            xaxis_title='Target Assets',
            yaxis_title='Source Assets'
        )

        return fig

    def create_attention_network(
        self,
        attention_stats: pd.DataFrame,
        threshold: float = 0.1
    ) -> Any:
        """
        Create interactive network graph of attention weights.
        
        Args:
            attention_stats: Attention statistics DataFrame
            threshold: Minimum attention weight threshold
            
        Returns:
            Plotly figure object
        """
        if not self.has_plotly:
            raise ImportError("Plotly required for interactive visualizations")

        import networkx as nx
        import plotly.graph_objects as go

        if not HAS_NETWORKX:
            raise ImportError("NetworkX required for network graphs")

        # Filter data
        data = attention_stats[attention_stats['attention_weight'] >= threshold]

        # Create NetworkX graph
        G = nx.DiGraph()
        for _, row in data.iterrows():
            G.add_edge(row['source'], row['target'], weight=row['attention_weight'])

        # Get positions
        pos = nx.spring_layout(G)

        # Extract node and edge traces
        node_trace = go.Scatter(
            x=[], y=[], mode='markers+text', text=[], textposition="middle center",
            marker=dict(size=10, color='lightblue', line=dict(width=1, color='black'))
        )

        edge_traces = []

        # Add nodes
        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['text'] += tuple([node])

        # Add edges
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = G[edge[0]][edge[1]]['weight']

            edge_trace = go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode='lines',
                line=dict(width=weight*10, color='gray'),
                hoverinfo='none'
            )
            edge_traces.append(edge_trace)

        # Create figure
        fig = go.Figure(data=[node_trace] + edge_traces)
        fig.update_layout(
            title='Interactive Attention Network',
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )

        return fig

    def save_dashboard_html(
        self,
        attention_stats: pd.DataFrame,
        output_path: Path | str,
        title: str = "GAT Attention Analysis Dashboard"
    ) -> None:
        """
        Save complete dashboard as HTML file.
        
        Args:
            attention_stats: Attention statistics DataFrame
            output_path: Path to save HTML file
            title: Dashboard title
        """
        if not self.has_plotly:
            raise ImportError("Plotly required for interactive dashboard")

        from plotly.subplots import make_subplots

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Attention Heatmap', 'Attention Distribution',
                          'Network Graph', 'Layer Comparison'),
            specs=[[{'type': 'heatmap'}, {'type': 'histogram'}],
                   [{'type': 'scatter'}, {'type': 'bar'}]]
        )

        # Add various plots to subplots
        # This is a simplified version - full implementation would add all visualizations

        fig.update_layout(
            height=800,
            title_text=title,
            showlegend=False
        )

        # Save as HTML
        fig.write_html(str(output_path))
