"""
Graph Attention Network (GAT) module for portfolio optimization.

This module provides GAT-based portfolio allocation using graph neural networks
to model complex asset relationships and optimize portfolio weights.
"""

from .gat_model import GATPortfolio, HeadCfg
from .graph_builder import GraphBuildConfig, build_graph_from_returns
from .model import GATModelConfig, GATPortfolioModel

__all__ = [
    "GATPortfolio",
    "HeadCfg",
    "GATPortfolioModel",
    "GATModelConfig",
    "GraphBuildConfig",
    "build_graph_from_returns",
]
