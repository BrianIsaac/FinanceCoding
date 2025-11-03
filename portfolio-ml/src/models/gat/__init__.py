"""Clean GAT Portfolio Optimization Model.

Research-proven implementation based on:
- Korangi et al. (2024): Logarithmic Sharpe, reduce-weight allocation, TMFG graphs
- FinGAT (2021): 8-head attention, proven hyperparameters

Simple, clean, working architecture with NO deprecated code.
"""

from .allocation import ReduceWeightAllocator, validate_weights
from .features import compute_node_features
from .gat_model import GATPortfolio
from .graph_constructor import build_graph
from .loss import NegativeSharpeLoss, PortfolioReturnLoss, get_loss_function
from .model import GATConfig, GATPortfolioModel

__all__ = [
    # Main interface
    "GATPortfolioModel",
    "GATConfig",
    # Core components
    "GATPortfolio",
    "ReduceWeightAllocator",
    "build_graph",
    "compute_node_features",
    # Loss functions
    "PortfolioReturnLoss",
    "NegativeSharpeLoss",
    "get_loss_function",
    # Utilities
    "validate_weights",
]

__version__ = "2.0.0"  # Clean rebuild
