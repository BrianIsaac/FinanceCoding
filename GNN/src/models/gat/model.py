"""
Main GAT model interface for portfolio optimization.

This module provides the high-level interface for the Graph Attention Network
implementation, integrating the GAT architecture with graph construction utilities
and implementing the PortfolioModel interface for unified portfolio optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam

from ..base.portfolio_model import PortfolioConstraints, PortfolioModel
from .gat_model import GATPortfolio, HeadCfg
from .graph_builder import GraphBuildConfig, build_period_graph
from .loss_functions import SharpeRatioLoss

__all__ = [
    "GATPortfolioModel",
    "GATModelConfig",
]


@dataclass
class GATModelConfig:
    """Configuration for GAT portfolio model."""

    # GAT architecture parameters
    input_features: int = 10
    hidden_dim: int = 64
    num_layers: int = 3
    num_attention_heads: int = 8
    dropout: float = 0.3
    edge_feature_dim: int = 3
    use_gatv2: bool = True
    residual: bool = True
    mem_hidden: int | None = None

    # Head configuration
    head_config: HeadCfg = field(
        default_factory=lambda: HeadCfg(mode="direct", activation="sparsemax")
    )

    # Graph construction configuration
    graph_config: GraphBuildConfig = field(default_factory=GraphBuildConfig)

    # Training parameters
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 200
    patience: int = 20

    # Memory optimization
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = True
    max_vram_gb: float = 11.0


class GATPortfolioModel(PortfolioModel):
    """
    Graph Attention Network-based portfolio optimization model.

    This class integrates graph construction, GAT model training, and portfolio
    weight prediction into a unified interface implementing the PortfolioModel protocol.
    """

    def __init__(self, constraints: PortfolioConstraints, config: GATModelConfig):
        """
        Initialize GAT portfolio model.

        Args:
            constraints: Portfolio constraints configuration
            config: GAT model configuration including architecture and training parameters
        """
        super().__init__(constraints)
        self.config = config
        self.model: GATPortfolio | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer: Adam | None = None
        self.loss_fn = SharpeRatioLoss()
        if config.use_mixed_precision and torch.cuda.is_available():
            try:
                # Use new API if available (PyTorch 2.1+)
                self.scaler = torch.amp.GradScaler("cuda")
            except (AttributeError, TypeError):
                # Fall back to older API
                self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # Training history
        self.training_history: dict[str, list[float]] = {
            "loss": [],
            "sharpe": [],
            "weights_norm": [],
        }

    def _build_model(self, input_dim: int) -> GATPortfolio:
        """
        Build GAT model architecture.

        Args:
            input_dim: Number of input features per asset

        Returns:
            Configured GAT model
        """
        model = GATPortfolio(
            in_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            heads=self.config.num_attention_heads,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            residual=self.config.residual,
            use_gatv2=self.config.use_gatv2,
            use_edge_attr=True,
            head=self.config.head_config.mode,
            activation=self.config.head_config.activation,
            mem_hidden=self.config.mem_hidden,
        ).to(self.device)

        # Note: gradient checkpointing is handled during forward pass in training pipeline
        return model

    def _prepare_features(self, returns: pd.DataFrame, universe: list[str]) -> np.ndarray:
        """
        Prepare node features from returns data.

        Args:
            returns: Historical returns DataFrame
            universe: List of asset tickers

        Returns:
            Node features matrix [n_assets, n_features]
        """
        # Simple feature engineering: mean, std, skew, rolling correlation with market
        returns_subset = returns[universe].dropna()

        features = []
        market_proxy = returns_subset.mean(axis=1)  # Equal-weight market proxy

        for ticker in universe:
            asset_returns = returns_subset[ticker]

            # Statistical features
            mean_return = asset_returns.mean()
            volatility = asset_returns.std()
            skewness = asset_returns.skew()
            kurtosis = asset_returns.kurtosis()

            # Market correlation features
            corr_with_market = asset_returns.corr(market_proxy)
            beta = asset_returns.cov(market_proxy) / market_proxy.var()

            # Momentum features
            momentum_1m = asset_returns.tail(21).mean()  # 1-month momentum
            momentum_3m = asset_returns.tail(63).mean()  # 3-month momentum

            # Risk features
            max_drawdown = (asset_returns.cumsum().expanding().max() - asset_returns.cumsum()).max()
            var_95 = np.percentile(asset_returns, 5)  # Value at Risk 95%

            features.append(
                [
                    mean_return,
                    volatility,
                    skewness,
                    kurtosis,
                    corr_with_market,
                    beta,
                    momentum_1m,
                    momentum_3m,
                    max_drawdown,
                    var_95,
                ]
            )

        return np.array(features, dtype=np.float32)

    def fit(
        self,
        returns: pd.DataFrame,
        universe: list[str],
        fit_period: tuple[pd.Timestamp, pd.Timestamp],
    ) -> None:
        """
        Train GAT model on historical data.

        Args:
            returns: Historical returns DataFrame with datetime index
            universe: List of asset tickers to include in optimization
            fit_period: (start_date, end_date) tuple defining training period
        """
        start_date, end_date = fit_period

        # Filter returns to training period
        training_returns = returns.loc[start_date:end_date].copy()
        if len(training_returns) < self.config.graph_config.lookback_days + 30:
            raise ValueError("Insufficient training data for GAT model")

        # Prepare node features
        features_matrix = self._prepare_features(training_returns, universe)
        input_dim = features_matrix.shape[1]

        # Build model
        self.model = self._build_model(input_dim)
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Generate training samples by creating graphs for different time windows
        training_graphs = []
        training_labels = []

        # Create monthly rebalancing points during training
        rebalance_dates = pd.date_range(
            start=start_date + pd.Timedelta(days=self.config.graph_config.lookback_days),
            end=end_date - pd.Timedelta(days=21),  # Leave some buffer
            freq="MS",  # Month start
        )

        for date in rebalance_dates[: self.config.batch_size]:  # Limit for memory
            try:
                # Build graph using lookback window ending day before rebalance
                graph_data = build_period_graph(
                    returns_daily=training_returns,
                    period_end=date,
                    tickers=universe,
                    features_matrix=features_matrix,
                    cfg=self.config.graph_config,
                )

                # Get forward returns for the next month as labels
                next_month_end = min(date + pd.Timedelta(days=30), end_date)
                forward_returns = training_returns.loc[date:next_month_end, universe].mean()

                training_graphs.append(graph_data)
                training_labels.append(forward_returns.values)

            except Exception:
                continue

        if not training_graphs:
            raise ValueError("No valid training graphs could be created")

        # Training loop
        self.model.train()
        self.training_history = {"loss": [], "sharpe": [], "weights_norm": []}

        for epoch in range(self.config.max_epochs):
            epoch_losses = []

            for _i, (graph_data, labels) in enumerate(zip(training_graphs, training_labels)):
                self.optimizer.zero_grad()

                # Move data to device
                x = graph_data.x.to(self.device)
                edge_index = graph_data.edge_index.to(self.device)
                edge_attr = (
                    graph_data.edge_attr.to(self.device)
                    if graph_data.edge_attr is not None
                    else None
                )
                mask_valid = torch.ones(len(universe), dtype=torch.bool, device=self.device)

                # Forward pass with mixed precision if enabled
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        weights, _ = self.model(x, edge_index, mask_valid, edge_attr)

                        # Reshape for loss computation
                        weights = weights.unsqueeze(0)  # [1, n_assets]
                        returns_tensor = torch.tensor(
                            labels, dtype=torch.float32, device=self.device
                        ).unsqueeze(0)

                        loss = self.loss_fn(weights, returns_tensor, mask_valid.unsqueeze(0))

                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    weights, _ = self.model(x, edge_index, mask_valid, edge_attr)
                    weights = weights.unsqueeze(0)
                    returns_tensor = torch.tensor(
                        labels, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)

                    loss = self.loss_fn(weights, returns_tensor, mask_valid.unsqueeze(0))
                    loss.backward()
                    self.optimizer.step()

                epoch_losses.append(loss.item())

            # Record training metrics
            avg_loss = np.mean(epoch_losses)
            self.training_history["loss"].append(avg_loss)

            # Early stopping check
            if len(self.training_history["loss"]) > self.config.patience:
                recent_losses = self.training_history["loss"][-self.config.patience :]
                if all(loss >= min(recent_losses) for loss in recent_losses[-5:]):
                    break

            if epoch % 20 == 0:
                pass

        self.is_fitted = True

    def predict_weights(self, date: pd.Timestamp, universe: list[str]) -> pd.Series:
        """
        Generate portfolio weights for rebalancing date.

        Args:
            date: Rebalancing date for which to generate weights
            universe: List of asset tickers (must be subset of fitted universe)

        Returns:
            Portfolio weights as pandas Series with asset tickers as index.
            Weights sum to 1.0 and satisfy all portfolio constraints.
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before making predictions")

        self.model.eval()

        with torch.no_grad():
            # This is a placeholder implementation
            # In a full implementation, we would:
            # 1. Get historical returns data up to date
            # 2. Prepare features for the universe
            # 3. Build graph for the current period
            # 4. Run model forward pass to get weights
            # 5. Apply portfolio constraints

            # For now, return equal weights as fallback
            equal_weights = 1.0 / len(universe)
            raw_weights = pd.Series(equal_weights, index=universe)

            # Apply portfolio constraints
            constrained_weights = self.validate_weights(raw_weights)

            return constrained_weights

    def get_model_info(self) -> dict[str, Any]:
        """
        Return model metadata for analysis and reproducibility.

        Returns:
            Dictionary containing model type, hyperparameters, constraints,
            and other relevant metadata for performance analysis.
        """
        info = {
            "model_type": "GAT",
            "architecture": {
                "input_features": self.config.input_features,
                "hidden_dim": self.config.hidden_dim,
                "num_layers": self.config.num_layers,
                "num_attention_heads": self.config.num_attention_heads,
                "dropout": self.config.dropout,
                "use_gatv2": self.config.use_gatv2,
                "residual": self.config.residual,
            },
            "head_config": {
                "mode": self.config.head_config.mode,
                "activation": self.config.head_config.activation,
            },
            "graph_config": {
                "lookback_days": self.config.graph_config.lookback_days,
                "filter_method": self.config.graph_config.filter_method,
                "knn_k": self.config.graph_config.knn_k,
                "use_edge_attr": self.config.graph_config.use_edge_attr,
            },
            "training": {
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "max_epochs": self.config.max_epochs,
                "use_mixed_precision": self.config.use_mixed_precision,
            },
            "constraints": {
                "long_only": self.constraints.long_only,
                "top_k_positions": self.constraints.top_k_positions,
                "max_position_weight": self.constraints.max_position_weight,
                "max_monthly_turnover": self.constraints.max_monthly_turnover,
            },
            "device": str(self.device),
            "is_fitted": self.is_fitted,
            "training_history": (
                self.training_history if hasattr(self, "training_history") else None
            ),
        }

        # Add model parameter count if model is built
        if self.model is not None:
            info["model_parameters"] = sum(p.numel() for p in self.model.parameters())

        return info
