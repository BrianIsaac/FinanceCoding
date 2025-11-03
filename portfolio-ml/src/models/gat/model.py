"""GAT Portfolio Model wrapper for backtest integration.

This wraps the clean GAT model and provides the interface needed by the
backtest engine (train, predict, etc.).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

from src.models.base.portfolio_model import PortfolioModel
from src.models.base.constraints import PortfolioConstraints

from .allocation import validate_weights
from .features import compute_node_features
from .gat_model import GATPortfolio
from .graph_constructor import build_graph
from .loss import get_loss_function

logger = logging.getLogger(__name__)


@dataclass
class GATConfig:
    """Configuration for GAT portfolio model."""

    # Graph construction
    graph_method: str = "mst"  # "mst", "knn", or "tmfg"
    lookback_days: int = 63  # Correlation lookback window
    knn_k: int = 5  # Number of neighbors for kNN

    # Model architecture
    n_features: int = 11  # Number of node features
    hidden_dim: int = 16  # Hidden dimension (proven in FinGAT)
    n_heads_layer1: int = 8  # Attention heads layer 1
    n_heads_layer2: int = 1  # Attention heads layer 2
    dropout: float = 0.3  # Dropout probability

    # Training
    learning_rate: float = 0.001  # Proven in FinGAT
    weight_decay: float = 1e-4  # L2 regularisation
    max_epochs: int = 5  # Quick retrain epochs
    patience: int = 3  # Early stopping patience
    gradient_clip: float = 1.0  # Gradient clipping max norm

    # Loss function
    loss_type: str = "return"  # "return" or "sharpe"
    vol_penalty: float = 0.5  # For return loss
    risk_free_rate: float = 0.03  # Annual risk-free rate

    # Constraints
    max_weight: float | None = 0.15  # Maximum weight per asset

    # Device
    device: str = "cpu"  # "cpu" or "cuda"


class GATPortfolioModel(PortfolioModel):
    """GAT Portfolio Model with backtest interface."""

    def __init__(
        self,
        constraints: PortfolioConstraints,
        config: GATConfig | None = None,
        **kwargs
    ):
        """Initialize GAT portfolio model.

        Args:
            constraints: Portfolio constraints
            config: Model configuration
            **kwargs: Additional arguments (absorbed for compatibility)
        """
        # Initialize base class
        super().__init__(constraints)

        self.config = config or GATConfig()

        # Create model
        self.model = GATPortfolio(
            n_features=self.config.n_features,
            hidden_dim=self.config.hidden_dim,
            n_heads_layer1=self.config.n_heads_layer1,
            n_heads_layer2=self.config.n_heads_layer2,
            dropout=self.config.dropout,
            max_weight=self.config.max_weight
        ).to(self.config.device)

        # Create loss function
        self.loss_fn = get_loss_function(
            loss_type=self.config.loss_type,
            vol_penalty=self.config.vol_penalty,
            risk_free_rate=self.config.risk_free_rate
        )

        # Create optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Training state
        self.training_history = []
        self.best_loss = float('inf')
        self.patience_counter = 0

        # Fitted state (for predict_weights interface)
        self._fitted_returns = None
        self._fitted_universe = None
        self.is_fitted = False

        # Scaler for returns normalization (prevents NaN losses)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

        logger.info(
            f"GAT-{self.config.graph_method.upper()} model created: "
            f"{self.model.count_parameters()['total']:,} parameters"
        )

    def fit(
        self,
        returns: pd.DataFrame,
        universe: list[str],
        fit_period: tuple[pd.Timestamp, pd.Timestamp],
        **kwargs
    ) -> None:
        """Train the GAT model.

        Args:
            returns: Historical returns DataFrame [time, assets]
            universe: List of asset tickers
            fit_period: (start_date, end_date) tuple defining training period
            **kwargs: Additional arguments (absorbed for compatibility)
        """
        # Filter data for training period
        start_date, end_date = fit_period
        time_mask = (returns.index >= start_date) & (returns.index <= end_date)
        period_returns = returns[time_mask]

        # Filter for universe assets
        available_assets = [asset for asset in universe if asset in period_returns.columns]
        if len(available_assets) < len(universe) * 0.8:
            raise ValueError(
                f"Insufficient asset coverage: {len(available_assets)}/{len(universe)}"
            )

        fitted_returns = period_returns[available_assets]

        # CRITICAL: Remove any assets with all-NaN values before normalization
        # MinMaxScaler will fail with "All-NaN slice encountered" if columns are all-NaN
        all_nan_mask = fitted_returns.isna().all()
        n_all_nan = all_nan_mask.sum()
        if n_all_nan > 0:
            logger.warning(f"Dropping {n_all_nan} assets with all-NaN returns from {len(available_assets)} total")
            fitted_returns = fitted_returns.loc[:, ~all_nan_mask]
            available_assets = [asset for asset in available_assets if not all_nan_mask[asset]]

        if len(available_assets) < 10:
            raise ValueError(
                f"Insufficient assets after NaN filtering: {len(available_assets)} remaining. "
                f"Need at least 10 assets for meaningful portfolio construction."
            )

        # CRITICAL: Impute NaN values BEFORE normalization
        # MinMaxScaler will produce NaN if input contains NaN
        from src.data.na_handling.imputation import simple_temporal_fill

        fitted_returns_imputed = simple_temporal_fill(
            fitted_returns,
            drop_all_na_first=False,  # Already filtered above
            allow_bfill=True  # Training context - can use all historical data
        )

        logger.info(
            f"Imputation applied: "
            f"pre_impute_nan_count={fitted_returns.isna().sum().sum()}, "
            f"post_impute_nan_count={fitted_returns_imputed.isna().sum().sum()}"
        )

        # Store fitted state for predict_weights
        self._fitted_returns = fitted_returns_imputed
        self._fitted_universe = available_assets

        # Set model to training mode
        self.model.train()

        # CRITICAL: Normalize returns for stable training (prevents NaN losses)
        # Fit scaler on training data (now guaranteed to have no NaN)
        normalized_returns_values = self.scaler.fit_transform(fitted_returns_imputed.values)
        normalized_returns = pd.DataFrame(
            normalized_returns_values,
            index=fitted_returns_imputed.index,
            columns=fitted_returns_imputed.columns
        )

        logger.info(
            f"Normalized returns with MinMaxScaler: "
            f"original_range=[{fitted_returns_imputed.min().min():.6f}, {fitted_returns_imputed.max().max():.6f}], "
            f"normalized_range=[{normalized_returns.min().min():.6f}, {normalized_returns.max().max():.6f}]"
        )

        # Build graph (use normalized returns for stability)
        edge_index = build_graph(
            returns=normalized_returns,
            universe=available_assets,
            method=self.config.graph_method,
            lookback_days=self.config.lookback_days,
            knn_k=self.config.knn_k
        ).to(self.config.device)

        # Compute node features (use normalized returns)
        node_features = compute_node_features(
            returns=normalized_returns,
            universe=available_assets,
            lookback=self.config.lookback_days
        ).to(self.config.device)

        # Prepare returns tensor for loss (use normalized returns)
        returns_tensor = torch.tensor(
            normalized_returns.values,
            dtype=torch.float32
        ).to(self.config.device)

        # Training loop
        epoch_losses = []
        for epoch in range(self.config.max_epochs):
            # Forward pass
            weights = self.model(node_features, edge_index)

            # Compute loss
            loss = self.loss_fn(weights, returns_tensor)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.gradient_clip
            )

            # Check gradient health
            total_grad_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** 0.5

            # Update weights
            self.optimizer.step()

            epoch_losses.append(loss.item())

            # Log every epoch
            logger.info(
                f"Epoch {epoch+1}/{self.config.max_epochs}: "
                f"loss={loss.item():.6f}, grad_norm={total_grad_norm:.6e}"
            )

            # Early stopping check
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Store training history
        self.training_history.append({
            "epoch_losses": epoch_losses,
            "final_loss": epoch_losses[-1],
            "grad_norm": total_grad_norm
        })

        # Mark as fitted
        self.is_fitted = True

        logger.info(
            f"GAT-{self.config.graph_method.upper()} training complete: "
            f"{len(epoch_losses)} epochs, final_loss={epoch_losses[-1]:.6f}, "
            f"grad_norm={total_grad_norm:.6e}"
        )

    def predict_weights(
        self,
        date: pd.Timestamp,
        universe: list[str]
    ) -> pd.Series:
        """Generate portfolio weights for rebalancing date.

        Args:
            date: Rebalancing date for which to generate weights
            universe: List of asset tickers

        Returns:
            Portfolio weights as pandas Series

        Raises:
            ValueError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating predictions")

        if self._fitted_returns is None:
            raise ValueError("Model state is invalid - refit required")

        # Handle dynamic universe - use overlap between fitted and current universe
        available_assets = [asset for asset in universe if asset in self._fitted_universe]

        if len(available_assets) == 0:
            logger.warning(f"No fitted assets in current universe. Using equal weights.")
            equal_weight = 1.0 / len(universe)
            return pd.Series(equal_weight, index=universe)

        logger.info(
            f"Asset alignment at {date.date()}: "
            f"requested_universe={len(universe)}, "
            f"fitted_universe={len(self._fitted_universe)}, "
            f"available_after_alignment={len(available_assets)}, "
            f"alignment_ratio={len(available_assets)/len(universe):.1%}"
        )

        # Set model to evaluation mode
        self.model.eval()

        with torch.no_grad():
            # Use fitted returns for graph and features
            # Get recent data (lookback window ending at date)
            end_date = date - pd.Timedelta(days=1)
            start_date = end_date - pd.Timedelta(days=self.config.lookback_days)

            # Filter fitted returns to lookback window
            recent_returns = self._fitted_returns.loc[
                (self._fitted_returns.index >= start_date) &
                (self._fitted_returns.index <= end_date)
            ]

            # Safety check: Drop any columns with all-NaN values (shouldn't happen after fit filtering)
            all_nan_mask = recent_returns.isna().all()
            if all_nan_mask.any():
                logger.warning(f"Dropping {all_nan_mask.sum()} assets with all-NaN in prediction window")
                recent_returns = recent_returns.loc[:, ~all_nan_mask]
                available_assets = [asset for asset in available_assets if not all_nan_mask.get(asset, False)]

            # CRITICAL: Impute NaN values BEFORE normalization (prediction context)
            from src.data.na_handling.imputation import simple_temporal_fill

            recent_returns_imputed = simple_temporal_fill(
                recent_returns,
                drop_all_na_first=False,  # Already filtered above
                allow_bfill=False  # Prediction context - no lookahead bias
            )

            # Normalize recent returns using fitted scaler (now guaranteed no NaN)
            normalized_recent_values = self.scaler.transform(recent_returns_imputed.values)
            normalized_recent_returns = pd.DataFrame(
                normalized_recent_values,
                index=recent_returns_imputed.index,
                columns=recent_returns_imputed.columns
            )

            # Build graph (use normalized returns)
            edge_index = build_graph(
                returns=normalized_recent_returns,
                universe=available_assets,
                method=self.config.graph_method,
                lookback_days=self.config.lookback_days,
                knn_k=self.config.knn_k
            ).to(self.config.device)

            # Compute node features (use normalized returns)
            node_features = compute_node_features(
                returns=normalized_recent_returns,
                universe=available_assets,
                lookback=self.config.lookback_days
            ).to(self.config.device)

            # Forward pass
            weights = self.model(node_features, edge_index)

            # Validate weights
            weights_valid = validate_weights(weights)
            if not weights_valid:
                logger.warning("Invalid weights detected, falling back to equal weights")
                weights = torch.ones_like(weights) / len(weights)

            # Convert to pandas Series with available_assets index
            weights_np = weights.cpu().numpy()
            weights_series = pd.Series(0.0, index=universe)
            weights_series[available_assets] = weights_np

        return weights_series

    def save(self, filepath: Path):
        """Save model weights and config."""
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config,
            "training_history": self.training_history
        }, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: Path):
        """Load model weights and config."""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.training_history = checkpoint.get("training_history", [])
        logger.info(f"Model loaded from {filepath}")

    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata for analysis and reproducibility.

        Returns:
            Dictionary containing model type, hyperparameters, constraints,
            and other relevant metadata for performance analysis.
        """
        from dataclasses import asdict

        return {
            "model_type": f"GAT-{self.config.graph_method.upper()}",
            "model_class": self.__class__.__name__,
            "is_fitted": self.is_fitted,
            "graph_method": self.config.graph_method,
            "config": asdict(self.config),
            "constraints": asdict(self.constraints) if hasattr(self.constraints, '__dict__') else {},
            "model_parameters": self.model.count_parameters() if hasattr(self.model, 'count_parameters') else None,
        }
