"""
Main GAT model interface for portfolio optimization.

This module provides the high-level interface for the Graph Attention Network
implementation, integrating the GAT architecture with graph construction utilities
and implementing the PortfolioModel interface for unified portfolio optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import logging

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam

from ..base.portfolio_model import PortfolioConstraints, PortfolioModel
from ..base.confidence_weighted_training import (
    ConfidenceWeightedTrainer,
    TrainingStrategy,
    create_confidence_weighted_trainer,
)
from .gat_model import GATPortfolio, HeadCfg
from .graph_builder import GraphBuildConfig, build_period_graph
from .loss_functions import SharpeRatioLoss, CorrelationAwareSharpeRatioLoss
from .diversification_loss import (
    CorrectedDiversificationLoss,
    AdaptiveDiversificationLoss,
    EntropyRegularizedSoftmax
)
from .diversification_gat import (
    DiversificationGAT,
    DiversificationLoss,
    CorrelationAwareGraphBuilder
)

logger = logging.getLogger(__name__)

# Import flexible academic validation
try:
    from ...evaluation.validation.flexible_academic_validator import (
        FlexibleAcademicValidator,
        AcademicValidationResult,
    )
    FLEXIBLE_VALIDATION_AVAILABLE = True
except ImportError:
    logger.info("Flexible validation not available for GAT, using standard validation")
    FLEXIBLE_VALIDATION_AVAILABLE = False

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

    # Diversification settings
    use_diversification_gat: bool = False  # Enable new diversification-aware GAT
    correlation_penalty: float = 0.5  # Weight for correlation penalty
    cluster_selection: bool = True  # Use cluster-based selection
    correlation_threshold: float = 0.7  # Threshold for high correlation


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
        self.logger = logger  # Use the module logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer: Adam | None = None

        # Initialize model based on configuration
        if config.use_diversification_gat:
            self.model: DiversificationGAT | None = None
            # Use adaptive loss for DiversificationGAT
            self.loss_fn = AdaptiveDiversificationLoss(
                min_effective_assets=15,
                initial_div_weight=2.0,
                final_div_weight=0.5,
                initial_sharpe_weight=0.5,
                final_sharpe_weight=1.5,
                warmup_epochs=5,
                total_epochs=config.max_epochs,
                debug_mode=False
            )
            self.correlation_graph_builder = CorrelationAwareGraphBuilder(
                correlation_threshold=config.correlation_threshold
            )
            logger.info("Using DiversificationGAT with CORRECTED adaptive loss")
        else:
            self.model: GATPortfolio | None = None

            # Adjust loss parameters based on graph type for better diversification
            graph_method = config.graph_config.filter_method.lower()

            if graph_method in ['knn', 'tmfg']:
                # STRONGER diversification for kNN and TMFG which tend to concentrate
                logger.info(f"Using ENHANCED diversification for GAT-{graph_method.upper()}")
                self.loss_fn = CorrectedDiversificationLoss(
                    min_effective_assets=20,  # Higher minimum (was 15)
                    sharpe_weight=0.5,        # Less emphasis on returns
                    diversification_weight=2.0,  # More emphasis on diversification
                    entropy_weight=0.2,       # More entropy regularization
                    concentration_penalty=3.0,  # Stronger penalty for concentration
                    debug_mode=False
                )
            else:
                # Standard diversification for MST which already diversifies well
                logger.info(f"Using standard diversification for GAT-{graph_method.upper()}")
                self.loss_fn = CorrectedDiversificationLoss(
                    min_effective_assets=15,
                    sharpe_weight=1.0,
                    diversification_weight=1.0,
                    entropy_weight=0.1,
                    concentration_penalty=2.0,
                    debug_mode=False
                )

            self.correlation_graph_builder = None
        # Gradient monitoring disabled for production
        if config.use_mixed_precision and torch.cuda.is_available():
            try:
                # Use new API if available (PyTorch 2.1+)
                self.scaler = torch.amp.GradScaler("cuda")
            except (AttributeError, TypeError):
                # Fall back to older API
                self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

        # Enhanced training history with gradient monitoring
        self.training_history: dict[str, list[float]] = {
            "loss": [],
            "sharpe": [],
            "weights_norm": [],
            "gradient_norm": [],
            "zero_grad_fraction": [],
            "constraint_violations": [],
        }

        # Confidence-weighted training support
        self.confidence_trainer = create_confidence_weighted_trainer()
        self.flexible_validator = (
            FlexibleAcademicValidator()
            if FLEXIBLE_VALIDATION_AVAILABLE
            else None
        )
        self.last_training_strategy: TrainingStrategy | None = None
        self.last_validation_result: AcademicValidationResult | None = None

    def supports_rolling_retraining(self) -> bool:
        """GAT supports rolling retraining with warm starts."""
        return True

    def rolling_fit(
        self,
        returns: pd.DataFrame,
        universe: list[str],
        rebalance_date: pd.Timestamp,
        lookback_months: int = 36,
        min_observations: int = 100,  # Reduced for flexible academic framework
    ) -> None:
        """
        Perform rolling fit for GAT model with warm start.

        GAT benefits from warm starts as graph structures evolve gradually
        and attention patterns can be fine-tuned on recent data.

        Args:
            returns: Full historical returns DataFrame
            universe: Dynamic universe for this rebalancing period
            rebalance_date: Date for which we're rebalancing
            lookback_months: Number of months to look back for training
            min_observations: Minimum number of observations required
        """
        # Calculate rolling window dates
        end_date = rebalance_date - pd.Timedelta(days=1)
        start_date = end_date - pd.Timedelta(days=lookback_months * 30)

        # Use existing _get_historical_returns method
        historical_data = self._get_historical_returns(rebalance_date, universe)

        if len(historical_data) < min_observations:
            # Try loading with extended window
            historical_data = self._load_historical_data_extended(
                returns, start_date, end_date, universe
            )

        if len(historical_data) < min_observations:
            raise ValueError(
                f"Insufficient data for rolling fit: {len(historical_data)} < {min_observations}"
            )

        # Quick retrain with warm start
        self._quick_retrain(historical_data, universe, max_epochs=10)

        # Update model state
        self.universe = universe.copy()
        self.fitted_period = (start_date, end_date)
        self.is_fitted = True

    def _load_historical_data_extended(
        self,
        returns: pd.DataFrame | Path | str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        universe: list[str],
    ) -> pd.DataFrame:
        """
        Load historical data with proper date range.

        Args:
            returns: Full historical returns or path to data
            start_date: Start of window
            end_date: End of window
            universe: Assets to include

        Returns:
            Filtered returns DataFrame
        """
        # If returns is a path, load from disk
        if isinstance(returns, (str, Path)):
            returns_path = Path(returns) if isinstance(returns, str) else returns
            if not returns_path.exists():
                returns_path = Path("data/final_new_pipeline/returns_daily_final.parquet")

            if returns_path.exists():
                returns = pd.read_parquet(returns_path)
            else:
                raise FileNotFoundError(f"Returns data not found at {returns_path}")

        # Filter by date range
        mask = (returns.index >= start_date) & (returns.index <= end_date)
        period_returns = returns[mask]

        # Filter for available universe assets
        available_assets = [asset for asset in universe if asset in period_returns.columns]

        if len(available_assets) == 0:
            raise ValueError("No assets from universe found in returns data")

        filtered_returns = period_returns[available_assets]

        # Clean data
        cleaned_returns = filtered_returns.ffill().fillna(0.0)

        return cleaned_returns

    def _quick_retrain(
        self,
        returns: pd.DataFrame,
        universe: list[str],
        max_epochs: int = 10,
    ) -> None:
        """
        Fast retraining for rolling updates using warm start.

        Args:
            returns: Training data for current window
            universe: Asset universe
            max_epochs: Maximum epochs for quick retraining
        """
        self.logger.info(
            f"GAT {self.config.graph_config.filter_method} _quick_retrain called - "
            f"universe={len(universe)}, returns_shape={returns.shape}, max_epochs={max_epochs}"
        )

        # Prepare features for current universe
        features_matrix = self._prepare_features(returns, universe)
        input_dim = features_matrix.shape[1]

        # If model exists, keep it (warm start), otherwise create new
        if self.model is None:
            self.model = self._build_model(input_dim)
            self.optimizer = Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

        # Store original max_epochs
        original_epochs = self.config.max_epochs
        self.config.max_epochs = max_epochs

        try:
            # Quick training loop with limited epochs
            self.model.train()

            # Get recent rebalancing dates for training
            lookback_days = self.config.graph_config.lookback_days
            valid_start = returns.index[0] + pd.Timedelta(days=lookback_days)
            valid_end = returns.index[-1] - pd.Timedelta(days=21)

            rebalance_dates = pd.date_range(
                start=valid_start,
                end=valid_end,
                freq="MS",  # Monthly
            )

            # Limit training samples for speed
            training_samples = min(len(rebalance_dates), 6)  # Max 6 months for quick retrain
            # Fix DatetimeIndex slicing to avoid boolean evaluation issues
            if len(rebalance_dates) > training_samples:
                selected_dates = rebalance_dates.tolist()[-training_samples:]
            else:
                selected_dates = rebalance_dates.tolist()

            for epoch in range(max_epochs):
                epoch_loss = 0.0

                for date in selected_dates:
                    try:
                        # Build graph for this date
                        graph_data = build_period_graph(
                            returns_daily=returns,
                            period_end=date,
                            tickers=universe,
                            features_matrix=features_matrix,
                            cfg=self.config.graph_config,
                        )

                        # Get forward returns as labels
                        next_month_end = min(date + pd.Timedelta(days=30), returns.index[-1])
                        forward_returns = returns.loc[date:next_month_end, universe].mean()

                        # Forward pass
                        self.optimizer.zero_grad()

                        if self.config.use_mixed_precision and self.scaler:
                            with torch.amp.autocast(device_type="cuda"):
                                # Properly unpack graph data for forward pass
                                x = graph_data.x.to(self.device)
                                edge_index = graph_data.edge_index.to(self.device)
                                edge_attr = graph_data.edge_attr.to(self.device) if graph_data.edge_attr is not None else None
                                mask_valid = torch.ones(len(universe), dtype=torch.bool, device=self.device)

                                weights, _, _ = self.model(x, edge_index, mask_valid, edge_attr)

                                # Prepare returns for loss computation
                                forward_returns_tensor = torch.tensor(
                                    forward_returns.values,
                                    dtype=torch.float32,
                                    device=self.device
                                ).unsqueeze(0)  # [1, n_assets]

                                # Always calculate correlation matrix for loss functions
                                corr = training_window.corr()
                                correlation_matrix = torch.tensor(
                                    corr.values, dtype=torch.float32, device=self.device
                                )

                                # Update epoch for adaptive losses
                                if hasattr(self.loss_fn, 'update_epoch'):
                                    self.loss_fn.update_epoch(epoch)

                                loss = self.loss_fn(
                                    weights.unsqueeze(0),
                                    forward_returns_tensor,
                                    constraints_mask=None,
                                    correlation_matrix=correlation_matrix
                                )

                            self.scaler.scale(loss).backward()
                            # Add gradient clipping to prevent exploding gradients
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            # Properly unpack graph data for forward pass
                            x = graph_data.x.to(self.device)
                            edge_index = graph_data.edge_index.to(self.device)
                            edge_attr = graph_data.edge_attr.to(self.device) if graph_data.edge_attr is not None else None
                            mask_valid = torch.ones(len(universe), dtype=torch.bool, device=self.device)

                            weights, _, _ = self.model(x, edge_index, mask_valid, edge_attr)

                            # Prepare returns for loss computation
                            forward_returns_tensor = torch.tensor(
                                forward_returns.values,
                                dtype=torch.float32,
                                device=self.device
                            ).unsqueeze(0)  # [1, n_assets]

                            # Always calculate correlation matrix for loss functions
                            corr = training_window.corr()
                            correlation_matrix = torch.tensor(
                                corr.values, dtype=torch.float32, device=self.device
                            )

                            # Update epoch for adaptive losses
                            if hasattr(self.loss_fn, 'update_epoch'):
                                self.loss_fn.update_epoch(epoch)

                            loss = self.loss_fn(
                                weights.unsqueeze(0),
                                forward_returns_tensor,
                                constraints_mask=None,
                                correlation_matrix=correlation_matrix
                            )
                            loss.backward()
                            # Add gradient clipping to prevent exploding gradients
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.optimizer.step()

                        epoch_loss += loss.item()

                    except Exception as e:
                        self.logger.warning(
                            f"GAT {self.config.graph_config.filter_method} training error at {date}: {str(e)}\n"
                            f"Error type: {type(e).__name__}"
                        )
                        logger.debug(f"Skipped training sample at {date}: {e}")
                        continue

                # Early stopping if loss is good enough
                avg_loss = epoch_loss / len(selected_dates) if selected_dates else float('inf')
                if avg_loss < 0.01:  # Good enough for quick retrain
                    break

        except Exception as e:
            logger.warning(f"Quick retrain failed: {e}, keeping existing weights")

        finally:
            # Restore original max_epochs
            self.config.max_epochs = original_epochs

    def _build_model(self, input_dim: int) -> GATPortfolio | DiversificationGAT:
        """
        Build GAT model architecture.

        Args:
            input_dim: Number of input features per asset

        Returns:
            Configured GAT model
        """
        if self.config.use_diversification_gat:
            model = DiversificationGAT(
                input_dim=input_dim,
                hidden_dim=self.config.hidden_dim,
                output_dim=1,
                num_heads=self.config.num_attention_heads,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
                correlation_penalty=self.config.correlation_penalty,
                cluster_selection=self.config.cluster_selection
            ).to(self.device)
        else:
            # Get graph type from config for proper projection head selection
            graph_type = self.config.graph_config.filter_method

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
                graph_type=graph_type,  # Pass graph type for proper projection head selection
            ).to(self.device)

        # Note: gradient checkpointing is handled during forward pass in training pipeline
        return model

    def _prepare_features(self, returns: pd.DataFrame, universe: list[str]) -> np.ndarray:
        """
        Prepare robust node features from returns data.

        Args:
            returns: Historical returns DataFrame
            universe: List of asset tickers

        Returns:
            Node features matrix [n_assets, n_features]
        """
        features = []

        # Filter to available universe assets
        available_universe = [ticker for ticker in universe if ticker in returns.columns]
        if not available_universe:
            logger.warning("No universe assets found in returns data")
            # Return minimum viable features to prevent dimension 0 errors
            # At least 1 asset with default features
            return np.ones((max(1, len(universe)), 10), dtype=np.float32) * 0.001

        # Log for debugging
        logger.info(f"GAT feature preparation: universe={len(universe)}, available={len(available_universe)}, returns_shape={returns.shape}")

        # Get returns for available assets only
        returns_subset = returns[available_universe]

        # Calculate market proxy from available data
        market_proxy = returns_subset.mean(axis=1)

        # Process each asset in the full universe
        for ticker in universe:
            if ticker not in available_universe:
                # Use default features for missing assets
                features.append([0.0, 0.02, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -0.02])
                continue

            # Get asset returns
            asset_returns = returns_subset[ticker]

            # Remove NaN values for statistics computation
            asset_returns_clean = asset_returns.dropna()

            # Require minimum data for reliable features
            if len(asset_returns_clean) < 20:
                # Use conservative defaults for insufficient data
                features.append([0.0, 0.02, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -0.02])
                continue

            # Robust statistical features with error handling
            try:
                mean_return = float(np.nanmean(asset_returns_clean))
                if np.isnan(mean_return) or np.isinf(mean_return):
                    mean_return = 0.0
            except:
                mean_return = 0.0

            try:
                volatility = float(np.nanstd(asset_returns_clean))
                if np.isnan(volatility) or np.isinf(volatility) or volatility < 1e-8:
                    volatility = 0.02  # Default 2% daily volatility
            except:
                volatility = 0.02

            try:
                from scipy import stats
                skewness = float(stats.skew(asset_returns_clean, nan_policy='omit'))
                if np.isnan(skewness) or np.isinf(skewness):
                    skewness = 0.0
            except:
                skewness = 0.0

            try:
                kurtosis = float(stats.kurtosis(asset_returns_clean, nan_policy='omit'))
                if np.isnan(kurtosis) or np.isinf(kurtosis):
                    kurtosis = 0.0
            except:
                kurtosis = 0.0

            # Market correlation with proper alignment
            try:
                # Get overlapping non-NaN indices
                valid_asset_idx = ~asset_returns.isna()
                valid_market_idx = ~market_proxy.isna()
                valid_idx = valid_asset_idx & valid_market_idx

                if valid_idx.sum() > 20:  # Need sufficient overlapping data
                    asset_aligned = asset_returns[valid_idx]
                    market_aligned = market_proxy[valid_idx]

                    # Calculate correlation with error handling
                    if len(asset_aligned) > 1 and asset_aligned.std() > 1e-8 and market_aligned.std() > 1e-8:
                        corr_with_market = float(np.corrcoef(asset_aligned, market_aligned)[0, 1])
                        if np.isnan(corr_with_market) or np.isinf(corr_with_market):
                            corr_with_market = 0.0
                        corr_with_market = np.clip(corr_with_market, -1.0, 1.0)
                    else:
                        corr_with_market = 0.0

                    # Calculate beta
                    market_var = market_aligned.var()
                    if market_var > 1e-8:
                        covariance = np.cov(asset_aligned, market_aligned)[0, 1]
                        beta = float(covariance / market_var)
                        if np.isnan(beta) or np.isinf(beta):
                            beta = 1.0
                        beta = np.clip(beta, -3.0, 3.0)  # Reasonable beta range
                    else:
                        beta = 1.0
                else:
                    corr_with_market = 0.0
                    beta = 1.0
            except Exception as e:
                logger.debug(f"Error calculating correlation for {ticker}: {e}")
                corr_with_market = 0.0
                beta = 1.0

            # Momentum features with robust calculation
            try:
                # 1-month momentum
                recent_21 = asset_returns_clean.tail(21) if len(asset_returns_clean) >= 21 else asset_returns_clean
                momentum_1m = float(np.nanmean(recent_21)) if len(recent_21) > 0 else 0.0
                if np.isnan(momentum_1m) or np.isinf(momentum_1m):
                    momentum_1m = 0.0

                # 3-month momentum
                recent_63 = asset_returns_clean.tail(63) if len(asset_returns_clean) >= 63 else asset_returns_clean
                momentum_3m = float(np.nanmean(recent_63)) if len(recent_63) > 0 else 0.0
                if np.isnan(momentum_3m) or np.isinf(momentum_3m):
                    momentum_3m = 0.0
            except:
                momentum_1m = 0.0
                momentum_3m = 0.0

            # Risk features
            try:
                # Maximum drawdown calculation
                cumulative = (1 + asset_returns_clean).cumprod()
                running_max = cumulative.expanding().max()
                drawdowns = (cumulative - running_max) / running_max
                max_drawdown = float(drawdowns.min())
                if np.isnan(max_drawdown) or np.isinf(max_drawdown):
                    max_drawdown = 0.0
                max_drawdown = np.clip(max_drawdown, -1.0, 0.0)

                # Value at Risk (5th percentile)
                var_95 = float(np.percentile(asset_returns_clean, 5))
                if np.isnan(var_95) or np.isinf(var_95):
                    var_95 = -0.02  # Default -2% VaR
                var_95 = np.clip(var_95, -0.10, 0.0)  # Reasonable VaR range
            except:
                max_drawdown = 0.0
                var_95 = -0.02

            # Append features with final validation
            feature_vector = [
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

            # Final NaN/Inf check
            feature_vector = [
                0.0 if (np.isnan(f) or np.isinf(f)) else float(f)
                for f in feature_vector
            ]

            features.append(feature_vector)

        feature_matrix = np.array(features, dtype=np.float32)

        # Validate feature matrix dimensions
        expected_shape = (len(universe), 10)
        if feature_matrix.shape != expected_shape:
            logger.error(f"Feature matrix shape mismatch: got {feature_matrix.shape}, expected {expected_shape}")
            # Create correctly shaped matrix with default features
            corrected_matrix = np.zeros(expected_shape, dtype=np.float32)
            # Copy available features
            copy_rows = min(feature_matrix.shape[0], expected_shape[0])
            copy_cols = min(feature_matrix.shape[1], expected_shape[1])
            corrected_matrix[:copy_rows, :copy_cols] = feature_matrix[:copy_rows, :copy_cols]
            feature_matrix = corrected_matrix

        logger.info(f"GAT features final shape: {feature_matrix.shape}")

        # Normalize features for better GAT performance
        for i in range(feature_matrix.shape[1]):
            col = feature_matrix[:, i]
            col_std = np.std(col)
            if col_std > 1e-8:
                feature_matrix[:, i] = (col - np.mean(col)) / col_std

        return feature_matrix

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

        self.logger.info(
            f"GAT {self.config.graph_config.filter_method} fit() called - "
            f"period={start_date} to {end_date}, universe={len(universe)}, "
            f"returns_shape={returns.shape}"
        )

        # Filter returns to training period
        training_returns = returns.loc[start_date:end_date].copy()
        if len(training_returns) < self.config.graph_config.lookback_days + 30:
            logger.warning(f"Insufficient training data for GAT model: {len(training_returns)} < {self.config.graph_config.lookback_days + 30}")
            # Use what's available instead of failing
            self.config.graph_config.lookback_days = max(60, len(training_returns) - 30)

        # Prepare node features with validation
        features_matrix = self._prepare_features(training_returns, universe)
        input_dim = features_matrix.shape[1]

        logger.info(f"GAT model setup: universe_size={len(universe)}, feature_dims={features_matrix.shape}, input_dim={input_dim}")

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

        # Fix DatetimeIndex slicing to avoid boolean evaluation issue
        limited_dates = rebalance_dates.tolist()[:self.config.batch_size] if len(rebalance_dates) > self.config.batch_size else rebalance_dates.tolist()
        for date in limited_dates:  # Limit for memory
            try:
                logger.debug(f"Building graph for date {date} with {len(universe)} assets")

                # Build graph using lookback window ending day before rebalance
                graph_data = build_period_graph(
                    returns_daily=training_returns,
                    period_end=date,
                    tickers=universe,
                    features_matrix=features_matrix,
                    cfg=self.config.graph_config,
                )

                logger.debug(f"Graph built successfully: nodes={graph_data.x.shape}, edges={graph_data.edge_index.shape}")

                # Get forward returns for the next month as labels
                next_month_end = min(date + pd.Timedelta(days=30), end_date)
                # Use the full time series of forward returns, not just the mean
                forward_returns_series = training_returns.loc[date:next_month_end, universe]

                # Only include if we have enough data points
                if len(forward_returns_series) >= 5:  # At least 5 days of forward returns
                    training_graphs.append(graph_data)
                    training_labels.append(forward_returns_series.values)  # Full time series

            except Exception as e:
                self.logger.error(
                    f"GAT {self.config.graph_config.filter_method} failed to build graph for date {date}: {str(e)}\n"
                    f"Error type: {type(e).__name__}\n"
                    f"Universe size: {len(universe)}, Features shape: {features_matrix.shape}"
                )
                self.logger.exception(f"Full traceback for graph building error:")
                logger.warning(f"Failed to build graph for date {date}: {e}")
                logger.debug(f"Graph building error details: universe_size={len(universe)}, features_shape={features_matrix.shape}", exc_info=True)
                continue

        if not training_graphs:
            raise ValueError("No valid training graphs could be created")

        # Training loop with enhanced monitoring
        self.model.train()
        self.training_history = {
            "loss": [], "sharpe": [], "weights_norm": [],
            "gradient_norm": [], "zero_grad_fraction": [], "constraint_violations": []
        }
        # Gradient monitoring disabled for production
        logger.info(f"Starting GAT training for {len(training_graphs)} graphs over {self.config.max_epochs} epochs")

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

                # Calculate correlation matrix if using DiversificationGAT
                correlation_matrix = None
                if self.config.use_diversification_gat:
                    # Get returns for correlation calculation
                    lookback_days = self.config.graph_config.lookback_days
                    corr_end = graph_data.period_end if hasattr(graph_data, 'period_end') else end_date
                    corr_start = corr_end - pd.Timedelta(days=lookback_days)
                    corr_returns = training_returns.loc[corr_start:corr_end, universe]
                    correlation_matrix = torch.tensor(
                        corr_returns.corr().fillna(0).values,
                        dtype=torch.float32,
                        device=self.device
                    )

                # Forward pass with mixed precision if enabled
                if self.scaler is not None:
                    with torch.amp.autocast("cuda"):
                        if self.config.use_diversification_gat:
                            weights, gat_scores = self.model(x, edge_index, edge_attr, correlation_matrix)
                        else:
                            weights, _, _ = self.model(x, edge_index, mask_valid, edge_attr)

                        # Fix tensor alignment for loss computation
                        weights = weights.unsqueeze(0) if weights.dim() == 1 else weights  # [1, n_assets]

                        # Ensure labels has correct shape and compute portfolio returns
                        if isinstance(labels, np.ndarray):
                            labels_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device)
                        else:
                            labels_tensor = labels

                        # Handle different label shapes
                        if labels_tensor.dim() == 2:  # [time_steps, n_assets]
                            # Compute portfolio returns: [time_steps, n_assets] @ [n_assets] = [time_steps]
                            portfolio_returns = torch.matmul(labels_tensor, weights.squeeze(0))  # [time_steps]
                            # Reshape to match expected format for Sharpe loss
                            portfolio_returns = portfolio_returns.unsqueeze(0)  # [1, time_steps]
                        elif labels_tensor.dim() == 1:  # [n_assets] - single period
                            # Single period return
                            portfolio_returns = torch.matmul(labels_tensor, weights.squeeze(0))  # scalar
                            portfolio_returns = portfolio_returns.unsqueeze(0).unsqueeze(0)  # [1, 1]
                        else:
                            raise ValueError(f"Unexpected labels shape: {labels_tensor.shape}")

                        # Fix tensor dimension mismatch: SharpeRatioLoss expects raw asset returns,
                        # not computed portfolio returns. Pass original labels_tensor as returns.
                        # GAT: weights [1, n_assets], labels_tensor [time_steps, n_assets] or [n_assets]
                        if labels_tensor.dim() == 2:  # [time_steps, n_assets]
                            # Reshape for batch processing: [time_steps, n_assets] -> [1, time_steps, n_assets]
                            asset_returns = labels_tensor.unsqueeze(0)  # [1, time_steps, n_assets]
                        else:  # [n_assets] - single period
                            # Reshape for batch processing: [n_assets] -> [1, n_assets]
                            asset_returns = labels_tensor.unsqueeze(0)  # [1, n_assets]

                        # Pass correlation matrix to loss if using DiversificationLoss
                        if self.config.use_diversification_gat:
                            loss = self.loss_fn(weights, asset_returns, correlation_matrix)
                        else:
                            loss = self.loss_fn(weights, asset_returns, constraints_mask=None)

                    # Proper mixed precision backward pass
                    if loss.requires_grad and torch.isfinite(loss):
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                    else:
                        # Handle case where loss doesn't require grad or is not finite
                        logger.warning(f"Skipping backward pass: loss.requires_grad={loss.requires_grad}, loss.isfinite={torch.isfinite(loss)}")
                        self.optimizer.zero_grad()
                else:
                    if self.config.use_diversification_gat:
                        weights, gat_scores = self.model(x, edge_index, edge_attr, correlation_matrix)
                    else:
                        weights, _, _ = self.model(x, edge_index, mask_valid, edge_attr)
                    weights = weights.unsqueeze(0) if weights.dim() == 1 else weights

                    # Fix tensor alignment for loss computation (standard precision)
                    if isinstance(labels, np.ndarray):
                        labels_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device)
                    else:
                        labels_tensor = labels

                    # Handle different label shapes
                    if labels_tensor.dim() == 2:  # [time_steps, n_assets]
                        # Compute portfolio returns: [time_steps, n_assets] @ [n_assets] = [time_steps]
                        portfolio_returns = torch.matmul(labels_tensor, weights.squeeze(0))  # [time_steps]
                        # Reshape to match expected format for Sharpe loss
                        portfolio_returns = portfolio_returns.unsqueeze(0)  # [1, time_steps]
                    elif labels_tensor.dim() == 1:  # [n_assets] - single period
                        # Single period return
                        portfolio_returns = torch.matmul(labels_tensor, weights.squeeze(0))  # scalar
                        portfolio_returns = portfolio_returns.unsqueeze(0).unsqueeze(0)  # [1, 1]
                    else:
                        raise ValueError(f"Unexpected labels shape: {labels_tensor.shape}")

                    # Fix tensor dimension mismatch: pass raw asset returns to loss function
                    if labels_tensor.dim() == 2:  # [time_steps, n_assets]
                        asset_returns = labels_tensor.unsqueeze(0)  # [1, time_steps, n_assets]
                    else:  # [n_assets] - single period
                        asset_returns = labels_tensor.unsqueeze(0)  # [1, n_assets]

                    # Pass correlation matrix to loss if using DiversificationLoss
                    if self.config.use_diversification_gat:
                        loss = self.loss_fn(weights, asset_returns, correlation_matrix)
                    else:
                        loss = self.loss_fn(weights, asset_returns, constraints_mask=None)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # Enhanced gradient flow monitoring and debugging
                loss_value = loss.item()

                # Monitor gradient flow after backward pass
                # Gradient monitoring disabled for production
                # grad_stats = self.gradient_monitor.check_gradient_flow(self.model, loss)
                grad_stats = None

                # Debug logging for first few iterations
                if epoch <= 2 and _i <= 2:
                    logger.debug(f"GAT Training Debug [Epoch {epoch}, Batch {_i}]:")
                    logger.debug(f"  Weights: shape={weights.shape}, sum={weights.sum():.6f}, mean={weights.mean():.6f}")
                    logger.debug(f"  Loss: {loss_value:.6f}")
                    if grad_stats:
                        logger.debug(f"  Gradient norm: {grad_stats['total_grad_norm']:.6f}")
                        logger.debug(f"  Zero grad fraction: {grad_stats['zero_grad_fraction']:.3f}")
                    logger.debug(f"  Asset returns shape: {asset_returns.shape}")

                # Check for problematic training dynamics
                training_issues = []
                if torch.isnan(loss) or torch.isinf(loss):
                    training_issues.append("NaN/Inf loss")
                if loss_value == 0.0:
                    training_issues.append("Zero loss")
                if grad_stats:
                    if grad_stats['total_grad_norm'] < 1e-8:
                        training_issues.append(f"Vanishing gradients (norm={grad_stats['total_grad_norm']:.2e})")
                    if grad_stats['zero_grad_fraction'] > 0.95:
                        training_issues.append(f"Dead neurons ({grad_stats['zero_grad_fraction']:.2f} zero grads)")

                if training_issues:
                    logger.warning(f"GAT Training Issues [Epoch {epoch}, Batch {_i}]: {', '.join(training_issues)}")

                    # Apply enhanced fallback when issues detected
                    if torch.isnan(loss) or torch.isinf(loss) or loss_value == 0.0:
                        # Comprehensive fallback loss computation
                        weight_sum_penalty = torch.mean(torch.abs(weights.sum(dim=-1) - 1.0))
                        negative_weight_penalty = torch.mean(torch.relu(-weights))
                        concentration_penalty = torch.mean(torch.sum(weights**2, dim=-1))

                        # Ensure some portfolio return variance
                        if asset_returns.numel() > 0:
                            portfolio_return = torch.sum(
                                weights.squeeze(0) * asset_returns.mean(dim=1 if asset_returns.dim() > 2 else 0),
                                dim=-1
                            )
                            return_variance_penalty = -torch.abs(portfolio_return).mean()
                        else:
                            return_variance_penalty = torch.tensor(0.0, device=weights.device)

                        # Construct meaningful fallback loss
                        fallback_loss = (
                            0.1 +  # Small base loss
                            10.0 * weight_sum_penalty +  # Strong normalisation constraint
                            10.0 * negative_weight_penalty +  # Strong non-negativity constraint
                            2.0 * concentration_penalty +  # Moderate diversification constraint
                            return_variance_penalty  # Encourage meaningful portfolio returns
                        )

                        # Ensure gradient flow by making fallback differentiable
                        loss = fallback_loss
                        loss_value = loss.item()

                        logger.debug(f"Applied fallback loss: {loss_value:.6f}")
                        logger.debug(f"  Components: sum={weight_sum_penalty:.6f}, neg={negative_weight_penalty:.6f}, conc={concentration_penalty:.6f}, ret={return_variance_penalty:.6f}")

                # Record detailed metrics
                epoch_losses.append(loss_value)

                # Store additional metrics for analysis
                if hasattr(self, '_batch_metrics'):
                    self._batch_metrics = []
                else:
                    self._batch_metrics = []

                self._batch_metrics.append({
                    'epoch': epoch,
                    'batch': _i,
                    'loss': loss_value,
                    'gradient_norm': grad_stats['total_grad_norm'] if grad_stats else 0.0,
                    'zero_grad_fraction': grad_stats['zero_grad_fraction'] if grad_stats else 0.0,
                    'weights_sum': weights.sum().item(),
                    'weights_mean': weights.mean().item(),
                    'max_weight': weights.max().item(),
                    'min_weight': weights.min().item(),
                })

            # Record comprehensive training metrics
            if epoch_losses:  # Only if we have valid losses
                avg_loss = np.mean(epoch_losses)
                self.training_history["loss"].append(avg_loss)

                # Calculate additional metrics from batch data
                if hasattr(self, '_batch_metrics') and self._batch_metrics:
                    epoch_batches = [m for m in self._batch_metrics if m['epoch'] == epoch]
                    if epoch_batches:
                        avg_grad_norm = np.mean([m['gradient_norm'] for m in epoch_batches])
                        avg_zero_fraction = np.mean([m['zero_grad_fraction'] for m in epoch_batches])
                        avg_weights_sum = np.mean([m['weights_sum'] for m in epoch_batches])

                        self.training_history["gradient_norm"].append(avg_grad_norm)
                        self.training_history["zero_grad_fraction"].append(avg_zero_fraction)
                        self.training_history["constraint_violations"].append(abs(avg_weights_sum - 1.0))

                        # Store weight statistics
                        avg_weights_norm = np.mean([np.sqrt(m['weights_mean']**2) for m in epoch_batches])
                        self.training_history["weights_norm"].append(avg_weights_norm)

                # Enhanced early stopping with gradient monitoring
                should_stop = False
                if len(self.training_history["loss"]) > self.config.patience:
                    recent_losses = self.training_history["loss"][-self.config.patience:]

                    # Traditional loss-based early stopping
                    loss_plateau = all(loss >= min(recent_losses) for loss in recent_losses[-5:])

                    # Gradient-based early stopping (if gradients are vanishing)
                    if len(self.training_history["gradient_norm"]) >= self.config.patience:
                        recent_grad_norms = self.training_history["gradient_norm"][-self.config.patience:]
                        vanishing_gradients = all(norm < 1e-6 for norm in recent_grad_norms[-3:])
                        should_stop = loss_plateau or vanishing_gradients

                        if vanishing_gradients:
                            logger.warning(f"Early stopping due to vanishing gradients at epoch {epoch+1}")
                    else:
                        should_stop = loss_plateau

                    if should_stop:
                        break

                # Enhanced progress logging
                if epoch % 10 == 0 or epoch < 5:
                    grad_info = ""
                    if len(self.training_history["gradient_norm"]) > 0:
                        current_grad_norm = self.training_history["gradient_norm"][-1]
                        current_zero_frac = self.training_history["zero_grad_fraction"][-1]
                        grad_info = f", Grad: {current_grad_norm:.2e}, Zero%: {current_zero_frac:.2f}"

                    logger.info(f"GAT Epoch {epoch+1}/{self.config.max_epochs}: Loss={avg_loss:.6f}{grad_info}")
            else:
                logger.warning(f"No valid losses recorded for epoch {epoch+1}")

        # Final training summary with gradient analysis
        final_epochs = len(self.training_history['loss'])
        logger.info(f"GAT training completed after {final_epochs} epochs")

        # Generate gradient flow summary - disabled for production
        if hasattr(self, 'gradient_monitor') and False:  # Disabled
            grad_summary = self.gradient_monitor.get_gradient_summary()
            if grad_summary:
                logger.info(f"Gradient Flow Summary:")
                logger.info(f"  Final loss: {self.training_history['loss'][-1]:.6f}")
                logger.info(f"  Average gradient norm: {grad_summary['gradient_norm_stats']['mean']:.2e}")
                logger.info(f"  Zero gradient fraction: {grad_summary['zero_gradient_stats']['mean_fraction']:.3f}")

                # Check for training quality issues
                if grad_summary['gradient_norm_stats']['mean'] < 1e-6:
                    logger.warning("Training may have suffered from vanishing gradients")
                if grad_summary['zero_gradient_stats']['mean_fraction'] > 0.9:
                    logger.warning("High fraction of dead neurons detected")
                if grad_summary['loss_stats']['std'] < 1e-6:
                    logger.warning("Loss showed little variation - potential convergence issues")
        self.is_fitted = True
        self.universe = universe  # Store fitted universe for filtering during inference
        self.fitted_period = fit_period  # Store fitted period for reference

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

        # Store original universe for final weight expansion
        original_universe = universe.copy()

        # Handle dynamic universe membership for GAT
        if hasattr(self, 'universe') and self.universe:
            available_assets = [asset for asset in universe if asset in self.universe]
            unavailable_assets = [asset for asset in universe if asset not in self.universe]

            # GAT handles variable graphs well, but we need some overlap
            if not available_assets:
                logger.warning(f"GAT model has no overlap with current universe. Using equal weights for {len(universe)} assets.")
                equal_weight = 1.0 / len(universe)
                return pd.Series(equal_weight, index=universe)

            # Log dynamic membership changes
            if unavailable_assets:
                logger.info(f"GAT adapting to dynamic universe: {len(available_assets)}/{len(universe)} assets available")
                logger.debug(f"New assets entering universe: {unavailable_assets[:5]}...")

            # Use available assets for prediction, will expand to original_universe later
            prediction_universe = available_assets
        else:
            prediction_universe = universe

        # For inference, we need historical returns data up to the prediction date
        # Since we don't have access to the training data here, we'll need to
        # implement a data retrieval mechanism or pass it as parameter
        # For now, we'll implement a fallback approach

        self.model.eval()

        with torch.no_grad():
            try:
                # Load actual historical returns data for prediction universe
                returns_data = self._get_historical_returns(date, prediction_universe)

                # Prepare node features
                features_matrix = self._prepare_features(returns_data, prediction_universe)

                # Build graph for current period using prediction universe
                graph_data = build_period_graph(
                    returns_daily=returns_data,
                    period_end=date,
                    tickers=prediction_universe,
                    features_matrix=features_matrix,
                    cfg=self.config.graph_config,
                )

                # Move data to device
                x = graph_data.x.to(self.device)
                edge_index = graph_data.edge_index.to(self.device)
                edge_attr = graph_data.edge_attr.to(self.device) if graph_data.edge_attr is not None else None

                # Create mask for valid assets (all valid for inference)
                mask_valid = torch.ones(len(prediction_universe), dtype=torch.bool, device=self.device)

                # Calculate correlation matrix if using DiversificationGAT
                correlation_matrix = None
                if self.config.use_diversification_gat:
                    correlation_matrix = torch.tensor(
                        returns_data.corr().fillna(0).values,
                        dtype=torch.float32,
                        device=self.device
                    )

                # Forward pass to get portfolio weights for prediction universe
                if self.config.use_diversification_gat:
                    weights, _ = self.model(x, edge_index, edge_attr, correlation_matrix)
                else:
                    # Model returns (weights, memory, regularization_loss)
                    weights, _, _ = self.model(x, edge_index, mask_valid, edge_attr)

                # Convert to pandas Series for prediction universe
                prediction_weights = pd.Series(
                    weights.cpu().numpy(),
                    index=prediction_universe
                )

                # Ensure non-negative and normalized weights
                prediction_weights = prediction_weights.clip(lower=0.0)
                weights_sum = prediction_weights.sum()

                if weights_sum > 0:
                    prediction_weights = prediction_weights / weights_sum
                else:
                    # Fallback to equal weights if all weights are zero/negative
                    self.logger.debug(f"GAT weights sum to {weights_sum}, using equal weights")
                    prediction_weights = pd.Series(1.0 / len(prediction_universe), index=prediction_universe)

                # Expand predictions to full requested universe
                raw_weights = self._expand_weights_to_universe(prediction_weights, original_universe)

            except Exception as e:
                # Log the actual error before falling back to equal weights
                self.logger.error(
                    f"GAT prediction failed with error: {str(e)}\n"
                    f"Error type: {type(e).__name__}\n"
                    f"Graph method: {self.config.graph_config.filter_method}\n"
                    f"Universe size: {len(original_universe)}\n"
                    f"Prediction universe size: {len(prediction_universe) if 'prediction_universe' in locals() else 'N/A'}"
                )
                self.logger.exception("Full GAT prediction exception traceback:")

                # Fallback to equal weights for full requested universe
                self.logger.warning(f"GAT {self.config.graph_config.filter_method} falling back to equal weights due to error")
                raw_weights = pd.Series(1.0 / len(original_universe), index=original_universe)

            # Apply portfolio constraints
            # CRITICAL FIX: Force hard constraint enforcement for GAT models
            # Override the base class method to ensure position limits are respected
            constrained_weights = self._enforce_hard_constraints(raw_weights)

            return constrained_weights

    def _enforce_hard_constraints(self, weights: pd.Series) -> pd.Series:
        """
        Enforce hard constraints on GAT model outputs.

        This method ensures that position limits are strictly enforced,
        preventing the extreme concentration issues we've observed.

        Args:
            weights: Raw model weights

        Returns:
            Constrained weights that respect all limits
        """
        import logging
        logger = logging.getLogger(__name__)

        # Start with a copy
        constrained = weights.copy()

        # 1. Ensure long-only first (no negative weights)
        if self.constraints.long_only:
            constrained = constrained.clip(lower=0)

        # 2. Remove positions below minimum threshold
        if self.constraints.min_weight_threshold > 0:
            constrained[constrained < self.constraints.min_weight_threshold] = 0

        # 3. Iteratively enforce max position weight
        # This is necessary because renormalization can push weights above limit
        max_position = self.constraints.max_position_weight
        if max_position < 1.0:
            max_iterations = 10  # Prevent infinite loops
            for iteration in range(max_iterations):
                violations = (constrained > max_position).sum()
                if violations == 0:
                    break

                if iteration == 0:
                    max_before = constrained.max()
                    logger.warning(f"GAT hard constraint: {violations} positions exceed {max_position:.1%} limit (max: {max_before:.1%})")

                # Clip weights to max position
                constrained = constrained.clip(upper=max_position)

                # Renormalize only the non-zero weights to maintain sum=1
                weight_sum = constrained.sum()
                if weight_sum > 0:
                    constrained = constrained / weight_sum
                else:
                    # Fallback to equal weights if all are zero
                    constrained = pd.Series(1.0 / len(constrained), index=constrained.index)
                    break

            # Final check
            if constrained.max() > max_position * 1.01:  # Allow 1% tolerance for numerical precision
                # If still violating, use a more aggressive approach
                # Distribute excess weight proportionally to all positions below limit
                excess = constrained[constrained > max_position] - max_position
                total_excess = excess.sum()
                constrained[constrained > max_position] = max_position

                below_limit = constrained[constrained < max_position]
                if len(below_limit) > 0:
                    # Add excess proportionally to positions below limit
                    available_space = (max_position - below_limit).sum()
                    if available_space > 0:
                        for idx in below_limit.index:
                            space = max_position - constrained[idx]
                            constrained[idx] += total_excess * (space / available_space)

                # Final renormalization
                constrained = constrained / constrained.sum()

        # 4. Final validation
            weight_sum = constrained.sum()
            if weight_sum > 0:
                constrained = constrained / weight_sum
            else:
                # Fallback to equal weights if all weights are invalid
                logger.warning("All weights invalid after constraints, using equal weights")
                constrained = pd.Series(1.0 / len(constrained), index=constrained.index)

        # 4. Final validation
        final_sum = constrained.sum()
        final_max = constrained.max()

        if abs(final_sum - 1.0) > 1e-6:
            logger.error(f"Weight sum violation: {final_sum:.6f} != 1.0")
            constrained = constrained / final_sum

        if final_max > max_position + 1e-6:
            logger.error(f"Max position violation after constraints: {final_max:.1%} > {max_position:.1%}")

        return constrained

    def _get_historical_returns(self, date: pd.Timestamp, universe: list[str]) -> pd.DataFrame:
        """
        Load historical returns data up to prediction date.

        Args:
            date: Prediction date
            universe: Asset universe

        Returns:
            Historical returns DataFrame
        """
        try:
            # Try to load from production dataset
            returns_path = Path("data/final_new_pipeline/returns_daily_final.parquet")
            if returns_path.exists():
                all_returns = pd.read_parquet(returns_path)

                # Filter to date range and universe
                end_date = date - pd.Timedelta(days=1)  # Day before prediction
                lookback_days = self.config.graph_config.lookback_days
                start_date = end_date - pd.Timedelta(days=lookback_days + 30)  # Extra buffer

                # Filter by date and available assets
                available_assets = [asset for asset in universe if asset in all_returns.columns]
                if not available_assets:
                    raise ValueError("No assets from universe found in historical data")

                historical_data = all_returns.loc[start_date:end_date, available_assets]

                # Forward fill missing values
                historical_data = historical_data.ffill().fillna(0.0)

                # Ensure we have enough data
                if len(historical_data) < lookback_days:
                    logger.warning(f"Limited historical data: {len(historical_data)} < {lookback_days}")
                    # Pad with zeros if needed
                    if len(historical_data) > 0:
                        padding_needed = lookback_days - len(historical_data)
                        padding_dates = pd.date_range(
                            end=historical_data.index[0] - pd.Timedelta(days=1),
                            periods=padding_needed,
                            freq='D'
                        )
                        padding_data = pd.DataFrame(
                            np.zeros((padding_needed, len(historical_data.columns))),
                            index=padding_dates,
                            columns=historical_data.columns
                        )
                        historical_data = pd.concat([padding_data, historical_data])

                # Take only the required lookback period
                return historical_data.tail(lookback_days)

            else:
                raise FileNotFoundError("Production returns data not found")

        except Exception as e:
            logger.warning(f"Failed to load historical data: {e}, using synthetic data")
            # Fallback to synthetic data
            lookback_days = self.config.graph_config.lookback_days
            date_range = pd.date_range(
                end=date - pd.Timedelta(days=1),
                periods=lookback_days,
                freq='D'
            )
            np.random.seed(int(date.timestamp()) % 2**32)
            synthetic_returns = pd.DataFrame(
                np.random.normal(0.001, 0.02, (lookback_days, len(universe))),
                index=date_range,
                columns=universe
            )
            return synthetic_returns

    def get_model_info(self) -> dict[str, Any]:
        """
        Return enhanced model metadata including gradient flow analysis.

        Returns:
            Dictionary containing model type, hyperparameters, constraints,
            training diagnostics, and other metadata for performance analysis.
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
            info["trainable_parameters"] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Add gradient flow analysis if available
        if hasattr(self, 'gradient_monitor') and self.gradient_monitor and False:  # Disabled
            gradient_summary = self.gradient_monitor.get_gradient_summary()
            if gradient_summary:
                info["gradient_analysis"] = gradient_summary

        # Add training quality assessment
        if hasattr(self, "training_history") and self.training_history:
            training_quality = {}

            # Loss convergence analysis
            if "loss" in self.training_history and len(self.training_history["loss"]) > 5:
                losses = self.training_history["loss"]
                training_quality["loss_convergence"] = {
                    "final_loss": losses[-1],
                    "min_loss": min(losses),
                    "loss_reduction": (losses[0] - losses[-1]) / max(losses[0], 1e-8),
                    "converged": abs(losses[-1] - losses[-2]) < 1e-6 if len(losses) > 1 else False,
                }

            # Gradient health analysis
            if "gradient_norm" in self.training_history and len(self.training_history["gradient_norm"]) > 0:
                grad_norms = self.training_history["gradient_norm"]
                zero_fractions = self.training_history.get("zero_grad_fraction", [])

                training_quality["gradient_health"] = {
                    "final_gradient_norm": grad_norms[-1] if grad_norms else 0.0,
                    "avg_gradient_norm": sum(grad_norms) / len(grad_norms) if grad_norms else 0.0,
                    "gradient_stability": len([g for g in grad_norms if 1e-6 < g < 100]) / max(len(grad_norms), 1),
                    "final_zero_fraction": zero_fractions[-1] if zero_fractions else 1.0,
                    "avg_zero_fraction": sum(zero_fractions) / len(zero_fractions) if zero_fractions else 1.0,
                }

            # Constraint compliance analysis
            if "constraint_violations" in self.training_history and len(self.training_history["constraint_violations"]) > 0:
                violations = self.training_history["constraint_violations"]
                training_quality["constraint_compliance"] = {
                    "final_violation": violations[-1],
                    "avg_violation": sum(violations) / len(violations),
                    "violation_trend": "improving" if violations[-1] < violations[0] else "worsening",
                }

            info["training_quality"] = training_quality

        return info

    def save_model(self, filepath: str) -> None:
        """Save complete model state including configuration and weights."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Cannot save unfitted model")

        model_state = {
            "model_state_dict": self.model.state_dict(),
            "config": {
                "input_features": self.config.input_features,
                "hidden_dim": self.config.hidden_dim,
                "num_layers": self.config.num_layers,
                "num_attention_heads": self.config.num_attention_heads,
                "dropout": self.config.dropout,
                "use_gatv2": self.config.use_gatv2,
                "residual": self.config.residual,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "max_epochs": self.config.max_epochs,
            },
            "constraints": {
                "long_only": self.constraints.long_only,
                "top_k_positions": self.constraints.top_k_positions,
                "max_position_weight": self.constraints.max_position_weight,
                "max_monthly_turnover": self.constraints.max_monthly_turnover,
                "min_weight_threshold": self.constraints.min_weight_threshold,
            },
            "model_type": "GATPortfolioModel",
            "universe": getattr(self, "universe", []),
            "fitted_period": getattr(self, "fitted_period", None),
            "training_history": getattr(self, "training_history", {}),
            "device": str(self.device),
        }

        torch.save(model_state, filepath)

        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"GAT model saved to {filepath}")

    def _expand_weights_to_universe(self, prediction_weights: pd.Series, target_universe: list[str]) -> pd.Series:
        """
        Expand weights from prediction universe to full target universe.

        For assets not in prediction universe, assign zero weight initially.
        The constraint engine will handle any necessary rebalancing.

        Args:
            prediction_weights: Weights for assets model can predict on
            target_universe: Full universe of assets needed

        Returns:
            Weights expanded to full target universe
        """
        # Initialize all assets with zero weight
        expanded_weights = pd.Series(0.0, index=target_universe)

        # Set weights for assets we have predictions for
        for asset in prediction_weights.index:
            if asset in target_universe:
                expanded_weights[asset] = prediction_weights[asset]

        # Handle assets not in predictions (new assets entering universe)
        missing_assets = [asset for asset in target_universe if asset not in prediction_weights.index]

        if missing_assets:
            # Allocate small weight to new assets (2% total, distributed equally)
            new_asset_allocation = 0.02
            remaining_allocation = 1.0 - new_asset_allocation

            # Scale down existing predictions
            if expanded_weights.sum() > 0:
                expanded_weights = expanded_weights * remaining_allocation / expanded_weights.sum()

            # Assign equal small weights to new assets
            new_asset_weight = new_asset_allocation / len(missing_assets)
            for asset in missing_assets:
                expanded_weights[asset] = new_asset_weight

        # Ensure weights sum to 1 (renormalize if needed)
        total_weight = expanded_weights.sum()
        if total_weight > 0:
            expanded_weights = expanded_weights / total_weight
        else:
            # If no predictions could be mapped, fall back to equal weights
            expanded_weights = pd.Series(1.0 / len(target_universe), index=target_universe)

        return expanded_weights

    def load_model(self, filepath: str) -> None:
        """Load complete model state from checkpoint."""
        import logging
        logger = logging.getLogger(__name__)

        try:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

            # Extract configuration if available
            if "config" in checkpoint:
                config_dict = checkpoint["config"]
                self.config.input_features = config_dict.get("input_features", self.config.input_features)
                self.config.hidden_dim = config_dict.get("hidden_dim", self.config.hidden_dim)
                self.config.num_layers = config_dict.get("num_layers", self.config.num_layers)
                self.config.num_attention_heads = config_dict.get("num_attention_heads", self.config.num_attention_heads)
                self.config.dropout = config_dict.get("dropout", self.config.dropout)

            # Build model with loaded configuration
            if self.model is None:
                self.model = self._build_model(self.config.input_features)

            # Load model weights
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                logger.info(f"Loaded GAT model weights from {filepath}")
            else:
                logger.warning(f"No model_state_dict found in {filepath}")

            # Load training metadata if available
            if "universe" in checkpoint:
                self.universe = checkpoint["universe"]
                logger.info(f"Loaded universe with {len(self.universe)} assets")

            if "fitted_period" in checkpoint:
                self.fitted_period = checkpoint["fitted_period"]
                logger.info(f"Loaded fitted period: {self.fitted_period}")

            if "training_history" in checkpoint:
                self.training_history = checkpoint["training_history"]
                logger.info(f"Loaded training history with {len(self.training_history.get('loss', []))} epochs")

            # Mark as fitted and pretrained
            self.is_fitted = True
            self._is_pretrained = True  # Skip retraining in backtest

            # Reinitialise gradient monitor for loaded model
            # Gradient monitoring disabled for production

            logger.info(f"GAT model loaded successfully from {filepath}")

        except Exception as e:
            logger.error(f"Failed to load GAT model from {filepath}: {e}")
            raise
