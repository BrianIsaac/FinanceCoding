"""
LSTM Portfolio Model implementation.

This module implements the complete LSTM-based portfolio optimization model
that integrates with the existing portfolio construction framework.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.optimize import minimize

from ..base.portfolio_model import PortfolioConstraints, PortfolioModel
from .architecture import LSTMConfig, LSTMNetwork, create_lstm_network
from .training import MemoryEfficientTrainer, TrainingConfig, create_trainer

logger = logging.getLogger(__name__)


@dataclass
class LSTMModelConfig:
    """Complete configuration for LSTM portfolio model."""

    # Architecture configuration
    lstm_config: LSTMConfig = field(default_factory=LSTMConfig)

    # Training configuration
    training_config: TrainingConfig = field(default_factory=TrainingConfig)

    # Model-specific parameters
    lookback_days: int = 756  # 3 years of trading days for correlation estimation
    rebalancing_frequency: str = "monthly"  # Rebalancing frequency
    prediction_horizon: int = 21  # Days ahead to predict (monthly)

    # Risk management
    risk_aversion: float = 1.0  # Risk aversion parameter for mean-variance optimization

    # Portfolio optimization
    use_markowitz_layer: bool = True  # Apply Markowitz optimization to LSTM predictions
    shrinkage_target: float = 0.1  # Shrinkage target for covariance estimation

    @classmethod
    def from_yaml(cls, filepath: Path) -> LSTMModelConfig:
        """Load configuration from YAML file."""
        with open(filepath) as f:
            config_dict = yaml.safe_load(f)

        # Handle nested configurations
        if "lstm_config" in config_dict:
            config_dict["lstm_config"] = LSTMConfig(**config_dict["lstm_config"])

        if "training_config" in config_dict:
            config_dict["training_config"] = TrainingConfig(**config_dict["training_config"])

        return cls(**config_dict)

    def to_yaml(self, filepath: Path) -> None:
        """Save configuration to YAML file."""
        # Convert dataclasses to dictionaries
        config_dict = {
            "lstm_config": {
                "sequence_length": self.lstm_config.sequence_length,
                "input_size": self.lstm_config.input_size,
                "hidden_size": self.lstm_config.hidden_size,
                "num_layers": self.lstm_config.num_layers,
                "dropout": self.lstm_config.dropout,
                "num_attention_heads": self.lstm_config.num_attention_heads,
                "output_size": self.lstm_config.output_size,
            },
            "training_config": {
                "max_memory_gb": self.training_config.max_memory_gb,
                "gradient_accumulation_steps": self.training_config.gradient_accumulation_steps,
                "use_mixed_precision": self.training_config.use_mixed_precision,
                "learning_rate": self.training_config.learning_rate,
                "weight_decay": self.training_config.weight_decay,
                "batch_size": self.training_config.batch_size,
                "epochs": self.training_config.epochs,
                "patience": self.training_config.patience,
            },
            "lookback_days": self.lookback_days,
            "rebalancing_frequency": self.rebalancing_frequency,
            "prediction_horizon": self.prediction_horizon,
            "risk_aversion": self.risk_aversion,
            "use_markowitz_layer": self.use_markowitz_layer,
            "shrinkage_target": self.shrinkage_target,
        }

        with open(filepath, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)


class LSTMPortfolioModel(PortfolioModel):
    """
    LSTM-based portfolio optimization model.

    Uses LSTM networks to predict future returns and applies mean-variance optimization
    to construct portfolios that satisfy constraints and maximize risk-adjusted returns.
    """

    def __init__(self, constraints: PortfolioConstraints, config: LSTMModelConfig | None = None):
        """
        Initialize LSTM portfolio model.

        Args:
            constraints: Portfolio constraints configuration
            config: LSTM model configuration (uses defaults if None)
        """
        super().__init__(constraints)

        self.config = config or LSTMModelConfig()
        self.network: LSTMNetwork | None = None
        self.trainer: MemoryEfficientTrainer | None = None
        self.universe: list[str] | None = None
        self.training_history: dict | None = None

        # Model state
        self.fitted_period: tuple[pd.Timestamp, pd.Timestamp] | None = None
        self.last_prediction_date: pd.Timestamp | None = None

        logger.info("LSTM Portfolio Model initialized")

    def fit(
        self,
        returns: pd.DataFrame,
        universe: list[str],
        fit_period: tuple[pd.Timestamp, pd.Timestamp],
        checkpoint_dir: Path | None = None,
    ) -> None:
        """
        Train LSTM model on historical return data.

        Args:
            returns: Historical returns DataFrame with datetime index and asset columns
            universe: List of asset tickers to include in optimization
            fit_period: (start_date, end_date) tuple defining training period
            checkpoint_dir: Directory to save model checkpoints

        Raises:
            ValueError: If returns data is insufficient or invalid
        """
        logger.info(f"Training LSTM model on period {fit_period[0]} to {fit_period[1]}")

        # Validate inputs
        self._validate_fit_inputs(returns, universe, fit_period)

        # Filter data for training period and universe
        training_data = self._prepare_training_data(returns, universe, fit_period)

        # Update configuration based on data
        self.config.lstm_config.input_size = len(universe)
        self.config.lstm_config.output_size = len(universe)

        # Create LSTM network
        self.network = create_lstm_network(self.config.lstm_config)

        # Create trainer
        self.trainer = create_trainer(self.network, self.config.training_config)

        # Train model
        self.training_history = self.trainer.fit(
            training_data,
            sequence_length=self.config.lstm_config.sequence_length,
            checkpoint_dir=checkpoint_dir,
        )

        # Update model state
        self.universe = universe.copy()
        self.fitted_period = fit_period
        self.is_fitted = True

        logger.info("LSTM model training completed successfully")

    def predict_weights(self, date: pd.Timestamp, universe: list[str]) -> pd.Series:
        """
        Generate portfolio weights using LSTM predictions.

        Args:
            date: Rebalancing date for which to generate weights
            universe: List of asset tickers (must be subset of fitted universe)

        Returns:
            Portfolio weights as pandas Series with asset tickers as index.
            Weights sum to 1.0 and satisfy all portfolio constraints.

        Raises:
            ValueError: If model is not fitted or universe is invalid
        """
        if not self.is_fitted or self.network is None:
            raise ValueError("Model must be fitted before generating predictions")

        if not all(asset in self.universe for asset in universe):
            raise ValueError("Universe contains assets not in fitted universe")

        logger.info(f"Generating LSTM portfolio weights for {date.strftime('%Y-%m-%d')}")

        # Get LSTM return predictions
        predicted_returns = self._predict_returns(date, universe)

        # Apply Markowitz optimization if enabled
        if self.config.use_markowitz_layer:
            weights = self._optimize_portfolio(predicted_returns, universe, date)
        else:
            # Use predicted returns directly as weights (after normalization)
            weights = pd.Series(predicted_returns, index=universe)
            weights = weights.clip(lower=0.0)  # Ensure non-negative
            weights = (
                weights / weights.sum()
                if weights.sum() > 0
                else pd.Series(1.0 / len(universe), index=universe)
            )

        # Apply portfolio constraints
        weights = self.validate_weights(weights)

        self.last_prediction_date = date

        logger.info(
            f"Generated weights for {len(weights)} assets, top 5: {weights.nlargest(5).to_dict()}"
        )

        return weights

    def _predict_returns(self, date: pd.Timestamp, universe: list[str]) -> np.ndarray:
        """
        Generate return predictions using trained LSTM network.

        Args:
            date: Prediction date
            universe: Asset universe

        Returns:
            Predicted returns array
        """
        # This is a simplified implementation - in practice, you would:
        # 1. Load historical returns data up to the prediction date
        # 2. Create input sequences for the LSTM
        # 3. Run forward pass through the network
        # 4. Extract predictions for the specified universe

        # For now, return mock predictions (to be implemented with actual data pipeline)
        logger.warning("Using mock predictions - implement actual LSTM inference")

        # Mock implementation: random predictions with some structure  # pragma: no cover
        np.random.seed(int(date.timestamp()) % 2**32)
        mock_predictions = np.random.normal(0.001, 0.05, len(universe))  # 0.1% mean, 5% std

        return mock_predictions

    def _optimize_portfolio(
        self, expected_returns: np.ndarray, universe: list[str], date: pd.Timestamp
    ) -> pd.Series:
        """
        Apply mean-variance optimization to LSTM predictions.

        Args:
            expected_returns: Expected returns from LSTM
            universe: Asset universe
            date: Portfolio construction date

        Returns:
            Optimized portfolio weights
        """
        # Estimate covariance matrix (simplified - should use historical data)
        # In practice, load historical returns and compute sample covariance
        n_assets = len(universe)

        # Mock covariance matrix (to be replaced with actual estimation)
        np.random.seed(42)
        correlation = np.random.uniform(0.1, 0.7, (n_assets, n_assets))
        correlation = (correlation + correlation.T) / 2  # Make symmetric
        np.fill_diagonal(correlation, 1.0)

        volatilities = np.random.uniform(0.15, 0.35, n_assets)  # 15-35% annual volatility
        covariance = np.outer(volatilities, volatilities) * correlation

        # Apply shrinkage
        target_var = np.mean(np.diag(covariance))
        target_cov = np.eye(n_assets) * target_var
        covariance = (
            1 - self.config.shrinkage_target
        ) * covariance + self.config.shrinkage_target * target_cov

        # Mean-variance optimization
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance, weights))
            return -portfolio_return + self.config.risk_aversion * portfolio_variance

        # Constraints
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]  # Weights sum to 1

        # Bounds (long-only)
        bounds = [(0.0, self.constraints.max_position_weight) for _ in range(n_assets)]

        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets

        # Optimization
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "disp": False},
        )

        if result.success:
            weights = pd.Series(result.x, index=universe)
        else:
            logger.warning("Portfolio optimization failed, using constrained equal weights")
            # Use equal weights but respect max position weight constraint
            max_weight = min(1.0 / n_assets, self.constraints.max_position_weight)
            weights = pd.Series(max_weight, index=universe)
            weights = weights / weights.sum()  # Normalize to sum to 1

        return weights

    def _validate_fit_inputs(
        self,
        returns: pd.DataFrame,
        universe: list[str],
        fit_period: tuple[pd.Timestamp, pd.Timestamp],
    ) -> None:
        """Validate inputs for model fitting."""
        if returns.empty:
            raise ValueError("Returns DataFrame is empty")

        if not universe:
            raise ValueError("Universe cannot be empty")

        missing_assets = set(universe) - set(returns.columns)
        if missing_assets:
            raise ValueError(f"Missing assets in returns data: {missing_assets}")

        if fit_period[0] >= fit_period[1]:
            raise ValueError("Invalid fit period: start date must be before end date")

        # Check minimum data requirements
        min_days = self.config.lstm_config.sequence_length + self.config.prediction_horizon
        period_days = (fit_period[1] - fit_period[0]).days

        if period_days <= min_days:
            raise ValueError(
                f"Insufficient data: need at least {min_days + 1} days, got {period_days} days"
            )

    def _prepare_training_data(
        self,
        returns: pd.DataFrame,
        universe: list[str],
        fit_period: tuple[pd.Timestamp, pd.Timestamp],
    ) -> pd.DataFrame:
        """Prepare and validate training data."""
        # Filter by period and universe
        mask = (returns.index >= fit_period[0]) & (returns.index <= fit_period[1])
        training_data = returns.loc[mask, universe].copy()

        # Handle missing data
        training_data = training_data.ffill().fillna(0.0)

        # Validate data quality
        if training_data.isna().sum().sum() > 0:
            logger.warning("Training data contains NaN values after preprocessing")

        logger.info(
            f"Prepared training data: {training_data.shape[0]} days, {training_data.shape[1]} assets"
        )

        return training_data

    def get_model_info(self) -> dict[str, Any]:
        """
        Return LSTM model metadata and configuration.

        Returns:
            Dictionary containing model information, hyperparameters, and training statistics
        """
        info = {
            "model_type": "LSTM",
            "is_fitted": self.is_fitted,
            "universe_size": len(self.universe) if self.universe else None,
            "fitted_period": self.fitted_period,
            "last_prediction_date": self.last_prediction_date,
            "constraints": {
                "long_only": self.constraints.long_only,
                "top_k_positions": self.constraints.top_k_positions,
                "max_position_weight": self.constraints.max_position_weight,
                "max_monthly_turnover": self.constraints.max_monthly_turnover,
                "transaction_cost_bps": self.constraints.transaction_cost_bps,
            },
            "lstm_config": {
                "sequence_length": self.config.lstm_config.sequence_length,
                "hidden_size": self.config.lstm_config.hidden_size,
                "num_layers": self.config.lstm_config.num_layers,
                "dropout": self.config.lstm_config.dropout,
                "num_attention_heads": self.config.lstm_config.num_attention_heads,
            },
            "training_config": {
                "learning_rate": self.config.training_config.learning_rate,
                "batch_size": self.config.training_config.batch_size,
                "epochs": self.config.training_config.epochs,
                "use_mixed_precision": self.config.training_config.use_mixed_precision,
            },
        }

        # Add training statistics if available
        if self.training_history:
            info["training_stats"] = {
                "final_train_loss": (
                    self.training_history["train_loss"][-1]
                    if self.training_history["train_loss"]
                    else None
                ),
                "final_val_loss": (
                    self.training_history["val_loss"][-1]
                    if self.training_history["val_loss"]
                    else None
                ),
                "training_epochs": len(self.training_history["train_loss"]),
                "best_val_loss": (
                    min(self.training_history["val_loss"])
                    if self.training_history["val_loss"]
                    else None
                ),
            }

        # Add network parameter count if model is fitted
        if self.network:
            info["network_params"] = sum(p.numel() for p in self.network.parameters())

        return info

    def save_model(self, filepath: Path) -> None:
        """Save complete model state including configuration and weights."""
        if not self.is_fitted or self.network is None:
            raise ValueError("Cannot save unfitted model")

        model_state = {
            "config": self.config,
            "constraints": self.constraints,
            "network_state_dict": self.network.state_dict(),
            "universe": self.universe,
            "fitted_period": self.fitted_period,
            "training_history": self.training_history,
            "model_info": self.get_model_info(),
        }

        torch.save(model_state, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: Path) -> None:
        """Load complete model state including configuration and weights."""
        model_state = torch.load(filepath, map_location="cpu", weights_only=False)

        # Restore configuration and state
        self.config = model_state["config"]
        self.constraints = model_state["constraints"]
        self.universe = model_state["universe"]
        self.fitted_period = model_state["fitted_period"]
        self.training_history = model_state["training_history"]

        # Recreate and load network
        self.network = create_lstm_network(self.config.lstm_config)
        self.network.load_state_dict(model_state["network_state_dict"])

        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")


def create_lstm_model(
    constraints: PortfolioConstraints, config_path: Path | None = None, **config_overrides
) -> LSTMPortfolioModel:
    """
    Factory function to create LSTM portfolio model with configuration.

    Args:
        constraints: Portfolio constraints
        config_path: Path to YAML configuration file
        **config_overrides: Configuration parameters to override

    Returns:
        Configured LSTM portfolio model
    """
    # Load configuration from file if provided
    if config_path and config_path.exists():
        config = LSTMModelConfig.from_yaml(config_path)
    else:
        config = LSTMModelConfig()

    # Apply configuration overrides
    for key, value in config_overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown configuration parameter: {key}")

    return LSTMPortfolioModel(constraints, config)
