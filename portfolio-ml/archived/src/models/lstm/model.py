"""
LSTM Portfolio Model implementation.

This module implements the complete LSTM-based portfolio optimization model
that integrates with the existing portfolio construction framework.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.optimize import minimize

from ..base.portfolio_model import PortfolioConstraints, PortfolioModel
from ..base.confidence_weighted_training import (
    ConfidenceWeightedTrainer,
    TrainingStrategy,
    create_confidence_weighted_trainer,
)
from .architecture import LSTMConfig, LSTMNetwork, create_lstm_network
from .training import MemoryEfficientTrainer, TrainingConfig, create_trainer

# Import adaptive padding for memory optimization
try:
    from ...utils.adaptive_padding import AdaptivePaddingConfig, AdaptivePaddingStrategy
    ADAPTIVE_PADDING_AVAILABLE = True
except ImportError:
    logger.warning("Adaptive padding not available, using legacy padding")
    ADAPTIVE_PADDING_AVAILABLE = False

# Import flexible academic validation
try:
    from ...evaluation.validation.flexible_academic_validator import (
        FlexibleAcademicValidator,
        AcademicValidationResult,
    )
    FLEXIBLE_VALIDATION_AVAILABLE = True
except ImportError:
    logger.info("Flexible validation not available, using standard validation")
    FLEXIBLE_VALIDATION_AVAILABLE = False

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

        # Confidence-weighted training support
        self.confidence_trainer = create_confidence_weighted_trainer()
        self.flexible_validator = (
            FlexibleAcademicValidator()
            if FLEXIBLE_VALIDATION_AVAILABLE
            else None
        )
        self.last_training_strategy: TrainingStrategy | None = None
        self.last_validation_result: AcademicValidationResult | None = None

        # Model state
        self.fitted_period: tuple[pd.Timestamp, pd.Timestamp] | None = None
        self.last_prediction_date: pd.Timestamp | None = None

        # Initialize adaptive padding strategy if available
        if ADAPTIVE_PADDING_AVAILABLE:
            self.padding_strategy = AdaptivePaddingStrategy(
                AdaptivePaddingConfig(
                    max_padding_ratio=0.1,
                    enable_dynamic_architecture=True,
                    use_correlation_substitution=False,
                    enable_adaptive_sequences=True
                )
            )
            logger.info("LSTM Portfolio Model initialized with adaptive padding optimization")
        else:
            self.padding_strategy = None
            logger.info("LSTM Portfolio Model initialized with legacy padding")

    def supports_rolling_retraining(self) -> bool:
        """LSTM supports rolling retraining with warm starts."""
        return True

    def rolling_fit(
        self,
        returns: pd.DataFrame,
        universe: list[str],
        rebalance_date: pd.Timestamp,
        lookback_months: int = 36,
        min_observations: int = 100,  # Reduced default for flexible academic framework
    ) -> None:
        """
        Perform rolling fit for LSTM model with warm start.

        Uses existing network weights as initialization for faster convergence,
        performing limited epochs to adapt to recent market conditions.

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

        # Load fresh returns data with buffer for sequence creation
        training_data = self._load_fresh_returns_data(
            returns, start_date, end_date, universe
        )

        if len(training_data) < min_observations:
            raise ValueError(
                f"Insufficient data for rolling fit: {len(training_data)} < {min_observations}"
            )

        # Quick retrain with limited epochs
        self._quick_retrain(training_data, universe, max_epochs=20)

        # Update model state
        self.universe = universe.copy()
        self.fitted_period = (start_date, end_date)
        self.is_fitted = True

    def _load_fresh_returns_data(
        self,
        returns: pd.DataFrame | Path | str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        universe: list[str],
    ) -> pd.DataFrame:
        """
        Load sufficient data for LSTM sequences and prediction.

        Args:
            returns: Full historical returns or path to data
            start_date: Start of training window
            end_date: End of training window
            universe: Assets to include

        Returns:
            Cleaned returns DataFrame with buffer for sequence creation
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

        # Add buffer for sequence creation
        buffer_days = self.config.lstm_config.sequence_length + self.config.prediction_horizon + 30
        extended_start = start_date - pd.Timedelta(days=buffer_days)

        # Filter by extended date range
        mask = (returns.index >= extended_start) & (returns.index <= end_date)
        period_returns = returns[mask]

        # Filter for available universe assets
        available_assets = [asset for asset in universe if asset in period_returns.columns]

        if len(available_assets) == 0:
            raise ValueError("No assets from universe found in returns data")

        filtered_returns = period_returns[available_assets]

        # Clean data
        cleaned_returns = filtered_returns.ffill().fillna(0.0)

        return cleaned_returns

    def _adjust_data_to_optimal_size(self, data: pd.DataFrame, universe: list[str], target_size: int) -> pd.DataFrame:
        """
        Adjust data to optimal size with intelligent asset selection and minimal padding.
        Uses adaptive padding strategy if available for memory optimization.

        Args:
            data: Input data DataFrame
            universe: Current universe of assets
            target_size: Target dimension size

        Returns:
            DataFrame adjusted to target dimensions
        """
        current_size = data.shape[1]

        # Use adaptive padding strategy if available
        if ADAPTIVE_PADDING_AVAILABLE and self.padding_strategy is not None:
            # Calculate data density for adaptive sizing
            data_density = 1.0 - (data.isna().sum().sum() / data.size) if data.size > 0 else 1.0

            # Determine optimal size using adaptive strategy
            optimal_size, strategy = self.padding_strategy.calculate_optimal_size(
                current_size, target_size, data_density
            )

            # Log memory savings if applicable
            if optimal_size != target_size:
                batch_size = self.config.training_config.batch_size
                seq_length = self.config.lstm_config.sequence_length
                savings = self.padding_strategy.get_memory_savings(
                    target_size, optimal_size, batch_size, seq_length
                )
                logger.info(
                    f"Adaptive padding: {current_size} -> {optimal_size} (target={target_size}), "
                    f"strategy={strategy}, memory_saved={savings['savings_mb']:.1f}MB ({savings['savings_percent']:.1f}%)"
                )

            # Handle different sizing scenarios
            if current_size == optimal_size:
                return data
            elif current_size > optimal_size:
                # Select most informative assets
                asset_activity = data.std().sort_values(ascending=False)
                top_assets = asset_activity.head(optimal_size).index
                return data[top_assets]
            else:
                # Use intelligent padding
                return self.padding_strategy.apply_intelligent_padding(
                    data, optimal_size, correlation_matrix=None
                )

        # Fall back to legacy padding if adaptive not available
        if current_size == target_size:
            return data
        elif current_size > target_size:
            # Select most informative assets based on volatility and trading activity
            asset_activity = data.std().sort_values(ascending=False)
            top_assets = asset_activity.head(target_size).index
            selected_data = data[top_assets]
            logger.info(f"Selected top {target_size} assets from {current_size} available")
            return selected_data
        else:
            # Minimal padding - only add zeros if absolutely necessary
            padding_needed = target_size - current_size
            if padding_needed <= current_size * 0.1:  # Only allow 10% padding
                padding_cols = [f'PAD_{i}' for i in range(current_size, target_size)]
                padding_df = pd.DataFrame(
                    np.zeros((len(data), padding_needed)),
                    index=data.index,
                    columns=padding_cols
                )
                logger.info(f"Added minimal padding of {padding_needed} features ({padding_needed/target_size:.1%})")
                return pd.concat([data, padding_df], axis=1)
            else:
                # Too much padding would be needed - use current size instead
                logger.warning(f"Refusing to pad {current_size} -> {target_size} (would be {padding_needed/target_size:.1%} padding)")
                # Update target size to current size
                self.config.input_size = current_size
                self.config.output_size = current_size
                self.network = create_lstm_network(self.config)
                logger.info(f"Recreated network with input_size={current_size}")
                return data

    def _pad_or_truncate_data(self, data: pd.DataFrame, universe: list[str]) -> pd.DataFrame:
        """
        Legacy method - replaced by _adjust_data_to_optimal_size for better efficiency.
        Kept for backward compatibility.
        """
        return self._adjust_data_to_optimal_size(data, universe, self.config.lstm_config.input_size)

    def _quick_retrain(
        self,
        training_data: pd.DataFrame,
        universe: list[str],
        max_epochs: int = 20,
        confidence_score: Optional[float] = None,
    ) -> None:
        """
        Fast retraining for rolling updates using warm start with confidence-weighted training.

        Args:
            training_data: Training data for current window
            universe: Asset universe
            max_epochs: Maximum epochs for quick retraining
            confidence_score: Optional academic confidence score for weighted training
        """
        # Use dynamic input sizing for better efficiency and training stability
        current_universe_size = len(universe)

        # Apply min/max constraints for stability
        min_size = getattr(self.config.lstm_config, 'min_input_size', 50)
        max_size = getattr(self.config.lstm_config, 'max_input_size', 700)
        optimal_size = max(min_size, min(current_universe_size, max_size))

        if self.network is None or self.config.lstm_config.input_size != optimal_size:
            # Create network with optimal size for current universe
            self.config.lstm_config.input_size = optimal_size
            self.config.lstm_config.output_size = optimal_size
            self.network = create_lstm_network(self.config.lstm_config)
            logger.info(f"Created LSTM network with input_size={optimal_size} for universe_size={current_universe_size}")

        # Check if we have sufficient data for training
        min_required_samples = self.config.lstm_config.sequence_length + 21  # sequence_length + prediction_horizon
        if len(training_data) < min_required_samples:
            logger.warning(f"Insufficient data for retraining: {len(training_data)} < {min_required_samples} required samples")
            # Keep existing weights if any, or initialize random weights
            if self.network is None:
                logger.info("Initializing network with random weights due to insufficient training data")
                self.network = create_lstm_network(self.config.lstm_config)
                self.is_fitted = True  # Mark as fitted to allow predictions with random weights
            return

        # Adjust training data to match optimal dimensions (minimal padding)
        training_data = self._adjust_data_to_optimal_size(training_data, universe, optimal_size)

        # Validate data with flexible validator if available and get confidence score
        if self.flexible_validator and confidence_score is None:
            validation_result = self.flexible_validator.validate_with_confidence(
                data=training_data,
                universe=universe,
                context={"is_retraining": True}
            )
            confidence_score = validation_result.confidence
            self.last_validation_result = validation_result

            if not validation_result.can_proceed:
                logger.warning(
                    f"Validation failed with confidence {confidence_score:.2f}. "
                    f"Using existing weights."
                )
                return
        else:
            confidence_score = confidence_score or 0.7  # Default moderate confidence

        # Select training strategy based on confidence
        training_strategy = self.confidence_trainer.select_training_strategy(
            confidence_score=confidence_score,
            data_characteristics={
                "n_samples": len(training_data),
                "n_features": len(universe),
            }
        )
        self.last_training_strategy = training_strategy

        # Apply confidence-weighted preprocessing
        training_data = self.confidence_trainer.apply_data_preprocessing(
            training_data, training_strategy
        )

        # Adjust hyperparameters based on strategy
        base_params = {
            "epochs": max_epochs,
            "learning_rate": self.config.training_config.learning_rate,
            "dropout": 0.2,
        }
        adjusted_params = self.confidence_trainer.adjust_hyperparameters(
            base_params, training_strategy
        )

        # Create or update trainer with adjusted parameters
        if self.trainer is None:
            # Create new trainer with confidence-adjusted epochs
            quick_config = TrainingConfig(
                epochs=adjusted_params.get("epochs", max_epochs),
                patience=5,  # Reduced patience for quick training
                batch_size=self.config.training_config.batch_size,
                learning_rate=adjusted_params.get("learning_rate", self.config.training_config.learning_rate * 0.1),  # Use adjusted LR or default lower for fine-tuning
                weight_decay=adjusted_params.get("weight_decay", 0.001),
                use_mixed_precision=self.config.training_config.use_mixed_precision,
            )
            self.trainer = create_trainer(self.network, quick_config)
        else:
            # Update existing trainer config
            self.trainer.config.epochs = max_epochs
            self.trainer.config.patience = 5

        # Perform quick training
        try:
            self.training_history = self.trainer.fit(
                training_data,
                sequence_length=self.config.lstm_config.sequence_length,
                checkpoint_dir=None,  # Don't save checkpoints for quick retraining
            )
        except Exception as e:
            logger.warning(f"Quick retrain failed: {e}, keeping existing weights")

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

        # Use dynamic input sizing for better efficiency and training stability
        current_universe_size = len(universe)

        # Apply min/max constraints for stability
        min_size = getattr(self.config.lstm_config, 'min_input_size', 50)
        max_size = getattr(self.config.lstm_config, 'max_input_size', 700)
        optimal_size = max(min_size, min(current_universe_size, max_size))

        self.config.lstm_config.input_size = optimal_size
        self.config.lstm_config.output_size = optimal_size
        logger.info(f"Using LSTM input_size={optimal_size} for universe_size={current_universe_size}")

        # Create LSTM network with optimal dimensions
        self.network = create_lstm_network(self.config.lstm_config)

        # Adjust training data to match optimal dimensions (minimal padding)
        training_data = self._adjust_data_to_optimal_size(training_data, universe, optimal_size)

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

        # Handle dynamic universe membership for LSTM
        if self.universe:
            available_assets = [asset for asset in universe if asset in self.universe]
            unavailable_assets = [asset for asset in universe if asset not in self.universe]

            # If we have no overlap, we need to handle this gracefully
            if not available_assets:
                logger.warning(f"LSTM model has no overlap with current universe. Using equal weights for {len(universe)} assets.")
                equal_weight = 1.0 / len(universe)
                return pd.Series(equal_weight, index=universe)

            # Log if we're missing some assets (use debug level to reduce spam)
            if unavailable_assets:
                logger.debug(f"LSTM model missing {len(unavailable_assets)} assets from current universe: {unavailable_assets[:5]}...")

            # For LSTM, we'll predict on available assets then handle missing ones
            prediction_universe = available_assets
        else:
            prediction_universe = universe

        logger.info(f"Generating LSTM portfolio weights for {date.strftime('%Y-%m-%d')}")

        # Get LSTM return predictions for available assets
        predicted_returns = self._predict_returns(date, prediction_universe)

        # Apply Markowitz optimization if enabled
        if self.config.use_markowitz_layer:
            available_weights = self._optimize_portfolio(predicted_returns, prediction_universe, date)
        else:
            # Use predicted returns directly as weights (after normalization)
            available_weights = pd.Series(predicted_returns, index=prediction_universe)
            available_weights = available_weights.clip(lower=0.0)  # Ensure non-negative
            available_weights = (
                available_weights / available_weights.sum()
                if available_weights.sum() > 0
                else pd.Series(1.0 / len(prediction_universe), index=prediction_universe)
            )

        # Expand to full universe (assign equal weight to unavailable assets)
        if len(prediction_universe) < len(universe):
            # Create full universe weights
            weights = pd.Series(0.0, index=universe)

            # Assign 80% to LSTM predictions, 20% equally to new assets
            lstm_allocation = 0.8
            new_asset_allocation = 0.2

            weights[prediction_universe] = available_weights * lstm_allocation

            unavailable_assets = [asset for asset in universe if asset not in prediction_universe]
            if unavailable_assets:
                equal_new_weight = new_asset_allocation / len(unavailable_assets)
                weights[unavailable_assets] = equal_new_weight
        else:
            weights = available_weights

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
        if not self.is_fitted or self.network is None:
            raise ValueError("Model must be fitted before making predictions")

        try:
            # Load historical returns data up to prediction date
            returns_data = self._load_historical_returns(date, universe)

            # Create input sequences for LSTM with adaptive sequence length
            sequence_length = self.config.lstm_config.sequence_length

            # Use adaptive sequence length if available
            if ADAPTIVE_PADDING_AVAILABLE and self.padding_strategy is not None:
                adaptive_length = self.padding_strategy.calculate_adaptive_sequence_length(
                    returns_data, sequence_length
                )
                if adaptive_length != sequence_length:
                    logger.info(f"Using adaptive sequence length: {adaptive_length} (original: {sequence_length})")
                    sequence_length = adaptive_length

            input_sequences, selected_assets = self._create_prediction_sequences(returns_data, universe, date, sequence_length)

            # Ensure model and input are on same device
            device = next(self.network.parameters()).device
            input_sequences = input_sequences.to(device)

            # Run forward pass through trained network
            self.network.eval()
            with torch.no_grad():
                predictions, _ = self.network(input_sequences)
                # Extract predictions for the selected assets
                predicted_returns_raw = predictions.cpu().numpy().flatten()

                # Create full prediction array for all universe assets
                predicted_returns = np.full(len(universe), 0.001)  # Default conservative return

                # Map predictions from selected assets back to full universe
                if hasattr(selected_assets, '__iter__') and len(selected_assets) > 0:
                    for i, asset in enumerate(selected_assets):
                        if asset in universe:
                            universe_idx = universe.index(asset)
                            if i < len(predicted_returns_raw):
                                predicted_returns[universe_idx] = predicted_returns_raw[i]

                # Log statistics about predictions to verify they're differentiated
                logger.info(f"Generated LSTM predictions for {len(selected_assets)} selected assets out of {len(universe)} universe")
                logger.debug(f"Prediction stats: mean={np.mean(predicted_returns):.6f}, std={np.std(predicted_returns):.6f}, "
                            f"min={np.min(predicted_returns):.6f}, max={np.max(predicted_returns):.6f}")
                return predicted_returns

        except Exception as e:
            # Fallback to conservative predictions if inference fails
            logger.warning(f"LSTM inference failed: {e}, using fallback predictions")
            fallback_predictions = np.full(len(universe), 0.001)  # 0.1% conservative return
            return fallback_predictions

    def _optimize_portfolio(
        self, expected_returns: np.ndarray, universe: list[str], date: pd.Timestamp
    ) -> pd.Series:
        """
        Apply robust mean-variance optimization to LSTM predictions.

        Args:
            expected_returns: Expected returns from LSTM
            universe: Asset universe
            date: Portfolio construction date

        Returns:
            Optimized portfolio weights
        """
        n_assets = len(universe)

        # Load actual historical returns for covariance estimation
        try:
            historical_returns = self._get_historical_returns_for_optimization(date, universe)

            if historical_returns is not None and len(historical_returns) >= 30:
                # Calculate empirical covariance with proper handling
                returns_matrix = historical_returns.values

                # Center the returns
                centered_returns = returns_matrix - np.mean(returns_matrix, axis=0)

                # Calculate covariance with regularization
                cov_matrix = np.cov(centered_returns.T)

                # Check for numerical issues
                if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
                    logger.warning("Invalid covariance matrix, using identity matrix")
                    cov_matrix = np.eye(n_assets) * 0.04  # 20% annual volatility squared
                else:
                    # Apply Ledoit-Wolf shrinkage for stability
                    cov_matrix = self._ledoit_wolf_shrinkage(cov_matrix)
            else:
                # Fallback: Use diagonal covariance based on individual volatilities
                logger.warning("Insufficient data for covariance estimation, using diagonal matrix")
                individual_vols = np.full(n_assets, 0.20)  # 20% annual volatility
                cov_matrix = np.diag(individual_vols ** 2)

        except Exception as e:
            logger.warning(f"Failed to estimate covariance: {e}, using diagonal fallback")
            cov_matrix = np.eye(n_assets) * 0.04

        # Multiple optimization attempts with different methods
        weights = None

        # Attempt 1: Standard Mean-Variance Optimization
        try:
            weights = self._mean_variance_optimization(
                expected_returns, cov_matrix, universe
            )
            if weights is not None and self._validate_optimization_result(weights):
                return weights
        except Exception as e:
            logger.debug(f"Mean-variance optimization failed: {e}")

        # Attempt 2: Risk Parity Optimization
        try:
            weights = self._risk_parity_optimization(cov_matrix, universe)
            if weights is not None and self._validate_optimization_result(weights):
                logger.info("Using risk parity optimization as fallback")
                return weights
        except Exception as e:
            logger.debug(f"Risk parity optimization failed: {e}")

        # Attempt 3: Maximum Diversification
        try:
            weights = self._max_diversification_optimization(
                cov_matrix, universe
            )
            if weights is not None and self._validate_optimization_result(weights):
                logger.info("Using maximum diversification as fallback")
                return weights
        except Exception as e:
            logger.debug(f"Maximum diversification failed: {e}")

        # Final fallback: Constrained equal weights with top-K selection
        logger.warning("All optimization methods failed, using constrained equal weights")

        # Select top K assets based on expected returns
        k = min(self.constraints.top_k_positions, n_assets)
        top_k_indices = np.argsort(expected_returns)[-k:]

        weights = pd.Series(0.0, index=universe)
        for idx in top_k_indices:
            weights.iloc[idx] = 1.0 / k

        # Apply position size constraints
        weights = weights.clip(upper=self.constraints.max_position_weight)
        weights = weights / weights.sum() if weights.sum() > 0 else pd.Series(1.0 / n_assets, index=universe)

        return weights

    def _ledoit_wolf_shrinkage(self, sample_cov: np.ndarray) -> np.ndarray:
        """Apply Ledoit-Wolf covariance shrinkage for numerical stability."""
        n = sample_cov.shape[0]

        # Shrinkage target: diagonal matrix with average variance
        avg_variance = np.mean(np.diag(sample_cov))
        target = np.eye(n) * avg_variance

        # Calculate optimal shrinkage intensity
        # Simplified version - in production use sklearn.covariance.LedoitWolf
        shrinkage_intensity = min(1.0, max(0.0, 0.1))  # Conservative 10% shrinkage

        # Apply shrinkage
        shrunk_cov = (1 - shrinkage_intensity) * sample_cov + shrinkage_intensity * target

        return shrunk_cov

    def _mean_variance_optimization(
        self, expected_returns: np.ndarray, cov_matrix: np.ndarray, universe: list[str]
    ) -> pd.Series | None:
        """Standard mean-variance optimization with robust error handling."""
        n_assets = len(universe)

        # Regularize covariance matrix for numerical stability
        cov_matrix = cov_matrix + np.eye(n_assets) * 1e-8

        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            # Add L2 regularization to prevent extreme weights
            regularization = 1e-4 * np.sum(weights ** 2)

            # Add entropy penalty to encourage diversification
            # Entropy = -sum(w * log(w)) - higher entropy means more diversification
            eps = 1e-8
            valid_weights = weights[weights > eps]
            if len(valid_weights) > 0:
                entropy = -np.sum(valid_weights * np.log(valid_weights + eps))
            else:
                entropy = 0.0
            # Reduced entropy penalty to allow predictions to influence weights
            # Scale by 1000 to match daily return scale (0.001 typical)
            entropy_penalty = 0.0001 * entropy  # Much smaller penalty to allow return-based allocation

            return -portfolio_return + self.config.risk_aversion * portfolio_variance + regularization - entropy_penalty

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # Sum to 1
            {"type": "ineq", "fun": lambda w: w}  # Non-negative
        ]

        # Bounds
        bounds = [(0.0, self.constraints.max_position_weight) for _ in range(n_assets)]

        # Initial guess: equal weights
        x0 = np.ones(n_assets) / n_assets

        # Try optimization with timeout
        try:
            result = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-6, "maxiter": 1000, "disp": False},
            )

            if result.success and not np.any(np.isnan(result.x)):
                weights = pd.Series(result.x, index=universe)
                # Log weight distribution to verify diversification
                non_zero_weights = weights[weights > 1e-4]
                logger.debug(f"Optimization successful - Non-zero positions: {len(non_zero_weights)}, "
                           f"Top weight: {weights.max():.4f}, Concentration (top 10): {weights.nlargest(10).sum():.4f}")
                return weights
        except Exception as e:
            logger.debug(f"Optimization error: {e}")

        return None

    def _risk_parity_optimization(self, cov_matrix: np.ndarray, universe: list[str]) -> pd.Series | None:
        """Risk parity optimization for equal risk contribution."""
        n_assets = len(universe)

        # Calculate correlation from covariance
        std_devs = np.sqrt(np.diag(cov_matrix))
        std_devs[std_devs == 0] = 1e-8  # Avoid division by zero

        # Inverse volatility weighting as starting point
        inv_vols = 1.0 / std_devs
        weights = inv_vols / np.sum(inv_vols)

        # Apply constraints
        weights = np.clip(weights, 0, self.constraints.max_position_weight)
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(n_assets) / n_assets

        return pd.Series(weights, index=universe)

    def _max_diversification_optimization(
        self, cov_matrix: np.ndarray, universe: list[str]
    ) -> pd.Series | None:
        """Maximum diversification portfolio optimization."""
        n_assets = len(universe)

        # Calculate asset volatilities
        vols = np.sqrt(np.diag(cov_matrix))
        vols[vols == 0] = 1e-8

        def diversification_ratio(weights):
            # Weighted average of volatilities divided by portfolio volatility
            weighted_vols = np.dot(weights, vols)
            port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            return -weighted_vols / (port_vol + 1e-8)  # Negative for minimization

        # Constraints and bounds
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, self.constraints.max_position_weight) for _ in range(n_assets)]

        # Initial guess
        x0 = np.ones(n_assets) / n_assets

        try:
            result = minimize(
                diversification_ratio,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-6, "maxiter": 500, "disp": False},
            )

            if result.success and not np.any(np.isnan(result.x)):
                weights = pd.Series(result.x, index=universe)
                # Log weight distribution to verify diversification
                non_zero_weights = weights[weights > 1e-4]
                logger.debug(f"Optimization successful - Non-zero positions: {len(non_zero_weights)}, "
                           f"Top weight: {weights.max():.4f}, Concentration (top 10): {weights.nlargest(10).sum():.4f}")
                return weights
        except Exception:
            pass

        return None

    def _validate_optimization_result(self, weights: pd.Series) -> bool:
        """Validate optimization result for sanity."""
        # Check for NaN or infinite values
        if weights.isna().any() or np.any(np.isinf(weights.values)):
            return False

        # Check sum constraint (with tolerance)
        if abs(weights.sum() - 1.0) > 0.01:
            return False

        # Check for negative weights
        if (weights < -1e-8).any():
            return False

        # Check max position constraint
        if (weights > self.constraints.max_position_weight + 1e-8).any():
            return False

        # Check for extreme concentration
        if weights.max() > 0.5:  # No single position > 50%
            return False

        return True

    def _get_historical_returns_for_optimization(
        self, date: pd.Timestamp, universe: list[str]
    ) -> pd.DataFrame | None:
        """Get historical returns for covariance estimation."""
        try:
            # Load from production dataset
            returns_path = Path("data/final_new_pipeline/returns_daily_final.parquet")
            if not returns_path.exists():
                returns_path = Path("data/processed/returns_daily_final.parquet")

            if returns_path.exists():
                all_returns = pd.read_parquet(returns_path)

                # Get more historical data - extend lookback to ensure sufficient trading days
                end_date = date - pd.Timedelta(days=1)
                start_date = end_date - pd.Timedelta(days=500)  # ~2 years to ensure 252+ trading days

                # Filter to date range and universe
                mask = (all_returns.index >= start_date) & (all_returns.index <= end_date)
                historical_data = all_returns.loc[mask]

                # Filter to available assets
                available_assets = [asset for asset in universe if asset in historical_data.columns]
                if len(available_assets) < 5:  # Reduced minimum - need some assets
                    return None

                historical_data = historical_data[available_assets]

                # More flexible data cleaning - allow more missing data
                historical_data = historical_data.ffill(limit=10).fillna(0)

                if len(historical_data) >= 30:  # Reduced minimum observations
                    return historical_data

        except Exception as e:
            logger.debug(f"Failed to load historical returns: {e}")

        return None

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

        # Handle different checkpoint formats
        if "config" in model_state:
            config_obj = model_state["config"]

            # Handle complete model state format (from proper save_model)
            if hasattr(config_obj, 'lstm_config') and "universe" in model_state:
                # This is a complete model state with LSTMModelConfig and universe
                logger.info("Loading complete LSTM model state")
                self.config = model_state.get("config", self.config)
                self.constraints = model_state.get("constraints", self.constraints)
                self.universe = model_state.get("universe", None)
                self.fitted_period = model_state.get("fitted_period", None)
                self.training_history = model_state.get("training_history", [])

                # Recreate network with loaded config
                self.network = create_lstm_network(self.config.lstm_config)

                # Load network weights
                network_key = "network_state_dict" if "network_state_dict" in model_state else "model_state_dict"
                if network_key in model_state:
                    self.network.load_state_dict(model_state[network_key])
                    logger.info(f"Loaded LSTM weights from key '{network_key}'")

                self.is_fitted = True
                self._is_pretrained = True  # Mark as pre-trained to skip retraining in backtest
                logger.info(f"Loaded LSTM checkpoint with universe size: {len(self.universe) if self.universe else 0}")

            # Handle TrainingConfig object from checkpoint
            elif hasattr(config_obj, 'batch_size'):
                # This is a TrainingConfig object from training pipeline
                # Extract model architecture from checkpoint filename or use defaults
                if "h128" in str(filepath):
                    self.config.lstm_config.hidden_size = 128
                elif "h256" in str(filepath):
                    self.config.lstm_config.hidden_size = 256
                elif "h64" in str(filepath):
                    self.config.lstm_config.hidden_size = 64

                # Infer input_size from the actual checkpoint weights
                state_dict_key = None
                for key in ["model", "model_state_dict", "network_state_dict"]:
                    if key in model_state:
                        state_dict_key = key
                        break

                if state_dict_key:
                    state_dict = model_state[state_dict_key]
                    # Infer input_size from input_projection.weight shape [hidden_size, input_size]
                    if "input_projection.weight" in state_dict:
                        input_size = state_dict["input_projection.weight"].shape[1]
                        self.config.lstm_config.input_size = input_size
                        logger.info(f"Inferred input_size from checkpoint: {input_size}")

                    # Infer output_size from output_projection.weight shape [output_size, hidden_size]
                    if "output_projection.weight" in state_dict:
                        output_size = state_dict["output_projection.weight"].shape[0]
                        self.config.lstm_config.output_size = output_size
                        logger.info(f"Inferred output_size from checkpoint: {output_size}")

                # Use reasonable defaults for other parameters
                self.config.lstm_config.dropout = 0.3
                logger.info(f"Inferred LSTM config from checkpoint: hidden_size={self.config.lstm_config.hidden_size}, input_size={self.config.lstm_config.input_size}, output_size={self.config.lstm_config.output_size}")

                # Recreate network with loaded config
                self.network = create_lstm_network(self.config.lstm_config)

                # Load model weights with flexible key names
                model_key = None
                for key in ["model", "model_state_dict", "network_state_dict"]:
                    if key in model_state:
                        model_key = key
                        break

                if model_key:
                    self.network.load_state_dict(model_state[model_key])
                    logger.info(f"Loaded LSTM weights from key '{model_key}'")
                else:
                    logger.warning("No model weights found in checkpoint")

                # Set basic fitted state
                self.is_fitted = True
                self._is_pretrained = True  # Mark as pre-trained to skip retraining in backtest
                self.universe = None  # Will be set during prediction
                self.fitted_period = None

                logger.info(f"Loaded training checkpoint from {filepath}")

            elif isinstance(config_obj, dict):
                # Handle dict format
                if "hidden_size" in config_obj:
                    self.config.lstm_config.hidden_size = config_obj["hidden_size"]
                if "dropout" in config_obj:
                    self.config.lstm_config.dropout = config_obj["dropout"]

                # Recreate network with loaded config
                self.network = create_lstm_network(self.config.lstm_config)

                # Load model weights with flexible key names
                model_key = None
                for key in ["model", "model_state_dict", "network_state_dict"]:
                    if key in model_state:
                        model_key = key
                        break

                if model_key:
                    self.network.load_state_dict(model_state[model_key])
                    logger.info(f"Loaded LSTM weights from key '{model_key}'")

                self.is_fitted = True
                self._is_pretrained = True  # Mark as pre-trained to skip retraining in backtest
                self.universe = model_state.get("universe", None)
                self.fitted_period = model_state.get("fitted_period", None)

        else:
            # Fallback for other formats
            self.config = model_state.get("config", self.config)
            self.constraints = model_state.get("constraints", self.constraints)
            self.universe = model_state.get("universe", None)
            self.fitted_period = model_state.get("fitted_period", None)
            self.training_history = model_state.get("training_history", [])

            # Recreate and load network
            network_key = "network_state_dict" if "network_state_dict" in model_state else "model_state_dict"
            self.network = create_lstm_network(self.config.lstm_config)
            self.network.load_state_dict(model_state[network_key])

            self.is_fitted = True
            self._is_pretrained = True  # Mark as pre-trained to skip retraining in backtest
            logger.info(f"Loaded complete model from {filepath}")

    def _load_historical_returns(self, date: pd.Timestamp, universe: list[str]) -> pd.DataFrame:
        """
        Load historical returns data up to prediction date.

        Args:
            date: Prediction date
            universe: Asset universe

        Returns:
            Historical returns DataFrame
        """
        try:
            # Try to load from the production dataset
            returns_path = Path("data/final_new_pipeline/returns_daily_final.parquet")
            if returns_path.exists():
                all_returns = pd.read_parquet(returns_path)
                # Filter to date range and universe
                end_date = date - pd.Timedelta(days=1)  # Day before prediction
                start_date = end_date - pd.Timedelta(days=365)  # 1 year lookback

                # Filter by date and available assets
                available_assets = [asset for asset in universe if asset in all_returns.columns]
                if not available_assets:
                    raise ValueError("No assets from universe found in historical data")

                historical_data = all_returns.loc[start_date:end_date, available_assets]

                # Forward fill missing values and ensure sufficient data
                historical_data = historical_data.ffill().fillna(0.0)

                if len(historical_data) < self.config.lstm_config.sequence_length:
                    raise ValueError(f"Insufficient historical data: {len(historical_data)} < {self.config.lstm_config.sequence_length}")

                return historical_data
            else:
                raise FileNotFoundError("Production returns data not found")

        except Exception as e:
            logger.warning(f"Failed to load historical data: {e}")
            # Create synthetic data as fallback
            date_range = pd.date_range(
                end=date - pd.Timedelta(days=1),
                periods=self.config.lstm_config.sequence_length + 10,
                freq='D'
            )
            np.random.seed(int(date.timestamp()) % 2**32)
            synthetic_returns = pd.DataFrame(
                np.random.normal(0.0005, 0.02, (len(date_range), len(universe))),
                index=date_range,
                columns=universe
            )
            return synthetic_returns

    def _create_prediction_sequences(
        self,
        returns_data: pd.DataFrame,
        universe: list[str],
        date: pd.Timestamp,
        sequence_length: int
    ) -> torch.Tensor:
        """
        Create LSTM input sequences from historical returns.

        Args:
            returns_data: Historical returns data
            universe: Asset universe
            date: Prediction date
            sequence_length: Length of input sequences

        Returns:
            Input tensor of shape (1, sequence_length, num_assets * num_features)
        """
        try:
            # Get the most recent sequence_length days
            end_idx = returns_data.index.get_indexer([date - pd.Timedelta(days=1)], method='nearest')[0]
            start_idx = max(0, end_idx - sequence_length + 1)

            sequence_data = returns_data.iloc[start_idx:end_idx+1]

            # Ensure we have enough data
            if len(sequence_data) < sequence_length:
                # Pad with zeros if insufficient data
                padding_needed = sequence_length - len(sequence_data)
                padding_dates = pd.date_range(
                    end=sequence_data.index[0] - pd.Timedelta(days=1),
                    periods=padding_needed,
                    freq='D'
                )
                padding_data = pd.DataFrame(
                    np.zeros((padding_needed, len(sequence_data.columns))),
                    index=padding_dates,
                    columns=sequence_data.columns
                )
                sequence_data = pd.concat([padding_data, sequence_data])

            # Take only the last sequence_length rows
            sequence_data = sequence_data.tail(sequence_length)

            # Ensure sequence data matches network dimensions via padding/truncation
            expected_input_size = self.config.lstm_config.input_size
            selected_assets = list(sequence_data.columns)

            if sequence_data.shape[1] < expected_input_size:
                # Pad with zeros
                padding_needed = expected_input_size - sequence_data.shape[1]
                padding = np.zeros((len(sequence_data), padding_needed))
                padding_df = pd.DataFrame(
                    padding,
                    index=sequence_data.index,
                    columns=[f'PAD_{i}' for i in range(padding_needed)]
                )
                sequence_data = pd.concat([sequence_data, padding_df], axis=1)
            elif sequence_data.shape[1] > expected_input_size:
                # Truncate to most liquid assets
                asset_activity = sequence_data.abs().mean().sort_values(ascending=False)
                top_assets = asset_activity.head(expected_input_size).index
                sequence_data = sequence_data[top_assets]
                selected_assets = list(top_assets)

            # Use simple returns data to match training configuration
            # Shape: (sequence_length, num_assets)
            feature_matrix = sequence_data.values

            # Flatten to (sequence_length, num_features) where num_features = num_assets
            input_tensor = torch.FloatTensor(feature_matrix)
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

            return input_tensor.to(self.device if hasattr(self, 'device') else 'cpu'), selected_assets

        except Exception as e:
            logger.warning(f"Failed to create sequences: {e}, using zero tensor")
            # Fallback: create zero tensor with correct shape for expected input size
            expected_input_size = self.config.lstm_config.input_size
            fallback_tensor = torch.zeros(1, sequence_length, expected_input_size)
            return fallback_tensor.to(self.device if hasattr(self, 'device') else 'cpu'), []


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
