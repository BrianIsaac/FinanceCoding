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
from .architecture import LSTMConfig
from .ragged_architecture import RaggedLSTMNetwork, create_ragged_lstm_network
from .ragged_utils import compute_sequence_statistics
from .training import MemoryEfficientTrainer, TrainingConfig, create_trainer

logger = logging.getLogger(__name__)

# Import NA handling for robust data filtering
try:
    from ...data.na_handling import (
        prepare_rolling_window_data,
        filter_zero_variance_assets,
    )
    NA_HANDLING_AVAILABLE = True
except ImportError:
    logger.warning("NA handling utilities not available, using basic filtering")
    NA_HANDLING_AVAILABLE = False

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
    portfolio_temperature: float = 3.0  # Temperature for softmax conversion of returns to weights
    top_k_assets: int = 0  # Select top K assets by prediction, then softmax (0 = no filtering)
    enforce_constraints: bool = True  # Enforce portfolio constraints (False = softmax naturally sums to 1)

    # Feature engineering
    use_technical_features: bool = True  # Use technical features instead of just returns
    feature_set: str = "standard"  # Options: "minimal" (7 features), "standard" (9), "full" (12)

    @property
    def actual_input_size(self) -> int:
        """Calculate input size based on feature configuration."""
        if not self.use_technical_features:
            return 1  # Just returns

        feature_counts = {
            "minimal": 7,
            "standard": 9,
            "full": 12,
        }
        return feature_counts.get(self.feature_set, 1)

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
            "portfolio_temperature": self.portfolio_temperature,
            "top_k_assets": self.top_k_assets,
            "enforce_constraints": self.enforce_constraints,
            "use_technical_features": self.use_technical_features,
            "feature_set": self.feature_set,
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
        self.network: RaggedLSTMNetwork | None = None
        self.trainer: MemoryEfficientTrainer | None = None
        self.universe: list[str] | None = None
        self.training_history: dict | None = None

        # CRITICAL: Expose enforce_constraints as direct attribute for rolling_engine
        # rolling_engine checks model.enforce_constraints not model.config.enforce_constraints
        self.enforce_constraints = self.config.enforce_constraints

        # CRITICAL FIX: Track device for network recreation
        # Auto-detect device (will be updated when trainer is created)
        import torch
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        lookback_months: int = 30,  # FIXED: Reduced from 36 to 30 months
        min_observations: int = 100,  # Reduced default for flexible academic framework
        max_epochs: int = 20,  # Maximum epochs for quick retraining
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
        training_data, valid_mask, sequence_lengths = self._load_fresh_returns_data(
            returns, start_date, end_date, universe
        )

        logger.debug(
            f"Loaded sequence_lengths distribution: min={sequence_lengths.min()}, "
            f"max={sequence_lengths.max()}, mean={sequence_lengths.mean():.1f}, "
            f"n_assets={len(sequence_lengths)}"
        )

        if len(training_data) < min_observations:
            raise ValueError(
                f"Insufficient data for rolling fit: {len(training_data)} < {min_observations}"
            )

        # Store valid mask and sequence lengths for later use in prediction
        self._last_valid_mask = valid_mask
        self._sequence_lengths = sequence_lengths

        # Quick retrain with limited epochs
        self._quick_retrain(training_data, universe, max_epochs=max_epochs)

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
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Load sufficient data for LSTM sequences and prediction with validity mask and sequence lengths.

        Args:
            returns: Full historical returns or path to data
            start_date: Start of training window
            end_date: End of training window
            universe: Assets to include

        Returns:
            Tuple of (cleaned_returns, valid_mask, sequence_lengths) where:
            - cleaned_returns: Imputed returns DataFrame
            - valid_mask: Boolean Series of assets meeting quality criteria
            - sequence_lengths: Series with actual sequence length per asset
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

        # Use unified NA handling pipeline
        from ...data.na_handling import (
            prepare_rolling_window_data,
            simple_temporal_fill,
            calculate_data_quality_metrics,
        )

        logger.info(
            f"LSTM model NA handling: coverage_threshold=0.80, "
            f"variance_threshold=1e-8, n_assets_before_filter={len(available_assets)}, "
            f"fit_period={start_date.date()} to {end_date.date()}"
        )

        prepared_returns, masks = prepare_rolling_window_data(
            returns_window=filtered_returns,
            universe=available_assets,
            coverage_threshold=0.80,  # Standardised: 80% coverage threshold (industry standard, ensures fair model comparison)
            variance_threshold=1e-8,
            return_masks=True,
        )

        # Compute sequence lengths (count non-NA values per asset)
        # CRITICAL FIX: Clamp to max_seq_len to prevent metadata mismatch after windowing
        max_seq_len = self.config.lstm_config.sequence_length
        sequence_lengths = (~prepared_returns.isna()).sum(axis=0).clip(upper=max_seq_len)

        logger.debug(
            f"Sequence lengths after clamping: min={sequence_lengths.min()}, "
            f"max={sequence_lengths.max()}, mean={sequence_lengths.mean():.1f}, "
            f"max_seq_len={max_seq_len}"
        )

        # Use simple temporal fill (training: allow bfill for complete historical window)
        cleaned_returns = simple_temporal_fill(prepared_returns, allow_bfill=True)

        # Store quality metrics for backtest tracking
        self._last_data_quality_metrics = calculate_data_quality_metrics(
            prepared_returns,
            available_assets,
            masks,
            date=end_date,  # Pass date for membership-aware verification
        )

        return cleaned_returns, masks['valid'], sequence_lengths

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
                # Too much padding would be needed - signal that network recreation is needed
                # This prevents shape mismatch errors from undersized data
                raise ValueError(
                    f"Padding would exceed 10% threshold (would need {padding_needed/target_size:.1%} padding "
                    f"to go from {current_size} to {target_size}). Network recreation required for input_size={current_size}."
                )

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
        # CRITICAL FIX: Filter zero-variance assets BEFORE sizing network
        # This prevents shape mismatch: network sized for N, receives N-M after variance filtering
        # Variance filtering happens in training.py:271-293, must also happen here before sizing
        std_threshold = 1e-5
        asset_stds = training_data.std()
        valid_mask = asset_stds >= std_threshold
        num_filtered = (~valid_mask).sum()

        if num_filtered > 0:
            logger.info(f"Pre-sizing variance filter: Removing {num_filtered} assets with std < {std_threshold:.2e}")
            training_data = training_data.loc[:, valid_mask]
            universe = [u for u, valid in zip(universe, valid_mask) if valid]
            logger.info(f"After pre-sizing filter: {len(training_data.columns)} assets remain")

        # Use dynamic input sizing for better efficiency and training stability
        # CRITICAL FIX: Use actual available assets in training_data, not desired universe
        # This prevents shape mismatch errors when available assets < desired universe
        # NOW sizing on CLEANED data (after variance filtering)
        n_assets = len(training_data.columns)

        # CRITICAL FIX: Calculate input_size and output_size correctly for multi-feature case
        # When technical features are enabled:
        #   - Features are flattened: [T, N, F] → [T, N*F]
        #   - input_size = N*F (LSTM learns from all flattened features)
        #   - output_size = N (predict one return per asset for portfolio weights)
        # When features disabled (returns only):
        #   - input_size = N
        #   - output_size = N
        if self.config.use_technical_features and hasattr(self, '_feature_names'):
            n_features_per_asset = len(self._feature_names)
            current_universe_size = n_assets * n_features_per_asset
            logger.info(f"Multi-feature mode: {n_assets} assets × {n_features_per_asset} features = {current_universe_size} input dimensions")
        else:
            current_universe_size = n_assets
            logger.info(f"Returns-only mode: {n_assets} assets = {current_universe_size} input dimensions")

        # Apply min/max constraints for stability
        min_size = getattr(self.config.lstm_config, 'min_input_size', 50)
        max_size = getattr(self.config.lstm_config, 'max_input_size', 700)
        optimal_input_size = max(min_size, min(current_universe_size, max_size))
        optimal_output_size = n_assets  # Always predict one return per asset

        # CRITICAL FIX: Network persistence - avoid recreation for small size changes
        # Only recreate if network doesn't exist OR size change is >10%
        network_recreated = False

        if self.network is None:
            # First time: create network
            self.config.lstm_config.input_size = optimal_input_size
            self.config.lstm_config.output_size = optimal_output_size
            self.network = create_ragged_lstm_network(self.config.lstm_config)

            # CRITICAL FIX: Immediately move network to correct device
            if hasattr(self, '_device'):
                self.network = self.network.to(self._device)
                # Ensure ALL buffers are on the correct device
                for name, buffer in self.network.named_buffers():
                    if buffer.device != self._device:
                        buffer.data = buffer.data.to(self._device)
                logger.debug(f"Moved recreated network and all buffers to {self._device}")

            network_recreated = True
            logger.info(f"Created LSTM network with input_size={optimal_input_size}, output_size={optimal_output_size} for {n_assets} assets")

        elif abs(self.config.lstm_config.input_size - optimal_input_size) / self.config.lstm_config.input_size > 0.10:
            # Significant size change (>10%): recreate network
            old_input_size = self.config.lstm_config.input_size
            old_output_size = self.config.lstm_config.output_size
            self.config.lstm_config.input_size = optimal_input_size
            self.config.lstm_config.output_size = optimal_output_size
            self.network = create_ragged_lstm_network(self.config.lstm_config)

            # Move to device
            if hasattr(self, '_device'):
                self.network = self.network.to(self._device)
                for name, buffer in self.network.named_buffers():
                    if buffer.device != self._device:
                        buffer.data = buffer.data.to(self._device)

            network_recreated = True
            logger.warning(
                f"Recreated LSTM network due to large size change: "
                f"input {old_input_size} → {optimal_input_size} ({abs(optimal_input_size - old_input_size) / old_input_size:.1%}), "
                f"output {old_output_size} → {optimal_output_size}"
            )
        else:
            # Small size change (<10%): keep network, pad data instead
            # This enables transfer learning between windows
            current_network_size = self.config.lstm_config.input_size
            if current_network_size != optimal_input_size:
                logger.info(
                    f"Network persistence: keeping network size={current_network_size}, "
                    f"padding data from {optimal_input_size} (size change: "
                    f"{abs(optimal_input_size - current_network_size) / current_network_size:.1%})"
                )
            # Adjust optimal_input_size to match network (data will be padded)
            optimal_input_size = current_network_size

        # Check if we have sufficient data for training
        min_required_samples = self.config.lstm_config.sequence_length + 21  # sequence_length + prediction_horizon
        if len(training_data) < min_required_samples:
            logger.warning(f"Insufficient data for retraining: {len(training_data)} < {min_required_samples} required samples")
            # Keep existing weights if any, or initialize random weights
            if self.network is None:
                logger.info("Initializing network with random weights due to insufficient training data")
                self.network = create_ragged_lstm_network(self.config.lstm_config)
                self.is_fitted = True  # Mark as fitted to allow predictions with random weights
            return

        # Adjust training data to match optimal dimensions (minimal padding)
        # SKIP padding adjustment when using technical features - data will be flattened later
        if not (self.config.use_technical_features and hasattr(self, '_feature_names')):
            # Only adjust data size when NOT using technical features
            try:
                training_data = self._adjust_data_to_optimal_size(training_data, universe, optimal_input_size)
            except ValueError as e:
                if "Padding would exceed" in str(e):
                    # Padding exceeded threshold - recreate network with correct size
                    logger.info(f"Recreating network due to padding limit: {e}")
                    optimal_input_size = len(training_data.columns)
                    self.config.lstm_config.input_size = optimal_input_size
                    self.config.lstm_config.output_size = n_assets  # Still output one per asset
                    self.network = create_ragged_lstm_network(self.config.lstm_config)

                    # Move to device
                    if hasattr(self, '_device'):
                        self.network = self.network.to(self._device)
                        for name, buffer in self.network.named_buffers():
                            if buffer.device != self._device:
                                buffer.data = buffer.data.to(self._device)

                    network_recreated = True
                    logger.info(f"Network recreated with input_size={optimal_input_size}, output_size={n_assets}")
                    # No padding needed now since network matches data
                else:
                    raise
        else:
            logger.info(f"Skipping data padding for technical features - data will be flattened to {optimal_input_size} dimensions after feature extraction")

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
        # CRITICAL FIX: Always recreate trainer if network was recreated
        # This ensures optimizer, scaler, and criterion are initialized with correct device
        if self.trainer is None or network_recreated:
            # Create new trainer with confidence-adjusted epochs
            quick_config = TrainingConfig(
                epochs=adjusted_params.get("epochs", max_epochs),
                patience=5,  # Reduced patience for quick training
                batch_size=self.config.training_config.batch_size,
                learning_rate=adjusted_params.get("learning_rate", self.config.training_config.learning_rate * 0.1),  # Use adjusted LR or default lower for fine-tuning
                weight_decay=adjusted_params.get("weight_decay", 0.001),
                use_mixed_precision=self.config.training_config.use_mixed_precision,
            )

            # Clean up old trainer if it exists
            if self.trainer is not None:
                logger.debug("Recreating trainer due to network architecture change")
                # Clear optimizer state
                if hasattr(self.trainer, 'optimizer'):
                    self.trainer.optimizer.zero_grad()
                # Clear CUDA cache
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self.trainer = create_trainer(self.network, quick_config)

            # CRITICAL FIX: Update device reference from trainer for network recreation
            self._device = self.trainer.device
            logger.info(f"Created new trainer for network with input_size={self.network.config.input_size} on device {self._device}")
        else:
            # Update existing trainer config only (network reference stays the same)
            self.trainer.config.epochs = max_epochs
            self.trainer.config.patience = 5
            logger.debug(
                f"Updated trainer config for network with input_size={self.network.config.input_size}"
            )

        # Perform quick training
        try:
            self.training_history = self.trainer.fit(
                training_data,
                sequence_length=self.config.lstm_config.sequence_length,
                checkpoint_dir=None,  # Don't save checkpoints for quick retraining
            )
        except Exception as e:
            logger.error(f"Quick retrain failed with {type(e).__name__}: {e}", exc_info=True)
            logger.warning("Keeping existing weights due to training failure")

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

        # Apply feature engineering if enabled
        if self.config.use_technical_features:
            logger.info(f"Extracting technical features using '{self.config.feature_set}' feature set")
            from src.features.technical_features import create_feature_extractor

            # Convert returns to prices for feature extraction
            prices = self._returns_to_prices(training_data)

            logger.debug(
                f"Shape tracking: training_data (returns)={training_data.shape}, "
                f"prices={prices.shape}"
            )

            # Extract features
            feature_extractor = create_feature_extractor(self.config.feature_set)
            logger.debug(
                f"Calling extract_features: prices_shape={prices.shape}, "
                f"returns_shape={training_data.shape}"
            )
            features_array, feature_names = feature_extractor.extract_features(
                prices=prices,
                returns=training_data,
                benchmark_prices=None  # TODO: Add SPY prices if available
            )

            # Log feature extraction results
            logger.info(
                f"Extracted {len(feature_names)} features: {feature_names} "
                f"with shape {features_array.shape}"
            )
            logger.debug(
                f"Feature array validation: shape={features_array.shape}, "
                f"dtype={features_array.dtype}, "
                f"has_nan={np.isnan(features_array).any()}, "
                f"n_features={len(feature_names)}, "
                f"expected_n_features={self.config.actual_input_size}"
            )

            # Store feature extractor for inference
            self._feature_extractor = feature_extractor
            self._feature_names = feature_names
            logger.info(
                f"Stored feature extractor (ID: {id(feature_extractor)}) "
                f"with {len(feature_names)} features: {feature_names}"
            )

            # CRITICAL FIX (Bug #5): Don't set input_size here yet
            # Features are reshaped to [T, N*F] at line 825, so input_size = N*F, not F
            # The network sizing logic below (lines 768-774) will set input_size correctly
            # based on the actual flattened dimension (optimal_size)
            logger.info(f"Feature extraction complete: {len(feature_names)} features per asset, will be flattened to N*{len(feature_names)}")
        else:
            # Single feature (returns only)
            self.config.lstm_config.input_size = 1
            self._feature_extractor = None
            self._feature_names = ["returns"]

        # CRITICAL FIX: Filter zero-variance assets BEFORE sizing network
        # This prevents shape mismatch: network sized for N, receives N-M after variance filtering
        # Variance filtering happens in training.py:271-293, must also happen here before sizing
        std_threshold = 1e-5
        asset_stds = training_data.std()
        valid_mask = asset_stds >= std_threshold
        num_filtered = (~valid_mask).sum()

        if num_filtered > 0:
            logger.info(f"Pre-sizing variance filter: Removing {num_filtered} assets with std < {std_threshold:.2e}")
            training_data = training_data.loc[:, valid_mask]
            universe = [u for u, valid in zip(universe, valid_mask) if valid]
            logger.info(f"After pre-sizing filter: {len(training_data.columns)} assets remain")

        # Use dynamic input sizing for better efficiency and training stability
        # CRITICAL FIX: Use actual available assets in training_data, not desired universe
        # This prevents shape mismatch errors when available assets < desired universe
        # NOW sizing on CLEANED data (after variance filtering)
        n_assets = len(training_data.columns)

        # CRITICAL FIX: Calculate input_size and output_size correctly for multi-feature case
        # When technical features are enabled:
        #   - Features are flattened: [T, N, F] → [T, N*F]
        #   - input_size = N*F (LSTM learns from all flattened features)
        #   - output_size = N (predict one return per asset for portfolio weights)
        # When features disabled (returns only):
        #   - input_size = N
        #   - output_size = N
        if self.config.use_technical_features and hasattr(self, '_feature_names'):
            n_features_per_asset = len(self._feature_names)
            current_universe_size = n_assets * n_features_per_asset
            logger.info(f"Multi-feature mode: {n_assets} assets × {n_features_per_asset} features = {current_universe_size} input dimensions")
        else:
            current_universe_size = n_assets
            logger.info(f"Returns-only mode: {n_assets} assets = {current_universe_size} input dimensions")

        # Apply min/max constraints for stability
        min_size = getattr(self.config.lstm_config, 'min_input_size', 50)
        max_size = getattr(self.config.lstm_config, 'max_input_size', 700)
        optimal_input_size = max(min_size, min(current_universe_size, max_size))
        optimal_output_size = n_assets  # Always predict one return per asset

        # CRITICAL FIX: Network persistence - reuse network if size is similar
        # Only recreate if network doesn't exist OR size change is >10%
        if self.network is None:
            # First time: create network
            self.config.lstm_config.input_size = optimal_input_size
            self.config.lstm_config.output_size = optimal_output_size
            logger.info(f"Creating LSTM network with input_size={optimal_input_size}, output_size={optimal_output_size} for {n_assets} assets")
            self.network = create_ragged_lstm_network(self.config.lstm_config)
        elif abs(self.config.lstm_config.input_size - optimal_input_size) / self.config.lstm_config.input_size > 0.10:
            # Significant size change (>10%): recreate network
            old_input_size = self.config.lstm_config.input_size
            old_output_size = self.config.lstm_config.output_size
            self.config.lstm_config.input_size = optimal_input_size
            self.config.lstm_config.output_size = optimal_output_size
            logger.warning(
                f"Recreating LSTM network due to large size change: "
                f"input {old_input_size} → {optimal_input_size} ({abs(optimal_input_size - old_input_size) / old_input_size:.1%}), "
                f"output {old_output_size} → {optimal_output_size}"
            )
            self.network = create_ragged_lstm_network(self.config.lstm_config)
        else:
            # Small size change (<10%): keep network for transfer learning
            logger.info(
                f"Network persistence: reusing existing network with input_size={self.config.lstm_config.input_size} "
                f"(requested size: {optimal_input_size}, change: {abs(optimal_input_size - self.config.lstm_config.input_size) / self.config.lstm_config.input_size:.1%})"
            )
            optimal_input_size = self.config.lstm_config.input_size  # Use existing network size

        # Adjust training data to match optimal dimensions (minimal padding)
        # SKIP padding adjustment when using technical features - data will be flattened later
        if not (self.config.use_technical_features and hasattr(self, '_feature_names')):
            # Only adjust data size when NOT using technical features
            try:
                training_data = self._adjust_data_to_optimal_size(training_data, universe, optimal_input_size)
            except ValueError as e:
                if "Padding would exceed" in str(e):
                    # Padding exceeded threshold - recreate network with correct size
                    logger.info(f"Recreating network due to padding limit: {e}")
                    optimal_input_size = len(training_data.columns)
                    self.config.lstm_config.input_size = optimal_input_size
                    self.config.lstm_config.output_size = n_assets  # Still output one per asset
                    self.network = create_ragged_lstm_network(self.config.lstm_config)
                    logger.info(f"Network recreated with input_size={optimal_input_size}, output_size={n_assets}")
                    # No padding needed now since network matches data
                else:
                    raise
        else:
            logger.info(f"Skipping data padding for technical features - data will be flattened to {optimal_input_size} dimensions after feature extraction")

        # Create trainer
        self.trainer = create_trainer(self.network, self.config.training_config)

        # CRITICAL FIX: Update device reference from trainer for network recreation
        self._device = self.trainer.device

        # Train model with appropriate data format
        if self.config.use_technical_features and hasattr(self, '_feature_extractor'):
            # Use extracted features for training
            # Note: features_array was already extracted above but we need to handle
            # the variance-filtered data properly
            if 'features_array' in locals():
                # FIX: Wrap numpy array back into DataFrame for trainer compatibility
                # Reshape from [T, N, F] to [T, N*F] and create proper column names
                T, N, F = features_array.shape
                logger.info(f"Wrapping features_array for trainer: shape={features_array.shape} (T={T}, N={N}, F={F})")

                # Reshape to [T, N*F]
                features_reshaped = features_array.reshape(T, N * F)

                # Create column names: asset_name + feature_name
                asset_names = training_data.columns.tolist()
                feature_names = self._feature_names
                column_names = [f"{asset}_{feature}" for asset in asset_names for feature in feature_names]

                # Wrap into DataFrame with proper index and columns
                features_df = pd.DataFrame(
                    features_reshaped,
                    index=training_data.index,
                    columns=column_names
                )
                logger.info(f"Wrapped features shape: {features_df.shape}, index range: {features_df.index[0]} to {features_df.index[-1]}")

                self.training_history = self.trainer.fit(
                    features_df,
                    sequence_length=self.config.lstm_config.sequence_length,
                    checkpoint_dir=checkpoint_dir,
                )
            else:
                # Re-extract for filtered data if needed
                prices = self._returns_to_prices(training_data)
                features_array, feature_names = self._feature_extractor.extract_features(
                    prices=prices,
                    returns=training_data,
                    benchmark_prices=None
                )

                # FIX: Wrap numpy array back into DataFrame for trainer compatibility
                T, N, F = features_array.shape
                logger.info(f"Wrapping re-extracted features: shape={features_array.shape} (T={T}, N={N}, F={F})")

                # Reshape to [T, N*F]
                features_reshaped = features_array.reshape(T, N * F)

                # Create column names: asset_name + feature_name
                asset_names = training_data.columns.tolist()
                column_names = [f"{asset}_{feature}" for asset in asset_names for feature in feature_names]

                # Wrap into DataFrame with proper index and columns
                features_df = pd.DataFrame(
                    features_reshaped,
                    index=training_data.index,
                    columns=column_names
                )
                logger.info(f"Wrapped features shape: {features_df.shape}, index range: {features_df.index[0]} to {features_df.index[-1]}")

                self.training_history = self.trainer.fit(
                    features_df,
                    sequence_length=self.config.lstm_config.sequence_length,
                    checkpoint_dir=checkpoint_dir,
                )
        else:
            # Use raw returns for training (single feature)
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

        Note:
            During inference, the model processes a single batch containing all assets.
            The lengths tensor has shape (1,) indicating the number of valid timesteps
            in the sequence. We use the minimum sequence length across all selected
            assets to ensure all data at each timestep is valid.
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
        predicted_returns_array = self._predict_returns(date, prediction_universe)

        # Apply Markowitz optimization if enabled
        if self.config.use_markowitz_layer:
            available_weights = self._optimize_portfolio(predicted_returns_array, prediction_universe, date)
        else:
            # Convert predicted returns to weights using softmax (preserves ranking)
            # CRITICAL FIX: Softmax handles both positive and negative returns correctly
            # Previous bug: clipping negative predictions to 0 destroyed 50% of signal
            temperature = self.config.portfolio_temperature

            # Top-K asset selection (if configured)
            if self.config.top_k_assets > 0 and len(prediction_universe) > self.config.top_k_assets:
                # Convert to pandas Series for easier manipulation
                predicted_returns_series = pd.Series(predicted_returns_array, index=prediction_universe)

                # Select top K assets by predicted returns
                top_k = min(self.config.top_k_assets, len(prediction_universe))
                top_k_series = predicted_returns_series.nlargest(top_k)

                logger.info(
                    f"Top-K selection: Selected {top_k} assets from {len(prediction_universe)} "
                    f"(top prediction: {top_k_series.max():.4f}, "
                    f"bottom of top-K: {top_k_series.min():.4f})"
                )

                # Use filtered predictions and universe
                predicted_returns_array = top_k_series.values
                prediction_universe = top_k_series.index.tolist()

            # Normalize for numerical stability
            pred_mean = predicted_returns_array.mean()
            pred_std = predicted_returns_array.std()
            if pred_std < 1e-8:
                # If all predictions are identical, use equal weights
                available_weights = pd.Series(1.0 / len(prediction_universe), index=prediction_universe)
            else:
                pred_normalized = (predicted_returns_array - pred_mean) / (pred_std + 1e-8)

                # Apply softmax to convert returns to weights
                exp_returns = np.exp(pred_normalized / temperature)
                available_weights = pd.Series(exp_returns / exp_returns.sum(), index=prediction_universe)

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

        # Apply portfolio constraints (optional for softmax path)
        if self.config.enforce_constraints:
            weights = self.validate_weights(weights)
            logger.info(f"Applied portfolio constraints (enforce_constraints=True)")
        else:
            logger.info(f"Skipped constraint enforcement (enforce_constraints=False) - softmax weights sum to 1 naturally")

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

            # Prepare lengths for batch (single sequence during inference)
            # Shape must be (batch_size,) = (1,) for inference
            if hasattr(self, '_sequence_lengths'):
                # Validate selected_assets before min() call
                if not selected_assets:
                    logger.error(
                        f"PREDICTION FAILED: selected_assets is empty. "
                        f"Input sequence creation returned no valid assets. "
                        f"Universe size: {len(universe)}, "
                        f"Network input_size: {self.network.config.input_size}"
                    )
                    raise ValueError("Cannot generate predictions with empty selected_assets list")

                # Use the minimum sequence length across selected assets
                # This ensures all assets have valid data for this many timesteps
                min_length = min(
                    self._sequence_lengths.loc[asset] if asset in self._sequence_lengths.index
                    else self.config.lstm_config.sequence_length
                    for asset in selected_assets
                )

                logger.debug(
                    f"Min_length calculation: min_length={min_length}, "
                    f"max_seq_len={self.config.lstm_config.sequence_length}, "
                    f"n_selected_assets={len(selected_assets)}"
                )
                pred_lengths = torch.tensor(
                    [min_length],  # Single value for batch_size=1
                    dtype=torch.long,
                    device=device
                )
            else:
                # Fallback: assume full length
                pred_lengths = torch.tensor(
                    [self.config.lstm_config.sequence_length],  # Shape is (1,)
                    dtype=torch.long,
                    device=device
                )

            # Run forward pass through trained network with ragged tensors
            self.network.eval()
            with torch.no_grad():
                predictions, _ = self.network(input_sequences, pred_lengths)
                # Extract predictions for the selected assets (on normalized scale)
                predicted_returns_normalized = predictions.cpu().numpy().flatten()

                # CRITICAL FIX: Denormalize predictions back to actual return scale
                # Network outputs normalized predictions, but portfolio optimization needs actual return scale
                if hasattr(self.network, 'normalization_stats'):
                    stats = self.network.normalization_stats
                    scaler = stats.get('scaler', None)

                    if scaler is not None:
                        # Use MinMaxScaler inverse_transform
                        try:
                            # Reshape for scaler: (n_assets,) -> (1, n_assets)
                            predicted_returns_raw = scaler.inverse_transform(
                                predicted_returns_normalized.reshape(1, -1)
                            ).flatten()

                            logger.debug(
                                f"Denormalized predictions with MinMaxScaler: "
                                f"normalized_range=[{predicted_returns_normalized.min():.6f}, {predicted_returns_normalized.max():.6f}], "
                                f"denormalized_range=[{predicted_returns_raw.min():.6f}, {predicted_returns_raw.max():.6f}]"
                            )
                        except Exception as e:
                            logger.warning(f"MinMaxScaler inverse_transform failed: {e}, using normalized predictions")
                            predicted_returns_raw = predicted_returns_normalized
                    elif 'mean' in stats and 'std' in stats:
                        # Legacy RobustScaler approach
                        mean = stats['mean'].flatten()  # [n_assets]
                        std = stats['std'].flatten()    # [n_assets]
                        epsilon = stats.get('epsilon', 1e-6)

                        # Denormalize: reverse the z-score normalization
                        n_predictions = len(predicted_returns_normalized)
                        if n_predictions <= len(mean):
                            # Use first n_predictions stats
                            predicted_returns_raw = (predicted_returns_normalized * (std[:n_predictions] + epsilon) +
                                                     mean[:n_predictions])
                        else:
                            # More predictions than stats - pad with mean/std
                            predicted_returns_raw = predicted_returns_normalized.copy()
                            predicted_returns_raw[:len(mean)] = (predicted_returns_normalized[:len(mean)] *
                                                                 (std + epsilon) + mean)

                        logger.debug(
                            f"Legacy denormalized predictions: "
                            f"normalized mean={predicted_returns_normalized.mean():.6f}, "
                        f"std={predicted_returns_normalized.std():.6f} -> "
                        f"actual mean={predicted_returns_raw.mean():.6f}, "
                        f"std={predicted_returns_raw.std():.6f}"
                    )
                else:
                    # CRITICAL FIX: Fail fast instead of silently continuing with wrong scale
                    raise RuntimeError(
                        "PREDICTION FAILED: No normalization stats available for denormalization. "
                        f"Network was trained with normalization but stats are missing. "
                        f"Predictions are on normalized scale (std≈1.0) instead of actual scale (std≈0.01-0.02). "
                        f"Portfolio optimization requires actual scale returns. "
                        f"This indicates a training or model loading error. "
                        f"Check that normalization_stats are properly stored during training."
                    )

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
                # CRITICAL FIX: Track available assets to align covariance with expected returns
                # Historical returns may be filtered to available assets in _get_historical_returns_for_optimization
                # This creates shape mismatch: cov_matrix is (M, M) but expected_returns is (N,) where M < N
                available_assets = list(historical_returns.columns)
                n_available = len(available_assets)

                # Create mapping from universe to available assets
                asset_to_idx = {asset: idx for idx, asset in enumerate(universe)}
                available_indices = [asset_to_idx[asset] for asset in available_assets if asset in asset_to_idx]

                # Filter expected returns to match available assets
                filtered_expected_returns = expected_returns[available_indices]

                logger.debug(
                    f"Covariance alignment: {n_assets} universe assets -> {n_available} available assets "
                    f"({n_available/n_assets*100:.1f}% coverage)"
                )

                # Calculate empirical covariance with proper handling
                returns_matrix = historical_returns.values

                # Center the returns
                centered_returns = returns_matrix - np.mean(returns_matrix, axis=0)

                # CRITICAL FIX: Use robust covariance estimation (same as HRP/GAT)
                # This provides optimal Ledoit-Wolf shrinkage, minimum variance floor, and PSD enforcement
                from ...data.processors.covariance import robust_covariance

                cov_matrix = robust_covariance(
                    data=centered_returns,
                    method="lw",  # Ledoit-Wolf with optimal shrinkage
                    shrink_to="diag",  # Shrink to diagonal matrix
                    min_var=1e-6,  # Minimum variance floor (prevents zero variance)
                )

                # Check for numerical issues
                if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
                    logger.warning("Invalid covariance matrix after robust estimation, using identity matrix")
                    cov_matrix = np.eye(n_available) * 0.04  # 20% annual volatility squared
                else:
                    # Log covariance quality metrics
                    eigenvalues = np.linalg.eigvalsh(cov_matrix)
                    condition_number = eigenvalues.max() / (eigenvalues.min() + 1e-16)
                    logger.debug(
                        f"Covariance matrix: shape={cov_matrix.shape}, "
                        f"condition_number={condition_number:.2e}, "
                        f"min_eig={eigenvalues.min():.2e}, max_eig={eigenvalues.max():.2e}"
                    )
            else:
                # Fallback: Use diagonal covariance based on individual volatilities
                logger.warning("Insufficient data for covariance estimation, using diagonal matrix")
                available_assets = universe
                filtered_expected_returns = expected_returns
                n_available = n_assets
                individual_vols = np.full(n_available, 0.20)  # 20% annual volatility
                cov_matrix = np.diag(individual_vols ** 2)

        except Exception as e:
            logger.warning(f"Failed to estimate covariance: {e}, using diagonal fallback")
            available_assets = universe
            filtered_expected_returns = expected_returns
            n_available = n_assets
            cov_matrix = np.eye(n_available) * 0.04

        # Multiple optimization attempts with different methods
        # Use filtered universe and expected returns to match covariance matrix shape
        weights = None

        # Attempt 1: Standard Mean-Variance Optimization
        try:
            logger.debug("Attempting mean-variance optimization")
            weights = self._mean_variance_optimization(
                filtered_expected_returns, cov_matrix, available_assets
            )
            if weights is not None and self._validate_optimization_result(weights):
                # Log success metrics
                self._log_weight_diagnostics(weights, "Mean-Variance")
                # Expand weights to full universe
                final_weights = self._expand_weights_to_universe(weights, universe, available_assets)
                logger.info(f"LSTM optimization: Mean-variance succeeded with {(final_weights > 1e-6).sum()} non-zero positions")
                return final_weights
            else:
                logger.debug(f"Mean-variance optimization produced invalid weights")
        except Exception as e:
            logger.warning(f"Mean-variance optimization failed: {type(e).__name__}: {str(e)[:200]}")

        # Attempt 2: Risk Parity Optimization
        try:
            logger.debug("Attempting risk parity optimization")
            weights = self._risk_parity_optimization(cov_matrix, available_assets)
            if weights is not None and self._validate_optimization_result(weights):
                # Log success metrics
                self._log_weight_diagnostics(weights, "Risk Parity")
                logger.info("Using risk parity optimization as fallback")
                # Expand weights to full universe
                final_weights = self._expand_weights_to_universe(weights, universe, available_assets)
                logger.info(f"LSTM optimization: Risk parity succeeded with {(final_weights > 1e-6).sum()} non-zero positions")
                return final_weights
            else:
                logger.debug(f"Risk parity optimization produced invalid weights")
        except Exception as e:
            logger.warning(f"Risk parity optimization failed: {type(e).__name__}: {str(e)[:200]}")

        # Attempt 3: Maximum Diversification
        try:
            logger.debug("Attempting maximum diversification optimization")
            weights = self._max_diversification_optimization(
                cov_matrix, available_assets
            )
            if weights is not None and self._validate_optimization_result(weights):
                # Log success metrics
                self._log_weight_diagnostics(weights, "Max Diversification")
                logger.info("Using maximum diversification as fallback")
                # Expand weights to full universe
                final_weights = self._expand_weights_to_universe(weights, universe, available_assets)
                logger.info(f"LSTM optimization: Max diversification succeeded with {(final_weights > 1e-6).sum()} non-zero positions")
                return final_weights
            else:
                logger.debug(f"Maximum diversification optimization produced invalid weights")
        except Exception as e:
            logger.warning(f"Maximum diversification failed: {type(e).__name__}: {str(e)[:200]}")

        # Final fallback: Constrained equal weights with top-K selection
        logger.warning(
            "LSTM OPTIMIZATION FAILURE: All three methods failed. "
            "Falling back to constrained equal weights. "
            "This indicates optimization configuration or data quality issues."
        )

        # Select top K assets based on expected returns
        # Handle None case: if no top_k limit, use all assets
        if self.constraints.top_k_positions is not None:
            k = min(self.constraints.top_k_positions, n_assets)
        else:
            k = n_assets
        top_k_indices = np.argsort(expected_returns)[-k:]

        weights = pd.Series(0.0, index=universe)
        for idx in top_k_indices:
            weights.iloc[idx] = 1.0 / k

        # Apply position size constraints
        weights = weights.clip(upper=self.constraints.max_position_weight)
        weights = weights / weights.sum() if weights.sum() > 0 else pd.Series(1.0 / n_assets, index=universe)

        # Detect and warn about equal weight fallback
        self._detect_equal_weight_fallback(weights, "Equal Weight Fallback")

        logger.info(f"LSTM fallback: Using {k} assets with equal weights")

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

    def _expand_weights_to_universe(
        self, weights: pd.Series, full_universe: list[str], available_assets: list[str]
    ) -> pd.Series:
        """
        Expand portfolio weights from available assets to full universe.

        This handles the case where covariance was computed for a filtered subset of assets
        but we need weights for the complete universe.

        Args:
            weights: Portfolio weights for available assets
            full_universe: Complete asset universe
            available_assets: Subset of assets with available data

        Returns:
            Expanded portfolio weights with zeros for unavailable assets
        """
        if len(available_assets) == len(full_universe):
            # No expansion needed
            return weights

        # Create zero weights for full universe
        expanded_weights = pd.Series(0.0, index=full_universe)

        # Fill in weights for available assets
        for asset in available_assets:
            if asset in weights.index and asset in expanded_weights.index:
                expanded_weights[asset] = weights[asset]

        # Renormalise to ensure weights sum to 1
        weights_sum = expanded_weights.sum()
        if weights_sum > 0:
            expanded_weights = expanded_weights / weights_sum
        else:
            # Fallback to equal weights if all weights are zero
            logger.warning(
                f"All weights zero after expansion. Using equal weights for {len(full_universe)} assets."
            )
            expanded_weights = pd.Series(1.0 / len(full_universe), index=full_universe)

        logger.debug(
            f"Expanded weights: {len(available_assets)} available -> {len(full_universe)} universe, "
            f"non-zero weights: {(expanded_weights > 1e-6).sum()}"
        )

        return expanded_weights

    def _log_weight_diagnostics(self, weights: pd.Series, method_name: str) -> None:
        """
        Log comprehensive weight diagnostics for optimization methods.

        Args:
            weights: Portfolio weights to analyze
            method_name: Name of optimization method for logging context
        """
        weight_values = weights.values
        weight_std = weight_values.std()
        weight_min = weight_values.min()
        weight_max = weight_values.max()
        num_nonzero = (weight_values > 1e-6).sum()
        num_total = len(weight_values)

        # Calculate effective number of assets (inverse HHI)
        hhi = (weight_values ** 2).sum()
        effective_assets = 1.0 / hhi if hhi > 0 else 0.0

        logger.info(
            f"LSTM {method_name} weights: "
            f"min={weight_min:.4f}, max={weight_max:.4f}, std={weight_std:.4f}, "
            f"non-zero={num_nonzero}/{num_total}, effective_assets={effective_assets:.1f}"
        )

    def _detect_equal_weight_fallback(self, weights: pd.Series, method_name: str) -> None:
        """
        Detect if portfolio weights are suspiciously uniform (equal weight fallback).

        This helps identify when optimization has silently failed and fallen back to equal weights.

        Args:
            weights: Portfolio weights to check
            method_name: Name of method for logging context
        """
        weight_values = weights[weights > 1e-6].values  # Only check non-zero weights
        if len(weight_values) == 0:
            logger.warning(f"LSTM {method_name}: All weights are zero")
            return

        weight_std = weight_values.std()
        expected_equal_weight = 1.0 / len(weight_values)
        mean_weight = weight_values.mean()

        # Check if weights are nearly uniform
        is_equal_weight = (weight_std < 1e-8) and (abs(mean_weight - expected_equal_weight) < 1e-8)

        if is_equal_weight:
            logger.warning(
                f"LSTM {method_name} EQUAL WEIGHT DETECTED: "
                f"std={weight_std:.2e}, mean={mean_weight:.6f}, expected={expected_equal_weight:.6f}. "
                f"Portfolio may be using naive equal weighting instead of optimized weights."
            )

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
            entropy_penalty = 0.01 * entropy  # Increased from 0.0001 to 0.01 (100x) to encourage diversification

            # CRITICAL FIX: Changed sign from - to + to penalise uniformity, not reward it
            # Entropy is maximised when weights are equal, so we ADD penalty to discourage uniformity
            return -portfolio_return + self.config.risk_aversion * portfolio_variance + regularization + entropy_penalty

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

                # Use simple temporal fill (production-safe: ffill only, no lookahead)
                from ...data.na_handling import simple_temporal_fill

                historical_data = simple_temporal_fill(historical_data, allow_bfill=False)

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

        # Use simple temporal fill (training: allow bfill for complete historical window)
        from ...data.na_handling import simple_temporal_fill

        training_data = simple_temporal_fill(training_data, allow_bfill=True)

        # CRITICAL FIX: Filter zero-variance assets to prevent gradient explosions
        # Zero-variance assets cause numerical instability in LSTM training
        if NA_HANDLING_AVAILABLE:
            training_data, variance_mask = filter_zero_variance_assets(
                training_data,
                variance_threshold=1e-8,
            )
            dropped_count = (~variance_mask).sum()
            if dropped_count > 0:
                logger.info(f"Dropped {dropped_count} zero-variance assets from training data")
        else:
            # Fallback: basic variance filtering
            asset_std = training_data.std()
            valid_assets = asset_std[asset_std > 1e-8].index.tolist()
            if len(valid_assets) < len(training_data.columns):
                dropped_count = len(training_data.columns) - len(valid_assets)
                logger.info(f"Dropped {dropped_count} zero-variance assets (fallback method)")
                training_data = training_data[valid_assets]

        # Validate data quality
        if training_data.isna().sum().sum() > 0:
            logger.warning("Training data contains NaN values after preprocessing")

        logger.info(
            f"Prepared training data: {training_data.shape[0]} days, {training_data.shape[1]} assets"
        )

        return training_data

    def _returns_to_prices(self, returns: pd.DataFrame, initial_price: float = 100.0) -> pd.DataFrame:
        """
        Convert returns to prices for feature extraction.

        Args:
            returns: DataFrame of returns with datetime index and asset columns
            initial_price: Initial price for all assets (default 100)

        Returns:
            DataFrame of prices with same shape as returns
        """
        # Convert returns to cumulative returns then to prices
        # Price_t = initial_price * (1 + r_1) * (1 + r_2) * ... * (1 + r_t)
        cumulative_returns = (1 + returns).cumprod()
        prices = cumulative_returns * initial_price

        # Prepend initial prices as first row
        initial_row = pd.DataFrame(
            initial_price,
            index=[returns.index[0] - pd.Timedelta(days=1)],
            columns=returns.columns
        )
        prices = pd.concat([initial_row, prices])

        logger.debug(f"Converted returns to prices: shape {prices.shape}")
        return prices

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
            "normalization_stats": getattr(self.network, 'normalization_stats', None),
        }

        # Save feature extractor configuration for restoration
        if hasattr(self, '_feature_extractor') and self._feature_extractor is not None:
            model_state['feature_extractor_config'] = {
                'feature_set': self.config.feature_set,
                'use_technical_features': self.config.use_technical_features,
                'feature_names': self._feature_names if hasattr(self, '_feature_names') else None,
            }
            logger.info(
                f"Saved feature extractor config: feature_set={self.config.feature_set}, "
                f"use_technical_features={self.config.use_technical_features}, "
                f"feature_names={self._feature_names if hasattr(self, '_feature_names') else None}"
            )

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
                self.network = create_ragged_lstm_network(self.config.lstm_config)

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
                self.network = create_ragged_lstm_network(self.config.lstm_config)

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
                self.network = create_ragged_lstm_network(self.config.lstm_config)

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
            self.network = create_ragged_lstm_network(self.config.lstm_config)
            self.network.load_state_dict(model_state[network_key])

            self.is_fitted = True
            self._is_pretrained = True  # Mark as pre-trained to skip retraining in backtest
            logger.info(f"Loaded complete model from {filepath}")

        # Restore normalization stats if available
        if "normalization_stats" in model_state and model_state["normalization_stats"] is not None:
            self.network.normalization_stats = model_state["normalization_stats"]
            logger.info("Restored normalization stats from checkpoint")

        # Restore feature extractor from saved configuration
        if 'feature_extractor_config' in model_state:
            config = model_state['feature_extractor_config']
            if config['use_technical_features']:
                from src.features.technical_features import create_feature_extractor
                self._feature_extractor = create_feature_extractor(config['feature_set'])
                self._feature_names = config.get('feature_names')
                logger.info(
                    f"Restored feature extractor: feature_set={config['feature_set']}, "
                    f"features={self._feature_names}"
                )
            else:
                self._feature_extractor = None
                self._feature_names = ["returns"]
        elif self.config.use_technical_features:
            # Fallback: recreate from config if no saved state available
            from src.features.technical_features import create_feature_extractor
            self._feature_extractor = create_feature_extractor(self.config.feature_set)
            logger.warning(
                f"Feature extractor config not found in checkpoint, "
                f"recreating from model config: {self.config.feature_set}"
            )
        else:
            self._feature_extractor = None
            self._feature_names = ["returns"]

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

                # Use simple temporal fill (production-safe: ffill only, no lookahead)
                from ...data.na_handling import simple_temporal_fill

                historical_data = simple_temporal_fill(historical_data, allow_bfill=False)

                # Allow sequences slightly shorter than configured length (99% threshold)
                # This handles edge cases like 251 days vs 252 days requirement
                min_required_length = int(self.config.lstm_config.sequence_length * 0.99)
                if len(historical_data) < min_required_length:
                    raise ValueError(
                        f"Insufficient historical data: {len(historical_data)} < {min_required_length} "
                        f"(99% of configured {self.config.lstm_config.sequence_length})"
                    )

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

            # CRITICAL FIX: Use training assets to prevent normalization mismatch
            # Instead of selecting top assets by activity (which creates misalignment),
            # use the same assets that were used during training with stored normalization stats
            expected_input_size = self.config.lstm_config.input_size
            selected_assets = list(sequence_data.columns)

            if hasattr(self.network, 'normalization_stats') and 'asset_names' in self.network.normalization_stats:
                training_assets = self.network.normalization_stats['asset_names']
                available_assets = list(sequence_data.columns)
                common_assets = [a for a in training_assets if a in available_assets]

                overlap_ratio = len(common_assets) / len(training_assets) if training_assets else 0
                logger.info(f"Asset selection for inference: training_assets={len(training_assets)}, "
                           f"available_assets={len(available_assets)}, "
                           f"common_assets={len(common_assets)}, "
                           f"overlap={overlap_ratio:.1%}")

                if len(common_assets) >= len(training_assets) * 0.8:
                    # Sufficient overlap - use training assets
                    sequence_data = sequence_data[common_assets]
                    selected_assets = common_assets
                    logger.info(f"Using {len(common_assets)} training assets for inference "
                               f"(overlap {overlap_ratio:.1%} >= 80% threshold)")
                else:
                    logger.warning(f"Only {len(common_assets)} of {len(training_assets)} training assets available "
                                  f"({overlap_ratio:.1%} < 80%), falling back to activity-based selection. "
                                  f"This may cause normalization mismatch.")
                    # Fall back to activity-based selection
                    if sequence_data.shape[1] > expected_input_size:
                        asset_activity = sequence_data.abs().mean().sort_values(ascending=False)
                        top_assets = asset_activity.head(expected_input_size).index
                        sequence_data = sequence_data[top_assets]
                        selected_assets = list(top_assets)
            else:
                logger.warning("No training asset names stored in normalization_stats, using activity-based selection")
                # Original logic: truncate by activity if needed
                if sequence_data.shape[1] > expected_input_size:
                    asset_activity = sequence_data.abs().mean().sort_values(ascending=False)
                    top_assets = asset_activity.head(expected_input_size).index
                    sequence_data = sequence_data[top_assets]
                    selected_assets = list(top_assets)

            # Pad if needed after asset selection
            if sequence_data.shape[1] < expected_input_size:
                padding_needed = expected_input_size - sequence_data.shape[1]
                padding = np.zeros((len(sequence_data), padding_needed))
                padding_df = pd.DataFrame(
                    padding,
                    index=sequence_data.index,
                    columns=[f'PAD_{i}' for i in range(padding_needed)]
                )
                sequence_data = pd.concat([sequence_data, padding_df], axis=1)

            # CRITICAL FIX: Extract technical features to match training
            # Training uses 9 features (returns + 8 technical), inference must use same
            if self.config.use_technical_features and hasattr(self, '_feature_extractor'):
                logger.info(
                    f"Feature extractor availability check: "
                    f"use_technical_features={self.config.use_technical_features}, "
                    f"has_extractor={hasattr(self, '_feature_extractor')}, "
                    f"extractor_is_not_none={self._feature_extractor is not None if hasattr(self, '_feature_extractor') else False}"
                )
                # Convert returns to prices for feature extraction
                initial_price = 100.0
                prices = pd.DataFrame(
                    initial_price * (1 + sequence_data).cumprod(),
                    index=sequence_data.index,
                    columns=sequence_data.columns
                )

                # Extract same features as training
                features_array, _ = self._feature_extractor.extract_features(
                    prices=prices,
                    returns=sequence_data,
                    benchmark_prices=None
                )
                # Shape: (sequence_length, num_assets, num_features=9 for "standard")
                feature_matrix = features_array
                logger.debug(f"Extracted {features_array.shape[2]} technical features for inference")
                logger.info(f"Inference features: extractor=YES, dim={features_array.shape[-1]}")

                if hasattr(features_array, 'shape') and features_array.shape[-1] != self.network.config.input_size:
                    logger.error(f"FEATURE MISMATCH: got {features_array.shape[-1]}, expected {self.network.config.input_size}")
            else:
                # Fallback: Use simple returns data
                # Shape: (sequence_length, num_assets)
                feature_matrix = sequence_data.values
                logger.warning(f"Inference features: extractor=NO, dim={feature_matrix.shape[-1]}, "
                               f"expected={self.network.config.input_size}")

            # CRITICAL FIX: Apply normalization using stored statistics (matching training)
            # Without this, inference uses raw data but network expects normalized data
            if hasattr(self.network, 'normalization_stats'):
                stats = self.network.normalization_stats
                scaler = stats.get('scaler', None)
                training_asset_names = stats.get('asset_names', None)

                # Check if we have MinMaxScaler (new approach) or legacy mean/std
                if scaler is not None:
                    # Use MinMaxScaler for normalization
                    n_features = feature_matrix.shape[1]
                    pre_norm_min = feature_matrix.min()
                    pre_norm_max = feature_matrix.max()

                    # Reshape for scaler: (sequence_length, n_assets) -> (sequence_length, n_assets)
                    # MinMaxScaler expects 2D array where each column is a feature
                    try:
                        normalized_features = scaler.transform(feature_matrix)
                        feature_matrix = normalized_features

                        post_norm_min = normalized_features.min()
                        post_norm_max = normalized_features.max()

                        logger.info(
                            f"MinMaxScaler normalization applied: "
                            f"pre=(min={pre_norm_min:.6f}, max={pre_norm_max:.6f}), "
                            f"post=(min={post_norm_min:.6f}, max={post_norm_max:.6f}), "
                            f"expected=(range≈[-1, 1])"
                        )
                    except Exception as e:
                        logger.error(f"MinMaxScaler transform failed: {e}, using raw features")
                elif 'mean' in stats and 'std' in stats:
                    # Legacy RobustScaler approach (fallback)
                    mean = stats['mean']
                    std = stats['std']
                    epsilon = stats.get('epsilon', 1e-6)
                    n_features = feature_matrix.shape[1]

                    # Handle 1D vs 2D stats
                    if mean.ndim == 1:
                        asset_alignment_ok = mean.shape[0] == n_features
                        mean = mean.reshape(1, -1)
                        std = std.reshape(1, -1)
                    else:
                        asset_alignment_ok = mean.shape[1] == n_features

                    if asset_alignment_ok:
                        normalized_features = (feature_matrix - mean) / (std + epsilon)
                        feature_matrix = normalized_features
                        logger.info(f"Legacy normalization applied (mean/std)")
                    else:
                        logger.warning(f"Feature dimension mismatch: {n_features} vs {mean.shape}, skipping normalization")
            else:
                logger.warning(
                    "No normalization stats found on network - using raw returns at inference! "
                    "This will cause 50x scale mismatch with training data. "
                    "Predictions will be meaningless."
                )

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
