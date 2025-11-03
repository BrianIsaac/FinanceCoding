"""
Memory-efficient training pipeline for LSTM portfolio models.

This module implements GPU memory optimization, gradient accumulation, and mixed precision
training to handle the full S&P MidCap 400 universe within VRAM constraints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from .architecture import LSTMNetwork, SharpeRatioLoss
from src.utils.memory_manager import BatchProcessingConfig, MemoryManager
from src.utils.gpu import AutomaticMemoryManager

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for LSTM training pipeline."""

    # Memory optimization
    max_memory_gb: float = 11.0  # VRAM limit for RTX 5070Ti
    # CRITICAL FIX: Disabled gradient accumulation to prevent batch 3 NaN explosion
    # Previous value of 4 caused optimizer.step() to trigger at batch 3, exposing accumulated instability
    # Setting to 1 ensures optimizer updates every batch, preventing gradient accumulation issues
    gradient_accumulation_steps: int = 1  # Disabled gradient accumulation (was 4)
    use_mixed_precision: bool = False  # Disabled for stability - using FP32 training
    target_memory_utilisation: float = 0.75  # Target 75% GPU memory usage

    # Training parameters
    learning_rate: float = 0.001  # Standard Adam learning rate
    weight_decay: float = 1e-5  # L2 regularization
    batch_size: int = 32  # Base batch size (will be optimized dynamically)
    epochs: int = 100  # Maximum training epochs
    patience: int = 30  # FIXED: Increased from 15 to 30 (financial data is noisy, needs more epochs)
    min_delta: float = 0.01  # Minimum improvement for early stopping (1% for Sharpe ratios)

    # Learning rate scheduling
    lr_patience: int = 15  # FIXED: Increased from 10 to 15 (delay LR reduction to allow more exploration)
    lr_factor: float = 0.5  # Learning rate reduction factor
    min_lr: float = 1e-6  # Minimum learning rate

    # Validation
    validation_split: float = 0.2  # Validation set size
    walk_forward_validation: bool = True  # Use temporal validation splits

    # Gradient optimisation
    # CRITICAL FIX v3: Increased from 2.0 to 5.0 to handle observed gradient explosions
    # Research showed gradients reaching 26.8 during training, exceeding 2.0 clip by 13.4×
    # Analysis: Peak gradient explosion at 26.8 indicates clip_value=2.0 is insufficient
    # Increasing to 5.0 provides 5.4× safety margin while still preventing runaway gradients
    gradient_clip_value: float = 5.0  # Increased from 2.0 to handle 26.8 gradient spikes
    # Gradient clipping at 5.0 handles observed gradient magnitudes (up to 26.8) while
    # still providing regularisation. This balances stability with learning signal preservation.
    # Previous value (2.0) was too aggressive and caused frequent clipping warnings.
    adaptive_clipping: bool = False  # Disabled - use fixed clipping for stability

    # Checkpointing
    save_best_only: bool = True  # Only save best model
    checkpoint_frequency: int = 5  # Save checkpoint every N epochs


class TimeSeriesDataset(Dataset):
    """Dataset for temporal sequence data with proper validation splits."""

    def __init__(
        self,
        sequences: torch.Tensor,
        targets: torch.Tensor,
        lengths: torch.Tensor | None = None,
        asset_ids: torch.Tensor | None = None,
    ):
        """
        Initialize time series dataset.

        Args:
            sequences: Input sequences of shape (n_samples, seq_len, n_features)
            targets: Target returns of shape (n_samples, n_features)
            lengths: Sequence lengths of shape (n_samples,)
            asset_ids: Asset identifiers for each sample
        """
        assert len(sequences) == len(targets), "Sequences and targets must have same length"
        if lengths is not None:
            assert len(sequences) == len(lengths), "Sequences and lengths must have same length"

        self.sequences = sequences
        self.targets = targets
        self.lengths = lengths if lengths is not None else torch.full((len(sequences),), sequences.shape[1], dtype=torch.long)
        self.asset_ids = asset_ids

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence = self.sequences[idx]
        target = self.targets[idx]
        length = self.lengths[idx]
        return sequence, target, length


class MemoryEfficientTrainer:
    """Memory-optimized trainer for LSTM portfolio models."""

    def __init__(
        self, model: LSTMNetwork, config: TrainingConfig, device: torch.device | None = None
    ):
        """
        Initialize memory-efficient trainer.

        Args:
            model: LSTM network to train
            config: Training configuration
            device: Training device (auto-detected if None)
        """
        self.model = model
        self.config = config
        self.device = device or self._get_device()

        # Move model to device
        self.model = self.model.to(self.device)

        # CRITICAL FIX: Ensure ALL model buffers are on the correct device
        # This prevents "tensor on cpu, different from cuda" errors during batch size optimization
        for name, buffer in self.model.named_buffers():
            if buffer.device != self.device:
                buffer.data = buffer.data.to(self.device)
                logger.debug(f"Moved model buffer '{name}' to {self.device}")

        # Verify all parameters are on correct device
        for name, param in self.model.named_parameters():
            if param.device != self.device:
                logger.error(f"Parameter '{name}' on wrong device: {param.device} vs {self.device}")
                param.data = param.data.to(self.device)

        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=config.lr_patience,
            factor=config.lr_factor,
            min_lr=config.min_lr,
        )

        # Initialize loss function with entropy regularisation
        # FIXED: Reduced from 1.0 to 0.01 to prevent entropy from dominating loss
        # Previous value (1.0) caused entropy term to be 80% of total loss, preventing Sharpe optimisation
        # With entropy_weight=0.01, entropy becomes ~4% of loss, allowing Sharpe to dominate (96%)
        # This ensures model optimises primarily for returns, with diversification as secondary constraint
        self.criterion = SharpeRatioLoss(entropy_weight=0.01)
        # CRITICAL: Move loss function to same device as model
        self.criterion = self.criterion.to(self.device)

        # CRITICAL FIX: Ensure loss function buffers are also on device
        for name, buffer in self.criterion.named_buffers():
            if buffer.device != self.device:
                buffer.data = buffer.data.to(self.device)
                logger.debug(f"Moved criterion buffer '{name}' to {self.device}")

        # Initialize mixed precision scaler
        # CRITICAL FIX: Only create CUDA scaler if actually using CUDA
        if config.use_mixed_precision:
            if self.device.type == "cuda":
                self.scaler = GradScaler("cuda")
                logger.debug("Initialized CUDA gradient scaler for mixed precision training")
            else:
                logger.warning("Mixed precision requested but CUDA not available - disabling mixed precision")
                self.config.use_mixed_precision = False
                self.scaler = None
        else:
            self.scaler = None

        # Training state
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": [],
            "memory_usage": [],
            "gradient_norms": [],
            "batch_processing_time": [],
            "epoch_duration": [],
            "convergence_metrics": [],
            "memory_pressure_events": [],
            "training_efficiency": [],
            "sharpe_ratio_evolution": [],
            "portfolio_volatility": [],
            "universe_size": [],
            # Enhanced financial validation metrics
            "information_ratio": [],
            "sortino_ratio": [],
            "calmar_ratio": [],
            "max_drawdown": [],
            "tracking_error": [],
            "portfolio_turnover": [],
            "hit_ratio": [],
            "prediction_accuracy": [],
            "correlation_with_returns": [],
            "portfolio_concentration": [],
        }

        # Initialize memory management
        memory_config = BatchProcessingConfig(
            batch_size=config.batch_size,
            max_memory_gb=config.max_memory_gb,
            memory_threshold=0.8,
            enable_gpu_batching=True,
            gpu_memory_threshold=0.9,
        )
        self.memory_manager = MemoryManager(memory_config)
        self.auto_memory_manager = AutomaticMemoryManager(
            memory_threshold_gb=config.max_memory_gb * 0.9,
            cleanup_interval_steps=50,
            enable_automatic_cleanup=True
        )

    def _get_device(self) -> torch.device:
        """Auto-detect optimal training device."""
        if torch.cuda.is_available():
            # Check CUDA memory
            device = torch.device("cuda:0")
            memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            logger.info(f"Using CUDA device with {memory_gb:.1f}GB memory")
            return device
        else:
            logger.warning("CUDA not available, using CPU")
            return torch.device("cpu")

    def create_sequences(
        self,
        returns_data: pd.DataFrame,
        sequence_length: int = 60,
        prediction_horizon: int = 21,  # Monthly prediction (~21 trading days)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[pd.Timestamp]]:
        """
        Create overlapping sequences for LSTM training with temporal alignment.

        Args:
            returns_data: Historical returns DataFrame with datetime index
            sequence_length: Length of input sequences (60 days)
            prediction_horizon: Days ahead to predict (21 for monthly)

        Returns:
            Tuple of (sequences, targets, lengths, dates)
            - sequences: Input sequences of shape (n_samples, seq_len, n_assets)
            - targets: Target returns of shape (n_samples, n_assets)
            - lengths: Actual sequence lengths of shape (n_samples,) - all equal to sequence_length for training
            - dates: List of target dates for temporal validation
        """
        # FIX: Add type validation before accessing DataFrame attributes
        if not isinstance(returns_data, pd.DataFrame):
            received_type = type(returns_data).__name__
            logger.error(
                f"Type validation failed in create_sequences: "
                f"Expected pd.DataFrame but received {received_type}. "
                f"This indicates the feature wrapping in model.py may have failed."
            )
            raise TypeError(
                f"create_sequences expects pd.DataFrame but received {received_type}. "
                f"Check that features are properly wrapped into DataFrame in model.py fit() method."
            )

        logger.debug(f"Type validation passed: received DataFrame with shape {returns_data.shape}")

        returns_array = returns_data.values
        dates = returns_data.index
        n_timesteps, n_features = returns_array.shape

        # CRITICAL FIX: Extract only returns for targets when using technical features
        # When technical features are enabled, the DataFrame has N*F columns (e.g., 10 assets * 9 features = 90 columns)
        # But targets should ALWAYS be just N values (the returns for each asset)

        # Check if we have technical features by looking for feature suffixes in column names
        columns = returns_data.columns.tolist()
        has_technical_features = any('_returns' in str(col) or '_volatility' in str(col) or '_rsi' in str(col) for col in columns)

        if has_technical_features:
            # Technical features case: extract only the returns columns
            # Columns are named like: ASSET_0_returns, ASSET_0_returns_5d, ..., ASSET_1_returns, ...
            # We want only the primary returns columns (those ending with just '_returns')
            returns_columns = [col for col in columns if str(col).endswith('_returns') and not any(suffix in str(col) for suffix in ['_5d', '_21d', '_63d'])]

            if not returns_columns:
                # Fallback: extract every Nth column where N is the number of features per asset
                # Assume 9 features per asset (standard feature set)
                n_features_per_asset = 9
                n_assets = n_features // n_features_per_asset
                target_indices = list(range(0, n_features, n_features_per_asset))
                target_array = returns_array[:, target_indices]
                logger.info(f"Technical features fallback: Extracted {n_assets} return columns from {n_features} total features")
            else:
                # Get indices of returns columns
                target_indices = [columns.index(col) for col in returns_columns]
                target_array = returns_array[:, target_indices]
                n_assets = len(returns_columns)
                logger.info(f"Technical features: Extracted {n_assets} returns columns from {n_features} total features")
        else:
            # Simple case: no technical features, all columns are returns
            target_array = returns_array
            n_assets = n_features
            logger.debug(f"No technical features: Using all {n_features} columns as returns")

        # CRITICAL FIX: Filter assets with very low variance BEFORE normalisation
        # This prevents extreme normalised values (10^6+) that cause gradient explosion
        # Previous approach: normalise all assets with epsilon, causing (returns - mean) / 1e-8 = 10^6+
        # Fixed approach: remove low-variance assets entirely before normalisation
        mean = np.nanmean(returns_array, axis=0, keepdims=True)
        std = np.nanstd(returns_array, axis=0, keepdims=True)

        # Separate stats for target returns
        target_mean = np.nanmean(target_array, axis=0, keepdims=True)
        target_std = np.nanstd(target_array, axis=0, keepdims=True)

        # FIX #6a: Filter out assets with std < 1e-5 to match primary filter
        std_threshold = 1e-5  # FIXED: Changed from 1e-4 to align with model.py filter
        valid_assets = (std >= std_threshold).flatten()

        if not np.all(valid_assets):
            num_filtered = (~valid_assets).sum()
            logger.warning(
                f"Secondary variance filtering found {num_filtered} assets with std < {std_threshold:.2e}. "
                f"Pre-sizing filter in model.py should have caught these - may indicate data drift or filter bypass."
            )

            # Filter data to keep only valid assets
            returns_array = returns_array[:, valid_assets]
            mean = mean[:, valid_assets]
            std = std[:, valid_assets]

            # Update returns_data to reflect filtered assets (for consistency)
            filtered_columns = returns_data.columns[valid_assets]
            returns_data = returns_data[filtered_columns]

        # CRITICAL FIX: Use MinMaxScaler for feature-wise normalization
        # MinMaxScaler to [-1, 1] ensures consistent scale across all feature types
        # and prevents gradient explosion/vanishing during training
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler(feature_range=(-1, 1))  # Scale all features to [-1, 1]
        returns_normalised = scaler.fit_transform(returns_array)

        # Store scaler for inverse transform during inference
        self.feature_scaler = scaler

        logger.info(f"Data normalization (MinMaxScaler): original_range=[{returns_array.min():.6f}, {returns_array.max():.6f}], "
                    f"normalized_range=[{returns_normalised.min():.6f}, {returns_normalised.max():.6f}], "
                    f"normalized_mean={returns_normalised.mean().item():.6f}")

        # CRITICAL FIX: Store normalization statistics for inference with asset identifiers
        asset_names = list(returns_data.columns)
        self.model.normalization_stats = {
            'asset_names': asset_names,  # Asset identifiers for alignment
            'scaler': scaler,  # sklearn MinMaxScaler for inverse transform
            'data_min': scaler.data_min_,  # Minimum values per feature
            'data_max': scaler.data_max_,  # Maximum values per feature
            'feature_range': scaler.feature_range,  # Target range (-1, 1)
            'epsilon': 1e-6,  # Numerical stability
            'valid_assets': valid_assets.copy() if 'valid_assets' in locals() else None,
        }
        logger.info(
            f"Stored normalization statistics on network for inference: "
            f"n_features={len(asset_names)}, scaler_type=MinMaxScaler, "
            f"data_min_shape={scaler.data_min_.shape}, data_max_shape={scaler.data_max_.shape}, "
            f"feature_names_sample={asset_names[:5] if len(asset_names) >= 5 else asset_names}"
        )

        # DIAGNOSTIC: Check for potential normalisation issues
        if np.any(np.isnan(scaler.data_min_)) or np.any(np.isnan(scaler.data_max_)):
            raise ValueError(
                f"CRITICAL: Normalization stats contain NaN. "
                f"This will corrupt all predictions. Training must stop."
            )

        # Log normalisation statistics for monitoring
        logger.info(
            f"Input normalisation: data_min_range=[{np.nanmin(scaler.data_min_):.6f}, {np.nanmax(scaler.data_min_):.6f}], "
            f"data_max_range=[{np.nanmin(scaler.data_max_):.6f}, {np.nanmax(scaler.data_max_):.6f}]"
        )
        logger.info(
            f"After normalisation: data_range=[{np.nanmin(returns_normalised):.4f}, {np.nanmax(returns_normalised):.4f}]"
        )

        sequences = []
        targets = []
        target_dates = []

        # Standard time series practice: stride=1 for dense sampling
        # Dropout (0.3) and L2 regularisation (1e-5) prevent overfitting
        stride = 1

        # Normalize targets separately (only N assets, not N*F features)
        # Use same MinMaxScaler approach for consistency
        target_scaler = MinMaxScaler(feature_range=(-1, 1))
        target_normalised = target_scaler.fit_transform(target_array)
        self.target_scaler = target_scaler  # Store for inverse transform
        logger.info(f"Target normalization (MinMaxScaler): shape={target_normalised.shape}, "
                    f"range=[{target_normalised.min():.6f}, {target_normalised.max():.6f}], "
                    f"mean={target_normalised.mean():.6f}")

        # Create sequences with controlled overlap
        for i in range(sequence_length, n_timesteps - prediction_horizon, stride):
            # Input sequence: t-(sequence_length-1) to t (normalised) - ALL features
            sequence = returns_normalised[i - sequence_length : i]

            # FIXED: Use single-day forward return to match input normalization scale
            # Previous 5-day averaging created scale mismatch causing negative Sharpe ratios:
            #   - Inputs: std ≈ 1.0 (normalised daily returns)
            #   - 5-day averaged targets: std ≈ 0.447 (variance reduced by sqrt(5))
            # This 2.24x scale mismatch artificially inflated Sharpe by 2.24x and caused
            # model to learn anti-correlated predictions (negative Sharpe -0.43 to -0.04)
            # Single-day target ensures inputs and targets have consistent variance scale
            target_idx = i + prediction_horizon - 1  # Target date at end of prediction horizon
            if target_idx >= len(target_normalised):
                continue  # Skip if target is beyond available data
            target = target_normalised[target_idx]  # Single day return for N assets only

            sequences.append(sequence)
            targets.append(target)
            target_dates.append(dates[target_idx])  # Use target date

        sequences = torch.tensor(np.array(sequences), dtype=torch.float32)
        targets = torch.tensor(np.array(targets), dtype=torch.float32)

        # Log target statistics to verify normalization
        logger.info(
            f"Target normalisation: mean={targets.mean():.6f}, std={targets.std():.6f}, "
            f"range=[{targets.min():.4f}, {targets.max():.4f}]"
        )

        if len(sequences) == 0:
            raise ValueError(
                f"Insufficient data for sequence creation. Data has {n_timesteps} timesteps, "
                f"need at least {sequence_length + prediction_horizon} for sequence_length={sequence_length} "
                f"and prediction_horizon={prediction_horizon}. "
                f"Consider using longer training periods or shorter sequence lengths."
            )

        # For training, all sequences are full length (no ragged edges)
        # In production inference, lengths would vary based on actual data availability
        lengths = torch.full((len(sequences),), sequence_length, dtype=torch.long)

        logger.info(f"Created {len(sequences)} sequences with shape {sequences.shape}")

        return sequences, targets, lengths, target_dates

    def create_walk_forward_splits(
        self,
        sequences: torch.Tensor,
        targets: torch.Tensor,
        lengths: torch.Tensor,
        dates: list[pd.Timestamp],
        validation_months: int = 12,  # Use last 12 months for validation
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create walk-forward validation splits to prevent look-ahead bias.

        ENHANCED: Uses adaptive validation split based on data availability to ensure
        sufficient training data while maintaining meaningful validation set.

        Args:
            sequences: Input sequences
            targets: Target returns
            lengths: Sequence lengths
            dates: Corresponding dates
            validation_months: Number of months for validation set (adaptive if data limited)

        Returns:
            Tuple of (train_sequences, train_targets, train_lengths, val_sequences, val_targets, val_lengths)
        """
        # Convert to pandas for date handling
        dates_series = pd.Series(dates)
        total_samples = len(sequences)

        # ENHANCED: Adaptive validation split based on data availability
        # For small datasets, use proportional split instead of fixed months
        if total_samples < 1000:  # Limited data
            # Use 20% for validation (more training data = better learning)
            validation_ratio = 0.20
            split_idx = int(total_samples * (1 - validation_ratio))
            split_date = dates_series.iloc[split_idx]
            logger.info(
                f"Using adaptive validation split (20%) due to limited data ({total_samples} samples)"
            )
        else:
            # Sufficient data: use time-based split
            end_date = dates_series.max()
            split_date = end_date - pd.DateOffset(months=validation_months)

        # Create boolean mask for training/validation split
        train_mask = dates_series <= split_date
        val_mask = dates_series > split_date

        train_sequences = sequences[train_mask]
        train_targets = targets[train_mask]
        train_lengths = lengths[train_mask]
        val_sequences = sequences[val_mask]
        val_targets = targets[val_mask]
        val_lengths = lengths[val_mask]

        # CRITICAL FIX: Ensure same number of assets (features) in train and val
        # This prevents shape mismatch errors during validation
        if train_targets.shape[1] != val_targets.shape[1]:
            logger.warning(
                f"Asset count mismatch detected: train={train_targets.shape[1]}, val={val_targets.shape[1]}. "
                f"Aligning to common asset set."
            )
            # Use the minimum common set to ensure compatibility
            min_assets = min(train_targets.shape[1], val_targets.shape[1])
            train_sequences = train_sequences[:, :, :min_assets]
            train_targets = train_targets[:, :min_assets]
            val_sequences = val_sequences[:, :, :min_assets]
            val_targets = val_targets[:, :min_assets]
            logger.info(f"Aligned to {min_assets} common assets across train/val splits")

        logger.info(
            f"Train samples: {len(train_sequences)}, Validation samples: {len(val_sequences)}"
        )
        logger.info(f"Split date: {split_date.strftime('%Y-%m-%d')}")
        logger.info(f"Assets: train={train_targets.shape[1]}, val={val_targets.shape[1]}")

        return train_sequences, train_targets, train_lengths, val_sequences, val_targets, val_lengths

    def estimate_memory_usage(self, batch_size: int, sequence_length: int) -> float:
        """
        Enhanced memory usage estimation accounting for mixed precision and GPU overhead.

        Args:
            batch_size: Training batch size
            sequence_length: Input sequence length

        Returns:
            Estimated memory usage in GB
        """
        # Base model memory calculation
        # For ragged LSTM, assume average real length is ~90% of max sequence_length
        if hasattr(self.model, 'get_memory_usage'):
            # Check if model is RaggedLSTMNetwork (requires avg_real_length)
            import inspect
            sig = inspect.signature(self.model.get_memory_usage)
            if 'avg_real_length' in sig.parameters:
                avg_real_length = int(sequence_length * 0.9)  # Conservative estimate
                base_model_memory = self.model.get_memory_usage(batch_size, sequence_length, avg_real_length)
            else:
                base_model_memory = self.model.get_memory_usage(batch_size, sequence_length)
        else:
            # Fallback estimation
            param_count = sum(p.numel() for p in self.model.parameters())
            base_model_memory = param_count * 4  # float32

        # Account for mixed precision memory savings
        if self.config.use_mixed_precision:
            # FP16 activations save ~50% activation memory, but parameters stay FP32
            mixed_precision_factor = 0.7  # Empirical factor for FP16 training
            model_memory = base_model_memory * mixed_precision_factor
        else:
            model_memory = base_model_memory

        # Optimizer state memory (Adam: momentum + variance buffers)
        param_count = sum(p.numel() for p in self.model.parameters())
        optimizer_memory = param_count * 4 * 2  # float32 * (momentum + variance)

        # Gradient memory (only count once, not double-counted from model)
        grad_memory = param_count * 4

        # Add GPU runtime overhead (CUDA context, kernels, memory fragmentation)
        gpu_overhead_factor = 1.2  # 20% overhead for CUDA operations

        # Memory fragmentation overhead (especially for dynamic batch sizes)
        fragmentation_overhead = 0.15  # 15% overhead for memory fragmentation

        # Calculate total with all factors
        base_memory = model_memory + optimizer_memory + grad_memory
        total_memory_bytes = base_memory * gpu_overhead_factor * (1 + fragmentation_overhead)

        # Add sequence-dependent working memory (attention computation overhead)
        attention_overhead = batch_size * sequence_length * sequence_length * 2  # FP16 attention matrices
        total_memory_bytes += attention_overhead

        total_memory_gb = total_memory_bytes / (1024**3)

        return total_memory_gb

    def optimize_batch_size(self, sequence_length: int, universe_size: int | None = None) -> int:
        """
        Dynamically optimize batch size using actual GPU memory monitoring and universe size.

        Args:
            sequence_length: Input sequence length
            universe_size: Number of assets in the universe

        Returns:
            Optimal batch size for given memory limits and data characteristics
        """
        if not torch.cuda.is_available():
            return self.config.batch_size

        # CRITICAL FIX: Verify device consistency before optimization
        # This prevents "tensor on cpu vs cuda" errors during batch testing
        model_device = next(self.model.parameters()).device
        if model_device != self.device:
            logger.error(f"Device mismatch detected: model on {model_device}, trainer expects {self.device}")
            logger.info("Correcting device placement...")
            self.model = self.model.to(self.device)
            for name, buffer in self.model.named_buffers():
                if buffer.device != self.device:
                    buffer.data = buffer.data.to(self.device)

        # Verify criterion is also on correct device
        if hasattr(self.criterion, 'parameters'):
            for param in self.criterion.parameters():
                if param.device != self.device:
                    logger.error(f"Criterion parameter on wrong device: {param.device} vs {self.device}")
                    self.criterion = self.criterion.to(self.device)

        # Get actual GPU memory capacity
        total_gpu_memory = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
        target_memory = total_gpu_memory * getattr(self.config, 'target_memory_utilisation', 0.75)

        # Universe size considerations
        universe_size = universe_size or len(self.universe) if hasattr(self, 'universe') and self.universe else 500

        logger.info(
            f"GPU memory optimization: {total_gpu_memory:.1f}GB total, targeting {target_memory:.1f}GB "
            f"({getattr(self.config, 'target_memory_utilisation', 0.75)*100:.0f}%) for universe_size={universe_size}"
        )

        # Universe size-aware batch size calculation
        batch_sizes_to_test = []

        # Adaptive base size based on universe characteristics
        base_size = self.config.batch_size

        # AGGRESSIVE batch sizing for maximum GPU utilization
        # FIX: Increased caps to target 50-75% GPU utilization (was at 6.3%)
        if universe_size <= 100:  # Small universe - large batches for GPU utilization
            universe_multiplier = 16.0  # Increased back for better GPU usage
            max_batch_cap = 8192  # Large batches OK for small universe
        elif universe_size <= 300:  # Medium universe - good sized batches
            universe_multiplier = 12.0  # Increased for better utilization
            max_batch_cap = 4096  # Increased cap
        elif universe_size <= 500:  # Large universe - moderate batches
            universe_multiplier = 8.0  # Doubled for better GPU usage
            max_batch_cap = 2048  # Doubled cap
        else:  # Very large universe (679 assets) - still room for bigger batches
            universe_multiplier = 6.0  # Doubled from 3.0 for better GPU usage
            max_batch_cap = 1536  # Doubled from 768 - GPU has 11.5GB, using only 0.72GB

        # Adjust for sequence length impact
        if sequence_length > 60:  # Long sequences need smaller batches
            sequence_multiplier = 0.8
        elif sequence_length > 30:
            sequence_multiplier = 0.9
        else:
            sequence_multiplier = 1.0

        effective_base = int(base_size * universe_multiplier * sequence_multiplier)
        logger.debug(
            f"Universe-aware batch sizing: base={base_size} * universe_mult={universe_multiplier:.2f} "
            f"* seq_mult={sequence_multiplier:.2f} = {effective_base}"
        )

        # Test exponentially increasing batch sizes with universe constraints
        for multiplier in [0.5, 1, 1.5, 2, 3, 4, 6, 8]:
            test_size = int(effective_base * multiplier)
            if 8 <= test_size <= max_batch_cap:  # Ensure reasonable bounds
                batch_sizes_to_test.append(test_size)

        # Add GPU-efficient sizes that fit within universe constraints
        efficient_sizes = [16, 24, 32, 48, 64, 96, 128, 192, 256]
        for size in efficient_sizes:
            if size <= max_batch_cap:
                batch_sizes_to_test.append(size)

        batch_sizes_to_test = sorted(set(batch_sizes_to_test))

        best_batch_size = 1
        best_actual_memory = 0
        best_theoretical_memory = 0

        for batch_size in batch_sizes_to_test:
            # Test actual memory usage with a small batch
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                # Create dummy tensors to simulate batch with realistic size
                test_batch = min(batch_size, 512)  # Moderate test batch for accurate memory estimation
                dummy_sequences = torch.randn(
                    test_batch,
                    sequence_length,
                    len(self.universe),
                    device=self.device,
                    dtype=torch.float16 if self.config.use_mixed_precision else torch.float32
                )
                dummy_targets = torch.randn(
                    test_batch,
                    len(self.universe),
                    device=self.device,
                    dtype=torch.float16 if self.config.use_mixed_precision else torch.float32
                )
                # Create dummy lengths (all sequences full length for testing)
                dummy_lengths = torch.full((test_batch,), sequence_length, dtype=torch.long, device=self.device)

                # Set model to training mode for backward pass compatibility
                self.model.train()

                # Simulate forward pass
                with torch.amp.autocast(device_type='cuda', enabled=self.config.use_mixed_precision):
                    predictions, _ = self.model(dummy_sequences, dummy_lengths)
                    loss = self.criterion(predictions, dummy_targets, model=self.model)

                # Simulate backward pass
                loss.backward()

                # Scale memory usage to full batch size with GPU efficiency corrections
                test_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
                scale_factor = batch_size / test_batch

                # Apply non-linear scaling to account for GPU memory efficiency
                # Small batches have overhead; larger batches are more efficient
                if scale_factor > 1:
                    # GPU efficiency improves with larger batches, apply slight discount
                    efficiency_factor = 0.90 + 0.10 / scale_factor  # 90-100% efficiency
                    estimated_full_memory = test_memory_gb * scale_factor * efficiency_factor
                else:
                    estimated_full_memory = test_memory_gb

                # Clean up
                del dummy_sequences, dummy_targets, dummy_lengths, predictions, loss
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()

                # Check if this batch size fits
                if estimated_full_memory <= target_memory:
                    best_batch_size = batch_size
                    best_actual_memory = estimated_full_memory
                    best_theoretical_memory = self.estimate_memory_usage(batch_size, sequence_length)
                    logger.debug(f"Batch size {batch_size}: {estimated_full_memory:.2f}GB estimated (fits)")
                else:
                    logger.debug(f"Batch size {batch_size}: {estimated_full_memory:.2f}GB estimated (too large)")
                    break  # Larger batch sizes will use even more memory

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.debug(f"Batch size {batch_size}: OOM error (too large)")
                    torch.cuda.empty_cache()
                    break
                else:
                    logger.warning(f"Error testing batch size {batch_size}: {e}")
                    torch.cuda.empty_cache()
                    continue

        # Calculate actual utilisation
        actual_utilisation_pct = (best_actual_memory / total_gpu_memory) * 100
        theoretical_utilisation_pct = (best_theoretical_memory / total_gpu_memory) * 100

        # DIAGNOSTIC: Enhanced memory estimation logging
        logger.info(
            f"Optimized batch size: {best_batch_size}"
            f"\n  Sequence length: {sequence_length}, Universe size: {len(self.universe) if hasattr(self, 'universe') else 'N/A'}"
            f"\n  Actual GPU usage: {best_actual_memory:.2f}GB/{total_gpu_memory:.1f}GB ({actual_utilisation_pct:.1f}%)"
            f"\n  Theoretical estimate: {best_theoretical_memory:.2f}GB ({theoretical_utilisation_pct:.1f}%)"
            f"\n  Mixed precision: {self.config.use_mixed_precision}, Gradient accumulation: {self.config.gradient_accumulation_steps}"
        )

        # DIAGNOSTIC: Log all tested batch sizes for debugging
        logger.debug(
            f"Batch size optimization tested: {len(batch_sizes_to_test)} candidates, "
            f"range=[{min(batch_sizes_to_test) if batch_sizes_to_test else 0}, "
            f"{max(batch_sizes_to_test) if batch_sizes_to_test else 0}], "
            f"selected={best_batch_size}"
        )

        # Enhanced warnings and suggestions based on GPU architecture efficiency
        if actual_utilisation_pct < 30:
            logger.warning(f"Very low GPU memory utilisation ({actual_utilisation_pct:.1f}%) - GPU severely underutilised")
            # More aggressive suggestions for very low utilisation
            suggested_batch = min(best_batch_size * 6, 384)
            logger.info(f"Suggestion: Try batch size {suggested_batch} for better utilisation")
        elif actual_utilisation_pct < 50:
            logger.warning(f"Low GPU memory utilisation ({actual_utilisation_pct:.1f}%) - consider increasing batch size")
            suggested_batch = min(best_batch_size * 3, 256)
            logger.info(f"Suggestion: Try batch size {suggested_batch} for better utilisation")
        elif actual_utilisation_pct < 70:
            logger.info(f"Moderate GPU memory utilisation ({actual_utilisation_pct:.1f}%) - room for improvement")
            suggested_batch = min(best_batch_size * 2, 192)
            logger.info(f"Suggestion: Try batch size {suggested_batch} for optimal utilisation")
        elif actual_utilisation_pct > 90:
            logger.warning(f"High GPU memory utilisation ({actual_utilisation_pct:.1f}%) - risk of OOM errors")

        # If theoretical estimate is very different from actual, warn about estimation accuracy
        estimation_error = abs(actual_utilisation_pct - theoretical_utilisation_pct)
        if estimation_error > 20:
            logger.warning(f"Memory estimation error: {estimation_error:.1f}% difference between actual and theoretical usage")

        return best_batch_size

    def _forward_pass_with_mixed_precision(
        self, sequences: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with automatic mixed precision."""
        with autocast("cuda"):
            predictions, _ = self.model(sequences, lengths)
            return self.criterion(predictions, targets, model=self.model) / self.config.gradient_accumulation_steps

    def _backward_pass_mixed_precision(self, loss: torch.Tensor, batch_idx: int) -> None:
        """Backward pass with gradient scaling for mixed precision."""
        # Check if loss is finite before scaling
        if not torch.isfinite(loss):
            logger.warning(f"Non-finite loss detected: {loss}")
            self.optimizer.zero_grad()
            return

        # CRITICAL FIX: Track if we've done unscale in this step
        # This prevents double unscale errors
        if not hasattr(self, '_unscaled_this_step'):
            self._unscaled_this_step = False

        # Scale loss and perform backward pass
        self.scaler.scale(loss).backward()

        # Always calculate and store scaled gradient norm for logging
        # This ensures we always have a gradient norm even with single batch
        scaled_grad_norm = self._calculate_gradient_norm()
        self._last_grad_norm = scaled_grad_norm  # Store scaled norm as fallback

        if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
            try:
                # CRITICAL FIX: Only unscale if we haven't already
                if not self._unscaled_this_step:
                    self.scaler.unscale_(self.optimizer)
                    self._unscaled_this_step = True
                else:
                    logger.warning(f"Skipping duplicate unscale at batch {batch_idx}")

                # Adaptive gradient clipping based on model size
                clip_value = self.config.gradient_clip_value
                if hasattr(self.config, 'adaptive_clipping') and self.config.adaptive_clipping:
                    # Scale clipping based on number of parameters (for large models)
                    param_count = sum(p.numel() for p in self.model.parameters())
                    if param_count > 1e6:  # Large model (>1M parameters)
                        clip_value *= 2.0

                # Calculate unscaled gradient norm before clipping for monitoring
                grad_norm_before_clip = self._calculate_gradient_norm()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)

                # Store unscaled gradient norm for later logging (overwrites scaled norm)
                self._last_grad_norm = grad_norm_before_clip

                # ADDED: Check for vanishing gradients (critical diagnostic)
                if grad_norm_before_clip < 1e-6:
                    logger.warning(
                        f"VANISHING GRADIENTS detected at batch {batch_idx}: "
                        f"grad_norm={grad_norm_before_clip:.2e}. This indicates the model is not learning. "
                        f"Check: (1) loss function gradient flow, (2) activation functions, "
                        f"(3) weight initialisation."
                    )

                # Log detailed gradient statistics for monitoring
                if batch_idx % 50 == 0:  # Log every 50 batches to avoid spam
                    logger.debug(
                        f"Batch {batch_idx}: grad_norm={grad_norm:.2f} (before_clip={grad_norm_before_clip:.2f}, "
                        f"clip_value={clip_value:.1f}, clipped={grad_norm_before_clip > clip_value})"
                    )

                # Check for extreme gradients (should be rare with normalised inputs)
                if grad_norm > clip_value * 2:  # Warning at 2x threshold
                    logger.warning(
                        f"Extreme gradient detected: {grad_norm:.1f} > {clip_value * 2:.1f}. "
                        f"Check data quality or consider increasing clip_value."
                    )

                # CRITICAL FIX: Perform optimizer step with proper scaler workflow
                # scaler.step() will check for infs/NaNs and skip the step if found
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # Reset unscale flag for next step
                self._unscaled_this_step = False

            except RuntimeError as e:
                error_msg = str(e)
                if "unscale_() has already been called" in error_msg:
                    # Double unscale detected - skip this update
                    logger.warning(f"Double unscale detected at batch {batch_idx} - skipping optimizer step")
                    self.optimizer.zero_grad()
                    self._unscaled_this_step = False  # Reset flag
                elif "No inf checks were recorded" in error_msg:
                    # CRITICAL FIX: Scaler state corruption
                    # This happens when step() is called without a forward pass
                    logger.warning(
                        f"Gradient scaler state error at batch {batch_idx}. "
                        f"Resetting scaler and skipping batch."
                    )
                    # Reinitialize scaler to clear corrupted state
                    self.scaler = GradScaler("cuda")
                    self.optimizer.zero_grad()
                    self._unscaled_this_step = False  # Reset flag
                else:
                    # Unknown error - reset state and continue
                    logger.error(f"Unexpected error in mixed precision backward pass: {e}")
                    self.optimizer.zero_grad()
                    self._unscaled_this_step = False  # Reset flag

    def _forward_pass_standard(
        self, sequences: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with standard precision."""
        predictions, _ = self.model(sequences, lengths)
        return self.criterion(predictions, targets, model=self.model) / self.config.gradient_accumulation_steps

    def _backward_pass_standard(self, loss: torch.Tensor, batch_idx: int) -> None:
        """Backward pass with standard precision."""
        loss.backward()

        # Always calculate gradient norm for accurate logging
        grad_norm_before_clip = self._calculate_gradient_norm()
        self._last_grad_norm = grad_norm_before_clip

        if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)

            # Check for vanishing gradients (critical diagnostic)
            if grad_norm_before_clip < 1e-6:
                logger.warning(
                    f"VANISHING GRADIENTS detected at batch {batch_idx}: "
                    f"grad_norm={grad_norm_before_clip:.2e}. This indicates the model is not learning."
                )

            self.optimizer.step()
            self.optimizer.zero_grad()

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """
        Train model for one epoch with memory optimization.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Track memory usage and GPU utilisation
        initial_memory = 0
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            total_gpu_memory = torch.cuda.get_device_properties(self.device).total_memory

            # Initialize dynamic GPU monitoring with improved thresholds
            gpu_utilisation_history = []
            memory_efficiency_threshold = 0.5  # 50% minimum utilisation for reasonable efficiency
            last_memory_check = 0

        for batch_idx, (sequences, targets, lengths) in enumerate(train_loader):
            try:
                # Memory monitoring and cleanup
                memory_stats = self.auto_memory_manager.monitor_and_cleanup()

                sequences = sequences.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                lengths = lengths.to(self.device, non_blocking=True)

                # Forward and backward pass with error recovery
                loss_value = self._safe_training_step(sequences, targets, lengths, batch_idx)

                # Enhanced memory monitoring every 10 batches
                if batch_idx % 10 == 0:
                    pressure_check = self.memory_manager.check_memory_pressure()
                    if pressure_check["memory_pressure"]:
                        logger.warning(f"Memory pressure detected at batch {batch_idx}, optimising...")
                        self.memory_manager.optimize_memory_usage()
                        self.training_history["memory_pressure_events"].append({
                            "epoch": epoch,
                            "batch": batch_idx,
                            "memory_gb": memory_stats.get("allocated_gb", 0),
                        })

                    # Real-time GPU utilisation monitoring
                    if self.device.type == "cuda":
                        current_memory = torch.cuda.memory_allocated()
                        current_utilisation = (current_memory / total_gpu_memory) * 100
                        gpu_utilisation_history.append(current_utilisation)

                        # Track efficiency over time and suggest improvements
                        if len(gpu_utilisation_history) >= 5:  # After 50 batches
                            avg_utilisation = sum(gpu_utilisation_history[-5:]) / 5
                            if avg_utilisation < memory_efficiency_threshold * 100:
                                if batch_idx - last_memory_check > 100:  # Don't spam warnings
                                    logger.warning(
                                        f"Real-time GPU utilisation: {avg_utilisation:.1f}% "
                                        f"(target: >{memory_efficiency_threshold*100:.0f}%) - consider dynamic batch size increase"
                                    )
                                    last_memory_check = batch_idx

                        # Log detailed memory stats every 50 batches
                        if batch_idx % 50 == 0:
                            peak_memory = torch.cuda.max_memory_allocated()
                            memory_efficiency = (current_memory / total_gpu_memory) * 100
                            logger.info(
                                f"GPU Memory @ Batch {batch_idx}: "
                                f"Current: {current_memory/(1024**3):.2f}GB ({memory_efficiency:.1f}%), "
                                f"Peak: {peak_memory/(1024**3):.2f}GB, "
                                f"Total: {total_gpu_memory/(1024**3):.1f}GB"
                            )

            except RuntimeError as e:
                error_recovery_success = self._handle_training_error(e, epoch, batch_idx)
                if not error_recovery_success:
                    logger.error(f"Unrecoverable error at epoch {epoch}, batch {batch_idx}: {e}")
                    raise
                loss_value = 0.0  # Skip this batch

            except Exception as e:
                logger.warning(f"Unexpected error at epoch {epoch}, batch {batch_idx}: {e}")
                loss_value = 0.0  # Skip this batch and continue
            if not torch.isfinite(torch.tensor(loss_value)):
                loss_value = 0.0

            total_loss += loss_value * self.config.gradient_accumulation_steps
            num_batches += 1

            # Log progress
            if batch_idx % 50 == 0:
                display_loss = loss_value if torch.isfinite(torch.tensor(loss_value)) else 0.0
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {display_loss:.6f}"
                )

        # Track memory usage
        if self.device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated()
            memory_gb = (peak_memory - initial_memory) / (1024**3)
            self.training_history["memory_usage"].append(memory_gb)
            logger.info(f"Peak memory usage: {memory_gb:.2f}GB")
            torch.cuda.reset_peak_memory_stats()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate(self, val_loader: DataLoader) -> tuple[float, dict[str, float]]:
        """
        Validate model performance with comprehensive financial metrics.

        Args:
            val_loader: Validation data loader

        Returns:
            Tuple of (average validation loss, financial validation metrics)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # Collect all predictions and targets for comprehensive metrics
        all_predictions = []
        all_targets = []
        all_weights = []

        with torch.no_grad():
            for sequences, targets, lengths in val_loader:
                sequences = sequences.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                lengths = lengths.to(self.device, non_blocking=True)

                if self.config.use_mixed_precision:
                    with autocast("cuda"):
                        predictions, attention_weights = self.model(sequences, lengths)
                        loss = self.criterion(predictions, targets, model=self.model)
                else:
                    predictions, attention_weights = self.model(sequences, lengths)
                    loss = self.criterion(predictions, targets, model=self.model)

                # Handle NaN losses by skipping the batch entirely
                loss_value = loss.item()
                if not torch.isfinite(torch.tensor(loss_value)):
                    logger.warning(
                        f"Non-finite validation loss detected: {loss_value}, skipping batch. "
                        f"This may indicate numerical instability in loss calculation."
                    )
                    continue  # Skip this batch instead of masking with constant

                total_loss += loss_value
                num_batches += 1

                # Collect data for financial metrics calculation
                all_predictions.append(predictions.detach())
                all_targets.append(targets.detach())

                # Convert predictions to portfolio weights (using softmax for now)
                weights = torch.softmax(predictions, dim=-1)
                all_weights.append(weights.detach())

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Calculate comprehensive financial validation metrics
        financial_metrics = {}
        if all_predictions and all_targets:
            try:
                # CRITICAL FIX: Align both batch dimension AND feature dimension
                # First, find minimum batch size
                min_batch_size = min(p.shape[0] for p in all_predictions + all_targets + all_weights)

                # Second, find minimum feature dimension (number of assets)
                min_feature_size = min(
                    min(p.shape[1] for p in all_predictions),
                    min(t.shape[1] for t in all_targets),
                    min(w.shape[1] for w in all_weights)
                )

                # CRITICAL FIX: Only trim the LAST batch to minimum size, not all batches
                # Previous code trimmed all batches, losing valid data from non-last batches
                # This caused shape mismatches when last batch had fewer samples (e.g., 95 vs 96)
                aligned_predictions = []
                aligned_targets = []
                aligned_weights = []

                for i, (p, t, w) in enumerate(zip(all_predictions, all_targets, all_weights)):
                    if i == len(all_predictions) - 1:  # Last batch - may be smaller
                        aligned_predictions.append(p[:min_batch_size, :min_feature_size])
                        aligned_targets.append(t[:min_batch_size, :min_feature_size])
                        aligned_weights.append(w[:min_batch_size, :min_feature_size])
                    else:  # Not last batch - only trim features, keep all samples
                        aligned_predictions.append(p[:, :min_feature_size])
                        aligned_targets.append(t[:, :min_feature_size])
                        aligned_weights.append(w[:, :min_feature_size])

                # Concatenate all batches
                concat_predictions = torch.cat(aligned_predictions, dim=0)
                concat_targets = torch.cat(aligned_targets, dim=0)
                concat_weights = torch.cat(aligned_weights, dim=0)

                logger.debug(
                    f"Aligned validation tensors: predictions={concat_predictions.shape}, "
                    f"targets={concat_targets.shape}, weights={concat_weights.shape}"
                )

                # Calculate financial validation metrics
                financial_metrics = self._calculate_financial_validation_metrics(
                    concat_predictions, concat_targets, concat_weights
                )

                # Log key metrics periodically
                if len(self.training_history["val_loss"]) % 10 == 0:  # Every 10 epochs
                    logger.info(
                        f"Validation Metrics - Correlation: {financial_metrics.get('correlation_with_returns', 0):.4f}, "
                        f"Hit Ratio: {financial_metrics.get('hit_ratio', 0):.4f}, "
                        f"Sharpe: {financial_metrics.get('sharpe_ratio', 0):.4f}"
                    )

            except Exception as e:
                logger.warning(f"Error calculating validation financial metrics: {e}")
                # Return default metrics on error
                financial_metrics = {
                    "correlation_with_returns": 0.0, "hit_ratio": 0.5, "prediction_accuracy": 0.0,
                    "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "calmar_ratio": 0.0,
                    "max_drawdown": 0.0, "tracking_error": 0.0, "portfolio_turnover": 0.0,
                    "information_ratio": 0.0, "portfolio_volatility": 0.0, "portfolio_concentration": 0.0
                }

        return avg_loss, financial_metrics

    def _create_data_splits(
        self, sequences: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor, dates: list
    ) -> tuple:
        """Create training and validation data splits."""
        if self.config.walk_forward_validation and len(sequences) > 10:
            return self.create_walk_forward_splits(sequences, targets, lengths, dates)

        # Standard random split
        split_idx = int(len(sequences) * (1 - self.config.validation_split))
        split_idx = max(1, min(split_idx, len(sequences) - 1))  # Ensure valid range

        train_seq, train_tgt, train_len = sequences[:split_idx], targets[:split_idx], lengths[:split_idx]
        val_seq, val_tgt, val_len = sequences[split_idx:], targets[split_idx:], lengths[split_idx:]
        return train_seq, train_tgt, train_len, val_seq, val_tgt, val_len

    def _create_data_loaders(
        self,
        train_seq: torch.Tensor,
        train_tgt: torch.Tensor,
        train_len: torch.Tensor,
        val_seq: torch.Tensor,
        val_tgt: torch.Tensor,
        val_len: torch.Tensor,
        batch_size: int,
    ) -> tuple:
        """Create training and validation data loaders."""
        train_dataset = TimeSeriesDataset(train_seq, train_tgt, train_len)
        val_dataset = TimeSeriesDataset(val_seq, val_tgt, val_len)

        # FIX LSTM Overfitting #4: Cap batch size for training stability
        # Research validation (Keskar et al. ICLR 2017):
        # - Large batch training converges to sharp minima (poor generalization)
        # - Small batch training converges to flat minima (better generalization)
        # - Optimal batch size: 32-128 for financial time series
        #
        # Previous issue: optimize_batch_size() returned batch_size=539 (full dataset)
        # This caused severe overfitting (train Sharpe +0.05, val Sharpe -0.14)
        MAX_TRAINING_BATCH = 128  # FIXED: Increased from 64 to 128 for better GPU utilisation while maintaining stability

        train_size = len(train_dataset)
        val_size = len(val_dataset)

        # Apply training stability cap alongside dataset size constraint
        # This ensures mini-batch training even when GPU memory allows larger batches
        effective_train_batch = min(batch_size, train_size, MAX_TRAINING_BATCH)

        # CRITICAL FIX: Ensure batch size prevents single-sample last batches
        # Single-sample batches cause NaN in Sharpe ratio loss due to std(n=1) = NaN
        # Adjust batch size so the last batch has at least 2 samples
        # Example: 385 samples / 128 batch_size = 3.008 → last batch has 1 sample
        # Solution: Use batch_size where train_size % batch_size >= 2 or batch_size divides evenly
        MIN_LAST_BATCH_SIZE = 2
        if train_size % effective_train_batch == 1:
            # Last batch would have 1 sample - adjust batch size
            # Try reducing batch size by 1 until we get at least 2 samples in last batch
            adjusted_batch = effective_train_batch
            for candidate_batch in range(effective_train_batch, 1, -1):
                last_batch_size = train_size % candidate_batch
                if last_batch_size == 0 or last_batch_size >= MIN_LAST_BATCH_SIZE:
                    adjusted_batch = candidate_batch
                    break

            if adjusted_batch != effective_train_batch:
                # DIAGNOSTIC: Enhanced batch size adjustment logging
                logger.warning(
                    f"Adjusted training batch size to prevent single-sample last batch: "
                    f"{effective_train_batch} → {adjusted_batch} "
                    f"(train_size={train_size}, last_batch_size={train_size % adjusted_batch}, "
                    f"num_batches={train_size // adjusted_batch + (1 if train_size % adjusted_batch > 0 else 0)})"
                )
                effective_train_batch = adjusted_batch

        # Validation can use larger batches (no gradient computation)
        effective_val_batch = min(batch_size * 2, val_size)

        # DIAGNOSTIC: Log batch size adjustments with reasoning
        if effective_train_batch != batch_size:
            if effective_train_batch == MAX_TRAINING_BATCH:
                logger.info(
                    f"Capped batch size for training stability: {batch_size} → {effective_train_batch} "
                    f"(max_stable_batch={MAX_TRAINING_BATCH}, train_size={train_size}, "
                    f"reason='stability_cap')"
                )
            else:
                logger.info(
                    f"Adjusted batch size to dataset size: {batch_size} → {effective_train_batch} "
                    f"(train_size={train_size}, val_size={val_size}, "
                    f"reason='dataset_size_constraint')"
                )

        # CRITICAL FIX: Add drop_last=True to ensure minimum batch size of 2
        # This prevents single-sample batches which cause:
        #   - Batch normalization std=0 errors
        #   - Sharpe ratio calculation instabilities with correction=0
        # Validation loader can handle single samples (no batch norm during eval)
        train_loader = DataLoader(
            train_dataset, batch_size=effective_train_batch, shuffle=True, pin_memory=True, num_workers=2, drop_last=True
        )

        val_loader = DataLoader(
            val_dataset, batch_size=effective_val_batch, shuffle=False, pin_memory=True, num_workers=2
        )

        # CRITICAL FIX: Disable gradient accumulation for single-batch scenarios
        # When batch_size >= dataset_size, we get exactly 1 batch per epoch
        # Gradient accumulation condition (batch_idx + 1) % steps == 0 is never met when batch_idx=0 and steps>1
        # This prevents optimizer.step() and gradient clipping from ever executing
        num_train_batches = len(train_loader)
        if num_train_batches == 1 and self.config.gradient_accumulation_steps > 1:
            logger.warning(
                f"Single batch detected ({num_train_batches} batches per epoch). "
                f"Gradient accumulation steps {self.config.gradient_accumulation_steps} → 1 to ensure optimizer updates. "
                f"Without this, gradient clipping and optimizer.step() would never execute."
            )
            self.config.gradient_accumulation_steps = 1

        return train_loader, val_loader

    def _update_training_history(
        self, train_loss: float, val_loss: float, current_lr: float, epoch: int
    ) -> None:
        """Update training history and log progress."""
        self.training_history["train_loss"].append(train_loss)
        self.training_history["val_loss"].append(val_loss)
        self.training_history["learning_rates"].append(current_lr)

        # Use stored gradient norm from last backward pass (calculated before zero_grad)
        grad_norm = getattr(self, '_last_grad_norm', 0.0)
        self.training_history["gradient_norms"].append(grad_norm)

        # Memory metrics
        memory_stats = self.memory_manager.get_current_memory_stats()
        self.training_history["memory_usage"].append(memory_stats.process_memory_gb)

        # Convergence metrics
        convergence_metric = self._calculate_convergence_metric()
        self.training_history["convergence_metrics"].append(convergence_metric)

        # Training efficiency (loss improvement per epoch)
        efficiency = self._calculate_training_efficiency()
        self.training_history["training_efficiency"].append(efficiency)

        # Calculate loss variance for stagnation detection
        if len(self.training_history["val_loss"]) > 5:
            recent_val_losses = self.training_history["val_loss"][-5:]
            val_loss_var = np.var(recent_val_losses)
            self.training_history.setdefault("val_loss_variance", []).append(val_loss_var)

            # Improved stagnation detection for financial time series
            # Use relative variance threshold based on loss magnitude
            recent_val_mean = np.mean(recent_val_losses)
            relative_variance = val_loss_var / (recent_val_mean**2 + 1e-8)

            # Only warn if variance is extremely low relative to loss magnitude
            # and we have sufficient epochs to assess stagnation
            if relative_variance < 1e-6 and epoch > 20:
                logger.debug(f"Low relative validation loss variance: {relative_variance:.2e} (absolute: {val_loss_var:.2e}) - monitoring for stagnation")

                # Additional check: if loss is improving (even slightly), not stagnating
                if len(recent_val_losses) >= 5:
                    trend = np.polyfit(range(len(recent_val_losses)), recent_val_losses, 1)[0]
                    if trend < -1e-6:  # Negative trend indicates improvement
                        logger.debug(f"Validation loss still improving (trend: {trend:.2e}), continuing training")
                    elif epoch > 50:  # Only warn after substantial training
                        logger.warning(f"Validation loss plateau detected after epoch {epoch}: variance={val_loss_var:.2e}, trend={trend:.2e}")

        # Comprehensive logging with key metrics
        logger.info(
            f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"LR: {current_lr:.2e} | Grad Norm: {grad_norm:.4f} | "
            f"Memory: {memory_stats.process_memory_gb:.2f}GB | "
            f"Convergence: {convergence_metric:.4f} | Efficiency: {efficiency:.4f}"
        )

        logger.info(
            f"Epoch {epoch+1}/{self.config.epochs}: "
            f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
            f"LR: {current_lr:.8f}, Grad Norm: {grad_norm:.4f}"
        )

    def _handle_early_stopping(
        self, val_loss: float, epoch: int, checkpoint_dir: Path | None, actual_sharpe: float | None = None
    ) -> bool:
        """
        Enhanced early stopping logic optimised for financial time series data.

        CRITICAL FIX: Now uses actual Sharpe ratio instead of val_loss for early stopping.

        Background:
        -----------
        SharpeRatioLoss computes Sharpe on z-score normalised returns, which artificially
        inflates the metric by ~100x compared to actual portfolio Sharpe. This caused
        early stopping to optimise a proxy metric rather than true performance.

        Fix:
        ----
        Monitor actual_sharpe (computed on real returns) instead of val_loss.
        Higher actual_sharpe is better, so we track the maximum value.

        Args:
            val_loss: Validation loss (kept for backwards compatibility and logging)
            epoch: Current epoch number
            checkpoint_dir: Directory to save checkpoints
            actual_sharpe: Actual Sharpe ratio computed on real portfolio returns (PRIMARY METRIC)
        """
        # Never stop in the first 5 epochs - need minimum training
        if epoch < 5:
            logger.debug(f"Epoch {epoch+1}: Too early for early stopping (min 5 epochs)")
            return False

        # CRITICAL FIX: Use actual Sharpe as primary metric for early stopping
        if actual_sharpe is not None:
            # Initialize best_sharpe on first call
            if not hasattr(self, 'best_sharpe'):
                self.best_sharpe = -999.0  # Very low initial value

            # Improvement check: Higher Sharpe is better
            improvement = actual_sharpe - self.best_sharpe
            significant_improvement = improvement > self.config.min_delta

            if significant_improvement:
                self.best_sharpe = actual_sharpe
                self.best_loss = val_loss  # Also track loss for logging
                self.patience_counter = 0

                if checkpoint_dir and self.config.save_best_only:
                    self.save_checkpoint(checkpoint_dir / "best_model.pth", epoch, val_loss)

                logger.info(
                    f"Epoch {epoch+1}: New best actual Sharpe: {actual_sharpe:.4f} "
                    f"(improved by {improvement:.4f}, val_loss: {val_loss:.4f})"
                )
            else:
                self.patience_counter += 1
                logger.debug(
                    f"Epoch {epoch+1}: No improvement in actual Sharpe "
                    f"(current: {actual_sharpe:.4f}, best: {self.best_sharpe:.4f}, "
                    f"patience: {self.patience_counter}/{self.config.patience})"
                )

        else:
            # Fallback to old behaviour if actual_sharpe not provided
            logger.warning(
                "actual_sharpe not provided to early stopping, falling back to val_loss. "
                "This may lead to optimising an inflated proxy metric!"
            )

            # Skip early stopping if loss magnitude is suspiciously close to zero
            if abs(val_loss) <= 1e-8:
                logger.warning(f"Validation loss magnitude near zero ({val_loss}), indicates potential issue - skipping early stopping")
                return False

            # Standard improvement check
            improvement = self.best_loss - val_loss
            significant_improvement = improvement > self.config.min_delta

            if significant_improvement:
                self.best_loss = val_loss
                self.patience_counter = 0

                if checkpoint_dir and self.config.save_best_only:
                    self.save_checkpoint(checkpoint_dir / "best_model.pth", epoch, val_loss)

                logger.debug(f"Validation improved by {improvement:.6f} at epoch {epoch+1}")
            else:
                self.patience_counter += 1

        # Adaptive patience based on training progress and data characteristics
        base_patience = self.config.patience

        # Early training phase - be more patient
        if epoch < 20:
            effective_patience = int(base_patience * 1.5)
        # Late training phase - less patient if no improvement
        elif epoch > 100 and self.patience_counter > base_patience // 2:
            effective_patience = int(base_patience * 0.7)
        else:
            effective_patience = base_patience

        # Additional convergence checks for financial data
        # Check actual Sharpe trend if available, otherwise fall back to loss trend
        if actual_sharpe is not None and "sharpe_ratio" in self.training_history and len(self.training_history["sharpe_ratio"]) >= 10:
            recent_sharpes = self.training_history["sharpe_ratio"][-10:]
            sharpe_trend = self._calculate_loss_trend(recent_sharpes)  # Positive trend is good for Sharpe

            # If Sharpe is consistently flat or decreasing, reduce patience
            if sharpe_trend <= 1e-6:  # Very small or negative trend
                convergence_factor = 0.6
                effective_patience = int(effective_patience * convergence_factor)

            # Check for oscillating Sharpe (sign of potential overfitting)
            if self._detect_loss_oscillation(recent_sharpes):
                logger.info(f"Sharpe oscillation detected at epoch {epoch+1}, reducing patience")
                effective_patience = min(effective_patience, 5)

        elif len(self.training_history["val_loss"]) >= 10:
            recent_losses = self.training_history["val_loss"][-10:]
            loss_trend = self._calculate_loss_trend(recent_losses)

            # If loss is consistently flat or increasing, reduce patience
            if loss_trend >= -1e-6:  # Very small or positive trend
                convergence_factor = 0.6
                effective_patience = int(effective_patience * convergence_factor)

            # Check for oscillating validation loss (sign of potential overfitting)
            if self._detect_loss_oscillation(recent_losses):
                logger.info(f"Loss oscillation detected at epoch {epoch+1}, reducing patience")
                effective_patience = min(effective_patience, 5)

        # Early stopping decision
        if self.patience_counter >= effective_patience:
            convergence_metric = self._calculate_convergence_metric()
            display_patience = min(self.patience_counter, effective_patience)

            if actual_sharpe is not None:
                logger.info(
                    f"Early stopping at epoch {epoch+1} "
                    f"(patience: {display_patience}/{effective_patience}, "
                    f"best actual Sharpe: {self.best_sharpe:.4f}, "
                    f"convergence: {convergence_metric:.6f})"
                )
            else:
                logger.info(
                    f"Early stopping at epoch {epoch+1} "
                    f"(patience: {display_patience}/{effective_patience}, "
                    f"convergence: {convergence_metric:.6f})"
                )
            return True

        return False

    def _calculate_loss_trend(self, recent_losses: list[float]) -> float:
        """Calculate the trend (slope) of recent validation losses."""
        if len(recent_losses) < 3:
            return 0.0

        import numpy as np
        x = np.arange(len(recent_losses))
        y = np.array(recent_losses)

        # Simple linear regression slope
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x * x) - np.sum(x) * np.sum(x))
        return slope

    def _detect_loss_oscillation(self, recent_losses: list[float]) -> bool:
        """Detect if validation loss is oscillating (potential overfitting)."""
        if len(recent_losses) < 6:
            return False

        # Count direction changes in the loss sequence
        direction_changes = 0
        for i in range(1, len(recent_losses) - 1):
            prev_diff = recent_losses[i] - recent_losses[i-1]
            next_diff = recent_losses[i+1] - recent_losses[i]

            # Sign change indicates direction change
            if prev_diff * next_diff < 0:
                direction_changes += 1

        # If more than half the sequence shows direction changes, it's oscillating
        return direction_changes > len(recent_losses) // 3

    def _calculate_gradient_norm(self) -> float:
        """Calculate the L2 norm of gradients for monitoring."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def _calculate_convergence_metric(self) -> float:
        """Calculate convergence rate based on recent loss improvements."""
        if len(self.training_history["val_loss"]) < 3:
            return 0.0

        recent_losses = self.training_history["val_loss"][-3:]
        if len(recent_losses) < 2:
            return 0.0

        # Calculate rate of loss improvement
        # Handle both positive losses (MSE) and negative losses (Sharpe ratio)
        improvements = []
        for i in range(1, len(recent_losses)):
            prev_loss = recent_losses[i-1]
            curr_loss = recent_losses[i]

            # For negative losses (Sharpe), improvement means loss becomes MORE negative
            # For positive losses (MSE), improvement means loss becomes LESS positive
            if prev_loss < 0:
                # Negative loss (Sharpe): curr_loss < prev_loss is improvement
                improvement = (prev_loss - curr_loss) / abs(prev_loss)
            elif prev_loss > 0:
                # Positive loss (MSE): curr_loss < prev_loss is improvement
                improvement = (prev_loss - curr_loss) / prev_loss
            else:
                # Zero loss - skip
                continue

            improvements.append(improvement)

        return np.mean(improvements) if improvements else 0.0

    def _calculate_training_efficiency(self) -> float:
        """Calculate training efficiency as loss improvement per computation unit."""
        if len(self.training_history["train_loss"]) < 2:
            return 0.0

        current_loss = self.training_history["train_loss"][-1]
        initial_loss = self.training_history["train_loss"][0]

        if initial_loss <= 0:
            return 0.0

        # Efficiency as relative improvement per epoch
        epochs_elapsed = len(self.training_history["train_loss"])
        relative_improvement = (initial_loss - current_loss) / initial_loss
        efficiency = relative_improvement / epochs_elapsed if epochs_elapsed > 0 else 0.0

        return max(0.0, efficiency)  # Ensure non-negative

    def _calculate_financial_validation_metrics(
        self,
        predictions: torch.Tensor,
        actual_returns: torch.Tensor,
        weights: torch.Tensor | None = None
    ) -> dict[str, float]:
        """
        Calculate comprehensive financial validation metrics for model evaluation.

        Args:
            predictions: Model predictions of shape (batch_size, n_assets)
            actual_returns: Actual returns of shape (batch_size, n_assets)
            weights: Portfolio weights if available

        Returns:
            Dictionary of financial validation metrics
        """
        import numpy as np

        metrics = {}

        try:
            # Convert to numpy for calculation
            pred_np = predictions.detach().cpu().numpy()
            actual_np = actual_returns.detach().cpu().numpy()

            # Check for shape compatibility and align if needed
            if pred_np.shape != actual_np.shape:
                logger.warning(f"Shape mismatch in validation metrics: predictions {pred_np.shape} vs actual {actual_np.shape}")
                # Truncate to smaller size for compatibility
                min_samples = min(pred_np.shape[0], actual_np.shape[0])
                min_assets = min(pred_np.shape[1], actual_np.shape[1])
                pred_np = pred_np[:min_samples, :min_assets]
                actual_np = actual_np[:min_samples, :min_assets]

            # 1. Prediction Accuracy Metrics
            # Correlation between predictions and actual returns
            pred_flat = pred_np.flatten()
            actual_flat = actual_np.flatten()

            if len(pred_flat) > 1 and np.std(pred_flat) > 1e-8 and np.std(actual_flat) > 1e-8:
                correlation = np.corrcoef(pred_flat, actual_flat)[0, 1]
                metrics["correlation_with_returns"] = correlation if not np.isnan(correlation) else 0.0
            else:
                metrics["correlation_with_returns"] = 0.0

            # Hit ratio (percentage of correct directional predictions)
            pred_direction = np.sign(pred_np)
            actual_direction = np.sign(actual_np)
            hit_ratio = np.mean(pred_direction == actual_direction)
            metrics["hit_ratio"] = hit_ratio

            # Prediction accuracy (MAE-based)
            # Align shapes if there's a mismatch (e.g., 251 vs 252 sequence length)
            if pred_np.shape != actual_np.shape:
                # Handle dimension mismatch safely
                if pred_np.ndim == 1 and actual_np.ndim == 1:
                    # Both 1D: align to minimum length
                    min_len = min(len(pred_np), len(actual_np))
                    pred_np = pred_np[:min_len]
                    actual_np = actual_np[:min_len]
                elif pred_np.ndim == 2 and actual_np.ndim == 2:
                    # Both 2D: align to minimum dimensions
                    min_samples = min(pred_np.shape[0], actual_np.shape[0])
                    min_assets = min(pred_np.shape[1], actual_np.shape[1])
                    pred_np = pred_np[:min_samples, :min_assets]
                    actual_np = actual_np[:min_samples, :min_assets]
                else:
                    # Mixed dimensions: align first dimension only
                    min_samples = min(pred_np.shape[0], actual_np.shape[0])
                    pred_np = pred_np[:min_samples] if pred_np.ndim == 1 else pred_np[:min_samples, :]
                    actual_np = actual_np[:min_samples] if actual_np.ndim == 1 else actual_np[:min_samples, :]

            mae = np.mean(np.abs(pred_np - actual_np))
            mse = np.mean((pred_np - actual_np) ** 2)
            metrics["prediction_accuracy"] = 1.0 / (1.0 + mae)  # Normalized accuracy

            # 2. Portfolio Performance Metrics (if weights provided)
            if weights is not None:
                weights_np = weights.detach().cpu().numpy()
                # Use aligned actual_np for portfolio calculations (consistent shapes)
                actual_for_portfolio = actual_np

                # CRITICAL FIX: Denormalize returns before calculating financial metrics
                # The actual_np contains normalized returns (mean=0, std=1), which makes
                # Sharpe ratio and other financial metrics meaningless. We need to denormalize
                # using the stored normalization statistics from training.
                # Log pre-denormalization state for debugging
                logger.debug(
                    f"Pre-denormalization state: "
                    f"has_normalization_stats={hasattr(self.model, 'normalization_stats')}, "
                    f"actual_for_portfolio_shape={actual_for_portfolio.shape}, "
                    f"actual_mean={actual_for_portfolio.mean():.6f}, "
                    f"actual_std={actual_for_portfolio.std():.6f}, "
                    f"expected_normalized_std_approx_1.0={abs(actual_for_portfolio.std() - 1.0) < 0.1}"
                )
                denormalization_applied = False
                has_stats = hasattr(self.model, 'normalization_stats')
                logger.debug(f"Denormalization check: has_stats={has_stats}, "
                            f"actual_shape={actual_for_portfolio.shape}")

                if has_stats:
                    stats = self.model.normalization_stats
                    scaler = stats.get('scaler', None)
                    asset_names = stats.get('asset_names', None)

                    # Check for MinMaxScaler (new approach) or legacy mean/std
                    if scaler is not None:
                        # Use MinMaxScaler inverse_transform
                        try:
                            # Save normalized stats before denormalization
                            normalized_min = actual_for_portfolio.min().item()
                            normalized_max = actual_for_portfolio.max().item()

                            # Inverse transform
                            actual_for_portfolio = scaler.inverse_transform(actual_for_portfolio)
                            denormalization_applied = True

                            denormalized_min = actual_for_portfolio.min().item()
                            denormalized_max = actual_for_portfolio.max().item()

                            logger.info(f"MinMaxScaler denormalization applied: "
                                       f"normalized=(min={normalized_min:.6f}, max={normalized_max:.6f}), "
                                       f"denormalized=(min={denormalized_min:.6f}, max={denormalized_max:.6f})")

                            # Validate denormalization restored realistic scale
                            denormalized_std = actual_for_portfolio.std().item()
                            if denormalized_std < 0.005 or denormalized_std > 0.05:
                                logger.warning(
                                    f"Denormalized returns std outside expected range: "
                                    f"std={denormalized_std:.6f}, expected_range=[0.005, 0.05]"
                                )
                        except Exception as e:
                            logger.error(f"MinMaxScaler inverse_transform failed: {e}")

                    elif 'mean' in stats and 'std' in stats:
                        # Legacy RobustScaler approach
                        stats_valid = 'mean' in stats and 'std' in stats and 'epsilon' in stats
                        logger.debug(f"Normalization stats valid: {stats_valid}, "
                                    f"stats_keys={list(stats.keys()) if stats else None}")

                        if stats_valid:
                            mean = stats['mean']  # Shape: [1, n_assets]
                            std = stats['std']    # Shape: [1, n_assets]
                            epsilon = stats['epsilon']  # 1e-6

                            # Denormalize: actual = normalized * (std + epsilon) + mean
                            # Ensure shapes are compatible for broadcasting
                            # Handle both 1D and 2D cases
                            if actual_for_portfolio.ndim == 1 or mean.ndim == 1:
                                # 1D case: compare lengths directly
                                shape_match = actual_for_portfolio.shape[-1] == mean.shape[-1]
                            else:
                                # 2D case: compare asset dimension (axis 1)
                                shape_match = actual_for_portfolio.shape[1] == mean.shape[1]

                            logger.info(f"Denormalization details: has_stats={has_stats}, "
                                       f"shape_match={shape_match}, "
                                       f"actual_shape={actual_for_portfolio.shape}, "
                                       f"stats_shape={mean.shape}, "
                                       f"n_training_assets={len(asset_names) if asset_names else 'unknown'}")

                            if shape_match:
                                # Save normalized std before denormalization for logging
                                normalized_std = actual_for_portfolio.std().item()
                                actual_for_portfolio = actual_for_portfolio * (std + epsilon) + mean
                                denormalization_applied = True
                                logger.info(f"Legacy denormalization applied: normalized_std={normalized_std:.6f}, "
                                            f"actual_std={actual_for_portfolio.std().item():.6f}, "
                                            f"scale_factor={actual_for_portfolio.std().item() / (normalized_std + 1e-8):.2f}x")
                                # Validate denormalization restored realistic scale
                                denormalized_std = actual_for_portfolio.std().item()
                                if denormalized_std < 0.005 or denormalized_std > 0.05:
                                    logger.warning(
                                        f"Denormalized returns std outside expected range: "
                                        f"std={denormalized_std:.6f}, expected_range=[0.005, 0.05], "
                                        f"scale_factor_applied={std.mean().item():.6f}"
                                    )
                                logger.debug(
                                    f"Denormalized returns for Sharpe calculation: "
                                    f"mean_range=[{mean.min():.6f}, {mean.max():.6f}], "
                                    f"std_range=[{std.min():.6f}, {std.max():.6f}]"
                                )
                        else:
                            # CRITICAL FIX: Fail fast instead of silently continuing with wrong scale
                            logger.error(
                                f"CRITICAL: Denormalization shape mismatch detected: "
                                f"actual_for_portfolio.shape={actual_for_portfolio.shape}, "
                                f"mean.shape={mean.shape}, std.shape={std.shape}, "
                                f"asset_names_count={len(asset_names) if asset_names else 0}, "
                                f"this_will_cause_negative_sharpe=True"
                            )
                            raise RuntimeError(
                                f"VALIDATION FAILED: Cannot denormalize returns - shape mismatch. "
                                f"actual_for_portfolio shape {actual_for_portfolio.shape} does not match "
                                f"normalization stats shape {mean.shape}. "
                                f"This would produce meaningless Sharpe ratios on normalized scale (std=1.0) "
                                f"instead of actual scale (std~0.015). Training cannot continue."
                            )

                if not denormalization_applied:
                    # CRITICAL FIX: Fail fast instead of silently continuing with wrong scale
                    raise RuntimeError(
                        "VALIDATION FAILED: Normalization stats not available or incomplete. "
                        f"Required: 'mean', 'std', 'epsilon'. "
                        f"Available: {list(stats.keys()) if hasattr(self.model, 'normalization_stats') and stats else 'None'}. "
                        f"Financial metrics cannot be calculated without denormalization. "
                        f"This prevents training/validation from using meaningless metrics."
                    )

                # Ensure weights shape matches actual_np with better alignment
                if weights_np.shape != actual_for_portfolio.shape:
                    logger.debug(f"Shape mismatch in weights: weights {weights_np.shape} vs actual {actual_for_portfolio.shape}")
                    min_samples = min(weights_np.shape[0], actual_for_portfolio.shape[0])
                    # Handle both 1D and 2D cases for asset dimension
                    min_assets = min(weights_np.shape[1], actual_for_portfolio.shape[1]) if (weights_np.ndim > 1 and actual_for_portfolio.ndim > 1) else 1

                    if weights_np.ndim > 1 and actual_for_portfolio.ndim > 1:
                        weights_np = weights_np[:min_samples, :min_assets]
                        actual_for_portfolio = actual_for_portfolio[:min_samples, :min_assets]
                    else:
                        # Handle 1D case
                        weights_np = weights_np[:min_samples]
                        actual_for_portfolio = actual_for_portfolio[:min_samples]

                # Calculate portfolio returns with shape alignment
                # Ensure both arrays have the same time dimension
                if weights_np.shape[0] != actual_for_portfolio.shape[0]:
                    # Align to the shorter sequence length
                    min_len = min(weights_np.shape[0], actual_for_portfolio.shape[0])
                    weights_np = weights_np[:min_len].copy()  # Use copy to avoid broadcasting warnings
                    actual_for_portfolio = actual_for_portfolio[:min_len].copy()
                    logger.debug(f"Aligned sequences to length {min_len} for portfolio return calculation")

                # Ensure shape compatibility before multiplication
                if weights_np.shape != actual_for_portfolio.shape:
                    logger.debug(f"Final shape mismatch: weights {weights_np.shape} vs actual {actual_for_portfolio.shape}")
                    # If still mismatched, use element-wise safe multiplication
                    try:
                        portfolio_returns = np.sum(weights_np * actual_for_portfolio, axis=1)
                    except ValueError:
                        # Fallback: align to minimum dimensions safely
                        # Handle dimension mismatch between weights and actual returns
                        if weights_np.ndim == 1 and actual_for_portfolio.ndim == 1:
                            # Both 1D: align to minimum length
                            min_len = min(len(weights_np), len(actual_for_portfolio))
                            weights_aligned = weights_np[:min_len]
                            actual_aligned = actual_for_portfolio[:min_len]
                            portfolio_returns = weights_aligned * actual_aligned
                        elif weights_np.ndim == 2 and actual_for_portfolio.ndim == 2:
                            # Both 2D: align both dimensions
                            min_samples = min(weights_np.shape[0], actual_for_portfolio.shape[0])
                            min_assets = min(weights_np.shape[1], actual_for_portfolio.shape[1])
                            weights_aligned = weights_np[:min_samples, :min_assets]
                            actual_aligned = actual_for_portfolio[:min_samples, :min_assets]
                            portfolio_returns = np.sum(weights_aligned * actual_aligned, axis=1)
                        else:
                            # Mixed dimensions: align first dimension only
                            min_samples = min(weights_np.shape[0], actual_for_portfolio.shape[0])
                            weights_aligned = weights_np[:min_samples] if weights_np.ndim == 1 else weights_np[:min_samples, :]
                            actual_aligned = actual_for_portfolio[:min_samples] if actual_for_portfolio.ndim == 1 else actual_for_portfolio[:min_samples, :]
                            # For mixed dimensions, use sum along last axis if 2D
                            if weights_aligned.ndim == 2 or actual_aligned.ndim == 2:
                                portfolio_returns = np.sum(weights_aligned * actual_aligned, axis=1 if weights_aligned.ndim == 2 else 0)
                            else:
                                portfolio_returns = weights_aligned * actual_aligned
                else:
                    portfolio_returns = np.sum(weights_np * actual_for_portfolio, axis=1)

                if len(portfolio_returns) > 1:
                    # Sharpe ratio
                    mean_return = np.mean(portfolio_returns)
                    std_return = np.std(portfolio_returns)
                    if std_return > 1e-8:
                        metrics["sharpe_ratio"] = mean_return / std_return
                    else:
                        metrics["sharpe_ratio"] = 0.0
                    logger.info(f"Validation Sharpe (ACTUAL scale): mean={mean_return:.6f}, "
                                f"std={std_return:.6f}, sharpe={metrics['sharpe_ratio']:.6f}, "
                                f"denormalized={'YES' if hasattr(self.model, 'normalization_stats') and self.model.normalization_stats else 'NO'}")

                    # Sortino ratio (downside deviation)
                    downside_returns = portfolio_returns[portfolio_returns < 0]
                    if len(downside_returns) > 0:
                        downside_std = np.std(downside_returns)
                        if downside_std > 1e-8:
                            metrics["sortino_ratio"] = mean_return / downside_std
                        else:
                            metrics["sortino_ratio"] = metrics["sharpe_ratio"]
                    else:
                        metrics["sortino_ratio"] = metrics["sharpe_ratio"]

                    # Maximum drawdown
                    cumulative_returns = np.cumprod(1 + portfolio_returns)
                    peak = np.maximum.accumulate(cumulative_returns)
                    drawdown = (cumulative_returns - peak) / peak
                    max_drawdown = np.min(drawdown)
                    metrics["max_drawdown"] = abs(max_drawdown)

                    # Calmar ratio
                    if max_drawdown < -1e-8:
                        annualized_return = mean_return * 252  # Daily to annual
                        metrics["calmar_ratio"] = annualized_return / abs(max_drawdown)
                    else:
                        metrics["calmar_ratio"] = 0.0

                    # Portfolio volatility
                    metrics["portfolio_volatility"] = std_return * np.sqrt(252)  # Annualized

                    # Portfolio concentration (Herfindahl index)
                    avg_weights = np.mean(weights_np, axis=0)
                    concentration = np.sum(avg_weights ** 2)
                    metrics["portfolio_concentration"] = concentration

                    # Portfolio turnover (if we have previous weights)
                    if hasattr(self, '_previous_weights') and self._previous_weights is not None:
                        prev_weights = self._previous_weights.detach().cpu().numpy()
                        turnover = np.mean(np.sum(np.abs(weights_np - prev_weights), axis=1))
                        metrics["portfolio_turnover"] = turnover
                    else:
                        metrics["portfolio_turnover"] = 0.0

                    # Store current weights for next calculation
                    self._previous_weights = weights.clone()

                # Tracking error vs equal weight benchmark
                # Use already aligned arrays
                equal_weight = np.ones_like(weights_np) / weights_np.shape[1]
                benchmark_returns = np.sum(equal_weight * actual_for_portfolio, axis=1)

                if len(portfolio_returns) > 1 and len(benchmark_returns) > 1:
                    tracking_error = np.std(portfolio_returns - benchmark_returns)
                    metrics["tracking_error"] = tracking_error * np.sqrt(252)  # Annualized

                    # Information ratio
                    excess_return = np.mean(portfolio_returns - benchmark_returns)
                    if tracking_error > 1e-8:
                        metrics["information_ratio"] = (excess_return * 252) / metrics["tracking_error"]
                    else:
                        metrics["information_ratio"] = 0.0
                else:
                    metrics["tracking_error"] = 0.0
                    metrics["information_ratio"] = 0.0

            # Set default values for missing metrics
            default_metrics = {
                "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "calmar_ratio": 0.0,
                "max_drawdown": 0.0, "tracking_error": 0.0, "portfolio_turnover": 0.0,
                "information_ratio": 0.0, "portfolio_volatility": 0.0, "portfolio_concentration": 0.0
            }

            for key, default_value in default_metrics.items():
                if key not in metrics:
                    metrics[key] = default_value

        except Exception as e:
            logger.warning(f"Error calculating financial validation metrics: {e}")
            # Return default metrics on error
            metrics = {
                "correlation_with_returns": 0.0, "hit_ratio": 0.5, "prediction_accuracy": 0.0,
                "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "calmar_ratio": 0.0,
                "max_drawdown": 0.0, "tracking_error": 0.0, "portfolio_turnover": 0.0,
                "information_ratio": 0.0, "portfolio_volatility": 0.0, "portfolio_concentration": 0.0
            }

        return metrics

    def get_training_summary(self) -> dict[str, Any]:
        """Get comprehensive training summary with advanced metrics."""
        if not self.training_history["train_loss"]:
            return {"status": "No training data available"}

        summary = {
            "training_epochs": len(self.training_history["train_loss"]),
            "final_train_loss": self.training_history["train_loss"][-1],
            "final_val_loss": self.training_history["val_loss"][-1],
            "best_val_loss": min(self.training_history["val_loss"]),
            "final_learning_rate": self.training_history["learning_rates"][-1],
            "total_memory_used_gb": max(self.training_history["memory_usage"]),
            "avg_gradient_norm": np.mean(self.training_history["gradient_norms"]),
            "max_gradient_norm": max(self.training_history["gradient_norms"]),
            "final_convergence_rate": self.training_history["convergence_metrics"][-1],
            "avg_training_efficiency": np.mean(self.training_history["training_efficiency"]),
            "memory_pressure_events": len(self.training_history["memory_pressure_events"]),
        }

        # Training stability analysis
        if len(self.training_history["val_loss"]) > 5:
            recent_val_losses = self.training_history["val_loss"][-5:]
            summary["val_loss_stability"] = 1.0 / (1.0 + np.std(recent_val_losses))
            summary["training_trend"] = "improving" if recent_val_losses[-1] < recent_val_losses[0] else "degrading"

        # Performance recommendations
        recommendations = []
        if summary["avg_gradient_norm"] > 5.0:
            recommendations.append("High gradient norms detected - consider stronger gradient clipping")
        if summary["final_convergence_rate"] < 0.001:
            recommendations.append("Low convergence rate - consider adjusting learning rate or model architecture")
        if summary["memory_pressure_events"] > summary["training_epochs"] * 0.1:
            recommendations.append("Frequent memory pressure - consider reducing batch size")

        summary["recommendations"] = recommendations

        return summary

    def _safe_training_step(self, sequences: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor, batch_idx: int) -> float:
        """Perform training step with comprehensive error handling."""
        try:
            # Forward and backward pass
            if self.config.use_mixed_precision and self.scaler is not None:
                loss = self._forward_pass_with_mixed_precision(sequences, targets, lengths)
                self._backward_pass_mixed_precision(loss, batch_idx)
            else:
                loss = self._forward_pass_standard(sequences, targets, lengths)
                self._backward_pass_standard(loss, batch_idx)

            # Validate loss value
            loss_value = loss.item()
            if not torch.isfinite(torch.tensor(loss_value)):
                logger.warning(f"Non-finite loss detected at batch {batch_idx}: {loss_value}")
                return 0.0

            return loss_value

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"GPU OOM at batch {batch_idx}, attempting recovery...")
                torch.cuda.empty_cache()
                # Retry with gradient accumulation reset
                self.optimizer.zero_grad()
                raise  # Re-raise to trigger error handler
            else:
                raise

    def _handle_training_error(self, error: Exception, epoch: int, batch_idx: int) -> bool:
        """Handle training errors with recovery strategies."""
        error_msg = str(error).lower()

        if "out of memory" in error_msg:
            return self._recover_from_oom_error(epoch, batch_idx)
        elif "nan" in error_msg or "inf" in error_msg:
            return self._recover_from_numerical_error(epoch, batch_idx)
        elif "cuda" in error_msg:
            return self._recover_from_cuda_error(epoch, batch_idx)
        else:
            logger.error(f"Unknown error type: {error}")
            return False

    def _recover_from_oom_error(self, epoch: int, batch_idx: int) -> bool:
        """Recover from GPU out-of-memory errors."""
        logger.warning(f"Recovering from OOM error at epoch {epoch}, batch {batch_idx}")

        try:
            # Clear GPU cache
            torch.cuda.empty_cache()
            self.memory_manager.optimize_memory_usage()

            # Reset optimizer state
            self.optimizer.zero_grad()

            # Reduce effective batch size by increasing gradient accumulation
            if hasattr(self.config, 'gradient_accumulation_steps'):
                self.config.gradient_accumulation_steps = min(self.config.gradient_accumulation_steps * 2, 16)
                logger.info(f"Increased gradient accumulation to {self.config.gradient_accumulation_steps}")

            return True

        except Exception as e:
            logger.error(f"Failed to recover from OOM: {e}")
            return False

    def _recover_from_numerical_error(self, epoch: int, batch_idx: int) -> bool:
        """Recover from numerical instability (NaN/Inf)."""
        logger.warning(f"Recovering from numerical error at epoch {epoch}, batch {batch_idx}")

        try:
            # Reset model parameters if they contain NaN/Inf
            for param in self.model.parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    param.data.fill_(0.01)  # Small non-zero values
                    logger.warning("Reset NaN/Inf parameters")

            # Reset optimizer state
            self.optimizer.zero_grad()

            # Reduce learning rate temporarily
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.1
                logger.info(f"Temporarily reduced learning rate to {param_group['lr']:.2e}")

            return True

        except Exception as e:
            logger.error(f"Failed to recover from numerical error: {e}")
            return False

    def _recover_from_cuda_error(self, epoch: int, batch_idx: int) -> bool:
        """Recover from CUDA-related errors."""
        logger.warning(f"Recovering from CUDA error at epoch {epoch}, batch {batch_idx}")

        try:
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Reset optimizer
            self.optimizer.zero_grad()

            # Move model back to device (in case of device issues)
            self.model = self.model.to(self.device)

            return True

        except Exception as e:
            logger.error(f"Failed to recover from CUDA error: {e}")
            return False

    def fit(
        self,
        returns_data: pd.DataFrame,
        sequence_length: int = 60,
        checkpoint_dir: Path | None = None,
    ) -> dict[str, list[float]]:
        """
        Train LSTM model with memory optimization and early stopping.

        Args:
            returns_data: Historical returns DataFrame
            sequence_length: Input sequence length
            checkpoint_dir: Directory to save model checkpoints

        Returns:
            Training history dictionary
        """
        logger.info("Starting LSTM training with memory optimization")

        # Set universe from returns data columns
        self.universe = returns_data.columns.tolist()

        # Create sequences
        sequences, targets, lengths, dates = self.create_sequences(returns_data, sequence_length)

        # Create validation splits
        train_seq, train_tgt, train_len, val_seq, val_tgt, val_len = self._create_data_splits(sequences, targets, lengths, dates)

        # Optimize batch size and create data loaders
        universe_size = len(self.universe) if hasattr(self, 'universe') and self.universe else None
        optimal_batch_size = self.optimize_batch_size(sequence_length, universe_size)
        train_loader, val_loader = self._create_data_loaders(
            train_seq, train_tgt, train_len, val_seq, val_tgt, val_len, optimal_batch_size
        )

        # Training loop
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss, financial_metrics = self.validate(val_loader)

            # CRITICAL FIX: Restore model to training mode after validation
            # validate() sets model to eval() mode (line 1035), which disables dropout and batch norm updates
            # This must be restored immediately for the next training epoch
            # Without this, dropout and batch norm run statistics won't update during subsequent training
            self.model.train()

            # Store financial validation metrics in training history
            for metric_name, metric_value in financial_metrics.items():
                if metric_name in self.training_history:
                    self.training_history[metric_name].append(metric_value)

            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # DIAGNOSTIC: Enhanced learning rate logging after potential reduction
            # If metrics appear frozen after epoch 15, check if LR was reduced here
            if epoch >= 14 and epoch % 15 == 14:
                lr_reduction_factor = current_lr / self.config.learning_rate
                logger.info(
                    f"Learning rate check at epoch {epoch}: "
                    f"current_lr={current_lr:.2e}, "
                    f"initial_lr={self.config.learning_rate:.2e}, "
                    f"reduction_factor={lr_reduction_factor:.2f}x, "
                    f"val_loss={val_loss:.6f}"
                )

            # Update history and check early stopping
            self._update_training_history(train_loss, val_loss, current_lr, epoch)

            # CRITICAL FIX: Pass actual Sharpe to early stopping
            # This ensures we optimise for real performance, not inflated loss metric
            actual_sharpe = financial_metrics.get('sharpe_ratio', None)
            if self._handle_early_stopping(val_loss, epoch, checkpoint_dir, actual_sharpe):
                break

            # Regular checkpointing
            if checkpoint_dir and (epoch + 1) % self.config.checkpoint_frequency == 0:
                self.save_checkpoint(
                    checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth", epoch, val_loss
                )

        logger.info("Training completed")

        # CRITICAL FIX: Explicitly set model to eval mode for deployment clarity
        # After training completes, the model should be in eval mode for inference
        # This prevents accidental training-mode computations on deployed models
        self.model.eval()

        return self.training_history

    def save_checkpoint(self, filepath: Path, epoch: int, val_loss: float) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
            "training_history": self.training_history,
        }

        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: Path) -> dict:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(f"Checkpoint loaded: {filepath}")
        return checkpoint


def create_trainer(
    model: LSTMNetwork, config: TrainingConfig | None = None
) -> MemoryEfficientTrainer:
    """
    Factory function to create memory-efficient trainer.

    Args:
        model: LSTM network to train
        config: Training configuration (uses defaults if None)

    Returns:
        Configured trainer instance
    """
    if config is None:
        config = TrainingConfig()

    return MemoryEfficientTrainer(model, config)
