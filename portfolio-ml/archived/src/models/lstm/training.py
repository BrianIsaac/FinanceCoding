"""
Memory-efficient training pipeline for LSTM portfolio models.

This module implements GPU memory optimization, gradient accumulation, and mixed precision
training to handle the full S&P MidCap 400 universe within VRAM constraints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

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
    gradient_accumulation_steps: int = 4  # Effective batch size multiplier
    use_mixed_precision: bool = True  # FP16 training for memory efficiency
    target_memory_utilisation: float = 0.75  # Target 75% GPU memory usage

    # Training parameters
    learning_rate: float = 0.001  # Standard Adam learning rate
    weight_decay: float = 1e-5  # L2 regularization
    batch_size: int = 32  # Base batch size (will be optimized dynamically)
    epochs: int = 100  # Maximum training epochs
    patience: int = 15  # Early stopping patience
    min_delta: float = 1e-6  # Minimum improvement for early stopping

    # Learning rate scheduling
    lr_patience: int = 10  # ReduceLROnPlateau patience
    lr_factor: float = 0.5  # Learning rate reduction factor
    min_lr: float = 1e-6  # Minimum learning rate

    # Validation
    validation_split: float = 0.2  # Validation set size
    walk_forward_validation: bool = True  # Use temporal validation splits

    # Gradient optimization
    gradient_clip_value: float = 1.0  # Gradient clipping threshold
    adaptive_clipping: bool = False  # Disable for stability

    # Checkpointing
    save_best_only: bool = True  # Only save best model
    checkpoint_frequency: int = 5  # Save checkpoint every N epochs


class TimeSeriesDataset(Dataset):
    """Dataset for temporal sequence data with proper validation splits."""

    def __init__(
        self,
        sequences: torch.Tensor,
        targets: torch.Tensor,
        asset_ids: torch.Tensor | None = None,
    ):
        """
        Initialize time series dataset.

        Args:
            sequences: Input sequences of shape (n_samples, seq_len, n_features)
            targets: Target returns of shape (n_samples, n_features)
            asset_ids: Asset identifiers for each sample
        """
        assert len(sequences) == len(targets), "Sequences and targets must have same length"

        self.sequences = sequences
        self.targets = targets
        self.asset_ids = asset_ids

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.sequences[idx]
        target = self.targets[idx]
        return sequence, target


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
        self.criterion = SharpeRatioLoss(entropy_weight=0.001)  # Mild diversification to allow prediction-based allocation

        # Initialize mixed precision scaler
        self.scaler = GradScaler("cuda") if config.use_mixed_precision else None

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

        # Initialize mixed precision training flag
        self._unscaled_in_step = False

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
    ) -> tuple[torch.Tensor, torch.Tensor, list[pd.Timestamp]]:
        """
        Create overlapping sequences for LSTM training with temporal alignment.

        Args:
            returns_data: Historical returns DataFrame with datetime index
            sequence_length: Length of input sequences (60 days)
            prediction_horizon: Days ahead to predict (21 for monthly)

        Returns:
            Tuple of (sequences, targets, dates)
            - sequences: Input sequences of shape (n_samples, seq_len, n_assets)
            - targets: Target returns of shape (n_samples, n_assets)
            - dates: List of target dates for temporal validation
        """
        returns_array = returns_data.values
        dates = returns_data.index
        n_timesteps, _ = returns_array.shape

        sequences = []
        targets = []
        target_dates = []

        # Create overlapping sequences
        for i in range(sequence_length, n_timesteps - prediction_horizon + 1):
            # Input sequence: t-59 to t
            sequence = returns_array[i - sequence_length : i]

            # Target: next month's return (average over prediction_horizon)
            target_start = i
            target_end = min(i + prediction_horizon, n_timesteps)
            target = returns_array[target_start:target_end].mean(axis=0)

            sequences.append(sequence)
            targets.append(target)
            target_dates.append(dates[target_start])

        sequences = torch.tensor(np.array(sequences), dtype=torch.float32)
        targets = torch.tensor(np.array(targets), dtype=torch.float32)

        if len(sequences) == 0:
            raise ValueError(
                f"Insufficient data for sequence creation. Data has {n_timesteps} timesteps, "
                f"need at least {sequence_length + prediction_horizon} for sequence_length={sequence_length} "
                f"and prediction_horizon={prediction_horizon}. "
                f"Consider using longer training periods or shorter sequence lengths."
            )

        logger.info(f"Created {len(sequences)} sequences with shape {sequences.shape}")

        return sequences, targets, target_dates

    def create_walk_forward_splits(
        self,
        sequences: torch.Tensor,
        targets: torch.Tensor,
        dates: list[pd.Timestamp],
        validation_months: int = 12,  # Use last 12 months for validation
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create walk-forward validation splits to prevent look-ahead bias.

        Args:
            sequences: Input sequences
            targets: Target returns
            dates: Corresponding dates
            validation_months: Number of months for validation set

        Returns:
            Tuple of (train_sequences, train_targets, val_sequences, val_targets)
        """
        # Convert to pandas for date handling
        dates_series = pd.Series(dates)

        # Find split date (validation_months ago from end)
        end_date = dates_series.max()
        split_date = end_date - pd.DateOffset(months=validation_months)

        # Create boolean mask for training/validation split
        train_mask = dates_series <= split_date
        val_mask = dates_series > split_date

        train_sequences = sequences[train_mask]
        train_targets = targets[train_mask]
        val_sequences = sequences[val_mask]
        val_targets = targets[val_mask]

        logger.info(
            f"Train samples: {len(train_sequences)}, Validation samples: {len(val_sequences)}"
        )
        logger.info(f"Split date: {split_date.strftime('%Y-%m-%d')}")

        return train_sequences, train_targets, val_sequences, val_targets

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
        base_model_memory = self.model.get_memory_usage(batch_size, sequence_length)

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
        # Moderate batch sizes to better utilize GPU (was at 10.6% utilization)
        if universe_size <= 100:  # Small universe - large batches for GPU utilization
            universe_multiplier = 8.0  # Reduced from 16x but still aggressive
            max_batch_cap = 4096  # Reduced from 8192 but still large
        elif universe_size <= 300:  # Medium universe - good sized batches
            universe_multiplier = 6.0  # Reduced from 12x
            max_batch_cap = 2048  # Reduced from 4096
        elif universe_size <= 500:  # Large universe - moderate batches
            universe_multiplier = 4.0  # Reduced from 8x
            max_batch_cap = 1024  # Reduced from 2048
        else:  # Very large universe (679 assets) - balanced for memory
            universe_multiplier = 3.0  # Reduced from 6x but still good for GPU
            max_batch_cap = 768  # Reduced from 1536 but reasonable

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

                # Set model to training mode for backward pass compatibility
                self.model.train()

                # Simulate forward pass
                with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
                    predictions, _ = self.model(dummy_sequences)
                    loss = self.criterion(predictions, dummy_targets)

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
                del dummy_sequences, dummy_targets, predictions, loss
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

        logger.info(
            f"Optimized batch size: {best_batch_size}"
            f"\n  Actual GPU usage: {best_actual_memory:.2f}GB/{total_gpu_memory:.1f}GB ({actual_utilisation_pct:.1f}%)"
            f"\n  Theoretical estimate: {best_theoretical_memory:.2f}GB ({theoretical_utilisation_pct:.1f}%)"
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
        self, sequences: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with automatic mixed precision."""
        with autocast("cuda"):
            predictions, _ = self.model(sequences)
            return self.criterion(predictions, targets) / self.config.gradient_accumulation_steps

    def _backward_pass_mixed_precision(self, loss: torch.Tensor, batch_idx: int) -> None:
        """Backward pass with gradient scaling for mixed precision."""
        # Check if loss is finite before scaling
        if not torch.isfinite(loss):
            logger.warning(f"Non-finite loss detected: {loss}")
            self.optimizer.zero_grad()
            return

        self.scaler.scale(loss).backward()

        if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
            # Check gradients are finite before unscaling
            try:
                # Check if unscale has already been called in this step
                # PyTorch tracks this internally, so we need to handle the error
                if not getattr(self, '_unscaled_in_step', False):
                    self.scaler.unscale_(self.optimizer)
                    self._unscaled_in_step = True

                # Adaptive gradient clipping based on model size
                clip_value = self.config.gradient_clip_value
                if hasattr(self.config, 'adaptive_clipping') and self.config.adaptive_clipping:
                    # Scale clipping based on number of parameters (for large models)
                    param_count = sum(p.numel() for p in self.model.parameters())
                    if param_count > 1e6:  # Large model (>1M parameters)
                        clip_value *= 2.0

                # Calculate gradient norm before clipping for monitoring
                grad_norm_before_clip = self._calculate_gradient_norm()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)

                # Store gradient norm for later logging
                self._last_grad_norm = grad_norm_before_clip

                # Additional check for exploding gradients
                if grad_norm > clip_value * 20:  # Extreme gradient explosion
                    logger.warning(f"Extreme gradient explosion detected: {grad_norm:.1f} > {clip_value * 20:.1f}")
                    # Apply more aggressive clipping and reduce learning rate temporarily
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value * 0.5)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 0.8

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self._unscaled_in_step = False  # Reset flag after optimizer step

            except RuntimeError as e:
                error_msg = str(e)
                if "unscale_() has already been called" in error_msg:
                    logger.warning("Double unscale detected - skipping gradient unscaling")
                    # Still perform the optimizer step without unscaling again
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self._unscaled_in_step = False
                elif "No inf checks were recorded" in error_msg or "non-finite" in error_msg.lower():
                    logger.warning("Scaler inf check error - skipping update and clearing gradients")
                    self.optimizer.zero_grad()
                    self._unscaled_in_step = False
                    # Reset the scaler to avoid persistent issues
                    self.scaler = torch.cuda.amp.GradScaler(enabled=True)
                else:
                    raise e

    def _forward_pass_standard(
        self, sequences: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with standard precision."""
        predictions, _ = self.model(sequences)
        return self.criterion(predictions, targets) / self.config.gradient_accumulation_steps

    def _backward_pass_standard(self, loss: torch.Tensor, batch_idx: int) -> None:
        """Backward pass with standard precision."""
        loss.backward()

        if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
            # Calculate gradient norm before clipping
            grad_norm_before_clip = self._calculate_gradient_norm()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)

            # Store gradient norm for later logging
            self._last_grad_norm = grad_norm_before_clip

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

        for batch_idx, (sequences, targets) in enumerate(train_loader):
            try:
                # Memory monitoring and cleanup
                memory_stats = self.auto_memory_manager.monitor_and_cleanup()

                sequences = sequences.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # Forward and backward pass with error recovery
                loss_value = self._safe_training_step(sequences, targets, batch_idx)

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
            for sequences, targets in val_loader:
                sequences = sequences.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                if self.config.use_mixed_precision:
                    with autocast("cuda"):
                        predictions, attention_weights = self.model(sequences)
                        loss = self.criterion(predictions, targets)
                else:
                    predictions, attention_weights = self.model(sequences)
                    loss = self.criterion(predictions, targets)

                # Handle NaN losses but maintain gradient flow
                loss_value = loss.item()
                if not torch.isfinite(torch.tensor(loss_value)):
                    logger.warning(f"Non-finite validation loss detected: {loss_value}")
                    loss_value = 1.0  # Use a reasonable default instead of 0

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
                # Check and align shapes before concatenation
                min_batch_size = min(p.shape[0] for p in all_predictions + all_targets + all_weights)

                # Trim all tensors to minimum batch size to ensure shape compatibility
                aligned_predictions = [p[:min_batch_size] for p in all_predictions]
                aligned_targets = [t[:min_batch_size] for t in all_targets]
                aligned_weights = [w[:min_batch_size] for w in all_weights]

                # Concatenate all batches
                concat_predictions = torch.cat(aligned_predictions, dim=0)
                concat_targets = torch.cat(aligned_targets, dim=0)
                concat_weights = torch.cat(aligned_weights, dim=0)

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
        self, sequences: torch.Tensor, targets: torch.Tensor, dates: list
    ) -> tuple:
        """Create training and validation data splits."""
        if self.config.walk_forward_validation and len(sequences) > 10:
            return self.create_walk_forward_splits(sequences, targets, dates)

        # Standard random split
        split_idx = int(len(sequences) * (1 - self.config.validation_split))
        split_idx = max(1, min(split_idx, len(sequences) - 1))  # Ensure valid range

        train_seq, train_tgt = sequences[:split_idx], targets[:split_idx]
        val_seq, val_tgt = sequences[split_idx:], targets[split_idx:]
        return train_seq, train_tgt, val_seq, val_tgt

    def _create_data_loaders(
        self,
        train_seq: torch.Tensor,
        train_tgt: torch.Tensor,
        val_seq: torch.Tensor,
        val_tgt: torch.Tensor,
        batch_size: int,
    ) -> tuple:
        """Create training and validation data loaders."""
        train_dataset = TimeSeriesDataset(train_seq, train_tgt)
        val_dataset = TimeSeriesDataset(val_seq, val_tgt)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2
        )

        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2
        )

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
        self, val_loss: float, epoch: int, checkpoint_dir: Path | None
    ) -> bool:
        """Enhanced early stopping logic optimized for financial time series data."""
        # Never stop in the first 5 epochs - need minimum training
        if epoch < 5:
            logger.debug(f"Epoch {epoch+1}: Too early for early stopping (min 5 epochs)")
            return False

        # Skip early stopping if loss magnitude is suspiciously close to zero
        # Note: SharpeRatioLoss returns negative values (we minimize negative Sharpe ratio)
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
        if len(self.training_history["val_loss"]) >= 10:
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
        # Ensure we cap patience_counter to avoid confusing display like "26/5"
        if self.patience_counter >= effective_patience:
            convergence_metric = self._calculate_convergence_metric()
            # Display capped patience counter for clarity
            display_patience = min(self.patience_counter, effective_patience)
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
        improvements = []
        for i in range(1, len(recent_losses)):
            if recent_losses[i-1] > 0:
                improvement = (recent_losses[i-1] - recent_losses[i]) / recent_losses[i-1]
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

            # Keep original for portfolio calculations
            actual_np_original = actual_np.copy()

            # Check for shape compatibility
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
                min_shape = tuple(min(s1, s2) for s1, s2 in zip(pred_np.shape, actual_np.shape))
                pred_np = pred_np[:min_shape[0]] if pred_np.ndim == 1 else pred_np[:min_shape[0], :min_shape[1]]
                actual_np = actual_np[:min_shape[0]] if actual_np.ndim == 1 else actual_np[:min_shape[0], :min_shape[1]]

            mae = np.mean(np.abs(pred_np - actual_np))
            mse = np.mean((pred_np - actual_np) ** 2)
            metrics["prediction_accuracy"] = 1.0 / (1.0 + mae)  # Normalized accuracy

            # 2. Portfolio Performance Metrics (if weights provided)
            if weights is not None:
                weights_np = weights.detach().cpu().numpy()
                # Use original actual_np for portfolio calculations
                actual_for_portfolio = actual_np_original

                # Ensure weights shape matches actual_np with better alignment
                if weights_np.shape != actual_for_portfolio.shape:
                    logger.debug(f"Shape mismatch in weights: weights {weights_np.shape} vs actual {actual_for_portfolio.shape}")
                    min_samples = min(weights_np.shape[0], actual_for_portfolio.shape[0])
                    min_assets = min(weights_np.shape[1], actual_for_portfolio.shape[1]) if weights_np.ndim > 1 else 1

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
                        # Fallback: align to minimum dimensions
                        min_shape = tuple(min(s1, s2) for s1, s2 in zip(weights_np.shape, actual_for_portfolio.shape))
                        weights_aligned = weights_np[:min_shape[0], :min_shape[1]] if weights_np.ndim > 1 else weights_np[:min_shape[0]]
                        actual_aligned = actual_for_portfolio[:min_shape[0], :min_shape[1]] if actual_for_portfolio.ndim > 1 else actual_for_portfolio[:min_shape[0]]
                        portfolio_returns = np.sum(weights_aligned * actual_aligned, axis=1)
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

    def _safe_training_step(self, sequences: torch.Tensor, targets: torch.Tensor, batch_idx: int) -> float:
        """Perform training step with comprehensive error handling."""
        try:
            # Forward and backward pass
            if self.config.use_mixed_precision and self.scaler is not None:
                loss = self._forward_pass_with_mixed_precision(sequences, targets)
                self._backward_pass_mixed_precision(loss, batch_idx)
            else:
                loss = self._forward_pass_standard(sequences, targets)
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
        sequences, targets, dates = self.create_sequences(returns_data, sequence_length)

        # Create validation splits
        train_seq, train_tgt, val_seq, val_tgt = self._create_data_splits(sequences, targets, dates)

        # Optimize batch size and create data loaders
        universe_size = len(self.universe) if hasattr(self, 'universe') and self.universe else None
        optimal_batch_size = self.optimize_batch_size(sequence_length, universe_size)
        train_loader, val_loader = self._create_data_loaders(
            train_seq, train_tgt, val_seq, val_tgt, optimal_batch_size
        )

        # Training loop
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss, financial_metrics = self.validate(val_loader)

            # Store financial validation metrics in training history
            for metric_name, metric_value in financial_metrics.items():
                if metric_name in self.training_history:
                    self.training_history[metric_name].append(metric_value)

            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Update history and check early stopping
            self._update_training_history(train_loss, val_loss, current_lr, epoch)

            if self._handle_early_stopping(val_loss, epoch, checkpoint_dir):
                break

            # Regular checkpointing
            if checkpoint_dir and (epoch + 1) % self.config.checkpoint_frequency == 0:
                self.save_checkpoint(
                    checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth", epoch, val_loss
                )

        logger.info("Training completed")
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
