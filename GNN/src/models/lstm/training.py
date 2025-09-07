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
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from .architecture import LSTMNetwork, SharpeRatioLoss

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for LSTM training pipeline."""

    # Memory optimization
    max_memory_gb: float = 11.0  # Conservative VRAM limit for RTX 5070Ti
    gradient_accumulation_steps: int = 4  # Effective batch size multiplier
    use_mixed_precision: bool = True  # FP16 training for memory efficiency

    # Training parameters
    learning_rate: float = 0.001  # Adam optimizer learning rate
    weight_decay: float = 1e-5  # L2 regularization
    batch_size: int = 32  # Base batch size
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

        # Initialize loss function
        self.criterion = SharpeRatioLoss()

        # Initialize mixed precision scaler
        self.scaler = GradScaler() if config.use_mixed_precision else None

        # Training state
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": [],
            "memory_usage": [],
        }

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
                f"No valid sequences created. Data has {n_timesteps} timesteps, "
                f"need at least {sequence_length + prediction_horizon} for sequence_length={sequence_length} "
                f"and prediction_horizon={prediction_horizon}"
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
        Estimate memory usage in GB for given batch configuration.

        Args:
            batch_size: Training batch size
            sequence_length: Input sequence length

        Returns:
            Estimated memory usage in GB
        """
        model_memory = self.model.get_memory_usage(batch_size, sequence_length)

        # Add optimizer state memory (approximately 2x model parameters for Adam)
        param_count = sum(p.numel() for p in self.model.parameters())
        optimizer_memory = param_count * 4 * 2  # float32 * (momentum + variance)

        # Add gradient accumulation memory
        grad_memory = param_count * 4 * self.config.gradient_accumulation_steps

        total_memory_bytes = model_memory + optimizer_memory + grad_memory
        total_memory_gb = total_memory_bytes / (1024**3)

        return total_memory_gb

    def optimize_batch_size(self, sequence_length: int) -> int:
        """
        Automatically optimize batch size based on memory constraints.

        Args:
            sequence_length: Input sequence length

        Returns:
            Optimal batch size for given memory limits
        """
        max_batch_size = self.config.batch_size

        for batch_size in range(max_batch_size, 0, -1):
            estimated_memory = self.estimate_memory_usage(batch_size, sequence_length)

            if estimated_memory <= self.config.max_memory_gb:
                logger.info(
                    f"Optimal batch size: {batch_size} (estimated memory: {estimated_memory:.2f}GB)"
                )
                return batch_size

        logger.warning("Could not find suitable batch size, using batch_size=1")
        return 1

    def _forward_pass_with_mixed_precision(
        self, sequences: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with automatic mixed precision."""
        with autocast():
            predictions, _ = self.model(sequences)
            return self.criterion(predictions, targets) / self.config.gradient_accumulation_steps

    def _backward_pass_mixed_precision(self, loss: torch.Tensor, batch_idx: int) -> None:
        """Backward pass with gradient scaling for mixed precision."""
        self.scaler.scale(loss).backward()

        if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)
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

        # Track memory usage
        initial_memory = 0
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences = sequences.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Forward and backward pass
            if self.config.use_mixed_precision and self.scaler is not None:
                loss = self._forward_pass_with_mixed_precision(sequences, targets)
                self._backward_pass_mixed_precision(loss, batch_idx)
            else:
                loss = self._forward_pass_standard(sequences, targets)
                self._backward_pass_standard(loss, batch_idx)

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

            # Log progress
            if batch_idx % 50 == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}"
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

    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate model performance.

        Args:
            val_loader: Validation data loader

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                if self.config.use_mixed_precision:
                    with autocast():
                        predictions, _ = self.model(sequences)
                        loss = self.criterion(predictions, targets)
                else:
                    predictions, _ = self.model(sequences)
                    loss = self.criterion(predictions, targets)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def _create_data_splits(self, sequences: list, targets: list, dates: list) -> tuple:
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
        self, train_seq: list, train_tgt: list, val_seq: list, val_tgt: list, batch_size: int
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

        logger.info(
            f"Epoch {epoch+1}/{self.config.epochs}: "
            f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.8f}"
        )

    def _handle_early_stopping(
        self, val_loss: float, epoch: int, checkpoint_dir: Path | None
    ) -> bool:
        """Handle early stopping logic. Returns True if training should stop."""
        if val_loss < self.best_loss - self.config.min_delta:
            self.best_loss = val_loss
            self.patience_counter = 0

            if checkpoint_dir and self.config.save_best_only:
                self.save_checkpoint(checkpoint_dir / "best_model.pth", epoch, val_loss)
        else:
            self.patience_counter += 1

            if self.patience_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                return True
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

        # Create sequences
        sequences, targets, dates = self.create_sequences(returns_data, sequence_length)

        # Create validation splits
        train_seq, train_tgt, val_seq, val_tgt = self._create_data_splits(sequences, targets, dates)

        # Optimize batch size and create data loaders
        optimal_batch_size = self.optimize_batch_size(sequence_length)
        train_loader, val_loader = self._create_data_loaders(
            train_seq, train_tgt, val_seq, val_tgt, optimal_batch_size
        )

        # Training loop
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)

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
