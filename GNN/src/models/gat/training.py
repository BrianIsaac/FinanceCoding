"""
Memory-efficient training pipeline for GAT portfolio models.

This module provides GPU memory optimization strategies, gradient accumulation,
mixed precision training, and batch processing utilities for handling large graphs
within VRAM constraints.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from .gat_model import GATPortfolio

__all__ = [
    "GATTrainingConfig",
    "GATTrainer",
    "PortfolioOptimizationLoss",
    "GPUMemoryManager",
]


@dataclass
class GATTrainingConfig:
    """Configuration for GAT model training with memory optimization."""

    # Training hyperparameters
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 200
    patience: int = 20
    gradient_clip_norm: float = 1.0

    # Memory optimization
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    max_vram_gb: float = 11.0

    # Loss function parameters
    risk_free_rate: float = 0.0
    constraint_penalty: float = 1.0
    regularization_weight: float = 0.1

    # Learning rate scheduling
    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # "cosine", "plateau", "step"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5

    # Validation and early stopping
    validation_split: float = 0.2
    early_stopping_metric: str = "loss"  # "loss", "sharpe", "returns"
    save_best_model: bool = True


class GPUMemoryManager:
    """GPU memory management utilities for efficient training."""

    def __init__(self, max_vram_gb: float = 11.0):
        """
        Initialize GPU memory manager.
        
        Args:
            max_vram_gb: Maximum VRAM to use in GB
        """
        self.max_memory = max_vram_gb * 1024**3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_memory_info(self) -> dict[str, float]:
        """Get current GPU memory usage information."""
        if not torch.cuda.is_available():
            return {"allocated": 0.0, "reserved": 0.0, "available": 0.0}

        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        available = self.max_memory / 1024**3 - allocated   # GB

        return {
            "allocated": allocated,
            "reserved": reserved,
            "available": available,
            "max_allowed": self.max_memory / 1024**3
        }

    def estimate_batch_memory(self, model: nn.Module, sample_batch_size: int = 1) -> float:
        """
        Estimate memory usage for a given batch size.
        
        Args:
            model: Model to estimate memory for
            sample_batch_size: Sample batch size for estimation
            
        Returns:
            Estimated memory usage in GB
        """
        if not torch.cuda.is_available():
            return 0.0

        # Get current memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()

        try:
            # Create dummy data similar to actual batch
            dummy_x = torch.randn(400, 10, device=self.device)  # 400 assets, 10 features
            dummy_edge_index = torch.randint(0, 400, (2, 8000), device=self.device)  # k=20 avg
            dummy_edge_attr = torch.randn(8000, 3, device=self.device)
            dummy_mask = torch.ones(400, dtype=torch.bool, device=self.device)

            # Forward pass
            model.eval()
            with torch.no_grad():
                _ = model(dummy_x, dummy_edge_index, dummy_mask, dummy_edge_attr)

            peak_memory = torch.cuda.max_memory_allocated()
            memory_per_sample = (peak_memory - initial_memory) / sample_batch_size

            return memory_per_sample * sample_batch_size / 1024**3  # GB

        except Exception:
            return 1.0  # Conservative estimate
        finally:
            torch.cuda.empty_cache()

    def get_optimal_batch_size(self, model: nn.Module, max_batch_size: int = 64) -> int:
        """
        Determine optimal batch size for memory constraints.
        
        Args:
            model: Model to optimize batch size for
            max_batch_size: Maximum batch size to consider
            
        Returns:
            Optimal batch size
        """
        if not torch.cuda.is_available():
            return max_batch_size

        # Binary search for optimal batch size
        left, right = 1, max_batch_size
        optimal_batch_size = 1

        while left <= right:
            mid = (left + right) // 2
            estimated_memory = self.estimate_batch_memory(model, mid)

            if estimated_memory <= self.max_memory / 1024**3 * 0.8:  # Use 80% of max memory
                optimal_batch_size = mid
                left = mid + 1
            else:
                right = mid - 1

        return max(1, optimal_batch_size)

    def clear_cache(self) -> None:
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class PortfolioOptimizationLoss(nn.Module):
    """Enhanced portfolio optimization loss combining Sharpe ratio and constraints."""

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        constraint_penalty: float = 1.0,
        regularization_weight: float = 0.1,
        target_volatility: float | None = None
    ):
        """
        Initialize portfolio optimization loss.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            constraint_penalty: Weight for constraint violation penalties
            regularization_weight: Weight for regularization terms
            target_volatility: Target portfolio volatility (optional)
        """
        super().__init__()
        self.risk_free_rate = risk_free_rate
        self.constraint_penalty = constraint_penalty
        self.regularization_weight = regularization_weight
        self.target_volatility = target_volatility

    def forward(
        self,
        portfolio_weights: torch.Tensor,
        returns: torch.Tensor,
        constraints_mask: torch.Tensor | None = None,
        prev_weights: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Compute portfolio optimization loss with detailed components.
        
        Args:
            portfolio_weights: Portfolio weights [batch_size, n_assets]
            returns: Asset returns [batch_size, n_assets]
            constraints_mask: Valid asset mask [batch_size, n_assets]
            prev_weights: Previous weights for turnover calculation
            
        Returns:
            Dictionary with loss components
        """
        # Apply constraints mask if provided
        if constraints_mask is not None:
            portfolio_weights = portfolio_weights * constraints_mask.float()
            returns = returns * constraints_mask.float()

        # Portfolio returns
        portfolio_returns = torch.sum(portfolio_weights * returns, dim=-1)
        excess_returns = portfolio_returns - self.risk_free_rate

        # Sharpe ratio components
        mean_return = excess_returns.mean()
        return_std = excess_returns.std() + 1e-8
        sharpe_ratio = mean_return / return_std

        # Portfolio volatility
        portfolio_vol = return_std

        # Constraint violations
        weight_sum_error = torch.mean(torch.abs(portfolio_weights.sum(dim=-1) - 1.0))
        negative_weight_penalty = torch.mean(torch.relu(-portfolio_weights))

        # Regularization terms
        concentration_penalty = torch.mean((portfolio_weights ** 2).sum(dim=-1))  # Herfindahl index

        # Turnover penalty
        turnover_penalty = torch.tensor(0.0, device=portfolio_weights.device)
        if prev_weights is not None:
            turnover_penalty = torch.mean(torch.abs(portfolio_weights - prev_weights).sum(dim=-1))

        # Volatility targeting
        volatility_penalty = torch.tensor(0.0, device=portfolio_weights.device)
        if self.target_volatility is not None:
            volatility_penalty = torch.abs(portfolio_vol - self.target_volatility)

        # Combined loss
        constraint_loss = weight_sum_error + negative_weight_penalty
        regularization_loss = concentration_penalty + turnover_penalty + volatility_penalty

        total_loss = (
            -sharpe_ratio +
            self.constraint_penalty * constraint_loss +
            self.regularization_weight * regularization_loss
        )

        return {
            "total_loss": total_loss,
            "sharpe_ratio": sharpe_ratio,
            "portfolio_return": mean_return,
            "portfolio_volatility": portfolio_vol,
            "constraint_loss": constraint_loss,
            "regularization_loss": regularization_loss,
            "weight_sum_error": weight_sum_error,
            "turnover": turnover_penalty,
        }


class GATTrainer:
    """Memory-efficient trainer for GAT portfolio models."""

    def __init__(
        self,
        model: GATPortfolio,
        config: GATTrainingConfig,
        device: torch.device | None = None
    ):
        """
        Initialize GAT trainer.
        
        Args:
            model: GAT portfolio model to train
            config: Training configuration
            device: Device to train on
        """
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model = self.model.to(self.device)

        # Initialize memory manager
        self.memory_manager = GPUMemoryManager(config.max_vram_gb)

        # Initialize optimizer
        if config.weight_decay > 0:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            self.optimizer = Adam(
                self.model.parameters(),
                lr=config.learning_rate
            )

        # Initialize loss function
        self.loss_fn = PortfolioOptimizationLoss(
            risk_free_rate=config.risk_free_rate,
            constraint_penalty=config.constraint_penalty,
            regularization_weight=config.regularization_weight
        )

        # Initialize learning rate scheduler
        self.scheduler = None
        if config.use_scheduler:
            if config.scheduler_type == "cosine":
                self.scheduler = CosineAnnealingLR(
                    self.optimizer, T_max=config.max_epochs
                )
            elif config.scheduler_type == "plateau":
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer,
                    mode="min",
                    factor=config.scheduler_factor,
                    patience=config.scheduler_patience,
                    verbose=True
                )

        # Initialize mixed precision scaler
        if config.use_mixed_precision and torch.cuda.is_available():
            try:
                # Use new API if available (PyTorch 2.1+)
                self.scaler = torch.amp.GradScaler('cuda')
            except (AttributeError, TypeError):
                # Fall back to older API
                self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # Training history
        self.history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [], "train_sharpe": [], "val_sharpe": [],
            "learning_rate": [], "memory_usage": []
        }

    def create_data_batches(
        self,
        graph_data_list: list[Any],
        batch_size: int | None = None
    ) -> Iterator[list[Any]]:
        """
        Create batches from graph data with memory constraints.
        
        Args:
            graph_data_list: List of graph data objects
            batch_size: Batch size (uses config if None)
            
        Yields:
            Batches of graph data
        """
        if batch_size is None:
            # Determine optimal batch size
            batch_size = self.memory_manager.get_optimal_batch_size(self.model, self.config.batch_size)

        for i in range(0, len(graph_data_list), batch_size):
            yield graph_data_list[i:i + batch_size]

    def train_epoch(self, train_data_list: list[Any]) -> dict[str, float]:
        """
        Train for one epoch with gradient accumulation and memory optimization.
        
        Args:
            train_data_list: List of training graph data
            
        Returns:
            Training metrics for the epoch
        """
        self.model.train()
        epoch_losses = []
        epoch_sharpe_ratios = []

        # Create batches
        batches = list(self.create_data_batches(train_data_list))
        accumulation_steps = max(1, self.config.gradient_accumulation_steps)

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(batches):
            try:
                # Process batch
                batch_losses = []
                batch_sharpe_ratios = []

                for graph_data, labels in batch:
                    # Move data to device
                    x = graph_data.x.to(self.device)
                    edge_index = graph_data.edge_index.to(self.device)
                    edge_attr = graph_data.edge_attr.to(self.device) if graph_data.edge_attr is not None else None
                    mask_valid = torch.ones(x.size(0), dtype=torch.bool, device=self.device)
                    returns_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device)

                    # Forward pass with mixed precision
                    if self.scaler is not None:
                        with torch.cuda.amp.autocast():
                            weights, _, reg_loss = self.model(x, edge_index, mask_valid, edge_attr)

                            # Compute portfolio loss
                            loss_dict = self.loss_fn(
                                weights.unsqueeze(0),
                                returns_tensor.unsqueeze(0),
                                mask_valid.unsqueeze(0)
                            )

                            total_loss = loss_dict["total_loss"]
                            if reg_loss is not None:
                                total_loss = total_loss + reg_loss

                            # Scale loss for gradient accumulation
                            total_loss = total_loss / accumulation_steps

                        # Backward pass with gradient scaling
                        self.scaler.scale(total_loss).backward()
                    else:
                        weights, _, reg_loss = self.model(x, edge_index, mask_valid, edge_attr)

                        loss_dict = self.loss_fn(
                            weights.unsqueeze(0),
                            returns_tensor.unsqueeze(0),
                            mask_valid.unsqueeze(0)
                        )

                        total_loss = loss_dict["total_loss"]
                        if reg_loss is not None:
                            total_loss = total_loss + reg_loss

                        total_loss = total_loss / accumulation_steps
                        total_loss.backward()

                    batch_losses.append(total_loss.item() * accumulation_steps)
                    batch_sharpe_ratios.append(loss_dict["sharpe_ratio"].item())

                # Accumulate gradients and update
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(batches):
                    if self.scaler is not None:
                        # Gradient clipping
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)

                        # Update parameters
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                        self.optimizer.step()

                    self.optimizer.zero_grad()

                epoch_losses.extend(batch_losses)
                epoch_sharpe_ratios.extend(batch_sharpe_ratios)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM error in batch {batch_idx}, clearing cache and skipping")
                    self.memory_manager.clear_cache()
                    self.optimizer.zero_grad()
                    continue
                else:
                    raise e

        return {
            "loss": np.mean(epoch_losses) if epoch_losses else float('inf'),
            "sharpe": np.mean(epoch_sharpe_ratios) if epoch_sharpe_ratios else 0.0,
            "batches_processed": len(batches)
        }

    def validate(self, val_data_list: list[Any]) -> dict[str, float]:
        """
        Validate model performance.
        
        Args:
            val_data_list: List of validation graph data
            
        Returns:
            Validation metrics
        """
        self.model.eval()
        val_losses = []
        val_sharpe_ratios = []

        with torch.no_grad():
            for batch in self.create_data_batches(val_data_list):
                try:
                    for graph_data, labels in batch:
                        x = graph_data.x.to(self.device)
                        edge_index = graph_data.edge_index.to(self.device)
                        edge_attr = graph_data.edge_attr.to(self.device) if graph_data.edge_attr is not None else None
                        mask_valid = torch.ones(x.size(0), dtype=torch.bool, device=self.device)
                        returns_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device)

                        weights, _, reg_loss = self.model(x, edge_index, mask_valid, edge_attr)

                        loss_dict = self.loss_fn(
                            weights.unsqueeze(0),
                            returns_tensor.unsqueeze(0),
                            mask_valid.unsqueeze(0)
                        )

                        total_loss = loss_dict["total_loss"]
                        if reg_loss is not None:
                            total_loss = total_loss + reg_loss

                        val_losses.append(total_loss.item())
                        val_sharpe_ratios.append(loss_dict["sharpe_ratio"].item())

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        self.memory_manager.clear_cache()
                        continue
                    else:
                        raise e

        return {
            "loss": np.mean(val_losses) if val_losses else float('inf'),
            "sharpe": np.mean(val_sharpe_ratios) if val_sharpe_ratios else 0.0
        }

    def train(
        self,
        train_data_list: list[Any],
        val_data_list: list[Any] | None = None
    ) -> dict[str, Any]:
        """
        Complete training loop with early stopping and memory optimization.
        
        Args:
            train_data_list: Training data
            val_data_list: Validation data (optional)
            
        Returns:
            Training history and final metrics
        """
        best_metric = float('inf')
        patience_counter = 0

        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.config.max_epochs):
            # Training
            train_metrics = self.train_epoch(train_data_list)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_sharpe"].append(train_metrics["sharpe"])

            # Validation
            val_metrics = {"loss": float('inf'), "sharpe": 0.0}
            if val_data_list:
                val_metrics = self.validate(val_data_list)
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_sharpe"].append(val_metrics["sharpe"])

            # Learning rate scheduling
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()
            self.history["learning_rate"].append(current_lr)

            # Memory usage tracking
            memory_info = self.memory_manager.get_memory_info()
            self.history["memory_usage"].append(memory_info["allocated"])

            # Early stopping
            current_metric = val_metrics["loss"] if val_data_list else train_metrics["loss"]
            if current_metric < best_metric:
                best_metric = current_metric
                patience_counter = 0

                if self.config.save_best_model:
                    # Save best model state
                    self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            # Progress logging
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: "
                      f"Train Loss: {train_metrics['loss']:.6f}, "
                      f"Val Loss: {val_metrics['loss']:.6f}, "
                      f"Train Sharpe: {train_metrics['sharpe']:.4f}, "
                      f"LR: {current_lr:.2e}, "
                      f"Memory: {memory_info['allocated']:.1f}GB")

            # Early stopping check
            if patience_counter >= self.config.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Load best model if available
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)

        return {
            "history": self.history,
            "best_metric": best_metric,
            "final_epoch": epoch,
            "memory_usage": memory_info
        }
