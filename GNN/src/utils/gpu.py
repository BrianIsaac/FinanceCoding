"""
GPU memory management utilities for PyTorch models.

This module provides GPU memory optimization utilities specifically
designed for the RTX GeForce 5070Ti (12GB VRAM) constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class GPUConfig:
    """Configuration for GPU memory management."""

    max_memory_gb: float = 11.0  # Reserve 1GB for system
    enable_mixed_precision: bool = True
    gradient_checkpointing: bool = False
    batch_size_auto_scale: bool = True


class GPUMemoryManager:
    """
    GPU memory manager for portfolio optimization models.

    Provides memory optimization specifically for GAT and LSTM models
    running on RTX GeForce 5070Ti with 12GB VRAM constraints.
    """

    def __init__(self, config: GPUConfig):
        """
        Initialize GPU memory manager.

        Args:
            config: GPU configuration parameters
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_optimal_batch_size(
        self, model: torch.nn.Module, input_shape: tuple, max_batch_size: int = 128
    ) -> int:
        """
        Determine optimal batch size for given model and input shape.

        Args:
            model: PyTorch model to optimize for
            input_shape: Input tensor shape (excluding batch dimension)
            max_batch_size: Maximum batch size to test

        Returns:
            Optimal batch size that fits in GPU memory

        Note:
            This is a stub implementation. Batch size optimization
            will be implemented in future stories.
        """
        # Stub implementation - returns conservative batch size
        return 32

    def setup_mixed_precision(self) -> torch.cuda.amp.GradScaler:
        """
        Setup mixed precision training for memory efficiency.

        Returns:
            GradScaler for mixed precision training

        Note:
            This is a stub implementation. Mixed precision setup
            will be implemented in future stories.
        """
        if self.config.enable_mixed_precision and self.device.type == "cuda":
            return torch.amp.GradScaler("cuda")
        else:
            return None

    def get_memory_stats(self) -> dict[str, Any]:
        """
        Get current GPU memory usage statistics.

        Returns:
            Dictionary containing memory usage information
        """
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
            cached = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            free = total - allocated
            utilization = (allocated / total) * 100

            return {
                "allocated_gb": round(allocated, 2),
                "cached_gb": round(cached, 2),
                "free_gb": round(free, 2),
                "total_gb": round(total, 2),
                "utilization_pct": round(utilization, 2),
            }
        else:
            return {"status": "CPU mode - no GPU stats available"}

    def clear_cache(self) -> None:
        """
        Clear GPU memory cache to free up memory.
        """
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def configure_memory_limit(self) -> None:
        """
        Configure GPU memory usage limit based on available VRAM.
        """
        if self.device.type == "cuda":
            # Set memory fraction to respect configured limit
            if hasattr(torch.cuda, "set_memory_fraction"):
                memory_fraction = self.config.max_memory_gb / 12.0  # 12GB total VRAM
                torch.cuda.set_memory_fraction(min(memory_fraction, 0.95))

    def is_gpu_available(self) -> bool:
        """
        Check if GPU is available and properly configured.

        Returns:
            True if GPU is available, False otherwise
        """
        return torch.cuda.is_available() and self.device.type == "cuda"
