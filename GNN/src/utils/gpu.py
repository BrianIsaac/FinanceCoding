"""
GPU memory management utilities for PyTorch models.

This module provides GPU memory optimization utilities specifically
designed for the RTX GeForce 5070Ti (12GB VRAM) constraints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
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
        """
        if not self.is_gpu_available():
            return 32  # Conservative fallback for CPU

        model.to(self.device)
        model.eval()  # Set to eval mode to disable dropout, etc.

        # Start with a small batch size and increase until we hit memory limits
        batch_size = 1
        max_safe_batch_size = 1

        # Memory threshold (90% of max to leave buffer)
        memory_threshold = self.config.max_memory_gb * 0.9 * (1024**3)

        while batch_size <= max_batch_size:
            try:
                # Clear cache before test
                torch.cuda.empty_cache()

                # Create test input batch
                if isinstance(input_shape, tuple):
                    test_input = torch.randn(batch_size, *input_shape, device=self.device)
                else:
                    # Handle more complex input shapes
                    test_input = torch.randn(batch_size, input_shape, device=self.device)

                # Forward pass to measure memory usage
                with torch.no_grad():
                    _ = model(test_input)

                # Check current memory usage
                current_memory = torch.cuda.memory_allocated()

                if current_memory <= memory_threshold:
                    max_safe_batch_size = batch_size
                    batch_size = min(batch_size * 2, max_batch_size)  # Double batch size
                else:
                    break  # Memory limit reached

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    break  # Memory limit reached
                else:
                    raise  # Re-raise if it's not a memory error
            except Exception:
                # Handle other potential errors gracefully
                break

        # Clean up
        torch.cuda.empty_cache()

        # Apply safety factor (use 80% of max found)
        optimal_batch_size = max(1, int(max_safe_batch_size * 0.8))

        return optimal_batch_size

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


class MemoryEfficientTrainer:
    """
    Memory-efficient trainer with gradient accumulation and mixed precision support.

    This trainer implements gradient accumulation, mixed precision training, and
    memory optimization specifically for 12GB VRAM constraints.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 max_memory_gb: float = 11.0,
                 gradient_accumulation_steps: int = 4,
                 enable_mixed_precision: bool = True,
                 gradient_clipping: float = 1.0):
        """
        Initialize memory-efficient trainer.

        Args:
            model: PyTorch model to train
            optimizer: Optimizer instance
            max_memory_gb: Maximum GPU memory to use (GB)
            gradient_accumulation_steps: Number of steps to accumulate gradients
            enable_mixed_precision: Whether to use mixed precision training
            gradient_clipping: Maximum gradient norm for clipping
        """
        self.model = model
        self.optimizer = optimizer
        self.max_memory = max_memory_gb * 1024**3  # Convert to bytes
        self.accumulation_steps = gradient_accumulation_steps
        self.enable_mixed_precision = enable_mixed_precision
        self.gradient_clipping = gradient_clipping

        # Initialize mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if (
            enable_mixed_precision and torch.cuda.is_available()
        ) else None

        # Memory monitoring
        self.memory_stats = []
        self._step_count = 0

    def train_epoch(self,
                   data_loader: torch.utils.data.DataLoader,
                   loss_fn: callable,
                   device: torch.device,
                   epoch: int = 0) -> dict[str, float]:
        """
        Train one epoch with memory-efficient techniques.

        Args:
            data_loader: Training data loader
            loss_fn: Loss function
            device: Training device
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        accumulated_steps = 0

        # Clear optimizer gradients at start
        self.optimizer.zero_grad()

        for batch_idx, batch_data in enumerate(data_loader):
            # Move data to device
            if isinstance(batch_data, (list, tuple)):
                batch_data = [item.to(device) if hasattr(item, 'to') else item
                             for item in batch_data]
            else:
                batch_data = batch_data.to(device)

            # Forward pass with optional mixed precision
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                if isinstance(batch_data, (list, tuple)):
                    outputs = self.model(*batch_data[:-1])  # Assume last item is target
                    targets = batch_data[-1]
                else:
                    # Handle single tensor case - may need adaptation based on model
                    outputs = self.model(batch_data)
                    targets = None  # Model-specific implementation needed

                # Compute loss
                if targets is not None:
                    loss = loss_fn(outputs, targets)
                else:
                    loss = loss_fn(outputs)  # For models with built-in targets

                # Scale loss for gradient accumulation
                loss = loss / self.accumulation_steps

            # Backward pass with mixed precision
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulated_steps += 1
            total_loss += loss.item() * self.accumulation_steps  # Unscale for logging

            # Gradient accumulation step
            if accumulated_steps % self.accumulation_steps == 0:
                # Gradient clipping
                if self.gradient_clipping > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                 self.gradient_clipping)

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

                # Memory cleanup every few steps
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Memory monitoring
                if batch_idx % 10 == 0:  # Monitor every 10 batches
                    self._record_memory_stats(batch_idx, epoch)

            num_batches += 1
            self._step_count += 1

        # Handle any remaining gradients if batch doesn't divide evenly
        if accumulated_steps % self.accumulation_steps != 0:
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {
            'epoch': epoch,
            'avg_loss': avg_loss,
            'num_batches': num_batches,
            'memory_peak_gb': self.get_peak_memory_usage(),
            'gradient_accumulation_steps': self.accumulation_steps
        }

    def _record_memory_stats(self, batch_idx: int, epoch: int) -> None:
        """Record current GPU memory statistics."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            cached = torch.cuda.memory_reserved() / (1024**3)

            self.memory_stats.append({
                'step': self._step_count,
                'epoch': epoch,
                'batch_idx': batch_idx,
                'allocated_gb': round(allocated, 2),
                'cached_gb': round(cached, 2),
                'timestamp': pd.Timestamp.now()
            })

    def get_peak_memory_usage(self) -> float:
        """Get peak memory usage in GB."""
        if not self.memory_stats:
            return 0.0
        return max(stat['allocated_gb'] for stat in self.memory_stats)

    def get_memory_report(self) -> dict[str, Any]:
        """Get comprehensive memory usage report."""
        if not self.memory_stats:
            return {'error': 'No memory statistics recorded'}

        allocated_values = [stat['allocated_gb'] for stat in self.memory_stats]
        cached_values = [stat['cached_gb'] for stat in self.memory_stats]

        return {
            'peak_allocated_gb': max(allocated_values),
            'avg_allocated_gb': round(np.mean(allocated_values), 2),
            'peak_cached_gb': max(cached_values),
            'avg_cached_gb': round(np.mean(cached_values), 2),
            'memory_limit_gb': self.max_memory / (1024**3),
            'utilization_peak_pct': round(
                (max(allocated_values) / (self.max_memory / (1024**3))) * 100, 2
            ),
            'gradient_accumulation_steps': self.accumulation_steps,
            'mixed_precision_enabled': self.scaler is not None,
            'total_training_steps': self._step_count
        }

    def optimize_for_memory_limit(self) -> dict[str, Any]:
        """
        Automatically optimize training parameters for memory constraints.

        Returns:
            Dictionary of optimization recommendations
        """
        current_usage = self.get_peak_memory_usage()
        memory_limit_gb = self.max_memory / (1024**3)

        recommendations = {
            'current_peak_gb': current_usage,
            'memory_limit_gb': memory_limit_gb,
            'memory_available': current_usage < memory_limit_gb * 0.9,
            'recommendations': []
        }

        if current_usage > memory_limit_gb * 0.9:  # Over 90% usage
            recommendations['recommendations'].append(
                f"High memory usage detected ({current_usage:.1f}GB). "
                f"Consider increasing gradient accumulation steps."
            )

            # Suggest doubling accumulation steps
            new_accumulation = self.accumulation_steps * 2
            recommendations['suggested_accumulation_steps'] = new_accumulation

        if not self.scaler and torch.cuda.is_available():
            recommendations['recommendations'].append(
                "Enable mixed precision training to reduce memory usage by ~50%"
            )

        if current_usage < memory_limit_gb * 0.5:  # Under 50% usage
            recommendations['recommendations'].append(
                "Memory usage is low. Consider reducing gradient accumulation steps "
                "or increasing batch size for better training efficiency."
            )

        return recommendations


class AutomaticMemoryManager:
    """
    Automatic memory management with monitoring and cleanup.

    This class provides automatic memory monitoring, cleanup, and
    optimization for continuous training processes.
    """

    def __init__(self,
                 memory_threshold_gb: float = 10.5,
                 cleanup_interval_steps: int = 50,
                 enable_automatic_cleanup: bool = True):
        """
        Initialize automatic memory manager.

        Args:
            memory_threshold_gb: Memory threshold for triggering cleanup
            cleanup_interval_steps: Steps between automatic cleanup
            enable_automatic_cleanup: Whether to enable automatic cleanup
        """
        self.memory_threshold = memory_threshold_gb * (1024**3)  # Convert to bytes
        self.cleanup_interval = cleanup_interval_steps
        self.enable_cleanup = enable_automatic_cleanup
        self.step_count = 0
        self.memory_history = []
        self.cleanup_events = []

    def monitor_and_cleanup(self, force_cleanup: bool = False) -> dict[str, Any]:
        """
        Monitor memory usage and perform cleanup if necessary.

        Args:
            force_cleanup: Force cleanup regardless of conditions

        Returns:
            Memory monitoring report
        """
        self.step_count += 1

        if not torch.cuda.is_available():
            return {'status': 'CPU mode - no GPU monitoring'}

        # Get current memory stats
        allocated = torch.cuda.memory_allocated()
        cached = torch.cuda.memory_reserved()

        memory_stats = {
            'step': self.step_count,
            'allocated_bytes': allocated,
            'cached_bytes': cached,
            'allocated_gb': allocated / (1024**3),
            'cached_gb': cached / (1024**3),
            'timestamp': pd.Timestamp.now()
        }

        self.memory_history.append(memory_stats)

        # Determine if cleanup is needed
        cleanup_needed = (
            force_cleanup or
            (self.enable_cleanup and allocated > self.memory_threshold) or
            (self.enable_cleanup and self.step_count % self.cleanup_interval == 0)
        )

        if cleanup_needed:
            cleanup_result = self._perform_cleanup()
            memory_stats['cleanup_performed'] = True
            memory_stats['cleanup_result'] = cleanup_result
        else:
            memory_stats['cleanup_performed'] = False

        return memory_stats

    def _perform_cleanup(self) -> dict[str, Any]:
        """Perform memory cleanup and record results."""
        if not torch.cuda.is_available():
            return {'status': 'CPU mode - no cleanup needed'}

        # Record pre-cleanup memory
        pre_allocated = torch.cuda.memory_allocated()
        pre_cached = torch.cuda.memory_reserved()

        # Perform cleanup
        torch.cuda.empty_cache()

        # Record post-cleanup memory
        post_allocated = torch.cuda.memory_allocated()
        post_cached = torch.cuda.memory_reserved()

        # Calculate cleanup effectiveness
        freed_allocated = pre_allocated - post_allocated
        freed_cached = pre_cached - post_cached

        cleanup_result = {
            'timestamp': pd.Timestamp.now(),
            'step': self.step_count,
            'pre_allocated_gb': pre_allocated / (1024**3),
            'post_allocated_gb': post_allocated / (1024**3),
            'pre_cached_gb': pre_cached / (1024**3),
            'post_cached_gb': post_cached / (1024**3),
            'freed_allocated_gb': freed_allocated / (1024**3),
            'freed_cached_gb': freed_cached / (1024**3),
            'cleanup_effectiveness': freed_cached / max(pre_cached, 1)  # Avoid division by zero
        }

        self.cleanup_events.append(cleanup_result)
        return cleanup_result

    def get_memory_trend_analysis(self,
                                 window_size: int = 100) -> dict[str, Any]:
        """
        Analyze memory usage trends over recent history.

        Args:
            window_size: Number of recent steps to analyze

        Returns:
            Trend analysis report
        """
        if len(self.memory_history) < 10:
            return {'status': 'Insufficient data for trend analysis'}

        # Get recent memory history
        recent_history = self.memory_history[-window_size:]
        allocated_values = [stat['allocated_gb'] for stat in recent_history]
        cached_values = [stat['cached_gb'] for stat in recent_history]

        # Calculate trend metrics
        if len(allocated_values) > 1:
            allocated_trend = np.polyfit(range(len(allocated_values)), allocated_values, 1)[0]
            cached_trend = np.polyfit(range(len(cached_values)), cached_values, 1)[0]
        else:
            allocated_trend = 0
            cached_trend = 0

        return {
            'window_size': len(recent_history),
            'current_allocated_gb': allocated_values[-1] if allocated_values else 0,
            'current_cached_gb': cached_values[-1] if cached_values else 0,
            'peak_allocated_gb': max(allocated_values) if allocated_values else 0,
            'avg_allocated_gb': np.mean(allocated_values) if allocated_values else 0,
            'allocated_trend_gb_per_step': allocated_trend,
            'cached_trend_gb_per_step': cached_trend,
            'memory_increasing': allocated_trend > 0.001,  # Growing by >1MB per step
            'cleanup_frequency': len(self.cleanup_events) / max(self.step_count, 1),
            'total_cleanups': len(self.cleanup_events)
        }

    def optimize_cleanup_strategy(self) -> dict[str, Any]:
        """
        Analyze cleanup effectiveness and suggest optimizations.

        Returns:
            Optimization recommendations
        """
        if not self.cleanup_events:
            return {
                'status': 'No cleanup events recorded',
                'recommendations': ['Enable automatic cleanup to gather data']
            }

        # Analyze cleanup effectiveness
        effectiveness_scores = [event['cleanup_effectiveness'] for event in self.cleanup_events]
        avg_effectiveness = np.mean(effectiveness_scores)

        # Analyze cleanup frequency vs memory growth
        trend_analysis = self.get_memory_trend_analysis()

        recommendations = []

        if avg_effectiveness < 0.1:  # Less than 10% cache freed on average
            recommendations.append(
                "Cleanup effectiveness is low. Consider increasing cleanup interval "
                "or addressing potential memory leaks."
            )

        if trend_analysis.get('memory_increasing', False):
            recommendations.append(
                "Memory usage is trending upward. Consider more frequent cleanup "
                "or reducing batch sizes."
            )

            # Suggest more frequent cleanup
            current_interval = self.cleanup_interval
            suggested_interval = max(10, current_interval // 2)
            recommendations.append(
                f"Consider reducing cleanup interval from {current_interval} to "
                f"{suggested_interval} steps."
            )

        # >20% of steps involve cleanup
        if len(self.cleanup_events) / max(self.step_count, 1) > 0.2:
            recommendations.append(
                "High cleanup frequency detected. Consider optimizing model or "
                "reducing memory pressure to improve training efficiency."
            )

        return {
            'avg_cleanup_effectiveness': round(avg_effectiveness, 3),
            'total_cleanups': len(self.cleanup_events),
            'cleanup_rate': round(len(self.cleanup_events) / max(self.step_count, 1), 3),
            'memory_trend': trend_analysis,
            'recommendations': recommendations,
            'suggested_cleanup_interval': (
                max(10, self.cleanup_interval // 2)
                if trend_analysis.get('memory_increasing')
                else self.cleanup_interval
            )
        }

    def export_memory_report(self, output_path: str) -> None:
        """Export comprehensive memory monitoring report."""
        report = {
            'generated_at': pd.Timestamp.now().isoformat(),
            'configuration': {
                'memory_threshold_gb': self.memory_threshold / (1024**3),
                'cleanup_interval_steps': self.cleanup_interval,
                'automatic_cleanup_enabled': self.enable_cleanup
            },
            'summary': {
                'total_steps': self.step_count,
                'total_cleanups': len(self.cleanup_events),
                'memory_history_length': len(self.memory_history)
            },
            'trend_analysis': self.get_memory_trend_analysis(),
            'optimization_analysis': self.optimize_cleanup_strategy(),
            'detailed_memory_history': self.memory_history[-1000:],  # Last 1000 steps
            'cleanup_events': self.cleanup_events
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)


# Enhanced batch size optimization utilities
def find_optimal_batch_size_binary_search(model: torch.nn.Module,
                                        input_shape: tuple,
                                        max_memory_gb: float = 11.0,
                                        max_batch_size: int = 256) -> dict[str, Any]:
    """
    Use binary search to find optimal batch size more efficiently.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape (excluding batch dimension)
        max_memory_gb: Maximum memory constraint
        max_batch_size: Maximum batch size to consider

    Returns:
        Dictionary with optimal batch size and analysis
    """
    if not torch.cuda.is_available():
        return {'optimal_batch_size': 32, 'method': 'CPU fallback'}

    device = torch.device('cuda')
    model = model.to(device)
    model.eval()

    memory_threshold = max_memory_gb * 0.9 * (1024**3)  # 90% of limit

    # Binary search bounds
    low, high = 1, max_batch_size
    optimal_batch_size = 1
    memory_usage_profile = []

    while low <= high:
        mid = (low + high) // 2

        try:
            torch.cuda.empty_cache()

            # Test batch size
            test_input = torch.randn(mid, *input_shape, device=device)

            with torch.no_grad():
                _ = model(test_input)

            current_memory = torch.cuda.memory_allocated()

            memory_usage_profile.append({
                'batch_size': mid,
                'memory_gb': current_memory / (1024**3),
                'success': True
            })

            if current_memory <= memory_threshold:
                optimal_batch_size = mid
                low = mid + 1  # Try larger batch size
            else:
                high = mid - 1  # Try smaller batch size

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                memory_usage_profile.append({
                    'batch_size': mid,
                    'memory_gb': None,
                    'success': False,
                    'error': 'OOM'
                })
                high = mid - 1  # Try smaller batch size
            else:
                raise

    # Clean up
    torch.cuda.empty_cache()

    # Apply safety factor
    safe_batch_size = max(1, int(optimal_batch_size * 0.8))

    return {
        'optimal_batch_size': safe_batch_size,
        'max_tested_batch_size': optimal_batch_size,
        'memory_usage_profile': memory_usage_profile,
        'method': 'binary_search',
        'memory_limit_gb': max_memory_gb,
        'safety_factor': 0.8
    }
