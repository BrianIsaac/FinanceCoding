"""
Memory management and performance optimization utilities.

This module provides comprehensive memory management capabilities for
rolling backtest execution including batch processing, memory monitoring,
and performance optimization for large-scale evaluations.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    timestamp: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    total_memory_gb: float = 0.0
    used_memory_gb: float = 0.0
    available_memory_gb: float = 0.0
    memory_percent: float = 0.0
    process_memory_gb: float = 0.0
    process_memory_percent: float = 0.0
    gpu_memory_gb: float = 0.0
    gpu_memory_percent: float = 0.0


@dataclass
class PerformanceStats:
    """Performance monitoring statistics."""

    timestamp: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    cpu_percent: float = 0.0
    cpu_count: int = 0
    load_average: tuple[float, float, float] = (0.0, 0.0, 0.0)
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_io_sent_mb: float = 0.0
    network_io_recv_mb: float = 0.0


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing."""

    batch_size: int = 32
    max_memory_gb: float = 16.0
    memory_threshold: float = 0.8  # 80% memory usage threshold
    enable_gpu_batching: bool = False
    gpu_memory_threshold: float = 0.9
    parallel_workers: int = 4
    prefetch_batches: int = 2
    enable_caching: bool = True
    cache_size_gb: float = 2.0


class MemoryManager:
    """
    Comprehensive memory management system for large-scale backtesting.

    This manager provides:
    - Real-time memory monitoring and alerting
    - Automatic garbage collection and cleanup
    - Memory-efficient data loading and processing
    - Batch processing for large datasets
    - GPU memory management integration
    - Performance optimization recommendations
    """

    def __init__(self, config: BatchProcessingConfig):
        """Initialize memory manager."""
        self.config = config
        self.monitoring_active = False
        self.monitoring_thread: threading.Thread | None = None

        # Statistics tracking
        self.memory_history: list[MemoryStats] = []
        self.performance_history: list[PerformanceStats] = []

        # Alerts and notifications
        self.alert_callbacks: list[callable] = []
        self.last_alert_time: pd.Timestamp | None = None
        self.alert_cooldown_seconds: int = 60

        # Cache management
        self.data_cache: dict[str, Any] = {}
        self.cache_access_times: dict[str, pd.Timestamp] = {}
        self.max_cache_size_bytes = int(config.cache_size_gb * 1024**3)

    def start_monitoring(self, interval_seconds: int = 30) -> None:
        """Start continuous memory monitoring."""

        if self.monitoring_active:
            logger.warning("Memory monitoring already active")
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, args=(interval_seconds,), daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Memory monitoring started")

    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Memory monitoring stopped")

    def get_current_memory_stats(self) -> MemoryStats:
        """Get current memory usage statistics."""

        # System memory
        memory = psutil.virtual_memory()

        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info()

        stats = MemoryStats(
            total_memory_gb=memory.total / (1024**3),
            used_memory_gb=memory.used / (1024**3),
            available_memory_gb=memory.available / (1024**3),
            memory_percent=memory.percent,
            process_memory_gb=process_memory.rss / (1024**3),
            process_memory_percent=(process_memory.rss / memory.total) * 100,
        )

        # GPU memory if available
        try:
            import torch

            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_stats()
                allocated = gpu_memory.get("allocated_bytes.all.current", 0) / (1024**3)
                cached = gpu_memory.get("reserved_bytes.all.current", 0) / (1024**3)
                stats.gpu_memory_gb = allocated + cached

                device_props = torch.cuda.get_device_properties(0)
                total_gpu_memory = device_props.total_memory / (1024**3)
                stats.gpu_memory_percent = (stats.gpu_memory_gb / total_gpu_memory) * 100
        except ImportError:
            pass

        return stats

    def get_current_performance_stats(self) -> PerformanceStats:
        """Get current performance statistics."""

        stats = PerformanceStats(
            cpu_percent=psutil.cpu_percent(interval=1),
            cpu_count=psutil.cpu_count(),
        )

        # Load average (Unix systems only)
        try:
            stats.load_average = psutil.getloadavg()
        except AttributeError:
            pass

        # Disk I/O
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                stats.disk_io_read_mb = disk_io.read_bytes / (1024**2)
                stats.disk_io_write_mb = disk_io.write_bytes / (1024**2)
        except Exception:
            pass

        # Network I/O
        try:
            net_io = psutil.net_io_counters()
            if net_io:
                stats.network_io_sent_mb = net_io.bytes_sent / (1024**2)
                stats.network_io_recv_mb = net_io.bytes_recv / (1024**2)
        except Exception:
            pass

        return stats

    def check_memory_pressure(self) -> dict[str, Any]:
        """Check for memory pressure and recommend actions."""

        stats = self.get_current_memory_stats()

        pressure_indicators = {
            "memory_pressure": False,
            "gpu_pressure": False,
            "recommendations": [],
        }

        # Check system memory pressure
        if stats.memory_percent > self.config.memory_threshold * 100:
            pressure_indicators["memory_pressure"] = True
            pressure_indicators["recommendations"].append("Reduce batch size")
            pressure_indicators["recommendations"].append("Enable aggressive garbage collection")

        # Check process memory usage
        if stats.process_memory_gb > self.config.max_memory_gb * 0.8:
            pressure_indicators["memory_pressure"] = True
            pressure_indicators["recommendations"].append("Clear data caches")
            pressure_indicators["recommendations"].append("Process data in smaller chunks")

        # Check GPU memory pressure
        if stats.gpu_memory_percent > self.config.gpu_memory_threshold * 100:
            pressure_indicators["gpu_pressure"] = True
            pressure_indicators["recommendations"].append("Clear GPU cache")
            pressure_indicators["recommendations"].append("Reduce model batch size")

        return pressure_indicators

    def optimize_memory_usage(self) -> dict[str, Any]:
        """Optimize memory usage with cleanup and tuning."""

        optimization_results = {
            "actions_taken": [],
            "memory_freed_gb": 0.0,
            "before_stats": self.get_current_memory_stats(),
        }

        # Clear Python garbage
        len(gc.get_objects())
        collected = gc.collect()
        optimization_results["actions_taken"].append(
            f"Garbage collection: {collected} objects collected"
        )

        # Clear data cache
        cache_size_before = len(self.data_cache)
        self._cleanup_cache()
        cache_size_after = len(self.data_cache)
        if cache_size_before > cache_size_after:
            optimization_results["actions_taken"].append(
                f"Cache cleanup: {cache_size_before - cache_size_after} items removed"
            )

        # Clear GPU cache if available
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                optimization_results["actions_taken"].append("GPU cache cleared")
        except ImportError:
            pass

        # Get final memory stats
        after_stats = self.get_current_memory_stats()
        memory_freed = (
            optimization_results["before_stats"].used_memory_gb - after_stats.used_memory_gb
        )
        optimization_results["memory_freed_gb"] = memory_freed
        optimization_results["after_stats"] = after_stats

        logger.info(f"Memory optimization freed {memory_freed:.2f} GB")

        return optimization_results

    def process_in_batches(
        self, data: pd.DataFrame | list[Any], processing_func: callable, **kwargs
    ) -> list[Any]:
        """Process data in memory-efficient batches."""

        if isinstance(data, pd.DataFrame):
            total_size = len(data)
            batch_generator = (
                data.iloc[i : i + self.config.batch_size]
                for i in range(0, total_size, self.config.batch_size)
            )
        else:
            total_size = len(data)
            batch_generator = (
                data[i : i + self.config.batch_size]
                for i in range(0, total_size, self.config.batch_size)
            )

        results = []
        processed_count = 0

        for batch_idx, batch in enumerate(batch_generator):
            # Check memory pressure before processing each batch
            if batch_idx % 10 == 0:  # Check every 10 batches
                pressure_check = self.check_memory_pressure()
                if pressure_check["memory_pressure"]:
                    logger.warning("Memory pressure detected, optimizing...")
                    self.optimize_memory_usage()

            try:
                batch_result = processing_func(batch, **kwargs)
                results.append(batch_result)
                processed_count += len(batch) if hasattr(batch, "__len__") else 1

                if batch_idx % 100 == 0:
                    logger.debug(
                        f"Processed {processed_count}/{total_size} items in {batch_idx+1} batches"
                    )

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue

        logger.info(f"Batch processing complete: {len(results)} batches processed")
        return results

    def cache_data(self, key: str, data: Any, force: bool = False) -> bool:
        """Cache data with automatic size management."""

        if not self.config.enable_caching and not force:
            return False

        # Estimate data size
        data_size = self._estimate_object_size(data)

        # Check if we have enough cache space
        current_cache_size = sum(
            self._estimate_object_size(obj) for obj in self.data_cache.values()
        )

        if current_cache_size + data_size > self.max_cache_size_bytes:
            # Clean up oldest items
            self._cleanup_cache_by_size(data_size)

        # Cache the data
        self.data_cache[key] = data
        self.cache_access_times[key] = pd.Timestamp.now()

        logger.debug(f"Cached data with key '{key}' ({data_size / (1024**2):.2f} MB)")
        return True

    def get_cached_data(self, key: str) -> Any | None:
        """Retrieve data from cache."""

        if key in self.data_cache:
            self.cache_access_times[key] = pd.Timestamp.now()
            return self.data_cache[key]

        return None

    def _monitoring_loop(self, interval_seconds: int) -> None:
        """Main monitoring loop running in background thread."""

        while self.monitoring_active:
            try:
                # Collect statistics
                memory_stats = self.get_current_memory_stats()
                performance_stats = self.get_current_performance_stats()

                # Store in history
                self.memory_history.append(memory_stats)
                self.performance_history.append(performance_stats)

                # Limit history size
                max_history = 1000
                if len(self.memory_history) > max_history:
                    self.memory_history = self.memory_history[-max_history:]
                if len(self.performance_history) > max_history:
                    self.performance_history = self.performance_history[-max_history:]

                # Check for alerts
                self._check_alerts(memory_stats, performance_stats)

                time.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)

    def _check_alerts(self, memory_stats: MemoryStats, performance_stats: PerformanceStats) -> None:
        """Check for alert conditions and trigger notifications."""

        current_time = pd.Timestamp.now()

        # Cooldown check
        if (
            self.last_alert_time
            and (current_time - self.last_alert_time).total_seconds() < self.alert_cooldown_seconds
        ):
            return

        alert_triggered = False

        # Memory alerts
        if memory_stats.memory_percent > 90:
            self._trigger_alert(f"Critical memory usage: {memory_stats.memory_percent:.1f}%")
            alert_triggered = True
        elif memory_stats.memory_percent > 80:
            self._trigger_alert(f"High memory usage: {memory_stats.memory_percent:.1f}%")
            alert_triggered = True

        # Process memory alerts
        if memory_stats.process_memory_gb > self.config.max_memory_gb:
            self._trigger_alert(
                f"Process memory limit exceeded: {memory_stats.process_memory_gb:.2f} GB"
            )
            alert_triggered = True

        # GPU memory alerts
        if memory_stats.gpu_memory_percent > 95:
            self._trigger_alert(
                f"Critical GPU memory usage: {memory_stats.gpu_memory_percent:.1f}%"
            )
            alert_triggered = True

        # Performance alerts
        if performance_stats.cpu_percent > 95:
            self._trigger_alert(f"High CPU usage: {performance_stats.cpu_percent:.1f}%")
            alert_triggered = True

        if alert_triggered:
            self.last_alert_time = current_time

    def _trigger_alert(self, message: str) -> None:
        """Trigger alert notification."""

        logger.warning(f"ALERT: {message}")

        for callback in self.alert_callbacks:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def add_alert_callback(self, callback: callable) -> None:
        """Add alert notification callback."""
        self.alert_callbacks.append(callback)

    def _cleanup_cache(self) -> None:
        """Clean up old cache entries."""

        if not self.cache_access_times:
            return

        current_time = pd.Timestamp.now()
        cache_ttl_hours = 2  # Time to live: 2 hours

        expired_keys = [
            key
            for key, access_time in self.cache_access_times.items()
            if (current_time - access_time).total_seconds() > cache_ttl_hours * 3600
        ]

        for key in expired_keys:
            if key in self.data_cache:
                del self.data_cache[key]
            if key in self.cache_access_times:
                del self.cache_access_times[key]

    def _cleanup_cache_by_size(self, required_size: int) -> None:
        """Clean up cache entries to make space for new data."""

        # Sort by access time (oldest first)
        sorted_keys = sorted(
            self.cache_access_times.keys(), key=lambda k: self.cache_access_times[k]
        )

        freed_size = 0
        for key in sorted_keys:
            if freed_size >= required_size:
                break

            if key in self.data_cache:
                data_size = self._estimate_object_size(self.data_cache[key])
                del self.data_cache[key]
                del self.cache_access_times[key]
                freed_size += data_size

    def _estimate_object_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""

        try:
            if isinstance(obj, pd.DataFrame):
                return obj.memory_usage(deep=True).sum()
            elif isinstance(obj, pd.Series):
                return obj.memory_usage(deep=True)
            elif isinstance(obj, np.ndarray):
                return obj.nbytes
            else:
                # Rough estimation using sys.getsizeof
                import sys

                return sys.getsizeof(obj)
        except Exception:
            return 1024  # Default 1KB estimate

    def get_monitoring_summary(self) -> dict[str, Any]:
        """Get monitoring summary statistics."""

        if not self.memory_history or not self.performance_history:
            return {"message": "No monitoring data available"}

        recent_memory = self.memory_history[-100:]  # Last 100 measurements
        recent_performance = self.performance_history[-100:]

        return {
            "monitoring_active": self.monitoring_active,
            "measurements_count": len(self.memory_history),
            "current_memory": {
                "used_gb": recent_memory[-1].used_memory_gb,
                "percent": recent_memory[-1].memory_percent,
                "process_gb": recent_memory[-1].process_memory_gb,
            },
            "memory_trends": {
                "avg_usage_percent": np.mean([m.memory_percent for m in recent_memory]),
                "max_usage_percent": max([m.memory_percent for m in recent_memory]),
                "avg_process_gb": np.mean([m.process_memory_gb for m in recent_memory]),
            },
            "performance_trends": {
                "avg_cpu_percent": np.mean([p.cpu_percent for p in recent_performance]),
                "max_cpu_percent": max([p.cpu_percent for p in recent_performance]),
            },
            "cache_stats": {
                "cached_items": len(self.data_cache),
                "cache_hit_ratio": 0.0,  # Would need to track hits/misses
            },
        }

    def create_performance_report(self, output_path: Path | None = None) -> dict[str, Any]:
        """Create comprehensive performance and memory report."""

        report = {
            "generated_at": pd.Timestamp.now().isoformat(),
            "monitoring_summary": self.get_monitoring_summary(),
            "memory_history": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "memory_percent": m.memory_percent,
                    "process_memory_gb": m.process_memory_gb,
                    "gpu_memory_percent": m.gpu_memory_percent,
                }
                for m in self.memory_history[-100:]  # Last 100 measurements
            ],
            "performance_history": [
                {
                    "timestamp": p.timestamp.isoformat(),
                    "cpu_percent": p.cpu_percent,
                    "disk_io_read_mb": p.disk_io_read_mb,
                    "disk_io_write_mb": p.disk_io_write_mb,
                }
                for p in self.performance_history[-100:]
            ],
        }

        if output_path:
            import json

            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Performance report saved to {output_path}")

        return report


class BatchProcessor:
    """
    High-performance batch processor for large-scale backtesting.

    Provides memory-efficient processing of large datasets with:
    - Automatic batch size optimization based on memory usage
    - Parallel processing capabilities
    - Progress tracking and monitoring
    - Error handling and recovery
    """

    def __init__(self, memory_manager: MemoryManager):
        """Initialize batch processor."""
        self.memory_manager = memory_manager
        self.processing_stats = {
            "batches_processed": 0,
            "items_processed": 0,
            "processing_time": 0.0,
            "errors": 0,
        }

    def process_backtest_splits(
        self,
        splits: list[Any],
        models: dict[str, Any],
        data: dict[str, pd.DataFrame],
        processing_func: callable,
    ) -> list[dict[str, Any]]:
        """Process backtest splits in memory-efficient batches."""

        logger.info(f"Starting batch processing of {len(splits)} splits")

        # Optimize batch size based on available memory
        optimal_batch_size = self._optimize_batch_size(splits[0] if splits else None, data)

        results = []
        start_time = pd.Timestamp.now()

        for i in range(0, len(splits), optimal_batch_size):
            batch_splits = splits[i : i + optimal_batch_size]

            try:
                # Process batch
                batch_results = processing_func(batch_splits, models, data)
                results.extend(batch_results)

                # Update statistics
                self.processing_stats["batches_processed"] += 1
                self.processing_stats["items_processed"] += len(batch_splits)

                # Log progress
                progress = (i + len(batch_splits)) / len(splits) * 100
                logger.info(f"Batch processing progress: {progress:.1f}%")

                # Check memory pressure
                if i % 5 == 0:  # Every 5 batches
                    pressure_check = self.memory_manager.check_memory_pressure()
                    if pressure_check["memory_pressure"]:
                        self.memory_manager.optimize_memory_usage()

            except Exception as e:
                logger.error(f"Error processing batch starting at index {i}: {e}")
                self.processing_stats["errors"] += 1
                continue

        # Update final statistics
        end_time = pd.Timestamp.now()
        self.processing_stats["processing_time"] = (end_time - start_time).total_seconds()

        logger.info(
            f"Batch processing complete: {len(results)} results, "
            f"{self.processing_stats['errors']} errors, "
            f"{self.processing_stats['processing_time']:.2f}s"
        )

        return results

    def _optimize_batch_size(self, sample_split: Any, data: dict[str, pd.DataFrame]) -> int:
        """Optimize batch size based on available memory and data characteristics."""

        base_batch_size = self.memory_manager.config.batch_size

        # Check current memory usage
        memory_stats = self.memory_manager.get_current_memory_stats()

        # If memory usage is high, reduce batch size
        if memory_stats.memory_percent > 70:
            return max(1, base_batch_size // 2)
        elif memory_stats.memory_percent < 40:
            return min(base_batch_size * 2, 64)  # Cap at 64

        return base_batch_size

    def get_processing_stats(self) -> dict[str, Any]:
        """Get processing statistics."""
        return self.processing_stats.copy()
