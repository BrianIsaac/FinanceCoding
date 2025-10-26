"""Batch Processing Architecture for Memory Constraint Compliance.

This module provides batch processing architecture design and validation ensuring
<12GB GPU and <32GB RAM compliance for comprehensive results validation.
"""

import gc
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import psutil
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BatchConfig:
    """Configuration for batch processing operations."""
    max_batch_size: int = 1000
    gpu_memory_limit_gb: float = 11.0  # Conservative limit
    system_memory_limit_gb: float = 30.0  # Conservative limit
    max_concurrent_batches: int = 2
    enable_memory_monitoring: bool = True
    garbage_collection_frequency: int = 5  # GC every N batches

@dataclass
class MemoryUsage:
    """Memory usage tracking data structure."""
    gpu_memory_gb: float = 0.0
    system_memory_gb: float = 0.0
    timestamp: datetime = None
    operation: str = ""

class BatchProcessor:
    """Memory-efficient batch processing architecture."""

    def __init__(self, config: Optional[BatchConfig] = None):
        """Initialise batch processor with memory constraints.

        Args:
            config: Batch processing configuration
        """
        self.config = config or BatchConfig()
        self.memory_history: list[MemoryUsage] = []
        self.processing_stats: dict[str, Any] = {}

        # Check if CUDA is available for GPU monitoring
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            logger.info(f"CUDA available. GPU count: {torch.cuda.device_count()}")
        else:
            logger.info("CUDA not available. GPU monitoring disabled.")

    def _monitor_memory_usage(self, operation: str = "") -> MemoryUsage:
        """Monitor current memory usage.

        Args:
            operation: Name of current operation

        Returns:
            Current memory usage data
        """
        # System memory
        process = psutil.Process()
        system_memory_gb = process.memory_info().rss / 1024 / 1024 / 1024

        # GPU memory
        gpu_memory_gb = 0.0
        if self.cuda_available and torch.cuda.is_available():
            try:
                gpu_memory_bytes = torch.cuda.memory_allocated()
                gpu_memory_gb = gpu_memory_bytes / 1024 / 1024 / 1024
            except Exception as e:
                logger.warning(f"Error reading GPU memory: {e}")

        usage = MemoryUsage(
            gpu_memory_gb=gpu_memory_gb,
            system_memory_gb=system_memory_gb,
            timestamp=datetime.now(),
            operation=operation
        )

        if self.config.enable_memory_monitoring:
            self.memory_history.append(usage)

        return usage

    def _check_memory_constraints(self, current_usage: MemoryUsage) -> dict[str, Any]:
        """Check if current memory usage meets constraints.

        Args:
            current_usage: Current memory usage

        Returns:
            Dictionary with constraint check results
        """
        gpu_compliant = (not self.cuda_available or
                        current_usage.gpu_memory_gb <= self.config.gpu_memory_limit_gb)
        system_compliant = current_usage.system_memory_gb <= self.config.system_memory_limit_gb

        return {
            'gpu_compliant': gpu_compliant,
            'system_compliant': system_compliant,
            'overall_compliant': gpu_compliant and system_compliant,
            'gpu_usage_gb': current_usage.gpu_memory_gb,
            'system_usage_gb': current_usage.system_memory_gb,
            'gpu_limit_gb': self.config.gpu_memory_limit_gb,
            'system_limit_gb': self.config.system_memory_limit_gb
        }

    def _force_garbage_collection(self):
        """Force garbage collection to free memory."""
        gc.collect()
        if self.cuda_available and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def create_data_batches(self, data: np.ndarray,
                           batch_size: Optional[int] = None) -> Iterator[tuple[int, np.ndarray]]:
        """Create memory-efficient data batches.

        Args:
            data: Input data array
            batch_size: Size of each batch (uses config default if None)

        Yields:
            Tuples of (batch_index, batch_data)
        """
        if batch_size is None:
            batch_size = self.config.max_batch_size

        n_samples = data.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        logger.info(f"Creating {n_batches} batches of size {batch_size} from {n_samples} samples")

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)

            batch_data = data[start_idx:end_idx]

            # Memory monitoring
            if i % self.config.garbage_collection_frequency == 0:
                self._force_garbage_collection()

            usage = self._monitor_memory_usage(f"batch_creation_{i}")
            constraint_check = self._check_memory_constraints(usage)

            if not constraint_check['overall_compliant']:
                logger.warning(f"Memory constraint violation at batch {i}: "
                             f"GPU {usage.gpu_memory_gb:.2f}GB, "
                             f"System {usage.system_memory_gb:.2f}GB")

            yield i, batch_data

    def process_performance_metrics_batch(self, returns_batch: np.ndarray) -> dict[str, np.ndarray]:
        """Process performance metrics for a batch of return data.

        Args:
            returns_batch: Batch of return data (samples x strategies)

        Returns:
            Dictionary of calculated performance metrics
        """
        batch_size, n_strategies = returns_batch.shape

        # Pre-allocate result arrays
        sharpe_ratios = np.zeros(n_strategies)
        information_ratios = np.zeros(n_strategies)
        max_drawdowns = np.zeros(n_strategies)
        var_95_values = np.zeros(n_strategies)
        cvar_95_values = np.zeros(n_strategies)

        # Calculate metrics for each strategy in the batch
        for strategy_idx in range(n_strategies):
            returns = returns_batch[:, strategy_idx]

            # Sharpe ratio
            sharpe_ratios[strategy_idx] = (np.mean(returns) / np.std(returns, ddof=1) *
                                         np.sqrt(252) if np.std(returns, ddof=1) > 0 else 0)

            # Information ratio (vs first strategy as benchmark)
            if strategy_idx > 0:
                benchmark_returns = returns_batch[:, 0]
                excess_returns = returns - benchmark_returns
                information_ratios[strategy_idx] = (np.mean(excess_returns) /
                                                  np.std(excess_returns, ddof=1) * np.sqrt(252)
                                                  if np.std(excess_returns, ddof=1) > 0 else 0)

            # Maximum drawdown
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdowns[strategy_idx] = np.min(drawdown)

            # VaR and CVaR
            var_95 = np.percentile(returns, 5)
            var_95_values[strategy_idx] = var_95
            tail_returns = returns[returns <= var_95]
            cvar_95_values[strategy_idx] = np.mean(tail_returns) if len(tail_returns) > 0 else var_95

        return {
            'sharpe_ratios': sharpe_ratios,
            'information_ratios': information_ratios,
            'max_drawdowns': max_drawdowns,
            'var_95_values': var_95_values,
            'cvar_95_values': cvar_95_values
        }

    def process_correlation_matrix_batch(self, returns_batch: np.ndarray) -> np.ndarray:
        """Process correlation matrix for a batch of return data.

        Args:
            returns_batch: Batch of return data (samples x strategies)

        Returns:
            Correlation matrix for the batch
        """
        try:
            correlation_matrix = np.corrcoef(returns_batch.T)
            return correlation_matrix
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            # Return identity matrix as fallback
            n_strategies = returns_batch.shape[1]
            return np.eye(n_strategies)

    def process_statistical_tests_batch(self, returns_batch1: np.ndarray,
                                       returns_batch2: np.ndarray) -> dict[str, float]:
        """Process statistical tests for a batch of return comparisons.

        Args:
            returns_batch1: First batch of returns
            returns_batch2: Second batch of returns

        Returns:
            Dictionary with statistical test results
        """
        try:
            from scipy import stats

            # Jobson-Korkie test components
            sharpe1 = np.mean(returns_batch1) / np.std(returns_batch1, ddof=1) * np.sqrt(252)
            sharpe2 = np.mean(returns_batch2) / np.std(returns_batch2, ddof=1) * np.sqrt(252)

            # Basic test statistic (simplified for batch processing)
            theta = sharpe1 - sharpe2

            # T-test for difference in means (as proxy for statistical significance)
            t_stat, p_value = stats.ttest_ind(returns_batch1, returns_batch2)

            return {
                'sharpe_difference': theta,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }

        except ImportError:
            logger.warning("scipy not available for statistical tests")
            return {
                'sharpe_difference': 0.0,
                't_statistic': 0.0,
                'p_value': 1.0,
                'significant': False
            }
        except Exception as e:
            logger.error(f"Error in statistical test: {e}")
            return {
                'sharpe_difference': 0.0,
                't_statistic': 0.0,
                'p_value': 1.0,
                'significant': False
            }

    def validate_batch_processing_architecture(self, test_data_size: int = 10000) -> dict[str, Any]:
        """Validate batch processing architecture with test data.

        Args:
            test_data_size: Size of test dataset

        Returns:
            Validation results for batch processing architecture
        """
        logger.info(f"Validating batch processing architecture with {test_data_size} samples")

        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'test_configuration': {
                'test_data_size': test_data_size,
                'batch_config': {
                    'max_batch_size': self.config.max_batch_size,
                    'gpu_memory_limit_gb': self.config.gpu_memory_limit_gb,
                    'system_memory_limit_gb': self.config.system_memory_limit_gb,
                    'max_concurrent_batches': self.config.max_concurrent_batches
                }
            },
            'validation_tests': {},
            'memory_compliance': {},
            'performance_results': {},
            'overall_status': 'unknown'
        }

        # Generate test data
        logger.info("Generating test return data...")
        test_returns = np.random.normal(0.001, 0.02, size=(test_data_size, 7))  # 7 strategies

        # Test 1: Performance metrics batch processing
        logger.info("Testing performance metrics batch processing...")
        performance_results = []
        memory_violations = 0

        for batch_idx, batch_data in self.create_data_batches(test_returns):
            self._monitor_memory_usage(f"batch_{batch_idx}_before")

            batch_metrics = self.process_performance_metrics_batch(batch_data)
            performance_results.append(batch_metrics)

            usage_after = self._monitor_memory_usage(f"batch_{batch_idx}_after")
            constraint_check = self._check_memory_constraints(usage_after)

            if not constraint_check['overall_compliant']:
                memory_violations += 1

        validation_results['validation_tests']['performance_metrics_batching'] = {
            'status': 'pass' if memory_violations == 0 else 'fail',
            'batches_processed': len(performance_results),
            'memory_violations': memory_violations,
            'error_message': None if memory_violations == 0 else f"{memory_violations} memory violations"
        }

        # Test 2: Correlation matrix batch processing
        logger.info("Testing correlation matrix batch processing...")
        correlation_violations = 0

        for batch_idx, batch_data in self.create_data_batches(test_returns, batch_size=500):  # Smaller batches for correlation
            self._monitor_memory_usage(f"corr_batch_{batch_idx}_before")

            self.process_correlation_matrix_batch(batch_data)

            usage_after = self._monitor_memory_usage(f"corr_batch_{batch_idx}_after")
            constraint_check = self._check_memory_constraints(usage_after)

            if not constraint_check['overall_compliant']:
                correlation_violations += 1

        validation_results['validation_tests']['correlation_matrix_batching'] = {
            'status': 'pass' if correlation_violations == 0 else 'fail',
            'batches_processed': batch_idx + 1,
            'memory_violations': correlation_violations,
            'error_message': None if correlation_violations == 0 else f"{correlation_violations} memory violations"
        }

        # Test 3: Memory usage analysis
        if self.memory_history:
            max_gpu_memory = max(usage.gpu_memory_gb for usage in self.memory_history)
            max_system_memory = max(usage.system_memory_gb for usage in self.memory_history)
            avg_gpu_memory = np.mean([usage.gpu_memory_gb for usage in self.memory_history])
            avg_system_memory = np.mean([usage.system_memory_gb for usage in self.memory_history])

            validation_results['memory_compliance'] = {
                'max_gpu_memory_gb': max_gpu_memory,
                'max_system_memory_gb': max_system_memory,
                'avg_gpu_memory_gb': avg_gpu_memory,
                'avg_system_memory_gb': avg_system_memory,
                'gpu_compliant': max_gpu_memory <= self.config.gpu_memory_limit_gb,
                'system_compliant': max_system_memory <= self.config.system_memory_limit_gb,
                'memory_samples': len(self.memory_history)
            }

        # Overall status
        all_tests_passed = all(
            test['status'] == 'pass'
            for test in validation_results['validation_tests'].values()
        )
        memory_compliant = (validation_results['memory_compliance'].get('gpu_compliant', True) and
                           validation_results['memory_compliance'].get('system_compliant', True))

        validation_results['overall_status'] = 'pass' if all_tests_passed and memory_compliant else 'fail'

        # Performance summary
        validation_results['performance_results'] = {
            'total_batches_processed': len(performance_results),
            'memory_monitoring_samples': len(self.memory_history),
            'architecture_validated': validation_results['overall_status'] == 'pass',
            'ready_for_production': validation_results['overall_status'] == 'pass'
        }

        logger.info(f"Batch processing architecture validation: {validation_results['overall_status']}")

        return validation_results

    def get_memory_usage_summary(self) -> pd.DataFrame:
        """Get memory usage history as DataFrame.

        Returns:
            DataFrame with memory usage history
        """
        if not self.memory_history:
            return pd.DataFrame()

        summary_data = []
        for usage in self.memory_history:
            summary_data.append({
                'Timestamp': usage.timestamp,
                'Operation': usage.operation,
                'GPU_Memory_GB': usage.gpu_memory_gb,
                'System_Memory_GB': usage.system_memory_gb,
                'GPU_Within_Limit': usage.gpu_memory_gb <= self.config.gpu_memory_limit_gb,
                'System_Within_Limit': usage.system_memory_gb <= self.config.system_memory_limit_gb
            })

        return pd.DataFrame(summary_data)

    def create_processing_recommendations(self) -> dict[str, Any]:
        """Create processing recommendations based on validation results.

        Returns:
            Dictionary with processing recommendations
        """
        recommendations = {
            'batch_size_recommendations': {
                'performance_metrics': self.config.max_batch_size,
                'correlation_matrices': min(self.config.max_batch_size, 500),  # Smaller for memory-intensive ops
                'statistical_tests': min(self.config.max_batch_size, 200)  # Even smaller for pairwise comparisons
            },
            'memory_management': {
                'garbage_collection_frequency': self.config.garbage_collection_frequency,
                'enable_memory_monitoring': True,
                'gpu_memory_buffer_gb': 1.0,  # Keep 1GB buffer
                'system_memory_buffer_gb': 2.0  # Keep 2GB buffer
            },
            'processing_strategy': {
                'concurrent_batches': 1 if self.cuda_available else self.config.max_concurrent_batches,
                'prioritize_gpu_operations': self.cuda_available,
                'enable_checkpointing': True,
                'fallback_cpu_processing': True
            }
        }

        return recommendations
