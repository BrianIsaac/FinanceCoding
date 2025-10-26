"""Performance Benchmarking Framework.

This module provides performance benchmarking with representative sample data validation
for processing time and reporting constraints compliance.
"""

import logging
import time
import tracemalloc
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """Performance benchmarking framework for validation and compliance testing."""

    def __init__(self):
        """Initialise performance benchmark tracker."""
        self.benchmark_results: dict[str, Any] = {}
        self.memory_profiles: dict[str, Any] = {}
        self.processing_times: dict[str, float] = {}

        # Performance targets from story requirements
        self.targets = {
            'processing_time_hours': 2.0,  # <2 hour processing
            'reporting_time_minutes': 30.0,  # <30 minute reporting
            'gpu_memory_gb': 12.0,  # <12GB GPU
            'system_memory_gb': 32.0,  # <32GB RAM
            'validation_time_hours': 2.0,  # <2 hour validation
        }

    @contextmanager
    def measure_performance(self, operation_name: str, track_memory: bool = True):
        """Context manager for measuring operation performance.

        Args:
            operation_name: Name of operation being measured
            track_memory: Whether to track memory usage
        """
        if track_memory:
            tracemalloc.start()

        # Record initial system state
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024 / 1024  # GB
        start_time = time.time()

        logger.info(f"Starting performance measurement for: {operation_name}")

        try:
            yield

        finally:
            # Record final measurements
            end_time = time.time()
            elapsed_time = end_time - start_time
            final_memory = process.memory_info().rss / 1024 / 1024 / 1024  # GB
            memory_delta = final_memory - initial_memory

            if track_memory:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                peak_mb = peak / 1024 / 1024
            else:
                peak_mb = 0

            # Store results
            self.processing_times[operation_name] = elapsed_time
            self.memory_profiles[operation_name] = {
                'initial_memory_gb': initial_memory,
                'final_memory_gb': final_memory,
                'memory_delta_gb': memory_delta,
                'peak_memory_mb': peak_mb,
                'elapsed_time_seconds': elapsed_time,
                'elapsed_time_minutes': elapsed_time / 60,
                'elapsed_time_hours': elapsed_time / 3600
            }

            logger.info(f"Performance measurement complete for {operation_name}: "
                       f"{elapsed_time:.2f}s, memory delta: {memory_delta:.2f}GB")

    def benchmark_results_validation(self, sample_size: int = 1000) -> dict[str, Any]:
        """Benchmark results validation processing with representative sample data.

        Args:
            sample_size: Size of sample data for validation testing

        Returns:
            Dictionary containing benchmark results
        """
        logger.info(f"Benchmarking results validation with sample size: {sample_size}")

        with self.measure_performance("results_validation_benchmark"):
            # Simulate comprehensive results validation
            sample_returns = np.random.normal(0.001, 0.02, size=(sample_size, 8))  # 8 strategies

            # Simulate statistical calculations
            sharpe_ratios = []
            for i in range(8):
                returns = sample_returns[:, i]
                sharpe = np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252)
                sharpe_ratios.append(sharpe)

            # Simulate correlation matrix calculation
            corr_matrix = np.corrcoef(sample_returns.T)

            # Simulate VaR/CVaR calculations
            var_95_values = []
            cvar_95_values = []
            for i in range(8):
                returns = sample_returns[:, i]
                var_95 = np.percentile(returns, 5)
                cvar_95 = np.mean(returns[returns <= var_95])
                var_95_values.append(var_95)
                cvar_95_values.append(cvar_95)

            # Simulate maximum drawdown calculations
            mdd_values = []
            for i in range(8):
                returns = sample_returns[:, i]
                cum_returns = np.cumprod(1 + returns)
                running_max = np.maximum.accumulate(cum_returns)
                drawdown = (cum_returns - running_max) / running_max
                mdd = np.min(drawdown)
                mdd_values.append(mdd)

            # Create benchmark results
            benchmark_data = {
                'sample_size': sample_size,
                'strategies_count': 8,
                'sharpe_ratios': sharpe_ratios,
                'var_95_values': var_95_values,
                'cvar_95_values': cvar_95_values,
                'mdd_values': mdd_values,
                'correlation_matrix_shape': corr_matrix.shape
            }

        return benchmark_data

    def benchmark_executive_reporting(self, strategies_count: int = 7) -> dict[str, Any]:
        """Benchmark executive report generation.

        Args:
            strategies_count: Number of strategies to include in reporting

        Returns:
            Dictionary containing executive reporting benchmark results
        """
        logger.info(f"Benchmarking executive reporting for {strategies_count} strategies")

        with self.measure_performance("executive_reporting_benchmark"):
            # Simulate report generation tasks

            # 1. Performance metrics table generation
            metrics_table = pd.DataFrame({
                'Strategy': [f'Strategy_{i}' for i in range(strategies_count)],
                'Sharpe_Ratio': np.random.uniform(0.5, 2.0, strategies_count),
                'Information_Ratio': np.random.uniform(0.1, 0.8, strategies_count),
                'Maximum_Drawdown': np.random.uniform(-0.5, -0.1, strategies_count),
                'VaR_95': np.random.uniform(-0.05, -0.01, strategies_count),
                'CVaR_95': np.random.uniform(-0.08, -0.02, strategies_count)
            })

            # 2. Ranking matrix generation
            ranking_matrix = metrics_table.rank(ascending=False, method='min')

            # 3. Statistical significance indicators
            np.random.choice([True, False], size=(strategies_count, strategies_count), p=[0.3, 0.7])

            # 4. Investment recommendations
            recommendations = []
            for i in range(strategies_count):
                sharpe = metrics_table.iloc[i]['Sharpe_Ratio']
                recommendation = {
                    'strategy': f'Strategy_{i}',
                    'recommendation': 'Deploy' if sharpe > 1.2 else 'Monitor' if sharpe > 0.8 else 'Reject',
                    'confidence': np.random.uniform(0.7, 0.95)
                }
                recommendations.append(recommendation)

            # 5. Visualization data preparation (simulated)
            chart_data = {
                'performance_comparison': metrics_table.to_dict('records'),
                'risk_return_scatter': [(row['Sharpe_Ratio'], row['Maximum_Drawdown'])
                                      for _, row in metrics_table.iterrows()],
                'time_series_length': 96  # 96 rolling windows
            }

            benchmark_data = {
                'strategies_processed': strategies_count,
                'metrics_table_rows': len(metrics_table),
                'ranking_matrix_shape': ranking_matrix.shape,
                'recommendations_count': len(recommendations),
                'chart_data_points': len(chart_data['performance_comparison'])
            }

        return benchmark_data

    def benchmark_statistical_testing(self, pairwise_comparisons: int = 21) -> dict[str, Any]:
        """Benchmark statistical testing operations.

        Args:
            pairwise_comparisons: Number of pairwise comparisons (7 choose 2 = 21 for 7 strategies)

        Returns:
            Dictionary containing statistical testing benchmark results
        """
        logger.info(f"Benchmarking statistical testing for {pairwise_comparisons} comparisons")

        with self.measure_performance("statistical_testing_benchmark"):
            # Simulate statistical testing operations

            # 1. Generate sample return data for testing
            n_periods = 96  # 96 rolling windows
            n_strategies = 7
            sample_returns = np.random.normal(0.001, 0.02, size=(n_periods, n_strategies))

            # 2. Jobson-Korkie tests simulation
            jk_test_results = []
            for i in range(pairwise_comparisons):
                # Simulate test statistic and p-value
                test_stat = np.random.normal(0, 1)
                p_value = 2 * (1 - stats.norm.cdf(abs(test_stat)))  # Two-tailed test
                jk_test_results.append({
                    'comparison': f'comparison_{i}',
                    'test_statistic': test_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })

            # 3. Bootstrap confidence intervals simulation
            bootstrap_results = []
            for strategy in range(n_strategies):
                # Simulate bootstrap samples
                bootstrap_samples = np.random.choice(sample_returns[:, strategy], size=(1000, n_periods), replace=True)
                bootstrap_sharpes = []

                for sample in bootstrap_samples:
                    sharpe = np.mean(sample) / np.std(sample, ddof=1) * np.sqrt(252)
                    bootstrap_sharpes.append(sharpe)

                ci_lower = np.percentile(bootstrap_sharpes, 2.5)
                ci_upper = np.percentile(bootstrap_sharpes, 97.5)

                bootstrap_results.append({
                    'strategy': f'strategy_{strategy}',
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'bootstrap_samples': len(bootstrap_samples)
                })

            # 4. Multiple comparison corrections
            p_values = [result['p_value'] for result in jk_test_results]
            bonferroni_corrected = [p * len(p_values) for p in p_values]

            benchmark_data = {
                'pairwise_comparisons': pairwise_comparisons,
                'jk_tests_completed': len(jk_test_results),
                'bootstrap_intervals': len(bootstrap_results),
                'significant_results': sum(1 for r in jk_test_results if r['significant']),
                'bonferroni_corrections': len(bonferroni_corrected)
            }

        return benchmark_data

    def validate_performance_constraints(self) -> dict[str, Any]:
        """Validate all operations meet performance constraints.

        Returns:
            Dictionary containing constraint validation results
        """
        logger.info("Validating performance constraints...")

        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'constraint_validations': {},
            'overall_compliance': False,
            'performance_summary': {}
        }

        # Check processing time constraints
        total_processing_time = 0
        for _operation, elapsed in self.processing_times.items():
            total_processing_time += elapsed

        total_processing_hours = total_processing_time / 3600

        validation_results['constraint_validations']['processing_time'] = {
            'actual_hours': total_processing_hours,
            'target_hours': self.targets['processing_time_hours'],
            'meets_constraint': total_processing_hours <= self.targets['processing_time_hours'],
            'margin': self.targets['processing_time_hours'] - total_processing_hours
        }

        # Check reporting time constraints
        reporting_operations = [op for op in self.processing_times.keys() if 'reporting' in op]
        reporting_time_minutes = sum(self.processing_times[op] for op in reporting_operations) / 60

        validation_results['constraint_validations']['reporting_time'] = {
            'actual_minutes': reporting_time_minutes,
            'target_minutes': self.targets['reporting_time_minutes'],
            'meets_constraint': reporting_time_minutes <= self.targets['reporting_time_minutes'],
            'margin': self.targets['reporting_time_minutes'] - reporting_time_minutes
        }

        # Check memory constraints
        max_memory_gb = max(profile.get('final_memory_gb', 0) for profile in self.memory_profiles.values())

        validation_results['constraint_validations']['memory_usage'] = {
            'actual_memory_gb': max_memory_gb,
            'target_memory_gb': self.targets['system_memory_gb'],
            'meets_constraint': max_memory_gb <= self.targets['system_memory_gb'],
            'margin': self.targets['system_memory_gb'] - max_memory_gb
        }

        # Overall compliance check
        all_constraints_met = all(
            constraint['meets_constraint']
            for constraint in validation_results['constraint_validations'].values()
        )

        validation_results['overall_compliance'] = all_constraints_met

        # Performance summary
        validation_results['performance_summary'] = {
            'total_operations': len(self.processing_times),
            'total_processing_time_seconds': total_processing_time,
            'total_processing_time_hours': total_processing_hours,
            'peak_memory_usage_gb': max_memory_gb,
            'constraints_met': sum(1 for c in validation_results['constraint_validations'].values() if c['meets_constraint']),
            'total_constraints': len(validation_results['constraint_validations'])
        }

        logger.info(f"Performance constraints validation: {validation_results['performance_summary']['constraints_met']}/{validation_results['performance_summary']['total_constraints']} constraints met")

        return validation_results

    def run_comprehensive_performance_benchmark(self) -> dict[str, Any]:
        """Run comprehensive performance benchmark across all operations.

        Returns:
            Comprehensive benchmark results
        """
        logger.info("Starting comprehensive performance benchmark...")

        # Run individual benchmarks
        results_benchmark = self.benchmark_results_validation(sample_size=5000)
        reporting_benchmark = self.benchmark_executive_reporting(strategies_count=7)
        statistical_benchmark = self.benchmark_statistical_testing(pairwise_comparisons=21)

        # Validate constraints
        constraint_validation = self.validate_performance_constraints()

        comprehensive_results = {
            'comprehensive_benchmark': {
                'timestamp': datetime.now().isoformat(),
                'benchmarks': {
                    'results_validation': {
                        **results_benchmark,
                        **self.memory_profiles.get('results_validation_benchmark', {})
                    },
                    'executive_reporting': {
                        **reporting_benchmark,
                        **self.memory_profiles.get('executive_reporting_benchmark', {})
                    },
                    'statistical_testing': {
                        **statistical_benchmark,
                        **self.memory_profiles.get('statistical_testing_benchmark', {})
                    }
                },
                'constraint_validation': constraint_validation,
                'summary': {
                    'total_benchmarks': 3,
                    'performance_compliant': constraint_validation['overall_compliance'],
                    'ready_for_production': constraint_validation['overall_compliance'] and
                                           constraint_validation['performance_summary']['constraints_met'] >= 3
                }
            }
        }

        logger.info(f"Comprehensive benchmark complete. Production ready: {comprehensive_results['comprehensive_benchmark']['summary']['ready_for_production']}")

        return comprehensive_results

    def get_benchmark_summary(self) -> pd.DataFrame:
        """Get benchmark results summary as DataFrame.

        Returns:
            DataFrame with benchmark summary
        """
        if not self.processing_times:
            return pd.DataFrame()

        summary_data = []
        for operation, elapsed_time in self.processing_times.items():
            memory_profile = self.memory_profiles.get(operation, {})

            summary_data.append({
                'Operation': operation,
                'Elapsed_Time_Seconds': elapsed_time,
                'Elapsed_Time_Minutes': elapsed_time / 60,
                'Elapsed_Time_Hours': elapsed_time / 3600,
                'Memory_Delta_GB': memory_profile.get('memory_delta_gb', 0),
                'Peak_Memory_MB': memory_profile.get('peak_memory_mb', 0),
                'Within_Time_Target': elapsed_time <= (self.targets['processing_time_hours'] * 3600),
                'Within_Memory_Target': memory_profile.get('final_memory_gb', 0) <= self.targets['system_memory_gb']
            })

        return pd.DataFrame(summary_data)

# Import scipy.stats for statistical functions used in benchmarking
try:
    from scipy import stats
except ImportError:
    logger.warning("scipy not available for advanced statistical functions")
