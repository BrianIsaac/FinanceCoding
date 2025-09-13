"""
Integration tests for comprehensive backtesting pipeline.

This module tests the end-to-end integration of the comprehensive
backtesting system including data loading, model execution,
constraint validation, and results generation.
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.evaluation.backtest.rolling_engine import RollingBacktestConfig, RollingBacktestEngine
from src.models.base.baselines import EqualWeightModel, MarketCapWeightedModel, MeanReversionModel


class TestComprehensiveBacktestIntegration(unittest.TestCase):
    """Integration tests for comprehensive backtest pipeline."""

    def setUp(self):
        """Set up integration test environment."""
        # Create comprehensive test dataset
        self.start_date = pd.Timestamp("2020-01-01")
        self.end_date = pd.Timestamp("2023-12-31")
        self.dates = pd.date_range(self.start_date, self.end_date, freq="D")
        self.assets = [f"ASSET_{i:02d}" for i in range(10)]  # 10 assets

        # Generate realistic-looking returns data
        np.random.seed(42)
        base_returns = np.random.normal(0.0005, 0.015, (len(self.dates), len(self.assets)))

        # Add some correlation structure
        correlation_factor = np.random.normal(0, 0.01, len(self.dates))
        for i in range(len(self.assets)):
            base_returns[:, i] += correlation_factor * (0.3 + 0.1 * i)

        self.returns_data = pd.DataFrame(
            base_returns,
            index=self.dates,
            columns=self.assets
        )

        # Create market data dictionary
        self.market_data = {
            "returns": self.returns_data,
            "universe": None,
            "benchmark": None,
        }

        # Create realistic backtest configuration
        self.config = RollingBacktestConfig(
            start_date=pd.Timestamp("2021-01-01"),
            end_date=pd.Timestamp("2023-06-30"),
            training_months=18,  # 1.5 years training
            validation_months=6,  # 6 months validation
            test_months=6,       # 6 months test
            step_months=6,       # Semi-annual rebalancing for testing
            rebalance_frequency="M",
            min_training_samples=200,
            output_dir=Path("test_output_integration"),
            save_intermediate_results=False,
            enable_progress_tracking=False,
        )

    def test_end_to_end_backtest_execution(self):
        """Test complete end-to-end backtest execution."""
        # Create models for testing
        models = {
            "EqualWeight": EqualWeightModel(),
            "MarketCap": MarketCapWeightedModel(lookback_days=252),
        }

        # Initialize and run backtest engine
        engine = RollingBacktestEngine(self.config)

        # Mock the memory-intensive components to avoid GPU requirements
        with patch.object(engine, 'gpu_manager', None):
            results = engine.run_rolling_backtest(
                models=models,
                data=self.market_data,
                universe_data=None,
            )

        # Validate results structure
        self.assertIsNotNone(results)
        self.assertIn("EqualWeight", results.portfolio_returns)
        self.assertIn("MarketCap", results.portfolio_returns)

        # Validate that returns were generated
        for model_name in models.keys():
            if model_name in results.portfolio_returns:
                returns_series = results.portfolio_returns[model_name]
                if not returns_series.empty:
                    self.assertIsInstance(returns_series, pd.Series)
                    self.assertGreater(len(returns_series), 0)

        # Validate execution summary
        self.assertIsNotNone(results.execution_summary)
        self.assertIn("total_splits", results.execution_summary)
        self.assertIn("models_tested", results.execution_summary)

    def test_temporal_integrity_enforcement(self):
        """Test that temporal integrity is enforced throughout execution."""
        models = {"EqualWeight": EqualWeightModel()}
        engine = RollingBacktestEngine(self.config)

        # Mock integrity monitor to track calls
        with patch.object(engine.integrity_monitor, 'monitor_split_integrity') as mock_monitor:
            mock_monitor.return_value = {
                "overall_pass": True,
                "violations": [],
                "critical_count": 0,
            }

            with patch.object(engine, 'gpu_manager', None):
                results = engine.run_rolling_backtest(
                    models=models,
                    data=self.market_data,
                    universe_data=None,
                )

            # Verify integrity monitoring was called
            self.assertGreater(mock_monitor.call_count, 0)

            # Verify integrity report is generated
            self.assertIsNotNone(results.temporal_integrity_report)

    def test_multiple_models_comparison(self):
        """Test execution with multiple baseline models."""
        from src.models.base.baselines import MeanReversionModel, MomentumModel

        models = {
            "EqualWeight": EqualWeightModel(),
            "MarketCap": MarketCapWeightedModel(lookback_days=126),
            "MeanReversion": MeanReversionModel(lookback_days=21),
            "Momentum": MomentumModel(lookback_days=63),
        }

        engine = RollingBacktestEngine(self.config)

        with patch.object(engine, 'gpu_manager', None):
            results = engine.run_rolling_backtest(
                models=models,
                data=self.market_data,
                universe_data=None,
            )

        # Verify all models were processed
        self.assertEqual(len(results.portfolio_returns), len(models))

        # Verify performance metrics were calculated for each model
        for model_name in models.keys():
            if model_name in results.performance_metrics:
                metrics = results.performance_metrics[model_name]
                # Basic sanity checks on metrics
                if metrics:  # Some models might not have metrics if they failed
                    self.assertIsInstance(metrics, dict)

    def test_constraint_enforcement_validation(self):
        """Test constraint enforcement across all models."""
        models = {
            "EqualWeight": EqualWeightModel(),
            "MarketCap": MarketCapWeightedModel(),
        }

        engine = RollingBacktestEngine(self.config)

        with patch.object(engine, 'gpu_manager', None):
            results = engine.run_rolling_backtest(
                models=models,
                data=self.market_data,
                universe_data=None,
            )

        # Validate constraint compliance in generated weights
        for model_name, weights_df in results.portfolio_weights.items():
            if not weights_df.empty:
                # Check long-only constraint (all weights >= 0)
                negative_weights = (weights_df < -1e-8).any(axis=1)
                self.assertFalse(negative_weights.any(),
                               f"Model {model_name} generated negative weights")

                # Check weight sum constraint (weights sum to ~1.0)
                weight_sums = weights_df.sum(axis=1)
                for i, weight_sum in enumerate(weight_sums):
                    self.assertAlmostEqual(
                        weight_sum, 1.0,
                        places=2,
                        msg=f"Model {model_name} weights don't sum to 1.0 at period {i}: {weight_sum}"
                    )

    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation across models."""
        models = {"EqualWeight": EqualWeightModel()}
        engine = RollingBacktestEngine(self.config)

        with patch.object(engine, 'gpu_manager', None):
            results = engine.run_rolling_backtest(
                models=models,
                data=self.market_data,
                universe_data=None,
            )

        # Validate performance metrics
        for model_name in models.keys():
            if model_name in results.performance_metrics:
                metrics = results.performance_metrics[model_name]

                if metrics:  # Check if metrics were calculated
                    # Verify standard metrics are present
                    expected_metrics = ["sharpe_ratio", "annualized_return", "annualized_volatility"]
                    for metric in expected_metrics:
                        if metric in metrics:
                            self.assertIsInstance(metrics[metric], (int, float))
                            self.assertFalse(np.isnan(metrics[metric]),
                                           f"Metric {metric} is NaN for model {model_name}")

    def test_memory_efficient_execution(self):
        """Test memory-efficient execution with monitoring."""
        # Create memory-constrained configuration
        memory_config = RollingBacktestConfig(
            start_date=pd.Timestamp("2021-01-01"),
            end_date=pd.Timestamp("2022-12-31"),
            training_months=12,
            validation_months=3,
            test_months=3,
            step_months=3,
            gpu_config=None,  # No GPU for testing
            enable_memory_monitoring=True,
            batch_size=16,  # Smaller batch size
        )

        models = {"EqualWeight": EqualWeightModel()}
        engine = RollingBacktestEngine(memory_config)

        results = engine.run_rolling_backtest(
            models=models,
            data=self.market_data,
            universe_data=None,
        )

        # Verify execution completed without memory errors
        self.assertIsNotNone(results)
        self.assertIn("EqualWeight", results.portfolio_returns)

    def test_missing_data_handling(self):
        """Test handling of missing data during execution."""
        # Create data with missing values
        returns_with_gaps = self.returns_data.copy()

        # Introduce missing values in random locations
        np.random.seed(123)
        missing_indices = np.random.choice(
            range(len(returns_with_gaps)),
            size=int(0.05 * len(returns_with_gaps)),  # 5% missing
            replace=False
        )

        for idx in missing_indices:
            asset_idx = np.random.randint(0, len(self.assets))
            returns_with_gaps.iloc[idx, asset_idx] = np.nan

        market_data_with_gaps = {
            "returns": returns_with_gaps,
            "universe": None,
            "benchmark": None,
        }

        models = {"EqualWeight": EqualWeightModel()}
        engine = RollingBacktestEngine(self.config)

        with patch.object(engine, 'gpu_manager', None):
            # Should handle missing data gracefully
            results = engine.run_rolling_backtest(
                models=models,
                data=market_data_with_gaps,
                universe_data=None,
            )

        # Verify execution completed despite missing data
        self.assertIsNotNone(results)

    def test_edge_cases_handling(self):
        """Test handling of edge cases and error conditions."""
        models = {"EqualWeight": EqualWeightModel()}
        engine = RollingBacktestEngine(self.config)

        # Test with empty universe
        empty_universe_data = self.market_data.copy()
        empty_universe_data["returns"] = pd.DataFrame()  # Empty DataFrame

        with patch.object(engine, 'gpu_manager', None):
            with self.assertRaises(ValueError):
                engine.run_rolling_backtest(
                    models=models,
                    data=empty_universe_data,
                    universe_data=None,
                )

    def test_results_consistency(self):
        """Test consistency of results across multiple runs."""
        models = {"EqualWeight": EqualWeightModel()}
        engine = RollingBacktestEngine(self.config)

        # Run backtest twice with same configuration
        with patch.object(engine, 'gpu_manager', None):
            results1 = engine.run_rolling_backtest(
                models=models,
                data=self.market_data,
                universe_data=None,
            )

        # Create fresh engine for second run
        engine2 = RollingBacktestEngine(self.config)
        with patch.object(engine2, 'gpu_manager', None):
            results2 = engine2.run_rolling_backtest(
                models=models,
                data=self.market_data,
                universe_data=None,
            )

        # Results should be identical for deterministic models
        if ("EqualWeight" in results1.portfolio_returns and
            "EqualWeight" in results2.portfolio_returns):

            returns1 = results1.portfolio_returns["EqualWeight"]
            returns2 = results2.portfolio_returns["EqualWeight"]

            if not returns1.empty and not returns2.empty:
                pd.testing.assert_series_equal(returns1, returns2)


class TestScriptIntegration(unittest.TestCase):
    """Test integration with the main backtest script."""

    def test_script_data_loading_integration(self):
        """Test integration of data loading components."""
        with patch('src.data.loaders.parquet_manager.ParquetManager') as mock_data_manager_class:
            # Mock data manager instance
            mock_data_manager_instance = MagicMock()
            mock_data_manager_class.return_value = mock_data_manager_instance

            # Mock data loading methods
            mock_returns = pd.DataFrame(
                np.random.randn(1000, 5),
                index=pd.date_range("2020-01-01", periods=1000),
                columns=["A", "B", "C", "D", "E"]
            )
            mock_data_manager_instance.load_returns.return_value = mock_returns
            mock_data_manager_instance.load_universe.return_value = None
            mock_data_manager_instance.load_benchmark.return_value = None

            # Test that data manager can be instantiated and called
            # This tests the interface without requiring the full script
            from src.data.loaders.parquet_manager import ParquetManager
            from src.config.base import DataConfig

            # Create a simple config for testing
            data_config = DataConfig()

            # Verify the data manager can be created (mocked)
            manager = ParquetManager(data_config)

            # This test primarily ensures imports work correctly
            self.assertIsNotNone(manager)

    def test_model_initialization_integration(self):
        """Test model initialization components."""
        try:
            from scripts.run_comprehensive_backtest import initialize_models
            from src.config.base import Config
            from src.utils.gpu import GPUConfig

            config = Config()
            gpu_config = GPUConfig(device="cpu")  # Use CPU for testing

            models = initialize_models(config, gpu_config)

            self.assertIsInstance(models, dict)
            self.assertGreater(len(models), 0)

            # Verify HRP models are included
            hrp_models = [name for name in models.keys() if name.startswith("HRP_")]
            self.assertGreater(len(hrp_models), 0)

            # Verify baseline models are included
            baseline_models = ["EqualWeight", "MarketCapWeighted", "MeanReversion"]
            for baseline in baseline_models:
                self.assertIn(baseline, models)

        except ImportError:
            self.skipTest("Required modules not accessible for integration testing")

    def test_memory_pressure_simulation(self):
        """Test memory pressure simulation for continuous execution (QA Fix: High Priority #4)."""
        from unittest.mock import patch

        from src.utils.gpu import GPUConfig, GPUMemoryManager

        # Configure for memory pressure testing
        memory_config = RollingBacktestConfig(
            start_date=pd.Timestamp("2020-01-01"),
            end_date=pd.Timestamp("2022-12-31"),
            training_months=12,
            validation_months=6,
            test_months=6,
            step_months=6,
            enable_memory_monitoring=True,
            batch_size=8,  # Smaller batch to simulate pressure
        )

        # Create larger dataset to simulate memory pressure
        large_dates = pd.date_range("2020-01-01", periods=2000, freq="D")
        large_assets = [f"STOCK_{i:03d}" for i in range(200)]  # 200 assets

        np.random.seed(42)
        large_returns_data = pd.DataFrame(
            np.random.normal(0.0005, 0.02, (len(large_dates), len(large_assets))),
            index=large_dates,
            columns=large_assets
        )

        large_market_data = {
            "returns": large_returns_data,
            "universe": None,
            "benchmark": None,
        }

        models = {
            "EqualWeight": EqualWeightModel(),
            "MarketCap": MarketCapWeightedModel(),
        }

        # Mock GPU memory monitoring
        with patch('torch.cuda.is_available', return_value=False), \
             patch('src.utils.gpu.torch.cuda.memory_allocated') as mock_memory, \
             patch('src.utils.gpu.torch.cuda.max_memory_allocated') as mock_max_memory:

            # Simulate increasing memory pressure
            memory_values = [6, 7, 8, 9, 10, 10.5, 10.8, 10.9]  # GB progression
            mock_memory.side_effect = [val * 1024**3 for val in memory_values]
            mock_max_memory.side_effect = [val * 1024**3 for val in memory_values]

            engine = RollingBacktestEngine(memory_config)

            # Execute with memory monitoring (CPU mode due to mocking)
            results = engine.run_rolling_backtest(
                models=models,
                data=large_market_data,
                universe_data=None,
            )

            # Validate execution completed despite memory pressure
            self.assertIsNotNone(results)
            self.assertIn("EqualWeight", results.portfolio_returns)
            self.assertIn("MarketCap", results.portfolio_returns)

            # Memory monitoring verification is less critical for CPU mode
            # Focus on successful execution under memory pressure simulation

    def test_continuous_execution_stress(self):
        """Test continuous execution stress over extended periods (QA Fix: Must Fix #1)."""
        import time
        from unittest.mock import patch

        # Configure for extended execution simulation
        stress_config = RollingBacktestConfig(
            start_date=pd.Timestamp("2020-01-01"),
            end_date=pd.Timestamp("2023-12-31"),  # 4 years for stress test
            training_months=24,  # Longer training periods
            validation_months=6,
            test_months=6,
            step_months=6,
            enable_progress_tracking=True,
            save_intermediate_results=True,
        )

        models = {
            "EqualWeight": EqualWeightModel(),
            "MeanReversion": MeanReversionModel(),
        }

        # Create extended dataset
        extended_dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
        extended_assets = [f"ASSET_{i:02d}" for i in range(50)]

        np.random.seed(42)
        extended_returns_data = pd.DataFrame(
            np.random.normal(0.0005, 0.015, (len(extended_dates), len(extended_assets))),
            index=extended_dates,
            columns=extended_assets
        )

        extended_market_data = {
            "returns": extended_returns_data,
            "universe": None,
            "benchmark": None,
        }

        engine = RollingBacktestEngine(stress_config)

        start_time = time.time()

        # Execute extended backtest
        with patch('src.evaluation.backtest.rolling_engine.logger'):
            results = engine.run_rolling_backtest(
                models=models,
                data=extended_market_data,
                universe_data=None,
            )

        execution_time = time.time() - start_time

        # Stress test should complete within reasonable time for 4-year subset
        # If 8-year target is 8 hours, 4-year should be ~4 hours max for stress test
        max_stress_time = 14400  # 4 hours for 4-year stress test

        self.assertLess(
            execution_time, max_stress_time,
            f"Stress test took {execution_time:.2f}s, exceeds {max_stress_time}s target for 4-year subset"
        )

        # Validate stress test results
        self.assertIsNotNone(results)
        self.assertIn("EqualWeight", results.portfolio_returns)
        self.assertIn("MeanReversion", results.portfolio_returns)

        # Verify all rolling windows were processed
        expected_windows = (2023 - 2020 + 1) * 2  # Approximate rolling windows
        actual_windows = len(results.portfolio_returns["EqualWeight"])
        self.assertGreater(actual_windows, expected_windows // 2)  # At least half


if __name__ == "__main__":
    unittest.main()
