"""
Integration tests for rolling backtest engine implementation.

Tests the complete integration of rolling window generation, model retraining,
backtest execution, memory management, and model pipeline integration.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest.execution_engine import BacktestExecutionConfig, BacktestExecutor
from src.evaluation.backtest.model_integration import IntegrationConfig, ModelTrainingIntegrator
from src.evaluation.backtest.model_retraining import ModelRetrainingEngine, RetrainingConfig
from src.evaluation.backtest.rolling_engine import RollingBacktestConfig, RollingBacktestEngine
from src.evaluation.validation.rolling_validation import BacktestConfig, RollingValidationEngine
from src.models.base import PortfolioModel
from src.utils.memory_manager import BatchProcessingConfig, MemoryManager


class MockPortfolioModel(PortfolioModel):
    """Mock portfolio model for comprehensive testing."""

    def __init__(self, name: str = "mock_model", model_type: str = "MOCK"):
        self.name = name
        self.model_type = model_type
        self.fitted = False
        self.fit_calls = []
        self.predict_calls = []

    def fit(self, returns: pd.DataFrame, universe: list[str], fit_period: tuple = None):
        """Mock fit method."""
        self.fitted = True
        self.fit_calls.append(
            {
                "returns_shape": returns.shape,
                "universe_size": len(universe),
                "fit_period": fit_period,
                "timestamp": pd.Timestamp.now(),
            }
        )

    def predict_weights(self, date: pd.Timestamp, universe: list[str]) -> pd.Series:
        """Mock weight prediction."""
        self.predict_calls.append(
            {
                "date": date,
                "universe_size": len(universe),
                "timestamp": pd.Timestamp.now(),
            }
        )

        # Return random weights that sum to 1
        np.random.seed(42)  # For reproducibility
        weights = np.random.dirichlet(np.ones(len(universe)))
        return pd.Series(weights, index=universe, name=f"weights_{date.strftime('%Y%m%d')}")


class TestRollingBacktestIntegration:
    """Test complete rolling backtest integration."""

    @pytest.fixture
    def sample_data(self) -> dict[str, pd.DataFrame]:
        """Generate comprehensive sample data for testing."""
        dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
        assets = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "META", "AMZN", "NFLX"]

        # Generate synthetic returns with realistic characteristics
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.multivariate_normal(
                mean=np.array([0.0008] * len(assets)),  # Daily returns ~20% annualized
                cov=np.eye(len(assets)) * 0.0004 + 0.0001,  # Correlated returns
                size=len(dates),
            ),
            index=dates,
            columns=assets,
        )

        # Generate universe data (dynamic membership)
        universe_dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="MS")
        universe_data = pd.DataFrame(True, index=universe_dates, columns=assets)

        # Simulate some assets leaving/joining the universe
        universe_data.loc["2021-01-01":, "TSLA"] = False  # TSLA leaves in 2021
        universe_data.loc["2022-01-01":, "TSLA"] = True  # TSLA returns in 2022

        return {
            "returns": returns,
            "universe": universe_data,
        }

    @pytest.fixture
    def mock_models(self) -> dict[str, MockPortfolioModel]:
        """Create mock models representing different types."""
        return {
            "equal_weight": MockPortfolioModel("equal_weight", "HRP"),
            "momentum": MockPortfolioModel("momentum", "LSTM"),
            "gat_model": MockPortfolioModel("gat_model", "GAT"),
        }

    def test_end_to_end_backtest_integration(
        self, sample_data: dict[str, pd.DataFrame], mock_models: dict[str, MockPortfolioModel]
    ):
        """Test complete end-to-end backtest integration."""

        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            # Configure rolling backtest
            config = RollingBacktestConfig(
                start_date=pd.Timestamp("2021-01-01"),
                end_date=pd.Timestamp("2023-12-31"),
                training_months=12,  # Shorter for testing
                validation_months=6,
                test_months=6,
                step_months=6,
                output_dir=output_dir,
                save_intermediate_results=True,
            )

            # Initialize rolling backtest engine
            engine = RollingBacktestEngine(config)

            # Execute complete backtest
            results = engine.run_rolling_backtest(
                models=mock_models,
                data=sample_data,
                universe_data=sample_data["universe"],
            )

            # Verify results structure
            assert isinstance(results, type(engine).__annotations__.get("return", object))
            assert len(results.splits) > 0
            assert len(results.portfolio_returns) == len(mock_models)
            assert len(results.portfolio_weights) == len(mock_models)
            assert len(results.performance_metrics) == len(mock_models)

            # Verify models were trained
            for model_name, model in mock_models.items():
                assert len(model.fit_calls) > 0, f"Model {model_name} was not trained"
                assert len(model.predict_calls) > 0, f"Model {model_name} made no predictions"

            # Verify temporal integrity
            assert "temporal_integrity_report" in results.__dict__

            # Verify execution summary
            assert "execution_summary" in results.__dict__
            summary = results.execution_summary
            assert "total_splits" in summary
            assert "models_tested" in summary

    def test_model_retraining_integration(
        self, sample_data: dict[str, pd.DataFrame], mock_models: dict[str, MockPortfolioModel]
    ):
        """Test model retraining integration."""

        with TemporaryDirectory() as temp_dir:
            # Configure retraining engine
            config = RetrainingConfig(
                enable_checkpointing=True,
                checkpoint_dir=Path(temp_dir) / "checkpoints",
                min_training_samples=100,  # Lower for testing
                max_missing_ratio=0.2,
            )

            engine = ModelRetrainingEngine(config)

            # Create a simple rolling split for testing
            from src.evaluation.validation.rolling_validation import RollSplit, ValidationPeriod

            split = RollSplit(
                ValidationPeriod(pd.Timestamp("2021-01-01"), pd.Timestamp("2022-01-01")),
                ValidationPeriod(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-07-01")),
                ValidationPeriod(pd.Timestamp("2022-07-01"), pd.Timestamp("2023-01-01")),
            )

            train_data = sample_data["returns"].loc["2021-01-01":"2021-12-31"]
            val_data = sample_data["returns"].loc["2022-01-01":"2022-06-30"]

            # Test retraining for each model
            for model_name, model in mock_models.items():
                result = engine.retrain_model(
                    model=model,
                    model_name=model_name,
                    split=split,
                    train_data=train_data,
                    val_data=val_data,
                    universe_data=sample_data["universe"],
                )

                assert result.success, f"Retraining failed for {model_name}: {result.error_message}"
                assert result.training_time_seconds > 0
                assert result.training_samples > 0
                assert result.training_assets > 0

                if config.enable_checkpointing:
                    assert result.model_checkpoint_path is not None
                    assert result.model_checkpoint_path.exists()

            # Verify retraining statistics
            stats = engine.get_retraining_stats()
            assert stats["total_models"] == len(mock_models)
            assert stats["total_retrains"] == len(mock_models)

    def test_backtest_execution_integration(
        self, sample_data: dict[str, pd.DataFrame], mock_models: dict[str, MockPortfolioModel]
    ):
        """Test backtest execution engine integration."""

        with TemporaryDirectory() as temp_dir:
            # Configure execution engine
            config = BacktestExecutionConfig(
                initial_capital=1000000.0,
                rebalance_frequency="M",
                enable_realistic_costs=True,
                track_positions=True,
                track_trades=True,
                output_dir=Path(temp_dir),
                save_detailed_logs=True,
            )

            executor = BacktestExecutor(config)

            # Generate rebalancing dates
            rebalance_dates = pd.date_range(
                start="2021-01-01", end="2023-12-31", freq="MS"
            ).tolist()

            # Test execution with first model
            model = list(mock_models.values())[0]
            results = executor.execute_backtest(
                model=model,
                returns_data=sample_data["returns"],
                rebalance_dates=rebalance_dates,
                universe_data=sample_data["universe"],
            )

            # Verify execution results
            assert "portfolio_returns" in results
            assert "performance_metrics" in results
            assert "trade_analysis" in results
            assert "position_analysis" in results
            assert "execution_summary" in results

            # Verify detailed logs were saved
            assert (Path(temp_dir) / "trade_log.csv").exists()
            assert (Path(temp_dir) / "position_history.csv").exists()
            assert (Path(temp_dir) / "performance_history.csv").exists()

            # Verify execution statistics
            summary = results["execution_summary"]
            assert summary["total_rebalances"] > 0
            assert summary["final_portfolio_value"] > 0

    def test_memory_management_integration(self, sample_data: dict[str, pd.DataFrame]):
        """Test memory management integration."""

        # Configure memory manager
        config = BatchProcessingConfig(
            batch_size=16,
            max_memory_gb=8.0,
            memory_threshold=0.7,
            enable_caching=True,
            cache_size_gb=1.0,
        )

        memory_manager = MemoryManager(config)

        # Test memory monitoring
        memory_stats = memory_manager.get_current_memory_stats()
        assert memory_stats.total_memory_gb > 0
        assert 0 <= memory_stats.memory_percent <= 100

        # Test performance monitoring
        perf_stats = memory_manager.get_current_performance_stats()
        assert perf_stats.cpu_count > 0

        # Test memory pressure checking
        pressure_check = memory_manager.check_memory_pressure()
        assert "memory_pressure" in pressure_check
        assert "recommendations" in pressure_check

        # Test data caching
        test_data = sample_data["returns"].head(100)
        cache_success = memory_manager.cache_data("test_data", test_data)
        assert cache_success

        retrieved_data = memory_manager.get_cached_data("test_data")
        assert retrieved_data is not None
        pd.testing.assert_frame_equal(retrieved_data, test_data)

        # Test batch processing
        def simple_processing_func(batch):
            return batch.sum().sum()  # Simple aggregation

        batch_results = memory_manager.process_in_batches(
            data=sample_data["returns"],
            processing_func=simple_processing_func,
        )

        assert len(batch_results) > 0
        assert all(isinstance(result, (int, float)) for result in batch_results)

    def test_model_training_pipeline_integration(
        self, sample_data: dict[str, pd.DataFrame], mock_models: dict[str, MockPortfolioModel]
    ):
        """Test integration with model training pipeline."""

        with TemporaryDirectory() as temp_dir:
            # Configure integration
            config = IntegrationConfig(
                enable_checkpointing=True,
                checkpoint_dir=Path(temp_dir) / "integration_checkpoints",
                enable_validation_metrics=True,
                enable_performance_attribution=True,
                enable_model_ensemble=True,
            )

            integrator = ModelTrainingIntegrator(config)

            # Configure rolling validation engine
            validation_config = BacktestConfig(
                start_date=pd.Timestamp("2021-01-01"),
                end_date=pd.Timestamp("2023-12-31"),
                training_months=12,
                validation_months=6,
                test_months=6,
                step_months=6,
            )

            rolling_engine = RollingValidationEngine(validation_config)

            # Test integration
            integration_results = integrator.integrate_with_rolling_validation(
                rolling_engine=rolling_engine,
                models=mock_models,
                data=sample_data,
            )

            # Verify integration results
            assert integration_results["models_integrated"] == len(mock_models)
            assert "validation_results" in integration_results
            assert "performance_metrics" in integration_results

            # Verify model-specific results
            for model_name in mock_models.keys():
                assert model_name in integration_results["validation_results"]
                model_results = integration_results["validation_results"][model_name]
                assert "splits_processed" in model_results
                assert model_results["splits_processed"] > 0

            # Verify ensemble creation
            if config.enable_model_ensemble:
                assert "ensemble_results" in integration_results
                ensemble_results = integration_results["ensemble_results"]
                assert "ensemble_weights" in ensemble_results
                assert len(ensemble_results["ensemble_weights"]) == len(mock_models)

            # Verify performance metrics
            perf_metrics = integration_results["performance_metrics"]
            assert "best_model" in perf_metrics
            assert "performance_comparison" in perf_metrics

            # Test checkpoint functionality
            for model_name in mock_models.keys():
                checkpoints = integrator.get_checkpoint_info(model_name)
                if checkpoints:  # May not have checkpoints if splits were insufficient
                    assert len(checkpoints) > 0
                    assert all(cp.checkpoint_path.exists() for cp in checkpoints)

    def test_temporal_integrity_across_integration(
        self, sample_data: dict[str, pd.DataFrame], mock_models: dict[str, MockPortfolioModel]
    ):
        """Test temporal integrity is maintained across all integration components."""

        # This test ensures that temporal integrity is preserved throughout
        # the entire integration pipeline

        config = RollingBacktestConfig(
            start_date=pd.Timestamp("2021-01-01"),
            end_date=pd.Timestamp("2023-12-31"),
            training_months=12,
            validation_months=6,
            test_months=6,
        )

        engine = RollingBacktestEngine(config)

        # Execute backtest and verify temporal integrity is maintained
        results = engine.run_rolling_backtest(
            models={"test_model": list(mock_models.values())[0]},
            data=sample_data,
        )

        # Check temporal integrity report
        integrity_report = results.temporal_integrity_report
        assert isinstance(integrity_report, dict)

        # If splits_monitored > 0, verify no critical violations
        if integrity_report.get("total_splits_monitored", 0) > 0:
            assert (
                integrity_report.get("splits_failed", 0) == 0
            ), "Temporal integrity violations detected"
            assert (
                integrity_report.get("success_rate", 0) == 100
            ), "Not all splits passed integrity checks"

    def test_performance_under_memory_constraints(
        self, sample_data: dict[str, pd.DataFrame], mock_models: dict[str, MockPortfolioModel]
    ):
        """Test system performance under memory constraints."""

        # Create larger dataset to stress test memory management
        large_dates = pd.date_range(start="2015-01-01", end="2023-12-31", freq="D")
        large_assets = [f"ASSET_{i:03d}" for i in range(50)]  # 50 assets

        # Generate larger synthetic dataset
        np.random.seed(42)
        large_returns = pd.DataFrame(
            np.random.normal(0.0005, 0.015, (len(large_dates), len(large_assets))),
            index=large_dates,
            columns=large_assets,
        )

        large_data = {"returns": large_returns}

        # Configure with memory constraints
        config = RollingBacktestConfig(
            start_date=pd.Timestamp("2021-01-01"),
            end_date=pd.Timestamp("2023-12-31"),
            training_months=6,  # Smaller windows for memory efficiency
            validation_months=3,
            test_months=3,
            step_months=3,
            enable_memory_monitoring=True,
        )

        engine = RollingBacktestEngine(config)

        # Use single model to reduce memory footprint
        single_model = {"test_model": list(mock_models.values())[0]}

        try:
            results = engine.run_rolling_backtest(
                models=single_model,
                data=large_data,
            )

            # Verify execution completed successfully
            assert len(results.splits) > 0
            assert "test_model" in results.portfolio_returns
            assert len(results.portfolio_returns["test_model"]) > 0

            # Check memory usage statistics
            if "memory_usage_stats" in results.__dict__:
                memory_stats = results.memory_usage_stats
                assert isinstance(memory_stats, dict)

        except MemoryError:
            pytest.skip("Insufficient memory for large dataset test")

    def test_error_handling_and_recovery(self, sample_data: dict[str, pd.DataFrame]):
        """Test error handling and recovery across integration components."""

        # Create a failing model to test error handling
        class FailingModel(MockPortfolioModel):
            def __init__(self, fail_on_fit=False, fail_on_predict=False):
                super().__init__("failing_model")
                self.fail_on_fit = fail_on_fit
                self.fail_on_predict = fail_on_predict

            def fit(self, returns, universe, fit_period=None):
                if self.fail_on_fit:
                    raise ValueError("Simulated training failure")
                super().fit(returns, universe, fit_period)

            def predict_weights(self, date, universe):
                if self.fail_on_predict:
                    raise ValueError("Simulated prediction failure")
                return super().predict_weights(date, universe)

        # Test with failing models
        failing_models = {
            "fail_fit": FailingModel(fail_on_fit=True),
            "fail_predict": FailingModel(fail_on_predict=True),
            "working_model": MockPortfolioModel("working"),
        }

        config = RollingBacktestConfig(
            start_date=pd.Timestamp("2021-01-01"),
            end_date=pd.Timestamp("2022-12-31"),
            training_months=6,
            validation_months=3,
            test_months=3,
        )

        engine = RollingBacktestEngine(config)

        # Execute backtest - should handle failures gracefully
        results = engine.run_rolling_backtest(
            models=failing_models,
            data=sample_data,
        )

        # Verify that working model succeeded and failing models were handled
        assert "working_model" in results.portfolio_returns
        assert len(results.portfolio_returns["working_model"]) > 0

        # Failing models should have empty or error results
        if "fail_fit" in results.portfolio_returns:
            assert (
                len(results.portfolio_returns["fail_fit"]) == 0
                or pd.isna(results.portfolio_returns["fail_fit"]).all()
            )

        if "fail_predict" in results.portfolio_returns:
            assert (
                len(results.portfolio_returns["fail_predict"]) == 0
                or pd.isna(results.portfolio_returns["fail_predict"]).all()
            )
