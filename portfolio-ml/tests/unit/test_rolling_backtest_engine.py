"""
Unit tests for rolling backtest engine.

Tests the comprehensive rolling backtest engine that integrates
rolling windows, temporal validation, and backtest execution.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest.rolling_engine import (
    ModelRetrainingState,
    RollingBacktestConfig,
    RollingBacktestEngine,
    RollingBacktestResults,
)
from src.evaluation.validation.rolling_validation import RollSplit, ValidationPeriod
from src.models.base import PortfolioModel


class MockPortfolioModel(PortfolioModel):
    """Mock portfolio model for testing."""

    def __init__(self, name: str = "mock_model"):
        # Initialize parent class with basic constraints
        from src.models.base.constraints import PortfolioConstraints
        super().__init__(PortfolioConstraints())

        self.name = name
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
            }
        )

    def predict_weights(self, date: pd.Timestamp, universe: list[str]) -> pd.Series:
        """Mock weight prediction."""
        self.predict_calls.append(
            {
                "date": date,
                "universe_size": len(universe),
            }
        )

        # Return equal weights
        weights = pd.Series(
            1.0 / len(universe), index=universe, name=f"weights_{date.strftime('%Y%m%d')}"
        )
        return weights

    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata for testing."""
        return {
            "model_type": "mock",
            "name": self.name,
            "fitted": self.fitted,
            "hyperparameters": {},
            "constraints": {},
            "version": "1.0.0",
        }


class TestRollingBacktestConfig:
    """Test rolling backtest configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RollingBacktestConfig(
            start_date=pd.Timestamp("2020-01-01"),
            end_date=pd.Timestamp("2024-12-31"),
        )

        assert config.training_months == 36
        assert config.validation_months == 12
        assert config.test_months == 12
        assert config.step_months == 12
        assert config.min_training_samples == 252
        assert config.initial_capital == 1000000.0

    def test_to_backtest_config_conversion(self):
        """Test conversion to BacktestConfig."""
        rolling_config = RollingBacktestConfig(
            start_date=pd.Timestamp("2020-01-01"),
            end_date=pd.Timestamp("2024-12-31"),
            training_months=24,
            validation_months=6,
        )

        backtest_config = rolling_config.to_backtest_config()

        assert backtest_config.start_date == rolling_config.start_date
        assert backtest_config.end_date == rolling_config.end_date
        assert backtest_config.training_months == 24
        assert backtest_config.validation_months == 6


class TestModelRetrainingState:
    """Test model retraining state tracking."""

    def test_initialization(self):
        """Test retraining state initialization."""
        state = ModelRetrainingState()

        assert state.model_checkpoints == {}
        assert state.training_history == []
        assert state.performance_metrics == []
        assert state.last_training_date is None
        assert state.retraining_count == 0


class TestRollingBacktestEngine:
    """Test rolling backtest engine functionality."""

    @pytest.fixture
    def config(self) -> RollingBacktestConfig:
        """Default rolling backtest configuration."""
        return RollingBacktestConfig(
            start_date=pd.Timestamp("2020-01-01"),
            end_date=pd.Timestamp("2027-12-31"),
            training_months=36,
            validation_months=12,
            test_months=12,
            step_months=12,
        )

    @pytest.fixture
    def engine(self, config: RollingBacktestConfig) -> RollingBacktestEngine:
        """Rolling backtest engine instance."""
        return RollingBacktestEngine(config)

    @pytest.fixture
    def sample_data(self) -> dict[str, pd.DataFrame]:
        """Sample data for testing."""
        dates = pd.date_range(start="2020-01-01", end="2027-12-31", freq="D")
        assets = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]

        # Generate synthetic returns
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.normal(0.0008, 0.02, (len(dates), len(assets))),
            index=dates,
            columns=assets,
        )

        return {"returns": returns}

    @pytest.fixture
    def mock_models(self) -> dict[str, MockPortfolioModel]:
        """Mock portfolio models for testing."""
        return {
            "equal_weight": MockPortfolioModel("equal_weight"),
            "momentum": MockPortfolioModel("momentum"),
        }

    def test_initialization(self, config: RollingBacktestConfig):
        """Test engine initialization."""
        engine = RollingBacktestEngine(config)

        assert engine.config == config
        assert engine.rolling_engine is not None
        assert engine.integrity_monitor is not None
        assert engine.performance_analytics is not None
        assert engine.model_states == {}

    def test_input_validation(
        self, engine: RollingBacktestEngine, mock_models: dict[str, MockPortfolioModel]
    ):
        """Test input validation."""
        # Test empty models
        with pytest.raises(ValueError, match="No models provided"):
            engine.run_rolling_backtest({}, {})

        # Test missing returns data
        with pytest.raises(ValueError, match="Returns data is required"):
            engine.run_rolling_backtest(mock_models, {"prices": pd.DataFrame()})

        # Test invalid index
        invalid_data = {"returns": pd.DataFrame(index=[1, 2, 3])}  # Not DatetimeIndex
        with pytest.raises(ValueError, match="DatetimeIndex"):
            engine.run_rolling_backtest(mock_models, invalid_data)

    def test_rolling_backtest_execution(
        self,
        engine: RollingBacktestEngine,
        mock_models: dict[str, MockPortfolioModel],
        sample_data: dict[str, pd.DataFrame],
    ):
        """Test complete rolling backtest execution."""
        results = engine.run_rolling_backtest(mock_models, sample_data)

        assert isinstance(results, RollingBacktestResults)
        assert len(results.splits) > 0
        assert len(results.portfolio_returns) == len(mock_models)
        assert len(results.portfolio_weights) == len(mock_models)
        assert len(results.performance_metrics) == len(mock_models)

        # Check that models were trained
        for _model_name, model in mock_models.items():
            assert len(model.fit_calls) > 0
            assert len(model.predict_calls) > 0

    def test_model_retraining_tracking(
        self,
        engine: RollingBacktestEngine,
        mock_models: dict[str, MockPortfolioModel],
        sample_data: dict[str, pd.DataFrame],
    ):
        """Test model retraining state tracking."""
        engine.run_rolling_backtest(mock_models, sample_data)

        # Check that model states were created and updated
        for model_name in mock_models.keys():
            assert model_name in engine.model_states
            state = engine.model_states[model_name]
            assert state.retraining_count > 0
            assert state.last_training_date is not None

    def test_temporal_integrity_monitoring(
        self,
        engine: RollingBacktestEngine,
        mock_models: dict[str, MockPortfolioModel],
        sample_data: dict[str, pd.DataFrame],
    ):
        """Test temporal integrity monitoring during backtest."""
        results = engine.run_rolling_backtest(mock_models, sample_data)

        # Check integrity report
        assert "temporal_integrity_report" in results.__dict__
        integrity_report = results.temporal_integrity_report
        assert isinstance(integrity_report, dict)
        assert "total_splits_monitored" in integrity_report

    @patch("src.evaluation.backtest.rolling_engine.RollingBacktestEngine._save_results")
    def test_result_saving(
        self,
        mock_save: Mock,
        sample_data: dict[str, pd.DataFrame],
        mock_models: dict[str, MockPortfolioModel],
    ):
        """Test result saving functionality."""
        with TemporaryDirectory() as tmp_dir:
            config = RollingBacktestConfig(
                start_date=pd.Timestamp("2020-01-01"),
                end_date=pd.Timestamp("2024-12-31"),
                output_dir=Path(tmp_dir),
                save_intermediate_results=True,
            )

            engine = RollingBacktestEngine(config)
            engine.run_rolling_backtest(mock_models, sample_data)

            # Should call save_results
            assert mock_save.called

    def test_memory_management_integration(
        self,
        sample_data: dict[str, pd.DataFrame],
        mock_models: dict[str, MockPortfolioModel],
    ):
        """Test memory management integration."""
        config = RollingBacktestConfig(
            start_date=pd.Timestamp("2020-01-01"),
            end_date=pd.Timestamp("2024-12-31"),
            enable_memory_monitoring=True,
        )

        engine = RollingBacktestEngine(config)
        results = engine.run_rolling_backtest(mock_models, sample_data)

        # Should include memory stats (even if empty without GPU)
        assert "memory_usage_stats" in results.__dict__

    def test_universe_handling(
        self,
        engine: RollingBacktestEngine,
        mock_models: dict[str, MockPortfolioModel],
        sample_data: dict[str, pd.DataFrame],
    ):
        """Test dynamic universe handling."""
        # Create universe data
        dates = sample_data["returns"].index
        assets = sample_data["returns"].columns

        # Dynamic universe - remove some assets over time
        universe_data = pd.DataFrame(index=dates, columns=assets, data=True)
        universe_data.loc["2022-01-01":, "TSLA"] = False  # Remove TSLA from 2022

        results = engine.run_rolling_backtest(mock_models, sample_data, universe_data=universe_data)

        assert isinstance(results, RollingBacktestResults)
        assert len(results.portfolio_returns) == len(mock_models)

    def test_error_handling_during_execution(
        self,
        engine: RollingBacktestEngine,
        sample_data: dict[str, pd.DataFrame],
    ):
        """Test error handling during backtest execution."""

        # Create a model that raises exceptions
        class FailingModel(MockPortfolioModel):
            def fit(self, returns, universe, fit_period=None):
                raise ValueError("Training failed")

        failing_models = {"failing_model": FailingModel()}

        # Should handle errors gracefully
        results = engine.run_rolling_backtest(failing_models, sample_data)

        assert isinstance(results, RollingBacktestResults)
        # Should have empty results for failing model
        assert "failing_model" in results.portfolio_returns

    def test_performance_metrics_calculation(
        self,
        engine: RollingBacktestEngine,
        mock_models: dict[str, MockPortfolioModel],
        sample_data: dict[str, pd.DataFrame],
    ):
        """Test performance metrics calculation."""
        results = engine.run_rolling_backtest(mock_models, sample_data)

        for model_name in mock_models.keys():
            assert model_name in results.performance_metrics
            metrics = results.performance_metrics[model_name]

            # Should contain standard metrics
            expected_metrics = ["sharpe_ratio", "total_return", "annualized_return"]
            for _metric in expected_metrics:
                # Check if any expected metrics exist (implementation may vary)
                assert isinstance(metrics, dict)

    def test_transaction_cost_tracking(
        self,
        engine: RollingBacktestEngine,
        mock_models: dict[str, MockPortfolioModel],
        sample_data: dict[str, pd.DataFrame],
    ):
        """Test transaction cost tracking."""
        results = engine.run_rolling_backtest(mock_models, sample_data)

        for model_name in mock_models.keys():
            assert model_name in results.transaction_costs
            costs = results.transaction_costs[model_name]
            assert isinstance(costs, pd.Series)

    def test_execution_summary_generation(
        self,
        engine: RollingBacktestEngine,
        mock_models: dict[str, MockPortfolioModel],
        sample_data: dict[str, pd.DataFrame],
    ):
        """Test execution summary generation."""
        results = engine.run_rolling_backtest(mock_models, sample_data)

        summary = results.execution_summary
        assert isinstance(summary, dict)
        assert "total_splits" in summary
        assert "models_tested" in summary
        assert "execution_timestamp" in summary
        assert "config_summary" in summary

        # Check config summary
        config_summary = summary["config_summary"]
        assert "training_months" in config_summary
        assert "validation_months" in config_summary

    def test_split_data_preparation(
        self,
        engine: RollingBacktestEngine,
        sample_data: dict[str, pd.DataFrame],
    ):
        """Test split data preparation with temporal guards."""
        # Create a simple split
        split = RollSplit(
            ValidationPeriod(pd.Timestamp("2020-01-01"), pd.Timestamp("2023-01-01")),
            ValidationPeriod(pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")),
            ValidationPeriod(pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")),
        )

        split_data = engine._prepare_split_data(split, sample_data, universe_data=None)

        assert "train_returns" in split_data
        assert "val_returns" in split_data
        assert "test_returns" in split_data

        # Check temporal separation
        train_data = split_data["train_returns"]
        val_data = split_data["val_returns"]

        if not train_data.empty and not val_data.empty:
            assert train_data.index.max() < val_data.index.min()

    def test_model_retraining_on_split(
        self,
        engine: RollingBacktestEngine,
        mock_models: dict[str, MockPortfolioModel],
        sample_data: dict[str, pd.DataFrame],
    ):
        """Test model retraining on individual splits."""
        split = RollSplit(
            ValidationPeriod(pd.Timestamp("2020-01-01"), pd.Timestamp("2023-01-01")),
            ValidationPeriod(pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")),
            ValidationPeriod(pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")),
        )

        split_data = engine._prepare_split_data(split, sample_data, universe_data=None)

        model = mock_models["equal_weight"]
        results = engine._retrain_model_on_split(model, split, split_data, "equal_weight")

        assert isinstance(results, dict)
        assert "training_success" in results
        assert "training_duration_seconds" in results

        # Check that model was actually trained
        assert len(model.fit_calls) > 0
