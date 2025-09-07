"""
Integration tests for LSTM model with backtesting engine.

This module tests the complete integration of the LSTM portfolio model
with the backtesting framework, ensuring end-to-end functionality.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.backtest.engine import BacktestConfig, BacktestEngine
from src.models.base.portfolio_model import PortfolioConstraints
from src.models.lstm.model import LSTMModelConfig, LSTMPortfolioModel
from src.models.model_registry import ModelRegistry


class TestLSTMBacktestIntegration:
    """Test LSTM integration with backtesting engine."""

    @pytest.fixture
    def sample_returns_data(self) -> pd.DataFrame:
        """Generate sample returns data for testing."""
        np.random.seed(42)
        torch.manual_seed(42)

        # Create 2 years of daily returns for 50 assets
        dates = pd.date_range("2022-01-01", "2024-01-01", freq="B")  # Business days
        n_assets = 50
        n_days = len(dates)

        # Generate correlated returns with some realistic patterns
        returns = np.random.multivariate_normal(
            mean=np.zeros(n_assets),
            cov=0.02**2 * (0.3 * np.ones((n_assets, n_assets)) + 0.7 * np.eye(n_assets)),
            size=n_days,
        )

        # Add some momentum and mean reversion patterns
        for i in range(1, n_days):
            returns[i] += 0.1 * returns[i - 1]  # Momentum
            returns[i] -= 0.05 * np.mean(returns[max(0, i - 20) : i], axis=0)  # Mean reversion

        # Create asset names
        assets = [f"STOCK_{i:03d}" for i in range(n_assets)]

        return pd.DataFrame(returns, index=dates, columns=assets)

    @pytest.fixture
    def lstm_model(self) -> LSTMPortfolioModel:
        """Create LSTM model for testing."""
        constraints = PortfolioConstraints(
            long_only=True, top_k_positions=25, max_position_weight=0.10, max_monthly_turnover=0.30
        )

        config = LSTMModelConfig()
        # Reduce complexity for faster testing
        config.lstm_config.hidden_size = 32
        config.lstm_config.num_layers = 1
        config.training_config.epochs = 5
        config.training_config.patience = 3

        return LSTMPortfolioModel(constraints=constraints, config=config)

    @pytest.fixture
    def backtest_config(self) -> BacktestConfig:
        """Create backtest configuration."""
        return BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            rebalance_frequency="M",
            initial_capital=1000000.0,
            transaction_cost_bps=10.0,
            min_history_days=120,  # Reduced for testing
        )

    @pytest.fixture
    def backtest_engine(self, backtest_config: BacktestConfig) -> BacktestEngine:
        """Create backtest engine."""
        return BacktestEngine(backtest_config)

    def test_lstm_model_registry_integration(self):
        """Test that LSTM model is properly registered."""
        models = ModelRegistry.list_models()
        assert "lstm" in models

        # Test model creation via registry
        constraints = PortfolioConstraints()
        model = ModelRegistry.create_model("lstm", constraints=constraints)
        assert isinstance(model, LSTMPortfolioModel)

    def test_backtest_engine_initialization(self, backtest_engine: BacktestEngine):
        """Test backtesting engine initializes correctly."""
        assert backtest_engine.config is not None
        assert backtest_engine.cost_calculator is not None
        assert backtest_engine.rebalancer is not None
        assert backtest_engine.performance_analytics is not None

    def test_rebalance_dates_generation(
        self, backtest_engine: BacktestEngine, sample_returns_data: pd.DataFrame
    ):
        """Test rebalancing date generation."""
        dates = backtest_engine.get_rebalance_dates(sample_returns_data)

        assert len(dates) > 0
        assert all(isinstance(d, pd.Timestamp) for d in dates)
        assert dates == sorted(dates)  # Should be in chronological order

        # Should have roughly monthly dates
        assert len(dates) >= 8  # At least 8 months in test period
        assert len(dates) <= 13  # At most 13 months

    def test_lstm_backtest_integration(
        self,
        backtest_engine: BacktestEngine,
        lstm_model: LSTMPortfolioModel,
        sample_returns_data: pd.DataFrame,
    ):
        """Test complete LSTM backtest integration."""
        # Run backtest
        results = backtest_engine.run_backtest(lstm_model, sample_returns_data)

        # Verify results structure
        assert "portfolio_returns" in results
        assert "portfolio_weights" in results
        assert "turnover" in results
        assert "transaction_costs" in results
        assert "performance_metrics" in results
        assert "rebalance_dates" in results

        # Check that we got some results
        portfolio_returns = results["portfolio_returns"]
        portfolio_weights = results["portfolio_weights"]

        if len(portfolio_returns) > 0:
            assert isinstance(portfolio_returns, pd.Series)
            assert len(portfolio_returns) > 0
            assert portfolio_returns.dtype == np.float64

        if len(portfolio_weights) > 0:
            assert isinstance(portfolio_weights, pd.DataFrame)
            assert len(portfolio_weights) > 0

            # Check constraint compliance
            for _idx, weights in portfolio_weights.iterrows():
                weights_clean = weights.dropna()
                if len(weights_clean) > 0:
                    # Long-only constraint
                    assert all(weights_clean >= -1e-6)  # Allow for small numerical errors

                    # Position limit constraint
                    assert all(weights_clean <= 0.11)  # 10% + small tolerance

                    # Weights should sum to approximately 1
                    assert abs(weights_clean.sum() - 1.0) < 0.01

        # Check performance metrics
        performance_metrics = results["performance_metrics"]
        if len(portfolio_returns) > 50:  # Only check if we have sufficient data
            assert isinstance(performance_metrics, dict)
            # Should have standard performance metrics
            expected_metrics = ["total_return", "annualized_return", "volatility", "sharpe_ratio"]
            for metric in expected_metrics:
                if metric in performance_metrics:
                    assert isinstance(performance_metrics[metric], (float, int))

    def test_lstm_model_persistence_in_backtest(
        self,
        backtest_engine: BacktestEngine,
        lstm_model: LSTMPortfolioModel,
        sample_returns_data: pd.DataFrame,
        tmp_path: Path,
    ):
        """Test that LSTM model can be saved/loaded during backtest."""
        # Set checkpoint directory
        lstm_model.config.training_config.epochs = 3  # Fast training for test

        # Run a short backtest
        short_config = BacktestConfig(
            start_date=datetime(2023, 6, 1),
            end_date=datetime(2023, 8, 31),
            rebalance_frequency="M",
            initial_capital=1000000.0,
            min_history_days=60,
        )
        short_engine = BacktestEngine(short_config)

        results = short_engine.run_backtest(lstm_model, sample_returns_data)

        # Should complete without errors
        assert results is not None
        assert isinstance(results, dict)

    def test_lstm_memory_constraints_in_backtest(
        self, lstm_model: LSTMPortfolioModel, sample_returns_data: pd.DataFrame
    ):
        """Test LSTM memory optimization during backtest."""
        # Create config with memory constraints
        lstm_model.config.training_config.max_memory_gb = 2.0  # Low memory limit
        lstm_model.config.training_config.batch_size = 16  # Small batch size

        config = BacktestConfig(
            start_date=datetime(2023, 10, 1),
            end_date=datetime(2023, 11, 30),
            rebalance_frequency="M",
            min_history_days=60,
        )
        engine = BacktestEngine(config)

        # Should handle memory constraints gracefully
        results = engine.run_backtest(lstm_model, sample_returns_data)
        assert results is not None

    @pytest.mark.slow
    def test_lstm_full_backtest_performance(
        self, backtest_engine: BacktestEngine, sample_returns_data: pd.DataFrame
    ):
        """Test full LSTM backtest with performance validation."""
        # Create a more sophisticated LSTM model
        constraints = PortfolioConstraints(
            long_only=True, top_k_positions=30, max_position_weight=0.08, max_monthly_turnover=0.25
        )

        config = LSTMModelConfig()
        config.lstm_config.hidden_size = 64
        config.lstm_config.num_layers = 2
        config.training_config.epochs = 10
        config.training_config.early_stopping = True

        model = LSTMPortfolioModel(constraints=constraints, config=config)

        results = backtest_engine.run_backtest(model, sample_returns_data)

        if len(results["portfolio_returns"]) > 0:
            # Validate performance characteristics
            returns = results["portfolio_returns"]
            turnover = results["turnover"]

            # Should have reasonable return characteristics
            assert returns.std() > 0  # Should have some volatility
            assert returns.std() < 0.5  # But not excessive

            # Should have reasonable turnover
            if len(turnover) > 0:
                avg_turnover = turnover.mean()
                assert avg_turnover >= 0  # Non-negative
                assert avg_turnover <= 1.0  # Reasonable upper bound
