"""
Unit tests for LSTM portfolio model integration.

Tests the complete LSTM portfolio model including constraint integration,
weight generation, and model persistence.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.models.base.portfolio_model import PortfolioConstraints
from src.models.lstm.architecture import LSTMConfig
from src.models.lstm.model import (
    LSTMModelConfig,
    LSTMPortfolioModel,
    create_lstm_model,
)
from src.models.lstm.training import TrainingConfig


class TestLSTMModelConfig:
    """Test LSTM model configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LSTMModelConfig()

        assert isinstance(config.lstm_config, LSTMConfig)
        assert isinstance(config.training_config, TrainingConfig)
        assert config.lookback_days == 756
        assert config.rebalancing_frequency == "monthly"
        assert config.prediction_horizon == 21
        assert config.risk_aversion == 1.0
        assert config.use_markowitz_layer is True

    def test_custom_config(self):
        """Test custom configuration values."""
        lstm_config = LSTMConfig(hidden_size=64, num_attention_heads=4)
        training_config = TrainingConfig(learning_rate=0.01, epochs=50)

        config = LSTMModelConfig(
            lstm_config=lstm_config,
            training_config=training_config,
            lookback_days=500,
            risk_aversion=2.0,
        )

        assert config.lstm_config == lstm_config
        assert config.training_config == training_config
        assert config.lookback_days == 500
        assert config.risk_aversion == 2.0

    def test_yaml_serialization(self):
        """Test YAML save/load functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"

            # Create and save config
            original_config = LSTMModelConfig(lookback_days=600, risk_aversion=1.5)
            original_config.to_yaml(config_path)

            assert config_path.exists()

            # Load config
            loaded_config = LSTMModelConfig.from_yaml(config_path)

            assert loaded_config.lookback_days == 600
            assert loaded_config.risk_aversion == 1.5
            assert loaded_config.lstm_config.hidden_size == 128  # Default value


class TestLSTMPortfolioModel:
    """Test LSTM portfolio model."""

    @pytest.fixture
    def constraints(self):
        """Create test portfolio constraints."""
        return PortfolioConstraints(
            long_only=True,
            top_k_positions=10,
            max_position_weight=0.80,  # Allow higher weights for test validity
            max_monthly_turnover=0.15,
        )

    @pytest.fixture
    def model_config(self):
        """Create test model configuration."""
        lstm_config = LSTMConfig(
            sequence_length=10,
            input_size=5,
            hidden_size=16,
            num_layers=1,
            dropout=0.1,
            num_attention_heads=2,
            output_size=5,
        )

        training_config = TrainingConfig(
            max_memory_gb=1.0, epochs=2, batch_size=4, use_mixed_precision=False
        )

        return LSTMModelConfig(
            lstm_config=lstm_config,
            training_config=training_config,
            prediction_horizon=1,  # Reduced for testing
        )

    @pytest.fixture
    def model(self, constraints, model_config):
        """Create LSTM portfolio model for testing."""
        return LSTMPortfolioModel(constraints, model_config)

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data for testing."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")  # Increased to 100 days
        assets = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

        np.random.seed(42)
        returns_data = np.random.normal(0.001, 0.02, (100, 5))  # Increased to 100 days

        return pd.DataFrame(returns_data, index=dates, columns=assets)

    def test_model_initialization(self, model, constraints, model_config):
        """Test model initialization."""
        assert model.constraints == constraints
        assert model.config == model_config
        assert model.network is None
        assert model.trainer is None
        assert model.universe is None
        assert not model.is_fitted

    def test_model_initialization_with_defaults(self, constraints):
        """Test model initialization with default configuration."""
        model = LSTMPortfolioModel(constraints)

        assert isinstance(model.config, LSTMModelConfig)
        assert not model.is_fitted

    def test_fit_input_validation(self, model, sample_returns):
        """Test fit method input validation."""
        universe = ["AAPL", "MSFT"]
        fit_period = (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-31"))

        # Test empty returns
        with pytest.raises(ValueError, match="Returns DataFrame is empty"):
            model.fit(pd.DataFrame(), universe, fit_period)

        # Test empty universe
        with pytest.raises(ValueError, match="Universe cannot be empty"):
            model.fit(sample_returns, [], fit_period)

        # Test missing assets
        with pytest.raises(ValueError, match="Missing assets in returns data"):
            model.fit(sample_returns, ["INVALID_ASSET"], fit_period)

        # Test invalid period
        invalid_period = (pd.Timestamp("2020-01-31"), pd.Timestamp("2020-01-01"))
        with pytest.raises(ValueError, match="Invalid fit period"):
            model.fit(sample_returns, universe, invalid_period)

    def test_fit_insufficient_data(self, model, sample_returns):
        """Test fit with insufficient data."""
        universe = ["AAPL", "MSFT"]
        # Period too short for sequence length + prediction horizon
        fit_period = (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-05"))

        with pytest.raises(ValueError, match="Insufficient data"):
            model.fit(sample_returns, universe, fit_period)

    @patch("src.models.lstm.model.create_trainer")
    @patch("src.models.lstm.model.create_lstm_network")
    def test_fit_success(self, mock_create_network, mock_create_trainer, model, sample_returns):
        """Test successful model fitting."""
        # Mock network
        mock_network = Mock()
        mock_create_network.return_value = mock_network

        # Mock trainer
        mock_trainer = Mock()
        mock_trainer.fit.return_value = {"train_loss": [0.1], "val_loss": [0.2]}
        mock_create_trainer.return_value = mock_trainer

        universe = ["AAPL", "MSFT", "GOOGL"]
        fit_period = (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-03-15"))  # Extended to 75 days

        # Fit model
        model.fit(sample_returns, universe, fit_period)

        # Verify model state
        assert model.is_fitted
        assert model.universe == universe
        assert model.fitted_period == fit_period
        assert model.network == mock_network
        assert model.training_history is not None

        # Verify creation functions were called
        mock_create_network.assert_called_once()
        mock_create_trainer.assert_called_once()
        mock_trainer.fit.assert_called_once()

    def test_predict_weights_not_fitted(self, model):
        """Test predict_weights on unfitted model."""
        date = pd.Timestamp("2020-01-15")
        universe = ["AAPL", "MSFT"]

        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict_weights(date, universe)

    def test_predict_weights_invalid_universe(self, model, sample_returns):
        """Test predict_weights with invalid universe."""
        # Mock fitted model
        model.is_fitted = True
        model.universe = ["AAPL", "MSFT"]
        model.network = Mock()

        date = pd.Timestamp("2020-01-15")
        invalid_universe = ["INVALID_ASSET"]

        with pytest.raises(ValueError, match="Universe contains assets not in fitted universe"):
            model.predict_weights(date, invalid_universe)

    @patch("src.models.lstm.model.LSTMPortfolioModel._predict_returns")
    @patch("src.models.lstm.model.LSTMPortfolioModel._optimize_portfolio")
    def test_predict_weights_with_markowitz(self, mock_optimize, mock_predict, model):
        """Test weight prediction with Markowitz optimization."""
        # Setup fitted model
        model.is_fitted = True
        model.universe = ["AAPL", "MSFT"]
        model.network = Mock()
        model.config.use_markowitz_layer = True

        # Mock return predictions and optimization
        mock_predict.return_value = np.array([0.01, 0.02])
        mock_optimize.return_value = pd.Series([0.25, 0.75], index=["AAPL", "MSFT"])

        date = pd.Timestamp("2020-01-15")
        universe = ["AAPL", "MSFT"]

        weights = model.predict_weights(date, universe)

        assert isinstance(weights, pd.Series)
        assert len(weights) == 2
        assert abs(weights.sum() - 1.0) < 1e-6  # Weights sum to 1

        mock_predict.assert_called_once_with(date, universe)
        mock_optimize.assert_called_once()

    @patch("src.models.lstm.model.LSTMPortfolioModel._predict_returns")
    def test_predict_weights_without_markowitz(self, mock_predict, model):
        """Test weight prediction without Markowitz optimization."""
        # Setup fitted model
        model.is_fitted = True
        model.universe = ["AAPL", "MSFT"]
        model.network = Mock()
        model.config.use_markowitz_layer = False

        # Mock return predictions
        mock_predict.return_value = np.array([0.01, 0.02])

        date = pd.Timestamp("2020-01-15")
        universe = ["AAPL", "MSFT"]

        weights = model.predict_weights(date, universe)

        assert isinstance(weights, pd.Series)
        assert len(weights) == 2
        assert abs(weights.sum() - 1.0) < 1e-6  # Weights sum to 1
        assert all(weights >= 0)  # Non-negative weights

    def test_predict_returns_mock_implementation(self, model):
        """Test mock return prediction implementation."""
        # Setup model
        model.universe = ["AAPL", "MSFT"]

        date = pd.Timestamp("2020-01-15")
        universe = ["AAPL", "MSFT"]

        predictions = model._predict_returns(date, universe)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(universe)
        assert predictions.dtype == np.float64

    def test_optimize_portfolio(self, model):
        """Test portfolio optimization functionality."""
        universe = ["AAPL", "MSFT", "GOOGL"]
        expected_returns = np.array([0.01, 0.02, 0.015])
        date = pd.Timestamp("2020-01-15")

        weights = model._optimize_portfolio(expected_returns, universe, date)

        assert isinstance(weights, pd.Series)
        assert len(weights) == len(universe)
        assert abs(weights.sum() - 1.0) < 1e-3  # Weights approximately sum to 1
        assert all(weights >= -1e-6)  # Non-negative (allowing small numerical errors)

    def test_get_model_info_unfitted(self, model):
        """Test model info for unfitted model."""
        info = model.get_model_info()

        assert info["model_type"] == "LSTM"
        assert info["is_fitted"] is False
        assert info["universe_size"] is None
        assert info["fitted_period"] is None
        assert "constraints" in info
        assert "lstm_config" in info
        assert "training_config" in info

    def test_get_model_info_fitted(self, model, sample_returns):
        """Test model info for fitted model."""
        # Mock fitted model
        model.is_fitted = True
        model.universe = ["AAPL", "MSFT"]
        model.fitted_period = (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-31"))
        model.training_history = {"train_loss": [0.1, 0.05], "val_loss": [0.2, 0.15]}
        model.network = Mock()
        # Mock parameters method
        mock_param1 = Mock()
        mock_param1.numel.return_value = 100  # 10*10
        mock_param2 = Mock()
        mock_param2.numel.return_value = 5
        model.network.parameters.return_value = [mock_param1, mock_param2]

        info = model.get_model_info()

        assert info["is_fitted"] is True
        assert info["universe_size"] == 2
        assert info["fitted_period"] is not None
        assert "training_stats" in info
        assert info["training_stats"]["final_train_loss"] == 0.05
        assert info["network_params"] == 105  # 10*10 + 5

    def test_save_load_model(self, model, sample_returns):
        """Test model save/load functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.pth"

            # Setup fitted model
            model.is_fitted = True
            model.universe = ["AAPL", "MSFT"]
            model.fitted_period = (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-31"))
            model.training_history = {"train_loss": [0.1], "val_loss": [0.2]}
            model.network = Mock()
            model.network.state_dict.return_value = {"param1": torch.tensor([1.0, 2.0])}
            # Mock parameters method for get_model_info
            mock_param1 = Mock()
            mock_param1.numel.return_value = 10
            mock_param2 = Mock()
            mock_param2.numel.return_value = 5
            model.network.parameters.return_value = [mock_param1, mock_param2]

            # Save model
            model.save_model(model_path)
            assert model_path.exists()

            # Create new model and load
            new_model = LSTMPortfolioModel(model.constraints, model.config)

            # Mock network creation for loading
            with patch("src.models.lstm.model.create_lstm_network") as mock_create:
                mock_network = Mock()
                mock_create.return_value = mock_network

                new_model.load_model(model_path)

            # Verify loaded model
            assert new_model.is_fitted
            assert new_model.universe == ["AAPL", "MSFT"]
            assert new_model.fitted_period == model.fitted_period

            mock_network.load_state_dict.assert_called_once()

    def test_save_model_unfitted(self, model):
        """Test saving unfitted model raises error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.pth"

            with pytest.raises(ValueError, match="Cannot save unfitted model"):
                model.save_model(model_path)

    def test_constraint_validation(self, model, sample_returns):
        """Test that portfolio constraints are properly applied."""
        # Setup fitted model
        model.is_fitted = True
        model.universe = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        model.network = Mock()

        # Mock return predictions that would violate constraints
        with patch.object(model, "_predict_returns") as mock_predict:
            mock_predict.return_value = np.array([0.1, 0.05, -0.02, 0.08])  # Negative return

            date = pd.Timestamp("2020-01-15")
            universe = ["AAPL", "MSFT", "GOOGL", "AMZN"]

            weights = model.predict_weights(date, universe)

            # Check constraint satisfaction
            assert all(weights >= 0)  # Long-only constraint
            assert abs(weights.sum() - 1.0) < 1e-6  # Weights sum to 1
            assert all(
                weights <= model.constraints.max_position_weight + 1e-6
            )  # Max position weight


class TestLSTMModelFactory:
    """Test LSTM model factory function."""

    def test_create_model_with_defaults(self):
        """Test creating model with default configuration."""
        constraints = PortfolioConstraints()

        model = create_lstm_model(constraints)

        assert isinstance(model, LSTMPortfolioModel)
        assert model.constraints == constraints
        assert isinstance(model.config, LSTMModelConfig)

    def test_create_model_with_config_file(self):
        """Test creating model with configuration file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"

            # Create test config file
            test_config = {
                "lookback_days": 500,
                "risk_aversion": 2.0,
                "lstm_config": {"hidden_size": 64, "num_layers": 3},
            }

            with open(config_path, "w") as f:
                yaml.dump(test_config, f)

            constraints = PortfolioConstraints()
            model = create_lstm_model(constraints, config_path)

            assert model.config.lookback_days == 500
            assert model.config.risk_aversion == 2.0
            assert model.config.lstm_config.hidden_size == 64

    def test_create_model_with_overrides(self):
        """Test creating model with configuration overrides."""
        constraints = PortfolioConstraints()

        model = create_lstm_model(constraints, risk_aversion=3.0, lookback_days=1000)

        assert model.config.risk_aversion == 3.0
        assert model.config.lookback_days == 1000

    def test_create_model_nonexistent_config(self):
        """Test creating model with non-existent config file."""
        constraints = PortfolioConstraints()
        nonexistent_path = Path("nonexistent_config.yaml")

        # Should use defaults when file doesn't exist
        model = create_lstm_model(constraints, nonexistent_path)

        assert isinstance(model, LSTMPortfolioModel)
        assert isinstance(model.config, LSTMModelConfig)


class TestLSTMModelIntegration:
    """Integration tests for LSTM model."""

    @pytest.mark.slow
    def test_minimal_end_to_end_workflow(self):
        """Test minimal end-to-end model workflow."""
        # Create model with minimal configuration
        constraints = PortfolioConstraints(long_only=True, max_position_weight=0.5)

        lstm_config = LSTMConfig(
            sequence_length=5,
            input_size=3,
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
            num_attention_heads=2,
            output_size=3,
        )

        training_config = TrainingConfig(epochs=1, batch_size=2, use_mixed_precision=False)

        config = LSTMModelConfig(lstm_config=lstm_config, training_config=training_config)

        model = LSTMPortfolioModel(constraints, config)

        # Create synthetic data
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        returns = pd.DataFrame(np.random.randn(30, 3) * 0.01, index=dates, columns=["A", "B", "C"])

        universe = ["A", "B", "C"]
        fit_period = (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-29"))  # Extended to 28 days

        # Fit model (this should complete without errors)
        model.fit(returns, universe, fit_period)

        # Generate weights
        prediction_date = pd.Timestamp("2020-01-26")
        weights = model.predict_weights(prediction_date, universe)

        # Verify results
        assert model.is_fitted
        assert isinstance(weights, pd.Series)
        assert len(weights) == 3
        assert abs(weights.sum() - 1.0) < 1e-6
        assert all(weights >= 0)  # Long-only

        # Get model info
        info = model.get_model_info()
        assert info["is_fitted"] is True
        assert info["model_type"] == "LSTM"
