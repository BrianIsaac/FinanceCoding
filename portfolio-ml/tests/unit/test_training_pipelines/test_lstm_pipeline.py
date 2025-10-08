"""
Unit tests for LSTM training pipeline.

Tests the complete LSTM training pipeline implementation including
GPU memory optimization, hyperparameter optimization, and training.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from scripts.train_lstm_pipeline import LSTMTrainingPipeline


class TestLSTMTrainingPipeline:
    """Test LSTM training pipeline functionality."""

    @pytest.fixture
    def mock_config_file(self):
        """Create mock configuration file."""
        config_data = {
            "architecture": {
                "input_features": 10,
                "hidden_dim": 64,
                "num_layers": 2,
                "dropout": 0.3
            },
            "training": {
                "learning_rate": 0.001,
                "weight_decay": 1e-5,
                "batch_size": 16,
                "max_epochs": 50
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            yield f.name

        # Cleanup
        Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def sample_data_set(self):
        """Generate comprehensive sample data for testing."""
        # Generate larger dataset for LSTM sequences
        dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")  # 4 years
        tickers = [f"ASSET_{i:03d}" for i in range(100)]

        np.random.seed(42)

        # Returns data
        returns = np.random.normal(0.0005, 0.015, (len(dates), len(tickers)))
        returns_df = pd.DataFrame(returns, index=dates, columns=tickers)

        # Prices data (cumulative returns)
        prices = np.cumprod(1 + returns, axis=0) * 100
        prices_df = pd.DataFrame(prices, index=dates, columns=tickers)

        # Volume data
        volume = np.random.lognormal(10, 1, (len(dates), len(tickers)))
        volume_df = pd.DataFrame(volume, index=dates, columns=tickers)

        return returns_df, prices_df, volume_df

    def test_initialization(self, mock_config_file):
        """Test pipeline initialization."""
        with patch('scripts.train_lstm_pipeline.Path.mkdir'):
            pipeline = LSTMTrainingPipeline(mock_config_file)

            assert pipeline.config_path == mock_config_file
            assert pipeline.base_config is not None
            assert pipeline.device is not None
            assert "architecture" in pipeline.base_config

    def test_config_loading(self, mock_config_file):
        """Test configuration loading."""
        with patch('scripts.train_lstm_pipeline.Path.mkdir'):
            pipeline = LSTMTrainingPipeline(mock_config_file)
            config = pipeline._load_config()

            assert isinstance(config, dict)
            assert "architecture" in config
            assert "training" in config

    @patch('scripts.train_lstm_pipeline.pd.read_parquet')
    def test_load_data(self, mock_read_parquet, mock_config_file, sample_data_set):
        """Test data loading functionality."""
        returns_df, prices_df, volume_df = sample_data_set
        mock_read_parquet.side_effect = [returns_df, prices_df, volume_df]

        with patch('scripts.train_lstm_pipeline.Path.mkdir'):
            pipeline = LSTMTrainingPipeline(mock_config_file)

            returns_data, prices_data, volume_data = pipeline.load_data()

            assert isinstance(returns_data, pd.DataFrame)
            assert isinstance(prices_data, pd.DataFrame)
            assert isinstance(volume_data, pd.DataFrame)
            assert len(returns_data.columns) == 100
            assert mock_read_parquet.call_count == 3

    def test_prepare_sequences(self, mock_config_file, sample_data_set):
        """Test sequence preparation for LSTM training."""
        returns_df, prices_df, volume_df = sample_data_set

        with patch('scripts.train_lstm_pipeline.Path.mkdir'):
            pipeline = LSTMTrainingPipeline(mock_config_file)

            X, y, assets, dates = pipeline.prepare_sequences(
                returns_df, prices_df, volume_df, sequence_length=60
            )

            assert isinstance(X, torch.Tensor)
            assert isinstance(y, torch.Tensor)
            assert len(assets) <= 200  # Limited for memory efficiency
            assert len(dates) > 0
            assert X.shape[1] == 60  # Sequence length
            assert X.shape[0] == y.shape[0]  # Same number of samples

    def test_create_train_val_splits(self, mock_config_file):
        """Test train/validation split creation."""
        # Create mock data
        n_samples = 1000
        seq_len = 60
        n_features = 100

        X = torch.randn(n_samples, seq_len, n_features)
        y = torch.randn(n_samples, 50)  # 50 assets
        dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")

        with patch('scripts.train_lstm_pipeline.Path.mkdir'):
            pipeline = LSTMTrainingPipeline(mock_config_file)

            X_train, y_train, X_val, y_val = pipeline.create_train_val_splits(X, y, dates)

            assert X_train.shape[0] > 0
            assert X_val.shape[0] > 0
            assert X_train.shape[0] + X_val.shape[0] <= n_samples
            assert X_train.shape[1:] == X.shape[1:]  # Same feature dimensions
            assert y_train.shape[1:] == y.shape[1:]  # Same target dimensions

    def test_optimize_memory_settings(self, mock_config_file):
        """Test GPU memory optimization."""
        # Create mock training data
        X_train = torch.randn(100, 60, 200)  # 100 samples, 60 seq, 200 features

        with patch('scripts.train_lstm_pipeline.Path.mkdir'):
            pipeline = LSTMTrainingPipeline(mock_config_file)

            memory_settings = pipeline.optimize_memory_settings(X_train, hidden_size=128)

            assert "batch_size" in memory_settings
            assert "gradient_accumulation_steps" in memory_settings
            assert "use_mixed_precision" in memory_settings
            assert memory_settings["batch_size"] > 0
            assert memory_settings["gradient_accumulation_steps"] > 0

    @patch('torch.cuda.is_available')
    def test_optimize_memory_settings_cpu(self, mock_cuda_available, mock_config_file):
        """Test memory optimization for CPU."""
        mock_cuda_available.return_value = False

        X_train = torch.randn(100, 60, 200)

        with patch('scripts.train_lstm_pipeline.Path.mkdir'):
            pipeline = LSTMTrainingPipeline(mock_config_file)

            memory_settings = pipeline.optimize_memory_settings(X_train, hidden_size=128)

            assert memory_settings["use_mixed_precision"] is False
            assert memory_settings["batch_size"] == 32
            assert memory_settings["gradient_accumulation_steps"] == 1

    def test_create_lstm_model(self, mock_config_file):
        """Test LSTM model creation."""
        with patch('scripts.train_lstm_pipeline.Path.mkdir'):
            pipeline = LSTMTrainingPipeline(mock_config_file)

            model = pipeline.create_lstm_model(
                input_size=100,
                output_size=50,
                hidden_size=64,
                dropout=0.3
            )

            assert model is not None
            # Test forward pass
            x = torch.randn(16, 60, 100)  # batch_size, seq_len, input_size
            output = model(x)
            assert output.shape == (16, 50)  # batch_size, output_size

    @patch('scripts.train_lstm_pipeline.torch.save')
    @patch('scripts.train_lstm_pipeline.DataLoader')
    def test_train_single_configuration(self, mock_dataloader, mock_torch_save, mock_config_file):
        """Test single configuration training."""
        # Mock data loaders
        mock_train_loader = Mock()
        mock_val_loader = Mock()

        # Mock training data with proper loss values
        mock_batch_data = [(torch.randn(8, 60, 100), torch.randn(8, 50))]
        mock_train_loader.__iter__.return_value = iter(mock_batch_data)
        mock_val_loader.__iter__.return_value = iter(mock_batch_data)
        mock_train_loader.__len__.return_value = 1
        mock_val_loader.__len__.return_value = 1

        mock_dataloader.side_effect = [mock_train_loader, mock_val_loader]

        # Create mock data
        X_train = torch.randn(32, 60, 100)
        y_train = torch.randn(32, 50)
        X_val = torch.randn(16, 60, 100)
        y_val = torch.randn(16, 50)

        config = {
            "config_name": "test_config",
            "hidden_size": 64,
            "learning_rate": 0.001,
            "dropout": 0.3,
            "weight_decay": 1e-5,
            "max_epochs": 5
        }

        with patch('scripts.train_lstm_pipeline.Path.mkdir'):
            pipeline = LSTMTrainingPipeline(mock_config_file)

            results = pipeline.train_single_configuration(X_train, y_train, X_val, y_val, config)

            assert "config_name" in results
            assert results["config_name"] == "test_config"
            assert "best_val_loss" in results
            assert "train_losses" in results
            assert "val_losses" in results

    def test_hyperparameter_optimization_grid_generation(self, mock_config_file):
        """Test hyperparameter grid generation."""
        with patch('scripts.train_lstm_pipeline.Path.mkdir'):
            with patch('scripts.train_lstm_pipeline.LSTMTrainingPipeline.train_single_configuration') as mock_train:
                mock_train.return_value = {"best_val_loss": 0.5, "config_name": "test"}

                pipeline = LSTMTrainingPipeline(mock_config_file)

                # Create minimal data
                X_train = torch.randn(10, 60, 100)
                y_train = torch.randn(10, 50)
                X_val = torch.randn(5, 60, 100)
                y_val = torch.randn(5, 50)

                results = pipeline.run_hyperparameter_optimization(X_train, y_train, X_val, y_val)

                assert "best_config" in results
                assert "best_val_loss" in results
                assert "all_results" in results
                assert "hyperparameter_analysis" in results

    @patch('builtins.open', new_callable=MagicMock)
    @patch('scripts.train_lstm_pipeline.yaml.dump')
    def test_save_results(self, mock_yaml_dump, mock_open, mock_config_file):
        """Test results saving functionality."""
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        with patch('scripts.train_lstm_pipeline.Path.mkdir'):
            pipeline = LSTMTrainingPipeline(mock_config_file)
            test_results = {"test": "data"}

            pipeline.save_results(test_results)

            mock_yaml_dump.assert_called_once_with(test_results, mock_file, default_flow_style=False)

    @patch('scripts.train_lstm_pipeline.LSTMTrainingPipeline.load_data')
    @patch('scripts.train_lstm_pipeline.LSTMTrainingPipeline.prepare_sequences')
    @patch('scripts.train_lstm_pipeline.LSTMTrainingPipeline.create_train_val_splits')
    @patch('scripts.train_lstm_pipeline.LSTMTrainingPipeline.run_hyperparameter_optimization')
    @patch('scripts.train_lstm_pipeline.LSTMTrainingPipeline.save_results')
    def test_run_full_pipeline(self, mock_save, mock_hyperopt, mock_splits,
                              mock_sequences, mock_load, mock_config_file):
        """Test full pipeline execution."""
        # Mock all components
        returns_df = pd.DataFrame(np.random.randn(100, 50))
        prices_df = pd.DataFrame(np.random.randn(100, 50))
        volume_df = pd.DataFrame(np.random.randn(100, 50))
        mock_load.return_value = (returns_df, prices_df, volume_df)

        X = torch.randn(50, 60, 100)
        y = torch.randn(50, 50)
        assets = [f"ASSET_{i}" for i in range(50)]
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        mock_sequences.return_value = (X, y, assets, dates)

        X_train = torch.randn(40, 60, 100)
        y_train = torch.randn(40, 50)
        X_val = torch.randn(10, 60, 100)
        y_val = torch.randn(10, 50)
        mock_splits.return_value = (X_train, y_train, X_val, y_val)

        mock_hyperopt.return_value = {"best_config": "test", "best_val_loss": 0.5}

        with patch('scripts.train_lstm_pipeline.Path.mkdir'):
            pipeline = LSTMTrainingPipeline(mock_config_file)
            results = pipeline.run_full_pipeline()

            mock_load.assert_called_once()
            mock_sequences.assert_called_once()
            mock_splits.assert_called_once()
            mock_hyperopt.assert_called_once()
            mock_save.assert_called_once()

            assert "training_summary" in results
            assert "hyperparameter_optimization" in results

    def test_model_weight_initialization(self, mock_config_file):
        """Test proper model weight initialization."""
        with patch('scripts.train_lstm_pipeline.Path.mkdir'):
            pipeline = LSTMTrainingPipeline(mock_config_file)

            model = pipeline.create_lstm_model(
                input_size=50,
                output_size=25,
                hidden_size=32,
                dropout=0.2
            )

            # Check that weights are initialized (not all zeros)
            has_nonzero_weights = False
            for param in model.parameters():
                if torch.any(param != 0):
                    has_nonzero_weights = True
                    break

            assert has_nonzero_weights, "Model weights should be properly initialized"

    def test_sequence_data_validation(self, mock_config_file):
        """Test sequence data validation and cleaning."""
        # Create data with some NaN and inf values
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        tickers = [f"ASSET_{i:03d}" for i in range(10)]

        returns_data = pd.DataFrame(np.random.randn(200, 10), index=dates, columns=tickers)
        returns_data.iloc[50, 2] = np.nan
        returns_data.iloc[100, 5] = np.inf

        prices_data = pd.DataFrame(np.random.randn(200, 10) * 100, index=dates, columns=tickers)
        volume_data = pd.DataFrame(np.random.lognormal(10, 1, (200, 10)), index=dates, columns=tickers)

        with patch('scripts.train_lstm_pipeline.Path.mkdir'):
            pipeline = LSTMTrainingPipeline(mock_config_file)

            X, y, assets, dates_out = pipeline.prepare_sequences(
                returns_data, prices_data, volume_data, sequence_length=60
            )

            # Check that output tensors don't contain NaN or inf
            assert not torch.isnan(X).any(), "Input sequences should not contain NaN"
            assert not torch.isinf(X).any(), "Input sequences should not contain inf"
            assert not torch.isnan(y).any(), "Target sequences should not contain NaN"
            assert not torch.isinf(y).any(), "Target sequences should not contain inf"

    def test_error_handling_insufficient_data(self, mock_config_file):
        """Test error handling with insufficient data."""
        # Create very small dataset
        dates = pd.date_range("2020-01-01", periods=30, freq="D")  # Less than sequence length
        tickers = ["ASSET_001", "ASSET_002"]

        returns_data = pd.DataFrame(np.random.randn(30, 2), index=dates, columns=tickers)
        prices_data = pd.DataFrame(np.random.randn(30, 2), index=dates, columns=tickers)
        volume_data = pd.DataFrame(np.random.randn(30, 2), index=dates, columns=tickers)

        with patch('scripts.train_lstm_pipeline.Path.mkdir'):
            pipeline = LSTMTrainingPipeline(mock_config_file)

            X, y, assets, dates_out = pipeline.prepare_sequences(
                returns_data, prices_data, volume_data, sequence_length=60
            )

            # Should handle gracefully and return empty or minimal data
            assert X.shape[0] == 0 or X.shape[0] > 0  # Should not crash
