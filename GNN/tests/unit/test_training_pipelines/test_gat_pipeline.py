"""
Unit tests for GAT training pipeline.

Tests the complete GAT training pipeline implementation including
multi-graph construction methods, memory optimization, and Sharpe optimization.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from scripts.train_gat_pipeline import GATTrainingPipeline


class TestGATTrainingPipeline:
    """Test GAT training pipeline functionality."""

    @pytest.fixture
    def mock_config_file(self):
        """Create mock configuration file."""
        config_data = {
            'architecture': {
                'input_features': 10,
                'hidden_dim': 64,
                'num_layers': 3,
                'num_attention_heads': 8,
                'dropout': 0.3,
                'edge_feature_dim': 3,
                'use_gatv2': True,
                'residual': True
            },
            'training': {
                'learning_rate': 0.001,
                'weight_decay': 1e-5,
                'batch_size': 16,
                'max_epochs': 100,
                'patience': 15,
                'use_mixed_precision': True,
                'gradient_accumulation_steps': 4
            },
            'graph_construction': {
                'lookback_days': 252,
                'methods': ['mst', 'tmfg', 'knn'],
                'knn_k_values': [5, 10, 15],
                'use_edge_attr': True
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            yield f.name

        # Cleanup
        Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def sample_returns_data(self):
        """Generate sample returns data for testing."""
        # Extend range to cover the method's expected date range (2016-2024)
        dates = pd.date_range("2015-01-01", "2024-01-01", freq="D")
        tickers = [f"ASSET_{i:03d}" for i in range(100)]

        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.015, (len(dates), len(tickers)))

        return pd.DataFrame(returns, index=dates, columns=tickers)

    def test_initialization(self, mock_config_file):
        """Test pipeline initialization."""
        with patch('scripts.train_gat_pipeline.Path.mkdir'):
            pipeline = GATTrainingPipeline(mock_config_file)

            assert pipeline.config_path == mock_config_file
            assert pipeline.base_config is not None
            assert "architecture" in pipeline.base_config
            assert "training" in pipeline.base_config
            assert "graph_construction" in pipeline.base_config

    def test_initialization_with_missing_config(self):
        """Test initialization with missing config file."""
        with patch('scripts.train_gat_pipeline.Path.mkdir'):
            pipeline = GATTrainingPipeline("nonexistent_config.yaml")

            # Should use default config
            assert pipeline.base_config is not None
            assert "architecture" in pipeline.base_config

    def test_config_loading(self, mock_config_file):
        """Test configuration loading."""
        with patch('scripts.train_gat_pipeline.Path.mkdir'):
            pipeline = GATTrainingPipeline(mock_config_file)
            config = pipeline._load_config()

            assert isinstance(config, dict)
            assert "architecture" in config
            assert "training" in config
            assert config["architecture"]["hidden_dim"] == 64

    def test_default_config_generation(self, mock_config_file):
        """Test default configuration generation."""
        with patch('scripts.train_gat_pipeline.Path.mkdir'):
            pipeline = GATTrainingPipeline(mock_config_file)
            default_config = pipeline._get_default_config()

            assert isinstance(default_config, dict)
            assert "architecture" in default_config
            assert "training" in default_config
            assert "graph_construction" in default_config

    @patch('scripts.train_gat_pipeline.pd.read_parquet')
    def test_load_data(self, mock_read_parquet, mock_config_file, sample_returns_data):
        """Test data loading functionality."""
        mock_read_parquet.return_value = sample_returns_data

        with patch('scripts.train_gat_pipeline.Path.mkdir'):
            pipeline = GATTrainingPipeline(mock_config_file)

            returns_data, universe_calendar = pipeline.load_data()

            assert isinstance(returns_data, pd.DataFrame)
            assert isinstance(universe_calendar, pd.DataFrame)
            assert len(returns_data.columns) == 100
            assert len(universe_calendar) > 0
            mock_read_parquet.assert_called_once()

    def test_prepare_graph_configurations(self, mock_config_file):
        """Test graph configuration preparation."""
        with patch('scripts.train_gat_pipeline.Path.mkdir'):
            pipeline = GATTrainingPipeline(mock_config_file)
            configs = pipeline.prepare_graph_configurations()

            assert len(configs) == 5  # mst, tmfg, knn_k5, knn_k10, knn_k15
            assert all("config_name" in config for config in configs)
            assert all("filter_method" in config for config in configs)

            # Check that knn configs have k values
            knn_configs = [c for c in configs if c["filter_method"] == "knn"]
            assert len(knn_configs) == 3
            assert all("knn_k" in config for config in knn_configs)

    def test_create_gat_model_config(self, mock_config_file):
        """Test GAT model configuration creation."""
        with patch('scripts.train_gat_pipeline.Path.mkdir'):
            pipeline = GATTrainingPipeline(mock_config_file)

            graph_config = {
                'filter_method': 'mst',
                'knn_k': None,
                'lookback_days': 252,
                'use_edge_attr': True,
                'config_name': 'mst'
            }

            model_config = pipeline.create_gat_model_config(graph_config)

            assert model_config is not None
            assert model_config.hidden_dim == 64
            assert model_config.num_layers == 3
            assert model_config.learning_rate == 0.001

    def test_prepare_features(self, mock_config_file, sample_returns_data):
        """Test feature preparation from returns data."""
        with patch('scripts.train_gat_pipeline.Path.mkdir'):
            pipeline = GATTrainingPipeline(mock_config_file)

            universe = sample_returns_data.columns[:20].tolist()
            returns_subset = sample_returns_data[universe].iloc[:500]  # Use subset for speed

            features = pipeline._prepare_features(returns_subset, universe)

            assert isinstance(features, np.ndarray)
            assert features.shape[0] == len(universe)
            assert features.shape[1] == 10  # Expected number of features
            assert not np.isnan(features).any()

    @patch('scripts.train_gat_pipeline.build_period_graph')
    def test_prepare_training_data(self, mock_build_graph, mock_config_file, sample_returns_data):
        """Test training data preparation with graph construction."""
        # Mock graph data
        mock_graph = Mock()
        mock_graph.x = torch.randn(50, 10)
        mock_graph.edge_index = torch.randint(0, 50, (2, 100))
        mock_graph.edge_attr = torch.randn(100, 3)
        mock_build_graph.return_value = mock_graph

        # Create universe calendar that matches the method's expected date range
        # The method looks for dates between 2016+lookback and 2022
        start_date = pd.Timestamp('2016-01-01') + pd.Timedelta(days=252)  # After lookback
        dates = pd.date_range(start_date, '2022-12-31', freq='MS')[:12]
        universe_calendar = pd.DataFrame({
            'date': dates,
            'tickers': [sample_returns_data.columns[:50].tolist()] * len(dates)
        })

        graph_config = {
            'filter_method': 'mst',
            'lookback_days': 252,
            'use_edge_attr': True,
            'config_name': 'mst'
        }

        with patch('scripts.train_gat_pipeline.Path.mkdir'):
            pipeline = GATTrainingPipeline(mock_config_file)

            training_samples = pipeline.prepare_training_data(
                sample_returns_data, universe_calendar, graph_config
            )

            assert len(training_samples) > 0
            assert all(len(sample) == 3 for sample in training_samples)  # (graph, returns, date)
            # Should be limited to 12 samples for memory efficiency
            assert len(training_samples) <= 12

    def test_train_gat_model_basic_validation(self, mock_config_file):
        """Test basic GAT model training parameter validation."""
        with patch('scripts.train_gat_pipeline.Path.mkdir'):
            pipeline = GATTrainingPipeline(mock_config_file)

            # Test that the method exists and accepts the expected parameters
            assert hasattr(pipeline, 'train_gat_model')

            # Test model config creation works
            graph_config = {
                'filter_method': 'mst',
                'lookback_days': 252,
                'use_edge_attr': True,
                'config_name': 'test'
            }
            model_config = pipeline.create_gat_model_config(graph_config)
            assert model_config is not None

    @patch('scripts.train_gat_pipeline.GATTrainingPipeline.load_data')
    @patch('scripts.train_gat_pipeline.GATTrainingPipeline.prepare_graph_configurations')
    @patch('scripts.train_gat_pipeline.GATTrainingPipeline.create_gat_model_config')
    @patch('scripts.train_gat_pipeline.GATTrainingPipeline.prepare_training_data')
    @patch('scripts.train_gat_pipeline.GATTrainingPipeline.train_gat_model')
    @patch('scripts.train_gat_pipeline.GATTrainingPipeline._save_training_results')
    def test_execute_full_training_pipeline(self, mock_save, mock_train, mock_prepare_data,
                                          mock_create_config, mock_graph_configs,
                                          mock_load, mock_config_file, sample_returns_data):
        """Test full training pipeline execution."""
        # Mock all components
        universe_calendar = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=5, freq='MS'),
            'tickers': [sample_returns_data.columns[:20].tolist()] * 5
        })
        mock_load.return_value = (sample_returns_data, universe_calendar)

        mock_graph_configs.return_value = [
            {'config_name': 'mst', 'filter_method': 'mst'},
            {'config_name': 'knn_k5', 'filter_method': 'knn', 'knn_k': 5}
        ]

        mock_create_config.return_value = Mock()
        mock_prepare_data.return_value = [Mock(), Mock()]  # Mock training samples
        mock_train.return_value = {'best_loss': 0.5, 'training_history': [0.8, 0.6, 0.5]}

        with patch('scripts.train_gat_pipeline.Path.mkdir'):
            pipeline = GATTrainingPipeline(mock_config_file)
            results = pipeline.execute_full_training_pipeline()

            mock_load.assert_called_once()
            mock_graph_configs.assert_called_once()
            assert len(results) == 2  # Two configurations trained
            mock_save.assert_called_once()

    @patch('builtins.open', new_callable=MagicMock)
    @patch('scripts.train_gat_pipeline.yaml.dump')
    def test_save_training_results(self, mock_yaml_dump, mock_open, mock_config_file):
        """Test training results saving."""
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        with patch('scripts.train_gat_pipeline.Path.mkdir'):
            pipeline = GATTrainingPipeline(mock_config_file)
            pipeline.training_results = {"test": "data"}

            pipeline._save_training_results()

            mock_yaml_dump.assert_called_once_with({"test": "data"}, mock_file, default_flow_style=False)

    def test_memory_manager_integration(self, mock_config_file):
        """Test GPU memory manager integration."""
        with patch('scripts.train_gat_pipeline.Path.mkdir'):
            pipeline = GATTrainingPipeline(mock_config_file)

            assert pipeline.memory_manager is not None
            # Test that memory manager has correct VRAM limit
            assert hasattr(pipeline.memory_manager, 'get_memory_info')

    def test_error_handling_in_training_data_preparation(self, mock_config_file, sample_returns_data):
        """Test error handling in training data preparation."""
        # Create universe calendar with invalid data
        universe_calendar = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=2, freq='MS'),
            'tickers': [[], ['INVALID_TICKER']]  # Empty and invalid tickers
        })

        graph_config = {
            'filter_method': 'mst',
            'lookback_days': 252,
            'config_name': 'test'
        }

        with patch('scripts.train_gat_pipeline.Path.mkdir'):
            pipeline = GATTrainingPipeline(mock_config_file)

            # Should handle errors gracefully and return fewer samples
            training_samples = pipeline.prepare_training_data(
                sample_returns_data, universe_calendar, graph_config
            )

            # Should not crash and may return empty list or partial results
            assert isinstance(training_samples, list)

    def test_graph_config_parameter_validation(self, mock_config_file):
        """Test graph configuration parameter validation."""
        with patch('scripts.train_gat_pipeline.Path.mkdir'):
            pipeline = GATTrainingPipeline(mock_config_file)

            # Test with knn method that has None knn_k (should be handled)
            graph_config = {
                'filter_method': 'knn',
                'knn_k': None,  # This should be handled
                'lookback_days': 252,
                'use_edge_attr': True,
                'config_name': 'knn_test'
            }

            model_config = pipeline.create_gat_model_config(graph_config)

            # Should set a default value for knn_k
            assert model_config.graph_config.knn_k is not None
            assert model_config.graph_config.knn_k > 0

    def test_feature_calculation_edge_cases(self, mock_config_file):
        """Test feature calculation with edge cases."""
        # Create data with edge cases
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        tickers = ["ASSET_001", "ASSET_002", "ASSET_003"]

        # Create data with some extreme values
        returns_data = pd.DataFrame(np.random.randn(100, 3), index=dates, columns=tickers)
        returns_data.iloc[50, 0] = 0.5  # Large positive return
        returns_data.iloc[51, 1] = -0.5  # Large negative return
        returns_data.iloc[:20, 2] = 0.0  # Constant returns

        with patch('scripts.train_gat_pipeline.Path.mkdir'):
            pipeline = GATTrainingPipeline(mock_config_file)

            features = pipeline._prepare_features(returns_data, tickers)

            assert isinstance(features, np.ndarray)
            assert features.shape[0] == len(tickers)
            assert not np.isnan(features).any()
            assert not np.isinf(features).any()

    def test_directory_setup(self, mock_config_file):
        """Test proper directory setup."""
        with patch('scripts.train_gat_pipeline.Path.mkdir') as mock_mkdir:
            pipeline = GATTrainingPipeline(mock_config_file)

            # Should create checkpoint and results directories
            assert hasattr(pipeline, 'checkpoint_dir')
            assert hasattr(pipeline, 'results_dir')
            # mkdir should be called for directory creation
            assert mock_mkdir.call_count >= 2
