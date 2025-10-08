"""
Unit tests for HRP training pipeline.

Tests the complete HRP training pipeline implementation to ensure
proper functionality and error handling.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.train_hrp_pipeline import HRPTrainingPipeline


class TestHRPTrainingPipeline:
    """Test HRP training pipeline functionality."""

    @pytest.fixture
    def mock_config_file(self):
        """Create mock configuration file."""
        config_data = {
            "constraints": {
                "long_only": True,
                "max_position_weight": 0.10,
                "top_k_positions": 50,
                "max_monthly_turnover": 0.20,
                "min_weight_threshold": 0.01
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
        dates = pd.date_range("2020-01-01", "2023-01-01", freq="D")
        tickers = [f"ASSET_{i:03d}" for i in range(50)]

        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.015, (len(dates), len(tickers)))

        return pd.DataFrame(returns, index=dates, columns=tickers)

    @pytest.fixture
    def mock_data_path(self, sample_returns_data):
        """Mock data path with sample data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)
            returns_path = data_path / "returns_daily_final.parquet"
            sample_returns_data.to_parquet(returns_path)
            yield data_path

    def test_initialization(self, mock_config_file):
        """Test pipeline initialization."""
        with patch('scripts.train_hrp_pipeline.Path.mkdir'):
            pipeline = HRPTrainingPipeline(mock_config_file)

            assert pipeline.config_path == mock_config_file
            assert pipeline.base_config is not None
            assert "constraints" in pipeline.base_config

    def test_config_loading(self, mock_config_file):
        """Test configuration loading."""
        with patch('scripts.train_hrp_pipeline.Path.mkdir'):
            pipeline = HRPTrainingPipeline(mock_config_file)
            config = pipeline._load_config()

            assert isinstance(config, dict)
            assert "constraints" in config
            assert config["constraints"]["long_only"] is True

    @patch('scripts.train_hrp_pipeline.pd.read_parquet')
    def test_load_data(self, mock_read_parquet, mock_config_file, sample_returns_data):
        """Test data loading functionality."""
        mock_read_parquet.return_value = sample_returns_data

        with patch('scripts.train_hrp_pipeline.Path.mkdir'):
            pipeline = HRPTrainingPipeline(mock_config_file)

            returns_data, universe_builder = pipeline.load_data()

            assert isinstance(returns_data, pd.DataFrame)
            assert len(returns_data.columns) == 50
            mock_read_parquet.assert_called_once()

    def test_generate_parameter_configurations(self, mock_config_file):
        """Test parameter configuration generation."""
        with patch('scripts.train_hrp_pipeline.Path.mkdir'):
            pipeline = HRPTrainingPipeline(mock_config_file)
            configs = pipeline._generate_parameter_configurations()

            assert len(configs) == 18  # 3 lookbacks * 3 linkages * 2 correlations
            assert all("config_name" in config for config in configs)
            assert all("lookback_days" in config for config in configs)
            assert all("linkage_method" in config for config in configs)

    @patch('scripts.train_hrp_pipeline.HRPModel')
    def test_train_single_configuration(self, mock_hrp_model, mock_config_file, sample_returns_data):
        """Test single configuration training."""
        # Mock model instance
        mock_model_instance = Mock()
        mock_model_instance.fit.return_value = None
        mock_model_instance.predict_weights.return_value = pd.Series([0.1, 0.2, 0.3, 0.4], name='weights')
        mock_model_instance.get_model_info.return_value = {"model_type": "HRP"}
        mock_hrp_model.return_value = mock_model_instance

        with patch('scripts.train_hrp_pipeline.Path.mkdir'):
            pipeline = HRPTrainingPipeline(mock_config_file)

            config = {
                "config_name": "test_config",
                "lookback_days": 252,
                "linkage_method": "single",
                "correlation_method": "pearson"
            }

            model, metrics = pipeline.train_single_configuration(sample_returns_data, config)

            assert mock_model_instance.fit.called
            assert metrics["fitted_successfully"] is True
            assert "weights_sum" in metrics

    @patch('scripts.train_hrp_pipeline.HRPClustering')
    def test_validate_clustering(self, mock_clustering, mock_config_file, sample_returns_data):
        """Test clustering validation."""
        # Mock clustering components
        mock_clustering_instance = Mock()
        mock_clustering_instance.build_correlation_distance.return_value = np.random.rand(10, 10)
        mock_clustering_instance.hierarchical_clustering.return_value = np.random.rand(9, 4)
        mock_clustering.return_value = mock_clustering_instance

        # Mock model with diagnostics
        mock_model = Mock()
        mock_model.get_clustering_diagnostics.return_value = {
            "n_assets": 10,
            "n_observations": 100,
            "linkage_method": "single"
        }

        with patch('scripts.train_hrp_pipeline.Path.mkdir'):
            pipeline = HRPTrainingPipeline(mock_config_file)

            config = {
                "config_name": "test_config",
                "lookback_days": 252,
                "linkage_method": "single",
                "correlation_method": "pearson"
            }

            validation_metrics = pipeline.validate_clustering(mock_model, sample_returns_data, config)

            assert "config_name" in validation_metrics
            assert validation_metrics["config_name"] == "test_config"

    def test_clustering_validation_error_handling(self, mock_config_file, sample_returns_data):
        """Test error handling in clustering validation."""
        # Mock model that returns error
        mock_model = Mock()
        mock_model.get_clustering_diagnostics.return_value = {"error": "Test error"}

        with patch('scripts.train_hrp_pipeline.Path.mkdir'):
            pipeline = HRPTrainingPipeline(mock_config_file)

            config = {"config_name": "test_config", "lookback_days": 252}
            validation_metrics = pipeline.validate_clustering(mock_model, sample_returns_data, config)

            assert "error" in validation_metrics
            assert validation_metrics["error"] == "Test error"

    @patch('scripts.train_hrp_pipeline.HRPTrainingPipeline.train_single_configuration')
    @patch('scripts.train_hrp_pipeline.HRPTrainingPipeline.validate_clustering')
    def test_run_parameter_validation(self, mock_validate, mock_train, mock_config_file, sample_returns_data):
        """Test parameter validation execution."""
        # Mock successful training and validation
        mock_train.return_value = (Mock(), {"fitted_successfully": True, "config_name": "test_config"})
        mock_validate.return_value = {"silhouette_score": 0.5, "config_name": "test_config"}

        with patch('scripts.train_hrp_pipeline.Path.mkdir'):
            with patch('scripts.train_hrp_pipeline.HRPTrainingPipeline._generate_parameter_configurations') as mock_configs:
                mock_configs.return_value = [
                    {"config_name": "test_config", "lookback_days": 252, "linkage_method": "single"}
                ]

                pipeline = HRPTrainingPipeline(mock_config_file)
                results = pipeline.run_parameter_validation(sample_returns_data)

                assert "training_results" in results
                assert "clustering_diagnostics" in results
                assert "best_configuration" in results
                assert results["best_configuration"] == "test_config"

    def test_parameter_comparison_generation(self, mock_config_file):
        """Test parameter comparison analysis."""
        with patch('scripts.train_hrp_pipeline.Path.mkdir'):
            pipeline = HRPTrainingPipeline(mock_config_file)

            # Mock validation results
            validation_results = {
                "training_results": {
                    "hrp_lb252_single_pearson": {"fitted_successfully": True},
                    "hrp_lb504_average_spearman": {"fitted_successfully": True}
                },
                "clustering_diagnostics": {
                    "hrp_lb252_single_pearson": {"silhouette_score": 0.1},
                    "hrp_lb504_average_spearman": {"silhouette_score": 0.2}
                }
            }

            comparison = pipeline._generate_parameter_comparison(validation_results)

            assert "lookback_analysis" in comparison
            assert "linkage_analysis" in comparison
            assert "total_configs_tested" in comparison

    @patch('builtins.open', new_callable=MagicMock)
    @patch('scripts.train_hrp_pipeline.yaml.dump')
    @patch('scripts.train_hrp_pipeline.Path.mkdir')
    def test_save_results(self, mock_mkdir, mock_yaml_dump, mock_open, mock_config_file):
        """Test results saving functionality."""
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        pipeline = HRPTrainingPipeline(mock_config_file)
        test_results = {"test": "data"}

        pipeline.save_results(test_results)

        mock_yaml_dump.assert_called_once_with(test_results, mock_file, default_flow_style=False)

    @patch('scripts.train_hrp_pipeline.HRPTrainingPipeline.load_data')
    @patch('scripts.train_hrp_pipeline.HRPTrainingPipeline.run_parameter_validation')
    @patch('scripts.train_hrp_pipeline.HRPTrainingPipeline.save_results')
    def test_run_full_pipeline(self, mock_save, mock_validate, mock_load, mock_config_file, sample_returns_data):
        """Test full pipeline execution."""
        # Mock all components
        mock_load.return_value = (sample_returns_data, None)
        mock_validate.return_value = {"test": "results"}

        with patch('scripts.train_hrp_pipeline.Path.mkdir'):
            pipeline = HRPTrainingPipeline(mock_config_file)
            results = pipeline.run_full_pipeline()

            mock_load.assert_called_once()
            mock_validate.assert_called_once_with(sample_returns_data)
            mock_save.assert_called_once()
            assert results == {"test": "results"}

    def test_error_handling_invalid_config(self):
        """Test error handling for invalid configuration."""
        with pytest.raises(FileNotFoundError):
            HRPTrainingPipeline("nonexistent_config.yaml")

    @patch('scripts.train_hrp_pipeline.logging.basicConfig')
    def test_logging_setup(self, mock_logging, mock_config_file):
        """Test logging configuration."""
        with patch('scripts.train_hrp_pipeline.Path.mkdir'):
            pipeline = HRPTrainingPipeline(mock_config_file)

            # Verify logger was created
            assert hasattr(pipeline, 'logger')
            mock_logging.assert_called_once()
