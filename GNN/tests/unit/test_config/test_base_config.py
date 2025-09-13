"""Tests for base configuration module."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest
import yaml

from src.config.base import (
    DataConfig,
    ModelConfig,
    ProjectConfig,
    load_config,
    save_config,
    validate_config,
    validate_config_comprehensive,
)


class TestProjectConfig:
    """Test ProjectConfig dataclass and methods."""

    def test_default_initialization(self):
        """Test default ProjectConfig initialization."""
        config = ProjectConfig()

        assert config.data_dir == "data"
        assert config.output_dir == "outputs"
        assert config.log_level == "INFO"
        assert config.gpu_memory_fraction == 0.9

    def test_custom_initialization(self):
        """Test ProjectConfig with custom parameters."""
        config = ProjectConfig(
            data_dir="custom_data",
            output_dir="custom_outputs",
            log_level="DEBUG",
            gpu_memory_fraction=0.7
        )

        assert config.data_dir == "custom_data"
        assert config.output_dir == "custom_outputs"
        assert config.log_level == "DEBUG"
        assert config.gpu_memory_fraction == 0.7


class TestDataConfig:
    """Test DataConfig dataclass and methods."""

    def test_default_initialization(self):
        """Test default DataConfig initialization."""
        config = DataConfig()

        assert config.universe == "midcap400"
        assert config.start_date == "2016-01-01"
        assert config.end_date == "2024-12-31"
        assert config.sources == ["stooq", "yfinance"]

    def test_custom_initialization(self):
        """Test DataConfig with custom parameters."""
        sources = ["stooq", "yfinance"]
        config = DataConfig(
            universe="sp500",
            start_date="2020-01-01",
            end_date="2023-12-31",
            sources=sources
        )

        assert config.universe == "sp500"
        assert config.start_date == "2020-01-01"
        assert config.end_date == "2023-12-31"
        assert config.sources == sources


class TestModelConfig:
    """Test ModelConfig dataclass and methods."""

    def test_default_initialization(self):
        """Test default ModelConfig initialization."""
        config = ModelConfig()

        assert config.batch_size == 32
        assert config.learning_rate == 0.001
        assert config.max_epochs == 100

    def test_custom_initialization(self):
        """Test ModelConfig with custom parameters."""
        config = ModelConfig(
            batch_size=64,
            learning_rate=0.01,
            max_epochs=50
        )

        assert config.batch_size == 64
        assert config.learning_rate == 0.01
        assert config.max_epochs == 50


class TestConfigIO:
    """Test configuration loading and saving functions."""

    def test_save_and_load_config(self):
        """Test saving and loading configuration files."""
        config_data = {
            "project": {
                "data_dir": "test_data",
                "output_dir": "test_outputs",
                "log_level": "DEBUG"
            },
            "data": {
                "universe": "test_universe",
                "start_date": "2020-01-01",
                "end_date": "2022-12-31"
            }
        }

        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"

            # Save configuration
            save_config(config_data, config_path)

            # Verify file exists
            assert config_path.exists()

            # Load configuration
            loaded_config = load_config(config_path)

            # Verify loaded data matches saved data
            assert loaded_config == config_data

    def test_load_nonexistent_config(self):
        """Test loading a non-existent configuration file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")

    def test_load_invalid_yaml(self):
        """Test loading invalid YAML configuration."""
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "invalid.yaml"
            config_path.write_text("invalid: yaml: content: [")

            with pytest.raises(yaml.YAMLError):
                load_config(config_path)


class TestConfigValidation:
    """Test configuration validation functions."""

    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config_data = {
            "project": {
                "data_dir": "data",
                "output_dir": "outputs",
                "log_level": "INFO"
            },
            "data": {
                "universe": "midcap400",
                "start_date": "2016-01-01",
                "end_date": "2024-12-31"
            }
        }

        # Should not raise any exceptions
        result = validate_config_comprehensive(config_data)
        assert result is True

    def test_validate_config_missing_section(self):
        """Test validation with missing required section."""
        config_data = {
            "project": {
                "data_dir": "data",
                "output_dir": "outputs"
            }
            # Missing "data" section
        }

        with pytest.raises(ValueError, match="Missing required section"):
            validate_config_comprehensive(config_data)

    def test_validate_config_invalid_log_level(self):
        """Test validation with invalid log level."""
        config_data = {
            "project": {
                "data_dir": "data",
                "output_dir": "outputs",
                "log_level": "INVALID_LEVEL"
            },
            "data": {
                "universe": "midcap400",
                "start_date": "2016-01-01",
                "end_date": "2024-12-31"
            }
        }

        with pytest.raises(ValueError, match="Invalid log level"):
            validate_config_comprehensive(config_data)

    def test_validate_config_invalid_date_format(self):
        """Test validation with invalid date format."""
        config_data = {
            "project": {
                "data_dir": "data",
                "output_dir": "outputs",
                "log_level": "INFO"
            },
            "data": {
                "universe": "midcap400",
                "start_date": "invalid-date",
                "end_date": "2024-12-31"
            }
        }

        with pytest.raises(ValueError, match="Invalid date format"):
            validate_config_comprehensive(config_data)

    def test_validate_config_end_before_start(self):
        """Test validation with end_date before start_date."""
        config_data = {
            "project": {
                "data_dir": "data",
                "output_dir": "outputs",
                "log_level": "INFO"
            },
            "data": {
                "universe": "midcap400",
                "start_date": "2024-01-01",
                "end_date": "2020-01-01"  # Before start_date
            }
        }

        with pytest.raises(ValueError, match="end_date must be after start_date"):
            validate_config_comprehensive(config_data)


@pytest.fixture
def sample_config_file():
    """Create a sample configuration file for testing."""
    config_data = {
        "project": {
            "data_dir": "data",
            "output_dir": "outputs",
            "log_level": "INFO",
            "gpu_memory_fraction": 0.9
        },
        "data": {
            "universe": "midcap400",
            "start_date": "2016-01-01",
            "end_date": "2024-12-31",
            "sources": ["stooq", "yfinance"]
        },
        "model": {
            "batch_size": 32,
            "learning_rate": 0.001,
            "max_epochs": 100
        }
    }

    with TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "sample_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        yield config_path


class TestIntegrationTests:
    """Integration tests for configuration system."""

    def test_full_config_workflow(self, sample_config_file):
        """Test complete configuration workflow."""
        # Load configuration
        config_data = load_config(sample_config_file)

        # Validate configuration
        assert validate_config_comprehensive(config_data) is True

        # Create configuration objects
        project_config = ProjectConfig(**config_data["project"])
        data_config = DataConfig(**config_data["data"])
        model_config = ModelConfig(**config_data["model"])

        # Verify configurations are correct
        assert project_config.data_dir == "data"
        assert data_config.universe == "midcap400"
        assert model_config.batch_size == 32

        # Test modification and save
        modified_config = config_data.copy()
        modified_config["model"]["batch_size"] = 64

        with TemporaryDirectory() as temp_dir:
            modified_path = Path(temp_dir) / "modified_config.yaml"
            save_config(modified_config, modified_path)

            # Verify modification was saved
            reloaded_config = load_config(modified_path)
            assert reloaded_config["model"]["batch_size"] == 64
