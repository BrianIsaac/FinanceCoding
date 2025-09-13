"""Base configuration classes for portfolio optimization ML framework.

This module provides core configuration classes and utilities for managing
project-wide settings, data configurations, and model parameters.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml


@dataclass
class ProjectConfig:
    """Core project configuration settings.

    Attributes:
        data_dir: Directory path for data storage
        output_dir: Directory path for output files and results
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        gpu_memory_fraction: Fraction of GPU memory to allocate (0.0-1.0)
    """

    data_dir: str = "data"
    output_dir: str = "outputs"
    log_level: str = "INFO"
    gpu_memory_fraction: float = 0.9


@dataclass
class DataConfig:
    """Data pipeline configuration settings.

    Attributes:
        universe: Asset universe identifier (e.g., 'midcap400', 'sp500')
        start_date: Start date for data in YYYY-MM-DD format
        end_date: End date for data in YYYY-MM-DD format
        sources: List of data sources to use
        lookback_window: Number of periods for rolling calculations
        min_history: Minimum number of periods required per asset
    """

    universe: str = "midcap400"
    start_date: str = "2016-01-01"
    end_date: str = "2024-12-31"
    sources: Optional[list] = field(default_factory=lambda: ["stooq", "yfinance"])
    lookback_window: int = 252
    min_history: int = 63


@dataclass
class ModelConfig:
    """Base model configuration settings.

    Attributes:
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        max_epochs: Maximum number of training epochs
        early_stopping_patience: Epochs to wait before early stopping
        validation_split: Fraction of data to use for validation
    """

    batch_size: int = 32
    learning_rate: float = 0.001
    max_epochs: int = 100
    early_stopping_patience: int = 10
    validation_split: float = 0.2


def load_config(config_path: Union[str, Path]) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing configuration parameters

    Raises:
        FileNotFoundError: If config_path doesn't exist
        yaml.YAMLError: If YAML file is malformed

    Example:
        >>> config = load_config('configs/models/gat_default.yaml')
        >>> print(config['hidden_dim'])
        64
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path) as file:
            config = yaml.safe_load(file)
            return config if config is not None else {}
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}") from e


def validate_config(config: dict[str, Any], required_keys: list) -> bool:
    """Validate configuration contains required keys.

    Args:
        config: Configuration dictionary to validate
        required_keys: List of required configuration keys

    Returns:
        True if all required keys are present

    Raises:
        ValueError: If required keys are missing

    Example:
        >>> config = {'model_type': 'gat', 'hidden_dim': 64}
        >>> validate_config(config, ['model_type', 'hidden_dim'])
        True
    """
    missing_keys = [key for key in required_keys if key not in config]

    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")

    return True


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple configuration dictionaries.

    Later dictionaries take precedence over earlier ones for conflicting keys.

    Args:
        *configs: Variable number of configuration dictionaries

    Returns:
        Merged configuration dictionary

    Example:
        >>> base = {'learning_rate': 0.001, 'batch_size': 32}
        >>> override = {'learning_rate': 0.01}
        >>> merged = merge_configs(base, override)
        >>> merged['learning_rate']
        0.01
    """
    merged = {}

    for config in configs:
        if isinstance(config, dict):
            merged.update(config)

    return merged


def setup_logging(config: ProjectConfig) -> logging.Logger:
    """Setup logging configuration.

    Args:
        config: Project configuration containing log_level

    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    return logging.getLogger(__name__)


def save_config(config: dict[str, Any], config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary to save
        config_path: Path where to save the YAML file

    Example:
        >>> config = {'model_type': 'gat', 'hidden_dim': 64}
        >>> save_config(config, 'configs/custom_config.yaml')
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, indent=2)


def validate_config_comprehensive(config: dict[str, Any]) -> bool:
    """Comprehensive configuration validation with specific rules.

    Args:
        config: Configuration dictionary to validate

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If configuration is invalid
    """
    # Check required sections
    required_sections = ['project', 'data']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")

    # Validate project section
    if 'project' in config:
        project = config['project']
        if 'log_level' in project:
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
            if project['log_level'] not in valid_levels:
                raise ValueError(f"Invalid log level: {project['log_level']}. Must be one of {valid_levels}")

    # Validate data section
    if 'data' in config:
        data = config['data']
        if 'start_date' in data and 'end_date' in data:
            try:
                from datetime import datetime
                start = datetime.strptime(data['start_date'], '%Y-%m-%d')
                end = datetime.strptime(data['end_date'], '%Y-%m-%d')
                if end <= start:
                    raise ValueError("end_date must be after start_date")
            except ValueError as e:
                if "time data" in str(e):
                    raise ValueError("Invalid date format. Use YYYY-MM-DD")
                raise

    return True
