"""Model-specific configuration classes.

This module provides configuration classes for different model types including
HRP (Hierarchical Risk Parity), LSTM, and GAT (Graph Attention Network) models.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from .base import ModelConfig


@dataclass
class HRPConfig(ModelConfig):
    """Hierarchical Risk Parity model configuration.

    Attributes:
        linkage_method: Clustering linkage method ('ward', 'complete', 'average')
        distance_metric: Distance metric for clustering ('euclidean', 'correlation')
        min_weight: Minimum weight constraint per asset
        max_weight: Maximum weight constraint per asset
        rebalance_frequency: Rebalancing frequency in days
    """
    linkage_method: str = "ward"
    distance_metric: str = "euclidean"
    min_weight: float = 0.01
    max_weight: float = 0.1
    rebalance_frequency: int = 21  # Monthly rebalancing


@dataclass
class LSTMConfig(ModelConfig):
    """LSTM model configuration for time series prediction.

    Attributes:
        hidden_size: LSTM hidden layer size
        num_layers: Number of LSTM layers
        dropout: Dropout rate for regularization
        sequence_length: Input sequence length
        prediction_horizon: Number of periods to predict
        features: List of feature names to use
    """
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    sequence_length: int = 60
    prediction_horizon: int = 1
    features: list[str] = field(default_factory=lambda: [
        "returns", "volume", "volatility", "momentum"
    ])


@dataclass
class GATConfig(ModelConfig):
    """Graph Attention Network model configuration.

    Attributes:
        hidden_dim: Hidden layer dimension
        num_heads: Number of attention heads
        num_layers: Number of GAT layers
        dropout: Dropout rate
        attention_dropout: Attention-specific dropout rate
        edge_dim: Edge feature dimension
        node_features: List of node feature names
        edge_features: List of edge feature names
        aggregation: Aggregation method ('mean', 'sum', 'max')
    """
    hidden_dim: int = 64
    num_heads: int = 8
    num_layers: int = 3
    dropout: float = 0.1
    attention_dropout: float = 0.1
    edge_dim: int = 32
    node_features: list[str] = field(default_factory=lambda: [
        "returns", "volatility", "volume", "market_cap", "momentum"
    ])
    edge_features: list[str] = field(default_factory=lambda: [
        "correlation", "covariance", "sector_similarity"
    ])
    aggregation: str = "mean"


def get_model_config(model_type: str, config_dict: Optional[dict[str, Any]] = None) -> ModelConfig:
    """Factory function to create model configuration objects.

    Args:
        model_type: Type of model ('hrp', 'lstm', 'gat')
        config_dict: Optional dictionary to override default values

    Returns:
        Configured model configuration object

    Raises:
        ValueError: If model_type is not supported

    Example:
        >>> config = get_model_config('gat', {'hidden_dim': 128})
        >>> config.hidden_dim
        128
    """
    model_configs = {
        'hrp': HRPConfig,
        'lstm': LSTMConfig,
        'gat': GATConfig
    }

    if model_type not in model_configs:
        raise ValueError(f"Unsupported model type: {model_type}. "
                        f"Supported types: {list(model_configs.keys())}")

    config_class = model_configs[model_type]

    if config_dict is None:
        return config_class()

    # Create config with overrides
    return config_class(**{
        k: v for k, v in config_dict.items()
        if k in config_class.__dataclass_fields__
    })


def validate_model_config(config: ModelConfig, model_type: str) -> bool:
    """Validate model-specific configuration parameters.

    Args:
        config: Model configuration object
        model_type: Type of model being validated

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If configuration parameters are invalid
    """
    # Common validation
    if config.learning_rate <= 0:
        raise ValueError("Learning rate must be positive")

    if config.batch_size <= 0:
        raise ValueError("Batch size must be positive")

    if config.max_epochs <= 0:
        raise ValueError("Max epochs must be positive")

    # Model-specific validation
    if model_type == 'gat' and isinstance(config, GATConfig):
        if config.hidden_dim <= 0:
            raise ValueError("Hidden dimension must be positive")

        if config.num_heads <= 0:
            raise ValueError("Number of attention heads must be positive")

        if not 0 <= config.dropout <= 1:
            raise ValueError("Dropout must be between 0 and 1")

    elif model_type == 'lstm' and isinstance(config, LSTMConfig):
        if config.hidden_size <= 0:
            raise ValueError("Hidden size must be positive")

        if config.sequence_length <= 0:
            raise ValueError("Sequence length must be positive")

    elif model_type == 'hrp' and isinstance(config, HRPConfig):
        if not 0 <= config.min_weight <= config.max_weight <= 1:
            raise ValueError("Weight constraints must satisfy 0 <= min_weight <= max_weight <= 1")

        if config.rebalance_frequency <= 0:
            raise ValueError("Rebalance frequency must be positive")

    return True
