"""Data pipeline configuration classes.

This module provides configuration classes for data collection, processing,
and preparation pipelines used in the portfolio optimization framework.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from .base import DataConfig


@dataclass
class DataPipelineConfig(DataConfig):
    """Extended data pipeline configuration with processing parameters.

    Attributes:
        processing_frequency: How often to process data ('daily', 'weekly', 'monthly')
        cache_enabled: Whether to enable data caching
        cache_dir: Directory for cached data
        parallel_processing: Enable parallel data processing
        num_workers: Number of worker processes for parallel processing
    """
    processing_frequency: str = "daily"
    cache_enabled: bool = True
    cache_dir: str = "data/cache"
    parallel_processing: bool = True
    num_workers: int = 4


@dataclass
class CollectorConfig:
    """Data collector configuration for external data sources.

    Attributes:
        source_name: Name of the data source ('stooq', 'yfinance', 'wikipedia')
        rate_limit: Rate limit for API calls (requests per second)
        timeout: Request timeout in seconds
        retry_attempts: Number of retry attempts for failed requests
        retry_delay: Delay between retry attempts in seconds
    """
    source_name: str
    rate_limit: float = 1.0
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class FeatureConfig:
    """Feature engineering configuration.

    Attributes:
        return_periods: List of periods for return calculations
        volatility_window: Window size for volatility calculations
        momentum_periods: List of periods for momentum features
        technical_indicators: List of technical indicators to compute
        correlation_window: Window size for correlation calculations
        outlier_threshold: Z-score threshold for outlier detection
    """
    return_periods: list[int] = field(default_factory=lambda: [1, 5, 21, 63])
    volatility_window: int = 21
    momentum_periods: list[int] = field(default_factory=lambda: [21, 63, 252])
    technical_indicators: list[str] = field(default_factory=lambda: [
        "rsi", "macd", "bollinger_bands", "moving_average"
    ])
    correlation_window: int = 252
    outlier_threshold: float = 3.0


@dataclass
class UniverseConfig:
    """Asset universe configuration.

    Attributes:
        universe_type: Type of universe ('midcap400', 'sp500', 'custom')
        custom_symbols: List of custom symbols if using custom universe
        min_market_cap: Minimum market cap filter (in millions)
        min_avg_volume: Minimum average daily volume filter
        exclude_sectors: List of sectors to exclude
        rebalance_frequency: How often to rebalance universe membership
    """
    universe_type: str = "midcap400"
    custom_symbols: Optional[list[str]] = None
    min_market_cap: Optional[float] = None
    min_avg_volume: Optional[float] = None
    exclude_sectors: Optional[list[str]] = None
    rebalance_frequency: str = "quarterly"


@dataclass
class ValidationConfig:
    """Data validation configuration.

    Attributes:
        missing_data_threshold: Maximum allowed percentage of missing data
        price_change_threshold: Maximum allowed daily price change percentage
        volume_threshold: Minimum volume threshold
        validate_business_days: Whether to validate business day alignment
        fill_method: Method for filling missing data ('forward', 'backward', 'interpolate')
    """
    missing_data_threshold: float = 0.1  # 10%
    price_change_threshold: float = 0.5   # 50%
    volume_threshold: int = 1000
    validate_business_days: bool = True
    fill_method: str = "forward"


def create_collector_config(source: str, **kwargs: Any) -> CollectorConfig:
    """Create a collector configuration for a specific data source.

    Args:
        source: Data source name
        **kwargs: Additional configuration parameters

    Returns:
        Configured CollectorConfig object

    Example:
        >>> config = create_collector_config('yfinance', rate_limit=5.0)
        >>> config.rate_limit
        5.0
    """
    # Default configurations for different sources
    source_defaults = {
        'yfinance': {'rate_limit': 5.0, 'timeout': 10},
        'stooq': {'rate_limit': 10.0, 'timeout': 15},
        'wikipedia': {'rate_limit': 1.0, 'timeout': 30}
    }

    # Merge defaults with provided kwargs
    config_params = source_defaults.get(source, {})
    config_params.update(kwargs)
    config_params['source_name'] = source

    return CollectorConfig(**config_params)


def validate_date_range(start_date: str, end_date: str) -> bool:
    """Validate date range parameters.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        True if date range is valid

    Raises:
        ValueError: If date format is invalid or end_date < start_date
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        if end <= start:
            raise ValueError("End date must be after start date")

        return True

    except ValueError as e:
        if "time data" in str(e):
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD format: {e}") from e
        raise


def get_universe_symbols(universe_config: UniverseConfig) -> list[str]:
    """Get symbols for the specified universe configuration.

    Args:
        universe_config: Universe configuration object

    Returns:
        List of symbol strings

    Raises:
        NotImplementedError: For non-custom universe types (requires data source integration)
    """
    if universe_config.universe_type == "custom":
        return universe_config.custom_symbols or []

    # For other universe types, this would integrate with data collectors
    # to fetch current universe membership
    raise NotImplementedError(
        f"Universe type '{universe_config.universe_type}' requires "
        "integration with data collectors. Use custom universe for now."
    )
