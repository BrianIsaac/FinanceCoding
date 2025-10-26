"""Shared configuration dataclasses for Hydra-based configuration management.

This module contains structured config dataclasses used by both:
1. Entry point scripts (via Hydra @hydra.main decorator)
2. Core modules (data collectors, processors, etc.)

These replace the old src/config/ directory which was archived in Phase 4.
Configuration values are now loaded via Hydra YAML configs and passed to modules.

Usage in scripts:
    @hydra.main(version_base=None, config_path="../configs/...", config_name="config")
    def main(cfg: DictConfig) -> None:
        collector_cfg = cfg.collectors.yfinance
        collector = YFinanceCollector(collector_cfg)

Usage in modules:
    from src.config_schemas import CollectorConfig

    class YFinanceCollector:
        def __init__(self, config: CollectorConfig):
            self.config = config
"""

from dataclasses import dataclass, field


@dataclass
class CollectorConfig:
    """Configuration for a data source collector.

    Attributes:
        source_name: Name of the data source (e.g., 'yfinance', 'polygon')
        rate_limit: Minimum seconds between API requests
        timeout: HTTP request timeout in seconds
        retry_attempts: Number of retry attempts for failed requests
        retry_delay: Delay between retry attempts in seconds
    """
    source_name: str
    rate_limit: float = 1.0
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class UniverseConfig:
    """Configuration for asset universe definition.

    Attributes:
        universe_type: Type of universe (e.g., 'midcap400', 'sp500')
        min_market_cap: Minimum market capitalisation filter (optional)
        min_avg_volume: Minimum average volume filter (optional)
        exclude_sectors: List of sectors to exclude from universe
    """
    universe_type: str = "midcap400"
    min_market_cap: float | None = None
    min_avg_volume: float | None = None
    exclude_sectors: list[str] = field(default_factory=list)


@dataclass
class ValidationConfig:
    """Configuration for data quality validation.

    Attributes:
        missing_data_threshold: Maximum allowed fraction of missing data
        price_change_threshold: Maximum allowed single-day price change
        volume_threshold: Minimum required trading volume
        validate_business_days: Whether to validate business day alignment
        fill_method: Method for filling missing data ('forward', 'interpolate')
        generate_reports: Whether to generate validation reports
        report_output_dir: Directory for validation reports
    """
    missing_data_threshold: float = 0.10
    price_change_threshold: float = 0.50
    volume_threshold: int = 1000
    validate_business_days: bool = True
    fill_method: str = "forward"
    generate_reports: bool = True
    report_output_dir: str = "logs/validation_reports"


def create_collector_config(source_name: str, **kwargs) -> CollectorConfig:
    """Create a CollectorConfig with default values for a given source.

    This helper function provides backward compatibility with old config system.

    Args:
        source_name: Name of the data source
        **kwargs: Override default config values

    Returns:
        CollectorConfig instance

    Example:
        >>> config = create_collector_config('yfinance', rate_limit=5.0)
        >>> collector = YFinanceCollector(config)
    """
    defaults = {
        'yfinance': {'rate_limit': 5.0, 'timeout': 10},
        'stooq': {'rate_limit': 10.0, 'timeout': 15},
        'tiingo': {'rate_limit': 72.0, 'timeout': 15},
        'polygon': {'rate_limit': 12.0, 'timeout': 15},
        'wikipedia': {'rate_limit': 1.0, 'timeout': 30},
    }

    config_dict = {'source_name': source_name}
    config_dict.update(defaults.get(source_name, {}))
    config_dict.update(kwargs)

    return CollectorConfig(**config_dict)
