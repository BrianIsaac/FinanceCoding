"""Unit tests for multi-source data pipeline components.

Tests for gap filling, data normalization, parquet management, and quality validation.
"""

import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config.data import CollectorConfig, ValidationConfig
from src.data.collectors.stooq import StooqCollector
from src.data.collectors.yfinance import YFinanceCollector
from src.data.loaders.parquet_manager import ParquetManager
from src.data.processors.data_normalization import DataNormalizer
from src.data.processors.data_quality_validator import DataQualityValidator
from src.data.processors.gap_filling import GapFiller


@pytest.fixture
def sample_price_data():
    """Create sample price data with gaps for testing."""
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]

    # Create base price data
    np.random.seed(42)
    prices = pd.DataFrame(
        index=dates,
        columns=tickers,
        data=np.random.randn(len(dates), len(tickers)).cumsum(axis=0) + 100,
    )

    # Add some gaps
    prices.loc["2023-03-15":"2023-03-20", "AAPL"] = np.nan
    prices.loc["2023-06-10":"2023-06-12", "MSFT"] = np.nan
    prices.loc["2023-09-01", "GOOGL"] = np.nan

    return prices


@pytest.fixture
def sample_volume_data(sample_price_data):
    """Create sample volume data corresponding to price data."""
    np.random.seed(123)
    volume = pd.DataFrame(
        index=sample_price_data.index,
        columns=sample_price_data.columns,
        data=np.random.randint(1000, 100000, sample_price_data.shape),
    )

    # Add some zero volume days
    volume.loc["2023-07-04", :] = 0  # Holiday
    volume.loc["2023-12-25", :] = 0  # Holiday

    return volume


@pytest.fixture
def validation_config():
    """Create validation configuration for testing."""
    return ValidationConfig(
        missing_data_threshold=0.05,
        price_change_threshold=0.3,
        volume_threshold=500,
        min_data_points=50,
        generate_reports=False,
    )


@pytest.fixture
def temp_directory():
    """Create temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestGapFilling:
    """Test gap filling functionality."""

    def test_gap_filler_initialization(self, validation_config):
        """Test gap filler initialization."""
        filler = GapFiller(validation_config)
        assert filler.config == validation_config

    def test_forward_fill(self, sample_price_data, validation_config):
        """Test forward fill functionality."""
        filler = GapFiller(validation_config)

        # Test single ticker
        aapl_prices = sample_price_data["AAPL"]
        filled = filler.forward_fill(aapl_prices, limit=10)

        # Should fill gaps up to limit
        assert filled.isna().sum() < aapl_prices.isna().sum()

        # Check specific gap is filled
        gap_period = pd.date_range("2023-03-15", "2023-03-20")
        original_na_count = aapl_prices.loc[gap_period].isna().sum()
        filled_na_count = filled.loc[gap_period].isna().sum()
        assert filled_na_count < original_na_count

    def test_linear_interpolate(self, sample_price_data, validation_config):
        """Test linear interpolation."""
        filler = GapFiller(validation_config)

        # Test single gap interpolation
        googl_prices = sample_price_data["GOOGL"]
        filled = filler.linear_interpolate(googl_prices, max_gap_days=5)

        # Single day gap should be filled
        assert not filled.loc["2023-09-01"].isna()

        # Value should be interpolated between neighbors
        before_val = googl_prices.loc["2023-08-31"]
        after_val = googl_prices.loc["2023-09-02"]
        filled_val = filled.loc["2023-09-01"]

        # Should be between the two values
        assert min(before_val, after_val) <= filled_val <= max(before_val, after_val)

    def test_hybrid_fill(self, sample_price_data, sample_volume_data, validation_config):
        """Test hybrid gap filling strategy."""
        filler = GapFiller(validation_config)

        # Test with volume validation
        aapl_prices = sample_price_data["AAPL"]
        aapl_volume = sample_volume_data["AAPL"]

        filled = filler.hybrid_fill(
            aapl_prices, volume_series=aapl_volume, small_gap_days=2, medium_gap_days=5
        )

        # Should fill some gaps
        assert filled.isna().sum() < aapl_prices.isna().sum()

    def test_process_dataframe(self, sample_price_data, sample_volume_data, validation_config):
        """Test DataFrame processing."""
        filler = GapFiller(validation_config)

        original_na_count = sample_price_data.isna().sum().sum()

        filled_df = filler.process_dataframe(
            sample_price_data, volume_df=sample_volume_data, method="hybrid"
        )

        # Should reduce missing data
        filled_na_count = filled_df.isna().sum().sum()
        assert filled_na_count < original_na_count

        # Should maintain shape
        assert filled_df.shape == sample_price_data.shape

    def test_validate_fill_quality(self, sample_price_data, validation_config):
        """Test fill quality validation."""
        filler = GapFiller(validation_config)

        filled_df = filler.process_dataframe(sample_price_data, method="forward")
        quality_report = filler.validate_fill_quality(sample_price_data, filled_df)

        assert "fill_rate" in quality_report
        assert "cells_filled" in quality_report
        assert "ticker_quality" in quality_report
        assert quality_report["fill_rate"] >= 0.0


class TestDataNormalization:
    """Test data normalization functionality."""

    def test_data_normalizer_initialization(self, validation_config):
        """Test data normalizer initialization."""
        normalizer = DataNormalizer(validation_config)
        assert normalizer.config == validation_config

    def test_calculate_daily_returns(self, sample_price_data, validation_config):
        """Test daily returns calculation."""
        normalizer = DataNormalizer(validation_config)

        returns = normalizer.calculate_daily_returns(sample_price_data, method="simple")

        # Should have same shape as prices (minus first row)
        assert returns.shape == sample_price_data.shape
        assert returns.index.equals(sample_price_data.index)

        # First row should be NaN
        assert returns.iloc[0].isna().all()

        # Returns should be reasonable (between -1 and 1 for daily)
        valid_returns = returns.dropna()
        assert (valid_returns >= -1).all().all()
        assert (valid_returns <= 1).all().all()

    def test_normalize_volume(self, sample_volume_data, sample_price_data, validation_config):
        """Test volume normalization."""
        normalizer = DataNormalizer(validation_config)

        # Test different normalization methods
        methods = ["raw", "dollar_volume", "log_volume", "z_score"]

        for method in methods:
            normalized = normalizer.normalize_volume(
                sample_volume_data, sample_price_data, method=method
            )

            assert normalized.shape == sample_volume_data.shape

            if method == "raw":
                pd.testing.assert_frame_equal(normalized, sample_volume_data)
            elif method == "dollar_volume":
                # Should be volume * price
                expected = sample_volume_data * sample_price_data
                pd.testing.assert_frame_equal(normalized, expected)

    def test_calculate_data_quality_score(
        self, sample_price_data, sample_volume_data, validation_config
    ):
        """Test data quality scoring."""
        normalizer = DataNormalizer(validation_config)

        # Create returns data
        returns = sample_price_data.pct_change()

        data_dict = {"close": sample_price_data, "returns": returns, "volume": sample_volume_data}

        quality_score = normalizer.calculate_data_quality_score(data_dict, sample_volume_data)

        assert "overall_score" in quality_score
        assert "component_scores" in quality_score
        assert "ticker_scores" in quality_score

        # Score should be between 0 and 1
        assert 0 <= quality_score["overall_score"] <= 1


class TestDataQualityValidator:
    """Test data quality validation functionality."""

    def test_validator_initialization(self, validation_config):
        """Test validator initialization."""
        validator = DataQualityValidator(validation_config)
        assert validator.config == validation_config

    def test_validate_price_data(self, sample_price_data, validation_config):
        """Test price data validation."""
        validator = DataQualityValidator(validation_config)

        results = validator.validate_price_data(sample_price_data)

        assert "data_completeness" in results
        assert "quality_score" in results
        assert "price_range_validation" in results

        # Should detect some data completeness issues due to gaps
        assert results["data_completeness"] < 1.0

        # Quality score should be reasonable
        assert 0 <= results["quality_score"] <= 1

    def test_validate_returns_data(self, sample_price_data, validation_config):
        """Test returns data validation."""
        validator = DataQualityValidator(validation_config)

        returns = sample_price_data.pct_change()
        results = validator.validate_returns_data(returns)

        assert "statistical_properties" in results
        assert "return_distribution" in results
        assert "quality_score" in results

        # Should have statistical properties for each ticker
        for ticker in sample_price_data.columns:
            if ticker in results["statistical_properties"]:
                stats = results["statistical_properties"][ticker]
                assert "mean" in stats
                assert "std" in stats

    def test_validate_volume_data(self, sample_volume_data, sample_price_data, validation_config):
        """Test volume data validation."""
        validator = DataQualityValidator(validation_config)

        results = validator.validate_volume_data(sample_volume_data, sample_price_data)

        assert "zero_volume_analysis" in results
        assert "quality_score" in results

        # Should detect zero volume days (holidays)
        assert any(
            info["zero_volume_days"] > 0 for info in results["zero_volume_analysis"].values()
        )

    def test_validate_complete_dataset(
        self, sample_price_data, sample_volume_data, validation_config
    ):
        """Test complete dataset validation."""
        validation_config.generate_reports = False  # Disable for testing
        validator = DataQualityValidator(validation_config)

        returns = sample_price_data.pct_change()
        data_dict = {"close": sample_price_data, "returns": returns, "volume": sample_volume_data}

        results = validator.validate_complete_dataset(data_dict, generate_report=False)

        assert "overall_quality_score" in results
        assert "component_scores" in results
        assert "recommendations" in results

        # Should have validation for each data type
        expected_components = ["prices", "returns", "volume", "cross_validation"]
        for component in expected_components:
            assert component in results["component_scores"]


class TestParquetManager:
    """Test parquet storage management."""

    def test_parquet_manager_initialization(self, temp_directory, validation_config):
        """Test parquet manager initialization."""
        manager = ParquetManager(temp_directory, validation_config=validation_config)
        assert manager.base_path == Path(temp_directory)
        assert manager.compression == "snappy"

    def test_save_and_load_dataframe(self, sample_price_data, temp_directory):
        """Test saving and loading DataFrames."""
        manager = ParquetManager(temp_directory)

        # Save DataFrame
        saved_path = manager.save_dataframe(
            sample_price_data, data_type="prices", partition_strategy="none"
        )

        assert Path(saved_path).exists()

        # Load DataFrame
        loaded_df = manager.load_dataframe("prices")

        # Should be identical (allowing for minor floating point differences)
        pd.testing.assert_frame_equal(sample_price_data, loaded_df, check_dtype=False, rtol=1e-10)

    def test_monthly_partitioning(self, sample_price_data, temp_directory):
        """Test monthly partitioning strategy."""
        manager = ParquetManager(temp_directory)

        # Save with monthly partitioning
        manager.save_dataframe(sample_price_data, data_type="prices", partition_strategy="monthly")

        # Should create multiple files
        price_dir = Path(temp_directory) / "prices"
        parquet_files = list(price_dir.glob("*.parquet"))
        assert len(parquet_files) > 1  # Should have multiple monthly files

        # Load back
        loaded_df = manager.load_dataframe("prices")

        # Should reconstruct original data
        pd.testing.assert_frame_equal(sample_price_data, loaded_df, check_dtype=False, rtol=1e-10)

    def test_date_filtering(self, sample_price_data, temp_directory):
        """Test date filtering during load."""
        manager = ParquetManager(temp_directory)

        # Save data
        manager.save_dataframe(sample_price_data, data_type="prices", partition_strategy="none")

        # Load with date filter
        filtered_df = manager.load_dataframe(
            "prices", start_date="2023-06-01", end_date="2023-08-31"
        )

        # Should only include data in specified range
        assert filtered_df.index.min() >= pd.to_datetime("2023-06-01")
        assert filtered_df.index.max() <= pd.to_datetime("2023-08-31")

    def test_get_data_info(self, sample_price_data, temp_directory):
        """Test data information retrieval."""
        manager = ParquetManager(temp_directory)

        # Save data
        manager.save_dataframe(sample_price_data, data_type="prices", partition_strategy="monthly")

        # Get data info
        info = manager.get_data_info("prices")

        assert info["exists"] is True
        assert info["num_files"] > 0
        assert info["total_size_mb"] > 0
        assert info["partition_strategy"] == "monthly"

    def test_data_loader_interface(self, sample_price_data, sample_volume_data, temp_directory):
        """Test data loader interface."""
        manager = ParquetManager(temp_directory)

        # Save multiple data types
        manager.save_dataframe(sample_price_data, "close", partition_strategy="none")
        manager.save_dataframe(sample_volume_data, "volume", partition_strategy="none")

        # Create data loader
        loader = manager.create_data_loading_interface()

        # Load training data
        training_data = loader.load_training_data(
            data_types=["close", "volume"], start_date="2023-01-01", end_date="2023-12-31"
        )

        assert "close" in training_data
        assert "volume" in training_data
        assert not training_data["close"].empty
        assert not training_data["volume"].empty


class TestIntegration:
    """Test integration between components."""

    def test_end_to_end_pipeline(
        self, sample_price_data, sample_volume_data, temp_directory, validation_config
    ):
        """Test complete end-to-end pipeline."""
        # Initialize components
        gap_filler = GapFiller(validation_config)
        normalizer = DataNormalizer(validation_config)
        validator = DataQualityValidator(validation_config)
        parquet_manager = ParquetManager(temp_directory)

        # Step 1: Fill gaps
        filled_prices = gap_filler.process_dataframe(
            sample_price_data, volume_df=sample_volume_data, method="hybrid"
        )

        # Step 2: Calculate returns
        returns = normalizer.calculate_daily_returns(filled_prices)

        # Step 3: Normalize volume
        normalized_volume = normalizer.normalize_volume(
            sample_volume_data, filled_prices, method="dollar_volume"
        )

        # Step 4: Validate quality
        data_dict = {"close": filled_prices, "returns": returns, "volume": normalized_volume}

        validation_config.generate_reports = False
        quality_results = validator.validate_complete_dataset(data_dict, generate_report=False)

        # Step 5: Save to parquet
        for data_type, df in data_dict.items():
            parquet_manager.save_dataframe(df, data_type=data_type, partition_strategy="monthly")

        # Step 6: Verify saved data
        loader = parquet_manager.create_data_loading_interface()
        loaded_data = loader.load_training_data(
            data_types=["close", "returns", "volume"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        # Verify pipeline results
        assert quality_results["overall_quality_score"] > 0.0
        assert all(data_type in loaded_data for data_type in data_dict.keys())
        assert all(not df.empty for df in loaded_data.values())

        # Gap filling should have improved data completeness
        original_completeness = 1.0 - (
            sample_price_data.isna().sum().sum() / sample_price_data.size
        )
        filled_completeness = 1.0 - (filled_prices.isna().sum().sum() / filled_prices.size)
        assert filled_completeness >= original_completeness

    def test_collector_integration(self, validation_config):
        """Test collector integration with configurations."""
        # Test Stooq collector
        stooq_config = CollectorConfig(source_name="stooq", rate_limit=0.1, retry_attempts=1)
        stooq_collector = StooqCollector(stooq_config)

        # Test basic functionality without actual API calls
        test_tickers = ["AAPL", "MSFT"]
        stooq_symbols = {t: stooq_collector._to_stooq_symbol(t) for t in test_tickers}

        assert stooq_symbols["AAPL"] == "aapl.us"
        assert stooq_symbols["MSFT"] == "msft.us"

        # Test Yahoo Finance collector
        yf_collector = YFinanceCollector(stooq_config)
        yahoo_symbols = {t: yf_collector._yahoo_symbol_map(t) for t in ["BRK.B", "BF.B"]}

        assert yahoo_symbols["BRK.B"] == "BRK-B"
        assert yahoo_symbols["BF.B"] == "BF-B"

    def test_configuration_validation(self):
        """Test configuration validation and defaults."""
        # Test ValidationConfig defaults
        config = ValidationConfig()

        assert config.missing_data_threshold == 0.1
        assert config.price_change_threshold == 0.5
        assert config.volume_threshold == 1000
        assert config.min_data_points == 100

        # Test custom configuration
        custom_config = ValidationConfig(
            missing_data_threshold=0.05, generate_reports=False, auto_fix_enabled=True
        )

        assert custom_config.missing_data_threshold == 0.05
        assert custom_config.generate_reports is False
        assert custom_config.auto_fix_enabled is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
