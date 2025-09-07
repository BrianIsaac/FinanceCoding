"""Integration tests for complete multi-source data pipeline.

Tests the end-to-end workflow from data collection to storage and validation.
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

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
def temp_directory():
    """Create temporary directory for integration testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def pipeline_configs():
    """Create configurations for pipeline components."""
    validation_config = ValidationConfig(
        missing_data_threshold=0.1,
        price_change_threshold=0.3,
        volume_threshold=1000,
        generate_reports=False,
        auto_fix_enabled=True,
    )

    collector_config = CollectorConfig(
        source_name="integration_test", rate_limit=0.1, timeout=5, retry_attempts=1
    )

    return {"validation": validation_config, "collector": collector_config}


@pytest.fixture
def mock_stooq_data():
    """Create mock Stooq data for testing."""
    dates = pd.date_range("2023-01-01", "2023-03-31", freq="D")
    tickers = ["AAPL", "MSFT", "GOOGL"]

    # Create realistic OHLCV data
    np.random.seed(42)
    base_prices = np.array([150.0, 250.0, 2800.0])  # Starting prices

    ohlcv_data = {}
    for i, ticker in enumerate(tickers):
        returns = np.random.normal(0.001, 0.02, len(dates))  # 0.1% daily return, 2% vol
        prices = base_prices[i] * (1 + returns).cumprod()

        # Create OHLCV with realistic relationships
        df = pd.DataFrame(index=dates)
        df["Open"] = prices * np.random.uniform(0.995, 1.005, len(dates))
        df["High"] = np.maximum(prices, df["Open"]) * np.random.uniform(1.0, 1.02, len(dates))
        df["Low"] = np.minimum(prices, df["Open"]) * np.random.uniform(0.98, 1.0, len(dates))
        df["Close"] = prices
        df["Volume"] = np.random.randint(1000000, 5000000, len(dates))

        # Add some gaps and outliers
        if ticker == "AAPL":
            df.loc["2023-01-15":"2023-01-18", ["Open", "High", "Low", "Close"]] = np.nan
            df.loc["2023-02-10", "Volume"] = 0

        ohlcv_data[ticker] = df

    return ohlcv_data


@pytest.fixture
def mock_yahoo_data():
    """Create mock Yahoo Finance data for testing."""
    dates = pd.date_range("2023-01-01", "2023-03-31", freq="D")
    tickers = ["AAPL", "TSLA"]  # Partial overlap with Stooq

    np.random.seed(123)
    yahoo_data = {}

    for ticker in tickers:
        base_price = 200.0 if ticker == "AAPL" else 800.0
        returns = np.random.normal(0.0008, 0.025, len(dates))
        prices = base_price * (1 + returns).cumprod()

        df = pd.DataFrame(index=dates)
        df["Close"] = prices  # Yahoo typically provides adjusted close
        df["Volume"] = np.random.randint(500000, 3000000, len(dates))

        yahoo_data[ticker] = df

    return yahoo_data


class TestCompleteDataPipeline:
    """Test complete data pipeline integration."""

    def test_pipeline_initialization(self, pipeline_configs, temp_directory):
        """Test pipeline component initialization."""
        gap_filler = GapFiller(pipeline_configs["validation"])
        normalizer = DataNormalizer(pipeline_configs["validation"])
        validator = DataQualityValidator(pipeline_configs["validation"])
        parquet_manager = ParquetManager(temp_directory)

        # All components should initialize successfully
        assert gap_filler.config == pipeline_configs["validation"]
        assert normalizer.config == pipeline_configs["validation"]
        assert validator.config == pipeline_configs["validation"]
        assert parquet_manager.base_path == Path(temp_directory)

    @patch.object(StooqCollector, "_fetch_stooq_csv")
    def test_stooq_data_collection_mock(self, mock_fetch, pipeline_configs, mock_stooq_data):
        """Test Stooq data collection with mocked responses."""

        # Configure mock to return our test data
        def mock_fetch_side_effect(symbol):
            ticker = symbol.replace(".us", "").upper()
            if ticker in mock_stooq_data:
                return mock_stooq_data[ticker]
            return None

        mock_fetch.side_effect = mock_fetch_side_effect

        # Test collection
        collector = StooqCollector(pipeline_configs["collector"])
        tickers = list(mock_stooq_data.keys())

        ohlcv_result = collector.collect_ohlcv_data(tickers)

        # Verify results
        assert "close" in ohlcv_result
        assert "volume" in ohlcv_result
        assert not ohlcv_result["close"].empty
        assert not ohlcv_result["volume"].empty

        # Should have data for all tickers
        for ticker in tickers:
            assert ticker in ohlcv_result["close"].columns
            assert ticker in ohlcv_result["volume"].columns

    @patch.object(YFinanceCollector, "download_batch_data")
    def test_yahoo_fallback_integration(self, mock_download, pipeline_configs, mock_yahoo_data):
        """Test Yahoo Finance fallback integration."""

        # Configure mock
        def mock_download_side_effect(tickers, start_date=None, end_date=None, batch_size=80):
            available_tickers = [t for t in tickers if t in mock_yahoo_data]
            if not available_tickers:
                return pd.DataFrame(), pd.DataFrame()

            prices_list = []
            volume_list = []

            for ticker in available_tickers:
                data = mock_yahoo_data[ticker]
                prices_list.append(data["Close"].rename(ticker))
                volume_list.append(data["Volume"].rename(ticker))

            prices_df = pd.concat(prices_list, axis=1)
            volume_df = pd.concat(volume_list, axis=1)

            return prices_df, volume_df

        mock_download.side_effect = mock_download_side_effect

        # Test fallback collection
        collector = YFinanceCollector(pipeline_configs["collector"])

        # Simulate primary data with gaps
        price_data = np.random.randn(90, 2).cumsum(axis=0) + 100
        primary_prices = pd.DataFrame(
            data=price_data,
            index=pd.date_range("2023-01-01", "2023-03-31"),
            columns=["AAPL", "MSFT"],
        )
        primary_prices.loc["2023-01-15":"2023-01-20", "AAPL"] = np.nan

        primary_volume = pd.DataFrame(
            index=primary_prices.index,
            columns=primary_prices.columns,
            data=np.random.randint(1000, 10000, primary_prices.shape),
        )

        # Execute fallback collection
        primary_data = {"close": primary_prices, "volume": primary_volume}
        universe_tickers = ["AAPL", "MSFT", "TSLA"]

        merged_data = collector.execute_fallback_collection(
            primary_data, universe_tickers, start_date="2023-01-01", end_date="2023-03-31"
        )

        # Verify fallback worked
        assert "close" in merged_data
        assert "volume" in merged_data
        assert "TSLA" in merged_data["close"].columns  # Should be added from Yahoo

    def test_gap_filling_pipeline(self, mock_stooq_data, pipeline_configs):
        """Test gap filling in pipeline context."""
        # Extract price data with gaps
        prices_df = pd.concat([data["Close"] for ticker, data in mock_stooq_data.items()], axis=1)
        prices_df.columns = list(mock_stooq_data.keys())

        volume_df = pd.concat([data["Volume"] for ticker, data in mock_stooq_data.items()], axis=1)
        volume_df.columns = list(mock_stooq_data.keys())

        # Test gap filling
        gap_filler = GapFiller(pipeline_configs["validation"])

        original_na_count = prices_df.isna().sum().sum()
        filled_prices = gap_filler.process_dataframe(
            prices_df, volume_df=volume_df, method="hybrid"
        )
        filled_na_count = filled_prices.isna().sum().sum()

        # Should fill some gaps
        assert filled_na_count < original_na_count

        # Validate fill quality
        quality_report = gap_filler.validate_fill_quality(prices_df, filled_prices, volume_df)
        assert quality_report["fill_rate"] > 0.0

    def test_data_normalization_pipeline(self, mock_stooq_data, pipeline_configs):
        """Test data normalization in pipeline context."""
        # Prepare data
        prices_df = pd.concat([data["Close"] for ticker, data in mock_stooq_data.items()], axis=1)
        prices_df.columns = list(mock_stooq_data.keys())

        volume_df = pd.concat([data["Volume"] for ticker, data in mock_stooq_data.items()], axis=1)
        volume_df.columns = list(mock_stooq_data.keys())

        # Test normalization
        normalizer = DataNormalizer(pipeline_configs["validation"])

        # Calculate returns
        returns = normalizer.calculate_daily_returns(prices_df, method="simple", handle_splits=True)
        assert returns.shape == prices_df.shape

        # Normalize volume
        normalized_volume = normalizer.normalize_volume(
            volume_df, prices_df, method="dollar_volume"
        )
        assert normalized_volume.shape == volume_df.shape

        # Calculate quality score
        data_dict = {"close": prices_df, "returns": returns, "volume": volume_df}

        quality_metrics = normalizer.calculate_data_quality_score(data_dict, volume_df)
        assert "overall_score" in quality_metrics
        assert 0 <= quality_metrics["overall_score"] <= 1

    def test_quality_validation_pipeline(self, mock_stooq_data, pipeline_configs):
        """Test quality validation in pipeline context."""
        # Prepare data
        prices_df = pd.concat([data["Close"] for ticker, data in mock_stooq_data.items()], axis=1)
        prices_df.columns = list(mock_stooq_data.keys())

        returns_df = prices_df.pct_change()
        volume_df = pd.concat([data["Volume"] for ticker, data in mock_stooq_data.items()], axis=1)
        volume_df.columns = list(mock_stooq_data.keys())

        # Test validation
        validator = DataQualityValidator(pipeline_configs["validation"])

        data_dict = {"close": prices_df, "returns": returns_df, "volume": volume_df}

        validation_results = validator.validate_complete_dataset(
            data_dict, universe_tickers=list(mock_stooq_data.keys()), generate_report=False
        )

        # Verify validation results
        assert "overall_quality_score" in validation_results
        assert "component_scores" in validation_results
        assert "recommendations" in validation_results

        # Should have validation for each data type
        expected_components = ["prices", "returns", "volume", "cross_validation"]
        for component in expected_components:
            assert component in validation_results["component_scores"]

    def test_parquet_storage_pipeline(self, mock_stooq_data, temp_directory, pipeline_configs):
        """Test parquet storage in pipeline context."""
        # Prepare data
        ohlcv_dict = {}
        for component in ["open", "high", "low", "close", "volume"]:
            component_data = pd.concat(
                [data[component.capitalize()] for ticker, data in mock_stooq_data.items()], axis=1
            )
            component_data.columns = list(mock_stooq_data.keys())
            ohlcv_dict[component] = component_data

        # Test parquet storage
        parquet_manager = ParquetManager(temp_directory)

        # Save all data types
        saved_paths = {}
        for data_type, df in ohlcv_dict.items():
            saved_path = parquet_manager.save_dataframe(
                df,
                data_type=data_type,
                partition_strategy="monthly",
                metadata={"test": "integration", "data_type": data_type},
            )
            saved_paths[data_type] = saved_path

        # Verify files were saved
        for data_type, path in saved_paths.items():
            assert os.path.exists(path)
            data_info = parquet_manager.get_data_info(data_type)
            assert data_info["exists"]
            assert data_info["num_files"] > 0

        # Test data loading
        loader = parquet_manager.create_data_loading_interface()
        loaded_data = loader.load_training_data(
            data_types=list(ohlcv_dict.keys()), start_date="2023-01-01", end_date="2023-03-31"
        )

        # Verify loaded data
        for data_type in ohlcv_dict.keys():
            assert data_type in loaded_data
            assert not loaded_data[data_type].empty

            # Should match original data (allowing for small differences)
            pd.testing.assert_frame_equal(
                ohlcv_dict[data_type], loaded_data[data_type], check_dtype=False, rtol=1e-10
            )

    def test_end_to_end_integration(self, temp_directory, pipeline_configs):
        """Test complete end-to-end integration."""
        # Create realistic test data
        dates = pd.date_range("2023-01-01", "2023-02-28", freq="D")
        tickers = ["AAPL", "MSFT", "GOOGL"]

        np.random.seed(42)

        # Generate correlated price data
        base_prices = np.array([150.0, 250.0, 2800.0])
        returns_data = np.random.multivariate_normal(
            mean=[0.001, 0.001, 0.001],
            cov=[[0.0004, 0.0001, 0.0002], [0.0001, 0.0003, 0.0001], [0.0002, 0.0001, 0.0005]],
            size=len(dates),
        )

        prices_df = pd.DataFrame(
            index=dates, columns=tickers, data=base_prices * (1 + returns_data).cumprod(axis=0)
        )

        volume_df = pd.DataFrame(
            index=dates,
            columns=tickers,
            data=np.random.randint(1000000, 5000000, (len(dates), len(tickers))),
        )

        # Introduce data quality issues
        prices_df.loc["2023-01-15":"2023-01-17", "AAPL"] = np.nan  # Gap
        prices_df.loc["2023-02-01", "MSFT"] = prices_df.loc["2023-01-31", "MSFT"] * 2  # Outlier
        volume_df.loc["2023-01-16", "AAPL"] = 0  # Zero volume

        # Initialize pipeline components
        gap_filler = GapFiller(pipeline_configs["validation"])
        normalizer = DataNormalizer(pipeline_configs["validation"])
        validator = DataQualityValidator(pipeline_configs["validation"])
        parquet_manager = ParquetManager(temp_directory)

        # Step 1: Gap filling
        filled_prices = gap_filler.process_dataframe(
            prices_df, volume_df=volume_df, method="hybrid"
        )

        gap_filler.validate_fill_quality(prices_df, filled_prices, volume_df)

        # Step 2: Data normalization
        returns = normalizer.calculate_daily_returns(
            filled_prices, method="simple", handle_splits=True
        )

        normalized_volume = normalizer.normalize_volume(
            volume_df, filled_prices, method="dollar_volume"
        )

        # Step 3: Quality validation
        data_dict = {"close": filled_prices, "returns": returns, "volume": normalized_volume}

        validation_results = validator.validate_complete_dataset(
            data_dict, universe_tickers=tickers, generate_report=False
        )

        # Step 4: Storage
        for data_type, df in data_dict.items():
            parquet_manager.save_dataframe(
                df,
                data_type=data_type,
                partition_strategy="none",  # Simple for small test data
                metadata={
                    "pipeline_version": "1.3.0",
                    "validation_score": validation_results["overall_quality_score"],
                    "test": "end_to_end_integration",
                },
            )

        # Step 5: Verification
        loader = parquet_manager.create_data_loading_interface()

        # Load aligned datasets
        aligned_data = loader.get_aligned_datasets(
            primary_data_type="close",
            start_date="2023-01-01",
            end_date="2023-02-28",
            required_data_types=["close", "returns", "volume"],
        )

        # Verify pipeline results
        assert all(data_type in aligned_data for data_type in data_dict.keys())
        assert all(not df.empty for df in aligned_data.values())

        # Quality should be reasonable
        assert validation_results["overall_quality_score"] > 0.5

        # Gap filling should have helped
        original_completeness = 1.0 - (prices_df.isna().sum().sum() / prices_df.size)
        final_completeness = 1.0 - (
            aligned_data["close"].isna().sum().sum() / aligned_data["close"].size
        )
        assert final_completeness >= original_completeness

        # Returns should be reasonable
        daily_vol = aligned_data["returns"].std().mean()
        annual_vol = daily_vol * np.sqrt(252)
        assert 0.05 < annual_vol < 1.0  # Reasonable annual volatility range

        return {
            "validation_results": validation_results,
            "final_data": aligned_data,
            "pipeline_metrics": {
                "original_completeness": original_completeness,
                "final_completeness": final_completeness,
                "annual_volatility": annual_vol,
            },
        }

    def test_performance_benchmarks(self, temp_directory, pipeline_configs):
        """Test pipeline performance with larger datasets."""
        import time

        # Create larger dataset for performance testing
        dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")  # 4 years
        tickers = [f"STOCK{i:03d}" for i in range(50)]  # 50 stocks

        np.random.seed(42)

        # Generate large price dataset
        start_time = time.time()

        returns_matrix = np.random.normal(0.001, 0.02, (len(dates), len(tickers)))
        prices_matrix = 100.0 * (1 + returns_matrix).cumprod(axis=0)

        prices_df = pd.DataFrame(index=dates, columns=tickers, data=prices_matrix)

        # Add some gaps
        for i in range(0, len(tickers), 5):  # Every 5th stock has gaps
            gap_start = np.random.randint(100, len(dates) - 10)
            gap_length = np.random.randint(1, 5)
            prices_df.iloc[gap_start : gap_start + gap_length, i] = np.nan

        time.time() - start_time

        # Test gap filling performance
        start_time = time.time()
        gap_filler = GapFiller(pipeline_configs["validation"])
        filled_prices = gap_filler.process_dataframe(prices_df, method="forward")
        gap_filling_time = time.time() - start_time

        # Test normalization performance
        start_time = time.time()
        normalizer = DataNormalizer(pipeline_configs["validation"])
        normalizer.calculate_daily_returns(filled_prices)
        normalization_time = time.time() - start_time

        # Test parquet storage performance
        start_time = time.time()
        parquet_manager = ParquetManager(temp_directory)
        parquet_manager.save_dataframe(
            filled_prices, data_type="prices", partition_strategy="yearly"
        )
        storage_time = time.time() - start_time

        # Test loading performance
        start_time = time.time()
        loaded_prices = parquet_manager.load_dataframe("prices")
        loading_time = time.time() - start_time

        # Verify data integrity
        pd.testing.assert_frame_equal(filled_prices, loaded_prices, check_dtype=False, rtol=1e-10)

        # Performance assertions (reasonable for CI environments)
        total_time = gap_filling_time + normalization_time + storage_time + loading_time
        data_points = len(dates) * len(tickers)

        # Should process at least 10K data points per second
        assert data_points / total_time > 10000, "Pipeline performance below threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
