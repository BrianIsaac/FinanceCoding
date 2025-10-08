"""Unit tests for universe builder functionality."""

from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.config.data import UniverseConfig
from src.data.processors.universe_builder import UniverseBuilder, create_universe_builder


@pytest.fixture
def universe_config():
    """Create a UniverseConfig for testing."""
    return UniverseConfig(universe_type="midcap400", rebalance_frequency="monthly")


@pytest.fixture
def universe_builder(universe_config, tmp_path):
    """Create a UniverseBuilder instance for testing."""
    return UniverseBuilder(universe_config, output_dir=str(tmp_path))


@pytest.fixture
def sample_membership_data():
    """Sample membership intervals data for testing."""
    return pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "start": pd.Timestamp("2020-01-01"),
                "end": pd.Timestamp("2021-06-01"),
                "index_name": "SP400",
            },
            {
                "ticker": "MSFT",
                "start": pd.Timestamp("2020-01-01"),
                "end": None,  # Still active
                "index_name": "SP400",
            },
            {
                "ticker": "IBM",
                "start": pd.Timestamp("2021-06-01"),
                "end": None,  # Still active
                "index_name": "SP400",
            },
        ]
    )


def test_universe_builder_initialization(universe_builder):
    """Test UniverseBuilder initialization."""
    assert universe_builder.universe_config.universe_type == "midcap400"
    assert universe_builder.universe_config.rebalance_frequency == "monthly"
    assert universe_builder.output_dir.exists()
    assert universe_builder.wikipedia_collector is not None


def test_get_index_key(universe_builder):
    """Test _get_index_key mapping."""
    assert universe_builder._get_index_key() == "sp400"

    # Test invalid universe type
    universe_builder.universe_config.universe_type = "invalid"
    with pytest.raises(ValueError, match="Universe type 'invalid' not supported"):
        universe_builder._get_index_key()


def test_create_monthly_snapshots(universe_builder, sample_membership_data):
    """Test create_monthly_snapshots functionality."""
    snapshots = universe_builder.create_monthly_snapshots(
        sample_membership_data, "2020-01-01", "2020-06-01"
    )

    # Should have multiple monthly snapshots
    assert len(snapshots) > 0
    assert "date" in snapshots.columns
    assert "ticker" in snapshots.columns
    assert "index_name" in snapshots.columns

    # Check that we have data for different months
    unique_dates = snapshots["date"].nunique()
    assert unique_dates >= 1

    # AAPL and MSFT should be present in early 2020
    unique_tickers = set(snapshots["ticker"].unique())
    assert "AAPL" in unique_tickers
    assert "MSFT" in unique_tickers


def test_monthly_snapshots_membership_changes(universe_builder, sample_membership_data):
    """Test that monthly snapshots respect membership changes."""
    # Test period that spans AAPL exit and IBM entry (around June 2021)
    snapshots = universe_builder.create_monthly_snapshots(
        sample_membership_data, "2021-05-01", "2021-07-01"
    )

    # Should have snapshots for May, June, July
    dates = snapshots["date"].unique()
    assert len(dates) >= 2

    # Get tickers for different dates
    may_tickers = set(snapshots[snapshots["date"] == dates[0]]["ticker"])
    later_tickers = set(snapshots[snapshots["date"] == dates[-1]]["ticker"])

    # MSFT should be present throughout (no end date)
    assert "MSFT" in may_tickers
    assert "MSFT" in later_tickers


def test_apply_universe_filters(universe_builder):
    """Test _apply_universe_filters functionality."""
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    date = pd.Timestamp("2023-01-01")

    # Test with no filters (should return sorted original list)
    filtered = universe_builder._apply_universe_filters(tickers, date)
    assert filtered == sorted(tickers)

    # Test with custom symbols filter
    universe_builder.universe_config.custom_symbols = ["AAPL", "MSFT"]
    filtered = universe_builder._apply_universe_filters(tickers, date)
    assert filtered == ["AAPL", "MSFT"]


def test_validate_universe_calendar(universe_builder):
    """Test validate_universe_calendar functionality."""
    # Create sample universe calendar
    calendar_data = []
    dates = pd.bdate_range("2023-01-01", "2023-06-01", freq="BMS")

    # Create 400 unique tickers for each date
    base_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    all_tickers = [f"{ticker}_{i}" for ticker in base_tickers for i in range(100)]

    for date in dates:
        for ticker in all_tickers:  # 400 tickers per month
            calendar_data.append({"date": date, "ticker": ticker, "index_name": "SP400"})

    universe_calendar = pd.DataFrame(calendar_data)

    validation_results = universe_builder.validate_universe_calendar(universe_calendar)

    assert validation_results["total_records"] == len(calendar_data)
    assert validation_results["unique_dates"] == len(dates)
    assert validation_results["min_constituents"] == 400
    assert validation_results["max_constituents"] == 400
    assert validation_results["dates_below_400"] == 0


def test_validate_universe_calendar_empty(universe_builder):
    """Test validate_universe_calendar with empty DataFrame."""
    empty_calendar = pd.DataFrame(columns=["date", "ticker", "index_name"])

    validation_results = universe_builder.validate_universe_calendar(empty_calendar)

    assert validation_results["total_records"] == 0
    assert validation_results["unique_dates"] == 0
    assert validation_results["unique_tickers"] == 0


@patch("src.data.processors.universe_builder.WikipediaCollector")
def test_build_membership_intervals(
    mock_wikipedia_collector, universe_builder, sample_membership_data
):
    """Test build_membership_intervals method."""
    # Mock the Wikipedia collector to return sample data
    mock_collector = Mock()
    mock_collector.build_membership.return_value = sample_membership_data
    universe_builder.wikipedia_collector = mock_collector

    result = universe_builder.build_membership_intervals("2020-01-01", "2021-12-31")

    # Should call Wikipedia collector with correct parameters
    mock_collector.build_membership.assert_called_once_with(
        index_key="sp400", end_cap="2021-12-31", seed_current=True
    )

    # Should return the membership data
    assert result.equals(sample_membership_data)


def test_save_universe_calendar(universe_builder, tmp_path):
    """Test save_universe_calendar functionality."""
    # Create sample universe calendar
    universe_calendar = pd.DataFrame(
        [
            {"date": pd.Timestamp("2023-01-01"), "ticker": "AAPL", "index_name": "SP400"},
            {"date": pd.Timestamp("2023-01-01"), "ticker": "MSFT", "index_name": "SP400"},
        ]
    )

    output_path = universe_builder.save_universe_calendar(universe_calendar)

    # Check file was created
    assert output_path.exists()
    assert output_path.suffix == ".parquet"

    # Check file can be read back
    loaded_data = pd.read_parquet(output_path)
    assert len(loaded_data) == 2
    assert "AAPL" in loaded_data["ticker"].values


def test_create_universe_builder_factory():
    """Test create_universe_builder factory function."""
    builder = create_universe_builder(universe_type="sp500", rebalance_frequency="quarterly")

    assert builder.universe_config.universe_type == "sp500"
    assert builder.universe_config.rebalance_frequency == "quarterly"


def test_create_universe_builder_with_kwargs():
    """Test create_universe_builder with additional kwargs."""
    builder = create_universe_builder(
        universe_type="midcap400", custom_symbols=["AAPL", "MSFT"], exclude_sectors=["Utilities"]
    )

    assert builder.universe_config.custom_symbols == ["AAPL", "MSFT"]
    assert builder.universe_config.exclude_sectors == ["Utilities"]


@patch("src.data.processors.universe_builder.WikipediaCollector")
def test_generate_universe_calendar_integration(
    mock_wikipedia_collector, universe_builder, sample_membership_data
):
    """Test generate_universe_calendar integration."""
    # Mock the Wikipedia collector
    mock_collector = Mock()
    mock_collector.build_membership.return_value = sample_membership_data
    universe_builder.wikipedia_collector = mock_collector

    calendar = universe_builder.generate_universe_calendar("2020-01-01", "2020-06-01")

    # Should have added metadata columns
    assert "universe_type" in calendar.columns
    assert "rebalance_frequency" in calendar.columns

    # Should have universe data
    assert len(calendar) > 0
    assert all(calendar["universe_type"] == "midcap400")
    assert all(calendar["rebalance_frequency"] == "monthly")


@patch("src.data.processors.universe_builder.WikipediaCollector")
def test_build_and_save_universe_workflow(
    mock_wikipedia_collector, universe_builder, sample_membership_data
):
    """Test complete build_and_save_universe workflow."""
    # Mock the Wikipedia collector
    mock_collector = Mock()
    mock_collector.build_membership.return_value = sample_membership_data
    universe_builder.wikipedia_collector = mock_collector

    output_path = universe_builder.build_and_save_universe("2020-01-01", "2020-06-01")

    # Should return a valid path
    assert isinstance(output_path, Path)
    assert output_path.exists()

    # Should be able to load the data
    loaded_data = pd.read_parquet(output_path)
    assert len(loaded_data) > 0
    assert "ticker" in loaded_data.columns
