"""Unit tests for Stooq collector functionality."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from src.data.collectors.stooq import StooqCollector
from src.config.data import CollectorConfig


@pytest.fixture
def stooq_collector():
    """Create a StooqCollector instance for testing."""
    config = CollectorConfig(source_name="stooq", rate_limit=10.0, timeout=20)
    return StooqCollector(config)


def test_stooq_collector_initialization(stooq_collector):
    """Test StooqCollector initialization."""
    assert stooq_collector.config.source_name == "stooq"
    assert stooq_collector.config.timeout == 20
    assert stooq_collector.config.rate_limit == 10.0


def test_to_stooq_symbol(stooq_collector):
    """Test _to_stooq_symbol ticker mapping."""
    assert stooq_collector._to_stooq_symbol("AAPL") == "aapl.us"
    assert stooq_collector._to_stooq_symbol("BRK.B") == "brk-b.us"
    assert stooq_collector._to_stooq_symbol("BF.B") == "bf-b.us"
    assert stooq_collector._to_stooq_symbol("  aapl  ") == "aapl.us"


def test_new_session(stooq_collector):
    """Test _new_session creates session with proper headers."""
    session = stooq_collector._new_session()
    assert "User-Agent" in session.headers
    assert "Mozilla" in session.headers["User-Agent"]


@patch("requests.Session")
def test_prewarm_session_success(mock_session_class, stooq_collector):
    """Test _prewarm_session successful prewarming."""
    mock_session = Mock()
    mock_session.get.return_value = Mock()

    stooq_collector._prewarm_session(mock_session)

    mock_session.get.assert_called_once_with("https://stooq.com/", timeout=10)


@patch("requests.Session")
def test_prewarm_session_failure(mock_session_class, stooq_collector):
    """Test _prewarm_session handles exceptions gracefully."""
    mock_session = Mock()
    mock_session.get.side_effect = Exception("Network error")

    # Should not raise exception
    stooq_collector._prewarm_session(mock_session)

    mock_session.get.assert_called_once()


def test_validate_data_coverage(stooq_collector):
    """Test validate_data_coverage functionality."""
    # Create sample price data
    dates = pd.date_range("2023-01-01", "2023-03-01", freq="D")
    prices_df = pd.DataFrame(
        {
            "AAPL": [100 + i for i in range(len(dates))],
            "MSFT": [200 + i * 0.5 for i in range(len(dates))],
            "SPARSE": [300] * 10 + [None] * (len(dates) - 10),  # Sparse data
        },
        index=dates,
    )

    required_tickers = ["AAPL", "MSFT", "GOOGL", "SPARSE"]
    validation = stooq_collector.validate_data_coverage(
        prices_df, required_tickers, min_data_points=50
    )

    assert validation["total_tickers"] == 3
    assert validation["missing_tickers"] == ["GOOGL"]
    assert validation["extra_tickers"] == []
    assert validation["coverage_ratio"] == 0.75  # 3 out of 4 tickers present
    assert "SPARSE" in validation["sparse_tickers"]
    assert validation["sparse_count"] == 1


def test_validate_data_coverage_empty(stooq_collector):
    """Test validate_data_coverage with empty DataFrame."""
    empty_df = pd.DataFrame()
    required_tickers = ["AAPL", "MSFT"]

    validation = stooq_collector.validate_data_coverage(empty_df, required_tickers)

    assert validation["total_tickers"] == 0
    assert validation["missing_tickers"] == ["AAPL", "MSFT"]
    assert validation["coverage_ratio"] == 0.0
    assert validation["date_range"] == (None, None)


@patch("src.data.collectors.stooq.StooqCollector._fetch_stooq_csv")
def test_collect_ohlcv_data_success(mock_fetch, stooq_collector):
    """Test collect_ohlcv_data with successful data retrieval."""
    # Mock successful OHLCV data
    sample_ohlcv = pd.DataFrame(
        {
            "Open": [100, 101, 102],
            "High": [105, 106, 107],
            "Low": [99, 100, 101],
            "Close": [104, 105, 106],
            "Volume": [1000, 1100, 1200],
        },
        index=pd.date_range("2023-01-01", periods=3),
    )

    mock_fetch.return_value = sample_ohlcv

    tickers = ["AAPL", "MSFT"]
    prices_df, volume_df = stooq_collector.collect_ohlcv_data(tickers, max_workers=1)

    # Should have called fetch for each ticker
    assert mock_fetch.call_count == len(tickers)

    # Should return proper DataFrames
    assert len(prices_df.columns) == len(tickers)
    assert len(volume_df.columns) == len(tickers)
    assert list(prices_df.index) == list(sample_ohlcv.index)


@patch("src.data.collectors.stooq.StooqCollector._fetch_stooq_csv")
def test_collect_ohlcv_data_with_failures(mock_fetch, stooq_collector):
    """Test collect_ohlcv_data handles partial failures."""

    # Mock mixed success/failure
    def side_effect(symbol):
        if "aapl" in symbol:
            return pd.DataFrame(
                {"Close": [100, 101], "Volume": [1000, 1100]},
                index=pd.date_range("2023-01-01", periods=2),
            )
        else:
            return None  # Failed fetch

    mock_fetch.side_effect = side_effect

    tickers = ["AAPL", "FAILED"]
    prices_df, volume_df = stooq_collector.collect_ohlcv_data(tickers, max_workers=1)

    # Should only have successful ticker
    assert list(prices_df.columns) == ["AAPL"]
    assert list(volume_df.columns) == ["AAPL"]


@patch("src.data.collectors.stooq.StooqCollector._fetch_stooq_csv")
def test_collect_ohlcv_data_no_data(mock_fetch, stooq_collector):
    """Test collect_ohlcv_data when no data is available."""
    mock_fetch.return_value = None

    tickers = ["INVALID"]
    prices_df, volume_df = stooq_collector.collect_ohlcv_data(tickers, max_workers=1)

    assert prices_df.empty
    assert volume_df.empty


def test_collect_ohlcv_data_date_filtering(stooq_collector):
    """Test collect_ohlcv_data applies date filters correctly."""
    # Create sample data spanning multiple months
    dates = pd.date_range("2023-01-01", "2023-06-30", freq="D")
    sample_data = pd.DataFrame(
        {"Close": range(len(dates)), "Volume": range(1000, 1000 + len(dates))}, index=dates
    )

    with patch.object(stooq_collector, "_fetch_stooq_csv", return_value=sample_data):
        prices_df, volume_df = stooq_collector.collect_ohlcv_data(
            ["TEST"], start_date="2023-02-01", end_date="2023-04-30", max_workers=1
        )

        # Should filter to requested date range
        assert prices_df.index.min() >= pd.Timestamp("2023-02-01")
        assert prices_df.index.max() <= pd.Timestamp("2023-04-30")


@patch("src.data.collectors.stooq.StooqCollector._fetch_stooq_csv")
def test_collect_single_ticker(mock_fetch, stooq_collector):
    """Test collect_single_ticker functionality."""
    sample_ohlcv = pd.DataFrame(
        {"Open": [100], "High": [105], "Low": [99], "Close": [104], "Volume": [1000]},
        index=pd.date_range("2023-01-01", periods=1),
    )

    mock_fetch.return_value = sample_ohlcv

    result = stooq_collector.collect_single_ticker("AAPL")

    assert result is not None
    assert len(result) == 1
    assert "Close" in result.columns
    mock_fetch.assert_called_once_with("aapl.us")
