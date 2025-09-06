"""Unit tests for Wikipedia scraping functionality."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from src.data.collectors.wikipedia import WikipediaCollector, ChangeEvent, WIKI_URLS
from src.config.data import CollectorConfig


@pytest.fixture
def wikipedia_collector():
    """Create a WikipediaCollector instance for testing."""
    config = CollectorConfig(source_name="wikipedia", rate_limit=1.0, timeout=30)
    return WikipediaCollector(config)


def test_wikipedia_collector_initialization(wikipedia_collector):
    """Test WikipediaCollector initialization."""
    assert wikipedia_collector.config.source_name == "wikipedia"
    assert wikipedia_collector.config.timeout == 30
    assert wikipedia_collector.session is not None


def test_change_event_dataclass():
    """Test ChangeEvent dataclass creation."""
    event = ChangeEvent(
        date=pd.Timestamp("2023-01-15"), added=["AAPL", "MSFT"], removed=["IBM"], index_name="SP400"
    )

    assert event.date == pd.Timestamp("2023-01-15")
    assert event.added == ["AAPL", "MSFT"]
    assert event.removed == ["IBM"]
    assert event.index_name == "SP400"


def test_wiki_urls_configuration():
    """Test that Wikipedia URLs are properly configured."""
    assert "sp400" in WIKI_URLS
    assert "sp500" in WIKI_URLS
    assert "S%26P_400" in WIKI_URLS["sp400"]
    assert "S%26P_500" in WIKI_URLS["sp500"]


def test_flatten_columns_single_index(wikipedia_collector):
    """Test _flatten_columns with single-level columns."""
    df = pd.DataFrame({"Symbol": [], "Company": [], "GICS Sector": []})
    flattened = wikipedia_collector._flatten_columns(df)

    expected = ["symbol", "company", "gics sector"]
    assert flattened == expected


def test_flatten_columns_multi_index(wikipedia_collector):
    """Test _flatten_columns with MultiIndex columns."""
    columns = pd.MultiIndex.from_tuples(
        [("Added", "Ticker"), ("Added", "Security"), ("Removed", "Ticker"), ("Date", "")]
    )
    df = pd.DataFrame(columns=columns)
    flattened = wikipedia_collector._flatten_columns(df)

    expected = ["added ticker", "added security", "removed ticker", "date"]
    # Handle potential whitespace differences in empty column names
    flattened_cleaned = [col.strip() for col in flattened]
    assert flattened_cleaned == expected


def test_clean_cell_to_tickers(wikipedia_collector):
    """Test _clean_cell_to_tickers ticker extraction."""
    # Test with parentheses pattern
    assert wikipedia_collector._clean_cell_to_tickers("Apple Inc. (AAPL)") == ["AAPL"]
    assert wikipedia_collector._clean_cell_to_tickers("Microsoft (MSFT), Meta (META)") == [
        "MSFT",
        "META",
    ]

    # Test with multiple patterns
    assert wikipedia_collector._clean_cell_to_tickers("Berkshire Hathaway (BRK.B)") == ["BRK.B"]

    # Test fallback to token parsing
    assert wikipedia_collector._clean_cell_to_tickers("AAPL MSFT") == ["AAPL", "MSFT"]

    # Test empty/None cases
    assert wikipedia_collector._clean_cell_to_tickers(None) == []
    assert wikipedia_collector._clean_cell_to_tickers("") == []


def test_pick_col(wikipedia_collector):
    """Test _pick_col column selection logic."""
    cols = ["date", "added ticker", "removed security", "notes"]

    assert wikipedia_collector._pick_col(cols, "date") == "date"
    assert wikipedia_collector._pick_col(cols, "added ticker", "added") == "added ticker"
    assert wikipedia_collector._pick_col(cols, "added", "added ticker") == "added ticker"
    assert wikipedia_collector._pick_col(cols, "nonexistent") is None


@patch("src.data.collectors.wikipedia.WikipediaCollector._fetch_html")
def test_collect_current_membership_invalid_index(mock_fetch, wikipedia_collector):
    """Test collect_current_membership with invalid index."""
    with pytest.raises(ValueError, match="index must be one of"):
        wikipedia_collector.collect_current_membership("invalid_index")


@patch("src.data.collectors.wikipedia.WikipediaCollector._fetch_html")
def test_collect_historical_changes_invalid_index(mock_fetch, wikipedia_collector):
    """Test collect_historical_changes with invalid index."""
    with pytest.raises(ValueError, match="index must be one of"):
        wikipedia_collector.collect_historical_changes("invalid_index")


@patch("src.data.collectors.wikipedia.WikipediaCollector._fetch_html")
def test_build_membership_invalid_index(mock_fetch, wikipedia_collector):
    """Test build_membership with invalid index."""
    with pytest.raises(ValueError, match="index must be one of"):
        wikipedia_collector.build_membership("invalid_index")


def test_events_to_membership_simple(wikipedia_collector):
    """Test _events_to_membership with simple events."""
    events = [
        ChangeEvent(
            date=pd.Timestamp("2023-01-01"), added=["AAPL"], removed=[], index_name="SP400"
        ),
        ChangeEvent(
            date=pd.Timestamp("2023-06-01"), added=["MSFT"], removed=["AAPL"], index_name="SP400"
        ),
    ]

    result = wikipedia_collector._events_to_membership(events, end_cap=pd.Timestamp("2023-12-31"))

    # Should have two intervals: AAPL (Jan-Jun), MSFT (Jun-Dec)
    assert len(result) == 2
    assert "AAPL" in result["ticker"].values
    assert "MSFT" in result["ticker"].values

    # Check AAPL interval
    aapl_row = result[result["ticker"] == "AAPL"].iloc[0]
    assert aapl_row["start"] == pd.Timestamp("2023-01-01")
    assert aapl_row["end"] == pd.Timestamp("2023-06-01")

    # Check MSFT interval
    msft_row = result[result["ticker"] == "MSFT"].iloc[0]
    assert msft_row["start"] == pd.Timestamp("2023-06-01")
    assert msft_row["end"] == pd.Timestamp("2023-12-31")


def test_session_headers(wikipedia_collector):
    """Test that session has proper User-Agent header."""
    headers = wikipedia_collector.session.headers
    assert "User-Agent" in headers
    assert "Mozilla" in headers["User-Agent"]


@pytest.mark.integration
@patch("requests.Session.get")
def test_fetch_html_success(mock_get, wikipedia_collector):
    """Test successful HTML fetching."""
    mock_response = Mock()
    mock_response.text = "<html><body>Test</body></html>"
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    html = wikipedia_collector._fetch_html("https://test.com")
    assert html == "<html><body>Test</body></html>"

    mock_get.assert_called_once_with("https://test.com", timeout=30)


@pytest.mark.integration
@patch("requests.Session.get")
def test_fetch_html_failure(mock_get, wikipedia_collector):
    """Test HTML fetching with HTTP error."""
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = Exception("HTTP Error")
    mock_get.return_value = mock_response

    with pytest.raises(Exception, match="HTTP Error"):
        wikipedia_collector._fetch_html("https://test.com")
