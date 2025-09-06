"""Unit tests for survivorship bias validation functionality."""

import pytest
import pandas as pd
from unittest.mock import patch
from src.data.processors.survivorship_validator import SurvivorshipValidator
from src.config.data import UniverseConfig


@pytest.fixture
def universe_config():
    """Create a UniverseConfig for testing."""
    return UniverseConfig(universe_type="midcap400", rebalance_frequency="monthly")


@pytest.fixture
def survivorship_validator(universe_config):
    """Create a SurvivorshipValidator instance for testing."""
    return SurvivorshipValidator(universe_config)


@pytest.fixture
def sample_membership_data():
    """Sample membership data with survivorship characteristics."""
    return pd.DataFrame(
        [
            {
                "ticker": "SURVIVOR1",
                "start": pd.Timestamp("2020-01-01"),
                "end": pd.NaT,  # Still active
                "index_name": "SP400",
            },
            {
                "ticker": "SURVIVOR2",
                "start": pd.Timestamp("2020-01-01"),
                "end": pd.NaT,  # Still active
                "index_name": "SP400",
            },
            {
                "ticker": "DELISTED1",
                "start": pd.Timestamp("2020-01-01"),
                "end": pd.Timestamp("2022-06-01"),  # Delisted
                "index_name": "SP400",
            },
            {
                "ticker": "DELISTED2",
                "start": pd.Timestamp("2019-01-01"),
                "end": pd.Timestamp("2021-12-01"),  # Delisted
                "index_name": "SP400",
            },
            {
                "ticker": "SHORT_LIVED",
                "start": pd.Timestamp("2023-01-01"),
                "end": pd.Timestamp("2023-03-01"),  # Short membership
                "index_name": "SP400",
            },
        ]
    )


@pytest.fixture
def sample_universe_calendar():
    """Sample universe calendar data."""
    calendar_data = []
    dates = pd.date_range("2020-01-01", "2020-06-01", freq="MS")  # Monthly start

    # Simulate changing universe composition
    for i, date in enumerate(dates):
        base_tickers = ["SURVIVOR1", "SURVIVOR2"]

        if i < 3:  # First 3 months have delisted companies
            base_tickers.extend(["DELISTED1", "DELISTED2"])
        if i >= 4:  # Later months have new additions
            base_tickers.append("NEW_TICKER")

        for ticker in base_tickers:
            calendar_data.append({"date": date, "ticker": ticker, "index_name": "SP400"})

    return pd.DataFrame(calendar_data)


def test_survivorship_validator_initialization(survivorship_validator, universe_config):
    """Test SurvivorshipValidator initialization."""
    assert survivorship_validator.universe_config == universe_config
    assert survivorship_validator.universe_config.universe_type == "midcap400"


def test_validate_membership_intervals(survivorship_validator, sample_membership_data):
    """Test validate_membership_intervals functionality."""
    validation = survivorship_validator.validate_membership_intervals(sample_membership_data)

    # Basic validation
    assert validation["total_intervals"] == 5
    assert validation["unique_tickers"] == 5

    # Survivorship analysis
    assert validation["active_intervals"] == 2  # SURVIVOR1, SURVIVOR2
    assert validation["historical_intervals"] == 3  # 3 delisted companies
    assert validation["historical_ratio"] == 0.6  # 3/5

    # Delisted ticker analysis
    delisted_analysis = validation["delisted_tickers"]
    assert delisted_analysis["delisted_count"] == 3
    assert "DELISTED1" in delisted_analysis["delisted_tickers"]
    assert "DELISTED2" in delisted_analysis["delisted_tickers"]
    assert "SHORT_LIVED" in delisted_analysis["delisted_tickers"]


def test_validate_membership_intervals_empty(survivorship_validator):
    """Test validate_membership_intervals with empty DataFrame."""
    empty_df = pd.DataFrame(columns=["ticker", "start", "end", "index_name"])

    validation = survivorship_validator.validate_membership_intervals(empty_df)

    assert validation["total_intervals"] == 0
    assert validation["unique_tickers"] == 0
    assert validation["active_intervals"] == 0
    assert validation["historical_intervals"] == 0
    assert validation["historical_ratio"] == 0.0


def test_identify_delisted_tickers(survivorship_validator, sample_membership_data):
    """Test _identify_delisted_tickers functionality."""
    delisted_analysis = survivorship_validator._identify_delisted_tickers(sample_membership_data)

    assert delisted_analysis["delisted_count"] == 3
    assert set(delisted_analysis["delisted_tickers"]) == {"DELISTED1", "DELISTED2", "SHORT_LIVED"}

    # Check yearly delistings
    yearly_delistings = delisted_analysis["yearly_delistings"]
    assert yearly_delistings[2022] == 1  # DELISTED1
    assert yearly_delistings[2021] == 1  # DELISTED2
    assert yearly_delistings[2023] == 1  # SHORT_LIVED

    # Check membership duration calculations
    assert delisted_analysis["avg_membership_duration_days"] > 0
    assert delisted_analysis["median_membership_duration_days"] > 0


def test_analyze_survivor_only_bias(survivorship_validator, sample_membership_data):
    """Test _analyze_survivor_only_bias functionality."""
    survivor_analysis = survivorship_validator._analyze_survivor_only_bias(sample_membership_data)

    assert survivor_analysis["current_survivor_count"] == 2  # 2 rows with NaT end dates
    assert survivor_analysis["unique_survivors"] == 2  # SURVIVOR1, SURVIVOR2
    assert survivor_analysis["unique_historical"] == 3  # DELISTED1, DELISTED2, SHORT_LIVED
    assert survivor_analysis["survivor_bias_ratio"] == 0.4  # 2/5 are survivors

    # Should have temporal analysis
    assert "temporal_membership_counts" in survivor_analysis
    temporal_counts = survivor_analysis["temporal_membership_counts"]
    assert len(temporal_counts) == 5  # 5 sample dates


def test_validate_universe_calendar(survivorship_validator, sample_universe_calendar):
    """Test validate_universe_calendar functionality."""
    validation = survivorship_validator.validate_universe_calendar(sample_universe_calendar)

    # Basic metrics
    assert validation["total_records"] > 0
    assert validation["unique_dates"] >= 4  # Several months of data
    assert validation["unique_tickers"] >= 3  # Multiple tickers

    # Universe size trend analysis
    assert "first_month_universe_size" in validation
    assert "last_month_universe_size" in validation
    assert "universe_size_trend" in validation

    # Should have turnover analysis
    assert "ticker_turnover_analysis" in validation
    turnover = validation["ticker_turnover_analysis"]
    assert "avg_monthly_additions" in turnover
    assert "avg_monthly_removals" in turnover


def test_validate_universe_calendar_empty(survivorship_validator):
    """Test validate_universe_calendar with empty DataFrame."""
    empty_calendar = pd.DataFrame(columns=["date", "ticker", "index_name"])

    validation = survivorship_validator.validate_universe_calendar(empty_calendar)
    assert "error" in validation
    assert validation["error"] == "Empty universe calendar"


def test_analyze_ticker_turnover(survivorship_validator, sample_universe_calendar):
    """Test _analyze_ticker_turnover functionality."""
    turnover_analysis = survivorship_validator._analyze_ticker_turnover(sample_universe_calendar)

    # Should have turnover statistics
    assert "avg_monthly_additions" in turnover_analysis
    assert "avg_monthly_removals" in turnover_analysis
    assert "avg_net_change" in turnover_analysis
    assert "max_monthly_turnover" in turnover_analysis

    # Should identify high turnover periods
    assert "high_turnover_dates" in turnover_analysis


def test_analyze_ticker_turnover_insufficient_data(survivorship_validator):
    """Test _analyze_ticker_turnover with insufficient data."""
    # Single date calendar
    single_date_calendar = pd.DataFrame(
        [{"date": pd.Timestamp("2023-01-01"), "ticker": "AAPL", "index_name": "SP400"}]
    )

    turnover_analysis = survivorship_validator._analyze_ticker_turnover(single_date_calendar)
    assert turnover_analysis["insufficient_data"] is True


def test_generate_survivorship_report(
    survivorship_validator, sample_membership_data, sample_universe_calendar
):
    """Test generate_survivorship_report functionality."""
    report = survivorship_validator.generate_survivorship_report(
        sample_membership_data, sample_universe_calendar
    )

    # Should have all main sections
    assert "universe_type" in report
    assert "validation_timestamp" in report
    assert "methodology" in report
    assert "membership_validation" in report
    assert "calendar_validation" in report
    assert "overall_assessment" in report

    # Methodology section
    methodology = report["methodology"]
    assert "description" in methodology
    assert "approach" in methodology
    assert "bias_mitigation" in methodology

    # Overall assessment
    assessment = report["overall_assessment"]
    assert "bias_score" in assessment
    assert "bias_grade" in assessment
    assert "recommendations" in assessment

    # Bias score should be between 0 and 1
    assert 0 <= assessment["bias_score"] <= 1
    assert assessment["bias_grade"] in ["A", "B", "C", "D", "F"]


def test_score_to_grade(survivorship_validator):
    """Test _score_to_grade conversion."""
    assert survivorship_validator._score_to_grade(0.95) == "A"
    assert survivorship_validator._score_to_grade(0.85) == "B"
    assert survivorship_validator._score_to_grade(0.75) == "C"
    assert survivorship_validator._score_to_grade(0.65) == "D"
    assert survivorship_validator._score_to_grade(0.45) == "F"


def test_generate_recommendations(
    survivorship_validator, sample_membership_data, sample_universe_calendar
):
    """Test _generate_recommendations functionality."""
    report = {
        "membership_validation": survivorship_validator.validate_membership_intervals(
            sample_membership_data
        ),
        "calendar_validation": survivorship_validator.validate_universe_calendar(
            sample_universe_calendar
        ),
    }

    recommendations = survivorship_validator._generate_recommendations(report)

    assert isinstance(recommendations, list)
    assert len(recommendations) > 0

    # All recommendations should be strings
    for rec in recommendations:
        assert isinstance(rec, str)
        assert len(rec) > 0


@patch("builtins.open")
@patch("json.dump")
@patch("pathlib.Path.mkdir")
def test_save_detailed_report(mock_mkdir, mock_json_dump, mock_open, survivorship_validator):
    """Test _save_detailed_report functionality."""
    report = {"test": "data"}
    output_path = "test_output.json"

    # Create mock file handle
    mock_file = mock_open.return_value.__enter__.return_value

    survivorship_validator._save_detailed_report(report, output_path)

    # Should create directories
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    # Should open file for writing (Path object)
    from pathlib import Path
    mock_open.assert_called_once_with(Path(output_path), "w")

    # Should write JSON data
    mock_json_dump.assert_called_once_with(report, mock_file, indent=2, default=str)


def test_generate_overall_assessment_high_bias(survivorship_validator):
    """Test _generate_overall_assessment with high bias scenario."""
    # Create report with high bias indicators
    report = {
        "membership_validation": {
            "historical_ratio": 0.1,  # Low historical ratio
            "unique_tickers": 100,
            "delisted_tickers": {"delisted_count": 5},  # Few delisted
        },
        "calendar_validation": {"universe_size_trend": 0.8},  # High size growth trend
    }

    assessment = survivorship_validator._generate_overall_assessment(report)

    # Should have low bias score (high bias)
    assert assessment["bias_score"] < 0.5
    assert assessment["bias_grade"] in ["D", "F"]
    assert len(assessment["recommendations"]) > 1  # Multiple recommendations for problems


def test_generate_overall_assessment_low_bias(survivorship_validator):
    """Test _generate_overall_assessment with low bias scenario."""
    # Create report with low bias indicators
    report = {
        "membership_validation": {
            "historical_ratio": 0.6,  # Good historical ratio
            "unique_tickers": 100,
            "delisted_tickers": {"delisted_count": 40},  # Many delisted
        },
        "calendar_validation": {
            "universe_size_trend": 0.1,  # Low size growth trend
            "survivorship_bias_warning": False,
        },
    }

    assessment = survivorship_validator._generate_overall_assessment(report)

    # Should have high bias score (low bias)
    assert assessment["bias_score"] > 0.7
    assert assessment["bias_grade"] in ["A", "B", "C"]
