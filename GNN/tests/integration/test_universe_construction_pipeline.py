"""Integration tests for complete universe construction pipeline."""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch

from src.config.data import UniverseConfig, CollectorConfig
from src.data.collectors.wikipedia import WikipediaCollector
from src.data.processors.universe_builder import UniverseBuilder
from src.data.processors.survivorship_validator import SurvivorshipValidator
from src.data.loaders.portfolio_data import PortfolioDataLoader


@pytest.fixture
def universe_config():
    """Universe configuration for testing."""
    return UniverseConfig(universe_type="midcap400", rebalance_frequency="monthly")


@pytest.fixture
def sample_wikipedia_membership():
    """Sample Wikipedia membership data for integration testing."""
    return pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "start": pd.Timestamp("2016-01-01"),
                "end": pd.NaT,  # Still active
                "index_name": "SP400",
            },
            {
                "ticker": "MSFT",
                "start": pd.Timestamp("2016-01-01"),
                "end": pd.NaT,  # Still active
                "index_name": "SP400",
            },
            {
                "ticker": "DELISTED1",
                "start": pd.Timestamp("2016-01-01"),
                "end": pd.Timestamp("2020-06-01"),  # Delisted
                "index_name": "SP400",
            },
            {
                "ticker": "DELISTED2",
                "start": pd.Timestamp("2018-01-01"),
                "end": pd.Timestamp("2022-03-01"),  # Delisted later
                "index_name": "SP400",
            },
            {
                "ticker": "NEW_MEMBER",
                "start": pd.Timestamp("2022-01-01"),
                "end": pd.NaT,  # Added recently
                "index_name": "SP400",
            },
        ]
    )


@pytest.mark.integration
class TestUniverseConstructionPipeline:
    """Integration tests for end-to-end universe construction."""

    @patch("src.data.collectors.wikipedia.WikipediaCollector.build_membership")
    def test_end_to_end_universe_construction(
        self, mock_build_membership, universe_config, sample_wikipedia_membership, tmp_path
    ):
        """Test complete pipeline from Wikipedia to universe calendar."""
        # Mock Wikipedia data
        mock_build_membership.return_value = sample_wikipedia_membership

        # Create universe builder with temp output
        universe_builder = UniverseBuilder(universe_config, str(tmp_path))

        # Generate universe calendar
        universe_calendar = universe_builder.generate_universe_calendar("2016-01-01", "2024-12-31")

        # Validate results
        assert not universe_calendar.empty
        assert "date" in universe_calendar.columns
        assert "ticker" in universe_calendar.columns
        assert "index_name" in universe_calendar.columns

        # Check temporal consistency
        dates = sorted(universe_calendar["date"].unique())
        assert len(dates) > 50  # Should have many monthly snapshots

        # Verify survivorship characteristics
        all_tickers = set(universe_calendar["ticker"].unique())
        assert "AAPL" in all_tickers  # Survivor
        assert "MSFT" in all_tickers  # Survivor
        assert "DELISTED1" in all_tickers  # Should appear in early periods
        assert "DELISTED2" in all_tickers  # Should appear in middle periods
        assert "NEW_MEMBER" in all_tickers  # Should appear in late periods

    @patch("src.data.collectors.wikipedia.WikipediaCollector.build_membership")
    def test_survivorship_validation_integration(
        self, mock_build_membership, universe_config, sample_wikipedia_membership
    ):
        """Test integration of survivorship validation with universe construction."""
        # Mock Wikipedia data
        mock_build_membership.return_value = sample_wikipedia_membership

        # Create components
        universe_builder = UniverseBuilder(universe_config, "data/processed")
        survivorship_validator = SurvivorshipValidator(universe_config)

        # Generate universe calendar (mocked data)
        universe_calendar = universe_builder.create_monthly_snapshots(
            sample_wikipedia_membership, "2016-01-01", "2024-12-31"
        )

        # Validate survivorship characteristics
        report = survivorship_validator.generate_survivorship_report(
            sample_wikipedia_membership, universe_calendar
        )

        # Check validation results
        assert report["universe_type"] == "midcap400"
        assert "membership_validation" in report
        assert "calendar_validation" in report
        assert "overall_assessment" in report

        # Check bias detection
        membership_val = report["membership_validation"]
        assert membership_val["historical_intervals"] >= 2  # Should detect delisted companies
        assert membership_val["active_intervals"] >= 3  # Should have active companies

        # Check assessment
        assessment = report["overall_assessment"]
        assert "bias_score" in assessment
        assert "bias_grade" in assessment
        assert assessment["bias_grade"] in ["A", "B", "C", "D", "F"]

    @patch("src.data.collectors.wikipedia.WikipediaCollector.build_membership")
    def test_portfolio_data_loader_integration(
        self, mock_build_membership, universe_config, sample_wikipedia_membership, tmp_path
    ):
        """Test integration of portfolio data loader with universe construction."""
        # Mock Wikipedia data
        mock_build_membership.return_value = sample_wikipedia_membership

        # Create portfolio data loader
        data_loader = PortfolioDataLoader(tmp_path, universe_config)

        # Build universe if missing (should create it)
        calendar_file = data_loader.build_universe_if_missing("2016-01-01", "2024-12-31")

        # Verify file was created
        assert calendar_file.exists()

        # Load the universe calendar
        universe_calendar = data_loader.load_universe_calendar("midcap400")

        assert not universe_calendar.empty
        assert "date" in universe_calendar.columns
        assert "ticker" in universe_calendar.columns

        # Test date-specific universe lookup
        universe_2020 = data_loader.get_universe_at_date("2020-01-01")
        universe_2023 = data_loader.get_universe_at_date("2023-01-01")

        assert isinstance(universe_2020, list)
        assert isinstance(universe_2023, list)

        # Should have differences due to membership changes
        assert "DELISTED1" in universe_2020  # Should be in 2020
        # DELISTED1 should NOT be in 2023 (delisted in 2020)

    def test_data_alignment_validation(self, tmp_path):
        """Test data alignment validation functionality."""
        # Create sample price and volume data
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        tickers = ["AAPL", "MSFT", "GOOGL"]

        prices_df = pd.DataFrame(
            data=[[100 + i] * len(tickers) for i in range(len(dates))], index=dates, columns=tickers
        )

        volumes_df = pd.DataFrame(
            data=[[1000 + i * 10] * len(tickers) for i in range(len(dates))],
            index=dates,
            columns=tickers,
        )

        # Create universe calendar
        universe_calendar = pd.DataFrame(
            [
                {"date": pd.Timestamp("2020-01-01"), "ticker": "AAPL", "index_name": "SP400"},
                {"date": pd.Timestamp("2020-01-01"), "ticker": "MSFT", "index_name": "SP400"},
                {"date": pd.Timestamp("2020-01-01"), "ticker": "GOOGL", "index_name": "SP400"},
            ]
        )

        # Test validation
        universe_config = UniverseConfig()
        data_loader = PortfolioDataLoader(tmp_path, universe_config)

        validation = data_loader.validate_data_alignment(prices_df, volumes_df, universe_calendar)

        # Check validation results
        assert validation["date_alignment"] is True
        assert validation["ticker_alignment"] is True
        assert validation["common_tickers"] == 3
        assert validation["universe_coverage_in_prices"] == 1.0  # Perfect coverage

    def test_misaligned_data_detection(self, tmp_path):
        """Test detection of data alignment issues."""
        # Create misaligned data
        dates1 = pd.date_range("2020-01-01", "2020-06-30", freq="D")
        dates2 = pd.date_range("2020-07-01", "2020-12-31", freq="D")

        prices_df = pd.DataFrame(
            data=[[100] * 2 for _ in range(len(dates1))], index=dates1, columns=["AAPL", "MSFT"]
        )

        volumes_df = pd.DataFrame(
            data=[[1000] * 3 for _ in range(len(dates2))],
            index=dates2,
            columns=["AAPL", "MSFT", "GOOGL"],  # Extra ticker
        )

        universe_calendar = pd.DataFrame(
            [
                {"date": pd.Timestamp("2020-01-01"), "ticker": "AAPL", "index_name": "SP400"},
                {"date": pd.Timestamp("2020-01-01"), "ticker": "MSFT", "index_name": "SP400"},
                {
                    "date": pd.Timestamp("2020-01-01"),
                    "ticker": "MISSING",
                    "index_name": "SP400",
                },  # Not in prices
            ]
        )

        universe_config = UniverseConfig()
        data_loader = PortfolioDataLoader(tmp_path, universe_config)

        validation = data_loader.validate_data_alignment(prices_df, volumes_df, universe_calendar)

        # Should detect misalignment
        assert validation["date_alignment"] is False
        assert validation["ticker_alignment"] is False
        assert validation["volume_only_tickers"] == 1  # GOOGL only in volumes
        assert "MISSING" in validation["missing_from_prices"]

    @patch("src.data.collectors.wikipedia.WikipediaCollector")
    def test_collector_integration(self, mock_wikipedia_collector_class):
        """Test integration between different data collectors."""
        # Mock Wikipedia collector
        mock_collector = Mock()
        mock_collector.build_membership.return_value = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "start": pd.Timestamp("2020-01-01"),
                    "end": pd.NaT,
                    "index_name": "SP400",
                }
            ]
        )
        mock_wikipedia_collector_class.return_value = mock_collector

        # Test collector config integration
        collector_config = CollectorConfig(
            source_name="wikipedia", rate_limit=1.0, timeout=30, retry_attempts=3
        )

        wikipedia_collector = WikipediaCollector(collector_config)

        # Verify collector uses configuration
        assert wikipedia_collector.config.source_name == "wikipedia"
        assert wikipedia_collector.config.rate_limit == 1.0
        assert wikipedia_collector.config.timeout == 30

    def test_configuration_integration(self, tmp_path):
        """Test that all components work together with unified configuration."""
        # Create comprehensive configuration
        universe_config = UniverseConfig(
            universe_type="midcap400",
            rebalance_frequency="monthly",
            min_market_cap=1000.0,
            exclude_sectors=["Utilities"],
        )

        collector_config = CollectorConfig(source_name="wikipedia", rate_limit=1.0, timeout=30)

        # Create integrated components
        data_loader = PortfolioDataLoader(tmp_path, universe_config)
        universe_builder = UniverseBuilder(universe_config, str(tmp_path))
        survivorship_validator = SurvivorshipValidator(universe_config)

        # Verify configuration propagation
        assert data_loader.universe_config.universe_type == "midcap400"
        assert data_loader.universe_config.rebalance_frequency == "monthly"
        assert universe_builder.universe_config.universe_type == "midcap400"
        assert survivorship_validator.universe_config.universe_type == "midcap400"


@pytest.mark.integration
class TestDataQualityValidation:
    """Integration tests for data quality and validation."""

    def test_empty_data_handling(self, tmp_path):
        """Test handling of empty or missing data files."""
        universe_config = UniverseConfig()
        data_loader = PortfolioDataLoader(tmp_path, universe_config)

        # Try loading non-existent files
        prices = data_loader.load_prices("nonexistent")
        volumes = data_loader.load_volumes("nonexistent")
        calendar = data_loader.load_universe_calendar("nonexistent")

        # Should return empty DataFrames without crashing
        assert prices.empty
        assert volumes.empty
        assert calendar.empty

        # Should handle empty universe lookup gracefully
        universe_tickers = data_loader.get_universe_at_date("2023-01-01")
        assert universe_tickers == []

    @patch("src.data.collectors.wikipedia.WikipediaCollector.build_membership")
    def test_data_quality_metrics(self, mock_build_membership, tmp_path):
        """Test comprehensive data quality validation."""
        # Create sample data with quality issues
        problematic_membership = pd.DataFrame(
            [
                {
                    "ticker": "GOOD_TICKER",
                    "start": pd.Timestamp("2020-01-01"),
                    "end": pd.NaT,
                    "index_name": "SP400",
                },
                {
                    "ticker": "BAD_TICKER",  # Very short membership
                    "start": pd.Timestamp("2023-01-01"),
                    "end": pd.Timestamp("2023-01-15"),
                    "index_name": "SP400",
                },
            ]
        )

        mock_build_membership.return_value = problematic_membership

        universe_config = UniverseConfig(universe_type="midcap400")
        universe_builder = UniverseBuilder(universe_config, str(tmp_path))
        survivorship_validator = SurvivorshipValidator(universe_config)

        # Generate universe calendar
        universe_calendar = universe_builder.generate_universe_calendar("2020-01-01", "2024-12-31")

        # Validate quality
        validation_results = universe_builder.validate_universe_calendar(universe_calendar)
        survivorship_report = survivorship_validator.generate_survivorship_report(
            problematic_membership, universe_calendar
        )

        # Check quality metrics
        assert "min_constituents" in validation_results
        assert "max_constituents" in validation_results
        assert "first_date" in validation_results
        assert "last_date" in validation_results

        # Check survivorship analysis
        assert "bias_score" in survivorship_report["overall_assessment"]
        assert "recommendations" in survivorship_report["overall_assessment"]

        # Should have recommendations due to data quality issues
        recommendations = survivorship_report["overall_assessment"]["recommendations"]
        assert len(recommendations) > 0
