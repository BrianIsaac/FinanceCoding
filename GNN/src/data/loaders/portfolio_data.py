"""Portfolio data loading interfaces.

This module provides clean interfaces for loading financial data used in
portfolio optimization, including prices, volumes, and universe membership data.
Integrates with the refactored data collection and universe construction pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.config.data import UniverseConfig
from src.data.processors.universe_builder import UniverseBuilder


class PortfolioDataLoader:
    """
    Data loader for portfolio optimization datasets.

    Provides consistent interfaces for loading prices, volumes, universe
    membership data, and other market data required for portfolio optimization.
    Integrates with the dynamic universe construction framework.
    """

    def __init__(
        self, data_path: Optional[Path] = None, universe_config: Optional[UniverseConfig] = None
    ):
        """
        Initialize portfolio data loader.

        Args:
            data_path: Base path for data files
            universe_config: Universe configuration for dynamic loading
        """
        self.data_path = data_path or Path("data")
        self.universe_config = universe_config or UniverseConfig()

        # Create universe builder for dynamic data loading
        self.universe_builder = UniverseBuilder(
            self.universe_config, str(self.data_path / "processed")
        )

    def load_prices(self, source: str = "merged") -> pd.DataFrame:
        """
        Load price data for portfolio optimization.

        Args:
            source: Data source ("merged", "stooq", "yfinance")

        Returns:
            DataFrame with dates as index and tickers as columns
        """
        price_file = self.data_path / source / "prices.parquet"

        if price_file.exists():
            df = pd.read_parquet(price_file)
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            return df.sort_index()
        else:
            print(f"Price file not found: {price_file}")
            return pd.DataFrame()

    def load_volumes(self, source: str = "merged") -> pd.DataFrame:
        """
        Load volume data for portfolio analysis.

        Args:
            source: Data source ("merged", "stooq", "yfinance")

        Returns:
            DataFrame with dates as index and tickers as columns
        """
        volume_file = self.data_path / source / "volume.parquet"

        if volume_file.exists():
            df = pd.read_parquet(volume_file)
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            return df.sort_index()
        else:
            print(f"Volume file not found: {volume_file}")
            return pd.DataFrame()

    def load_membership_intervals(self, universe: str = "sp400") -> pd.DataFrame:
        """
        Load universe membership interval data.

        Args:
            universe: Universe identifier ("sp400", "sp500")

        Returns:
            DataFrame with ticker, start, end, index_name columns
        """
        membership_file = self.data_path / "processed" / f"membership_intervals_{universe}.parquet"

        if membership_file.exists():
            return pd.read_parquet(membership_file)
        else:
            print(f"Membership intervals file not found: {membership_file}")
            return pd.DataFrame(columns=["ticker", "start", "end", "index_name"])

    def load_universe_calendar(self, universe: str = "sp400") -> pd.DataFrame:
        """
        Load universe calendar with monthly snapshots.

        Args:
            universe: Universe identifier ("sp400", "sp500")

        Returns:
            DataFrame with date, ticker, index_name columns
        """
        calendar_file = self.data_path / "processed" / f"universe_calendar_{universe}.parquet"

        if calendar_file.exists():
            df = pd.read_parquet(calendar_file)
            # Ensure datetime column
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            return df.sort_values(["date", "ticker"]).reset_index(drop=True)
        else:
            print(f"Universe calendar file not found: {calendar_file}")
            return pd.DataFrame(columns=["date", "ticker", "index_name"])

    def load_market_data(
        self, source: str = "merged"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load complete market dataset for portfolio optimization.

        Args:
            source: Data source for prices and volumes

        Returns:
            Tuple of (prices, volumes, universe_calendar) DataFrames
        """
        prices = self.load_prices(source)
        volumes = self.load_volumes(source)
        universe_calendar = self.load_universe_calendar(self.universe_config.universe_type)

        return prices, volumes, universe_calendar

    def get_universe_at_date(self, date: str) -> List[str]:
        """
        Get universe members at a specific date.

        Args:
            date: Date in YYYY-MM-DD format

        Returns:
            List of ticker symbols active at the date
        """
        universe_calendar = self.load_universe_calendar(self.universe_config.universe_type)

        if universe_calendar.empty:
            return []

        date_ts = pd.to_datetime(date)

        # Find the most recent rebalancing date <= target date
        available_dates = universe_calendar[universe_calendar["date"] <= date_ts]["date"].unique()

        if len(available_dates) == 0:
            return []

        target_date = max(available_dates)
        universe_at_date = universe_calendar[universe_calendar["date"] == target_date]

        return sorted(universe_at_date["ticker"].tolist())

    def validate_data_alignment(
        self, prices: pd.DataFrame, volumes: pd.DataFrame, universe_calendar: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Validate alignment between different data sources.

        Args:
            prices: Price data DataFrame
            volumes: Volume data DataFrame
            universe_calendar: Universe calendar DataFrame

        Returns:
            Dictionary with validation results
        """
        validation = {}

        # Basic shape validation
        validation["prices_shape"] = prices.shape
        validation["volumes_shape"] = volumes.shape
        validation["calendar_records"] = len(universe_calendar)

        # Date alignment
        if not prices.empty and not volumes.empty:
            prices_dates = set(prices.index)
            volumes_dates = set(volumes.index)
            validation["date_alignment"] = prices_dates == volumes_dates
            validation["price_date_range"] = (prices.index.min(), prices.index.max())
            validation["volume_date_range"] = (volumes.index.min(), volumes.index.max())

        # Ticker alignment
        if not prices.empty and not volumes.empty:
            prices_tickers = set(prices.columns)
            volumes_tickers = set(volumes.columns)
            validation["ticker_alignment"] = prices_tickers == volumes_tickers
            validation["common_tickers"] = len(prices_tickers & volumes_tickers)
            validation["price_only_tickers"] = len(prices_tickers - volumes_tickers)
            validation["volume_only_tickers"] = len(volumes_tickers - prices_tickers)

        # Universe coverage
        if not universe_calendar.empty:
            universe_tickers = set(universe_calendar["ticker"].unique())
            if not prices.empty:
                price_tickers = set(prices.columns)
                validation["universe_coverage_in_prices"] = (
                    len(universe_tickers & price_tickers) / len(universe_tickers)
                    if universe_tickers
                    else 0.0
                )
                validation["missing_from_prices"] = sorted(list(universe_tickers - price_tickers))

        return validation

    def build_universe_if_missing(self, start_date: str, end_date: str) -> Path:
        """
        Build universe calendar if it doesn't exist.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Path to universe calendar file
        """
        calendar_file = (
            self.data_path
            / "processed"
            / f"universe_calendar_{self.universe_config.universe_type}.parquet"
        )

        if not calendar_file.exists():
            print(f"Universe calendar not found, building from Wikipedia...")
            return self.universe_builder.build_and_save_universe(start_date, end_date)
        else:
            return calendar_file
