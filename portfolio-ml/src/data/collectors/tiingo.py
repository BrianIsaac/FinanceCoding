"""Tiingo data collector for historical OHLCV data.

This module handles downloading historical price and volume data from Tiingo API.
Tiingo provides 30+ years of historical OHLCV data with excellent quality.

Free tier limitations:
- 50 requests/hour OR 1,000 requests/day
- 500 unique symbols per month (critical limitation)
- 5 GB data transfer per month
"""

from __future__ import annotations

import logging
import os
import time

import pandas as pd
from dotenv import load_dotenv
from tiingo import TiingoClient

from src.config.data import CollectorConfig

# Load environment variables from .env file
load_dotenv()


class TiingoCollector:
    """Collector for Tiingo financial data API.

    Tiingo provides 30+ years of historical OHLCV data with excellent quality.
    Free tier: 50 symbols/hour, 500 symbol lookups/month.

    Args:
        config: Collector configuration with rate limiting settings

    Example:
        >>> from dotenv import load_dotenv
        >>> load_dotenv()  # Loads TIINGO_API_KEY from .env
        >>> config = CollectorConfig(source_name='tiingo', rate_limit=72.0)
        >>> collector = TiingoCollector(config)
        >>> ticker_periods = [
        ...     {'ticker': 'AAPL', 'start': '2020-01-01', 'end': '2020-12-31'}
        ... ]
        >>> data = collector.collect_ohlcv_data(ticker_periods)
    """

    def __init__(self, config: CollectorConfig):
        """Initialise Tiingo collector.

        Args:
            config: Collector configuration

        Raises:
            ValueError: If TIINGO_API_KEY not found in environment
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Check for API key in environment
        api_key = os.getenv("TIINGO_API_KEY")
        if not api_key:
            raise ValueError(
                "TIINGO_API_KEY not found in environment. "
                "Please add it to .env file at project root."
            )

        # Initialise Tiingo client (reads API key from TIINGO_API_KEY env var)
        self.client = TiingoClient()
        self.logger.info("TiingoCollector initialised successfully")

    def collect_ohlcv_data(
        self,
        ticker_periods: list[dict[str, str]],
    ) -> dict[str, pd.DataFrame]:
        """Collect OHLCV data for tickers with individual date ranges.

        Args:
            ticker_periods: List of dicts with keys:
                - ticker: Symbol to collect
                - start: Start date (YYYY-MM-DD) or None
                - end: End date (YYYY-MM-DD) or None

        Returns:
            Dictionary with keys ['open', 'high', 'low', 'close', 'volume'],
            each containing a DataFrame with shape (Dates × Tickers)

        Note:
            Rate limit: 50 symbols/hour on free tier
            Uses sequential processing with rate limiting
            Free tier limited to 500 unique symbols per month
        """
        all_data: dict[str, dict[str, pd.Series]] = {
            "open": {},
            "high": {},
            "low": {},
            "close": {},
            "volume": {},
        }
        successful: list[str] = []
        failed: list[str] = []

        self.logger.info(
            f"Collecting data for {len(ticker_periods)} ticker-period combinations via Tiingo"
        )

        # Track unique symbols for monthly limit
        unique_symbols = {p["ticker"] for p in ticker_periods}
        if len(unique_symbols) > 500:
            self.logger.warning(
                f"Requesting {len(unique_symbols)} unique symbols exceeds "
                "free tier limit of 500/month. Some requests may fail."
            )
        elif len(unique_symbols) > 400:
            self.logger.warning(
                f"Requesting {len(unique_symbols)} unique symbols "
                f"({len(unique_symbols)/500*100:.1f}% of monthly limit)"
            )

        for i, period in enumerate(ticker_periods, 1):
            ticker = period["ticker"]
            start_date = period.get("start")
            end_date = period.get("end")

            # Progress logging
            if i % 25 == 0 or i == len(ticker_periods):
                progress_pct = (i / len(ticker_periods)) * 100
                self.logger.info(
                    f"Progress: {i}/{len(ticker_periods)} ({progress_pct:.1f}%) - "
                    f"{len(successful)} successful, {len(failed)} failed"
                )

            try:
                # Rate limiting (72 seconds = 50 symbols/hour)
                time.sleep(self.config.rate_limit)

                # Fetch data using Tiingo client
                df = self.client.get_dataframe(
                    ticker,
                    startDate=start_date,
                    endDate=end_date,
                    frequency="daily",
                )

                if df is None or df.empty:
                    self.logger.debug(f"{ticker}: No data returned")
                    failed.append(ticker)
                    continue

                # Tiingo returns multi-index DataFrame for single ticker
                # Reset index to get date as column
                if isinstance(df.index, pd.MultiIndex):
                    df = df.reset_index(level=0, drop=True)

                # Convert timezone-aware index to timezone-naive for compatibility
                # Tiingo returns UTC timezone, but we need naive for concat with other sources
                if hasattr(df.index, "tz") and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)

                # Store OHLCV data (Tiingo columns: open, high, low, close, volume)
                for field in ["open", "high", "low", "close", "volume"]:
                    if field in df.columns:
                        all_data[field][ticker] = df[field]

                successful.append(ticker)

            except Exception as e:
                self.logger.warning(f"{ticker}: Collection failed - {e}")
                failed.append(ticker)

        self.logger.info(
            f"Tiingo collection complete: {len(successful)}/{len(ticker_periods)} successful"
        )

        if failed:
            self.logger.warning(f"Failed tickers ({len(failed)}): {', '.join(failed[:10])}")
            if len(failed) > 10:
                self.logger.warning(f"... and {len(failed) - 10} more")

        # Combine into wide DataFrames
        result = {}
        for field in ["open", "high", "low", "close", "volume"]:
            if all_data[field]:
                result[field] = pd.DataFrame(all_data[field])
                self.logger.info(
                    f"Built {field} DataFrame: {result[field].shape[1]} tickers × "
                    f"{result[field].shape[0]} dates"
                )
            else:
                result[field] = pd.DataFrame()

        return result
