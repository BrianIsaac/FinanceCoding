"""Stooq data collector for stock price and volume data.

This module handles downloading OHLCV data from Stooq.com for US equities.
Extracted and refactored from scripts/download_stooq.py.
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Iterable
from typing import Any

import pandas as pd
import requests

from src.config_schemas import CollectorConfig


class StooqCollector:
    """
    Collector for stock price and volume data from Stooq.com.

    Handles multi-threaded downloading of OHLCV data with session management
    and robust error handling for the Stooq public API.
    """

    def __init__(self, config: CollectorConfig):
        """
        Initialize Stooq collector.

        Args:
            config: Collector configuration with rate limits and timeouts
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _to_stooq_symbol(self, ticker: str) -> str:
        """Map a US ticker to Stooq symbol form.

        Args:
            ticker: US ticker symbol

        Returns:
            Stooq symbol (lowercase + .us, dots become dashes)

        Examples:
            'AAPL' -> 'aapl.us'
            'BRK.B' -> 'brk-b.us'
        """
        t = ticker.strip().upper().replace(".", "-")
        return f"{t.lower()}.us"

    def _new_session(self) -> requests.Session:
        """Create a requests Session with browser-like headers.

        Returns:
            Configured requests session
        """
        s = requests.Session()
        s.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"
                )
            }
        )
        return s

    def _prewarm_session(self, session: requests.Session) -> None:
        """Visit stooq.com homepage to set cookies before CSV fetches.

        Args:
            session: Session to prewarm
        """
        try:
            session.get("https://stooq.com/", timeout=min(10, self.config.timeout))
        except Exception:
            # Non-fatal: if it fails, we still try CSV endpoints
            pass

    def _fetch_stooq_csv(self, symbol: str) -> tuple[pd.DataFrame | None, str]:
        """Download daily CSV for a single Stooq symbol with enhanced error handling.

        Args:
            symbol: Stooq symbol like 'aapl.us'

        Returns:
            Tuple of (DataFrame with OHLCV data or None, failure reason string)
        """
        url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
        last_error = "Unknown error"

        for attempt in range(self.config.retry_attempts + 1):
            try:
                # Rate limiting between attempts
                if attempt > 0:
                    time.sleep(self.config.retry_delay * attempt)

                sess = self._new_session()
                self._prewarm_session(sess)
                r = sess.get(url, timeout=self.config.timeout)

                # Enhanced response validation
                if r.status_code == 429:  # Rate limited
                    last_error = "Rate limited (429)"
                    # Sleep longer when rate limited (2 seconds by default)
                    time.sleep(2.0 if self.config.rate_limit > 0 else 2.0)
                    continue

                if r.status_code == 403:  # Forbidden - might need session warmup
                    last_error = "Forbidden (403)"
                    continue

                if r.status_code != 200:
                    last_error = f"HTTP {r.status_code}"
                    continue

                if not r.text:
                    last_error = "Empty response"
                    continue

                if r.text.strip().lower().startswith("<!doctype"):
                    last_error = "HTML instead of CSV"
                    continue

                if "error" in r.text.lower():
                    last_error = "Error in response text"
                    continue

                df = pd.read_csv(io.StringIO(r.text))

                # Validate required columns
                required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
                if not all(col in df.columns for col in ["Date", "Close"]):
                    last_error = f"Missing required columns (have: {list(df.columns)})"
                    continue

                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

                # Validate data quality
                if df.empty:
                    last_error = "No valid dates after parsing"
                    continue

                if len(df) < 10:
                    last_error = f"Insufficient data points ({len(df)} < 10)"
                    continue

                # Return available OHLCV columns
                available_cols = [col for col in required_cols[1:] if col in df.columns]
                return df[available_cols], "Success"

            except requests.RequestException as e:
                last_error = f"Network error: {str(e)}"
                if attempt < self.config.retry_attempts:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
            except Exception as e:
                last_error = f"Parse error: {str(e)}"
                if attempt < self.config.retry_attempts:
                    continue

        return None, last_error

    def _fetch_stooq_csv_with_rate_limit(self, symbol: str) -> tuple[pd.DataFrame | None, str]:
        """Wrapper for _fetch_stooq_csv with global rate limiting.

        Args:
            symbol: Stooq symbol like 'aapl.us'

        Returns:
            Tuple of (DataFrame with OHLCV data or None, failure reason string)
        """
        # Apply rate limiting before each request (rate_limit = requests per second)
        if self.config.rate_limit > 0:
            time.sleep(1.0 / self.config.rate_limit)
        return self._fetch_stooq_csv(symbol)

    def _fetch_single_ticker_legacy_style(
        self, ticker: str, start_date: str | None = None, end_date: str | None = None
    ) -> pd.DataFrame | None:
        """Download data for single ticker using legacy approach with fresh session per request.

        Args:
            ticker: US ticker symbol
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        symbol = self._to_stooq_symbol(ticker)
        url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"

        for attempt in range(self.config.retry_attempts + 1):
            try:
                # Create fresh session for each request (legacy approach)
                sess = self._new_session()

                # Prewarm session by visiting homepage first
                try:
                    sess.get("https://stooq.com/", timeout=min(10, self.config.timeout))
                except Exception:
                    pass  # Non-fatal

                # Fetch the CSV data
                r = sess.get(url, timeout=self.config.timeout)

                # Check for rate limiting or HTML responses
                if (r.status_code != 200 or not r.text or
                    r.text.strip().lower().startswith("<!doctype") or
                    "exceeded the daily hits limit" in r.text.lower()):
                    if attempt < self.config.retry_attempts:
                        time.sleep(self.config.retry_delay * (attempt + 1))
                        continue
                    return None

                df = pd.read_csv(io.StringIO(r.text))

                if "Date" not in df.columns or "Close" not in df.columns:
                    continue

                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

                if df.empty:
                    continue

                # Return available OHLCV columns
                available_cols = [col for col in ["Open", "High", "Low", "Close", "Volume"]
                                if col in df.columns]
                return df[available_cols]

            except Exception:
                if attempt < self.config.retry_attempts:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue

        return None

    def check_rate_limit_status(self) -> bool:
        """Check if Stooq API is rate limited by testing with a single ticker.

        Returns:
            True if API is available, False if rate limited
        """
        try:
            # Test with a common ticker
            test_result = self._fetch_single_ticker_legacy_style("AAPL")
            return test_result is not None
        except Exception:
            return False

    def collect_ohlcv_data(
        self,
        ticker_periods: list[dict[str, str]],
    ) -> dict[str, pd.DataFrame]:
        """Collect OHLCV data for tickers with individual date ranges.

        Args:
            ticker_periods: List of dicts with keys:
                - ticker: Symbol to collect
                - start: Start date (YYYY-MM-DD) or None for all history
                - end: End date (YYYY-MM-DD) or None for present

        Returns:
            Dictionary with keys ['open', 'high', 'low', 'close', 'volume'],
            each containing a DataFrame with shape (Dates × Tickers)

        Note:
            Uses SEQUENTIAL processing to respect Stooq API rate limits.
            Parallel requests will trigger 429 errors and kill the API.
        """
        # Storage for results
        ohlcv_data: dict[str, dict[str, pd.Series]] = {
            "open": {},
            "high": {},
            "low": {},
            "close": {},
            "volume": {},
        }
        successful_tickers = []
        failed_tickers = []

        self.logger.info(f"Collecting data for {len(ticker_periods)} ticker-period combinations (SEQUENTIAL)")

        # Sequential iteration (NO ThreadPoolExecutor)
        for i, period in enumerate(ticker_periods, 1):
            ticker = period["ticker"]
            start_date = period.get("start")
            end_date = period.get("end")

            # Per-ticker progress logging
            self.logger.info(f"[{i}/{len(ticker_periods)}] Processing {ticker}...")

            try:
                # Fetch ticker data
                symbol = self._to_stooq_symbol(ticker)
                df, reason = self._fetch_stooq_csv(symbol)

                # Apply rate limiting AFTER receiving response
                if self.config.rate_limit > 0:
                    time.sleep(1.0 / self.config.rate_limit)

                if df is None or df.empty:
                    failed_tickers.append(ticker)
                    self.logger.info(f"  └─ Failed: {ticker} - {reason}")
                    continue

                # Apply per-ticker date filtering
                if start_date:
                    df = df.loc[pd.to_datetime(start_date) :]
                if end_date:
                    df = df.loc[: pd.to_datetime(end_date)]

                if df.empty:
                    self.logger.debug(f"{ticker}: No data in range {start_date} to {end_date}")
                    failed_tickers.append(ticker)
                    continue

                # Store OHLCV components
                if "Open" in df.columns:
                    ohlcv_data["open"][ticker] = df["Open"].rename(ticker)
                if "High" in df.columns:
                    ohlcv_data["high"][ticker] = df["High"].rename(ticker)
                if "Low" in df.columns:
                    ohlcv_data["low"][ticker] = df["Low"].rename(ticker)
                if "Close" in df.columns:
                    ohlcv_data["close"][ticker] = df["Close"].rename(ticker)
                if "Volume" in df.columns:
                    ohlcv_data["volume"][ticker] = df["Volume"].rename(ticker)

                successful_tickers.append(ticker)
                self.logger.info(f"  └─ Success: {ticker} ({len(df)} rows, {df.index.min().date()} to {df.index.max().date()})")

            except Exception as e:
                self.logger.info(f"  └─ Error: {ticker} - {e}")
                failed_tickers.append(ticker)

        if not ohlcv_data["close"]:
            self.logger.warning("No valid price data collected from any tickers")
            return {k: pd.DataFrame() for k in ohlcv_data.keys()}

        self.logger.info(
            f"Stooq collection completed: {len(successful_tickers)}/{len(ticker_periods)} tickers successful"
        )
        if failed_tickers:
            self.logger.warning(
                f"Failed tickers ({len(failed_tickers)}): {', '.join(failed_tickers[:10])}"
                f"{'...' if len(failed_tickers) > 10 else ''}"
            )

        # Combine into wide DataFrames
        result = {}
        for component, series_dict in ohlcv_data.items():
            if series_dict:
                df = pd.concat(series_dict.values(), axis=1).sort_index()
                result[component] = df
                self.logger.info(f"Built {component} DataFrame: {df.shape[1]} tickers × {df.shape[0]} dates")
            else:
                result[component] = pd.DataFrame()

        return result

    def collect_prices_and_volume(
        self,
        tickers: Iterable[str],
        start_date: str | None = None,
        end_date: str | None = None,
        max_workers: int = 8,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Download price and volume data (backward compatibility method).

        Args:
            tickers: Iterable of US ticker symbols
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            max_workers: Ignored (sequential processing used)

        Returns:
            Tuple of (prices_df, volume_df) where each is Date × Tickers
        """
        # Convert old signature to new ticker_periods format
        ticker_periods = [
            {"ticker": ticker, "start": start_date, "end": end_date}
            for ticker in tickers
        ]
        ohlcv = self.collect_ohlcv_data(ticker_periods)
        return ohlcv["close"], ohlcv["volume"]

    def collect_single_ticker(self, ticker: str) -> pd.DataFrame | None:
        """Collect OHLCV data for a single ticker.

        Args:
            ticker: US ticker symbol

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        symbol = self._to_stooq_symbol(ticker)
        df, _ = self._fetch_stooq_csv(symbol)  # Ignore reason for backwards compatibility
        return df

    def validate_data_coverage(
        self, prices_df: pd.DataFrame, required_tickers: list[str], min_data_points: int = 100
    ) -> dict[str, Any]:
        """Validate data coverage and quality.

        Args:
            prices_df: Prices DataFrame to validate
            required_tickers: List of tickers that should be present
            min_data_points: Minimum number of data points per ticker

        Returns:
            Dictionary with validation results
        """
        validation_results = {}

        # Basic coverage metrics
        validation_results["total_tickers"] = len(prices_df.columns)
        validation_results["date_range"] = (
            (prices_df.index.min(), prices_df.index.max()) if not prices_df.empty else (None, None)
        )
        validation_results["total_dates"] = len(prices_df.index)

        # Ticker coverage
        available_tickers = set(prices_df.columns)
        required_set = set(required_tickers)
        validation_results["missing_tickers"] = sorted(required_set - available_tickers)
        validation_results["extra_tickers"] = sorted(available_tickers - required_set)
        validation_results["coverage_ratio"] = (
            len(available_tickers & required_set) / len(required_set) if required_set else 0.0
        )

        # Data quality per ticker
        sparse_tickers = []
        for ticker in prices_df.columns:
            non_na_count = prices_df[ticker].notna().sum()
            if non_na_count < min_data_points:
                sparse_tickers.append(ticker)

        validation_results["sparse_tickers"] = sparse_tickers
        validation_results["sparse_count"] = len(sparse_tickers)

        return validation_results

    def collect_universe_data(
        self,
        universe_builder: Any,
        date_range: tuple[str, str] | None = None,
        max_workers: int = 8,
    ) -> dict[str, pd.DataFrame]:
        """Collect OHLCV data for S&P MidCap 400 universe with batch processing.

        Args:
            universe_builder: UniverseBuilder instance
            date_range: Optional (start_date, end_date) tuple
            max_workers: Ignored (sequential processing used)

        Returns:
            Dictionary with full OHLCV data for universe constituents
        """
        if date_range:
            start_date, end_date = date_range
            universe_df = universe_builder.get_universe_for_period(start_date, end_date)
            tickers = sorted(universe_df["ticker"].unique())
        else:
            tickers = sorted(universe_builder.get_current_constituents())

        # Convert to ticker_periods format
        ticker_periods = [
            {
                "ticker": ticker,
                "start": date_range[0] if date_range else None,
                "end": date_range[1] if date_range else None,
            }
            for ticker in tickers
        ]

        return self.collect_ohlcv_data(ticker_periods)

    def batch_collect_with_retry(
        self,
        tickers: list[str],
        batch_size: int = 50,
        max_workers: int = 8,
        max_retries: int = 3,
    ) -> dict[str, pd.DataFrame]:
        """Collect data in batches with comprehensive retry logic.

        Args:
            tickers: List of ticker symbols
            batch_size: Number of tickers per batch
            max_workers: Ignored (sequential processing used)
            max_retries: Maximum retry attempts for failed tickers

        Returns:
            Dictionary with full OHLCV data
        """
        all_data: dict[str, dict[str, pd.Series]] = {
            "open": {},
            "high": {},
            "low": {},
            "close": {},
            "volume": {},
        }
        failed_tickers = set()

        # Process in batches
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]

            # Convert to ticker_periods format
            ticker_periods = [{"ticker": ticker, "start": None, "end": None} for ticker in batch]
            batch_data = self.collect_ohlcv_data(ticker_periods)

            # Merge batch results
            for component in all_data.keys():
                if component in batch_data and not batch_data[component].empty:
                    for ticker in batch_data[component].columns:
                        all_data[component][ticker] = batch_data[component][ticker]

            # Track failed tickers
            successful_tickers = set(batch_data.get("close", pd.DataFrame()).columns)
            batch_failed = set(batch) - successful_tickers
            failed_tickers.update(batch_failed)

        # Retry failed tickers with individual downloads
        retry_count = 0
        while failed_tickers and retry_count < max_retries:
            retry_count += 1

            retry_tickers = list(failed_tickers)
            failed_tickers.clear()

            for ticker in retry_tickers:
                df = self.collect_single_ticker(ticker)
                if df is not None and not df.empty:
                    all_data["open"][ticker] = df["Open"].rename(ticker)
                    all_data["high"][ticker] = df["High"].rename(ticker)
                    all_data["low"][ticker] = df["Low"].rename(ticker)
                    all_data["close"][ticker] = df["Close"].rename(ticker)
                    all_data["volume"][ticker] = df["Volume"].rename(ticker)
                else:
                    failed_tickers.add(ticker)

        if failed_tickers:
            pass

        # Combine final results
        result = {}
        for component, series_dict in all_data.items():
            if series_dict:
                result[component] = pd.concat(series_dict.values(), axis=1).sort_index()
            else:
                result[component] = pd.DataFrame()

        return result
