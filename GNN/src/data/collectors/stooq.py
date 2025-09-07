"""Stooq data collector for stock price and volume data.

This module handles downloading OHLCV data from Stooq.com for US equities.
Extracted and refactored from scripts/download_stooq.py.
"""

from __future__ import annotations

import concurrent.futures as cf
import io
import time
from collections.abc import Iterable
from typing import Any

import pandas as pd
import requests

from src.config.data import CollectorConfig


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

    def _fetch_stooq_csv(self, symbol: str) -> pd.DataFrame | None:
        """Download daily CSV for a single Stooq symbol with enhanced error handling.

        Args:
            symbol: Stooq symbol like 'aapl.us'

        Returns:
            DataFrame with OHLCV data or None if fetch failed
        """
        url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"

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
                    time.sleep(self.config.rate_limit * 2)
                    continue

                if r.status_code == 403:  # Forbidden - might need session warmup
                    continue

                if (
                    r.status_code != 200
                    or not r.text
                    or r.text.strip().lower().startswith("<!doctype")
                    or "error" in r.text.lower()
                ):
                    continue

                df = pd.read_csv(io.StringIO(r.text))

                # Validate required columns
                required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
                if not all(col in df.columns for col in ["Date", "Close"]):
                    continue

                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

                # Validate data quality
                if df.empty or len(df) < 10:  # Minimum data points
                    continue

                # Return available OHLCV columns
                available_cols = [col for col in required_cols[1:] if col in df.columns]
                return df[available_cols]

            except requests.RequestException:
                if attempt < self.config.retry_attempts:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
            except Exception:
                if attempt < self.config.retry_attempts:
                    continue

        return None

    def _fetch_stooq_csv_with_rate_limit(self, symbol: str) -> pd.DataFrame | None:
        """Wrapper for _fetch_stooq_csv with global rate limiting.

        Args:
            symbol: Stooq symbol like 'aapl.us'

        Returns:
            DataFrame with OHLCV data or None if fetch failed
        """
        # Apply rate limiting before each request
        time.sleep(self.config.rate_limit)
        return self._fetch_stooq_csv(symbol)

    def collect_ohlcv_data(
        self,
        tickers: Iterable[str],
        start_date: str | None = None,
        end_date: str | None = None,
        max_workers: int = 8,
    ) -> dict[str, pd.DataFrame]:
        """Download full OHLCV data for multiple tickers and build wide panels.

        Args:
            tickers: Iterable of US ticker symbols
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            max_workers: Number of parallel download threads

        Returns:
            Dictionary with 'open', 'high', 'low', 'close', 'volume' DataFrames (Date × Tickers)
        """
        ohlcv_data: dict[str, dict[str, pd.Series]] = {
            "open": {},
            "high": {},
            "low": {},
            "close": {},
            "volume": {},
        }

        symbols = {t: self._to_stooq_symbol(t) for t in tickers}

        # Download in parallel with thread pool and rate limiting
        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            # Submit all download tasks
            futs = {
                ex.submit(self._fetch_stooq_csv_with_rate_limit, sym): tkr
                for tkr, sym in symbols.items()
            }

            # Process completed downloads
            for fut in cf.as_completed(futs):
                tkr = futs[fut]
                df = fut.result()

                if df is None or df.empty:
                    continue

                # Store all OHLCV components that are available
                if "Open" in df.columns:
                    ohlcv_data["open"][tkr] = df["Open"].rename(tkr)
                if "High" in df.columns:
                    ohlcv_data["high"][tkr] = df["High"].rename(tkr)
                if "Low" in df.columns:
                    ohlcv_data["low"][tkr] = df["Low"].rename(tkr)
                if "Close" in df.columns:
                    ohlcv_data["close"][tkr] = df["Close"].rename(tkr)
                if "Volume" in df.columns:
                    ohlcv_data["volume"][tkr] = df["Volume"].rename(tkr)

        if not ohlcv_data["close"]:
            return {k: pd.DataFrame() for k in ohlcv_data.keys()}

        # Combine into wide panels for each OHLCV component
        result = {}
        for component, series_dict in ohlcv_data.items():
            df = pd.concat(series_dict.values(), axis=1).sort_index()

            # Apply date filters if specified
            if start_date:
                df = df.loc[pd.to_datetime(start_date) :]
            if end_date:
                df = df.loc[: pd.to_datetime(end_date)]

            result[component] = df

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
            max_workers: Number of parallel download threads

        Returns:
            Tuple of (prices_df, volume_df) where each is Date × Tickers
        """
        ohlcv = self.collect_ohlcv_data(tickers, start_date, end_date, max_workers)
        return ohlcv["close"], ohlcv["volume"]

    def collect_single_ticker(self, ticker: str) -> pd.DataFrame | None:
        """Collect OHLCV data for a single ticker.

        Args:
            ticker: US ticker symbol

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        symbol = self._to_stooq_symbol(ticker)
        return self._fetch_stooq_csv(symbol)

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
        universe_builder,
        date_range: tuple[str, str] | None = None,
        max_workers: int = 8,
    ) -> dict[str, pd.DataFrame]:
        """Collect OHLCV data for S&P MidCap 400 universe with batch processing.

        Args:
            universe_builder: UniverseBuilder instance
            date_range: Optional (start_date, end_date) tuple
            max_workers: Number of parallel download threads

        Returns:
            Dictionary with full OHLCV data for universe constituents
        """
        if date_range:
            start_date, end_date = date_range
            universe_df = universe_builder.get_universe_for_period(start_date, end_date)
            tickers = sorted(universe_df["ticker"].unique())
        else:
            tickers = sorted(universe_builder.get_current_constituents())

        return self.collect_ohlcv_data(
            tickers=tickers,
            start_date=date_range[0] if date_range else None,
            end_date=date_range[1] if date_range else None,
            max_workers=max_workers,
        )

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
            max_workers: Parallel workers per batch
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

            batch_data = self.collect_ohlcv_data(batch, max_workers=max_workers)

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
