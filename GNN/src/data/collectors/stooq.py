"""Stooq data collector for stock price and volume data.

This module handles downloading OHLCV data from Stooq.com for US equities.
Extracted and refactored from scripts/download_stooq.py.
"""

from __future__ import annotations

import concurrent.futures as cf
import io
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

    def _fetch_stooq_csv(self, symbol: str) -> Optional[pd.DataFrame]:
        """Download daily CSV for a single Stooq symbol.

        Args:
            symbol: Stooq symbol like 'aapl.us'

        Returns:
            DataFrame with OHLCV data or None if fetch failed
        """
        url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"

        for attempt in range(self.config.retry_attempts + 1):
            try:
                sess = self._new_session()
                self._prewarm_session(sess)
                r = sess.get(url, timeout=self.config.timeout)

                if (
                    r.status_code != 200
                    or not r.text
                    or r.text.strip().lower().startswith("<!doctype")
                ):
                    continue

                df = pd.read_csv(io.StringIO(r.text))

                if "Date" not in df.columns or "Close" not in df.columns:
                    continue

                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

                return df[["Open", "High", "Low", "Close", "Volume"]]

            except Exception:
                # Try again if attempts left
                if attempt < self.config.retry_attempts:
                    continue

        return None

    def collect_ohlcv_data(
        self,
        tickers: Iterable[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_workers: int = 8,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Download OHLCV data for multiple tickers and build wide panels.

        Args:
            tickers: Iterable of US ticker symbols
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            max_workers: Number of parallel download threads

        Returns:
            Tuple of (prices_df, volume_df) where each is Date × Tickers
        """
        prices: Dict[str, pd.Series] = {}
        volumes: Dict[str, pd.Series] = {}

        symbols = {t: self._to_stooq_symbol(t) for t in tickers}

        # Download in parallel with thread pool
        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            # Submit all download tasks
            futs = {ex.submit(self._fetch_stooq_csv, sym): tkr for tkr, sym in symbols.items()}

            # Process completed downloads
            for fut in cf.as_completed(futs):
                tkr = futs[fut]
                df = fut.result()

                if df is None or df.empty:
                    print(f"[WARN] No data for {tkr}", file=sys.stderr)
                    continue

                prices[tkr] = df["Close"].rename(tkr)
                volumes[tkr] = df["Volume"].rename(tkr)

        if not prices:
            return pd.DataFrame(), pd.DataFrame()

        # Combine into wide panels
        px = pd.concat(prices.values(), axis=1).sort_index()
        vol = pd.concat(volumes.values(), axis=1).reindex(px.index)

        # Apply date filters if specified
        if start_date:
            px = px.loc[pd.to_datetime(start_date) :]
            vol = vol.loc[px.index]
        if end_date:
            px = px.loc[: pd.to_datetime(end_date)]
            vol = vol.loc[px.index]

        return px, vol

    def collect_single_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        """Collect OHLCV data for a single ticker.

        Args:
            ticker: US ticker symbol

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        symbol = self._to_stooq_symbol(ticker)
        return self._fetch_stooq_csv(symbol)

    def validate_data_coverage(
        self, prices_df: pd.DataFrame, required_tickers: List[str], min_data_points: int = 100
    ) -> Dict[str, Any]:
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
