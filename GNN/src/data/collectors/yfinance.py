"""Yahoo Finance data collector with splice-fill functionality.

This module handles downloading data from Yahoo Finance and provides
splice-fill methodology to merge with other data sources.
Extracted and refactored from scripts/augment_with_yfinance.py.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from src.config.data import CollectorConfig


class YFinanceCollector:
    """
    Collector for stock data from Yahoo Finance with splice-fill capabilities.

    Provides methods to download adjusted close and volume data, and merge
    with existing data sources using scaling to avoid discontinuities.
    """

    def __init__(self, config: CollectorConfig):
        """
        Initialize Yahoo Finance collector.

        Args:
            config: Collector configuration with rate limits and timeouts
        """
        self.config = config

    def _yahoo_symbol_map(self, ticker: str) -> str:
        """Map US ticker to Yahoo Finance symbol format.

        Args:
            ticker: US ticker symbol

        Returns:
            Yahoo Finance symbol (dots become dashes)

        Examples:
            'BRK.B' -> 'BRK-B'
            'BF.B' -> 'BF-B'
        """
        return ticker.strip().upper().replace(".", "-")

    def download_batch_data(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        batch_size: int = 80,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Download adjusted close and volume data for multiple tickers.

        Args:
            tickers: List of US ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            batch_size: Number of tickers to download per batch

        Returns:
            Tuple of (prices_df, volume_df) where each is Date × Tickers
        """
        prices_list: List[pd.DataFrame] = []
        volume_list: List[pd.DataFrame] = []

        # Process tickers in batches for better performance
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            syms = [self._yahoo_symbol_map(t) for t in batch]

            try:
                data = yf.download(
                    syms,
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                    progress=False,
                    group_by="ticker",
                    threads=True,
                )

                # Handle multi-ticker vs single-ticker response format
                if isinstance(data.columns, pd.MultiIndex):
                    # Multi-ticker response
                    closes, vols = [], []
                    top_level = data.columns.get_level_values(0)

                    for sym, orig in zip(syms, batch):
                        if sym in top_level:
                            # Extract adjusted close and volume
                            closes.append(data[sym]["Close"].rename(orig))
                            vols.append(data[sym]["Volume"].rename(orig))

                    if closes:
                        p = pd.concat(closes, axis=1)
                        v = pd.concat(vols, axis=1)
                        prices_list.append(p)
                        volume_list.append(v)

                else:
                    # Single-ticker response fallback
                    if not data.empty and len(batch) == 1:
                        orig = batch[0]
                        p = data["Close"].to_frame(orig)
                        v = data["Volume"].to_frame(orig)
                        prices_list.append(p)
                        volume_list.append(v)

            except Exception as e:
                print(f"[WARN] Failed to download batch starting at {batch[0]}: {e}")
                continue

        # Combine all batches
        if prices_list:
            prices_df = pd.concat(prices_list, axis=1).sort_index()
            volume_df = pd.concat(volume_list, axis=1).reindex(prices_df.index)
            return prices_df, volume_df
        else:
            return pd.DataFrame(), pd.DataFrame()

    def splice_fill_series(self, primary: pd.Series, donor: pd.Series) -> pd.Series:
        """Fill NaNs in primary series with donor series, scaling to avoid jumps.

        Args:
            primary: Primary data series (takes precedence where available)
            donor: Donor series to fill gaps (will be scaled to match primary)

        Returns:
            Combined series with gaps filled

        Logic:
            - Find overlapping dates where both series have values
            - Compute scaling factor = median(primary / donor) over overlap
            - Scale donor by this factor before filling
            - Fill only where primary is NaN
        """
        s = primary.copy()
        d = donor.copy()

        # Align indices
        idx = s.index.union(d.index)
        s = s.reindex(idx)
        d = d.reindex(idx)

        # Find overlap where both have valid, non-zero values
        overlap = s.notna() & d.notna() & (d != 0)

        if overlap.any():
            # Calculate scaling factor from overlap period
            ratio = (s[overlap] / d[overlap]).replace([np.inf, -np.inf], np.nan).dropna()

            if not ratio.empty and ratio.median() != 0 and not math.isinf(ratio.median()):
                factor = ratio.median()
                d = d * factor

        # Fill only missing primary values with scaled donor
        s = s.where(s.notna(), d)
        return s

    def merge_with_primary_source(
        self,
        primary_prices: pd.DataFrame,
        primary_volume: pd.DataFrame,
        yahoo_prices: pd.DataFrame,
        yahoo_volume: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Merge Yahoo Finance data with primary data source using splice-fill.

        Args:
            primary_prices: Primary source price data (e.g., from Stooq)
            primary_volume: Primary source volume data
            yahoo_prices: Yahoo Finance price data
            yahoo_volume: Yahoo Finance volume data

        Returns:
            Tuple of merged (prices_df, volume_df)
        """
        # Create union of indices and columns
        idx = primary_prices.index.union(yahoo_prices.index)
        cols = sorted(set(primary_prices.columns).union(set(yahoo_prices.columns)))

        # Initialize output DataFrames
        merged_prices = pd.DataFrame(index=idx, columns=cols, dtype=float)
        merged_volume = pd.DataFrame(index=idx, columns=cols, dtype=float)

        # Align all input DataFrames
        px_primary = primary_prices.reindex(index=idx, columns=cols)
        vol_primary = primary_volume.reindex(index=idx, columns=cols)
        px_yahoo = yahoo_prices.reindex(index=idx, columns=cols)
        vol_yahoo = yahoo_volume.reindex(index=idx, columns=cols)

        # Splice-fill each ticker individually
        for ticker in cols:
            # Prices: use splice-fill with scaling
            primary_series = px_primary[ticker]
            yahoo_series = px_yahoo[ticker]
            merged_prices[ticker] = self.splice_fill_series(primary_series, yahoo_series)

            # Volume: simple fill without scaling (volume scaling doesn't make sense)
            primary_vol = vol_primary[ticker]
            yahoo_vol = vol_yahoo[ticker]
            merged_volume[ticker] = primary_vol.where(primary_vol.notna(), yahoo_vol)

        return merged_prices, merged_volume

    def identify_missing_tickers(
        self, primary_prices: pd.DataFrame, universe_tickers: List[str], min_data_points: int = 100
    ) -> List[str]:
        """Identify tickers that are missing or sparse in primary data source.

        Args:
            primary_prices: Primary data source prices DataFrame
            universe_tickers: Complete list of tickers that should be available
            min_data_points: Minimum number of non-NA data points required

        Returns:
            List of tickers that need Yahoo Finance augmentation
        """
        primary_columns = set(map(str, primary_prices.columns.tolist()))
        needed_tickers = []

        for ticker in universe_tickers:
            if ticker not in primary_columns:
                # Completely missing
                needed_tickers.append(ticker)
            else:
                # Check if too sparse
                non_na_count = int(primary_prices[ticker].notna().sum())
                if non_na_count < min_data_points:
                    needed_tickers.append(ticker)

        return sorted(needed_tickers)

    def augment_data_source(
        self,
        primary_prices: pd.DataFrame,
        primary_volume: pd.DataFrame,
        universe_tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_data_points: int = 100,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Complete workflow to augment primary data source with Yahoo Finance.

        Args:
            primary_prices: Primary source price data
            primary_volume: Primary source volume data
            universe_tickers: Complete universe of tickers
            start_date: Start date for Yahoo Finance data
            end_date: End date for Yahoo Finance data
            min_data_points: Minimum data points threshold

        Returns:
            Tuple of augmented (prices_df, volume_df)
        """
        # Identify tickers needing Yahoo Finance data
        needed_tickers = self.identify_missing_tickers(
            primary_prices, universe_tickers, min_data_points
        )

        if not needed_tickers:
            print("No tickers need Yahoo Finance augmentation")
            return primary_prices, primary_volume

        print(f"Downloading Yahoo Finance data for {len(needed_tickers)} tickers")

        # Download Yahoo Finance data for needed tickers
        yahoo_prices, yahoo_volume = self.download_batch_data(needed_tickers, start_date, end_date)

        if yahoo_prices.empty:
            print("No Yahoo Finance data downloaded")
            return primary_prices, primary_volume

        # Merge using splice-fill methodology
        merged_prices, merged_volume = self.merge_with_primary_source(
            primary_prices, primary_volume, yahoo_prices, yahoo_volume
        )

        return merged_prices, merged_volume
