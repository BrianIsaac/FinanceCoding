"""Data download and preparation utilities for portfolio GNN experiments.

This module:
- Loads the trading universe (from CSV or an inline list).
- Downloads adjusted prices and volumes via yfinance.
- Aligns calendars, forward-fills short gaps, and cleans sparse tickers.
- Computes daily log returns and month-end rebalance dates.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


def _load_universe(universe_csv: str | None, tickers_inline: list[str]) -> pd.DataFrame:
    """Loads the trading universe from CSV if present, otherwise from an inline list.

    Args:
        universe_csv: Path to a CSV containing at least a ``ticker`` column.
        tickers_inline: Fallback list of ticker symbols if the CSV is absent.

    Returns:
        A DataFrame with columns ``ticker`` and (optionally) other metadata
        such as ``sector``. Ticker symbols are upper-cased and stripped.

    Raises:
        ValueError: If the CSV exists but lacks a ``ticker`` column.
    """
    path = to_absolute_path(universe_csv) if universe_csv else ""
    if universe_csv and os.path.exists(path):
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        if "ticker" not in df.columns:
            raise ValueError("universe.csv must contain a 'ticker' column")
        df["ticker"] = df["ticker"].str.upper().str.strip()
        cols = ["ticker"] + [c for c in df.columns if c != "ticker"]
        return df[cols]
    # Fallback to inline list.
    return pd.DataFrame({"ticker": [t.upper() for t in tickers_inline], "sector": np.nan})


def _download_prices(tickers: list[str], start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Downloads adjusted close and volume for the given tickers via yfinance.

    Args:
        tickers: List of ticker symbols.
        start: ISO date (YYYY-MM-DD) for the start of the period.
        end: ISO date (YYYY-MM-DD) for the end of the period.

    Returns:
        Tuple of two DataFrames:
        - prices: Adjusted close prices with Date index and tickers as columns.
        - volume: Daily volumes aligned to prices.

    Notes:
        - ``auto_adjust=True`` is used, so splits/dividends are accounted for.
        - MultiIndex columns from yfinance are flattened into wide frames.
    """
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        closes, vols = [], []
        top = data.columns.get_level_values(0)
        for t in tickers:
            if t in top:
                closes.append(data[t]["Close"].rename(t))
                vols.append(data[t]["Volume"].rename(t))
        prices = pd.concat(closes, axis=1) if closes else pd.DataFrame()
        volume = pd.concat(vols, axis=1) if vols else pd.DataFrame()
    else:
        # Single-ticker fallback.
        prices = data["Close"].to_frame(tickers[0])
        volume = data["Volume"].to_frame(tickers[0])

    prices = prices[~prices.index.duplicated(keep="first")].sort_index()
    volume = volume.reindex(prices.index)
    return prices, volume


def _align_calendar(prices: pd.DataFrame, how: str = "union", ffill_limit: int = 5) -> pd.DataFrame:
    """Aligns all tickers onto a common trading calendar.

    Args:
        prices: Wide price DataFrame (Date index, tickers as columns).
        how: ``'union'`` forward-fills short gaps; ``'intersection'`` keeps only
            dates present for all tickers.
        ffill_limit: Maximum number of consecutive days to forward-fill per ticker
            after its first valid observation.

    Returns:
        A calendar-aligned price DataFrame.
    """
    if how == "intersection":
        mask = prices.notna().all(axis=1)
        return prices.loc[mask]

    aligned = prices.copy()
    for col in aligned.columns:
        first_valid = aligned[col].first_valid_index()
        if first_valid is not None:
            aligned.loc[first_valid:, col] = aligned.loc[first_valid:, col].ffill(limit=ffill_limit)
    return aligned


def _basic_clean(
    prices: pd.DataFrame, min_history_days: int, max_missing_ratio: float
) -> pd.DataFrame:
    """Cleans the price panel by history length and missing data thresholds.

    Args:
        prices: Calendar-aligned price DataFrame.
        min_history_days: Minimum non-NaN observations required to keep a ticker.
        max_missing_ratio: Maximum fraction of missing observations allowed.

    Returns:
        A cleaned price DataFrame with sparse tickers removed.
    """
    ok_history = prices.notna().sum() >= min_history_days
    cleaned = prices.loc[:, ok_history]

    miss_ratio = cleaned.isna().mean()
    cleaned = cleaned.loc[:, miss_ratio <= max_missing_ratio]

    cleaned = cleaned.dropna(how="all")
    return cleaned


def _compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Computes daily arithmetic returns.

    Args:
        prices: Cleaned price DataFrame.

    Returns:
        Daily log returns aligned to ``prices``.
    """
    return prices.pct_change()


def _make_rebalance_dates(prices: pd.DataFrame, freq: str = "M") -> pd.Series:
    """Gets the last trading day of each period as rebalance dates.

    Args:
        prices: Cleaned price DataFrame.
        freq: Pandas offset alias (e.g., ``'M'`` for month-end, ``'W'`` for weekly).

    Returns:
        Series of rebalance dates (DatetimeIndex) named ``rebalance_date``.
    """
    per = prices.index.to_period(freq)
    last = prices.groupby(per).apply(lambda d: d.index.max())
    rb = pd.DatetimeIndex(sorted(set(last)))
    return pd.Series(rb, name="rebalance_date")


def run_data_pipeline(cfg: DictConfig) -> dict[str, Any]:
    """Runs the full data pipeline and writes artefacts using Hydra paths.

    Args:
        cfg: Hydra configuration containing sections:
            - ``data``: start/end, universe_csv, tickers_inline.
            - ``cleaning``: min_history_days, max_missing_ratio, calendar, ffill_limit.
            - ``paths``: output locations for parquet/CSV files.
            - ``rebalance`` (optional): frequency key ``freq``.

    Returns:
        Dictionary containing:
            - ``universe``: Universe DataFrame.
            - ``prices``: Cleaned prices.
            - ``returns``: Daily log returns.
            - ``rebalance_dates``: Series of month-end (or chosen freq) dates.

    Side Effects:
        Writes interim and processed artefacts into the current Hydra run dir.
    """
    uni_df = _load_universe(cfg.data.universe_csv, cfg.data.tickers_inline)
    tickers = uni_df["ticker"].tolist()

    prices, volume = _download_prices(tickers, cfg.data.start, cfg.data.end)

    # Persist raw caches.
    os.makedirs(os.path.dirname(cfg.paths.prices_cache), exist_ok=True)
    prices.to_parquet(cfg.paths.prices_cache)
    volume.to_parquet(cfg.paths.volume_cache)

    # Align & clean.
    prices = _align_calendar(prices, cfg.cleaning.calendar, cfg.cleaning.ffill_limit)
    prices = _basic_clean(prices, cfg.cleaning.min_history_days, cfg.cleaning.max_missing_ratio)

    returns = _compute_daily_returns(prices)

    # Save processed artefacts.
    os.makedirs(os.path.dirname(cfg.paths.prices_processed), exist_ok=True)
    prices.to_parquet(cfg.paths.prices_processed)
    returns.to_parquet(cfg.paths.returns_processed)

    freq = cfg.rebalance.freq if "rebalance" in cfg else "M"
    rb = _make_rebalance_dates(prices, freq=freq)
    os.makedirs(os.path.dirname(cfg.paths.rebalance_csv), exist_ok=True)
    rb.to_frame().to_csv(cfg.paths.rebalance_csv, index=False)

    return {"universe": uni_df, "prices": prices, "returns": returns, "rebalance_dates": rb}
