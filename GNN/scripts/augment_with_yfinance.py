#!/usr/bin/env python3
"""
Augment a Stooq price panel with Yahoo Finance (yfinance) to improve coverage.

What it does:
  1) Loads membership CSV and Stooq wide parquets (Close/Volume).
  2) Finds tickers missing in Stooq or too sparse (few non-NaNs).
  3) Downloads Yahoo OHLCV for the needed tickers (batched).
  4) Splices Yahoo series into Stooq:
       - If there is overlap, scales Yahoo segment to match Stooq level
         (median ratio on overlap) to avoid jumps.
       - Fills only where Stooq is NaN; Stooq remains the primary source.
  5) Writes merged parquets to an output directory.

Usage:
  python scripts/augment_with_yfinance.py ^
    --membership data/processed/universe_membership_wiki_sp400.csv ^
    --stooq-prices data/stooq/prices.parquet ^
    --stooq-volume data/stooq/volume.parquet ^
    --out-dir data/merged ^
    --start 2010-01-01 --end 2024-12-31

Notes:
  - yfinance fetch uses auto_adjust=True (adjusted prices). Stooq "Close" may not
    be identically adjusted, so we splice with a scaling factor at the overlap
    to reduce discontinuities.
  - If a ticker exists in Stooq but has gaps, those gaps can be filled from
    Yahoo. If there is no overlap window at all, we take Yahoo as-is.
"""

from __future__ import annotations

import argparse
import math
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


def _load_membership(path: str) -> List[str]:
    """Load and normalize tickers from the membership CSV."""
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        raise ValueError("membership CSV must contain a 'ticker' column")
    return sorted(set(df["ticker"].astype(str).str.upper().tolist()))


def _load_stooq(prices_path: str, volume_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Stooq wide parquets; ensure Date index and aligned rows."""
    px = pd.read_parquet(prices_path)
    vol = pd.read_parquet(volume_path)
    if not isinstance(px.index, pd.DatetimeIndex):
        px.index = pd.to_datetime(px.index, errors="coerce")
    if not isinstance(vol.index, pd.DatetimeIndex):
        vol.index = pd.to_datetime(vol.index, errors="coerce")
    px = px.sort_index().sort_index(axis=1)
    vol = vol.reindex(px.index).sort_index(axis=1)
    return px, vol


def _yahoo_symbol_map(ticker: str) -> str:
    """Map a US ticker to Yahoo symbol (dots → dashes for share classes)."""
    # Examples: BRK.B -> BRK-B, BF.B -> BF-B
    return ticker.strip().upper().replace(".", "-")


def _download_yahoo(
    tickers: List[str],
    start: Optional[str],
    end: Optional[str],
    batch_size: int = 80,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download Adjusted Close and Volume for tickers via yfinance in batches.

    Returns:
        prices_y: Date × Tickers (Adj Close)
        volume_y: Date × Tickers (Volume)
    """
    prices_list: List[pd.DataFrame] = []
    volume_list: List[pd.DataFrame] = []

    # yfinance works well with multi-ticker batches; we’ll do ~80 at a time.
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        syms = [_yahoo_symbol_map(t) for t in batch]

        data = yf.download(
            syms, start=start, end=end, auto_adjust=True, progress=False, group_by="ticker", threads=True
        )

        # Normalize to wide frames Date × Ticker
        if isinstance(data.columns, pd.MultiIndex):
            # Extract Adj Close (Close if auto_adjust=True) and Volume per ticker
            closes, vols = [], []
            top = data.columns.get_level_values(0)
            for sym, orig in zip(syms, batch):
                if sym in top:
                    # With auto_adjust=True, 'Close' is adjusted close
                    closes.append(data[sym]["Close"].rename(orig))
                    vols.append(data[sym]["Volume"].rename(orig))
            if closes:
                p = pd.concat(closes, axis=1)
                v = pd.concat(vols, axis=1)
                prices_list.append(p)
                volume_list.append(v)
        else:
            # Single-ticker fallback
            # NB: when a single symbol has no data, yfinance may return empty frame
            if not data.empty:
                orig = batch[0]
                p = data["Close"].to_frame(orig)
                v = data["Volume"].to_frame(orig)
                prices_list.append(p)
                volume_list.append(v)

    if prices_list:
        prices_y = pd.concat(prices_list, axis=1).sort_index()
        volume_y = pd.concat(volume_list, axis=1).reindex(prices_y.index)
        return prices_y, volume_y
    else:
        return pd.DataFrame(), pd.DataFrame()


def _splice_fill(primary: pd.Series, donor: pd.Series) -> pd.Series:
    """Fill NaNs in `primary` with `donor`, scaling donor at overlap to avoid jumps.

    Logic:
      - Find overlapping dates where both series have values.
      - Compute scaling factor = median(primary / donor) over overlap (donor!=0).
      - Multiply donor by factor before filling.
      - Fill only where primary is NaN. Return the combined series.

    If no overlap or donor all zeros/NaNs, returns primary with donor fill (no scaling).
    """
    s = primary.copy()
    d = donor.copy()

    # Align indices
    idx = s.index.union(d.index)
    s = s.reindex(idx)
    d = d.reindex(idx)

    overlap = s.notna() & d.notna() & (d != 0)
    if overlap.any():
        ratio = (s[overlap] / d[overlap]).replace([np.inf, -np.inf], np.nan).dropna()
        if not ratio.empty and ratio.median() != 0 and not math.isinf(ratio.median()):
            factor = ratio.median()
            d = d * factor

    # Fill only missing primary values
    s = s.where(s.notna(), d)
    return s


def _needs_yahoo(px: pd.DataFrame, universe: Iterable[str], min_non_na: int) -> List[str]:
    """Return tickers missing or too sparse in the Stooq panel."""
    px_cols = set(map(str, px.columns.tolist()))
    need: List[str] = []

    for t in universe:
        if t not in px_cols:
            need.append(t)
            continue
        non_na = int(px[t].notna().sum())
        if non_na < min_non_na:
            need.append(t)
    return sorted(need)


def _merge_panels(
    px_stooq: pd.DataFrame,
    vol_stooq: pd.DataFrame,
    px_yahoo: pd.DataFrame,
    vol_yahoo: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splice-fill Stooq with Yahoo on a per-ticker basis and return merged panels."""
    # Union index & columns
    idx = px_stooq.index.union(px_yahoo.index)
    cols = sorted(set(px_stooq.columns).union(set(px_yahoo.columns)))

    px_out = pd.DataFrame(index=idx, columns=cols, dtype=float)
    vol_out = pd.DataFrame(index=idx, columns=cols, dtype=float)

    # Pre-align
    px_s = px_stooq.reindex(index=idx, columns=cols)
    vol_s = vol_stooq.reindex(index=idx, columns=cols)
    px_y = px_yahoo.reindex(index=idx, columns=cols)
    vol_y = vol_yahoo.reindex(index=idx, columns=cols)

    for c in cols:
        s = px_s[c]
        d = px_y[c]
        px_out[c] = _splice_fill(s, d)

        # Volume: no scaling; just fill missing from Yahoo
        sv = vol_s[c]
        dv = vol_y[c]
        vol_out[c] = sv.where(sv.notna(), dv)

    # Sort for cleanliness
    px_out = px_out.sort_index().sort_index(axis=1)
    vol_out = vol_out.reindex(px_out.index).sort_index(axis=1)
    return px_out, vol_out


def main() -> None:
    ap = argparse.ArgumentParser(description="Augment Stooq panel with Yahoo Finance to improve coverage.")
    ap.add_argument("--membership", required=True, help="membership CSV (from Wikipedia scraper)")
    ap.add_argument("--stooq-prices", required=True, help="Stooq prices parquet (Date × Tickers, Close)")
    ap.add_argument("--stooq-volume", required=True, help="Stooq volume parquet (Date × Tickers, Volume)")
    ap.add_argument("--out-dir", required=True, help="Directory to write merged prices/volume parquets")
    ap.add_argument("--start", default=None, help="Optional ISO start date for Yahoo fetch (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="Optional ISO end date for Yahoo fetch (YYYY-MM-DD)")
    ap.add_argument("--min-non-na", type=int, default=60, help="Tickers with < this many data points will be fetched from Yahoo")
    ap.add_argument("--batch-size", type=int, default=80, help="yfinance multi-ticker batch size")
    args = ap.parse_args()

    universe = _load_membership(args.membership)
    px_s, vol_s = _load_stooq(args.stooq_prices, args.stooq_volume)

    need = _needs_yahoo(px_s, universe, args.min_non_na)
    print(f"Tickers in membership: {len(universe)}")
    print(f"Tickers already decent in Stooq: {len(set(universe) - set(need))}")
    print(f"Tickers to fetch from Yahoo: {len(need)}")

    if need:
        px_y, vol_y = _download_yahoo(need, start=args.start, end=args.end, batch_size=args.batch_size)
        if px_y.empty:
            print("No Yahoo data retrieved; writing out original Stooq panels.")
            px_out, vol_out = px_s, vol_s
        else:
            # Ensure columns use membership tickers (not Yahoo symbol variants)
            # Our loader already used membership tickers as column names.
            px_out, vol_out = _merge_panels(px_s, vol_s, px_y, vol_y)
    else:
        px_out, vol_out = px_s, vol_s

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    prices_path = f"{args.out_dir}/prices.parquet"
    volume_path = f"{args.out_dir}/volume.parquet"
    px_out.to_parquet(prices_path)
    vol_out.to_parquet(volume_path)

    # Simple stats
    stooq_cov = int(px_s.notna().sum().sum())
    merged_cov = int(px_out.notna().sum().sum())
    print(f"Saved merged prices  -> {prices_path}  shape={px_out.shape}")
    print(f"Saved merged volume  -> {volume_path}  shape={vol_out.shape}")
    print(f"Filled {merged_cov - stooq_cov:,} additional price cells using Yahoo.")
    print("Done.")
    

if __name__ == "__main__":
    main()
