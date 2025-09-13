#!/usr/bin/env python3
"""
Coverage checker for your data pipeline.

Given:
  - a membership CSV (ticker,start,end,index_name) from the Wikipedia scraper
  - a wide prices parquet (Date x Tickers) from Stooq (after ingest/clean)
  - optional rebalance_dates.csv (one date per row)

It reports:
  - # tickers in membership vs # tickers in data, intersection, and missing
  - date range, # trading days
  - missingness stats in the price panel
  - per-rebalance active counts (membership at t) and covered counts (in data)
  - writes a CSV summary per rebalance if --out is provided

Usage:
  python scripts/check_coverage.py ^
    --membership data/processed/universe_membership_wiki_sp400.csv ^
    --prices processed/prices_clean.parquet ^
    --rebalances processed/rebalance_dates.csv ^
    --out processed/coverage_by_rebalance.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def _load_membership(path: str) -> pd.DataFrame:
    """Load membership intervals.

    Args:
        path: CSV path with columns ['ticker','start','end',...].

    Returns:
        DataFrame with parsed dates and uppercase tickers.
    """
    m = pd.read_csv(path)
    if "ticker" not in m.columns or "start" not in m.columns:
        raise ValueError("membership CSV must have at least 'ticker' and 'start' columns")
    # Normalize
    m["ticker"] = m["ticker"].astype(str).str.upper()
    m["start"] = pd.to_datetime(m["start"], errors="coerce")
    if "end" in m.columns:
        m["end"] = pd.to_datetime(m.get("end"), errors="coerce")
    else:
        m["end"] = pd.NaT
    return m


def _load_prices(path: str) -> pd.DataFrame:
    """Load wide price panel (Date × Tickers).

    Args:
        path: Parquet path with a DatetimeIndex and ticker columns.

    Returns:
        Sorted price DataFrame.
    """
    px = pd.read_parquet(path)
    if not isinstance(px.index, pd.DatetimeIndex):
        px.index = pd.to_datetime(px.index, errors="coerce")
    px = px.sort_index().sort_index(axis=1)
    return px


def _load_rebalances(path: str | None) -> pd.Series | None:
    """Load rebalance dates if provided.

    Args:
        path: CSV with a column 'rebalance_date'.

    Returns:
        Series of Timestamps or None.
    """
    if not path:
        return None
    df = pd.read_csv(path)
    col = "rebalance_date" if "rebalance_date" in df.columns else df.columns[0]
    s = pd.to_datetime(df[col], errors="coerce")
    s = s.dropna().sort_values().reset_index(drop=True)
    return s


def _membership_active_at(m: pd.DataFrame, ts: pd.Timestamp) -> set[str]:
    """Tickers active at timestamp ts per membership intervals.

    Args:
        m: Membership DataFrame.
        ts: Timestamp (rebalance date).

    Returns:
        Set of active tickers.
    """
    # Treat NaT end as open-ended
    mask = (m["start"] <= ts) & ((m["end"].isna()) | (m["end"] >= ts))
    return set(m.loc[mask, "ticker"].astype(str).tolist())


def _missingness_stats(px: pd.DataFrame) -> dict[str, float]:
    """Compute simple panel missingness stats.

    Args:
        px: Price panel.

    Returns:
        Dict with overall fraction, median per-column, and share of columns with any NaNs.
    """
    total = px.size
    n_missing = int(px.isna().sum().sum())
    frac_overall = n_missing / total if total > 0 else np.nan

    col_miss = px.isna().mean()
    median_per_col = float(col_miss.median()) if not col_miss.empty else np.nan
    share_cols_any = float((col_miss > 0).mean()) if not col_miss.empty else np.nan
    return {
        "overall_missing_frac": frac_overall,
        "median_missing_per_col": median_per_col,
        "share_columns_with_any_nans": share_cols_any,
    }


def _per_rebalance_coverage(
    m: pd.DataFrame,
    rebalances: pd.Series,
    available_tickers: set[str],
) -> pd.DataFrame:
    """Compute active vs covered counts at each rebalance date.

    Args:
        m: Membership DataFrame.
        rebalances: Series of Timestamps.
        available_tickers: Set of tickers present in the price panel.

    Returns:
        DataFrame with columns: ['rebalance_date','active','covered','coverage_ratio'].
    """
    rows: list[dict[str, object]] = []
    for ts in pd.to_datetime(rebalances):
        active = _membership_active_at(m, ts)
        covered = active & available_tickers
        rows.append(
            {
                "rebalance_date": ts,
                "active": len(active),
                "covered": len(covered),
                "coverage_ratio": (len(covered) / len(active)) if active else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values("rebalance_date").reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Check coverage of membership vs price panel.")
    parser.add_argument(
        "--membership", required=True, help="membership CSV (from Wikipedia scraper)"
    )
    parser.add_argument("--prices", required=True, help="processed/prices_clean.parquet")
    parser.add_argument(
        "--rebalances", default=None, help="processed/rebalance_dates.csv (optional)"
    )
    parser.add_argument(
        "--out", default=None, help="Write per-rebalance coverage CSV here (optional)"
    )
    parser.add_argument(
        "--top-missing",
        type=int,
        default=20,
        help="Show top-N membership tickers missing from data",
    )
    args = parser.parse_args()

    # Load inputs
    m = _load_membership(args.membership)
    px = _load_prices(args.prices)
    rb = _load_rebalances(args.rebalances)

    # Sets
    membership_set = set(m["ticker"].astype(str).tolist())
    data_set = set(map(str, px.columns.tolist()))

    # Basic counts
    len(membership_set)
    len(data_set)
    len(membership_set & data_set)
    missing = sorted(membership_set - data_set)

    # Date range
    str(px.index.min().date()) if not px.empty else "NA"
    str(px.index.max().date()) if not px.empty else "NA"
    px.shape[0]
    miss_stats = _missingness_stats(px)

    if missing:
        missing[: args.top_missing]

    for _k, _v in miss_stats.items():
        pass

    # Per-rebalance coverage
    if rb is not None and not rb.empty:
        cov = _per_rebalance_coverage(m, rb, data_set)
        if args.out:
            cov.to_csv(args.out, index=False)
    else:
        pass


if __name__ == "__main__":
    main()
