#!/usr/bin/env python3
"""
Compute next-period (quarter-ahead) returns per ticker for each rebalance date.

- Reads: processed/returns_daily.parquet (Date x Ticker), processed/rebalance_dates.csv
- For each rebalance date t_i, aggregates daily log-returns from the NEXT business day
  up to (and including) the day BEFORE t_{i+1}. If t_{i+1} is missing, use the panel end.
- Writes: processed/labels/labels_YYYY-MM-DD.parquet (index=tickers, col='r_next')

Run:
  python scripts/make_labels.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROCESSED = Path("processed")
RET_PATH = PROCESSED / "returns_daily.parquet"
REB_PATH = PROCESSED / "rebalance_dates.csv"
OUT_DIR = PROCESSED / "labels"


def _next_trading_day(idx: pd.DatetimeIndex, d: pd.Timestamp) -> pd.Timestamp:
    # first date strictly greater than d
    pos = idx.searchsorted(d, side="right")
    if pos >= len(idx):
        return idx[-1]
    return idx[pos]


def _prev_trading_day(idx: pd.DatetimeIndex, d: pd.Timestamp) -> pd.Timestamp:
    # last date strictly less than d
    pos = idx.searchsorted(d, side="left") - 1
    if pos < 0:
        return idx[0]
    return idx[pos]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rets = pd.read_parquet(RET_PATH).sort_index()
    dates: pd.DatetimeIndex = rets.index
    rebal = pd.read_csv(REB_PATH, parse_dates=["rebalance_date"]).sort_values("rebalance_date")
    rb: list[pd.Timestamp] = rebal["rebalance_date"].tolist()

    for i, t in enumerate(rb):
        start = _next_trading_day(dates, t)
        if i + 1 < len(rb):
            # end (inclusive): last trading day strictly before next rebalance date
            end = _prev_trading_day(dates, rb[i + 1])
        else:
            end = dates[-1]
        if end < start:
            # no window (shouldn’t happen often)
            continue

        # arithmetic compounding over the forward window
        window = rets.loc[start:end].fillna(0.0)  # arithmetic daily returns
        r_next = (1.0 + window).prod(axis=0) - 1.0

        df = pd.DataFrame({"r_next": r_next.astype(float)})
        out = OUT_DIR / f"labels_{t.date()}.parquet"
        df.to_parquet(out)


if __name__ == "__main__":
    main()
