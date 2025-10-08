"""
Download daily OHLCV from Stooq for a set of tickers and write wide parquet panels.

This script reads tickers either from a dynamic membership CSV (ticker,start,end,index_name)
or from a command-line list, downloads daily CSVs from Stooq's public endpoint for each
ticker, and writes:
    - prices parquet: Date x Tickers (Close)
    - volume parquet: Date x Tickers (Volume)

Updates:
    - Uses requests.Session with a browser-like User-Agent.
    - "Pre-warms" the session by visiting https://stooq.com/ before CSV fetches,
      which avoids occasional 403/empty responses some users see when hitting the
      CSV endpoint directly.

Usage:
    # From membership (recommended)
    python scripts/download_stooq.py \
        --membership data/processed/universe_membership_wiki_sp400.csv \
        --out-dir data/stooq --start 2010-01-01 --end 2024-12-31

    # From explicit tickers
    python scripts/download_stooq.py --tickers AAPL,MSFT,NVDA,AMZN --out-dir data/stooq
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import os
from collections.abc import Iterable

import pandas as pd
import requests


def _to_stooq_symbol(ticker: str) -> str:
    """Map a US ticker to Stooq symbol form (lowercase + .us, '.' -> '-').

    Examples:
        'AAPL'  -> 'aapl.us'
        'BRK.B' -> 'brk-b.us'
        'BF.B'  -> 'bf-b.us'

    Args:
        ticker: Upper/lowercase US ticker.

    Returns:
        Stooq symbol.
    """
    t = ticker.strip().upper().replace(".", "-")
    return f"{t.lower()}.us"


def _new_session() -> requests.Session:
    """Create a requests Session with a browser-like User-Agent.

    Returns:
        A configured requests.Session instance.
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


def _prewarm_session(session: requests.Session, timeout: int = 15) -> None:
    """Visit stooq.com homepage to set any cookies before CSV fetches.

    Args:
        session: requests.Session to prewarm.
        timeout: Request timeout in seconds.
    """
    try:
        session.get("https://stooq.com/", timeout=timeout)
    except Exception:
        # Non-fatal: if it fails, we still try CSV endpoints.
        pass


def _fetch_stooq_csv(symbol: str, timeout: int = 20, retries: int = 2) -> pd.DataFrame | None:
    """Download the daily CSV for a single Stooq symbol with a fresh prewarmed session.

    Args:
        symbol: Stooq symbol like 'aapl.us'.
        timeout: HTTP timeout in seconds.
        retries: Number of retry attempts on failure.

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume; or None if not found.
    """
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"

    for _attempt in range(retries + 1):
        try:
            sess = _new_session()
            _prewarm_session(sess, timeout=min(10, timeout))
            r = sess.get(url, timeout=timeout)
            if r.status_code != 200 or not r.text or r.text.strip().lower().startswith("<!doctype"):
                continue
            df = pd.read_csv(io.StringIO(r.text))
            if "Date" not in df.columns or "Close" not in df.columns:
                continue
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
            return df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception:
            # try again if attempts left
            continue
    return None


def _load_membership_tickers(membership_csv: str) -> list[str]:
    """Load unique tickers from a membership CSV.

    Args:
        membership_csv: Path to CSV with column 'ticker'.

    Returns:
        Sorted unique tickers.
    """
    df = pd.read_csv(membership_csv)
    if "ticker" not in df.columns:
        raise ValueError("membership CSV must contain a 'ticker' column")
    return sorted(set(df["ticker"].astype(str).str.upper().tolist()))


def _collect_panels(
    tickers: Iterable[str],
    start: str | None,
    end: str | None,
    max_workers: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download OHLCV per ticker and build wide Close/Volume panels.

    Args:
        tickers: Iterable of ticker symbols (US).
        start: Optional ISO start date filter.
        end: Optional ISO end date filter.
        max_workers: Thread pool size.

    Returns:
        Tuple (prices, volume) where each is Date × Tickers.
    """
    prices: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}

    symbols = {t: _to_stooq_symbol(t) for t in tickers}

    # Download in parallel; each task uses its own (prewarmed) Session.
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_fetch_stooq_csv, sym): tkr for tkr, sym in symbols.items()}
        for fut in cf.as_completed(futs):
            tkr = futs[fut]
            df = fut.result()
            if df is None or df.empty:
                continue
            prices[tkr] = df["Close"].rename(tkr)
            volumes[tkr] = df["Volume"].rename(tkr)

    if not prices:
        return pd.DataFrame(), pd.DataFrame()

    px = pd.concat(prices.values(), axis=1).sort_index()
    vol = pd.concat(volumes.values(), axis=1).reindex(px.index)

    if start:
        px = px.loc[pd.to_datetime(start) :]
        vol = vol.loc[px.index]
    if end:
        px = px.loc[: pd.to_datetime(end)]
        vol = vol.loc[px.index]

    return px, vol


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download daily OHLCV from Stooq and write wide parquet panels."
    )
    parser.add_argument("--membership", help="CSV with 'ticker' column (from Wikipedia scraper).")
    parser.add_argument(
        "--tickers",
        help="Comma-separated tickers (e.g., AAPL,MSFT,NVDA). Used if --membership is absent.",
    )
    parser.add_argument(
        "--out-dir", required=True, help="Directory to write prices.parquet and volume.parquet"
    )
    parser.add_argument("--start", default=None, help="ISO start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="ISO end date (YYYY-MM-DD)")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel downloads")
    args = parser.parse_args()

    if args.membership:
        tickers = _load_membership_tickers(args.membership)
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        raise SystemExit("Provide either --membership or --tickers")

    prices, volume = _collect_panels(tickers, args.start, args.end, args.max_workers)
    if prices.empty:
        raise SystemExit("No data collected from Stooq. Check tickers or try a smaller set.")

    os.makedirs(args.out_dir, exist_ok=True)
    prices.to_parquet(f"{args.out_dir}/prices.parquet")
    volume.to_parquet(f"{args.out_dir}/volume.parquet")


if __name__ == "__main__":
    main()
