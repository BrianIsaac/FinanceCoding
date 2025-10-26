"""
Build a dynamic universe membership CSV from Wikipedia index change logs.

Fixes:
- Handles MultiIndex headers in the 'Selected changes' tables
  (e.g., 'Added' → ['Ticker','Security']).
- Wraps HTML in StringIO for pandas.read_html to avoid deprecation warnings.

Usage:
  python scripts/build_membership_from_wikipedia.py --index sp400 \
    --out data/processed/universe_membership_wiki_sp400.csv
  python scripts/build_membership_from_wikipedia.py --index sp500 \
    --out data/processed/universe_membership_wiki_sp500.csv
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from dataclasses import dataclass
from io import StringIO

import pandas as pd
import requests

WIKI_URLS: dict[str, str] = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
}

TICKER_RE = re.compile(r"\(([A-Z][A-Z0-9.\-]{0,9})\)")  # captures ABC, BRK.B, BF-B, etc.


@dataclass(frozen=True)
class ChangeEvent:
    date: pd.Timestamp
    added: list[str]
    removed: list[str]
    index_name: str  # "SP500" or "SP400"


def _new_session() -> requests.Session:
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


def _scrape_current_constituents(html: str) -> list[str]:
    """Extract current constituents from the main constituents table.

    Robust to:
      - MultiIndex headers (we flatten them)
      - Header variants: 'Symbol', 'Ticker symbol', 'Ticker'
      - Unicode dashes, footnote clutter
    """
    tables = pd.read_html(StringIO(html), flavor="lxml")
    best: list[str] | None = None

    def _pick_symbol_col(cols: list[str]) -> str | None:
        # prefer exact matches in this order, then any header containing 'symbol' or 'ticker'
        for exact in ("symbol", "ticker symbol", "ticker"):
            if exact in cols:
                return exact
        for c in cols:
            if "symbol" in c or "ticker" in c:
                return c
        return None

    for df in tables:
        cols = _flatten_columns(df)
        # Heuristic: a real constituents table usually has a symbol/ticker col
        # AND a company/sector col
        has_id = any(("symbol" in c) or ("ticker" in c) for c in cols)
        has_meta = any(
            ("security" in c) or ("company" in c) or ("gics" in c) or ("sector" in c) for c in cols
        )
        if not (has_id and has_meta):
            continue

        df = df.copy()
        df.columns = cols
        symcol = _pick_symbol_col(cols)
        if symcol is None or symcol not in df.columns:
            continue

        s = df[symcol].astype(str)

        # Clean up: remove spaces/footnotes, unify dashes, keep only safe ticker chars
        s = (
            s.str.upper()
            .str.replace(r"\s+", "", regex=True)
            .str.replace("–", "-", regex=False)
            .str.replace("—", "-", regex=False)
            .str.replace(r"\[.*?\]", "", regex=True)  # drop [1], [a], etc.
            .str.replace(r"\(.*?\)", "", regex=True)  # drop (...) notes
            .str.replace(r"[^A-Z0-9.\-]", "", regex=True)  # keep A-Z0-9.- only
        )
        syms = s[s.str.fullmatch(r"[A-Z0-9.\-]{1,6}")].tolist()
        uniq = sorted(set(syms))

        # Prefer the biggest plausible constituents table
        if len(uniq) >= 300:
            return uniq
        if best is None or len(uniq) > len(best):
            best = uniq

    return best or []


def _clean_cell_to_tickers(cell: str) -> list[str]:
    """Extract tickers from a cell: prefer '(TICKER)' patterns; fallback to token heuristics."""
    if cell is None:
        return []
    s = str(cell)
    m = TICKER_RE.findall(s.upper())
    if m:
        return [x.replace("–", "-").replace("—", "-").strip() for x in m]
    tokens = re.split(r"[,\;/\s]+", s.upper())
    out: list[str] = []
    for tok in tokens:
        tok = tok.strip()
        if 1 <= len(tok) <= 6 and re.fullmatch(r"[A-Z0-9.\-]+", tok or ""):
            out.append(tok)
    return out


def _fetch_html(url: str, session: requests.Session, timeout: int = 30) -> str:
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def _flatten_columns(df: pd.DataFrame) -> list[str]:
    """Return a list of normalized, lower-cased column names (handles MultiIndex)."""
    if isinstance(df.columns, pd.MultiIndex):
        flat = []
        for tup in df.columns:  # type: ignore[assignment]
            parts = [
                str(p).strip().lower()
                for p in tuple(tup)
                if p is not None and str(p).strip().lower() != "nan"
            ]
            flat.append(" ".join(parts))
        return flat
    else:
        return [str(c).strip().lower() for c in df.columns]


def _extract_change_tables(html: str) -> list[pd.DataFrame]:
    """Extract candidate change-history tables via pandas.read_html (MultiIndex-safe)."""
    all_tables = pd.read_html(StringIO(html), flavor="lxml")
    keep: list[pd.DataFrame] = []
    for df in all_tables:
        cols = _flatten_columns(df)
        # Must have a 'date' column and at least one 'added …' and one 'removed …' column.
        has_date = any(c.startswith("date") for c in cols)
        has_added = any(c.startswith("added") for c in cols)
        has_removed = any(c.startswith("removed") for c in cols)
        if has_date and has_added and has_removed:
            df.columns = cols  # set flattened names
            keep.append(df)
    return keep


def _pick_col(cols: list[str], *candidates: str) -> str | None:
    """Choose the first matching column name by prefix among candidates."""
    for cand in candidates:
        cand = cand.lower()
        for c in cols:
            if c == cand or c.startswith(cand):
                return c
    return None


def _tables_to_events(tables: Iterable[pd.DataFrame], index_name: str) -> list[ChangeEvent]:
    events: list[ChangeEvent] = []
    for tbl in tables:
        df = tbl.copy()
        cols = list(df.columns)

        date_col = _pick_col(cols, "date")
        # Prefer the explicit 'ticker' subcolumn if present (e.g., 'added ticker')
        added_col = _pick_col(cols, "added ticker", "added")
        removed_col = _pick_col(cols, "removed ticker", "removed")

        if date_col is None or added_col is None or removed_col is None:
            continue

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=False)
        df = df.dropna(subset=[date_col])

        # cells may be objects/NaN; cast to str
        df[added_col] = df[added_col].astype(str)
        df[removed_col] = df[removed_col].astype(str)

        for _, row in df.iterrows():
            added = _clean_cell_to_tickers(row[added_col])
            removed = _clean_cell_to_tickers(row[removed_col])
            if not added and not removed:
                continue
            events.append(
                ChangeEvent(
                    date=pd.Timestamp(row[date_col]),
                    added=added,
                    removed=removed,
                    index_name=index_name,
                )
            )

    events.sort(key=lambda e: e.date)
    return events


def _events_to_membership(events: list[ChangeEvent], end_cap: pd.Timestamp | None) -> pd.DataFrame:
    """Turn add/remove events into membership intervals per ticker."""
    start_map: dict[str, pd.Timestamp] = {}
    intervals: list[tuple[str, pd.Timestamp, pd.Timestamp | None, str]] = []

    for ev in events:
        for t in ev.added:
            start_map.setdefault(t, ev.date)
        for t in ev.removed:
            s = start_map.pop(t, None)
            if s is None:
                intervals.append((t, ev.date, ev.date, ev.index_name))
            else:
                intervals.append((t, s, ev.date, ev.index_name))

    for t, s in start_map.items():
        intervals.append((t, s, end_cap, events[-1].index_name if events else "UNKNOWN"))

    out = pd.DataFrame(intervals, columns=["ticker", "start", "end", "index_name"])
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out["start"] = pd.to_datetime(out["start"])
    if out["end"].notna().any():
        out["end"] = pd.to_datetime(out["end"])
    return out.sort_values(["ticker", "start"]).reset_index(drop=True)


def build_membership(
    index_key: str, end_cap: str | None, seed_current: bool = False
) -> pd.DataFrame:
    if index_key not in WIKI_URLS:
        raise ValueError("index must be one of: sp500, sp400")

    session = _new_session()
    html = _fetch_html(WIKI_URLS[index_key], session=session, timeout=30)
    tables = _extract_change_tables(html)
    events = _tables_to_events(tables, index_name=index_key.upper())
    cap = pd.to_datetime(end_cap) if end_cap else None

    if not seed_current or not events:
        # fallback: original behavior (no seeding)
        return _events_to_membership(events, end_cap=cap)

    # --- seeded method ---
    # 1) current full set
    current = set(_scrape_current_constituents(html))
    if not current:
        # if we couldn't find the table, fallback
        return _events_to_membership(events, end_cap=cap)

    # 2) roll BACKWARD to get roster at the earliest change date
    # Forward event means: at date d, Added A, Removed R
    # Backward step means: pre-d roster = (post-d roster - A) ∪ R
    roster = set(current)
    for ev in reversed(events):
        roster = (roster - set(ev.added)) | set(ev.removed)

    earliest = events[0].date

    # 3) roll FORWARD from earliest to now, creating intervals
    start_map: dict[str, pd.Timestamp] = dict.fromkeys(roster, earliest)
    intervals: list[tuple[str, pd.Timestamp, pd.Timestamp | None, str]] = []

    active = set(roster)
    for ev in events:
        d = ev.date

        # close intervals for names that are about to be removed (forward removal)
        for t in ev.removed:
            if t in active:
                # they are currently active; removal at d closes interval
                s = start_map.get(t, earliest)
                intervals.append((t, s, d, index_key.upper()))
                active.remove(t)
                start_map.pop(t, None)
            else:
                # removed but not currently active: ignore
                pass

        # open intervals for names that are about to be added (forward addition)
        for t in ev.added:
            if t not in active:
                active.add(t)
                start_map[t] = d  # start at the addition date

    # 4) close any open intervals at cap (or leave None if no cap)
    for t in sorted(active):
        s = start_map.get(t, earliest)
        intervals.append((t, s, cap, index_key.upper()))

    out = pd.DataFrame(intervals, columns=["ticker", "start", "end", "index_name"])
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out["start"] = pd.to_datetime(out["start"])
    if out["end"].notna().any():
        out["end"] = pd.to_datetime(out["end"])
    out = out.sort_values(["ticker", "start"]).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build dynamic membership from Wikipedia change logs."
    )
    parser.add_argument("--index", choices=["sp500", "sp400"], required=True)
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument(
        "--end-cap",
        default=None,
        help="Optional ISO date to close open intervals (e.g., 2025-08-10).",
    )
    parser.add_argument(
        "--seed-current",
        choices=["yes", "no"],
        default="yes",
        help=(
            "Seed membership from the current constituents table and roll changes "
            "backward/forward."
        ),
    )
    args = parser.parse_args()

    df = build_membership(args.index, args.end_cap, seed_current=(args.seed_current == "yes"))
    df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
