"""Wikipedia data collector for S&P index membership data.

This module handles scraping and processing of S&P index historical
membership data from Wikipedia to support dynamic universe management.
Extracted and refactored from scripts/build_membership_from_wikipedia.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import StringIO
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

from src.config.data import CollectorConfig

# Wikipedia URLs for different indices
WIKI_URLS: Dict[str, str] = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
}

# Regex pattern for ticker extraction from Wikipedia text
TICKER_RE = re.compile(r"\(([A-Z][A-Z0-9.\-]{0,9})\)")  # captures ABC, BRK.B, BF-B, etc.


@dataclass(frozen=True)
class ChangeEvent:
    """Represents a membership change event from Wikipedia scraping.

    Attributes:
        date: Date when the change occurred
        added: List of tickers that were added
        removed: List of tickers that were removed
        index_name: Name of the index (SP500, SP400, etc.)
    """

    date: pd.Timestamp
    added: List[str]
    removed: List[str]
    index_name: str


class WikipediaCollector:
    """
    Collector for S&P index membership data from Wikipedia.

    Handles web scraping, HTML parsing, and processing of historical
    membership information to support time-varying universe construction.
    """

    def __init__(self, config: CollectorConfig):
        """
        Initialize Wikipedia collector.

        Args:
            config: Collector configuration with rate limits and timeouts
        """
        self.config = config
        self.session = self._new_session()

    def _new_session(self) -> requests.Session:
        """Create a new requests session with proper headers.

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

    def _fetch_html(self, url: str) -> str:
        """Fetch HTML content from Wikipedia URL.

        Args:
            url: Wikipedia URL to fetch

        Returns:
            HTML content as string

        Raises:
            requests.HTTPError: If request fails
        """
        r = self.session.get(url, timeout=self.config.timeout)
        r.raise_for_status()
        return r.text

    def _flatten_columns(self, df: pd.DataFrame) -> List[str]:
        """Return normalized, lower-cased column names (handles MultiIndex).

        Args:
            df: DataFrame with potentially MultiIndex columns

        Returns:
            List of flattened column names
        """
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

    def _scrape_current_constituents(self, html: str) -> List[str]:
        """Extract current constituents from the main constituents table.

        Args:
            html: Raw HTML from Wikipedia page

        Returns:
            List of current ticker symbols
        """
        tables = pd.read_html(StringIO(html), flavor="lxml")
        best: List[str] | None = None

        def _pick_symbol_col(cols: List[str]) -> Optional[str]:
            # prefer exact matches in this order, then any header containing 'symbol' or 'ticker'
            for exact in ("symbol", "ticker symbol", "ticker"):
                if exact in cols:
                    return exact
            for c in cols:
                if "symbol" in c or "ticker" in c:
                    return c
            return None

        for df in tables:
            cols = self._flatten_columns(df)
            # Heuristic: a real constituents table usually has a symbol/ticker col AND a company/sector col
            has_id = any(("symbol" in c) or ("ticker" in c) for c in cols)
            has_meta = any(
                ("security" in c) or ("company" in c) or ("gics" in c) or ("sector" in c)
                for c in cols
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

    def _clean_cell_to_tickers(self, cell: str) -> List[str]:
        """Extract tickers from a table cell.

        Args:
            cell: Cell content as string

        Returns:
            List of extracted ticker symbols
        """
        if cell is None:
            return []
        s = str(cell)
        m = TICKER_RE.findall(s.upper())
        if m:
            return [x.replace("–", "-").replace("—", "-").strip() for x in m]
        tokens = re.split(r"[,\;/\s]+", s.upper())
        out: List[str] = []
        for tok in tokens:
            tok = tok.strip()
            if 1 <= len(tok) <= 6 and re.fullmatch(r"[A-Z0-9.\-]+", tok or ""):
                out.append(tok)
        return out

    def _extract_change_tables(self, html: str) -> List[pd.DataFrame]:
        """Extract change history tables from Wikipedia HTML.

        Args:
            html: Raw HTML from Wikipedia page

        Returns:
            List of DataFrames containing change history
        """
        all_tables = pd.read_html(StringIO(html), flavor="lxml")
        keep: List[pd.DataFrame] = []
        for df in all_tables:
            cols = self._flatten_columns(df)
            # Must have a 'date' column and at least one 'added …' and one 'removed …' column.
            has_date = any(c.startswith("date") for c in cols)
            has_added = any(c.startswith("added") for c in cols)
            has_removed = any(c.startswith("removed") for c in cols)
            if has_date and has_added and has_removed:
                df.columns = cols  # set flattened names
                keep.append(df)
        return keep

    def _pick_col(self, cols: List[str], *candidates: str) -> Optional[str]:
        """Choose the first matching column name by prefix among candidates.

        Args:
            cols: List of available column names
            *candidates: Candidate column names to search for

        Returns:
            First matching column name or None
        """
        for cand in candidates:
            cand = cand.lower()
            for c in cols:
                if c == cand or c.startswith(cand):
                    return c
        return None

    def _tables_to_events(
        self, tables: Iterable[pd.DataFrame], index_name: str
    ) -> List[ChangeEvent]:
        """Convert change tables to ChangeEvent objects.

        Args:
            tables: Iterable of change history DataFrames
            index_name: Name of the index (e.g., 'SP400')

        Returns:
            List of ChangeEvent objects sorted by date
        """
        events: List[ChangeEvent] = []
        for tbl in tables:
            df = tbl.copy()
            cols = list(df.columns)

            date_col = self._pick_col(cols, "date")
            # Prefer the explicit 'ticker' subcolumn if present (e.g., 'added ticker')
            added_col = self._pick_col(cols, "added ticker", "added")
            removed_col = self._pick_col(cols, "removed ticker", "removed")

            if date_col is None or added_col is None or removed_col is None:
                continue

            df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=False)
            df = df.dropna(subset=[date_col])

            # cells may be objects/NaN; cast to str
            df[added_col] = df[added_col].astype(str)
            df[removed_col] = df[removed_col].astype(str)

            for _, row in df.iterrows():
                added = self._clean_cell_to_tickers(row[added_col])
                removed = self._clean_cell_to_tickers(row[removed_col])
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

    def _events_to_membership(
        self, events: List[ChangeEvent], end_cap: Optional[pd.Timestamp]
    ) -> pd.DataFrame:
        """Convert ChangeEvents to membership intervals DataFrame.

        Args:
            events: List of ChangeEvent objects
            end_cap: Optional end date to close open intervals

        Returns:
            DataFrame with columns: ticker, start, end, index_name
        """
        start_map: Dict[str, pd.Timestamp] = {}
        intervals: List[Tuple[str, pd.Timestamp, Optional[pd.Timestamp], str]] = []

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
        self, index_key: str, end_cap: Optional[str] = None, seed_current: bool = True
    ) -> pd.DataFrame:
        """Build complete membership DataFrame from Wikipedia data.

        Args:
            index_key: Index identifier ('sp400', 'sp500')
            end_cap: Optional ISO date string to close open intervals
            seed_current: Whether to seed from current constituents and roll backward

        Returns:
            DataFrame with membership intervals

        Raises:
            ValueError: If index_key is not supported
        """
        if index_key not in WIKI_URLS:
            raise ValueError(f"index must be one of: {list(WIKI_URLS.keys())}")

        html = self._fetch_html(WIKI_URLS[index_key])
        tables = self._extract_change_tables(html)
        events = self._tables_to_events(tables, index_name=index_key.upper())
        cap = pd.to_datetime(end_cap) if end_cap else None

        if not seed_current or not events:
            # fallback: original behavior (no seeding)
            return self._events_to_membership(events, end_cap=cap)

        # Seeded method: roll backward from current, then forward
        current = set(self._scrape_current_constituents(html))
        if not current:
            return self._events_to_membership(events, end_cap=cap)

        # Roll BACKWARD to get roster at the earliest change date
        roster = set(current)
        for ev in reversed(events):
            roster = (roster - set(ev.added)) | set(ev.removed)

        earliest = events[0].date

        # Roll FORWARD from earliest to now, creating intervals
        start_map: Dict[str, pd.Timestamp] = {t: earliest for t in roster}
        intervals: List[Tuple[str, pd.Timestamp, Optional[pd.Timestamp], str]] = []

        active = set(roster)
        for ev in events:
            d = ev.date

            # close intervals for names that are about to be removed
            for t in ev.removed:
                if t in active:
                    s = start_map.get(t, earliest)
                    intervals.append((t, s, d, index_key.upper()))
                    active.remove(t)
                    start_map.pop(t, None)

            # open intervals for names that are about to be added
            for t in ev.added:
                if t not in active:
                    active.add(t)
                    start_map[t] = d

        # close any open intervals at cap (or leave None if no cap)
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

    def collect_current_membership(self, index_key: str = "sp400") -> pd.DataFrame:
        """Collect current index membership data.

        Args:
            index_key: Index identifier ('sp400', 'sp500')

        Returns:
            DataFrame with current ticker symbols
        """
        if index_key not in WIKI_URLS:
            raise ValueError(f"index must be one of: {list(WIKI_URLS.keys())}")

        html = self._fetch_html(WIKI_URLS[index_key])
        tickers = self._scrape_current_constituents(html)

        return pd.DataFrame({"ticker": tickers, "index_name": index_key.upper()})

    def collect_historical_changes(
        self,
        index_key: str = "sp400",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[ChangeEvent]:
        """Collect historical membership changes.

        Args:
            index_key: Index identifier ('sp400', 'sp500')
            start_date: Optional start date filter (ISO format)
            end_date: Optional end date filter (ISO format)

        Returns:
            List of ChangeEvent objects
        """
        if index_key not in WIKI_URLS:
            raise ValueError(f"index must be one of: {list(WIKI_URLS.keys())}")

        html = self._fetch_html(WIKI_URLS[index_key])
        tables = self._extract_change_tables(html)
        events = self._tables_to_events(tables, index_name=index_key.upper())

        # Apply date filters if provided
        if start_date:
            start_ts = pd.to_datetime(start_date)
            events = [e for e in events if e.date >= start_ts]
        if end_date:
            end_ts = pd.to_datetime(end_date)
            events = [e for e in events if e.date <= end_ts]

        return events
