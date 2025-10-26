"""Wikipedia data collector for S&P index membership data.

This module handles scraping and processing of S&P index historical
membership data from Wikipedia to support dynamic universe management.
Extracted and refactored from scripts/build_membership_from_wikipedia.py.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from io import StringIO

import pandas as pd
import requests

from src.config.data import CollectorConfig

# Wikipedia URLs for different indices
WIKI_URLS: dict[str, str] = {
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
    added: list[str]
    removed: list[str]
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

    def _flatten_columns(self, df: pd.DataFrame) -> list[str]:
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

    def _scrape_current_constituents(self, html: str) -> list[str]:
        """Extract current constituents from the main constituents table.

        Args:
            html: Raw HTML from Wikipedia page

        Returns:
            List of current ticker symbols
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
            cols = self._flatten_columns(df)
            # Heuristic: a real constituents table usually has a symbol/ticker col
            # AND a company/sector col
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

    def _clean_cell_to_tickers(self, cell: str) -> list[str]:
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
        out: list[str] = []
        for tok in tokens:
            tok = tok.strip()
            if 1 <= len(tok) <= 6 and re.fullmatch(r"[A-Z0-9.\-]+", tok or ""):
                out.append(tok)
        return out

    def _extract_change_tables(self, html: str) -> list[pd.DataFrame]:
        """Extract change history tables from Wikipedia HTML.

        Args:
            html: Raw HTML from Wikipedia page

        Returns:
            List of DataFrames containing change history
        """
        all_tables = pd.read_html(StringIO(html), flavor="lxml")
        keep: list[pd.DataFrame] = []
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

    def _pick_col(self, cols: list[str], *candidates: str) -> str | None:
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
    ) -> list[ChangeEvent]:
        """Convert change tables to ChangeEvent objects.

        Args:
            tables: Iterable of change history DataFrames
            index_name: Name of the index (e.g., 'SP400')

        Returns:
            List of ChangeEvent objects sorted by date
        """
        events: list[ChangeEvent] = []
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
        self, events: list[ChangeEvent], end_cap: pd.Timestamp | None
    ) -> pd.DataFrame:
        """Convert ChangeEvents to membership intervals DataFrame.

        Args:
            events: List of ChangeEvent objects
            end_cap: Optional end date to close open intervals

        Returns:
            DataFrame with columns: ticker, start, end, index_name
        """
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
        self,
        index_key: str,
        end_cap: str | None = None,
        seed_current: bool = True,
        start_date: str | None = None,
    ) -> pd.DataFrame:
        """Build membership DataFrame using forward-only tracking.

        This method tracks membership changes forward from the earliest available
        change event, avoiding the "zombie ticker" problem that occurred with
        backward reconstruction.

        Strategy:
        1. Get current constituents (known to be active now)
        2. Track all change events forward from earliest available
        3. Only include tickers with verified activity in requested date range
        4. Mark tickers with unknown start dates for transparency

        Args:
            index_key: Index identifier ('sp400', 'sp500')
            end_cap: Optional ISO date string to close open intervals (legacy parameter)
            seed_current: Legacy parameter, kept for backwards compatibility
            start_date: Optional start of analysis period (YYYY-MM-DD). If None, uses earliest event.

        Returns:
            DataFrame with columns: ticker, start, end, index_name, start_verified

        Raises:
            ValueError: If index_key is not supported

        Example:
            >>> collector = WikipediaCollector(config)
            >>> membership = collector.build_membership(
            ...     index_key='sp400',
            ...     end_cap='2024-12-01',
            ...     start_date='2016-01-01'
            ... )
            >>> # Now produces ~400 constituents per snapshot, not 853
        """
        if index_key not in WIKI_URLS:
            raise ValueError(f"index must be one of: {list(WIKI_URLS.keys())}")

        # Parse dates
        start_ts = pd.to_datetime(start_date) if start_date else None
        end_ts = pd.to_datetime(end_cap) if end_cap else None

        # Fetch data
        html = self._fetch_html(WIKI_URLS[index_key])
        tables = self._extract_change_tables(html)
        events = self._tables_to_events(tables, index_name=index_key.upper())
        current_constituents = set(self._scrape_current_constituents(html))

        if not events:
            # No change history - just use current constituents
            return self._current_only_membership(
                current_constituents, index_key, start_ts, end_ts
            )

        earliest_event = events[0].date

        # Build membership using forward-only tracking
        return self._build_forward_only_intervals(
            events=events,
            current_constituents=current_constituents,
            earliest_event=earliest_event,
            start_date=start_ts,
            end_date=end_ts,
            index_name=index_key.upper(),
        )

    def _current_only_membership(
        self,
        current_constituents: set[str],
        index_key: str,
        start_date: pd.Timestamp | None,
        end_date: pd.Timestamp | None,
    ) -> pd.DataFrame:
        """Create membership DataFrame from current constituents only.

        Used when no change history is available.

        Args:
            current_constituents: Set of current ticker symbols
            index_key: Index identifier
            start_date: Start of analysis period
            end_date: Optional end date

        Returns:
            DataFrame with membership intervals
        """
        intervals = []
        for ticker in sorted(current_constituents):
            intervals.append({
                "ticker": ticker,
                "start": start_date if start_date else pd.Timestamp.now(),
                "end": end_date,
                "index_name": index_key.upper(),
                "start_verified": False,
            })

        return pd.DataFrame(intervals)

    def _build_forward_only_intervals(
        self,
        events: list[ChangeEvent],
        current_constituents: set[str],
        earliest_event: pd.Timestamp,
        start_date: pd.Timestamp | None,
        end_date: pd.Timestamp | None,
        index_name: str,
    ) -> pd.DataFrame:
        """Build membership intervals using forward-only tracking.

        Algorithm:
        1. Track additions/removals forward from earliest event
        2. For tickers added >= start_date: use actual addition date (verified)
        3. For current members never explicitly added: use earliest_event as proxy (unverified)
        4. Exclude tickers removed before start_date
        5. Close intervals at end_date if provided

        Args:
            events: Sorted list of ChangeEvent objects
            current_constituents: Set of current member tickers
            earliest_event: Date of first change event in history
            start_date: Start of analysis period
            end_date: Optional end date for closing intervals
            index_name: Index name for labelling

        Returns:
            DataFrame with columns: ticker, start, end, index_name, start_verified
        """
        # Track ticker lifecycles
        ticker_added: dict[str, pd.Timestamp] = {}  # ticker -> addition date
        ticker_removed: dict[str, pd.Timestamp] = {}  # ticker -> removal date
        tickers_ever_added: set[str] = set()

        # Process all change events
        for event in events:
            for ticker in event.added:
                tickers_ever_added.add(ticker)
                if ticker not in ticker_added:
                    # Record first addition date
                    ticker_added[ticker] = event.date

            for ticker in event.removed:
                # Record removal date (may have multiple, use latest)
                ticker_removed[ticker] = event.date

        # Build intervals
        intervals = []

        # 1. Process tickers with explicit addition dates
        for ticker, add_date in ticker_added.items():
            removal_date = ticker_removed.get(ticker)

            # Skip if ticker was removed before our analysis period
            if start_date and removal_date and removal_date < start_date:
                continue

            # Skip if ticker was added after our analysis period
            if end_date and add_date > end_date:
                continue

            intervals.append({
                "ticker": ticker,
                "start": add_date,
                "end": removal_date if removal_date else end_date,
                "index_name": index_name,
                "start_verified": True,
            })

        # 2. Process current constituents that were never explicitly added
        # These existed before earliest_event date
        for ticker in current_constituents:
            if ticker in ticker_added:
                # Already processed above
                continue

            # This ticker is currently active but has no addition record
            # It must have existed before earliest_event
            # Use earliest_event as proxy start date (unverified)

            removal_date = ticker_removed.get(ticker)

            # If it has a removal date, something is wrong (it's supposed to be current)
            if removal_date:
                # Skip - inconsistent data
                continue

            intervals.append({
                "ticker": ticker,
                "start": max(earliest_event, start_date) if start_date else earliest_event,
                "end": end_date,
                "index_name": index_name,
                "start_verified": False,  # We don't know actual start date
            })

        # 3. Handle tickers that were removed but never added (pre-2014 members)
        # These are the "zombie tickers" that cause the 853 problem
        pre_existing_removed = set(ticker_removed.keys()) - tickers_ever_added

        for ticker in pre_existing_removed:
            removal_date = ticker_removed[ticker]

            # Only include if removal happened during or after our analysis period
            if start_date and removal_date >= start_date:
                intervals.append({
                    "ticker": ticker,
                    "start": max(earliest_event, start_date),
                    "end": removal_date,
                    "index_name": index_name,
                    "start_verified": False,
                })
            elif not start_date:
                # No start_date filter, include all removals
                intervals.append({
                    "ticker": ticker,
                    "start": earliest_event,
                    "end": removal_date,
                    "index_name": index_name,
                    "start_verified": False,
                })

        # Convert to DataFrame and sort
        df = pd.DataFrame(intervals)

        if df.empty:
            return df

        df["ticker"] = df["ticker"].astype(str).str.upper()
        df["start"] = pd.to_datetime(df["start"])
        df["end"] = pd.to_datetime(df["end"], errors='coerce')  # May have None values

        df = df.sort_values(["ticker", "start"]).reset_index(drop=True)

        return df

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
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[ChangeEvent]:
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
