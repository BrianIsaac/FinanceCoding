"""Universe construction engine for dynamic S&P index membership.

This module processes Wikipedia membership data to create time-varying
universe snapshots for backtesting and portfolio optimization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config_schemas import UniverseConfig, create_collector_config
from src.data.collectors.wikipedia import WikipediaCollector


class UniverseBuilder:
    """
    Dynamic universe construction engine for S&P indices.

    Manages time-varying universe membership by processing Wikipedia
    historical change data and generating monthly snapshots for backtesting.
    """

    def __init__(
        self,
        universe_config: UniverseConfig,
        output_dir: str = "data/processed",
        strict_validation: bool = True,
    ):
        """Initialise universe builder.

        Args:
            universe_config: Universe configuration specifying type and parameters
            output_dir: Directory for processed universe data
            strict_validation: Whether to enforce strict validation (raise on anomalies)
        """
        self.universe_config = universe_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.strict_validation = strict_validation

        # Create Wikipedia collector for membership data
        collector_config = create_collector_config("wikipedia")
        self.wikipedia_collector = WikipediaCollector(collector_config)

        # Map universe types to Wikipedia index keys
        self._universe_map = {
            "midcap400": "sp400",
            "sp500": "sp500",
        }

    def _get_index_key(self) -> str:
        """Get Wikipedia index key for universe type.

        Returns:
            Index key for Wikipedia collector

        Raises:
            ValueError: If universe type is not supported
        """
        if self.universe_config.universe_type not in self._universe_map:
            raise ValueError(
                f"Universe type '{self.universe_config.universe_type}' not supported. "
                f"Available types: {list(self._universe_map.keys())}"
            )
        return self._universe_map[self.universe_config.universe_type]

    def build_membership_intervals(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Build membership intervals DataFrame from Wikipedia data.

        Uses forward-only tracking algorithm to avoid "zombie ticker" issue
        where pre-2014 members are incorrectly assigned start=2014-09-30.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with columns: ticker, start, end, index_name, start_verified
        """
        index_key = self._get_index_key()

        # Use refactored build_membership with start_date filtering
        membership_df = self.wikipedia_collector.build_membership(
            index_key=index_key,
            end_cap=end_date,
            start_date=start_date,  # New parameter for filtering
        )

        return membership_df

    def create_monthly_snapshots(
        self, membership_df: pd.DataFrame, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Create monthly universe snapshots from membership intervals.

        Args:
            membership_df: DataFrame with membership intervals
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with columns: date, ticker, index_name
        """
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)

        # Generate monthly dates (first business day of each month)
        monthly_dates = pd.bdate_range(
            start=start_ts.replace(day=1), end=end_ts, freq="BMS"  # Business month start
        )

        snapshots = []

        for date in monthly_dates:
            # Find tickers active on this date
            active_mask = (membership_df["start"] <= date) & (
                (membership_df["end"].isna()) | (membership_df["end"] > date)
            )
            active_tickers = membership_df[active_mask]["ticker"].unique()

            # Apply universe filters if specified
            filtered_tickers = self._apply_universe_filters(active_tickers, date)

            # Create snapshot records
            for ticker in filtered_tickers:
                snapshots.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "index_name": (
                            membership_df["index_name"].iloc[0]
                            if len(membership_df) > 0
                            else self.universe_config.universe_type.upper()
                        ),
                    }
                )

        snapshot_df = pd.DataFrame(snapshots)

        # Sort by date and ticker
        if not snapshot_df.empty:
            snapshot_df = snapshot_df.sort_values(["date", "ticker"]).reset_index(drop=True)

        return snapshot_df

    def _apply_universe_filters(self, tickers: list[str], date: pd.Timestamp) -> list[str]:
        """Apply universe configuration filters to ticker list.

        Args:
            tickers: List of ticker symbols to filter
            date: Date for context (future enhancement for market cap/volume filters)

        Returns:
            Filtered list of ticker symbols
        """
        filtered = list(tickers)

        # Apply sector exclusions if specified
        if self.universe_config.exclude_sectors:
            # Note: This would require sector mapping data in future enhancement
            pass

        # Apply custom symbol restrictions if specified
        if self.universe_config.custom_symbols:
            filtered = [t for t in filtered if t in self.universe_config.custom_symbols]

        return sorted(filtered)

    def generate_universe_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate complete universe calendar with monthly snapshots.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with universe calendar data including quality metadata
        """
        # Build membership intervals from Wikipedia
        membership_df = self.build_membership_intervals(start_date, end_date)

        # Create monthly snapshots
        universe_calendar = self.create_monthly_snapshots(membership_df, start_date, end_date)

        # Add metadata
        universe_calendar["universe_type"] = self.universe_config.universe_type
        universe_calendar["rebalance_frequency"] = self.universe_config.rebalance_frequency
        universe_calendar["data_source"] = "wikipedia"
        universe_calendar["algorithm"] = "forward_only"

        # Add start_verified flag from membership data if available
        if "start_verified" in membership_df.columns:
            # Create lookup for start_verified status
            ticker_verified = membership_df.groupby("ticker")["start_verified"].first()
            universe_calendar["start_verified"] = universe_calendar["ticker"].map(ticker_verified)
        else:
            universe_calendar["start_verified"] = True  # Default to verified

        return universe_calendar

    def save_universe_calendar(
        self, universe_calendar: pd.DataFrame, filename: str | None = None
    ) -> Path:
        """Save universe calendar to parquet file.

        Args:
            universe_calendar: Universe calendar DataFrame
            filename: Optional filename (defaults to universe type)

        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"universe_calendar_{self.universe_config.universe_type}.parquet"

        output_path = self.output_dir / filename
        universe_calendar.to_parquet(output_path, index=False)

        return output_path

    def validate_universe_calendar(self, universe_calendar: pd.DataFrame) -> dict[str, Any]:
        """Validate universe calendar data quality with strict enforcement.

        Args:
            universe_calendar: Universe calendar DataFrame to validate

        Returns:
            Dictionary with validation metrics

        Raises:
            ValueError: If strict_validation=True and critical issues found
        """
        validation_results = {}

        # Check basic structure
        validation_results["total_records"] = len(universe_calendar)
        validation_results["unique_dates"] = universe_calendar["date"].nunique()
        validation_results["unique_tickers"] = universe_calendar["ticker"].nunique()

        # Check minimum constituents per month
        monthly_counts = universe_calendar.groupby("date")["ticker"].nunique()
        validation_results["min_constituents"] = int(monthly_counts.min())
        validation_results["max_constituents"] = int(monthly_counts.max())
        validation_results["avg_constituents"] = float(monthly_counts.mean())

        # Check for data quality issues (S&P 400 should have ~400 constituents)
        # Note: Index can temporarily have >400 due to corporate actions
        expected_min = 380
        expected_max = 425

        dates_below_threshold = (monthly_counts < expected_min).sum()
        dates_above_threshold = (monthly_counts > expected_max).sum()

        validation_results["dates_below_threshold"] = int(dates_below_threshold)
        validation_results["dates_above_threshold"] = int(dates_above_threshold)
        validation_results["anomalous_dates"] = int(dates_below_threshold + dates_above_threshold)

        # Check date coverage
        if not universe_calendar.empty:
            first_date = pd.to_datetime(universe_calendar["date"].min())
            last_date = pd.to_datetime(universe_calendar["date"].max())
            validation_results["first_date"] = first_date.isoformat()
            validation_results["last_date"] = last_date.isoformat()

            # Check for missing months
            expected_months = pd.bdate_range(
                start=first_date.replace(day=1),
                end=last_date,
                freq="BMS",
            )
            actual_dates = set(universe_calendar["date"].unique())
            missing_dates = [d for d in expected_months if d not in actual_dates]
            validation_results["missing_months"] = len(missing_dates)

        # Check for duplicates
        duplicates = universe_calendar.groupby(["date", "ticker"]).size()
        duplicate_count = (duplicates > 1).sum()
        validation_results["duplicate_entries"] = int(duplicate_count)

        # Strict validation enforcement
        if self.strict_validation:
            issues = []

            if validation_results["min_constituents"] < expected_min:
                issues.append(
                    f"Minimum constituents ({validation_results['min_constituents']}) "
                    f"below threshold ({expected_min})"
                )

            if validation_results["max_constituents"] > expected_max:
                issues.append(
                    f"Maximum constituents ({validation_results['max_constituents']}) "
                    f"above threshold ({expected_max})"
                )

            if validation_results["anomalous_dates"] > 0:
                issues.append(
                    f"{validation_results['anomalous_dates']} dates have anomalous "
                    f"constituent counts (expected {expected_min}-{expected_max})"
                )

            if validation_results["duplicate_entries"] > 0:
                issues.append(
                    f"{validation_results['duplicate_entries']} duplicate ticker entries per date"
                )

            if issues:
                error_msg = "Universe calendar validation failed:\n" + "\n".join(
                    f"  - {issue}" for issue in issues
                )
                raise ValueError(error_msg)

        validation_results["validation_passed"] = (
            validation_results["anomalous_dates"] == 0
            and validation_results["duplicate_entries"] == 0
        )

        return validation_results

    def build_and_save_universe(self, start_date: str, end_date: str) -> Path:
        """Complete workflow to build and save universe calendar.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Path to saved universe calendar file

        Raises:
            ValueError: If validation fails and strict_validation=True
        """
        # Generate universe calendar
        universe_calendar = self.generate_universe_calendar(start_date, end_date)

        # Validate data quality (may raise ValueError)
        validation_results = self.validate_universe_calendar(universe_calendar)

        # Log validation results
        print("\nUniverse Calendar Validation Results:")  # noqa: T201
        print("  Data Source: Wikipedia (forward-only)")  # noqa: T201
        print(f"  Total Records: {validation_results['total_records']}")  # noqa: T201
        print(f"  Date Range: {validation_results.get('first_date')} to {validation_results.get('last_date')}")  # noqa: T201
        print(f"  Unique Tickers: {validation_results['unique_tickers']}")  # noqa: T201
        print(f"  Avg Constituents: {validation_results['avg_constituents']:.1f}")  # noqa: T201
        print(f"  Min Constituents: {validation_results['min_constituents']}")  # noqa: T201
        print(f"  Max Constituents: {validation_results['max_constituents']}")  # noqa: T201
        print(f"  Anomalous Dates: {validation_results['anomalous_dates']}")  # noqa: T201
        print(f"  Duplicate Entries: {validation_results['duplicate_entries']}")  # noqa: T201
        print(f"  Validation Passed: {validation_results['validation_passed']}")  # noqa: T201

        # Save to parquet
        output_path = self.save_universe_calendar(universe_calendar)

        return output_path


def create_universe_builder(
    universe_type: str = "midcap400",
    rebalance_frequency: str = "monthly",
    output_dir: str = "data/processed",
    strict_validation: bool = True,
    **kwargs: Any,
) -> UniverseBuilder:
    """Factory function to create UniverseBuilder with sensible defaults.

    Args:
        universe_type: Type of universe ('midcap400', 'sp500')
        rebalance_frequency: Rebalancing frequency
        output_dir: Output directory for processed data
        strict_validation: Whether to enforce strict validation
        **kwargs: Additional universe configuration parameters

    Returns:
        Configured UniverseBuilder instance
    """
    universe_config = UniverseConfig(
        universe_type=universe_type,
        rebalance_frequency=rebalance_frequency,
        **kwargs,
    )

    return UniverseBuilder(
        universe_config,
        output_dir,
        strict_validation=strict_validation,
    )
