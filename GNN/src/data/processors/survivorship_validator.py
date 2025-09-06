"""Survivorship bias validation framework.

This module provides validation and analysis tools to detect and report
survivorship bias in dynamic universe construction and backtesting.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.config.data import UniverseConfig


class SurvivorshipValidator:
    """
    Framework for validating survivorship bias in universe construction.

    Provides methods to detect delisted companies, validate historical
    accuracy, and generate comprehensive reports on anti-survivorship
    bias methodology.
    """

    def __init__(self, universe_config: UniverseConfig):
        """
        Initialize survivorship validator.

        Args:
            universe_config: Universe configuration for validation context
        """
        self.universe_config = universe_config

    def validate_membership_intervals(self, membership_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate membership intervals for survivorship bias issues.

        Args:
            membership_df: DataFrame with ticker, start, end, index_name columns

        Returns:
            Dictionary with validation metrics and survivorship analysis
        """
        validation_results = {}

        # Basic interval validation
        validation_results["total_intervals"] = len(membership_df)
        validation_results["unique_tickers"] = membership_df["ticker"].nunique()

        # Temporal coverage analysis
        if not membership_df.empty:
            validation_results["date_range"] = (
                membership_df["start"].min(),
                membership_df["end"].max() if membership_df["end"].notna().any() else "Present",
            )

            # Count active vs historical intervals
            active_intervals = membership_df["end"].isna().sum()
            historical_intervals = membership_df["end"].notna().sum()

            validation_results["active_intervals"] = active_intervals
            validation_results["historical_intervals"] = historical_intervals
            validation_results["historical_ratio"] = historical_intervals / len(membership_df)

            # Survivorship bias indicators
            validation_results["delisted_tickers"] = self._identify_delisted_tickers(membership_df)
            validation_results["survivor_only_analysis"] = self._analyze_survivor_only_bias(
                membership_df
            )

        else:
            validation_results["date_range"] = (None, None)
            validation_results["active_intervals"] = 0
            validation_results["historical_intervals"] = 0
            validation_results["historical_ratio"] = 0.0

        return validation_results

    def _identify_delisted_tickers(self, membership_df: pd.DataFrame) -> Dict[str, Any]:
        """Identify tickers that were delisted (have end dates).

        Args:
            membership_df: Membership intervals DataFrame

        Returns:
            Dictionary with delisting analysis
        """
        delisted_analysis = {}

        # Find tickers with end dates (delisted)
        delisted_mask = membership_df["end"].notna()
        delisted_df = membership_df[delisted_mask].copy()

        if not delisted_df.empty:
            delisted_analysis["delisted_count"] = len(delisted_df)
            delisted_analysis["delisted_tickers"] = sorted(delisted_df["ticker"].unique().tolist())

            # Analyze delisting patterns by year
            delisted_df["delisting_year"] = delisted_df["end"].dt.year
            yearly_delistings = delisted_df.groupby("delisting_year").size().to_dict()
            delisted_analysis["yearly_delistings"] = yearly_delistings

            # Calculate membership duration for delisted companies
            delisted_df["membership_duration"] = (delisted_df["end"] - delisted_df["start"]).dt.days
            delisted_analysis["avg_membership_duration_days"] = delisted_df[
                "membership_duration"
            ].mean()
            delisted_analysis["median_membership_duration_days"] = delisted_df[
                "membership_duration"
            ].median()

        else:
            delisted_analysis["delisted_count"] = 0
            delisted_analysis["delisted_tickers"] = []
            delisted_analysis["yearly_delistings"] = {}

        return delisted_analysis

    def _analyze_survivor_only_bias(self, membership_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze potential survivor-only bias in the dataset.

        Args:
            membership_df: Membership intervals DataFrame

        Returns:
            Dictionary with survivor bias analysis
        """
        survivor_analysis = {}

        # Current survivors (no end date)
        current_survivors = membership_df[membership_df["end"].isna()]
        survivor_analysis["current_survivor_count"] = len(current_survivors)

        # Historical vs current ratio
        total_unique_tickers = membership_df["ticker"].nunique()
        current_unique_survivors = current_survivors["ticker"].nunique()

        survivor_analysis["unique_survivors"] = current_unique_survivors
        survivor_analysis["unique_historical"] = total_unique_tickers - current_unique_survivors
        survivor_analysis["survivor_bias_ratio"] = (
            current_unique_survivors / total_unique_tickers if total_unique_tickers > 0 else 0.0
        )

        # Temporal bias analysis - check if early periods have fewer companies
        if not membership_df.empty:
            # Sample periods at different time points
            earliest_date = membership_df["start"].min()
            latest_date = (
                membership_df["end"].max()
                if membership_df["end"].notna().any()
                else pd.Timestamp.now()
            )

            # Count active companies at different time points
            sample_dates = pd.date_range(earliest_date, latest_date, periods=5)
            temporal_counts = {}

            for sample_date in sample_dates:
                active_mask = (membership_df["start"] <= sample_date) & (
                    (membership_df["end"].isna()) | (membership_df["end"] > sample_date)
                )
                active_count = active_mask.sum()
                temporal_counts[sample_date.strftime("%Y-%m-%d")] = active_count

            survivor_analysis["temporal_membership_counts"] = temporal_counts

        return survivor_analysis

    def validate_universe_calendar(self, universe_calendar: pd.DataFrame) -> Dict[str, Any]:
        """Validate universe calendar for survivorship bias.

        Args:
            universe_calendar: Universe calendar with date, ticker, index_name

        Returns:
            Dictionary with calendar-specific survivorship validation
        """
        calendar_validation = {}

        if universe_calendar.empty:
            calendar_validation["error"] = "Empty universe calendar"
            return calendar_validation

        # Basic metrics
        calendar_validation["total_records"] = len(universe_calendar)
        calendar_validation["date_range"] = (
            universe_calendar["date"].min(),
            universe_calendar["date"].max(),
        )
        calendar_validation["unique_dates"] = universe_calendar["date"].nunique()
        calendar_validation["unique_tickers"] = universe_calendar["ticker"].nunique()

        # Survivorship bias detection in calendar
        monthly_stats = (
            universe_calendar.groupby("date").agg({"ticker": ["nunique", "count"]}).round(2)
        )
        monthly_stats.columns = ["unique_tickers", "total_records"]

        # Check for increasing universe size over time (potential survivorship bias indicator)
        first_month_size = monthly_stats["unique_tickers"].iloc[0]
        last_month_size = monthly_stats["unique_tickers"].iloc[-1]
        size_trend = (
            (last_month_size - first_month_size) / first_month_size if first_month_size > 0 else 0.0
        )

        calendar_validation["first_month_universe_size"] = first_month_size
        calendar_validation["last_month_universe_size"] = last_month_size
        calendar_validation["universe_size_trend"] = size_trend

        # Flag potential bias if universe grows significantly over time without explanation
        if size_trend > 0.2:  # 20% growth
            calendar_validation["survivorship_bias_warning"] = True
            calendar_validation["bias_warning_message"] = (
                f"Universe size increased by {size_trend:.1%} over time. "
                "This may indicate survivorship bias if not due to genuine index changes."
            )
        else:
            calendar_validation["survivorship_bias_warning"] = False

        # Analyze ticker turnover
        calendar_validation["ticker_turnover_analysis"] = self._analyze_ticker_turnover(
            universe_calendar
        )

        return calendar_validation

    def _analyze_ticker_turnover(self, universe_calendar: pd.DataFrame) -> Dict[str, Any]:
        """Analyze ticker turnover patterns in universe calendar.

        Args:
            universe_calendar: Universe calendar DataFrame

        Returns:
            Dictionary with turnover analysis
        """
        turnover_analysis = {}

        # Get unique dates sorted
        unique_dates = sorted(universe_calendar["date"].unique())

        if len(unique_dates) < 2:
            turnover_analysis["insufficient_data"] = True
            return turnover_analysis

        # Calculate monthly additions and removals
        monthly_changes = []

        for i in range(1, len(unique_dates)):
            current_date = unique_dates[i]
            previous_date = unique_dates[i - 1]

            current_tickers = set(
                universe_calendar[universe_calendar["date"] == current_date]["ticker"]
            )
            previous_tickers = set(
                universe_calendar[universe_calendar["date"] == previous_date]["ticker"]
            )

            added = current_tickers - previous_tickers
            removed = previous_tickers - current_tickers

            monthly_changes.append(
                {
                    "date": current_date,
                    "added_count": len(added),
                    "removed_count": len(removed),
                    "added_tickers": sorted(list(added)),
                    "removed_tickers": sorted(list(removed)),
                    "net_change": len(added) - len(removed),
                }
            )

        turnover_df = pd.DataFrame(monthly_changes)

        # Aggregate turnover statistics
        turnover_analysis["avg_monthly_additions"] = turnover_df["added_count"].mean()
        turnover_analysis["avg_monthly_removals"] = turnover_df["removed_count"].mean()
        turnover_analysis["avg_net_change"] = turnover_df["net_change"].mean()
        turnover_analysis["max_monthly_turnover"] = (
            turnover_df["added_count"] + turnover_df["removed_count"]
        ).max()

        # Identify periods of high turnover
        high_turnover_threshold = turnover_analysis["max_monthly_turnover"] * 0.7
        high_turnover_dates = turnover_df[
            (turnover_df["added_count"] + turnover_df["removed_count"]) >= high_turnover_threshold
        ]["date"].tolist()

        turnover_analysis["high_turnover_dates"] = [
            d.strftime("%Y-%m-%d") for d in high_turnover_dates
        ]

        return turnover_analysis

    def generate_survivorship_report(
        self,
        membership_df: pd.DataFrame,
        universe_calendar: pd.DataFrame,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive survivorship bias validation report.

        Args:
            membership_df: Membership intervals DataFrame
            universe_calendar: Universe calendar DataFrame
            output_path: Optional path to save detailed report

        Returns:
            Dictionary with complete survivorship analysis
        """
        report = {
            "universe_type": self.universe_config.universe_type,
            "validation_timestamp": datetime.now().isoformat(),
            "methodology": {
                "description": "Anti-survivorship bias methodology validation",
                "approach": "Wikipedia-based historical membership reconstruction",
                "bias_mitigation": "Includes delisted companies in historical periods",
            },
        }

        # Validate membership intervals
        report["membership_validation"] = self.validate_membership_intervals(membership_df)

        # Validate universe calendar
        report["calendar_validation"] = self.validate_universe_calendar(universe_calendar)

        # Overall assessment
        report["overall_assessment"] = self._generate_overall_assessment(report)

        # Save detailed report if requested
        if output_path:
            self._save_detailed_report(report, output_path)

        return report

    def _generate_overall_assessment(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall survivorship bias assessment.

        Args:
            report: Validation report dictionary

        Returns:
            Dictionary with overall assessment
        """
        assessment = {}

        membership_val = report["membership_validation"]
        calendar_val = report["calendar_validation"]

        # Calculate bias score (0 = high bias, 1 = low bias)
        bias_indicators = []

        # Historical ratio indicator
        historical_ratio = membership_val.get("historical_ratio", 0.0)
        bias_indicators.append(min(historical_ratio * 2, 1.0))  # Good if >= 0.5

        # Delisted companies indicator
        delisted_count = membership_val.get("delisted_tickers", {}).get("delisted_count", 0)
        total_tickers = membership_val.get("unique_tickers", 1)
        delisted_ratio = delisted_count / total_tickers
        bias_indicators.append(min(delisted_ratio * 3, 1.0))  # Good if >= 0.33

        # Universe size trend indicator (penalize excessive growth)
        size_trend = abs(calendar_val.get("universe_size_trend", 0.0))
        trend_score = max(0, 1 - size_trend * 2)  # Penalize trends > 0.5
        bias_indicators.append(trend_score)

        # Calculate overall bias score
        assessment["bias_score"] = (
            sum(bias_indicators) / len(bias_indicators) if bias_indicators else 0.0
        )
        assessment["bias_grade"] = self._score_to_grade(assessment["bias_score"])

        # Detailed recommendations
        assessment["recommendations"] = self._generate_recommendations(report)

        return assessment

    def _score_to_grade(self, score: float) -> str:
        """Convert bias score to letter grade.

        Args:
            score: Bias score between 0 and 1

        Returns:
            Letter grade (A-F)
        """
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"

    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results.

        Args:
            report: Complete validation report

        Returns:
            List of recommendation strings
        """
        recommendations = []

        membership_val = report["membership_validation"]
        calendar_val = report["calendar_validation"]

        # Historical ratio recommendations
        historical_ratio = membership_val.get("historical_ratio", 0.0)
        if historical_ratio < 0.3:
            recommendations.append(
                "Low historical ratio detected. Verify Wikipedia scraping includes "
                "comprehensive change history and delisted companies."
            )

        # Delisting recommendations
        delisted_count = membership_val.get("delisted_tickers", {}).get("delisted_count", 0)
        if delisted_count < 10:
            recommendations.append(
                "Few delisted companies found. Cross-reference with known major "
                "delistings to ensure completeness."
            )

        # Universe size trend recommendations
        if calendar_val.get("survivorship_bias_warning", False):
            recommendations.append(
                "Significant universe size growth detected. Verify this reflects "
                "genuine index composition changes rather than survivorship bias."
            )

        # Turnover recommendations
        turnover = calendar_val.get("ticker_turnover_analysis", {})
        avg_removals = turnover.get("avg_monthly_removals", 0)
        if avg_removals < 1:
            recommendations.append(
                "Very low monthly removal rate. Ensure methodology captures "
                "companies that exit the index due to delisting or other factors."
            )

        if not recommendations:
            recommendations.append(
                "Survivorship bias validation passed. Universe construction "
                "methodology appears robust against survivorship bias."
            )

        return recommendations

    def _save_detailed_report(self, report: Dict[str, Any], output_path: str) -> None:
        """Save detailed survivorship report to file.

        Args:
            report: Complete validation report
            output_path: Path to save report
        """
        import json

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"Detailed survivorship report saved to {output_path}")
