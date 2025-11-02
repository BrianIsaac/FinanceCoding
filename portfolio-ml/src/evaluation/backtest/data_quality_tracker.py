"""Data quality tracking and persistence for backtest analysis."""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataQualityTracker:
    """
    Track and persist data quality metrics across rebalancing periods.

    Provides CSV log output for post-hoc analysis of data quality
    degradation over time.
    """

    def __init__(self, output_path: Optional[str] = None):
        """
        Initialise data quality tracker.

        Args:
            output_path: Path to save CSV log (default: None, no persistence)
        """
        self.metrics_history: List[Dict[str, Any]] = []
        self.output_path = output_path

    def record_rebalance_metrics(
        self,
        date: pd.Timestamp,
        model_name: str,
        metrics: Dict[str, Any],
    ) -> None:
        """
        Record data quality metrics for a rebalancing period.

        Args:
            date: Rebalance date
            model_name: Name of the model
            metrics: Data quality metrics dictionary
        """
        record = {
            'date': date,
            'model': model_name,
            'requested_assets': metrics.get('requested_assets', 0),
            'available_assets': metrics.get('available_assets', 0),
            'valid_assets': metrics.get('valid_assets', 0),
            'coverage_ratio': metrics.get('coverage_ratio', 0.0),
            'na_count': metrics.get('na_count', 0),
            'na_ratio': metrics.get('na_ratio', 0.0),
        }

        self.metrics_history.append(record)

        logger.debug(
            f"Recorded quality metrics for {model_name} at {date}: "
            f"{record['valid_assets']}/{record['requested_assets']} valid "
            f"({record['coverage_ratio']:.1%} coverage)"
        )

    def save_to_csv(self) -> None:
        """
        Save accumulated metrics to CSV file.

        Raises:
            ValueError: If no output path configured
        """
        if self.output_path is None:
            logger.warning("No output path configured, skipping CSV save")
            return

        if len(self.metrics_history) == 0:
            logger.warning("No metrics to save")
            return

        df = pd.DataFrame(self.metrics_history)

        # Ensure output directory exists
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(self.output_path, index=False)

        logger.info(
            f"Saved {len(self.metrics_history)} data quality records to {self.output_path}"
        )

    def generate_degradation_report(self) -> pd.DataFrame:
        """
        Generate summary report of data quality degradation over time.

        Returns:
            DataFrame with aggregated quality metrics by time period
        """
        if len(self.metrics_history) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(self.metrics_history)
        df['date'] = pd.to_datetime(df['date'])

        # Group by month and model
        df['month'] = df['date'].dt.to_period('M')

        summary = df.groupby(['month', 'model']).agg({
            'coverage_ratio': ['mean', 'std', 'min', 'max'],
            'valid_assets': ['mean', 'min', 'max'],
            'na_ratio': ['mean', 'max'],
        }).round(4)

        return summary

    def get_metrics_dataframe(self) -> pd.DataFrame:
        """
        Get all recorded metrics as a DataFrame.

        Returns:
            DataFrame containing all recorded metrics
        """
        if len(self.metrics_history) == 0:
            return pd.DataFrame()

        return pd.DataFrame(self.metrics_history)
