"""
Portfolio export and analysis framework.

This module provides functionality to export portfolio positions,
weights, and rebalancing schedules to various formats for analysis
and verification purposes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.metrics.returns import ReturnAnalyzer, ReturnMetricsConfig


@dataclass
class ExportConfig:
    """Configuration for portfolio export functionality."""

    output_directory: str = "output/portfolio_analysis"
    export_formats: list[str] = None  # ["parquet", "csv", "json"]
    include_metadata: bool = True
    compress_output: bool = True
    date_format: str = "%Y-%m-%d"

    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ["parquet", "csv"]


class PortfolioExporter:
    """
    Portfolio export and analysis framework.

    Provides comprehensive export functionality for portfolio weights,
    rebalancing history, and performance analytics in multiple formats.
    """

    def __init__(self, config: ExportConfig):
        """
        Initialize portfolio exporter.

        Args:
            config: Configuration for export behavior
        """
        self.config = config
        self.output_path = Path(config.output_directory)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def export_portfolio_weights(
        self,
        weights_history: list[dict[str, Any]],
        filename: str = "portfolio_weights"
    ) -> dict[str, str]:
        """
        Export portfolio weights history to configured formats.

        Args:
            weights_history: List of weight dictionaries with dates and weights
            filename: Base filename for export

        Returns:
            Dictionary mapping format names to exported file paths
        """
        # Convert to DataFrame
        weights_df = self._weights_history_to_dataframe(weights_history)

        exported_files = {}

        for format_type in self.config.export_formats:
            if format_type == "parquet":
                filepath = self.output_path / f"{filename}.parquet"
                weights_df.to_parquet(
                    filepath,
                    compression='snappy' if self.config.compress_output else None,
                    index=True
                )
                exported_files["parquet"] = str(filepath)

            elif format_type == "csv":
                filepath = self.output_path / f"{filename}.csv"
                weights_df.to_csv(filepath, index=True)
                exported_files["csv"] = str(filepath)

            elif format_type == "json":
                filepath = self.output_path / f"{filename}.json"
                # Convert to JSON-serializable format
                json_data = self._dataframe_to_json(weights_df)
                with open(filepath, 'w') as f:
                    json.dump(json_data, f, indent=2, default=str)
                exported_files["json"] = str(filepath)

        return exported_files

    def export_rebalancing_schedule(
        self,
        rebalancing_history: list[dict[str, Any]],
        filename: str = "rebalancing_schedule"
    ) -> dict[str, str]:
        """
        Export rebalancing schedule and statistics.

        Args:
            rebalancing_history: List of rebalancing event dictionaries
            filename: Base filename for export

        Returns:
            Dictionary mapping format names to exported file paths
        """
        # Convert to DataFrame
        rebalancing_df = pd.DataFrame(rebalancing_history)

        if 'date' in rebalancing_df.columns:
            rebalancing_df['date'] = pd.to_datetime(rebalancing_df['date'])
            rebalancing_df.set_index('date', inplace=True)

        exported_files = {}

        for format_type in self.config.export_formats:
            if format_type == "parquet":
                filepath = self.output_path / f"{filename}.parquet"
                rebalancing_df.to_parquet(
                    filepath,
                    compression='snappy' if self.config.compress_output else None,
                    index=True
                )
                exported_files["parquet"] = str(filepath)

            elif format_type == "csv":
                filepath = self.output_path / f"{filename}.csv"
                rebalancing_df.to_csv(filepath, index=True)
                exported_files["csv"] = str(filepath)

            elif format_type == "json":
                filepath = self.output_path / f"{filename}.json"
                # Convert timestamps to strings for JSON serialization
                json_data = {}
                for idx, row in rebalancing_df.iterrows():
                    key = str(idx)  # Convert timestamp index to string
                    json_data[key] = row.to_dict()

                with open(filepath, 'w') as f:
                    json.dump(json_data, f, indent=2, default=str)
                exported_files["json"] = str(filepath)

        return exported_files

    def create_portfolio_analytics_dashboard(
        self,
        weights_history: list[dict[str, Any]],
        returns_history: pd.Series | None = None,
        rebalancing_history: list[dict[str, Any]] | None = None,
        model_info: dict[str, Any] | None = None
    ) -> str:
        """
        Create comprehensive portfolio analytics dashboard.

        Args:
            weights_history: Portfolio weights over time
            returns_history: Portfolio returns time series
            rebalancing_history: Rebalancing events and statistics
            model_info: Model metadata and configuration

        Returns:
            Path to generated analytics dashboard file
        """
        dashboard_data = {
            "metadata": {
                "generation_date": pd.Timestamp.now().strftime(self.config.date_format),
                "model_info": model_info or {},
            },
            "summary_statistics": {},
            "portfolio_analysis": {},
            "rebalancing_analysis": {},
        }

        # Portfolio summary statistics
        weights_df = self._weights_history_to_dataframe(weights_history)

        dashboard_data["summary_statistics"] = {
            "total_periods": len(weights_df),
            "date_range": {
                "start": weights_df.index.min().strftime(self.config.date_format),
                "end": weights_df.index.max().strftime(self.config.date_format),
            },
            "average_positions": (weights_df > 0).sum(axis=1).mean(),
            "max_positions": (weights_df > 0).sum(axis=1).max(),
            "weight_concentration": (weights_df ** 2).sum(axis=1).mean(),  # Herfindahl index
        }

        # Performance analysis (if returns provided)
        if returns_history is not None and not returns_history.empty:
            analyzer = ReturnAnalyzer(ReturnMetricsConfig())
            performance_metrics = analyzer.calculate_basic_metrics(returns_history)
            dashboard_data["performance_metrics"] = performance_metrics

        # Rebalancing analysis
        if rebalancing_history:
            rebalancing_stats = self._analyze_rebalancing_history(rebalancing_history)
            dashboard_data["rebalancing_analysis"] = rebalancing_stats

        # Export dashboard
        dashboard_file = self.output_path / "portfolio_analytics_dashboard.json"
        with open(dashboard_file, 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)

        return str(dashboard_file)

    def export_position_analysis(
        self,
        weights_history: list[dict[str, Any]],
        filename: str = "position_analysis"
    ) -> str:
        """
        Export detailed position-level analysis.

        Args:
            weights_history: Portfolio weights over time
            filename: Base filename for export

        Returns:
            Path to position analysis file
        """
        weights_df = self._weights_history_to_dataframe(weights_history)

        position_analysis = {}

        for asset in weights_df.columns:
            asset_weights = weights_df[asset]
            active_periods = asset_weights[asset_weights > 0]

            if len(active_periods) > 0:
                position_analysis[asset] = {
                    "periods_active": len(active_periods),
                    "periods_total": len(asset_weights),
                    "activity_rate": len(active_periods) / len(asset_weights),
                    "average_weight": active_periods.mean(),
                    "max_weight": active_periods.max(),
                    "weight_volatility": active_periods.std(),
                    "first_appearance": active_periods.index.min().strftime(self.config.date_format),
                    "last_appearance": active_periods.index.max().strftime(self.config.date_format),
                }

        # Export position analysis
        analysis_file = self.output_path / f"{filename}.json"
        with open(analysis_file, 'w') as f:
            json.dump(position_analysis, f, indent=2, default=str)

        return str(analysis_file)

    def _weights_history_to_dataframe(self, weights_history: list[dict[str, Any]]) -> pd.DataFrame:
        """Convert weights history to DataFrame format."""
        if not weights_history:
            return pd.DataFrame()

        # Extract dates and weights
        dates = [entry["date"] for entry in weights_history]
        weights_data = [entry["weights"] for entry in weights_history]

        # Create DataFrame
        weights_df = pd.DataFrame(weights_data, index=pd.to_datetime(dates))
        weights_df = weights_df.fillna(0.0)
        weights_df.index.name = "date"

        return weights_df

    def _dataframe_to_json(self, df: pd.DataFrame) -> dict[str, Any]:
        """Convert DataFrame to JSON-serializable format."""
        return {
            "index": [str(idx) for idx in df.index],
            "columns": df.columns.tolist(),
            "data": df.values.tolist(),
        }

    def _analyze_rebalancing_history(self, rebalancing_history: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze rebalancing history for dashboard."""
        if not rebalancing_history:
            return {}

        executed_rebalances = [r for r in rebalancing_history if r.get("rebalanced", False)]

        total_costs = sum(
            r.get("transaction_costs", {}).get("total_cost", 0.0)
            for r in executed_rebalances
        )

        average_turnover = (
            sum(r.get("transaction_costs", {}).get("turnover", 0.0) for r in executed_rebalances)
            / len(executed_rebalances)
            if executed_rebalances else 0.0
        )

        return {
            "total_periods": len(rebalancing_history),
            "executed_rebalances": len(executed_rebalances),
            "rebalancing_rate": len(executed_rebalances) / len(rebalancing_history),
            "total_transaction_costs": total_costs,
            "average_turnover": average_turnover,
            "cost_per_rebalance": total_costs / len(executed_rebalances) if executed_rebalances else 0.0,
        }
