"""
Performance comparison tables framework with comprehensive ranking and statistical integration.

This module provides comprehensive ranking tables with sortable columns for all key metrics,
statistical significance indicators, and relative performance metrics across all approaches.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    from IPython.display import HTML, display

    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

from src.evaluation.validation.reporting import PublicationReadyStatisticalReporting
from src.evaluation.validation.significance import StatisticalValidation


@dataclass
class TableConfig:
    """Configuration for performance comparison tables."""

    decimal_places: int = 4
    significance_levels: list[float] = None
    include_confidence_intervals: bool = True
    include_rankings: bool = True
    sortable: bool = True
    export_formats: list[str] = None

    def __post_init__(self):
        if self.significance_levels is None:
            self.significance_levels = [0.001, 0.01, 0.05]
        if self.export_formats is None:
            self.export_formats = ["html", "latex", "csv"]


class PerformanceComparisonTables:
    """
    Comprehensive performance comparison tables framework.

    Creates sortable ranking tables with statistical significance indicators,
    relative performance metrics, and comprehensive formatting options.
    """

    def __init__(self, config: TableConfig = None):
        """
        Initialize performance comparison tables framework.

        Args:
            config: Configuration for table generation and formatting
        """
        self.config = config or TableConfig()
        self.statistical_reporter = PublicationReadyStatisticalReporting(
            significance_levels=self.config.significance_levels,
            decimal_places=self.config.decimal_places,
        )
        self.statistical_validator = StatisticalValidation()

    def create_performance_ranking_table(
        self,
        performance_results: dict[str, dict[str, float]],
        statistical_results: dict[str, dict[str, Any]] | None = None,
        baseline_comparisons: dict[str, dict[str, float]] | None = None,
    ) -> pd.DataFrame:
        """
        Create comprehensive performance ranking table with all key metrics.

        Args:
            performance_results: Dictionary mapping approach names to performance metrics
            statistical_results: Optional statistical significance test results
            baseline_comparisons: Optional relative performance vs baselines

        Returns:
            Formatted performance ranking DataFrame
        """
        # Convert performance results to DataFrame
        perf_df = pd.DataFrame(performance_results).T

        # Add statistical significance indicators
        if statistical_results:
            perf_df = self._add_significance_indicators(perf_df, statistical_results)

        # Add relative performance metrics
        if baseline_comparisons:
            perf_df = self._add_baseline_comparisons(perf_df, baseline_comparisons)

        # Add rankings
        if self.config.include_rankings:
            perf_df = self._add_metric_rankings(perf_df)

        # Format numerical values
        perf_df = self._format_numerical_columns(perf_df)

        # Sort by primary ranking metric (Sharpe ratio by default)
        if "sharpe_ratio" in perf_df.columns:
            perf_df = perf_df.sort_values("sharpe_ratio", ascending=False)
        elif "information_ratio" in perf_df.columns:
            perf_df = perf_df.sort_values("information_ratio", ascending=False)

        return perf_df

    def create_risk_adjusted_ranking_table(
        self,
        performance_results: dict[str, dict[str, float]],
        risk_results: dict[str, dict[str, float]],
        statistical_results: dict[str, dict[str, Any]] | None = None,
    ) -> pd.DataFrame:
        """
        Create risk-adjusted performance ranking table.

        Args:
            performance_results: Performance metrics by approach
            risk_results: Risk metrics by approach
            statistical_results: Statistical significance test results

        Returns:
            Risk-adjusted performance ranking DataFrame
        """
        # Combine performance and risk metrics
        combined_results = {}
        for approach in performance_results:
            combined_results[approach] = {
                **performance_results[approach],
                **risk_results.get(approach, {}),
            }

        # Create ranking table
        ranking_df = self.create_performance_ranking_table(combined_results, statistical_results)

        # Add risk-adjusted composite score
        ranking_df["risk_adjusted_score"] = self._calculate_risk_adjusted_score(ranking_df)

        # Sort by risk-adjusted score
        ranking_df = ranking_df.sort_values("risk_adjusted_score", ascending=False)

        return ranking_df

    def create_operational_efficiency_table(
        self,
        operational_results: dict[str, dict[str, float]],
        performance_results: dict[str, dict[str, float]],
    ) -> pd.DataFrame:
        """
        Create operational efficiency comparison table.

        Args:
            operational_results: Operational metrics by approach
            performance_results: Performance metrics for context

        Returns:
            Operational efficiency comparison DataFrame
        """
        # Combine operational and key performance metrics
        combined_results = {}
        key_performance_metrics = ["sharpe_ratio", "total_return", "max_drawdown"]

        for approach in operational_results:
            combined_results[approach] = operational_results[approach].copy()

            # Add key performance metrics for context
            perf_data = performance_results.get(approach, {})
            for metric in key_performance_metrics:
                if metric in perf_data:
                    combined_results[approach][f"perf_{metric}"] = perf_data[metric]

        # Create table
        ops_df = pd.DataFrame(combined_results).T

        # Calculate efficiency ratios
        if "avg_monthly_turnover" in ops_df.columns and "perf_sharpe_ratio" in ops_df.columns:
            ops_df["sharpe_per_turnover"] = ops_df["perf_sharpe_ratio"] / (
                ops_df["avg_monthly_turnover"] + 1e-6
            )

        if "total_transaction_costs" in ops_df.columns and "perf_total_return" in ops_df.columns:
            ops_df["return_after_costs"] = (
                ops_df["perf_total_return"] - ops_df["total_transaction_costs"]
            )

        # Format and sort
        ops_df = self._format_numerical_columns(ops_df)

        if "sharpe_per_turnover" in ops_df.columns:
            ops_df = ops_df.sort_values("sharpe_per_turnover", ascending=False)

        return ops_df

    def create_rolling_window_summary_table(
        self,
        rolling_results: dict[str, dict[str, list[float]]],
        statistical_results: dict[str, dict[str, Any]] | None = None,
    ) -> pd.DataFrame:
        """
        Create aggregated performance summary across rolling windows.

        Args:
            rolling_results: Rolling window results by approach
            statistical_results: Statistical significance across windows

        Returns:
            Rolling window summary DataFrame
        """
        summary_results = {}

        for approach, metrics in rolling_results.items():
            approach_summary = {}

            for metric_name, values in metrics.items():
                if values:  # Check if list is not empty
                    values_array = np.array(values)
                    approach_summary.update(
                        {
                            f"{metric_name}_mean": np.mean(values_array),
                            f"{metric_name}_std": np.std(values_array),
                            f"{metric_name}_median": np.median(values_array),
                            f"{metric_name}_min": np.min(values_array),
                            f"{metric_name}_max": np.max(values_array),
                            f"{metric_name}_consistency": 1.0
                            - (np.std(values_array) / (np.abs(np.mean(values_array)) + 1e-6)),
                        }
                    )

            summary_results[approach] = approach_summary

        # Create summary table
        summary_df = pd.DataFrame(summary_results).T

        # Add statistical significance if available
        if statistical_results:
            summary_df = self._add_significance_indicators(summary_df, statistical_results)

        # Format and sort
        summary_df = self._format_numerical_columns(summary_df)

        # Sort by consistency of primary metric (Sharpe ratio)
        if "sharpe_ratio_consistency" in summary_df.columns:
            summary_df = summary_df.sort_values("sharpe_ratio_consistency", ascending=False)

        return summary_df

    def _add_significance_indicators(
        self, df: pd.DataFrame, statistical_results: dict[str, dict[str, Any]]
    ) -> pd.DataFrame:
        """Add statistical significance indicators to table."""
        df_with_sig = df.copy()

        for approach in df.index:
            approach_stats = statistical_results.get(approach, {})

            for metric in df.columns:
                if metric in approach_stats:
                    stat_data = approach_stats[metric]
                    p_value = stat_data.get("p_value", 1.0)

                    # Add significance indicator
                    significance_symbol = self._get_significance_symbol(p_value)
                    if significance_symbol:
                        original_value = df_with_sig.loc[approach, metric]
                        # Convert to string to avoid dtype compatibility issues
                        df_with_sig.loc[approach, metric] = str(original_value) + significance_symbol

        return df_with_sig

    def _add_baseline_comparisons(
        self, df: pd.DataFrame, baseline_comparisons: dict[str, dict[str, float]]
    ) -> pd.DataFrame:
        """Add relative performance vs baseline metrics."""
        df_with_baselines = df.copy()

        for approach in df.index:
            baseline_data = baseline_comparisons.get(approach, {})

            for baseline_name, metrics in baseline_data.items():
                for metric_name, value in metrics.items():
                    col_name = f"vs_{baseline_name}_{metric_name}"
                    df_with_baselines.loc[approach, col_name] = value

        return df_with_baselines

    def _add_metric_rankings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ranking columns for key metrics."""
        df_with_ranks = df.copy()

        # Define metrics and their ranking order (higher is better vs lower is better)
        ranking_metrics = {
            "sharpe_ratio": False,  # Higher is better
            "information_ratio": False,
            "total_return": False,
            "max_drawdown": True,  # Lower is better (less negative)
            "volatility": True,  # Lower is better
            "tracking_error": True,  # Lower is better
            "avg_monthly_turnover": True,  # Lower is better
        }

        for metric, ascending in ranking_metrics.items():
            if metric in df.columns:
                # Extract numeric values for ranking (remove significance symbols)
                numeric_values = pd.to_numeric(
                    df[metric].astype(str).str.replace(r"[^\d\.\-]", "", regex=True),
                    errors="coerce",
                )
                df_with_ranks[f"{metric}_rank"] = numeric_values.rank(ascending=ascending)

        return df_with_ranks

    def _calculate_risk_adjusted_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate composite risk-adjusted performance score."""
        score_components = {}

        # Sharpe ratio (40% weight)
        if "sharpe_ratio" in df.columns:
            sharpe_values = pd.to_numeric(
                df["sharpe_ratio"].astype(str).str.replace(r"[^\d\.\-]", "", regex=True),
                errors="coerce",
            )
            score_components["sharpe"] = sharpe_values * 0.4

        # Information ratio (30% weight)
        if "information_ratio" in df.columns:
            info_values = pd.to_numeric(
                df["information_ratio"].astype(str).str.replace(r"[^\d\.\-]", "", regex=True),
                errors="coerce",
            )
            score_components["info_ratio"] = info_values * 0.3

        # Maximum drawdown penalty (20% weight, inverted)
        if "max_drawdown" in df.columns:
            drawdown_values = pd.to_numeric(
                df["max_drawdown"].astype(str).str.replace(r"[^\d\.\-]", "", regex=True),
                errors="coerce",
            )
            # Convert to positive score (less negative drawdown = higher score)
            score_components["drawdown"] = (-drawdown_values / drawdown_values.abs().max()) * 0.2

        # Volatility penalty (10% weight, inverted)
        if "volatility" in df.columns:
            vol_values = pd.to_numeric(
                df["volatility"].astype(str).str.replace(r"[^\d\.\-]", "", regex=True),
                errors="coerce",
            )
            score_components["volatility"] = (1 - vol_values / vol_values.max()) * 0.1

        # Combine scores
        total_score = pd.Series(0.0, index=df.index)
        for component_score in score_components.values():
            total_score += component_score.fillna(0)

        return total_score

    def _get_significance_symbol(self, p_value: float) -> str:
        """Get significance symbol based on p-value."""
        if p_value <= 0.001:
            return "***"
        elif p_value <= 0.01:
            return "**"
        elif p_value <= 0.05:
            return "*"
        else:
            return ""

    def _format_numerical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format numerical columns with appropriate precision."""
        formatted_df = df.copy()

        for col in formatted_df.columns:
            if col.endswith("_rank"):
                # Format ranks as integers, handling NaN values
                if formatted_df[col].dtype in ["float64", "int64"]:
                    # Convert to nullable integer type to handle NaN
                    formatted_df[col] = formatted_df[col].astype('Int64')
            elif not col.endswith(("_symbol", "_indicator")):
                # Format other numerical columns
                if formatted_df[col].dtype in ["float64", "int64"]:
                    formatted_df[col] = formatted_df[col].round(self.config.decimal_places)

        return formatted_df

    def export_table(
        self,
        table_df: pd.DataFrame,
        filename: str,
        title: str = "Performance Comparison Table",
        formats: list[str] | None = None,
    ) -> dict[str, str]:
        """
        Export table in multiple formats.

        Args:
            table_df: Table DataFrame to export
            filename: Base filename for export
            title: Table title for formatted exports
            formats: Export formats (defaults to config)

        Returns:
            Dictionary mapping format names to file paths
        """
        export_formats = formats or self.config.export_formats
        exported_files = {}

        for format_type in export_formats:
            if format_type == "csv":
                filepath = f"{filename}.csv"
                table_df.to_csv(filepath)
                exported_files["csv"] = filepath

            elif format_type == "html":
                filepath = f"{filename}.html"
                html_content = self._generate_html_table(table_df, title)
                with open(filepath, "w") as f:
                    f.write(html_content)
                exported_files["html"] = filepath

            elif format_type == "latex":
                filepath = f"{filename}.tex"
                latex_content = self._generate_latex_table(table_df, title)
                with open(filepath, "w") as f:
                    f.write(latex_content)
                exported_files["latex"] = filepath

        return exported_files

    def _generate_html_table(self, df: pd.DataFrame, title: str) -> str:
        """Generate HTML table with styling and interactivity."""
        html_style = """
        <style>
        .performance-table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            font-family: Arial, sans-serif;
        }
        .performance-table th,
        .performance-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: right;
        }
        .performance-table th {
            background-color: #f2f2f2;
            font-weight: bold;
            cursor: pointer;
        }
        .performance-table th:hover {
            background-color: #e0e0e0;
        }
        .performance-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .performance-table tr:hover {
            background-color: #f5f5f5;
        }
        .significance-symbol {
            color: #d32f2f;
            font-weight: bold;
        }
        .table-title {
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        </style>
        """

        # Convert DataFrame to HTML
        table_html = df.to_html(
            classes="performance-table", escape=False, table_id="performanceTable"
        )

        # Add JavaScript for sorting (if enabled)
        sorting_js = ""
        if self.config.sortable:
            sorting_js = """
            <script>
            function sortTable(columnIndex) {
                var table = document.getElementById("performanceTable");
                var switching = true;
                var dir = "asc";
                var switchcount = 0;

                while (switching) {
                    switching = false;
                    var rows = table.rows;

                    for (var i = 1; i < (rows.length - 1); i++) {
                        var shouldSwitch = false;
                        var x = rows[i].getElementsByTagName("TD")[columnIndex];
                        var y = rows[i + 1].getElementsByTagName("TD")[columnIndex];

                        var xContent = x.innerHTML.toLowerCase();
                        var yContent = y.innerHTML.toLowerCase();

                        // Try to parse as numbers
                        var xNum = parseFloat(xContent.replace(/[^0-9.-]/g, ''));
                        var yNum = parseFloat(yContent.replace(/[^0-9.-]/g, ''));

                        if (!isNaN(xNum) && !isNaN(yNum)) {
                            if (dir == "asc" && xNum > yNum) shouldSwitch = true;
                            if (dir == "desc" && xNum < yNum) shouldSwitch = true;
                        } else {
                            if (dir == "asc" && xContent > yContent) shouldSwitch = true;
                            if (dir == "desc" && xContent < yContent) shouldSwitch = true;
                        }

                        if (shouldSwitch) {
                            rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
                            switching = true;
                            switchcount++;
                        }
                    }

                    if (switchcount == 0 && dir == "asc") {
                        dir = "desc";
                        switching = true;
                    }
                }
            }

            // Add click handlers to headers
            document.addEventListener("DOMContentLoaded", function() {
                var headers = document.querySelectorAll("#performanceTable th");
                headers.forEach(function(header, index) {
                    header.addEventListener("click", function() {
                        sortTable(index);
                    });
                    header.title = "Click to sort";
                });
            });
            </script>
            """

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            {html_style}
        </head>
        <body>
            <div class="table-title">{title}</div>
            {table_html}
            {sorting_js}
        </body>
        </html>
        """

    def _generate_latex_table(self, df: pd.DataFrame, title: str) -> str:
        """Generate LaTeX table with proper formatting."""
        # Convert DataFrame to LaTeX
        latex_table = df.to_latex(
            escape=False,
            column_format="|" + "c|" * (len(df.columns) + 1),
            caption=title,
            label=f"tab:{title.lower().replace(' ', '_')}",
        )

        # Add document structure
        latex_document = f"""
        \\documentclass{{article}}
        \\usepackage{{booktabs}}
        \\usepackage{{array}}
        \\usepackage{{longtabu}}
        \\usepackage{{geometry}}

        \\begin{{document}}

        {latex_table}

        \\textit{{Note: *, **, *** indicate statistical significance at
                    5\\%, 1\\%, and 0.1\\% levels respectively.}}

        \\end{{document}}
        """

        return latex_document

    def display_interactive_table(
        self, table_df: pd.DataFrame, title: str = "Performance Table"
    ) -> None:
        """Display interactive table in Jupyter notebook environment."""
        if not HAS_IPYTHON:
            warnings.warn("IPython not available. Cannot display interactive table.", stacklevel=2)
            return

        html_content = self._generate_html_table(table_df, title)
        display(HTML(html_content))
