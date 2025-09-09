"""
Comprehensive performance analysis framework for detailed statistical reporting.

This module provides comprehensive performance analysis including statistical significance,
confidence intervals, bootstrap distributions, performance attribution, and risk-adjusted
return analysis for institutional-grade reporting.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("Matplotlib/Seaborn not available. Static plotting disabled.", stacklevel=2)

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    warnings.warn("Plotly not available. Interactive plotting disabled.", stacklevel=2)

from src.evaluation.reporting.tables import PerformanceComparisonTables, TableConfig
from src.evaluation.validation.reporting import PublicationReadyStatisticalReporting
from src.evaluation.validation.significance import StatisticalValidation


@dataclass
class PerformanceAnalysisConfig:
    """Configuration for comprehensive performance analysis."""

    confidence_levels: list[float] = None
    bootstrap_iterations: int = 10000
    rolling_window_days: int = 252
    significance_threshold: float = 0.05
    decimal_places: int = 4
    include_attribution: bool = True
    risk_free_rate: float = 0.02

    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.90, 0.95, 0.99]


class ComprehensivePerformanceAnalyzer:
    """
    Comprehensive performance analyzer for institutional reporting.

    Provides detailed statistical performance analysis with confidence intervals,
    bootstrap distributions, performance attribution, and risk-adjusted metrics.
    """

    def __init__(self, config: PerformanceAnalysisConfig = None):
        """
        Initialize comprehensive performance analyzer.

        Args:
            config: Performance analysis configuration
        """
        self.config = config or PerformanceAnalysisConfig()
        self.table_generator = PerformanceComparisonTables(
            TableConfig(
                decimal_places=self.config.decimal_places,
                significance_levels=[self.config.significance_threshold],
                include_confidence_intervals=True,
                include_rankings=True,
            )
        )
        self.statistical_reporter = PublicationReadyStatisticalReporting(
            significance_levels=[self.config.significance_threshold],
            decimal_places=self.config.decimal_places,
        )
        self.statistical_validator = StatisticalValidation()

    def generate_comprehensive_performance_tables(
        self,
        performance_results: dict[str, pd.DataFrame],
        statistical_results: dict[str, Any],
        baseline_results: dict[str, pd.DataFrame] = None,
        output_dir: Path = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Generate comprehensive performance tables with statistical indicators.

        Args:
            performance_results: Model performance results by approach
            statistical_results: Statistical significance test results
            baseline_results: Baseline performance results for comparison
            output_dir: Optional output directory for saving tables

        Returns:
            Dictionary containing various performance analysis tables
        """
        tables = {}

        # Main performance summary table
        tables["performance_summary"] = self._create_performance_summary_table(
            performance_results, statistical_results, baseline_results
        )

        # Risk-adjusted returns table with confidence intervals
        tables["risk_adjusted_analysis"] = self._create_risk_adjusted_analysis_table(
            performance_results, statistical_results
        )

        # Statistical significance detailed table
        tables["statistical_significance"] = self._create_statistical_significance_table(
            performance_results, statistical_results
        )

        # Rolling window consistency analysis
        tables["consistency_analysis"] = self._create_consistency_analysis_table(
            performance_results
        )

        # Performance attribution table (if attribution data available)
        if self.config.include_attribution:
            tables["attribution_analysis"] = self._create_attribution_analysis_table(
                performance_results
            )

        # Bootstrap confidence intervals table
        tables["confidence_intervals"] = self._create_confidence_intervals_table(
            performance_results, statistical_results
        )

        if output_dir:
            self._save_performance_tables(tables, output_dir)

        return tables

    def _create_performance_summary_table(
        self,
        performance_results: dict[str, pd.DataFrame],
        statistical_results: dict[str, Any],
        baseline_results: dict[str, pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Create main performance summary table."""
        summary_data = []

        for model_name, results in performance_results.items():
            if results.empty:
                continue

            # Calculate summary statistics
            metrics = self._calculate_summary_metrics(results)

            # Add statistical significance
            sig_indicator = self._get_significance_indicator(model_name, statistical_results)

            # Add baseline comparison if available
            baseline_comparison = ""
            if baseline_results and "equal_weight" in baseline_results:
                baseline_metrics = self._calculate_summary_metrics(baseline_results["equal_weight"])
                sharpe_diff = metrics["sharpe_ratio"] - baseline_metrics["sharpe_ratio"]
                baseline_comparison = (
                    f"+{sharpe_diff:.3f}" if sharpe_diff > 0 else f"{sharpe_diff:.3f}"
                )

            summary_data.append(
                {
                    "Model": model_name,
                    "Sharpe Ratio": f"{metrics['sharpe_ratio']:.4f}{sig_indicator}",
                    "vs Baseline": baseline_comparison,
                    "Information Ratio": f"{metrics['information_ratio']:.4f}",
                    "Annual Return (%)": f"{metrics['annual_return']*100:.2f}",
                    "Volatility (%)": f"{metrics['volatility']*100:.2f}",
                    "Max Drawdown (%)": f"{abs(metrics['max_drawdown'])*100:.2f}",
                    "Calmar Ratio": f"{metrics['calmar_ratio']:.4f}",
                    "Sortino Ratio": f"{metrics['sortino_ratio']:.4f}",
                    "Win Rate (%)": f"{metrics['win_rate']*100:.1f}",
                    "Avg Win/Loss": f"{metrics['avg_win_loss']:.2f}",
                }
            )

        summary_df = pd.DataFrame(summary_data)

        # Add rankings
        if not summary_df.empty:
            numeric_cols = ["Sharpe Ratio", "Information Ratio", "Calmar Ratio", "Sortino Ratio"]
            for col in numeric_cols:
                if col in summary_df.columns:
                    # Extract numeric values for ranking
                    numeric_values = summary_df[col].str.extract(r"(\d+\.\d+)")[0].astype(float)
                    summary_df[f"{col} Rank"] = numeric_values.rank(ascending=False).astype(int)

        return summary_df

    def _create_risk_adjusted_analysis_table(
        self, performance_results: dict[str, pd.DataFrame], statistical_results: dict[str, Any]
    ) -> pd.DataFrame:
        """Create risk-adjusted returns analysis table."""
        risk_analysis_data = []

        for model_name, results in performance_results.items():
            if results.empty:
                continue

            metrics = self._calculate_summary_metrics(results)

            # Calculate risk metrics
            returns = (
                results["returns"]
                if "returns" in results.columns
                else results.get("daily_returns", pd.Series())
            )
            if returns.empty:
                continue

            # Value at Risk calculations
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            cvar_95 = returns[returns <= var_95].mean()
            cvar_99 = returns[returns <= var_99].mean()

            # Tail ratio
            tail_ratio = abs(returns.quantile(0.95)) / abs(returns.quantile(0.05))

            # Statistical significance confidence intervals
            confidence_intervals = self._get_confidence_intervals(model_name, statistical_results)

            risk_analysis_data.append(
                {
                    "Model": model_name,
                    "Sharpe Ratio": f"{metrics['sharpe_ratio']:.4f}",
                    "Sharpe 95% CI": confidence_intervals.get("sharpe_95_ci", "N/A"),
                    "Information Ratio": f"{metrics['information_ratio']:.4f}",
                    "Info 95% CI": confidence_intervals.get("information_95_ci", "N/A"),
                    "VaR 95% (%)": f"{var_95*100:.2f}",
                    "CVaR 95% (%)": f"{cvar_95*100:.2f}",
                    "VaR 99% (%)": f"{var_99*100:.2f}",
                    "CVaR 99% (%)": f"{cvar_99*100:.2f}",
                    "Tail Ratio": f"{tail_ratio:.2f}",
                    "Skewness": f"{returns.skew():.3f}",
                    "Kurtosis": f"{returns.kurtosis():.3f}",
                }
            )

        return pd.DataFrame(risk_analysis_data)

    def _create_statistical_significance_table(
        self, performance_results: dict[str, pd.DataFrame], statistical_results: dict[str, Any]
    ) -> pd.DataFrame:
        """Create detailed statistical significance table."""
        significance_data = []

        for model_name in performance_results.keys():
            if model_name not in statistical_results:
                continue

            model_stats = statistical_results[model_name]

            significance_data.append(
                {
                    "Model": model_name,
                    "vs Equal Weight p-value": f"{model_stats.get('pvalue_vs_equal_weight', 1.0):.6f}",
                    "vs Market Cap p-value": f"{model_stats.get('pvalue_vs_market_cap', 1.0):.6f}",
                    "Jobson-Korkie Test": f"{model_stats.get('jobson_korkie_pvalue', 1.0):.6f}",
                    "Bootstrap p-value": f"{model_stats.get('bootstrap_pvalue', 1.0):.6f}",
                    "Effect Size": f"{model_stats.get('effect_size', 0.0):.4f}",
                    "Significance Level": self._classify_significance(
                        model_stats.get("pvalue_vs_equal_weight", 1.0)
                    ),
                    "Bonferroni Adjusted": f"{model_stats.get('bonferroni_adjusted_pvalue', 1.0):.6f}",
                    "FDR Adjusted": f"{model_stats.get('fdr_adjusted_pvalue', 1.0):.6f}",
                }
            )

        return pd.DataFrame(significance_data)

    def _create_consistency_analysis_table(
        self, performance_results: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Create rolling window consistency analysis table."""
        consistency_data = []

        for model_name, results in performance_results.items():
            if results.empty or "rolling_sharpe" not in results.columns:
                continue

            rolling_sharpe = results["rolling_sharpe"].dropna()
            results.get("rolling_returns", pd.Series()).dropna()

            if rolling_sharpe.empty:
                continue

            # Consistency metrics
            sharpe_std = rolling_sharpe.std()
            sharpe_consistency = (
                1 - (sharpe_std / abs(rolling_sharpe.mean())) if rolling_sharpe.mean() != 0 else 0
            )
            positive_periods = (rolling_sharpe > 0).sum() / len(rolling_sharpe)

            # Performance persistence
            above_median = rolling_sharpe > rolling_sharpe.median()
            persistence = (
                sum(above_median[1:] == above_median[:-1]) / (len(above_median) - 1)
                if len(above_median) > 1
                else 0
            )

            consistency_data.append(
                {
                    "Model": model_name,
                    "Avg Rolling Sharpe": f"{rolling_sharpe.mean():.4f}",
                    "Sharpe Volatility": f"{sharpe_std:.4f}",
                    "Consistency Score": f"{sharpe_consistency:.4f}",
                    "Positive Periods (%)": f"{positive_periods*100:.1f}",
                    "Performance Persistence": f"{persistence:.4f}",
                    "Best Rolling Period": f"{rolling_sharpe.max():.4f}",
                    "Worst Rolling Period": f"{rolling_sharpe.min():.4f}",
                    "Sharpe Quartile Range": f"{rolling_sharpe.quantile(0.75) - rolling_sharpe.quantile(0.25):.4f}",
                }
            )

        return pd.DataFrame(consistency_data)

    def _create_attribution_analysis_table(
        self, performance_results: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Create performance attribution analysis table."""
        attribution_data = []

        for model_name, results in performance_results.items():
            if results.empty:
                continue

            # Calculate attribution metrics where possible
            returns = results.get("returns", pd.Series())
            if returns.empty:
                continue

            # Sector attribution (if sector data available)
            sector_contribution = "N/A"
            if "sector_returns" in results.columns:
                sector_returns = results["sector_returns"]
                if not sector_returns.empty:
                    sector_contribution = f"{sector_returns.mean()*100:.2f}"

            # Factor attribution (if factor loadings available)
            factor_attribution = "N/A"
            if "factor_loadings" in results.columns:
                factor_loadings = results["factor_loadings"]
                if not factor_loadings.empty:
                    factor_attribution = f"{factor_loadings.mean():.3f}"

            # Alpha and beta calculation (if benchmark available)
            alpha, beta = "N/A", "N/A"
            if "benchmark_returns" in results.columns:
                benchmark_returns = results["benchmark_returns"]
                if not benchmark_returns.empty and len(returns) == len(benchmark_returns):
                    try:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            benchmark_returns, returns
                        )
                        alpha = f"{intercept*252*100:.2f}%"  # Annualized alpha
                        beta = f"{slope:.3f}"
                    except Exception:
                        pass

            attribution_data.append(
                {
                    "Model": model_name,
                    "Total Return (%)": f"{returns.sum()*100:.2f}",
                    "Alpha (%)": alpha,
                    "Beta": beta,
                    "Sector Contribution": sector_contribution,
                    "Factor Loading": factor_attribution,
                    "Active Share": "N/A",  # Would need benchmark weights
                    "Tracking Error (%)": f"{returns.std()*np.sqrt(252)*100:.2f}",
                    "Information Ratio": f"{(returns.mean()/returns.std())*np.sqrt(252):.4f}",
                }
            )

        return pd.DataFrame(attribution_data)

    def _create_confidence_intervals_table(
        self, performance_results: dict[str, pd.DataFrame], statistical_results: dict[str, Any]
    ) -> pd.DataFrame:
        """Create bootstrap confidence intervals table."""
        confidence_data = []

        for model_name in performance_results.keys():
            if model_name not in statistical_results:
                continue

            model_stats = statistical_results[model_name]
            bootstrap_results = model_stats.get("bootstrap_results", {})

            # Extract confidence intervals for key metrics
            confidence_data.append(
                {
                    "Model": model_name,
                    "Sharpe Ratio 90% CI": self._format_confidence_interval(
                        bootstrap_results.get("sharpe_90_ci", [np.nan, np.nan])
                    ),
                    "Sharpe Ratio 95% CI": self._format_confidence_interval(
                        bootstrap_results.get("sharpe_95_ci", [np.nan, np.nan])
                    ),
                    "Sharpe Ratio 99% CI": self._format_confidence_interval(
                        bootstrap_results.get("sharpe_99_ci", [np.nan, np.nan])
                    ),
                    "Annual Return 95% CI": self._format_confidence_interval(
                        bootstrap_results.get("return_95_ci", [np.nan, np.nan]), percentage=True
                    ),
                    "Volatility 95% CI": self._format_confidence_interval(
                        bootstrap_results.get("volatility_95_ci", [np.nan, np.nan]), percentage=True
                    ),
                    "Max Drawdown 95% CI": self._format_confidence_interval(
                        bootstrap_results.get("drawdown_95_ci", [np.nan, np.nan]), percentage=True
                    ),
                    "Bootstrap Iterations": bootstrap_results.get(
                        "iterations", self.config.bootstrap_iterations
                    ),
                }
            )

        return pd.DataFrame(confidence_data)

    def _calculate_summary_metrics(self, results: pd.DataFrame) -> dict[str, float]:
        """Calculate summary performance metrics."""
        if results.empty:
            return {}

        returns = results.get("returns", results.get("daily_returns", pd.Series()))
        if returns.empty:
            return {}

        # Basic metrics
        annual_return = returns.mean() * 252
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (
            (annual_return - self.config.risk_free_rate) / volatility if volatility > 0 else 0
        )

        # Drawdown calculation
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()

        # Other ratios
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Sortino ratio (using downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = (
            downside_returns.std() * np.sqrt(252) if not downside_returns.empty else volatility
        )
        sortino_ratio = (
            (annual_return - self.config.risk_free_rate) / downside_deviation
            if downside_deviation > 0
            else 0
        )

        # Win rate and average win/loss
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        avg_win_loss = (
            abs(positive_returns.mean() / negative_returns.mean())
            if not positive_returns.empty
            and not negative_returns.empty
            and negative_returns.mean() != 0
            else 0
        )

        # Information ratio (assuming benchmark return of 0 for simplicity)
        information_ratio = sharpe_ratio  # Simplified - would need actual benchmark

        return {
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "information_ratio": information_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "sortino_ratio": sortino_ratio,
            "win_rate": win_rate,
            "avg_win_loss": avg_win_loss,
        }

    def _get_significance_indicator(
        self, model_name: str, statistical_results: dict[str, Any]
    ) -> str:
        """Get statistical significance indicator for a model."""
        if not statistical_results or model_name not in statistical_results:
            return ""

        pvalue = statistical_results[model_name].get("pvalue_vs_equal_weight", 1.0)

        if pvalue < 0.001:
            return "***"
        elif pvalue < 0.01:
            return "**"
        elif pvalue < 0.05:
            return "*"
        else:
            return ""

    def _get_confidence_intervals(
        self, model_name: str, statistical_results: dict[str, Any]
    ) -> dict[str, str]:
        """Get confidence intervals for a model."""
        if not statistical_results or model_name not in statistical_results:
            return {}

        bootstrap_results = statistical_results[model_name].get("bootstrap_results", {})

        intervals = {}
        if "sharpe_95_ci" in bootstrap_results:
            intervals["sharpe_95_ci"] = self._format_confidence_interval(
                bootstrap_results["sharpe_95_ci"]
            )
        if "information_95_ci" in bootstrap_results:
            intervals["information_95_ci"] = self._format_confidence_interval(
                bootstrap_results["information_95_ci"]
            )

        return intervals

    def _classify_significance(self, pvalue: float) -> str:
        """Classify statistical significance level."""
        if pvalue < 0.001:
            return "Highly Significant (p<0.001)"
        elif pvalue < 0.01:
            return "Very Significant (p<0.01)"
        elif pvalue < 0.05:
            return "Significant (p<0.05)"
        elif pvalue < 0.10:
            return "Marginally Significant (p<0.10)"
        else:
            return "Not Significant (p≥0.10)"

    def _format_confidence_interval(self, ci: list[float], percentage: bool = False) -> str:
        """Format confidence interval for display."""
        if not ci or len(ci) != 2 or any(np.isnan(ci)):
            return "N/A"

        if percentage:
            return f"[{ci[0]*100:.2f}%, {ci[1]*100:.2f}%]"
        else:
            return f"[{ci[0]:.4f}, {ci[1]:.4f}]"

    def _save_performance_tables(self, tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
        """Save performance analysis tables to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for table_name, table_df in tables.items():
            if not table_df.empty:
                # Save as CSV
                csv_path = output_dir / f"{table_name}.csv"
                table_df.to_csv(csv_path, index=False)

                # Save as HTML for better formatting
                html_path = output_dir / f"{table_name}.html"
                table_df.to_html(
                    html_path, index=False, escape=False, classes="table table-striped"
                )
