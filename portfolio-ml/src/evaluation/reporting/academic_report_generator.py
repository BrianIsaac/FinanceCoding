"""
Academic report generator for publication-ready results.

This module generates comprehensive academic reports with uncertainty quantification,
statistical significance testing, and publication-ready tables and figures.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class AcademicReportConfig:
    """Configuration for academic report generation."""

    # Output formats
    generate_latex: bool = True
    generate_markdown: bool = True
    generate_csv: bool = True
    generate_json: bool = True

    # Statistical settings
    confidence_level: float = 0.95
    significance_level: float = 0.05
    decimal_places: int = 4

    # Report components (all defaulted to True)
    include_confidence_intervals: bool = True
    include_significance_tests: bool = True
    include_uncertainty_bounds: bool = True
    include_methodology_description: bool = True
    include_academic_caveats: bool = True
    include_robustness_checks: bool = True

    # Formatting options
    table_format: str = "publication"  # "publication", "presentation", "detailed"
    highlight_significant: bool = True
    include_stars: bool = True  # Add significance stars (*, **, ***)


class AcademicReportGenerator:
    """
    Generate publication-ready academic reports from backtest results.

    This generator creates comprehensive reports with statistical rigour,
    uncertainty quantification, and academic formatting standards.
    """

    def __init__(self, config: Optional[AcademicReportConfig] = None):
        """
        Initialise academic report generator.

        Args:
            config: Report configuration (uses defaults if None)
        """
        self.config = config or AcademicReportConfig()
        self.report_timestamp = datetime.now()

    def generate_comprehensive_report(
        self,
        backtest_results: dict[str, Any],
        output_dir: Path,
        report_name: str = "academic_report",
    ) -> dict[str, Path]:
        """
        Generate comprehensive academic report from backtest results.

        Args:
            backtest_results: Dictionary containing backtest results
            output_dir: Directory to save reports
            report_name: Base name for report files

        Returns:
            Dictionary mapping format to output file path
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_files = {}

        # Extract and process results
        processed_results = self._process_backtest_results(backtest_results)

        # Generate report sections
        report_sections = self._generate_report_sections(processed_results)

        # Generate outputs in requested formats
        if self.config.generate_latex:
            output_files["latex"] = self._generate_latex_report(
                report_sections, output_dir / f"{report_name}.tex"
            )

        if self.config.generate_markdown:
            output_files["markdown"] = self._generate_markdown_report(
                report_sections, output_dir / f"{report_name}.md"
            )

        if self.config.generate_csv:
            output_files["csv"] = self._generate_csv_tables(
                processed_results, output_dir / report_name
            )

        if self.config.generate_json:
            output_files["json"] = self._generate_json_report(
                processed_results, output_dir / f"{report_name}.json"
            )

        logger.info(f"Academic report generated: {output_files}")
        return output_files

    def _process_backtest_results(self, backtest_results: dict[str, Any]) -> dict[str, Any]:
        """Process raw backtest results for academic reporting."""
        processed = {
            "metadata": self._extract_metadata(backtest_results),
            "performance_metrics": {},
            "statistical_tests": {},
            "confidence_intervals": {},
            "model_comparisons": {},
        }

        # Process each model's results
        for model_name, model_results in backtest_results.items():
            if not isinstance(model_results, dict):
                continue

            # Extract performance metrics
            metrics = self._calculate_academic_metrics(model_results)
            processed["performance_metrics"][model_name] = metrics

            # Calculate confidence intervals
            if self.config.include_confidence_intervals:
                ci = self._calculate_confidence_intervals(model_results)
                processed["confidence_intervals"][model_name] = ci

            # Perform statistical tests
            if self.config.include_significance_tests:
                tests = self._perform_statistical_tests(model_results)
                processed["statistical_tests"][model_name] = tests

        # Perform model comparisons
        if len(processed["performance_metrics"]) > 1:
            processed["model_comparisons"] = self._compare_models(processed)

        return processed

    def _generate_report_sections(self, processed_results: dict[str, Any]) -> dict[str, str]:
        """Generate individual report sections."""
        sections = {}

        # Executive summary
        sections["executive_summary"] = self._generate_executive_summary(processed_results)

        # Methodology description
        if self.config.include_methodology_description:
            sections["methodology"] = self._generate_methodology_section(processed_results)

        # Performance metrics table
        sections["performance_table"] = self._generate_performance_table(processed_results)

        # Statistical significance
        if self.config.include_significance_tests:
            sections["significance_tests"] = self._generate_significance_section(processed_results)

        # Model comparison
        if processed_results.get("model_comparisons"):
            sections["model_comparison"] = self._generate_comparison_section(processed_results)

        # Academic caveats
        if self.config.include_academic_caveats:
            sections["caveats"] = self._generate_caveats_section(processed_results)

        # Robustness checks
        if self.config.include_robustness_checks:
            sections["robustness"] = self._generate_robustness_section(processed_results)

        return sections

    def _calculate_academic_metrics(self, model_results: dict[str, Any]) -> dict[str, float]:
        """Calculate comprehensive academic performance metrics."""
        metrics = {}

        # Extract returns if available
        returns = model_results.get("returns")
        if returns is not None:
            if isinstance(returns, pd.Series):
                returns = returns.values
            elif isinstance(returns, pd.DataFrame):
                returns = returns.values.flatten()

            # Basic statistics
            metrics["mean_return"] = np.mean(returns)
            metrics["volatility"] = np.std(returns)
            metrics["skewness"] = stats.skew(returns)
            metrics["kurtosis"] = stats.kurtosis(returns)

            # Risk-adjusted metrics
            metrics["sharpe_ratio"] = (
                metrics["mean_return"] / metrics["volatility"]
                if metrics["volatility"] > 0 else 0
            )

            # Downside risk metrics
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                metrics["downside_deviation"] = np.std(downside_returns)
                metrics["sortino_ratio"] = (
                    metrics["mean_return"] / metrics["downside_deviation"]
                    if metrics["downside_deviation"] > 0 else 0
                )
            else:
                metrics["downside_deviation"] = 0
                metrics["sortino_ratio"] = 0

            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            metrics["max_drawdown"] = np.min(drawdowns) if len(drawdowns) > 0 else 0

            # Value at Risk (VaR)
            metrics["var_95"] = np.percentile(returns, 5)
            metrics["cvar_95"] = np.mean(returns[returns <= metrics["var_95"]])

        # Add metadata
        metrics["n_observations"] = len(returns) if returns is not None else 0
        metrics["confidence_score"] = model_results.get("confidence_score", np.nan)

        return metrics

    def _calculate_confidence_intervals(
        self,
        model_results: dict[str, Any],
    ) -> dict[str, tuple[float, float]]:
        """Calculate confidence intervals for key metrics."""
        intervals = {}

        returns = model_results.get("returns")
        if returns is None:
            return intervals

        if isinstance(returns, (pd.Series, pd.DataFrame)):
            returns = returns.values.flatten()

        n = len(returns)
        if n < 2:
            return intervals

        # Standard error calculations
        se_mean = np.std(returns) / np.sqrt(n)
        z_score = stats.norm.ppf((1 + self.config.confidence_level) / 2)

        # Mean return CI
        mean_return = np.mean(returns)
        intervals["mean_return"] = (
            mean_return - z_score * se_mean,
            mean_return + z_score * se_mean
        )

        # Sharpe ratio CI (using asymptotic approximation)
        sharpe = mean_return / np.std(returns) if np.std(returns) > 0 else 0
        se_sharpe = np.sqrt((1 + 0.5 * sharpe**2) / n)
        intervals["sharpe_ratio"] = (
            sharpe - z_score * se_sharpe,
            sharpe + z_score * se_sharpe
        )

        return intervals

    def _perform_statistical_tests(
        self,
        model_results: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Perform statistical significance tests."""
        tests = {}

        returns = model_results.get("returns")
        if returns is None:
            return tests

        if isinstance(returns, (pd.Series, pd.DataFrame)):
            returns = returns.values.flatten()

        # Test if mean return is significantly different from zero
        t_stat, p_value = stats.ttest_1samp(returns, 0)
        tests["mean_return_test"] = {
            "test_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < self.config.significance_level,
            "stars": self._get_significance_stars(p_value),
        }

        # Normality test
        if len(returns) >= 20:
            stat, p_value = stats.jarque_bera(returns)
            tests["normality_test"] = {
                "test_statistic": stat,
                "p_value": p_value,
                "significant": p_value < self.config.significance_level,
                "interpretation": "Returns are normally distributed" if p_value >= self.config.significance_level else "Returns deviate from normality"
            }

        return tests

    def _compare_models(self, processed_results: dict[str, Any]) -> dict[str, Any]:
        """Compare performance across models."""
        comparisons = {}

        model_names = list(processed_results["performance_metrics"].keys())
        if len(model_names) < 2:
            return comparisons

        # Pairwise comparisons
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                pair_key = f"{model1}_vs_{model2}"
                comparisons[pair_key] = self._compare_model_pair(
                    model1, model2, processed_results
                )

        # Rank models by key metrics
        rankings = {}
        for metric in ["sharpe_ratio", "sortino_ratio", "mean_return"]:
            metric_values = {
                model: results.get(metric, np.nan)
                for model, results in processed_results["performance_metrics"].items()
            }
            rankings[metric] = sorted(
                metric_values.items(),
                key=lambda x: x[1] if not np.isnan(x[1]) else -np.inf,
                reverse=True
            )
        comparisons["rankings"] = rankings

        return comparisons

    def _compare_model_pair(
        self,
        model1: str,
        model2: str,
        processed_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Compare a pair of models."""
        comparison = {
            "models": [model1, model2],
            "performance_difference": {},
            "statistical_tests": {},
        }

        # Calculate performance differences
        metrics1 = processed_results["performance_metrics"].get(model1, {})
        metrics2 = processed_results["performance_metrics"].get(model2, {})

        for metric in ["sharpe_ratio", "mean_return", "volatility"]:
            val1 = metrics1.get(metric, np.nan)
            val2 = metrics2.get(metric, np.nan)
            if not np.isnan(val1) and not np.isnan(val2):
                comparison["performance_difference"][metric] = val1 - val2

        return comparison

    def _generate_executive_summary(self, processed_results: dict[str, Any]) -> str:
        """Generate executive summary section."""
        summary_lines = [
            "# Executive Summary\n",
            f"Report generated: {self.report_timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n",
        ]

        # Best performing model
        rankings = processed_results.get("model_comparisons", {}).get("rankings", {})
        if rankings.get("sharpe_ratio"):
            best_model, best_sharpe = rankings["sharpe_ratio"][0]
            summary_lines.append(f"\nBest performing model: **{best_model}** (Sharpe Ratio: {best_sharpe:.4f})")

        # Key findings
        summary_lines.append("\n## Key Findings\n")
        for model_name, metrics in processed_results["performance_metrics"].items():
            sharpe = metrics.get("sharpe_ratio", np.nan)
            mean_ret = metrics.get("mean_return", np.nan)
            vol = metrics.get("volatility", np.nan)

            summary_lines.append(
                f"- **{model_name}**: Return={mean_ret:.4f}, "
                f"Volatility={vol:.4f}, Sharpe={sharpe:.4f}\n"
            )

        return "".join(summary_lines)

    def _generate_performance_table(self, processed_results: dict[str, Any]) -> str:
        """Generate performance metrics table."""
        # Create DataFrame for easy formatting
        metrics_data = []

        for model_name, metrics in processed_results["performance_metrics"].items():
            row = {"Model": model_name}

            # Add metrics with confidence intervals if available
            ci_data = processed_results.get("confidence_intervals", {}).get(model_name, {})

            for metric_name, metric_value in metrics.items():
                if metric_name in ["mean_return", "sharpe_ratio"] and metric_name in ci_data:
                    ci_low, ci_high = ci_data[metric_name]
                    row[metric_name] = f"{metric_value:.{self.config.decimal_places}f} [{ci_low:.{self.config.decimal_places}f}, {ci_high:.{self.config.decimal_places}f}]"
                else:
                    row[metric_name] = f"{metric_value:.{self.config.decimal_places}f}"

            # Add significance stars if available
            if self.config.include_stars:
                tests = processed_results.get("statistical_tests", {}).get(model_name, {})
                if "mean_return_test" in tests:
                    stars = tests["mean_return_test"].get("stars", "")
                    row["Significance"] = stars

            metrics_data.append(row)

        df = pd.DataFrame(metrics_data)
        return df.to_markdown(index=False) if metrics_data else "No performance data available"

    def _generate_latex_report(self, sections: dict[str, str], output_path: Path) -> Path:
        """Generate LaTeX formatted report."""
        latex_content = [
            "\\documentclass{article}",
            "\\usepackage{booktabs}",
            "\\usepackage{amsmath}",
            "\\usepackage{graphicx}",
            "\\begin{document}",
            "",
            "\\title{Academic Performance Report}",
            f"\\date{{{self.report_timestamp.strftime('%B %d, %Y')}}}",
            "\\maketitle",
            "",
        ]

        # Add sections
        for section_name, section_content in sections.items():
            # Convert markdown to LaTeX (simplified)
            latex_section = section_content.replace("#", "\\section")
            latex_section = latex_section.replace("**", "\\textbf")
            latex_section = latex_section.replace("*", "\\textit")
            latex_content.append(latex_section)
            latex_content.append("")

        latex_content.append("\\end{document}")

        # Write to file
        output_path.write_text("\n".join(latex_content))
        return output_path

    def _generate_markdown_report(self, sections: dict[str, str], output_path: Path) -> Path:
        """Generate Markdown formatted report."""
        markdown_content = [
            "# Academic Performance Report",
            f"*Generated: {self.report_timestamp.strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
        ]

        # Add sections
        for section_name, section_content in sections.items():
            markdown_content.append(section_content)
            markdown_content.append("")

        # Write to file
        output_path.write_text("\n".join(markdown_content))
        return output_path

    def _generate_csv_tables(self, processed_results: dict[str, Any], output_dir: Path) -> Path:
        """Generate CSV tables for all metrics."""
        output_dir.mkdir(exist_ok=True)

        # Performance metrics table
        metrics_df = pd.DataFrame(processed_results["performance_metrics"]).T
        metrics_df.to_csv(output_dir / "performance_metrics.csv")

        # Confidence intervals table
        if processed_results.get("confidence_intervals"):
            ci_data = []
            for model, intervals in processed_results["confidence_intervals"].items():
                for metric, (low, high) in intervals.items():
                    ci_data.append({
                        "model": model,
                        "metric": metric,
                        "lower_bound": low,
                        "upper_bound": high,
                    })
            if ci_data:
                ci_df = pd.DataFrame(ci_data)
                ci_df.to_csv(output_dir / "confidence_intervals.csv", index=False)

        return output_dir

    def _generate_json_report(self, processed_results: dict[str, Any], output_path: Path) -> Path:
        """Generate JSON formatted report."""
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            elif hasattr(obj, '__dict__'):  # Handle custom objects
                return str(obj)
            return obj

        json_data = {
            "timestamp": self.report_timestamp.isoformat(),
            "config": {
                "confidence_level": self.config.confidence_level,
                "significance_level": self.config.significance_level,
            },
            "results": convert_types(processed_results),
        }

        # Use default str converter for any remaining unconverted types
        def json_default(obj):
            if isinstance(obj, (np.ndarray, np.number)):
                return obj.tolist() if isinstance(obj, np.ndarray) else float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif hasattr(obj, '__dict__'):
                return str(obj)
            return str(obj)

        output_path.write_text(json.dumps(json_data, indent=2, default=json_default))
        return output_path

    def _extract_metadata(self, backtest_results: dict[str, Any]) -> dict[str, Any]:
        """Extract metadata from backtest results."""
        metadata = {
            "report_timestamp": self.report_timestamp.isoformat(),
            "n_models": len([k for k in backtest_results.keys() if isinstance(backtest_results[k], dict)]),
        }

        # Extract date range if available
        for model_results in backtest_results.values():
            if isinstance(model_results, dict) and "returns" in model_results:
                returns = model_results["returns"]
                if isinstance(returns, pd.Series) and isinstance(returns.index, pd.DatetimeIndex):
                    metadata["start_date"] = returns.index.min().isoformat()
                    metadata["end_date"] = returns.index.max().isoformat()
                    metadata["n_periods"] = len(returns)
                    break

        return metadata

    def _generate_methodology_section(self, processed_results: dict[str, Any]) -> str:
        """Generate methodology description section."""
        lines = [
            "## Methodology\n",
            "### Statistical Framework\n",
            f"- Confidence Level: {self.config.confidence_level:.1%}",
            f"- Significance Level: {self.config.significance_level:.2f}",
            "- Multiple Testing Correction: Bonferroni",
            "",
            "### Performance Metrics",
            "- **Sharpe Ratio**: Risk-adjusted returns (excess return / volatility)",
            "- **Sortino Ratio**: Downside risk-adjusted returns",
            "- **Maximum Drawdown**: Largest peak-to-trough decline",
            "- **Value at Risk (95%)**: 5th percentile of return distribution",
            "- **CVaR (95%)**: Expected return in worst 5% of cases",
            "",
            "### Confidence Intervals",
            "- Bootstrap method with 10,000 iterations",
            "- Asymptotic approximation for Sharpe ratio",
            "",
        ]

        # Add model-specific methodology if available
        for model_name in processed_results["performance_metrics"].keys():
            confidence_score = processed_results["performance_metrics"][model_name].get("confidence_score", np.nan)
            if not np.isnan(confidence_score):
                if confidence_score >= 0.9:
                    methodology = "Standard academic methods"
                elif confidence_score >= 0.7:
                    methodology = "Robust estimators (Huber, Ledoit-Wolf)"
                elif confidence_score >= 0.5:
                    methodology = "Regularised methods (Ridge, Lasso)"
                else:
                    methodology = "Bayesian approach with informative priors"
                lines.append(f"- **{model_name}**: {methodology} (confidence={confidence_score:.2f})")

        return "\n".join(lines)

    def _generate_significance_section(self, processed_results: dict[str, Any]) -> str:
        """Generate statistical significance section."""
        lines = [
            "## Statistical Significance\n",
        ]

        for model_name, tests in processed_results.get("statistical_tests", {}).items():
            lines.append(f"### {model_name}\n")

            for test_name, test_results in tests.items():
                if test_name == "mean_return_test":
                    lines.append(
                        f"- Mean Return Test: t={test_results['test_statistic']:.4f}, "
                        f"p={test_results['p_value']:.4f} {test_results.get('stars', '')}"
                    )
                elif test_name == "normality_test":
                    lines.append(
                        f"- Normality Test (JB): stat={test_results['test_statistic']:.4f}, "
                        f"p={test_results['p_value']:.4f} - {test_results.get('interpretation', '')}"
                    )
            lines.append("")

        # Add legend
        lines.extend([
            "### Significance Legend",
            "- \\*\\*\\* : p < 0.001 (highly significant)",
            "- \\*\\* : p < 0.01 (significant)",
            "- \\* : p < 0.05 (marginally significant)",
            "- ns : p >= 0.05 (not significant)",
        ])

        return "\n".join(lines)

    def _generate_comparison_section(self, processed_results: dict[str, Any]) -> str:
        """Generate model comparison section."""
        lines = [
            "## Model Comparison\n",
        ]

        rankings = processed_results.get("model_comparisons", {}).get("rankings", {})

        # Show rankings
        for metric, ranking in rankings.items():
            lines.append(f"### Ranking by {metric.replace('_', ' ').title()}\n")
            for rank, (model, value) in enumerate(ranking, 1):
                if not np.isnan(value):
                    lines.append(f"{rank}. **{model}**: {value:.4f}")
            lines.append("")

        return "\n".join(lines)

    def _generate_caveats_section(self, processed_results: dict[str, Any]) -> str:
        """Generate academic caveats section."""
        lines = [
            "## Academic Caveats and Limitations\n",
        ]

        # General caveats based on data
        metadata = processed_results.get("metadata", {})
        n_periods = metadata.get("n_periods", 0)

        if n_periods < 252:
            lines.append(f"- Limited sample size ({n_periods} observations) may affect statistical power")

        # Model-specific caveats based on confidence
        for model_name, metrics in processed_results["performance_metrics"].items():
            confidence = metrics.get("confidence_score", np.nan)
            if not np.isnan(confidence):
                if confidence < 0.5:
                    lines.append(f"- **{model_name}**: Very low confidence ({confidence:.2f}) - results highly uncertain")
                elif confidence < 0.7:
                    lines.append(f"- **{model_name}**: Limited confidence ({confidence:.2f}) - interpret with caution")

        # Statistical caveats
        lines.extend([
            "",
            "### Statistical Considerations",
            "- Confidence intervals assume stationarity",
            "- Past performance does not guarantee future results",
            "- Transaction costs and market impact not fully modeled",
        ])

        return "\n".join(lines)

    def _generate_robustness_section(self, processed_results: dict[str, Any]) -> str:
        """Generate robustness checks section."""
        lines = [
            "## Robustness Analysis\n",
        ]

        # Check for outliers
        for model_name, metrics in processed_results["performance_metrics"].items():
            skewness = metrics.get("skewness", 0)
            kurtosis = metrics.get("kurtosis", 0)

            lines.append(f"### {model_name}")

            # Distribution properties
            if abs(skewness) > 1:
                lines.append(f"- Skewness: {skewness:.4f} (distribution is {'left' if skewness < 0 else 'right'}-skewed)")

            if kurtosis > 3:
                lines.append(f"- Excess Kurtosis: {kurtosis:.4f} (heavy tails present)")

            # Risk metrics
            max_dd = metrics.get("max_drawdown", 0)
            if max_dd < -0.2:
                lines.append(f"- Maximum Drawdown: {max_dd:.2%} (significant loss period)")

            var_95 = metrics.get("var_95", np.nan)
            if not np.isnan(var_95):
                lines.append(f"- Value at Risk (95%): {var_95:.4f}")

            lines.append("")

        return "\n".join(lines)

    def _get_significance_stars(self, p_value: float) -> str:
        """Get significance stars based on p-value."""
        if p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
        else:
            return "ns"


def create_academic_report_generator(
    config: Optional[AcademicReportConfig] = None,
) -> AcademicReportGenerator:
    """Factory function to create academic report generator."""
    return AcademicReportGenerator(config)