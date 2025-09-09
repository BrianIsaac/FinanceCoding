"""Comprehensive sensitivity analysis reporting and visualization framework."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .constraints import ConstraintAnalyzer
from .hyperparameters import HyperparameterTester
from .portfolio_size import PortfolioSizeAnalyzer
from .transaction_costs import TransactionCostAnalyzer

logger = logging.getLogger(__name__)


class SensitivityReporter:
    """Comprehensive sensitivity analysis reporting framework."""

    def __init__(
        self,
        hyperparameter_tester: Optional[HyperparameterTester] = None,
        transaction_cost_analyzer: Optional[TransactionCostAnalyzer] = None,
        portfolio_size_analyzer: Optional[PortfolioSizeAnalyzer] = None,
        constraint_analyzer: Optional[ConstraintAnalyzer] = None,
    ):
        """Initialize sensitivity reporter.

        Args:
            hyperparameter_tester: Hyperparameter sensitivity analyzer
            transaction_cost_analyzer: Transaction cost analyzer
            portfolio_size_analyzer: Portfolio size analyzer
            constraint_analyzer: Constraint violation analyzer
        """
        self.hyperparameter_tester = hyperparameter_tester
        self.transaction_cost_analyzer = transaction_cost_analyzer
        self.portfolio_size_analyzer = portfolio_size_analyzer
        self.constraint_analyzer = constraint_analyzer

    def generate_comprehensive_report(
        self,
        output_dir: str,
        target_metric: str = "sharpe_ratio",
        include_visualizations: bool = True,
    ) -> dict[str, str]:
        """Generate comprehensive sensitivity analysis report.

        Args:
            output_dir: Directory to save report and data files
            target_metric: Performance metric for analysis
            include_visualizations: Whether to generate visualization data

        Returns:
            Dictionary mapping report sections to file paths
        """
        logger.info(f"Generating comprehensive sensitivity analysis report in {output_dir}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report_files = {}

        # 1. Executive Summary
        summary_file = output_path / "executive_summary.md"
        self._generate_executive_summary(summary_file, target_metric)
        report_files["executive_summary"] = str(summary_file)

        # 2. Hyperparameter Analysis Report
        if self.hyperparameter_tester:
            hyperparam_file = output_path / "hyperparameter_analysis.md"
            self._generate_hyperparameter_report(hyperparam_file, target_metric)
            report_files["hyperparameter_analysis"] = str(hyperparam_file)

        # 3. Transaction Cost Analysis Report
        if self.transaction_cost_analyzer:
            cost_file = output_path / "transaction_cost_analysis.md"
            self._generate_transaction_cost_report(cost_file, target_metric)
            report_files["transaction_cost_analysis"] = str(cost_file)

        # 4. Portfolio Size Analysis Report
        if self.portfolio_size_analyzer:
            size_file = output_path / "portfolio_size_analysis.md"
            self._generate_portfolio_size_report(size_file, target_metric)
            report_files["portfolio_size_analysis"] = str(size_file)

        # 5. Constraint Analysis Report
        if self.constraint_analyzer:
            constraint_file = output_path / "constraint_analysis.md"
            self._generate_constraint_analysis_report(constraint_file, target_metric)
            report_files["constraint_analysis"] = str(constraint_file)

        # 6. Robustness Summary
        robustness_file = output_path / "robustness_summary.md"
        self._generate_robustness_summary(robustness_file, target_metric)
        report_files["robustness_summary"] = str(robustness_file)

        # 7. Export visualization data if requested
        if include_visualizations:
            viz_dir = output_path / "visualization_data"
            viz_dir.mkdir(exist_ok=True)
            viz_files = self._export_visualization_data(viz_dir, target_metric)
            report_files.update(viz_files)

        # 8. Export raw data
        data_dir = output_path / "raw_data"
        data_dir.mkdir(exist_ok=True)
        data_files = self._export_raw_data(data_dir)
        report_files.update(data_files)

        logger.info(f"Comprehensive report generated with {len(report_files)} files")

        return report_files

    def _generate_executive_summary(self, output_file: Path, target_metric: str) -> None:
        """Generate executive summary report.

        Args:
            output_file: Path to save executive summary
            target_metric: Performance metric for analysis
        """
        sections = []

        sections.append("# Sensitivity Analysis Executive Summary")
        sections.append("")
        sections.append(f"**Target Metric**: {target_metric}")
        sections.append(f"**Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        sections.append("")

        # Overall findings
        sections.append("## Key Findings")
        sections.append("")

        # Hyperparameter findings
        if self.hyperparameter_tester and self.hyperparameter_tester.test_results:
            sections.append("### Hyperparameter Sensitivity")

            total_models = len(self.hyperparameter_tester.test_results)
            sections.append(f"- **Models Analyzed**: {total_models}")

            # Count significant parameters across all models
            significant_params = 0
            total_params = 0

            for model_results in self.hyperparameter_tester.test_results.values():
                total_params += len(model_results)
                significant_params += len(
                    [
                        r
                        for r in model_results
                        if r.statistical_significance.get("p_value", 1.0) < 0.05
                    ]
                )

            significance_rate = (significant_params / total_params * 100) if total_params > 0 else 0
            sections.append(
                f"- **Parameter Significance Rate**: {significance_rate:.1f}% ({significant_params}/{total_params})"
            )

        # Transaction cost findings
        if self.transaction_cost_analyzer and self.transaction_cost_analyzer.cost_impact_results:
            sections.append("")
            sections.append("### Transaction Cost Impact")

            cost_summary = self.transaction_cost_analyzer.get_cost_sensitivity_summary(
                target_metric
            )
            sections.append(f"- **Models Analyzed**: {cost_summary['models_analyzed']}")
            sections.append(f"- **Most Sensitive Model**: {cost_summary['most_sensitive_model']}")
            sections.append(
                f"- **Ranking Stability Score**: {cost_summary['ranking_stability_score']:.3f}"
            )

        # Portfolio size findings
        if self.portfolio_size_analyzer and self.portfolio_size_analyzer.size_analysis_results:
            sections.append("")
            sections.append("### Portfolio Size Analysis")

            recommendations = (
                self.portfolio_size_analyzer.get_optimal_portfolio_size_recommendations()
            )
            sections.append(f"- **Models Analyzed**: {len(recommendations)}")

            optimal_sizes = [r["optimal_size"] for r in recommendations.values()]
            if optimal_sizes:
                avg_optimal_size = np.mean(optimal_sizes)
                sections.append(f"- **Average Optimal Size**: {avg_optimal_size:.0f} positions")

        # Constraint findings
        if self.constraint_analyzer and self.constraint_analyzer.constraint_analysis_results:
            sections.append("")
            sections.append("### Constraint Analysis")

            constraint_recommendations = (
                self.constraint_analyzer.get_constraint_optimization_recommendations()
            )
            sections.append(f"- **Models Analyzed**: {len(constraint_recommendations)}")

            avg_robustness = np.mean(
                [r["optimal_robustness_score"] for r in constraint_recommendations.values()]
            )
            sections.append(f"- **Average Robustness Score**: {avg_robustness:.3f}")

        # Recommendations
        sections.append("")
        sections.append("## Recommendations")
        sections.append("")

        if self.hyperparameter_tester and self.hyperparameter_tester.test_results:
            sections.append("### Hyperparameter Tuning")
            for model_type in self.hyperparameter_tester.test_results.keys():
                optimal_params = self.hyperparameter_tester.get_optimal_hyperparameters(model_type)
                if optimal_params:
                    sections.append(
                        f"- **{model_type.upper()}**: {len(optimal_params)} significant parameters identified"
                    )

        sections.append("")
        sections.append("### Implementation Priority")
        sections.append(
            "1. Apply identified optimal hyperparameters for significant performance improvements"
        )
        sections.append("2. Implement appropriate transaction cost modeling based on trading style")
        sections.append("3. Set portfolio size constraints based on diversification analysis")
        sections.append(
            "4. Configure constraint enforcement levels based on robustness requirements"
        )

        # Write report
        with open(output_file, "w") as f:
            f.write("\n".join(sections))

        logger.info(f"Executive summary written to {output_file}")

    def _generate_hyperparameter_report(self, output_file: Path, target_metric: str) -> None:
        """Generate detailed hyperparameter analysis report.

        Args:
            output_file: Path to save hyperparameter report
            target_metric: Performance metric for analysis
        """
        if not self.hyperparameter_tester or not self.hyperparameter_tester.test_results:
            return

        sections = []

        sections.append("# Hyperparameter Sensitivity Analysis")
        sections.append("")
        sections.append(f"**Target Metric**: {target_metric}")
        sections.append("")

        for model_type, results in self.hyperparameter_tester.test_results.items():
            sections.append(f"## {model_type.upper()} Model Analysis")
            sections.append("")

            # Parameter importance ranking
            importance_df = self.hyperparameter_tester.compare_hyperparameter_importance(
                model_type, target_metric
            )

            sections.append("### Parameter Importance Ranking")
            sections.append("")
            sections.append("| Parameter | Effect Size | P-Value | Significance | Optimal Value |")
            sections.append("|-----------|-------------|---------|--------------|---------------|")

            for _, row in importance_df.iterrows():
                significance = "✓" if row["significance"] == "Yes" else "✗"
                sections.append(
                    f"| {row['parameter']} | {row['effect_size']:.4f} | {row['p_value']:.6f} | {significance} | {row['optimal_value']} |"
                )

            sections.append("")

            # Optimal configuration
            optimal_params = self.hyperparameter_tester.get_optimal_hyperparameters(model_type)
            if optimal_params:
                sections.append("### Recommended Configuration")
                sections.append("")
                for param, value in optimal_params.items():
                    sections.append(f"- **{param}**: {value}")
                sections.append("")

        # Statistical summary
        sections.append("## Statistical Summary")
        sections.append("")

        all_p_values = []
        all_effect_sizes = []

        for results in self.hyperparameter_tester.test_results.values():
            for result in results:
                all_p_values.append(result.statistical_significance.get("p_value", 1.0))
                all_effect_sizes.append(result.effect_size)

        if all_p_values:
            significant_count = sum(1 for p in all_p_values if p < 0.05)
            sections.append(f"- **Total Parameters Tested**: {len(all_p_values)}")
            sections.append(
                f"- **Significant Parameters**: {significant_count} ({significant_count/len(all_p_values)*100:.1f}%)"
            )
            sections.append(f"- **Average Effect Size**: {np.mean(all_effect_sizes):.4f}")
            sections.append(f"- **Median P-Value**: {np.median(all_p_values):.6f}")

        # Write report
        with open(output_file, "w") as f:
            f.write("\n".join(sections))

        logger.info(f"Hyperparameter analysis report written to {output_file}")

    def _generate_transaction_cost_report(self, output_file: Path, target_metric: str) -> None:
        """Generate transaction cost analysis report.

        Args:
            output_file: Path to save transaction cost report
            target_metric: Performance metric for analysis
        """
        if (
            not self.transaction_cost_analyzer
            or not self.transaction_cost_analyzer.cost_impact_results
        ):
            return

        sections = []

        sections.append("# Transaction Cost Impact Analysis")
        sections.append("")
        sections.append(f"**Target Metric**: {target_metric}")
        sections.append("")

        # Cost sensitivity summary
        cost_summary = self.transaction_cost_analyzer.get_cost_sensitivity_summary(target_metric)

        sections.append("## Cost Sensitivity Summary")
        sections.append("")
        sections.append(f"- **Models Analyzed**: {cost_summary['models_analyzed']}")
        sections.append(f"- **Cost Scenarios**: {cost_summary['cost_scenarios']}")
        sections.append(f"- **Most Sensitive Model**: {cost_summary['most_sensitive_model']}")
        sections.append(f"- **Least Sensitive Model**: {cost_summary['least_sensitive_model']}")
        sections.append(
            f"- **Ranking Stability Score**: {cost_summary['ranking_stability_score']:.3f}"
        )
        sections.append("")

        # Average impact by scenario
        sections.append("## Impact by Cost Scenario")
        sections.append("")
        sections.append("| Cost Scenario | Avg Impact (%) |")
        sections.append("|---------------|----------------|")

        for scenario, impact in cost_summary["avg_impact_by_scenario"].items():
            sections.append(f"| {scenario.title()} | {impact:.2f}% |")

        sections.append("")

        # Ranking stability analysis
        stability_df = self.transaction_cost_analyzer.analyze_ranking_stability(target_metric)

        sections.append("## Model Ranking Stability")
        sections.append("")
        sections.append(
            "| Model | Max Ranking Change | Avg Performance Impact | Significant Impacts |"
        )
        sections.append(
            "|-------|--------------------|-------------------------|---------------------|"
        )

        # Group by model to get unique entries
        model_summary = (
            stability_df.groupby("model_type")
            .agg(
                {
                    "ranking_change": lambda x: max(abs(x)),
                    "performance_impact_pct": lambda x: np.mean(abs(x)),
                    "is_significant": "sum",
                }
            )
            .reset_index()
        )

        for _, row in model_summary.iterrows():
            sections.append(
                f"| {row['model_type'].upper()} | {row['ranking_change']:.0f} | {row['performance_impact_pct']:.2f}% | {row['is_significant']:.0f} |"
            )

        sections.append("")

        # Recommendations
        sections.append("## Recommendations")
        sections.append("")
        sections.append("### Trading Style Recommendations")

        if cost_summary["most_sensitive_model"]:
            sections.append(
                f"- **{cost_summary['most_sensitive_model'].upper()}**: Consider lower turnover strategies due to high cost sensitivity"
            )

        if cost_summary["least_sensitive_model"]:
            sections.append(
                f"- **{cost_summary['least_sensitive_model'].upper()}**: Can accommodate higher frequency trading with minimal impact"
            )

        # Write report
        with open(output_file, "w") as f:
            f.write("\n".join(sections))

        logger.info(f"Transaction cost analysis report written to {output_file}")

    def _generate_portfolio_size_report(self, output_file: Path, target_metric: str) -> None:
        """Generate portfolio size analysis report.

        Args:
            output_file: Path to save portfolio size report
            target_metric: Performance metric for analysis
        """
        if (
            not self.portfolio_size_analyzer
            or not self.portfolio_size_analyzer.size_analysis_results
        ):
            return

        sections = []

        sections.append("# Portfolio Size Sensitivity Analysis")
        sections.append("")
        sections.append(f"**Target Metric**: {target_metric}")
        sections.append("")

        # Optimal size recommendations
        recommendations = self.portfolio_size_analyzer.get_optimal_portfolio_size_recommendations()

        sections.append("## Optimal Portfolio Size Recommendations")
        sections.append("")
        sections.append(
            "| Model | Optimal Size | Performance | Diversification Ratio | Effective Stocks | Recommendation Strength |"
        )
        sections.append(
            "|-------|-------------|-------------|----------------------|------------------|------------------------|"
        )

        for model_type, rec in recommendations.items():
            sections.append(
                f"| {model_type.upper()} | {rec['optimal_size']} | {rec['performance']:.4f} | {rec['diversification_ratio']:.3f} | {rec['effective_stocks']:.1f} | {rec['recommendation_strength']} |"
            )

        sections.append("")

        # Diversification analysis
        tradeoff_df = self.portfolio_size_analyzer.analyze_diversification_tradeoffs(target_metric)

        sections.append("## Diversification Trade-off Analysis")
        sections.append("")

        # Summary by portfolio size
        size_summary = (
            tradeoff_df.groupby("portfolio_size")
            .agg(
                {
                    "performance": "mean",
                    "diversification_ratio": "mean",
                    "concentration_hhi": "mean",
                    "idiosyncratic_risk": "mean",
                }
            )
            .reset_index()
        )

        sections.append(
            "| Portfolio Size | Avg Performance | Diversification Ratio | Concentration (HHI) | Idiosyncratic Risk |"
        )
        sections.append(
            "|---------------|-----------------|----------------------|---------------------|-------------------|"
        )

        for _, row in size_summary.iterrows():
            sections.append(
                f"| {row['portfolio_size']:.0f} | {row['performance']:.4f} | {row['diversification_ratio']:.3f} | {row['concentration_hhi']:.3f} | {row['idiosyncratic_risk']:.3f} |"
            )

        sections.append("")

        # Model-specific analysis
        for model_type in tradeoff_df["model_type"].unique():
            model_data = tradeoff_df[tradeoff_df["model_type"] == model_type]
            optimal_row = (
                model_data[model_data["is_optimal"]].iloc[0]
                if any(model_data["is_optimal"])
                else None
            )

            sections.append(f"### {model_type.upper()} Analysis")
            sections.append("")

            if optimal_row is not None:
                sections.append(
                    f"- **Optimal Size**: {optimal_row['portfolio_size']:.0f} positions"
                )
                sections.append(f"- **Performance**: {optimal_row['performance']:.4f}")
                sections.append(f"- **Diversification**: {optimal_row['expected_diversification']}")
                sections.append(
                    f"- **Statistical Significance**: {'Yes' if optimal_row['is_significant'] else 'No'}"
                )

            sections.append("")

        # Write report
        with open(output_file, "w") as f:
            f.write("\n".join(sections))

        logger.info(f"Portfolio size analysis report written to {output_file}")

    def _generate_constraint_analysis_report(self, output_file: Path, target_metric: str) -> None:
        """Generate constraint analysis report.

        Args:
            output_file: Path to save constraint analysis report
            target_metric: Performance metric for analysis
        """
        if not self.constraint_analyzer or not self.constraint_analyzer.constraint_analysis_results:
            return

        sections = []

        sections.append("# Constraint Violation Analysis")
        sections.append("")
        sections.append(f"**Target Metric**: {target_metric}")
        sections.append("")

        # Constraint optimization recommendations
        recommendations = self.constraint_analyzer.get_constraint_optimization_recommendations()

        sections.append("## Constraint Optimization Recommendations")
        sections.append("")
        sections.append(
            "| Model | Robustness Score | Violation Freq | Performance Impact | Most Violated Constraint | Recommendation |"
        )
        sections.append(
            "|-------|-----------------|----------------|-------------------|-------------------------|----------------|"
        )

        for model_type, rec in recommendations.items():
            sections.append(
                f"| {model_type.upper()} | {rec['optimal_robustness_score']:.3f} | {rec['avg_violation_frequency']:.3f} | {rec['avg_performance_impact']:.2f}% | {rec['most_violated_constraint']} | {rec['constraint_recommendation']} |"
            )

        sections.append("")

        # Violation analysis by constraint type
        viz_data = self.constraint_analyzer.create_constraint_analysis_visualization_data()

        if "violations" in viz_data and not viz_data["violations"].empty:
            sections.append("## Constraint Violation Analysis")
            sections.append("")

            violation_summary = (
                viz_data["violations"]
                .groupby("constraint_type")
                .agg({"violation_frequency": "mean", "violation_magnitude": "mean"})
                .reset_index()
            )

            sections.append("| Constraint Type | Avg Frequency | Avg Magnitude |")
            sections.append("|-----------------|---------------|---------------|")

            for _, row in violation_summary.iterrows():
                sections.append(
                    f"| {row['constraint_type'].replace('_', ' ').title()} | {row['violation_frequency']:.3f} | {row['violation_magnitude']:.4f} |"
                )

            sections.append("")

        # Robustness ranking
        if "robustness" in viz_data and not viz_data["robustness"].empty:
            sections.append("## Model Robustness Ranking")
            sections.append("")

            robustness_ranking = viz_data["robustness"].sort_values(
                "best_robustness", ascending=False
            )

            sections.append("| Rank | Model | Best Robustness | Avg Robustness |")
            sections.append("|------|-------|-----------------|----------------|")

            for rank, (_, row) in enumerate(robustness_ranking.iterrows(), 1):
                sections.append(
                    f"| {rank} | {row['model'].upper()} | {row['best_robustness']:.3f} | {row['avg_robustness']:.3f} |"
                )

            sections.append("")

        # Write report
        with open(output_file, "w") as f:
            f.write("\n".join(sections))

        logger.info(f"Constraint analysis report written to {output_file}")

    def _generate_robustness_summary(self, output_file: Path, target_metric: str) -> None:
        """Generate overall robustness summary.

        Args:
            output_file: Path to save robustness summary
            target_metric: Performance metric for analysis
        """
        sections = []

        sections.append("# Overall Robustness Summary")
        sections.append("")
        sections.append(f"**Target Metric**: {target_metric}")
        sections.append("")

        # Collect robustness scores from all analyzers
        robustness_scores = {}

        # Hyperparameter robustness (based on effect sizes)
        if self.hyperparameter_tester and self.hyperparameter_tester.test_results:
            for model_type, results in self.hyperparameter_tester.test_results.items():
                effect_sizes = [r.effect_size for r in results]
                # Higher effect sizes = less robust to hyperparameter changes
                hyperparam_robustness = 1.0 - min(np.mean(effect_sizes), 1.0)

                if model_type not in robustness_scores:
                    robustness_scores[model_type] = {}
                robustness_scores[model_type]["hyperparameter"] = hyperparam_robustness

        # Transaction cost robustness
        if self.transaction_cost_analyzer and self.transaction_cost_analyzer.cost_impact_results:
            cost_summary = self.transaction_cost_analyzer.get_cost_sensitivity_summary(
                target_metric
            )

            # Lower ranking stability score = more robust
            cost_robustness = 1.0 - min(cost_summary["ranking_stability_score"] / 5.0, 1.0)

            for model_type in self.transaction_cost_analyzer.cost_impact_results.keys():
                if model_type not in robustness_scores:
                    robustness_scores[model_type] = {}
                robustness_scores[model_type]["transaction_cost"] = cost_robustness

        # Portfolio size robustness
        if self.portfolio_size_analyzer and self.portfolio_size_analyzer.size_analysis_results:
            recommendations = (
                self.portfolio_size_analyzer.get_optimal_portfolio_size_recommendations()
            )

            for model_type, rec in recommendations.items():
                # Lower performance sensitivity = more robust
                sensitivity = rec.get("performance_sensitivity", 0.0)
                size_robustness = 1.0 - min(sensitivity / 0.1, 1.0)  # Normalize by 10% sensitivity

                if model_type not in robustness_scores:
                    robustness_scores[model_type] = {}
                robustness_scores[model_type]["portfolio_size"] = size_robustness

        # Constraint robustness
        if self.constraint_analyzer and self.constraint_analyzer.constraint_analysis_results:
            constraint_recommendations = (
                self.constraint_analyzer.get_constraint_optimization_recommendations()
            )

            for model_type, rec in constraint_recommendations.items():
                if model_type not in robustness_scores:
                    robustness_scores[model_type] = {}
                robustness_scores[model_type]["constraints"] = rec["optimal_robustness_score"]

        # Calculate overall robustness scores
        sections.append("## Model Robustness Rankings")
        sections.append("")
        sections.append(
            "| Model | Hyperparameter | Transaction Cost | Portfolio Size | Constraints | Overall |"
        )
        sections.append(
            "|-------|----------------|------------------|----------------|-------------|---------|"
        )

        overall_scores = []

        for model_type, scores in robustness_scores.items():
            hyperparam = scores.get("hyperparameter", 0.5)
            cost = scores.get("transaction_cost", 0.5)
            size = scores.get("portfolio_size", 0.5)
            constraint = scores.get("constraints", 0.5)

            # Calculate weighted overall score
            overall = hyperparam * 0.3 + cost * 0.25 + size * 0.25 + constraint * 0.2
            overall_scores.append((model_type, overall))

            sections.append(
                f"| {model_type.upper()} | {hyperparam:.3f} | {cost:.3f} | {size:.3f} | {constraint:.3f} | {overall:.3f} |"
            )

        sections.append("")

        # Sort by overall robustness
        overall_scores.sort(key=lambda x: x[1], reverse=True)

        sections.append("## Robustness Ranking")
        sections.append("")

        for rank, (model_type, score) in enumerate(overall_scores, 1):
            sections.append(f"{rank}. **{model_type.upper()}** - Overall Robustness: {score:.3f}")

        sections.append("")

        # Recommendations based on robustness
        sections.append("## Robustness-Based Recommendations")
        sections.append("")

        if overall_scores:
            best_model, best_score = overall_scores[0]
            worst_model, worst_score = overall_scores[-1]

            sections.append(
                f"### Most Robust Model: {best_model.upper()} (Score: {best_score:.3f})"
            )
            sections.append("- Recommended for production deployment with minimal parameter tuning")
            sections.append("- Suitable for diverse market conditions")
            sections.append("")

            if len(overall_scores) > 1:
                sections.append(
                    f"### Least Robust Model: {worst_model.upper()} (Score: {worst_score:.3f})"
                )
                sections.append("- Requires careful hyperparameter tuning")
                sections.append("- Consider ensemble approaches or additional regularization")
                sections.append("")

        # Implementation guidance
        sections.append("## Implementation Guidance")
        sections.append("")
        sections.append(
            "1. **High Robustness Models (>0.7)**: Deploy with confidence using identified optimal parameters"
        )
        sections.append(
            "2. **Medium Robustness Models (0.5-0.7)**: Implement with additional monitoring and periodic retuning"
        )
        sections.append(
            "3. **Low Robustness Models (<0.5)**: Consider alternative approaches or significant model improvements"
        )

        # Write report
        with open(output_file, "w") as f:
            f.write("\n".join(sections))

        logger.info(f"Robustness summary written to {output_file}")

    def _export_visualization_data(self, output_dir: Path, target_metric: str) -> dict[str, str]:
        """Export visualization data files.

        Args:
            output_dir: Directory to save visualization data
            target_metric: Performance metric for visualizations

        Returns:
            Dictionary mapping data types to file paths
        """
        viz_files = {}

        # Hyperparameter visualization data
        if self.hyperparameter_tester and self.hyperparameter_tester.test_results:
            for model_type in self.hyperparameter_tester.test_results.keys():
                importance_df = self.hyperparameter_tester.compare_hyperparameter_importance(
                    model_type, target_metric
                )
                hyperparam_file = output_dir / f"{model_type}_hyperparameter_importance.csv"
                importance_df.to_csv(hyperparam_file, index=False)
                viz_files[f"{model_type}_hyperparameter"] = str(hyperparam_file)

        # Transaction cost visualization data
        if self.transaction_cost_analyzer and self.transaction_cost_analyzer.cost_impact_results:
            cost_viz_data = self.transaction_cost_analyzer.create_cost_impact_visualization_data(
                target_metric
            )

            for data_type, df in cost_viz_data.items():
                cost_file = output_dir / f"transaction_cost_{data_type}.csv"
                df.to_csv(cost_file, index=False)
                viz_files[f"cost_{data_type}"] = str(cost_file)

        # Portfolio size visualization data
        if self.portfolio_size_analyzer and self.portfolio_size_analyzer.size_analysis_results:
            size_viz_data = self.portfolio_size_analyzer.create_portfolio_size_visualization_data(
                target_metric
            )

            for data_type, df in size_viz_data.items():
                size_file = output_dir / f"portfolio_size_{data_type}.csv"
                df.to_csv(size_file, index=False)
                viz_files[f"size_{data_type}"] = str(size_file)

        # Constraint visualization data
        if self.constraint_analyzer and self.constraint_analyzer.constraint_analysis_results:
            constraint_viz_data = (
                self.constraint_analyzer.create_constraint_analysis_visualization_data()
            )

            for data_type, df in constraint_viz_data.items():
                constraint_file = output_dir / f"constraint_{data_type}.csv"
                df.to_csv(constraint_file, index=False)
                viz_files[f"constraint_{data_type}"] = str(constraint_file)

        return viz_files

    def _export_raw_data(self, output_dir: Path) -> dict[str, str]:
        """Export raw analysis data files.

        Args:
            output_dir: Directory to save raw data

        Returns:
            Dictionary mapping data types to file paths
        """
        data_files = {}

        # Export hyperparameter results
        if self.hyperparameter_tester and self.hyperparameter_tester.test_results:
            hyperparam_file = output_dir / "hyperparameter_results.csv"

            # Flatten hyperparameter results
            hyperparam_data = []
            for model_type, results in self.hyperparameter_tester.test_results.items():
                for result in results:
                    row = {
                        "model_type": model_type,
                        "parameter_name": result.parameter_name,
                        "effect_size": result.effect_size,
                        "optimal_value": result.optimal_value,
                        "p_value": result.statistical_significance.get("p_value", 1.0),
                        "is_significant": result.statistical_significance.get("p_value", 1.0)
                        < 0.05,
                    }
                    hyperparam_data.append(row)

            pd.DataFrame(hyperparam_data).to_csv(hyperparam_file, index=False)
            data_files["hyperparameter_results"] = str(hyperparam_file)

        # Export transaction cost results
        if self.transaction_cost_analyzer and self.transaction_cost_analyzer.cost_impact_results:
            cost_file = output_dir / "transaction_cost_results.csv"
            self.transaction_cost_analyzer.export_cost_analysis_results(str(cost_file))
            data_files["transaction_cost_results"] = str(cost_file)

        # Export portfolio size results
        if self.portfolio_size_analyzer and self.portfolio_size_analyzer.size_analysis_results:
            size_file = output_dir / "portfolio_size_results.csv"
            self.portfolio_size_analyzer.export_portfolio_size_results(str(size_file))
            data_files["portfolio_size_results"] = str(size_file)

        # Export constraint results
        if self.constraint_analyzer and self.constraint_analyzer.constraint_analysis_results:
            constraint_file = output_dir / "constraint_results.csv"
            self.constraint_analyzer.export_constraint_analysis_results(str(constraint_file))
            data_files["constraint_results"] = str(constraint_file)

        return data_files
