"""Executive Summary Report Generation Framework.

This module provides executive summary report generation with approach rankings,
investment recommendations, and comprehensive performance analysis for portfolio managers.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceRanking:
    """Individual strategy performance ranking."""
    strategy_name: str
    rank: int
    sharpe_ratio: float
    information_ratio: float
    max_drawdown: float
    annual_turnover: float
    risk_adjusted_score: float
    meets_criteria: bool

@dataclass
class InvestmentRecommendation:
    """Investment recommendation for a strategy."""
    strategy_name: str
    recommendation: str  # "Deploy", "Monitor", "Reject"
    confidence_level: float
    key_strengths: list[str]
    key_risks: list[str]
    rationale: str

@dataclass
class ExecutiveSummary:
    """Complete executive summary data structure."""
    timestamp: datetime
    total_strategies_evaluated: int
    recommended_strategies: int
    performance_rankings: list[PerformanceRanking]
    investment_recommendations: list[InvestmentRecommendation]
    key_findings: list[str]
    production_readiness: str

class ExecutiveSummaryGenerator:
    """Generator for executive summary reports and investment recommendations."""

    def __init__(self, base_path: Optional[str] = None):
        """Initialise executive summary generator.

        Args:
            base_path: Base path for data files (defaults to current directory)
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()

        # Investment criteria thresholds
        self.criteria = {
            'min_sharpe_improvement': 0.2,  # ≥0.2 Sharpe improvement
            'max_annual_turnover': 2.4,     # ≤20% monthly = 240% annual
            'max_drawdown_threshold': -0.4,  # Maximum acceptable drawdown -40%
            'min_information_ratio': 0.1,   # Minimum information ratio
            'confidence_threshold': 0.7      # Minimum confidence for deployment
        }

        # Baseline performance (using equal weight as baseline)
        self.baseline_sharpe = 0.5  # Assumed baseline Sharpe ratio

    def load_performance_data(self) -> dict[str, Any]:
        """Load performance analytics data.

        Returns:
            Dictionary containing performance data
        """
        results_file = self.base_path / 'results' / 'performance_analytics' / 'performance_analytics_results.json'

        if not results_file.exists():
            logger.error(f"Performance results file not found: {results_file}")
            raise FileNotFoundError(f"Performance results file not found: {results_file}")

        try:
            with open(results_file) as f:
                data = json.load(f)
            logger.info(f"Loaded performance data from {results_file}")
            return data

        except Exception as e:
            logger.error(f"Error loading performance data: {e}")
            raise

    def calculate_risk_adjusted_score(self, sharpe: float, info_ratio: float,
                                    max_drawdown: float, turnover: float) -> float:
        """Calculate composite risk-adjusted performance score.

        Args:
            sharpe: Sharpe ratio
            info_ratio: Information ratio
            max_drawdown: Maximum drawdown (negative value)
            turnover: Annual turnover

        Returns:
            Risk-adjusted performance score (0-100)
        """
        try:
            # Normalise components (0-1 scale)
            sharpe_score = min(max(sharpe / 3.0, 0), 1)  # Cap at 3.0 Sharpe
            ir_score = min(max(info_ratio / 1.0, 0), 1)  # Cap at 1.0 IR
            dd_score = min(max((max_drawdown + 0.5) / 0.5, 0), 1)  # -50% to 0% drawdown
            turnover_score = min(max((5.0 - turnover) / 5.0, 0), 1)  # Lower turnover is better

            # Weighted composite score
            composite_score = (
                0.4 * sharpe_score +      # 40% weight on Sharpe ratio
                0.25 * ir_score +         # 25% weight on Information ratio
                0.25 * dd_score +         # 25% weight on drawdown control
                0.1 * turnover_score      # 10% weight on turnover efficiency
            )

            return min(max(composite_score * 100, 0), 100)  # Scale to 0-100

        except Exception as e:
            logger.warning(f"Error calculating risk-adjusted score: {e}")
            return 0.0

    def generate_approach_rankings(self, performance_data: dict[str, Any]) -> list[PerformanceRanking]:
        """Generate approach rankings matrix based on risk-adjusted performance.

        Args:
            performance_data: Performance analytics data

        Returns:
            List of performance rankings sorted by score
        """
        logger.info("Generating approach rankings matrix...")

        performance_metrics = performance_data.get('performance_metrics', {})
        rankings = []

        # Extract baseline performance
        baseline_metrics = performance_metrics.get('EqualWeight', {})
        baseline_sharpe = baseline_metrics.get('Sharpe', self.baseline_sharpe)

        for strategy_name, metrics in performance_metrics.items():
            try:
                # Extract key metrics
                sharpe = metrics.get('Sharpe', 0.0)
                info_ratio = metrics.get('information_ratio', 0.0)
                max_drawdown = metrics.get('MDD', 0.0)
                annual_turnover = metrics.get('annual_turnover', 0.0)

                # Calculate risk-adjusted score
                risk_score = self.calculate_risk_adjusted_score(
                    sharpe, info_ratio, max_drawdown, annual_turnover
                )

                # Check if meets investment criteria
                sharpe_improvement = sharpe - baseline_sharpe
                meets_criteria = (
                    sharpe_improvement >= self.criteria['min_sharpe_improvement'] and
                    annual_turnover <= self.criteria['max_annual_turnover'] and
                    max_drawdown >= self.criteria['max_drawdown_threshold'] and
                    info_ratio >= self.criteria['min_information_ratio']
                )

                ranking = PerformanceRanking(
                    strategy_name=strategy_name,
                    rank=0,  # Will be set after sorting
                    sharpe_ratio=sharpe,
                    information_ratio=info_ratio,
                    max_drawdown=max_drawdown,
                    annual_turnover=annual_turnover,
                    risk_adjusted_score=risk_score,
                    meets_criteria=meets_criteria
                )

                rankings.append(ranking)

            except Exception as e:
                logger.warning(f"Error processing strategy {strategy_name}: {e}")
                continue

        # Sort by risk-adjusted score (descending)
        rankings.sort(key=lambda x: x.risk_adjusted_score, reverse=True)

        # Assign ranks
        for i, ranking in enumerate(rankings):
            ranking.rank = i + 1

        logger.info(f"Generated rankings for {len(rankings)} strategies")
        return rankings

    def generate_investment_recommendations(self, rankings: list[PerformanceRanking]) -> list[InvestmentRecommendation]:
        """Generate investment recommendations with clear selection criteria.

        Args:
            rankings: List of performance rankings

        Returns:
            List of investment recommendations
        """
        logger.info("Generating investment recommendations...")

        recommendations = []

        for ranking in rankings:
            try:
                # Determine recommendation category
                if ranking.meets_criteria and ranking.risk_adjusted_score >= 70:
                    recommendation = "Deploy"
                    confidence = min(0.7 + (ranking.risk_adjusted_score - 70) / 100, 0.95)
                elif ranking.meets_criteria or ranking.risk_adjusted_score >= 50:
                    recommendation = "Monitor"
                    confidence = min(0.5 + (ranking.risk_adjusted_score - 50) / 100, 0.8)
                else:
                    recommendation = "Reject"
                    confidence = 0.3 + (50 - ranking.risk_adjusted_score) / 100

                # Generate key strengths
                strengths = []
                if ranking.sharpe_ratio > 1.0:
                    strengths.append(f"Strong risk-adjusted returns (Sharpe: {ranking.sharpe_ratio:.2f})")
                if ranking.information_ratio > 0.2:
                    strengths.append(f"Consistent alpha generation (IR: {ranking.information_ratio:.2f})")
                if ranking.max_drawdown > -0.3:
                    strengths.append(f"Good downside protection (MDD: {ranking.max_drawdown:.1%})")
                if ranking.annual_turnover < 2.0:
                    strengths.append(f"Low transaction costs (Turnover: {ranking.annual_turnover:.1f}x)")

                # Generate key risks
                risks = []
                if ranking.sharpe_ratio < 0.8:
                    risks.append(f"Below-average risk-adjusted returns (Sharpe: {ranking.sharpe_ratio:.2f})")
                if ranking.max_drawdown < -0.4:
                    risks.append(f"High maximum drawdown ({ranking.max_drawdown:.1%})")
                if ranking.annual_turnover > 3.0:
                    risks.append(f"High transaction costs (Turnover: {ranking.annual_turnover:.1f}x)")
                if ranking.information_ratio < 0.1:
                    risks.append(f"Limited alpha generation (IR: {ranking.information_ratio:.2f})")

                # Generate rationale
                if recommendation == "Deploy":
                    rationale = f"Strong performer meeting all investment criteria with risk-adjusted score of {ranking.risk_adjusted_score:.0f}/100"
                elif recommendation == "Monitor":
                    rationale = f"Promising approach requiring further evaluation with risk-adjusted score of {ranking.risk_adjusted_score:.0f}/100"
                else:
                    rationale = f"Underperforms investment criteria with risk-adjusted score of {ranking.risk_adjusted_score:.0f}/100"

                rec = InvestmentRecommendation(
                    strategy_name=ranking.strategy_name,
                    recommendation=recommendation,
                    confidence_level=confidence,
                    key_strengths=strengths[:3],  # Top 3 strengths
                    key_risks=risks[:3],  # Top 3 risks
                    rationale=rationale
                )

                recommendations.append(rec)

            except Exception as e:
                logger.warning(f"Error generating recommendation for {ranking.strategy_name}: {e}")
                continue

        logger.info(f"Generated recommendations for {len(recommendations)} strategies")
        return recommendations

    def generate_key_findings(self, rankings: list[PerformanceRanking],
                            recommendations: list[InvestmentRecommendation]) -> list[str]:
        """Generate key findings for executive summary.

        Args:
            rankings: Performance rankings
            recommendations: Investment recommendations

        Returns:
            List of key findings
        """
        findings = []

        try:
            # Strategy count analysis
            total_strategies = len(rankings)
            deploy_count = sum(1 for rec in recommendations if rec.recommendation == "Deploy")
            monitor_count = sum(1 for rec in recommendations if rec.recommendation == "Monitor")

            findings.append(f"Evaluated {total_strategies} portfolio strategies with {deploy_count} recommended for deployment and {monitor_count} for monitoring")

            # Top performer analysis
            if rankings:
                top_strategy = rankings[0]
                findings.append(f"Top performer: {top_strategy.strategy_name} with Sharpe ratio of {top_strategy.sharpe_ratio:.2f} and risk score of {top_strategy.risk_adjusted_score:.0f}/100")

            # Performance criteria analysis
            criteria_met = sum(1 for ranking in rankings if ranking.meets_criteria)
            findings.append(f"{criteria_met}/{total_strategies} strategies meet all investment criteria (≥0.2 Sharpe improvement, ≤20% turnover)")

            # Risk-return trade-off analysis
            high_sharpe_strategies = [r for r in rankings if r.sharpe_ratio > 1.2]
            if high_sharpe_strategies:
                findings.append(f"{len(high_sharpe_strategies)} strategies demonstrate superior risk-adjusted returns (Sharpe > 1.2)")

            # Turnover efficiency analysis
            low_turnover = [r for r in rankings if r.annual_turnover < 2.0]
            findings.append(f"{len(low_turnover)} strategies achieve low transaction costs with annual turnover below 200%")

            # Drawdown control analysis
            good_drawdown = [r for r in rankings if r.max_drawdown > -0.3]
            findings.append(f"{len(good_drawdown)} strategies demonstrate strong downside protection with maximum drawdown below 30%")

        except Exception as e:
            logger.warning(f"Error generating key findings: {e}")
            findings.append("Analysis completed with some limitations due to data processing constraints")

        return findings

    def generate_executive_summary(self) -> ExecutiveSummary:
        """Generate comprehensive executive summary.

        Returns:
            Complete executive summary
        """
        logger.info("Generating comprehensive executive summary...")

        try:
            # Load performance data
            performance_data = self.load_performance_data()

            # Generate rankings
            rankings = self.generate_approach_rankings(performance_data)

            # Generate recommendations
            recommendations = self.generate_investment_recommendations(rankings)

            # Generate key findings
            key_findings = self.generate_key_findings(rankings, recommendations)

            # Determine production readiness
            deploy_count = sum(1 for rec in recommendations if rec.recommendation == "Deploy")
            high_confidence = sum(1 for rec in recommendations
                                if rec.recommendation == "Deploy" and rec.confidence_level >= 0.8)

            if high_confidence >= 2:
                production_readiness = "Ready - Multiple high-confidence strategies available for deployment"
            elif deploy_count >= 1:
                production_readiness = "Ready - At least one strategy recommended for deployment"
            elif sum(1 for rec in recommendations if rec.recommendation == "Monitor") >= 2:
                production_readiness = "Conditional - Promising strategies require additional validation"
            else:
                production_readiness = "Not Ready - No strategies meet deployment criteria"

            summary = ExecutiveSummary(
                timestamp=datetime.now(),
                total_strategies_evaluated=len(rankings),
                recommended_strategies=deploy_count,
                performance_rankings=rankings,
                investment_recommendations=recommendations,
                key_findings=key_findings,
                production_readiness=production_readiness
            )

            logger.info(f"Executive summary generated: {deploy_count} strategies recommended for deployment")
            return summary

        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            raise

    def create_executive_dashboard_data(self, summary: ExecutiveSummary) -> dict[str, Any]:
        """Create executive dashboard data for visualisation.

        Args:
            summary: Executive summary

        Returns:
            Dictionary containing dashboard data
        """
        dashboard_data = {
            'summary_metrics': {
                'total_strategies': summary.total_strategies_evaluated,
                'recommended_strategies': summary.recommended_strategies,
                'recommendation_rate': summary.recommended_strategies / summary.total_strategies_evaluated if summary.total_strategies_evaluated > 0 else 0,
                'production_readiness': summary.production_readiness
            },
            'performance_rankings': {
                'top_5_strategies': [
                    {
                        'name': ranking.strategy_name,
                        'rank': ranking.rank,
                        'sharpe_ratio': ranking.sharpe_ratio,
                        'risk_score': ranking.risk_adjusted_score,
                        'meets_criteria': ranking.meets_criteria
                    }
                    for ranking in summary.performance_rankings[:5]
                ]
            },
            'recommendation_breakdown': {
                'deploy': sum(1 for rec in summary.investment_recommendations if rec.recommendation == "Deploy"),
                'monitor': sum(1 for rec in summary.investment_recommendations if rec.recommendation == "Monitor"),
                'reject': sum(1 for rec in summary.investment_recommendations if rec.recommendation == "Reject")
            },
            'risk_return_profile': [
                {
                    'strategy': ranking.strategy_name,
                    'sharpe_ratio': ranking.sharpe_ratio,
                    'max_drawdown': ranking.max_drawdown,
                    'annual_turnover': ranking.annual_turnover,
                    'recommendation': next(
                        (rec.recommendation for rec in summary.investment_recommendations
                         if rec.strategy_name == ranking.strategy_name),
                        "Unknown"
                    )
                }
                for ranking in summary.performance_rankings
            ]
        }

        return dashboard_data

    def export_executive_summary(self, summary: ExecutiveSummary,
                                output_dir: Optional[str] = None) -> dict[str, str]:
        """Export executive summary to various formats.

        Args:
            summary: Executive summary to export
            output_dir: Output directory (defaults to results/executive_summary)

        Returns:
            Dictionary mapping format to file path
        """
        if output_dir is None:
            output_dir = self.base_path / 'results' / 'executive_summary'
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        exported_files = {}

        try:
            # Export as JSON
            json_file = output_dir / 'executive_summary.json'
            with open(json_file, 'w') as f:
                json.dump(asdict(summary), f, indent=2, default=str)
            exported_files['json'] = str(json_file)

            # Export rankings as CSV
            rankings_df = pd.DataFrame([asdict(ranking) for ranking in summary.performance_rankings])
            csv_file = output_dir / 'performance_rankings.csv'
            rankings_df.to_csv(csv_file, index=False)
            exported_files['rankings_csv'] = str(csv_file)

            # Export recommendations as CSV
            recommendations_df = pd.DataFrame([asdict(rec) for rec in summary.investment_recommendations])
            rec_csv_file = output_dir / 'investment_recommendations.csv'
            recommendations_df.to_csv(rec_csv_file, index=False)
            exported_files['recommendations_csv'] = str(rec_csv_file)

            # Export dashboard data
            dashboard_data = self.create_executive_dashboard_data(summary)
            dashboard_file = output_dir / 'executive_dashboard_data.json'
            with open(dashboard_file, 'w') as f:
                json.dump(dashboard_data, f, indent=2, default=str)
            exported_files['dashboard_json'] = str(dashboard_file)

            logger.info(f"Executive summary exported to {len(exported_files)} files in {output_dir}")

        except Exception as e:
            logger.error(f"Error exporting executive summary: {e}")
            raise

        return exported_files
