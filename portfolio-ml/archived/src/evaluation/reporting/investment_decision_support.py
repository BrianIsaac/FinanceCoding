"""
Investment Decision Support Framework.

This module provides comprehensive investment decision support by translating
research findings into actionable recommendations with risk assessments,
implementation guidance, and monitoring protocols.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.reporting.comparative_analysis import ComparativeAnalysisEngine
from src.evaluation.validation.project_goals_validator import ProjectGoalsValidator
from src.evaluation.validation.robustness_testing import RobustnessTestingFramework


@dataclass
class InvestmentRecommendation:
    """Investment recommendation with supporting evidence."""
    strategy_name: str
    allocation_percentage: float
    confidence_level: float
    expected_return: float
    expected_volatility: float
    max_drawdown_estimate: float
    implementation_priority: str
    supporting_evidence: list[str]
    risk_considerations: list[str]
    monitoring_metrics: list[str]


@dataclass
class RiskAssessment:
    """Comprehensive risk assessment for investment strategies."""
    market_risk_score: float
    operational_risk_score: float
    model_risk_score: float
    liquidity_risk_score: float
    overall_risk_rating: str
    risk_mitigation_measures: list[str]
    stress_test_results: dict[str, float]


@dataclass
class ImplementationGuidance:
    """Implementation guidance for investment strategies."""
    deployment_timeline: str
    resource_requirements: list[str]
    technical_prerequisites: list[str]
    regulatory_considerations: list[str]
    monitoring_frequency: str
    rebalancing_schedule: str
    performance_review_cycle: str


@dataclass
class InvestmentDecisionReport:
    """Comprehensive investment decision support report."""
    primary_recommendation: InvestmentRecommendation
    alternative_recommendations: list[InvestmentRecommendation]
    risk_assessment: RiskAssessment
    implementation_guidance: ImplementationGuidance
    research_summary: dict[str, Any]
    confidence_metrics: dict[str, float]
    generated_date: str


class InvestmentDecisionSupportEngine:
    """
    Engine for generating investment decision support reports.

    Translates research findings into actionable investment recommendations
    with comprehensive risk assessments and implementation guidance.
    """

    def __init__(self, results_dir: str = "results"):
        """
        Initialise investment decision support engine.

        Args:
            results_dir: Directory containing analysis results
        """
        self.results_dir = Path(results_dir)
        self.logger = logging.getLogger(__name__)

        # Initialise component engines
        self.comparative_engine = ComparativeAnalysisEngine(results_dir)
        self.goals_validator = ProjectGoalsValidator(results_dir)
        self.robustness_tester = RobustnessTestingFramework(results_dir)

        # Risk assessment parameters
        self.risk_thresholds = {
            'low_risk': 0.3,
            'medium_risk': 0.6,
            'high_risk': 0.8
        }

        # Performance expectations
        self.performance_expectations = {
            'min_sharpe_ratio': 1.0,
            'max_drawdown_tolerance': -0.25,
            'min_information_ratio': 0.1,
            'min_calmar_ratio': 0.5
        }

    def generate_investment_recommendations(self) -> list[InvestmentRecommendation]:
        """
        Generate investment recommendations based on research findings.

        Returns:
            List of investment recommendations ordered by priority
        """
        # Load comparative analysis results
        comparative_results = self._load_comparative_results()
        strategy_rankings = self._load_strategy_rankings()

        recommendations = []

        # Generate primary recommendation
        primary_strategy = comparative_results.get('primary_recommendation')
        if primary_strategy:
            primary_rec = self._create_recommendation(
                primary_strategy,
                strategy_rankings,
                is_primary=True
            )
            recommendations.append(primary_rec)

        # Generate alternative recommendations
        top_strategies = strategy_rankings.head(3)
        for _, strategy_data in top_strategies.iterrows():
            if strategy_data['strategy_name'] != primary_strategy:
                alt_rec = self._create_recommendation(
                    strategy_data['strategy_name'],
                    strategy_rankings,
                    is_primary=False
                )
                recommendations.append(alt_rec)

        return recommendations

    def _create_recommendation(
        self,
        strategy_name: str,
        rankings_df: pd.DataFrame,
        is_primary: bool = False
    ) -> InvestmentRecommendation:
        """Create investment recommendation for a strategy."""
        strategy_data = rankings_df[
            rankings_df['strategy_name'] == strategy_name
        ].iloc[0]

        # Determine allocation based on confidence and performance
        if is_primary:
            allocation = min(100.0, max(60.0, strategy_data['overall_score'] * 100))
        else:
            allocation = min(30.0, max(10.0, strategy_data['overall_score'] * 50))

        # Calculate confidence based on multiple factors
        confidence = self._calculate_recommendation_confidence(strategy_data)

        # Generate supporting evidence
        evidence = self._generate_supporting_evidence(strategy_data)

        # Assess risks
        risks = self._assess_strategy_risks(strategy_data)

        # Define monitoring metrics
        monitoring = self._define_monitoring_metrics(strategy_name)

        return InvestmentRecommendation(
            strategy_name=strategy_name,
            allocation_percentage=allocation,
            confidence_level=confidence,
            expected_return=strategy_data.get('sharpe_ratio', 0.0) * 0.15,  # Approximate
            expected_volatility=0.15,  # Standard market volatility
            max_drawdown_estimate=strategy_data.get('max_drawdown', -0.2),
            implementation_priority=strategy_data.get('recommendation', 'Monitor'),
            supporting_evidence=evidence,
            risk_considerations=risks,
            monitoring_metrics=monitoring
        )

    def _calculate_recommendation_confidence(self, strategy_data: pd.Series) -> float:
        """Calculate confidence level for recommendation."""
        # Base confidence from overall score
        base_confidence = strategy_data.get('overall_score', 0.0)

        # Adjust for statistical significance
        sig_count = strategy_data.get('statistical_significance_count', 0)
        sig_boost = min(0.2, sig_count * 0.01)

        # Adjust for dominance score
        dominance = strategy_data.get('dominance_score', 0.0)
        dominance_boost = dominance * 0.1

        # Penalty for high drawdown
        drawdown = abs(strategy_data.get('max_drawdown', 0.0))
        drawdown_penalty = max(0, (drawdown - 0.2) * 0.5)

        confidence = base_confidence + sig_boost + dominance_boost - drawdown_penalty
        return max(0.0, min(1.0, confidence))

    def _generate_supporting_evidence(self, strategy_data: pd.Series) -> list[str]:
        """Generate supporting evidence for recommendation."""
        evidence = []

        # Performance evidence
        sharpe = strategy_data.get('sharpe_ratio', 0.0)
        if sharpe > 1.0:
            evidence.append(f"Strong risk-adjusted returns: Sharpe {sharpe:.3f}")

        # Statistical evidence
        sig_count = strategy_data.get('statistical_significance_count', 0)
        if sig_count > 10:
            evidence.append(f"Multiple significant outperformance: {sig_count} metrics")

        # Risk evidence
        drawdown = strategy_data.get('max_drawdown', 0.0)
        if drawdown > -0.3:
            evidence.append(f"Controlled downside risk: {drawdown:.1%} max drawdown")

        # Consistency evidence
        dominance = strategy_data.get('dominance_score', 0.0)
        if dominance > 0.7:
            evidence.append(f"Statistical dominance: {dominance:.1%} win rate")

        # Transaction cost evidence
        turnover = strategy_data.get('annual_turnover', 0.0)
        if turnover < 2.0:
            evidence.append(f"Low transaction costs: {turnover:.1f}x annual turnover")

        return evidence

    def _assess_strategy_risks(self, strategy_data: pd.Series) -> list[str]:
        """Assess risks for strategy."""
        risks = []

        # Drawdown risk
        drawdown = abs(strategy_data.get('max_drawdown', 0.0))
        if drawdown > 0.4:
            risks.append(f"High maximum drawdown risk: {drawdown:.1%}")

        # Volatility risk (estimated)
        sharpe = strategy_data.get('sharpe_ratio', 0.0)
        if sharpe < 0.8:
            risks.append("Moderate risk-adjusted performance")

        # Model complexity risk
        strategy_name = strategy_data.get('strategy_name', '')
        if 'GAT' in strategy_name or 'LSTM' in strategy_name:
            risks.append("Model complexity and overfitting risk")

        # Market regime risk
        if strategy_data.get('overall_score', 0.0) < 0.3:
            risks.append("Sensitivity to market regime changes")

        return risks

    def _define_monitoring_metrics(self, strategy_name: str) -> list[str]:
        """Define monitoring metrics for strategy."""
        base_metrics = [
            "Monthly Sharpe ratio",
            "Rolling 6-month returns",
            "Maximum drawdown tracking",
            "Portfolio turnover monitoring"
        ]

        # Add strategy-specific metrics
        if 'HRP' in strategy_name:
            base_metrics.extend([
                "Correlation matrix stability",
                "Clustering consistency"
            ])
        elif 'GAT' in strategy_name or 'LSTM' in strategy_name:
            base_metrics.extend([
                "Model prediction accuracy",
                "Feature importance drift",
                "Training data staleness"
            ])

        return base_metrics

    def assess_comprehensive_risk(self, recommendations: list[InvestmentRecommendation]) -> RiskAssessment:
        """Assess comprehensive risk across all recommendations."""
        # Load robustness test results
        robustness_results = self._load_robustness_results()

        # Calculate risk scores
        market_risk = self._calculate_market_risk(recommendations)
        operational_risk = self._calculate_operational_risk(recommendations)
        model_risk = self._calculate_model_risk(recommendations)
        liquidity_risk = self._calculate_liquidity_risk(recommendations)

        # Overall risk rating
        overall_score = np.mean([market_risk, operational_risk, model_risk, liquidity_risk])
        if overall_score < self.risk_thresholds['low_risk']:
            risk_rating = "Low"
        elif overall_score < self.risk_thresholds['medium_risk']:
            risk_rating = "Medium"
        elif overall_score < self.risk_thresholds['high_risk']:
            risk_rating = "High"
        else:
            risk_rating = "Very High"

        # Risk mitigation measures
        mitigation_measures = self._generate_risk_mitigation_measures(
            market_risk, operational_risk, model_risk, liquidity_risk
        )

        # Stress test results
        stress_results = self._extract_stress_test_results(robustness_results)

        return RiskAssessment(
            market_risk_score=market_risk,
            operational_risk_score=operational_risk,
            model_risk_score=model_risk,
            liquidity_risk_score=liquidity_risk,
            overall_risk_rating=risk_rating,
            risk_mitigation_measures=mitigation_measures,
            stress_test_results=stress_results
        )

    def _calculate_market_risk(self, recommendations: list[InvestmentRecommendation]) -> float:
        """Calculate market risk score."""
        if not recommendations:
            return 0.5

        # Average maximum drawdown across recommendations
        avg_drawdown = np.mean([
            abs(rec.max_drawdown_estimate) for rec in recommendations
        ])

        # Risk score based on drawdown (higher drawdown = higher risk)
        return min(1.0, avg_drawdown * 2.5)

    def _calculate_operational_risk(self, recommendations: list[InvestmentRecommendation]) -> float:
        """Calculate operational risk score."""
        # Base operational risk
        base_risk = 0.2

        # Increase risk for complex strategies
        complex_strategies = sum(1 for rec in recommendations
                               if 'GAT' in rec.strategy_name or 'LSTM' in rec.strategy_name)
        complexity_risk = complex_strategies * 0.15

        return min(1.0, base_risk + complexity_risk)

    def _calculate_model_risk(self, recommendations: list[InvestmentRecommendation]) -> float:
        """Calculate model risk score."""
        if not recommendations:
            return 0.5

        # Higher risk for ML-based strategies
        ml_allocation = sum(rec.allocation_percentage for rec in recommendations
                          if 'GAT' in rec.strategy_name or 'LSTM' in rec.strategy_name)

        # Risk increases with ML allocation
        return min(1.0, ml_allocation / 100.0 * 0.8)

    def _calculate_liquidity_risk(self, recommendations: list[InvestmentRecommendation]) -> float:
        """Calculate liquidity risk score."""
        # Base liquidity risk (assuming liquid equity markets)
        return 0.1

    def _generate_risk_mitigation_measures(
        self,
        market_risk: float,
        operational_risk: float,
        model_risk: float,
        liquidity_risk: float
    ) -> list[str]:
        """Generate risk mitigation measures."""
        measures = []

        if market_risk > 0.5:
            measures.append("Implement dynamic position sizing based on volatility")
            measures.append("Consider portfolio hedging during high volatility periods")

        if operational_risk > 0.4:
            measures.append("Establish robust model monitoring and alerting systems")
            measures.append("Implement automated failover to simpler strategies")

        if model_risk > 0.5:
            measures.append("Regular model retraining and validation")
            measures.append("Maintain ensemble of models for robustness")

        if liquidity_risk > 0.3:
            measures.append("Monitor daily trading volumes and market impact")

        # Always include these
        measures.extend([
            "Diversify across multiple strategy types",
            "Implement gradual deployment with performance monitoring",
            "Maintain cash reserves for rebalancing flexibility"
        ])

        return measures

    def _extract_stress_test_results(self, robustness_results: dict) -> dict[str, float]:
        """Extract stress test results from robustness testing."""
        if not robustness_results:
            return {}

        return {
            "High transaction costs": robustness_results.get('transaction_cost_sensitivity', {}).get('high_cost_performance', 0.0),
            "Parameter uncertainty": robustness_results.get('parameter_robustness', {}).get('robustness_score', 0.0),
            "Market stress": robustness_results.get('market_regime_stress', {}).get('stress_score', 0.0),
            "Overall robustness": robustness_results.get('overall_robustness_score', 0.0)
        }

    def generate_implementation_guidance(self, recommendations: list[InvestmentRecommendation]) -> ImplementationGuidance:
        """Generate implementation guidance."""
        # Determine deployment timeline based on complexity
        has_ml_strategies = any('GAT' in rec.strategy_name or 'LSTM' in rec.strategy_name
                              for rec in recommendations)

        timeline = "6-8 weeks" if has_ml_strategies else "3-4 weeks"

        # Resource requirements
        resources = [
            "Quantitative analyst for strategy implementation",
            "Risk management oversight",
            "Technology infrastructure for backtesting",
            "Data feed subscriptions"
        ]

        if has_ml_strategies:
            resources.extend([
                "Machine learning engineering support",
                "GPU computing resources for model training",
                "MLOps infrastructure for model deployment"
            ])

        # Technical prerequisites
        technical = [
            "Historical market data (minimum 5 years)",
            "Portfolio management system integration",
            "Risk monitoring dashboard",
            "Performance reporting capabilities"
        ]

        # Regulatory considerations
        regulatory = [
            "Investment committee approval",
            "Risk committee review",
            "Compliance documentation",
            "Audit trail requirements"
        ]

        return ImplementationGuidance(
            deployment_timeline=timeline,
            resource_requirements=resources,
            technical_prerequisites=technical,
            regulatory_considerations=regulatory,
            monitoring_frequency="Daily",
            rebalancing_schedule="Monthly",
            performance_review_cycle="Quarterly"
        )

    def generate_comprehensive_report(self) -> InvestmentDecisionReport:
        """Generate comprehensive investment decision support report."""
        # Generate recommendations
        recommendations = self.generate_investment_recommendations()

        if not recommendations:
            raise ValueError("No recommendations could be generated")

        primary_rec = recommendations[0]
        alternative_recs = recommendations[1:] if len(recommendations) > 1 else []

        # Assess risks
        risk_assessment = self.assess_comprehensive_risk(recommendations)

        # Generate implementation guidance
        implementation = self.generate_implementation_guidance(recommendations)

        # Compile research summary
        research_summary = self._compile_research_summary()

        # Calculate confidence metrics
        confidence_metrics = self._calculate_confidence_metrics(recommendations)

        return InvestmentDecisionReport(
            primary_recommendation=primary_rec,
            alternative_recommendations=alternative_recs,
            risk_assessment=risk_assessment,
            implementation_guidance=implementation,
            research_summary=research_summary,
            confidence_metrics=confidence_metrics,
            generated_date=datetime.now().isoformat()
        )

    def _compile_research_summary(self) -> dict[str, Any]:
        """Compile research summary from all analysis components."""
        try:
            # Load project goals validation
            goals_results = self._load_goals_validation_results()

            # Load comparative analysis
            comparative_results = self._load_comparative_results()

            # Load robustness results
            robustness_results = self._load_robustness_results()

            return {
                "project_goals_achievement": goals_results.get('overall_success_rate', 0.0),
                "optimal_strategy_identified": comparative_results.get('primary_recommendation', 'Unknown'),
                "strategy_confidence": comparative_results.get('confidence_level', 0.0),
                "robustness_score": robustness_results.get('overall_robustness_score', 0.0),
                "statistical_significance": comparative_results.get('statistical_significance_rate', 0.0),
                "number_of_strategies_tested": len(self._load_strategy_rankings())
            }
        except Exception as e:
            self.logger.warning(f"Could not compile complete research summary: {e}")
            return {"error": "Research summary compilation failed"}

    def _calculate_confidence_metrics(self, recommendations: list[InvestmentRecommendation]) -> dict[str, float]:
        """Calculate overall confidence metrics."""
        if not recommendations:
            return {}

        # Average confidence across recommendations
        avg_confidence = np.mean([rec.confidence_level for rec in recommendations])

        # Evidence strength (number of supporting evidence points)
        evidence_strength = np.mean([len(rec.supporting_evidence) for rec in recommendations]) / 5.0

        # Risk-adjusted confidence
        risk_scores = [len(rec.risk_considerations) for rec in recommendations]
        risk_penalty = np.mean(risk_scores) * 0.05
        risk_adjusted_confidence = max(0.0, avg_confidence - risk_penalty)

        return {
            "overall_confidence": avg_confidence,
            "evidence_strength": min(1.0, evidence_strength),
            "risk_adjusted_confidence": risk_adjusted_confidence,
            "recommendation_consistency": self._calculate_consistency_score(recommendations)
        }

    def _calculate_consistency_score(self, recommendations: list[InvestmentRecommendation]) -> float:
        """Calculate consistency score across recommendations."""
        if len(recommendations) < 2:
            return 1.0

        # Check if recommendations are consistent in their approach
        allocation_variance = np.var([rec.allocation_percentage for rec in recommendations])
        confidence_variance = np.var([rec.confidence_level for rec in recommendations])

        # Lower variance = higher consistency
        consistency = 1.0 - min(1.0, (allocation_variance + confidence_variance * 100) / 1000)
        return max(0.0, consistency)

    def export_decision_report(self, report: InvestmentDecisionReport, output_path: str) -> None:
        """Export investment decision report to JSON."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dictionary for JSON serialisation
        report_dict = asdict(report)

        with open(output_file, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)

        self.logger.info(f"Investment decision report exported to {output_file}")

    def _load_comparative_results(self) -> dict:
        """Load comparative analysis results."""
        try:
            results_file = self.results_dir / "comparative_analysis" / "optimal_approach_recommendation.json"
            if results_file.exists():
                with open(results_file) as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load comparative results: {e}")
        return {}

    def _load_strategy_rankings(self) -> pd.DataFrame:
        """Load strategy rankings."""
        try:
            rankings_file = self.results_dir / "comparative_analysis" / "strategy_rankings.csv"
            if rankings_file.exists():
                return pd.read_csv(rankings_file)
        except Exception as e:
            self.logger.warning(f"Could not load strategy rankings: {e}")
        return pd.DataFrame()

    def _load_goals_validation_results(self) -> dict:
        """Load project goals validation results."""
        try:
            results_file = self.results_dir / "validation" / "project_goals_validation_summary.json"
            if results_file.exists():
                with open(results_file) as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load goals validation results: {e}")
        return {}

    def _load_robustness_results(self) -> dict:
        """Load robustness testing results."""
        try:
            results_file = self.results_dir / "validation" / "robustness_test_summary.json"
            if results_file.exists():
                with open(results_file) as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load robustness results: {e}")
        return {}


def main():
    """Main execution function for investment decision support."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialise engine
    engine = InvestmentDecisionSupportEngine()

    try:
        # Generate comprehensive investment decision report
        report = engine.generate_comprehensive_report()

        # Export results
        output_path = "results/investment_decision/investment_decision_report.json"
        engine.export_decision_report(report, output_path)


        # Print key findings

        return True

    except Exception as e:
        logging.error(f"Investment decision support generation failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
