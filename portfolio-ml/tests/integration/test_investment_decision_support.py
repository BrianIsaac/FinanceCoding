"""
Integration tests for Investment Decision Support Engine.

This module tests the investment decision support framework to ensure
proper generation of actionable investment recommendations.
"""

import json
import logging
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch

from src.evaluation.reporting.investment_decision_support import (
    InvestmentDecisionSupportEngine,
    InvestmentDecisionReport,
    InvestmentRecommendation,
    RiskAssessment,
    ImplementationGuidance
)


@pytest.fixture
def decision_engine():
    """Create decision support engine fixture for testing."""
    return InvestmentDecisionSupportEngine(results_dir="tests/fixtures/results")


@pytest.fixture
def mock_results_dir(tmp_path):
    """Create mock results directory structure with realistic data."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # Create subdirectories
    (results_dir / "comparative_analysis").mkdir()
    (results_dir / "validation").mkdir()
    (results_dir / "investment_decision").mkdir()
    
    # Create mock comparative analysis results
    comparative_data = {
        "primary_recommendation": "HRP_average_correlation_756",
        "confidence_level": 0.849,
        "strategy_names": ["HRP_average_correlation_756", "GAT_MST", "LSTM_Model", "Equal_Weight"],
        "statistical_significance_rate": 0.75
    }
    
    with open(results_dir / "comparative_analysis" / "optimal_approach_recommendation.json", "w") as f:
        json.dump(comparative_data, f)
    
    # Create mock strategy rankings
    strategy_rankings = pd.DataFrame({
        'strategy_name': [
            'HRP_average_correlation_756',
            'GAT_MST', 
            'LSTM_Model',
            'Equal_Weight',
            'Market_Cap_Weighted'
        ],
        'overall_score': [0.88, 0.82, 0.76, 0.65, 0.58],
        'sharpe_ratio': [1.25, 1.18, 1.12, 0.89, 0.85],
        'max_drawdown': [-0.15, -0.18, -0.21, -0.22, -0.25],
        'recommendation': ['Implement', 'Implement', 'Monitor', 'Baseline', 'Baseline'],
        'statistical_significance_count': [15, 12, 10, 5, 3],
        'dominance_score': [0.75, 0.68, 0.62, 0.45, 0.38],
        'annual_turnover': [1.2, 1.8, 2.1, 0.0, 0.0]
    })
    
    strategy_rankings.to_csv(results_dir / "comparative_analysis" / "strategy_rankings.csv", index=False)
    
    # Create mock goals validation results
    goals_data = {
        "overall_success_rate": 0.82,
        "goal_results": [
            {"goal": "outperform_benchmark", "status": "pass", "score": 0.85},
            {"goal": "risk_control", "status": "pass", "score": 0.78}
        ]
    }
    
    with open(results_dir / "validation" / "project_goals_validation_summary.json", "w") as f:
        json.dump(goals_data, f)
    
    # Create mock robustness results
    robustness_data = {
        "overall_robustness_score": 0.85,
        "test_results": [
            {"test": "transaction_cost_stress", "result": 0.82},
            {"test": "parameter_sensitivity", "result": 0.88}
        ],
        "transaction_cost_sensitivity": {"high_cost_performance": 0.75},
        "parameter_robustness": {"robustness_score": 0.82},
        "market_regime_stress": {"stress_score": 0.78}
    }
    
    with open(results_dir / "validation" / "robustness_test_summary.json", "w") as f:
        json.dump(robustness_data, f)
    
    return results_dir


class TestInvestmentDecisionSupportEngine:
    """Test suite for Investment Decision Support Engine."""
    
    def test_engine_initialization(self, tmp_path):
        """Test decision support engine initialises correctly."""
        engine = InvestmentDecisionSupportEngine(str(tmp_path))
        
        assert engine.results_dir == tmp_path
        assert hasattr(engine, 'comparative_engine')
        assert hasattr(engine, 'goals_validator')
        assert hasattr(engine, 'robustness_tester')
        assert engine.risk_thresholds['low_risk'] == 0.3
        assert engine.risk_thresholds['medium_risk'] == 0.6
        assert engine.risk_thresholds['high_risk'] == 0.8
        assert engine.performance_expectations['min_sharpe_ratio'] == 1.0
    
    def test_investment_recommendations_generation(self, mock_results_dir):
        """Test generation of investment recommendations."""
        engine = InvestmentDecisionSupportEngine(str(mock_results_dir))
        
        recommendations = engine.generate_investment_recommendations()
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Check primary recommendation
        primary_rec = recommendations[0]
        assert isinstance(primary_rec, InvestmentRecommendation)
        assert primary_rec.strategy_name == "HRP_average_correlation_756"
        assert 60.0 <= primary_rec.allocation_percentage <= 100.0  # Primary allocation range
        assert 0.0 <= primary_rec.confidence_level <= 1.0
        assert isinstance(primary_rec.supporting_evidence, list)
        assert isinstance(primary_rec.risk_considerations, list)
        assert isinstance(primary_rec.monitoring_metrics, list)
        
        # Check alternative recommendations if present
        if len(recommendations) > 1:
            for alt_rec in recommendations[1:]:
                assert isinstance(alt_rec, InvestmentRecommendation)
                assert 10.0 <= alt_rec.allocation_percentage <= 30.0  # Alternative allocation range
                assert alt_rec.strategy_name != primary_rec.strategy_name
    
    def test_recommendation_confidence_calculation(self, mock_results_dir):
        """Test recommendation confidence calculation."""
        engine = InvestmentDecisionSupportEngine(str(mock_results_dir))
        
        # Load strategy data for testing
        rankings_df = engine._load_strategy_rankings()
        strategy_data = rankings_df.iloc[0]  # Best performing strategy
        
        confidence = engine._calculate_recommendation_confidence(strategy_data)
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        
        # High-performing strategies should have higher confidence
        assert confidence > 0.5  # Should be reasonably confident for top strategy
    
    def test_supporting_evidence_generation(self, mock_results_dir):
        """Test supporting evidence generation for recommendations."""
        engine = InvestmentDecisionSupportEngine(str(mock_results_dir))
        
        rankings_df = engine._load_strategy_rankings()
        strategy_data = rankings_df.iloc[0]  # Best performing strategy
        
        evidence = engine._generate_supporting_evidence(strategy_data)
        
        assert isinstance(evidence, list)
        assert len(evidence) > 0
        
        # Check evidence contains relevant performance metrics
        evidence_text = " ".join(evidence)
        assert any(keyword in evidence_text.lower() for keyword in [
            'sharpe', 'performance', 'risk', 'drawdown', 'significant', 'dominance'
        ])
    
    def test_strategy_risk_assessment(self, mock_results_dir):
        """Test strategy-specific risk assessment."""
        engine = InvestmentDecisionSupportEngine(str(mock_results_dir))
        
        rankings_df = engine._load_strategy_rankings()
        strategy_data = rankings_df.iloc[0]
        
        risks = engine._assess_strategy_risks(strategy_data)
        
        assert isinstance(risks, list)
        # Risks could be empty for well-performing strategies
        
        # Test high-risk strategy
        high_risk_data = pd.Series({
            'strategy_name': 'GAT_Complex',
            'max_drawdown': -0.45,  # High drawdown
            'sharpe_ratio': 0.6,    # Low Sharpe
            'overall_score': 0.2    # Low score
        })
        
        high_risks = engine._assess_strategy_risks(high_risk_data)
        assert len(high_risks) > 0
        assert any('drawdown' in risk.lower() for risk in high_risks)
    
    def test_monitoring_metrics_definition(self, mock_results_dir):
        """Test monitoring metrics definition for different strategies."""
        engine = InvestmentDecisionSupportEngine(str(mock_results_dir))
        
        # Test HRP strategy
        hrp_metrics = engine._define_monitoring_metrics("HRP_average_correlation_756")
        assert isinstance(hrp_metrics, list)
        assert any('correlation' in metric.lower() for metric in hrp_metrics)
        assert any('clustering' in metric.lower() for metric in hrp_metrics)
        
        # Test GAT strategy
        gat_metrics = engine._define_monitoring_metrics("GAT_MST")
        assert isinstance(gat_metrics, list)
        assert any('accuracy' in metric.lower() for metric in gat_metrics)
        assert any('feature' in metric.lower() for metric in gat_metrics)
        
        # All strategies should have base metrics
        for metrics in [hrp_metrics, gat_metrics]:
            assert any('sharpe' in metric.lower() for metric in metrics)
            assert any('drawdown' in metric.lower() for metric in metrics)
    
    def test_comprehensive_risk_assessment(self, mock_results_dir):
        """Test comprehensive risk assessment across recommendations."""
        engine = InvestmentDecisionSupportEngine(str(mock_results_dir))
        
        # Generate recommendations first
        recommendations = engine.generate_investment_recommendations()
        
        # Assess comprehensive risk
        risk_assessment = engine.assess_comprehensive_risk(recommendations)
        
        assert isinstance(risk_assessment, RiskAssessment)
        assert 0.0 <= risk_assessment.market_risk_score <= 1.0
        assert 0.0 <= risk_assessment.operational_risk_score <= 1.0
        assert 0.0 <= risk_assessment.model_risk_score <= 1.0
        assert 0.0 <= risk_assessment.liquidity_risk_score <= 1.0
        assert risk_assessment.overall_risk_rating in ["Low", "Medium", "High", "Very High"]
        assert isinstance(risk_assessment.risk_mitigation_measures, list)
        assert len(risk_assessment.risk_mitigation_measures) > 0
        assert isinstance(risk_assessment.stress_test_results, dict)
    
    def test_market_risk_calculation(self, mock_results_dir):
        """Test market risk calculation."""
        engine = InvestmentDecisionSupportEngine(str(mock_results_dir))
        
        # Create test recommendations
        test_recommendations = [
            InvestmentRecommendation(
                strategy_name="Test_Strategy",
                allocation_percentage=50.0,
                confidence_level=0.8,
                expected_return=0.12,
                expected_volatility=0.15,
                max_drawdown_estimate=-0.2,
                implementation_priority="Implement",
                supporting_evidence=[],
                risk_considerations=[],
                monitoring_metrics=[]
            )
        ]
        
        market_risk = engine._calculate_market_risk(test_recommendations)
        
        assert isinstance(market_risk, float)
        assert 0.0 <= market_risk <= 1.0
        
        # Higher drawdown should result in higher risk
        high_drawdown_recs = [
            InvestmentRecommendation(
                strategy_name="High_Risk_Strategy",
                allocation_percentage=50.0,
                confidence_level=0.6,
                expected_return=0.15,
                expected_volatility=0.25,
                max_drawdown_estimate=-0.4,  # High drawdown
                implementation_priority="Monitor",
                supporting_evidence=[],
                risk_considerations=[],
                monitoring_metrics=[]
            )
        ]
        
        high_market_risk = engine._calculate_market_risk(high_drawdown_recs)
        assert high_market_risk > market_risk
    
    def test_implementation_guidance_generation(self, mock_results_dir):
        """Test implementation guidance generation."""
        engine = InvestmentDecisionSupportEngine(str(mock_results_dir))
        
        # Generate recommendations
        recommendations = engine.generate_investment_recommendations()
        
        # Generate implementation guidance
        guidance = engine.generate_implementation_guidance(recommendations)
        
        assert isinstance(guidance, ImplementationGuidance)
        assert guidance.deployment_timeline in ["3-4 weeks", "6-8 weeks"]
        assert isinstance(guidance.resource_requirements, list)
        assert len(guidance.resource_requirements) > 0
        assert isinstance(guidance.technical_prerequisites, list)
        assert len(guidance.technical_prerequisites) > 0
        assert isinstance(guidance.regulatory_considerations, list)
        assert len(guidance.regulatory_considerations) > 0
        assert guidance.monitoring_frequency == "Daily"
        assert guidance.rebalancing_schedule == "Monthly"
        assert guidance.performance_review_cycle == "Quarterly"
        
        # Check if ML strategies require additional resources
        has_ml_strategies = any('GAT' in rec.strategy_name or 'LSTM' in rec.strategy_name
                              for rec in recommendations)
        if has_ml_strategies:
            assert guidance.deployment_timeline == "6-8 weeks"
            resources_text = " ".join(guidance.resource_requirements).lower()
            assert any(keyword in resources_text for keyword in [
                'machine learning', 'gpu', 'mlops'
            ])
    
    def test_comprehensive_report_generation(self, mock_results_dir):
        """Test comprehensive investment decision report generation."""
        engine = InvestmentDecisionSupportEngine(str(mock_results_dir))
        
        report = engine.generate_comprehensive_report()
        
        assert isinstance(report, InvestmentDecisionReport)
        assert isinstance(report.primary_recommendation, InvestmentRecommendation)
        assert isinstance(report.alternative_recommendations, list)
        assert isinstance(report.risk_assessment, RiskAssessment)
        assert isinstance(report.implementation_guidance, ImplementationGuidance)
        assert isinstance(report.research_summary, dict)
        assert isinstance(report.confidence_metrics, dict)
        assert report.generated_date is not None
        
        # Check research summary content
        assert 'project_goals_achievement' in report.research_summary
        assert 'optimal_strategy_identified' in report.research_summary
        assert 'strategy_confidence' in report.research_summary
        assert 'robustness_score' in report.research_summary
        
        # Check confidence metrics
        assert 'overall_confidence' in report.confidence_metrics
        assert 'evidence_strength' in report.confidence_metrics
        assert 'risk_adjusted_confidence' in report.confidence_metrics
        assert 'recommendation_consistency' in report.confidence_metrics
        
        for metric in report.confidence_metrics.values():
            assert isinstance(metric, (int, float))
            assert 0.0 <= metric <= 1.0
    
    def test_confidence_metrics_calculation(self, mock_results_dir):
        """Test confidence metrics calculation."""
        engine = InvestmentDecisionSupportEngine(str(mock_results_dir))
        
        # Create test recommendations
        test_recommendations = [
            InvestmentRecommendation(
                strategy_name="Strategy_1",
                allocation_percentage=60.0,
                confidence_level=0.85,
                expected_return=0.12,
                expected_volatility=0.15,
                max_drawdown_estimate=-0.18,
                implementation_priority="Implement",
                supporting_evidence=["High Sharpe ratio", "Low drawdown", "Consistent performance"],
                risk_considerations=["Model complexity"],
                monitoring_metrics=["Daily Sharpe", "Monthly returns"]
            ),
            InvestmentRecommendation(
                strategy_name="Strategy_2",
                allocation_percentage=25.0,
                confidence_level=0.78,
                expected_return=0.10,
                expected_volatility=0.14,
                max_drawdown_estimate=-0.16,
                implementation_priority="Monitor",
                supporting_evidence=["Stable performance", "Good risk control"],
                risk_considerations=["Market sensitivity", "Parameter drift"],
                monitoring_metrics=["Weekly review", "Risk tracking"]
            )
        ]
        
        metrics = engine._calculate_confidence_metrics(test_recommendations)
        
        assert isinstance(metrics, dict)
        assert 'overall_confidence' in metrics
        assert 'evidence_strength' in metrics
        assert 'risk_adjusted_confidence' in metrics
        assert 'recommendation_consistency' in metrics
        
        # Overall confidence should be average of individual confidences
        expected_confidence = (0.85 + 0.78) / 2
        assert abs(metrics['overall_confidence'] - expected_confidence) < 0.01
        
        # Evidence strength should reflect number of evidence points
        assert metrics['evidence_strength'] > 0.0
        
        # Risk-adjusted confidence should be lower than overall confidence
        assert metrics['risk_adjusted_confidence'] <= metrics['overall_confidence']
    
    def test_decision_report_export(self, mock_results_dir, tmp_path):
        """Test decision report export functionality."""
        engine = InvestmentDecisionSupportEngine(str(mock_results_dir))
        
        # Generate comprehensive report
        report = engine.generate_comprehensive_report()
        
        # Export to temporary location
        output_path = tmp_path / "test_decision_report.json"
        engine.export_decision_report(report, str(output_path))
        
        # Verify file was created and contains valid JSON
        assert output_path.exists()
        
        with open(output_path) as f:
            exported_data = json.load(f)
        
        assert 'primary_recommendation' in exported_data
        assert 'risk_assessment' in exported_data
        assert 'implementation_guidance' in exported_data
        assert 'research_summary' in exported_data
        assert 'confidence_metrics' in exported_data
    
    def test_error_handling_missing_files(self, tmp_path):
        """Test error handling when result files are missing."""
        # Create empty results directory
        engine = InvestmentDecisionSupportEngine(str(tmp_path))
        
        # These should handle missing files gracefully
        comparative_results = engine._load_comparative_results()
        rankings_df = engine._load_strategy_rankings()
        goals_results = engine._load_goals_validation_results()
        robustness_results = engine._load_robustness_results()
        
        # Should return valid structures even with missing data
        assert isinstance(comparative_results, dict)
        assert isinstance(rankings_df, pd.DataFrame)
        assert isinstance(goals_results, dict)
        assert isinstance(robustness_results, dict)
        
        # Empty DataFrame should be handled
        assert len(rankings_df) == 0 or not rankings_df.empty
    
    def test_error_handling_no_recommendations(self, tmp_path):
        """Test error handling when no recommendations can be generated."""
        # Create engine with empty results directory
        engine = InvestmentDecisionSupportEngine(str(tmp_path))
        
        # Should raise ValueError when no recommendations can be generated
        with pytest.raises(ValueError, match="No recommendations could be generated"):
            engine.generate_comprehensive_report()
    
    def test_risk_mitigation_measures_generation(self, mock_results_dir):
        """Test risk mitigation measures generation."""
        engine = InvestmentDecisionSupportEngine(str(mock_results_dir))
        
        # Test with various risk levels
        measures_low = engine._generate_risk_mitigation_measures(0.2, 0.3, 0.2, 0.1)
        measures_high = engine._generate_risk_mitigation_measures(0.8, 0.7, 0.8, 0.5)
        
        assert isinstance(measures_low, list)
        assert isinstance(measures_high, list)
        
        # High-risk scenarios should have more mitigation measures
        assert len(measures_high) >= len(measures_low)
        
        # Always should include base measures
        for measures in [measures_low, measures_high]:
            measures_text = " ".join(measures).lower()
            assert any(keyword in measures_text for keyword in [
                'diversify', 'monitor', 'cash', 'deployment'
            ])
    
    def test_graceful_handling_missing_files(self, tmp_path):
        """Test graceful handling of missing files without crashing."""
        engine = InvestmentDecisionSupportEngine(str(tmp_path))
        
        # Try to load non-existent files - should not crash
        comparative_result = engine._load_comparative_results()
        rankings_result = engine._load_strategy_rankings()
        goals_result = engine._load_goals_validation_results()
        robustness_result = engine._load_robustness_results()
        
        assert isinstance(comparative_result, dict)
        assert isinstance(rankings_result, pd.DataFrame)
        assert isinstance(goals_result, dict)
        assert isinstance(robustness_result, dict)


def test_integration_with_main_framework():
    """Test integration with the main testing framework."""
    # This test verifies that the decision support engine can be instantiated
    # and doesn't conflict with other components
    try:
        engine = InvestmentDecisionSupportEngine()
        assert engine is not None
        assert hasattr(engine, 'generate_comprehensive_report')
        assert hasattr(engine, 'generate_investment_recommendations')
        assert hasattr(engine, 'assess_comprehensive_risk')
    except Exception as e:
        pytest.fail(f"Decision support engine integration failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])