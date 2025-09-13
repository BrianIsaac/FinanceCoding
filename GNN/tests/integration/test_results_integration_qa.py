"""
Integration tests for Results Integration QA Engine.

This module tests the comprehensive QA framework to ensure proper
validation of all research components and integration quality.
"""

import json
import logging
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from src.evaluation.validation.results_integration_qa import (
    ResultsIntegrationQAEngine,
    QualityAssuranceReport,
    QualityMetric,
    IntegrationValidation
)


@pytest.fixture
def qa_engine():
    """Create QA engine fixture for testing."""
    return ResultsIntegrationQAEngine(results_dir="tests/fixtures/results")


@pytest.fixture
def mock_results_dir(tmp_path):
    """Create mock results directory structure."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # Create subdirectories
    (results_dir / "performance_analytics").mkdir()
    (results_dir / "comparative_analysis").mkdir()
    (results_dir / "validation").mkdir()
    (results_dir / "investment_decision").mkdir()
    (results_dir / "qa").mkdir()
    
    # Create mock data files
    perf_data = {
        "sharpe_ratio": 1.25,
        "information_ratio": 0.45,
        "calmar_ratio": 0.67,
        "returns": [0.01, 0.02, -0.005, 0.015],
        "strategy_returns": [0.012, 0.018, -0.002, 0.020],
        "benchmark_returns": [0.010, 0.015, -0.003, 0.018],
        "annual_return": 0.125,
        "max_drawdown": -0.15,
        "strategy_names": ["HRP_Strategy", "GAT_Strategy"]
    }
    
    with open(results_dir / "performance_analytics" / "performance_analytics_results.json", "w") as f:
        json.dump(perf_data, f)
    
    comparative_data = {
        "primary_recommendation": "HRP_Strategy",
        "confidence_level": 0.849,
        "strategy_names": ["HRP_Strategy", "GAT_Strategy"],
        "statistical_significance_rate": 0.75,
        "strategies": {
            "HRP_Strategy": {"sharpe_ratio": 1.25},
            "GAT_Strategy": {"sharpe_ratio": 1.18}
        }
    }
    
    with open(results_dir / "comparative_analysis" / "optimal_approach_recommendation.json", "w") as f:
        json.dump(comparative_data, f)
    
    goals_data = {
        "overall_success_rate": 0.82,
        "goal_results": [{"goal": "test1", "status": "pass"}, {"goal": "test2", "status": "pass"}]
    }
    
    with open(results_dir / "validation" / "project_goals_validation_summary.json", "w") as f:
        json.dump(goals_data, f)
    
    robustness_data = {
        "overall_robustness_score": 0.85,
        "test_results": [{"test": "stress1", "result": 0.8}],
        "transaction_cost_sensitivity": {"high_cost_performance": 0.75},
        "parameter_robustness": {"robustness_score": 0.82},
        "market_regime_stress": {"stress_score": 0.78}
    }
    
    with open(results_dir / "validation" / "robustness_test_summary.json", "w") as f:
        json.dump(robustness_data, f)
    
    investment_data = {
        "primary_recommendation": "HRP_Strategy",
        "risk_assessment": {"overall_risk": "Medium"},
        "implementation_guidance": {"timeline": "6-8 weeks"}
    }
    
    with open(results_dir / "investment_decision" / "investment_decision_report.json", "w") as f:
        json.dump(investment_data, f)
    
    return results_dir


class TestResultsIntegrationQAEngine:
    """Test suite for Results Integration QA Engine."""
    
    def test_qa_engine_initialization(self, tmp_path):
        """Test QA engine initialises correctly."""
        engine = ResultsIntegrationQAEngine(str(tmp_path))
        
        assert engine.results_dir == tmp_path
        assert engine.qa_thresholds['statistical_accuracy'] == 0.001
        assert engine.qa_thresholds['data_consistency'] == 0.95
        assert engine.qa_thresholds['performance_benchmark'] == 0.9
        assert hasattr(engine, 'goals_validator')
        assert hasattr(engine, 'robustness_tester')
        assert hasattr(engine, 'comparative_engine')
        assert hasattr(engine, 'decision_engine')
    
    def test_statistical_accuracy_validation(self, mock_results_dir):
        """Test statistical accuracy validation against reference implementations."""
        engine = ResultsIntegrationQAEngine(str(mock_results_dir))
        
        results = engine._validate_statistical_accuracy()
        
        assert isinstance(results, dict)
        assert 'overall_statistical_accuracy' in results
        assert results['overall_statistical_accuracy'] >= 0.0
        assert results['overall_statistical_accuracy'] <= 1.0
        
        # Check individual metric validations
        for metric in ['sharpe_ratio', 'information_ratio', 'calmar_ratio']:
            assert f'{metric}_error' in results
            assert f'{metric}_pass' in results
            assert isinstance(results[f'{metric}_error'], float)
            assert isinstance(results[f'{metric}_pass'], float)
    
    def test_data_consistency_validation(self, mock_results_dir):
        """Test data consistency validation across components."""
        engine = ResultsIntegrationQAEngine(str(mock_results_dir))
        
        results = engine._validate_data_consistency()
        
        assert isinstance(results, dict)
        assert 'overall_consistency' in results
        assert 'strategy_names_consistent' in results
        assert 'temporal_data_consistent' in results
        assert 'missing_data_handling_consistent' in results
        
        # Check consistency metrics
        assert isinstance(results['overall_consistency'], float)
        assert results['overall_consistency'] >= 0.0
        assert results['overall_consistency'] <= 1.0
    
    def test_performance_benchmarks_validation(self, mock_results_dir):
        """Test performance benchmarks compliance validation."""
        engine = ResultsIntegrationQAEngine(str(mock_results_dir))
        
        results = engine._validate_performance_benchmarks()
        
        assert isinstance(results, dict)
        assert 'overall_performance_score' in results
        assert 'memory_efficiency' in results
        assert 'processing_time_compliance' in results
        assert 'computational_efficiency' in results
        assert 'scalability_compliance' in results
        
        # Check all scores are valid
        for key, value in results.items():
            if isinstance(value, (int, float)):
                assert 0.0 <= value <= 1.0
    
    def test_component_integration_validation(self, mock_results_dir):
        """Test integration validation between components."""
        engine = ResultsIntegrationQAEngine(str(mock_results_dir))
        
        results = engine._validate_component_integration()
        
        assert isinstance(results, list)
        assert len(results) == 4  # Goals, Robustness, Comparative, Decision
        
        for validation in results:
            assert isinstance(validation, IntegrationValidation)
            assert validation.component_name in [
                "Project Goals Validator",
                "Robustness Testing Framework", 
                "Comparative Analysis Engine",
                "Investment Decision Support"
            ]
            assert validation.validation_status in ['pass', 'fail', 'error']
            assert isinstance(validation.error_tolerance, float)
            assert isinstance(validation.actual_error, float)
    
    def test_quality_metrics_assessment(self, mock_results_dir):
        """Test quality metrics assessment."""
        engine = ResultsIntegrationQAEngine(str(mock_results_dir))
        
        # Mock component results
        statistical_results = {'overall_statistical_accuracy': 0.99}
        consistency_results = {'overall_consistency': 0.96}
        performance_results = {'overall_performance_score': 0.91}
        integration_results = [
            IntegrationValidation(
                component_name="Test Component",
                validation_status="pass",
                error_tolerance=0.01,
                actual_error=0.005,
                reference_value=1.0,
                computed_value=0.995,
                validation_details={}
            )
        ]
        
        metrics = engine._assess_quality_metrics(
            statistical_results, consistency_results,
            performance_results, integration_results
        )
        
        assert isinstance(metrics, list)
        assert len(metrics) == 4  # Statistical, Consistency, Performance, Integration
        
        for metric in metrics:
            assert isinstance(metric, QualityMetric)
            assert metric.name in [
                "Statistical Accuracy",
                "Data Consistency", 
                "Performance Benchmark",
                "Component Integration"
            ]
            assert metric.status in ['pass', 'fail']
            assert 0.0 <= metric.value <= 1.0
    
    def test_expert_review_checklist_generation(self, mock_results_dir):
        """Test expert review checklist generation."""
        engine = ResultsIntegrationQAEngine(str(mock_results_dir))
        
        checklist = engine._generate_expert_review_checklist()
        
        assert isinstance(checklist, dict)
        
        expected_checks = [
            'statistical_methodology_valid',
            'research_design_sound',
            'results_interpretation_appropriate',
            'publication_standards_met',
            'risk_assessment_comprehensive',
            'implementation_feasible'
        ]
        
        for check in expected_checks:
            if check in checklist:
                assert isinstance(checklist[check], bool)
    
    def test_comprehensive_qa_validation(self, mock_results_dir):
        """Test complete comprehensive QA validation workflow."""
        engine = ResultsIntegrationQAEngine(str(mock_results_dir))
        
        report = engine.execute_comprehensive_qa_validation()
        
        assert isinstance(report, QualityAssuranceReport)
        assert report.overall_qa_status in ['PASS', 'FAIL', 'WARNING']
        assert 0.0 <= report.quality_score <= 1.0
        assert isinstance(report.validation_summary, dict)
        assert isinstance(report.quality_metrics, list)
        assert isinstance(report.integration_validations, list)
        assert isinstance(report.statistical_accuracy_results, dict)
        assert isinstance(report.data_consistency_results, dict)
        assert isinstance(report.performance_benchmarks, dict)
        assert isinstance(report.expert_review_checklist, dict)
        assert isinstance(report.final_recommendations, list)
        assert report.generated_date is not None
    
    def test_qa_report_export(self, mock_results_dir, tmp_path):
        """Test QA report export functionality."""
        engine = ResultsIntegrationQAEngine(str(mock_results_dir))
        
        # Generate a report
        report = engine.execute_comprehensive_qa_validation()
        
        # Export to temporary location
        output_path = tmp_path / "test_qa_report.json"
        engine.export_qa_report(report, str(output_path))
        
        # Verify file was created and contains valid JSON
        assert output_path.exists()
        
        with open(output_path) as f:
            exported_data = json.load(f)
        
        assert 'overall_qa_status' in exported_data
        assert 'quality_score' in exported_data
        assert 'validation_summary' in exported_data
        assert 'quality_metrics' in exported_data
    
    def test_reference_metric_calculations(self, mock_results_dir):
        """Test reference metric calculations for validation."""
        engine = ResultsIntegrationQAEngine(str(mock_results_dir))
        
        # Test data
        test_data = {
            'returns': [0.01, 0.02, -0.005, 0.015, 0.008],
            'strategy_returns': [0.012, 0.018, -0.002, 0.020, 0.010],
            'benchmark_returns': [0.010, 0.015, -0.003, 0.018, 0.008],
            'annual_return': 0.125,
            'max_drawdown': -0.15
        }
        
        # Test Sharpe ratio calculation
        sharpe = engine._calculate_reference_metric('sharpe_ratio', test_data)
        assert isinstance(sharpe, (float, type(None)))
        if sharpe is not None:
            assert sharpe > 0  # Should be positive for these returns
        
        # Test Information ratio calculation
        ir = engine._calculate_reference_metric('information_ratio', test_data)
        assert isinstance(ir, (float, type(None)))
        
        # Test Calmar ratio calculation
        calmar = engine._calculate_reference_metric('calmar_ratio', test_data)
        assert isinstance(calmar, (float, type(None)))
        if calmar is not None:
            assert calmar > 0  # Should be positive
    
    def test_overall_qa_status_assessment(self, mock_results_dir):
        """Test overall QA status and recommendation generation."""
        engine = ResultsIntegrationQAEngine(str(mock_results_dir))
        
        # Mock quality metrics
        quality_metrics = [
            QualityMetric(
                name="Test Metric",
                value=0.95,
                threshold=0.9,
                status="pass",
                description="Test description",
                validation_method="Test method"
            )
        ]
        
        integration_results = [
            IntegrationValidation(
                component_name="Test Component",
                validation_status="pass",
                error_tolerance=0.01,
                actual_error=0.005,
                reference_value=1.0,
                computed_value=0.995,
                validation_details={}
            )
        ]
        
        status, score, summary = engine._assess_overall_qa_status(
            quality_metrics, integration_results
        )
        
        assert status in ['PASS', 'FAIL', 'WARNING']
        assert 0.0 <= score <= 1.0
        assert isinstance(summary, dict)
        assert 'Overall Status' in summary
        assert 'Quality Score' in summary
        assert 'Recommendation' in summary
    
    def test_error_handling_missing_files(self, tmp_path):
        """Test error handling when result files are missing."""
        # Create empty results directory
        engine = ResultsIntegrationQAEngine(str(tmp_path))
        
        # These should handle missing files gracefully
        statistical_results = engine._validate_statistical_accuracy()
        consistency_results = engine._validate_data_consistency()
        performance_results = engine._validate_performance_benchmarks()
        integration_results = engine._validate_component_integration()
        
        # Should return valid structures even with missing data
        assert isinstance(statistical_results, dict)
        assert isinstance(consistency_results, dict)
        assert isinstance(performance_results, dict)
        assert isinstance(integration_results, list)
        
        # Should handle errors gracefully
        assert 'validation_error' in statistical_results or 'overall_statistical_accuracy' in statistical_results
    
    def test_graceful_handling_missing_files(self, tmp_path):
        """Test graceful handling of missing files without crashing."""
        engine = ResultsIntegrationQAEngine(str(tmp_path))
        
        # Try to load non-existent performance data - should not crash
        result = engine._load_performance_analytics()
        assert result == {}
        
        # Try other load methods - should handle gracefully
        comparative_result = engine._load_comparative_results()
        goals_result = engine._load_goals_validation_results()
        robustness_result = engine._load_robustness_results()
        
        assert isinstance(comparative_result, dict)
        assert isinstance(goals_result, dict)
        assert isinstance(robustness_result, dict)


def test_integration_with_main_framework():
    """Test integration with the main testing framework."""
    # This test verifies that the QA engine can be instantiated
    # and doesn't conflict with other components
    try:
        engine = ResultsIntegrationQAEngine()
        assert engine is not None
        assert hasattr(engine, 'execute_comprehensive_qa_validation')
    except Exception as e:
        pytest.fail(f"QA engine integration failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])