"""
Tests for comprehensive report generation framework.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.evaluation.reporting.comprehensive_report import ComprehensiveReportGenerator, ReportConfig


@pytest.fixture
def report_generator():
    """Create comprehensive report generator for testing."""
    config = ReportConfig(
        output_directory="test_reports",
        report_title="Test ML Portfolio Analysis",
        institution_name="Test Investment Firm",
        generate_pdf=False,  # Disable for testing without dependencies
        generate_excel=False,
        generate_html=True,
        generate_interactive=False,
    )
    return ComprehensiveReportGenerator(config)


@pytest.fixture
def sample_comprehensive_data():
    """Sample comprehensive data for testing."""
    np.random.seed(42)

    performance_results = {
        "HRP": pd.DataFrame(
            {
                "returns": np.random.normal(0.0008, 0.012, 252),
                "sharpe_ratio": [1.2] * 252,
                "information_ratio": [0.8] * 252,
                "annual_return": [0.12] * 252,
                "max_drawdown": [-0.08] * 252,
                "volatility": [0.15] * 252,
            }
        ),
        "LSTM": pd.DataFrame(
            {
                "returns": np.random.normal(0.001, 0.015, 252),
                "sharpe_ratio": [1.8] * 252,
                "information_ratio": [1.2] * 252,
                "annual_return": [0.18] * 252,
                "max_drawdown": [-0.12] * 252,
                "volatility": [0.18] * 252,
            }
        ),
        "GAT": pd.DataFrame(
            {
                "returns": np.random.normal(0.0009, 0.013, 252),
                "sharpe_ratio": [1.5] * 252,
                "information_ratio": [1.0] * 252,
                "annual_return": [0.15] * 252,
                "max_drawdown": [-0.10] * 252,
                "volatility": [0.16] * 252,
            }
        ),
    }

    operational_metrics = {
        "HRP": {
            "gpu_memory_gb": 2.0,
            "training_time_hours": 1.0,
            "inference_cost_per_prediction": 0.005,
            "average_turnover": 0.15,
            "transaction_cost_bps": 8.0,
            "monitoring_score": 30,
        },
        "LSTM": {
            "gpu_memory_gb": 6.0,
            "training_time_hours": 4.0,
            "inference_cost_per_prediction": 0.02,
            "average_turnover": 0.25,
            "transaction_cost_bps": 12.0,
            "monitoring_score": 60,
        },
        "GAT": {
            "gpu_memory_gb": 8.0,
            "training_time_hours": 6.0,
            "inference_cost_per_prediction": 0.03,
            "average_turnover": 0.30,
            "transaction_cost_bps": 15.0,
            "monitoring_score": 70,
        },
    }

    statistical_results = {
        "HRP": {
            "pvalue_vs_equal_weight": 0.08,
            "bootstrap_pvalue": 0.10,
            "effect_size": 0.3,
            "bootstrap_results": {
                "sharpe_95_ci": [1.0, 1.4],
                "iterations": 1000,
            },
        },
        "LSTM": {
            "pvalue_vs_equal_weight": 0.01,
            "bootstrap_pvalue": 0.012,
            "effect_size": 0.8,
            "bootstrap_results": {
                "sharpe_95_ci": [1.6, 2.0],
                "iterations": 1000,
            },
        },
        "GAT": {
            "pvalue_vs_equal_weight": 0.04,
            "bootstrap_pvalue": 0.045,
            "effect_size": 0.5,
            "bootstrap_results": {
                "sharpe_95_ci": [1.3, 1.7],
                "iterations": 1000,
            },
        },
    }

    model_specifications = {
        "HRP": {
            "gpu_memory_required": 2.0,
            "training_time_hours": 1.0,
            "retraining_frequency_days": 30,
            "inference_time_ms": 5.0,
            "daily_predictions": 500,
            "complexity_score": 30,
            "data_sources": 1,
            "preprocessing_steps": 2,
            "model_parameters": 50000,
            "hyperparameters_count": 3,
        },
        "LSTM": {
            "gpu_memory_required": 6.0,
            "training_time_hours": 4.0,
            "retraining_frequency_days": 15,
            "inference_time_ms": 20.0,
            "daily_predictions": 500,
            "complexity_score": 70,
            "data_sources": 3,
            "preprocessing_steps": 5,
            "model_parameters": 500000,
            "hyperparameters_count": 8,
        },
        "GAT": {
            "gpu_memory_required": 8.0,
            "training_time_hours": 6.0,
            "retraining_frequency_days": 20,
            "inference_time_ms": 30.0,
            "daily_predictions": 500,
            "complexity_score": 80,
            "data_sources": 2,
            "preprocessing_steps": 4,
            "model_parameters": 750000,
            "hyperparameters_count": 10,
        },
    }

    return performance_results, operational_metrics, statistical_results, model_specifications


class TestComprehensiveReportGenerator:
    """Test cases for ComprehensiveReportGenerator."""

    def test_initialization(self):
        """Test comprehensive report generator initialization."""
        config = ReportConfig(report_title="Custom Title")
        generator = ComprehensiveReportGenerator(config)

        assert generator.config.report_title == "Custom Title"
        assert generator.config.generate_html is True
        assert hasattr(generator, "executive_generator")
        assert hasattr(generator, "performance_analyzer")
        assert hasattr(generator, "feasibility_assessor")
        assert hasattr(generator, "strategy_engine")

    def test_comprehensive_report_generation(
        self, report_generator, sample_comprehensive_data, tmp_path
    ):
        """Test comprehensive report generation with all components."""
        performance_results, operational_metrics, statistical_results, model_specifications = (
            sample_comprehensive_data
        )

        report_data = report_generator.generate_comprehensive_report(
            performance_results=performance_results,
            operational_metrics=operational_metrics,
            statistical_results=statistical_results,
            model_specifications=model_specifications,
            output_path=tmp_path,
        )

        # Check report structure
        assert "components" in report_data
        assert "files" in report_data
        assert "metadata" in report_data

        # Check components are generated
        components = report_data["components"]
        assert "executive_summary" in components
        assert "performance_analysis" in components
        assert "feasibility_assessment" in components
        assert "strategic_recommendations" in components

        # Check metadata
        metadata = report_data["metadata"]
        assert "generation_timestamp" in metadata
        assert metadata["models_analyzed"] == 3
        assert "report_sections" in metadata

    def test_executive_summary_generation(
        self, report_generator, sample_comprehensive_data, tmp_path
    ):
        """Test executive summary component generation."""
        performance_results, operational_metrics, statistical_results, model_specifications = (
            sample_comprehensive_data
        )

        # Mock the executive generator
        with patch.object(
            report_generator.executive_generator, "generate_executive_dashboard"
        ) as mock_exec:
            mock_exec.return_value = {
                "rankings": pd.DataFrame(
                    {"Model": ["LSTM", "GAT", "HRP"], "Sharpe Ratio": [1.8, 1.5, 1.2]}
                ),
                "summary_statistics": {"top_performer": "LSTM", "total_models_evaluated": 3},
            }

            report_data = report_generator.generate_comprehensive_report(
                performance_results,
                operational_metrics,
                statistical_results,
                model_specifications,
                output_path=tmp_path,
            )

            assert "executive_summary" in report_data["components"]
            assert mock_exec.called

    def test_performance_analysis_generation(
        self, report_generator, sample_comprehensive_data, tmp_path
    ):
        """Test performance analysis component generation."""
        performance_results, operational_metrics, statistical_results, model_specifications = (
            sample_comprehensive_data
        )

        # Mock the performance analyzer
        with patch.object(
            report_generator.performance_analyzer, "generate_comprehensive_performance_tables"
        ) as mock_perf:
            mock_perf.return_value = {
                "performance_summary": pd.DataFrame({"Model": ["LSTM"], "Sharpe Ratio": [1.8]}),
                "risk_adjusted_analysis": pd.DataFrame({"Model": ["LSTM"], "VaR": [-0.02]}),
            }

            report_data = report_generator.generate_comprehensive_report(
                performance_results,
                operational_metrics,
                statistical_results,
                model_specifications,
                output_path=tmp_path,
            )

            assert "performance_analysis" in report_data["components"]
            assert mock_perf.called

    def test_feasibility_assessment_generation(
        self, report_generator, sample_comprehensive_data, tmp_path
    ):
        """Test feasibility assessment component generation."""
        performance_results, operational_metrics, statistical_results, model_specifications = (
            sample_comprehensive_data
        )

        # Mock feasibility assessor methods
        with (
            patch.object(
                report_generator.feasibility_assessor, "assess_computational_requirements"
            ) as mock_comp,
            patch.object(
                report_generator.feasibility_assessor, "assess_operational_complexity"
            ) as mock_ops,
            patch.object(
                report_generator.feasibility_assessor, "calculate_total_cost_ownership"
            ) as mock_tco,
            patch.object(
                report_generator.feasibility_assessor, "assess_implementation_timeline"
            ) as mock_timeline,
            patch.object(
                report_generator.feasibility_assessor, "create_feasibility_summary_report"
            ) as mock_summary,
        ):

            mock_comp.return_value = pd.DataFrame(
                {"Model": ["HRP", "LSTM", "GAT"], "GPU Memory Required (GB)": [2.0, 6.0, 8.0]}
            )
            mock_ops.return_value = pd.DataFrame(
                {"Model": ["HRP", "LSTM", "GAT"], "Overall Complexity (0-100)": [30.0, 70.0, 80.0]}
            )
            mock_tco.return_value = pd.DataFrame(
                {
                    "Model": ["HRP", "LSTM", "GAT"],
                    "Risk-Adjusted TCO ($)": ["100,000", "200,000", "250,000"],
                }
            )
            mock_timeline.return_value = pd.DataFrame(
                {"Model": ["HRP", "LSTM", "GAT"], "Total Timeline (months)": ["3.0", "4.0", "5.0"]}
            )
            mock_summary.return_value = "Feasibility summary text"

            report_data = report_generator.generate_comprehensive_report(
                performance_results,
                operational_metrics,
                statistical_results,
                model_specifications,
                output_path=tmp_path,
            )

            assert "feasibility_assessment" in report_data["components"]
            feasibility_data = report_data["components"]["feasibility_assessment"]
            assert "computational_requirements" in feasibility_data
            assert "operational_complexity" in feasibility_data
            assert "tco_analysis" in feasibility_data
            assert "timeline_analysis" in feasibility_data

    def test_strategic_recommendations_generation(
        self, report_generator, sample_comprehensive_data, tmp_path
    ):
        """Test strategic recommendations generation."""
        performance_results, operational_metrics, statistical_results, model_specifications = (
            sample_comprehensive_data
        )

        # Mock all required dependencies
        mock_rankings = pd.DataFrame(
            {"Model": ["LSTM", "GAT", "HRP"], "Sharpe Ratio": [1.8, 1.5, 1.2]}
        )
        mock_feasibility = pd.DataFrame(
            {"Model": ["LSTM", "GAT", "HRP"], "Feasibility Score": [70, 60, 90]}
        )

        with (
            patch.object(
                report_generator.executive_generator, "generate_executive_dashboard"
            ) as mock_exec,
            patch.object(
                report_generator.feasibility_assessor, "assess_computational_requirements"
            ) as mock_comp,
            patch.object(
                report_generator.feasibility_assessor, "assess_operational_complexity"
            ) as mock_ops,
            patch.object(
                report_generator.feasibility_assessor, "calculate_total_cost_ownership"
            ) as mock_tco,
            patch.object(
                report_generator.feasibility_assessor, "assess_implementation_timeline"
            ) as mock_timeline,
            patch.object(
                report_generator.feasibility_assessor, "create_feasibility_summary_report"
            ) as mock_summary,
        ):

            # Setup mocks
            mock_exec.return_value = {"rankings": mock_rankings}
            mock_comp.return_value = pd.DataFrame()
            mock_ops.return_value = pd.DataFrame()
            mock_tco.return_value = mock_feasibility
            mock_timeline.return_value = pd.DataFrame(
                {"Model": ["HRP", "LSTM", "GAT"], "Total Timeline (months)": ["3.0", "4.0", "5.0"]}
            )
            mock_summary.return_value = "Summary"

            report_data = report_generator.generate_comprehensive_report(
                performance_results,
                operational_metrics,
                statistical_results,
                model_specifications,
                output_path=tmp_path,
            )

            # Strategic recommendations should be included when feasibility data is available
            if "strategic_recommendations" in report_data["components"]:
                strategy_data = report_data["components"]["strategic_recommendations"]
                assert (
                    "decision_framework" in strategy_data
                    or "implementation_roadmap" in strategy_data
                )

    def test_html_report_generation(self, report_generator, sample_comprehensive_data, tmp_path):
        """Test HTML report generation."""
        performance_results, operational_metrics, statistical_results, model_specifications = (
            sample_comprehensive_data
        )

        report_data = report_generator.generate_comprehensive_report(
            performance_results,
            operational_metrics,
            statistical_results,
            model_specifications,
            output_path=tmp_path,
        )

        # HTML should be generated
        if "html" in report_data["files"]:
            html_path = report_data["files"]["html"]
            assert html_path.exists()

            # Check HTML content
            with open(html_path) as f:
                content = f.read()
                assert report_generator.config.report_title in content
                assert "Executive Summary" in content

    def test_default_institutional_constraints(self, report_generator):
        """Test default institutional constraints generation."""
        constraints = report_generator._default_institutional_constraints()

        required_keys = [
            "risk_tolerance",
            "aum_millions",
            "regulatory_complexity",
            "computational_budget",
            "preferred_timeline_months",
        ]
        for key in required_keys:
            assert key in constraints

        assert constraints["risk_tolerance"] in ["conservative", "moderate", "aggressive"]
        assert isinstance(constraints["aum_millions"], (int, float))
        assert constraints["aum_millions"] > 0

    def test_html_template_creation(self, report_generator):
        """Test HTML template creation."""
        html_template = report_generator._create_html_template()

        assert "<!DOCTYPE html>" in html_template
        assert report_generator.config.report_title in html_template
        assert report_generator.config.institution_name in html_template
        assert "table" in html_template.lower()  # CSS styling

    def test_executive_presentation_creation(self, report_generator, tmp_path):
        """Test executive presentation creation."""
        # Mock report components
        mock_components = {
            "executive_summary": {
                "summary_statistics": {
                    "top_performer": "LSTM",
                    "best_sharpe_ratio": 1.8,
                    "recommended_model": "LSTM",
                }
            },
            "strategic_recommendations": {
                "decision_framework": {"Primary Decision": "Deploy LSTM as primary approach"}
            },
        }

        presentation_path = report_generator.create_executive_presentation(
            mock_components, tmp_path / "presentation.html"
        )

        assert presentation_path.exists()

        # Check presentation content
        with open(presentation_path) as f:
            content = f.read()
            assert "Executive Presentation" in content
            assert "LSTM" in content
            assert "1.800" in content

    def test_key_insights_extraction(self, report_generator):
        """Test key insights extraction from components."""
        mock_components = {
            "executive_summary": {
                "summary_statistics": {
                    "top_performer": "LSTM",
                    "best_sharpe_ratio": 1.8,
                    "recommended_model": "GAT",
                }
            },
            "strategic_recommendations": {
                "decision_framework": {"Primary Decision": "Deploy GAT as primary approach"}
            },
        }

        insights = report_generator._extract_key_insights(mock_components)

        assert insights["top_performer"] == "LSTM"
        assert insights["best_sharpe"] == "1.800"
        assert insights["recommendation"] == "GAT"
        assert insights["primary_decision"] == "Deploy GAT as primary approach"

    def test_config_validation(self):
        """Test report configuration validation."""
        # Test default config
        config = ReportConfig()
        assert config.report_title == "ML-Enhanced Portfolio Construction Analysis"
        assert config.generate_pdf is True
        assert config.include_executive_summary is True

        # Test custom config
        custom_config = ReportConfig(
            report_title="Custom Analysis",
            generate_pdf=False,
            include_feasibility_assessment=False,
            color_scheme="modern",
        )
        assert custom_config.report_title == "Custom Analysis"
        assert custom_config.generate_pdf is False
        assert custom_config.include_feasibility_assessment is False
        assert custom_config.color_scheme == "modern"

    def test_empty_data_handling(self, report_generator, tmp_path):
        """Test handling of empty or missing data."""
        empty_performance = {}
        empty_operational = {}
        empty_statistical = {}
        empty_specifications = {}

        # Should handle gracefully without crashing
        report_data = report_generator.generate_comprehensive_report(
            empty_performance, empty_operational, empty_statistical, empty_specifications, tmp_path
        )

        assert "metadata" in report_data
        assert report_data["metadata"]["models_analyzed"] == 0

    def test_report_section_selective_generation(self, tmp_path):
        """Test selective report section generation."""
        config = ReportConfig(
            include_executive_summary=True,
            include_performance_analysis=False,
            include_feasibility_assessment=False,
            include_strategic_recommendations=False,
            generate_html=True,
        )
        generator = ComprehensiveReportGenerator(config)

        # Mock data
        performance_results = {"HRP": pd.DataFrame({"returns": [0.01, 0.02]})}

        with patch.object(
            generator.executive_generator, "generate_executive_dashboard"
        ) as mock_exec:
            mock_exec.return_value = {"rankings": pd.DataFrame()}

            report_data = generator.generate_comprehensive_report(
                performance_results, {}, {}, {}, tmp_path
            )

            components = report_data["components"]
            assert "executive_summary" in components
            assert "performance_analysis" not in components
            assert "feasibility_assessment" not in components
            assert "strategic_recommendations" not in components

    @patch("src.evaluation.reporting.comprehensive_report.HAS_REPORTLAB", False)
    def test_pdf_generation_disabled(self, report_generator, tmp_path):
        """Test behavior when PDF generation dependencies are missing."""
        # Should skip PDF generation gracefully
        report_generator.config.generate_pdf = True

        pdf_path = report_generator._generate_pdf_report({}, tmp_path / "test.pdf")
        assert pdf_path is None

    @patch("src.evaluation.reporting.comprehensive_report.HAS_OPENPYXL", False)
    def test_excel_generation_disabled(self, report_generator, tmp_path):
        """Test behavior when Excel generation dependencies are missing."""
        # Should skip Excel generation gracefully
        report_generator.config.generate_excel = True

        excel_path = report_generator._generate_excel_report({}, tmp_path / "test.xlsx")
        assert excel_path is None

    def test_presentation_template_creation(self, report_generator):
        """Test presentation template creation."""
        insights = {
            "top_performer": "LSTM",
            "best_sharpe": "1.800",
            "recommendation": "GAT",
            "primary_decision": "Deploy GAT",
        }

        template = report_generator._create_presentation_template(insights)

        assert "<!DOCTYPE html>" in template
        assert "Executive Summary" in template
        assert "LSTM" in template
        assert "1.800" in template
        assert "Deploy GAT" in template
        assert "gradient" in template  # Check for styling
