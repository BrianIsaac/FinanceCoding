"""
Tests for executive summary report generation.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.evaluation.reporting.executive import ExecutiveConfig, ExecutiveSummaryGenerator


@pytest.fixture
def executive_generator():
    """Create executive summary generator for testing."""
    config = ExecutiveConfig(
        figsize=(10, 6), significance_threshold=0.05, top_n_models=3, decimal_places=4
    )
    return ExecutiveSummaryGenerator(config)


@pytest.fixture
def sample_performance_results():
    """Sample performance results for testing."""
    return {
        "HRP": pd.DataFrame(
            {
                "sharpe_ratio": [1.2, 1.3, 1.1, 1.4],
                "information_ratio": [0.8, 0.9, 0.7, 1.0],
                "annual_return": [0.12, 0.14, 0.11, 0.15],
                "max_drawdown": [-0.08, -0.06, -0.09, -0.07],
                "volatility": [0.15, 0.14, 0.16, 0.13],
            }
        ),
        "LSTM": pd.DataFrame(
            {
                "sharpe_ratio": [1.8, 1.7, 1.9, 1.6],
                "information_ratio": [1.2, 1.1, 1.3, 1.0],
                "annual_return": [0.18, 0.17, 0.19, 0.16],
                "max_drawdown": [-0.12, -0.11, -0.13, -0.10],
                "volatility": [0.18, 0.17, 0.19, 0.16],
            }
        ),
        "GAT": pd.DataFrame(
            {
                "sharpe_ratio": [1.5, 1.4, 1.6, 1.3],
                "information_ratio": [1.0, 0.9, 1.1, 0.8],
                "annual_return": [0.15, 0.14, 0.16, 0.13],
                "max_drawdown": [-0.10, -0.09, -0.11, -0.08],
                "volatility": [0.16, 0.15, 0.17, 0.14],
            }
        ),
    }


@pytest.fixture
def sample_operational_metrics():
    """Sample operational metrics for testing."""
    return {
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


@pytest.fixture
def sample_statistical_results():
    """Sample statistical test results for testing."""
    return {
        "HRP": {
            "pvalue_vs_baseline": 0.12,
            "pvalue_vs_equal_weight": 0.08,
            "bootstrap_pvalue": 0.09,
            "effect_size": 0.3,
        },
        "LSTM": {
            "pvalue_vs_baseline": 0.02,
            "pvalue_vs_equal_weight": 0.01,
            "bootstrap_pvalue": 0.015,
            "effect_size": 0.8,
        },
        "GAT": {
            "pvalue_vs_baseline": 0.06,
            "pvalue_vs_equal_weight": 0.04,
            "bootstrap_pvalue": 0.05,
            "effect_size": 0.5,
        },
    }


class TestExecutiveSummaryGenerator:
    """Test cases for ExecutiveSummaryGenerator."""

    def test_initialization(self):
        """Test executive generator initialization."""
        config = ExecutiveConfig(top_n_models=5)
        generator = ExecutiveSummaryGenerator(config)

        assert generator.config.top_n_models == 5
        assert generator.config.significance_threshold == 0.05
        assert generator.config.decimal_places == 3

    def test_performance_rankings_calculation(
        self, executive_generator, sample_performance_results, sample_statistical_results
    ):
        """Test performance rankings calculation."""
        rankings = executive_generator._calculate_performance_rankings(
            sample_performance_results, sample_statistical_results
        )

        assert not rankings.empty
        assert len(rankings) == 3  # Three models
        assert "Model" in rankings.columns
        assert "Sharpe Ratio" in rankings.columns
        assert "Overall Rank" in rankings.columns
        assert "Statistically Significant" in rankings.columns

        # Check that LSTM (highest Sharpe) is ranked first
        top_model = rankings.iloc[0]["Model"]
        assert top_model == "LSTM"

        # Check statistical significance marking
        lstm_significant = rankings[rankings["Model"] == "LSTM"]["Statistically Significant"].iloc[
            0
        ]
        assert lstm_significant  # p-value < 0.05

    def test_feasibility_scores_calculation(self, executive_generator, sample_operational_metrics):
        """Test feasibility scores calculation."""
        feasibility_scores = executive_generator._calculate_feasibility_scores(
            sample_operational_metrics
        )

        assert not feasibility_scores.empty
        assert len(feasibility_scores) == 3
        assert "Model" in feasibility_scores.columns
        assert "Feasibility Score" in feasibility_scores.columns
        assert "Feasibility Rank" in feasibility_scores.columns

        # HRP should have highest feasibility (lowest resource requirements)
        hrp_score = feasibility_scores[feasibility_scores["Model"] == "HRP"][
            "Feasibility Score"
        ].iloc[0]
        gat_score = feasibility_scores[feasibility_scores["Model"] == "GAT"][
            "Feasibility Score"
        ].iloc[0]
        assert hrp_score > gat_score

    @patch("src.evaluation.reporting.executive.HAS_PLOTLY", True)
    def test_executive_dashboard_generation(
        self,
        executive_generator,
        sample_performance_results,
        sample_operational_metrics,
        sample_statistical_results,
    ):
        """Test executive dashboard generation."""
        dashboard_data = executive_generator.generate_executive_dashboard(
            sample_performance_results, sample_operational_metrics, sample_statistical_results
        )

        assert "rankings" in dashboard_data
        assert "feasibility_scores" in dashboard_data
        assert "summary_statistics" in dashboard_data

        # Check summary statistics
        summary_stats = dashboard_data["summary_statistics"]
        assert "total_models_evaluated" in summary_stats
        assert "top_performer" in summary_stats
        assert "recommended_model" in summary_stats
        assert summary_stats["total_models_evaluated"] == 3
        assert summary_stats["top_performer"] == "LSTM"

    def test_recommendation_matrix_generation(
        self, executive_generator, sample_performance_results, sample_operational_metrics
    ):
        """Test recommendation matrix generation."""
        # First calculate rankings and feasibility
        rankings = executive_generator._calculate_performance_rankings(
            sample_performance_results, {}
        )
        feasibility_scores = executive_generator._calculate_feasibility_scores(
            sample_operational_metrics
        )

        recommendation_matrix = executive_generator.generate_recommendation_matrix(
            rankings, feasibility_scores
        )

        assert not recommendation_matrix.empty
        assert "Model" in recommendation_matrix.columns
        assert "Use Case" in recommendation_matrix.columns
        assert "Suitability Score" in recommendation_matrix.columns

        # Check that we have multiple use cases
        use_cases = recommendation_matrix["Use Case"].unique()
        assert len(use_cases) >= 4

    def test_management_summary_creation(
        self,
        executive_generator,
        sample_performance_results,
        sample_operational_metrics,
        sample_statistical_results,
    ):
        """Test management summary creation."""
        rankings = executive_generator._calculate_performance_rankings(
            sample_performance_results, sample_statistical_results
        )
        feasibility_scores = executive_generator._calculate_feasibility_scores(
            sample_operational_metrics
        )

        summary_text = executive_generator.create_management_summary(
            rankings, feasibility_scores, sample_statistical_results
        )

        assert "EXECUTIVE SUMMARY" in summary_text
        assert "PERFORMANCE OVERVIEW" in summary_text
        assert "STRATEGIC RECOMMENDATION" in summary_text
        assert "NEXT STEPS" in summary_text
        assert "LSTM" in summary_text  # Top performer

    def test_statistical_significance_checking(
        self, executive_generator, sample_statistical_results
    ):
        """Test statistical significance checking."""
        # LSTM should be significant (p < 0.05)
        lstm_significant = executive_generator._check_statistical_significance(
            "LSTM", sample_statistical_results
        )
        assert lstm_significant is True

        # HRP should not be significant (p > 0.05)
        hrp_significant = executive_generator._check_statistical_significance(
            "HRP", sample_statistical_results
        )
        assert hrp_significant is False

    def test_suitability_scoring_methods(self, executive_generator):
        """Test different suitability scoring methods."""
        model_data = pd.Series(
            {
                "Sharpe Ratio": 1.5,
                "CAGR (%)": 15.0,
                "Max Drawdown (%)": 10.0,
                "Feasibility Score": 75.0,
                "Statistically Significant": True,
            }
        )

        # Conservative scoring
        conservative_score = executive_generator._score_conservative_suitability(model_data)
        assert 0 <= conservative_score <= 100

        # Aggressive scoring
        aggressive_score = executive_generator._score_aggressive_suitability(model_data)
        assert 0 <= aggressive_score <= 100

        # Aggressive should score higher for high-performance models
        high_perf_data = model_data.copy()
        high_perf_data["Sharpe Ratio"] = 2.5
        high_perf_data["CAGR (%)"] = 25.0

        aggressive_high = executive_generator._score_aggressive_suitability(high_perf_data)
        assert aggressive_high > aggressive_score

    def test_primary_strength_identification(self, executive_generator):
        """Test primary strength identification logic."""
        # High Sharpe ratio model
        high_sharpe_data = pd.Series(
            {
                "Sharpe Ratio": 2.1,
                "CAGR (%)": 15.0,
                "Max Drawdown (%)": 12.0,
                "Feasibility Score": 60.0,
            }
        )
        strength = executive_generator._identify_primary_strength(high_sharpe_data)
        assert "Exceptional Risk-Adjusted Returns" in strength

        # High return model
        high_return_data = pd.Series(
            {
                "Sharpe Ratio": 1.5,
                "CAGR (%)": 26.0,
                "Max Drawdown (%)": 15.0,
                "Feasibility Score": 60.0,
            }
        )
        strength = executive_generator._identify_primary_strength(high_return_data)
        assert "High Absolute Returns" in strength

        # Low drawdown model
        low_dd_data = pd.Series(
            {
                "Sharpe Ratio": 1.3,
                "CAGR (%)": 12.0,
                "Max Drawdown (%)": 8.0,
                "Feasibility Score": 60.0,
            }
        )
        strength = executive_generator._identify_primary_strength(low_dd_data)
        assert "Low Downside Risk" in strength

    def test_key_consideration_identification(self, executive_generator):
        """Test key consideration identification logic."""
        # Low feasibility model
        low_feasibility_data = pd.Series(
            {
                "Feasibility Score": 40.0,
                "Max Drawdown (%)": 10.0,
                "Statistically Significant": True,
                "Volatility (%)": 15.0,
            }
        )
        consideration = executive_generator._identify_key_consideration(low_feasibility_data)
        assert "High Implementation Complexity" in consideration

        # High drawdown model
        high_dd_data = pd.Series(
            {
                "Feasibility Score": 70.0,
                "Max Drawdown (%)": 25.0,
                "Statistically Significant": True,
                "Volatility (%)": 15.0,
            }
        )
        consideration = executive_generator._identify_key_consideration(high_dd_data)
        assert "Significant Drawdown Risk" in consideration

    def test_empty_data_handling(self, executive_generator):
        """Test handling of empty data inputs."""
        empty_performance = {}
        empty_operational = {}
        empty_statistical = {}

        dashboard_data = executive_generator.generate_executive_dashboard(
            empty_performance, empty_operational, empty_statistical
        )

        # Should handle gracefully
        assert "rankings" in dashboard_data
        assert dashboard_data["rankings"].empty

    def test_config_validation(self):
        """Test configuration validation."""
        # Test default config
        config = ExecutiveConfig()
        assert config.significance_threshold == 0.05
        assert config.top_n_models == 5
        assert config.decimal_places == 3

        # Test custom config
        custom_config = ExecutiveConfig(
            significance_threshold=0.01, top_n_models=10, decimal_places=2
        )
        assert custom_config.significance_threshold == 0.01
        assert custom_config.top_n_models == 10
        assert custom_config.decimal_places == 2

    def test_dashboard_saving(
        self,
        executive_generator,
        sample_performance_results,
        sample_operational_metrics,
        sample_statistical_results,
        tmp_path,
    ):
        """Test dashboard saving functionality."""
        output_dir = tmp_path / "test_output"

        executive_generator.generate_executive_dashboard(
            sample_performance_results,
            sample_operational_metrics,
            sample_statistical_results,
            output_dir,
        )

        # Check that files are created
        assert (output_dir / "executive_rankings.csv").exists()
        assert (output_dir / "feasibility_scores.csv").exists()
        assert (output_dir / "executive_summary_stats.json").exists()
