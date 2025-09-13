"""
Tests for HRP clustering analysis.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.interpretability.hrp_analysis import (
    HRPAnalysisConfig,
    HRPAnalyzer,
)
from src.models.base.constraints import PortfolioConstraints
from src.models.hrp.clustering import ClusteringConfig, HRPClustering
from src.models.hrp.model import HRPConfig, HRPModel


class TestHRPAnalyzer:
    """Test suite for HRP analysis functionality."""

    @pytest.fixture
    def sample_hrp_model(self):
        """Create sample fitted HRP model for testing."""
        constraints = PortfolioConstraints()
        hrp_config = HRPConfig(
            lookback_days=100,
            clustering_config=ClusteringConfig(
                linkage_method="single",
                min_observations=50,
            ),
        )
        model = HRPModel(constraints, hrp_config)
        model.is_fitted = True  # Mock fitted state

        return model

    @pytest.fixture
    def analysis_config(self):
        """Create HRP analysis configuration."""
        return HRPAnalysisConfig(
            dendrogram_levels=5,
            cluster_threshold=0.4,
            correlation_threshold=0.3,
        )

    @pytest.fixture
    def sample_returns_data(self):
        """Create sample returns data for testing."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        assets = ["AAPL", "MSFT", "GOOGL", "TSLA", "JPM"]

        # Generate correlated returns
        np.random.seed(42)
        base_returns = np.random.normal(0.001, 0.02, (100, 5))

        # Add some correlation structure
        base_returns[:, 1] += 0.3 * base_returns[:, 0]  # MSFT correlated with AAPL
        base_returns[:, 2] += 0.2 * base_returns[:, 0]  # GOOGL weakly correlated

        return pd.DataFrame(base_returns, index=dates, columns=assets)

    def test_hrp_analyzer_initialization(self, sample_hrp_model, analysis_config):
        """Test HRP analyzer initialization."""
        analyzer = HRPAnalyzer(sample_hrp_model, analysis_config)

        assert analyzer.model is sample_hrp_model
        assert analyzer.config is analysis_config

    def test_initialization_with_unfitted_model(self):
        """Test that initialization fails with unfitted model."""
        constraints = PortfolioConstraints()
        unfitted_model = HRPModel(constraints)

        with pytest.raises(ValueError, match="HRP model must be fitted"):
            HRPAnalyzer(unfitted_model)

    def test_clustering_structure_analysis(
        self, sample_hrp_model, sample_returns_data, analysis_config
    ):
        """Test clustering structure analysis."""
        analyzer = HRPAnalyzer(sample_hrp_model, analysis_config)
        universe = list(sample_returns_data.columns)

        # Analyze clustering structure
        results = analyzer.analyze_clustering_structure(sample_returns_data, universe)

        # Check required keys
        required_keys = [
            "distance_matrix",
            "linkage_matrix",
            "dendrogram_data",
            "cluster_labels",
            "cluster_analysis",
            "quality_metrics",
            "asset_names",
        ]
        for key in required_keys:
            assert key in results

        # Check data types and shapes
        assert isinstance(results["distance_matrix"], np.ndarray)
        assert results["distance_matrix"].shape == (len(universe), len(universe))

        assert isinstance(results["linkage_matrix"], np.ndarray)
        assert results["linkage_matrix"].shape[1] == 4  # Standard linkage format

        assert isinstance(results["cluster_labels"], np.ndarray)
        assert len(results["cluster_labels"]) == len(universe)

        assert isinstance(results["cluster_analysis"], dict)
        assert isinstance(results["quality_metrics"], dict)

    def test_correlation_pattern_analysis(
        self, sample_hrp_model, sample_returns_data, analysis_config
    ):
        """Test correlation pattern analysis."""
        analyzer = HRPAnalyzer(sample_hrp_model, analysis_config)
        universe = list(sample_returns_data.columns)

        # Analyze correlation patterns
        results = analyzer.analyze_correlation_patterns(sample_returns_data, universe)

        # Check required keys
        required_keys = [
            "correlation_matrix",
            "high_correlation_pairs",
            "correlation_blocks",
            "correlation_statistics",
        ]
        for key in required_keys:
            assert key in results

        # Check correlation matrix
        corr_matrix = results["correlation_matrix"]
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape == (len(universe), len(universe))
        assert (np.diag(corr_matrix.values) == 1.0).all()  # Diagonal should be 1

        # Check high correlation pairs
        high_corr = results["high_correlation_pairs"]
        assert isinstance(high_corr, list)
        for pair in high_corr:
            assert "asset1" in pair and "asset2" in pair
            assert "correlation" in pair and "abs_correlation" in pair

        # Check correlation statistics
        stats = results["correlation_statistics"]
        assert isinstance(stats, dict)
        assert all(key in stats for key in [
            "mean_correlation", "std_correlation", "min_correlation", "max_correlation"
        ])

    def test_allocation_logic_analysis(
        self, sample_hrp_model, sample_returns_data, analysis_config
    ):
        """Test allocation logic analysis."""
        analyzer = HRPAnalyzer(sample_hrp_model, analysis_config)
        universe = list(sample_returns_data.columns)
        target_date = sample_returns_data.index[-1]

        # Mock the predict_weights method
        sample_weights = pd.Series({
            asset: 1.0 / len(universe) + np.random.normal(0, 0.1)
            for asset in universe
        })
        sample_weights = sample_weights.abs()
        sample_weights = sample_weights / sample_weights.sum()

        # Mock the model's predict_weights method
        def mock_predict_weights(date, universe_subset):
            return sample_weights[universe_subset]

        sample_hrp_model.predict_weights = mock_predict_weights

        # Analyze allocation logic
        results = analyzer.analyze_allocation_logic(
            sample_returns_data, universe, target_date
        )

        # Check required keys
        required_keys = [
            "portfolio_weights",
            "allocation_tree",
            "allocation_analysis",
            "baseline_comparison",
            "clustering_rationale",
        ]
        for key in required_keys:
            assert key in results

        # Check portfolio weights
        weights = results["portfolio_weights"]
        assert isinstance(weights, pd.Series)
        assert len(weights) == len(universe)
        assert abs(weights.sum() - 1.0) < 1e-6  # Should sum to 1

        # Check allocation analysis
        alloc_analysis = results["allocation_analysis"]
        assert isinstance(alloc_analysis, dict)
        assert "herfindahl_hirschman_index" in alloc_analysis
        assert "effective_n_stocks" in alloc_analysis

        # Check baseline comparison
        baseline_comp = results["baseline_comparison"]
        assert isinstance(baseline_comp, dict)
        assert "active_share" in baseline_comp
        assert "equal_weight_baseline" in baseline_comp

    def test_sector_alignment_analysis(
        self, sample_hrp_model, analysis_config
    ):
        """Test sector alignment analysis."""
        analyzer = HRPAnalyzer(sample_hrp_model, analysis_config)

        universe = ["AAPL", "MSFT", "GOOGL", "TSLA", "JPM"]
        cluster_labels = np.array([0, 0, 0, 1, 2])  # Mock cluster assignments

        # Analyze sector alignment
        results = analyzer.analyze_sector_alignment(universe, cluster_labels)

        # Check required keys
        required_keys = [
            "sector_mapping",
            "alignment_analysis",
            "alignment_metrics",
        ]
        for key in required_keys:
            assert key in results

        # Check sector mapping
        sector_mapping = results["sector_mapping"]
        assert isinstance(sector_mapping, dict)
        assert all(asset in sector_mapping for asset in universe)

        # Check alignment analysis
        alignment_analysis = results["alignment_analysis"]
        assert isinstance(alignment_analysis, dict)

        # Check alignment metrics
        alignment_metrics = results["alignment_metrics"]
        assert isinstance(alignment_metrics, dict)
        assert "alignment_score" in alignment_metrics
        assert 0 <= alignment_metrics["alignment_score"] <= 1

    def test_high_correlation_pairs_identification(
        self, sample_hrp_model, analysis_config
    ):
        """Test identification of high correlation pairs."""
        analyzer = HRPAnalyzer(sample_hrp_model, analysis_config)

        # Create sample correlation matrix
        assets = ["A", "B", "C", "D"]
        corr_data = np.array([
            [1.0, 0.8, 0.2, -0.1],
            [0.8, 1.0, 0.1, 0.0],
            [0.2, 0.1, 1.0, 0.7],
            [-0.1, 0.0, 0.7, 1.0],
        ])
        correlation_matrix = pd.DataFrame(corr_data, index=assets, columns=assets)

        # Find high correlation pairs
        high_corr_pairs = analyzer._find_high_correlation_pairs(
            correlation_matrix, threshold=0.5
        )

        # Should find pairs (A,B) with 0.8 and (C,D) with 0.7
        assert len(high_corr_pairs) == 2

        # Check that pairs are sorted by absolute correlation
        correlations = [pair["abs_correlation"] for pair in high_corr_pairs]
        assert correlations == sorted(correlations, reverse=True)

        # Verify specific high correlations
        pair_correlations = {
            (pair["asset1"], pair["asset2"]): pair["correlation"]
            for pair in high_corr_pairs
        }
        assert abs(pair_correlations.get(("A", "B"), 0) - 0.8) < 1e-6
        assert abs(pair_correlations.get(("C", "D"), 0) - 0.7) < 1e-6

    def test_correlation_statistics_calculation(
        self, sample_hrp_model, analysis_config
    ):
        """Test correlation statistics calculation."""
        analyzer = HRPAnalyzer(sample_hrp_model, analysis_config)

        # Create test correlation matrix
        assets = ["A", "B", "C"]
        corr_data = np.array([
            [1.0, 0.5, -0.3],
            [0.5, 1.0, 0.2],
            [-0.3, 0.2, 1.0],
        ])
        correlation_matrix = pd.DataFrame(corr_data, index=assets, columns=assets)

        # Calculate statistics
        stats = analyzer._calculate_correlation_statistics(correlation_matrix)

        # Check that all expected statistics are present
        expected_keys = [
            "mean_correlation", "std_correlation", "median_correlation",
            "min_correlation", "max_correlation", "mean_abs_correlation"
        ]
        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], float)

        # Verify some calculations
        off_diag_values = [0.5, -0.3, 0.2]
        assert abs(stats["mean_correlation"] - np.mean(off_diag_values)) < 1e-6
        assert abs(stats["max_correlation"] - 0.5) < 1e-6
        assert abs(stats["min_correlation"] - (-0.3)) < 1e-6

    def test_allocation_concentration_analysis(
        self, sample_hrp_model, analysis_config
    ):
        """Test allocation concentration analysis."""
        analyzer = HRPAnalyzer(sample_hrp_model, analysis_config)

        # Test with equal weights (low concentration)
        equal_weights = pd.Series([0.25, 0.25, 0.25, 0.25], index=["A", "B", "C", "D"])
        equal_analysis = analyzer._analyze_allocation_concentration(equal_weights)

        # Test with concentrated weights (high concentration)
        concentrated_weights = pd.Series([0.7, 0.1, 0.1, 0.1], index=["A", "B", "C", "D"])
        concentrated_analysis = analyzer._analyze_allocation_concentration(concentrated_weights)

        # Check required metrics
        for analysis in [equal_analysis, concentrated_analysis]:
            assert "herfindahl_hirschman_index" in analysis
            assert "effective_n_stocks" in analysis
            assert "gini_coefficient" in analysis
            assert "top_5_concentration" in analysis

        # Concentrated portfolio should have higher HHI and lower effective N stocks
        assert concentrated_analysis["herfindahl_hirschman_index"] > equal_analysis["herfindahl_hirschman_index"]
        assert concentrated_analysis["effective_n_stocks"] < equal_analysis["effective_n_stocks"]
        assert concentrated_analysis["gini_coefficient"] > equal_analysis["gini_coefficient"]

    def test_baseline_comparison(
        self, sample_hrp_model, analysis_config
    ):
        """Test baseline comparison functionality."""
        analyzer = HRPAnalyzer(sample_hrp_model, analysis_config)

        # Test with equal weights (should match baseline exactly)
        n_assets = 4
        equal_weights = pd.Series([1.0/n_assets] * n_assets, index=["A", "B", "C", "D"])
        equal_comparison = analyzer._compare_to_baseline(equal_weights)

        assert equal_comparison["active_share"] == 0.0
        assert equal_comparison["equal_weight_baseline"] == 0.25
        assert equal_comparison["n_overweight"] == 0
        assert equal_comparison["n_underweight"] == 0

        # Test with tilted weights
        tilted_weights = pd.Series([0.4, 0.3, 0.2, 0.1], index=["A", "B", "C", "D"])
        tilted_comparison = analyzer._compare_to_baseline(tilted_weights)

        assert tilted_comparison["active_share"] > 0
        assert tilted_comparison["n_overweight"] > 0
        assert tilted_comparison["n_underweight"] > 0
        assert tilted_comparison["max_overweight"] > 0
        assert tilted_comparison["max_underweight"] > 0

    def test_edge_cases(self, sample_hrp_model):
        """Test edge cases and error handling."""
        analyzer = HRPAnalyzer(sample_hrp_model)

        # Test with minimal data
        minimal_returns = pd.DataFrame({
            "A": [0.01, 0.02],
            "B": [-0.01, 0.03],
        })

        # Should handle small datasets gracefully
        try:
            results = analyzer.analyze_correlation_patterns(minimal_returns, ["A", "B"])
            assert isinstance(results, dict)
        except ValueError:
            # Acceptable if insufficient data
            pass

        # Test empty universe
        with pytest.raises((ValueError, IndexError)):
            analyzer.analyze_clustering_structure(minimal_returns, [])

    def test_configuration_effects(self, sample_hrp_model, sample_returns_data):
        """Test that configuration parameters affect results."""
        universe = list(sample_returns_data.columns)

        # Test different correlation thresholds
        low_threshold_config = HRPAnalysisConfig(correlation_threshold=0.1)
        high_threshold_config = HRPAnalysisConfig(correlation_threshold=0.8)

        analyzer_low = HRPAnalyzer(sample_hrp_model, low_threshold_config)
        analyzer_high = HRPAnalyzer(sample_hrp_model, high_threshold_config)

        results_low = analyzer_low.analyze_correlation_patterns(sample_returns_data, universe)
        results_high = analyzer_high.analyze_correlation_patterns(sample_returns_data, universe)

        # Higher threshold should result in fewer high correlation pairs
        assert len(results_high["high_correlation_pairs"]) <= len(results_low["high_correlation_pairs"])
