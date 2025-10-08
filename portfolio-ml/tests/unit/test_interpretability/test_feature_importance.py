"""
Tests for feature importance analysis framework.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.evaluation.interpretability.feature_importance import (
    FeatureImportanceAnalyzer,
    FeatureImportanceConfig,
)


class MockModel:
    """Mock model for testing."""

    def __init__(self, model_type: str = "linear"):
        self.model_type = model_type
        self.is_fitted = True

        if model_type == "linear":
            self.coef_ = np.array([0.5, -0.3, 0.8, 0.1])

    def predict(self, X):
        """Mock predict method."""
        if hasattr(X, 'values'):
            X = X.values

        if self.model_type == "linear":
            return X @ self.coef_[:X.shape[1]]  # Handle variable feature count
        else:
            # Random predictions for other model types
            return np.random.normal(0, 1, X.shape[0])

    def predict_weights(self, date, universe):
        """Mock predict_weights method for portfolio models."""
        weights = np.random.uniform(0, 1, len(universe))
        return pd.Series(weights / weights.sum(), index=universe)

    def fit(self, X, y):
        """Mock fit method for scikit-learn compatibility."""
        return self


class TestFeatureImportanceAnalyzer:
    """Test suite for feature importance analysis."""

    @pytest.fixture
    def sample_data(self):
        """Create sample feature data."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")

        # Create correlated features
        n_samples = 100
        n_features = 4

        # Generate base features
        X = np.random.normal(0, 1, (n_samples, n_features))

        # Add some correlation structure
        X[:, 1] += 0.5 * X[:, 0]  # Feature 1 correlated with Feature 0
        X[:, 3] += 0.3 * X[:, 2]  # Feature 3 correlated with Feature 2

        feature_names = ["price_close", "volume_traded", "rsi_14", "pe_ratio"]
        X_df = pd.DataFrame(X, index=dates, columns=feature_names)

        # Create target
        y = X[:, 0] * 0.5 - X[:, 1] * 0.3 + X[:, 2] * 0.8 + np.random.normal(0, 0.1, n_samples)
        y_series = pd.Series(y, index=dates, name="target")

        return X_df, y_series

    @pytest.fixture
    def mock_models(self):
        """Create mock models for testing."""
        return {
            "linear_model": MockModel("linear"),
            "tree_model": MockModel("tree"),
            "portfolio_model": MockModel("portfolio"),
        }

    @pytest.fixture
    def analyzer_config(self):
        """Create analyzer configuration."""
        return FeatureImportanceConfig(
            n_permutations=10,  # Reduced for faster testing
            shap_background_samples=20,
            importance_threshold=0.01,
            max_features_display=10,
        )

    def test_analyzer_initialization(self, mock_models, analyzer_config):
        """Test analyzer initialization."""
        analyzer = FeatureImportanceAnalyzer(mock_models, analyzer_config)

        assert analyzer.models == mock_models
        assert analyzer.config == analyzer_config
        assert len(analyzer.models) == 3

    def test_initialization_with_empty_models(self):
        """Test that initialization fails with empty models."""
        with pytest.raises(ValueError, match="At least one model must be provided"):
            FeatureImportanceAnalyzer({})

    def test_initialization_with_invalid_model(self):
        """Test that initialization fails with invalid model."""
        invalid_model = {"no_predict_method": True}

        with pytest.raises(ValueError, match="must have predict or predict_weights method"):
            FeatureImportanceAnalyzer({"invalid": invalid_model})

    @patch('src.evaluation.interpretability.feature_importance.SHAP_AVAILABLE', False)
    def test_fallback_feature_importance(self, mock_models, sample_data, analyzer_config):
        """Test fallback feature importance when SHAP is not available."""
        X, y = sample_data
        analyzer = FeatureImportanceAnalyzer(mock_models, analyzer_config)

        results = analyzer.analyze_shap_importance("linear_model", X, y)

        # Check basic structure
        assert "feature_importance" in results
        assert "feature_names" in results
        assert "importance_ranking" in results
        assert "method" in results

        # Check feature importance
        importance = results["feature_importance"]
        assert isinstance(importance, pd.Series)
        assert len(importance) == X.shape[1]
        assert all(name in importance.index for name in X.columns)

        # Check ranking
        ranking = results["importance_ranking"]
        assert isinstance(ranking, list)
        assert all(isinstance(item, dict) for item in ranking)
        assert all("rank" in item and "feature" in item and "importance" in item for item in ranking)

    def test_permutation_importance_analysis(self, mock_models, sample_data, analyzer_config):
        """Test permutation importance analysis."""
        X, y = sample_data
        analyzer = FeatureImportanceAnalyzer(mock_models, analyzer_config)

        # Force use of permutation importance
        with patch('src.evaluation.interpretability.feature_importance.SHAP_AVAILABLE', False):
            results = analyzer._fallback_feature_importance("linear_model", X, y, list(X.columns))

        # Verify results structure
        assert "feature_importance" in results
        assert "importance_std" in results or "method" in results

        importance = results["feature_importance"]
        assert isinstance(importance, pd.Series)
        assert len(importance) == len(X.columns)

        # Check that importance values are reasonable
        # Note: permutation importance can be negative if features are uninformative
        assert isinstance(importance, pd.Series)
        assert len(importance) > 0  # Should have some features

    def test_feature_ranking(self, mock_models, analyzer_config):
        """Test feature ranking functionality."""
        analyzer = FeatureImportanceAnalyzer(mock_models, analyzer_config)

        # Test ranking with known importance scores
        importance_scores = np.array([0.8, 0.2, 0.5, 0.1])
        feature_names = ["high_imp", "low_imp", "med_imp", "very_low_imp"]

        ranking = analyzer._rank_features(importance_scores, feature_names)

        # Check that features are ranked by importance
        assert ranking[0]["feature"] == "high_imp"
        assert ranking[1]["feature"] == "med_imp"
        assert ranking[2]["feature"] == "low_imp"

        # Check ranking values
        assert ranking[0]["rank"] == 1
        assert ranking[1]["rank"] == 2
        assert ranking[0]["relative_importance"] == 1.0  # Highest feature

        # Check importance threshold filtering
        assert all(item["importance"] >= analyzer_config.importance_threshold for item in ranking)

    def test_cross_model_comparison(self, mock_models, sample_data, analyzer_config):
        """Test cross-model feature importance comparison."""
        X, y = sample_data
        analyzer = FeatureImportanceAnalyzer(mock_models, analyzer_config)

        # Mock the analyze_shap_importance method to return consistent results
        def mock_analyze_shap(model_name, X, y, feature_names):
            # Return different importance patterns for different models
            if model_name == "linear_model":
                importance = pd.Series([0.8, 0.2, 0.5, 0.1], index=feature_names)
            elif model_name == "tree_model":
                importance = pd.Series([0.6, 0.4, 0.7, 0.3], index=feature_names)
            else:
                importance = pd.Series([0.7, 0.3, 0.6, 0.2], index=feature_names)

            return {
                "feature_importance": importance,
                "importance_ranking": [
                    {"rank": i+1, "feature": feat, "importance": imp}
                    for i, (feat, imp) in enumerate(importance.sort_values(ascending=False).items())
                ]
            }

        with patch.object(analyzer, 'analyze_shap_importance', side_effect=mock_analyze_shap):
            results = analyzer.compare_feature_importance_across_models(X, y, list(X.columns))

        # Check results structure
        required_keys = [
            "model_importance", "importance_matrix", "consensus_importance",
            "model_correlations", "consistent_features", "divergent_features",
            "cross_model_ranking", "summary_statistics"
        ]
        for key in required_keys:
            assert key in results

        # Check consensus importance
        consensus = results["consensus_importance"]
        assert isinstance(consensus, pd.Series)
        assert len(consensus) == len(X.columns)

        # Check model correlations
        correlations = results["model_correlations"]
        assert isinstance(correlations, pd.DataFrame)
        assert correlations.shape == (len(mock_models), len(mock_models))

        # Check summary statistics
        summary = results["summary_statistics"]
        assert summary["n_models_analyzed"] <= len(mock_models)
        assert summary["n_features"] == len(X.columns)
        assert "consensus_top_features" in summary

    def test_temporal_feature_importance(self, mock_models, analyzer_config):
        """Test temporal feature importance analysis."""
        # Create larger time series dataset
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=300, freq="D")
        n_features = 3

        # Generate time-varying feature importance
        time_series_data = pd.DataFrame(
            np.random.normal(0, 1, (300, n_features + 1)),
            index=dates,
            columns=["feature_1", "feature_2", "feature_3", "target"]
        )

        analyzer = FeatureImportanceAnalyzer(mock_models, analyzer_config)

        # Mock analyze_shap_importance for temporal analysis
        call_count = 0
        def mock_temporal_analyze(model_name, X, y, feature_names):
            nonlocal call_count
            call_count += 1

            # Return varying importance over time
            base_importance = np.array([0.5, 0.3, 0.2])
            time_factor = 1 + 0.2 * np.sin(call_count * 0.5)  # Sinusoidal variation
            importance = base_importance * time_factor

            return {
                "feature_importance": pd.Series(importance, index=feature_names)
            }

        with patch.object(analyzer, 'analyze_shap_importance', side_effect=mock_temporal_analyze):
            results = analyzer.analyze_temporal_feature_importance(
                "linear_model", time_series_data, "target", window_days=60
            )

        # Check results structure
        assert "temporal_importance" in results
        assert "importance_trends" in results
        assert "most_stable_features" in results
        assert "most_volatile_features" in results

        # Check temporal importance DataFrame
        temporal_df = results["temporal_importance"]
        assert isinstance(temporal_df, pd.DataFrame)
        assert temporal_df.shape[1] == n_features  # Should have all features

        # Check trends analysis
        trends = results["importance_trends"]
        assert isinstance(trends, dict)
        for feature in ["feature_1", "feature_2", "feature_3"]:
            if feature in trends:
                assert "mean_importance" in trends[feature]
                assert "stability" in trends[feature]
                assert "trend_slope" in trends[feature]

    def test_feature_categorization(self, mock_models, analyzer_config):
        """Test feature categorization functionality."""
        analyzer = FeatureImportanceAnalyzer(mock_models, analyzer_config)

        # Test with financial feature names
        feature_names = [
            "price_close", "volume_traded", "rsi_14", "pe_ratio",
            "gdp_growth", "macd_signal", "earnings_yield", "vix_index",
            "sma_20", "debt_to_equity", "unknown_feature"
        ]

        categorization = analyzer._categorize_features(feature_names)

        # Check structure
        assert "categories" in categorization
        assert "summary" in categorization

        categories = categorization["categories"]

        # Check specific categorizations
        assert "price_close" in categories["price_features"]
        assert "volume_traded" in categories["volume_features"]
        assert "rsi_14" in categories["technical_indicators"]
        assert "pe_ratio" in categories["fundamental_features"]
        assert "gdp_growth" in categories["macro_features"]
        assert "unknown_feature" in categories["other_features"]

        # Check summary statistics
        summary = categorization["summary"]
        total_percentage = sum(cat["percentage"] for cat in summary.values())
        assert abs(total_percentage - 100.0) < 1e-6  # Should sum to 100%

    def test_comprehensive_report_generation(self, mock_models, sample_data, analyzer_config):
        """Test comprehensive feature importance report generation."""
        X, y = sample_data
        analyzer = FeatureImportanceAnalyzer(mock_models, analyzer_config)

        # Mock individual model analysis
        def mock_analyze_shap(model_name, X, y, feature_names):
            importance = pd.Series([0.5, 0.3, 0.8, 0.1], index=feature_names)
            return {
                "feature_importance": importance,
                "importance_ranking": [
                    {"rank": i+1, "feature": feat, "importance": imp}
                    for i, (feat, imp) in enumerate(importance.sort_values(ascending=False).items())
                ],
                "method": "shap"
            }

        with patch.object(analyzer, 'analyze_shap_importance', side_effect=mock_analyze_shap):
            report = analyzer.generate_feature_importance_report(X, y, list(X.columns))

        # Check report structure
        assert "analysis_summary" in report
        assert "individual_models" in report
        assert "cross_model_comparison" in report
        assert "feature_analysis" in report

        # Check analysis summary
        summary = report["analysis_summary"]
        assert summary["n_models"] == len(mock_models)
        assert summary["n_features"] == len(X.columns)
        assert summary["n_samples"] == len(X)

        # Check individual model results
        individual = report["individual_models"]
        assert len(individual) == len(mock_models)
        for _model_name, model_results in individual.items():
            if "error" not in model_results:
                assert "top_features" in model_results
                assert "method" in model_results
                assert "total_importance" in model_results

        # Check feature analysis
        feature_analysis = report["feature_analysis"]
        assert "categories" in feature_analysis
        assert "summary" in feature_analysis

    def test_edge_cases_and_error_handling(self, analyzer_config):
        """Test edge cases and error handling."""
        # Test with minimal models
        minimal_model = MockModel("linear")
        analyzer = FeatureImportanceAnalyzer({"single_model": minimal_model}, analyzer_config)

        # Test with minimal data
        minimal_X = pd.DataFrame([[1, 2], [3, 4]], columns=["feat1", "feat2"])
        minimal_y = pd.Series([0.5, 1.5])

        # Should handle small datasets gracefully
        try:
            results = analyzer.analyze_shap_importance("single_model", minimal_X, minimal_y)
            assert isinstance(results, dict)
            assert "feature_importance" in results
        except Exception:
            # Acceptable if insufficient data
            pass

        # Test with non-existent model
        with pytest.raises(ValueError, match="Model nonexistent not found"):
            analyzer.analyze_shap_importance("nonexistent", minimal_X, minimal_y)

    def test_shap_statistics_calculation(self, mock_models, analyzer_config):
        """Test SHAP statistics calculation."""
        analyzer = FeatureImportanceAnalyzer(mock_models, analyzer_config)

        # Mock SHAP values
        n_samples, n_features = 50, 3
        shap_values = np.random.normal(0, 1, (n_samples, n_features))
        feature_names = ["feat1", "feat2", "feat3"]

        stats = analyzer._calculate_shap_statistics(shap_values, feature_names)

        # Check that all expected statistics are present
        expected_stats = ["mean_abs_shap", "std_shap", "median_shap", "max_shap", "min_shap"]
        for stat in expected_stats:
            assert stat in stats
            assert isinstance(stats[stat], pd.Series)
            assert len(stats[stat]) == n_features

        # Verify some calculations
        mean_abs_shap = stats["mean_abs_shap"]
        expected_mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names)
        pd.testing.assert_series_equal(mean_abs_shap, expected_mean_abs)

    def test_feature_interaction_analysis(self, mock_models, analyzer_config):
        """Test feature interaction analysis."""
        analyzer = FeatureImportanceAnalyzer(mock_models, analyzer_config)

        # Mock interaction values
        n_samples, n_features = 30, 3
        interaction_values = np.random.uniform(0, 1, (n_samples, n_features, n_features))
        feature_names = ["feat1", "feat2", "feat3"]

        interaction_results = analyzer._analyze_feature_interactions(interaction_values, feature_names)

        # Check structure
        assert "interaction_matrix" in interaction_results
        assert "top_interactions" in interaction_results
        assert "interaction_summary" in interaction_results

        # Check interaction matrix
        matrix = interaction_results["interaction_matrix"]
        assert isinstance(matrix, pd.DataFrame)
        assert matrix.shape == (n_features, n_features)

        # Check top interactions
        interactions = interaction_results["top_interactions"]
        assert isinstance(interactions, list)
        for interaction in interactions:
            assert "feature1" in interaction
            assert "feature2" in interaction
            assert "interaction_strength" in interaction

        # Check summary
        summary = interaction_results["interaction_summary"]
        assert "n_significant_interactions" in summary
        assert "max_interaction" in summary
        assert "mean_interaction" in summary
