"""
Tests for risk factor attribution analysis framework.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.evaluation.interpretability.factor_attribution import (
    FactorAttributionConfig,
    FactorAttributor,
)


class TestFactorAttributor:
    """Test suite for factor attribution analysis."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")

        # Create sample returns data
        n_assets = 5
        asset_names = [f"ASSET_{i}" for i in range(n_assets)]

        # Generate correlated returns
        returns_data = np.random.normal(0.001, 0.02, (100, n_assets))
        returns = pd.DataFrame(returns_data, index=dates, columns=asset_names)

        # Create portfolio weights
        weights = pd.Series(
            np.random.uniform(0.1, 0.3, n_assets),
            index=asset_names
        )
        weights = weights / weights.sum()

        # Create benchmark weights (equal weight)
        benchmark_weights = pd.Series(
            [1.0 / n_assets] * n_assets,
            index=asset_names
        )

        return returns, weights, benchmark_weights

    @pytest.fixture
    def config(self):
        """Create configuration for testing."""
        return FactorAttributionConfig(
            min_observations=30,  # Reduced for testing
            significance_level=0.1,  # Less strict for testing
        )

    def test_initialization(self, config):
        """Test factor attributor initialization."""
        attributor = FactorAttributor(config)

        assert attributor.config == config
        assert attributor.factor_data_provider is None
        assert attributor._cached_factor_data == {}

    def test_portfolio_returns_calculation(self, sample_data, config):
        """Test portfolio returns calculation."""
        returns, weights, _ = sample_data
        attributor = FactorAttributor(config)

        portfolio_returns = attributor._calculate_portfolio_returns(weights, returns)

        # Check basic properties
        assert isinstance(portfolio_returns, pd.Series)
        assert len(portfolio_returns) == len(returns)
        assert portfolio_returns.index.equals(returns.index)

        # Check that returns are reasonable
        assert not portfolio_returns.isna().any()
        assert abs(portfolio_returns.mean()) < 0.1  # Should be small daily returns

    def test_portfolio_returns_no_common_assets(self, config):
        """Test portfolio returns with no common assets."""
        attributor = FactorAttributor(config)

        returns = pd.DataFrame(
            [[0.01, 0.02], [0.02, -0.01]],
            columns=["A", "B"],
            index=pd.date_range("2023-01-01", periods=2)
        )
        weights = pd.Series([0.5, 0.5], index=["C", "D"])

        with pytest.raises(ValueError, match="No common assets"):
            attributor._calculate_portfolio_returns(weights, returns)

    def test_synthetic_factor_generation(self, sample_data, config):
        """Test synthetic factor generation."""
        returns, _, _ = sample_data
        attributor = FactorAttributor(config)

        factors = attributor._generate_synthetic_factors(returns)

        # Check basic structure
        assert isinstance(factors, pd.DataFrame)
        assert len(factors) == len(returns)
        assert factors.index.equals(returns.index)

        # Check expected factors
        expected_factors = ["Market", "Size", "Momentum", "Quality"]
        for factor in expected_factors:
            assert factor in factors.columns

        # Check no NaN values
        assert not factors.isna().any().any()

    def test_factor_loadings_estimation(self, sample_data, config):
        """Test factor loadings estimation."""
        returns, weights, _ = sample_data
        attributor = FactorAttributor(config)

        portfolio_returns = attributor._calculate_portfolio_returns(weights, returns)
        factor_returns = attributor._generate_synthetic_factors(returns)

        loadings = attributor._estimate_factor_loadings(portfolio_returns, factor_returns)

        # Check structure
        assert isinstance(loadings, dict)
        assert "alpha" in loadings
        assert "betas" in loadings
        assert "r_squared" in loadings

        # Handle case where estimation might fail due to singular matrix
        if "error" in loadings:
            # If there's an error, check that it's handled gracefully
            assert loadings["r_squared"] == 0.0
            assert loadings["betas"] == {}
        else:
            # If successful, check full structure
            assert "coefficients" in loadings
            assert "p_values" in loadings

            # Check that we have beta for each factor
            for factor in factor_returns.columns:
                assert factor in loadings["betas"]

            # Check R-squared is reasonable
            assert 0 <= loadings["r_squared"] <= 1

    def test_factor_loadings_insufficient_data(self, config):
        """Test factor loadings with insufficient data."""
        attributor = FactorAttributor(config)

        # Create minimal data (less than min_observations)
        short_returns = pd.Series(
            [0.01, 0.02],
            index=pd.date_range("2023-01-01", periods=2)
        )
        short_factors = pd.DataFrame(
            [[0.01], [0.02]],
            columns=["Factor1"],
            index=pd.date_range("2023-01-01", periods=2)
        )

        with pytest.raises(ValueError, match="Insufficient observations"):
            attributor._estimate_factor_loadings(short_returns, short_factors)

    def test_factor_contributions_calculation(self, sample_data, config):
        """Test factor contributions calculation."""
        returns, weights, _ = sample_data
        attributor = FactorAttributor(config)

        portfolio_returns = attributor._calculate_portfolio_returns(weights, returns)
        factor_returns = attributor._generate_synthetic_factors(returns)
        factor_loadings = attributor._estimate_factor_loadings(portfolio_returns, factor_returns)

        contributions = attributor._calculate_factor_contributions(
            factor_loadings, factor_returns, portfolio_returns
        )

        # Check structure
        assert isinstance(contributions, dict)
        assert "contributions" in contributions
        assert "total_explained" in contributions
        assert "explanation_ratio" in contributions

        # Check contributions for each factor
        for factor in factor_returns.columns:
            if factor in contributions["contributions"]:
                factor_contrib = contributions["contributions"][factor]
                assert "total_contribution" in factor_contrib
                assert "beta" in factor_contrib
                assert isinstance(factor_contrib["total_contribution"], float)

    def test_factor_contributions_empty_betas(self, sample_data, config):
        """Test factor contributions with empty betas."""
        returns, weights, _ = sample_data
        attributor = FactorAttributor(config)

        portfolio_returns = attributor._calculate_portfolio_returns(weights, returns)
        factor_returns = attributor._generate_synthetic_factors(returns)

        # Mock empty factor loadings
        empty_loadings = {"betas": {}}

        contributions = attributor._calculate_factor_contributions(
            empty_loadings, factor_returns, portfolio_returns
        )

        assert contributions["contributions"] == {}
        assert contributions["total_explained"] == 0.0

    def test_active_exposures_calculation(self, sample_data, config):
        """Test active exposures calculation."""
        returns, weights, benchmark_weights = sample_data
        attributor = FactorAttributor(config)

        portfolio_returns = attributor._calculate_portfolio_returns(weights, returns)
        benchmark_returns = attributor._calculate_portfolio_returns(benchmark_weights, returns)
        factor_returns = attributor._generate_synthetic_factors(returns)

        portfolio_loadings = attributor._estimate_factor_loadings(portfolio_returns, factor_returns)
        benchmark_loadings = attributor._estimate_factor_loadings(benchmark_returns, factor_returns)

        active_exposures = attributor._calculate_active_exposures(
            portfolio_loadings, benchmark_loadings
        )

        # Check structure
        assert isinstance(active_exposures, dict)
        assert "active_betas" in active_exposures
        assert "total_active_risk" in active_exposures
        assert "largest_active_exposures" in active_exposures

        # Check that active betas are calculated
        for factor in factor_returns.columns:
            assert factor in active_exposures["active_betas"]

        # Check that total active risk is non-negative
        assert active_exposures["total_active_risk"] >= 0

    def test_risk_attribution_calculation(self, sample_data, config):
        """Test risk attribution calculation."""
        returns, weights, _ = sample_data
        attributor = FactorAttributor(config)

        portfolio_returns = attributor._calculate_portfolio_returns(weights, returns)
        factor_returns = attributor._generate_synthetic_factors(returns)
        factor_loadings = attributor._estimate_factor_loadings(portfolio_returns, factor_returns)

        risk_attribution = attributor._calculate_risk_attribution(factor_loadings, factor_returns)

        # Check structure
        assert isinstance(risk_attribution, dict)
        assert "factor_risks" in risk_attribution
        assert "systematic_variance" in risk_attribution
        assert "idiosyncratic_variance" in risk_attribution
        assert "systematic_ratio" in risk_attribution

        # Check that systematic ratio is between 0 and 1
        assert 0 <= risk_attribution["systematic_ratio"] <= 1

        # Check that variances are non-negative
        assert risk_attribution["systematic_variance"] >= 0
        assert risk_attribution["idiosyncratic_variance"] >= 0

    def test_risk_attribution_empty_betas(self, sample_data, config):
        """Test risk attribution with empty betas."""
        returns, _, _ = sample_data
        attributor = FactorAttributor(config)

        factor_returns = attributor._generate_synthetic_factors(returns)
        empty_loadings = {"betas": {}}

        risk_attribution = attributor._calculate_risk_attribution(empty_loadings, factor_returns)

        assert risk_attribution["factor_risks"] == {}
        assert risk_attribution["total_systematic_risk"] == 0.0

    def test_factor_timing_analysis(self, sample_data, config):
        """Test factor timing analysis."""
        returns, weights, _ = sample_data
        attributor = FactorAttributor(config)

        factor_returns = attributor._generate_synthetic_factors(returns)

        timing_analysis = attributor._analyze_factor_timing(weights, returns, factor_returns)

        # Check structure
        assert isinstance(timing_analysis, dict)
        assert "factor_correlations" in timing_analysis
        assert "timing_signals" in timing_analysis

        # Check correlations for each factor
        for factor in factor_returns.columns:
            assert factor in timing_analysis["factor_correlations"]
            corr = timing_analysis["factor_correlations"][factor]
            assert -1 <= corr <= 1  # Correlation should be between -1 and 1

    def test_summary_statistics_calculation(self, sample_data, config):
        """Test summary statistics calculation."""
        returns, weights, _ = sample_data
        attributor = FactorAttributor(config)

        portfolio_returns = attributor._calculate_portfolio_returns(weights, returns)
        factor_returns = attributor._generate_synthetic_factors(returns)
        factor_loadings = attributor._estimate_factor_loadings(portfolio_returns, factor_returns)

        factor_contributions = attributor._calculate_factor_contributions(
            factor_loadings, factor_returns, portfolio_returns
        )
        risk_attribution = attributor._calculate_risk_attribution(factor_loadings, factor_returns)

        summary = attributor._calculate_summary_statistics(
            factor_loadings, factor_contributions, risk_attribution
        )

        # Check structure
        assert isinstance(summary, dict)
        assert "model_quality" in summary
        assert "risk_decomposition" in summary

        # Check model quality metrics
        model_quality = summary["model_quality"]
        assert "r_squared" in model_quality
        assert "significant_factors" in model_quality
        assert "explanation_ratio" in model_quality

        # Check risk decomposition
        risk_decomp = summary["risk_decomposition"]
        assert "systematic_ratio" in risk_decomp
        assert "systematic_volatility" in risk_decomp
        assert "idiosyncratic_volatility" in risk_decomp

    def test_full_factor_exposure_analysis(self, sample_data, config):
        """Test full factor exposure analysis."""
        returns, weights, benchmark_weights = sample_data
        attributor = FactorAttributor(config)

        analysis = attributor.analyze_factor_exposure(
            weights, returns, benchmark_weights
        )

        # Check all expected components are present
        expected_keys = [
            "factor_loadings", "factor_contributions", "active_exposures",
            "risk_attribution", "timing_analysis", "portfolio_returns",
            "factor_returns", "summary_statistics"
        ]

        for key in expected_keys:
            assert key in analysis

        # Check that analysis components are non-empty
        assert len(analysis["factor_loadings"]) > 0
        assert len(analysis["factor_returns"].columns) > 0
        assert len(analysis["portfolio_returns"]) == len(returns)

    def test_analysis_without_benchmark(self, sample_data, config):
        """Test analysis without benchmark weights."""
        returns, weights, _ = sample_data
        attributor = FactorAttributor(config)

        analysis = attributor.analyze_factor_exposure(weights, returns)

        # Should still work without benchmark
        assert "factor_loadings" in analysis
        assert "factor_contributions" in analysis
        assert analysis["active_exposures"] == {}  # Should be empty without benchmark

    def test_analysis_with_provided_factors(self, sample_data, config):
        """Test analysis with provided factor returns."""
        returns, weights, _ = sample_data
        attributor = FactorAttributor(config)

        # Create custom factor returns
        custom_factors = pd.DataFrame({
            "CustomFactor1": np.random.normal(0, 0.01, len(returns)),
            "CustomFactor2": np.random.normal(0, 0.01, len(returns)),
        }, index=returns.index)

        analysis = attributor.analyze_factor_exposure(
            weights, returns, factor_returns=custom_factors
        )

        # Should use provided factors
        assert list(analysis["factor_returns"].columns) == list(custom_factors.columns)
        assert "CustomFactor1" in analysis["factor_loadings"]["betas"]
        assert "CustomFactor2" in analysis["factor_loadings"]["betas"]

    def test_edge_cases_and_error_handling(self, config):
        """Test edge cases and error handling."""
        attributor = FactorAttributor(config)

        # Test with minimal data
        minimal_returns = pd.DataFrame(
            [[0.01, 0.02]],
            columns=["A", "B"],
            index=pd.date_range("2023-01-01", periods=1)
        )
        minimal_weights = pd.Series([0.5, 0.5], index=["A", "B"])

        # Should handle small datasets gracefully or raise appropriate errors
        try:
            analysis = attributor.analyze_factor_exposure(minimal_weights, minimal_returns)
            # If it succeeds, check it returns valid structure
            assert isinstance(analysis, dict)
        except ValueError:
            # Acceptable if insufficient data
            pass

    @patch('src.evaluation.interpretability.factor_attribution.SKLEARN_AVAILABLE', False)
    def test_fallback_without_sklearn(self, sample_data, config):
        """Test fallback functionality without sklearn."""
        returns, weights, _ = sample_data
        attributor = FactorAttributor(config)

        # Should still work without sklearn using numpy fallback
        portfolio_returns = attributor._calculate_portfolio_returns(weights, returns)
        factor_returns = attributor._generate_synthetic_factors(returns)

        # This should not fail even without sklearn
        loadings = attributor._estimate_factor_loadings(portfolio_returns, factor_returns)

        assert isinstance(loadings, dict)
        assert "alpha" in loadings
        assert "betas" in loadings

    def test_configuration_defaults(self):
        """Test configuration defaults."""
        config = FactorAttributionConfig()

        # Check default values
        assert config.lookback_window == 252
        assert config.min_observations == 60
        assert config.significance_level == 0.05
        assert config.risk_free_rate == 0.02

        # Check default factor models
        expected_models = ["fama_french_3", "momentum", "quality", "size", "value"]
        assert config.factor_models == expected_models

    def test_custom_configuration(self):
        """Test custom configuration."""
        custom_config = FactorAttributionConfig(
            factor_models=["custom_factor"],
            min_observations=30,
            significance_level=0.1,
        )

        assert custom_config.factor_models == ["custom_factor"]
        assert custom_config.min_observations == 30
        assert custom_config.significance_level == 0.1
