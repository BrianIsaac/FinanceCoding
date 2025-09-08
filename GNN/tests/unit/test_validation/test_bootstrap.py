"""Unit tests for bootstrap methodology framework."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.evaluation.validation.bootstrap import (
    BootstrapMethodology,
    MultiMetricBootstrap,
    calmar_ratio,
    maximum_drawdown,
    sharpe_ratio,
    sortino_ratio,
)


class TestBootstrapMethodology:
    """Test cases for BootstrapMethodology class."""

    @pytest.fixture
    def bootstrap_framework(self):
        """Create BootstrapMethodology instance."""
        return BootstrapMethodology(n_bootstrap=100, random_state=42)

    @pytest.fixture
    def sample_returns(self):
        """Generate sample return data."""
        np.random.seed(42)
        return np.random.normal(0.001, 0.02, 252)

    def test_bootstrap_confidence_intervals_basic(self, bootstrap_framework, sample_returns):
        """Test basic bootstrap confidence interval calculation."""
        lower, upper, info = bootstrap_framework.bootstrap_confidence_intervals(
            sharpe_ratio, sample_returns, confidence_level=0.95
        )

        assert isinstance(lower, (float, np.floating))
        assert isinstance(upper, (float, np.floating))
        assert isinstance(info, dict)

        # Lower bound should be less than upper bound
        assert lower < upper

        # Check info dictionary
        required_keys = [
            "original_statistic",
            "bootstrap_mean",
            "bootstrap_std",
            "bootstrap_samples",
        ]
        for key in required_keys:
            assert key in info

        assert info["bootstrap_samples"] <= 100
        assert info["confidence_level"] == 0.95

    def test_bootstrap_confidence_intervals_methods(self, bootstrap_framework, sample_returns):
        """Test different bootstrap CI methods."""
        methods = ["percentile", "bias_corrected", "bca"]

        for method in methods:
            lower, upper, info = bootstrap_framework.bootstrap_confidence_intervals(
                sharpe_ratio, sample_returns, method=method
            )

            assert not np.isnan(lower)
            assert not np.isnan(upper)
            assert lower < upper
            assert info["method"] == method

    def test_bootstrap_confidence_intervals_different_levels(
        self, bootstrap_framework, sample_returns
    ):
        """Test CI calculation with different confidence levels."""
        confidence_levels = [0.80, 0.90, 0.95, 0.99]
        intervals = []

        for level in confidence_levels:
            lower, upper, info = bootstrap_framework.bootstrap_confidence_intervals(
                sharpe_ratio, sample_returns, confidence_level=level
            )
            intervals.append((lower, upper, level))

        # Higher confidence levels should produce wider intervals
        widths = [(upper - lower, level) for lower, upper, level in intervals]
        widths.sort(key=lambda x: x[1])  # Sort by confidence level

        for i in range(1, len(widths)):
            assert widths[i][0] >= widths[i - 1][0]  # Width should increase

    def test_bootstrap_confidence_intervals_pandas_series(self, bootstrap_framework):
        """Test with pandas Series input."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        returns_series = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)

        lower, upper, info = bootstrap_framework.bootstrap_confidence_intervals(
            sharpe_ratio, returns_series
        )

        assert not np.isnan(lower)
        assert not np.isnan(upper)
        assert lower < upper

    def test_bootstrap_significance_test_basic(self, bootstrap_framework):
        """Test basic bootstrap significance test."""
        np.random.seed(42)
        returns_a = np.random.normal(0.001, 0.02, 100)
        returns_b = np.random.normal(0.0005, 0.025, 100)

        result = bootstrap_framework.bootstrap_significance_test(returns_a, returns_b, sharpe_ratio)

        required_keys = [
            "original_diff",
            "bootstrap_p_value",
            "bootstrap_mean_diff",
            "valid_samples",
        ]
        for key in required_keys:
            assert key in result

        assert 0 <= result["bootstrap_p_value"] <= 1
        assert result["valid_samples"] > 0

    def test_bootstrap_significance_test_alternatives(self, bootstrap_framework):
        """Test different alternative hypotheses."""
        np.random.seed(42)
        returns_a = np.random.normal(0.002, 0.02, 100)  # Higher mean
        returns_b = np.random.normal(0.001, 0.02, 100)

        alternatives = ["two-sided", "greater", "less"]

        for alt in alternatives:
            result = bootstrap_framework.bootstrap_significance_test(
                returns_a, returns_b, sharpe_ratio, alternative=alt
            )

            assert "bootstrap_p_value" in result
            assert result["alternative"] == alt
            assert 0 <= result["bootstrap_p_value"] <= 1

    def test_bootstrap_significance_test_identical_series(self, bootstrap_framework):
        """Test significance test with identical return series."""
        returns = np.random.normal(0.001, 0.02, 100)

        result = bootstrap_framework.bootstrap_significance_test(returns, returns, sharpe_ratio)

        # Difference should be essentially zero
        assert abs(result["original_diff"]) < 1e-10
        # P-value should be high (not significant)
        assert result["bootstrap_p_value"] > 0.05

    def test_bootstrap_significance_test_unequal_lengths(self, bootstrap_framework):
        """Test that unequal length series raise ValueError."""
        returns_a = np.random.normal(0.001, 0.02, 100)
        returns_b = np.random.normal(0.001, 0.02, 50)

        with pytest.raises(ValueError, match="Return series must have equal length"):
            bootstrap_framework.bootstrap_significance_test(returns_a, returns_b, sharpe_ratio)

    def test_paired_bootstrap_test_basic(self, bootstrap_framework):
        """Test basic paired bootstrap test."""
        np.random.seed(42)
        returns_a = np.random.normal(0.001, 0.02, 100)
        returns_b = np.random.normal(0.0005, 0.025, 100)

        result = bootstrap_framework.paired_bootstrap_test(returns_a, returns_b, sharpe_ratio)

        required_keys = [
            "original_diff",
            "paired_bootstrap_p_value",
            "bootstrap_mean_diff",
            "valid_samples",
        ]
        for key in required_keys:
            assert key in result

        assert 0 <= result["paired_bootstrap_p_value"] <= 1
        assert result["valid_samples"] > 0

    def test_paired_bootstrap_test_temporal_structure(self, bootstrap_framework):
        """Test that paired bootstrap maintains temporal structure."""
        # Create returns with temporal dependence
        np.random.seed(42)
        base_trend = np.linspace(0, 0.002, 100)
        returns_a = base_trend + np.random.normal(0, 0.01, 100)
        returns_b = base_trend * 0.5 + np.random.normal(0, 0.012, 100)

        result = bootstrap_framework.paired_bootstrap_test(returns_a, returns_b, sharpe_ratio)

        # Should complete without errors and provide meaningful results
        assert "paired_bootstrap_p_value" in result
        assert not np.isnan(result["paired_bootstrap_p_value"])

    def test_acceleration_calculation_private_method(self, bootstrap_framework):
        """Test acceleration calculation for BCa method."""
        returns = np.random.normal(0.001, 0.02, 50)  # Smaller sample for speed

        # Access private method for testing
        acceleration = bootstrap_framework._calculate_acceleration(sharpe_ratio, returns)

        assert isinstance(acceleration, (float, np.floating))
        # Acceleration should be finite
        assert np.isfinite(acceleration)

    def test_bootstrap_with_failing_metric(self, bootstrap_framework, sample_returns):
        """Test bootstrap behavior when performance metric fails."""

        def failing_metric(returns):
            if len(returns) < 50:  # Fail for small samples
                raise ValueError("Insufficient data")
            return np.mean(returns) / np.std(returns)

        # Should handle failures gracefully
        lower, upper, info = bootstrap_framework.bootstrap_confidence_intervals(
            failing_metric, sample_returns
        )

        # Should still return results, possibly with warnings
        assert isinstance(lower, (float, np.floating))
        assert isinstance(upper, (float, np.floating))


class TestMultiMetricBootstrap:
    """Test cases for MultiMetricBootstrap class."""

    @pytest.fixture
    def bootstrap_framework(self):
        """Create BootstrapMethodology instance."""
        return BootstrapMethodology(n_bootstrap=50, random_state=42)

    @pytest.fixture
    def multi_metric_bootstrap(self, bootstrap_framework):
        """Create MultiMetricBootstrap instance."""
        return MultiMetricBootstrap(bootstrap_framework)

    @pytest.fixture
    def sample_returns(self):
        """Generate sample return data."""
        np.random.seed(42)
        return np.random.normal(0.001, 0.02, 252)

    def test_multi_metric_confidence_intervals(self, multi_metric_bootstrap, sample_returns):
        """Test confidence intervals for multiple metrics."""
        metrics = {
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "maximum_drawdown": maximum_drawdown,
        }

        result = multi_metric_bootstrap.multi_metric_confidence_intervals(sample_returns, metrics)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(metrics)

        required_columns = ["metric", "original_value", "ci_lower", "ci_upper"]
        for col in required_columns:
            assert col in result.columns

        # Check that all metrics are present
        metric_names = set(result["metric"].tolist())
        assert metric_names == set(metrics.keys())

        # Check that confidence intervals are properly ordered
        for _, row in result.iterrows():
            assert row["ci_lower"] <= row["original_value"] <= row["ci_upper"]

    def test_metric_comparison_matrix(self, multi_metric_bootstrap):
        """Test metric comparison matrix generation."""
        np.random.seed(42)
        returns_dict = {
            "Portfolio_A": np.random.normal(0.001, 0.02, 100),
            "Portfolio_B": np.random.normal(0.0008, 0.022, 100),
            "Benchmark": np.random.normal(0.0005, 0.018, 100),
        }

        metrics = {"sharpe_ratio": sharpe_ratio, "sortino_ratio": sortino_ratio}

        result = multi_metric_bootstrap.metric_comparison_matrix(returns_dict, metrics)

        assert isinstance(result, dict)
        assert set(result.keys()) == set(metrics.keys())

        for _metric_name, matrices in result.items():
            assert "p_values" in matrices
            assert "differences" in matrices

            p_matrix = matrices["p_values"]
            diff_matrix = matrices["differences"]

            assert isinstance(p_matrix, pd.DataFrame)
            assert isinstance(diff_matrix, pd.DataFrame)

            # Matrices should be square
            assert p_matrix.shape[0] == p_matrix.shape[1]
            assert diff_matrix.shape[0] == diff_matrix.shape[1]

            # Diagonal should be NaN for p-values and 0 for differences
            np.testing.assert_array_equal(np.diag(diff_matrix.values), [0.0, 0.0, 0.0])


class TestPerformanceMetricFunctions:
    """Test the performance metric utility functions."""

    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns with known properties."""
        np.random.seed(42)
        return np.random.normal(0.001, 0.02, 252)

    def test_sharpe_ratio_function(self, sample_returns):
        """Test Sharpe ratio calculation function."""
        sharpe = sharpe_ratio(sample_returns)

        expected_sharpe = np.mean(sample_returns) / np.std(sample_returns)
        assert abs(sharpe - expected_sharpe) < 1e-10

        # Test edge cases
        assert sharpe_ratio(np.array([])) == 0.0
        assert sharpe_ratio(np.ones(100)) == 0.0  # Zero volatility

    def test_sortino_ratio_function(self, sample_returns):
        """Test Sortino ratio calculation function."""
        sortino = sortino_ratio(sample_returns)

        assert isinstance(sortino, (float, np.floating))
        assert np.isfinite(sortino)

        # Test with different target returns
        sortino_custom = sortino_ratio(sample_returns, target_return=0.0005)
        assert isinstance(sortino_custom, (float, np.floating))

        # Test edge cases
        positive_returns = np.abs(sample_returns) + 0.001
        sortino_pos = sortino_ratio(positive_returns)
        assert np.isinf(sortino_pos) or sortino_pos > 100  # Should be very high

    def test_maximum_drawdown_function(self, sample_returns):
        """Test maximum drawdown calculation function."""
        mdd = maximum_drawdown(sample_returns)

        assert isinstance(mdd, (float, np.floating))
        assert mdd >= 0  # Drawdown should be positive

        # Test with known sequence
        known_returns = np.array([0.1, -0.05, -0.1, 0.2, -0.15])
        known_mdd = maximum_drawdown(known_returns)
        assert known_mdd > 0

        # Test edge cases
        assert maximum_drawdown(np.array([])) == 0.0

        constant_positive = np.ones(100) * 0.01
        assert maximum_drawdown(constant_positive) == 0.0

    def test_calmar_ratio_function(self, sample_returns):
        """Test Calmar ratio calculation function."""
        calmar = calmar_ratio(sample_returns)

        assert isinstance(calmar, (float, np.floating))
        # Calmar ratio can be infinite if no drawdown
        assert np.isfinite(calmar) or np.isinf(calmar)

        # Test with known drawdown
        returns_with_drawdown = np.array([0.01, -0.02, 0.005] * 50)
        calmar_known = calmar_ratio(returns_with_drawdown)
        assert np.isfinite(calmar_known)


class TestBootstrapEdgeCases:
    """Test edge cases and error conditions for bootstrap methods."""

    def test_very_small_sample_size(self):
        """Test bootstrap with very small sample sizes."""
        bootstrap = BootstrapMethodology(n_bootstrap=10, random_state=42)
        small_returns = np.array([0.01, -0.02, 0.005])

        lower, upper, info = bootstrap.bootstrap_confidence_intervals(sharpe_ratio, small_returns)

        # Should complete but may have warnings
        assert isinstance(lower, (float, np.floating))
        assert isinstance(upper, (float, np.floating))

    def test_extreme_outliers(self):
        """Test bootstrap with extreme outliers."""
        bootstrap = BootstrapMethodology(n_bootstrap=50, random_state=42)
        returns_with_outliers = np.concatenate(
            [np.random.normal(0.001, 0.02, 98), [10.0, -10.0]]  # Extreme outliers
        )

        lower, upper, info = bootstrap.bootstrap_confidence_intervals(
            sharpe_ratio, returns_with_outliers
        )

        # Should handle outliers without crashing
        assert isinstance(lower, (float, np.floating))
        assert isinstance(upper, (float, np.floating))

    def test_all_zero_returns(self):
        """Test bootstrap with all zero returns."""
        bootstrap = BootstrapMethodology(n_bootstrap=50, random_state=42)
        zero_returns = np.zeros(100)

        lower, upper, info = bootstrap.bootstrap_confidence_intervals(sharpe_ratio, zero_returns)

        # Should handle zero returns
        assert lower == 0.0
        assert upper == 0.0
        assert info["original_statistic"] == 0.0

    def test_high_bootstrap_samples(self):
        """Test with high number of bootstrap samples."""
        bootstrap = BootstrapMethodology(n_bootstrap=2000, random_state=42)
        returns = np.random.normal(0.001, 0.02, 100)

        lower, upper, info = bootstrap.bootstrap_confidence_intervals(sharpe_ratio, returns)

        assert info["bootstrap_samples"] == 2000
        assert not np.isnan(lower)
        assert not np.isnan(upper)

    def test_random_state_reproducibility(self):
        """Test that random state produces reproducible results."""
        returns = np.random.normal(0.001, 0.02, 100)

        bootstrap1 = BootstrapMethodology(n_bootstrap=100, random_state=42)
        lower1, upper1, _ = bootstrap1.bootstrap_confidence_intervals(sharpe_ratio, returns)

        bootstrap2 = BootstrapMethodology(n_bootstrap=100, random_state=42)
        lower2, upper2, _ = bootstrap2.bootstrap_confidence_intervals(sharpe_ratio, returns)

        # Results should be identical with same random state
        assert abs(lower1 - lower2) < 1e-10
        assert abs(upper1 - upper2) < 1e-10


class TestBootstrapAccuracy:
    """Test statistical accuracy of bootstrap methods."""

    def test_coverage_probability(self):
        """Test that confidence intervals have correct coverage."""
        # This is a simulation-based test
        bootstrap = BootstrapMethodology(n_bootstrap=200, random_state=42)

        # Known population parameters
        true_mean = 0.001
        true_std = 0.02
        true_sharpe = true_mean / true_std

        coverage_count = 0
        n_simulations = 50  # Reduced for test speed

        for i in range(n_simulations):
            np.random.seed(i)  # Different seed for each simulation
            sample_returns = np.random.normal(true_mean, true_std, 252)

            lower, upper, _ = bootstrap.bootstrap_confidence_intervals(
                sharpe_ratio, sample_returns, confidence_level=0.95
            )

            if lower <= true_sharpe <= upper:
                coverage_count += 1

        coverage_rate = coverage_count / n_simulations

        # Should be close to 0.95, allowing for Monte Carlo variation
        assert 0.80 < coverage_rate < 1.0  # Relaxed bounds for test stability

    def test_bootstrap_vs_theoretical_variance(self):
        """Test bootstrap standard error against theoretical values."""
        # For large samples, bootstrap SE should approximate theoretical SE
        bootstrap = BootstrapMethodology(n_bootstrap=500, random_state=42)

        # Large sample
        np.random.seed(42)
        large_sample = np.random.normal(0.001, 0.02, 1000)

        _, _, info = bootstrap.bootstrap_confidence_intervals(sharpe_ratio, large_sample)

        bootstrap_se = info["bootstrap_std"]

        # Theoretical SE for Sharpe ratio (approximate)
        sample_sharpe = sharpe_ratio(large_sample)
        theoretical_se = np.sqrt((1 + 0.5 * sample_sharpe**2) / len(large_sample))

        # Should be reasonably close
        assert abs(bootstrap_se - theoretical_se) < theoretical_se * 0.5
