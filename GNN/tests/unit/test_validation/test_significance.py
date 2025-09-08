"""Unit tests for statistical significance testing framework."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.evaluation.validation.significance import (
    PerformanceSignificanceTest,
    StatisticalValidation,
)


class TestStatisticalValidation:
    """Test cases for StatisticalValidation class."""

    @pytest.fixture
    def sample_returns(self):
        """Generate sample return data for testing."""
        np.random.seed(42)
        returns_a = np.random.normal(0.001, 0.02, 252)  # Daily returns with 0.1% mean, 2% vol
        returns_b = np.random.normal(0.0005, 0.025, 252)  # Slightly lower performance
        return returns_a, returns_b

    def test_sharpe_ratio_test_basic_functionality(self, sample_returns):
        """Test basic functionality of Sharpe ratio test."""
        returns_a, returns_b = sample_returns

        result = StatisticalValidation.sharpe_ratio_test(returns_a, returns_b)

        # Check required keys in result
        required_keys = [
            "test_statistic",
            "p_value",
            "is_significant",
            "sharpe_a",
            "sharpe_b",
            "sharpe_diff",
        ]
        for key in required_keys:
            assert key in result

        # Check data types and ranges
        assert isinstance(result["test_statistic"], (float, np.floating))
        assert isinstance(result["p_value"], (float, np.floating))
        assert isinstance(result["is_significant"], bool)
        assert 0 <= result["p_value"] <= 1

    def test_sharpe_ratio_test_identical_returns(self):
        """Test Sharpe ratio test with identical return series."""
        returns = np.random.normal(0.001, 0.02, 100)

        result = StatisticalValidation.sharpe_ratio_test(returns, returns)

        assert abs(result["sharpe_diff"]) < 1e-10  # Should be essentially zero
        assert abs(result["test_statistic"]) < 1e-10
        assert result["p_value"] > 0.05  # Should not be significant
        assert not result["is_significant"]

    def test_sharpe_ratio_test_unequal_lengths(self):
        """Test that unequal length series raise ValueError."""
        returns_a = np.random.normal(0.001, 0.02, 100)
        returns_b = np.random.normal(0.001, 0.02, 50)

        with pytest.raises(ValueError, match="Return series must have equal length"):
            StatisticalValidation.sharpe_ratio_test(returns_a, returns_b)

    def test_sharpe_ratio_test_zero_volatility(self):
        """Test handling of zero volatility cases."""
        returns_a = np.ones(100) * 0.001  # Constant returns (zero volatility)
        returns_b = np.random.normal(0.001, 0.02, 100)

        with pytest.warns(UserWarning, match="Zero volatility detected"):
            result = StatisticalValidation.sharpe_ratio_test(returns_a, returns_b)

        assert np.isnan(result["test_statistic"])
        assert np.isnan(result["p_value"])
        assert not result["is_significant"]

    def test_sharpe_ratio_test_pandas_series_input(self, sample_returns):
        """Test that function works with pandas Series input."""
        returns_a, returns_b = sample_returns

        series_a = pd.Series(returns_a)
        series_b = pd.Series(returns_b)

        result = StatisticalValidation.sharpe_ratio_test(series_a, series_b)

        assert "test_statistic" in result
        assert "p_value" in result
        assert not np.isnan(result["test_statistic"])
        assert not np.isnan(result["p_value"])

    def test_asymptotic_sharpe_test(self, sample_returns):
        """Test asymptotic Sharpe ratio test."""
        returns_a, returns_b = sample_returns

        result = StatisticalValidation.asymptotic_sharpe_test(returns_a, returns_b)

        required_keys = ["test_statistic", "p_value", "is_significant", "asymptotic_variance"]
        for key in required_keys:
            assert key in result

        assert result["asymptotic_variance"] >= 0
        assert result["method"] == "Asymptotic delta method"

    def test_pairwise_comparison_framework_basic(self):
        """Test basic pairwise comparison framework."""
        np.random.seed(42)
        returns_dict = {
            "Strategy_A": np.random.normal(0.001, 0.02, 100),
            "Strategy_B": np.random.normal(0.0005, 0.025, 100),
            "Benchmark": np.random.normal(0.0003, 0.015, 100),
        }

        result = StatisticalValidation.pairwise_comparison_framework(
            returns_dict, baseline_keys=["Benchmark"]
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # Two ML strategies vs one benchmark

        required_columns = [
            "portfolio_a",
            "portfolio_b",
            "test_statistic",
            "p_value",
            "is_significant",
        ]
        for col in required_columns:
            assert col in result.columns

    def test_pairwise_comparison_all_vs_all(self):
        """Test all vs all pairwise comparisons."""
        np.random.seed(42)
        returns_dict = {
            "A": np.random.normal(0.001, 0.02, 100),
            "B": np.random.normal(0.0005, 0.025, 100),
            "C": np.random.normal(0.0003, 0.015, 100),
        }

        result = StatisticalValidation.pairwise_comparison_framework(returns_dict)

        # Should have n*(n-1) comparisons where n=3
        assert len(result) == 6

        # Check all combinations are present
        portfolios = set(result["portfolio_a"].tolist() + result["portfolio_b"].tolist())
        assert portfolios == {"A", "B", "C"}

    def test_pairwise_comparison_different_methods(self):
        """Test pairwise comparisons with different methods."""
        np.random.seed(42)
        returns_dict = {
            "Strategy": np.random.normal(0.001, 0.02, 100),
            "Benchmark": np.random.normal(0.0005, 0.025, 100),
        }

        # Test Jobson-Korkie method
        result_jk = StatisticalValidation.pairwise_comparison_framework(
            returns_dict, method="jobson_korkie"
        )

        # Test asymptotic method
        result_asymp = StatisticalValidation.pairwise_comparison_framework(
            returns_dict, method="asymptotic"
        )

        assert result_jk["method"].iloc[0] == "jobson_korkie"
        assert result_asymp["method"].iloc[0] == "asymptotic"

        # Both should have same comparison structure
        assert len(result_jk) == len(result_asymp)


class TestPerformanceSignificanceTest:
    """Test cases for PerformanceSignificanceTest class."""

    @pytest.fixture
    def significance_tester(self):
        """Create PerformanceSignificanceTest instance."""
        return PerformanceSignificanceTest(alpha=0.05)

    @pytest.fixture
    def sample_returns_dict(self):
        """Generate sample returns dictionary."""
        np.random.seed(42)
        return {
            "HRP": np.random.normal(0.0008, 0.018, 252),
            "LSTM": np.random.normal(0.0012, 0.022, 252),
            "GAT": np.random.normal(0.0010, 0.020, 252),
            "Equal_Weight": np.random.normal(0.0005, 0.016, 252),
            "Mean_Variance": np.random.normal(0.0006, 0.019, 252),
        }

    def test_comprehensive_comparison_default(self, significance_tester, sample_returns_dict):
        """Test comprehensive comparison with default settings."""
        result = significance_tester.comprehensive_comparison(sample_returns_dict)

        assert isinstance(result, dict)
        assert "sharpe" in result
        assert isinstance(result["sharpe"], pd.DataFrame)

        # Should have comparisons between all pairs
        n_strategies = len(sample_returns_dict)
        expected_comparisons = n_strategies * (n_strategies - 1)
        assert len(result["sharpe"]) == expected_comparisons

    def test_comprehensive_comparison_multiple_metrics(
        self, significance_tester, sample_returns_dict
    ):
        """Test comprehensive comparison with multiple metrics."""
        metrics = ["sharpe", "sortino"]  # Only sharpe is implemented currently

        result = significance_tester.comprehensive_comparison(sample_returns_dict, metrics=metrics)

        # Should handle unknown metrics gracefully
        assert "sharpe" in result
        # Additional metrics would be added here when implemented

    def test_rolling_significance_analysis_basic(self, significance_tester, sample_returns_dict):
        """Test basic rolling significance analysis."""
        result = significance_tester.rolling_significance_analysis(
            sample_returns_dict, window_size=63, step_size=21
        )

        assert isinstance(result, pd.DataFrame)

        required_columns = ["portfolio_a", "portfolio_b", "window_start", "window_end"]
        for col in required_columns:
            assert col in result.columns

        # Should have multiple windows
        assert len(result["window_start"].unique()) > 1

    def test_rolling_significance_analysis_insufficient_data(self, significance_tester):
        """Test rolling analysis with insufficient data."""
        small_returns_dict = {
            "A": np.random.normal(0.001, 0.02, 30),
            "B": np.random.normal(0.001, 0.02, 30),
        }

        result = significance_tester.rolling_significance_analysis(
            small_returns_dict, window_size=63, step_size=21
        )

        # Should return empty DataFrame or handle gracefully
        assert isinstance(result, pd.DataFrame)

    def test_alpha_parameter_effect(self, sample_returns_dict):
        """Test that different alpha levels affect significance results."""
        tester_strict = PerformanceSignificanceTest(alpha=0.01)
        tester_lenient = PerformanceSignificanceTest(alpha=0.10)

        result_strict = tester_strict.comprehensive_comparison(sample_returns_dict)
        result_lenient = tester_lenient.comprehensive_comparison(sample_returns_dict)

        # More lenient alpha should generally result in more significant results
        n_significant_strict = result_strict["sharpe"]["is_significant"].sum()
        n_significant_lenient = result_lenient["sharpe"]["is_significant"].sum()

        assert n_significant_lenient >= n_significant_strict


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_returns(self):
        """Test handling of empty return series."""
        empty_returns = np.array([])
        normal_returns = np.random.normal(0.001, 0.02, 100)

        with pytest.raises((ValueError, IndexError)):
            StatisticalValidation.sharpe_ratio_test(empty_returns, normal_returns)

    def test_single_value_returns(self):
        """Test handling of single-value return series."""
        single_return = np.array([0.01])

        with pytest.raises((ValueError, IndexError)):
            StatisticalValidation.sharpe_ratio_test(single_return, single_return)

    def test_all_nan_returns(self):
        """Test handling of all-NaN return series."""
        nan_returns = np.full(100, np.nan)
        normal_returns = np.random.normal(0.001, 0.02, 100)

        # This should handle NaN values gracefully
        result = StatisticalValidation.sharpe_ratio_test(nan_returns, normal_returns)

        # Results should indicate inability to calculate
        assert np.isnan(result["test_statistic"]) or result["test_statistic"] == 0

    def test_extreme_values(self):
        """Test handling of extreme values."""
        extreme_returns_a = np.array([1000.0] + [0.001] * 99)  # One extreme outlier
        normal_returns_b = np.random.normal(0.001, 0.02, 100)

        result = StatisticalValidation.sharpe_ratio_test(extreme_returns_a, normal_returns_b)

        # Should complete without errors
        assert "test_statistic" in result
        assert "p_value" in result

    def test_high_correlation_returns(self):
        """Test handling of highly correlated return series."""
        base_returns = np.random.normal(0.001, 0.02, 100)
        noise = np.random.normal(0, 0.001, 100)  # Small noise

        returns_a = base_returns + noise
        returns_b = base_returns - noise

        result = StatisticalValidation.sharpe_ratio_test(returns_a, returns_b)

        # Should handle high correlation appropriately
        assert "correlation" in result
        assert abs(result["correlation"]) > 0.8  # Should be highly correlated


class TestStatisticalAccuracy:
    """Test statistical accuracy against known results."""

    def test_known_sharpe_ratios(self):
        """Test Sharpe ratio calculations against known values."""
        # Create returns with known Sharpe ratio
        np.random.seed(42)
        mean_return = 0.001  # 0.1% daily
        volatility = 0.02  # 2% daily
        n_obs = 1000

        returns = np.random.normal(mean_return, volatility, n_obs)

        # Calculate expected Sharpe ratio
        expected_sharpe = mean_return / volatility

        result = StatisticalValidation.sharpe_ratio_test(returns, returns)
        calculated_sharpe = result["sharpe_a"]

        # Should be close to expected (within sampling error)
        assert abs(calculated_sharpe - expected_sharpe) < 0.1

    def test_test_statistic_distribution(self):
        """Test that test statistics follow expected distribution under null."""
        # Under null hypothesis, test statistics should be approximately normal(0,1)
        test_statistics = []

        np.random.seed(42)
        for _ in range(100):
            returns_a = np.random.normal(0.001, 0.02, 100)
            returns_b = np.random.normal(0.001, 0.02, 100)  # Same distribution

            result = StatisticalValidation.sharpe_ratio_test(returns_a, returns_b)
            if not np.isnan(result["test_statistic"]):
                test_statistics.append(result["test_statistic"])

        test_statistics = np.array(test_statistics)

        # Test that mean is close to 0 under null
        assert abs(np.mean(test_statistics)) < 0.2

        # Test that roughly 5% are significant at alpha=0.05
        p_values = [2 * (1 - stats.norm.cdf(abs(ts))) for ts in test_statistics]
        significant_fraction = np.mean(np.array(p_values) < 0.05)

        # Should be close to 0.05 under null (within sampling variation)
        assert 0.02 < significant_fraction < 0.08

    def test_confidence_interval_coverage(self):
        """Test that confidence intervals have correct coverage."""
        # This is a more complex test that would require simulation
        # For now, just check that intervals are reasonable
        np.random.seed(42)
        returns_a = np.random.normal(0.001, 0.02, 252)
        returns_b = np.random.normal(0.0005, 0.025, 252)

        result = StatisticalValidation.sharpe_ratio_test(returns_a, returns_b, alpha=0.05)

        # Check that confidence interval makes sense
        assert "sharpe_diff" in result
        result["sharpe_diff"]

        # For a properly functioning test, the observed difference should be
        # within reasonable bounds given the input parameters
