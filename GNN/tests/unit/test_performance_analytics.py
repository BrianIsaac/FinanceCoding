"""
Unit tests for performance analytics modules.

Tests the comprehensive performance analytics functionality including
returns, risk metrics, operational metrics, rolling analysis, attribution,
and benchmark comparison.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics.attribution import AttributionConfig, PerformanceAttributionAnalyzer
from src.evaluation.metrics.benchmarks import BenchmarkComparator, BenchmarkConfig
from src.evaluation.metrics.operational import OperationalAnalytics, OperationalMetricsConfig
from src.evaluation.metrics.returns import PerformanceAnalytics, ReturnAnalyzer, ReturnMetricsConfig
from src.evaluation.metrics.risk import RiskAnalytics, RiskMetricsConfig
from src.evaluation.metrics.rolling_analysis import (
    RollingAnalysisConfig,
    RollingPerformanceAnalyzer,
)


class TestPerformanceAnalytics:
    """Test suite for PerformanceAnalytics class."""

    @pytest.fixture
    def sample_returns(self):
        """Generate sample return data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
        returns = np.random.normal(0.0005, 0.015, len(dates))  # Daily returns with realistic params
        return pd.Series(returns, index=dates)

    @pytest.fixture
    def performance_analytics(self):
        """Create PerformanceAnalytics instance for testing."""
        config = ReturnMetricsConfig(risk_free_rate=0.02, trading_days_per_year=252)
        return PerformanceAnalytics(config)

    def test_calculate_portfolio_metrics_basic(self, performance_analytics, sample_returns):
        """Test basic portfolio metrics calculation."""
        metrics = performance_analytics.calculate_portfolio_metrics(sample_returns)

        # Check that all expected metrics are present
        expected_metrics = [
            "total_return",
            "annualized_return",
            "volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "var_95",
            "cvar_95",
            "win_rate",
            "profit_factor",
            "skewness",
            "kurtosis",
            "calmar_ratio",
            "num_observations",
        ]

        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"

        # Basic sanity checks
        assert isinstance(metrics["total_return"], (int, float))
        assert metrics["num_observations"] == len(sample_returns)
        assert 0 <= metrics["win_rate"] <= 1

    def test_calculate_portfolio_metrics_empty_data(self, performance_analytics):
        """Test metrics calculation with empty data."""
        empty_returns = pd.Series([], dtype=float)
        metrics = performance_analytics.calculate_portfolio_metrics(empty_returns)

        assert metrics == {}

    def test_calculate_portfolio_metrics_insufficient_data(self, performance_analytics):
        """Test metrics calculation with insufficient data."""
        short_returns = pd.Series([0.01], index=[pd.Timestamp("2020-01-01")])
        metrics = performance_analytics.calculate_portfolio_metrics(short_returns)

        assert "error" in metrics


class TestReturnAnalyzer:
    """Test suite for ReturnAnalyzer class."""

    @pytest.fixture
    def return_analyzer(self):
        """Create ReturnAnalyzer instance for testing."""
        config = ReturnMetricsConfig()
        return ReturnAnalyzer(config)

    @pytest.fixture
    def sample_returns(self):
        """Generate sample return data."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", "2021-12-31", freq="D")
        returns = np.random.normal(0.0008, 0.012, len(dates))
        return pd.Series(returns, index=dates)

    def test_calculate_basic_metrics(self, return_analyzer, sample_returns):
        """Test basic metrics calculation."""
        metrics = return_analyzer.calculate_basic_metrics(sample_returns)

        expected_keys = [
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "calmar_ratio",
            "skewness",
            "kurtosis",
            "var_95",
            "cvar_95",
        ]

        for key in expected_keys:
            assert key in metrics

        # Sanity checks
        assert metrics["annualized_volatility"] > 0
        assert isinstance(metrics["sharpe_ratio"], (int, float))

    def test_calculate_maximum_drawdown(self, return_analyzer, sample_returns):
        """Test maximum drawdown calculation."""
        max_dd, peak_date, trough_date = return_analyzer.calculate_maximum_drawdown(sample_returns)

        assert isinstance(max_dd, (int, float))
        assert max_dd <= 0  # Drawdown should be negative or zero

        if max_dd < 0:  # If there's a drawdown, dates should be provided
            assert isinstance(peak_date, pd.Timestamp)
            assert isinstance(trough_date, pd.Timestamp)
            assert peak_date <= trough_date

    def test_calculate_sharpe_ratio(self, return_analyzer, sample_returns):
        """Test Sharpe ratio calculation."""
        sharpe = return_analyzer.calculate_sharpe_ratio(sample_returns)

        assert isinstance(sharpe, (int, float))
        assert not np.isnan(sharpe)


class TestRiskAnalytics:
    """Test suite for RiskAnalytics class."""

    @pytest.fixture
    def risk_analytics(self):
        """Create RiskAnalytics instance for testing."""
        return RiskAnalytics()

    @pytest.fixture
    def sample_data(self):
        """Generate sample portfolio and benchmark returns."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")

        portfolio_returns = pd.Series(np.random.normal(0.0008, 0.015, len(dates)), index=dates)
        benchmark_returns = pd.Series(np.random.normal(0.0006, 0.012, len(dates)), index=dates)

        return portfolio_returns, benchmark_returns

    def test_calculate_tracking_error(self, risk_analytics, sample_data):
        """Test tracking error calculation."""
        portfolio_returns, benchmark_returns = sample_data

        tracking_error = risk_analytics.calculate_tracking_error(
            portfolio_returns, benchmark_returns
        )

        assert isinstance(tracking_error, (int, float))
        assert tracking_error >= 0

    def test_calculate_information_ratio(self, risk_analytics, sample_data):
        """Test Information Ratio calculation."""
        portfolio_returns, benchmark_returns = sample_data

        info_ratio = risk_analytics.calculate_information_ratio(
            portfolio_returns, benchmark_returns
        )

        assert isinstance(info_ratio, (int, float))
        assert not np.isnan(info_ratio)

    def test_calculate_win_rate_analysis(self, risk_analytics, sample_data):
        """Test win rate analysis."""
        portfolio_returns, benchmark_returns = sample_data

        win_rates = risk_analytics.calculate_win_rate_analysis(portfolio_returns, benchmark_returns)

        # Check that win rates are calculated
        expected_keys = ["monthly_win_rate", "quarterly_win_rate", "annual_win_rate"]
        for key in expected_keys:
            assert key in win_rates
            assert 0 <= win_rates[key] <= 1

    def test_calculate_downside_deviation(self, risk_analytics, sample_data):
        """Test downside deviation calculation."""
        portfolio_returns, _ = sample_data

        downside_dev = risk_analytics.calculate_downside_deviation(portfolio_returns)

        assert isinstance(downside_dev, (int, float))
        assert downside_dev >= 0

    def test_comprehensive_risk_metrics(self, risk_analytics, sample_data):
        """Test comprehensive risk metrics calculation."""
        portfolio_returns, benchmark_returns = sample_data

        risk_metrics = risk_analytics.calculate_comprehensive_risk_metrics(
            portfolio_returns, benchmark_returns
        )

        # Check required components
        assert "var_metrics" in risk_metrics
        assert "win_rates" in risk_metrics
        assert "tracking_error" in risk_metrics
        assert "information_ratio" in risk_metrics


class TestOperationalAnalytics:
    """Test suite for OperationalAnalytics class."""

    @pytest.fixture
    def operational_analytics(self):
        """Create OperationalAnalytics instance for testing."""
        return OperationalAnalytics()

    @pytest.fixture
    def sample_portfolio_data(self):
        """Generate sample portfolio weights and returns."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", "2022-12-31", freq="ME")
        assets = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

        # Create random portfolio weights that sum to 1
        weights_data = np.random.dirichlet(np.ones(len(assets)), len(dates))
        portfolio_weights = pd.DataFrame(weights_data, index=dates, columns=assets)

        # Create portfolio returns
        portfolio_returns = pd.Series(np.random.normal(0.01, 0.05, len(dates)), index=dates)

        return portfolio_weights, portfolio_returns

    def test_calculate_monthly_turnover_metrics(self, operational_analytics, sample_portfolio_data):
        """Test monthly turnover metrics calculation."""
        portfolio_weights, portfolio_returns = sample_portfolio_data

        turnover_metrics = operational_analytics.calculate_monthly_turnover_metrics(
            portfolio_weights, portfolio_returns
        )

        # Check structure
        assert "turnover_statistics" in turnover_metrics
        assert "turnover_history" in turnover_metrics
        assert "asset_analysis" in turnover_metrics

        # Check statistics
        stats = turnover_metrics["turnover_statistics"]
        assert "avg_monthly_turnover" in stats
        assert stats["avg_monthly_turnover"] >= 0

    def test_calculate_constraint_compliance(self, operational_analytics, sample_portfolio_data):
        """Test constraint compliance calculation."""
        portfolio_weights, _ = sample_portfolio_data

        compliance = operational_analytics.calculate_constraint_compliance(portfolio_weights)

        assert "constraint_compliance" in compliance
        assert "overall_compliance_rate" in compliance
        assert 0 <= compliance["overall_compliance_rate"] <= 1

    def test_calculate_operational_efficiency(self, operational_analytics, sample_portfolio_data):
        """Test operational efficiency calculation."""
        portfolio_weights, portfolio_returns = sample_portfolio_data

        efficiency = operational_analytics.calculate_operational_efficiency(
            portfolio_weights, portfolio_returns
        )

        expected_keys = [
            "return_per_turnover",
            "sharpe_per_turnover",
            "avg_implementation_cost",
            "rebalancing_frequency",
            "efficiency_score",
        ]

        for key in expected_keys:
            assert key in efficiency


class TestRollingPerformanceAnalyzer:
    """Test suite for RollingPerformanceAnalyzer class."""

    @pytest.fixture
    def rolling_analyzer(self):
        """Create RollingPerformanceAnalyzer instance for testing."""
        return RollingPerformanceAnalyzer()

    @pytest.fixture
    def sample_returns(self):
        """Generate sample return data with sufficient length."""
        np.random.seed(42)
        dates = pd.date_range("2018-01-01", "2023-12-31", freq="D")
        returns = np.random.normal(0.0005, 0.015, len(dates))
        return pd.Series(returns, index=dates)

    def test_calculate_rolling_metrics(self, rolling_analyzer, sample_returns):
        """Test rolling metrics calculation."""
        rolling_metrics = rolling_analyzer.calculate_rolling_metrics(
            sample_returns, window_size=252
        )

        assert isinstance(rolling_metrics, pd.DataFrame)
        if not rolling_metrics.empty:
            # Check that key metrics are present
            expected_columns = ["annualized_return", "annualized_volatility", "sharpe_ratio"]
            for col in expected_columns:
                assert col in rolling_metrics.columns

    def test_calculate_time_varying_performance(self, rolling_analyzer, sample_returns):
        """Test time-varying performance analysis."""
        time_varying = rolling_analyzer.calculate_time_varying_performance(sample_returns)

        assert isinstance(time_varying, dict)
        # Should have results for different window sizes
        assert len(time_varying) > 0

    def test_detect_market_regimes(self, rolling_analyzer, sample_returns):
        """Test market regime detection."""
        regime_df, regime_performance = rolling_analyzer.detect_market_regimes(sample_returns)

        assert isinstance(regime_df, pd.DataFrame)
        assert isinstance(regime_performance, dict)

        # Check regime classification columns
        assert "returns" in regime_df.columns
        assert "regime" in regime_df.columns

    def test_calculate_performance_stability(self, rolling_analyzer, sample_returns):
        """Test performance stability analysis."""
        stability = rolling_analyzer.calculate_performance_stability(sample_returns)

        assert isinstance(stability, dict)
        # Should include stability metrics
        expected_keys = ["sharpe_stability", "volatility_stability", "return_stability"]
        for key in expected_keys:
            if key in stability:  # Some keys might not be present with insufficient data
                assert isinstance(stability[key], (int, float))


class TestPerformanceAttributionAnalyzer:
    """Test suite for PerformanceAttributionAnalyzer class."""

    @pytest.fixture
    def attribution_analyzer(self):
        """Create PerformanceAttributionAnalyzer instance for testing."""
        return PerformanceAttributionAnalyzer()

    @pytest.fixture
    def sample_factor_data(self):
        """Generate sample portfolio returns and factor returns."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")

        # Portfolio returns
        portfolio_returns = pd.Series(np.random.normal(0.0008, 0.015, len(dates)), index=dates)

        # Factor returns
        factors = ["market", "size", "value", "momentum"]
        factor_data = {}
        for factor in factors:
            factor_data[factor] = np.random.normal(0.0005, 0.01, len(dates))

        factor_returns = pd.DataFrame(factor_data, index=dates)

        # Benchmark returns
        benchmark_returns = pd.Series(np.random.normal(0.0006, 0.012, len(dates)), index=dates)

        return portfolio_returns, factor_returns, benchmark_returns

    def test_calculate_factor_based_attribution(self, attribution_analyzer, sample_factor_data):
        """Test factor-based attribution analysis."""
        portfolio_returns, factor_returns, benchmark_returns = sample_factor_data

        attribution = attribution_analyzer.calculate_factor_based_attribution(
            portfolio_returns, factor_returns, benchmark_returns
        )

        if "error" not in attribution:
            # Check required components
            assert "factor_exposures" in attribution
            assert "factor_contributions" in attribution
            assert "alpha" in attribution
            assert "r_squared" in attribution

    def test_decompose_alpha_beta(self, attribution_analyzer, sample_factor_data):
        """Test alpha/beta decomposition."""
        portfolio_returns, _, benchmark_returns = sample_factor_data

        decomposition = attribution_analyzer.decompose_alpha_beta(
            portfolio_returns, benchmark_returns
        )

        if "error" not in decomposition:
            # Check required metrics
            assert "market_beta" in decomposition
            assert "market_alpha" in decomposition
            assert "jensen_alpha" in decomposition
            assert "information_ratio" in decomposition

    def test_track_risk_factor_exposures(self, attribution_analyzer, sample_factor_data):
        """Test risk factor exposure tracking."""
        portfolio_returns, factor_returns, _ = sample_factor_data

        exposures = attribution_analyzer.track_risk_factor_exposures(
            portfolio_returns, factor_returns, window_size=252
        )

        assert isinstance(exposures, pd.DataFrame)
        if not exposures.empty:
            # Should have exposure columns for each factor
            for factor in factor_returns.columns:
                exposure_col = f"{factor}_exposure"
                assert exposure_col in exposures.columns or len(exposures) == 0


class TestBenchmarkComparator:
    """Test suite for BenchmarkComparator class."""

    @pytest.fixture
    def benchmark_comparator(self):
        """Create BenchmarkComparator instance for testing."""
        return BenchmarkComparator()

    @pytest.fixture
    def sample_benchmark_data(self):
        """Generate sample data for benchmark testing."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")

        # Portfolio returns
        portfolio_returns = pd.Series(np.random.normal(0.0008, 0.015, len(dates)), index=dates)

        # Benchmark returns
        sp_midcap_returns = pd.Series(np.random.normal(0.0006, 0.012, len(dates)), index=dates)

        # Asset returns for constructing benchmarks
        assets = [f"ASSET_{i:03d}" for i in range(10)]
        asset_returns_data = {}
        for asset in assets:
            asset_returns_data[asset] = np.random.normal(0.0005, 0.018, len(dates))

        asset_returns = pd.DataFrame(asset_returns_data, index=dates)

        return portfolio_returns, sp_midcap_returns, asset_returns

    def test_integrate_sp_midcap_tracking(self, benchmark_comparator, sample_benchmark_data):
        """Test S&P MidCap 400 tracking and comparison."""
        portfolio_returns, sp_midcap_returns, _ = sample_benchmark_data

        comparison = benchmark_comparator.integrate_sp_midcap_tracking(
            sp_midcap_returns, portfolio_returns
        )

        if "error" not in comparison:
            assert "tracking_metrics" in comparison
            assert "performance_comparison" in comparison
            assert "attribution" in comparison

    def test_create_equal_weight_benchmark(self, benchmark_comparator, sample_benchmark_data):
        """Test equal-weight benchmark creation."""
        _, _, asset_returns = sample_benchmark_data

        # Create simple universe (all assets available)
        asset_universe = asset_returns.notna()

        equal_weight = benchmark_comparator.create_equal_weight_benchmark(
            asset_universe, asset_returns
        )

        if "error" not in equal_weight:
            assert "returns" in equal_weight
            assert "performance_metrics" in equal_weight
            assert isinstance(equal_weight["returns"], pd.Series)

    def test_create_multi_benchmark_ranking(self, benchmark_comparator, sample_benchmark_data):
        """Test multi-benchmark ranking."""
        portfolio_returns, sp_midcap_returns, _ = sample_benchmark_data

        benchmarks = {
            "S&P MidCap 400": sp_midcap_returns,
            "Random Benchmark": pd.Series(
                np.random.normal(0.0004, 0.014, len(portfolio_returns)),
                index=portfolio_returns.index,
            ),
        }

        ranking = benchmark_comparator.create_multi_benchmark_ranking(portfolio_returns, benchmarks)

        if "error" not in ranking:
            assert "performance_metrics" in ranking
            assert "rankings" in ranking
            assert "pairwise_comparisons" in ranking


if __name__ == "__main__":
    pytest.main([__file__])
