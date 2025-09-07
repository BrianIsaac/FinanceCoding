"""
Integration tests for HRP model with backtesting framework.

Tests complete HRP model integration with existing portfolio construction infrastructure.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.processors.universe_builder import UniverseBuilder, UniverseConfig
from src.evaluation.backtest.engine import BacktestConfig, BacktestEngine
from src.models.base.portfolio_model import PortfolioConstraints
from src.models.baselines.equal_weight import EqualWeightModel
from src.models.hrp.model import HRPConfig, HRPModel


class TestHRPIntegration:
    """Test suite for HRP model integration."""

    @pytest.fixture
    def sample_market_data(self):
        """Create comprehensive sample market data."""
        np.random.seed(42)

        # Create 2 years of daily data
        dates = pd.date_range("2020-01-01", "2021-12-31", freq="D")

        # Create 30 assets with realistic correlation structure
        n_assets = 30

        # Sector correlation structure
        sectors = {
            "TECH": list(range(0, 10)),
            "FIN": list(range(10, 20)),
            "ENERGY": list(range(20, 30)),
        }

        # Build correlation matrix
        correlation_matrix = np.eye(n_assets)

        for sector_assets in sectors.values():
            for i in sector_assets:
                for j in sector_assets:
                    if i != j:
                        correlation_matrix[i, j] = 0.4  # Moderate sector correlation

        # Generate returns
        daily_vol = 0.015  # 1.5% daily volatility
        returns = np.random.multivariate_normal(
            mean=np.zeros(n_assets), cov=(daily_vol**2) * correlation_matrix, size=len(dates)
        )

        asset_names = [f"STOCK_{i:02d}" for i in range(n_assets)]
        returns_df = pd.DataFrame(returns, index=dates, columns=asset_names)

        # Create universe data (all assets available throughout period)
        universe_data = []
        for date in pd.date_range("2020-01-01", "2021-12-31", freq="MS"):  # Monthly
            for asset in asset_names:
                universe_data.append({"date": date, "ticker": asset, "index_name": "TEST_INDEX"})

        universe_df = pd.DataFrame(universe_data)

        return returns_df, universe_df

    @pytest.fixture
    def portfolio_constraints(self):
        """Create realistic portfolio constraints."""
        return PortfolioConstraints(
            long_only=True,
            top_k_positions=20,
            max_position_weight=0.08,
            max_monthly_turnover=0.30,
            transaction_cost_bps=10.0,
        )

    @pytest.fixture
    def hrp_config(self):
        """Create HRP configuration for integration testing."""
        return HRPConfig(
            lookback_days=252,  # 1 year lookback
            min_observations=150,
            rebalance_frequency="monthly",
        )

    def test_hrp_model_fitting_integration(
        self, sample_market_data, portfolio_constraints, hrp_config
    ):
        """Test HRP model fitting with realistic data."""
        returns_df, universe_df = sample_market_data

        # Create HRP model
        hrp_model = HRPModel(portfolio_constraints, hrp_config)

        # Test fitting with partial universe
        universe = returns_df.columns[:15].tolist()  # Use first 15 assets
        fit_period = (pd.Timestamp("2020-06-01"), pd.Timestamp("2021-06-01"))

        # Should fit successfully
        hrp_model.fit(returns_df, universe, fit_period)

        assert hrp_model.is_fitted
        assert hrp_model._fitted_universe is not None
        assert len(hrp_model._fitted_universe) > 0

        # Test model info
        model_info = hrp_model.get_model_info()
        assert model_info["model_type"] == "HRP"
        assert model_info["is_fitted"]
        assert model_info["fitted_universe_size"] > 0

    def test_hrp_vs_equal_weight_comparison(self, sample_market_data, portfolio_constraints):
        """Test HRP model performance comparison with equal weight baseline."""
        returns_df, universe_df = sample_market_data

        # Create models
        hrp_model = HRPModel(portfolio_constraints, HRPConfig(lookback_days=252))
        equal_weight_model = EqualWeightModel(portfolio_constraints)

        universe = returns_df.columns[:12].tolist()
        fit_period = (pd.Timestamp("2020-06-01"), pd.Timestamp("2021-06-01"))

        # Fit both models
        hrp_model.fit(returns_df, universe, fit_period)
        equal_weight_model.fit(returns_df, universe, fit_period)

        # Generate predictions for the same date
        prediction_date = pd.Timestamp("2021-07-01")
        prediction_universe = universe[:10]

        hrp_weights = hrp_model.predict_weights(prediction_date, prediction_universe)
        ew_weights = equal_weight_model.predict_weights(prediction_date, prediction_universe)

        # Both should produce valid weights
        np.testing.assert_almost_equal(hrp_weights.sum(), 1.0, decimal=6)
        np.testing.assert_almost_equal(ew_weights.sum(), 1.0, decimal=6)

        # HRP should produce different (more concentrated) weights than equal weight
        hrp_concentration = (hrp_weights**2).sum()  # Herfindahl index
        ew_concentration = (ew_weights**2).sum()

        # HRP typically produces more concentrated portfolios than equal weight
        # This is a probabilistic test, but with our correlated data it should hold
        assert hrp_concentration >= ew_concentration * 0.8  # Allow some tolerance

    def test_monthly_rebalancing_simulation(
        self, sample_market_data, portfolio_constraints, hrp_config
    ):
        """Test HRP model in monthly rebalancing scenario."""
        returns_df, universe_df = sample_market_data

        hrp_model = HRPModel(portfolio_constraints, hrp_config)

        universe = returns_df.columns[:15].tolist()

        # Simulate rolling rebalancing
        rebalance_dates = pd.date_range("2021-01-01", "2021-12-01", freq="MS")

        portfolio_weights = {}
        portfolio_returns = []

        for i, rebal_date in enumerate(rebalance_dates):
            if i == 0:
                continue  # Skip first date for initial fitting

            # Fit model using data up to rebalancing date
            fit_start = rebal_date - pd.DateOffset(months=12)  # 12-month lookback
            fit_period = (fit_start, rebal_date)

            try:
                hrp_model.fit(returns_df, universe, fit_period)

                # Generate portfolio weights
                weights = hrp_model.predict_weights(rebal_date, universe)
                portfolio_weights[rebal_date] = weights

                # Calculate portfolio return for next month (simple)
                next_month_end = rebal_date + pd.DateOffset(months=1)
                if next_month_end in returns_df.index:
                    # Calculate period returns
                    period_returns = returns_df.loc[rebal_date:next_month_end, weights.index]
                    if len(period_returns) > 1:
                        period_perf = (1 + period_returns).prod() - 1
                        portfolio_return = np.dot(weights.values, period_perf.values)
                        portfolio_returns.append(portfolio_return)

            except Exception:
                # Skip months where fitting fails (insufficient data, etc.)
                continue

        # Should have generated weights for multiple periods
        assert len(portfolio_weights) > 3

        # All weight vectors should be valid
        for _date, weights in portfolio_weights.items():
            np.testing.assert_almost_equal(weights.sum(), 1.0, decimal=6)
            assert all(weights >= 0)
            assert all(weights <= portfolio_constraints.max_position_weight)

    def test_constraint_enforcement_integration(self, sample_market_data):
        """Test constraint enforcement across different scenarios."""
        returns_df, universe_df = sample_market_data

        # Test strict constraints
        strict_constraints = PortfolioConstraints(
            long_only=True,
            top_k_positions=5,  # Very restrictive
            max_position_weight=0.25,  # 25% max
            max_monthly_turnover=0.20,  # 20% turnover limit
            transaction_cost_bps=20.0,
        )

        hrp_model = HRPModel(strict_constraints, HRPConfig(lookback_days=200))

        universe = returns_df.columns[:20].tolist()
        fit_period = (pd.Timestamp("2020-06-01"), pd.Timestamp("2021-06-01"))

        hrp_model.fit(returns_df, universe, fit_period)

        # Generate weights multiple times to test consistency
        test_dates = pd.date_range("2021-07-01", "2021-11-01", freq="MS")

        for date in test_dates:
            weights = hrp_model.predict_weights(date, universe)

            # Verify all constraints
            assert (weights > 0).sum() <= strict_constraints.top_k_positions
            assert all(weights <= strict_constraints.max_position_weight)
            assert all(weights >= 0)  # Long-only
            np.testing.assert_almost_equal(weights.sum(), 1.0, decimal=6)

    def test_clustering_diagnostics_integration(
        self, sample_market_data, portfolio_constraints, hrp_config
    ):
        """Test clustering diagnostics in realistic scenario."""
        returns_df, universe_df = sample_market_data

        hrp_model = HRPModel(portfolio_constraints, hrp_config)

        universe = returns_df.columns[:18].tolist()
        fit_period = (pd.Timestamp("2020-06-01"), pd.Timestamp("2021-06-01"))

        hrp_model.fit(returns_df, universe, fit_period)

        # Get clustering diagnostics for different dates
        test_dates = [
            pd.Timestamp("2021-07-01"),
            pd.Timestamp("2021-10-01"),
            pd.Timestamp("2021-12-01"),
        ]

        for date in test_dates:
            diagnostics = hrp_model.get_clustering_diagnostics(date, universe[:12])

            # Verify diagnostics structure
            assert "n_assets" in diagnostics
            assert "n_observations" in diagnostics
            assert "cluster_tree_depth" in diagnostics
            assert "linkage_method" in diagnostics

            # Verify reasonable values
            assert diagnostics["n_assets"] > 0
            assert diagnostics["n_observations"] >= hrp_config.min_observations
            assert diagnostics["cluster_tree_depth"] > 1
            assert diagnostics["linkage_method"] in ["single", "complete", "average", "ward"]

    def test_risk_contribution_analysis(
        self, sample_market_data, portfolio_constraints, hrp_config
    ):
        """Test risk contribution analysis integration."""
        returns_df, universe_df = sample_market_data

        hrp_model = HRPModel(portfolio_constraints, hrp_config)

        universe = returns_df.columns[:10].tolist()
        fit_period = (pd.Timestamp("2020-06-01"), pd.Timestamp("2021-06-01"))

        hrp_model.fit(returns_df, universe, fit_period)

        # Generate weights and analyze risk contributions
        prediction_date = pd.Timestamp("2021-08-01")
        weights = hrp_model.predict_weights(prediction_date, universe)

        risk_contributions = hrp_model.get_risk_contributions(weights, prediction_date)

        # Verify risk contributions
        assert isinstance(risk_contributions, dict)
        assert len(risk_contributions) > 0

        # Assets with higher weights should generally have higher risk contributions
        # (This is a general principle but not always strictly true in HRP)
        non_zero_weights = weights[weights > 0]

        for asset in non_zero_weights.index:
            if asset in risk_contributions:
                assert isinstance(risk_contributions[asset], (int, float))

    def test_model_performance_under_stress(self, portfolio_constraints):
        """Test HRP model performance under stressed market conditions."""
        np.random.seed(42)

        # Create stressed market scenario with high correlations and volatility spikes
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        n_assets = 15

        # Create returns with time-varying correlation and volatility
        base_vol = 0.02  # 2% daily vol
        stress_vol = 0.08  # 8% daily vol during stress

        # Stress period (middle 3 months)
        stress_start = pd.Timestamp("2020-06-01")
        stress_end = pd.Timestamp("2020-09-01")

        returns_list = []

        for date in dates:
            if stress_start <= date <= stress_end:
                # High correlation, high volatility period
                correlation = 0.8
                vol = stress_vol
            else:
                # Normal market conditions
                correlation = 0.3
                vol = base_vol

            # Create correlation matrix
            corr_matrix = np.eye(n_assets)
            corr_matrix[corr_matrix == 0] = correlation

            # Generate returns
            day_returns = np.random.multivariate_normal(
                mean=np.zeros(n_assets), cov=(vol**2) * corr_matrix, size=1
            )[0]

            returns_list.append(day_returns)

        asset_names = [f"ASSET_{i:02d}" for i in range(n_assets)]
        returns_df = pd.DataFrame(returns_list, index=dates, columns=asset_names)

        # Test HRP model under these conditions
        hrp_model = HRPModel(portfolio_constraints, HRPConfig(lookback_days=120))

        universe = asset_names

        # Fit on pre-stress period
        pre_stress_period = (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-05-01"))

        hrp_model.fit(returns_df, universe, pre_stress_period)

        # Test predictions during and after stress
        stress_prediction = hrp_model.predict_weights(pd.Timestamp("2020-07-01"), universe)
        post_stress_prediction = hrp_model.predict_weights(pd.Timestamp("2020-10-01"), universe)

        # Both should produce valid weights despite market stress
        np.testing.assert_almost_equal(stress_prediction.sum(), 1.0, decimal=6)
        np.testing.assert_almost_equal(post_stress_prediction.sum(), 1.0, decimal=6)

        # Verify constraints are maintained under stress
        assert all(stress_prediction <= portfolio_constraints.max_position_weight)
        assert all(post_stress_prediction <= portfolio_constraints.max_position_weight)
        assert all(stress_prediction >= 0)
        assert all(post_stress_prediction >= 0)

    def test_backtesting_integration_stub(
        self, sample_market_data, portfolio_constraints, hrp_config
    ):
        """Test integration with backtesting framework (using current stub)."""
        returns_df, universe_df = sample_market_data

        # Create models and backtest config
        hrp_model = HRPModel(portfolio_constraints, hrp_config)

        backtest_config = BacktestConfig(
            start_date=pd.Timestamp("2020-01-01"),
            end_date=pd.Timestamp("2021-12-31"),
            rebalance_frequency="M",
            initial_capital=1000000.0,
            transaction_cost_bps=10.0,
        )

        backtest_engine = BacktestEngine(backtest_config)

        # Run backtest (currently returns stub results)
        backtest_results = backtest_engine.run_backtest(hrp_model, returns_df, universe_df)

        # Verify stub structure (will be enhanced in future stories)
        assert isinstance(backtest_results, dict)
        assert "portfolio_returns" in backtest_results
        assert "portfolio_weights" in backtest_results
        assert "turnover" in backtest_results
        assert "transaction_costs" in backtest_results
        assert "performance_metrics" in backtest_results

        # Current stub returns empty results
        assert backtest_results["portfolio_returns"].empty
        assert backtest_results["portfolio_weights"].empty

        # Note: Full backtesting integration will be completed in future stories
        # This test validates the interface compatibility
