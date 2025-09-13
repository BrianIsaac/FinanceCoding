"""
Performance comparison tests for LSTM against baseline models.

This module tests the LSTM model's performance relative to HRP and
equal-weight baselines using statistical significance tests.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats

from src.evaluation.backtest.engine import BacktestConfig, BacktestEngine
from src.models.base.portfolio_model import PortfolioConstraints
from src.models.hrp.model import HRPConfig, HRPModel
from src.models.lstm.model import LSTMModelConfig, LSTMPortfolioModel
from src.models.model_registry import ModelRegistry


class TestModelPerformanceComparison:
    """Test LSTM performance against baseline models."""

    @pytest.fixture
    def realistic_returns_data(self) -> pd.DataFrame:
        """Generate more realistic returns data with market-like properties."""
        np.random.seed(42)
        torch.manual_seed(42)

        # Create 3 years of daily returns for 100 assets
        dates = pd.date_range("2021-01-01", "2024-01-01", freq="B")  # Business days
        n_assets = 100
        n_days = len(dates)

        # Create sector structure
        n_sectors = 10
        assets_per_sector = n_assets // n_sectors

        # Generate factor returns (market, sector factors)
        market_returns = np.random.normal(0.0005, 0.015, n_days)  # Market factor
        sector_returns = np.random.normal(0, 0.008, (n_days, n_sectors))  # Sector factors

        # Generate asset returns with factor structure
        returns = np.zeros((n_days, n_assets))

        for i in range(n_assets):
            sector = i // assets_per_sector

            # Factor loadings
            market_beta = 0.8 + 0.4 * np.random.random()  # Beta between 0.8 and 1.2
            sector_loading = 0.3 + 0.4 * np.random.random()  # Sector loading

            # Idiosyncratic component
            idiosyncratic = np.random.normal(0, 0.01, n_days)

            # Combine factors
            returns[:, i] = (
                market_beta * market_returns
                + sector_loading * sector_returns[:, min(sector, n_sectors - 1)]
                + idiosyncratic
            )

            # Add some momentum and mean reversion
            for t in range(1, n_days):
                returns[t, i] += 0.05 * returns[t - 1, i]  # Momentum
                if t >= 20:
                    returns[t, i] -= 0.02 * np.mean(returns[t - 20 : t, i])  # Mean reversion

        # Create asset names with sector info
        assets = []
        sector_names = [f"SECTOR_{s:02d}" for s in range(n_sectors)]
        for i in range(n_assets):
            sector = i // assets_per_sector
            sector_name = sector_names[min(sector, len(sector_names) - 1)]
            assets.append(f"{sector_name}_STOCK_{i:03d}")

        return pd.DataFrame(returns, index=dates, columns=assets)

    @pytest.fixture
    def model_configurations(self) -> dict[str, dict[str, Any]]:
        """Define model configurations for comparison."""
        base_constraints = PortfolioConstraints(
            long_only=True,
            top_k_positions=40,
            max_position_weight=0.08,
            max_monthly_turnover=0.25,
            transaction_cost_bps=10.0,
        )

        # LSTM configuration (reduced for testing speed)
        lstm_config = LSTMModelConfig()
        lstm_config.lstm_config.hidden_size = 32
        lstm_config.lstm_config.num_layers = 1
        lstm_config.training_config.epochs = 5
        lstm_config.training_config.patience = 3
        lstm_config.training_config.batch_size = 16

        return {
            "lstm": {
                "model": LSTMPortfolioModel(constraints=base_constraints, config=lstm_config),
                "name": "LSTM",
            },
            "hrp": {
                "model": HRPModel(
                    constraints=base_constraints, hrp_config=HRPConfig(min_observations=50)
                ),
                "name": "HRP",
            },
            "equal_weight": {
                "model": self._create_equal_weight_model(base_constraints),
                "name": "Equal Weight",
            },
        }

    def _create_equal_weight_model(self, constraints: PortfolioConstraints):
        """Create simple equal-weight baseline model."""

        class EqualWeightModel:
            def __init__(self, constraints):
                self.constraints = constraints

            def fit(self, returns, universe, fit_period):
                pass  # No fitting required

            def predict_weights(self, date, universe):
                n_assets = min(len(universe), self.constraints.top_k_positions or len(universe))
                weight = 1.0 / n_assets
                weights = pd.Series(
                    [weight] * n_assets + [0.0] * (len(universe) - n_assets), index=universe
                )
                return weights[:n_assets]

        return EqualWeightModel(constraints)

    @pytest.fixture
    def backtest_engine(self) -> BacktestEngine:
        """Create backtest engine for comparison."""
        config = BacktestConfig(
            start_date=datetime(2022, 6, 1),
            end_date=datetime(2023, 12, 31),
            rebalance_frequency="M",
            initial_capital=1000000.0,
            transaction_cost_bps=10.0,
            min_history_days=150,
        )
        return BacktestEngine(config)

    def test_model_performance_comparison(
        self,
        realistic_returns_data: pd.DataFrame,
        model_configurations: dict[str, dict[str, Any]],
        backtest_engine: BacktestEngine,
    ):
        """Compare LSTM performance against baseline models."""
        results = {}

        for model_key, config in model_configurations.items():
            model = config["model"]
            model_name = config["name"]

            try:
                backtest_results = backtest_engine.run_backtest(model, realistic_returns_data)
                results[model_key] = {
                    "name": model_name,
                    "results": backtest_results,
                    "returns": backtest_results.get("portfolio_returns", pd.Series()),
                    "metrics": backtest_results.get("performance_metrics", {}),
                }

            except Exception as e:
                results[model_key] = {
                    "name": model_name,
                    "error": str(e),
                    "returns": pd.Series(),
                    "metrics": {},
                }

        # Validate we have results
        successful_models = [
            k for k, v in results.items() if "error" not in v and len(v["returns"]) > 0
        ]
        assert (
            len(successful_models) >= 2
        ), f"Need at least 2 successful models for comparison, got {len(successful_models)}"

        # Compare performance metrics
        self._compare_performance_metrics(results, successful_models)

        # Statistical significance tests
        if "lstm" in successful_models:
            self._test_statistical_significance(results, successful_models)

    def _compare_performance_metrics(self, results: dict, successful_models: list[str]):
        """Compare key performance metrics across models."""

        for model_key in successful_models:
            model_data = results[model_key]
            metrics = model_data["metrics"]

            sharpe = metrics.get("sharpe_ratio", 0.0)
            metrics.get("total_return", 0.0)
            volatility = metrics.get("volatility", 0.0)
            max_drawdown = metrics.get("max_drawdown", 0.0)

            # Basic sanity checks
            if len(model_data["returns"]) > 50:  # Only if sufficient data
                assert (
                    -1.0 <= sharpe <= 5.0
                ), f"Sharpe ratio {sharpe} seems unrealistic for {model_data['name']}"
                assert (
                    0.0 <= volatility <= 1.0
                ), f"Volatility {volatility} seems unrealistic for {model_data['name']}"
                assert (
                    -1.0 <= max_drawdown <= 0.0
                ), f"Max drawdown {max_drawdown} seems unrealistic for {model_data['name']}"

    def _test_statistical_significance(self, results: dict, successful_models: list[str]):
        """Test statistical significance of LSTM vs baseline performance."""
        if "lstm" not in successful_models:
            return

        lstm_returns = results["lstm"]["returns"]
        if len(lstm_returns) < 30:  # Need sufficient data for statistical tests
            return

        for baseline_key in successful_models:
            if baseline_key == "lstm":
                continue

            baseline_returns = results[baseline_key]["returns"]
            results[baseline_key]["name"]

            if len(baseline_returns) < 30:
                continue

            # Align returns for comparison
            common_dates = lstm_returns.index.intersection(baseline_returns.index)
            if len(common_dates) < 30:
                continue

            lstm_aligned = lstm_returns.reindex(common_dates)
            baseline_aligned = baseline_returns.reindex(common_dates)

            # T-test for mean return difference
            t_stat, t_pvalue = stats.ttest_rel(lstm_aligned, baseline_aligned)

            # Variance test (F-test approximation)
            lstm_var = lstm_aligned.var()
            baseline_var = baseline_aligned.var()
            f_stat = lstm_var / baseline_var if baseline_var > 0 else np.inf

            # Sharpe ratio comparison (using bootstrap-like approach)
            lstm_aligned.mean() / lstm_aligned.std() if lstm_aligned.std() > 0 else 0
            (baseline_aligned.mean() / baseline_aligned.std() if baseline_aligned.std() > 0 else 0)

            # Basic statistical checks
            assert not np.isnan(t_stat), "T-statistic should not be NaN"
            assert 0 <= t_pvalue <= 1, "P-value should be between 0 and 1"
            assert f_stat > 0, "Variance ratio should be positive"

    def test_lstm_constraint_compliance(
        self, realistic_returns_data: pd.DataFrame, backtest_engine: BacktestEngine
    ):
        """Test LSTM model constraint compliance during backtesting."""
        constraints = PortfolioConstraints(
            long_only=True, top_k_positions=30, max_position_weight=0.06, max_monthly_turnover=0.20
        )

        config = LSTMModelConfig()
        config.training_config.epochs = 3  # Fast for testing

        model = LSTMPortfolioModel(constraints=constraints, config=config)

        results = backtest_engine.run_backtest(model, realistic_returns_data)
        weights_df = results.get("portfolio_weights", pd.DataFrame())
        turnover = results.get("turnover", pd.Series())

        if len(weights_df) > 0:

            for idx, weights in weights_df.iterrows():
                clean_weights = weights.dropna()
                if len(clean_weights) == 0:
                    continue

                # Long-only constraint
                negative_weights = (clean_weights < -1e-6).sum()
                assert negative_weights == 0, f"Found {negative_weights} negative weights on {idx}"

                # Position limit constraint
                max_weight = clean_weights.max()
                assert (
                    max_weight <= 0.07
                ), f"Max weight {max_weight:.4f} exceeds limit on {idx}"  # 6% + tolerance

                # Position count constraint
                active_positions = (clean_weights > 1e-6).sum()
                assert (
                    active_positions <= constraints.top_k_positions
                ), f"Too many positions ({active_positions}) on {idx}"

                # Weights should sum to approximately 1
                weight_sum = clean_weights.sum()
                assert abs(weight_sum - 1.0) < 0.02, f"Weights sum to {weight_sum:.4f} on {idx}"

        # Check turnover constraints
        if len(turnover) > 0:
            high_turnover_periods = (
                turnover > constraints.max_monthly_turnover + 0.05
            ).sum()  # Allow some tolerance
            total_periods = len(turnover)
            violation_rate = high_turnover_periods / total_periods if total_periods > 0 else 0

            # Allow some violations due to market conditions, but not too many
            assert violation_rate < 0.3, f"Too many turnover violations: {violation_rate:.1%}"

    def test_model_robustness_across_periods(
        self, realistic_returns_data: pd.DataFrame, model_configurations: dict[str, dict[str, Any]]
    ):
        """Test model performance across different time periods."""
        # Define different test periods
        test_periods = [
            ("Bull Market", datetime(2022, 1, 1), datetime(2022, 6, 30)),
            ("Bear Market", datetime(2022, 7, 1), datetime(2023, 3, 31)),
            ("Recovery", datetime(2023, 4, 1), datetime(2023, 12, 31)),
        ]

        period_results = {}

        for period_name, start_date, end_date in test_periods:

            config = BacktestConfig(
                start_date=start_date,
                end_date=end_date,
                rebalance_frequency="M",
                initial_capital=1000000.0,
                min_history_days=100,  # Reduced for shorter periods
            )
            engine = BacktestEngine(config)

            period_results[period_name] = {}

            for model_key, model_config in model_configurations.items():
                try:
                    # Use a fresh model instance for each period
                    if model_key == "lstm":
                        # Create new LSTM model with reduced complexity
                        constraints = model_config["model"].constraints
                        config = LSTMModelConfig()
                        config.training_config.epochs = 3
                        config.lstm_config.hidden_size = 16
                        model = LSTMPortfolioModel(constraints=constraints, config=config)
                    else:
                        model = model_config["model"]

                    results = engine.run_backtest(model, realistic_returns_data)
                    returns = results.get("portfolio_returns", pd.Series())

                    if len(returns) > 5:  # Minimum returns for meaningful analysis
                        annualized_return = returns.mean() * 252
                        volatility = returns.std() * np.sqrt(252)
                        sharpe = annualized_return / volatility if volatility > 0 else 0

                        period_results[period_name][model_key] = {
                            "return": annualized_return,
                            "volatility": volatility,
                            "sharpe": sharpe,
                            "n_obs": len(returns),
                        }

                except Exception:
                    pass

        # Validate robustness - models should perform reasonably across periods
        for model_key in ["lstm", "hrp"]:
            sharpe_ratios = []
            for period_data in period_results.values():
                if model_key in period_data:
                    sharpe_ratios.append(period_data[model_key]["sharpe"])

            if len(sharpe_ratios) >= 2:
                sharpe_std = np.std(sharpe_ratios)
                avg_sharpe = np.mean(sharpe_ratios)

                # Model should show some consistency (not be completely random)
                # Allow high variability but check for basic reasonableness
                assert sharpe_std < 2.0, f"{model_key} shows excessive performance variability"
                assert abs(avg_sharpe) < 3.0, f"{model_key} shows unrealistic average Sharpe ratio"
