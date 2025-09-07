"""
Integration tests for end-to-end portfolio construction.

Tests validate the complete workflow from model fitting through
rebalancing to performance calculation and export.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest.rebalancing import PortfolioRebalancer, RebalancingConfig
from src.evaluation.backtest.transaction_costs import (
    TransactionCostCalculator,
    TransactionCostConfig,
)
from src.evaluation.metrics.returns import ReturnAnalyzer, ReturnMetricsConfig
from src.evaluation.reporting.export import ExportConfig, PortfolioExporter
from src.models.base.portfolio_model import PortfolioConstraints
from src.models.baselines.equal_weight import EqualWeightModel


class TestPortfolioConstructionIntegration:
    """Integration tests for complete portfolio construction workflow."""

    @pytest.fixture
    def sample_universe(self):
        """Create sample S&P MidCap 400 style universe."""
        return [f"MIDCAP_{i:03d}" for i in range(100)]

    @pytest.fixture
    def sample_returns_data(self, sample_universe):
        """Create sample returns data for backtesting."""
        # Generate 2 years of daily returns data
        start_date = pd.Timestamp("2022-01-01")
        end_date = pd.Timestamp("2023-12-31")
        dates = pd.date_range(start=start_date, end=end_date, freq="B")  # Business days

        np.random.seed(42)  # For reproducibility

        # Generate correlated returns with some market factor
        market_factor = np.random.normal(0.0008, 0.015, len(dates))

        returns_data = {}
        for asset in sample_universe:
            # Each asset has exposure to market factor plus idiosyncratic noise
            beta = np.random.uniform(0.7, 1.3)  # Random beta
            idiosyncratic = np.random.normal(0.0, 0.01, len(dates))
            asset_returns = beta * market_factor + idiosyncratic
            returns_data[asset] = asset_returns

        return pd.DataFrame(returns_data, index=dates)

    @pytest.fixture
    def portfolio_constraints(self):
        """Create realistic portfolio constraints."""
        return PortfolioConstraints(
            long_only=True,
            top_k_positions=50,
            max_position_weight=0.05,  # 5% max position
            max_monthly_turnover=0.25,
            transaction_cost_bps=10.0,
            min_weight_threshold=0.005,  # 0.5% minimum
        )

    @pytest.fixture
    def mock_universe_calendar(self, sample_universe):
        """Create mock universe calendar for testing."""

        class MockUniverseCalendar:
            def __init__(self, universe):
                self.universe = universe

            def get_active_tickers(self, date):
                # Return full universe (simplified for testing)
                return self.universe

            def is_active_date(self, date):
                # All dates are active (simplified)
                return True

        return MockUniverseCalendar(sample_universe)

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_end_to_end_equal_weight_portfolio(
        self,
        sample_returns_data,
        sample_universe,
        portfolio_constraints,
        mock_universe_calendar,
        temp_output_dir,
    ):
        """Test complete equal-weight portfolio construction workflow."""

        # 1. Initialize equal-weight model
        model = EqualWeightModel(constraints=portfolio_constraints, top_k=50)

        # 2. Fit model on training data (first year)
        train_start = sample_returns_data.index[0]
        train_end = sample_returns_data.index[len(sample_returns_data) // 2]

        model.fit(
            returns=sample_returns_data,
            universe=sample_universe,
            fit_period=(train_start, train_end),
        )

        assert model.is_fitted
        assert len(model.fitted_universe) > 0

        # 3. Setup rebalancing configuration
        transaction_config = TransactionCostConfig(linear_cost_bps=10.0)
        rebalancing_config = RebalancingConfig(
            frequency="monthly",
            transaction_cost_config=transaction_config,
            enable_transaction_costs=True,
        )

        rebalancer = PortfolioRebalancer(rebalancing_config)

        # 4. Generate rebalancing schedule
        backtest_start = train_end + pd.Timedelta(days=1)
        backtest_end = sample_returns_data.index[-1]

        rebalance_dates = rebalancer.generate_rebalancing_schedule(
            backtest_start, backtest_end, mock_universe_calendar
        )

        assert len(rebalance_dates) > 0

        # 5. Execute rebalancing workflow
        portfolio_weights_history = []
        current_weights = pd.Series(dtype=float)

        for rebalance_date in rebalance_dates[:3]:  # Test first 3 rebalances
            universe_at_date = mock_universe_calendar.get_active_tickers(rebalance_date)

            new_weights, rebalancing_info = rebalancer.execute_rebalancing(
                rebalance_date=rebalance_date,
                model=model,
                current_weights=current_weights,
                universe=universe_at_date,
                portfolio_value=1000000.0,
            )

            # Validate rebalancing results
            assert isinstance(new_weights, pd.Series)
            assert abs(new_weights.sum() - 1.0) < 1e-10  # Weights sum to 1
            assert all(new_weights >= 0)  # Long-only
            assert (new_weights > 0).sum() <= portfolio_constraints.top_k_positions

            # Store for analysis
            portfolio_weights_history.append(
                {
                    "date": rebalance_date,
                    "weights": new_weights.to_dict(),
                }
            )

            current_weights = new_weights

        assert len(portfolio_weights_history) == 3

        # 6. Calculate portfolio returns (simplified)
        portfolio_returns = []
        for i, weights_entry in enumerate(portfolio_weights_history):
            # Calculate period return using next month's data
            period_start = weights_entry["date"]
            if i < len(portfolio_weights_history) - 1:
                period_end = portfolio_weights_history[i + 1]["date"]
            else:
                period_end = period_start + pd.DateOffset(months=1)

            # Get returns for this period
            period_mask = (sample_returns_data.index >= period_start) & (
                sample_returns_data.index < period_end
            )
            period_returns = sample_returns_data.loc[period_mask]

            if not period_returns.empty:
                # Calculate weighted portfolio return
                weights = pd.Series(weights_entry["weights"])
                common_assets = weights.index.intersection(period_returns.columns)

                if len(common_assets) > 0:
                    period_portfolio_returns = (
                        period_returns[common_assets]
                        .multiply(weights[common_assets], axis=1)
                        .sum(axis=1)
                    )

                    total_period_return = period_portfolio_returns.sum()
                    portfolio_returns.append(total_period_return)

        # 7. Performance analysis
        if portfolio_returns:
            returns_series = pd.Series(portfolio_returns)
            analyzer = ReturnAnalyzer(ReturnMetricsConfig())

            performance_metrics = analyzer.calculate_basic_metrics(returns_series)

            # Validate performance metrics
            assert "annualized_return" in performance_metrics
            assert "annualized_volatility" in performance_metrics
            assert "sharpe_ratio" in performance_metrics
            assert "max_drawdown" in performance_metrics

            # Basic sanity checks
            assert isinstance(performance_metrics["annualized_return"], float)
            assert performance_metrics["annualized_volatility"] >= 0

        # 8. Export functionality
        export_config = ExportConfig(
            output_directory=temp_output_dir,
            export_formats=["parquet", "csv", "json"],
        )
        exporter = PortfolioExporter(export_config)

        # Export portfolio weights
        exported_files = exporter.export_portfolio_weights(
            portfolio_weights_history, filename="test_portfolio_weights"
        )

        assert "parquet" in exported_files
        assert "csv" in exported_files
        assert "json" in exported_files

        # Verify files exist
        for file_path in exported_files.values():
            assert Path(file_path).exists()

        # Export rebalancing history
        rebalancing_files = exporter.export_rebalancing_schedule(
            rebalancer.rebalancing_history, filename="test_rebalancing"
        )

        for file_path in rebalancing_files.values():
            assert Path(file_path).exists()

        # Create analytics dashboard
        dashboard_path = exporter.create_portfolio_analytics_dashboard(
            weights_history=portfolio_weights_history,
            returns_history=pd.Series(portfolio_returns) if portfolio_returns else None,
            rebalancing_history=rebalancer.rebalancing_history,
            model_info=model.get_model_info(),
        )

        assert Path(dashboard_path).exists()

        # Validate dashboard content
        with open(dashboard_path) as f:
            dashboard_data = json.load(f)

        assert "metadata" in dashboard_data
        assert "summary_statistics" in dashboard_data
        assert "model_info" in dashboard_data["metadata"]
        assert dashboard_data["metadata"]["model_info"]["model_type"] == "EqualWeight"

    def test_portfolio_constraint_enforcement_integration(
        self, sample_returns_data, sample_universe, temp_output_dir
    ):
        """Test that constraints are properly enforced throughout the workflow."""

        # Create strict constraints
        strict_constraints = PortfolioConstraints(
            long_only=True,
            top_k_positions=20,  # Only 20 positions
            max_position_weight=0.08,  # Max 8% per position
            max_monthly_turnover=0.15,  # Lower turnover limit
            transaction_cost_bps=15.0,
        )

        model = EqualWeightModel(constraints=strict_constraints, top_k=20)

        # Fit model
        train_start = sample_returns_data.index[0]
        train_end = sample_returns_data.index[len(sample_returns_data) // 3]

        model.fit(
            returns=sample_returns_data,
            universe=sample_universe[:50],  # Smaller universe
            fit_period=(train_start, train_end),
        )

        # Generate weights
        test_date = train_end + pd.Timedelta(days=30)
        weights = model.predict_weights(test_date, sample_universe[:50])

        # Validate constraint enforcement
        assert all(weights >= 0), "Long-only constraint violated"
        assert (weights > 0).sum() <= 20, "Top-k constraint violated"
        assert weights.max() <= 0.08, "Max position weight constraint violated"
        assert abs(weights.sum() - 1.0) < 1e-10, "Weights don't sum to 1"

    def test_transaction_cost_impact_integration(
        self, sample_returns_data, sample_universe, portfolio_constraints
    ):
        """Test that transaction costs properly impact portfolio performance."""

        model = EqualWeightModel(constraints=portfolio_constraints)

        # Fit model
        train_period = (sample_returns_data.index[0], sample_returns_data.index[100])
        model.fit(sample_returns_data, sample_universe[:30], train_period)

        # Create high-cost and low-cost configurations
        high_cost_config = TransactionCostConfig(linear_cost_bps=50.0)  # 0.5% per trade
        low_cost_config = TransactionCostConfig(linear_cost_bps=5.0)  # 0.05% per trade

        high_cost_calc = TransactionCostCalculator(high_cost_config)
        low_cost_calc = TransactionCostCalculator(low_cost_config)

        # Generate two different weight scenarios
        current_weights = pd.Series([0.1] * 10, index=sample_universe[:10])
        target_weights = model.predict_weights(sample_returns_data.index[150], sample_universe[:30])

        # Calculate costs for both scenarios
        high_costs = high_cost_calc.calculate_transaction_costs(current_weights, target_weights)
        low_costs = low_cost_calc.calculate_transaction_costs(current_weights, target_weights)

        # High cost should be significantly higher than low cost
        assert high_costs["total_cost"] > low_costs["total_cost"]
        assert high_costs["linear_cost"] > low_costs["linear_cost"]

        # Apply costs to returns
        mock_return = 0.02  # 2% period return

        high_cost_net, _ = high_cost_calc.apply_transaction_costs(
            mock_return, current_weights, target_weights
        )
        low_cost_net, _ = low_cost_calc.apply_transaction_costs(
            mock_return, current_weights, target_weights
        )

        # Net returns should reflect cost differences
        assert high_cost_net < low_cost_net
        assert high_cost_net < mock_return
        assert low_cost_net < mock_return

    def test_performance_metrics_integration(self, sample_returns_data):
        """Test integration of performance metrics calculation."""

        # Create sample portfolio returns
        np.random.seed(123)
        portfolio_returns = pd.Series(
            np.random.normal(0.0008, 0.012, 250),  # Daily returns for ~1 year
            index=pd.date_range("2022-01-01", periods=250, freq="B"),
        )

        # Initialize analyzer
        config = ReturnMetricsConfig(risk_free_rate=0.02)
        analyzer = ReturnAnalyzer(config)

        # Calculate comprehensive metrics
        basic_metrics = analyzer.calculate_basic_metrics(portfolio_returns)

        # Validate metrics are reasonable
        assert -0.5 < basic_metrics["annualized_return"] < 0.5  # Reasonable annual return range
        assert 0 < basic_metrics["annualized_volatility"] < 1.0  # Reasonable volatility
        assert basic_metrics["max_drawdown"] <= 0  # Drawdown should be negative or zero

        # Test maximum drawdown calculation specifically
        max_dd, peak_date, trough_date = analyzer.calculate_maximum_drawdown(portfolio_returns)

        assert max_dd <= 0  # Drawdown is negative
        if max_dd < 0:  # If there was a drawdown
            assert peak_date is not None
            assert trough_date is not None
            assert peak_date <= trough_date  # Peak comes before trough

    def test_export_functionality_integration(self, temp_output_dir):
        """Test comprehensive export functionality."""

        # Create sample data
        dates = pd.date_range("2022-01-01", periods=12, freq="M")
        assets = [f"ASSET_{i}" for i in range(20)]

        weights_history = []
        for date in dates:
            # Generate random weights that sum to 1
            raw_weights = np.random.exponential(1.0, len(assets))
            normalized_weights = raw_weights / raw_weights.sum()

            weights_history.append({"date": date, "weights": dict(zip(assets, normalized_weights))})

        # Setup exporter
        export_config = ExportConfig(
            output_directory=temp_output_dir,
            export_formats=["parquet", "csv", "json"],
            include_metadata=True,
        )
        exporter = PortfolioExporter(export_config)

        # Test portfolio weights export
        weight_files = exporter.export_portfolio_weights(weights_history)

        for format_name, file_path in weight_files.items():
            assert Path(file_path).exists()
            assert format_name in file_path

        # Test position analysis
        position_analysis_file = exporter.export_position_analysis(weights_history)
        assert Path(position_analysis_file).exists()

        # Validate position analysis content
        with open(position_analysis_file) as f:
            position_data = json.load(f)

        assert len(position_data) > 0
        for _asset, analysis in position_data.items():
            assert "periods_active" in analysis
            assert "average_weight" in analysis
            assert analysis["periods_active"] > 0
            assert analysis["average_weight"] > 0
