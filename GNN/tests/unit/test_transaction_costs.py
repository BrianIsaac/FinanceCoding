"""
Unit tests for transaction cost modeling.

Tests verify cost calculations, turnover computation,
and integration with portfolio performance calculations.
"""

import pytest
import pandas as pd
import numpy as np

from src.evaluation.backtest.transaction_costs import (
    TransactionCostCalculator,
    TransactionCostConfig,
)


class TestTransactionCostCalculator:
    """Test suite for TransactionCostCalculator class."""

    @pytest.fixture
    def default_config(self):
        """Create default transaction cost configuration."""
        return TransactionCostConfig()

    @pytest.fixture
    def custom_config(self):
        """Create custom transaction cost configuration."""
        return TransactionCostConfig(
            linear_cost_bps=20.0,  # 0.2% linear cost
            fixed_cost_per_trade=5.0,
            bid_ask_spread_bps=8.0,
        )

    @pytest.fixture
    def calculator(self, default_config):
        """Create TransactionCostCalculator with default config."""
        return TransactionCostCalculator(default_config)

    @pytest.fixture
    def sample_weights(self):
        """Create sample current portfolio weights."""
        assets = [f"ASSET_{i:03d}" for i in range(5)]
        weights = [0.20, 0.20, 0.20, 0.20, 0.20]
        return pd.Series(weights, index=assets)

    @pytest.fixture
    def target_weights(self):
        """Create sample target portfolio weights."""
        assets = [f"ASSET_{i:03d}" for i in range(5)]
        weights = [0.30, 0.25, 0.15, 0.15, 0.15]
        return pd.Series(weights, index=assets)

    def test_calculator_initialization(self, default_config):
        """Test proper calculator initialization."""
        calc = TransactionCostCalculator(default_config)
        
        assert calc.config == default_config
        assert calc.config.linear_cost_bps == 10.0
        assert calc.config.fixed_cost_per_trade == 0.0

    def test_calculate_turnover_one_way(self, calculator, sample_weights, target_weights):
        """Test one-way turnover calculation."""
        turnover = calculator.calculate_turnover(
            sample_weights, target_weights, method="one_way"
        )
        
        # Expected turnover: sum of absolute weight changes
        expected = np.abs(target_weights - sample_weights).sum()
        assert abs(turnover - expected) < 1e-10

    def test_calculate_turnover_two_way(self, calculator, sample_weights, target_weights):
        """Test two-way turnover calculation."""
        turnover = calculator.calculate_turnover(
            sample_weights, target_weights, method="two_way"
        )
        
        # Calculate expected two-way turnover
        weight_changes = target_weights - sample_weights
        buys = weight_changes[weight_changes > 0].sum()
        sells = np.abs(weight_changes[weight_changes < 0]).sum()
        expected = buys + sells
        
        assert abs(turnover - expected) < 1e-10

    def test_calculate_turnover_invalid_method(self, calculator, sample_weights, target_weights):
        """Test turnover calculation with invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown turnover method"):
            calculator.calculate_turnover(
                sample_weights, target_weights, method="invalid_method"
            )

    def test_calculate_turnover_disjoint_assets(self, calculator):
        """Test turnover calculation with completely different asset sets."""
        current = pd.Series([0.5, 0.5], index=["ASSET_A", "ASSET_B"])
        target = pd.Series([0.4, 0.6], index=["ASSET_C", "ASSET_D"])
        
        turnover = calculator.calculate_turnover(current, target)
        
        # Should be 2.0 (complete portfolio change)
        assert abs(turnover - 2.0) < 1e-10

    def test_calculate_transaction_costs_basic(self, calculator, sample_weights, target_weights):
        """Test basic transaction cost calculation."""
        costs = calculator.calculate_transaction_costs(
            sample_weights, target_weights, portfolio_value=1000000.0
        )
        
        # Check required fields
        assert "total_cost" in costs
        assert "linear_cost" in costs
        assert "fixed_cost" in costs
        assert "turnover" in costs
        assert "num_trades" in costs
        assert "cost_per_turnover" in costs
        
        # Linear cost should be turnover * cost_bps
        expected_turnover = np.abs(target_weights - sample_weights).sum()
        expected_linear_cost = expected_turnover * (calculator.config.linear_cost_bps / 10000.0)
        
        assert abs(costs["turnover"] - expected_turnover) < 1e-10
        assert abs(costs["linear_cost"] - expected_linear_cost) < 1e-10

    def test_calculate_transaction_costs_with_fixed_costs(self, custom_config):
        """Test transaction cost calculation with fixed costs."""
        calc = TransactionCostCalculator(custom_config)
        
        current = pd.Series([0.5, 0.5], index=["ASSET_A", "ASSET_B"])
        target = pd.Series([0.3, 0.7], index=["ASSET_A", "ASSET_B"])
        
        costs = calc.calculate_transaction_costs(current, target, portfolio_value=100000.0)
        
        # Should have both linear and fixed costs
        assert costs["linear_cost"] > 0
        assert costs["fixed_cost"] > 0  # Due to fixed_cost_per_trade = 5.0
        assert costs["total_cost"] == costs["linear_cost"] + costs["fixed_cost"]

    def test_calculate_transaction_costs_zero_turnover(self, calculator):
        """Test transaction costs when there's no turnover."""
        identical_weights = pd.Series([0.25, 0.25, 0.25, 0.25], 
                                    index=[f"ASSET_{i}" for i in range(4)])
        
        costs = calculator.calculate_transaction_costs(identical_weights, identical_weights)
        
        assert costs["turnover"] == 0.0
        assert costs["total_cost"] == 0.0
        assert costs["linear_cost"] == 0.0
        assert costs["cost_per_turnover"] == 0.0

    def test_apply_transaction_costs_to_returns(self, calculator, sample_weights, target_weights):
        """Test applying transaction costs to portfolio returns."""
        gross_return = 0.02  # 2% gross return
        returns_series = pd.Series([gross_return])
        
        net_return, cost_breakdown = calculator.apply_transaction_costs(
            returns_series, sample_weights, target_weights, portfolio_value=1000000.0
        )
        
        # Net return should be gross return minus transaction costs
        expected_net = gross_return - cost_breakdown["total_cost"]
        assert abs(net_return - expected_net) < 1e-10
        
        # Cost breakdown should include both gross and net returns
        assert "gross_return" in cost_breakdown
        assert "net_return" in cost_breakdown
        assert cost_breakdown["gross_return"] == gross_return
        assert cost_breakdown["net_return"] == net_return

    def test_estimate_annual_costs(self, calculator):
        """Test annual cost estimation."""
        monthly_turnover = 0.15
        annual_estimates = calculator.estimate_annual_costs(
            monthly_turnover, rebalancing_frequency=12
        )
        
        # Check required fields
        assert "annual_turnover" in annual_estimates
        assert "annual_linear_cost" in annual_estimates
        assert "annual_total_cost" in annual_estimates
        assert "cost_drag_bps" in annual_estimates
        
        # Annual turnover should be monthly * frequency
        expected_annual_turnover = monthly_turnover * 12
        assert abs(annual_estimates["annual_turnover"] - expected_annual_turnover) < 1e-10
        
        # Cost drag in basis points
        assert annual_estimates["cost_drag_bps"] > 0

    def test_count_trades(self, calculator, sample_weights, target_weights):
        """Test trade counting functionality."""
        num_trades = calculator._count_trades(sample_weights, target_weights)
        
        # Should count trades for all assets with weight changes above threshold
        weight_changes = np.abs(target_weights - sample_weights)
        expected_trades = (weight_changes > 0.0001).sum()  # Above minimum threshold
        
        assert num_trades == expected_trades

    def test_count_trades_small_changes(self, calculator):
        """Test that very small weight changes don't count as trades."""
        current = pd.Series([0.25, 0.25, 0.25, 0.25], index=[f"ASSET_{i}" for i in range(4)])
        # Very small changes (below threshold)
        target = pd.Series([0.250001, 0.249999, 0.25, 0.25], index=[f"ASSET_{i}" for i in range(4)])
        
        num_trades = calculator._count_trades(current, target)
        
        # Changes are below minimum trade threshold (0.0001), so no trades
        assert num_trades == 0

    def test_create_cost_report_stub(self, calculator):
        """Test cost report creation (stub implementation)."""
        # Create dummy data
        rebalancing_history = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=12, freq='M'),
            'weights': [{}] * 12,
            'costs': [0.001] * 12,
        })
        
        returns_history = pd.Series(
            np.random.normal(0.001, 0.02, 100),
            index=pd.date_range('2020-01-01', periods=100)
        )
        
        report = calculator.create_cost_report(rebalancing_history, returns_history)
        
        # Should return basic report structure (stub implementation)
        assert "total_periods" in report
        assert report["total_periods"] == len(rebalancing_history)

    def test_transaction_costs_edge_cases(self, calculator):
        """Test transaction cost calculation edge cases."""
        # Empty weights
        empty = pd.Series([], dtype=float)
        costs = calculator.calculate_transaction_costs(empty, empty)
        assert costs["total_cost"] == 0.0
        
        # Single asset
        single_current = pd.Series([1.0], index=["ASSET_A"])
        single_target = pd.Series([1.0], index=["ASSET_A"])
        costs = calculator.calculate_transaction_costs(single_current, single_target)
        assert costs["total_cost"] == 0.0

    def test_portfolio_value_scaling(self, calculator, sample_weights, target_weights):
        """Test that costs scale properly with portfolio value."""
        small_portfolio = 10000.0
        large_portfolio = 10000000.0
        
        costs_small = calculator.calculate_transaction_costs(
            sample_weights, target_weights, portfolio_value=small_portfolio
        )
        costs_large = calculator.calculate_transaction_costs(
            sample_weights, target_weights, portfolio_value=large_portfolio
        )
        
        # Linear costs should be the same (percentage-based)
        assert abs(costs_small["linear_cost"] - costs_large["linear_cost"]) < 1e-10
        
        # Fixed costs should scale inversely with portfolio value
        if calculator.config.fixed_cost_per_trade > 0:
            ratio = costs_small["fixed_cost"] / costs_large["fixed_cost"]
            expected_ratio = large_portfolio / small_portfolio
            assert abs(ratio - expected_ratio) < 1e-6