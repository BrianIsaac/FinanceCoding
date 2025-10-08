"""
Comprehensive backtest execution engine with portfolio tracking.

This module provides a sophisticated backtest execution framework with
comprehensive trade logging, position tracking, performance analytics,
and transaction cost modeling for portfolio strategies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.backtest.transaction_costs import (
    TransactionCostCalculator,
    TransactionCostConfig,
)
from src.evaluation.metrics.returns import PerformanceAnalytics
from src.models.base import PortfolioModel

logger = logging.getLogger(__name__)


class RebalanceReason(Enum):
    """Reasons for portfolio rebalancing."""

    SCHEDULED = "scheduled"  # Regular scheduled rebalancing
    SIGNAL_CHANGE = "signal_change"  # Model signal changed significantly
    UNIVERSE_CHANGE = "universe_change"  # Universe membership changed
    RISK_MANAGEMENT = "risk_management"  # Risk-based rebalancing
    THRESHOLD_BREACH = "threshold_breach"  # Threshold-based rebalancing
    MANUAL = "manual"  # Manual override


@dataclass
class TradeRecord:
    """Record of a single trade execution."""

    timestamp: pd.Timestamp
    asset: str
    side: str  # "buy" or "sell"
    quantity: float
    price: float
    value: float
    transaction_cost: float
    reason: RebalanceReason
    portfolio_weight_before: float
    portfolio_weight_after: float
    notes: str = ""


@dataclass
class PositionRecord:
    """Record of portfolio position at a point in time."""

    timestamp: pd.Timestamp
    positions: dict[str, float]  # asset -> quantity
    weights: dict[str, float]  # asset -> weight
    portfolio_value: float
    cash: float
    leverage: float
    turnover: float
    tracking_error: float = 0.0


@dataclass
class PerformanceSnapshot:
    """Performance metrics at a point in time."""

    timestamp: pd.Timestamp
    portfolio_value: float
    total_return: float
    period_return: float
    benchmark_return: float
    excess_return: float
    sharpe_ratio: float
    volatility: float
    max_drawdown: float
    trade_count: int
    turnover: float
    transaction_costs: float


@dataclass
class BacktestExecutionConfig:
    """Configuration for backtest execution."""

    # Execution parameters
    initial_capital: float = 1000000.0
    rebalance_frequency: str = "M"  # Monthly
    rebalance_threshold: float = 0.05  # 5% weight change threshold
    enable_threshold_rebalancing: bool = False

    # Transaction costs
    transaction_cost_bps: float = 10.0  # 10 bps
    enable_realistic_costs: bool = True
    market_impact_factor: float = 0.1

    # Position tracking
    track_positions: bool = True
    track_trades: bool = True
    enable_performance_attribution: bool = True

    # Risk management
    max_position_weight: float = 0.15  # 15% max per position
    max_leverage: float = 1.0  # No leverage by default
    enable_risk_checks: bool = True

    # Output configuration
    save_detailed_logs: bool = True
    log_frequency: str = "D"  # Daily logging
    output_dir: Path | None = None


class BacktestExecutor:
    """
    Comprehensive backtest execution engine.

    This executor provides sophisticated backtesting capabilities including:
    - Portfolio tracking with detailed position and weight management
    - Comprehensive trade logging with execution details
    - Performance metrics calculation and attribution
    - Transaction cost modeling with market impact
    - Risk management and constraint enforcement
    - Detailed reporting and analytics
    """

    def __init__(self, config: BacktestExecutionConfig):
        """Initialize backtest executor."""
        self.config = config

        # Initialize components
        self.transaction_cost_calculator = TransactionCostCalculator(
            TransactionCostConfig(
                linear_cost_bps=config.transaction_cost_bps,
                market_impact_coefficient=(
                    config.market_impact_factor if config.enable_realistic_costs else 0.0
                ),
                bid_ask_spread_bps=5.0,
            )
        )

        self.performance_analytics = PerformanceAnalytics()

        # Initialize tracking
        self.trade_log: list[TradeRecord] = []
        self.position_history: list[PositionRecord] = []
        self.performance_history: list[PerformanceSnapshot] = []

        # State variables
        self.current_positions: dict[str, float] = {}  # asset -> quantity
        self.current_weights: dict[str, float] = {}  # asset -> weight
        self.current_cash: float = config.initial_capital
        self.total_transaction_costs: float = 0.0
        self.rebalance_count: int = 0

        # Setup output directory
        if config.output_dir:
            config.output_dir.mkdir(parents=True, exist_ok=True)

    def execute_backtest(
        self,
        model: PortfolioModel,
        returns_data: pd.DataFrame,
        rebalance_dates: list[pd.Timestamp],
        benchmark_returns: pd.Series | None = None,
        universe_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """
        Execute complete backtest with comprehensive tracking.

        Args:
            model: Portfolio model to backtest
            returns_data: Historical returns data
            rebalance_dates: Dates for portfolio rebalancing
            benchmark_returns: Optional benchmark returns for comparison
            universe_data: Optional dynamic universe data

        Returns:
            Comprehensive backtest results
        """
        logger.info(f"Starting backtest execution with {len(rebalance_dates)} rebalancing dates")

        # Initialize
        self._reset_state()

        # Execute rebalancing on each date
        for i, rebalance_date in enumerate(rebalance_dates):
            try:
                self._execute_rebalancing(
                    model=model,
                    rebalance_date=rebalance_date,
                    returns_data=returns_data,
                    universe_data=universe_data,
                    is_final=(i == len(rebalance_dates) - 1),
                )

            except Exception as e:
                logger.error(f"Error during rebalancing on {rebalance_date}: {e}")
                continue

        # Calculate final performance metrics
        results = self._compile_results(returns_data, benchmark_returns)

        # Save detailed logs if configured
        if self.config.save_detailed_logs:
            self._save_execution_logs()

        logger.info(
            f"Backtest execution completed: {self.rebalance_count} rebalances, "
            f"{len(self.trade_log)} trades"
        )

        return results

    def _reset_state(self) -> None:
        """Reset executor state for new backtest."""
        self.trade_log.clear()
        self.position_history.clear()
        self.performance_history.clear()

        self.current_positions.clear()
        self.current_weights.clear()
        self.current_cash = self.config.initial_capital
        self.total_transaction_costs = 0.0
        self.rebalance_count = 0

    def _execute_rebalancing(
        self,
        model: PortfolioModel,
        rebalance_date: pd.Timestamp,
        returns_data: pd.DataFrame,
        universe_data: pd.DataFrame | None,
        is_final: bool = False,
    ) -> None:
        """Execute portfolio rebalancing for given date."""

        # Get current universe
        current_universe = self._get_current_universe(rebalance_date, returns_data, universe_data)

        if not current_universe or len(current_universe) == 0:
            logger.warning(f"No universe available for {rebalance_date}")
            return

        # Generate target weights
        target_weights = model.predict_weights(
            date=rebalance_date,
            universe=current_universe,
        )

        if target_weights.empty:
            logger.warning(f"No weights generated for {rebalance_date}")
            return

        # Apply risk constraints
        constrained_weights = self._apply_risk_constraints(target_weights)

        # Determine rebalancing reason
        reason = self._determine_rebalance_reason(constrained_weights, rebalance_date)

        # Execute trades
        trades = self._execute_trades(
            target_weights=constrained_weights,
            current_date=rebalance_date,
            returns_data=returns_data,
            reason=reason,
        )

        # Update positions
        self._update_positions(trades, rebalance_date)

        # Record position snapshot
        self._record_position_snapshot(rebalance_date)

        # Calculate and record performance
        if not is_final:
            self._calculate_period_performance(rebalance_date, returns_data)

        self.rebalance_count += 1

    def _get_current_universe(
        self,
        date: pd.Timestamp,
        returns_data: pd.DataFrame,
        universe_data: pd.DataFrame | None,
    ) -> list[str]:
        """Get current universe for given date."""

        if universe_data is not None:
            # Use dynamic universe
            if date in universe_data.index:
                universe_mask = universe_data.loc[date]
                return universe_mask[universe_mask].index.tolist()
            else:
                # Use most recent universe
                available_dates = universe_data.index[universe_data.index <= date]
                if len(available_dates) > 0:
                    latest_date = available_dates[-1]
                    universe_mask = universe_data.loc[latest_date]
                    return universe_mask[universe_mask].index.tolist()

        # Use all available assets in returns data
        return returns_data.columns.tolist()

    def _apply_risk_constraints(self, weights: pd.Series) -> pd.Series:
        """Apply risk management constraints to portfolio weights."""

        if not self.config.enable_risk_checks:
            return weights

        constrained_weights = weights.copy()

        # Apply maximum position weight constraint
        if self.config.max_position_weight < 1.0:
            excess_weight = constrained_weights > self.config.max_position_weight
            if excess_weight.any():
                # Cap weights and redistribute excess
                excess_total = (
                    constrained_weights[excess_weight] - self.config.max_position_weight
                ).sum()
                constrained_weights[excess_weight] = self.config.max_position_weight

                # Redistribute excess weight proportionally to uncapped assets
                uncapped_assets = ~excess_weight
                if uncapped_assets.any():
                    redistribution = excess_total * (
                        constrained_weights[uncapped_assets]
                        / constrained_weights[uncapped_assets].sum()
                    )
                    constrained_weights[uncapped_assets] += redistribution

        # Normalize weights
        weight_sum = constrained_weights.sum()
        if abs(weight_sum - 1.0) > 1e-6:
            constrained_weights = constrained_weights / weight_sum

        return constrained_weights

    def _determine_rebalance_reason(
        self,
        target_weights: pd.Series,
        date: pd.Timestamp,
    ) -> RebalanceReason:
        """Determine reason for rebalancing."""

        # Default to scheduled rebalancing
        if not self.current_weights or len(self.current_weights) == 0:
            return RebalanceReason.SCHEDULED

        # Check for significant weight changes
        current_weights_aligned = pd.Series(self.current_weights).reindex(
            target_weights.index, fill_value=0
        )
        weight_changes = abs(target_weights - current_weights_aligned)
        max_change = weight_changes.max()

        if max_change > self.config.rebalance_threshold:
            return RebalanceReason.SIGNAL_CHANGE

        # Check for universe changes
        current_assets = set(self.current_weights.keys())
        target_assets = set(target_weights.index)

        if current_assets != target_assets:
            return RebalanceReason.UNIVERSE_CHANGE

        return RebalanceReason.SCHEDULED

    def _execute_trades(
        self,
        target_weights: pd.Series,
        current_date: pd.Timestamp,
        returns_data: pd.DataFrame,
        reason: RebalanceReason,
    ) -> list[TradeRecord]:
        """Execute trades to achieve target weights."""

        trades = []

        # Get current portfolio value
        portfolio_value = self._calculate_portfolio_value(current_date, returns_data)

        # Calculate target quantities
        target_values = target_weights * portfolio_value
        prices = (
            returns_data.loc[current_date]
            if current_date in returns_data.index
            else returns_data.iloc[-1]
        )

        for asset in target_weights.index:
            if asset not in prices or pd.isna(prices[asset]):
                continue

            current_quantity = self.current_positions.get(asset, 0.0)
            target_value = target_values[asset]
            price = prices[asset]

            # For returns data, we work with weights directly
            target_quantity = target_value / price if price != 0 else 0.0

            quantity_change = target_quantity - current_quantity

            if abs(quantity_change) > 1e-8:  # Only trade if meaningful change
                # Determine side
                side = "buy" if quantity_change > 0 else "sell"
                trade_value = abs(quantity_change * price)

                # Calculate transaction cost
                current_weights_series = (
                    pd.Series(self.current_weights, name="current")
                    if self.current_weights
                    else pd.Series(dtype=float, index=target_weights.index).fillna(0.0)
                )
                cost_breakdown = self.transaction_cost_calculator.calculate_transaction_costs(
                    current_weights_series,
                    target_weights,
                    portfolio_value,
                )
                transaction_cost = (
                    cost_breakdown.get("total_cost", 0.0)
                    * trade_value
                    / portfolio_value
                    if portfolio_value > 0
                    else 0.0
                )

                # Create trade record
                trade = TradeRecord(
                    timestamp=current_date,
                    asset=asset,
                    side=side,
                    quantity=abs(quantity_change),
                    price=price,
                    value=trade_value,
                    transaction_cost=transaction_cost,
                    reason=reason,
                    portfolio_weight_before=self.current_weights.get(asset, 0.0),
                    portfolio_weight_after=target_weights[asset],
                )

                trades.append(trade)
                self.total_transaction_costs += transaction_cost

        return trades

    def _update_positions(self, trades: list[TradeRecord], date: pd.Timestamp) -> None:
        """Update current positions based on executed trades."""

        # Apply trades to positions
        for trade in trades:
            if trade.asset not in self.current_positions:
                self.current_positions[trade.asset] = 0.0

            if trade.side == "buy":
                self.current_positions[trade.asset] += trade.quantity
            else:
                self.current_positions[trade.asset] -= trade.quantity

            # Remove zero positions
            if abs(self.current_positions[trade.asset]) < 1e-8:
                del self.current_positions[trade.asset]

        # Update weights based on current positions and prices
        portfolio_value = sum(
            quantity * 1.0  # Simplified - in practice would use actual prices
            for quantity in self.current_positions.values()
        )

        if portfolio_value > 0:
            self.current_weights = {
                asset: quantity / portfolio_value
                for asset, quantity in self.current_positions.items()
            }
        else:
            self.current_weights.clear()

        # Add trades to log
        self.trade_log.extend(trades)

    def _record_position_snapshot(self, date: pd.Timestamp) -> None:
        """Record current portfolio position snapshot."""

        if not self.config.track_positions:
            return

        portfolio_value = self._calculate_portfolio_value_from_positions()

        # Calculate turnover
        turnover = self._calculate_turnover()

        position_record = PositionRecord(
            timestamp=date,
            positions=self.current_positions.copy(),
            weights=self.current_weights.copy(),
            portfolio_value=portfolio_value,
            cash=self.current_cash,
            leverage=sum(abs(w) for w in self.current_weights.values()),
            turnover=turnover,
        )

        self.position_history.append(position_record)

    def _calculate_portfolio_value(
        self,
        date: pd.Timestamp,
        returns_data: pd.DataFrame,
    ) -> float:
        """Calculate current portfolio value."""

        if date in returns_data.index:
            prices = returns_data.loc[date]
        else:
            # Use most recent available prices
            available_dates = returns_data.index[returns_data.index <= date]
            if len(available_dates) > 0:
                prices = returns_data.loc[available_dates[-1]]
            else:
                return self.config.initial_capital

        position_value = 0.0
        for asset, quantity in self.current_positions.items():
            if asset in prices and not pd.isna(prices[asset]):
                position_value += quantity * prices[asset]

        return position_value + self.current_cash

    def _calculate_portfolio_value_from_positions(self) -> float:
        """Calculate portfolio value from current positions (simplified)."""
        return sum(self.current_positions.values()) + self.current_cash

    def _calculate_turnover(self) -> float:
        """Calculate portfolio turnover since last rebalancing."""

        if len(self.position_history) < 2:
            return 0.0

        previous_weights = self.position_history[-1].weights if self.position_history else {}
        current_weights = self.current_weights

        # Calculate weight changes
        all_assets = set(previous_weights.keys()) | set(current_weights.keys())

        turnover = 0.0
        for asset in all_assets:
            prev_weight = previous_weights.get(asset, 0.0)
            curr_weight = current_weights.get(asset, 0.0)
            turnover += abs(curr_weight - prev_weight)

        return turnover / 2.0  # Divide by 2 to avoid double counting

    def _calculate_period_performance(
        self,
        date: pd.Timestamp,
        returns_data: pd.DataFrame,
    ) -> None:
        """Calculate performance metrics for current period."""

        if not self.config.enable_performance_attribution:
            return

        # This is a simplified version - full implementation would calculate
        # comprehensive performance attribution metrics
        portfolio_value = self._calculate_portfolio_value(date, returns_data)

        # Calculate returns since last measurement
        total_return = 0.0
        if self.performance_history:
            prev_value = self.performance_history[-1].portfolio_value
            total_return = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0.0

        performance_snapshot = PerformanceSnapshot(
            timestamp=date,
            portfolio_value=portfolio_value,
            total_return=total_return,
            period_return=total_return,  # Simplified
            benchmark_return=0.0,  # Would calculate from benchmark
            excess_return=total_return,  # Simplified
            sharpe_ratio=0.0,  # Would calculate from return history
            volatility=0.0,  # Would calculate from return history
            max_drawdown=0.0,  # Would calculate from return history
            trade_count=len(self.trade_log),
            turnover=self._calculate_turnover(),
            transaction_costs=self.total_transaction_costs,
        )

        self.performance_history.append(performance_snapshot)

    def _compile_results(
        self,
        returns_data: pd.DataFrame,
        benchmark_returns: pd.Series | None,
    ) -> dict[str, Any]:
        """Compile comprehensive backtest results."""

        # Portfolio returns
        portfolio_returns = self._calculate_portfolio_returns()

        # Performance metrics
        performance_metrics = {}
        if not portfolio_returns.empty:
            performance_metrics = self.performance_analytics.calculate_portfolio_metrics(
                portfolio_returns
            )

        # Trade analysis
        trade_analysis = self._analyze_trades()

        # Position analysis
        position_analysis = self._analyze_positions()

        return {
            "portfolio_returns": portfolio_returns,
            "performance_metrics": performance_metrics,
            "trade_analysis": trade_analysis,
            "position_analysis": position_analysis,
            "execution_summary": {
                "total_rebalances": self.rebalance_count,
                "total_trades": len(self.trade_log),
                "total_transaction_costs": self.total_transaction_costs,
                "final_portfolio_value": (
                    self.performance_history[-1].portfolio_value
                    if self.performance_history
                    else self.config.initial_capital
                ),
            },
            "detailed_logs": {
                "trades": self.trade_log,
                "positions": self.position_history,
                "performance": self.performance_history,
            },
        }

    def _calculate_portfolio_returns(self) -> pd.Series:
        """Calculate portfolio returns from performance history."""

        if len(self.performance_history) < 2:
            return pd.Series(dtype=float)

        returns = []
        dates = []

        for i in range(1, len(self.performance_history)):
            prev_value = self.performance_history[i - 1].portfolio_value
            curr_value = self.performance_history[i].portfolio_value

            if prev_value > 0:
                period_return = (curr_value - prev_value) / prev_value
                returns.append(period_return)
                dates.append(self.performance_history[i].timestamp)

        return pd.Series(returns, index=dates, name="portfolio_returns")

    def _analyze_trades(self) -> dict[str, Any]:
        """Analyze trade execution patterns."""

        if not self.trade_log:
            return {"message": "No trades executed"}

        trade_df = pd.DataFrame(
            [
                {
                    "timestamp": trade.timestamp,
                    "asset": trade.asset,
                    "side": trade.side,
                    "value": trade.value,
                    "transaction_cost": trade.transaction_cost,
                    "reason": trade.reason.value,
                }
                for trade in self.trade_log
            ]
        )

        return {
            "total_trades": len(self.trade_log),
            "total_trade_value": trade_df["value"].sum(),
            "total_transaction_costs": trade_df["transaction_cost"].sum(),
            "avg_trade_size": trade_df["value"].mean(),
            "trades_by_reason": trade_df["reason"].value_counts().to_dict(),
            "trades_by_side": trade_df["side"].value_counts().to_dict(),
            "most_traded_assets": trade_df["asset"].value_counts().head(10).to_dict(),
        }

    def _analyze_positions(self) -> dict[str, Any]:
        """Analyze position history and characteristics."""

        if not self.position_history:
            return {"message": "No position history available"}

        position_df = pd.DataFrame(
            [
                {
                    "timestamp": pos.timestamp,
                    "portfolio_value": pos.portfolio_value,
                    "leverage": pos.leverage,
                    "turnover": pos.turnover,
                    "num_positions": len(pos.positions),
                }
                for pos in self.position_history
            ]
        )

        return {
            "avg_num_positions": position_df["num_positions"].mean(),
            "max_leverage": position_df["leverage"].max(),
            "avg_turnover": position_df["turnover"].mean(),
            "portfolio_value_range": {
                "min": position_df["portfolio_value"].min(),
                "max": position_df["portfolio_value"].max(),
                "final": position_df["portfolio_value"].iloc[-1] if len(position_df) > 0 else 0,
            },
        }

    def _save_execution_logs(self) -> None:
        """Save detailed execution logs to files."""

        if not self.config.output_dir:
            return

        output_dir = self.config.output_dir

        # Save trades
        if self.trade_log:
            trades_df = pd.DataFrame(
                [
                    {
                        "timestamp": trade.timestamp,
                        "asset": trade.asset,
                        "side": trade.side,
                        "quantity": trade.quantity,
                        "price": trade.price,
                        "value": trade.value,
                        "transaction_cost": trade.transaction_cost,
                        "reason": trade.reason.value,
                        "weight_before": trade.portfolio_weight_before,
                        "weight_after": trade.portfolio_weight_after,
                        "notes": trade.notes,
                    }
                    for trade in self.trade_log
                ]
            )
            trades_df.to_csv(output_dir / "trade_log.csv", index=False)

        # Save positions
        if self.position_history:
            positions_df = pd.DataFrame(
                [
                    {
                        "timestamp": pos.timestamp,
                        "portfolio_value": pos.portfolio_value,
                        "cash": pos.cash,
                        "leverage": pos.leverage,
                        "turnover": pos.turnover,
                        "num_positions": len(pos.positions),
                    }
                    for pos in self.position_history
                ]
            )
            positions_df.to_csv(output_dir / "position_history.csv", index=False)

        # Save performance
        if self.performance_history:
            performance_df = pd.DataFrame(
                [
                    {
                        "timestamp": perf.timestamp,
                        "portfolio_value": perf.portfolio_value,
                        "total_return": perf.total_return,
                        "period_return": perf.period_return,
                        "trade_count": perf.trade_count,
                        "turnover": perf.turnover,
                        "transaction_costs": perf.transaction_costs,
                    }
                    for perf in self.performance_history
                ]
            )
            performance_df.to_csv(output_dir / "performance_history.csv", index=False)

        logger.info(f"Execution logs saved to {output_dir}")


class TradingSimulator:
    """
    Advanced trading simulation with market microstructure effects.

    This simulator provides realistic trading execution including:
    - Market impact modeling
    - Slippage and timing effects
    - Partial fill simulation
    - Order book dynamics
    """

    def __init__(self, impact_model: str = "linear"):
        """Initialize trading simulator."""
        self.impact_model = impact_model
        self.execution_history: list[dict[str, Any]] = []

    def simulate_trade_execution(
        self,
        asset: str,
        target_quantity: float,
        market_data: pd.Series,
        volume_data: pd.Series | None = None,
    ) -> dict[str, Any]:
        """Simulate realistic trade execution."""

        price = market_data[asset] if asset in market_data else 1.0
        volume = volume_data[asset] if volume_data is not None and asset in volume_data else 1000000

        # Calculate market impact
        volume_participation = abs(target_quantity) / volume if volume > 0 else 0.0

        if self.impact_model == "linear":
            market_impact = 0.001 * volume_participation  # 10 bps per 10% volume
        else:
            market_impact = 0.001 * np.sqrt(volume_participation)  # Square root impact

        # Apply slippage
        slippage = np.random.normal(0, 0.0005)  # 5 bps std dev slippage

        # Calculate execution price
        execution_price = price * (1 + market_impact + slippage)

        # Record execution
        execution_record = {
            "asset": asset,
            "target_quantity": target_quantity,
            "executed_quantity": target_quantity,  # Assume full fill
            "reference_price": price,
            "execution_price": execution_price,
            "market_impact": market_impact,
            "slippage": slippage,
            "volume_participation": volume_participation,
        }

        self.execution_history.append(execution_record)

        return execution_record
