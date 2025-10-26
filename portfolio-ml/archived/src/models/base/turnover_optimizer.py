"""Turnover-Aware Portfolio Optimization Framework.

This module provides turnover-aware optimization to ensure portfolio rebalancing
stays within institutional constraints while maintaining performance objectives.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TurnoverConstraints:
    """Turnover constraint specifications."""

    max_monthly_turnover: float = 0.20  # 20% monthly limit
    max_quarterly_turnover: float = 0.40  # 40% quarterly limit
    max_annual_turnover: float = 1.20   # 120% annual limit
    turnover_penalty_lambda: float = 1.0  # Penalty weight for turnover
    position_change_threshold: float = 0.001  # Minimum change to trigger trade


@dataclass
class OptimizationResult:
    """Portfolio optimization result with turnover control."""

    optimized_weights: np.ndarray
    original_weights: np.ndarray
    previous_weights: np.ndarray
    turnover_achieved: float
    turnover_constraint: float
    constraint_met: bool
    optimization_success: bool
    iterations: int
    objective_value: float
    turnover_penalty: float


class TurnoverAwareOptimizer:
    """Turnover-aware portfolio optimizer that respects institutional constraints.

    This optimizer modifies portfolio weights to ensure turnover constraints are met
    while preserving as much of the original signal as possible.
    """

    def __init__(self, constraints: Optional[TurnoverConstraints] = None):
        """Initialise turnover-aware optimizer.

        Args:
            constraints: Turnover constraint specifications
        """
        self.constraints = constraints or TurnoverConstraints()
        self.optimization_history: list[OptimizationResult] = []

    def calculate_turnover(self, weights_current: np.ndarray, weights_previous: np.ndarray) -> float:
        """Calculate portfolio turnover between two weight vectors.

        Args:
            weights_current: Current period weights
            weights_previous: Previous period weights

        Returns:
            Turnover rate as percentage
        """
        return np.sum(np.abs(weights_current - weights_previous)) / 2.0

    def _turnover_penalty_objective(self, weights: np.ndarray,
                                   target_weights: np.ndarray,
                                   previous_weights: np.ndarray) -> float:
        """Objective function with turnover penalty.

        Args:
            weights: Current optimization weights
            target_weights: Target weights from original model
            previous_weights: Previous period weights

        Returns:
            Penalized objective value
        """
        # Tracking error from target weights (minimize deviation from model signal)
        tracking_error = np.sum((weights - target_weights) ** 2)

        # Turnover penalty
        turnover = self.calculate_turnover(weights, previous_weights)
        turnover_penalty = self.constraints.turnover_penalty_lambda * max(0, turnover - self.constraints.max_monthly_turnover) ** 2

        return tracking_error + turnover_penalty

    def _create_constraints_for_optimization(self, previous_weights: np.ndarray,
                                           max_turnover: float) -> list[dict[str, Any]]:
        """Create optimization constraints.

        Args:
            previous_weights: Previous period weights
            max_turnover: Maximum allowed turnover

        Returns:
            List of constraint dictionaries for scipy.optimize
        """
        constraints = []

        # Budget constraint (weights sum to 1)
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        })

        # Turnover constraint
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: max_turnover - self.calculate_turnover(w, previous_weights)
        })

        # Non-negative weights (long-only)
        for i in range(len(previous_weights)):
            constraints.append({
                'type': 'ineq',
                'fun': lambda w, i=i: w[i]
            })

        # Maximum position weight (10% limit)
        for i in range(len(previous_weights)):
            constraints.append({
                'type': 'ineq',
                'fun': lambda w, i=i: 0.10 - w[i]
            })

        return constraints

    def optimize_with_turnover_constraint(self,
                                        target_weights: np.ndarray,
                                        previous_weights: np.ndarray,
                                        max_turnover: Optional[float] = None) -> OptimizationResult:
        """Optimize portfolio weights with turnover constraint.

        Args:
            target_weights: Target weights from ML model
            previous_weights: Previous period weights
            max_turnover: Maximum allowed turnover (defaults to monthly limit)

        Returns:
            Optimization result with turnover-constrained weights
        """
        if max_turnover is None:
            max_turnover = self.constraints.max_monthly_turnover

        n_assets = len(target_weights)

        # Initial guess: blend between previous and target weights
        initial_blend_ratio = 0.5  # Start with 50% blend
        initial_weights = initial_blend_ratio * previous_weights + (1 - initial_blend_ratio) * target_weights
        initial_weights = initial_weights / np.sum(initial_weights)  # Normalize

        # Check if target weights already meet turnover constraint
        target_turnover = self.calculate_turnover(target_weights, previous_weights)

        if target_turnover <= max_turnover:
            # No optimization needed
            result = OptimizationResult(
                optimized_weights=target_weights,
                original_weights=target_weights,
                previous_weights=previous_weights,
                turnover_achieved=target_turnover,
                turnover_constraint=max_turnover,
                constraint_met=True,
                optimization_success=True,
                iterations=0,
                objective_value=0.0,
                turnover_penalty=0.0
            )
            self.optimization_history.append(result)
            return result

        # Set up optimization
        bounds = [(0.0, 0.10) for _ in range(n_assets)]  # 0% to 10% per asset
        constraints = self._create_constraints_for_optimization(previous_weights, max_turnover)

        try:
            # Run optimization
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                optimization_result = minimize(
                    fun=self._turnover_penalty_objective,
                    x0=initial_weights,
                    args=(target_weights, previous_weights),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-6}
                )

            optimized_weights = optimization_result.x
            optimized_weights = optimized_weights / np.sum(optimized_weights)  # Ensure normalization

            # Calculate results
            achieved_turnover = self.calculate_turnover(optimized_weights, previous_weights)
            constraint_met = achieved_turnover <= max_turnover + 1e-6  # Small tolerance

            turnover_penalty = self.constraints.turnover_penalty_lambda * max(0, achieved_turnover - max_turnover) ** 2

            result = OptimizationResult(
                optimized_weights=optimized_weights,
                original_weights=target_weights,
                previous_weights=previous_weights,
                turnover_achieved=achieved_turnover,
                turnover_constraint=max_turnover,
                constraint_met=constraint_met,
                optimization_success=optimization_result.success,
                iterations=optimization_result.nit,
                objective_value=optimization_result.fun,
                turnover_penalty=turnover_penalty
            )

            logger.info(f"Turnover optimization: {target_turnover:.3f} -> {achieved_turnover:.3f} "
                       f"(limit: {max_turnover:.3f}, met: {constraint_met})")

        except Exception as e:
            logger.error(f"Turnover optimization failed: {e}")

            # Fallback: simple blending approach
            blend_ratio = min(0.8, max_turnover / target_turnover) if target_turnover > 0 else 0.5
            fallback_weights = blend_ratio * previous_weights + (1 - blend_ratio) * target_weights
            fallback_weights = fallback_weights / np.sum(fallback_weights)

            achieved_turnover = self.calculate_turnover(fallback_weights, previous_weights)
            constraint_met = achieved_turnover <= max_turnover + 1e-6

            result = OptimizationResult(
                optimized_weights=fallback_weights,
                original_weights=target_weights,
                previous_weights=previous_weights,
                turnover_achieved=achieved_turnover,
                turnover_constraint=max_turnover,
                constraint_met=constraint_met,
                optimization_success=False,
                iterations=0,
                objective_value=float('inf'),
                turnover_penalty=0.0
            )

            logger.warning(f"Using fallback blending: {target_turnover:.3f} -> {achieved_turnover:.3f}")

        self.optimization_history.append(result)
        return result

    def batch_optimize_time_series(self,
                                 weights_time_series: np.ndarray,
                                 rebalancing_frequency: str = 'monthly') -> tuple[np.ndarray, list[OptimizationResult]]:
        """Optimize a time series of portfolio weights with turnover constraints.

        Args:
            weights_time_series: Array of shape (n_periods, n_assets) with target weights
            rebalancing_frequency: 'monthly', 'quarterly', or 'annual'

        Returns:
            Tuple of (optimized_weights_series, optimization_results)
        """
        n_periods, n_assets = weights_time_series.shape

        # Set turnover limit based on frequency
        turnover_limits = {
            'monthly': self.constraints.max_monthly_turnover,
            'quarterly': self.constraints.max_quarterly_turnover,
            'annual': self.constraints.max_annual_turnover
        }
        max_turnover = turnover_limits.get(rebalancing_frequency, self.constraints.max_monthly_turnover)

        logger.info(f"Batch optimizing {n_periods} periods with {rebalancing_frequency} "
                   f"rebalancing (max turnover: {max_turnover:.1%})")

        optimized_series = np.zeros_like(weights_time_series)
        optimization_results = []

        # Initialize with equal weights
        previous_weights = np.ones(n_assets) / n_assets

        for period in range(n_periods):
            target_weights = weights_time_series[period]

            # Optimize current period
            result = self.optimize_with_turnover_constraint(
                target_weights, previous_weights, max_turnover
            )

            optimized_series[period] = result.optimized_weights
            optimization_results.append(result)

            # Update for next period
            previous_weights = result.optimized_weights

            if period % 10 == 0:
                logger.info(f"Optimized period {period}/{n_periods}: "
                           f"turnover {result.turnover_achieved:.3f}")

        # Summary statistics
        turnovers = [r.turnover_achieved for r in optimization_results]
        constraints_met = sum(1 for r in optimization_results if r.constraint_met)

        logger.info(f"Batch optimization complete: {constraints_met}/{n_periods} periods compliant, "
                   f"avg turnover: {np.mean(turnovers):.3f}")

        return optimized_series, optimization_results

    def adaptive_rebalancing_frequency(self,
                                     weights_time_series: np.ndarray) -> tuple[str, float]:
        """Determine optimal rebalancing frequency based on turnover patterns.

        Args:
            weights_time_series: Time series of target weights

        Returns:
            Tuple of (recommended_frequency, expected_turnover)
        """
        n_periods = len(weights_time_series)

        # Calculate turnovers for different frequencies
        frequencies = ['monthly', 'quarterly', 'annual']
        frequency_analysis = {}

        for freq in frequencies:
            # Sample periods based on frequency
            if freq == 'monthly':
                sample_periods = list(range(min(n_periods, 24)))  # Up to 24 months
            elif freq == 'quarterly':
                sample_periods = list(range(0, min(n_periods, 96), 3))  # Every 3rd period
            else:  # annual
                sample_periods = list(range(0, min(n_periods, 96), 12))  # Every 12th period

            if len(sample_periods) < 2:
                continue

            # Calculate average turnover for this frequency
            turnovers = []
            previous_weights = np.ones(weights_time_series.shape[1]) / weights_time_series.shape[1]

            for period in sample_periods[1:]:
                current_weights = weights_time_series[period]
                turnover = self.calculate_turnover(current_weights, previous_weights)
                turnovers.append(turnover)
                previous_weights = current_weights

            avg_turnover = np.mean(turnovers)
            turnover_limit = {
                'monthly': self.constraints.max_monthly_turnover,
                'quarterly': self.constraints.max_quarterly_turnover,
                'annual': self.constraints.max_annual_turnover
            }[freq]

            frequency_analysis[freq] = {
                'avg_turnover': avg_turnover,
                'turnover_limit': turnover_limit,
                'compliance_rate': np.mean([t <= turnover_limit for t in turnovers]),
                'feasible': avg_turnover <= turnover_limit
            }

        # Find optimal frequency
        feasible_frequencies = [(freq, data) for freq, data in frequency_analysis.items() if data['feasible']]

        if feasible_frequencies:
            # Choose highest frequency that's feasible
            frequency_priority = {'monthly': 3, 'quarterly': 2, 'annual': 1}
            best_freq = max(feasible_frequencies, key=lambda x: frequency_priority[x[0]])
            recommended_freq = best_freq[0]
            expected_turnover = best_freq[1]['avg_turnover']
        else:
            # If no frequency is feasible, recommend annual with optimization
            recommended_freq = 'annual'
            expected_turnover = frequency_analysis.get('annual', {}).get('avg_turnover', 0.5)

        logger.info(f"Recommended rebalancing frequency: {recommended_freq} "
                   f"(expected turnover: {expected_turnover:.3f})")

        return recommended_freq, expected_turnover

    def generate_compliance_report(self) -> dict[str, Any]:
        """Generate turnover compliance report from optimization history.

        Returns:
            Dictionary containing compliance analysis
        """
        if not self.optimization_history:
            return {'error': 'No optimization history available'}

        results = self.optimization_history

        # Calculate statistics
        turnovers = [r.turnover_achieved for r in results]
        original_turnovers = [self.calculate_turnover(r.original_weights, r.previous_weights) for r in results]
        constraints_met = sum(1 for r in results if r.constraint_met)
        successful_optimizations = sum(1 for r in results if r.optimization_success)

        # Turnover reduction analysis
        turnover_reductions = [orig - opt for orig, opt in zip(original_turnovers, turnovers)]

        compliance_report = {
            'summary_statistics': {
                'total_optimizations': len(results),
                'successful_optimizations': successful_optimizations,
                'constraints_met': constraints_met,
                'compliance_rate': constraints_met / len(results),
                'success_rate': successful_optimizations / len(results)
            },
            'turnover_analysis': {
                'original_avg_turnover': np.mean(original_turnovers),
                'optimized_avg_turnover': np.mean(turnovers),
                'avg_turnover_reduction': np.mean(turnover_reductions),
                'max_turnover_achieved': np.max(turnovers),
                'min_turnover_achieved': np.min(turnovers),
                'turnover_volatility': np.std(turnovers)
            },
            'constraint_compliance': {
                'monthly_limit': self.constraints.max_monthly_turnover,
                'violations': len(results) - constraints_met,
                'violation_rate': (len(results) - constraints_met) / len(results),
                'avg_violation_magnitude': np.mean([max(0, t - self.constraints.max_monthly_turnover)
                                                   for t in turnovers])
            },
            'optimization_performance': {
                'avg_iterations': np.mean([r.iterations for r in results if r.optimization_success]),
                'avg_objective_value': np.mean([r.objective_value for r in results if r.optimization_success and r.objective_value < float('inf')]),
                'avg_turnover_penalty': np.mean([r.turnover_penalty for r in results])
            }
        }

        return compliance_report

    def get_optimization_history_dataframe(self) -> pd.DataFrame:
        """Get optimization history as DataFrame.

        Returns:
            DataFrame with optimization results
        """
        if not self.optimization_history:
            return pd.DataFrame()

        history_data = []
        for i, result in enumerate(self.optimization_history):
            original_turnover = self.calculate_turnover(result.original_weights, result.previous_weights)

            history_data.append({
                'Period': i,
                'Original_Turnover': original_turnover,
                'Optimized_Turnover': result.turnover_achieved,
                'Turnover_Reduction': original_turnover - result.turnover_achieved,
                'Constraint_Met': result.constraint_met,
                'Optimization_Success': result.optimization_success,
                'Iterations': result.iterations,
                'Objective_Value': result.objective_value,
                'Turnover_Penalty': result.turnover_penalty
            })

        return pd.DataFrame(history_data)
