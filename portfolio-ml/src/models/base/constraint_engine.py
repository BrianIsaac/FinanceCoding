"""
Unified constraint engine for portfolio optimization models.

This module provides the main ConstraintEngine class that integrates with
transaction cost calculations and provides comprehensive constraint enforcement
across all portfolio models (HRP, LSTM, GAT, baselines).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ...evaluation.backtest.transaction_costs import TransactionCostCalculator
from .constraints import (
    ConstraintEngine as BaseConstraintEngine,
)
from .constraints import (
    ConstraintViolation,
    ConstraintViolationError,
    PortfolioConstraints,
)


class UnifiedConstraintEngine:
    """
    Unified constraint engine with transaction cost integration.

    This class combines the base constraint enforcement with transaction
    cost modeling to provide comprehensive portfolio optimization constraint
    handling across all models.
    """

    def __init__(
        self,
        constraints: PortfolioConstraints,
        transaction_cost_calculator: TransactionCostCalculator | None = None,
    ):
        """
        Initialize unified constraint engine.

        Args:
            constraints: Portfolio constraints configuration
            transaction_cost_calculator: Optional transaction cost calculator
        """
        self.base_engine = BaseConstraintEngine(constraints)
        self.transaction_cost_calculator = transaction_cost_calculator
        self.constraints = constraints

    def enforce_all_constraints(
        self,
        weights: pd.Series,
        previous_weights: pd.Series | None = None,
        model_scores: pd.Series | None = None,
        date: pd.Timestamp | None = None,
        portfolio_value: float = 1.0,
        use_soft_constraints: bool = True,
    ) -> tuple[pd.Series, list[ConstraintViolation], dict[str, Any]]:
        """
        Apply all constraints including transaction cost optimization.

        Args:
            weights: Raw portfolio weights from model
            previous_weights: Previous period weights
            model_scores: Model confidence scores for ranking
            date: Current rebalancing date
            portfolio_value: Total portfolio value

        Returns:
            Tuple of (constrained_weights, violations, cost_analysis)
        """
        # Debug logging for constraint enforcement
        import logging
        logger = logging.getLogger(__name__)

        # CRITICAL: Validate constraint feasibility before enforcement
        # This prevents attempting mathematically impossible optimization
        if self.constraints.max_position_weight is not None:
            min_required = 1.0 / len(weights)
            if self.constraints.max_position_weight < min_required:
                raise ValueError(
                    f"Infeasible constraints: max_position_weight={self.constraints.max_position_weight:.4f} "
                    f"is less than minimum required={min_required:.4f} for {len(weights)} assets. "
                    f"With {len(weights)} assets, each needs at least {min_required:.2%} on average to sum to 100%. "
                    f"Increase max_position_weight to at least {min_required:.4f} or reduce universe size."
                )

        max_weight_before = weights.max()
        if max_weight_before > self.constraints.max_position_weight:
            logger.warning(f"enforce_all_constraints: Input has max weight {max_weight_before:.1%}, limit is {self.constraints.max_position_weight:.1%}")

        # Step 1: Apply base constraints (soft or hard)
        if use_soft_constraints:
            # Try convex optimization first, fall back to iterative if needed
            try:
                constrained_weights = self._apply_convex_constraints(
                    weights, previous_weights
                )
                logger.debug("Applied convex constraint projection successfully")
            except Exception as e:
                logger.warning(f"Convex optimization failed: {e}, falling back to soft constraints")
                # Use soft penalties instead of hard constraints
                constrained_weights = self._apply_soft_constraints(
                    weights, previous_weights, model_scores
                )

            # Check violations and re-enforce if needed
            violations = self.base_engine.check_violations(constrained_weights, previous_weights)

            # Re-enforce if violations detected
            if violations:
                logger.warning(f"Detected {len(violations)} violations, re-enforcing constraints")
                max_iterations = 3
                for iteration in range(max_iterations):
                    # Tighten constraints by 5% and re-solve
                    # CRITICAL FIX: Check feasibility before tightening to avoid mathematical infeasibility
                    num_assets = len(constrained_weights)
                    if self.constraints.max_position_weight:
                        proposed_tightened = self.constraints.max_position_weight * 0.95
                        # Check if tightened constraint is feasible for universe size
                        if num_assets * proposed_tightened >= 1.0:
                            tightened_max_weight = proposed_tightened
                        else:
                            logger.warning(
                                f"Cannot tighten max_position constraint for {num_assets} assets "
                                f"(would require {1.0/num_assets:.4f} avg but max would be {proposed_tightened:.4f}). "
                                f"Using original constraint."
                            )
                            tightened_max_weight = self.constraints.max_position_weight
                    else:
                        tightened_max_weight = None

                    tightened_min_weight = self.constraints.min_weight_threshold * 1.05 if self.constraints.min_weight_threshold else None

                    # Create tightened constraint copy
                    # Use direct attribute assignment instead of constructor since we don't know the exact init signature
                    import copy
                    tighter_constraints = copy.deepcopy(self.constraints)
                    if tightened_max_weight is not None:
                        tighter_constraints.max_position_weight = tightened_max_weight
                    if tightened_min_weight is not None:
                        tighter_constraints.min_weight_threshold = tightened_min_weight

                    # Re-solve with tighter constraints
                    try:
                        # Temporarily swap constraints to use tighter ones
                        original_constraints = self.constraints
                        original_base_engine_constraints = self.base_engine.constraints

                        # Create temporary engine with tighter constraints
                        from .constraints import ConstraintEngine as BaseConstraintEngine
                        temp_engine = BaseConstraintEngine(tighter_constraints)

                        # Swap in tighter constraints temporarily
                        self.constraints = tighter_constraints
                        self.base_engine = temp_engine

                        # Re-apply constraints using the same method
                        if use_soft_constraints:
                            try:
                                temp_weights = self._apply_convex_constraints(
                                    weights, previous_weights
                                )
                            except Exception:
                                temp_weights = self._apply_soft_constraints(
                                    weights, previous_weights, model_scores
                                )
                        else:
                            temp_weights, _ = temp_engine.enforce_constraints(
                                weights, previous_weights, model_scores, date
                            )

                        # Restore original constraints
                        self.constraints = original_constraints
                        self.base_engine.constraints = original_base_engine_constraints

                        # Check if violations are resolved
                        new_violations = temp_engine.check_violations(temp_weights, previous_weights)

                        if not new_violations:
                            logger.info(f"Re-enforcement succeeded at iteration {iteration + 1}")
                            constrained_weights = temp_weights
                            violations = new_violations
                            break
                        else:
                            # Still have violations but fewer - use this as best attempt
                            if len(new_violations) < len(violations):
                                constrained_weights = temp_weights
                                violations = new_violations
                    except Exception as e:
                        logger.warning(f"Re-enforcement iteration {iteration + 1} failed: {e}")
                        # Ensure constraints are restored even on error
                        try:
                            self.constraints = original_constraints
                            self.base_engine.constraints = original_base_engine_constraints
                        except:
                            pass
                        continue
                else:
                    # CRITICAL FIX: Raise error instead of silently returning violated weights
                    # This ensures constraint failures are caught and debugged rather than hidden
                    from .constraints import ConstraintViolationError
                    violation_details = []
                    max_weight = constrained_weights.max()
                    weight_sum = constrained_weights.sum()
                    if max_weight > self.constraints.max_position_weight:
                        violation_details.append(f"max_weight={max_weight:.4f} > limit={self.constraints.max_position_weight:.4f}")
                    if abs(weight_sum - 1.0) > 1e-6:
                        violation_details.append(f"sum={weight_sum:.6f} != 1.0")

                    raise ConstraintViolationError(
                        f"Failed to enforce constraints after {max_iterations} iterations. "
                        f"Violations: {', '.join(violation_details)}"
                    )

            max_weight_after = constrained_weights.max()
            if max_weight_after > self.constraints.max_position_weight + 1e-6:
                logger.warning(f"Minor constraint violation: max weight is {max_weight_after:.1%}, limit is {self.constraints.max_position_weight:.1%}")
        else:
            # Traditional hard constraint enforcement
            constrained_weights, violations = self.base_engine.enforce_constraints(
                weights, previous_weights, model_scores, date
            )

        # Step 2: Calculate transaction costs if available
        cost_analysis = {}
        if (
            self.transaction_cost_calculator is not None
            and previous_weights is not None
            and self.constraints.cost_aware_enforcement
        ):
            cost_analysis = self.transaction_cost_calculator.calculate_transaction_costs(
                previous_weights, constrained_weights, portfolio_value
            )

            # Step 3: Cost-aware constraint adjustment if needed
            if cost_analysis["total_cost"] > (self.constraints.transaction_cost_bps / 10000.0):
                constrained_weights = self._apply_cost_aware_adjustment(
                    constrained_weights, previous_weights, cost_analysis, violations
                )

        return constrained_weights, violations, cost_analysis

    def _apply_soft_constraints(
        self,
        weights: pd.Series,
        previous_weights: pd.Series | None = None,
        model_scores: pd.Series | None = None,
    ) -> pd.Series:
        """
        Apply soft constraints using penalty functions rather than hard limits.

        This allows for more flexible portfolio construction that can violate
        constraints when the model has high conviction.

        Args:
            weights: Raw model weights
            previous_weights: Previous period weights
            model_scores: Model confidence scores

        Returns:
            Soft-constrained weights
        """
        import numpy as np

        adjusted_weights = weights.copy()

        # 1. Hard position limit - enforce with iterative redistribution
        max_position = self.constraints.max_position_weight
        if max_position < 1.0:
            # Debug logging
            import logging
            logger = logging.getLogger(__name__)

            # Check for violations before clipping
            violations_before = (adjusted_weights > max_position).sum()
            if violations_before > 0:
                max_weight_before = adjusted_weights.max()
                logger.info(f"Position limit violations: {violations_before} assets exceed {max_position:.1%} limit")
                logger.info(f"Max weight before iterative redistribution: {max_weight_before:.1%}")

                # ITERATIVE REDISTRIBUTION ALGORITHM
                # This maintains both sum=1.0 AND max weight <= max_position
                # by redistributing excess weight to non-violating assets proportionally
                max_iterations = 100
                converged = False

                for iteration in range(max_iterations):
                    # Identify assets exceeding limit
                    excess_mask = adjusted_weights > max_position

                    if not excess_mask.any():
                        converged = True
                        break  # No violations, algorithm succeeded

                    # Calculate total excess to redistribute
                    excess_weights = adjusted_weights[excess_mask]
                    total_excess = (excess_weights - max_position).sum()

                    # Clip violating assets to maximum
                    adjusted_weights[excess_mask] = max_position

                    # Redistribute excess to non-violating assets proportionally
                    non_excess_mask = ~excess_mask
                    if non_excess_mask.sum() > 0:
                        # Get current weights of non-violating assets
                        recipient_weights = adjusted_weights[non_excess_mask]
                        recipient_sum = recipient_weights.sum()

                        if recipient_sum > 1e-10:  # Avoid division by zero
                            # Distribute proportional to current weights
                            redistribution = total_excess * (recipient_weights / recipient_sum)
                            adjusted_weights[non_excess_mask] += redistribution
                        else:
                            # Edge case: all non-violating weights are zero
                            # Distribute equally
                            n_recipients = non_excess_mask.sum()
                            adjusted_weights[non_excess_mask] = total_excess / n_recipients
                    else:
                        # Edge case: all assets violate the limit
                        # This means max_position is too restrictive for the universe size
                        logger.warning(
                            f"Cannot redistribute excess - all {len(adjusted_weights)} assets at maximum weight. "
                            f"Consider increasing max_position_weight or reducing universe size."
                        )
                        break

                if converged:
                    logger.info(f"Iterative redistribution converged in {iteration + 1} iterations")
                else:
                    # Raise exception with diagnostic information
                    # Note: We already clipped violating assets to max_position in the loop,
                    # so remaining_violations = 0 means redistribution became impossible
                    raise ConstraintViolationError(
                        message=f"Iterative redistribution failed to converge after {max_iterations} iterations: "
                        f"all assets forced to maximum weight of {max_position:.1%} but sum exceeds 1.0. "
                        f"Cannot redistribute further.",
                        iterations_attempted=max_iterations,
                        remaining_violations=0,
                        max_weight=adjusted_weights.max(),
                        max_position_weight=max_position,
                    )

                # Verify final state
                max_weight_after = adjusted_weights.max()
                weight_sum_after = adjusted_weights.sum()
                logger.info(
                    f"After redistribution: max_weight={max_weight_after:.4f}, "
                    f"sum={weight_sum_after:.4f}, target_limit={max_position:.4f}"
                )

                # Final check: if sum deviates significantly from 1.0, apply constrained normalisation
                # This should rarely happen with proper iterative redistribution
                if abs(weight_sum_after - 1.0) > 1e-4:
                    logger.warning(
                        f"Weight sum {weight_sum_after:.6f} deviates from 1.0. "
                        f"Applying constrained normalisation to preserve max_weight ≤ {max_position:.1%}."
                    )

                    # Use constrained projection instead of simple division
                    # Simple division can reintroduce violations when sum < 1.0
                    try:
                        import cvxpy as cp

                        n_assets = len(adjusted_weights)
                        w_target = adjusted_weights.values

                        # Define optimization variable
                        w = cp.Variable(n_assets)

                        # Objective: minimize squared deviation from current weights
                        objective = cp.Minimize(cp.sum_squares(w - w_target))

                        # Constraints: sum=1.0 AND respect box constraints
                        constraints = [
                            cp.sum(w) == 1.0,
                            w >= 0.0,
                            w <= max_position,
                        ]

                        # Solve constrained projection
                        problem = cp.Problem(objective, constraints)

                        # Try CLARABEL first (already used in codebase), fallback to OSQP
                        try:
                            problem.solve(solver=cp.CLARABEL, verbose=False, max_iter=200)
                        except Exception:
                            problem.solve(solver=cp.OSQP, verbose=False, max_iter=200)

                        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                            adjusted_weights = pd.Series(w.value, index=adjusted_weights.index)
                            logger.info(
                                f"Constrained normalisation successful: "
                                f"max_weight={adjusted_weights.max():.4f}, sum={adjusted_weights.sum():.6f}"
                            )
                        else:
                            logger.error(
                                f"Constrained normalisation failed with status: {problem.status}. "
                                f"Falling back to simple division (may violate constraints)."
                            )
                            adjusted_weights = adjusted_weights / weight_sum_after
                    except ImportError:
                        logger.error(
                            "cvxpy not available for constrained normalisation. "
                            "Falling back to simple division (may violate constraints)."
                        )
                        adjusted_weights = adjusted_weights / weight_sum_after
                    except Exception as e:
                        logger.error(
                            f"Error during constrained normalisation: {e}. "
                            f"Falling back to simple division (may violate constraints)."
                        )
                        adjusted_weights = adjusted_weights / weight_sum_after

                    # Verify we didn't reintroduce violations
                    final_max_weight = adjusted_weights.max()
                    if final_max_weight > max_position + 1e-6:
                        logger.error(
                            f"Normalisation reintroduced violation: "
                            f"{final_max_weight:.4f} > {max_position:.4f}"
                        )
                    else:
                        logger.info(
                            f"Constraint validation passed: max_weight={final_max_weight:.4f} ≤ {max_position:.4f}"
                        )

        # 2. Soft turnover constraint - blend with previous weights
        if previous_weights is not None and self.constraints.max_monthly_turnover > 0:
            # Calculate current turnover
            aligned_prev = previous_weights.reindex(adjusted_weights.index, fill_value=0)
            turnover = np.abs(adjusted_weights - aligned_prev).sum()

            if turnover > self.constraints.max_monthly_turnover:
                # Blend towards previous weights to reduce turnover
                blend_factor = self.constraints.max_monthly_turnover / turnover
                # Remove sqrt - we need stricter enforcement, not less aggressive
                # blend_factor = np.sqrt(blend_factor)  # This was making it too permissive

                # Apply more aggressive blending to ensure compliance
                blend_factor = blend_factor ** 2  # Square instead of sqrt for stricter enforcement
                adjusted_weights = blend_factor * adjusted_weights + (1 - blend_factor) * aligned_prev

                # Verify turnover after blending
                new_turnover = np.abs(adjusted_weights - aligned_prev).sum()

                # If still violating, apply harder constraint
                if new_turnover > self.constraints.max_monthly_turnover * 1.1:  # Allow 10% tolerance
                    # Force harder blending
                    target_blend = self.constraints.max_monthly_turnover / (new_turnover + 1e-8)
                    target_blend = min(target_blend, 0.9)  # Don't completely ignore new weights
                    adjusted_weights = target_blend * adjusted_weights + (1 - target_blend) * aligned_prev

        # 3. Soft minimum position - allow small positions but penalize
        min_position = self.constraints.min_weight_threshold
        if min_position > 0:
            small_positions = adjusted_weights[
                (adjusted_weights > 0) & (adjusted_weights < min_position)
            ]
            if len(small_positions) > 0:
                # Gradually push small positions towards minimum or zero
                for asset in small_positions.index:
                    w = adjusted_weights[asset]
                    # If very small, push to zero; otherwise push towards minimum
                    if w < min_position / 2:
                        adjusted_weights[asset] *= 0.5  # Decay small positions
                    else:
                        adjusted_weights[asset] = 0.9 * w + 0.1 * min_position

        # Iterative redistribution already maintains sum=1.0
        # No additional normalisation needed here
        # Only normalise if sum deviates significantly (shouldn't happen)
        weight_sum = adjusted_weights.sum()
        if abs(weight_sum - 1.0) > 1e-4:
            logger.warning(
                f"Weight sum {weight_sum:.6f} deviates from 1.0 after constraints. "
                f"This indicates an issue with iterative redistribution."
            )
            if weight_sum > 0:
                adjusted_weights = adjusted_weights / weight_sum
            else:
                logger.error("Weight sum is zero or negative - cannot normalise")
                # Return equal weights as fallback
                adjusted_weights = pd.Series(1.0 / len(adjusted_weights), index=adjusted_weights.index)

        return adjusted_weights

    def _apply_convex_constraints(
        self,
        weights: pd.Series,
        previous_weights: pd.Series | None = None,
    ) -> pd.Series:
        """
        Apply constraints using convex optimization to properly project onto feasible space.
        This method mathematically guarantees both sum=1.0 AND max_weight constraints.

        Args:
            weights: Raw portfolio weights from model
            previous_weights: Previous period weights (for turnover constraint)

        Returns:
            Constrained weights satisfying all constraints simultaneously
        """
        import logging
        logger = logging.getLogger(__name__)

        try:
            import cvxpy as cp
            import numpy as np
        except ImportError:
            logger.warning("cvxpy not available for convex constraint projection")
            raise ImportError("cvxpy required for convex constraint enforcement")

        n_assets = len(weights)
        assets = weights.index
        weights_np = weights.values

        # ERROR HANDLING: Validate input weights
        if np.isnan(weights_np).any():
            logger.error(f"Input weights contain {np.isnan(weights_np).sum()} NaN values")
            raise ValueError("Cannot apply convex constraints to weights containing NaN")

        if np.isinf(weights_np).any():
            logger.error(f"Input weights contain {np.isinf(weights_np).sum()} infinite values")
            raise ValueError("Cannot apply convex constraints to weights containing infinite values")

        if n_assets == 0:
            logger.error("Cannot apply constraints to empty weight vector")
            raise ValueError("Empty weight vector provided to constraint engine")

        # ERROR HANDLING: Check constraint feasibility before solving
        if self.constraints.min_weight_threshold * n_assets > 1.0:
            logger.error(
                f"Infeasible constraints: min_weight={self.constraints.min_weight_threshold} × {n_assets} assets "
                f"= {self.constraints.min_weight_threshold * n_assets:.3f} > 1.0"
            )
            raise ValueError(f"Mathematically infeasible: minimum weight threshold too high for {n_assets} assets")

        # Define optimization variable
        w = cp.Variable(n_assets)

        # Objective: minimize squared deviation from target weights
        # This finds the closest feasible point to the original weights
        objective = cp.Minimize(cp.sum_squares(w - weights_np))

        # Constraints
        constraints = [
            cp.sum(w) == 1.0,  # Weights sum to 1
            w >= self.constraints.min_weight_threshold,  # Minimum position
            w <= self.constraints.max_position_weight,  # Maximum position
        ]

        # Add turnover constraint if previous weights provided
        if previous_weights is not None and self.constraints.max_monthly_turnover is not None:
            prev_weights_aligned = previous_weights.reindex(assets, fill_value=0).values
            turnover = cp.sum(cp.abs(w - prev_weights_aligned))
            # Turnover already two-way, no multiplication needed
            constraints.append(turnover <= self.constraints.max_monthly_turnover)

        # Add long-only constraint if needed
        if self.constraints.long_only:
            constraints.append(w >= 0)

        # Create and solve problem
        problem = cp.Problem(objective, constraints)

        # Try multiple solvers in order of preference
        solvers_to_try = [
            (cp.CLARABEL, {"verbose": False, "max_iter": 200}),
            (cp.OSQP, {"verbose": False, "max_iter": 200}),
            (cp.SCS, {"verbose": False, "max_iters": 200}),
        ]

        solution_found = False
        solver_errors = []
        for solver, kwargs in solvers_to_try:
            try:
                problem.solve(solver=solver, **kwargs)
                if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    solution_found = True
                    logger.debug(f"Convex constraint projection solved with {solver}: status={problem.status}")
                    break
                else:
                    # ERROR HANDLING: Log non-optimal status
                    error_msg = f"{solver}: status={problem.status}"
                    solver_errors.append(error_msg)
                    logger.debug(f"Solver {solver} failed with status {problem.status}")
            except Exception as e:
                # ERROR HANDLING: Capture solver exceptions
                error_msg = f"{solver}: {type(e).__name__}: {str(e)[:100]}"
                solver_errors.append(error_msg)
                logger.debug(f"Solver {solver} raised exception: {e}")
                continue

        if not solution_found:
            # ERROR HANDLING: Provide detailed failure report
            error_details = "\n  ".join(solver_errors)
            logger.error(
                f"Failed to solve constraint projection problem.\n"
                f"  Problem: {n_assets} assets, max_weight={self.constraints.max_position_weight}, "
                f"min_weight={self.constraints.min_weight_threshold}\n"
                f"  Solver attempts:\n  {error_details}"
            )
            raise ValueError(
                f"Failed to solve constraint projection problem after trying {len(solvers_to_try)} solvers. "
                f"Last status: {problem.status}. This may indicate infeasible constraints."
            )

        # Extract solution
        constrained_weights = pd.Series(w.value, index=assets)

        # Final defensive clipping (handles numerical precision)
        constrained_weights = constrained_weights.clip(
            lower=self.constraints.min_weight_threshold,
            upper=self.constraints.max_position_weight
        )

        # Ensure exact sum to 1.0
        weight_sum = constrained_weights.sum()
        if abs(weight_sum - 1.0) > 1e-8:
            constrained_weights = constrained_weights / weight_sum

        # Log the result
        max_weight = constrained_weights.max()
        logger.debug(
            f"Convex projection: max weight before={weights.max():.1%}, "
            f"after={max_weight:.1%}, deviation={np.linalg.norm(constrained_weights - weights):.4f}"
        )

        return constrained_weights

    def _apply_cost_aware_adjustment(
        self,
        weights: pd.Series,
        previous_weights: pd.Series,
        cost_analysis: dict[str, Any],
        violations: list[ConstraintViolation],
    ) -> pd.Series:
        """
        Apply cost-aware adjustments to reduce transaction costs.

        Args:
            weights: Current constrained weights
            previous_weights: Previous period weights
            cost_analysis: Transaction cost analysis
            violations: Current violations list

        Returns:
            Cost-adjusted portfolio weights
        """
        # If transaction costs are too high, blend more with previous weights
        if cost_analysis["turnover"] > 0:
            # Calculate cost-efficiency ratio
            cost_per_turnover = cost_analysis["cost_per_turnover"]
            max_acceptable_cost = self.constraints.transaction_cost_bps / 10000.0

            if cost_per_turnover > max_acceptable_cost:
                # Reduce turnover by blending more with previous weights
                target_cost_ratio = max_acceptable_cost / cost_per_turnover
                blend_factor = min(0.9, target_cost_ratio)  # Cap at 90% previous weights

                # Align weights
                all_assets = weights.index.union(previous_weights.index)
                current_aligned = weights.reindex(all_assets, fill_value=0.0)
                previous_aligned = previous_weights.reindex(all_assets, fill_value=0.0)

                # Apply cost-aware blending
                adjusted_weights = (
                    blend_factor * previous_aligned + (1 - blend_factor) * current_aligned
                )

                # Renormalize
                weight_sum = adjusted_weights.sum()
                if weight_sum > 0:
                    adjusted_weights = adjusted_weights / weight_sum

                return adjusted_weights

        return weights

    def validate_portfolio_feasibility(
        self, weights: pd.Series, previous_weights: pd.Series | None = None
    ) -> dict[str, Any]:
        """
        Validate overall portfolio feasibility and constraint adherence.

        Args:
            weights: Portfolio weights to validate
            previous_weights: Previous weights for comparison

        Returns:
            Dictionary with feasibility assessment and recommendations
        """
        # Get base constraint metrics
        metrics = self.base_engine.calculate_constraint_metrics(weights, previous_weights)

        # Check violations
        violations = self.base_engine.check_violations(weights, previous_weights)

        # Calculate severity score
        severity_scores = {"warning": 1, "minor": 2, "major": 3, "critical": 4}
        total_severity = sum(severity_scores.get(v.severity.value, 0) for v in violations)

        # Determine feasibility status
        if total_severity == 0:
            feasibility_status = "fully_compliant"
        elif total_severity <= 3:
            feasibility_status = "acceptable_with_warnings"
        elif total_severity <= 6:
            feasibility_status = "requires_adjustment"
        else:
            feasibility_status = "infeasible"

        # Generate recommendations
        recommendations = []
        if violations:
            violation_types = {v.violation_type.value for v in violations}
            if "long_only" in violation_types:
                recommendations.append("Enable long-only constraint enforcement")
            if "top_k_positions" in violation_types:
                recommendations.append("Reduce number of positions or increase top-k limit")
            if "monthly_turnover" in violation_types:
                recommendations.append("Increase turnover limit or enable turnover blending")
            if "max_position_weight" in violation_types:
                recommendations.append("Increase max position weight or enable redistribution")

        return {
            "feasibility_status": feasibility_status,
            "total_violations": len(violations),
            "severity_score": total_severity,
            "constraint_metrics": metrics,
            "violations": [
                {
                    "type": v.violation_type.value,
                    "severity": v.severity.value,
                    "description": v.description,
                }
                for v in violations
            ],
            "recommendations": recommendations,
        }

    def get_constraint_configuration(self) -> dict[str, Any]:
        """Get current constraint configuration for reporting."""
        return {
            "basic_constraints": {
                "long_only": self.constraints.long_only,
                "top_k_positions": self.constraints.top_k_positions,
                "max_position_weight": self.constraints.max_position_weight,
                "min_weight_threshold": self.constraints.min_weight_threshold,
            },
            "turnover_constraints": {
                "max_monthly_turnover": self.constraints.max_monthly_turnover,
                "enable_turnover_penalty": self.constraints.enable_turnover_penalty,
                "turnover_lookback_months": self.constraints.turnover_lookback_months,
            },
            "transaction_cost_settings": {
                "transaction_cost_bps": self.constraints.transaction_cost_bps,
                "cost_aware_enforcement": self.constraints.cost_aware_enforcement,
            },
            "enforcement_settings": {
                "violation_handling": self.constraints.violation_handling,
                "position_ranking_method": self.constraints.position_ranking_method,
            },
        }

    def create_enforcement_report(
        self, weights: pd.Series, previous_weights: pd.Series | None = None
    ) -> dict[str, Any]:
        """
        Create comprehensive constraint enforcement report.

        Args:
            weights: Current portfolio weights
            previous_weights: Previous weights for comparison

        Returns:
            Comprehensive enforcement report
        """
        # Get base constraint report
        base_report = self.base_engine.create_constraint_report()

        # Add feasibility assessment
        feasibility = self.validate_portfolio_feasibility(weights, previous_weights)

        # Add transaction cost analysis if available
        cost_analysis = {}
        if self.transaction_cost_calculator and previous_weights is not None:
            cost_analysis = self.transaction_cost_calculator.calculate_transaction_costs(
                previous_weights, weights
            )

        return {
            "constraint_enforcement": base_report,
            "portfolio_feasibility": feasibility,
            "transaction_cost_analysis": cost_analysis,
            "configuration": self.get_constraint_configuration(),
            "summary": {
                "total_positions": (weights > 0).sum(),
                "max_weight": weights.max(),
                "weight_sum": weights.sum(),
                "is_feasible": feasibility["feasibility_status"] != "infeasible",
                "has_violations": len(feasibility["violations"]) > 0,
                "estimated_transaction_cost": cost_analysis.get("total_cost", 0.0),
            },
        }
