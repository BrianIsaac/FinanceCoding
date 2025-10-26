"""
HRP allocation engine using recursive bisection algorithm.

This module implements the recursive bisection allocation algorithm that
distributes capital based on hierarchical clustering structure and risk parity principles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AllocationConfig:
    """Configuration for HRP allocation algorithm."""

    risk_measure: str = "variance"  # variance, vol, equal
    min_allocation: float = 0.001  # Minimum allocation per asset
    max_allocation: float = 0.5  # Maximum allocation per asset
    allocation_precision: int = 6  # Decimal precision for allocations


class HRPAllocation:
    """
    Recursive bisection allocation engine for HRP portfolio construction.

    Implements the hierarchical risk parity allocation algorithm that distributes
    capital through the cluster tree using equal risk contribution principles.
    """

    def __init__(self, config: AllocationConfig | None = None):
        """
        Initialize HRP allocation engine.

        Args:
            config: Allocation configuration parameters
        """
        self.config = config or AllocationConfig()

    def recursive_bisection(
        self,
        covariance_matrix: pd.DataFrame,
        cluster_tree: dict,
        initial_weights: pd.Series | None = None,
    ) -> pd.Series:
        """
        Allocate capital through recursive cluster bisection.

        Args:
            covariance_matrix: Asset covariance matrix
            cluster_tree: Hierarchical cluster tree structure
            initial_weights: Optional initial weight distribution (defaults to equal)

        Returns:
            Portfolio weights as pandas Series

        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        self._validate_inputs(covariance_matrix, cluster_tree)

        # Get asset list from cluster tree
        asset_names = cluster_tree.get("assets", [])

        if not asset_names:
            raise ValueError("Empty asset list in cluster tree")

        # Initialize equal weights if not provided
        if initial_weights is None:
            initial_weights = pd.Series(1.0 / len(asset_names), index=asset_names, dtype=float)

        # Ensure covariance matrix alignment
        aligned_cov = self._align_covariance_matrix(covariance_matrix, asset_names)

        # Start recursive bisection from root
        weights = self._bisect_cluster(
            cluster_tree, aligned_cov, initial_weights.sum()  # Total allocation for this cluster
        )

        # Apply allocation constraints
        constrained_weights = self._apply_allocation_constraints(weights)

        return constrained_weights

    def _bisect_cluster(
        self, cluster_node: dict, covariance_matrix: pd.DataFrame, total_allocation: float
    ) -> pd.Series:
        """
        Recursively bisect cluster and allocate capital.

        Args:
            cluster_node: Current cluster node
            covariance_matrix: Aligned covariance matrix
            total_allocation: Total allocation available for this cluster

        Returns:
            Allocation weights for assets in this cluster
        """
        if cluster_node["type"] == "leaf":
            # Leaf node: return single asset allocation
            asset = cluster_node["asset"]
            return pd.Series([total_allocation], index=[asset])

        # Internal node: split allocation between left and right subclusters
        left_node = cluster_node["left"]
        right_node = cluster_node["right"]

        # Calculate risk-based allocation split
        left_risk = self._calculate_cluster_risk(left_node["assets"], covariance_matrix)
        right_risk = self._calculate_cluster_risk(right_node["assets"], covariance_matrix)

        # Inverse risk weighting for risk parity
        if left_risk > 0 and right_risk > 0:
            inv_left_risk = 1.0 / left_risk
            inv_right_risk = 1.0 / right_risk
            risk_sum = inv_left_risk + inv_right_risk

            left_allocation = total_allocation * (inv_left_risk / risk_sum)
            right_allocation = total_allocation * (inv_right_risk / risk_sum)
        else:
            # Fallback to equal allocation if risk calculation fails
            left_allocation = total_allocation * 0.5
            right_allocation = total_allocation * 0.5

        # Recursively allocate to subclusters
        left_weights = self._bisect_cluster(left_node, covariance_matrix, left_allocation)
        right_weights = self._bisect_cluster(right_node, covariance_matrix, right_allocation)

        # Combine weights
        combined_weights = pd.concat([left_weights, right_weights])
        return combined_weights

    def _calculate_cluster_risk(self, assets: list[str], covariance_matrix: pd.DataFrame) -> float:
        """
        Calculate risk measure for a cluster of assets.

        Args:
            assets: List of asset names in cluster
            covariance_matrix: Full covariance matrix

        Returns:
            Risk measure for the cluster
        """
        if not assets or len(assets) == 0:
            return 0.0

        try:
            # Get cluster covariance submatrix
            cluster_assets = [a for a in assets if a in covariance_matrix.index]
            if not cluster_assets:
                return 0.0

            cluster_cov = covariance_matrix.loc[cluster_assets, cluster_assets]

            if cluster_cov.empty:
                return 0.0

            # Calculate risk based on configuration
            if self.config.risk_measure == "variance":
                if len(cluster_assets) == 1:
                    # Single asset variance
                    return cluster_cov.iloc[0, 0]
                else:
                    # Equal-weighted portfolio variance
                    n = len(cluster_assets)
                    equal_weights = np.ones(n) / n
                    portfolio_var = np.dot(equal_weights, np.dot(cluster_cov.values, equal_weights))
                    return max(portfolio_var, 1e-8)  # Avoid zero risk

            elif self.config.risk_measure == "vol":
                # Recursively call with variance and take sqrt
                variance_risk = self._calculate_cluster_risk_variance(assets, covariance_matrix)
                return np.sqrt(max(variance_risk, 1e-8))

            elif self.config.risk_measure == "equal":
                return 1.0  # Equal risk treatment

            else:
                raise ValueError(f"Unknown risk measure: {self.config.risk_measure}")

        except Exception as e:
            # Log the failure and fallback to equal risk
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Cluster risk calculation failed: {str(e)}, using equal risk")
            return 1.0

    def _calculate_cluster_risk_variance(
        self, assets: list[str], covariance_matrix: pd.DataFrame
    ) -> float:
        """Helper method for variance risk calculation."""
        if not assets or len(assets) == 0:
            return 0.0

        try:
            cluster_assets = [a for a in assets if a in covariance_matrix.index]
            if not cluster_assets:
                return 0.0

            cluster_cov = covariance_matrix.loc[cluster_assets, cluster_assets]

            if cluster_cov.empty:
                return 0.0

            if len(cluster_assets) == 1:
                return cluster_cov.iloc[0, 0]
            else:
                n = len(cluster_assets)
                equal_weights = np.ones(n) / n
                portfolio_var = np.dot(equal_weights, np.dot(cluster_cov.values, equal_weights))
                return max(portfolio_var, 1e-8)

        except Exception as e:
            # Log the failure
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Variance risk calculation failed: {str(e)}")
            return 1.0

    def calculate_risk_budgets(
        self, weights: pd.Series, covariance_matrix: pd.DataFrame, cluster_tree: dict
    ) -> dict[str, float]:
        """
        Calculate risk budgeting for equal risk contribution validation.

        Args:
            weights: Portfolio weights
            covariance_matrix: Asset covariance matrix
            cluster_tree: Cluster tree structure

        Returns:
            Dictionary mapping assets to their risk contributions
        """
        # Ensure alignment
        common_assets = weights.index.intersection(covariance_matrix.index)
        aligned_weights = weights.reindex(common_assets, fill_value=0.0)
        aligned_cov = covariance_matrix.loc[common_assets, common_assets]

        if len(aligned_weights) == 0 or aligned_cov.empty:
            return {}

        # Calculate portfolio variance
        portfolio_var = np.dot(
            aligned_weights.values, np.dot(aligned_cov.values, aligned_weights.values)
        )

        if portfolio_var <= 0:
            return dict.fromkeys(aligned_weights.index, 0.0)

        # Calculate marginal risk contributions
        marginal_contributions = np.dot(aligned_cov.values, aligned_weights.values)

        # Calculate risk contributions (weight * marginal contribution / portfolio risk)
        portfolio_vol = np.sqrt(portfolio_var)
        risk_contributions = {}

        for i, asset in enumerate(aligned_weights.index):
            risk_contrib = (aligned_weights.iloc[i] * marginal_contributions[i]) / portfolio_vol
            risk_contributions[asset] = risk_contrib

        return risk_contributions

    def optimize_cluster_allocation(
        self, assets: list[str], covariance_matrix: pd.DataFrame, target_allocation: float
    ) -> pd.Series:
        """
        Optimize allocation within a single cluster for risk parity.

        Args:
            assets: List of assets in cluster
            covariance_matrix: Full covariance matrix
            target_allocation: Total allocation target for this cluster

        Returns:
            Optimized weights for cluster assets
        """
        if not assets:
            return pd.Series(dtype=float)

        cluster_assets = [a for a in assets if a in covariance_matrix.index]
        if not cluster_assets:
            return pd.Series(dtype=float)

        cluster_cov = covariance_matrix.loc[cluster_assets, cluster_assets]

        # For single asset, return target allocation
        if len(cluster_assets) == 1:
            return pd.Series([target_allocation], index=cluster_assets)

        # Calculate inverse volatility weights for risk parity
        try:
            asset_vols = np.sqrt(np.diag(cluster_cov.values))
            asset_vols = np.maximum(asset_vols, 1e-8)  # Avoid division by zero

            inv_vol_weights = 1.0 / asset_vols
            normalized_weights = inv_vol_weights / inv_vol_weights.sum()

            # Scale to target allocation
            scaled_weights = normalized_weights * target_allocation

            return pd.Series(scaled_weights, index=cluster_assets)

        except Exception as e:
            # Log allocation failure and fallback to equal weights
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Allocation failed for {len(cluster_assets)} assets: {str(e)}")
            equal_weight = target_allocation / len(cluster_assets)
            return pd.Series([equal_weight] * len(cluster_assets), index=cluster_assets)

    def _validate_inputs(self, covariance_matrix: pd.DataFrame, cluster_tree: dict) -> None:
        """Validate allocation inputs."""
        if covariance_matrix.empty:
            raise ValueError("Empty covariance matrix")

        if not isinstance(cluster_tree, dict):
            raise ValueError("Cluster tree must be a dictionary")

        if "assets" not in cluster_tree:
            raise ValueError("Cluster tree must contain 'assets' field")

    def _align_covariance_matrix(
        self, covariance_matrix: pd.DataFrame, asset_names: list[str]
    ) -> pd.DataFrame:
        """Align covariance matrix with asset names from cluster tree."""
        common_assets = [a for a in asset_names if a in covariance_matrix.index]

        if not common_assets:
            raise ValueError("No common assets between covariance matrix and cluster tree")

        return covariance_matrix.loc[common_assets, common_assets]

    def _apply_allocation_constraints(self, weights: pd.Series) -> pd.Series:
        """Apply allocation constraints and normalize."""
        # Handle empty weights
        if weights.empty:
            return weights

        # Iteratively enforce constraints to handle normalization issues
        constrained = weights.copy()
        max_iters = 10

        # For single-asset portfolios, allow allocation up to 1.0
        effective_max_allocation = 1.0 if len(weights) == 1 else self.config.max_allocation

        for _iteration in range(max_iters):
            # Apply minimum/maximum allocation constraints
            constrained = constrained.clip(
                lower=self.config.min_allocation, upper=effective_max_allocation
            )
            # Remove tiny allocations
            constrained = constrained.where(constrained >= self.config.min_allocation, 0.0)

            # Normalize to sum to 1.0
            weight_sum = constrained.sum()
            if weight_sum <= 0:
                # Equal weights fallback
                constrained = pd.Series(1.0 / len(constrained), index=constrained.index)
                break

            constrained = constrained / weight_sum

            # Check if constraints are satisfied after normalization (with small tolerance)
            if (constrained <= effective_max_allocation + 1e-8).all():
                break

            # If normalization re-violated max constraints, redistribute excess
            violating_mask = constrained > effective_max_allocation
            if violating_mask.any():
                excess_weight = (constrained[violating_mask] - effective_max_allocation).sum()
                constrained[violating_mask] = effective_max_allocation

                # Redistribute excess to non-violating assets proportionally
                non_violating_mask = ~violating_mask
                if non_violating_mask.any():
                    redistribution_base = constrained[non_violating_mask]
                    if redistribution_base.sum() > 0:
                        redistribution_weights = redistribution_base / redistribution_base.sum()
                        constrained[non_violating_mask] += excess_weight * redistribution_weights

        # Final constraint enforcement with precise clipping
        constrained = constrained.clip(upper=effective_max_allocation)

        # Final normalization
        final_sum = constrained.sum()
        if final_sum > 0:
            constrained = constrained / final_sum
            # Ensure no constraint violations remain after final normalization
            constrained = constrained.clip(upper=effective_max_allocation)

        # Round to specified precision
        constrained = constrained.round(self.config.allocation_precision)

        return constrained

    def handle_edge_cases(
        self, cluster_tree: dict, covariance_matrix: pd.DataFrame
    ) -> tuple[bool, str]:
        """
        Handle edge cases for allocation (single assets, empty clusters).

        Args:
            cluster_tree: Cluster tree structure
            covariance_matrix: Asset covariance matrix

        Returns:
            Tuple of (needs_special_handling, handling_type)
        """
        assets = cluster_tree.get("assets", [])

        # Single asset case
        if len(assets) == 1:
            return True, "single_asset"

        # Empty cluster case
        if len(assets) == 0:
            return True, "empty_cluster"

        # Check for assets missing from covariance matrix
        available_assets = [a for a in assets if a in covariance_matrix.index]
        if len(available_assets) < len(assets) * 0.5:  # Less than 50% coverage
            return True, "insufficient_coverage"

        # Check for degenerate covariance matrix
        try:
            cluster_cov = covariance_matrix.loc[available_assets, available_assets]
            if np.any(np.diag(cluster_cov.values) <= 0):
                return True, "zero_variance_assets"
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Covariance validation error: {str(e)}")
            return True, "covariance_error"

        return False, "normal"
