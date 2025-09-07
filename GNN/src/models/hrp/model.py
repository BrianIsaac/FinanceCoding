"""
HRP portfolio model implementing the PortfolioModel interface.

This module provides the main HRP model class that integrates clustering,
allocation, and constraint enforcement for portfolio construction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from src.models.base.portfolio_model import PortfolioConstraints, PortfolioModel
from src.models.hrp.allocation import AllocationConfig, HRPAllocation
from src.models.hrp.clustering import ClusteringConfig, HRPClustering
from src.models.hrp.universe_integration import HRPUniverseIntegration


@dataclass
class HRPConfig:
    """Complete HRP model configuration."""

    lookback_days: int = 756  # 3 years of daily data
    clustering_config: ClusteringConfig = None
    allocation_config: AllocationConfig = None
    min_observations: int = 252  # Minimum data overlap
    correlation_method: str = "pearson"  # Correlation calculation method
    rebalance_frequency: str = "monthly"  # Rebalancing frequency

    def __post_init__(self):
        """Initialize sub-configurations if not provided."""
        if self.clustering_config is None:
            self.clustering_config = ClusteringConfig(
                min_observations=self.min_observations, correlation_method=self.correlation_method
            )
        if self.allocation_config is None:
            self.allocation_config = AllocationConfig()


class HRPModel(PortfolioModel):
    """
    Hierarchical Risk Parity portfolio model.

    Implements HRP allocation using correlation distance clustering and
    recursive bisection with integrated constraint enforcement.
    """

    def __init__(
        self,
        constraints: PortfolioConstraints,
        hrp_config: HRPConfig | None = None,
        universe_integration: HRPUniverseIntegration | None = None,
    ):
        """
        Initialize HRP portfolio model.

        Args:
            constraints: Portfolio constraints configuration
            hrp_config: HRP-specific configuration parameters
            universe_integration: Optional universe integration component
        """
        super().__init__(constraints)

        self.hrp_config = hrp_config or HRPConfig()
        self.clustering_engine = HRPClustering(self.hrp_config.clustering_config)
        self.allocation_engine = HRPAllocation(self.hrp_config.allocation_config)
        self.universe_integration = universe_integration

        # Model state
        self._fitted_covariance: pd.DataFrame | None = None
        self._fitted_returns: pd.DataFrame | None = None
        self._fitted_universe: list[str] | None = None
        self._fit_period: tuple[pd.Timestamp, pd.Timestamp] | None = None

    def fit(
        self,
        returns: pd.DataFrame,
        universe: list[str],
        fit_period: tuple[pd.Timestamp, pd.Timestamp],
    ) -> None:
        """
        Train HRP model on historical correlation patterns.

        Args:
            returns: Historical returns DataFrame with datetime index and asset columns
            universe: List of asset tickers to include in optimization
            fit_period: (start_date, end_date) tuple defining training period

        Raises:
            ValueError: If returns data is insufficient or invalid
        """
        # Validate inputs
        self._validate_fit_inputs(returns, universe, fit_period)

        # Filter returns for fit period and universe
        start_date, end_date = fit_period
        time_mask = (returns.index >= start_date) & (returns.index <= end_date)
        period_returns = returns[time_mask]

        # Filter for universe assets
        available_assets = [asset for asset in universe if asset in period_returns.columns]
        if len(available_assets) < len(universe) * 0.8:  # Require 80% coverage
            missing_assets = set(universe) - set(available_assets)
            raise ValueError(
                f"Insufficient asset coverage: {len(available_assets)}/{len(universe)}. "
                f"Missing: {list(missing_assets)[:10]}"
            )

        fitted_returns = period_returns[available_assets]

        # Validate data sufficiency
        if len(fitted_returns) < self.hrp_config.min_observations:
            raise ValueError(
                f"Insufficient observations: {len(fitted_returns)} < {self.hrp_config.min_observations}"
            )

        # Remove assets with too much missing data
        coverage_threshold = 0.8
        asset_coverage = fitted_returns.count() / len(fitted_returns)
        sufficient_assets = asset_coverage[asset_coverage >= coverage_threshold].index.tolist()

        min_assets = min(2, len(universe))  # At least 2 assets for clustering, or size of universe
        if len(sufficient_assets) < min_assets:
            raise ValueError(f"Too few assets with sufficient data: {len(sufficient_assets)}")

        final_returns = fitted_returns[sufficient_assets]

        # Calculate covariance matrix for allocation
        covariance_matrix = final_returns.cov()

        # Validate covariance matrix
        is_valid, error_msg = self.clustering_engine.validate_correlation_matrix(
            final_returns.corr()
        )
        if not is_valid:
            raise ValueError(f"Invalid correlation matrix: {error_msg}")

        # Store fitted state
        self._fitted_returns = final_returns
        self._fitted_covariance = covariance_matrix
        self._fitted_universe = sufficient_assets
        self._fit_period = fit_period
        self.is_fitted = True

    def predict_weights(self, date: pd.Timestamp, universe: list[str]) -> pd.Series:
        """
        Generate HRP portfolio weights for rebalancing date.

        Args:
            date: Rebalancing date for which to generate weights
            universe: List of asset tickers (must be subset of fitted universe)

        Returns:
            Portfolio weights as pandas Series with asset tickers as index.
            Weights sum to 1.0 and satisfy all portfolio constraints.

        Raises:
            ValueError: If model is not fitted or universe is invalid
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating predictions")

        if self._fitted_returns is None or self._fitted_covariance is None:
            raise ValueError("Model state is invalid - refit required")

        # Validate universe compatibility
        available_assets = [asset for asset in universe if asset in self._fitted_universe]
        if len(available_assets) == 0:
            raise ValueError("No assets in common between prediction universe and fitted universe")

        # Handle universe integration if available
        if self.universe_integration is not None:
            try:
                # Get universe-aligned returns for clustering
                aligned_returns = self.universe_integration.align_returns_with_universe(
                    self._fitted_returns, date, lookback_days=self.hrp_config.lookback_days
                )
                clustering_universe = aligned_returns.columns.tolist()

                # Filter for prediction universe
                prediction_assets = [
                    asset for asset in clustering_universe if asset in available_assets
                ]
            except Exception:
                # Fallback to available assets if universe integration fails
                prediction_assets = available_assets
        else:
            prediction_assets = available_assets

        if len(prediction_assets) < 2:
            # Single asset case - return equal weight
            return pd.Series(1.0, index=prediction_assets)

        # Use lookback window around prediction date
        end_date = date
        start_date = date - pd.Timedelta(days=self.hrp_config.lookback_days)

        # Filter fitted returns for lookback period
        lookback_mask = (self._fitted_returns.index >= start_date) & (
            self._fitted_returns.index < end_date
        )
        lookback_returns = self._fitted_returns[lookback_mask]

        # Ensure sufficient data
        if len(lookback_returns) < self.hrp_config.min_observations:
            # Use full fitted returns if lookback is insufficient
            lookback_returns = self._fitted_returns

        # Filter for prediction assets
        prediction_returns = lookback_returns[prediction_assets]

        # Build correlation distance matrix
        try:
            distance_matrix = self.clustering_engine.build_correlation_distance(prediction_returns)
        except ValueError:
            # Fallback to equal weights if clustering fails
            return pd.Series(1.0 / len(prediction_assets), index=prediction_assets)

        # Perform hierarchical clustering
        try:
            linkage_matrix = self.clustering_engine.hierarchical_clustering(distance_matrix)
            cluster_tree = self.clustering_engine.build_cluster_tree(
                prediction_assets, linkage_matrix
            )
        except Exception:
            # Fallback to equal weights if clustering fails
            return pd.Series(1.0 / len(prediction_assets), index=prediction_assets)

        # Calculate covariance matrix for allocation
        prediction_covariance = prediction_returns.cov()

        # Perform recursive bisection allocation
        try:
            raw_weights = self.allocation_engine.recursive_bisection(
                prediction_covariance, cluster_tree
            )
        except Exception:
            # Fallback to equal weights if allocation fails
            raw_weights = pd.Series(1.0 / len(prediction_assets), index=prediction_assets)

        # Apply portfolio constraints using base class method
        constrained_weights = self.validate_weights(raw_weights)

        # Ensure minimum weight threshold is applied
        if (
            hasattr(self.constraints, "min_weight_threshold")
            and self.constraints.min_weight_threshold > 0
        ):
            constrained_weights = constrained_weights.where(
                constrained_weights >= self.constraints.min_weight_threshold, 0.0
            )
            # Renormalize after applying threshold
            weight_sum = constrained_weights.sum()
            if weight_sum > 0:
                constrained_weights = constrained_weights / weight_sum

        return constrained_weights

    def get_model_info(self) -> dict[str, Any]:
        """
        Return HRP model metadata for analysis and reproducibility.

        Returns:
            Dictionary containing model type, hyperparameters, constraints,
            and other relevant metadata for performance analysis.
        """
        info = {
            "model_type": "HRP",
            "model_class": self.__class__.__name__,
            "is_fitted": self.is_fitted,
            "hrp_config": asdict(self.hrp_config),
            "constraints": asdict(self.constraints),
        }

        if self.is_fitted:
            info.update(
                {
                    "fitted_universe_size": (
                        len(self._fitted_universe) if self._fitted_universe else 0
                    ),
                    "fit_period_start": (
                        self._fit_period[0].strftime("%Y-%m-%d") if self._fit_period else None
                    ),
                    "fit_period_end": (
                        self._fit_period[1].strftime("%Y-%m-%d") if self._fit_period else None
                    ),
                    "training_observations": (
                        len(self._fitted_returns) if self._fitted_returns is not None else 0
                    ),
                }
            )

        return info

    def get_clustering_diagnostics(self, date: pd.Timestamp, universe: list[str]) -> dict[str, Any]:
        """
        Get detailed clustering diagnostics for analysis.

        Args:
            date: Analysis date
            universe: Asset universe

        Returns:
            Dictionary with clustering metrics and tree structure
        """
        if not self.is_fitted:
            return {"error": "Model not fitted"}

        try:
            # Get prediction assets
            available_assets = [asset for asset in universe if asset in self._fitted_universe]

            if len(available_assets) < 2:
                return {"error": "Insufficient assets for clustering"}

            # Get lookback returns
            end_date = date
            start_date = date - pd.Timedelta(days=self.hrp_config.lookback_days)

            lookback_mask = (self._fitted_returns.index >= start_date) & (
                self._fitted_returns.index < end_date
            )
            lookback_returns = self._fitted_returns[lookback_mask]
            prediction_returns = lookback_returns[available_assets]

            # Build clustering
            distance_matrix = self.clustering_engine.build_correlation_distance(prediction_returns)
            linkage_matrix = self.clustering_engine.hierarchical_clustering(distance_matrix)
            cluster_tree = self.clustering_engine.build_cluster_tree(
                available_assets, linkage_matrix
            )

            # Calculate diagnostics
            diagnostics = {
                "n_assets": len(available_assets),
                "n_observations": len(prediction_returns),
                "linkage_method": self.hrp_config.clustering_config.linkage_method,
                "correlation_method": self.hrp_config.clustering_config.correlation_method,
                "cluster_tree_depth": self._calculate_tree_depth(cluster_tree),
                "distance_matrix_shape": distance_matrix.shape,
                "linkage_matrix_shape": linkage_matrix.shape,
                "assets": available_assets,
            }

            return diagnostics

        except Exception as e:
            return {"error": str(e)}

    def _validate_fit_inputs(
        self,
        returns: pd.DataFrame,
        universe: list[str],
        fit_period: tuple[pd.Timestamp, pd.Timestamp],
    ) -> None:
        """Validate inputs for fit method."""
        if returns.empty:
            raise ValueError("Empty returns DataFrame")

        if not universe:
            raise ValueError("Empty universe list")

        start_date, end_date = fit_period
        if start_date >= end_date:
            raise ValueError("Invalid fit period: start_date >= end_date")

        # Check minimum period length
        period_days = (end_date - start_date).days
        if period_days < self.hrp_config.min_observations:
            raise ValueError(
                f"Fit period too short: {period_days} days < {self.hrp_config.min_observations}"
            )

    def _calculate_tree_depth(self, cluster_tree: dict) -> int:
        """Calculate the depth of the cluster tree."""
        if cluster_tree["type"] == "leaf":
            return 1

        left_depth = self._calculate_tree_depth(cluster_tree["left"])
        right_depth = self._calculate_tree_depth(cluster_tree["right"])

        return max(left_depth, right_depth) + 1

    def get_risk_contributions(self, weights: pd.Series, date: pd.Timestamp) -> dict[str, float]:
        """
        Calculate risk contributions for portfolio weights.

        Args:
            weights: Portfolio weights
            date: Date for covariance estimation

        Returns:
            Dictionary mapping assets to risk contributions
        """
        if not self.is_fitted or self._fitted_returns is None:
            return {}

        try:
            # Get covariance matrix for the date
            end_date = date
            start_date = date - pd.Timedelta(days=self.hrp_config.lookback_days)

            lookback_mask = (self._fitted_returns.index >= start_date) & (
                self._fitted_returns.index < end_date
            )
            lookback_returns = self._fitted_returns[lookback_mask]

            # Filter for weight assets
            common_assets = weights.index.intersection(lookback_returns.columns)
            if len(common_assets) == 0:
                return {}

            filtered_returns = lookback_returns[common_assets]
            covariance_matrix = filtered_returns.cov()
            filtered_weights = weights.reindex(common_assets, fill_value=0.0)

            # Build cluster tree for risk contribution calculation
            distance_matrix = self.clustering_engine.build_correlation_distance(filtered_returns)
            linkage_matrix = self.clustering_engine.hierarchical_clustering(distance_matrix)
            cluster_tree = self.clustering_engine.build_cluster_tree(
                common_assets.tolist(), linkage_matrix
            )

            # Calculate risk contributions
            risk_contributions = self.allocation_engine.calculate_risk_budgets(
                filtered_weights, covariance_matrix, cluster_tree
            )

            return risk_contributions

        except Exception:
            return {}
