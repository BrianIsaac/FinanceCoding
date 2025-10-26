"""
HRP portfolio model implementing the PortfolioModel interface.

This module provides the main HRP model class that integrates clustering,
allocation, and constraint enforcement for portfolio construction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional
import logging

import numpy as np
import pandas as pd

from pathlib import Path

logger = logging.getLogger(__name__)

from src.models.base.portfolio_model import PortfolioConstraints, PortfolioModel
from src.models.base.confidence_weighted_training import (
    ConfidenceWeightedTrainer,
    TrainingStrategy,
    create_confidence_weighted_trainer,
)
from src.models.hrp.allocation import AllocationConfig, HRPAllocation
from src.models.hrp.clustering import ClusteringConfig, HRPClustering
from src.models.hrp.universe_integration import HRPUniverseIntegration

# Import flexible academic validation
try:
    from src.evaluation.validation.flexible_academic_validator import (
        FlexibleAcademicValidator,
        AcademicValidationResult,
    )
    FLEXIBLE_VALIDATION_AVAILABLE = True
except ImportError:
    logger.info("Flexible validation not available for HRP, using standard validation")
    FLEXIBLE_VALIDATION_AVAILABLE = False


@dataclass
class HRPConfig:
    """Complete HRP model configuration."""

    lookback_days: int = 756  # 3 years of daily data
    clustering_config: ClusteringConfig = None
    allocation_config: AllocationConfig = None
    min_observations: int = 100  # Reduced for flexible academic framework
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

        # Confidence-weighted training support
        self.confidence_trainer = create_confidence_weighted_trainer()
        self.flexible_validator = (
            FlexibleAcademicValidator()
            if FLEXIBLE_VALIDATION_AVAILABLE
            else None
        )
        self.last_training_strategy: TrainingStrategy | None = None
        self.last_validation_result: AcademicValidationResult | None = None

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

        # Perform flexible academic validation if available
        confidence_score = 0.7  # Default moderate confidence
        if self.flexible_validator:
            validation_result = self.flexible_validator.validate_with_confidence(
                data=returns,
                universe=universe,
                context={"fit_period": fit_period, "model": "HRP"}
            )
            confidence_score = validation_result.confidence
            self.last_validation_result = validation_result

            if not validation_result.can_proceed:
                logger.warning(
                    f"HRP validation failed with confidence {confidence_score:.2f}. "
                    f"Using conservative defaults."
                )
                # Continue with conservative settings

        # Select training strategy based on confidence
        training_strategy = self.confidence_trainer.select_training_strategy(
            confidence_score=confidence_score,
            data_characteristics={
                "n_samples": len(returns),
                "n_features": len(universe),
            }
        )
        self.last_training_strategy = training_strategy

        # Apply confidence-based preprocessing
        returns = self.confidence_trainer.apply_data_preprocessing(
            returns, training_strategy
        )

        # Select appropriate covariance estimator based on confidence
        cov_estimator = self.confidence_trainer.create_robust_covariance_estimator(
            confidence_score
        )

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

        # Remove assets with zero variance to avoid NaN in correlation matrix
        returns_std = final_returns.std()
        non_zero_variance_assets = returns_std[returns_std > 1e-8].index.tolist()

        if len(non_zero_variance_assets) < 2:
            raise ValueError(f"Too few assets with non-zero variance: {len(non_zero_variance_assets)}")

        final_returns = final_returns[non_zero_variance_assets]

        # Calculate covariance matrix with shrinkage for large universes
        covariance_matrix = self._calculate_robust_covariance(final_returns)

        # Validate covariance matrix
        is_valid, error_msg = self.clustering_engine.validate_correlation_matrix(
            final_returns.corr()
        )
        if not is_valid:
            raise ValueError(f"Invalid correlation matrix: {error_msg}")

        # Store fitted state
        self._fitted_returns = final_returns
        self._fitted_covariance = covariance_matrix
        self._fitted_universe = non_zero_variance_assets
        self._fit_period = fit_period
        self.is_fitted = True

    def supports_rolling_retraining(self) -> bool:
        """HRP supports efficient rolling retraining since it's non-parametric."""
        return True

    def rolling_fit(
        self,
        returns: pd.DataFrame,
        universe: list[str],
        rebalance_date: pd.Timestamp,
        lookback_months: int = 36,
        min_observations: int = 100,  # Reduced for flexible academic framework
    ) -> None:
        """
        Perform rolling fit for HRP model using fresh data.

        HRP is particularly well-suited for rolling retraining since it's
        non-parametric and only requires correlation matrix calculation.

        Args:
            returns: Full historical returns DataFrame
            universe: Dynamic universe for this rebalancing period
            rebalance_date: Date for which we're rebalancing
            lookback_months: Number of months to look back for correlation calculation
            min_observations: Minimum number of observations required
        """
        # Calculate rolling window dates
        end_date = rebalance_date - pd.Timedelta(days=1)

        # Adaptive lookback based on data availability
        lookback_months_adaptive = self._get_adaptive_lookback(
            returns, end_date, universe, lookback_months, min_observations
        )

        start_date = end_date - pd.Timedelta(days=lookback_months_adaptive * 30)

        # Load fresh returns data for rolling window
        rolling_returns = self._load_fresh_returns_data(
            returns, start_date, end_date, universe
        )

        if len(rolling_returns) < min_observations:
            raise ValueError(
                f"Insufficient data for rolling fit: {len(rolling_returns)} < {min_observations}"
            )

        # Update fitted state with fresh data and robust covariance
        self._fitted_returns = rolling_returns
        self._fitted_covariance = self._calculate_robust_covariance(rolling_returns)
        self._fitted_universe = rolling_returns.columns.tolist()
        self._fit_period = (start_date, end_date)
        self.is_fitted = True

    def _load_fresh_returns_data(
        self,
        returns: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        universe: list[str],
    ) -> pd.DataFrame:
        """
        Load fresh returns data for rolling window.

        Args:
            returns: Full historical returns DataFrame
            start_date: Start of rolling window
            end_date: End of rolling window
            universe: Assets to include

        Returns:
            Filtered and cleaned returns DataFrame
        """
        # If returns is a path, load from disk
        if isinstance(returns, (str, Path)):
            returns_path = Path(returns)
            if returns_path.exists():
                returns = pd.read_parquet(returns_path)
            else:
                # Try default path
                returns_path = Path("data/final_new_pipeline/returns_daily_final.parquet")
                if returns_path.exists():
                    returns = pd.read_parquet(returns_path)
                else:
                    raise FileNotFoundError(f"Returns data not found at {returns_path}")

        # Filter by date range
        mask = (returns.index >= start_date) & (returns.index <= end_date)
        period_returns = returns[mask]

        # Filter for available universe assets
        available_assets = [asset for asset in universe if asset in period_returns.columns]

        if len(available_assets) == 0:
            raise ValueError("No assets from universe found in returns data")

        filtered_returns = period_returns[available_assets]

        # Clean data: forward fill then drop remaining NaN
        cleaned_returns = filtered_returns.ffill().dropna(axis=1, how='all')

        # Remove assets with zero variance
        returns_std = cleaned_returns.std()
        non_zero_variance_assets = returns_std[returns_std > 1e-8].index.tolist()

        if len(non_zero_variance_assets) < 2:
            raise ValueError(f"Too few assets with non-zero variance: {len(non_zero_variance_assets)}")

        return cleaned_returns[non_zero_variance_assets]

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

        # Handle dynamic universe membership
        # For HRP, we can work with any subset of assets that have sufficient data
        available_assets = [asset for asset in universe if asset in self._fitted_universe]

        # For dynamic membership, we should be more flexible
        if len(available_assets) == 0:
            # Try to find any overlap with the current universe
            fitted_assets = set(self._fitted_universe)
            universe_assets = set(universe)
            available_assets = list(fitted_assets.intersection(universe_assets))

            if len(available_assets) == 0:
                # As last resort, use equal weights for current universe
                logger.warning(f"No fitted assets in current universe {len(universe)} assets. Using equal weights.")
                equal_weight = 1.0 / len(universe)
                return pd.Series(equal_weight, index=universe)

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
            except Exception as e:
                logger.debug(f"Universe integration failed: {str(e)}, using available assets")
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
            distance_matrix, valid_assets = self.clustering_engine.build_correlation_distance(prediction_returns)

            # Check if any assets were dropped
            if len(valid_assets) < len(prediction_assets):
                dropped_assets = set(prediction_assets) - set(valid_assets)
                logger.info(f"Dropped {len(dropped_assets)} assets with zero variance during clustering")

            # Update prediction_assets to only include valid ones
            prediction_assets = valid_assets

            if len(prediction_assets) < 2:
                logger.warning(f"Only {len(prediction_assets)} valid assets after filtering, using equal weights")
                return pd.Series(1.0 / len(prediction_assets), index=prediction_assets)

        except ValueError as e:
            logger.warning(f"HRP clustering failed for {len(prediction_assets)} assets: {str(e)}")
            logger.info("Falling back to equal weights due to clustering failure")
            return pd.Series(1.0 / len(prediction_assets), index=prediction_assets)

        # Perform hierarchical clustering
        try:
            linkage_matrix = self.clustering_engine.hierarchical_clustering(distance_matrix)
            cluster_tree = self.clustering_engine.build_cluster_tree(
                prediction_assets, linkage_matrix
            )
        except Exception as e:
            logger.warning(f"HRP cluster tree building failed: {str(e)}")
            logger.warning(f"Asset count: {len(prediction_assets)}, distance matrix shape: {distance_matrix.shape}")
            logger.warning(f"Linkage matrix shape: {linkage_matrix.shape if 'linkage_matrix' in locals() else 'Not created'}")
            logger.info("Falling back to equal weights due to cluster tree failure")
            return pd.Series(1.0 / len(prediction_assets), index=prediction_assets)

        # Calculate covariance matrix for allocation
        prediction_covariance = prediction_returns.cov()

        # Perform recursive bisection allocation
        try:
            raw_weights = self.allocation_engine.recursive_bisection(
                prediction_covariance, cluster_tree
            )
        except Exception as e:
            logger.warning(f"HRP allocation failed: {str(e)}")
            logger.info(f"Using equal weights for {len(prediction_assets)} assets")
            raw_weights = pd.Series(1.0 / len(prediction_assets), index=prediction_assets)

        # Apply portfolio constraints using base class method
        # This handles all constraints including min_weight_threshold, top_k_positions,
        # max_position_weight, and proper normalization
        constrained_weights = self.validate_weights(raw_weights)

        # Fallback for impossible constraints (when all weights become zero or very small)
        if constrained_weights.sum() < 0.5:  # Weights sum is too small to be valid
            # Use equal weights as fallback when constraints are impossible to satisfy
            n_assets = len(prediction_assets)
            fallback_weights = pd.Series(1.0 / n_assets, index=prediction_assets)
            # Try to apply constraints again, but if they fail, return equal weights
            try:
                constrained_weights = self.validate_weights(fallback_weights)
                if constrained_weights.sum() < 0.5:
                    # Even equal weights fail constraints, return equal weights anyway
                    constrained_weights = fallback_weights
            except Exception as e:
                logger.debug(f"Constraint validation failed even for equal weights: {str(e)}")
                constrained_weights = fallback_weights

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

    def _get_adaptive_lookback(
        self,
        returns: pd.DataFrame,
        end_date: pd.Timestamp,
        universe: list[str],
        target_lookback_months: int,
        min_observations: int
    ) -> int:
        """
        Determine adaptive lookback period based on data availability.

        Args:
            returns: Historical returns data
            end_date: End date for lookback period
            universe: Current universe of assets
            target_lookback_months: Desired lookback period
            min_observations: Minimum required observations

        Returns:
            Adaptive lookback period in months
        """
        # Start with target lookback
        lookback = target_lookback_months

        # Check data availability with progressively shorter windows
        # More aggressive adaptation: try even shorter periods if needed
        # Adding 0.5 month (15 days) and 0.25 month (7 days) for extreme cases
        for months in [target_lookback_months, 24, 18, 12, 6, 3, 2, 1, 0.5, 0.25]:
            test_start = end_date - pd.Timedelta(days=int(months * 30))
            mask = (returns.index >= test_start) & (returns.index <= end_date)
            test_data = returns[mask]

            # Check if we have enough data - reduced from 50% to 30% of universe
            available_assets = [a for a in universe if a in test_data.columns]
            if len(available_assets) < len(universe) * 0.3:
                continue  # Need at least 30% of universe (was 50%)

            # Check observation count
            if len(test_data) >= min_observations:
                lookback = months
                if months < target_lookback_months:
                    logger.info(f"Adaptive lookback: reduced from {target_lookback_months} to {months} months due to data availability")
                break
        else:
            # If even 3 months doesn't work, use whatever we can get
            lookback = 3
            logger.warning(f"Insufficient data even for 3 months, using minimum lookback")

        return lookback

    def _calculate_robust_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate robust covariance matrix with shrinkage for large universes.

        Uses Ledoit-Wolf shrinkage to handle high-dimensional covariance matrices
        that become singular with many assets relative to observations.

        Args:
            returns: Returns DataFrame

        Returns:
            Robust covariance matrix
        """
        n_obs, n_assets = returns.shape

        # If we have enough observations, use sample covariance
        if n_obs > n_assets * 2:
            return returns.cov()

        try:
            # Use Ledoit-Wolf shrinkage for better conditioning
            from sklearn.covariance import LedoitWolf

            # Remove any NaN values
            clean_returns = returns.dropna()
            if len(clean_returns) < 2:
                logger.warning("Insufficient clean data for covariance, using simple cov")
                return returns.cov()

            # Apply shrinkage
            lw = LedoitWolf(store_precision=False, assume_centered=False)
            cov_shrunk, shrinkage = lw.fit(clean_returns.values).covariance_, lw.shrinkage_

            logger.debug(f"Applied Ledoit-Wolf shrinkage: {shrinkage:.3f} for {n_assets} assets")

            # Convert back to DataFrame
            return pd.DataFrame(
                cov_shrunk,
                index=returns.columns,
                columns=returns.columns
            )

        except Exception as e:
            logger.warning(f"Shrinkage failed: {str(e)}, using simple covariance")
            # Fallback to simple covariance with regularization
            cov = returns.cov()

            # Add small regularization to diagonal for numerical stability
            if n_assets > 100:
                reg = 1e-6 * np.trace(cov.values) / n_assets
                np.fill_diagonal(cov.values, cov.values.diagonal() + reg)
                logger.debug(f"Added regularization {reg:.2e} to covariance diagonal")

            return cov

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
