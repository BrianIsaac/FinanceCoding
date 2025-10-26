"""
Universe Management Utilities for Training-Backtest Compatibility.

This module provides standardised universe management to ensure consistency
between model training and backtesting phases, handling dynamic asset universes,
dimension alignment, and model compatibility checks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class UniverseManager:
    """
    Centralised universe management for consistent handling across training and backtesting.

    Handles:
    - Dynamic universe membership based on index constituents
    - Model-universe compatibility validation
    - Dimension alignment for LSTM/GAT models
    - Universe intersection and expansion logic
    """

    def __init__(
        self,
        universe_data_path: Path | str | None = None,
        default_universe_size: int = 400,  # S&P MidCap 400
        min_assets: int = 10,
        max_assets: int = 600,
    ):
        """
        Initialise universe manager.

        Args:
            universe_data_path: Path to universe membership data
            default_universe_size: Default expected universe size
            min_assets: Minimum required assets for valid universe
            max_assets: Maximum allowed assets (memory constraint)
        """
        self.universe_data_path = Path(universe_data_path) if universe_data_path else None
        self.default_universe_size = default_universe_size
        self.min_assets = min_assets
        self.max_assets = max_assets
        self.universe_membership = None

        # Load universe membership data if available
        if self.universe_data_path and self.universe_data_path.exists():
            self._load_universe_membership()

    def _load_universe_membership(self) -> None:
        """Load universe membership data from CSV."""
        try:
            if self.universe_data_path.suffix == '.csv':
                self.universe_membership = pd.read_csv(self.universe_data_path)
                if 'date' in self.universe_membership.columns:
                    self.universe_membership['date'] = pd.to_datetime(self.universe_membership['date'])
                logger.info(f"Loaded universe membership data: {self.universe_membership.shape}")
            else:
                logger.warning(f"Unsupported universe data format: {self.universe_data_path.suffix}")
        except Exception as e:
            logger.error(f"Failed to load universe membership data: {e}")
            self.universe_membership = None

    def get_training_universe(
        self,
        returns_data: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        min_coverage: float = 0.8,
    ) -> list[str]:
        """
        Get appropriate universe for model training using only membership rules (no look-ahead bias).

        In real trading, you cannot know ahead of time which assets will have good data coverage,
        so we only use universe membership rules that would be known at training time.

        Args:
            returns_data: Historical returns DataFrame
            start_date: Training period start
            end_date: Training period end
            min_coverage: Minimum data coverage required (DEPRECATED - causes look-ahead bias)

        Returns:
            List of asset tickers suitable for training (based on membership only)

        Raises:
            ValueError: If insufficient valid assets for training

        Note:
            This method ensures consistency between training and backtesting by using
            only membership-based filtering that would be available in real-time.
        """
        # Apply dynamic membership filtering if available
        if self.universe_membership is not None:
            try:
                # Handle start/end date format (current data structure)
                if 'start' in self.universe_membership.columns and 'end' in self.universe_membership.columns:
                    # Filter assets that were in universe during any part of training period
                    membership_mask = (
                        (self.universe_membership['start'] <= end_date) &
                        (self.universe_membership['end'] >= start_date)
                    )
                    period_members = self.universe_membership[membership_mask]
                    universe_tickers = period_members['ticker'].unique().tolist()

                    # Only use assets that exist in the price data (basic availability check)
                    available_assets = [asset for asset in universe_tickers if asset in returns_data.columns]
                    logger.info(f"Universe membership filter: {len(available_assets)} available assets (from {len(universe_tickers)} universe members)")
                    return available_assets

                # Handle date/in_universe format (legacy/alternative format)
                elif 'date' in self.universe_membership.columns and 'in_universe' in self.universe_membership.columns:
                    membership_mask = (
                        (self.universe_membership['date'] >= start_date) &
                        (self.universe_membership['date'] <= end_date)
                    )
                    period_members = self.universe_membership[membership_mask]

                    # Get all assets that were ever in universe during period (dynamic membership)
                    universe_tickers = period_members[period_members['in_universe'] == 1]['ticker'].unique().tolist()

                    # Only use assets that exist in the price data (basic availability check)
                    available_assets = [asset for asset in universe_tickers if asset in returns_data.columns]
                    logger.info(f"Dynamic membership filter: {len(available_assets)} available assets")
                    return available_assets
                else:
                    logger.warning("Universe membership data format not recognised - using all available assets")

            except Exception as e:
                logger.warning(f"Failed to apply universe membership filtering: {e} - using all available assets")

        # Fallback to all available assets if no universe membership data
        logger.warning("No universe membership data available - using all assets in price data")
        available_assets = list(returns_data.columns)

        # Apply size constraints if needed
        if len(available_assets) > self.max_assets:
            # Get period data for liquidity calculation
            mask = (returns_data.index >= start_date) & (returns_data.index <= end_date)
            period_data = returns_data.loc[mask]

            # Keep most liquid assets (highest mean absolute returns)
            liquidity = period_data[available_assets].abs().mean().sort_values(ascending=False)
            available_assets = liquidity.head(self.max_assets).index.tolist()
            logger.info(f"Reduced universe to {self.max_assets} most liquid assets")

        # Validate minimum universe size
        if len(available_assets) < self.min_assets:
            raise ValueError(f"Insufficient valid assets for training: {len(available_assets)} < {self.min_assets}")

        # Warn if universe is very small for ML training
        if len(available_assets) < 100:
            logger.warning(f"Very small universe size: {len(available_assets)} assets - consider reviewing filtering criteria")

        # Add consistency validation
        self._validate_universe_consistency(available_assets, start_date, end_date)

        # Log final universe statistics
        logger.info(f"Final training universe: {len(available_assets)} assets (membership-based, no look-ahead bias)")
        return available_assets

    def get_backtest_universe(
        self,
        returns_data: pd.DataFrame,
        date: pd.Timestamp,
        lookback_days: int = 60,
    ) -> list[str]:
        """
        Get appropriate universe for backtesting at specific date.

        Args:
            returns_data: Historical returns DataFrame
            date: Backtesting date
            lookback_days: Days to look back for data validation

        Returns:
            List of asset tickers available for backtesting
        """
        # Get recent data for validation
        end_date = date
        start_date = date - pd.Timedelta(days=lookback_days)

        mask = (returns_data.index >= start_date) & (returns_data.index <= end_date)
        recent_data = returns_data.loc[mask]

        if len(recent_data) == 0:
            logger.error(f"No data available for backtesting at {date}")
            return []

        # Get assets with sufficient recent data
        coverage = recent_data.notna().mean()
        valid_assets = coverage[coverage >= 0.5].index.tolist()  # More lenient for backtest

        # Apply point-in-time membership if available
        if self.universe_membership is not None:
            try:
                # Get membership at specific date
                date_membership = self.universe_membership[
                    self.universe_membership['date'] == date
                ]

                if not date_membership.empty and 'ticker' in date_membership.columns:
                    current_members = date_membership[
                        date_membership.get('in_universe', 1) == 1
                    ]['ticker'].tolist()

                    # Intersect with valid data
                    valid_assets = [asset for asset in valid_assets if asset in current_members]
                    logger.debug(f"Applied point-in-time membership: {len(valid_assets)} assets at {date}")
            except Exception as e:
                logger.warning(f"Failed to apply point-in-time membership: {e}")

        return valid_assets

    def align_model_universe(
        self,
        model_universe: list[str] | None,
        target_universe: list[str],
        allow_subset: bool = True,
    ) -> tuple[list[str], dict[str, Any]]:
        """
        Align model's trained universe with target universe for inference.

        Args:
            model_universe: Universe model was trained on
            target_universe: Universe requested for prediction
            allow_subset: Whether to allow predictions on subset of assets

        Returns:
            Tuple of (aligned_universe, alignment_info)
        """
        if model_universe is None:
            # Model has no fixed universe, can adapt to any
            return target_universe, {"type": "adaptive", "coverage": 1.0}

        # Calculate intersection
        common_assets = [asset for asset in target_universe if asset in model_universe]

        coverage = len(common_assets) / len(target_universe) if target_universe else 0

        alignment_info = {
            "type": "intersection",
            "model_assets": len(model_universe),
            "target_assets": len(target_universe),
            "common_assets": len(common_assets),
            "coverage": coverage,
            "missing_assets": [asset for asset in target_universe if asset not in model_universe],
        }

        if not common_assets:
            logger.error("No common assets between model and target universe")
            if not allow_subset:
                raise ValueError("No universe overlap between model and target")
            # Return empty universe, let caller handle fallback
            return [], alignment_info

        if coverage < 0.5:
            logger.warning(f"Low universe coverage: {coverage:.1%} ({len(common_assets)}/{len(target_universe)} assets)")

        return common_assets, alignment_info

    def expand_weights_to_universe(
        self,
        weights: pd.Series,
        target_universe: list[str],
        fill_value: float = 0.0,
    ) -> pd.Series:
        """
        Expand weight predictions to full target universe.

        Args:
            weights: Weights for subset of assets
            target_universe: Full universe to expand to
            fill_value: Value for assets without predictions

        Returns:
            Weights expanded to full universe
        """
        # Initialise with fill value
        expanded_weights = pd.Series(fill_value, index=target_universe)

        # Set weights for assets we have predictions for
        for asset in weights.index:
            if asset in target_universe:
                expanded_weights[asset] = weights[asset]

        # Renormalise if needed
        total_weight = expanded_weights.sum()
        if total_weight > 0 and abs(total_weight - 1.0) > 1e-6:
            expanded_weights = expanded_weights / total_weight
        elif total_weight == 0:
            # Fallback to equal weights
            expanded_weights = pd.Series(1.0 / len(target_universe), index=target_universe)

        return expanded_weights

    def validate_model_compatibility(
        self,
        model_info: dict[str, Any],
        target_universe: list[str],
    ) -> dict[str, Any]:
        """
        Validate model compatibility with target universe.

        Args:
            model_info: Model metadata dictionary
            target_universe: Target universe for inference

        Returns:
            Compatibility report dictionary
        """
        compatibility = {
            "compatible": True,
            "warnings": [],
            "errors": [],
        }

        # Check model type
        model_type = model_info.get("model_type", "unknown")

        if model_type == "LSTM":
            # Check dimension constraints
            input_size = model_info.get("lstm_config", {}).get("input_size", 0)
            if input_size > 0 and len(target_universe) > input_size:
                compatibility["warnings"].append(
                    f"LSTM expects {input_size} assets, got {len(target_universe)}. Will truncate."
                )

        elif model_type == "GAT":
            # Check universe overlap
            trained_universe = model_info.get("universe", [])
            if trained_universe:
                common_assets = [a for a in target_universe if a in trained_universe]
                if not common_assets:
                    compatibility["compatible"] = False
                    compatibility["errors"].append("No common assets with GAT training universe")
                elif len(common_assets) < len(target_universe) * 0.5:
                    compatibility["warnings"].append(
                        f"Only {len(common_assets)}/{len(target_universe)} assets available for GAT"
                    )

        elif model_type == "HRP":
            # HRP is generally compatible with any universe
            if len(target_universe) < 2:
                compatibility["compatible"] = False
                compatibility["errors"].append("HRP requires at least 2 assets")

        # Check constraint compatibility
        constraints = model_info.get("constraints", {})
        if constraints.get("top_k_positions", 0) > len(target_universe):
            compatibility["warnings"].append(
                f"top_k constraint ({constraints['top_k_positions']}) > universe size ({len(target_universe)})"
            )

        return compatibility

    def get_dimension_padding_strategy(
        self,
        model_type: str,
        model_dims: int,
        universe_size: int,
    ) -> dict[str, Any]:
        """
        Get appropriate padding strategy for dimension mismatches.

        Args:
            model_type: Type of model (LSTM, GAT, etc.)
            model_dims: Model's expected dimensions
            universe_size: Actual universe size

        Returns:
            Padding strategy configuration
        """
        strategy = {
            "type": "none",
            "padding_needed": 0,
            "truncation_needed": 0,
        }

        if model_type == "LSTM":
            if universe_size < model_dims:
                # Need padding
                strategy["type"] = "zero_padding"
                strategy["padding_needed"] = model_dims - universe_size
                strategy["pad_value"] = 0.0
            elif universe_size > model_dims:
                # Need truncation
                strategy["type"] = "liquidity_truncation"
                strategy["truncation_needed"] = universe_size - model_dims
                strategy["keep_top_k"] = model_dims

        elif model_type == "GAT":
            # GAT can handle variable sizes, but may need feature padding
            strategy["type"] = "adaptive"
            strategy["notes"] = "GAT adapts to graph size dynamically"

        return strategy

    def validate_universe_evolution(
        self,
        training_universe: list[str],
        inference_universe: list[str],
        model_type: str = "unknown",
        warn_threshold: float = 0.3,
        error_threshold: float = 0.7,
    ) -> dict[str, Any]:
        """
        Validate universe evolution between training and inference periods.

        Args:
            training_universe: Assets used during training
            inference_universe: Assets available during inference
            model_type: Type of model being validated
            warn_threshold: Threshold for raising warnings (30% change)
            error_threshold: Threshold for raising errors (70% change)

        Returns:
            Validation report with warnings and recommendations
        """
        if not training_universe or not inference_universe:
            return {
                "status": "error",
                "message": "Empty universe provided",
                "recommendations": ["Check data availability"],
            }

        # Calculate universe overlap metrics
        training_set = set(training_universe)
        inference_set = set(inference_universe)

        intersection = training_set.intersection(inference_set)
        training_only = training_set - inference_set
        inference_only = inference_set - training_set

        # Calculate change ratios
        universe_overlap_ratio = len(intersection) / len(training_set) if training_set else 0
        new_assets_ratio = len(inference_only) / len(inference_set) if inference_set else 0
        dropped_assets_ratio = len(training_only) / len(training_set) if training_set else 0

        # Determine validation status
        status = "pass"
        warnings = []
        errors = []
        recommendations = []

        # Check for significant universe changes
        if universe_overlap_ratio < (1.0 - error_threshold):
            status = "error"
            errors.append(f"Severe universe drift: only {universe_overlap_ratio:.1%} overlap")
            recommendations.append("Consider model retraining with recent data")
        elif universe_overlap_ratio < (1.0 - warn_threshold):
            status = "warning"
            warnings.append(f"Moderate universe drift: {universe_overlap_ratio:.1%} overlap")
            recommendations.append("Monitor model performance carefully")

        # Check for model-specific issues
        if model_type == "LSTM":
            if len(inference_universe) != len(training_universe):
                size_change = abs(len(inference_universe) - len(training_universe))
                change_ratio = size_change / len(training_universe)

                if change_ratio > error_threshold:
                    status = "error"
                    errors.append(f"LSTM universe size changed dramatically: {len(training_universe)} -> {len(inference_universe)}")
                    recommendations.append("LSTM models sensitive to dimension changes - retrain recommended")
                elif change_ratio > warn_threshold:
                    if status != "error":
                        status = "warning"
                    warnings.append(f"LSTM universe size changed: {len(training_universe)} -> {len(inference_universe)}")
                    recommendations.append("Consider padding/truncation strategy")

        elif model_type == "GAT":
            if len(intersection) < 5:  # GAT needs some common nodes for graph structure
                status = "error"
                errors.append(f"GAT has insufficient common assets: {len(intersection)}")
                recommendations.append("GAT requires stable core assets - retrain recommended")

        # Asset turnover analysis
        if new_assets_ratio > warn_threshold:
            if status != "error":
                status = "warning"
            warnings.append(f"High new asset ratio: {new_assets_ratio:.1%}")
            recommendations.append("New assets may lack sufficient training history")

        if dropped_assets_ratio > warn_threshold:
            if status != "error":
                status = "warning"
            warnings.append(f"High dropped asset ratio: {dropped_assets_ratio:.1%}")
            recommendations.append("Check for data quality issues in dropped assets")

        return {
            "status": status,
            "overlap_ratio": universe_overlap_ratio,
            "new_assets_ratio": new_assets_ratio,
            "dropped_assets_ratio": dropped_assets_ratio,
            "intersection_size": len(intersection),
            "training_only_count": len(training_only),
            "inference_only_count": len(inference_only),
            "warnings": warnings,
            "errors": errors,
            "recommendations": recommendations,
            "training_only_assets": list(training_only)[:10],  # Sample for debugging
            "inference_only_assets": list(inference_only)[:10],  # Sample for debugging
        }

    def _validate_universe_consistency(
        self,
        available_assets: list[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> None:
        """
        Validate universe consistency to ensure standardised selection logic.

        Args:
            available_assets: Selected assets for training
            start_date: Training period start
            end_date: Training period end

        Raises:
            Warning: If universe selection shows potential inconsistencies
        """
        # Check for reasonable asset distribution
        if len(available_assets) > self.default_universe_size * 1.5:
            logger.warning(
                f"Universe size ({len(available_assets)}) significantly exceeds expected "
                f"size ({self.default_universe_size}) - verify membership data"
            )

        # Validate asset naming consistency
        invalid_assets = [asset for asset in available_assets if not asset or len(asset) > 10]
        if invalid_assets:
            logger.warning(f"Found {len(invalid_assets)} assets with unusual tickers: {invalid_assets[:5]}")

        # Check for period consistency
        period_days = (end_date - start_date).days
        if period_days < 30:
            logger.warning(f"Very short training period: {period_days} days - may affect universe stability")
        elif period_days > 1095:  # 3 years
            logger.warning(f"Very long training period: {period_days} days - universe may drift significantly")

        logger.debug(f"Universe consistency validation passed for {len(available_assets)} assets")

    def track_universe_stability(
        self,
        returns_data: pd.DataFrame,
        window_months: int = 12,
        overlap_threshold: float = 0.8,
    ) -> dict[str, Any]:
        """
        Track universe stability over time to identify periods of high volatility.

        Args:
            returns_data: Historical returns data
            window_months: Window size in months for stability analysis
            overlap_threshold: Minimum overlap ratio for stable periods

        Returns:
            Universe stability report over time
        """
        if returns_data.empty:
            return {"status": "error", "message": "No data provided"}

        # Generate monthly windows
        date_range = pd.date_range(
            start=returns_data.index.min(),
            end=returns_data.index.max(),
            freq='MS'  # Month start
        )

        stability_report = {
            "periods": [],
            "stability_score": 0.0,
            "unstable_periods": [],
            "avg_universe_size": 0.0,
        }

        previous_universe = None
        universe_sizes = []
        overlap_ratios = []

        for i, period_start in enumerate(date_range[:-1]):
            period_end = date_range[i + 1] - pd.Timedelta(days=1)

            # Get universe for this period
            current_universe = self.get_training_universe(
                returns_data,
                period_start,
                period_end,
                min_coverage=0.6  # More lenient for stability tracking
            )

            universe_sizes.append(len(current_universe))

            period_info = {
                "start_date": period_start.strftime('%Y-%m-%d'),
                "end_date": period_end.strftime('%Y-%m-%d'),
                "universe_size": len(current_universe),
                "overlap_ratio": None,
                "stability": "stable",
            }

            # Calculate overlap with previous period
            if previous_universe is not None:
                current_set = set(current_universe)
                previous_set = set(previous_universe)

                if previous_set:
                    overlap = len(current_set.intersection(previous_set)) / len(previous_set)
                    period_info["overlap_ratio"] = overlap
                    overlap_ratios.append(overlap)

                    if overlap < overlap_threshold:
                        period_info["stability"] = "unstable"
                        stability_report["unstable_periods"].append({
                            "period": period_start.strftime('%Y-%m'),
                            "overlap_ratio": overlap,
                            "size_change": len(current_universe) - len(previous_universe),
                        })

            stability_report["periods"].append(period_info)
            previous_universe = current_universe

        # Calculate summary statistics
        if overlap_ratios:
            stability_report["stability_score"] = np.mean(overlap_ratios)

        if universe_sizes:
            stability_report["avg_universe_size"] = np.mean(universe_sizes)
            stability_report["universe_size_std"] = np.std(universe_sizes)

        # Add recommendations
        stability_report["recommendations"] = []

        if stability_report["stability_score"] < 0.7:
            stability_report["recommendations"].append(
                "Low universe stability detected - consider shorter retraining periods"
            )

        if len(stability_report["unstable_periods"]) > len(date_range) * 0.3:
            stability_report["recommendations"].append(
                "Frequent universe changes - investigate data quality or market regime changes"
            )

        return stability_report


def create_default_manager() -> UniverseManager:
    """
    Create universe manager with default configuration.

    Returns:
        Configured UniverseManager instance
    """
    # Try to find universe membership data (prefer daily format)
    potential_paths = [
        "data/processed/universe_membership_daily.csv",  # New daily format
        "data/processed/universe_membership_clean.csv", # Original format
        "data/processed/universe_membership.csv",
        "data/universe/membership.csv",
    ]

    universe_path = None
    for path in potential_paths:
        if Path(path).exists():
            universe_path = path
            break

    return UniverseManager(
        universe_data_path=universe_path,
        default_universe_size=400,  # S&P MidCap 400
        min_assets=50,  # Increased minimum for meaningful ML training
        max_assets=600,  # Conservative limit for 11GB GPU
    )