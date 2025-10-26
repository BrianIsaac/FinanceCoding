"""
Confidence-weighted training methodologies for flexible academic framework.

This module provides adaptive training strategies based on academic confidence
scores, implementing robust methods when data quality or quantity is limited.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, MinCovDet
from sklearn.linear_model import HuberRegressor, Ridge, Lasso
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


class TrainingMethodology(Enum):
    """Available training methodologies based on confidence levels."""

    STANDARD = "standard_academic_methods"
    ROBUST = "robust_academic_estimators"
    REGULARISED = "regularised_academic_methods"
    BAYESIAN = "bayesian_academic_approach"
    EXPLORATORY = "exploratory_analysis_only"


@dataclass
class TrainingStrategy:
    """Training strategy with confidence-based adjustments."""

    methodology: TrainingMethodology
    confidence: float
    adjustments: dict[str, Any]
    caveats: list[str]

    @property
    def should_regularise(self) -> bool:
        """Check if regularisation should be applied."""
        return self.methodology in [
            TrainingMethodology.REGULARISED,
            TrainingMethodology.BAYESIAN,
        ]

    @property
    def should_use_robust_estimators(self) -> bool:
        """Check if robust estimators should be used."""
        return self.methodology in [
            TrainingMethodology.ROBUST,
            TrainingMethodology.REGULARISED,
        ]


class ConfidenceWeightedTrainer:
    """
    Implements confidence-weighted training strategies for portfolio models.

    This trainer adapts the training methodology based on the academic confidence
    score, using more conservative and robust methods when confidence is lower.
    """

    def __init__(
        self,
        high_confidence_threshold: float = 0.9,
        moderate_confidence_threshold: float = 0.7,
        low_confidence_threshold: float = 0.5,
    ):
        """
        Initialise confidence-weighted trainer.

        Args:
            high_confidence_threshold: Threshold for standard methods
            moderate_confidence_threshold: Threshold for robust methods
            low_confidence_threshold: Threshold for regularised methods
        """
        self.high_threshold = high_confidence_threshold
        self.moderate_threshold = moderate_confidence_threshold
        self.low_threshold = low_confidence_threshold

    def select_training_strategy(
        self,
        confidence_score: float,
        data_characteristics: Optional[dict[str, Any]] = None,
    ) -> TrainingStrategy:
        """
        Select appropriate training strategy based on confidence.

        Args:
            confidence_score: Academic confidence score from validation
            data_characteristics: Optional data characteristics

        Returns:
            TrainingStrategy with methodology and adjustments
        """
        data_characteristics = data_characteristics or {}

        if confidence_score >= self.high_threshold:
            return self._standard_strategy(confidence_score, data_characteristics)
        elif confidence_score >= self.moderate_threshold:
            return self._robust_strategy(confidence_score, data_characteristics)
        elif confidence_score >= self.low_threshold:
            return self._regularised_strategy(confidence_score, data_characteristics)
        else:
            return self._bayesian_strategy(confidence_score, data_characteristics)

    def _standard_strategy(
        self,
        confidence: float,
        data_characteristics: dict[str, Any],
    ) -> TrainingStrategy:
        """Create standard training strategy for high confidence."""
        return TrainingStrategy(
            methodology=TrainingMethodology.STANDARD,
            confidence=confidence,
            adjustments={
                "learning_rate": 1.0,  # Full learning rate
                "epochs_multiplier": 1.0,  # Standard epochs
                "regularisation": 0.0,  # No regularisation needed
                "dropout_rate": 0.1,  # Minimal dropout
                "use_sample_weights": False,
            },
            caveats=[],
        )

    def _robust_strategy(
        self,
        confidence: float,
        data_characteristics: dict[str, Any],
    ) -> TrainingStrategy:
        """Create robust training strategy for moderate confidence."""
        return TrainingStrategy(
            methodology=TrainingMethodology.ROBUST,
            confidence=confidence,
            adjustments={
                "learning_rate": 0.8,  # Slightly reduced learning rate
                "epochs_multiplier": 1.2,  # More epochs for convergence
                "regularisation": 0.01,  # Light regularisation
                "dropout_rate": 0.2,  # Moderate dropout
                "use_sample_weights": True,  # Weight recent samples more
                "outlier_threshold": 3.0,  # Remove 3-sigma outliers
            },
            caveats=[
                "Using robust estimators due to moderate confidence",
                "Outliers may be down-weighted or removed",
            ],
        )

    def _regularised_strategy(
        self,
        confidence: float,
        data_characteristics: dict[str, Any],
    ) -> TrainingStrategy:
        """Create regularised training strategy for low confidence."""
        return TrainingStrategy(
            methodology=TrainingMethodology.REGULARISED,
            confidence=confidence,
            adjustments={
                "learning_rate": 0.5,  # Conservative learning rate
                "epochs_multiplier": 1.5,  # Extended training
                "regularisation": 0.1,  # Strong regularisation
                "dropout_rate": 0.3,  # Higher dropout
                "use_sample_weights": True,
                "outlier_threshold": 2.5,  # More aggressive outlier removal
                "shrinkage_factor": 0.3,  # Shrink towards prior
            },
            caveats=[
                "Strong regularisation applied due to limited confidence",
                "Results may be conservative due to shrinkage",
                "Consider ensemble methods for stability",
            ],
        )

    def _bayesian_strategy(
        self,
        confidence: float,
        data_characteristics: dict[str, Any],
    ) -> TrainingStrategy:
        """Create Bayesian training strategy for very low confidence."""
        return TrainingStrategy(
            methodology=TrainingMethodology.BAYESIAN,
            confidence=confidence,
            adjustments={
                "learning_rate": 0.3,  # Very conservative
                "epochs_multiplier": 2.0,  # Extended training
                "regularisation": 0.2,  # Very strong regularisation
                "dropout_rate": 0.4,  # High dropout
                "use_sample_weights": True,
                "outlier_threshold": 2.0,
                "shrinkage_factor": 0.5,  # Strong shrinkage
                "use_prior": True,  # Use informative priors
                "prior_weight": 0.3,  # Weight of prior vs data
            },
            caveats=[
                "Bayesian approach with informative priors due to low confidence",
                "Results heavily influenced by prior assumptions",
                "High uncertainty in parameter estimates",
                "Consider deferring decisions if possible",
            ],
        )

    def create_robust_covariance_estimator(
        self,
        confidence_score: float,
    ) -> Any:
        """
        Create appropriate covariance estimator based on confidence.

        Args:
            confidence_score: Academic confidence score

        Returns:
            Sklearn covariance estimator
        """
        if confidence_score >= self.high_threshold:
            # Standard empirical covariance
            from sklearn.covariance import EmpiricalCovariance
            return EmpiricalCovariance()
        elif confidence_score >= self.moderate_threshold:
            # Ledoit-Wolf shrinkage estimator
            return LedoitWolf()
        else:
            # Minimum Covariance Determinant (robust to outliers)
            return MinCovDet(support_fraction=0.7)

    def create_regression_estimator(
        self,
        confidence_score: float,
        alpha: float = 1.0,
    ) -> Any:
        """
        Create appropriate regression estimator based on confidence.

        Args:
            confidence_score: Academic confidence score
            alpha: Regularisation strength

        Returns:
            Sklearn regression estimator
        """
        if confidence_score >= self.high_threshold:
            # Standard OLS (through Ridge with alpha=0)
            return Ridge(alpha=0.0)
        elif confidence_score >= self.moderate_threshold:
            # Huber regression (robust to outliers)
            return HuberRegressor(epsilon=1.35, alpha=alpha * 0.01)
        elif confidence_score >= self.low_threshold:
            # Ridge regression with regularisation
            return Ridge(alpha=alpha * 0.1)
        else:
            # Lasso for feature selection with sparse data
            return Lasso(alpha=alpha * 0.1)

    def apply_data_preprocessing(
        self,
        data: pd.DataFrame,
        strategy: TrainingStrategy,
    ) -> pd.DataFrame:
        """
        Apply confidence-based data preprocessing.

        Args:
            data: Input data
            strategy: Training strategy

        Returns:
            Preprocessed data
        """
        processed_data = data.copy()

        # Apply outlier handling for robust/regularised strategies
        if strategy.should_use_robust_estimators:
            outlier_threshold = strategy.adjustments.get("outlier_threshold", 3.0)

            # Use robust scaler for numeric columns
            numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                scaler = RobustScaler()
                scaled_data = scaler.fit_transform(processed_data[numeric_cols])

                # Identify and clip outliers
                outlier_mask = np.abs(scaled_data) > outlier_threshold
                scaled_data[outlier_mask] = np.sign(scaled_data[outlier_mask]) * outlier_threshold

                # Transform back
                processed_data[numeric_cols] = scaler.inverse_transform(scaled_data)

                outlier_count = outlier_mask.sum()
                if outlier_count > 0:
                    logger.info(f"Clipped {outlier_count} outliers using threshold {outlier_threshold}")

        return processed_data

    def calculate_sample_weights(
        self,
        data: pd.DataFrame,
        strategy: TrainingStrategy,
        decay_factor: float = 0.95,
    ) -> np.ndarray:
        """
        Calculate sample weights based on strategy.

        Args:
            data: Input data with time index
            strategy: Training strategy
            decay_factor: Exponential decay factor for time weighting

        Returns:
            Sample weights array
        """
        n_samples = len(data)

        if not strategy.adjustments.get("use_sample_weights", False):
            return np.ones(n_samples)

        # Exponential decay weights (more recent = higher weight)
        time_weights = decay_factor ** np.arange(n_samples - 1, -1, -1)

        # Normalise weights
        time_weights = time_weights / time_weights.sum() * n_samples

        # Additional adjustments for low confidence
        if strategy.confidence < self.low_threshold:
            # Further emphasise recent data
            time_weights = time_weights ** 1.5
            time_weights = time_weights / time_weights.sum() * n_samples

        return time_weights

    def adjust_hyperparameters(
        self,
        base_params: dict[str, Any],
        strategy: TrainingStrategy,
    ) -> dict[str, Any]:
        """
        Adjust model hyperparameters based on training strategy.

        Args:
            base_params: Base hyperparameters
            strategy: Training strategy

        Returns:
            Adjusted hyperparameters
        """
        adjusted_params = base_params.copy()

        # Apply strategy adjustments
        for key, adjustment in strategy.adjustments.items():
            if key == "learning_rate" and "learning_rate" in adjusted_params:
                adjusted_params["learning_rate"] *= adjustment
            elif key == "epochs_multiplier" and "epochs" in adjusted_params:
                adjusted_params["epochs"] = int(adjusted_params["epochs"] * adjustment)
            elif key == "regularisation":
                adjusted_params["weight_decay"] = adjustment
                adjusted_params["l2_lambda"] = adjustment
            elif key == "dropout_rate" and "dropout" in adjusted_params:
                adjusted_params["dropout"] = adjustment

        return adjusted_params

    def generate_training_report(
        self,
        strategy: TrainingStrategy,
        training_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Generate comprehensive training report with confidence context.

        Args:
            strategy: Training strategy used
            training_metrics: Metrics from training

        Returns:
            Training report dictionary
        """
        return {
            "methodology": strategy.methodology.value,
            "confidence_score": strategy.confidence,
            "confidence_level": self._get_confidence_level(strategy.confidence),
            "adjustments_applied": strategy.adjustments,
            "academic_caveats": strategy.caveats,
            "training_metrics": training_metrics,
            "recommendations": self._get_recommendations(strategy),
        }

    def _get_confidence_level(self, confidence: float) -> str:
        """Get human-readable confidence level."""
        if confidence >= self.high_threshold:
            return "HIGH"
        elif confidence >= self.moderate_threshold:
            return "MODERATE"
        elif confidence >= self.low_threshold:
            return "LIMITED"
        else:
            return "INSUFFICIENT"

    def _get_recommendations(self, strategy: TrainingStrategy) -> list[str]:
        """Get recommendations based on strategy."""
        recommendations = []

        if strategy.confidence < self.moderate_threshold:
            recommendations.append("Consider collecting more data if possible")
            recommendations.append("Use ensemble methods for improved stability")

        if strategy.confidence < self.low_threshold:
            recommendations.append("Results should be interpreted with caution")
            recommendations.append("Consider deferring critical decisions")
            recommendations.append("Validate results with alternative methods")

        if strategy.methodology == TrainingMethodology.BAYESIAN:
            recommendations.append("Sensitivity analysis on prior assumptions recommended")

        return recommendations


def create_confidence_weighted_trainer() -> ConfidenceWeightedTrainer:
    """Factory function to create default confidence-weighted trainer."""
    return ConfidenceWeightedTrainer()