"""
Flexible Academic Validator for adaptive validation with confidence scoring.

This module implements a flexible academic framework that maintains statistical
rigour while adapting to real-world data characteristics through confidence scoring
and uncertainty quantification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AcademicValidationResult:
    """Result of flexible academic validation with confidence scoring."""

    threshold: int
    confidence: float
    can_proceed: bool
    methodology_recommendations: list[str]
    academic_caveats: list[str]
    statistical_power: float
    data_quality_score: float
    universe_stability_score: float


@dataclass
class AcademicStandardsConfig:
    """Configuration for academic standards and thresholds."""

    # Minimum sample thresholds for different confidence levels
    high_confidence_samples: int = 120  # Reduced from 200 (6 months)
    moderate_confidence_samples: int = 60  # Reduced from 120 (3 months)
    low_confidence_samples: int = 30  # Reduced from 60 (1.5 months)
    minimum_samples: int = 10  # Reduced from 30 - train on ANY available data

    # Statistical power requirements
    target_power: float = 0.8
    minimum_power: float = 0.6

    # Confidence scoring weights
    data_quality_weight: float = 0.3
    sample_size_weight: float = 0.4
    universe_stability_weight: float = 0.3

    # Confidence thresholds
    high_confidence_threshold: float = 0.9
    moderate_confidence_threshold: float = 0.7
    low_confidence_threshold: float = 0.5

    @classmethod
    def load(cls, standards_level: str = "moderate") -> AcademicStandardsConfig:
        """Load configuration based on standards level."""
        configs = {
            "strict": cls(
                high_confidence_samples=180,  # Reduced from 252
                moderate_confidence_samples=120,  # Reduced from 180
                low_confidence_samples=60,  # Reduced from 90
                minimum_samples=20,  # Allow training with less data
                target_power=0.8,  # Reduced from 0.9
            ),
            "moderate": cls(),  # Use defaults
            "relaxed": cls(
                high_confidence_samples=60,  # Reduced from 150
                moderate_confidence_samples=30,  # Reduced from 90
                low_confidence_samples=15,  # Reduced from 45
                minimum_samples=5,  # Train on almost any data
                target_power=0.6,  # Reduced from 0.7
            ),
        }
        return configs.get(standards_level, cls())


class FlexibleAcademicValidator:
    """
    Flexible academic validator with confidence scoring.

    This validator adapts academic requirements based on data characteristics
    while maintaining statistical validity through confidence scoring.
    """

    def __init__(
        self,
        min_confidence: float = 0.6,
        academic_standards: str = "moderate",
    ):
        """
        Initialise flexible academic validator.

        Args:
            min_confidence: Minimum confidence score to proceed
            academic_standards: Level of academic standards ("strict", "moderate", "relaxed")
        """
        self.min_confidence = min_confidence
        self.standards = AcademicStandardsConfig.load(academic_standards)
        self.validation_history: list[AcademicValidationResult] = []

    def validate_with_confidence(
        self,
        data: pd.DataFrame,
        universe: list[str],
        context: Optional[dict[str, Any]] = None,
    ) -> AcademicValidationResult:
        """
        Validate data with confidence scoring.

        Args:
            data: Training data to validate
            universe: List of assets in the universe
            context: Additional context for validation

        Returns:
            AcademicValidationResult with confidence scores and recommendations
        """
        context = context or {}

        # Calculate adaptive threshold
        threshold = self.calculate_adaptive_threshold(data, universe, context)

        # Assess data quality components
        data_quality = self.assess_data_quality(data)
        universe_stability = self.assess_universe_stability(universe, data)
        sample_size_score = self.assess_sample_size(len(data), threshold)

        # Calculate overall academic confidence
        confidence = self.calculate_academic_confidence(
            sample_size_score, universe_stability, data_quality
        )

        # Calculate statistical power
        statistical_power = self.calculate_statistical_power(data, universe)

        # Generate methodology recommendations based on confidence
        methodology = self.recommend_methods(confidence, statistical_power)

        # Generate academic caveats
        caveats = self.generate_caveats(confidence, data, universe, statistical_power)

        # Determine if we can proceed
        can_proceed = confidence >= self.min_confidence and len(data) >= self.standards.minimum_samples

        result = AcademicValidationResult(
            threshold=threshold,
            confidence=confidence,
            can_proceed=can_proceed,
            methodology_recommendations=methodology,
            academic_caveats=caveats,
            statistical_power=statistical_power,
            data_quality_score=data_quality,
            universe_stability_score=universe_stability,
        )

        # Store validation result for tracking
        self.validation_history.append(result)

        # Log validation outcome
        self._log_validation_result(result, len(data))

        return result

    def calculate_adaptive_threshold(
        self,
        data: pd.DataFrame,
        universe: list[str],
        context: dict[str, Any],
    ) -> int:
        """
        Calculate adaptive threshold based on data characteristics.

        Args:
            data: Training data
            universe: Asset universe
            context: Additional context

        Returns:
            Adaptive threshold for minimum samples
        """
        # Base calculation factors
        universe_size = len(universe)
        data_periods = len(data) if not data.empty else 0

        # Market regime detection
        market_volatility = self._detect_market_volatility(data)
        is_crisis_period = context.get("is_crisis", False) or market_volatility > 0.3

        # Calculate base threshold
        if is_crisis_period:
            # Lower requirements during market stress
            base_threshold = self.standards.low_confidence_samples
        elif universe_size < 50:
            # Small universe requires fewer samples
            base_threshold = self.standards.low_confidence_samples
        elif universe_size > 200:
            # Large universe benefits from more samples
            base_threshold = self.standards.high_confidence_samples
        else:
            # Standard case
            base_threshold = self.standards.moderate_confidence_samples

        # Adjust for data characteristics
        if data_periods > 0:
            # Check data density
            non_null_ratio = 1 - (data.isna().sum().sum() / (data.shape[0] * data.shape[1]))
            if non_null_ratio < 0.8:
                # Sparse data requires adjustment
                base_threshold = int(base_threshold * 0.8)

        # Apply minimum constraint
        return max(base_threshold, self.standards.minimum_samples)

    def assess_data_quality(self, data: pd.DataFrame) -> float:
        """
        Assess quality of the data.

        Args:
            data: Data to assess

        Returns:
            Data quality score between 0 and 1
        """
        if data.empty:
            return 0.0

        scores = []

        # Completeness score
        completeness = 1 - (data.isna().sum().sum() / (data.shape[0] * data.shape[1]))
        scores.append(completeness)

        # Consistency score (check for extreme values)
        if data.select_dtypes(include=[np.number]).shape[1] > 0:
            numeric_data = data.select_dtypes(include=[np.number])
            z_scores = np.abs((numeric_data - numeric_data.mean()) / numeric_data.std())
            outlier_ratio = (z_scores > 5).sum().sum() / z_scores.size
            consistency = 1 - min(outlier_ratio * 10, 1.0)  # Scale outliers impact
            scores.append(consistency)

        # Temporal consistency (for time series)
        if isinstance(data.index, pd.DatetimeIndex):
            expected_freq = pd.infer_freq(data.index[:10])
            if expected_freq:
                actual_gaps = (data.index[1:] - data.index[:-1]).value_counts()
                temporal_consistency = 1.0 if len(actual_gaps) == 1 else 0.8
                scores.append(temporal_consistency)

        return float(np.mean(scores)) if scores else 0.5

    def assess_universe_stability(self, universe: list[str], data: pd.DataFrame) -> float:
        """
        Assess stability of the universe.

        Args:
            universe: Asset universe
            data: Training data

        Returns:
            Universe stability score between 0 and 1
        """
        if not universe or data.empty:
            return 0.5

        # Check overlap between universe and data columns
        if hasattr(data, 'columns'):
            data_assets = set(data.columns)
            universe_assets = set(universe)
            overlap_ratio = len(data_assets & universe_assets) / len(universe_assets) if universe_assets else 0
        else:
            overlap_ratio = 1.0

        # Assess universe size stability
        universe_size = len(universe)
        if universe_size < 30:
            size_score = 0.5  # Too small
        elif universe_size > 500:
            size_score = 0.8  # Large but manageable
        else:
            size_score = 1.0  # Optimal size

        return float(np.mean([overlap_ratio, size_score]))

    def assess_sample_size(self, sample_size: int, threshold: int) -> float:
        """
        Assess sample size adequacy.

        Args:
            sample_size: Number of samples
            threshold: Adaptive threshold

        Returns:
            Sample size score between 0 and 1
        """
        if sample_size == 0:
            return 0.0

        # Calculate score based on how much we exceed threshold
        if sample_size >= self.standards.high_confidence_samples:
            return 1.0
        elif sample_size >= threshold:
            # Linear interpolation between threshold and high confidence
            range_size = self.standards.high_confidence_samples - threshold
            excess = sample_size - threshold
            return 0.7 + 0.3 * (excess / range_size) if range_size > 0 else 0.7
        else:
            # Below threshold, score proportionally
            return 0.7 * (sample_size / threshold) if threshold > 0 else 0.0

    def calculate_academic_confidence(
        self,
        sample_size_score: float,
        universe_stability: float,
        data_quality: float,
    ) -> float:
        """
        Calculate overall academic confidence score.

        Args:
            sample_size_score: Sample size assessment score
            universe_stability: Universe stability score
            data_quality: Data quality score

        Returns:
            Academic confidence score between 0 and 1
        """
        # Weighted average based on configuration
        confidence = (
            self.standards.sample_size_weight * sample_size_score
            + self.standards.universe_stability_weight * universe_stability
            + self.standards.data_quality_weight * data_quality
        )

        # Apply non-linear transformation for more conservative scoring
        # This ensures high confidence requires excellence in all areas
        confidence = confidence ** 1.2

        return float(np.clip(confidence, 0, 1))

    def calculate_statistical_power(self, data: pd.DataFrame, universe: list[str]) -> float:
        """
        Calculate statistical power of the analysis.

        Args:
            data: Training data
            universe: Asset universe

        Returns:
            Estimated statistical power between 0 and 1
        """
        if data.empty:
            return 0.0

        n_samples = len(data)
        n_features = len(universe)

        # Simplified power calculation based on sample size and features
        # Using approximation for multivariate analysis
        if n_features > 0:
            samples_per_feature = n_samples / n_features
            if samples_per_feature >= 10:
                power = 0.9
            elif samples_per_feature >= 5:
                power = 0.7 + 0.2 * ((samples_per_feature - 5) / 5)
            else:
                power = 0.7 * (samples_per_feature / 5)
        else:
            power = 0.5

        return float(np.clip(power, 0, 1))

    def recommend_methods(self, confidence: float, statistical_power: float) -> list[str]:
        """
        Recommend methodologies based on confidence and power.

        Args:
            confidence: Academic confidence score
            statistical_power: Statistical power estimate

        Returns:
            List of recommended methodologies
        """
        recommendations = []

        if confidence >= self.standards.high_confidence_threshold:
            recommendations.extend([
                "standard_academic_methods",
                "maximum_likelihood_estimation",
                "classical_hypothesis_testing",
            ])
        elif confidence >= self.standards.moderate_confidence_threshold:
            recommendations.extend([
                "robust_academic_estimators",
                "huber_regression",
                "bootstrap_confidence_intervals",
                "regularisation_techniques",
            ])
        elif confidence >= self.standards.low_confidence_threshold:
            recommendations.extend([
                "bayesian_academic_approach",
                "informative_priors",
                "shrinkage_estimators",
                "ensemble_methods",
            ])
        else:
            recommendations.extend([
                "exploratory_data_analysis",
                "descriptive_statistics_only",
                "qualitative_assessment",
            ])

        # Add power-specific recommendations
        if statistical_power < self.standards.minimum_power:
            recommendations.append("increase_sample_size_recommended")
            recommendations.append("dimension_reduction_techniques")

        return recommendations

    def generate_caveats(
        self,
        confidence: float,
        data: pd.DataFrame,
        universe: list[str],
        statistical_power: float,
    ) -> list[str]:
        """
        Generate academic caveats based on validation results.

        Args:
            confidence: Academic confidence score
            data: Training data
            universe: Asset universe
            statistical_power: Statistical power estimate

        Returns:
            List of academic caveats
        """
        caveats = []

        # Confidence-based caveats
        if confidence < self.standards.high_confidence_threshold:
            caveats.append(f"Academic confidence below high threshold: {confidence:.2f}")

        if confidence < self.standards.moderate_confidence_threshold:
            caveats.append("Results should be interpreted with caution due to limited confidence")

        # Power-based caveats
        if statistical_power < self.standards.target_power:
            caveats.append(f"Statistical power below target: {statistical_power:.2f} < {self.standards.target_power}")

        # Data-based caveats
        if len(data) < self.standards.moderate_confidence_samples:
            caveats.append(f"Limited sample size: {len(data)} observations")

        # Universe-based caveats
        if len(universe) > 200:
            caveats.append(f"Large universe size ({len(universe)}) may impact estimation precision")
        elif len(universe) < 30:
            caveats.append(f"Small universe size ({len(universe)}) limits diversification analysis")

        # Data quality caveats
        if not data.empty:
            missing_ratio = data.isna().sum().sum() / (data.shape[0] * data.shape[1])
            if missing_ratio > 0.1:
                caveats.append(f"Significant missing data: {missing_ratio:.1%}")

        return caveats

    def _detect_market_volatility(self, data: pd.DataFrame) -> float:
        """
        Detect market volatility level from data.

        Args:
            data: Market data

        Returns:
            Volatility score between 0 and 1
        """
        if data.empty:
            return 0.5

        # Simple volatility detection based on returns dispersion
        if data.select_dtypes(include=[np.number]).shape[1] > 0:
            numeric_data = data.select_dtypes(include=[np.number])
            # Calculate rolling volatility proxy
            volatilities = numeric_data.std()
            # Normalise to 0-1 scale (assuming 50% annualised vol is extreme)
            avg_vol = volatilities.mean()
            normalised_vol = min(avg_vol / 0.5, 1.0) if not pd.isna(avg_vol) else 0.5
            return float(normalised_vol)

        return 0.5

    def _log_validation_result(self, result: AcademicValidationResult, sample_size: int) -> None:
        """Log validation result for tracking."""
        if result.can_proceed:
            logger.info(
                f"Validation PASSED - Confidence: {result.confidence:.2f}, "
                f"Samples: {sample_size}, Threshold: {result.threshold}"
            )
        else:
            logger.warning(
                f"Validation FAILED - Confidence: {result.confidence:.2f}, "
                f"Samples: {sample_size}, Threshold: {result.threshold}"
            )

        if result.academic_caveats:
            logger.info(f"Academic caveats: {', '.join(result.academic_caveats[:3])}")

    def get_validation_summary(self) -> dict[str, Any]:
        """Get summary of validation history."""
        if not self.validation_history:
            return {"validations": 0}

        confidences = [r.confidence for r in self.validation_history]
        pass_rate = sum(1 for r in self.validation_history if r.can_proceed) / len(self.validation_history)

        return {
            "validations": len(self.validation_history),
            "pass_rate": pass_rate,
            "avg_confidence": np.mean(confidences),
            "min_confidence": np.min(confidences),
            "max_confidence": np.max(confidences),
        }