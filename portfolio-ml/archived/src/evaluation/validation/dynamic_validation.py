"""
Dynamic validation configuration for sparse financial data.

This module provides adaptive validation thresholds based on
universe characteristics and data availability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DynamicValidationConfig:
    """Configuration with dynamic thresholds for sparse data validation."""

    # Base requirements
    base_sequence_length: int = 60  # LSTM sequence length
    base_prediction_horizon: int = 21  # Monthly prediction

    # Minimum absolute requirements
    absolute_minimum_samples: int = 81  # sequence + prediction
    absolute_maximum_samples: int = 252  # 1 year cap

    # Adjustment factors
    sparse_data_factor: float = 0.5  # Factor for sparse periods
    crisis_period_factor: float = 0.3  # Factor for crisis periods

    def calculate_dynamic_minimum(
        self,
        universe_size: int,
        membership_duration: int,
        date: Optional[pd.Timestamp] = None,
        data_availability_ratio: float = 1.0
    ) -> int:
        """
        Calculate asset-specific minimum training requirements.

        Args:
            universe_size: Number of assets in universe at this time
            membership_duration: Days asset has been in universe
            date: Current date for crisis detection
            data_availability_ratio: Ratio of non-NaN data points

        Returns:
            Dynamic minimum training samples required
        """
        # Base requirement: sequence + prediction horizon
        base_minimum = self.base_sequence_length + self.base_prediction_horizon

        # Adjustment based on universe size
        # Small universes need less data for meaningful patterns
        universe_factor = min(1.0, max(0.3, universe_size / 100))

        # Adjustment based on membership duration
        # Newer assets can work with less historical data
        membership_factor = min(1.0, max(0.5, membership_duration / 252))

        # Adjustment for data sparsity
        # Sparse data periods get lower requirements
        sparsity_factor = max(self.sparse_data_factor, data_availability_ratio)

        # Crisis period detection (2008, 2020 COVID, etc.)
        crisis_factor = 1.0
        if date:
            crisis_periods = [
                (pd.Timestamp("2008-01-01"), pd.Timestamp("2009-12-31")),  # Financial crisis
                (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31")),  # COVID-19
            ]
            for start, end in crisis_periods:
                if start <= date <= end:
                    crisis_factor = self.crisis_period_factor
                    logger.debug(f"Crisis period detected for {date}, applying factor {crisis_factor}")
                    break

        # Calculate dynamic minimum with all factors
        dynamic_min = int(
            base_minimum * universe_factor * membership_factor * sparsity_factor * crisis_factor
        )

        # Apply floor and ceiling
        dynamic_min = max(self.absolute_minimum_samples, dynamic_min)
        dynamic_min = min(self.absolute_maximum_samples, dynamic_min)

        logger.debug(
            f"Dynamic minimum calculated: {dynamic_min} "
            f"(universe: {universe_size}, membership: {membership_duration} days, "
            f"sparsity: {data_availability_ratio:.2%})"
        )

        return dynamic_min


class AdaptiveValidator:
    """Validator with adaptive thresholds for dynamic universes."""

    def __init__(self, config: Optional[DynamicValidationConfig] = None):
        """
        Initialise adaptive validator.

        Args:
            config: Dynamic validation configuration
        """
        self.config = config or DynamicValidationConfig()

    def validate_training_data(
        self,
        train_data: pd.DataFrame,
        universe_df: pd.DataFrame,
        date: pd.Timestamp,
        strict: bool = False
    ) -> tuple[bool, int, str]:
        """
        Validate training data with adaptive thresholds.

        Args:
            train_data: Training data DataFrame
            universe_df: Universe membership DataFrame
            date: Current date for validation
            strict: Use strict validation (original 252 samples)

        Returns:
            Tuple of (is_valid, actual_samples, validation_message)
        """
        actual_samples = len(train_data)

        if strict:
            # Original strict validation
            required_samples = 252
            is_valid = actual_samples >= required_samples
            message = f"Strict validation: {actual_samples}/{required_samples} samples"
        else:
            # Calculate universe characteristics
            universe_at_date = self._get_universe_at_date(universe_df, date)
            universe_size = len(universe_at_date)

            # Calculate average membership duration
            avg_membership = self._calculate_average_membership(universe_df, universe_at_date, date)

            # Calculate data availability ratio
            non_nan_ratio = 1.0 - (train_data.isna().sum().sum() / train_data.size)

            # Get dynamic threshold
            required_samples = self.config.calculate_dynamic_minimum(
                universe_size=universe_size,
                membership_duration=avg_membership,
                date=date,
                data_availability_ratio=non_nan_ratio
            )

            is_valid = actual_samples >= required_samples
            message = (
                f"Adaptive validation: {actual_samples}/{required_samples} samples "
                f"(universe: {universe_size} assets, availability: {non_nan_ratio:.1%})"
            )

        if not is_valid:
            logger.warning(message)
        else:
            logger.debug(message)

        return is_valid, actual_samples, message

    def _get_universe_at_date(
        self,
        universe_df: pd.DataFrame,
        date: pd.Timestamp
    ) -> list[str]:
        """Get list of assets in universe at specific date."""
        mask = (universe_df['start'] <= date) & (universe_df['end'] >= date)
        return universe_df[mask]['ticker'].unique().tolist()

    def _calculate_average_membership(
        self,
        universe_df: pd.DataFrame,
        tickers: list[str],
        date: pd.Timestamp
    ) -> int:
        """Calculate average membership duration for assets."""
        if not tickers:
            return 0

        durations = []
        for ticker in tickers:
            ticker_data = universe_df[universe_df['ticker'] == ticker]
            for _, row in ticker_data.iterrows():
                if row['start'] <= date <= row['end']:
                    duration = (date - row['start']).days
                    durations.append(duration)
                    break

        return int(sum(durations) / len(durations)) if durations else 0