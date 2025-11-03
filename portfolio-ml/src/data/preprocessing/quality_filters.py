"""Data quality filters and preprocessing for backtest data.

Implements recommendations from data_exploration_findings.md:
1. Forward-fill short gaps (<30 days)
2. Exclude tickers with <80% coverage during membership
3. Handle extreme outliers (fat-tailed distributions)
4. Validate data quality before training

References:
- data_exploration_findings.md Section 4: Gap analysis
- data_exploration_findings.md Section 5: Distribution analysis
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataQualityFilter:
    """Filter and preprocess data based on quality metrics.

    Args:
        min_coverage: Minimum coverage rate during membership (default: 0.80)
        max_gap_days: Maximum gap to forward-fill (default: 30)
        price_bounds: Tuple of (min_price, max_price) for validity (default: (0.01, 10000))
        volume_bounds: Tuple of (min_volume, max_volume) for validity (default: (0, 1e9))
        winsorize_percentiles: Tuple of (lower, upper) percentiles for winsorization (default: (0.01, 0.99))
    """

    def __init__(
        self,
        min_coverage: float = 0.80,
        max_gap_days: int = 30,
        price_bounds: Tuple[float, float] = (0.01, 10000.0),
        volume_bounds: Tuple[float, float] = (0, 1e9),
        winsorize_percentiles: Tuple[float, float] = (0.01, 0.99),
    ):
        """Initialise data quality filter."""
        self.min_coverage = min_coverage
        self.max_gap_days = max_gap_days
        self.price_bounds = price_bounds
        self.volume_bounds = volume_bounds
        self.winsorize_percentiles = winsorize_percentiles

        logger.info(
            f"Initialised DataQualityFilter: min_coverage={min_coverage}, "
            f"max_gap_days={max_gap_days}, price_bounds={price_bounds}"
        )

    def filter_by_coverage(
        self,
        prices: pd.DataFrame,
        membership_mask: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """Filter tickers by coverage rate during membership periods.

        Args:
            prices: Price data [T, N]
            membership_mask: Boolean mask of membership [T, N]

        Returns:
            filtered_prices: Prices with low-coverage tickers removed
            kept_tickers: List of tickers that passed filter
            removed_tickers: List of tickers that failed filter with reasons
        """
        if membership_mask is None:
            # No membership data, use simple non-NaN coverage
            coverage = prices.notna().sum() / len(prices)
        else:
            # Membership-aware coverage
            expected_data = membership_mask.sum()
            actual_data = (prices.notna() & membership_mask).sum()
            coverage = actual_data / (expected_data + 1e-8)

        # Filter tickers
        kept_mask = coverage >= self.min_coverage
        kept_tickers = coverage[kept_mask].index.tolist()
        removed_tickers = coverage[~kept_mask].index.tolist()

        logger.info(f"Coverage filter: {len(kept_tickers)}/{len(coverage)} tickers passed (>={self.min_coverage:.1%})")
        logger.info(f"Removed {len(removed_tickers)} tickers with low coverage")

        if removed_tickers:
            logger.warning(f"Sample of removed tickers: {removed_tickers[:10]}")
            for ticker in removed_tickers[:5]:
                logger.debug(f"  {ticker}: {coverage[ticker]:.2%} coverage")

        filtered_prices = prices[kept_tickers].copy()

        return filtered_prices, kept_tickers, removed_tickers

    def forward_fill_gaps(
        self,
        data: pd.DataFrame,
        max_gap_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """Forward-fill gaps up to max_gap_days.

        Args:
            data: Data with potential gaps [T, N]
            max_gap_days: Maximum gap to fill (default: self.max_gap_days)

        Returns:
            filled_data: Data with short gaps filled
        """
        max_gap = max_gap_days or self.max_gap_days
        filled_data = data.copy()

        # Track filling statistics
        total_filled = 0
        total_gaps = 0

        for col in filled_data.columns:
            series = filled_data[col]

            # Find gaps
            is_missing = series.isna()
            gap_starts = is_missing & ~is_missing.shift(1, fill_value=False)
            gap_ends = is_missing & ~is_missing.shift(-1, fill_value=False)

            # Calculate gap lengths
            if gap_starts.any():
                gap_start_indices = np.where(gap_starts)[0]
                gap_end_indices = np.where(gap_ends)[0]

                for start_idx, end_idx in zip(gap_start_indices, gap_end_indices):
                    gap_length = end_idx - start_idx + 1
                    total_gaps += 1

                    if gap_length <= max_gap:
                        # Forward fill this gap
                        if start_idx > 0:
                            fill_value = series.iloc[start_idx - 1]
                            filled_data.loc[series.index[start_idx:end_idx+1], col] = fill_value
                            total_filled += 1

        logger.info(f"Gap filling: {total_filled}/{total_gaps} gaps filled (<={max_gap} days)")

        return filled_data

    def winsorize_outliers(
        self,
        data: pd.DataFrame,
        lower_percentile: Optional[float] = None,
        upper_percentile: Optional[float] = None,
    ) -> pd.DataFrame:
        """Winsorize extreme outliers to reduce fat-tail effects.

        Args:
            data: Data to winsorize [T, N]
            lower_percentile: Lower percentile bound (default: self.winsorize_percentiles[0])
            upper_percentile: Upper percentile bound (default: self.winsorize_percentiles[1])

        Returns:
            winsorized_data: Data with outliers clipped to percentile bounds
        """
        lower = lower_percentile or self.winsorize_percentiles[0]
        upper = upper_percentile or self.winsorize_percentiles[1]

        # Calculate percentiles across all data
        lower_bound = data.quantile(lower)
        upper_bound = data.quantile(upper)

        # Clip data
        winsorized_data = data.clip(lower=lower_bound, upper=upper_bound, axis=1)

        # Track winsorization statistics
        n_lower_clipped = (data < lower_bound).sum().sum()
        n_upper_clipped = (data > upper_bound).sum().sum()
        total_clipped = n_lower_clipped + n_upper_clipped
        total_values = data.notna().sum().sum()

        logger.info(
            f"Winsorization: {total_clipped}/{total_values} values clipped "
            f"({total_clipped/total_values:.2%}) at [{lower:.1%}, {upper:.1%}] percentiles"
        )

        return winsorized_data

    def validate_data_bounds(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Validate data is within acceptable bounds.

        Args:
            prices: Price data [T, N]
            volumes: Volume data [T, N] (optional)

        Returns:
            validated_prices: Prices with out-of-bounds values set to NaN
            validated_volumes: Volumes with out-of-bounds values set to NaN (if provided)
        """
        validated_prices = prices.copy()

        # Validate prices
        min_price, max_price = self.price_bounds
        invalid_prices = (prices < min_price) | (prices > max_price)
        n_invalid_prices = invalid_prices.sum().sum()

        if n_invalid_prices > 0:
            logger.warning(
                f"Found {n_invalid_prices} prices outside bounds [{min_price}, {max_price}], "
                f"setting to NaN"
            )
            validated_prices[invalid_prices] = np.nan

        validated_volumes = None
        if volumes is not None:
            validated_volumes = volumes.copy()

            # Validate volumes
            min_volume, max_volume = self.volume_bounds
            invalid_volumes = (volumes < min_volume) | (volumes > max_volume)
            n_invalid_volumes = invalid_volumes.sum().sum()

            if n_invalid_volumes > 0:
                logger.warning(
                    f"Found {n_invalid_volumes} volumes outside bounds [{min_volume}, {max_volume}], "
                    f"setting to NaN"
                )
                validated_volumes[invalid_volumes] = np.nan

        return validated_prices, validated_volumes

    def preprocess_pipeline(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
        membership_mask: Optional[pd.DataFrame] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Run complete preprocessing pipeline.

        Pipeline steps:
        1. Validate data bounds
        2. Filter by coverage
        3. Forward-fill short gaps
        4. Winsorize returns (not prices)

        Args:
            prices: Price data [T, N]
            returns: Return data [T, N]
            volumes: Volume data [T, N] (optional)
            membership_mask: Membership mask [T, N] (optional)

        Returns:
            Dictionary with preprocessed data:
                - prices: Cleaned prices
                - returns: Cleaned and winsorized returns
                - volumes: Cleaned volumes (if provided)
                - kept_tickers: List of tickers kept
                - removed_tickers: List of tickers removed
        """
        logger.info("Starting data quality preprocessing pipeline")

        # Step 1: Validate bounds
        logger.info("Step 1: Validating data bounds")
        validated_prices, validated_volumes = self.validate_data_bounds(prices, volumes)

        # Step 2: Filter by coverage
        logger.info("Step 2: Filtering by coverage")
        filtered_prices, kept_tickers, removed_tickers = self.filter_by_coverage(
            validated_prices, membership_mask
        )

        # Filter other data to match kept tickers
        filtered_returns = returns[kept_tickers].copy()
        if validated_volumes is not None:
            filtered_volumes = validated_volumes[kept_tickers].copy()
        else:
            filtered_volumes = None

        # Step 3: Forward-fill short gaps
        logger.info("Step 3: Forward-filling short gaps")
        filled_prices = self.forward_fill_gaps(filtered_prices)
        filled_returns = self.forward_fill_gaps(filtered_returns)
        if filtered_volumes is not None:
            filled_volumes = self.forward_fill_gaps(filtered_volumes)

        # Step 4: Winsorize returns (handle fat tails)
        logger.info("Step 4: Winsorizing returns to handle fat tails")
        winsorized_returns = self.winsorize_outliers(filled_returns)

        logger.info("Data quality preprocessing complete")
        logger.info(f"Final universe size: {len(kept_tickers)} tickers")

        return {
            "prices": filled_prices,
            "returns": winsorized_returns,
            "volumes": filled_volumes if filled_volumes is not None else None,
            "kept_tickers": kept_tickers,
            "removed_tickers": removed_tickers,
        }


def create_quality_filter(preset: str = "standard") -> DataQualityFilter:
    """Factory function to create quality filter with preset configurations.

    Args:
        preset: Configuration preset ("strict", "standard", "relaxed")

    Returns:
        Configured DataQualityFilter instance
    """
    presets = {
        "strict": {
            "min_coverage": 0.90,
            "max_gap_days": 20,
            "winsorize_percentiles": (0.005, 0.995),
        },
        "standard": {
            "min_coverage": 0.80,
            "max_gap_days": 30,
            "winsorize_percentiles": (0.01, 0.99),
        },
        "relaxed": {
            "min_coverage": 0.70,
            "max_gap_days": 45,
            "winsorize_percentiles": (0.02, 0.98),
        },
    }

    if preset not in presets:
        raise ValueError(f"preset must be one of {list(presets.keys())}, got {preset}")

    return DataQualityFilter(**presets[preset])
