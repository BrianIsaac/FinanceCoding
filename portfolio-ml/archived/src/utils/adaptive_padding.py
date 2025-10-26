"""
Adaptive padding strategies for dynamic universe handling.

This module provides intelligent padding and truncation strategies
that minimize computational waste while maintaining model performance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


@dataclass
class AdaptivePaddingConfig:
    """Configuration for adaptive padding strategies."""

    # Padding thresholds
    max_padding_ratio: float = 0.1  # Maximum 10% padding allowed
    min_assets_for_training: int = 30  # Minimum assets needed for meaningful patterns

    # Dynamic sizing
    enable_dynamic_architecture: bool = True  # Allow runtime architecture adjustments
    size_granularity: int = 10  # Round sizes to nearest 10 for consistency

    # Asset selection
    use_correlation_substitution: bool = True  # Use correlated assets as substitutes
    correlation_threshold: float = 0.7  # Minimum correlation for substitution

    # Sequence length adaptation
    enable_adaptive_sequences: bool = True  # Vary sequence length based on data density
    min_sequence_length: int = 30  # Minimum viable sequence
    max_sequence_length: int = 90  # Maximum sequence to prevent overfitting

    # Memory optimization
    use_sparse_tensors: bool = False  # Use sparse tensors for padded regions
    cache_padding_masks: bool = True  # Cache masks to avoid recomputation


class AdaptivePaddingStrategy:
    """Intelligent padding and dimension management for dynamic universes."""

    def __init__(self, config: Optional[AdaptivePaddingConfig] = None):
        """
        Initialize adaptive padding strategy.

        Args:
            config: Adaptive padding configuration
        """
        self.config = config or AdaptivePaddingConfig()
        self._padding_mask_cache = {}
        self._correlation_cache = {}

    def calculate_optimal_size(
        self,
        current_size: int,
        target_size: int,
        data_density: float = 1.0
    ) -> Tuple[int, str]:
        """
        Calculate optimal dimension size based on current universe.

        Args:
            current_size: Current number of assets
            target_size: Target/expected size
            data_density: Ratio of non-missing data

        Returns:
            Tuple of (optimal_size, strategy_used)
        """
        if not self.config.enable_dynamic_architecture:
            return target_size, "fixed_architecture"

        # Check if current size is viable
        if current_size < self.config.min_assets_for_training:
            logger.warning(
                f"Universe too small ({current_size} < {self.config.min_assets_for_training}), "
                "may need to skip this period"
            )
            return current_size, "insufficient_assets"

        # Calculate padding ratio
        padding_ratio = (target_size - current_size) / target_size if target_size > current_size else 0

        # Determine strategy based on padding requirements
        if padding_ratio <= self.config.max_padding_ratio:
            # Acceptable padding level
            optimal = target_size
            strategy = "minimal_padding"
        elif padding_ratio <= 0.25:
            # Moderate padding - round to granularity
            optimal = self._round_to_granularity(current_size)
            strategy = "rounded_dynamic"
        else:
            # Excessive padding - use current size
            optimal = current_size
            strategy = "no_padding"

        # Adjust for data density
        if data_density < 0.5 and self.config.enable_adaptive_sequences:
            # Very sparse data - prefer smaller dimensions
            optimal = int(optimal * (0.5 + data_density * 0.5))
            optimal = max(self.config.min_assets_for_training, optimal)
            strategy += "_sparse_adjusted"

        logger.debug(
            f"Optimal size calculation: current={current_size}, target={target_size}, "
            f"optimal={optimal}, strategy={strategy}"
        )

        return optimal, strategy

    def apply_intelligent_padding(
        self,
        data: pd.DataFrame,
        target_size: int,
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Apply intelligent padding with correlation-based substitution.

        Args:
            data: Input data to pad
            target_size: Target dimension size
            correlation_matrix: Optional correlation matrix for substitution

        Returns:
            Padded DataFrame
        """
        current_size = data.shape[1]

        if current_size >= target_size:
            # No padding needed
            return data.iloc[:, :target_size]

        padding_needed = target_size - current_size

        if self.config.use_correlation_substitution and correlation_matrix is not None:
            # Use correlation-based substitution
            padded_data = self._correlation_based_padding(
                data, padding_needed, correlation_matrix
            )
        else:
            # Use smart zero padding
            padded_data = self._smart_zero_padding(data, padding_needed)

        return padded_data

    def _correlation_based_padding(
        self,
        data: pd.DataFrame,
        padding_needed: int,
        correlation_matrix: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Pad using correlated assets as proxies.

        Args:
            data: Original data
            padding_needed: Number of padding columns needed
            correlation_matrix: Asset correlation matrix

        Returns:
            Padded DataFrame with correlation-based substitutes
        """
        existing_assets = data.columns.tolist()
        padded_columns = []

        # Find highly correlated assets not in current universe
        for i in range(padding_needed):
            best_proxy = None
            best_correlation = 0

            for asset in existing_assets:
                if asset in correlation_matrix.index:
                    correlations = correlation_matrix.loc[asset]
                    # Find best correlated asset not already used
                    for candidate, corr in correlations.items():
                        if (candidate not in existing_assets and
                            candidate not in [col for col, _ in padded_columns] and
                            abs(corr) > self.config.correlation_threshold and
                            abs(corr) > best_correlation):
                            best_proxy = (candidate, asset, corr)
                            best_correlation = abs(corr)

            if best_proxy:
                proxy_name, source_asset, corr = best_proxy
                # Use correlated asset's data as proxy
                proxy_data = data[source_asset] * corr  # Scale by correlation
                padded_columns.append((f"PROXY_{proxy_name}", proxy_data))
                logger.debug(f"Using {source_asset} as proxy for {proxy_name} (corr={corr:.3f})")
            else:
                # Fall back to zero padding
                padded_columns.append((f"PAD_{i}", pd.Series(0, index=data.index)))

        # Combine original and padded data
        result = data.copy()
        for col_name, col_data in padded_columns:
            result[col_name] = col_data

        return result

    def _smart_zero_padding(self, data: pd.DataFrame, padding_needed: int) -> pd.DataFrame:
        """
        Apply smart zero padding with noise to prevent dead neurons.

        Args:
            data: Original data
            padding_needed: Number of padding columns needed

        Returns:
            Padded DataFrame with smart zeros
        """
        # Add small random noise to prevent dead neurons
        noise_scale = data.values.std() * 0.001  # 0.1% of data std

        padding = np.random.normal(0, noise_scale, (len(data), padding_needed))
        padding_df = pd.DataFrame(
            padding,
            index=data.index,
            columns=[f'PAD_{i}' for i in range(padding_needed)]
        )

        return pd.concat([data, padding_df], axis=1)

    def create_padding_mask(
        self,
        data_shape: Tuple[int, int],
        actual_assets: int
    ) -> torch.Tensor:
        """
        Create mask to ignore padded dimensions during training.

        Args:
            data_shape: Shape of padded data (batch, features)
            actual_assets: Number of real (non-padded) assets

        Returns:
            Boolean mask tensor
        """
        cache_key = (data_shape, actual_assets)

        if self.config.cache_padding_masks and cache_key in self._padding_mask_cache:
            return self._padding_mask_cache[cache_key]

        mask = torch.ones(data_shape, dtype=torch.bool)
        if actual_assets < data_shape[1]:
            mask[:, actual_assets:] = False

        if self.config.cache_padding_masks:
            self._padding_mask_cache[cache_key] = mask

        return mask

    def calculate_adaptive_sequence_length(
        self,
        data: pd.DataFrame,
        target_length: int
    ) -> int:
        """
        Calculate adaptive sequence length based on data characteristics.

        Args:
            data: Historical data
            target_length: Target sequence length

        Returns:
            Optimal sequence length
        """
        if not self.config.enable_adaptive_sequences:
            return target_length

        # Calculate data density
        non_nan_ratio = 1.0 - (data.isna().sum().sum() / data.size)

        # Calculate effective data points
        effective_rows = len(data) * non_nan_ratio

        if effective_rows < self.config.min_sequence_length:
            # Too little data - use minimum
            optimal_length = self.config.min_sequence_length
        elif effective_rows > self.config.max_sequence_length * 2:
            # Plenty of data - use target or max
            optimal_length = min(target_length, self.config.max_sequence_length)
        else:
            # Scale based on available data
            scale_factor = effective_rows / (self.config.max_sequence_length * 2)
            optimal_length = int(
                self.config.min_sequence_length +
                (self.config.max_sequence_length - self.config.min_sequence_length) * scale_factor
            )

        # Round to nearest 5 for consistency
        optimal_length = int(np.round(optimal_length / 5) * 5)

        logger.debug(
            f"Adaptive sequence length: target={target_length}, "
            f"effective_rows={effective_rows:.0f}, optimal={optimal_length}"
        )

        return optimal_length

    def _round_to_granularity(self, size: int) -> int:
        """Round size to nearest granularity level."""
        return int(np.round(size / self.config.size_granularity) * self.config.size_granularity)

    def get_memory_savings(
        self,
        original_size: int,
        optimized_size: int,
        batch_size: int,
        sequence_length: int,
        dtype_bytes: int = 4
    ) -> dict:
        """
        Calculate memory savings from optimization.

        Args:
            original_size: Original dimension size
            optimized_size: Optimized dimension size
            batch_size: Training batch size
            sequence_length: LSTM sequence length
            dtype_bytes: Bytes per element (4 for float32)

        Returns:
            Dictionary with memory statistics
        """
        original_memory = batch_size * sequence_length * original_size * dtype_bytes
        optimized_memory = batch_size * sequence_length * optimized_size * dtype_bytes

        savings_bytes = original_memory - optimized_memory
        savings_percent = (savings_bytes / original_memory) * 100 if original_memory > 0 else 0

        return {
            "original_memory_mb": original_memory / (1024 * 1024),
            "optimized_memory_mb": optimized_memory / (1024 * 1024),
            "savings_mb": savings_bytes / (1024 * 1024),
            "savings_percent": savings_percent,
            "dimension_reduction": original_size - optimized_size,
        }