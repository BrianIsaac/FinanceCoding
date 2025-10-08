"""
Universe alignment utilities for handling mismatches between dynamic universe and cleaned data.

This module provides functions to align the dynamic universe membership with
available cleaned returns data, preventing missing asset errors during portfolio construction.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class UniverseAlignmentManager:
    """Manages alignment between dynamic universe and available data."""

    def __init__(self):
        """Initialise universe alignment manager."""
        self.alignment_stats = {
            "total_alignments": 0,
            "assets_removed": 0,
            "assets_missing": 0,
        }
        self._misalignment_warning_shown = False  # Track if warning already shown

    def align_universe_with_data(
        self,
        universe: list[str],
        returns_data: pd.DataFrame,
        allow_partial: bool = True,
    ) -> tuple[list[str], dict[str, Any]]:
        """
        Align universe with available returns data.

        Args:
            universe: List of assets in the theoretical universe
            returns_data: DataFrame containing actual returns data
            allow_partial: Whether to allow partial universe (default True)

        Returns:
            Tuple of (aligned_universe, alignment_info)
        """
        self.alignment_stats["total_alignments"] += 1

        # Get available assets from returns data
        available_assets = set(returns_data.columns)
        requested_assets = set(universe)

        # Find intersection and differences
        aligned_assets = list(requested_assets & available_assets)
        missing_assets = list(requested_assets - available_assets)
        extra_assets = list(available_assets - requested_assets)

        self.alignment_stats["assets_missing"] += len(missing_assets)

        # Sort for consistency
        aligned_assets.sort()

        alignment_info = {
            "requested_count": len(universe),
            "available_count": len(available_assets),
            "aligned_count": len(aligned_assets),
            "missing_count": len(missing_assets),
            "missing_assets": missing_assets[:10] if missing_assets else [],  # First 10 for logging
            "alignment_ratio": len(aligned_assets) / len(universe) if universe else 0,
        }

        # Log significant misalignments (only once per instance to avoid spam)
        if alignment_info["alignment_ratio"] < 0.8:
            if not self._misalignment_warning_shown:
                logger.warning(
                    f"Significant universe misalignment detected (showing first occurrence only): "
                    f"{alignment_info['aligned_count']}/{alignment_info['requested_count']} "
                    f"assets available ({alignment_info['alignment_ratio']:.1%})"
                )
                if missing_assets:
                    logger.debug(f"Sample missing assets: {missing_assets[:5]}")
                self._misalignment_warning_shown = True
            else:
                # Log at debug level for subsequent occurrences
                logger.debug(
                    f"Universe misalignment: {alignment_info['aligned_count']}/{alignment_info['requested_count']} "
                    f"({alignment_info['alignment_ratio']:.1%})"
                )

        # Check if we have enough assets to proceed
        if not allow_partial and alignment_info["alignment_ratio"] < 0.95:
            logger.error(
                f"Universe alignment below threshold: {alignment_info['alignment_ratio']:.1%} < 95%"
            )
            raise ValueError("Insufficient universe alignment for strict mode")

        if len(aligned_assets) < 10:
            logger.error(f"Too few aligned assets: {len(aligned_assets)} < 10 minimum")
            raise ValueError("Insufficient assets after alignment")

        return aligned_assets, alignment_info

    def create_aligned_returns(
        self,
        returns_data: pd.DataFrame,
        universe: list[str],
        fill_missing: bool = False,
    ) -> pd.DataFrame:
        """
        Create returns DataFrame aligned with specified universe.

        Args:
            returns_data: Original returns data
            universe: Target universe
            fill_missing: Whether to fill missing assets with zeros

        Returns:
            Aligned returns DataFrame
        """
        # Get aligned universe
        aligned_universe, info = self.align_universe_with_data(
            universe, returns_data, allow_partial=True
        )

        # Create aligned returns
        aligned_returns = returns_data[aligned_universe].copy()

        # Optionally add missing assets with zeros (for compatibility)
        if fill_missing and info["missing_count"] > 0:
            missing_assets = info["missing_assets"]
            for asset in missing_assets:
                aligned_returns[asset] = 0.0
            logger.info(f"Filled {len(missing_assets)} missing assets with zeros")

        return aligned_returns

    def get_alignment_summary(self) -> dict[str, Any]:
        """Get summary statistics of alignment operations."""
        return {
            "total_operations": self.alignment_stats["total_alignments"],
            "total_missing_assets": self.alignment_stats["assets_missing"],
            "avg_missing_per_operation": (
                self.alignment_stats["assets_missing"] / self.alignment_stats["total_alignments"]
                if self.alignment_stats["total_alignments"] > 0
                else 0
            ),
        }


def align_portfolio_universe(
    model_universe: list[str],
    data_universe: list[str],
    min_overlap: float = 0.5,
) -> list[str]:
    """
    Quick utility function to align portfolio universe with data universe.

    Args:
        model_universe: Universe expected by the model
        data_universe: Universe available in the data
        min_overlap: Minimum required overlap ratio

    Returns:
        Aligned universe list

    Raises:
        ValueError: If overlap is below minimum threshold
    """
    model_set = set(model_universe)
    data_set = set(data_universe)
    aligned = list(model_set & data_set)

    overlap_ratio = len(aligned) / len(model_universe) if model_universe else 0

    if overlap_ratio < min_overlap:
        raise ValueError(
            f"Insufficient universe overlap: {overlap_ratio:.1%} < {min_overlap:.1%} minimum"
        )

    return sorted(aligned)