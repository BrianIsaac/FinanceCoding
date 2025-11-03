"""Validation utilities for ML models."""
import logging
from typing import Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


def validate_no_nan(arr: Union[torch.Tensor, np.ndarray], name: str = "array") -> None:
    """Validate tensor/array contains no NaN values.

    Args:
        arr: Array or tensor to validate
        name: Name of the array for error messages

    Raises:
        ValueError: If array contains NaN values
    """
    if torch.is_tensor(arr):
        if torch.isnan(arr).any():
            raise ValueError(f"{name} contains {torch.isnan(arr).sum()} NaN values")
    else:
        if np.isnan(arr).any():
            raise ValueError(f"{name} contains {np.isnan(arr).sum()} NaN values")


def validate_shape(arr: Union[torch.Tensor, np.ndarray], expected_shape: tuple, name: str = "tensor") -> None:
    """Validate tensor shape matches expected dimensions.

    Args:
        arr: Array or tensor to validate
        expected_shape: Expected shape (use None for dimensions to skip)
        name: Name of the tensor for error messages

    Raises:
        ValueError: If shape doesn't match expectations
    """
    if len(arr.shape) != len(expected_shape):
        raise ValueError(f"{name} has {len(arr.shape)} dims, expected {len(expected_shape)}")
    for i, (actual, expected) in enumerate(zip(arr.shape, expected_shape)):
        if expected is not None and actual != expected:
            raise ValueError(f"{name} dim {i}: expected {expected}, got {actual}")


def validate_universe_size(universe, min_size: int = 5, operation: str = "") -> None:
    """Validate universe has minimum required assets.

    Args:
        universe: Collection of assets to validate
        min_size: Minimum required number of assets
        operation: Description of operation for error message

    Raises:
        ValueError: If universe size is below minimum
    """
    if len(universe) < min_size:
        raise ValueError(
            f"Universe size {len(universe)} below minimum {min_size} for {operation}"
        )


def validate_finite(arr: Union[torch.Tensor, np.ndarray], name: str = "array", allow_nan: bool = False) -> None:
    """Validate array contains finite values.

    Args:
        arr: Array to validate
        name: Name of the array for error messages
        allow_nan: Whether to allow NaN values

    Raises:
        ValueError: If array contains NaN or infinite values
    """
    if not allow_nan and np.isnan(arr).any():
        raise ValueError(f"{name} contains {np.isnan(arr).sum()} NaN values")
    if np.isinf(arr).any():
        raise ValueError(f"{name} contains {np.isinf(arr).sum()} infinite values")
