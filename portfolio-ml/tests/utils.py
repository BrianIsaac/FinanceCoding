"""Test utilities and helper functions."""

from __future__ import annotations

import torch
import pandas as pd
import numpy as np
from typing import Tuple


def assert_tensor_shape(
    tensor: torch.Tensor,
    expected_shape: Tuple[int, ...],
    name: str = "tensor"
) -> None:
    """Assert tensor has expected shape."""
    assert tensor.shape == expected_shape, \
        f"{name} shape {tensor.shape} != expected {expected_shape}"


def assert_no_nan_inf(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Assert tensor contains no NaN or Inf values."""
    assert torch.isfinite(tensor).all(), \
        f"{name} contains NaN or Inf values"


def assert_gradients_exist(model: torch.nn.Module) -> None:
    """Assert all model parameters have gradients."""
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(param.grad).all(), \
            f"Gradient for {name} contains NaN/Inf"


def create_mock_packed_sequence(
    batch_size: int = 4,
    max_length: int = 60,
    features: int = 10,
    min_length: int = 30
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create mock data for packed sequence testing.

    Returns:
        Tuple of (padded_sequences, lengths)
    """
    sequences = torch.randn(batch_size, max_length, features)
    lengths = torch.randint(min_length, max_length + 1, (batch_size,))

    return sequences, lengths


def calculate_padding_ratio(
    sequences: torch.Tensor,
    lengths: torch.Tensor
) -> float:
    """Calculate ratio of padded values in sequences."""
    batch_size, max_len, features = sequences.shape
    total_values = batch_size * max_len * features
    actual_values = int(lengths.sum()) * features
    return 1.0 - (actual_values / total_values)
