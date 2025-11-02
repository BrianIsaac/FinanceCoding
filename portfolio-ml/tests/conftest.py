"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Tuple


@pytest.fixture
def sample_returns_data() -> pd.DataFrame:
    """
    Create sample returns data for testing.

    Returns:
        DataFrame with 252 days x 100 assets of synthetic returns
    """
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='B')
    tickers = [f'ASSET_{i:03d}' for i in range(100)]

    # Generate correlated returns
    returns = np.random.randn(252, 100) * 0.02

    return pd.DataFrame(returns, index=dates, columns=tickers)


@pytest.fixture
def sample_returns_with_gaps() -> pd.DataFrame:
    """
    Create sample returns with realistic missing data patterns.

    Returns:
        DataFrame with gaps simulating real financial data
    """
    # Generate sample data directly (not calling fixture)
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='B')
    tickers = [f'ASSET_{i:03d}' for i in range(100)]
    returns = pd.DataFrame(np.random.randn(252, 100) * 0.02, index=dates, columns=tickers)

    # Introduce random gaps (10-30% missing per asset)
    for col in returns.columns:
        mask_prob = np.random.uniform(0.1, 0.3)
        mask = np.random.random(len(returns)) < mask_prob
        returns.loc[mask, col] = np.nan

    return returns


@pytest.fixture
def sample_variable_length_sequences() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create sample sequences with variable lengths for ragged tensor testing.

    Returns:
        Tuple of (sequences, lengths) where:
        - sequences: [batch=32, max_seq_len=60, features=10]
        - lengths: [batch=32] with values in range [30, 60]
    """
    batch_size = 32
    max_seq_len = 60
    features = 10

    torch.manual_seed(42)
    sequences = torch.randn(batch_size, max_seq_len, features)
    lengths = torch.randint(30, max_seq_len + 1, (batch_size,))

    return sequences, lengths


@pytest.fixture
def sample_universe() -> list[str]:
    """Create sample universe of assets."""
    return [f'ASSET_{i:03d}' for i in range(50)]


@pytest.fixture
def temp_results_dir(tmp_path: Path) -> Path:
    """Create temporary directory for test results."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    return results_dir
