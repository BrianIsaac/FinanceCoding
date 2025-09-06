"""
Global test configuration and fixtures for the GNN portfolio optimization project.

This module provides common fixtures and configuration used across all test modules.
"""

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture(scope="session")
def gpu_available() -> bool:
    """
    Check if GPU is available for testing.

    Returns:
        True if CUDA GPU is available, False otherwise
    """
    return torch.cuda.is_available()


@pytest.fixture
def temp_data_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory for test data.

    Yields:
        Path to temporary directory
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_returns_data() -> pd.DataFrame:
    """
    Generate sample market returns data for testing.

    Returns:
        DataFrame with daily returns for multiple assets
    """
    np.random.seed(42)
    
    # Generate 252 trading days (1 year) of data
    dates = pd.date_range(
        start=datetime(2023, 1, 1), 
        periods=252, 
        freq='B'  # Business days
    )
    
    # Generate returns for 50 assets
    n_assets = 50
    assets = [f"ASSET_{i:03d}" for i in range(n_assets)]
    
    # Create correlated returns with realistic volatility
    base_vol = 0.15  # 15% annual volatility
    daily_vol = base_vol / np.sqrt(252)
    
    # Generate some correlation structure
    correlation_matrix = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i+1, min(i+10, n_assets)):  # Local correlation
            corr = 0.3 * np.exp(-0.1 * (j - i))
            correlation_matrix[i, j] = corr
            correlation_matrix[j, i] = corr
    
    # Generate returns using multivariate normal
    returns = np.random.multivariate_normal(
        mean=np.zeros(n_assets),
        cov=daily_vol**2 * correlation_matrix,
        size=len(dates)
    )
    
    return pd.DataFrame(
        data=returns,
        index=dates,
        columns=assets
    )


@pytest.fixture
def sample_universe() -> list[str]:
    """
    Generate sample asset universe for portfolio testing.

    Returns:
        List of asset tickers
    """
    return [f"ASSET_{i:03d}" for i in range(50)]


@pytest.fixture
def sample_config():
    """
    Generate sample configuration for testing.

    Returns:
        Dictionary with test configuration parameters
    """
    return {
        "data": {
            "lookback_days": 252,
            "min_observations": 100,
            "universe_size": 50
        },
        "model": {
            "hidden_dim": 64,
            "num_heads": 4,
            "dropout": 0.1,
            "learning_rate": 0.001
        },
        "training": {
            "batch_size": 16,
            "max_epochs": 10,
            "patience": 5
        },
        "gpu": {
            "max_memory_gb": 11.0,
            "enable_mixed_precision": True
        }
    }


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Skip GPU tests if CUDA is not available
    if not torch.cuda.is_available():
        config.addinivalue_line(
            "markers", "gpu: mark test as requiring GPU (skipped if no GPU)"
        )


def pytest_runtest_setup(item):
    """Setup logic run before each test."""
    # Skip GPU tests if no GPU available
    if "gpu" in item.keywords and not torch.cuda.is_available():
        pytest.skip("GPU not available")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names/paths."""
    for item in items:
        # Add unit marker for tests in unit/ directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker for tests in integration/ directory
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker for tests with "slow" in name
        if "slow" in item.name:
            item.add_marker(pytest.mark.slow)