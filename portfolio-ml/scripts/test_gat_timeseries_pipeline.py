"""Test GAT time-series data pipeline."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.models.gat.model import GATPortfolioModel, GATModelConfig
from src.models.base.portfolio_model import PortfolioConstraints


def test_timeseries_feature_preparation():
    """Test _prepare_timeseries_features method."""
    print("=" * 60)
    print("GAT TIME-SERIES FEATURE PREPARATION TEST")
    print("=" * 60)

    # Create model with default config
    constraints = PortfolioConstraints()
    config = GATModelConfig()
    model = GATPortfolioModel(constraints, config)

    # Create dummy returns data (300 days, 10 assets)
    dates = pd.date_range("2023-01-01", periods=300, freq="D")
    tickers = [f"ASSET{i:02d}" for i in range(10)]
    returns = pd.DataFrame(
        np.random.randn(300, 10) * 0.02,
        index=dates,
        columns=tickers,
    )

    print(f"\nReturns shape: {returns.shape}")
    print(f"Universe size: {len(tickers)}")

    # Test with different window lengths and features
    test_cases = [
        (60, ["volatility"], 1),
        (60, ["returns"], 1),
        (60, ["volatility", "returns"], 2),
        (60, ["volatility", "returns", "momentum"], 3),
        (120, ["volatility"], 1),
    ]

    for window_length, features, expected_features in test_cases:
        print(f"\nTesting window_length={window_length}, features={features}:")

        timeseries_features = model._prepare_timeseries_features(
            returns, tickers, window_length=window_length, features=features
        )

        print(f"  Output shape: {timeseries_features.shape}")
        print(f"  Expected: ({len(tickers)}, {window_length}, {expected_features})")

        # Verify shape
        assert timeseries_features.shape == (
            len(tickers),
            window_length,
            expected_features,
        ), f"Wrong shape: {timeseries_features.shape}"

        # Verify no NaN or Inf
        assert not np.isnan(timeseries_features).any(), "Contains NaN"
        assert not np.isinf(timeseries_features).any(), "Contains Inf"

        # Check normalisation (should be roughly z-scored)
        for i in range(len(tickers)):
            for j in range(expected_features):
                feat = timeseries_features[i, :, j]
                mean = feat.mean()
                std = feat.std()
                print(
                    f"    Asset {i}, Feature {j}: mean={mean:.3f}, std={std:.3f}"
                )
                # Should be roughly normalized
                assert abs(mean) < 0.1, f"Mean too large: {mean}"
                if std > 0:
                    assert 0.5 < std < 1.5, f"Std should be ~1.0, got {std}"

        print(f"  ✅ Test passed for {features}")

    print("\n" + "=" * 60)
    print("ALL TIME-SERIES FEATURE TESTS PASSED")
    print("=" * 60)


def test_node_features_routing():
    """Test _get_node_features routing method."""
    print("\n" + "=" * 60)
    print("NODE FEATURES ROUTING TEST")
    print("=" * 60)

    # Create returns data
    dates = pd.date_range("2023-01-01", periods=300, freq="D")
    tickers = [f"ASSET{i:02d}" for i in range(10)]
    returns = pd.DataFrame(
        np.random.randn(300, 10) * 0.02,
        index=dates,
        columns=tickers,
    )

    # Test 1: Static features (default)
    print("\nTest 1: Static features (default)")
    constraints = PortfolioConstraints()
    config = GATModelConfig(node_feature_type="static")
    model = GATPortfolioModel(constraints, config)

    features = model._get_node_features(returns, tickers)
    print(f"  Output shape: {features.shape}")
    print(f"  Expected: ({len(tickers)}, 10) - static features")

    assert features.ndim == 2, f"Static features should be 2D, got {features.ndim}D"
    assert features.shape[0] == len(tickers), f"Wrong number of assets"
    print("  ✅ Static features routing works")

    # Test 2: Time-series features
    print("\nTest 2: Time-series features")
    constraints = PortfolioConstraints()
    config = GATModelConfig(
        node_feature_type="timeseries",
        timeseries_length=60,
        timeseries_features=["volatility"],
    )
    model = GATPortfolioModel(constraints, config)

    features = model._get_node_features(returns, tickers)
    print(f"  Output shape: {features.shape}")
    print(f"  Expected: ({len(tickers)}, 60, 1) - time-series features")

    assert features.ndim == 3, f"Time-series features should be 3D, got {features.ndim}D"
    assert features.shape == (
        len(tickers),
        60,
        1,
    ), f"Wrong shape: {features.shape}"
    print("  ✅ Time-series features routing works")

    # Test 3: Multiple time-series features
    print("\nTest 3: Multiple time-series features")
    constraints = PortfolioConstraints()
    config = GATModelConfig(
        node_feature_type="timeseries",
        timeseries_length=60,
        timeseries_features=["volatility", "returns"],
    )
    model = GATPortfolioModel(constraints, config)

    features = model._get_node_features(returns, tickers)
    print(f"  Output shape: {features.shape}")
    print(f"  Expected: ({len(tickers)}, 60, 2) - multiple time-series features")

    assert features.shape == (
        len(tickers),
        60,
        2,
    ), f"Wrong shape: {features.shape}"
    print("  ✅ Multiple time-series features routing works")

    print("\n" + "=" * 60)
    print("ALL ROUTING TESTS PASSED")
    print("=" * 60)


def main():
    """Run all data pipeline tests."""
    test_timeseries_feature_preparation()
    test_node_features_routing()

    print("\n" + "=" * 60)
    print("GAT TIME-SERIES DATA PIPELINE: ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
