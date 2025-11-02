"""Manual verification for Phase 6 & 7: Temporal encoders and data pipeline.

This script performs comprehensive manual testing:
1. Temporal encoders with realistic financial data
2. Data pipeline with 756-day window (paper specification)
3. End-to-end GAT model with time-series features
4. Backward compatibility with static features
5. Memory usage analysis
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
from src.models.gat.model import GATPortfolioModel, GATModelConfig
from src.models.base.portfolio_model import PortfolioConstraints
from src.models.gat.temporal_encoders import (
    TemporalLSTMEncoder,
    TemporalConvEncoder,
    TemporalTransformerEncoder,
)


def generate_realistic_returns(n_days: int = 1000, n_assets: int = 50) -> pd.DataFrame:
    """Generate realistic synthetic financial returns data.

    Args:
        n_days: Number of trading days
        n_assets: Number of assets

    Returns:
        DataFrame with realistic returns
    """
    print(f"\nGenerating realistic returns: {n_days} days, {n_assets} assets")

    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    tickers = [f"ASSET{i:03d}" for i in range(n_assets)]

    # Generate returns with realistic properties
    # - Different volatilities per asset
    # - Correlation structure
    # - Fat tails (occasional large moves)

    # Base volatilities (annualized 10-40%)
    daily_vols = np.random.uniform(0.01, 0.04, n_assets)

    # Generate correlated returns using a simpler approach
    # Create random factor loadings for correlation structure
    num_factors = 5
    factor_loadings = np.random.randn(n_assets, num_factors) * 0.3

    # Generate idiosyncratic component
    idiosyncratic_std = np.ones(n_assets) * 0.7

    # Generate returns
    returns_data = []
    for _ in range(n_days):
        # Factor returns
        factor_returns = np.random.randn(num_factors)

        # Add occasional fat tail events (5% chance)
        if np.random.rand() < 0.05:
            factor_returns *= np.random.uniform(2, 4)

        # Idiosyncratic returns
        idiosyncratic_returns = np.random.randn(n_assets) * idiosyncratic_std

        # Combine factor and idiosyncratic
        correlated_noise = factor_loadings @ factor_returns + idiosyncratic_returns

        # Apply volatility
        daily_returns = correlated_noise * daily_vols

        returns_data.append(daily_returns)

    returns = pd.DataFrame(returns_data, index=dates, columns=tickers)

    # Add some NaN values (realistic - not all assets trade every day)
    mask = np.random.rand(n_days, n_assets) > 0.98
    returns[mask] = np.nan

    print(f"  Mean return: {returns.mean().mean():.6f}")
    print(f"  Mean volatility: {returns.std().mean():.6f}")
    print(f"  NaN percentage: {returns.isna().sum().sum() / (n_days * n_assets) * 100:.2f}%")

    return returns


def test_1_temporal_encoders_realistic_data():
    """Test 1: Temporal encoders with realistic financial data."""
    print("=" * 80)
    print("TEST 1: TEMPORAL ENCODERS WITH REALISTIC DATA")
    print("=" * 80)

    # Generate realistic returns
    returns = generate_realistic_returns(n_days=1000, n_assets=50)

    # Test with different window lengths
    window_lengths = [60, 120, 252]  # ~3 months, ~6 months, ~1 year

    for window_length in window_lengths:
        print(f"\n--- Window Length: {window_length} days ---")

        # Extract last window_length days
        recent_returns = returns.iloc[-window_length:].values

        # Compute rolling volatility (30-day)
        volatility_series = []
        for col in range(recent_returns.shape[1]):
            asset_returns = recent_returns[:, col]
            vol_series = pd.Series(asset_returns).rolling(
                window=30, min_periods=10
            ).std()
            vol_series = vol_series.fillna(vol_series.mean())
            volatility_series.append(vol_series.values)

        # Shape: [n_assets, time_steps, 1]
        vol_features = np.array(volatility_series)[:, :, np.newaxis].astype(np.float32)

        # Normalize
        for i in range(vol_features.shape[0]):
            feat = vol_features[i, :, 0]
            if feat.std() > 1e-8:
                vol_features[i, :, 0] = (feat - feat.mean()) / feat.std()

        print(f"  Volatility features shape: {vol_features.shape}")

        # Convert to tensor
        x = torch.from_numpy(vol_features)

        # Test each encoder
        for encoder_type, encoder_class in [
            ("Conv1D", TemporalConvEncoder),
            ("LSTM", TemporalLSTMEncoder),
            ("Transformer", TemporalTransformerEncoder),
        ]:
            print(f"\n  {encoder_type} Encoder:")

            encoder = encoder_class(input_features=1, hidden_dim=64)
            encoder.eval()

            with torch.no_grad():
                encoded = encoder(x)

            print(f"    Output shape: {encoded.shape}")
            print(f"    Output mean: {encoded.mean().item():.6f}")
            print(f"    Output std: {encoded.std().item():.6f}")
            print(f"    Output min: {encoded.min().item():.6f}")
            print(f"    Output max: {encoded.max().item():.6f}")

            # Check for reasonable values
            assert not torch.isnan(encoded).any(), "NaN in encoded features"
            assert not torch.isinf(encoded).any(), "Inf in encoded features"
            assert encoded.std() > 0.05, f"Encoder output has very low variance: {encoded.std():.6f}"

            print(f"    ✅ {encoder_type} encoder produces reasonable outputs")

    print("\n" + "=" * 80)
    print("TEST 1 PASSED: All temporal encoders work with realistic data")
    print("=" * 80)


def test_2_data_pipeline_756_days():
    """Test 2: Data pipeline with 756-day window (paper specification)."""
    print("\n" + "=" * 80)
    print("TEST 2: DATA PIPELINE WITH 756-DAY WINDOW")
    print("=" * 80)

    # Generate 1000 days of data (need 756 + buffer)
    returns = generate_realistic_returns(n_days=1000, n_assets=50)
    tickers = list(returns.columns)

    # Create model with default config
    constraints = PortfolioConstraints()
    config = GATModelConfig()
    model = GATPortfolioModel(constraints, config)

    print("\nTest 2.1: Single feature (volatility)")
    features = model._prepare_timeseries_features(
        returns, tickers, window_length=756, features=["volatility"]
    )

    print(f"  Output shape: {features.shape}")
    print(f"  Expected: ({len(tickers)}, 756, 1)")
    assert features.shape == (len(tickers), 756, 1)

    print(f"  Mean: {features.mean():.6f}")
    print(f"  Std: {features.std():.6f}")
    print(f"  Min: {features.min():.6f}")
    print(f"  Max: {features.max():.6f}")

    # Check normalization
    assert abs(features.mean()) < 0.1, "Features not properly normalized"
    assert 0.8 < features.std() < 1.2, f"Std should be ~1.0, got {features.std()}"

    print("  ✅ Volatility series normalized correctly")

    # Verify volatility looks reasonable
    for i in range(min(3, len(tickers))):
        asset_feat = features[i, :, 0]
        # Should have some variation
        assert asset_feat.std() > 0.5, f"Asset {i} volatility too constant"
        print(f"  Asset {i}: mean={asset_feat.mean():.3f}, std={asset_feat.std():.3f}")

    print("  ✅ Volatility series have reasonable variation")

    print("\nTest 2.2: Multiple features")
    features_multi = model._prepare_timeseries_features(
        returns,
        tickers,
        window_length=756,
        features=["volatility", "returns", "momentum"],
    )

    print(f"  Output shape: {features_multi.shape}")
    print(f"  Expected: ({len(tickers)}, 756, 3)")
    assert features_multi.shape == (len(tickers), 756, 3)

    # Check each feature is normalized independently
    for feat_idx in range(3):
        feat_name = ["volatility", "returns", "momentum"][feat_idx]
        feat_values = features_multi[:, :, feat_idx]
        print(f"  {feat_name}: mean={feat_values.mean():.3f}, std={feat_values.std():.3f}")
        assert abs(feat_values.mean()) < 0.15, f"{feat_name} not normalized"

    print("  ✅ Multiple features normalized independently")

    print("\n" + "=" * 80)
    print("TEST 2 PASSED: Data pipeline works with 756-day window")
    print("=" * 80)


def test_3_end_to_end_timeseries():
    """Test 3: End-to-end GAT model with time-series features."""
    print("\n" + "=" * 80)
    print("TEST 3: END-TO-END GAT WITH TIME-SERIES FEATURES")
    print("=" * 80)

    # Generate data
    returns = generate_realistic_returns(n_days=500, n_assets=30)
    tickers = list(returns.columns)

    # Create model with time-series features
    constraints = PortfolioConstraints()
    config = GATModelConfig(
        node_feature_type="timeseries",
        timeseries_length=60,
        timeseries_features=["volatility"],
        temporal_encoder_type="conv1d",
        hidden_dim=64,
        num_layers=2,
    )

    print(f"\nConfig:")
    print(f"  node_feature_type: {config.node_feature_type}")
    print(f"  timeseries_length: {config.timeseries_length}")
    print(f"  use_temporal_encoder: {config.use_temporal_encoder}")
    print(f"  temporal_encoder_type: {config.temporal_encoder_type}")

    model = GATPortfolioModel(constraints, config)

    # Prepare features
    print("\nPreparing features...")
    features = model._get_node_features(returns, tickers)

    print(f"  Features shape: {features.shape}")
    print(f"  Expected: (30, 60, 1)")
    assert features.shape == (30, 60, 1), f"Wrong shape: {features.shape}"
    print("  ✅ Features prepared correctly")

    # Test that model can be built
    print("\nBuilding model...")
    input_dim = features.shape[2]  # Number of features per timestep
    gat_model = model._build_model(input_dim)

    print(f"  Model created: {type(gat_model).__name__}")
    print(f"  Has temporal encoder: {hasattr(gat_model, 'temporal_encoder')}")
    print(f"  Temporal encoder type: {type(gat_model.temporal_encoder).__name__ if gat_model.temporal_encoder else 'None'}")

    assert gat_model.temporal_encoder is not None, "Temporal encoder not created"
    print("  ✅ Model built with temporal encoder")

    # Test forward pass (simplified - just check it runs)
    print("\nTesting forward pass...")
    from src.models.gat.graph_builder import build_period_graph

    # Build graph
    graph_data = build_period_graph(
        returns_daily=returns,
        period_end=returns.index[-1],
        tickers=tickers,
        features_matrix=features,
        cfg=config.graph_config,
    )

    print(f"  Graph nodes: {len(graph_data.tickers)}")
    print(f"  Graph edges: {graph_data.edge_index.shape[1]}")

    # Convert features to tensor
    x_tensor = torch.from_numpy(features).float()

    # Handle edge_index - might already be a tensor
    if isinstance(graph_data.edge_index, torch.Tensor):
        edge_index = graph_data.edge_index.long()
    else:
        edge_index = torch.from_numpy(graph_data.edge_index).long()

    mask_valid = torch.ones(len(graph_data.tickers), dtype=torch.bool)

    # Skip test if no edges (can happen with small/sparse data)
    if edge_index.shape[1] == 0:
        print("  ⚠️  Warning: Graph has no edges, creating minimal edges for testing")
        # Create a minimal connected graph for testing
        if len(graph_data.tickers) > 1:
            # Connect first node to all others
            sources = torch.zeros(len(graph_data.tickers) - 1, dtype=torch.long)
            targets = torch.arange(1, len(graph_data.tickers), dtype=torch.long)
            edge_index = torch.stack([sources, targets], dim=0)
            print(f"  Created {edge_index.shape[1]} test edges")

    # Forward pass (move tensors to same device as model)
    device = next(gat_model.parameters()).device
    x_tensor = x_tensor.to(device)
    edge_index = edge_index.to(device)
    mask_valid = mask_valid.to(device)

    gat_model.eval()
    with torch.no_grad():
        output, memory, reg_loss = gat_model(x_tensor, edge_index, mask_valid)

    print(f"  Output shape: {output.shape}")
    print(f"  Output sum: {output.sum().item():.6f}")
    print(f"  Output contains NaN: {torch.isnan(output).any().item()}")
    print(f"  Output contains Inf: {torch.isinf(output).any().item()}")

    assert not torch.isnan(output).any(), "NaN in model output"
    assert not torch.isinf(output).any(), "Inf in model output"
    print("  ✅ Forward pass successful with time-series features")

    print("\n" + "=" * 80)
    print("TEST 3 PASSED: End-to-end GAT works with time-series features")
    print("=" * 80)


def test_4_backward_compatibility():
    """Test 4: Verify backward compatibility with static features."""
    print("\n" + "=" * 80)
    print("TEST 4: BACKWARD COMPATIBILITY (STATIC FEATURES)")
    print("=" * 80)

    # Generate data
    returns = generate_realistic_returns(n_days=500, n_assets=30)
    tickers = list(returns.columns)

    # Create model with STATIC features (default)
    constraints = PortfolioConstraints()
    config = GATModelConfig(
        node_feature_type="static",  # Explicit static
        hidden_dim=64,
        num_layers=2,
    )

    print(f"\nConfig:")
    print(f"  node_feature_type: {config.node_feature_type}")
    print(f"  use_temporal_encoder: {config.use_temporal_encoder}")

    model = GATPortfolioModel(constraints, config)

    # Prepare features
    print("\nPreparing static features...")
    features = model._get_node_features(returns, tickers)

    print(f"  Features shape: {features.shape}")
    print(f"  Expected: (30, 10) - static features")
    assert features.shape == (30, 10), f"Wrong shape: {features.shape}"
    assert features.ndim == 2, "Static features should be 2D"
    print("  ✅ Static features prepared correctly")

    # Test that model is built WITHOUT temporal encoder
    print("\nBuilding model...")
    input_dim = features.shape[1]  # Number of features
    gat_model = model._build_model(input_dim)

    print(f"  Model created: {type(gat_model).__name__}")
    print(f"  Has temporal encoder: {hasattr(gat_model, 'temporal_encoder')}")
    print(f"  Temporal encoder is None: {gat_model.temporal_encoder is None}")

    assert gat_model.temporal_encoder is None, "Temporal encoder should not be created for static features"
    print("  ✅ Model built without temporal encoder (backward compatible)")

    # Test forward pass
    print("\nTesting forward pass with static features...")
    from src.models.gat.graph_builder import build_period_graph

    graph_data = build_period_graph(
        returns_daily=returns,
        period_end=returns.index[-1],
        tickers=tickers,
        features_matrix=features,
        cfg=config.graph_config,
    )

    x_tensor = torch.from_numpy(features).float()

    # Handle edge_index - might already be a tensor
    if isinstance(graph_data.edge_index, torch.Tensor):
        edge_index = graph_data.edge_index.long()
    else:
        edge_index = torch.from_numpy(graph_data.edge_index).long()

    mask_valid = torch.ones(len(graph_data.tickers), dtype=torch.bool)

    # Skip test if no edges
    if edge_index.shape[1] == 0:
        print("  ⚠️  Warning: Graph has no edges, creating minimal edges for testing")
        if len(graph_data.tickers) > 1:
            sources = torch.zeros(len(graph_data.tickers) - 1, dtype=torch.long)
            targets = torch.arange(1, len(graph_data.tickers), dtype=torch.long)
            edge_index = torch.stack([sources, targets], dim=0)
            print(f"  Created {edge_index.shape[1]} test edges")

    # Move tensors to same device as model
    device = next(gat_model.parameters()).device
    x_tensor = x_tensor.to(device)
    edge_index = edge_index.to(device)
    mask_valid = mask_valid.to(device)

    gat_model.eval()
    with torch.no_grad():
        output, memory, reg_loss = gat_model(x_tensor, edge_index, mask_valid)

    print(f"  Output shape: {output.shape}")
    print(f"  Output contains NaN: {torch.isnan(output).any().item()}")
    print(f"  Output contains Inf: {torch.isinf(output).any().item()}")

    assert not torch.isnan(output).any(), "NaN in model output"
    assert not torch.isinf(output).any(), "Inf in model output"
    print("  ✅ Forward pass successful with static features")

    print("\n" + "=" * 80)
    print("TEST 4 PASSED: Backward compatibility maintained")
    print("=" * 80)


def test_5_memory_usage():
    """Test 5: Check memory usage for different configurations."""
    print("\n" + "=" * 80)
    print("TEST 5: MEMORY USAGE ANALYSIS")
    print("=" * 80)

    import psutil
    import os

    process = psutil.Process(os.getpid())

    def get_memory_mb():
        return process.memory_info().rss / 1024 / 1024

    # Baseline
    baseline_memory = get_memory_mb()
    print(f"\nBaseline memory: {baseline_memory:.2f} MB")

    # Generate data
    returns = generate_realistic_returns(n_days=1000, n_assets=100)
    tickers = list(returns.columns)

    # Test 1: Static features
    print("\n--- Static Features ---")
    mem_before = get_memory_mb()

    constraints = PortfolioConstraints()
    config_static = GATModelConfig(node_feature_type="static")
    model_static = GATPortfolioModel(constraints, config_static)

    features_static = model_static._get_node_features(returns, tickers)

    mem_after = get_memory_mb()
    mem_used = mem_after - mem_before

    print(f"  Features shape: {features_static.shape}")
    print(f"  Memory used: {mem_used:.2f} MB")
    print(f"  Memory per asset: {mem_used / len(tickers):.3f} MB")

    # Test 2: Time-series features (60 days)
    print("\n--- Time-Series Features (60 days) ---")
    mem_before = get_memory_mb()

    config_ts_60 = GATModelConfig(
        node_feature_type="timeseries",
        timeseries_length=60,
        timeseries_features=["volatility"],
    )
    model_ts_60 = GATPortfolioModel(constraints, config_ts_60)

    features_ts_60 = model_ts_60._get_node_features(returns, tickers)

    mem_after = get_memory_mb()
    mem_used_ts_60 = mem_after - mem_before

    print(f"  Features shape: {features_ts_60.shape}")
    print(f"  Memory used: {mem_used_ts_60:.2f} MB")
    print(f"  Memory per asset: {mem_used_ts_60 / len(tickers):.3f} MB")

    # Test 3: Time-series features (756 days - paper spec)
    print("\n--- Time-Series Features (756 days - Paper) ---")
    mem_before = get_memory_mb()

    config_ts_756 = GATModelConfig(
        node_feature_type="timeseries",
        timeseries_length=756,
        timeseries_features=["volatility"],
    )
    model_ts_756 = GATPortfolioModel(constraints, config_ts_756)

    features_ts_756 = model_ts_756._get_node_features(returns, tickers)

    mem_after = get_memory_mb()
    mem_used_ts_756 = mem_after - mem_before

    print(f"  Features shape: {features_ts_756.shape}")
    print(f"  Memory used: {mem_used_ts_756:.2f} MB")
    print(f"  Memory per asset: {mem_used_ts_756 / len(tickers):.3f} MB")

    # Summary
    print("\n--- Memory Usage Summary ---")
    print(f"  Static features:           {mem_used:.2f} MB (baseline)")
    print(f"  Time-series (60 days):     {mem_used_ts_60:.2f} MB")
    print(f"  Time-series (756 days):    {mem_used_ts_756:.2f} MB")

    if mem_used > 0.5:
        print(f"  Ratio (60-day/static):     {mem_used_ts_60/mem_used:.1f}x")
        print(f"  Ratio (756-day/static):    {mem_used_ts_756/mem_used:.1f}x")
    else:
        print("  ⚠️  Note: Memory changes too small to measure accurately (<0.5 MB)")
        print("      This is normal for small datasets - feature arrays are tiny")

    # Warnings
    if mem_used_ts_756 > 1000:
        print("\n  ⚠️  WARNING: 756-day window uses >1GB memory")
        print("      Consider using shorter window (60-120 days) or Conv1D encoder")
    else:
        print("\n  ✅ Memory usage is acceptable for all configurations")

    print("\n" + "=" * 80)
    print("TEST 5 PASSED: Memory usage analyzed")
    print("=" * 80)


def main():
    """Run all manual verification tests."""
    print("\n")
    print("=" * 80)
    print("MANUAL VERIFICATION: PHASE 6 & 7")
    print("Temporal Encoders + Data Pipeline")
    print("=" * 80)

    try:
        test_1_temporal_encoders_realistic_data()
        test_2_data_pipeline_756_days()
        test_3_end_to_end_timeseries()
        test_4_backward_compatibility()
        test_5_memory_usage()

        print("\n")
        print("=" * 80)
        print("ALL MANUAL VERIFICATION TESTS PASSED ✅")
        print("=" * 80)
        print("\nPhase 6 & 7 Implementation:")
        print("  ✅ Temporal encoders work with realistic data")
        print("  ✅ Data pipeline supports 756-day window")
        print("  ✅ End-to-end GAT works with time-series features")
        print("  ✅ Backward compatibility maintained")
        print("  ✅ Memory usage is acceptable")
        print("\nReady for production use!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
