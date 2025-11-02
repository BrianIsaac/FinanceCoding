"""Debug script to find source of NaN in time-series features."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.models.gat.model import GATPortfolioModel, GATModelConfig
from src.models.base.portfolio_model import PortfolioConstraints

# Generate small test case
np.random.seed(42)
dates = pd.date_range("2024-01-01", periods=100, freq="D")
tickers = ["ASSET1", "ASSET2", "ASSET3"]

# Create returns with some NaN
returns = pd.DataFrame(
    np.random.randn(100, 3) * 0.02,
    index=dates,
    columns=tickers,
)

# Add NaN values
returns.iloc[10:15, 0] = np.nan  # ASSET1 missing days 10-14
returns.iloc[20, 1] = np.nan      # ASSET2 missing day 20

print("Test returns:")
print(f"  Shape: {returns.shape}")
print(f"  NaN count: {returns.isna().sum().sum()}")
print(f"  NaN locations:")
for col in returns.columns:
    nan_indices = returns[returns[col].isna()].index.tolist()
    if nan_indices:
        print(f"    {col}: {len(nan_indices)} NaN values")

# Test time-series feature preparation
constraints = PortfolioConstraints()
config = GATModelConfig()
model = GATPortfolioModel(constraints, config)

print("\n" + "="*60)
print("Testing each feature type:")
print("="*60)

# Test 1: Volatility only
print("\n1. Volatility only:")
features_vol = model._prepare_timeseries_features(
    returns, tickers, window_length=60, features=["volatility"]
)
print(f"  Output shape: {features_vol.shape}")
print(f"  Contains NaN: {np.isnan(features_vol).any()}")
print(f"  NaN count: {np.isnan(features_vol).sum()}")

# Test 2: Returns only
print("\n2. Returns only:")
features_ret = model._prepare_timeseries_features(
    returns, tickers, window_length=60, features=["returns"]
)
print(f"  Output shape: {features_ret.shape}")
print(f"  Contains NaN: {np.isnan(features_ret).any()}")
print(f"  NaN count: {np.isnan(features_ret).sum()}")
if np.isnan(features_ret).any():
    print(f"  NaN locations:")
    for i in range(features_ret.shape[0]):
        nan_count = np.isnan(features_ret[i, :, 0]).sum()
        if nan_count > 0:
            print(f"    Asset {i} ({tickers[i]}): {nan_count} NaN values")

# Test 3: Momentum only
print("\n3. Momentum only:")
features_mom = model._prepare_timeseries_features(
    returns, tickers, window_length=60, features=["momentum"]
)
print(f"  Output shape: {features_mom.shape}")
print(f"  Contains NaN: {np.isnan(features_mom).any()}")
print(f"  NaN count: {np.isnan(features_mom).sum()}")

# Test 4: All features
print("\n4. All features (volatility, returns, momentum):")
features_all = model._prepare_timeseries_features(
    returns, tickers, window_length=60, features=["volatility", "returns", "momentum"]
)
print(f"  Output shape: {features_all.shape}")
print(f"  Contains NaN: {np.isnan(features_all).any()}")
print(f"  NaN count: {np.isnan(features_all).sum()}")
if np.isnan(features_all).any():
    print(f"  NaN by feature:")
    feature_names = ["volatility", "returns", "momentum"]
    for j in range(features_all.shape[2]):
        nan_count = np.isnan(features_all[:, :, j]).sum()
        print(f"    {feature_names[j]}: {nan_count} NaN values")

print("\n" + "="*60)
print("Conclusion:")
print("="*60)
print("The NaN values in the output come from the feature computation,")
print("specifically when normalizing features that contain NaN values.")
print("Using nanmean/nanstd should handle this, but NaN values in the")
print("raw data will still be present in the normalized output.")
