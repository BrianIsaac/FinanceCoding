"""Node feature computation for GAT portfolio model.

Simple, proven features from research papers (FinGAT, Korangi et al.).
No complex temporal encoders - just statistical features that work.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


def compute_node_features(
    returns: pd.DataFrame,
    universe: list[str],
    lookback: int = 60
) -> torch.Tensor:
    """Compute simple statistical node features for each asset.

    Args:
        returns: Historical returns DataFrame [time, assets]
        universe: List of asset tickers to compute features for
        lookback: Number of days to look back for feature computation

    Returns:
        Node features tensor [n_assets, n_features]
        Features (11 total):
        - Returns statistics (5): mean, std, min, max, median
        - Momentum (3): 20d, 60d, 120d cumulative returns
        - Volatility (3): 20d, 60d, 120d realized volatility
    """
    features_list = []

    for ticker in universe:
        if ticker not in returns.columns:
            # Asset not in returns - use zero features
            features_list.append(np.zeros(11))
            continue

        asset_returns = returns[ticker].iloc[-lookback:].values

        # Handle missing data
        asset_returns = np.nan_to_num(asset_returns, nan=0.0)

        if len(asset_returns) < 20:
            # Insufficient data - use zero features
            features_list.append(np.zeros(11))
            continue

        # Returns statistics (past lookback days)
        feat = []
        feat.append(np.mean(asset_returns))
        feat.append(np.std(asset_returns) + 1e-8)
        feat.append(np.min(asset_returns))
        feat.append(np.max(asset_returns))
        feat.append(np.median(asset_returns))

        # Momentum (cumulative returns at different windows)
        feat.append(np.sum(asset_returns[-20:]))    # 20-day momentum
        feat.append(np.sum(asset_returns[-60:]) if len(asset_returns) >= 60 else 0.0)  # 60-day
        feat.append(np.sum(asset_returns[-120:]) if len(asset_returns) >= 120 else 0.0)  # 120-day

        # Realized volatility at different windows
        feat.append(np.std(asset_returns[-20:]) + 1e-8)
        feat.append(np.std(asset_returns[-60:]) + 1e-8 if len(asset_returns) >= 60 else 1e-8)
        feat.append(np.std(asset_returns[-120:]) + 1e-8 if len(asset_returns) >= 120 else 1e-8)

        features_list.append(feat)

    # Convert to numpy array
    features_array = np.array(features_list)

    # CRITICAL: Pre-process to handle NaN and inf values BEFORE normalization
    # Replace any NaN or inf values that might have been computed
    features_array = np.nan_to_num(features_array, nan=0.0, posinf=1e6, neginf=-1e6)

    # Check for constant features (all values same) which will cause MinMaxScaler to produce NaN
    # For each feature column, if std is near 0, add small noise to avoid division by zero
    for col_idx in range(features_array.shape[1]):
        col_data = features_array[:, col_idx]
        if np.std(col_data) < 1e-8:
            # Add tiny noise to break constant column
            features_array[:, col_idx] = col_data + np.random.normal(0, 1e-6, len(col_data))

    # CRITICAL: Normalize all features with MinMaxScaler to [-1, 1] range
    # This ensures consistent scale across all feature types (returns stats, momentum, volatility)
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # Handle edge cases: constant features or insufficient data
    try:
        features_normalized = scaler.fit_transform(features_array)

        # Double-check: replace any NaN values with 0 (occurs when feature is constant across all assets)
        if np.any(np.isnan(features_normalized)):
            nan_cols = np.any(np.isnan(features_normalized), axis=0)
            n_nan_cols = np.sum(nan_cols)
            # Set NaN columns to 0 (neutral value in [-1, 1] range)
            features_normalized[:, nan_cols] = 0.0

    except Exception as e:
        # Fallback: if scaling fails completely, use raw features with manual normalization
        # Clip to reasonable range and scale to [-1, 1]
        features_normalized = np.clip(features_array, -10, 10) / 10.0

    # Final safety check: replace any remaining NaN with 0
    features_normalized = np.nan_to_num(features_normalized, nan=0.0, posinf=1.0, neginf=-1.0)

    # Convert to tensor
    features_tensor = torch.tensor(features_normalized, dtype=torch.float32)

    return features_tensor
