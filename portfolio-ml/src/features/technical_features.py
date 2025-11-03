"""Technical feature extraction from price/return data.

This module implements feature engineering for LSTM models, extracting
momentum, volatility, mean reversion, and cross-sectional signals from
price and return data.

Research support:
- Jegadeesh & Titman (1993): Momentum effects
- Ang et al. (2006): Low volatility anomaly
- Lehmann (1990): Short-term mean reversion
- DeMiguel et al. (2009): Cross-sectional signals
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TechnicalFeatureExtractor:
    """Extract technical features from price/return data for LSTM.

    Features extracted:
    - Minimal (7): Multi-horizon returns (3) + volatility (2) + returns (1) + vol momentum (1)
    - Standard (9): Minimal + mean reversion (1) + RSI (1)
    - Full (12): Standard + cross-sectional ranks (2) + market regime (1)

    Args:
        feature_set: Which features to extract ("minimal", "standard", "full")
        lookback_windows: List of lookback periods for momentum features
    """

    def __init__(
        self,
        feature_set: str = "standard",
        lookback_windows: Optional[List[int]] = None,
    ):
        """Initialise feature extractor.

        Args:
            feature_set: Which features to extract
                - minimal: 7 features (multi-horizon returns + volatility)
                - standard: 9 features (+ mean reversion, RSI)
                - full: 12 features (+ cross-sectional ranks, market regime)
            lookback_windows: List of lookback periods for momentum
        """
        if feature_set not in ["minimal", "standard", "full"]:
            raise ValueError(f"feature_set must be 'minimal', 'standard', or 'full', got {feature_set}")

        self.feature_set = feature_set
        self.lookback_windows = lookback_windows or [5, 21, 63]

        logger.info(f"Initialised {feature_set} feature extractor with lookback windows: {self.lookback_windows}")

    def extract_features(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        benchmark_prices: Optional[pd.Series] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract all features and stack into [T, N, F] array.

        Args:
            prices: Asset prices [T, N]
            returns: Asset returns [T, N]
            benchmark_prices: Market benchmark prices [T] for regime features

        Returns:
            features: [T, N, F] array where F = number of features
            feature_names: List of feature names in order
        """
        logger.info(f"Extracting {self.feature_set} features from data with shape {returns.shape}")
        logger.debug(f"Input dimensions: prices={prices.shape}, returns={returns.shape}")

        features_dict = {}

        # Feature 1: Current returns (baseline) - keep as DataFrame for alignment
        features_dict['returns'] = returns

        if self.feature_set in ['minimal', 'standard', 'full']:
            # Features 2-4: Multi-horizon returns (momentum)
            for window in self.lookback_windows:
                feature_name = f'returns_{window}d'
                momentum = prices.pct_change(window)
                features_dict[feature_name] = momentum  # Keep as DataFrame
                logger.debug(
                    f"Extracted {feature_name}: shape={momentum.shape}, "
                    f"mean={momentum.mean().mean():.4f}, std={momentum.std().mean():.4f}"
                )
                logger.debug(f"Feature '{feature_name}' before alignment: shape={momentum.shape}")

            # Feature 5: Realised volatility (21-day)
            vol_21d = returns.rolling(21, min_periods=10).std()
            features_dict['volatility_21d'] = vol_21d  # Keep as DataFrame

            # Feature 6: Long-term volatility (63-day)
            vol_63d = returns.rolling(63, min_periods=20).std()
            features_dict['volatility_63d'] = vol_63d  # Keep as DataFrame

            # Feature 7: Volatility momentum (regime changes)
            vol_change = (vol_21d / (vol_63d + 1e-8)) - 1.0
            features_dict['vol_momentum'] = vol_change  # Keep as DataFrame

        if self.feature_set in ['standard', 'full']:
            # Feature 8: Mean reversion (z-score from 21-day MA)
            ma_21 = prices.rolling(21, min_periods=10).mean()
            std_21 = prices.rolling(21, min_periods=10).std()
            z_score = (prices - ma_21) / (std_21 + 1e-8)
            features_dict['z_score'] = z_score  # Keep as DataFrame

            # Feature 9: RSI (14-day Relative Strength Index)
            rsi = self._calculate_rsi(returns, window=14)
            features_dict['rsi'] = rsi  # Keep as DataFrame

        if self.feature_set == 'full':
            # Feature 10: Cross-sectional return rank
            return_rank = returns.apply(lambda row: row.rank(pct=True), axis=1)
            features_dict['return_rank'] = return_rank  # Keep as DataFrame

            # Feature 11: Cross-sectional volatility rank
            vol_rank = vol_21d.apply(lambda row: row.rank(pct=True), axis=1)
            features_dict['vol_rank'] = vol_rank  # Keep as DataFrame

            # Feature 12: Market regime (if benchmark provided)
            if benchmark_prices is not None:
                bench_returns = benchmark_prices.pct_change()
                market_vol = bench_returns.rolling(21, min_periods=10).std()
                # Broadcast to all assets
                market_vol_broadcast = pd.DataFrame(
                    np.tile(market_vol.values[:, None], (1, returns.shape[1])),
                    index=returns.index,
                    columns=returns.columns
                )
                features_dict['market_vol'] = market_vol_broadcast  # Keep as DataFrame
            else:
                logger.warning("Benchmark prices not provided, skipping market regime feature")

        # Stack features along last dimension: [T, N, F]
        # Ensure all features are aligned to returns index
        feature_names = list(features_dict.keys())
        aligned_features = []
        for name in feature_names:
            feat = features_dict[name]
            if isinstance(feat, pd.DataFrame):
                # Reindex to match returns index
                feat_aligned = feat.reindex(returns.index, fill_value=np.nan)
                logger.debug(
                    f"Aligning feature '{name}': original_shape={feat.shape}, "
                    f"aligned_shape={feat_aligned.shape}"
                )
                aligned_features.append(feat_aligned.values)
            else:
                logger.debug(f"Feature '{name}' already aligned: shape={feat.shape}")
                aligned_features.append(feat)

        # FIX: Add comprehensive shape validation logging BEFORE np.stack()
        logger.info(f"Shape validation before stacking {len(aligned_features)} features:")
        for i, (name, feat) in enumerate(zip(feature_names, aligned_features)):
            logger.info(f"  Feature {i}: '{name}' shape={feat.shape}")

        # Verify all features have the same shape before stacking
        expected_shape = aligned_features[0].shape
        for i, (name, feat) in enumerate(zip(feature_names, aligned_features)):
            if feat.shape != expected_shape:
                raise ValueError(
                    f"Shape mismatch in feature stacking: "
                    f"Feature '{name}' at index {i} has shape {feat.shape} but expected {expected_shape}. "
                    f"All features must have the same [T, N] dimensions before stacking along axis=-1."
                )

        feature_array = np.stack(aligned_features, axis=-1)

        # FIX: Add assertion to verify final stacked shape matches expected
        expected_final_shape = (expected_shape[0], expected_shape[1], len(aligned_features))
        assert feature_array.shape == expected_final_shape, (
            f"Feature stacking produced unexpected shape: "
            f"got {feature_array.shape} but expected {expected_final_shape} "
            f"([T={expected_shape[0]}, N={expected_shape[1]}, F={len(aligned_features)}])"
        )
        logger.info(f"Feature stacking successful: final shape={feature_array.shape} matches expected {expected_final_shape}")

        # Handle NaN (forward fill then backward fill, then zero)
        original_nans = np.isnan(feature_array).sum()
        for i in range(feature_array.shape[-1]):
            feature_slice = feature_array[:, :, i]
            df_temp = pd.DataFrame(feature_slice)
            df_temp = df_temp.ffill().bfill().fillna(0.0)
            feature_array[:, :, i] = df_temp.values

        final_nans = np.isnan(feature_array).sum()
        logger.info(f"Feature extraction complete: {len(feature_names)} features, shape {feature_array.shape}")
        logger.info(f"NaN handling: {original_nans} → {final_nans} (filled {original_nans - final_nans})")

        return feature_array, feature_names

    def _calculate_rsi(self, returns: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index.

        Args:
            returns: Daily returns [T, N]
            window: RSI calculation window (typically 14)

        Returns:
            rsi: RSI values [T, N], range [0, 100]
        """
        # Separate gains and losses
        gains = returns.clip(lower=0)
        losses = -returns.clip(upper=0)

        # Calculate rolling averages
        avg_gain = gains.rolling(window, min_periods=max(1, window // 2)).mean()
        avg_loss = losses.rolling(window, min_periods=max(1, window // 2)).mean()

        # Calculate RS and RSI
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))

        return rsi


def create_feature_extractor(config_name: str = "standard") -> TechnicalFeatureExtractor:
    """Factory function to create feature extractor with preset configs.

    Args:
        config_name: Preset configuration ("minimal", "standard", "full")

    Returns:
        Configured TechnicalFeatureExtractor instance
    """
    configs = {
        "minimal": {
            "feature_set": "minimal",
            "lookback_windows": [5, 21, 63],
        },
        "standard": {
            "feature_set": "standard",
            "lookback_windows": [5, 21, 63],
        },
        "full": {
            "feature_set": "full",
            "lookback_windows": [5, 21, 63],
        },
    }

    if config_name not in configs:
        raise ValueError(f"config_name must be one of {list(configs.keys())}, got {config_name}")

    return TechnicalFeatureExtractor(**configs[config_name])
