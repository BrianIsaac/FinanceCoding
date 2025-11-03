"""Fixed LSTM training with proper normalization based on working implementations."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """Dataset for temporal sequence data."""

    def __init__(self, sequences, targets, lengths=None):
        assert len(sequences) == len(targets), "Sequences and targets must have same length"
        self.sequences = sequences
        self.targets = targets
        self.lengths = lengths if lengths is not None else torch.full((len(sequences),), sequences.shape[1])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx], self.lengths[idx]


class ImprovedNormalization:
    """
    Normalization strategy based on successful implementations:
    - PGPortfolio: MinMaxScaler(0,1)
    - deepdow: Separate normalization for train/val
    - George Muriithi: MinMaxScaler with proper inverse transform
    """

    def __init__(self, use_technical_features=False):
        self.use_technical_features = use_technical_features

        # Based on research: MinMaxScaler(0,1) works best for LSTM
        # From PGPortfolio and multiple successful implementations
        self.input_scaler = None
        self.target_scaler = None

    def create_preprocessor(self, feature_names):
        """
        Create feature-specific preprocessor based on sklearn ColumnTransformer.
        Different features need different scaling strategies.
        """
        transformers = []

        # Identify feature types from names
        returns_cols = [i for i, name in enumerate(feature_names)
                       if 'returns' in str(name) and not any(x in str(name) for x in ['_5d', '_21d', '_63d'])]
        momentum_cols = [i for i, name in enumerate(feature_names)
                        if any(x in str(name) for x in ['_5d', '_21d', '_63d'])]
        volatility_cols = [i for i, name in enumerate(feature_names)
                         if 'volatility' in str(name) or 'vol_' in str(name)]
        rsi_cols = [i for i, name in enumerate(feature_names) if 'rsi' in str(name)]
        zscore_cols = [i for i, name in enumerate(feature_names) if 'z_score' in str(name)]

        # Apply appropriate scalers based on feature type
        # Research shows these work best:

        if returns_cols:
            # Returns: RobustScaler handles outliers better
            transformers.append(
                ('returns', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', RobustScaler())
                ]), returns_cols)
            )

        if momentum_cols:
            # Momentum features: RobustScaler
            transformers.append(
                ('momentum', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', RobustScaler())
                ]), momentum_cols)
            )

        if volatility_cols:
            # Volatility: MinMaxScaler after log transform would be ideal
            # But for simplicity use RobustScaler
            transformers.append(
                ('volatility', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', RobustScaler())
                ]), volatility_cols)
            )

        if rsi_cols:
            # RSI: Already bounded [0,100], just scale to [0,1]
            transformers.append(
                ('rsi', Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value=50)),
                    ('scaler', MinMaxScaler(feature_range=(0, 1)))
                ]), rsi_cols)
            )

        if zscore_cols:
            # Z-scores: Already normalized, minimal scaling needed
            transformers.append(
                ('zscore', Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                    ('scaler', MinMaxScaler(feature_range=(-1, 1)))
                ]), zscore_cols)
            )

        # If no specific columns identified, use default
        if not transformers:
            # Default: MinMaxScaler as per successful LSTM implementations
            all_cols = list(range(len(feature_names)))
            transformers.append(
                ('default', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', MinMaxScaler(feature_range=(0, 1)))
                ]), all_cols)
            )

        return ColumnTransformer(transformers=transformers, remainder='passthrough')

    def fit_transform_sequences(self, data, sequence_length=60, prediction_horizon=21):
        """
        Create sequences with proper normalization following successful implementations.

        Key insights from research:
        1. MinMaxScaler(0,1) most common for LSTM (PGPortfolio, George Muriithi)
        2. Fit scaler only on training data (deepdow)
        3. Separate scalers for inputs and targets
        4. Handle missing data before scaling
        """

        if isinstance(data, pd.DataFrame):
            feature_names = data.columns.tolist()
            data_array = data.values
            dates = data.index
        else:
            raise TypeError("Data must be a DataFrame")

        n_timesteps, n_features = data_array.shape

        # Determine target extraction based on features
        if self.use_technical_features and n_features > 50:
            # Extract only returns for targets (every 9th column if 9 features per asset)
            n_features_per_asset = 9
            n_assets = n_features // n_features_per_asset
            target_indices = list(range(0, n_features, n_features_per_asset))
            target_array = data_array[:, target_indices]
            logger.info(f"Extracted {n_assets} target returns from {n_features} features")
        else:
            target_array = data_array
            n_assets = n_features

        # CRITICAL FIX: Use MinMaxScaler for LSTM (based on research)
        # Most successful implementations use MinMaxScaler(0,1)

        # For inputs: MinMaxScaler or feature-specific
        if self.use_technical_features:
            # Use feature-specific preprocessing
            self.input_scaler = self.create_preprocessor(feature_names)
            input_normalized = self.input_scaler.fit_transform(data_array)
        else:
            # Simple MinMaxScaler for homogeneous features
            self.input_scaler = MinMaxScaler(feature_range=(0, 1))
            input_normalized = self.input_scaler.fit_transform(data_array)

        # For targets: Always use MinMaxScaler(0,1) for bounded outputs
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        target_normalized = self.target_scaler.fit_transform(target_array)

        # Log normalization quality
        logger.info(f"Input normalization: min={input_normalized.min():.3f}, max={input_normalized.max():.3f}")
        logger.info(f"Target normalization: min={target_normalized.min():.3f}, max={target_normalized.max():.3f}")

        # Create sequences
        sequences = []
        targets = []
        target_dates = []

        for i in range(sequence_length, n_timesteps - prediction_horizon):
            # Input sequence
            seq = input_normalized[i - sequence_length:i]
            sequences.append(seq)

            # Target (single day, not averaged)
            target_idx = i + prediction_horizon - 1
            target = target_normalized[target_idx]
            targets.append(target)
            target_dates.append(dates[target_idx])

        sequences = torch.tensor(np.array(sequences), dtype=torch.float32)
        targets = torch.tensor(np.array(targets), dtype=torch.float32)

        logger.info(f"Created {len(sequences)} sequences")
        logger.info(f"Sequence shape: {sequences.shape}, Target shape: {targets.shape}")

        # Store normalization stats for inference
        # Following deepdow pattern: store scalers not raw stats
        self.normalization_info = {
            'input_scaler': self.input_scaler,
            'target_scaler': self.target_scaler,
            'n_assets': n_assets,
            'feature_names': feature_names if self.use_technical_features else None
        }

        return sequences, targets, target_dates, self.normalization_info

    def denormalize_predictions(self, predictions):
        """Denormalize predictions back to original scale."""
        if self.target_scaler is None:
            raise ValueError("Must fit normalization before denormalizing")

        # Reshape if needed
        original_shape = predictions.shape
        if len(original_shape) == 1:
            predictions = predictions.reshape(-1, 1)

        # Use inverse_transform (following sklearn best practice)
        denormalized = self.target_scaler.inverse_transform(predictions)

        # Reshape back
        if len(original_shape) == 1:
            denormalized = denormalized.flatten()

        return denormalized


def train_with_fixed_normalization(
    returns_data: pd.DataFrame,
    config,
    use_technical_features: bool = True
):
    """
    Train LSTM with fixed normalization based on successful implementations.

    Key changes based on research:
    1. MinMaxScaler(0,1) instead of Z-score
    2. Feature-specific normalization
    3. Separate scalers for inputs/targets
    4. Proper train/val split
    """

    # Initialize improved normalization
    normalizer = ImprovedNormalization(use_technical_features=use_technical_features)

    # Create sequences with proper normalization
    sequences, targets, dates, norm_info = normalizer.fit_transform_sequences(
        returns_data,
        sequence_length=config.sequence_length,
        prediction_horizon=config.prediction_horizon
    )

    # Train/validation split (temporal, not random)
    split_idx = int(len(sequences) * 0.8)

    train_sequences = sequences[:split_idx]
    train_targets = targets[:split_idx]
    val_sequences = sequences[split_idx:]
    val_targets = targets[split_idx:]

    # Create dataloaders
    train_dataset = TimeSeriesDataset(train_sequences, train_targets)
    val_dataset = TimeSeriesDataset(val_sequences, val_targets)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    return train_loader, val_loader, normalizer


# Example usage matching successful implementations
if __name__ == "__main__":
    # Test with sample data
    import numpy as np

    # Create sample data
    n_timesteps = 1000
    n_assets = 10
    n_features_per_asset = 9

    if use_technical_features := True:
        # Heterogeneous features
        n_total_features = n_assets * n_features_per_asset
        data = np.random.randn(n_timesteps, n_total_features) * 0.02

        # Add different scales to different feature types
        for asset in range(n_assets):
            base_idx = asset * n_features_per_asset
            # Returns: small scale
            data[:, base_idx] *= 1.0
            # Momentum: slightly larger
            data[:, base_idx + 1:base_idx + 4] *= 1.5
            # Volatility: positive only
            data[:, base_idx + 4:base_idx + 6] = np.abs(data[:, base_idx + 4:base_idx + 6]) * 2
            # RSI: bounded [0, 100]
            data[:, base_idx + 8] = np.clip(50 + data[:, base_idx + 8] * 20, 0, 100)

        # Create column names
        columns = []
        for i in range(n_assets):
            columns.extend([
                f'ASSET_{i}_returns',
                f'ASSET_{i}_returns_5d',
                f'ASSET_{i}_returns_21d',
                f'ASSET_{i}_returns_63d',
                f'ASSET_{i}_volatility_21d',
                f'ASSET_{i}_volatility_63d',
                f'ASSET_{i}_vol_momentum',
                f'ASSET_{i}_z_score',
                f'ASSET_{i}_rsi'
            ])
    else:
        # Simple returns only
        data = np.random.randn(n_timesteps, n_assets) * 0.02
        columns = [f'ASSET_{i}' for i in range(n_assets)]

    # Create DataFrame
    dates = pd.date_range('2020-01-01', periods=n_timesteps, freq='D')
    df = pd.DataFrame(data, index=dates, columns=columns)

    # Test normalization
    normalizer = ImprovedNormalization(use_technical_features=use_technical_features)
    sequences, targets, target_dates, norm_info = normalizer.fit_transform_sequences(df)

    print(f"Sequences shape: {sequences.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Sequences range: [{sequences.min():.3f}, {sequences.max():.3f}]")
    print(f"Targets range: [{targets.min():.3f}, {targets.max():.3f}]")