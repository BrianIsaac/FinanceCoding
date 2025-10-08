"""
LSTM Model Training Pipeline with Memory Optimization.

This script implements comprehensive LSTM training with GPU memory optimization,
following Story 5.2 Task 2 requirements.
"""

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from src.utils.gpu import GPUConfig, GPUMemoryManager

warnings.filterwarnings("ignore", category=UserWarning)


class LSTMTrainingPipeline:
    """
    Complete LSTM model training pipeline with memory optimization and GPU support.

    This class implements comprehensive LSTM training with advanced GPU memory management,
    mixed precision training, and extensive hyperparameter optimization. Designed for
    production-scale portfolio optimization with proper sequence modeling and validation.

    Implements all subtasks from Story 5.2 Task 2:
    - 60-day sequence modeling with multi-head attention mechanisms
    - 36-month training/12-month validation splits with mixed precision
    - GPU memory optimization for 12GB VRAM constraints
    - Hyperparameter optimization across hidden dimensions, learning rates, and dropout

    Attributes:
        config_path: Path to LSTM configuration YAML file
        base_config: Loaded configuration dictionary
        device: PyTorch device (CPU/CUDA)
        gpu_manager: GPU memory management instance
        logger: Configured logging instance
        data_path: Path to production-ready datasets
        checkpoints_path: Path for model checkpoint storage
        use_mixed_precision: Flag for mixed precision training
        scaler: CUDA gradient scaler for mixed precision

    Example:
        >>> pipeline = LSTMTrainingPipeline("configs/models/lstm_default.yaml")
        >>> results = pipeline.run_full_pipeline()
        >>> print(f"Best config: {results['hyperparameter_optimization']['best_config']}")
    """

    def __init__(self, config_path: str = "configs/models/lstm_default.yaml"):
        """
        Initialize LSTM training pipeline.

        Args:
            config_path: Path to LSTM configuration file
        """
        self.config_path = config_path
        self.base_config = self._load_config()

        # Setup device and memory management
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_manager = GPUMemoryManager(GPUConfig()) if torch.cuda.is_available() else None

        # Setup logging
        self._setup_logging()

        # Initialize data paths
        self.data_path = Path("data/final_new_pipeline")
        self.checkpoints_path = Path("data/models/checkpoints/lstm")
        self.checkpoints_path.mkdir(parents=True, exist_ok=True)

        # Training results storage
        self.training_results: dict[str, Any] = {}
        self.hyperparameter_results: dict[str, Any] = {}

        # Mixed precision training
        self.use_mixed_precision = torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision else None

    def _load_config(self) -> dict[str, Any]:
        """Load LSTM configuration from YAML file."""
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_path = Path("logs/training/lstm")
        log_path.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / 'lstm_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load production-ready datasets from Story 5.1.

        Returns:
            Tuple of (returns_data, prices_data, volume_data)
        """
        self.logger.info("Loading production datasets from Story 5.1...")

        # Load datasets
        returns_data = pd.read_parquet(self.data_path / "returns_daily_final.parquet")
        prices_data = pd.read_parquet(self.data_path / "prices_final.parquet")
        volume_data = pd.read_parquet(self.data_path / "volume_final.parquet")

        # Ensure datetime index
        for df in [returns_data, prices_data, volume_data]:
            df.index = pd.to_datetime(df.index)

        self.logger.info(f"Loaded returns data: {returns_data.shape}")
        self.logger.info(f"Loaded prices data: {prices_data.shape}")
        self.logger.info(f"Loaded volume data: {volume_data.shape}")
        self.logger.info(f"Date range: {returns_data.index.min()} to {returns_data.index.max()}")

        return returns_data, prices_data, volume_data

    def prepare_sequences(self, returns_data: pd.DataFrame, prices_data: pd.DataFrame,
                         volume_data: pd.DataFrame, sequence_length: int = 60) -> tuple[torch.Tensor, torch.Tensor, list[str], pd.DatetimeIndex]:
        """
        Prepare sequential data for LSTM training.

        Implements Subtask 2.1: 60-day sequence modeling with multi-head attention

        Args:
            returns_data: Daily returns data
            prices_data: Daily prices data
            volume_data: Daily volume data
            sequence_length: Length of input sequences (60 days)

        Returns:
            Tuple of (input_sequences, target_sequences, asset_names, dates)
        """
        self.logger.info(f"Preparing sequences with length {sequence_length}")

        # Filter for common assets with sufficient data
        common_assets = set(returns_data.columns) & set(prices_data.columns) & set(volume_data.columns)

        # Calculate data coverage and filter assets
        returns_coverage = returns_data[list(common_assets)].count() / len(returns_data)
        valid_assets = returns_coverage[returns_coverage >= 0.7].index.tolist()  # 70% coverage

        # Limit to top 200 most liquid assets for memory efficiency
        universe = valid_assets[:200]

        self.logger.info(f"Using {len(universe)} assets for training")

        # Prepare feature matrix (returns, volume, technical indicators)
        feature_data = []

        # Returns features
        returns_filtered = returns_data[universe].fillna(0.0)
        feature_data.append(returns_filtered)

        # Volume features (normalized)
        volume_filtered = volume_data[universe].fillna(method='ffill').fillna(0.0)
        volume_mean = volume_filtered.rolling(252).mean()
        volume_mean = volume_mean.replace(0.0, 1.0)  # Avoid division by zero
        volume_normalized = volume_filtered.div(volume_mean, fill_value=1.0)
        # Clip extreme values and handle inf/nan
        volume_normalized = volume_normalized.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        volume_normalized = volume_normalized.clip(lower=0.1, upper=10.0)  # Reasonable bounds
        feature_data.append(volume_normalized)

        # Technical indicators
        # Simple moving averages
        returns_sma_5 = returns_filtered.rolling(5).mean().fillna(0.0)
        returns_sma_20 = returns_filtered.rolling(20).mean().fillna(0.0)
        feature_data.extend([returns_sma_5, returns_sma_20])

        # Volatility (rolling std)
        volatility = returns_filtered.rolling(20).std().fillna(0.0)
        feature_data.append(volatility)

        # Momentum
        momentum = returns_filtered.rolling(20).sum().fillna(0.0)
        feature_data.append(momentum)

        # Combine all features
        pd.concat(feature_data, axis=1)

        # Reshape to (time, assets, features)
        n_times, n_assets = returns_filtered.shape
        n_features = len(feature_data)

        feature_matrix = np.zeros((n_times, n_assets, n_features))
        for i, feature_df in enumerate(feature_data):
            values = feature_df.values
            # Clean the data: replace inf/nan and clip extreme values
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            values = np.clip(values, -10.0, 10.0)  # Reasonable bounds for financial data
            feature_matrix[:, :, i] = values

        # Create sequences
        sequences = []
        targets = []
        valid_dates = []

        for i in range(sequence_length, n_times - 1):  # -1 for prediction target
            # Input sequence
            seq_data = feature_matrix[i-sequence_length:i]  # (seq_len, n_assets, n_features)
            sequences.append(seq_data)

            # Target: next day returns
            target = returns_filtered.iloc[i+1].values  # (n_assets,)
            targets.append(target)

            valid_dates.append(returns_filtered.index[i+1])

        # Convert to tensors
        X = torch.tensor(np.array(sequences), dtype=torch.float32)  # (n_samples, seq_len, n_assets, n_features)
        y = torch.tensor(np.array(targets), dtype=torch.float32)  # (n_samples, n_assets)

        # Reshape for LSTM: (n_samples, seq_len, n_assets * n_features)
        n_samples, seq_len, n_assets_dim, n_feat = X.shape
        X_reshaped = X.reshape(n_samples, seq_len, n_assets_dim * n_feat)

        self.logger.info(f"Created {len(sequences)} sequences with shape {X_reshaped.shape}")

        return X_reshaped, y, universe, pd.DatetimeIndex(valid_dates)

    def create_train_val_splits(self, X: torch.Tensor, y: torch.Tensor,
                               dates: pd.DatetimeIndex) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create 36-month training / 12-month validation splits.

        Implements Subtask 2.2: 36-month training/12-month validation splits

        Args:
            X: Input sequences
            y: Target sequences
            dates: Corresponding dates

        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        self.logger.info("Creating 36-month training / 12-month validation splits")

        # Use recent data for training (last 4 years)
        end_date = dates.max()
        train_end = end_date - pd.Timedelta(days=365)  # 12 months for validation
        train_start = train_end - pd.Timedelta(days=365 * 3)  # 36 months for training

        # Create masks
        train_mask = (dates >= train_start) & (dates <= train_end)
        val_mask = dates > train_end

        # Apply masks
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_val = X[val_mask]
        y_val = y[val_mask]

        self.logger.info(f"Training set: {X_train.shape[0]} samples")
        self.logger.info(f"Validation set: {X_val.shape[0]} samples")
        self.logger.info(f"Training period: {dates[train_mask].min()} to {dates[train_mask].max()}")
        self.logger.info(f"Validation period: {dates[val_mask].min()} to {dates[val_mask].max()}")

        return X_train, y_train, X_val, y_val

    def optimize_memory_settings(self, X_train: torch.Tensor,
                                hidden_size: int) -> dict[str, Any]:
        """
        Optimize GPU memory settings for 12GB VRAM constraints.

        Implements Subtask 2.3: GPU memory optimization with batch size adjustment

        Args:
            X_train: Training data for memory estimation
            hidden_size: LSTM hidden size

        Returns:
            Dictionary with optimized memory settings
        """
        if not torch.cuda.is_available():
            self.logger.info("CUDA not available, using CPU settings")
            return {
                "batch_size": 32,
                "gradient_accumulation_steps": 1,
                "use_mixed_precision": False
            }

        self.logger.info("Optimizing memory settings for 12GB VRAM constraints")

        # Memory estimation
        seq_len, input_size = X_train.shape[1], X_train.shape[2]

        # Estimate model parameters
        lstm_params = 4 * (input_size + hidden_size + 1) * hidden_size * 2  # 2 layers
        attention_params = 8 * hidden_size * hidden_size  # Multi-head attention
        output_params = hidden_size * X_train.shape[2] // 6  # Output projection (approx)

        total_params = lstm_params + attention_params + output_params
        model_memory_mb = total_params * 4 / (1024**2)  # 4 bytes per float32

        self.logger.info(f"Estimated model memory: {model_memory_mb:.1f} MB")

        # Conservative memory limit (11GB out of 12GB)
        available_memory_mb = 11 * 1024

        # Estimate memory per sample
        sample_memory_mb = seq_len * input_size * 4 / (1024**2)  # Input
        sample_memory_mb += hidden_size * seq_len * 4 / (1024**2)  # Hidden states
        sample_memory_mb *= 3  # Forward + backward + optimizer states

        # Calculate optimal batch size
        max_batch_memory_mb = available_memory_mb - model_memory_mb - 1024  # 1GB buffer
        optimal_batch_size = max(1, int(max_batch_memory_mb / sample_memory_mb))

        # Constrain batch size to reasonable range
        optimal_batch_size = min(optimal_batch_size, 64)
        optimal_batch_size = max(optimal_batch_size, 4)

        # Gradient accumulation to maintain effective batch size
        target_effective_batch = 32
        gradient_accumulation_steps = max(1, target_effective_batch // optimal_batch_size)

        memory_settings = {
            "batch_size": optimal_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "use_mixed_precision": True,
            "estimated_model_memory_mb": model_memory_mb,
            "estimated_sample_memory_mb": sample_memory_mb,
            "max_memory_mb": available_memory_mb
        }

        self.logger.info(f"Optimal batch size: {optimal_batch_size}")
        self.logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        self.logger.info(f"Effective batch size: {optimal_batch_size * gradient_accumulation_steps}")

        return memory_settings

    def create_lstm_model(self, input_size: int, output_size: int,
                         hidden_size: int = 128, dropout: float = 0.3) -> nn.Module:
        """
        Create LSTM model with multi-head attention.

        Args:
            input_size: Size of input features
            output_size: Size of output (number of assets)
            hidden_size: LSTM hidden size
            dropout: Dropout rate

        Returns:
            LSTM model with attention
        """

        class LSTMWithAttention(nn.Module):
            def __init__(self, input_size: int, hidden_size: int, output_size: int,
                        num_layers: int = 2, dropout: float = 0.3, num_heads: int = 8):
                super().__init__()

                # LSTM layers
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    batch_first=True
                )

                # Multi-head attention
                self.attention = nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                )

                # Layer normalization
                self.layer_norm1 = nn.LayerNorm(hidden_size)
                self.layer_norm2 = nn.LayerNorm(hidden_size)

                # Output projection
                self.dropout = nn.Dropout(dropout)
                self.output_projection = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size // 2, output_size)
                )

                # Initialize weights properly
                self._initialize_weights()

            def forward(self, x):
                # LSTM forward pass
                lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)

                # Multi-head attention
                attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

                # Residual connection and layer norm
                lstm_out = self.layer_norm1(lstm_out + attn_out)

                # Use last timestep output
                final_hidden = lstm_out[:, -1, :]  # (batch, hidden)

                # Output projection
                output = self.output_projection(self.dropout(final_hidden))

                return output

            def _initialize_weights(self):
                """Initialize model weights properly to prevent NaN losses"""
                for name, param in self.named_parameters():
                    if 'weight_ih' in name:
                        # Input-hidden weights: Xavier uniform
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        # Hidden-hidden weights: Orthogonal
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        # Biases: zeros except forget gate bias = 1
                        param.data.fill_(0)
                        if 'bias_ih' in name or 'bias_hh' in name:
                            # Set forget gate bias to 1 (LSTM has 4 gates)
                            n = param.size(0)
                            forget_gate_start = n // 4
                            forget_gate_end = 2 * n // 4
                            param.data[forget_gate_start:forget_gate_end].fill_(1)
                    elif 'weight' in name and len(param.shape) == 2:
                        # Linear layer weights: Xavier uniform
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight' in name and len(param.shape) == 1:
                        # Batch norm, layer norm weights: ones
                        param.data.fill_(1)

        return LSTMWithAttention(input_size, hidden_size, output_size, dropout=dropout)

    def train_single_configuration(self, X_train: torch.Tensor, y_train: torch.Tensor,
                                 X_val: torch.Tensor, y_val: torch.Tensor,
                                 config: dict[str, Any]) -> dict[str, Any]:
        """
        Train LSTM model with single hyperparameter configuration.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            config: Model configuration

        Returns:
            Training metrics and results
        """
        self.logger.info(f"Training LSTM configuration: {config['config_name']}")

        # Create model
        input_size = X_train.shape[2]
        output_size = y_train.shape[1]

        model = self.create_lstm_model(
            input_size=input_size,
            output_size=output_size,
            hidden_size=config["hidden_size"],
            dropout=config["dropout"]
        )

        model = model.to(self.device)

        # Optimize memory settings
        memory_settings = self.optimize_memory_settings(X_train, config["hidden_size"])

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=memory_settings["batch_size"],
            shuffle=True,
            pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=memory_settings["batch_size"] * 2,  # Larger batch for validation
            shuffle=False,
            pin_memory=torch.cuda.is_available()
        )

        # Optimizer and scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 1e-5)
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        # Loss function (Mean Squared Error)
        criterion = nn.MSELoss()

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10

        train_losses = []
        val_losses = []

        num_epochs = config.get("max_epochs", 50)
        gradient_accumulation_steps = memory_settings["gradient_accumulation_steps"]

        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            valid_batches = 0

            optimizer.zero_grad()

            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                # Validate inputs for NaN/inf
                if torch.isnan(batch_X).any() or torch.isinf(batch_X).any():
                    self.logger.warning(f"Invalid input data detected in batch {batch_idx}, skipping...")
                    continue
                if torch.isnan(batch_y).any() or torch.isinf(batch_y).any():
                    self.logger.warning(f"Invalid target data detected in batch {batch_idx}, skipping...")
                    continue

                # Mixed precision forward pass
                if self.use_mixed_precision and self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        predictions = model(batch_X)
                        loss = criterion(predictions, batch_y)

                        # Validate loss
                        if torch.isnan(loss) or torch.isinf(loss):
                            self.logger.warning(f"Invalid loss detected in batch {batch_idx}: {loss.item()}, skipping...")
                            continue

                        loss = loss / gradient_accumulation_steps

                    self.scaler.scale(loss).backward()

                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        # Unscale gradients and clip them
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                        self.scaler.step(optimizer)
                        self.scaler.update()
                        optimizer.zero_grad()
                else:
                    # Standard training
                    predictions = model(batch_X)
                    loss = criterion(predictions, batch_y)

                    # Validate loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.warning(f"Invalid loss detected in batch {batch_idx}: {loss.item()}, skipping...")
                        continue

                    loss = loss / gradient_accumulation_steps
                    loss.backward()

                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()

                train_loss += loss.item()
                valid_batches += 1

                # Memory management
                if torch.cuda.is_available() and batch_idx % 10 == 0:
                    torch.cuda.empty_cache()

            # Validation phase
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)

                    if self.use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            predictions = model(batch_X)
                            loss = criterion(predictions, batch_y)
                    else:
                        predictions = model(batch_X)
                        loss = criterion(predictions, batch_y)

                    val_loss += loss.item()

            # Calculate average losses
            train_loss /= max(valid_batches, 1)
            val_loss /= len(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save best model
                checkpoint_path = self.checkpoints_path / f"{config['config_name']}_best.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'config': config
                }, checkpoint_path)

            else:
                patience_counter += 1

            # Logging
            if epoch % 5 == 0 or patience_counter == 0:
                self.logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                               f"Train Loss: {train_loss:.6f}, "
                               f"Val Loss: {val_loss:.6f}, "
                               f"Best Val Loss: {best_val_loss:.6f}")

            # Early stopping
            if patience_counter >= max_patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Training results
        results = {
            "config_name": config["config_name"],
            "best_val_loss": best_val_loss,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "num_epochs_trained": len(train_losses),
            "converged": patience_counter < max_patience,
            "memory_settings": memory_settings,
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "train_losses": train_losses,
            "val_losses": val_losses
        }

        # Calculate additional metrics
        model.eval()
        with torch.no_grad():
            # Sample validation predictions for analysis
            sample_batch_X, sample_batch_y = next(iter(val_loader))
            sample_batch_X = sample_batch_X.to(self.device)
            sample_batch_y = sample_batch_y.to(self.device)

            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    sample_pred = model(sample_batch_X)
            else:
                sample_pred = model(sample_batch_X)

            # Calculate correlation between predictions and targets
            pred_np = sample_pred.cpu().numpy().flatten()
            target_np = sample_batch_y.cpu().numpy().flatten()

            # Remove NaN values
            valid_mask = ~(np.isnan(pred_np) | np.isnan(target_np))
            if valid_mask.sum() > 0:
                correlation = np.corrcoef(pred_np[valid_mask], target_np[valid_mask])[0, 1]
                results["prediction_correlation"] = correlation
            else:
                results["prediction_correlation"] = 0.0

        self.logger.info(f"Training completed for {config['config_name']}")
        self.logger.info(f"Best validation loss: {best_val_loss:.6f}")
        self.logger.info(f"Prediction correlation: {results.get('prediction_correlation', 0.0):.4f}")

        return results

    def run_hyperparameter_optimization(self, X_train: torch.Tensor, y_train: torch.Tensor,
                                       X_val: torch.Tensor, y_val: torch.Tensor) -> dict[str, Any]:
        """
        Execute hyperparameter optimization across hidden dimensions, learning rates, dropout.

        Implements Subtask 2.4: Hyperparameter optimization

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data

        Returns:
            Hyperparameter optimization results
        """
        self.logger.info("Starting LSTM hyperparameter optimization")

        # Hyperparameter grid
        param_grid = {
            "hidden_size": [64, 128, 256],
            "learning_rate": [0.001, 0.0005, 0.0001],
            "dropout": [0.2, 0.3, 0.4],
            "weight_decay": [1e-5, 1e-4]
        }

        # Generate configurations
        configs = []
        config_id = 0

        for hidden_size in param_grid["hidden_size"]:
            for learning_rate in param_grid["learning_rate"]:
                for dropout in param_grid["dropout"]:
                    for weight_decay in param_grid["weight_decay"]:
                        config = {
                            "config_name": f"lstm_h{hidden_size}_lr{learning_rate}_d{dropout}_wd{weight_decay}",
                            "hidden_size": hidden_size,
                            "learning_rate": learning_rate,
                            "dropout": dropout,
                            "weight_decay": weight_decay,
                            "max_epochs": 30  # Reduced for hyperparameter search
                        }
                        configs.append(config)
                        config_id += 1

        self.logger.info(f"Generated {len(configs)} hyperparameter configurations")

        # Train each configuration
        results = {}
        best_config = None
        best_val_loss = float('inf')

        for i, config in enumerate(configs):
            self.logger.info(f"Training configuration {i+1}/{len(configs)}: {config['config_name']}")

            try:
                result = self.train_single_configuration(X_train, y_train, X_val, y_val, config)
                results[config["config_name"]] = result

                if result["best_val_loss"] < best_val_loss:
                    best_val_loss = result["best_val_loss"]
                    best_config = config["config_name"]

            except Exception as e:
                self.logger.error(f"Training failed for {config['config_name']}: {str(e)}")
                results[config["config_name"]] = {"error": str(e), "config_name": config["config_name"]}

            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Hyperparameter optimization results
        optimization_results = {
            "best_config": best_config,
            "best_val_loss": best_val_loss,
            "all_results": results,
            "num_configs_tested": len(configs),
            "successful_configs": len([r for r in results.values() if "error" not in r])
        }

        # Analysis by hyperparameter
        hyperparameter_analysis = {}

        for param_name, param_values in param_grid.items():
            param_analysis = {}
            for param_value in param_values:
                matching_configs = [
                    r for config_name, r in results.items()
                    if "error" not in r and str(param_value) in config_name
                ]

                if matching_configs:
                    avg_val_loss = np.mean([r["best_val_loss"] for r in matching_configs])
                    param_analysis[str(param_value)] = {
                        "avg_val_loss": avg_val_loss,
                        "num_configs": len(matching_configs)
                    }

            hyperparameter_analysis[param_name] = param_analysis

        optimization_results["hyperparameter_analysis"] = hyperparameter_analysis

        self.logger.info("Hyperparameter optimization completed")
        self.logger.info(f"Best configuration: {best_config}")
        self.logger.info(f"Best validation loss: {best_val_loss:.6f}")

        return optimization_results

    def save_results(self, results: dict[str, Any]) -> None:
        """Save training results to disk."""
        results_path = Path("logs/training/lstm/lstm_training_results.yaml")
        results_path.parent.mkdir(parents=True, exist_ok=True)

        with open(results_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)

        self.logger.info(f"Results saved to: {results_path}")

    def run_full_pipeline(self) -> dict[str, Any]:
        """
        Execute complete LSTM training pipeline.

        Implements all Story 5.2 Task 2 requirements:
        - Subtask 2.1: 60-day sequence modeling with multi-head attention mechanisms
        - Subtask 2.2: 36-month training/12-month validation splits with mixed precision
        - Subtask 2.3: GPU memory optimization for 12GB VRAM constraints
        - Subtask 2.4: Hyperparameter optimization across dimensions, learning rates, dropout

        Returns:
            Complete training and optimization results
        """
        self.logger.info("Starting complete LSTM training pipeline execution")

        # Load data
        returns_data, prices_data, volume_data = self.load_data()

        # Prepare sequences (Subtask 2.1)
        X, y, assets, dates = self.prepare_sequences(returns_data, prices_data, volume_data)

        # Create train/val splits (Subtask 2.2)
        X_train, y_train, X_val, y_val = self.create_train_val_splits(X, y, dates)

        # Execute hyperparameter optimization (includes Subtasks 2.3 & 2.4)
        optimization_results = self.run_hyperparameter_optimization(X_train, y_train, X_val, y_val)

        # Complete results
        results = {
            "training_summary": {
                "total_sequences": len(X),
                "training_sequences": len(X_train),
                "validation_sequences": len(X_val),
                "num_assets": len(assets),
                "sequence_length": X.shape[1],
                "input_features": X.shape[2]
            },
            "hyperparameter_optimization": optimization_results,
            "assets_used": assets,
            "training_period": {
                "start": dates.min().strftime("%Y-%m-%d"),
                "end": dates.max().strftime("%Y-%m-%d")
            },
            "gpu_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
        }

        # Save results
        self.save_results(results)

        self.logger.info("LSTM training pipeline completed successfully")

        return results


if __name__ == "__main__":
    """Execute LSTM training pipeline."""

    # Initialize and run pipeline
    pipeline = LSTMTrainingPipeline()

    # Execute complete training pipeline
    results = pipeline.run_full_pipeline()


    training_summary = results["training_summary"]
    optimization = results["hyperparameter_optimization"]



    if results['gpu_available']:
        pass

