#!/usr/bin/env python3
"""
Aggressive LSTM Training Pipeline with Larger Models and Extended Sequences
Enhanced version with deeper networks, longer sequences, and more comprehensive hyperparameter search
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.amp import GradScaler

from src.data.loaders.parquet_manager import ParquetManager
from src.models.lstm.architecture import LSTMConfig, LSTMNetwork


class AggressiveLSTMTraining:
    """Aggressive LSTM training with extended architecture and sequence lengths."""

    def __init__(self):
        self.setup_logging()
        self.data_manager = ParquetManager()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_mixed_precision = torch.cuda.is_available()
        self.scaler = GradScaler("cuda") if self.use_mixed_precision else None

        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Mixed precision: {self.use_mixed_precision}")

    def setup_logging(self):
        """Setup comprehensive logging."""
        log_dir = Path("logs/training/lstm_aggressive")
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "lstm_aggressive_training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def generate_aggressive_configs(self) -> list[dict[str, Any]]:
        """
        Generate aggressive LSTM hyperparameter configurations.

        Returns:
            List of aggressive configuration dictionaries
        """
        configurations = []

        # Aggressive architecture parameters
        hidden_sizes = [128, 256, 384, 512, 640, 768]  # Much larger models
        num_layers = [2, 3, 4, 5]  # Deeper networks
        sequence_lengths = [60, 90, 120, 180, 252]  # Longer sequences (up to 1 year)
        attention_heads = [4, 6, 8, 12, 16]  # More attention heads
        dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Aggressive optimization parameters
        learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
        weight_decays = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
        batch_sizes = [32, 64, 128]  # Will be optimized based on GPU memory
        gradient_accumulation_steps = [1, 2, 4, 8]

        # Advanced training strategies
        warmup_epochs = [5, 10, 15, 20]
        scheduler_types = ['reduce_on_plateau', 'cosine_annealing', 'exponential']
        early_stopping_patience = [15, 20, 25, 30]

        config_id = 0

        # Generate comprehensive configuration space
        for hidden_size in hidden_sizes:
            for num_layer in num_layers:
                for seq_len in sequence_lengths:
                    for n_heads in attention_heads:
                        for dropout in dropout_rates:
                            for lr in learning_rates:
                                for wd in weight_decays:
                                    for batch_size in batch_sizes:
                                        for grad_accum in gradient_accumulation_steps:
                                            for warmup in warmup_epochs:
                                                for scheduler in scheduler_types:
                                                    for patience in early_stopping_patience:
                                                        # Skip some combinations to manage size
                                                        if config_id % 5 != 0:  # Sample every 5th config
                                                            config_id += 1
                                                            continue

                                                        # Memory constraint checks
                                                        estimated_params = (
                                                            hidden_size * hidden_size * num_layer * 4 +  # LSTM weights
                                                            hidden_size * n_heads * 64 * 2 +  # Attention weights
                                                            hidden_size * 200  # Output layer (assuming 200 assets)
                                                        )

                                                        # Skip if model is too large for 12GB VRAM
                                                        estimated_memory_mb = (
                                                            estimated_params * 4 * batch_size * seq_len / (1024 * 1024)
                                                        )

                                                        if estimated_memory_mb > 8000:  # Conservative limit
                                                            config_id += 1
                                                            continue

                                                        config_id += 1
                                                        config = {
                                                            'config_id': f"aggressive_lstm_{config_id:04d}",
                                                            'hidden_size': hidden_size,
                                                            'num_layers': num_layer,
                                                            'sequence_length': seq_len,
                                                            'attention_heads': n_heads,
                                                            'dropout': dropout,
                                                            'learning_rate': lr,
                                                            'weight_decay': wd,
                                                            'batch_size': batch_size,
                                                            'gradient_accumulation_steps': grad_accum,
                                                            'warmup_epochs': warmup,
                                                            'scheduler_type': scheduler,
                                                            'early_stopping_patience': patience,
                                                            'max_epochs': 50,  # Longer training
                                                            'use_residual_connections': True,
                                                            'use_layer_normalization': True,
                                                            'use_gradient_clipping': True,
                                                            'gradient_clip_value': 1.0,
                                                            'use_attention_dropout': True,
                                                            'attention_dropout': dropout * 0.5,
                                                            'use_advanced_regularization': True
                                                        }
                                                        configurations.append(config)

        # Add some extreme configurations for testing limits
        extreme_configs = [
            {
                'config_id': 'extreme_deep',
                'hidden_size': 256,
                'num_layers': 8,
                'sequence_length': 120,
                'attention_heads': 16,
                'dropout': 0.4,
                'learning_rate': 0.0001,
                'weight_decay': 1e-4,
                'batch_size': 16,
                'gradient_accumulation_steps': 8,
                'warmup_epochs': 20,
                'scheduler_type': 'cosine_annealing',
                'early_stopping_patience': 40,
                'max_epochs': 100
            },
            {
                'config_id': 'extreme_wide',
                'hidden_size': 1024,
                'num_layers': 2,
                'sequence_length': 60,
                'attention_heads': 32,
                'dropout': 0.5,
                'learning_rate': 0.0005,
                'weight_decay': 1e-3,
                'batch_size': 8,
                'gradient_accumulation_steps': 16,
                'warmup_epochs': 15,
                'scheduler_type': 'reduce_on_plateau',
                'early_stopping_patience': 30,
                'max_epochs': 75
            },
            {
                'config_id': 'extreme_long_sequence',
                'hidden_size': 384,
                'num_layers': 3,
                'sequence_length': 504,  # 2 years of daily data
                'attention_heads': 12,
                'dropout': 0.3,
                'learning_rate': 0.00005,
                'weight_decay': 1e-5,
                'batch_size': 4,
                'gradient_accumulation_steps': 32,
                'warmup_epochs': 25,
                'scheduler_type': 'exponential',
                'early_stopping_patience': 50,
                'max_epochs': 150
            }
        ]

        configurations.extend(extreme_configs)

        self.logger.info(f"Generated {len(configurations)} aggressive LSTM configurations")
        return configurations

    def prepare_aggressive_sequences(self, returns_data: pd.DataFrame,
                                   prices_data: pd.DataFrame,
                                   volume_data: pd.DataFrame,
                                   sequence_length: int = 120,
                                   n_assets: int = 400) -> tuple[torch.Tensor, torch.Tensor, list[str], pd.DatetimeIndex]:
        """
        Prepare aggressive training sequences with extended features.

        Args:
            returns_data: Returns data
            prices_data: Price data
            volume_data: Volume data
            sequence_length: Sequence length
            n_assets: Number of assets to use

        Returns:
            Tuple of (sequences, targets, universe, dates)
        """
        self.logger.info(f"Preparing aggressive sequences with length {sequence_length}")

        # Enhanced universe selection - use more assets with broader criteria
        universe = self.select_aggressive_universe(returns_data, n_assets, coverage_threshold=0.75)
        self.logger.info(f"Selected aggressive universe: {len(universe)} assets")

        # Filter data
        returns_filtered = returns_data[universe].fillna(0.0)
        prices_filtered = prices_data[universe].ffill().fillna(method='bfill')
        volume_filtered = volume_data[universe].ffill().fillna(0.0)

        # Create comprehensive feature matrix with more indicators
        feature_list = []

        # Returns (multiple timeframes)
        feature_list.append(returns_filtered.values)  # Daily returns
        feature_list.append(returns_filtered.rolling(5).mean().fillna(0).values)  # Weekly avg
        feature_list.append(returns_filtered.rolling(21).mean().fillna(0).values)  # Monthly avg

        # Volatility (multiple timeframes)
        feature_list.append(returns_filtered.rolling(5).std().fillna(0).values)
        feature_list.append(returns_filtered.rolling(21).std().fillna(0).values)
        feature_list.append(returns_filtered.rolling(63).std().fillna(0).values)

        # Price-based features
        feature_list.append((prices_filtered / prices_filtered.rolling(21).mean()).fillna(1).values)  # Price momentum
        feature_list.append((prices_filtered / prices_filtered.rolling(63).mean()).fillna(1).values)  # Long-term momentum

        # Volume-based features
        volume_ma = volume_filtered.rolling(21).mean()
        feature_list.append((volume_filtered / volume_ma).fillna(1).values)  # Volume ratio

        # Technical indicators
        # RSI-like indicator
        returns_pos = returns_filtered.clip(lower=0)
        returns_neg = (-returns_filtered).clip(lower=0)
        rs = returns_pos.rolling(14).mean() / (returns_neg.rolling(14).mean() + 1e-8)
        rsi = (100 - 100 / (1 + rs)).fillna(50).values
        feature_list.append(rsi / 100)  # Normalize to [0,1]

        # Bollinger Band position
        price_ma = prices_filtered.rolling(20).mean()
        price_std = prices_filtered.rolling(20).std()
        bb_position = ((prices_filtered - price_ma) / (2 * price_std + 1e-8)).fillna(0).values
        feature_list.append(np.clip(bb_position, -2, 2) / 2)  # Normalize to [-1,1]

        # Cross-sectional rank features
        returns_rank = returns_filtered.rank(axis=1, pct=True).fillna(0.5).values
        volume_rank = volume_filtered.rank(axis=1, pct=True).fillna(0.5).values
        feature_list.append(returns_rank)
        feature_list.append(volume_rank)

        # Stack all features
        feature_matrix = np.stack(feature_list, axis=-1)  # (time, assets, features)

        # Create sequences with extended length
        sequences = []
        targets = []
        valid_dates = []

        n_times = len(feature_matrix)

        for i in range(sequence_length, n_times - 1):
            # Input sequence - more comprehensive
            seq_data = feature_matrix[i-sequence_length:i]  # (seq_len, n_assets, n_features)
            sequences.append(seq_data)

            # Target: multi-period forward returns (1, 5, 21 days)
            target_1d = returns_filtered.iloc[i+1].values
            target_5d = returns_filtered.iloc[i+1:i+6].mean().values if i+6 < n_times else target_1d
            target_21d = returns_filtered.iloc[i+1:i+22].mean().values if i+22 < n_times else target_1d

            # Combine targets
            target = np.stack([target_1d, target_5d, target_21d], axis=1)  # (n_assets, 3)
            targets.append(target)

            valid_dates.append(returns_filtered.index[i+1])

        # Convert to tensors
        X = torch.tensor(np.array(sequences), dtype=torch.float32)  # (n_samples, seq_len, n_assets, n_features)
        y = torch.tensor(np.array(targets), dtype=torch.float32)    # (n_samples, n_assets, 3)

        # Reshape for LSTM: (n_samples, seq_len, n_assets * n_features)
        n_samples, seq_len, n_assets_dim, n_feat = X.shape
        X_reshaped = X.reshape(n_samples, seq_len, n_assets_dim * n_feat)

        # Use only 1-day ahead target for now (can be extended)
        y_1d = y[:, :, 0]  # (n_samples, n_assets)

        self.logger.info(f"Created {len(sequences)} aggressive sequences with shape {X_reshaped.shape}")
        self.logger.info(f"Feature dimensions: {n_feat} features per asset")

        return X_reshaped, y_1d, universe, pd.DatetimeIndex(valid_dates)

    def select_aggressive_universe(self, returns_data: pd.DataFrame,
                                 n_assets: int = 400,
                                 coverage_threshold: float = 0.75) -> list[str]:
        """Select aggressive universe with more assets and lower coverage threshold."""
        coverage = returns_data.count() / len(returns_data)
        valid_assets = coverage[coverage >= coverage_threshold].index.tolist()

        if len(valid_assets) < n_assets:
            # Gradually lower threshold
            for threshold in [0.70, 0.65, 0.60, 0.55, 0.50]:
                valid_assets = coverage[coverage >= threshold].index.tolist()
                if len(valid_assets) >= n_assets:
                    break

        # Select top n_assets by coverage
        if len(valid_assets) > n_assets:
            coverage_sorted = coverage[valid_assets].sort_values(ascending=False)
            valid_assets = coverage_sorted.head(n_assets).index.tolist()

        return valid_assets

    def train_aggressive_configuration(self, config: dict[str, Any],
                                     X_train: torch.Tensor, y_train: torch.Tensor,
                                     X_val: torch.Tensor, y_val: torch.Tensor) -> dict[str, Any]:
        """
        Train LSTM with aggressive configuration.

        Args:
            config: Training configuration
            X_train, y_train: Training data
            X_val, y_val: Validation data

        Returns:
            Training results dictionary
        """
        try:
            self.logger.info(f"Training aggressive LSTM: {config['config_id']}")

            # Memory optimization
            memory_info = self.optimize_aggressive_memory(config, X_train.shape)

            # Create enhanced model architecture
            input_size = X_train.shape[-1]
            output_size = y_train.shape[-1]

            model_config = LSTMConfig(
                input_size=input_size,
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                output_size=output_size,
                dropout=config['dropout'],
                use_attention=True,
                attention_heads=config.get('attention_heads', 8),
                use_residual=config.get('use_residual_connections', True),
                use_layer_norm=config.get('use_layer_normalization', True)
            )

            model = LSTMNetwork(model_config).to(self.device)

            # Enhanced optimizer with more aggressive settings
            if config.get('weight_decay', 0) > 0:
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=config['learning_rate'],
                    weight_decay=config['weight_decay'],
                    betas=(0.9, 0.999),
                    eps=1e-8
                )
            else:
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=config['learning_rate'],
                    betas=(0.9, 0.999),
                    eps=1e-8
                )

            # Advanced scheduler
            if config.get('scheduler_type') == 'cosine_annealing':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=config['max_epochs'],
                    eta_min=config['learning_rate'] * 0.01
                )
            elif config.get('scheduler_type') == 'exponential':
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
            else:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.5,
                    patience=config.get('early_stopping_patience', 20) // 3
                )

            # Enhanced loss function with multi-task capability
            criterion = nn.MSELoss()

            # Training loop with aggressive settings
            best_val_loss = float('inf')
            patience_counter = 0
            training_history = []

            for epoch in range(config['max_epochs']):
                # Training
                model.train()
                train_loss = 0.0
                n_batches = 0

                # Use optimal batch size from memory optimization
                batch_size = memory_info['optimal_batch_size']
                grad_accum_steps = config.get('gradient_accumulation_steps', 1)

                for i in range(0, len(X_train), batch_size):
                    batch_X = X_train[i:i+batch_size].to(self.device)
                    batch_y = y_train[i:i+batch_size].to(self.device)

                    if self.use_mixed_precision:
                        with torch.amp.autocast('cuda'):
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y) / grad_accum_steps

                        self.scaler.scale(loss).backward()

                        if (n_batches + 1) % grad_accum_steps == 0:
                            if config.get('use_gradient_clipping', True):
                                self.scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(
                                    model.parameters(),
                                    config.get('gradient_clip_value', 1.0)
                                )
                            self.scaler.step(optimizer)
                            self.scaler.update()
                            optimizer.zero_grad()
                    else:
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y) / grad_accum_steps
                        loss.backward()

                        if (n_batches + 1) % grad_accum_steps == 0:
                            if config.get('use_gradient_clipping', True):
                                torch.nn.utils.clip_grad_norm_(
                                    model.parameters(),
                                    config.get('gradient_clip_value', 1.0)
                                )
                            optimizer.step()
                            optimizer.zero_grad()

                    train_loss += loss.item() * grad_accum_steps
                    n_batches += 1

                # Validation
                model.eval()
                val_loss = 0.0
                val_batches = 0

                with torch.no_grad():
                    for i in range(0, len(X_val), batch_size):
                        batch_X = X_val[i:i+batch_size].to(self.device)
                        batch_y = y_val[i:i+batch_size].to(self.device)

                        if self.use_mixed_precision:
                            with torch.amp.autocast('cuda'):
                                outputs = model(batch_X)
                                loss = criterion(outputs, batch_y)
                        else:
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y)

                        val_loss += loss.item()
                        val_batches += 1

                train_loss /= n_batches
                val_loss /= val_batches

                # Learning rate scheduling
                if config.get('scheduler_type') == 'reduce_on_plateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                training_history.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'lr': optimizer.param_groups[0]['lr']
                })

                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}/{config['max_epochs']}: "
                                   f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

                if patience_counter >= config.get('early_stopping_patience', 25):
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break

            # Calculate final validation metrics
            final_metrics = self.calculate_validation_metrics(
                model, X_val, y_val, criterion
            )

            result = {
                'config': config,
                'status': 'success',
                'best_val_loss': best_val_loss,
                'final_epoch': epoch,
                'training_history': training_history,
                'validation_metrics': final_metrics,
                'memory_info': memory_info,
                'model_parameters': sum(p.numel() for p in model.parameters()),
                'training_timestamp': datetime.now().isoformat()
            }

            self.logger.info(f"Successfully trained {config['config_id']}: "
                           f"Best Val Loss={best_val_loss:.6f}, "
                           f"Epochs={epoch}, Params={result['model_parameters']:,}")

            return result

        except Exception as e:
            self.logger.error(f"Training failed for {config['config_id']}: {str(e)}")
            return {
                'config': config,
                'status': 'failed',
                'error': str(e),
                'training_timestamp': datetime.now().isoformat()
            }

    def optimize_aggressive_memory(self, config: dict[str, Any],
                                 input_shape: tuple) -> dict[str, Any]:
        """Optimize memory settings for aggressive configurations."""

        # Estimate model memory requirements
        n_params = (
            config['hidden_size'] ** 2 * config['num_layers'] * 4 +  # LSTM weights
            config['hidden_size'] * config.get('attention_heads', 8) * 64 * 2 +  # Attention
            input_shape[-1] * config['hidden_size'] +  # Input projection
            config['hidden_size'] * input_shape[-1]   # Output projection
        )

        param_memory_mb = n_params * 4 / (1024 ** 2)  # 4 bytes per float32

        # Account for activations and gradients
        total_memory_mb = param_memory_mb * 4  # Conservative estimate

        # Optimize batch size
        available_memory_mb = 10000  # Conservative 10GB limit
        max_batch_memory = available_memory_mb - total_memory_mb

        sequence_memory_per_sample = (
            config.get('sequence_length', 120) *
            input_shape[-1] * 4 / (1024 ** 2)
        )

        optimal_batch_size = max(1, int(max_batch_memory / (sequence_memory_per_sample * 2)))
        optimal_batch_size = min(optimal_batch_size, config.get('batch_size', 64))

        return {
            'estimated_param_memory_mb': param_memory_mb,
            'estimated_total_memory_mb': total_memory_mb,
            'optimal_batch_size': optimal_batch_size,
            'gradient_accumulation_needed': config.get('batch_size', 64) > optimal_batch_size
        }

    def calculate_validation_metrics(self, model: nn.Module,
                                   X_val: torch.Tensor,
                                   y_val: torch.Tensor,
                                   criterion: nn.Module) -> dict[str, float]:
        """Calculate comprehensive validation metrics."""
        model.eval()

        with torch.no_grad():
            if self.use_mixed_precision:
                with torch.amp.autocast('cuda'):
                    predictions = model(X_val.to(self.device))
            else:
                predictions = model(X_val.to(self.device))

            # Move to CPU for calculations
            predictions = predictions.cpu()

            # Loss
            val_loss = criterion(predictions, y_val).item()

            # Correlation metrics
            pred_flat = predictions.flatten().numpy()
            true_flat = y_val.flatten().numpy()

            correlation = np.corrcoef(pred_flat, true_flat)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0

            # Directional accuracy
            pred_signs = np.sign(pred_flat)
            true_signs = np.sign(true_flat)
            directional_accuracy = np.mean(pred_signs == true_signs)

            # MSE and MAE
            mse = np.mean((pred_flat - true_flat) ** 2)
            mae = np.mean(np.abs(pred_flat - true_flat))

        return {
            'validation_loss': val_loss,
            'correlation': correlation,
            'directional_accuracy': directional_accuracy,
            'mse': mse,
            'mae': mae
        }

    def run_aggressive_training(self) -> None:
        """Run aggressive LSTM training pipeline."""

        self.logger.info("Starting Aggressive LSTM Training Pipeline")
        self.logger.info("="*80)

        try:
            # Load data
            self.logger.info("Loading production datasets...")
            returns_data = self.data_manager.load_returns()
            prices_data = self.data_manager.load_prices()
            volume_data = self.data_manager.load_volume()

            self.logger.info(f"Loaded data shapes: Returns {returns_data.shape}, "
                           f"Prices {prices_data.shape}, Volume {volume_data.shape}")

        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            return

        # Generate configurations
        configs = self.generate_aggressive_configs()

        results = []
        successful = 0
        failed = 0

        for i, config in enumerate(configs, 1):
            self.logger.info(f"\nProcessing configuration {i}/{len(configs)}: {config['config_id']}")

            try:
                # Prepare sequences with config-specific parameters
                X, y, universe, dates = self.prepare_aggressive_sequences(
                    returns_data, prices_data, volume_data,
                    sequence_length=config.get('sequence_length', 120),
                    n_assets=400
                )

                # Train/validation split (70/30)
                split_idx = int(len(X) * 0.7)
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]

                # Train configuration
                result = self.train_aggressive_configuration(
                    config, X_train, y_train, X_val, y_val
                )

                results.append(result)

                if result['status'] == 'success':
                    successful += 1
                else:
                    failed += 1

            except Exception as e:
                self.logger.error(f"Failed to process config {config['config_id']}: {e}")
                failed += 1
                results.append({
                    'config': config,
                    'status': 'failed',
                    'error': str(e)
                })

        # Save results
        self.save_results(results)

        # Summary
        self.logger.info("\n" + "="*80)
        self.logger.info("Aggressive LSTM Training Completed!")
        self.logger.info(f"Successful: {successful}, Failed: {failed}")

        if successful > 0:
            successful_results = [r for r in results if r['status'] == 'success']
            best_result = min(successful_results,
                            key=lambda x: x['best_val_loss'])

            self.logger.info(f"Best configuration: {best_result['config']['config_id']}")
            self.logger.info(f"Best validation loss: {best_result['best_val_loss']:.6f}")

    def save_results(self, results: list[dict]) -> None:
        """Save training results."""
        results_dir = Path("data/models/checkpoints/lstm_aggressive")
        results_dir.mkdir(parents=True, exist_ok=True)

        logs_dir = Path("logs/training/lstm_aggressive")

        results_file = logs_dir / "lstm_aggressive_results.yaml"
        with open(results_file, 'w') as f:
            yaml.dump({
                'training_summary': {
                    'total_configs': len(results),
                    'successful_configs': sum(1 for r in results if r['status'] == 'success'),
                    'failed_configs': sum(1 for r in results if r['status'] == 'failed'),
                    'training_completed': datetime.now().isoformat()
                },
                'results': results
            }, f, default_flow_style=False)

        self.logger.info(f"Results saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(description='Aggressive LSTM Training Pipeline')
    parser.add_argument('--full-training', action='store_true',
                       help='Run full aggressive training')

    parser.parse_args()

    trainer = AggressiveLSTMTraining()
    trainer.run_aggressive_training()


if __name__ == "__main__":
    main()
