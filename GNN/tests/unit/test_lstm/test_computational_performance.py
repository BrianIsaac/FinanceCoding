"""
Computational performance tests for LSTM model.

This module tests LSTM training and inference performance to ensure
it meets computational constraints and time requirements.
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import pytest
import torch

from src.models.base.portfolio_model import PortfolioConstraints
from src.models.lstm.model import LSTMModelConfig, LSTMPortfolioModel
from src.models.lstm.training import MemoryEfficientTrainer


class TestLSTMComputationalPerformance:
    """Test LSTM computational performance and constraints."""

    @pytest.fixture
    def performance_data(self) -> pd.DataFrame:
        """Generate realistic-sized data for performance testing."""
        np.random.seed(42)
        torch.manual_seed(42)

        # S&P MidCap 400 sized universe over 5 years
        dates = pd.date_range("2019-01-01", "2024-01-01", freq="B")  # ~1300 trading days
        n_assets = 400

        # Generate returns with factor structure for realism
        market_factor = np.random.normal(0.0005, 0.012, len(dates))
        returns = np.random.multivariate_normal(
            mean=np.zeros(n_assets),
            cov=0.015**2 * (0.2 * np.ones((n_assets, n_assets)) + 0.8 * np.eye(n_assets)),
            size=len(dates),
        )

        # Add market factor exposure
        betas = 0.5 + 0.8 * np.random.random(n_assets)  # Beta between 0.5 and 1.3
        for i in range(n_assets):
            returns[:, i] += betas[i] * market_factor

        assets = [f"MIDCAP_{i:03d}" for i in range(n_assets)]
        return pd.DataFrame(returns, index=dates, columns=assets)

    @pytest.fixture
    def memory_constrained_config(self) -> LSTMModelConfig:
        """Configuration optimized for memory constraints."""
        config = LSTMModelConfig()

        # Reasonable settings for 12GB GPU
        config.lstm_config.hidden_size = 128
        config.lstm_config.num_layers = 2
        config.lstm_config.dropout = 0.3

        # Memory optimization settings
        config.training_config.batch_size = 32
        config.training_config.max_memory_gb = 10.0  # Conservative limit
        config.training_config.use_mixed_precision = True
        config.training_config.gradient_accumulation_steps = 2

        # Reasonable training duration
        config.training_config.epochs = 20
        config.training_config.patience = 5

        return config

    @pytest.fixture
    def constraints(self) -> PortfolioConstraints:
        """Standard portfolio constraints."""
        return PortfolioConstraints(
            long_only=True, top_k_positions=50, max_position_weight=0.08, max_monthly_turnover=0.20
        )

    def test_memory_usage_estimation(self, memory_constrained_config: LSTMModelConfig):
        """Test LSTM memory usage estimation accuracy."""
        config = memory_constrained_config
        trainer = MemoryEfficientTrainer(config.training_config, None, None, None)

        # Test different batch sizes
        sequence_length = 60
        n_assets = 400

        batch_sizes = [8, 16, 32, 64]
        estimated_memory = {}

        for batch_size in batch_sizes:
            memory_gb = trainer.estimate_memory_usage(batch_size, sequence_length, n_assets)
            estimated_memory[batch_size] = memory_gb

            # Memory should increase with batch size
            assert memory_gb > 0, f"Memory estimate should be positive for batch_size={batch_size}"
            assert memory_gb < 50.0, f"Memory estimate {memory_gb}GB seems unrealistic"

        # Memory should scale roughly with batch size
        for i in range(1, len(batch_sizes)):
            prev_batch = batch_sizes[i - 1]
            curr_batch = batch_sizes[i]

            prev_memory = estimated_memory[prev_batch]
            curr_memory = estimated_memory[curr_batch]

            assert (
                curr_memory > prev_memory
            ), f"Memory should increase with batch size: {prev_memory} -> {curr_memory}"

            # Should roughly double (with some overhead)
            ratio = curr_memory / prev_memory
            assert 1.5 < ratio < 3.0, f"Memory scaling ratio {ratio:.2f} seems unrealistic"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_gpu_memory_constraints(
        self, constraints: PortfolioConstraints, memory_constrained_config: LSTMModelConfig
    ):
        """Test LSTM respects GPU memory constraints."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Clear GPU memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()

        model = LSTMPortfolioModel(constraints=constraints, config=memory_constrained_config)

        # Generate small dataset for memory testing
        dates = pd.date_range("2023-01-01", "2023-06-30", freq="B")
        n_assets = 200
        returns = pd.DataFrame(
            np.random.normal(0, 0.02, (len(dates), n_assets)),
            index=dates,
            columns=[f"ASSET_{i:03d}" for i in range(n_assets)],
        )

        try:
            # This should not exceed memory limits
            model.fit(
                returns=returns,
                universe=returns.columns.tolist(),
                fit_period=(returns.index[0], returns.index[-1]),
            )

            # Check final memory usage
            peak_memory = torch.cuda.max_memory_allocated()
            memory_used_gb = (peak_memory - initial_memory) / (1024**3)

            # Should respect the configured limit with some tolerance
            memory_limit = memory_constrained_config.training_config.max_memory_gb
            assert (
                memory_used_gb <= memory_limit + 1.0
            ), f"Memory usage {memory_used_gb:.2f}GB exceeded limit {memory_limit}GB"

        finally:
            # Cleanup
            torch.cuda.empty_cache()

    def test_training_time_performance(
        self, performance_data: pd.DataFrame, constraints: PortfolioConstraints
    ):
        """Test LSTM training time is reasonable for production use."""
        # Use smaller dataset and faster config for timing test
        subset_data = performance_data.iloc[:500, :100]  # 500 days, 100 assets

        fast_config = LSTMModelConfig()
        fast_config.lstm_config.hidden_size = 64
        fast_config.lstm_config.num_layers = 1
        fast_config.training_config.epochs = 10
        fast_config.training_config.batch_size = 16
        fast_config.training_config.patience = 3

        model = LSTMPortfolioModel(constraints=constraints, config=fast_config)

        # Time the training process
        start_time = time.time()

        model.fit(
            returns=subset_data,
            universe=subset_data.columns.tolist(),
            fit_period=(subset_data.index[0], subset_data.index[-1]),
        )

        training_time = time.time() - start_time

        # Should complete training in reasonable time
        # Allow up to 5 minutes for this test configuration
        assert training_time < 300, f"Training took too long: {training_time:.2f}s"

        # Should be faster than 1 second per day of data (very conservative)
        assert training_time < len(
            subset_data
        ), f"Training too slow: {training_time/len(subset_data):.3f}s per day"

    def test_inference_speed(
        self,
        performance_data: pd.DataFrame,
        constraints: PortfolioConstraints,
        memory_constrained_config: LSTMModelConfig,
    ):
        """Test LSTM inference speed for real-time portfolio rebalancing."""
        # Use reduced data for faster testing
        subset_data = performance_data.iloc[:300, :200]  # 300 days, 200 assets

        # Quick training config
        config = memory_constrained_config
        config.training_config.epochs = 5
        config.training_config.patience = 2

        model = LSTMPortfolioModel(constraints=constraints, config=config)

        # Train model
        train_end = subset_data.index[-60]  # Leave 60 days for inference testing
        train_data = subset_data.loc[:train_end]

        model.fit(
            returns=train_data,
            universe=train_data.columns.tolist(),
            fit_period=(train_data.index[0], train_end),
        )

        # Test inference speed on multiple dates
        inference_dates = subset_data.index[-30:]  # Last 30 days
        inference_times = []

        for date in inference_dates:
            start_time = time.time()

            weights = model.predict_weights(date=date, universe=subset_data.columns.tolist())

            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            # Verify we got valid weights
            assert isinstance(weights, pd.Series), "Should return pandas Series"
            assert len(weights) > 0, "Should return non-empty weights"
            assert not weights.isna().all(), "Should not return all NaN weights"

        avg_inference_time = np.mean(inference_times)
        max_inference_time = np.max(inference_times)

        # Should be fast enough for real-time use
        assert avg_inference_time < 1.0, f"Average inference too slow: {avg_inference_time:.4f}s"
        assert max_inference_time < 2.0, f"Maximum inference too slow: {max_inference_time:.4f}s"

        # Should be consistent (no huge outliers)
        inference_std = np.std(inference_times)
        assert inference_std < 0.5, f"Inference times too variable: std={inference_std:.4f}s"

    def test_memory_efficiency_during_backtesting(
        self, performance_data: pd.DataFrame, constraints: PortfolioConstraints
    ):
        """Test memory efficiency during simulated backtesting scenario."""
        # Monitor system memory during multiple fit/predict cycles
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        memory_samples = [initial_memory]

        # Fast config for testing
        config = LSTMModelConfig()
        config.lstm_config.hidden_size = 32
        config.training_config.epochs = 3
        config.training_config.batch_size = 8

        model = LSTMPortfolioModel(constraints=constraints, config=config)

        # Simulate backtesting with multiple retraining periods
        subset_data = performance_data.iloc[:400, :50]  # Smaller data for testing
        n_rebalances = 5

        for i in range(n_rebalances):
            start_idx = i * 60
            end_idx = start_idx + 200

            if end_idx >= len(subset_data):
                break

            window_data = subset_data.iloc[start_idx:end_idx]

            # Retrain model
            model.fit(
                returns=window_data,
                universe=window_data.columns.tolist(),
                fit_period=(window_data.index[0], window_data.index[-1]),
            )

            # Generate predictions
            for j in range(3):  # Multiple predictions per rebalance
                pred_date = window_data.index[-10 + j]
                model.predict_weights(date=pred_date, universe=window_data.columns.tolist())

                # Sample memory usage
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)

        final_memory = process.memory_info().rss / 1024 / 1024
        peak_memory = max(memory_samples)
        memory_growth = final_memory - initial_memory

        # Should not have excessive memory growth (some growth expected due to caching)
        assert memory_growth < 500, f"Excessive memory growth: {memory_growth:.1f}MB"

        # Peak memory should be reasonable
        assert peak_memory < initial_memory + 1000, f"Peak memory too high: {peak_memory:.1f}MB"

    def test_scalability_with_universe_size(self, constraints: PortfolioConstraints):
        """Test how performance scales with universe size."""
        universe_sizes = [50, 100, 200, 400]
        training_times = {}
        inference_times = {}

        for n_assets in universe_sizes:

            # Generate data for this universe size
            dates = pd.date_range("2023-01-01", "2023-06-30", freq="B")
            returns = pd.DataFrame(
                np.random.normal(0, 0.02, (len(dates), n_assets)),
                index=dates,
                columns=[f"ASSET_{i:03d}" for i in range(n_assets)],
            )

            # Fast config scaled for universe size
            config = LSTMModelConfig()
            config.lstm_config.hidden_size = min(64, max(32, n_assets // 8))  # Scale hidden size
            config.training_config.epochs = 5
            config.training_config.batch_size = max(
                8, min(32, 1000 // n_assets)
            )  # Scale batch size

            model = LSTMPortfolioModel(constraints=constraints, config=config)

            # Time training
            start_time = time.time()
            model.fit(
                returns=returns,
                universe=returns.columns.tolist(),
                fit_period=(returns.index[0], returns.index[-1]),
            )
            training_times[n_assets] = time.time() - start_time

            # Time inference
            start_time = time.time()
            model.predict_weights(date=returns.index[-1], universe=returns.columns.tolist())
            inference_times[n_assets] = time.time() - start_time

        # Analyze scaling behavior
        for i in range(1, len(universe_sizes)):
            prev_size = universe_sizes[i - 1]
            curr_size = universe_sizes[i]

            train_ratio = training_times[curr_size] / training_times[prev_size]
            inference_ratio = inference_times[curr_size] / inference_times[prev_size]

            size_ratio = curr_size / prev_size

            # Scaling should be reasonable (not exponential)
            assert (
                train_ratio < size_ratio**1.5
            ), f"Training scaling too poor: {train_ratio:.2f}x for {size_ratio:.1f}x size increase"
            assert (
                inference_ratio < size_ratio**1.2
            ), (f"Inference scaling too poor: {inference_ratio:.2f}x for "
               f"{size_ratio:.1f}x size increase")

        # All configurations should complete in reasonable time
        max_training_time = max(training_times.values())
        max_inference_time = max(inference_times.values())

        assert (
            max_training_time < 180
        ), f"Training too slow for largest universe: {max_training_time:.2f}s"
        assert (
            max_inference_time < 1.0
        ), f"Inference too slow for largest universe: {max_inference_time:.4f}s"
