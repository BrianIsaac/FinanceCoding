"""
Unit tests for GPU utilities module.

Tests GPU memory management, configuration, and utility functions.
"""

import pytest
import torch

from src.utils.gpu import GPUConfig, GPUMemoryManager


class TestGPUConfig:
    """Test GPU configuration dataclass."""

    def test_default_config(self):
        """Test default GPU configuration values."""
        config = GPUConfig()
        
        assert config.max_memory_gb == 11.0
        assert config.enable_mixed_precision is True
        assert config.gradient_checkpointing is False
        assert config.batch_size_auto_scale is True

    def test_custom_config(self):
        """Test custom GPU configuration values."""
        config = GPUConfig(
            max_memory_gb=8.0,
            enable_mixed_precision=False,
            gradient_checkpointing=True,
            batch_size_auto_scale=False
        )
        
        assert config.max_memory_gb == 8.0
        assert config.enable_mixed_precision is False
        assert config.gradient_checkpointing is True
        assert config.batch_size_auto_scale is False


class TestGPUMemoryManager:
    """Test GPU memory manager functionality."""

    def test_init_cpu_mode(self):
        """Test GPU manager initialization in CPU mode."""
        config = GPUConfig()
        manager = GPUMemoryManager(config)
        
        assert manager.config == config
        # Device should be cuda if available, cpu otherwise
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        assert manager.device.type == expected_device

    def test_is_gpu_available(self):
        """Test GPU availability detection."""
        config = GPUConfig()
        manager = GPUMemoryManager(config)
        
        # Should match torch.cuda.is_available()
        assert manager.is_gpu_available() == torch.cuda.is_available()

    def test_get_optimal_batch_size(self):
        """Test optimal batch size determination."""
        config = GPUConfig()
        manager = GPUMemoryManager(config)
        
        # Mock model and input shape
        model = torch.nn.Linear(10, 1)
        input_shape = (10,)
        
        batch_size = manager.get_optimal_batch_size(model, input_shape)
        
        # Should return conservative default for now
        assert batch_size == 32
        assert isinstance(batch_size, int)

    @pytest.mark.gpu
    def test_setup_mixed_precision_gpu(self, gpu_available):
        """Test mixed precision setup on GPU."""
        if not gpu_available:
            pytest.skip("GPU not available")
        
        config = GPUConfig(enable_mixed_precision=True)
        manager = GPUMemoryManager(config)
        
        scaler = manager.setup_mixed_precision()
        assert scaler is not None
        assert isinstance(scaler, torch.amp.GradScaler)

    def test_setup_mixed_precision_cpu(self):
        """Test mixed precision setup on CPU (should return None)."""
        config = GPUConfig(enable_mixed_precision=True)
        manager = GPUMemoryManager(config)
        
        # Force CPU mode for this test
        manager.device = torch.device("cpu")
        
        scaler = manager.setup_mixed_precision()
        assert scaler is None

    @pytest.mark.gpu
    def test_get_memory_stats_gpu(self, gpu_available):
        """Test GPU memory statistics retrieval."""
        if not gpu_available:
            pytest.skip("GPU not available")
        
        config = GPUConfig()
        manager = GPUMemoryManager(config)
        
        stats = manager.get_memory_stats()
        
        # Check required keys exist
        required_keys = ["allocated_gb", "cached_gb", "free_gb", "total_gb", "utilization_pct"]
        for key in required_keys:
            assert key in stats
        
        # Check reasonable values
        assert stats["total_gb"] > 0
        assert 0 <= stats["utilization_pct"] <= 100
        assert stats["free_gb"] >= 0

    def test_get_memory_stats_cpu(self):
        """Test memory statistics on CPU."""
        config = GPUConfig()
        manager = GPUMemoryManager(config)
        
        # Force CPU mode
        manager.device = torch.device("cpu")
        
        stats = manager.get_memory_stats()
        assert "status" in stats
        assert stats["status"] == "CPU mode - no GPU stats available"

    @pytest.mark.gpu
    def test_clear_cache_gpu(self, gpu_available):
        """Test GPU cache clearing."""
        if not gpu_available:
            pytest.skip("GPU not available")
        
        config = GPUConfig()
        manager = GPUMemoryManager(config)
        
        # This should not raise an exception
        manager.clear_cache()

    def test_clear_cache_cpu(self):
        """Test cache clearing on CPU (should be no-op)."""
        config = GPUConfig()
        manager = GPUMemoryManager(config)
        
        # Force CPU mode
        manager.device = torch.device("cpu")
        
        # This should not raise an exception
        manager.clear_cache()

    @pytest.mark.gpu
    def test_configure_memory_limit_gpu(self, gpu_available):
        """Test GPU memory limit configuration."""
        if not gpu_available:
            pytest.skip("GPU not available")
        
        config = GPUConfig(max_memory_gb=8.0)
        manager = GPUMemoryManager(config)
        
        # This should not raise an exception
        manager.configure_memory_limit()