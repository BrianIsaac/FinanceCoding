# Testing and Quality Assurance

## Comprehensive Testing Strategy

```python
# tests/unit/test_models/test_gat.py
import pytest
import torch
import pandas as pd
from src.models.gat.model import GATPortfolioModel

class TestGATModel:
    @pytest.fixture
    def sample_data(self):
        n_assets, n_features = 50, 10
        node_features = torch.randn(n_assets, n_features)
        edge_index = torch.randint(0, n_assets, (2, 100))
        edge_attr = torch.randn(100, 3)
        asset_mask = torch.ones(n_assets, dtype=torch.bool)
        
        return {
            "node_features": node_features,
            "edge_index": edge_index, 
            "edge_attr": edge_attr,
            "asset_mask": asset_mask
        }
    
    def test_forward_pass(self, sample_data):
        """Test basic forward pass functionality."""
        model = GATPortfolioModel(
            input_features=10,
            hidden_dim=32,
            num_layers=2
        )
        
        weights = model(**sample_data)
        
        # Verify output properties
        assert weights.shape[0] == sample_data["node_features"].shape[0]
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
        assert (weights >= 0).all()
    
    def test_memory_efficiency(self, sample_data):
        """Test GPU memory usage within limits.""" 
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        model = GATPortfolioModel(input_features=10).cuda()
        
        # Measure memory before and after forward pass
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
        
        with torch.no_grad():
            weights = model(**{k: v.cuda() for k, v in sample_data.items()})
        
        peak_memory = torch.cuda.max_memory_allocated()
        memory_used_gb = (peak_memory - initial_memory) / 1024**3
        
        assert memory_used_gb < 2.0, f"Memory usage {memory_used_gb:.2f}GB exceeds limit"

# tests/integration/test_pipeline/test_end_to_end.py  
class TestEndToEndPipeline:
    def test_full_pipeline_execution(self, sample_config):
        """Test complete pipeline from data to results."""
        orchestrator = ExperimentOrchestrator(sample_config)
        results = orchestrator.run_full_experiment()
        
        # Verify results structure
        assert "backtest_results" in results
        assert "analysis_results" in results
        
        # Verify performance metrics are calculated
        for model_name in sample_config.models:
            assert model_name in results.analysis_results
            metrics = results.analysis_results[model_name]
            assert "sharpe_ratio" in metrics
            assert "max_drawdown" in metrics
```

---
