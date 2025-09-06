# Development Guide - Epic 1 ML Workflows

## Epic 1 Development Environment

Epic 1 builds upon the foundation established in Epic 0 (Stories 0.1-0.3) with enhanced ML development capabilities and configuration management.

### Prerequisites

Epic 0 completed the following foundation:
- ✅ Python environment with uv package manager
- ✅ CUDA 12.8 and PyTorch 2.8.0+cu128 installation  
- ✅ Repository structure and module organization
- ✅ Testing framework and code quality tools
- ✅ GPU optimization utilities

Story 1.1 adds:
- ✅ Configuration management system
- ✅ ML-optimized directory structure
- ✅ YAML-based configuration files
- ✅ Epic 1 development workflows

## Configuration Management

### Configuration System Architecture

The configuration system uses a hierarchical approach:

```
configs/
├── data/                    # Data pipeline configurations
│   ├── default.yaml         # Base data settings
│   └── midcap400.yaml      # S&P MidCap 400 specific
├── models/                  # Model configurations
│   ├── hrp_default.yaml    # Hierarchical Risk Parity
│   ├── lstm_default.yaml   # LSTM time series model
│   └── gat_default.yaml    # Graph Attention Network
└── experiments/             # Experiment workflows
    └── baseline_comparison.yaml
```

### Configuration Usage Patterns

#### Loading Configurations

```python
from src.config.base import load_config, merge_configs
from src.config.models import get_model_config

# Load YAML configuration
config_dict = load_config('configs/models/gat_default.yaml')

# Create Python configuration object
gat_config = get_model_config('gat', config_dict['architecture'])

# Merge multiple configurations
base_config = load_config('configs/data/default.yaml')
override_config = {'learning_rate': 0.01}
merged_config = merge_configs(base_config, override_config)
```

#### Configuration Inheritance

```python
from src.config.base import ProjectConfig, DataConfig
from src.config.models import GATConfig
from src.config.data import DataPipelineConfig

# Create configurations with defaults
project_config = ProjectConfig(gpu_memory_fraction=0.8)
data_config = DataPipelineConfig(num_workers=8)
model_config = GATConfig(hidden_dim=128, num_heads=12)
```

### Model Development Workflow

#### 1. Configuration Setup

```python
# Load model configuration
model_config_path = 'configs/models/gat_default.yaml'
config_dict = load_config(model_config_path)
model_config = get_model_config('gat', config_dict)

# Load data configuration
data_config = load_config('configs/data/midcap400.yaml')
```

#### 2. Data Pipeline Integration

```python
from src.data.loaders.portfolio_data import PortfolioDataLoader
from src.data.processors.features import FeatureProcessor

# Initialize data pipeline
data_loader = PortfolioDataLoader(data_config)
feature_processor = FeatureProcessor(data_config['features'])

# Load and process data
returns_data = data_loader.load_returns(
    start_date=data_config['start_date'],
    end_date=data_config['end_date']
)
features = feature_processor.compute_features(returns_data)
```

#### 3. Model Training

```python
from src.models.gat.model import GATModel
from src.utils.gpu import GPUMemoryManager, GPUConfig

# Setup GPU environment
gpu_config = GPUConfig(max_memory_gb=10.0)
gpu_manager = GPUMemoryManager(gpu_config)

# Initialize and train model
model = GATModel(model_config)
model.to(gpu_manager.device)

# Training loop with configuration
training_results = model.fit(
    train_data=features,
    validation_split=model_config.validation_split,
    max_epochs=model_config.max_epochs,
    batch_size=model_config.batch_size
)
```

#### 4. Model Evaluation

```python
from src.evaluation.backtest.engine import BacktestEngine
from src.evaluation.metrics.portfolio_metrics import PortfolioMetrics

# Setup evaluation
backtest_engine = BacktestEngine(config_dict['evaluation'])
metrics_calculator = PortfolioMetrics()

# Run backtest
backtest_results = backtest_engine.run_backtest(
    model=model,
    returns_data=returns_data,
    start_date=data_config['test_start'],
    end_date=data_config['test_end']
)

# Calculate performance metrics
performance_metrics = metrics_calculator.calculate_metrics(backtest_results)
```

## Development Scripts

### Training Script Pattern

```python
#!/usr/bin/env python3
"""Training script for portfolio optimization models."""

import argparse
from pathlib import Path
from src.config.base import load_config
from src.config.models import get_model_config

def main():
    parser = argparse.ArgumentParser(description='Train portfolio model')
    parser.add_argument('--model-config', type=str, required=True,
                       help='Path to model configuration file')
    parser.add_argument('--data-config', type=str, required=True,
                       help='Path to data configuration file')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load configurations
    model_config_dict = load_config(args.model_config)
    data_config = load_config(args.data_config)
    
    # Create model configuration object
    model_type = model_config_dict['model_type']
    model_config = get_model_config(model_type, model_config_dict)
    
    # Training logic here...
    
if __name__ == '__main__':
    main()
```

### Experiment Script Pattern

```python
#!/usr/bin/env python3
"""Experiment runner for baseline comparison."""

import yaml
from src.config.base import load_config
from src.models import get_model_class

def run_experiment(config_path: str):
    """Run baseline comparison experiment."""
    exp_config = load_config(config_path)
    
    # Initialize models
    models = {}
    for model_name, model_settings in exp_config['models'].items():
        if model_settings['enabled']:
            model_config = load_config(model_settings['config_file'])
            model_class = get_model_class(model_name)
            
            for variant in model_settings['variants']:
                variant_config = merge_configs(
                    model_config, 
                    variant.get('overrides', {})
                )
                models[variant['name']] = model_class(variant_config)
    
    # Run training and evaluation for all models...
```

## Import Patterns

Epic 1 uses the following import structure established in Epic 0:

```python
# Configuration imports
from src.config.base import ProjectConfig, load_config
from src.config.models import GATConfig, get_model_config
from src.config.data import DataPipelineConfig

# Model imports
from src.models.gat.model import GATModel
from src.models.hrp import HRPModel  
from src.models.lstm.model import LSTMModel

# Data pipeline imports
from src.data.loaders.portfolio_data import PortfolioDataLoader
from src.data.processors.features import FeatureProcessor
from src.data.collectors.wikipedia import WikipediaCollector

# Evaluation imports
from src.evaluation.backtest.engine import BacktestEngine
from src.evaluation.metrics.portfolio_metrics import PortfolioMetrics
from src.evaluation.reporting import ReportGenerator

# Utility imports
from src.utils.gpu import GPUMemoryManager, GPUConfig
from src.utils.io import save_results, load_results
```

### Package Structure
```
src/
├── config/          # Configuration management
├── data/           # Data processing pipeline  
├── models/         # ML model implementations
├── evaluation/     # Backtesting and evaluation
└── utils/          # Shared utilities

configs/
├── data/           # Data configurations
├── models/         # Model configurations
└── experiments/    # Experiment configurations
```

## Command Line Tools

### Development Commands

```bash
# Train single model
python scripts/train_model.py \
  --model-config configs/models/gat_default.yaml \
  --data-config configs/data/midcap400.yaml \
  --output-dir outputs/gat_experiment

# Run experiment
python scripts/run_experiment.py \
  --experiment-config configs/experiments/baseline_comparison.yaml

# Validate configuration
python scripts/validate_config.py configs/models/gat_default.yaml

# Run backtest
python scripts/run_backtest.py \
  --model-path outputs/gat_experiment/model.pkl \
  --data-config configs/data/midcap400.yaml \
  --start-date 2023-01-01 \
  --end-date 2024-12-31
```

### Environment Commands

```bash
# Install Epic 1 dependencies
uv sync

# Run tests
pytest tests/ -v

# Check code quality
ruff check src/
mypy src/

# GPU memory check
python -c "from src.utils.gpu import GPUMemoryManager; print(GPUMemoryManager().get_memory_stats())"

# Running with uv
uv run python your_script.py
```

## Testing Integration

### Configuration Testing

```python
def test_model_config_loading():
    """Test model configuration loading and validation."""
    config = load_config('configs/models/gat_default.yaml')
    gat_config = get_model_config('gat', config)
    
    assert isinstance(gat_config, GATConfig)
    assert gat_config.hidden_dim > 0
    assert gat_config.num_heads > 0
```

### Integration Testing

```python
def test_end_to_end_workflow():
    """Test complete model training workflow."""
    # Load test configurations
    data_config = load_config('configs/data/default.yaml')
    model_config = load_config('configs/models/gat_default.yaml')
    
    # Run abbreviated training
    model = GATModel(get_model_config('gat', model_config))
    # ... test implementation
```

## Best Practices

### 1. Configuration Management
- Always use YAML files for configuration
- Create configuration objects from YAML for type safety
- Use configuration inheritance and merging for variants
- Validate configurations before training

### 2. Model Development
- Follow the PortfolioModel interface pattern
- Use configuration classes for all hyperparameters
- Implement proper GPU memory management
- Add comprehensive docstrings and type hints

### 3. Data Pipeline
- Use configuration-driven data processing
- Implement caching for expensive operations
- Follow feature naming conventions
- Validate data quality at each step

### 4. Experimentation
- Define experiments in YAML configuration files
- Use consistent metrics across all models
- Save intermediate results for analysis
- Document experiment hypotheses and results

### 5. Code Quality
- Follow Google-style docstrings with ML-specific patterns
- Use type hints for all functions and methods
- Write tests for configuration loading and model training
- Run linting and type checking before commits