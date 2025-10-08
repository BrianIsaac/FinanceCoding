# Developer Setup Guide
## Portfolio Optimization with Machine Learning Techniques

**Version:** 1.0  
**Date:** September 5, 2025  
**Environment:** Linux (RTX GeForce 5070Ti, 12GB VRAM)

---

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **GPU**: RTX GeForce 5070Ti (12GB VRAM) with CUDA 11.8+
- **RAM**: 32GB+ recommended for data processing
- **Storage**: 50GB+ free space for datasets and models
- **Python**: 3.9+ (will be managed by uv)

### Required System Packages
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
    git \
    curl \
    build-essential \
    python3-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    pkg-config

# NVIDIA drivers and CUDA (if not already installed)
# Check existing installation
nvidia-smi
nvcc --version

# If CUDA not installed, follow NVIDIA CUDA installation guide
```

## Step 1: Install uv Package Manager

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Reload shell or add to PATH
source ~/.bashrc
# OR
export PATH="$HOME/.local/bin:$PATH"

# Verify installation
uv --version
```

## Step 2: Project Repository Setup

### Clone/Initialize Repository
```bash
# If starting from existing directory (your case)
cd /home/brian-isaac/Documents/personal/FinanceCoding/GNN

# Initialize git if not already done
git init
git add .
git commit -m "Initial commit: existing GAT implementation"

# Create main branch and set upstream
git branch -M main
```

### Create Project Structure
```bash
# Create the complete directory structure
mkdir -p {data/{raw/{membership,stooq,yfinance},processed,graphs/snapshots},src/{config,data/{collectors,processors,loaders},models/{base,hrp,lstm,gat,baselines},evaluation/{backtest,metrics,validation,reporting},utils},scripts/{experiments},configs/{data,models,experiments},tests/{unit/{test_data,test_models,test_evaluation},integration/{test_pipeline,test_backtest},fixtures},notebooks,docs/{api,tutorials,deployment,research}}

# Create __init__.py files for Python packages
find src -type d -exec touch {}/__init__.py \;
find tests -type d -exec touch {}/__init__.py \;

# Create initial configuration files
touch configs/data/default.yaml
touch configs/data/midcap400.yaml
touch configs/models/hrp_default.yaml
touch configs/models/lstm_default.yaml
touch configs/models/gat_default.yaml
touch configs/experiments/baseline_comparison.yaml
touch configs/experiments/full_evaluation.yaml
```

## Step 3: Python Environment Setup with uv

### Initialize uv Project
```bash
# Initialize uv project (this creates pyproject.toml)
uv init --python 3.9

# Verify Python version
uv run python --version
```

### Configure pyproject.toml
```bash
# Replace the generated pyproject.toml with our configuration
cat > pyproject.toml << 'EOF'
[project]
name = "portfolio-optimization-ml"
version = "1.0.0"
requires-python = ">=3.9"
description = "ML-Based Portfolio Optimization Research Framework"
authors = [
    {name = "Research Team", email = "research@example.com"},
]
readme = "README.md"

[dependency-groups]
core = [
    "numpy>=1.24.0",
    "pandas>=2.0.0", 
    "scipy>=1.10.0",
    "scikit-learn>=1.3.0",
    "networkx>=3.1",
]

deep-learning = [
    "torch>=2.0.0",
    "torch-geometric>=2.3.0",
    "torchvision>=0.15.0",
    "torch-scatter>=2.1.0",
    "torch-sparse>=0.6.0",
]

data-processing = [
    "pyarrow>=12.0.0",
    "yfinance>=0.2.18",
    "lxml>=4.9.0",
    "requests>=2.31.0",
    "beautifulsoup4>=4.12.0",
]

optimization = [
    "cvxpy>=1.3.0",
    "cvxopt>=1.3.0",
]

analysis = [
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "plotly>=5.15.0",
    "jupyter>=1.0.0",
    "jupyterlab>=4.0.0",
]

configuration = [
    "hydra-core>=1.3.0",
    "omegaconf>=2.3.0",
]

testing = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-xdist>=3.3.0",
    "pytest-mock>=3.11.0",
]

development = [
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.5.0",
    "pre-commit>=3.3.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "portfolio-optimization-ml[core,deep-learning,data-processing,optimization,analysis,configuration,testing,development]"
]

[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"
explicit = true

[tool.uv.sources]
torch = [
    { index = "pytorch-cu118" },
]
torch-geometric = [
    { index = "pytorch-cu118" },
]
torchvision = [
    { index = "pytorch-cu118" },
]

[tool.black]
line-length = 100
target-version = ['py39']

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
EOF
```

### Install Dependencies
```bash
# Install all dependencies with CUDA support
uv sync --all-groups

# Verify PyTorch CUDA installation
uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Verify torch-geometric installation
uv run python -c "import torch_geometric; print(f'PyTorch Geometric version: {torch_geometric.__version__}')"
```

## Step 4: Hydra Configuration Setup

### Create Base Configuration Structure
```bash
# Main Hydra config
cat > configs/config.yaml << 'EOF'
defaults:
  - data: midcap400
  - models: [hrp_default, lstm_default, gat_default]
  - experiment: baseline_comparison
  - _self_

# Global settings
project_name: portfolio_optimization_ml
output_dir: outputs
seed: 42

# Logging configuration
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# GPU settings
gpu:
  enabled: true
  device_id: 0
  memory_limit_gb: 11.0
EOF

# Data configuration
cat > configs/data/midcap400.yaml << 'EOF'
# S&P MidCap 400 Data Configuration
_target_: src.data.loaders.universe.UniverseLoader

universe_name: "sp_midcap_400"
start_date: "2016-01-01"
end_date: "2024-12-31"
rebalance_frequency: "M"  # Monthly

data_sources:
  membership:
    source: "wikipedia"
    url: "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
  
  prices:
    primary: "stooq"
    fallback: "yfinance"
    
  volume:
    primary: "stooq"
    fallback: "yfinance"

processing:
  min_history_days: 100
  max_missing_pct: 0.2
  outlier_method: "winsorize"
  outlier_limits: [0.01, 0.99]

storage:
  format: "parquet"
  compression: "gzip"
  base_path: "data/processed"
EOF

# Model configurations (HRP)
cat > configs/models/hrp_default.yaml << 'EOF'
_target_: src.models.hrp.model.HRPPortfolioModel

data_config:
  lookback_days: 756  # 3 years
  min_observations: 252
  correlation_method: "pearson"
  handle_missing: "pairwise"
  min_asset_coverage: 0.8
  winsorize_returns: true
  winsorize_quantiles: [0.01, 0.99]

clustering_config:
  linkage_method: "single"

allocation_config:
  risk_measure: "variance"

constraints:
  long_only: true
  top_k_positions: 50
  max_position_weight: 0.10
  max_monthly_turnover: 0.20
  transaction_cost_bps: 10.0
EOF

# LSTM configuration  
cat > configs/models/lstm_default.yaml << 'EOF'
_target_: src.models.lstm.model.LSTMPortfolioModel

model_config:
  input_features: 5
  hidden_size: 128
  num_layers: 3
  dropout: 0.3
  bidirectional: true
  attention_heads: 8
  attention_dropout: 0.1
  forecast_horizon: 21
  temperature: 1.0
  learning_rate: 0.001
  weight_decay: 0.00001
  batch_size: 32
  max_epochs: 100
  patience: 15
  gradient_accumulation_steps: 4
  mixed_precision: true
  gradient_clip_norm: 1.0

data_config:
  sequence_length: 60
  forecast_horizon: 21
  feature_set: ['returns', 'volatility_20d', 'momentum_20d', 'rsi_14d', 'volume_ratio_20d']
  min_history_days: 100
  standardize_features: true
  handle_missing: 'forward_fill'

constraints:
  long_only: true
  top_k_positions: 50
  max_position_weight: 0.10
  max_monthly_turnover: 0.20
  transaction_cost_bps: 10.0
EOF

# GAT configuration
cat > configs/models/gat_default.yaml << 'EOF'
_target_: src.models.gat.model.GATPortfolioModel

model_config:
  input_features: 10
  hidden_dim: 64
  num_attention_heads: 8
  num_layers: 3
  dropout: 0.3
  edge_feature_dim: 3
  constraint_layer: true
  learning_rate: 0.001
  weight_decay: 0.00001
  batch_size: 16
  max_epochs: 100
  patience: 15
  mixed_precision: true

graph_config:
  lookback_days: 252
  cov_method: "lw"
  filter_method: "tmfg"
  edge_attributes: true
  min_observations: 100

constraints:
  long_only: true
  top_k_positions: 50
  max_position_weight: 0.10
  max_monthly_turnover: 0.20
  transaction_cost_bps: 10.0
EOF

# Experiment configuration
cat > configs/experiments/baseline_comparison.yaml << 'EOF'
# Baseline Comparison Experiment
experiment_name: "baseline_comparison"

# Backtesting configuration
backtest:
  training_months: 36
  validation_months: 12
  test_months: 12
  step_months: 12
  transaction_cost_bps: 10.0

# Performance evaluation
evaluation:
  metrics:
    - sharpe_ratio
    - max_drawdown
    - information_ratio
    - calmar_ratio
    - sortino_ratio
    - volatility
    - var_95
    - cvar_95
  
  significance_tests:
    - sharpe_ratio_test
    - bootstrap_ci
  
  benchmark: "equal_weight"

# Output configuration
output:
  save_results: true
  save_plots: true
  generate_report: true
  export_weights: false
EOF
```

### Create Hydra Application Structure
```bash
# Create main configuration manager
cat > src/config/__init__.py << 'EOF'
"""Configuration management using Hydra."""
EOF

cat > src/config/base.py << 'EOF'
"""Base configuration classes and utilities."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    universe_name: str
    start_date: str
    end_date: str
    rebalance_frequency: str
    data_sources: Dict[str, Any]
    processing: Dict[str, Any]
    storage: Dict[str, Any]


@dataclass
class ModelConfig:
    """Model configuration base class."""
    _target_: str
    constraints: Dict[str, Any]


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    experiment_name: str
    backtest: Dict[str, Any]
    evaluation: Dict[str, Any]
    output: Dict[str, Any]


@dataclass
class AppConfig:
    """Main application configuration."""
    project_name: str
    output_dir: str
    seed: int
    logging: Dict[str, Any]
    gpu: Dict[str, Any]
    data: DataConfig
    models: List[ModelConfig]
    experiment: ExperimentConfig


# Register configurations with Hydra
cs = ConfigStore.instance()
cs.store(name="base_config", node=AppConfig)
EOF
```

## Step 5: Environment Variables and Secrets

```bash
# Create environment file
cat > .env.example << 'EOF'
# Data Sources (if API keys needed in future)
# ALPHA_VANTAGE_API_KEY=your_key_here
# QUANDL_API_KEY=your_key_here

# Logging
LOG_LEVEL=INFO

# GPU Settings  
CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_MEMORY_FRACTION=0.9

# Output Directories
OUTPUT_DIR=outputs
DATA_DIR=data
CACHE_DIR=.cache

# Development
PYTHONPATH=src:$PYTHONPATH
EOF

# Copy to actual .env file
cp .env.example .env

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Data files
data/raw/
data/processed/
*.parquet
*.csv
*.h5
*.hdf5

# Model artifacts
*.pth
*.pkl
*.joblib
models/saved/
checkpoints/

# Outputs
outputs/
logs/
.cache/
.hrp_cache/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# OS
.DS_Store
Thumbs.db

# Hydra
.hydra/
multirun/

# GPU memory dumps
*.gpu

# Temporary files
tmp/
temp/
EOF
```

## Step 6: Development Tools Setup

### Code Quality Tools
```bash
# Install pre-commit hooks
uv run pre-commit install

# Create pre-commit configuration
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3.9
        args: [--line-length=100]

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: [--profile=black, --line-length=100]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=100, --ignore=E203,W503,F401]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-requests, types-PyYAML]
        args: [--ignore-missing-imports]
EOF

# Run initial code formatting
uv run black src/ tests/
uv run isort src/ tests/
```

### Testing Setup
```bash
# Create pytest configuration
cat > pytest.ini << 'EOF'
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v 
    --tb=short
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow tests (require significant computation)
    gpu: Tests requiring GPU
EOF

# Create initial test files
cat > tests/conftest.py << 'EOF'
"""Test configuration and fixtures."""
import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path


@pytest.fixture
def sample_returns_data():
    """Generate sample returns data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='B')  # Business days
    assets = [f'ASSET_{i:03d}' for i in range(50)]
    
    # Generate correlated returns
    n_dates, n_assets = len(dates), len(assets)
    returns = np.random.multivariate_normal(
        mean=[0.0005] * n_assets,
        cov=0.0004 * (0.3 * np.ones((n_assets, n_assets)) + 0.7 * np.eye(n_assets)),
        size=n_dates
    )
    
    return pd.DataFrame(returns, index=dates, columns=assets)


@pytest.fixture
def sample_universe():
    """Sample asset universe for testing."""
    return [f'ASSET_{i:03d}' for i in range(50)]


@pytest.fixture
def gpu_available():
    """Check if GPU is available for testing."""
    return torch.cuda.is_available()


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir
EOF

# Create sample unit tests
cat > tests/unit/test_config.py << 'EOF'
"""Test configuration management."""
import pytest
from hydra import compose, initialize
from omegaconf import DictConfig


def test_hydra_config_loading():
    """Test that Hydra configurations load correctly."""
    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="config")
        
        assert isinstance(cfg, DictConfig)
        assert "project_name" in cfg
        assert "data" in cfg
        assert "models" in cfg
        assert "experiment" in cfg


def test_model_configs():
    """Test individual model configurations."""
    with initialize(config_path="../../configs", version_base=None):
        # Test HRP config
        hrp_cfg = compose(config_name="config", overrides=["models=[hrp_default]"])
        assert hrp_cfg.models[0]._target_ == "src.models.hrp.model.HRPPortfolioModel"
        
        # Test LSTM config
        lstm_cfg = compose(config_name="config", overrides=["models=[lstm_default]"])
        assert lstm_cfg.models[0]._target_ == "src.models.lstm.model.LSTMPortfolioModel"
        
        # Test GAT config
        gat_cfg = compose(config_name="config", overrides=["models=[gat_default]"])
        assert gat_cfg.models[0]._target_ == "src.models.gat.model.GATPortfolioModel"
EOF
```

## Step 7: Jupyter Lab Setup

```bash
# Install and configure JupyterLab
uv add --group analysis "jupyterlab>=4.0.0"
uv add --group analysis "ipywidgets>=8.0.0"
uv add --group analysis "jupyter-ai>=2.0.0"

# Create Jupyter configuration
mkdir -p ~/.jupyter

cat > ~/.jupyter/jupyter_lab_config.py << 'EOF'
c = get_config()  # noqa

# Server configuration
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True

# Security
c.ServerApp.token = ''
c.ServerApp.password = ''

# Resource management
c.ResourceUseDisplay.track_cpu_percent = True
c.ResourceUseDisplay.mem_limit = 32212254720  # 32GB in bytes

# Extensions
c.LabApp.collaborative = True
EOF

# Create sample notebook
cat > notebooks/00_environment_verification.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Environment Verification\n",
    "This notebook verifies that the development environment is properly configured."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Core imports\n",
    "import sys\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import torch\n",
    "import torch_geometric\n",
    "from hydra import compose, initialize\n",
    "\n",
    "print(f\"Python version: {sys.version}\")\n",
    "print(f\"NumPy version: {np.__version__}\")\n",
    "print(f\"Pandas version: {pd.__version__}\")\n",
    "print(f\"PyTorch version: {torch.__version__}\")\n",
    "print(f\"PyTorch Geometric version: {torch_geometric.__version__}\")\n",
    "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
    "if torch.cuda.is_available():\n",
    "    print(f\"CUDA version: {torch.version.cuda}\")\n",
    "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")\n",
    "    print(f\"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test Hydra configuration loading\n",
    "with initialize(config_path=\"../configs\", version_base=None):\n",
    "    cfg = compose(config_name=\"config\")\n",
    "    print(f\"Project: {cfg.project_name}\")\n",
    "    print(f\"Models: {[model._target_ for model in cfg.models]}\")\n",
    "    print(\"✅ Hydra configuration loaded successfully\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test PyTorch GPU functionality\n",
    "if torch.cuda.is_available():\n",
    "    device = torch.device(\"cuda\")\n",
    "    x = torch.randn(1000, 1000).to(device)\n",
    "    y = torch.randn(1000, 1000).to(device)\n",
    "    z = torch.mm(x, y)\n",
    "    print(f\"GPU computation result shape: {z.shape}\")\n",
    "    print(f\"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB\")\n",
    "    print(\"✅ GPU computation working correctly\")\n",
    "else:\n",
    "    print(\"❌ CUDA not available - check GPU setup\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF
```

## Step 8: Executable Scripts

```bash
# Make scripts directory executable
chmod +x scripts/

# Create main execution scripts
cat > scripts/setup_project.py << 'EOF'
#!/usr/bin/env python3
"""Project setup and initialization script."""
import logging
from pathlib import Path
import sys
sys.path.append("src")

from src.utils.logging import setup_logging


def main():
    """Initialize project structure and verify setup."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Setting up Portfolio Optimization ML project...")
    
    # Verify directory structure
    required_dirs = [
        "data/processed",
        "src/models", 
        "configs",
        "tests",
        "notebooks",
        "outputs"
    ]
    
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Directory verified: {dir_path}")
    
    logger.info("✅ Project setup complete!")


if __name__ == "__main__":
    main()
EOF

cat > scripts/run_experiment.py << 'EOF'
#!/usr/bin/env python3
"""Main experiment execution script."""
import hydra
from omegaconf import DictConfig
import logging
import sys
sys.path.append("src")

from src.utils.logging import setup_logging


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run portfolio optimization experiment."""
    setup_logging(cfg.logging.level)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting experiment: {cfg.experiment.experiment_name}")
    logger.info(f"Models: {[model._target_ for model in cfg.models]}")
    
    # TODO: Import and run experiment orchestrator
    # from src.evaluation.orchestrator import ExperimentOrchestrator
    # orchestrator = ExperimentOrchestrator(cfg)
    # results = orchestrator.run_full_experiment()
    
    logger.info("✅ Experiment completed!")


if __name__ == "__main__":
    main()
EOF

# Make scripts executable
chmod +x scripts/*.py
```

## Step 9: Verification and Testing

```bash
# Run environment verification
uv run python -c "
import torch
import torch_geometric
import hydra
import pandas as pd
import numpy as np
print('✅ All core dependencies imported successfully')
print(f'PyTorch CUDA: {torch.cuda.is_available()}')
print(f'PyTorch version: {torch.__version__}')
print(f'PyTorch Geometric version: {torch_geometric.__version__}')
"

# Test Hydra configuration
cd /home/brian-isaac/Documents/personal/FinanceCoding/GNN
uv run python -c "
from hydra import compose, initialize
with initialize(config_path='configs', version_base=None):
    cfg = compose(config_name='config')
    print('✅ Hydra configuration loaded successfully')
    print(f'Project: {cfg.project_name}')
"

# Run tests
uv run pytest tests/ -v

# Test Jupyter environment
uv run jupyter lab --generate-config
# Note: Run 'uv run jupyter lab' to start Jupyter server

# Run project setup
uv run python scripts/setup_project.py
```

## Step 10: IDE Configuration (VS Code)

```bash
# Create VS Code workspace configuration
mkdir -p .vscode

cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.pytestArgs": ["tests"],
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=100"],
    "python.sortImports.args": ["--profile=black"],
    "[python]": {
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        }
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/.pytest_cache": true,
        "**/.mypy_cache": true,
        ".coverage": true,
        "htmlcov/": true,
        "outputs/": true,
        ".hydra/": true
    }
}
EOF

cat > .vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            }
        },
        {
            "name": "Run Experiment",
            "type": "python", 
            "request": "launch",
            "program": "${workspaceFolder}/scripts/run_experiment.py",
            "console": "integratedTerminal",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            }
        },
        {
            "name": "Pytest",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["tests/", "-v"],
            "console": "integratedTerminal",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            }
        }
    ]
}
EOF

cat > .vscode/tasks.json << 'EOF'
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Install Dependencies",
            "type": "shell",
            "command": "uv sync --all-groups",
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "Run Tests",
            "type": "shell",
            "command": "uv run pytest tests/ -v",
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "Format Code",
            "type": "shell",
            "command": "uv run black src/ tests/ && uv run isort src/ tests/",
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "Start Jupyter Lab",
            "type": "shell",
            "command": "uv run jupyter lab",
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        }
    ]
}
EOF
```

## Final Verification Checklist

Run through this checklist to ensure everything is working:

```bash
# 1. Environment verification
echo "1. Testing core environment..."
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 2. Dependencies verification  
echo "2. Testing dependencies..."
uv run python -c "import pandas, numpy, torch, torch_geometric, hydra; print('✅ Core deps OK')"

# 3. Configuration verification
echo "3. Testing Hydra config..."
uv run python -c "
from hydra import compose, initialize
with initialize(config_path='configs'):
    cfg = compose(config_name='config')
    print('✅ Hydra OK')
"

# 4. Project structure verification
echo "4. Verifying project structure..."
uv run python scripts/setup_project.py

# 5. Test suite verification
echo "5. Running test suite..."
uv run pytest tests/ -v

# 6. Jupyter verification
echo "6. Testing Jupyter..."
uv run jupyter --version

echo "
✅ Development environment setup complete!

Next steps:
1. Run: uv run jupyter lab (start development environment)
2. Open: notebooks/00_environment_verification.ipynb
3. Begin implementation following Epic 1 specifications

Key commands:
- uv run python scripts/run_experiment.py  # Run experiments
- uv run pytest tests/                     # Run tests  
- uv run jupyter lab                       # Start notebooks
- uv sync --all-groups                     # Update dependencies
"
```

Your development environment is now fully configured with uv dependency management, Hydra configuration, GPU optimization, and ready for the ML portfolio optimization implementation.