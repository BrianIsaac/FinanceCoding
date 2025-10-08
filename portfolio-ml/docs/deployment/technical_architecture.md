# Technical Architecture Documentation

## System Overview

The Graph Neural Network (GNN) Portfolio Optimization System is a comprehensive machine learning framework designed for institutional portfolio management. The system integrates multiple advanced methodologies including Hierarchical Risk Parity (HRP), Long Short-Term Memory (LSTM) networks, and Graph Attention Networks (GAT) to deliver superior risk-adjusted returns.

## System Requirements

### Hardware Specifications

#### Minimum Requirements
- **GPU**: RTX GeForce 5070Ti or equivalent with 12GB VRAM minimum
- **RAM**: 32GB system memory
- **Storage**: 500GB SSD for data and model storage
- **CPU**: 8-core processor (Intel i7 or AMD Ryzen 7 equivalent)

#### Recommended Specifications
- **GPU**: RTX GeForce 5070Ti with 12GB VRAM (conservative 11GB limit for memory management)
- **RAM**: 64GB system memory for large universe processing
- **Storage**: 1TB NVMe SSD for optimal data pipeline performance
- **CPU**: 16-core processor for parallel processing capabilities

### Software Dependencies

#### Core Environment
```toml
[project]
name = "gnn"
version = "0.1.0"
requires-python = ">=3.12"

[dependency-groups]
data = [
    "cvxpy>=1.7.1",
    "cvxpylayers>=0.1.6", 
    "dcor>=0.6",
    "diffcp>=1.0.23",
    "matplotlib>=3.9.4",
    "numpy>=2.0.2",
    "pandas>=2.3.1",
    "python-dotenv>=1.1.1",
    "scikit-learn>=1.6.1",
    "scipy>=1.13.1",
    "scs>=3.2.8",
    "yfinance>=0.2.65",
]

graph = [
    "networkx>=3.2.1",
    "torch>=2.7.0",
    "torch-geometric>=2.6.1", 
    "torchvision>=0.22.0",
]

requests = [
    "lxml>=6.0.0",
    "omegaconf>=2.3.0",
    "plotly>=6.3.0",
    "pyarrow>=21.0.0",
    "seaborn>=0.13.2",
]
```

#### Development Dependencies
```toml
dev = [
    "black>=25.1.0",
    "ruff>=0.12.12",
    "ipykernel>=6.30.1",
    "isort>=6.0.1",
    "mypy>=1.17.1",
    "pre-commit>=4.3.0",
    "pytest>=8.4.2",
    "pytest-cov>=6.2.1",
    "pytest-mock>=3.15.0",
    "pytest-xdist>=3.8.0",
]
```

## Architecture Components

### Data Architecture

#### Data Pipeline Structure
```
src/data/
├── collectors/           # Market data collection modules
│   ├── yfinance.py      # Yahoo Finance data collector
│   └── base.py          # Abstract data collector interface
├── loaders/             # Data loading and management
│   ├── parquet_manager.py    # Efficient parquet-based storage
│   └── portfolio_data.py     # Portfolio data loader
└── processors/          # Data transformation and validation
    ├── data_pipeline.py      # Main data processing pipeline
    ├── data_normalization.py # Feature normalization
    ├── data_quality_validator.py # Data quality checks
    ├── gap_filling.py        # Missing data handling
    ├── universe_builder.py   # Investment universe construction
    └── survivorship_validator.py # Survivorship bias validation
```

#### Data Storage Strategy
- **Format**: Apache Parquet for efficient storage and fast I/O operations
- **Structure**: Date-partitioned data organized by symbol and date
- **Caching**: Intelligent caching of processed features and intermediate results
- **Validation**: Comprehensive data quality validation with automated gap filling

### Model Architecture

#### Hierarchical Risk Parity (HRP)
```python
# Location: src/models/hrp/
class HRPModel:
    """
    Hierarchical Risk Parity implementation with advanced clustering methods.
    
    Features:
    - Multiple linkage methods: single, complete, average, ward
    - Distance metrics: correlation, angular, absolute_correlation
    - Lookback periods: 252, 504, 756, 1008 days
    """
```

#### LSTM Temporal Networks
```python
# Location: src/models/lstm/
class LSTMModel:
    """
    Long Short-Term Memory network for temporal pattern recognition.
    
    Architecture:
    - Sequence lengths: 30, 45, 60, 90 days
    - Hidden sizes: 64, 128, 256 units
    - Layer configurations: 1, 2, 3 layers
    - Dropout rates: 0.1, 0.3, 0.5
    """
```

#### Graph Attention Networks (GAT)
```python
# Location: src/models/gat/
class GATModel:
    """
    Graph Attention Network for relationship modeling.
    
    Configuration:
    - Attention heads: 2, 4, 8
    - Hidden dimensions: 64, 128, 256
    - Graph construction: k-NN, MST, TMFG methods
    - Learning rates: 0.0001, 0.001, 0.01
    """
```

### Evaluation Framework

#### Rolling Backtest Engine
```python
# Location: src/evaluation/backtest/
class RollingBacktestEngine:
    """
    Production-grade rolling window backtesting framework.
    
    Features:
    - Strict no-look-ahead bias prevention
    - Memory-efficient processing for large datasets
    - GPU optimization for neural network models
    - Statistical significance testing integration
    """
```

#### Performance Analytics
```python
# Location: src/evaluation/metrics/
class PerformanceAnalytics:
    """
    Institutional-grade performance metric calculations.
    
    Metrics:
    - Sharpe ratio with Jobson-Korkie statistical testing
    - Information ratio and tracking error
    - Maximum drawdown and Value at Risk (VaR)
    - Conditional Value at Risk (CVaR)
    """
```

## GPU Memory Management

### Memory Optimization Strategies

#### Conservative Memory Limits
```python
class GPUMemoryManager:
    """
    RTX GeForce 5070Ti optimization with 11GB conservative limit.
    
    Strategies:
    - Mixed precision training (FP16)
    - Gradient checkpointing for memory efficiency
    - Automatic memory cleanup after each model training
    - 90% usage threshold monitoring with alerts
    """
    
    VRAM_LIMIT = 11 * 1024**3  # 11GB conservative limit
    USAGE_THRESHOLD = 0.9      # 90% usage alert threshold
```

#### Training Performance Targets
- **HRP Model**: 2 minutes maximum per fold
- **LSTM Model**: 4 hours maximum per fold
- **GAT Model**: 6 hours maximum per fold
- **Production Rebalancing**: 10 minutes maximum
- **Full Backtest**: 8 hours maximum

### Batch Processing Strategy
```python
class MemoryEfficientTrainer:
    """
    Gradient accumulation and mixed precision training.
    
    Features:
    - Automatic batch size adjustment based on available memory
    - Gradient accumulation for large effective batch sizes
    - Model checkpoint saving for recovery from memory errors
    """
```

## Scalability Considerations

### Universe Size Scalability

#### S&P MidCap 400 (Current Target)
- **Memory Usage**: 8GB GPU, 16GB System RAM
- **Training Time**: 4-6 hours full pipeline
- **Inference Time**: <10 minutes for portfolio generation

#### S&P 500 Extension
- **Memory Usage**: 11GB GPU, 32GB System RAM
- **Training Time**: 8-12 hours full pipeline
- **Computational Scaling**: Linear for LSTM, quadratic for graph methods

#### Russell 2000 (Future Consideration)
- **Memory Usage**: Requires system architecture optimization
- **Recommendation**: Multi-GPU setup or cloud deployment for large universes

### Performance Scaling Characteristics

#### Model Complexity Scaling
```python
# Memory complexity by model type
HRP_COMPLEXITY = O(n²)      # Covariance matrix operations
LSTM_COMPLEXITY = O(n×T)    # Linear in assets and time
GAT_COMPLEXITY = O(n²×H)    # Quadratic in assets, linear in hidden size
```

## Configuration Management

### Hierarchical Configuration Structure
```yaml
# configs/production/production_config.yaml
data:
  universe: "sp_midcap_400"
  lookback_days: 252
  refresh_frequency: "daily"

models:
  hrp:
    linkage_method: "ward"
    distance_metric: "correlation"
    lookback_days: 252
  
  lstm:
    sequence_length: 60
    hidden_size: 128
    num_layers: 2
    dropout: 0.3
  
  gat:
    attention_heads: 4
    hidden_dim: 128
    graph_method: "k_nn"
    dropout: 0.3

constraints:
  long_only: true
  top_k_positions: 50
  max_position_weight: 0.10
  max_monthly_turnover: 0.20
  transaction_cost_bps: 10.0
```

## Security and Data Privacy

### Data Protection Framework
```python
class SecureDataManager:
    """
    Comprehensive data security and privacy management.
    
    Features:
    - Local-only processing (no cloud data transmission)
    - Encryption at rest for sensitive portfolio data
    - Access logging and audit trails
    - 365-day data retention policy
    - Rate-limited API access with respectful delays
    """
```

### Privacy Configuration
```python
class PrivacyConfig:
    """
    Institutional compliance settings.
    
    Settings:
    - Data anonymization for backtesting
    - Secure model checkpointing
    - Audit trail generation
    - Compliance reporting integration
    """
```

## Dependency Management with uv

### Installation and Environment Setup
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create project environment
uv python install 3.12
uv venv --python 3.12

# Install production dependencies
uv pip install -e ".[data,graph,requests]"

# Install development dependencies  
uv pip install -e ".[dev,logging]"
```

### Dependency Groups Strategy
The system uses uv's dependency groups for modular installation:
- **data**: Core data processing and optimization libraries
- **graph**: PyTorch and graph neural network dependencies
- **requests**: Web scraping and visualization libraries
- **dev**: Development, testing, and code quality tools
- **logging**: Experiment tracking and progress monitoring

### CUDA Dependencies
```toml
[tool.uv.sources]
torch = [
  { index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
torchvision = [
  { index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

## System Integration Points

### External Data Sources
- **Yahoo Finance**: Primary market data source with rate limiting
- **Parquet Storage**: Local caching for offline processing
- **Configuration Files**: YAML-based parameter management

### Performance Monitoring
- **Real-time Metrics**: GPU memory, training progress, performance metrics
- **Alert Systems**: Threshold-based notifications for system health
- **Dashboard Integration**: Web-based monitoring interface

### Backup and Recovery
- **Model Checkpoints**: Automatic saving during training
- **Data Backup**: Daily parquet file backup procedures
- **Configuration Versioning**: Git-based configuration management

## Production Deployment Considerations

### Environment Isolation
```bash
# Production environment setup
uv venv --python 3.12 production-env
source production-env/bin/activate
uv pip install -e ".[data,graph,requests,logging]"
```

### Resource Monitoring
```python
class ProductionMonitor:
    """
    Production system monitoring and alerting.
    
    Monitors:
    - GPU memory usage and temperature
    - Training progress and convergence
    - Model performance degradation
    - Data quality issues
    """
```

### Maintenance Procedures
- **Weekly**: Data quality validation and model performance review
- **Monthly**: Full model retraining and portfolio rebalancing
- **Quarterly**: System performance optimization and hardware assessment
- **Annually**: Full system architecture review and upgrade planning

This technical architecture provides the foundation for enterprise-grade deployment of the GNN portfolio optimization system with institutional-level reliability, security, and scalability requirements.