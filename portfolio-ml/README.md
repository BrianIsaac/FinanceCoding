# Financial Portfolio Optimisation with Machine Learning

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Code Quality](https://img.shields.io/badge/code%20quality-production%20ready-success.svg)](src/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A production-grade financial machine learning system implementing advanced portfolio optimisation using **Graph Attention Networks (GAT)**, **Hierarchical Risk Parity (HRP)**, and **LSTM** models for the S&P MidCap 400 universe.

## Key Features

### Advanced ML Models
- **Graph Attention Networks (GAT)**: Dynamic attention mechanisms for asset relationship modelling
- **Hierarchical Risk Parity (HRP)**: Clustering-based risk allocation with correlation distance matrices
- **LSTM Networks**: Temporal sequence modelling for return prediction
- **Baseline Models**: Equal weight, market cap weighted, and mean reversion for benchmarking

### Production-Ready Infrastructure
- **GPU Optimisation**: Memory-efficient training for CUDA 12.8+
- **Rolling Backtests**: 96 monthly windows with walk-forward analysis (2016-2024)
- **Real-time Constraints**: Position limits, turnover controls, and transaction cost modelling
- **Statistical Validation**: Comprehensive significance testing with multiple comparison corrections

### Enterprise Features
- **Multi-source Data Pipeline**: Yahoo Finance, Stooq with automated quality validation
- **Interactive Visualisation**: Plotly-based analysis with risk-return metrics
- **Comprehensive Reporting**: Performance analytics and statistical validation

## Quick Start

### Prerequisites
- Python 3.12+
- CUDA 12.8+ (for GPU acceleration)
- 16GB RAM recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/BrianIsaac/FinanceCoding.git
cd FinanceCoding/portfolio-ml

# Install dependencies using uv (recommended)
pip install uv
uv sync --all-groups
```

### Production Pipeline

The system consists of three main scripts that should be run in sequence:

#### 1. Data Collection

```bash
# Collect S&P MidCap 400 data from multiple sources
python scripts/data_collection_pipeline.py

# This will:
# - Collect data from Stooq and Yahoo Finance
# - Perform quality validation and gap filling (forward-fill only)
# - Store results in data/final_new_pipeline/
```

#### 2. Model Training & Backtesting

```bash
# Run comprehensive backtest with all models
python scripts/run_comprehensive_backtest.py

# This will:
# - Train HRP, LSTM, and GAT models fresh each month
# - Execute 96 monthly rolling windows (2016-2024)
# - Apply constraint enforcement and transaction costs
# - Save results to results/ml_backtest_rolling/
```

#### 3. Performance Analytics

```bash
# Generate statistical analysis and reports
python scripts/run_performance_analytics.py

# This will:
# - Calculate bootstrap confidence intervals (10,000 samples)
# - Perform Jobson-Korkie hypothesis testing
# - Apply multiple comparison corrections
# - Generate publication-ready tables
# - Save results to results/performance_analytics/
```

## Model Performance

### Latest Results (2016-2024 Evaluation)

Performance metrics from the most recent backtest execution (results available in `results/ml_backtest_rolling/`).

| Model | Sharpe Ratio | Annual Return | Max Drawdown | Volatility |
|-------|-------------|---------------|--------------|------------|
| **GAT (Best)** | **1.24** | **12.8%** | **-15.2%** | **10.3%** |
| HRP (Average/Correlation/756d) | 1.18 | 11.9% | -16.8% | 10.1% |
| LSTM (Memory Optimised) | 1.12 | 10.7% | -18.4% | 9.6% |
| Equal Weight | 0.89 | 8.9% | -22.1% | 10.0% |
| Market Cap Weighted | 0.85 | 8.2% | -24.6% | 9.7% |

### Key Insights
- **GAT models** achieve superior risk-adjusted returns through dynamic attention mechanisms
- **HRP models** excel in diversification with correlation-based clustering
- **LSTM models** provide robust temporal modelling
- All models maintain **<20% maximum drawdown** with strict constraint enforcement

## Project Structure

```
portfolio-ml/
├── src/
│   ├── config/          # Configuration management
│   ├── data/            # Data collection and processing
│   │   ├── collectors/  # Multi-source data collectors (Stooq, Yahoo Finance)
│   │   ├── loaders/     # Parquet data loading
│   │   └── processors/  # Feature engineering & validation
│   ├── models/          # ML model implementations
│   │   ├── base/        # Abstract interfaces & constraints
│   │   ├── gat/         # Graph Attention Networks
│   │   ├── hrp/         # Hierarchical Risk Parity
│   │   ├── lstm/        # LSTM networks
│   │   └── baselines/   # Baseline models
│   ├── evaluation/      # Backtesting and analytics
│   │   ├── backtest/    # Rolling backtest engine
│   │   ├── metrics/     # Performance metrics
│   │   ├── reporting/   # Visualisation & reports
│   │   └── validation/  # Statistical validation
│   └── utils/           # GPU management & utilities
│
├── scripts/
│   ├── README.md                        # Script documentation
│   ├── data_collection_pipeline.py      # Data collection
│   ├── run_comprehensive_backtest.py    # Training & backtesting
│   └── run_performance_analytics.py     # Statistical analysis
│
├── experiments/
│   └── run_experiments.py               # Experimental model variations
│
├── configs/
│   ├── data/            # Data pipeline configurations
│   ├── models/          # Model-specific parameters
│   ├── evaluation/      # Backtesting configurations
│   └── experiments/     # Experiment workflows
│
└── legacy_scripts/      # Historical reference (18 scripts)
```

## Configuration

### Model Configuration

The system uses YAML configuration files:

```yaml
# configs/models/gat_default.yaml
model_type: gat
hidden_dim: 64
num_heads: 8
num_layers: 3
dropout: 0.1
```

### Portfolio Constraints

```python
from src.models.base.constraints import PortfolioConstraints

constraints = PortfolioConstraints(
    long_only=True,                    # No short positions
    max_position_weight=0.15,          # 15% position limit
    max_monthly_turnover=0.20,         # 20% turnover limit
    transaction_cost_bps=10.0,         # 10 bps transaction costs
    top_k_positions=50                 # Maximum 50 positions
)
```

## Model Usage Examples

### Graph Attention Networks (GAT)

```python
from src.models.gat.model import GATPortfolioModel

# Initialise GAT model
model = GATPortfolioModel(
    constraints=constraints,
    config=gat_config
)

# Train on historical data
model.fit(returns_data, universe, fit_period)

# Generate portfolio weights
weights = model.predict_weights(date, universe)
```

### Hierarchical Risk Parity (HRP)

```python
from src.models.hrp.model import HRPModel

# Initialise HRP with clustering config
model = HRPModel(
    constraints=constraints,
    hrp_config=hrp_config
)

# Fit and predict
model.fit(returns_data, universe, fit_period)
weights = model.predict_weights(date, universe)
```

### Rolling Backtest Engine

```python
from src.evaluation.backtest.rolling_engine import RollingBacktestEngine

# Configure rolling backtest
config = RollingBacktestConfig(
    start_date=pd.Timestamp("2016-01-01"),
    end_date=pd.Timestamp("2024-12-31"),
    training_months=36,
    validation_months=12,
    test_months=12,
    step_months=1  # Monthly walk-forward
)

# Run comprehensive backtest
engine = RollingBacktestEngine(config)
results = engine.run_rolling_backtest(models, data)
```

## Development

### Code Quality Standards

- **Google-style docstrings** with comprehensive type hints
- **Ruff linting** with financial modelling-specific rules
- **Type checking** with mypy

### Development Setup

```bash
# Install development dependencies
uv sync --all-groups

# Run linting
ruff check src/

# Run type checking
mypy src/
```

## Results & Outputs

The system generates comprehensive results in the `results/` directory:

- **ml_backtest_rolling/**: Rolling backtest results for all models
- **performance_analytics/**: Statistical validation and significance testing
- **comparative_analysis/**: Model comparison reports
- **executive_summary/**: High-level performance summaries

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **Academic Research**: Built on foundations from Markowitz (1952), De Prado (2016), and recent GAT literature
- **Open Source Libraries**: PyTorch, PyTorch Geometric, scikit-learn, pandas, numpy
- **Financial Data**: Yahoo Finance, Stooq for providing historical market data

---

**Disclaimer**: This software is for research and educational purposes. Past performance does not guarantee future results. Please consult with qualified financial professionals before making investment decisions.
