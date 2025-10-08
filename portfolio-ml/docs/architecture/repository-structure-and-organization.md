# Repository Structure and Organization

## Current State Analysis

The existing codebase contains substantial implementation but requires architectural reorganization for production readiness:

**Existing Assets:**
- ✅ Complete GAT implementation with multiple graph construction methods
- ✅ Comprehensive data pipeline with Wikipedia scraping and multi-source integration
- ✅ Graph construction utilities (MST, TMFG, k-NN filtering)
- ✅ Basic evaluation and backtesting framework
- 🔄 Scattered code requiring modular organization
- ❌ LSTM and HRP modules need formal implementation
- ❌ Unified constraint system requires completion

## Target Monorepo Structure

```
portfolio-optimization-ml/
├── README.md
├── pyproject.toml                    # uv dependency management
├── .env.example                      # Environment variables template
├── .gitignore
│
├── data/                             # Data storage layer
│   ├── raw/                          # Original downloaded data
│   │   ├── membership/               # S&P MidCap 400 historical membership
│   │   ├── stooq/                    # Raw Stooq downloads
│   │   └── yfinance/                 # Raw Yahoo Finance data
│   ├── processed/                    # Clean, analysis-ready datasets
│   │   ├── prices.parquet            # Aligned price panel
│   │   ├── volume.parquet            # Aligned volume panel
│   │   ├── returns_daily.parquet     # Daily returns matrix
│   │   └── universe_calendar.parquet # Dynamic membership calendar
│   └── graphs/                       # Pre-built graph snapshots
│       └── snapshots/                # Monthly correlation graphs
│
├── src/                              # Source code modules
│   ├── __init__.py
│   ├── config/                       # Configuration management
│   │   ├── __init__.py
│   │   ├── base.py                   # Base configuration classes
│   │   ├── models.py                 # Model-specific configurations
│   │   └── data.py                   # Data pipeline configurations
│   │
│   ├── data/                         # Data processing pipeline
│   │   ├── __init__.py
│   │   ├── collectors/               # Data collection modules
│   │   │   ├── __init__.py
│   │   │   ├── wikipedia.py          # S&P MidCap 400 membership scraping
│   │   │   ├── stooq.py              # Stooq data integration
│   │   │   └── yfinance.py           # Yahoo Finance augmentation
│   │   ├── processors/               # Data cleaning and preparation
│   │   │   ├── __init__.py
│   │   │   ├── alignment.py          # Calendar alignment utilities
│   │   │   ├── cleaning.py           # Data quality and cleaning
│   │   │   └── features.py           # Feature engineering
│   │   └── loaders/                  # Data loading utilities
│   │       ├── __init__.py
│   │       ├── parquet.py            # Parquet I/O operations
│   │       └── universe.py           # Dynamic universe management
│   │
│   ├── models/                       # ML model implementations
│   │   ├── __init__.py
│   │   ├── base/                     # Base classes and interfaces
│   │   │   ├── __init__.py
│   │   │   ├── portfolio_model.py    # Abstract portfolio model interface
│   │   │   └── constraints.py        # Unified constraint system
│   │   ├── hrp/                      # Hierarchical Risk Parity
│   │   │   ├── __init__.py
│   │   │   ├── clustering.py         # Correlation-based clustering
│   │   │   ├── allocation.py         # Recursive bisection allocation
│   │   │   └── model.py              # Main HRP model class
│   │   ├── lstm/                     # LSTM temporal networks
│   │   │   ├── __init__.py
│   │   │   ├── architecture.py       # LSTM network architectures
│   │   │   ├── training.py           # Training and validation logic
│   │   │   └── model.py              # Main LSTM model class
│   │   ├── gat/                      # Graph Attention Networks
│   │   │   ├── __init__.py
│   │   │   ├── gat_model.py          # GAT architecture (existing)
│   │   │   ├── graph_builder.py      # Graph construction (existing)
│   │   │   └── model.py              # Main GAT model class
│   │   └── baselines/                # Baseline comparisons
│   │       ├── __init__.py
│   │       ├── equal_weight.py       # Equal-weight portfolio
│   │       └── mean_variance.py      # Classical mean-variance optimization
│   │
│   ├── evaluation/                   # Backtesting and evaluation
│   │   ├── __init__.py
│   │   ├── backtest/                 # Backtesting engine
│   │   │   ├── __init__.py
│   │   │   ├── engine.py             # Main backtesting orchestration
│   │   │   ├── rebalancing.py        # Monthly rebalancing logic
│   │   │   └── transaction_costs.py  # Cost modeling
│   │   ├── metrics/                  # Performance analytics
│   │   │   ├── __init__.py
│   │   │   ├── returns.py            # Return-based metrics
│   │   │   ├── risk.py               # Risk metrics and drawdowns
│   │   │   └── attribution.py        # Performance attribution
│   │   ├── validation/               # Statistical validation
│   │   │   ├── __init__.py
│   │   │   ├── rolling.py            # Rolling window validation
│   │   │   ├── significance.py       # Statistical significance testing
│   │   │   └── bootstrap.py          # Bootstrap confidence intervals
│   │   └── reporting/                # Results visualization
│   │       ├── __init__.py
│   │       ├── charts.py             # Performance charts
│   │       ├── tables.py             # Summary tables
│   │       └── export.py             # Report generation
│   │
│   └── utils/                        # Shared utilities
│       ├── __init__.py
│       ├── io.py                     # File I/O helpers
│       ├── dates.py                  # Date handling utilities
│       ├── math.py                   # Mathematical utilities
│       ├── gpu.py                    # GPU memory management
│       └── logging.py                # Logging configuration
│
├── scripts/                          # Execution scripts
│   ├── data_pipeline.py              # End-to-end data processing
│   ├── train_models.py               # Model training orchestration
│   ├── run_backtest.py               # Backtesting execution
│   ├── generate_reports.py           # Report generation
│   └── experiments/                  # Experimental scripts
│       ├── hyperparameter_tuning.py
│       ├── sensitivity_analysis.py
│       └── ensemble_experiments.py
│
├── configs/                          # Configuration files
│   ├── data/                         # Data pipeline configs
│   │   ├── default.yaml
│   │   └── midcap400.yaml
│   ├── models/                       # Model configurations
│   │   ├── hrp_default.yaml
│   │   ├── lstm_default.yaml
│   │   └── gat_default.yaml
│   └── experiments/                  # Experiment configurations
│       ├── baseline_comparison.yaml
│       └── full_evaluation.yaml
│
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── unit/                         # Unit tests
│   │   ├── test_data/
│   │   ├── test_models/
│   │   └── test_evaluation/
│   ├── integration/                  # Integration tests
│   │   ├── test_pipeline/
│   │   └── test_backtest/
│   └── fixtures/                     # Test data fixtures
│       ├── sample_data.parquet
│       └── mock_responses/
│
├── notebooks/                        # Research notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_development.ipynb
│   ├── 03_backtesting_analysis.ipynb
│   └── 04_results_visualization.ipynb
│
└── docs/                             # Documentation
    ├── api/                          # API documentation
    ├── tutorials/                    # Usage tutorials
    ├── deployment/                   # Deployment guides
    └── research/                     # Research notes and papers
```

---
