# Integration and Deployment Architecture

## Environment Management

The system uses `uv` for consistent dependency management across environments:

```toml
# pyproject.toml - Enhanced dependency configuration
[project]
name = "portfolio-optimization-ml"
version = "1.0.0"
requires-python = ">=3.9"
description = "ML-Based Portfolio Optimization Research Framework"

[dependency-groups]
core = [
    "numpy>=1.24.0",
    "pandas>=2.0.0", 
    "scipy>=1.10.0",
    "scikit-learn>=1.3.0",
]

deep-learning = [
    "torch>=2.0.0",
    "torch-geometric>=2.3.0",
    "torchvision>=0.15.0",
]

data-processing = [
    "pyarrow>=12.0.0",        # Parquet I/O
    "yfinance>=0.2.18",       # Yahoo Finance data
    "lxml>=4.9.0",            # XML parsing for web scraping
    "requests>=2.31.0",       # HTTP requests
]

optimization = [
    "cvxpy>=1.3.0",           # Convex optimization
    "cvxopt>=1.3.0",          # Optimization solvers
]

analysis = [
    "matplotlib>=3.7.0",     # Plotting
    "seaborn>=0.12.0",        # Statistical visualization
    "plotly>=5.15.0",         # Interactive plots
    "jupyter>=1.0.0",         # Notebook environment
]

testing = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-xdist>=3.3.0",   # Parallel testing
]

[tool.uv.sources]
torch = [
    { index = "pytorch-cu118", marker = "sys_platform == 'linux'" },
    { index = "pytorch-cu118", marker = "sys_platform == 'win32'" },
]

[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"
explicit = true
```

## Configuration Management

Hierarchical configuration system supporting experiment management:

```python
# src/config/base.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml

@dataclass 
class DataConfig:
    data_path: str = "data/"
    start_date: str = "2016-01-01"
    end_date: str = "2024-12-31"
    universe: str = "sp_midcap_400"
    rebalance_frequency: str = "M"  # Monthly
    lookback_days: int = 252
    min_history_days: int = 100

@dataclass
class ModelConfig:
    name: str
    hyperparameters: Dict = field(default_factory=dict)
    training_config: Dict = field(default_factory=dict)
    constraint_config: Dict = field(default_factory=dict)

@dataclass 
class BacktestConfig:
    training_months: int = 36
    validation_months: int = 12
    test_months: int = 12
    step_months: int = 12
    transaction_cost_bps: float = 10.0
    
@dataclass
class ExperimentConfig:
    experiment_name: str
    data: DataConfig = field(default_factory=DataConfig)
    models: List[ModelConfig] = field(default_factory=list)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    output_path: str = "results/"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "ExperimentConfig":
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
```

## Execution Pipeline Architecture

Modular execution system supporting both interactive and batch processing:

```python
# scripts/orchestrator.py
class ExperimentOrchestrator:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = self._setup_logging()
        
    def run_full_experiment(self) -> ExperimentResults:
        """Execute complete experiment pipeline."""
        
        # Stage 1: Data preparation
        self.logger.info("Stage 1: Data pipeline execution")
        data_pipeline = DataPipeline(self.config.data)
        datasets = data_pipeline.execute()
        
        # Stage 2: Model training and validation  
        self.logger.info("Stage 2: Model training")
        trained_models = {}
        for model_config in self.config.models:
            model = self._initialize_model(model_config)
            trained_models[model_config.name] = model
            
        # Stage 3: Rolling backtesting
        self.logger.info("Stage 3: Rolling backtesting")
        backtest_engine = RollingBacktestEngine(self.config.backtest)
        backtest_results = backtest_engine.run_backtest(trained_models, datasets)
        
        # Stage 4: Performance analysis
        self.logger.info("Stage 4: Performance analysis")
        analyzer = PerformanceAnalyzer()
        analysis_results = analyzer.analyze_results(backtest_results)
        
        # Stage 5: Report generation
        self.logger.info("Stage 5: Report generation")
        report_generator = ReportGenerator(self.config.output_path)
        report_generator.generate_comprehensive_report(analysis_results)
        
        return ExperimentResults(
            backtest_results=backtest_results,
            analysis_results=analysis_results,
            model_configs=self.config.models
        )
```

---
