# Reproducible Research Package Guide

## Overview

This guide provides comprehensive instructions for reproducing all research results presented in the ML-based portfolio optimization study. The package ensures complete reproducibility through deterministic execution, exact environment specification, and automated experiment orchestration.

## 1. Environment Setup and Dependencies

### 1.1 System Requirements

**Hardware Requirements:**
- **CPU**: Modern multi-core processor (Intel/AMD)
- **RAM**: 32GB minimum, 64GB recommended
- **GPU**: NVIDIA GPU with 12GB+ VRAM (GTX 4070 Ti SUPER or equivalent)
- **Storage**: 100GB free space for data and results

**Software Requirements:**
- **Operating System**: Linux (Ubuntu 22.04+), macOS (12.0+), or Windows 11
- **Python**: 3.9 to 3.12 (3.11 recommended)
- **CUDA**: 11.8 or 12.1 (for GPU acceleration)
- **Git**: Latest version for repository management

### 1.2 Python Environment Setup

**Using UV Package Manager (Recommended):**
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd portfolio-optimization-ml

# Create and activate virtual environment
uv venv --python 3.11
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows

# Install dependencies with exact versions
uv sync --frozen
```

**Alternative: Using pip with requirements.txt:**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or venv\Scripts\activate  # Windows

# Install exact dependencies
pip install -r requirements-frozen.txt
```

### 1.3 GPU Setup and Validation

**CUDA Installation Validation:**
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"

# Check GPU memory
nvidia-smi
```

**GPU Memory Management:**
The framework automatically manages GPU memory, but you can configure limits:
```yaml
# configs/environment/gpu_config.yaml
gpu_settings:
  max_memory_fraction: 0.9
  memory_growth: true
  device_id: 0  # Use specific GPU
```

## 2. Data Setup and Preparation

### 2.1 Data Sources and Collection

**Primary Data Sources:**
- **S&P MidCap 400 Membership**: Wikipedia historical data
- **Price Data**: Stooq API (primary), Yahoo Finance (fallback)
- **Market Data**: Risk-free rates, market indices

**Automated Data Collection:**
```bash
# Run complete data pipeline (requires ~2-4 hours)
python scripts/collect_data.py --config configs/data/midcap400.yaml

# Alternative: Download preprocessed data package
python scripts/download_data_package.py --verify-checksums
```

### 2.2 Data Validation and Quality Checks

**Data Integrity Validation:**
```bash
# Validate data completeness and quality
python scripts/validate_data.py --data-path data/processed/

# Generate data quality report
python scripts/generate_data_report.py --output reports/data_quality.html
```

**Expected Data Structure:**
```
data/
├── processed/
│   ├── prices.parquet          # OHLCV price data
│   ├── returns_daily.parquet   # Daily returns
│   ├── volume.parquet          # Trading volume
│   └── universe_calendar.parquet # Dynamic universe membership
├── external/
│   └── sp_midcap_400_history.csv
└── interim/
    └── collection_logs/
```

## 3. Experiment Configuration and Execution

### 3.1 Configuration System

**Hierarchical Configuration Structure:**
```
configs/
├── environment/
│   ├── development.yaml        # Development settings
│   ├── production.yaml         # Production settings
│   └── gpu_config.yaml         # GPU-specific configuration
├── models/
│   ├── hrp/
│   │   ├── default.yaml        # Default HRP parameters
│   │   └── sensitivity.yaml    # Parameter grid for sensitivity
│   ├── lstm/
│   │   ├── default.yaml
│   │   └── architecture_grid.yaml
│   └── gat/
│       ├── default.yaml
│       └── graph_config.yaml
├── evaluation/
│   ├── backtest_config.yaml    # Rolling backtest settings
│   ├── statistical_tests.yaml  # Statistical testing parameters
│   └── performance_metrics.yaml
└── experiments/
    ├── baseline_comparison.yaml
    ├── sensitivity_analysis.yaml
    └── full_evaluation.yaml
```

### 3.2 Experiment Orchestration Scripts

**Complete Research Reproduction:**
```bash
# Run full research pipeline (requires 8-24 hours depending on hardware)
python scripts/reproduce_research.py --experiment full_evaluation

# Monitor progress
tail -f logs/experiment_progress.log
```

**Individual Experiment Execution:**
```bash
# Baseline model comparison
python scripts/run_experiment.py --config experiments/baseline_comparison.yaml

# Sensitivity analysis
python scripts/run_experiment.py --config experiments/sensitivity_analysis.yaml

# Statistical significance testing
python scripts/run_experiment.py --config experiments/statistical_tests.yaml
```

### 3.3 Experiment Templates and Customization

**Creating Custom Experiments:**
```yaml
# experiments/custom_experiment.yaml
name: "Custom Portfolio Optimization Experiment"
description: "Custom configuration for specific research questions"

models:
  - name: "hrp"
    config: "configs/models/hrp/default.yaml"
    parameters:
      lookback_days: [252, 504]
      linkage_method: ["ward", "complete"]
  
  - name: "lstm"
    config: "configs/models/lstm/default.yaml"
    parameters:
      sequence_length: [60, 90]
      hidden_size: [128, 256]

evaluation:
  backtest:
    start_date: "2020-01-01"
    end_date: "2024-12-31"
    rebalance_frequency: "monthly"
  
  statistical_tests:
    significance_level: 0.05
    bootstrap_iterations: 10000
    
output:
  results_dir: "results/custom_experiment"
  generate_report: true
  export_formats: ["html", "pdf", "csv"]
```

## 4. Reproducibility Validation

### 4.1 Deterministic Execution

**Random Seed Management:**
All experiments use fixed seeds for reproducibility:
```python
# Set in all experiment scripts
import numpy as np
import torch
import random

def set_random_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Configuration Checksums:**
```bash
# Verify configuration file integrity
python scripts/verify_configs.py --generate-checksums
python scripts/verify_configs.py --validate-checksums checksums.yaml
```

### 4.2 Result Validation

**Expected Results Validation:**
```bash
# Compare results against reference outputs
python scripts/validate_results.py \
    --results results/full_evaluation/ \
    --reference reference_results/ \
    --tolerance 1e-6
```

**Performance Benchmarks:**
Reference performance metrics for validation:
```yaml
# reference_results/performance_benchmarks.yaml
sharpe_ratios:
  hrp_default: 0.847 ± 0.023
  lstm_default: 0.891 ± 0.031  
  gat_default: 0.863 ± 0.028

statistical_significance:
  lstm_vs_hrp: 
    p_value: 0.023
    effect_size: 0.34
  gat_vs_hrp:
    p_value: 0.156
    effect_size: 0.19
```

### 4.3 Environment Validation

**Dependency Verification:**
```bash
# Validate all dependencies are correctly installed
python scripts/validate_environment.py

# Check GPU setup and memory availability
python scripts/check_gpu_setup.py

# Test critical functionality
python -m pytest tests/integration/test_reproducibility.py -v
```

## 5. Data Processing Pipeline Documentation

### 5.1 Universe Construction Process

**S&P MidCap 400 Dynamic Universe:**
```python
# Reproduce universe construction
from src.data.universe import UniverseBuilder

# Initialize universe builder
builder = UniverseBuilder(
    start_date="2016-01-01",
    end_date="2024-12-31"
)

# Collect historical membership
membership_data = builder.collect_wikipedia_data()

# Generate dynamic universe calendar
universe_calendar = builder.build_calendar(
    rebalance_frequency="monthly"
)

# Validate against known index changes
validation_report = builder.validate_membership()
```

### 5.2 Data Collection and Gap Filling

**Multi-Source Data Pipeline:**
```python
# Reproduce data collection
from src.data.collectors import StooqCollector, YahooFinanceCollector
from src.data.processors import GapFiller, DataValidator

# Primary data collection (Stooq)
stooq = StooqCollector(api_key=None)  # Free tier
price_data = stooq.collect_universe_data(
    universe_calendar, 
    start_date="2016-01-01"
)

# Gap filling with Yahoo Finance
yahoo = YahooFinanceCollector()
filled_data = GapFiller.fill_gaps(
    primary_data=price_data,
    fallback_collector=yahoo
)

# Data quality validation
validator = DataValidator()
quality_report = validator.validate_data(filled_data)
```

### 5.3 Feature Engineering Pipeline

**Return and Risk Metrics Calculation:**
```python
# Generate features for model training
from src.data.processors import FeatureEngineer

engineer = FeatureEngineer()

# Calculate returns
daily_returns = engineer.calculate_returns(
    price_data, 
    method="simple"
)

# Generate risk metrics
risk_features = engineer.calculate_risk_metrics(
    returns=daily_returns,
    lookback_windows=[21, 63, 252]
)

# Create model-ready datasets
model_data = engineer.create_model_features(
    returns=daily_returns,
    prices=price_data,
    volume=volume_data
)
```

## 6. Model Training and Evaluation Pipeline

### 6.1 Individual Model Training

**HRP Model Training:**
```python
from src.models.hrp import HRPPortfolioModel

# Initialize with configuration
hrp_model = HRPPortfolioModel.from_config(
    "configs/models/hrp/default.yaml"
)

# Train on historical data
hrp_model.fit(
    returns=daily_returns,
    start_date="2020-01-01",
    end_date="2022-12-31"
)

# Generate predictions
weights = hrp_model.predict_weights(
    returns=test_returns
)
```

**LSTM Model Training:**
```python
from src.models.lstm import LSTMPortfolioModel

# GPU-optimized training
lstm_model = LSTMPortfolioModel.from_config(
    "configs/models/lstm/default.yaml"
)

# Memory-efficient training
lstm_model.fit(
    returns=daily_returns,
    sequence_length=60,
    batch_size=32,
    epochs=100,
    validation_split=0.2
)
```

**GAT Model Training:**
```python
from src.models.gat import GATPortfolioModel

# Graph-based training
gat_model = GATPortfolioModel.from_config(
    "configs/models/gat/default.yaml"
)

# Train with graph construction
gat_model.fit(
    returns=daily_returns,
    graph_method="mst",
    attention_heads=4
)
```

### 6.2 Rolling Backtest Execution

**Automated Rolling Validation:**
```python
from src.evaluation.backtest import RollingBacktestEngine

# Configure backtest engine
engine = RollingBacktestEngine(
    models=[hrp_model, lstm_model, gat_model],
    start_date="2023-01-01",
    end_date="2024-12-31",
    rebalance_frequency="monthly",
    training_window=756  # 3 years
)

# Execute rolling backtest
results = engine.run_backtest()

# Generate performance analytics
performance = engine.calculate_performance_metrics(results)
```

### 6.3 Statistical Analysis Execution

**Comprehensive Statistical Testing:**
```python
from src.evaluation.validation import StatisticalValidator

validator = StatisticalValidator(
    significance_level=0.05,
    bootstrap_iterations=10000
)

# Sharpe ratio significance testing
sharpe_tests = validator.test_sharpe_ratios(
    returns_dict=model_returns,
    method="jobson_korkie"
)

# Bootstrap confidence intervals
confidence_intervals = validator.bootstrap_confidence_intervals(
    performance_metrics,
    confidence_level=0.95
)

# Multiple comparison corrections
corrected_p_values = validator.apply_multiple_corrections(
    p_values, method="holm_sidak"
)
```

## 7. Results Generation and Reporting

### 7.1 Automated Report Generation

**Comprehensive Reporting Pipeline:**
```python
from src.evaluation.reporting import ComprehensiveReportGenerator

# Initialize report generator
generator = ComprehensiveReportGenerator(
    results_dir="results/full_evaluation",
    template_dir="templates/reports"
)

# Generate complete research report
generator.generate_comprehensive_report(
    performance_results=performance,
    statistical_tests=statistical_results,
    sensitivity_analysis=sensitivity_results,
    output_formats=["html", "pdf", "latex"]
)
```

### 7.2 Interactive Dashboard Generation

**Research Dashboard Creation:**
```bash
# Launch interactive dashboard
python scripts/launch_dashboard.py \
    --results results/full_evaluation/ \
    --port 8050

# Access at http://localhost:8050
```

### 7.3 Publication-Ready Export

**Academic Publication Formats:**
```python
# Generate LaTeX tables and figures
generator.export_publication_materials(
    output_dir="publication_materials/",
    formats={
        "tables": "latex",
        "figures": "pdf", 
        "statistics": "csv"
    }
)
```

## 8. Troubleshooting and Common Issues

### 8.1 Environment Issues

**CUDA/GPU Problems:**
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Memory Issues:**
```yaml
# Reduce memory usage in config
memory_optimization:
  batch_size: 16  # Reduce from default 32
  gradient_checkpointing: true
  mixed_precision: true
```

### 8.2 Data Issues

**Missing Data Handling:**
```bash
# Check data completeness
python scripts/diagnose_data_issues.py --data-path data/processed/

# Re-run data collection for specific periods
python scripts/collect_data.py --start-date 2020-01-01 --end-date 2020-12-31
```

**Universe Construction Problems:**
```bash
# Validate universe calendar
python scripts/validate_universe.py --calendar data/processed/universe_calendar.parquet

# Regenerate with different source
python scripts/rebuild_universe.py --source manual --config configs/data/manual_universe.yaml
```

### 8.3 Performance Issues

**Training Time Optimization:**
```yaml
# Enable performance optimizations
training_optimizations:
  compile_model: true  # PyTorch 2.0+
  channels_last: true  # Memory layout optimization
  cpu_threads: 8       # Parallel data loading
```

**Memory Optimization:**
```python
# Enable memory-efficient settings
torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
torch.set_float32_matmul_precision('medium')  # Faster matmul
```

## 9. Extension and Customization

### 9.1 Adding New Models

**Model Integration Template:**
```python
from src.models.base import PortfolioModel

class CustomPortfolioModel(PortfolioModel):
    def __init__(self, config: Dict):
        super().__init__(config)
        # Initialize custom model
        
    def fit(self, returns: pd.DataFrame, **kwargs):
        # Implement training logic
        pass
        
    def predict_weights(self, returns: pd.DataFrame) -> np.ndarray:
        # Implement prediction logic
        pass
```

### 9.2 Custom Evaluation Metrics

**Adding New Performance Metrics:**
```python
from src.evaluation.metrics import PerformanceAnalytics

class CustomMetrics(PerformanceAnalytics):
    def calculate_custom_ratio(self, returns: pd.Series) -> float:
        # Implement custom metric calculation
        return custom_value
```

### 9.3 Configuration Customization

**Environment-Specific Configurations:**
```yaml
# configs/environment/custom.yaml
name: "Custom Research Environment"

hardware:
  gpu_memory: "24GB"  # Adjust for different hardware
  cpu_cores: 16
  
optimization:
  batch_size: 64      # Larger batch for more powerful GPUs
  parallel_workers: 4
```

## 10. Validation and Quality Assurance

### 10.1 Automated Testing

**Comprehensive Test Suite:**
```bash
# Run full test suite
python -m pytest tests/ -v --cov=src --cov-report=html

# Test specific components
python -m pytest tests/unit/test_models/ -v
python -m pytest tests/integration/test_backtest/ -v
python -m pytest tests/test_reproducibility/ -v
```

### 10.2 Reproducibility Validation

**Cross-Platform Testing:**
```bash
# Validate results across different systems
python scripts/cross_platform_validation.py \
    --reference-platform ubuntu \
    --test-platforms macos,windows \
    --tolerance 1e-6
```

### 10.3 Performance Regression Testing

**Performance Benchmarking:**
```bash
# Run performance benchmarks
python scripts/benchmark_performance.py \
    --save-baseline results/performance_baseline.json

# Compare against baseline
python scripts/compare_performance.py \
    --baseline results/performance_baseline.json \
    --current results/latest_run/
```

## Summary

This reproducible research package provides complete instructions and automation for reproducing all research results. The framework ensures deterministic execution, exact environment specification, and comprehensive validation while maintaining flexibility for customization and extension.

Key features include:
- **Complete Environment Specification**: Exact dependency versions and system requirements
- **Automated Experiment Orchestration**: One-command reproduction of all results
- **Deterministic Execution**: Fixed seeds and reproducible configurations
- **Comprehensive Validation**: Result verification and performance benchmarking
- **Extensible Framework**: Easy customization and new model integration

Following this guide ensures complete reproducibility of research findings and enables reliable extension for future research directions.