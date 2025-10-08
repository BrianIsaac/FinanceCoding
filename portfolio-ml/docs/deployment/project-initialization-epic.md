# Epic 0: Project Initialization & Foundation Setup
## Portfolio Optimization with Machine Learning Techniques

**Epic Goal:** Establish complete project foundation with organized repository structure, development environment, configuration management, and initial codebase organization. This epic ensures all developers can immediately begin productive work with a consistent, well-structured development environment.

**Priority:** Critical - Must be completed before any other epics  
**Timeline:** Week 0 (before Epic 1)  
**Dependencies:** None  
**Enables:** All subsequent epics (1-4)

---

## Story 0.1: Repository Structure and Code Organization

**As a** developer,  
**I want** a well-organized repository structure with proper Python packaging,  
**so that** all code is maintainable, discoverable, and follows Python best practices.

### Acceptance Criteria

1. **Complete directory structure created** following technical architecture specification:
   ```
   portfolio-optimization-ml/
   ├── src/{config,data,models,evaluation,utils}/
   ├── configs/{data,models,experiments}/
   ├── tests/{unit,integration,fixtures}/
   ├── scripts/
   ├── notebooks/
   ├── data/{raw,processed,graphs}/
   └── docs/
   ```

2. **Python package structure established**:
   - All directories have `__init__.py` files
   - Proper import paths configured
   - PYTHONPATH setup in development environment

3. **Existing GAT implementation organized**:
   - Move GAT code from root to `src/models/gat/`
   - Preserve all functionality while improving structure
   - Update import statements throughout codebase

4. **Core module stubs created**:
   - Base model interfaces (`src/models/base/`)
   - Data pipeline stubs (`src/data/`)
   - Evaluation framework stubs (`src/evaluation/`)
   - Utility modules (`src/utils/`)

5. **Documentation structure established**:
   - Move existing docs to proper locations
   - Create placeholders for missing documentation
   - Establish documentation standards and templates

### Implementation Tasks

```bash
# 1. Create complete directory structure
mkdir -p {data/{raw/{membership,stooq,yfinance},processed,graphs/snapshots},src/{config,data/{collectors,processors,loaders},models/{base,hrp,lstm,gat,baselines},evaluation/{backtest,metrics,validation,reporting},utils},scripts/{experiments},configs/{data,models,experiments},tests/{unit/{test_data,test_models,test_evaluation},integration/{test_pipeline,test_backtest},fixtures},notebooks,docs/{api,tutorials,deployment,research}}

# 2. Create __init__.py files
find src -type d -exec touch {}/__init__.py \;
find tests -type d -exec touch {}/__init__.py \;

# 3. Move and organize existing code
# [Detailed in Story 0.2]

# 4. Create module stubs
# [Detailed in Story 0.3]
```

---

## Story 0.2: Existing Codebase Refactoring and Organization

**As a** developer,  
**I want** existing GAT implementation and data pipeline code properly organized,  
**so that** I can build upon existing work without technical debt.

### Acceptance Criteria

1. **GAT implementation relocated and organized**:
   - Move GAT model code to `src/models/gat/`
   - Separate concerns: model architecture, training, graph construction
   - Maintain all existing functionality
   - Update all import statements

2. **Data pipeline code organized**:
   - Move data collection scripts to `src/data/collectors/`
   - Move processing utilities to `src/data/processors/`
   - Create clean interfaces for data loading
   - Preserve Wikipedia scraping and multi-source integration

3. **Graph construction utilities organized**:
   - Move to `src/models/gat/graph_builder.py`
   - Clean separation of MST, TMFG, k-NN methods
   - Maintain correlation-based edge construction

4. **Evaluation code organized**:
   - Move backtesting logic to `src/evaluation/backtest/`
   - Move performance metrics to `src/evaluation/metrics/`
   - Clean interfaces for rolling validation

5. **All existing functionality preserved**:
   - No breaking changes to core algorithms
   - All existing notebooks continue to work
   - Comprehensive testing of refactored code

### Implementation Guide

```python
# Example refactoring approach for GAT model
# Before: gat_model.py in root
# After: src/models/gat/

# src/models/gat/__init__.py
from .model import GATPortfolioModel
from .gat_model import GATLayer, MultiHeadGATLayer
from .graph_builder import GraphBuilder

# src/models/gat/model.py
from ..base.portfolio_model import PortfolioModel

class GATPortfolioModel(PortfolioModel):
    # Implement unified interface while preserving GAT functionality
    pass
```

---

## Story 0.3: Development Environment and Dependency Management

**As a** developer,  
**I want** a consistent, reproducible development environment with uv and proper dependency management,  
**so that** all developers work with identical configurations and dependencies.

### Acceptance Criteria

1. **uv environment management configured**:
   - `pyproject.toml` with all dependency groups
   - PyTorch with CUDA 11.8 support configured
   - All ML dependencies (torch-geometric, scikit-learn, etc.)
   - Development tools (pytest, black, mypy, etc.)

2. **GPU environment verified**:
   - CUDA 11.8+ installation verified
   - PyTorch GPU acceleration working
   - 12GB VRAM memory constraints respected
   - GPU memory optimization utilities available

3. **Python environment properly configured**:
   - Python 3.9+ with uv management
   - PYTHONPATH configured for src/ imports
   - Environment variables managed with .env
   - Development vs production configurations

4. **Code quality tools integrated**:
   - Black code formatting (100 char line length)
   - isort import sorting
   - flake8 linting
   - mypy type checking
   - pre-commit hooks configured

5. **Testing framework established**:
   - pytest configuration with coverage
   - Test fixtures for common data
   - GPU vs CPU test selection
   - Integration test framework

### Dependency Groups

```toml
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
]

# ... other groups as specified in setup guide
```

---

## Story 0.4: Configuration Management with Hydra

**As a** developer,  
**I want** flexible configuration management using Hydra,  
**so that** I can easily manage different model configurations, experiments, and environments.

### Acceptance Criteria

1. **Hydra framework integrated**:
   - Main configuration structure (`configs/config.yaml`)
   - Hierarchical configuration loading
   - Override capabilities for experiments
   - Configuration validation

2. **Model configurations created**:
   - HRP default configuration (`configs/models/hrp_default.yaml`)
   - LSTM default configuration (`configs/models/lstm_default.yaml`) 
   - GAT default configuration (`configs/models/gat_default.yaml`)
   - Constraint configurations

3. **Data pipeline configurations**:
   - S&P MidCap 400 universe configuration
   - Data source configurations (Stooq, Yahoo Finance)
   - Processing parameters and quality controls
   - Storage configurations

4. **Experiment configurations**:
   - Baseline comparison experiment
   - Full evaluation experiment
   - Backtesting parameters
   - Performance evaluation metrics

5. **Configuration management utilities**:
   - Configuration validation functions
   - Environment-specific overrides
   - Logging configuration
   - GPU and memory management settings

### Configuration Structure

```yaml
# configs/config.yaml
defaults:
  - data: midcap400
  - models: [hrp_default, lstm_default, gat_default]
  - experiment: baseline_comparison
  - _self_

project_name: portfolio_optimization_ml
output_dir: outputs
seed: 42

gpu:
  enabled: true
  device_id: 0
  memory_limit_gb: 11.0
```

---

## Story 0.5: Base Model Interfaces and Constraint System

**As a** developer,  
**I want** unified base interfaces and constraint system,  
**so that** all ML models implement consistent APIs and respect identical portfolio constraints.

### Acceptance Criteria

1. **Abstract base model interface created**:
   - `PortfolioModel` abstract base class
   - Standard `fit()`, `predict_weights()`, `get_model_info()` methods
   - Type hints and comprehensive docstrings
   - Integration with Hydra configuration

2. **Unified constraint system implemented**:
   - `PortfolioConstraints` dataclass
   - Long-only, position limits, turnover controls
   - Transaction cost modeling
   - Constraint validation and enforcement

3. **Model factory and registry**:
   - Dynamic model loading from configuration
   - Model registration system
   - Error handling for invalid configurations
   - Model metadata and versioning

4. **Integration testing framework**:
   - Base model interface compliance tests
   - Constraint enforcement validation
   - Configuration loading tests
   - Mock model implementations for testing

### Interface Definition

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import pandas as pd

@dataclass
class PortfolioConstraints:
    long_only: bool = True
    top_k_positions: Optional[int] = None
    max_position_weight: float = 0.10
    max_monthly_turnover: float = 0.20
    transaction_cost_bps: float = 10.0

class PortfolioModel(ABC):
    def __init__(self, constraints: PortfolioConstraints):
        self.constraints = constraints
        
    @abstractmethod
    def fit(self, returns: pd.DataFrame, universe: List[str], 
            fit_period: Tuple[pd.Timestamp, pd.Timestamp]) -> None:
        """Train model on historical data."""
        pass
        
    @abstractmethod 
    def predict_weights(self, date: pd.Timestamp, universe: List[str]) -> pd.Series:
        """Generate portfolio weights for rebalancing date."""
        pass
        
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata for analysis."""
        pass
```

---

## Story 0.6: Initial Testing and Validation Framework

**As a** developer,  
**I want** comprehensive testing framework with data fixtures,  
**so that** all code is properly tested and validated throughout development.

### Acceptance Criteria

1. **Test framework configured**:
   - pytest with coverage reporting
   - Test discovery and execution
   - Parallel test execution
   - GPU vs CPU test markers

2. **Test fixtures created**:
   - Sample returns data generator
   - Mock universe data
   - Configuration fixtures
   - Temporary data directories

3. **Unit test structure established**:
   - Model unit tests (`tests/unit/test_models/`)
   - Data pipeline unit tests (`tests/unit/test_data/`)
   - Configuration unit tests (`tests/unit/test_config/`)
   - Utility unit tests (`tests/unit/test_utils/`)

4. **Integration test framework**:
   - End-to-end pipeline tests
   - Model integration tests
   - Configuration integration tests
   - Memory and performance tests

5. **Continuous testing setup**:
   - Pre-commit hook integration
   - Coverage reporting (80%+ target)
   - Performance benchmarks
   - Memory usage monitoring

### Test Configuration

```ini
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
addopts = 
    -v 
    --cov=src
    --cov-report=html
    --cov-fail-under=80
markers =
    unit: Unit tests
    integration: Integration tests  
    slow: Slow tests
    gpu: Tests requiring GPU
```

---

## Story 0.7: Documentation and Developer Onboarding

**As a** new developer,  
**I want** comprehensive setup documentation and onboarding materials,  
**so that** I can quickly become productive in the development environment.

### Acceptance Criteria

1. **Complete setup documentation**:
   - Step-by-step environment setup guide
   - Dependency installation instructions
   - GPU setup and verification
   - Common troubleshooting scenarios

2. **Development workflow documentation**:
   - Code organization principles
   - Git workflow and branching strategy
   - Testing procedures and standards
   - Code review guidelines

3. **API documentation foundation**:
   - Docstring standards established
   - API documentation generation setup
   - Module and class documentation
   - Usage examples and tutorials

4. **Onboarding verification**:
   - Environment verification notebook
   - "Hello World" example implementations
   - Configuration loading examples
   - Basic model training examples

5. **IDE configuration**:
   - VS Code workspace configuration
   - Python path and interpreter setup
   - Debugging configurations
   - Extension recommendations

### Verification Checklist

- [ ] New developer can setup environment in <30 minutes
- [ ] All dependencies install without errors
- [ ] GPU functionality verified and working
- [ ] Tests pass on clean installation
- [ ] Configuration loading works correctly
- [ ] Basic model interfaces functional

---

## Epic 0 Success Criteria

### Technical Deliverables

1. **Organized Repository Structure**: Complete directory hierarchy with proper Python packaging
2. **Working Development Environment**: uv, Hydra, PyTorch GPU, all dependencies configured
3. **Refactored Existing Code**: GAT implementation and data pipeline properly organized
4. **Base Interfaces**: Model interfaces and constraint system implemented
5. **Testing Framework**: pytest, fixtures, unit and integration test structure
6. **Documentation**: Complete setup guide, API documentation foundation

### Validation Requirements

1. **Environment Verification**: New developer setup completes successfully
2. **Code Organization**: All imports work, no circular dependencies
3. **Test Suite**: All tests pass, coverage >80%
4. **Configuration Loading**: Hydra configs load without errors
5. **GPU Functionality**: PyTorch GPU acceleration verified
6. **Model Interfaces**: Base classes implement correctly

### Timeline and Dependencies

- **Duration**: 3-5 days (before Epic 1 begins)
- **Parallel Work**: Stories 0.1-0.3 can be done in parallel
- **Critical Path**: Environment setup → Code organization → Base interfaces
- **Blocking**: Epic 1 cannot begin until Epic 0 is complete

### Risk Mitigation

1. **GPU Setup Issues**: Provide detailed CUDA installation guide
2. **Dependency Conflicts**: Use uv lock file for reproducible installs  
3. **Code Refactoring**: Comprehensive testing before and after moves
4. **Configuration Complexity**: Start with simple configs, expand gradually

This Epic 0 provides the complete foundation needed to address the critical gaps identified in the PO checklist evaluation, ensuring all subsequent development can proceed smoothly with proper organization, tooling, and documentation.