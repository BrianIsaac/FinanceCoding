# Technical Assumptions

## Repository Structure: Monorepo

The project will use a monorepo structure with organized module hierarchy:
- `data/` - Raw and processed datasets (prices.parquet, volume.parquet, returns_daily.parquet)
- `src/models/` - Model implementations (GAT complete, LSTM and HRP to be formalized)
- `src/preprocessing/` - Data collection and cleaning utilities
- `src/evaluation/` - Backtesting and performance analytics frameworks
- `src/utils/` - Shared utilities and helper functions

## Service Architecture

**Monolithic Research Framework**: Single integrated Python application optimized for academic research and experimentation. All ML models, data processing, and evaluation components operate within unified codebase to facilitate rapid prototyping and consistent data flow. No distributed services or microservices architecture given local development constraints and research focus.

## Testing Requirements

**Unit + Integration Testing**: Implement comprehensive testing pyramid including:
- Unit tests for individual model components (HRP clustering, LSTM layers, GAT attention mechanisms)
- Integration tests for end-to-end pipeline validation (data ingestion → model training → portfolio generation)
- Performance regression tests to ensure model outputs remain consistent across code changes
- Data quality validation tests for input pipeline integrity

## Additional Technical Assumptions and Requests

- **Python 3.9+** with uv environment management for dependency handling
- **PyTorch** for deep learning implementations (GAT, LSTM) with CUDA support
- **scikit-learn** for traditional ML components (HRP clustering algorithms)
- **NetworkX** for graph analysis and construction in GAT module
- **pandas/numpy** for data manipulation and numerical computing
- **Parquet file format** for efficient storage and retrieval of historical price/volume panels
- **Jupyter Lab** for interactive development and model experimentation
- **Local filesystem-based data management** - no external database dependencies
- **GPU acceleration** optimized for RTX GeForce 5070Ti with 12GB VRAM constraints
- **Matplotlib/seaborn** for visualization and results presentation
- **Google-style docstrings** and comprehensive type hints following user preferences
