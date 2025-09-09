# Contributing to ML Portfolio Optimization Research

We welcome contributions to this open-source machine learning portfolio optimization research framework! This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. We are committed to providing a welcoming and inclusive environment for all contributors.

## How to Contribute

### Reporting Issues

- Use the GitHub issue tracker to report bugs or request features
- Check existing issues before creating a new one
- Provide detailed information including steps to reproduce
- Include system information (OS, Python version, GPU specifications)

### Submitting Changes

1. **Fork the repository** and create a feature branch
2. **Follow coding standards** (see below)
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Submit a pull request** with detailed description

## Development Setup

### Prerequisites

- Python 3.9+ 
- NVIDIA GPU with 8GB+ VRAM (recommended)
- 16GB+ RAM

### Installation

```bash
# Clone your fork
git clone https://github.com/your-username/portfolio-optimization-ml.git
cd portfolio-optimization-ml

# Install development dependencies
uv sync --dev

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories  
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## Coding Standards

### Python Style

- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Maximum line length: 100 characters
- Use docstrings following Google format

### Documentation

- All public functions must have comprehensive docstrings
- Include usage examples in docstrings
- Update relevant documentation in `docs/` directory
- Add entry to CHANGELOG.md for significant changes

### Example Code Style

```python
def calculate_sharpe_ratio(
    returns: pd.Series, 
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """Calculate annualized Sharpe ratio for return series.
    
    Args:
        returns: Daily return series
        risk_free_rate: Annual risk-free rate (default: 2%)
        periods_per_year: Trading periods per year (default: 252)
        
    Returns:
        Annualized Sharpe ratio
        
    Example:
        >>> returns = pd.Series([0.01, -0.005, 0.02, 0.001])
        >>> sharpe = calculate_sharpe_ratio(returns)
        >>> print(f"Sharpe ratio: {sharpe:.3f}")
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
```

## Contributing Areas

### Research Contributions

- **New Models**: Implement additional ML approaches
- **Evaluation Methods**: Enhance statistical testing frameworks  
- **Data Sources**: Add support for alternative data
- **Benchmarks**: Implement additional baseline strategies

### Technical Improvements

- **Performance**: Optimize computational efficiency
- **Scalability**: Enhance support for larger universes
- **Reliability**: Improve error handling and robustness
- **Documentation**: Expand tutorials and examples

### Testing and Quality

- **Unit Tests**: Increase test coverage
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Benchmark and regression testing
- **Documentation Tests**: Validate examples and tutorials

## Pull Request Process

1. **Create Feature Branch**: `git checkout -b feature/your-feature-name`
2. **Make Changes**: Implement your contribution following coding standards
3. **Add Tests**: Ensure new functionality is thoroughly tested
4. **Update Documentation**: Add/update relevant documentation
5. **Run Tests**: Verify all tests pass locally
6. **Submit PR**: Open pull request with detailed description

### PR Description Template

```
## Description
Brief description of changes and motivation

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update

## Testing
- [ ] New tests added for new functionality
- [ ] All existing tests pass
- [ ] Manual testing performed

## Documentation  
- [ ] Code comments updated
- [ ] Documentation updated
- [ ] Changelog entry added
```

## Recognition

Contributors will be acknowledged in:
- AUTHORS.md file
- Release notes for significant contributions
- Academic publications (for research contributions)
- Project website and documentation

Thank you for contributing to advancing machine learning research in portfolio optimization!