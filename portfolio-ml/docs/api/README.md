# API Documentation Directory

This directory will contain API reference documentation and integration guides.

## Planned Contents

### Module Reference Documentation
- **Data Pipeline API**: Classes and functions for data collection, processing, and loading
- **Model API**: Base classes and interfaces for portfolio models (HRP, LSTM, GAT)
- **Evaluation API**: Backtesting engine, performance analytics, and statistical validation
- **Utility API**: Shared utilities for GPU management, I/O operations, and mathematical functions

### Integration Guides
- **Model Development Guide**: Creating new portfolio optimization models
- **Data Source Integration**: Adding new data providers and sources
- **Custom Evaluation Metrics**: Implementing additional performance measurements
- **Deployment Integration**: API interfaces for production systems

## Documentation Generation

API documentation will be automatically generated from source code docstrings using:
- **Sphinx**: Documentation generation framework
- **Google Style Docstrings**: Consistent documentation format
- **Type Hints**: Full type annotation for all public interfaces
- **Examples**: Usage examples for all major classes and functions

## Structure (Planned)

```
api/
├── data/                    # Data pipeline API reference
├── models/                  # Model interface documentation  
├── evaluation/              # Evaluation framework API
├── utils/                   # Utility module reference
├── examples/                # Code examples and tutorials
└── integration/             # Integration and deployment guides
```

## Target Audience

- API Consumers
- Integration Developers
- Third-party Tool Developers
- Advanced Users

## Generation Commands (Future)

```bash
# Generate API documentation
uv run sphinx-build -b html docs/api/_source docs/api/_build

# Update API reference
uv run sphinx-apidoc -o docs/api/_source src/

# Serve documentation locally
uv run python -m http.server -d docs/api/_build
```

## Related Documentation

- [Technical Architecture](../technical-architecture.md) - System design and module organization
- [Research Specifications](../research/) - Implementation details for ML models
- [Developer Setup](../tutorials/) - Development environment for API usage