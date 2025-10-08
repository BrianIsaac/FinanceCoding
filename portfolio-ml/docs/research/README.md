# Research Directory

This directory contains research methodology, technical specifications, and implementation details for machine learning models.

## Contents

### [hrp-implementation-spec.md](./hrp-implementation-spec.md)
Technical specification for Hierarchical Risk Parity implementation:
- Theoretical foundation and mathematical framework
- Data integration with existing pipeline
- Algorithm implementation details
- Performance optimization and GPU considerations
- Testing and validation procedures

### [lstm-implementation-spec.md](./lstm-implementation-spec.md)
Technical specification for LSTM temporal network implementation:
- Sequence-to-sequence architecture design
- Feature engineering pipeline
- Memory optimization for 12GB VRAM constraints
- Training strategies and hyperparameter tuning
- Integration with portfolio optimization framework

## Research Framework

Both specifications are designed to integrate with the unified evaluation framework:
- Consistent constraint application across all ML approaches
- Fair comparison methodology with identical backtesting protocols
- GPU-optimized implementations within hardware constraints
- Integration with existing data pipeline and dynamic universe management

## Implementation Order

1. **HRP Implementation**: Start with HRP as it's the least computationally intensive
2. **LSTM Implementation**: Implement after data pipeline is stable
3. **GAT Enhancement**: Leverage existing GAT implementation as baseline

## Target Audience

- Machine Learning Engineers
- Quantitative Developers  
- Research Scientists
- Portfolio Managers (for methodology understanding)

## Related Documentation

- [Technical Architecture](../technical-architecture.md) - Overall system design
- [Developer Setup Guide](../tutorials/developer-setup-guide.md) - Environment configuration
- [Product Requirements](../prd.md) - Success criteria and performance targets