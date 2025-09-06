# Documentation Standards

## Google-Style Docstrings

All functions, classes, and modules must use Google-style docstrings with comprehensive type hints.

### Function Example

```python
def calculate_portfolio_weights(
    returns: pd.DataFrame,
    constraints: PortfolioConstraints,
    lookback_days: int = 252
) -> pd.Series:
    """
    Calculate optimal portfolio weights using specified constraints.
    
    This function implements portfolio optimization with risk constraints
    and regulatory compliance requirements.
    
    Args:
        returns: Historical returns DataFrame with datetime index and asset columns
        constraints: Portfolio constraints including position limits and turnover
        lookback_days: Number of days to use for optimization window
        
    Returns:
        Portfolio weights as pandas Series with asset tickers as index.
        Weights sum to 1.0 and satisfy all constraints.
        
    Raises:
        ValueError: If returns data is insufficient or constraints are invalid
        
    Example:
        >>> returns = pd.DataFrame(...)  # Historical returns data
        >>> constraints = PortfolioConstraints(max_position_weight=0.1)
        >>> weights = calculate_portfolio_weights(returns, constraints)
        >>> print(weights.sum())  # Should be 1.0
    """
```

### Class Example

```python
class PortfolioModel(ABC):
    """
    Abstract base class for portfolio optimization models.
    
    This class defines the unified interface that all portfolio models must
    implement, ensuring consistent APIs across different ML approaches.
    
    Attributes:
        constraints: Portfolio constraints configuration
        is_fitted: Boolean indicating if model has been trained
        
    Example:
        >>> class MyModel(PortfolioModel):
        ...     def fit(self, returns, universe, fit_period):
        ...         # Implementation here
        ...         pass
    """
```

### Module Example

```python
"""
Portfolio constraint system for unified risk management.

This module provides constraint enforcement and validation utilities
that ensure all portfolio models comply with risk management requirements
and regulatory constraints.

Key components:
    - ConstraintEngine: Main constraint enforcement class
    - TurnoverConstraints: Turnover-based constraint configuration
    - RiskConstraints: Risk-based constraint configuration
    
Example:
    >>> from src.models.base.constraints import ConstraintEngine
    >>> engine = ConstraintEngine()
    >>> constrained_weights = engine.apply_constraints(raw_weights)
"""
```

## Type Hints

- Use comprehensive type hints for all function parameters and return values
- Import types from `typing` module: `List`, `Dict`, `Optional`, `Union`, `Tuple`
- Use `pd.DataFrame`, `pd.Series`, `np.ndarray` for data structures
- Use `datetime` for date/time parameters
- Use `PathLike` for file path parameters

## ML-Specific Documentation Patterns

### Configuration Classes

```python
@dataclass
class GATConfig(ModelConfig):
    """Graph Attention Network model configuration.
    
    Extends base ModelConfig with GAT-specific parameters for graph neural
    network training and portfolio optimization applications.
    
    Attributes:
        hidden_dim: Hidden layer dimension for graph convolutions
        num_heads: Number of attention heads in each layer
        num_layers: Number of GAT layers in the network
        node_features: List of node feature names to extract
        edge_features: List of edge feature names to compute
        
    Example:
        >>> config = GATConfig(hidden_dim=128, num_heads=12)
        >>> model = GATModel(config)
        >>> model.fit(graph_data, returns_data)
    """
```

### ML Pipeline Functions

```python
def create_portfolio_graph(
    returns: pd.DataFrame,
    features: List[str],
    correlation_threshold: float = 0.3
) -> torch_geometric.data.Data:
    """Create portfolio graph for GAT model training.
    
    Constructs graph representation of portfolio assets with node features
    representing asset characteristics and edge features representing
    asset relationships (correlations, sector similarities).
    
    Args:
        returns: Asset returns with datetime index and asset columns
        features: List of node feature names to include
        correlation_threshold: Minimum correlation for edge creation
        
    Returns:
        PyTorch Geometric Data object with:
        - x: Node features tensor [num_assets, num_features]
        - edge_index: Graph connectivity [2, num_edges]  
        - edge_attr: Edge features tensor [num_edges, edge_dim]
        
    Raises:
        ValueError: If correlation_threshold not in [0, 1] range
        KeyError: If required features not found in returns data
        
    Example:
        >>> returns = load_asset_returns('2020-01-01', '2024-01-01')
        >>> features = ['volatility', 'momentum', 'market_cap']
        >>> graph = create_portfolio_graph(returns, features, 0.4)
        >>> print(graph.x.shape)  # [400, 3] for MidCap 400 universe
    """
```

### Model Training Functions

```python
def train_portfolio_model(
    model: PortfolioModel,
    config: ModelConfig,
    train_data: Dict[str, Any],
    val_data: Optional[Dict[str, Any]] = None
) -> TrainingResults:
    """Train portfolio optimization model with validation.
    
    Implements unified training loop for all portfolio models (HRP, LSTM, GAT)
    with consistent metrics tracking, early stopping, and checkpoint saving.
    
    Args:
        model: Portfolio model instance implementing PortfolioModel interface
        config: Model configuration with training hyperparameters
        train_data: Training data dictionary with required keys:
            - 'returns': Asset returns DataFrame
            - 'features': Feature DataFrame (if applicable)
            - 'graph': Graph data (for GAT models)
        val_data: Optional validation data with same structure
        
    Returns:
        TrainingResults object containing:
        - training_loss: List of training losses by epoch
        - validation_metrics: Validation metrics by epoch
        - best_epoch: Epoch with best validation performance
        - model_checkpoint: Path to saved best model
        
    Example:
        >>> model = GATModel(gat_config)
        >>> results = train_portfolio_model(model, config, train_data, val_data)
        >>> print(f"Best Sharpe ratio: {results.best_sharpe:.3f}")
    """
```

## Documentation Requirements

1. **All public functions** must have complete docstrings
2. **All classes** must have class-level docstrings  
3. **All modules** must have module-level docstrings
4. **Complex algorithms** should include mathematical notation or references
5. **Examples** should be provided for non-trivial functionality
6. **ML models** must document architecture, training process, and expected inputs/outputs
7. **Configuration classes** must document all parameters and provide usage examples
8. **Graph operations** must specify tensor dimensions and graph structure expectations