# LSTM Portfolio Model Documentation

## Overview

The LSTM (Long Short-Term Memory) Portfolio Model is a deep learning approach to portfolio optimization that leverages temporal patterns in financial data to predict future returns and construct optimal portfolios. This model is designed for the S&P MidCap 400 universe and integrates seamlessly with the existing portfolio construction framework.

## Key Features

- **Temporal Pattern Recognition**: 60-day lookback windows capture complex temporal dependencies
- **Multi-Head Attention**: Focuses on relevant historical periods for improved predictions
- **Memory Optimization**: GPU memory management for 400+ asset universes within 12GB VRAM
- **Constraint Integration**: Full compliance with portfolio constraints (long-only, position limits, turnover)
- **Production Ready**: Real-time inference capabilities for monthly rebalancing

## Architecture

### Network Structure

```python
class LSTMNetwork(nn.Module):
    """
    Sequence-to-sequence LSTM with attention mechanism.
    
    Components:
    - LSTM layers (configurable depth)
    - Multi-head attention (8 heads default)
    - Output projection layer
    - Dropout and batch normalization
    """
```

### Key Components

1. **LSTM Backbone**: 2-layer LSTM with 128 hidden units (configurable)
2. **Attention Layer**: 8-head attention mechanism for temporal focus
3. **Output Projection**: Linear layer mapping to return predictions
4. **Regularization**: Dropout (30%) and batch normalization for stability

### Loss Function

Custom Sharpe ratio loss for direct portfolio optimization:

```python
def sharpe_loss(predicted_returns, actual_returns, portfolio_weights):
    """Compute negative Sharpe ratio as loss for direct optimization."""
    portfolio_returns = (portfolio_weights * actual_returns).sum(dim=1)
    excess_returns = portfolio_returns - risk_free_rate
    sharpe_ratio = excess_returns.mean() / excess_returns.std()
    return -sharpe_ratio  # Negative for minimization
```

## Configuration

### Default Configuration

```yaml
# LSTM architecture
lstm_config:
  hidden_size: 128
  num_layers: 2
  dropout: 0.3
  sequence_length: 60
  prediction_horizon: 21

# Training configuration  
training_config:
  learning_rate: 0.001
  batch_size: 32
  epochs: 100
  patience: 15
  max_memory_gb: 10.0
  use_mixed_precision: true
  gradient_accumulation_steps: 2

# Portfolio configuration
portfolio_config:
  lookback_days: 756
  rebalancing_frequency: monthly
  risk_aversion: 1.0
  use_markowitz_layer: true
```

### Memory Optimization Settings

For different GPU memory constraints:

#### 8GB GPU (RTX 3070)
```yaml
training_config:
  max_memory_gb: 6.0
  batch_size: 16
  gradient_accumulation_steps: 4
  use_mixed_precision: true
```

#### 12GB GPU (RTX 4070 Ti)  
```yaml
training_config:
  max_memory_gb: 10.0
  batch_size: 32
  gradient_accumulation_steps: 2
  use_mixed_precision: true
```

#### 24GB GPU (RTX 4090)
```yaml
training_config:
  max_memory_gb: 20.0
  batch_size: 64
  gradient_accumulation_steps: 1
  use_mixed_precision: false
```

## Hyperparameter Tuning Guidelines

### Architecture Tuning

#### Hidden Size (`hidden_size`)
- **Range**: 32-256
- **Default**: 128
- **Guidance**: 
  - Smaller (64): Faster training, less overfitting, may miss complex patterns
  - Larger (256): More capacity, risk of overfitting, requires more memory
  - Rule of thumb: ~1/3 to 1/2 of universe size

#### Number of Layers (`num_layers`)
- **Range**: 1-4  
- **Default**: 2
- **Guidance**:
  - 1 layer: Fastest, good for smaller universes (<100 assets)
  - 2 layers: Good balance for most use cases
  - 3+ layers: Only for very large universes (500+ assets)

#### Dropout Rate (`dropout`)
- **Range**: 0.1-0.5
- **Default**: 0.3
- **Guidance**:
  - Lower (0.1-0.2): Less regularization, may overfit
  - Higher (0.4-0.5): Strong regularization, may underfit
  - Increase if validation loss plateaus above training loss

### Training Tuning

#### Learning Rate (`learning_rate`)
- **Range**: 1e-5 to 1e-2
- **Default**: 1e-3
- **Guidance**:
  - Start with 1e-3 and adjust based on training curves
  - If loss oscillates: reduce by factor of 3-10
  - If training too slow: increase by factor of 2-3
  - Use learning rate scheduler for fine-tuning

#### Batch Size (`batch_size`)
- **Range**: 8-128 (memory dependent)
- **Default**: 32
- **Guidance**:
  - Larger batches: More stable gradients, better GPU utilization
  - Smaller batches: More gradient updates, may converge faster
  - Optimize based on memory constraints and gradient accumulation

#### Sequence Length (`sequence_length`)
- **Range**: 20-120 trading days
- **Default**: 60 days (~3 months)
- **Guidance**:
  - Shorter (20-40): Less memory, faster training, may miss long-term patterns
  - Longer (80-120): Captures longer patterns, more memory intensive
  - Consider market regime stability and rebalancing frequency

### Portfolio Integration Tuning

#### Risk Aversion (`risk_aversion`)
- **Range**: 0.5-5.0
- **Default**: 1.0
- **Guidance**:
  - Lower (0.5-0.8): More aggressive, higher expected returns and risk
  - Higher (2.0-5.0): More conservative, lower risk but may sacrifice returns
  - Adjust based on risk tolerance and market conditions

#### Prediction Horizon (`prediction_horizon`)
- **Range**: 5-63 trading days
- **Default**: 21 days (monthly)
- **Guidance**:
  - Should match rebalancing frequency
  - Shorter horizons: More reactive to market changes
  - Longer horizons: More stable but less responsive

## Performance Optimization

### Memory Optimization

1. **Mixed Precision Training**: Use FP16 to reduce memory by ~50%
2. **Gradient Accumulation**: Simulate larger batches without memory cost  
3. **Batch Size Optimization**: Automatically find optimal batch size for hardware
4. **Gradient Checkpointing**: Trade compute for memory in deep networks

### Training Speed Optimization

1. **Learning Rate Scheduling**: ReduceLROnPlateau for faster convergence
2. **Early Stopping**: Prevent overfitting and reduce training time
3. **Data Loading**: Optimized data loaders with proper num_workers
4. **GPU Utilization**: Pin memory and non-blocking transfers

### Inference Optimization

1. **Model Compilation**: TorchScript compilation for production deployment
2. **Batch Prediction**: Process multiple dates simultaneously when possible
3. **Caching**: Cache model states between similar prediction requests
4. **Quantization**: Post-training quantization for memory-constrained deployment

## Usage Examples

### Basic Training

```python
from src.models.lstm.model import LSTMPortfolioModel, LSTMModelConfig
from src.models.base.constraints import PortfolioConstraints

# Configure constraints
constraints = PortfolioConstraints(
    long_only=True,
    top_k_positions=50,
    max_position_weight=0.08,
    max_monthly_turnover=0.20
)

# Configure model
config = LSTMModelConfig()
model = LSTMPortfolioModel(constraints=constraints, config=config)

# Train model
model.fit(
    returns=returns_data,
    universe=asset_universe,
    fit_period=(start_date, end_date)
)

# Generate portfolio weights
weights = model.predict_weights(
    date=rebalance_date,
    universe=current_universe
)
```

### Custom Configuration

```python
config = LSTMModelConfig()

# Customize architecture
config.lstm_config.hidden_size = 64
config.lstm_config.num_layers = 1
config.lstm_config.dropout = 0.2

# Customize training
config.training_config.learning_rate = 0.0005
config.training_config.epochs = 50
config.training_config.batch_size = 16

# Memory optimization
config.training_config.max_memory_gb = 8.0
config.training_config.use_mixed_precision = True

model = LSTMPortfolioModel(constraints=constraints, config=config)
```

### Backtesting Integration

```python
from src.evaluation.backtest.engine import BacktestEngine, BacktestConfig

# Configure backtest
backtest_config = BacktestConfig(
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 1, 1),
    rebalance_frequency="M",
    initial_capital=1000000.0
)

# Run backtest
engine = BacktestEngine(backtest_config)
results = engine.run_backtest(model, returns_data, universe_data)

# Analyze results
performance_metrics = results["performance_metrics"]
print(f"Sharpe Ratio: {performance_metrics['sharpe_ratio']:.3f}")
print(f"Annual Return: {performance_metrics['annualized_return']:.3f}")
```

### Experiment Framework

```python
from src.experiments import run_experiment_from_config

# Run comprehensive experiment
results = run_experiment_from_config("configs/experiments/lstm_comparison.yaml")

# Compare with baselines
for model_name, model_results in results["models"].items():
    best_variant = model_results["best_variant"]
    best_sharpe = model_results["best_sharpe"]
    print(f"{model_name}: {best_variant} (Sharpe: {best_sharpe:.3f})")
```

## Troubleshooting

### Common Issues

#### Out of Memory Errors
```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Reduce batch_size: Try 16, 8, or 4
2. Enable mixed precision: `use_mixed_precision: true`
3. Increase gradient accumulation: `gradient_accumulation_steps: 4`
4. Reduce sequence length: Try 40 or 30 days
5. Reduce hidden_size: Try 64 or 32

#### Slow Convergence
```
Validation loss not improving after many epochs
```

**Solutions**:
1. Increase learning rate: Try 0.003 or 0.005
2. Reduce dropout: Try 0.2 or 0.1
3. Add learning rate scheduling
4. Check data quality and preprocessing
5. Increase model capacity (hidden_size)

#### Poor Performance
```
Sharpe ratio significantly worse than baselines
```

**Solutions**:
1. Increase training data: Use longer historical periods
2. Tune hyperparameters systematically
3. Check feature engineering and data preprocessing
4. Verify constraint implementation
5. Consider ensemble methods

#### Training Instability
```
Loss exploding or oscillating wildly
```

**Solutions**:
1. Reduce learning rate: Try 0.0001
2. Add gradient clipping: Already implemented
3. Check data normalization
4. Reduce sequence length
5. Increase dropout for regularization

### Performance Debugging

#### Memory Profiling
```python
import torch

# Monitor GPU memory during training
torch.cuda.empty_cache()
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
```

#### Training Monitoring
```python
# Enable detailed logging
import logging
logging.getLogger('src.models.lstm').setLevel(logging.DEBUG)

# Monitor loss curves and training metrics
# Use tensorboard or wandb for visualization
```

## Best Practices

### Data Preparation
1. **Clean Data**: Remove or interpolate missing values appropriately
2. **Outlier Handling**: Cap extreme returns to prevent training instability  
3. **Normalization**: Use robust scaling methods for features
4. **Time Alignment**: Ensure proper temporal alignment across assets

### Model Development
1. **Start Simple**: Begin with smaller models and scale up
2. **Systematic Tuning**: Use grid search or Bayesian optimization
3. **Cross Validation**: Use walk-forward validation for temporal data
4. **Ensemble Methods**: Combine multiple model variants for robustness

### Production Deployment  
1. **Model Versioning**: Track model versions and configurations
2. **Performance Monitoring**: Monitor model performance in production
3. **Retraining Schedule**: Regular retraining based on performance decay
4. **Risk Management**: Implement safeguards and circuit breakers

### Computational Efficiency
1. **Hardware Optimization**: Match configuration to available hardware
2. **Parallel Processing**: Use multiple GPUs when available
3. **Checkpointing**: Regular checkpoints for long training runs
4. **Resource Monitoring**: Monitor GPU utilization and memory usage

## Advanced Topics

### Custom Loss Functions

For specific investment objectives, custom loss functions can be implemented:

```python
class CustomLoss(nn.Module):
    def __init__(self, target_volatility=0.15):
        super().__init__()
        self.target_vol = target_volatility
    
    def forward(self, predicted_returns, actual_returns, weights):
        portfolio_returns = (weights * actual_returns).sum(dim=1)
        
        # Penalize deviation from target volatility
        portfolio_vol = portfolio_returns.std()
        vol_penalty = abs(portfolio_vol - self.target_vol)
        
        # Maximize return subject to volatility constraint
        expected_return = portfolio_returns.mean()
        
        return -expected_return + vol_penalty
```

### Multi-Objective Optimization

Combine multiple objectives using weighted loss:

```python
def multi_objective_loss(pred_returns, actual_returns, weights, 
                        alpha=0.7, beta=0.2, gamma=0.1):
    """
    alpha: Weight for Sharpe ratio
    beta: Weight for maximum drawdown
    gamma: Weight for turnover penalty
    """
    portfolio_returns = (weights * actual_returns).sum(dim=1)
    
    # Sharpe component
    sharpe = portfolio_returns.mean() / portfolio_returns.std()
    
    # Drawdown component  
    cumulative = (1 + portfolio_returns).cumprod()
    drawdown = (cumulative / cumulative.cummax() - 1).min()
    
    # Turnover component (requires previous weights)
    # turnover = abs(weights - prev_weights).sum()
    
    return -(alpha * sharpe - beta * abs(drawdown))
```

### Transfer Learning

Leverage pre-trained models for new universes:

```python
# Load pre-trained model
pretrained_model = LSTMPortfolioModel.load_checkpoint("models/pretrained_sp500.pth")

# Fine-tune for new universe
pretrained_model.config.training_config.learning_rate = 0.0001  # Lower LR
pretrained_model.config.training_config.epochs = 20  # Fewer epochs

# Fine-tune on new data
pretrained_model.fit(new_returns_data, new_universe, fit_period)
```

## References

1. Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.
2. Vaswani, A. et al. (2017). Attention is All You Need. NeurIPS.
3. Markowitz, H. (1952). Portfolio Selection. Journal of Finance.
4. López de Prado, M. (2018). Advances in Financial Machine Learning.

## Support

For questions or issues:
1. Check the troubleshooting section above
2. Review test cases in `tests/unit/test_lstm/`
3. Examine configuration files in `configs/models/`
4. Consult the experiment results in `outputs/experiments/`

## Changelog

- **v1.0**: Initial LSTM implementation with basic architecture
- **v1.1**: Added memory optimization and mixed precision training  
- **v1.2**: Integrated with backtesting framework and experiment orchestration
- **v1.3**: Added performance comparison tests and computational validation
- **v1.4**: Complete documentation and hyperparameter tuning guidelines