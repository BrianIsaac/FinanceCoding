# Methodology Documentation: ML-Based Portfolio Optimization

## Overview

This document provides comprehensive methodology documentation for machine learning-based portfolio optimization approaches implemented in this research framework. The study compares three distinct approaches: Hierarchical Risk Parity (HRP), Long Short-Term Memory networks (LSTM), and Graph Attention Networks (GAT) for portfolio optimization on the S&P MidCap 400 universe.

## 1. Hierarchical Risk Parity (HRP) Implementation

### 1.1 Mathematical Foundation

Hierarchical Risk Parity (HRP) applies hierarchical clustering to asset correlation structures, then allocates capital using recursive bisection based on cluster risk contributions.

**Algorithm Steps:**
1. **Tree Clustering**: Apply hierarchical clustering to correlation distance matrix
2. **Quasi-Diagonalization**: Reorder correlation matrix based on clustering results  
3. **Recursive Bisection**: Recursively allocate capital between clusters based on risk parity principles

### 1.2 Implementation Details

**Distance Metric Calculation:**
```
d[i,j] = √((1 - ρ[i,j]) / 2)
```
Where ρ[i,j] is the correlation between assets i and j.

**Clustering Algorithm:**
- **Linkage Methods**: Single, Complete, Average, Ward linkage supported
- **Distance Metrics**: Correlation, Angular, Absolute correlation distance
- **Lookback Periods**: 252, 504, 756, 1008 trading days

**Recursive Allocation:**
At each cluster split, capital allocation follows:
```
α_left = 1 / (1 + (σ_right / σ_left))
α_right = 1 - α_left  
```
Where σ represents cluster volatility.

### 1.3 Technical Implementation

**Key Classes:**
- `HRPModel`: Main implementation with correlation calculation and clustering
- `HRPConfig`: Configuration parameters for lookback periods and clustering methods
- `RecursiveBisection`: Capital allocation algorithm implementation

**Performance Optimizations:**
- Efficient correlation matrix computation using NumPy
- Cached clustering results for rolling window applications
- Memory-efficient recursive allocation implementation

## 2. LSTM Temporal Network Architecture

### 2.1 Mathematical Foundation

LSTM networks model temporal dependencies in asset return sequences for portfolio weight prediction, incorporating multi-head attention mechanisms for enhanced pattern recognition.

**LSTM Cell Equations:**
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)  # Forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)  # Input gate  
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)  # Candidate values
C_t = f_t * C_{t-1} + i_t * C̃_t  # Cell state
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)  # Output gate
h_t = o_t * tanh(C_t)  # Hidden state
```

### 2.2 Network Architecture

**Multi-Layer LSTM Structure:**
- **Input Layer**: Time series of asset returns (sequence_length × n_assets)
- **LSTM Layers**: 1-3 layers with hidden dimensions 64, 128, or 256
- **Attention Layer**: Multi-head attention for temporal pattern emphasis
- **Output Layer**: Portfolio weight prediction with L1 normalization

**Multi-Head Attention:**
```
Attention(Q,K,V) = softmax(QK^T / √d_k)V
MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
```

### 2.3 Training Protocol

**Loss Function - Sharpe Ratio Optimization:**
```
L = -E[r_p] / σ[r_p] + λ||w||_1
```
Where r_p is portfolio return, w are portfolio weights, λ is L1 regularization.

**Training Configuration:**
- **Sequence Lengths**: 30, 45, 60, 90 trading days
- **Batch Sizes**: 32, 64, 128 (GPU memory dependent)
- **Learning Rates**: 0.0001, 0.001, 0.01 with Adam optimizer
- **Dropout**: 0.1, 0.3, 0.5 for regularization

**Memory Optimization:**
- Gradient checkpointing for reduced memory consumption
- Dynamic batch sizing based on GPU memory availability
- Efficient data loading with sliding window batching

### 2.4 Technical Implementation

**Key Classes:**
- `LSTMNetwork`: Core neural network architecture
- `MultiHeadAttention`: Attention mechanism implementation
- `TimeSeriesDataset`: Efficient data loading for temporal sequences
- `MemoryEfficientTrainer`: GPU-optimized training with memory management
- `SharpeRatioLoss`: Custom loss function for portfolio optimization

## 3. Graph Attention Network (GAT) Architecture

### 3.1 Mathematical Foundation

GAT models asset relationships through graph neural networks, using attention mechanisms to weight connections between assets based on feature similarity and correlation patterns.

**Graph Attention Mechanism:**
```
e_ij = a^T[Wh_i || Wh_j]  # Attention coefficients
α_ij = softmax_j(e_ij)  # Normalized attention weights  
h_i' = σ(Σ_j α_ij W h_j)  # Output features
```

### 3.2 Graph Construction Methods

**K-Nearest Neighbors (k-NN):**
- Connect each asset to k most correlated assets
- Parameters: k ∈ {5, 10, 15, 20}
- Bidirectional edges with correlation-based weights

**Minimum Spanning Tree (MST):**
- Connect assets using minimum correlation distance
- Creates tree structure ensuring connectivity
- Edge weights: 1 - |correlation|

**Triangulated Maximally Filtered Graph (TMFG):**
- Preserves maximum correlation information
- Creates planar graph with 3(n-2) edges
- Maintains topological constraints

### 3.3 Network Architecture

**Multi-Layer GAT Structure:**
- **Input Layer**: Asset features (returns, volatility, market cap)
- **GAT Layers**: 2-4 layers with attention heads 2, 4, 8
- **Attention Heads**: Multi-head attention for diverse relationship modeling
- **Output Layer**: Portfolio weight prediction with constraints

**Attention Head Configuration:**
```
MultiHeadGAT(H) = ||_{k=1}^K σ(Σ_j α_ij^k W^k h_j)  # Concatenation
FinalLayer(H) = σ(Σ_j α_ij W h_j)  # Averaging for output
```

### 3.4 Training Protocol

**Loss Function:**
```
L = -Sharpe(r_p) + λ_1||w||_1 + λ_2||Aw - w||_2^2
```
Where A is the adjacency matrix encouraging smooth allocations across connected assets.

**Training Configuration:**
- **Hidden Dimensions**: 64, 128, 256
- **Attention Heads**: 2, 4, 8
- **Learning Rates**: 0.0001, 0.001, 0.01
- **Graph Construction**: Systematic comparison across methods

### 3.5 Technical Implementation

**Key Classes:**
- `GATPortfolio`: Main GAT model implementation
- `GraphBuildConfig`: Graph construction parameter configuration
- `build_graph_from_returns`: Graph construction algorithms (k-NN, MST, TMFG)
- `GATPortfolioModel`: Integration with portfolio optimization framework

## 4. Evaluation Protocols and Framework

### 4.1 Rolling Window Validation

**Temporal Validation Protocol:**
- **Training Windows**: 252, 504, 756 trading days
- **Rebalancing Frequency**: Monthly (end of month)
- **Rolling Forward**: 21 trading days (1 month) increments
- **No Look-Ahead**: Strict temporal integrity maintained

**Data Split Strategy:**
- **Training Period**: 2016-2020 (4 years)
- **Validation Period**: 2021-2022 (2 years)  
- **Test Period**: 2023-2024 (2 years)
- **Total Evaluation**: 8 years with 96 rebalancing periods

### 4.2 Performance Analytics Framework

**Risk-Adjusted Metrics:**
- **Sharpe Ratio**: E[r_p - r_f] / σ[r_p]
- **Information Ratio**: E[r_p - r_b] / σ[r_p - r_b]
- **Maximum Drawdown**: max_t(max_{s≤t} P_s - P_t) / max_{s≤t} P_s
- **Calmar Ratio**: CAGR / Maximum Drawdown
- **Sortino Ratio**: E[r_p - r_f] / σ_downside[r_p]

**Risk Metrics:**
- **Value at Risk (VaR)**: 5th percentile of return distribution
- **Conditional VaR (CVaR)**: Expected return below VaR threshold
- **Volatility**: Standard deviation of portfolio returns
- **Beta**: Systematic risk relative to market benchmark

### 4.3 Statistical Testing Framework

**Hypothesis Testing:**
- **Sharpe Ratio Tests**: Jobson-Korkie test with Memmel correction
- **Bootstrap Methods**: 10,000 bootstrap samples for confidence intervals
- **Multiple Comparison Corrections**: Bonferroni and Holm-Sidak methods
- **Effect Size Analysis**: Practical significance assessment

**Significance Levels:**
- **Primary Tests**: α = 0.05
- **Conservative Tests**: α = 0.01 (after multiple comparison correction)
- **Confidence Intervals**: 95% bootstrap confidence intervals

### 4.4 Benchmark Comparisons

**Baseline Strategies:**
- **Equal Weight**: 1/n allocation across all assets
- **Market Cap Weight**: Allocation proportional to market capitalization
- **Mean Variance Optimization**: Classical Markowitz optimization
- **Risk Parity**: Volatility-based risk contribution parity

### 4.5 Implementation Requirements

**Computational Constraints:**
- **GPU Memory**: 12GB VRAM (GTX 4070 Ti SUPER)
- **System Memory**: 32GB RAM limit
- **Training Time**: <2 hours per model configuration
- **Inference Time**: <5 seconds for portfolio rebalancing

**Data Requirements:**
- **Universe Size**: ~400 assets (S&P MidCap 400)
- **Data Coverage**: >95% availability required
- **Quality Thresholds**: <10% missing data per asset
- **Temporal Range**: 2016-2024 (8 years)

## 5. Technical Implementation Architecture

### 5.1 Model Integration Framework

**Unified Interface:**
All models implement consistent `PortfolioModel` interface:
```python
@abstractmethod
def fit(self, returns: pd.DataFrame, **kwargs) -> None:
    """Train model on historical return data"""
    
@abstractmethod 
def predict_weights(self, returns: pd.DataFrame) -> np.ndarray:
    """Generate portfolio weights for given returns"""
    
@abstractmethod
def get_feature_importance(self) -> Dict[str, float]:
    """Return model-specific feature importance metrics"""
```

### 5.2 Configuration Management

**Hierarchical Configuration System:**
- Model-specific parameters in `configs/models/`
- Evaluation settings in `configs/evaluation/` 
- Experiment configurations in `configs/experiments/`
- Environment specifications in `configs/environment/`

### 5.3 Memory Management and Optimization

**GPU Memory Optimization:**
- Dynamic batch sizing based on available VRAM
- Model checkpointing for large networks
- Efficient data loading with prefetching
- Memory cleanup between model training runs

**CPU Optimization:**
- Parallel processing for non-ML computations
- Efficient pandas operations for data processing
- Cached computation results for repeated operations
- Memory-mapped file access for large datasets

## 6. Quality Assurance and Validation

### 6.1 Testing Framework

**Unit Testing:**
- Model implementation correctness
- Mathematical computation validation
- Configuration parameter validation
- Data processing pipeline testing

**Integration Testing:**
- End-to-end evaluation pipeline
- Multi-model comparison framework
- Statistical significance testing validation
- Performance metric calculation verification

**Reproducibility Testing:**
- Deterministic model training with fixed seeds
- Consistent evaluation results across runs
- Environment specification validation
- Configuration reproducibility verification

### 6.2 Documentation Standards

**Code Documentation:**
- Google-style docstrings for all functions
- Type hints for all function parameters
- Comprehensive class documentation
- Usage examples for public APIs

**Research Documentation:**
- Mathematical formulation documentation
- Algorithm implementation details
- Parameter selection rationale
- Performance optimization explanations

## Summary

This methodology framework provides comprehensive documentation for three distinct ML-based portfolio optimization approaches: HRP clustering-based allocation, LSTM temporal modeling, and GAT graph-based relationship modeling. Each approach is implemented with rigorous evaluation protocols, statistical testing frameworks, and performance optimization techniques suitable for institutional portfolio management applications.

The framework emphasizes reproducibility, statistical rigor, and practical implementation considerations while maintaining academic research standards for peer review and publication readiness.