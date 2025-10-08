# Epic 2: Machine Learning Model Implementation

**Epic Goal:** Implement and validate the three ML portfolio allocation approaches (HRP, LSTM, GAT) with unified constraint system and optimization framework. This epic delivers the core machine learning capabilities that differentiate the research from traditional approaches, enabling direct comparison of clustering, temporal, and graph-based allocation methods.

## Story 2.1: Hierarchical Risk Parity (HRP) Implementation

As a **quantitative researcher**,  
I want **HRP allocation using correlation distance clustering and recursive bisection**,  
so that **I can test clustering-aware portfolio construction without covariance matrix inversion**.

### Acceptance Criteria
1. Correlation distance matrix calculation using (1 - correlation)/2 transformation
2. Hierarchical clustering implementation with linkage methods (single, complete, average)
3. Recursive bisection algorithm allocates capital according to cluster tree structure
4. HRP allocation respects unified constraints (long-only, top-k limits, turnover controls)
5. Model outputs portfolio weights for monthly rebalancing with performance tracking
6. Unit tests validate clustering behavior and allocation logic against known examples

## Story 2.2: LSTM Temporal Pattern Recognition Module

As a **portfolio manager**,  
I want **LSTM networks with 60-day lookback windows for return forecasting**,  
so that **I can capture temporal dependencies and tilt portfolios toward predicted outperformers**.

### Acceptance Criteria
1. Sequence-to-sequence LSTM architecture processes 60-day historical return windows
2. Return forecasting model predicts next-month expected returns for all securities
3. Portfolio allocation tilts weights based on LSTM return predictions within constraint system
4. GPU memory optimization handles full S&P MidCap 400 universe within 12GB VRAM limits
5. Training pipeline implements proper validation splits to prevent look-ahead bias
6. Model checkpointing enables retraining and hyperparameter experimentation

## Story 2.3: Graph Attention Network (GAT) Relationship Modeling

As a **quantitative researcher**,  
I want **GAT implementation with multiple graph construction methods and end-to-end Sharpe optimization**,  
so that **I can model complex asset relationships and optimize directly for risk-adjusted returns**.

### Acceptance Criteria
1. Graph construction supports k-NN, MST, and TMFG methods using correlation matrices
2. Multi-head attention mechanism learns adaptive relationship weights between securities
3. End-to-end training optimizes Sharpe ratio objective function rather than intermediate forecasts
4. Graph neural network architecture handles variable universe size and membership changes
5. GPU acceleration leverages existing GAT framework with memory optimization
6. Attention weight visualization enables interpretation of learned asset relationships

## Story 2.4: Unified Constraint System Integration

As a **portfolio manager**,  
I want **consistent constraint application across all ML approaches**,  
so that **fair performance comparisons isolate model effectiveness from implementation differences**.

### Acceptance Criteria
1. Constraint enforcement module applies identical rules to HRP, LSTM, and GAT allocations
2. Top-k position limits (k ∈ {20, 30, 50, 75, 100}) implemented for all approaches
3. Monthly turnover tracking with ≤20% limits enforced consistently
4. Long-only constraint prevents negative weights across all model outputs
5. Transaction cost modeling (0.1% linear) applied uniformly to all approaches
6. Constraint violation logging and handling maintains portfolio feasibility

## Story 2.5: Model Training and Validation Pipeline

As a **developer**,  
I want **automated training pipelines with proper validation splits**,  
so that **all models train consistently with out-of-sample validation and no look-ahead bias**.

### Acceptance Criteria
1. 36-month training / 12-month validation / 12-month testing protocol implemented
2. Walk-forward analysis maintains temporal data integrity across rolling windows
3. Model hyperparameter optimization within validation framework
4. Training automation handles GPU memory constraints and batch processing
5. Model serialization enables consistent backtesting across time periods
6. Validation metrics track overfitting and generalization performance
