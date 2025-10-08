# Evaluation Protocols and Statistical Testing Framework

## Overview

This document details the comprehensive evaluation protocols and statistical testing framework used to assess machine learning-based portfolio optimization approaches. The framework ensures academic rigor, temporal integrity, and institutional-grade performance validation.

## 1. Rolling Window Backtesting Protocol

### 1.1 Temporal Validation Design

**Rolling Window Configuration:**
- **Training Windows**: 252, 504, 756 trading days (1, 2, 3 years)
- **Rebalancing Frequency**: Monthly end-of-month rebalancing
- **Rolling Increment**: 21 trading days (1 month forward)
- **Total Evaluation Period**: January 2016 - December 2024 (8 years)

**Data Split Strategy:**
```
Training Phase:   2016-01-01 to 2020-12-31 (4 years)
Validation Phase: 2021-01-01 to 2022-12-31 (2 years)
Testing Phase:    2023-01-01 to 2024-12-31 (2 years)
```

### 1.2 No Look-Ahead Bias Prevention

**Strict Temporal Integrity:**
- Model training uses only historical data available at time t
- Universe membership reflects historical index composition
- No future information leakage in feature engineering
- Transaction costs applied to actual rebalancing decisions

**Implementation Safeguards:**
- Date-aware data splitting with explicit temporal boundaries
- Automated validation of temporal data integrity
- Universe calendar alignment with actual index changes
- Business day calendar validation for realistic timing

### 1.3 Performance Attribution Framework

**Rolling Window Performance Tracking:**
- Monthly portfolio return calculation
- Risk metric evolution over time
- Performance attribution by market regime
- Model decision analysis and interpretation

## 2. Statistical Testing Framework

### 2.1 Sharpe Ratio Significance Testing

**Jobson-Korkie Test Implementation:**
```
Test Statistic: T_JK = (SR_1 - SR_2) / √(Var[SR_1 - SR_2])
Where: Var[SR_1 - SR_2] = (1/T)[2 - 2ρ + 0.5(SR_1² + SR_2²) - 1.5ρ(SR_1·SR_2)]
```

**Memmel Correction for Finite Samples:**
- Adjusts for small sample bias in Sharpe ratio comparisons
- Provides more conservative significance testing
- Essential for realistic portfolio evaluation periods

### 2.2 Bootstrap Confidence Intervals

**Bootstrap Methodology:**
- **Bootstrap Samples**: 10,000 iterations
- **Sampling Strategy**: Block bootstrap preserving temporal structure
- **Block Length**: 21 trading days (monthly blocks)
- **Confidence Levels**: 90%, 95%, 99%

**Bootstrap Applications:**
- Sharpe ratio confidence intervals
- Maximum drawdown uncertainty quantification
- Performance metric stability assessment
- Robustness analysis across parameter configurations

### 2.3 Multiple Comparison Corrections

**Multiple Testing Problem:**
When comparing multiple models simultaneously, Type I error inflation requires correction:

**Bonferroni Correction:**
```
α_corrected = α / n_comparisons
```
Conservative approach ensuring family-wise error rate control.

**Holm-Sidak Correction:**
```
α_i = 1 - (1-α)^(1/(n-i+1))
```
Less conservative while maintaining error rate control.

### 2.4 Effect Size Analysis

**Practical Significance Assessment:**
Beyond statistical significance, we assess practical significance:

**Cohen's d for Performance Differences:**
```
d = (μ_1 - μ_2) / σ_pooled
Where: σ_pooled = √((σ_1² + σ_2²)/2)
```

**Effect Size Interpretation:**
- Small: d = 0.2 (2% annualized return difference)
- Medium: d = 0.5 (5% annualized return difference)
- Large: d = 0.8 (8% annualized return difference)

## 3. Performance Analytics Suite

### 3.1 Risk-Adjusted Performance Metrics

**Primary Metrics:**
- **Sharpe Ratio**: (E[r_p] - r_f) / σ[r_p]
- **Information Ratio**: (E[r_p] - r_b) / σ[r_p - r_b]
- **Calmar Ratio**: CAGR / Maximum Drawdown
- **Sortino Ratio**: E[r_p] / σ_downside[r_p]

**Risk Metrics:**
- **Maximum Drawdown**: Maximum peak-to-trough decline
- **VaR (5%)**: 5th percentile of return distribution
- **CVaR (5%)**: Expected return below VaR threshold
- **Volatility**: Annualized standard deviation

### 3.2 Benchmark Comparison Framework

**Baseline Strategies:**
1. **Equal Weight Portfolio**: 1/n allocation
2. **Market Cap Weighted**: Proportional to market capitalization
3. **Mean Variance Optimization**: Classical Markowitz approach
4. **Risk Parity**: Equal risk contribution allocation

**Statistical Comparison Protocol:**
- Paired t-tests for return series comparison
- Wilcoxon signed-rank test for non-parametric validation
- Bootstrap confidence intervals for performance differences
- Multiple comparison corrections for simultaneous testing

### 3.3 Market Regime Analysis

**Regime Identification:**
- **Bull Markets**: S&P 500 returns > 10% annualized
- **Bear Markets**: S&P 500 returns < -10% annualized  
- **Volatile Markets**: S&P 500 volatility > 20% annualized
- **Sideways Markets**: All other periods

**Regime-Specific Performance:**
- Model performance within each market regime
- Regime transition analysis and adaptation capabilities
- Statistical significance testing within regime periods
- Cross-regime consistency validation

## 4. Robustness and Sensitivity Analysis

### 4.1 Parameter Sensitivity Testing

**Hyperparameter Grids:**

**HRP Parameters:**
- Lookback days: {252, 504, 756, 1008}
- Linkage methods: {'single', 'complete', 'average', 'ward'}
- Distance metrics: {'correlation', 'angular', 'absolute_correlation'}

**LSTM Parameters:**
- Sequence lengths: {30, 45, 60, 90}
- Hidden sizes: {64, 128, 256}
- Number of layers: {1, 2, 3}
- Dropout rates: {0.1, 0.3, 0.5}
- Learning rates: {0.0001, 0.001, 0.01}

**GAT Parameters:**
- Attention heads: {2, 4, 8}
- Hidden dimensions: {64, 128, 256}
- Graph construction: {'k_nn', 'mst', 'tmfg'}
- Learning rates: {0.0001, 0.001, 0.01}

### 4.2 Transaction Cost Sensitivity

**Cost Scenarios:**
- **Aggressive Trading**: 5 basis points (0.05%)
- **Baseline Trading**: 10 basis points (0.10%)
- **Conservative Trading**: 20 basis points (0.20%)

**Impact Analysis:**
- Performance ranking stability across cost levels
- Optimal allocation changes under different cost assumptions
- Break-even analysis for model complexity vs. cost efficiency

### 4.3 Portfolio Size Sensitivity

**Top-K Analysis:**
- Portfolio sizes: k ∈ {20, 30, 50, 75, 100}
- Diversification vs. concentration trade-offs
- Statistical significance of performance differences
- Optimal portfolio size recommendations

## 5. Data Quality and Integrity Framework

### 5.1 Data Coverage Requirements

**Minimum Standards:**
- **Overall Coverage**: >95% data availability
- **Individual Asset**: >90% data availability over evaluation period
- **Daily Volume**: >1,000 shares minimum trading volume
- **Price Continuity**: <50% maximum daily price change threshold

### 5.2 Gap Filling and Quality Control

**Gap Filling Methodology:**
- **Forward Fill**: Primary method for short gaps (<5 days)
- **Backward Fill**: Secondary method for data initialization
- **Interpolation**: Linear interpolation for medium gaps (5-10 days)
- **Exclusion**: Assets with >10% missing data excluded from analysis

**Quality Control Checks:**
- Statistical outlier detection using 3-sigma thresholds
- Volume validation for realistic trading scenarios
- Corporate action detection and adjustment
- Business day calendar alignment validation

### 5.3 Universe Construction Validation

**S&P MidCap 400 Historical Membership:**
- Wikipedia-sourced historical membership data
- Monthly rebalancing aligned with index methodology
- Survivorship bias prevention through dynamic universe
- Historical accuracy validation against known index changes

## 6. Computational Performance and Scalability

### 6.1 Performance Benchmarks

**Target Performance Standards:**
- **Model Training**: <2 hours per configuration
- **Portfolio Rebalancing**: <5 seconds per month
- **Full Backtest Execution**: <8 hours for complete evaluation
- **Memory Usage**: <32GB RAM, <12GB VRAM

### 6.2 Optimization Strategies

**Memory Management:**
- Dynamic batch sizing based on available GPU memory
- Gradient checkpointing for LSTM training
- Efficient data loading with prefetching
- Model checkpointing and state management

**Parallel Processing:**
- Multi-process parameter grid exploration
- Parallel bootstrap confidence interval calculation
- Concurrent statistical test execution
- GPU/CPU workload balancing

### 6.3 Scalability Considerations

**Universe Size Scaling:**
- Memory complexity analysis for larger universes
- Computational time scaling with asset count
- GPU memory constraints for large portfolios
- Alternative optimization strategies for scale

## 7. Reproducibility Framework

### 7.1 Deterministic Execution

**Random Seed Management:**
- Fixed seeds for NumPy, PyTorch, and random modules
- Deterministic GPU operations where possible
- Consistent initialization across model training runs
- Reproducible bootstrap sampling sequences

### 7.2 Environment Specification

**Complete Dependency Management:**
- Python version specification (>=3.9)
- Exact package versions using uv lock files
- GPU driver and CUDA version requirements
- Operating system compatibility testing

### 7.3 Configuration Management

**Experiment Configuration:**
- YAML-based configuration files for all experiments
- Parameter grid specifications for sensitivity analysis
- Model architecture configurations
- Evaluation protocol specifications

## 8. Quality Assurance and Validation

### 8.1 Testing Framework

**Statistical Test Validation:**
- Known statistical distributions for test validation
- Monte Carlo simulations for framework verification
- Edge case testing for numerical stability
- Performance regression testing

### 8.2 Code Quality Standards

**Implementation Standards:**
- Type hints for all function parameters
- Comprehensive docstrings following Google format
- Unit test coverage >90% for critical components
- Integration test validation for end-to-end workflows

### 8.3 Documentation Standards

**Research Documentation:**
- Mathematical formulation accuracy
- Algorithm implementation completeness
- Parameter selection rationale
- Performance interpretation guidelines

## Summary

This evaluation protocol framework ensures rigorous, reproducible, and statistically sound assessment of machine learning-based portfolio optimization approaches. The framework addresses temporal integrity, statistical significance testing, robustness analysis, and computational efficiency while maintaining academic research standards suitable for peer review and institutional implementation.

The comprehensive testing methodology provides confidence in model comparisons and enables evidence-based decision making for portfolio optimization approach selection in practical investment applications.