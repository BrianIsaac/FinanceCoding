# Academic Performance Report
*Generated: 2025-09-21 17:19:13*

# Executive Summary
Report generated: 2025-09-21 17:19:13

Best performing model: **LSTM** (Sharpe Ratio: 0.0446)
## Key Findings
- **HRP**: Return=0.0002, Volatility=0.0068, Sharpe=0.0301
- **LSTM**: Return=0.0001, Volatility=0.0032, Sharpe=0.0446
- **GAT-MST**: Return=0.0002, Volatility=0.0080, Sharpe=0.0222
- **GAT-kNN**: Return=0.0001, Volatility=0.0098, Sharpe=0.0103
- **GAT-TMFG**: Return=0.0002, Volatility=0.0068, Sharpe=0.0365
- **EqualWeight**: Return=0.0001, Volatility=0.0041, Sharpe=0.0270
- **MarketCapWeighted**: Return=0.0002, Volatility=0.0053, Sharpe=0.0333
- **MeanReversion**: Return=0.0001, Volatility=0.0042, Sharpe=0.0291


## Methodology

### Statistical Framework

- Confidence Level: 95.0%
- Significance Level: 0.05
- Multiple Testing Correction: Bonferroni

### Performance Metrics
- **Sharpe Ratio**: Risk-adjusted returns (excess return / volatility)
- **Sortino Ratio**: Downside risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Value at Risk (95%)**: 5th percentile of return distribution
- **CVaR (95%)**: Expected return in worst 5% of cases

### Confidence Intervals
- Bootstrap method with 10,000 iterations
- Asymptotic approximation for Sharpe ratio


| Model             | mean_return              |   volatility |   skewness |   kurtosis | sharpe_ratio             |   downside_deviation |   sortino_ratio |   max_drawdown |   var_95 |   cvar_95 |   n_observations |   confidence_score | Significance   |
|:------------------|:-------------------------|-------------:|-----------:|-----------:|:-------------------------|---------------------:|----------------:|---------------:|---------:|----------:|-----------------:|-------------------:|:---------------|
| HRP               | 0.0002 [-0.0002, 0.0006] |       0.0068 |    -0.7005 |    15.1153 | 0.0301 [-0.0223, 0.0825] |               0.0058 |          0.0354 |        -0.2298 |  -0.0084 |   -0.0165 |             1399 |                nan | ns             |
| LSTM              | 0.0001 [-0.0000, 0.0003] |       0.0032 |     0.5876 |     8.3396 | 0.0446 [-0.0079, 0.0970] |               0.0023 |          0.0623 |        -0.0837 |  -0.005  |   -0.0073 |             1399 |                nan | ns             |
| GAT-MST           | 0.0002 [-0.0002, 0.0006] |       0.008  |     0.0443 |     6.1793 | 0.0222 [-0.0302, 0.0746] |               0.0061 |          0.0295 |        -0.3553 |  -0.0125 |   -0.0191 |             1399 |                nan | ns             |
| GAT-kNN           | 0.0001 [-0.0004, 0.0006] |       0.0098 |     0.4908 |    18.5911 | 0.0103 [-0.0421, 0.0627] |               0.0077 |          0.0132 |        -0.3381 |  -0.013  |   -0.0222 |             1399 |                nan | ns             |
| GAT-TMFG          | 0.0002 [-0.0001, 0.0006] |       0.0068 |     0.5186 |    14.3615 | 0.0365 [-0.0159, 0.0890] |               0.0051 |          0.0485 |        -0.1681 |  -0.0092 |   -0.0151 |             1399 |                nan | ns             |
| EqualWeight       | 0.0001 [-0.0001, 0.0003] |       0.0041 |    -0.0278 |     3.935  | 0.0270 [-0.0255, 0.0794] |               0.0029 |          0.0387 |        -0.108  |  -0.0063 |   -0.0092 |             1399 |                nan | ns             |
| MarketCapWeighted | 0.0002 [-0.0001, 0.0005] |       0.0053 |    -0.0475 |     3.5396 | 0.0333 [-0.0191, 0.0857] |               0.0036 |          0.0482 |        -0.1358 |  -0.0083 |   -0.0116 |             1399 |                nan | ns             |
| MeanReversion     | 0.0001 [-0.0001, 0.0003] |       0.0042 |     0.0068 |     5.0549 | 0.0291 [-0.0233, 0.0815] |               0.003  |          0.0415 |        -0.1168 |  -0.0064 |   -0.0094 |             1399 |                nan | ns             |

## Statistical Significance

### HRP

- Mean Return Test: t=1.1247, p=0.2609 ns
- Normality Test (JB): stat=13432.3844, p=0.0000 - Returns deviate from normality

### LSTM

- Mean Return Test: t=1.6667, p=0.0958 ns
- Normality Test (JB): stat=4134.6706, p=0.0000 - Returns deviate from normality

### GAT-MST

- Mean Return Test: t=0.8311, p=0.4061 ns
- Normality Test (JB): stat=2226.2511, p=0.0000 - Returns deviate from normality

### GAT-kNN

- Mean Return Test: t=0.3858, p=0.6997 ns
- Normality Test (JB): stat=20203.4873, p=0.0000 - Returns deviate from normality

### GAT-TMFG

- Mean Return Test: t=1.3660, p=0.1721 ns
- Normality Test (JB): stat=12085.4963, p=0.0000 - Returns deviate from normality

### EqualWeight

- Mean Return Test: t=1.0080, p=0.3136 ns
- Normality Test (JB): stat=902.7929, p=0.0000 - Returns deviate from normality

### MarketCapWeighted

- Mean Return Test: t=1.2450, p=0.2133 ns
- Normality Test (JB): stat=730.8670, p=0.0000 - Returns deviate from normality

### MeanReversion

- Mean Return Test: t=1.0892, p=0.2763 ns
- Normality Test (JB): stat=1489.5049, p=0.0000 - Returns deviate from normality

### Significance Legend
- \*\*\* : p < 0.001 (highly significant)
- \*\* : p < 0.01 (significant)
- \* : p < 0.05 (marginally significant)
- ns : p >= 0.05 (not significant)

## Model Comparison

### Ranking by Sharpe Ratio

1. **LSTM**: 0.0446
2. **GAT-TMFG**: 0.0365
3. **MarketCapWeighted**: 0.0333
4. **HRP**: 0.0301
5. **MeanReversion**: 0.0291
6. **EqualWeight**: 0.0270
7. **GAT-MST**: 0.0222
8. **GAT-kNN**: 0.0103

### Ranking by Sortino Ratio

1. **LSTM**: 0.0623
2. **GAT-TMFG**: 0.0485
3. **MarketCapWeighted**: 0.0482
4. **MeanReversion**: 0.0415
5. **EqualWeight**: 0.0387
6. **HRP**: 0.0354
7. **GAT-MST**: 0.0295
8. **GAT-kNN**: 0.0132

### Ranking by Mean Return

1. **GAT-TMFG**: 0.0002
2. **HRP**: 0.0002
3. **GAT-MST**: 0.0002
4. **MarketCapWeighted**: 0.0002
5. **LSTM**: 0.0001
6. **MeanReversion**: 0.0001
7. **EqualWeight**: 0.0001
8. **GAT-kNN**: 0.0001


## Academic Caveats and Limitations

- Limited sample size (0 observations) may affect statistical power

### Statistical Considerations
- Confidence intervals assume stationarity
- Past performance does not guarantee future results
- Transaction costs and market impact not fully modeled

## Robustness Analysis

### HRP
- Excess Kurtosis: 15.1153 (heavy tails present)
- Maximum Drawdown: -22.98% (significant loss period)
- Value at Risk (95%): -0.0084

### LSTM
- Excess Kurtosis: 8.3396 (heavy tails present)
- Value at Risk (95%): -0.0050

### GAT-MST
- Excess Kurtosis: 6.1793 (heavy tails present)
- Maximum Drawdown: -35.53% (significant loss period)
- Value at Risk (95%): -0.0125

### GAT-kNN
- Excess Kurtosis: 18.5911 (heavy tails present)
- Maximum Drawdown: -33.81% (significant loss period)
- Value at Risk (95%): -0.0130

### GAT-TMFG
- Excess Kurtosis: 14.3615 (heavy tails present)
- Value at Risk (95%): -0.0092

### EqualWeight
- Excess Kurtosis: 3.9350 (heavy tails present)
- Value at Risk (95%): -0.0063

### MarketCapWeighted
- Excess Kurtosis: 3.5396 (heavy tails present)
- Value at Risk (95%): -0.0083

### MeanReversion
- Excess Kurtosis: 5.0549 (heavy tails present)
- Value at Risk (95%): -0.0064

