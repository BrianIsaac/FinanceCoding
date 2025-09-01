
```mermaid
flowchart TD
  A[Month-end snapshot t] --> B[Build 60-day return features and z-score]
  A --> C[Compute 60-day correlations]
  C --> D1[kNN graph]
  C --> D2[MST graph]
  C --> D3[TMFG graph]
  B --> E[GAT encoder with attention]
  D1 --> E
  D2 --> E
  D3 --> E
  E --> F[Simplex projection long-only sum=1]
  F --> G[Top-K sparsity and renormalise]
  G --> H[Trade at t+1 then hold to next rebalance]
  H --> I[Daily P&L to test metrics]
  I --> J[Rolling evaluation train val test]
  J --> K[Select best by validation Sharpe]
  K --> L[Report Sharpe CAGR Drawdown Turnover]
```
---


---


---

```mermaid
flowchart TD
  subgraph Snapshot
    A[Month-end snapshot t]
  end

  subgraph Features_and_Corr
    B[Build 60d return features and z-score]
    C[Compute 60d correlations]
  end

  subgraph Graph_Filter
    D1[kNN graph]
    D2[MST graph]
    D3[TMFG graph]
  end

  subgraph Model
    E[GAT with attention]
    F[Simplex projection long only sum=1]
    G[Top K sparsity and renormalise]
  end

  subgraph Backtest
    H[Trade at t+1 and hold to next rebalance]
  end

  subgraph Evaluation
    I[Daily PnL]
    J[Test metrics Sharpe CAGR Drawdown]
    K[Rolling evaluation train val test]
    L[Select best by validation Sharpe]
    M[Report figures and tables]
  end

  A --> B
  A --> C
  C --> D1
  C --> D2
  C --> D3
  B --> E
  D1 --> E
  D2 --> E
  D3 --> E
  E --> F
  F --> G
  G --> H
  H --> I
  I --> J
  J --> K
  K --> L
  L --> M
  ```

  ```mermaid
  flowchart TD
  subgraph Backtest
    H[Trade at t+1<br/>Hold to next rebalance]
  end

  subgraph Evaluation
    I[Daily PnL]
    J[Test metrics:<br/>Sharpe, CAGR, Drawdown]
    K[Rolling evaluation:<br/>train / val / test]
    L[Model selection:<br/>best validation Sharpe]
    M[Reporting:<br/>figures and tables]
  end

  H --> I --> J --> K --> L --> M
```

```mermaid
flowchart TB
  A["Snapshot - month end t"] --> B["Universe and hygiene - liquidity filters"]
  B --> C["Prepare returns window - 36 months daily"]
  C --> D["Covariance estimation - shrinkage (Ledoit-Wolf)"]
  C --> E["Expected returns - hist mean (0 for MinVar)"]

  %% EW path
  B --> EW1["Select top-K by liquidity"]
  EW1 --> EW2["Equal weights = 1/K"]
  EW2 --> W_EW["EW weights (long-only, sum=1)"]

  %% MV / MinVar path
  D --> MV1["Solve MV or MinVar - with weight caps, sum=1"]
  E --> MV1
  MV1 --> MV2["Top-K prune tiny weights and renormalise"]
  MV2 --> W_MV["MV/MinVar weights (long-only, sum=1)"]
```

```mermaid
flowchart TB
  A1["Snapshot - month end t"] --> B1["Prepare returns window - 36 months daily"]
  B1 --> C1["Correlation matrix (rho)"]
  C1 --> D1["Distance matrix d(i,j)=sqrt(0.5*(1-rho(i,j)))"]
  D1 --> E1["Hierarchical clustering - linkage on distance"]
  E1 --> F1["Quasi-diagonalise Sigma - reorder by dendrogram"]
  F1 --> G1["Recursive bisection - cluster risk parity"]
  G1 --> H1["Raw HRP weights - long-only, sum=1"]
  H1 --> I1["Top-K prune tiny weights and renormalise"]
  I1 --> W_HRP["HRP weights (long-only, sum=1)"]
```

```mermaid
flowchart TB
  A2["Snapshot - month end t"] --> B2["Build sequences per asset - lookback L (e.g., 60 days)"]
  B2 --> C2["Feature scaling and options - returns, volatility, sector, liquidity"]
  C2 --> D2["Train LSTM on train window - rolling, no look-ahead"]
  D2 --> E2["Select checkpoint - best validation Sharpe"]
  E2 --> P2["At each rebalance - predict next-period returns"]
  P2 --> R2["Rank predictions and select top-K (long-only)"]
  R2 --> W2["Weighting rule - softmax or proportional to positive forecast"]
  W2 --> W_LSTM["LSTM weights (long-only, sum=1)"]
```
