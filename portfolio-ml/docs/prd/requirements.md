# Requirements

## Functional

FR1: The system shall construct a dynamic S&P MidCap 400 universe from Wikipedia historical membership data, avoiding survivorship bias and capturing realistic index dynamics from 2016-2024.

FR2: The system shall implement a multi-source data pipeline integrating Stooq and Yahoo Finance with gap-filling, normalization, and quality assurance processes producing clean daily returns and volume panels.

FR3: The system shall implement a Hierarchical Risk Parity (HRP) module with correlation distance matrices and recursive bisection allocation.

FR4: The system shall implement an LSTM module with sequence-to-sequence networks using 60-day lookback windows for temporal pattern recognition and return forecasting.

FR5: The system shall implement a Graph Attention Network (GAT) module with correlation-based edge construction (k-NN, MST, TMFG options) and end-to-end Sharpe optimization.

FR6: The system shall apply unified constraints consistently across all approaches including long-only positions, top-k position limits (k ∈ {20, 30, 50, 75, 100}), monthly turnover controls, and linear transaction cost modeling at 0.1% per trade.

FR7: The system shall execute rolling backtests using 36-month training / 12-month validation / 12-month testing protocol with walk-forward analysis and strict no-look-ahead validation.

FR8: The system shall calculate performance analytics including Sharpe ratio, CAGR, maximum drawdown, turnover, and statistical significance testing across all approaches.

FR9: The system shall implement baseline comparison frameworks including equal-weight and mean-variance benchmarks with identical constraint sets.

FR10: The system shall generate comparative analysis reports with clear recommendations on which approach merits further development for production deployment.

## Non Functional

NFR1: The system must process 400+ asset S&P MidCap 400 universe with daily data processing within 12GB VRAM constraints using GPU acceleration.

NFR2: The system must complete end-to-end pipeline processing and monthly rebalancing within 4 hours computational time limits.

NFR3: The system must achieve monthly portfolio turnover ≤20% to ensure realistic implementation costs.

NFR4: The system must maintain reproducible results across different execution environments using local filesystem-based data management.

NFR5: The system must optimize memory usage for GAT and LSTM training through batch processing, gradient checkpointing, and mixed-precision training where applicable.

NFR6: The system must provide clear documentation enabling deployment within 6 months of adoption decision for implementation teams.

NFR7: The system must support academic-level computational resources with local development environment constraints.
