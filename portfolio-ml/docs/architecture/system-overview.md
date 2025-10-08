# System Overview

## Architecture Philosophy

The architecture follows a **modular research framework** design that prioritizes:
1. **Reproducibility**: All experiments can be exactly replicated across different environments
2. **Fair Comparison**: Identical constraint systems and evaluation protocols across all approaches
3. **Scalability**: GPU memory optimization handles 400+ asset universe within hardware constraints
4. **Academic Rigor**: Strict no-look-ahead validation with statistical significance testing
5. **Production Readiness**: Clean interfaces enable institutional deployment within 6 months

## High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Portfolio Optimization Framework             │
├─────────────────────────────────────────────────────────────────┤
│  Data Pipeline           │  ML Models              │ Evaluation   │
│  ┌─────────────────┐     │  ┌─────────────────┐    │ ┌──────────┐ │
│  │ Wikipedia       │────▶│  │ Hierarchical    │    │ │ Rolling  │ │
│  │ Scraper         │     │  │ Risk Parity     │    │ │ Backtest │ │
│  └─────────────────┘     │  └─────────────────┘    │ │ Engine   │ │
│  ┌─────────────────┐     │  ┌─────────────────┐    │ └──────────┘ │
│  │ Multi-Source    │────▶│  │ LSTM Temporal   │    │ ┌──────────┐ │
│  │ Market Data     │     │  │ Networks        │    │ │ Performance│ │
│  └─────────────────┘     │  └─────────────────┘    │ │ Analytics │ │
│  ┌─────────────────┐     │  ┌─────────────────┐    │ └──────────┘ │
│  │ Graph           │────▶│  │ Graph Attention │    │ ┌──────────┐ │
│  │ Construction    │     │  │ Networks        │    │ │ Statistical│ │
│  └─────────────────┘     │  └─────────────────┘    │ │ Testing   │ │
│                          │                         │ └──────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---
