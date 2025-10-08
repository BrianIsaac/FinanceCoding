# Executive Summary

This document outlines the technical architecture for a comprehensive machine learning research framework designed to evaluate and compare three advanced portfolio optimization approaches: Hierarchical Risk Parity (HRP), Long Short-Term Memory (LSTM) networks, and Graph Attention Networks (GATs). The system targets institutional portfolio managers seeking improved risk-adjusted returns for US mid-cap equity portfolios while maintaining operational feasibility within realistic constraints.

**Key Architecture Highlights:**
- **Monorepo Structure**: Organized research framework optimized for rapid experimentation and reproducible results
- **GPU-Optimized ML Pipeline**: PyTorch-based implementation designed for RTX GeForce 5070Ti (12GB VRAM) constraints
- **Dynamic Universe Management**: Handles time-varying S&P MidCap 400 membership without survivorship bias
- **Parquet-Based Data Architecture**: Efficient storage and retrieval of large-scale financial time series data
- **Unified Constraint System**: Consistent application of real-world portfolio constraints across all ML approaches
- **Rolling Validation Framework**: Academic-grade backtesting with strict temporal data integrity

---
