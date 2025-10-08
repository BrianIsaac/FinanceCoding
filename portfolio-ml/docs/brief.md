# Project Brief: Optimising Portfolio Allocation with Machine Learning Techniques

## Executive Summary

This project develops and evaluates machine learning-based portfolio allocation strategies for US mid-cap equities that outperform traditional mean-variance optimization approaches. The core concept addresses the fundamental portfolio optimization problem of maximizing risk-adjusted returns while satisfying real-world constraints like long-only positions, transaction costs, and limited turnover.

The project targets the limitations of classical Modern Portfolio Theory (MPT), which assumes normal return distributions and requires accurate estimation of expected returns and covariances - leading to poor out-of-sample performance due to estimation errors. Our solution investigates three advanced ML approaches: Hierarchical Risk Parity (HRP) for clustering-aware allocation, Long Short-Term Memory (LSTM) networks for temporal pattern recognition, and Graph Attention Networks (GATs) for relationship-aware portfolio construction.

The target market is institutional portfolio managers and quantitative investment firms managing mid-cap equity portfolios, specifically those seeking to improve Sharpe ratios while maintaining operational feasibility. The key value proposition is delivering superior risk-adjusted returns through data-driven allocation methods that capture complex asset relationships and temporal dependencies that traditional approaches miss.

## Problem Statement

Traditional portfolio optimization faces critical practical limitations that undermine its effectiveness in real-world applications. The classical Markowitz mean-variance model, while foundational, suffers from several key weaknesses:

**Current State and Pain Points:**
- Estimation errors in expected returns and covariances lead to poor out-of-sample performance, with naive equal-weighted portfolios often outperforming optimized portfolios
- Assumption of normally distributed returns is frequently violated in real markets, especially during periods of stress
- High sensitivity to input parameters results in unstable portfolio weights that require frequent, costly rebalancing
- Large-scale portfolios face dimensionality challenges where estimation errors are amplified
- Static approaches fail to capture evolving market relationships and temporal dependencies

**Impact (Quantified):**
- Research shows traditional mean-variance optimization can underperform simple benchmarks by significant margins out-of-sample
- High portfolio turnover from unstable weights can erode returns through transaction costs
- Concentrated allocations from estimation errors expose portfolios to unnecessary concentration risk
- The S&P MidCap 400 universe with ~400 dynamic constituents exemplifies the scale where these problems become acute

**Why Existing Solutions Fall Short:**
- Robust optimization techniques add complexity but don't fundamentally solve the temporal and relational blind spots
- Risk parity approaches improve stability but ignore return forecasting entirely
- Traditional factor models rely on static relationships that may not adapt to changing market conditions

**Urgency and Importance:**
Modern markets generate vast amounts of high-frequency data with complex interdependencies that traditional linear models cannot capture. The advent of machine learning provides tools to model these relationships more effectively, but requires systematic evaluation to determine practical value in portfolio construction.

## Proposed Solution

Our solution implements a systematic framework to evaluate three machine learning approaches for portfolio allocation, each addressing different limitations of traditional methods:

**Core Concept and Approach:**

1. **Hierarchical Risk Parity (HRP)**: A clustering-based allocation technique that builds asset hierarchies using correlation distance and allocates capital according to tree structures. This avoids unstable covariance matrix inversion while respecting natural asset groupings and relationships.

2. **Long Short-Term Memory (LSTM)**: Recurrent neural networks designed to capture long-term dependencies in sequential financial data. LSTMs learn from historical return patterns to forecast future performance and dynamically tilt portfolios toward predicted outperformers.

3. **Graph Attention Networks (GATs)**: Advanced deep learning models that operate on graph-structured data, learning asset relationships through attention mechanisms. GATs can uncover complex, non-linear interdependencies and adapt the importance of different relationships based on market conditions.

**Key Differentiators:**
- **Dynamic Universe Handling**: Unlike static approaches, our framework accommodates changing index membership (S&P MidCap 400) to avoid survivorship bias
- **Unified Evaluation Framework**: All approaches tested under identical constraints (long-only, top-k holdings, transaction costs, turnover limits)
- **End-to-End Optimization**: GAT approach trains directly on Sharpe ratio objective rather than return forecasting as intermediate step
- **Practical Constraints Integration**: Real-world considerations like liquidity, transaction costs, and operational complexity built into model design

**Why This Solution Will Succeed:**
- **Comprehensive Comparison**: Systematic evaluation identifies which ML capabilities (clustering, temporal modeling, relational learning) add genuine value
- **Realistic Testing Environment**: Rolling backtests with out-of-sample validation mirror live deployment conditions
- **Academic Foundation**: Approaches grounded in peer-reviewed research with documented success in similar applications
- **Risk Management**: Maintains diversification principles while exploring enhanced allocation methods

**High-Level Vision:**
Deliver a production-ready framework that institutional portfolio managers can deploy to systematically improve risk-adjusted returns. The system provides clear guidance on when and how to apply advanced ML techniques, with built-in safeguards and performance monitoring.

## Target Users

Our solution serves institutional investment professionals who manage quantitative equity strategies, specifically focusing on two primary user segments:

### Primary User Segment: Quantitative Portfolio Managers

**Demographic/Firmographic Profile:**
- Senior portfolio managers and quantitative researchers at asset management firms, hedge funds, and institutional investment offices
- Teams managing $100M+ in mid-cap equity allocations with dedicated quantitative research capabilities
- Organizations with existing systematic trading infrastructure and risk management frameworks
- Firms already using factor-based or optimization-driven allocation approaches

**Current Behaviors and Workflows:**
- Monthly or quarterly portfolio rebalancing based on risk models and return forecasts
- Extensive backtesting and validation before deploying new strategies
- Multi-layered risk management with position limits, sector constraints, and turnover controls
- Performance attribution and factor exposure monitoring
- Integration with execution management systems for cost-efficient trading

**Specific Needs and Pain Points:**
- Improved risk-adjusted returns (Sharpe ratios) without significantly increasing operational complexity
- Better diversification that adapts to changing market relationships
- Reduced sensitivity to estimation errors that plague traditional mean-variance approaches
- Enhanced ability to capture temporal patterns in asset returns
- Transparent, interpretable allocation logic for client reporting and regulatory compliance

**Goals They're Trying to Achieve:**
- Consistent outperformance of benchmark indices and peer strategies
- Stable, implementable portfolio weights with reasonable turnover
- Scalable approaches that work across different market conditions
- Integration of alternative data and advanced analytics into investment process

### Secondary User Segment: Quantitative Researchers & Risk Analysts

**Demographic/Firmographic Profile:**
- PhD-level quantitative researchers, risk analysts, and portfolio construction specialists
- Academic researchers and graduate students focusing on computational finance
- Financial technology firms developing portfolio optimization solutions
- Regulatory bodies and consultants evaluating investment strategies

**Current Behaviors and Workflows:**
- Research and development of new allocation methodologies
- Academic paper publication and peer review processes
- Strategy validation through rigorous backtesting and statistical analysis
- Collaboration with portfolio management teams on implementation

**Specific Needs and Pain Points:**
- Access to cutting-edge techniques with reproducible research frameworks
- Comprehensive evaluation methodologies that account for practical constraints
- Understanding of when and why advanced techniques add value over simpler approaches
- Clean, bias-free datasets for strategy development and testing

**Goals They're Trying to Achieve:**
- Advancement of portfolio optimization science through rigorous empirical research
- Development of implementable solutions that bridge academic theory and practical application
- Publication of significant findings that advance the field

## Goals & Success Metrics

Our project establishes clear, measurable objectives that align with both academic rigor and practical business value:

### Business Objectives

- **Primary Performance Goal**: Achieve annualized Sharpe ratio improvement of ≥0.2 over equal-weight baseline across all ML approaches, measured net of transaction costs in rolling out-of-sample tests
- **Risk Management**: Maintain maximum drawdown within 5% of benchmark levels while improving risk-adjusted returns 
- **Operational Feasibility**: Demonstrate monthly portfolio turnover ≤20% to ensure realistic implementation costs and capacity
- **Robustness**: Show consistent outperformance across at least 75% of rolling evaluation windows (2016-2024)
- **Scalability**: Complete end-to-end pipeline processing and rebalancing within acceptable computational time limits (<4 hours monthly)

### User Success Metrics

- **Portfolio Managers**: Net Sharpe ratio improvement of ≥0.15 with max drawdown increase ≤3% vs current approaches
- **Risk Teams**: Successful integration with existing risk monitoring systems and constraint compliance ≥99.5%
- **Research Teams**: Publication-ready framework with full reproducibility and statistical significance testing
- **Implementation Teams**: Clear documentation and codebase enabling deployment within 6 months of adoption decision

### Key Performance Indicators (KPIs)

- **Annualized Sharpe Ratio**: Net risk-adjusted return metric, target ≥1.0 for best-performing ML approach vs baseline 0.8
- **Compound Annual Growth Rate (CAGR)**: Target 12-15% depending on market conditions and volatility regime
- **Maximum Drawdown (MDD)**: Risk control metric, maintain ≤25% during stress periods
- **Information Ratio vs S&P MidCap 400**: Target ≥0.5 for active return generation capability
- **Average Monthly Turnover**: Operational efficiency metric, target ≤15% for cost control
- **Tracking Error vs Benchmark**: Risk budget utilization, target 8-12% annual
- **Win Rate**: Percentage of months with positive excess returns, target ≥55%
- **Implementation Shortfall**: Trading cost impact, target ≤0.5% annual drag

## MVP Scope

The minimum viable product delivers a complete comparative framework for evaluating machine learning portfolio allocation approaches with realistic constraints and robust evaluation methodology.

### Core Features (Must Have)

- **Dynamic Universe Construction**: Build time-varying S&P MidCap 400 universe from Wikipedia historical membership data, avoiding survivorship bias and capturing realistic index dynamics from 2016-2024

- **Multi-Source Data Pipeline**: Integrated data collection from Stooq and Yahoo Finance with gap-filling, normalization, and quality assurance processes producing clean daily returns and volume panels

- **Three ML Implementation Frameworks**:
  - **HRP Module**: Hierarchical clustering-based allocation with correlation distance matrices and recursive bisection
  - **LSTM Module**: Sequence-to-sequence networks with 60-day lookback windows for temporal pattern recognition and return forecasting
  - **GAT Module**: Graph attention networks with correlation-based edge construction (k-NN, MST, TMFG options) and end-to-end Sharpe optimization

- **Unified Constraint System**: Long-only, top-k position limits, monthly turnover controls, and transaction cost modeling applied consistently across all approaches

- **Rolling Backtest Engine**: 36-month training / 12-month validation / 12-month testing protocol with walk-forward analysis and strict no-look-ahead validation

- **Performance Analytics Dashboard**: Automated calculation of Sharpe ratio, CAGR, maximum drawdown, turnover, and statistical significance testing across all approaches

- **Baseline Comparison Framework**: Equal-weight and mean-variance benchmarks implemented with identical constraint sets for fair comparison

### Out of Scope for MVP

- Real-time data feeds and live trading integration
- Alternative asset classes beyond US mid-cap equities
- High-frequency (daily/intraday) rebalancing strategies
- Advanced transaction cost modeling beyond linear approximation
- Multi-objective optimization (ESG, sector constraints, etc.)
- Interactive parameter tuning interface
- Alternative ML approaches beyond HRP/LSTM/GAT

### MVP Success Criteria

The MVP successfully demonstrates whether machine learning approaches can systematically improve risk-adjusted returns over traditional methods under realistic constraints. Success is defined as:

1. **Technical Implementation**: All three ML approaches execute without errors on full historical dataset with reproducible results
2. **Performance Validation**: At least one ML approach achieves statistically significant Sharpe ratio improvement ≥0.15 over both equal-weight and mean-variance baselines
3. **Operational Feasibility**: Monthly turnover remains ≤20% and computational processing completes within reasonable time limits
4. **Business Insight**: Clear recommendation on which approach (if any) merits further development for production deployment

## Post-MVP Vision

Following successful MVP validation, the framework can be expanded to address broader portfolio optimization challenges and serve additional market segments.

### Phase 2 Features

**Enhanced ML Capabilities:**
- Ensemble methods combining HRP, LSTM, and GAT predictions through weighted averaging or stacking approaches
- Alternative graph construction methods including sector-based, fundamental similarity, and dynamic correlation networks
- Advanced deep learning architectures such as Transformer networks for sequence modeling and Graph Transformers for enhanced attention mechanisms
- Multi-horizon forecasting with different rebalancing frequencies (weekly, daily) and adaptive timing

**Expanded Universe Coverage:**
- Large-cap (S&P 500) and small-cap (Russell 2000) equity universes with cross-market validation
- International developed and emerging market equity portfolios
- Multi-asset allocation including bonds, commodities, and alternative investments
- Sector-specific and thematic portfolio construction (technology, healthcare, ESG)

**Advanced Risk Management:**
- Regime-aware models that adapt allocation based on market volatility and correlation patterns
- Downside risk optimization targeting Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR)
- Dynamic hedging strategies incorporating derivatives and alternative risk management tools
- Real-time risk monitoring with early warning systems and position limit enforcement

### Long-term Vision

Transform portfolio optimization from a static, periodically-updated process to a dynamic, continuously-learning system that adapts to evolving market conditions. The platform becomes the standard infrastructure for quantitative portfolio management, supporting:

**Institutional Adoption:**
- Cloud-based SaaS platform serving asset managers, pension funds, and family offices
- API integration with existing portfolio management systems and execution platforms
- Regulatory compliance tools for risk reporting and audit trails
- Client reporting dashboards with performance attribution and explanation

**Research Platform:**
- Open-source academic research environment for portfolio optimization innovation
- Crowdsourced model development with community contributions and peer review
- Integration with alternative data sources (satellite imagery, social media, supply chain data)
- Collaborative research initiatives with academic institutions and industry partners

### Expansion Opportunities

**Vertical Integration:**
- Execution optimization combining portfolio construction with smart order routing
- Tax optimization for after-tax return maximization in taxable accounts
- Liability-driven investment solutions for pension funds and insurance companies
- Personalized portfolio management for high-net-worth individuals

**Technology Platform:**
- Real-time portfolio monitoring and alerts via mobile applications
- Natural language interfaces for portfolio managers to query and modify strategies
- Automated model retraining and performance monitoring with A/B testing frameworks
- Integration with ESG data providers for sustainable investing constraints

**Market Expansion:**
- Private market allocation strategies for venture capital, private equity, and real estate
- Cryptocurrency and digital asset portfolio optimization
- Commodity and agricultural portfolio management
- Cross-border currency hedging and international tax optimization

## Technical Considerations

The implementation requires careful attention to computational architecture, data infrastructure, and technology choices that support both research flexibility and production scalability.

### Platform Requirements

- **Target Platforms**: Local Linux-based development environment with GPU acceleration for deep learning models
- **Hardware Specifications**: RTX GeForce 5070Ti laptop GPU with 12GB VRAM supporting CUDA-accelerated PyTorch operations
- **Performance Requirements**: Handle 400+ asset S&P MidCap 400 universe with daily data processing; optimize memory usage for GAT training within 12GB VRAM constraints; leverage GPU acceleration for LSTM and GAT model training

### Technology Preferences

- **Frontend**: Python-based Jupyter notebooks for research, analysis, and model development; matplotlib/seaborn for visualization; Streamlit application for interactive dashboard (future consideration)
- **Backend**: Python 3.9+ with PyTorch for deep learning (GAT, LSTM), scikit-learn for traditional ML (HRP), NetworkX for graph analysis, and pandas/numpy for data manipulation
- **Database**: Parquet files for efficient storage and retrieval of historical price/volume panels; local filesystem-based data management
- **Development Environment**: Local development with uv environment management; Jupyter Lab for interactive development

### Architecture Considerations

- **Repository Structure**: Current monorepo requires reorganization into proper module structure:
  - `data/` - Raw and processed datasets (prices.parquet, volume.parquet, returns_daily.parquet)
  - `src/models/` - Model implementations (GAT complete, LSTM and HRP to be formalized)
  - `src/preprocessing/` - Data collection and cleaning utilities (existing scripts need organization)
  - `src/evaluation/` - Backtesting and performance analytics frameworks
  - `src/utils/` - Shared utilities and helper functions
  
- **Current Implementation Status**:
  - ✅ **Data Collection Pipeline**: Complete implementation with Wikipedia scraping, Stooq/Yahoo Finance integration, and gap-filling
  - ✅ **GAT Framework**: Functional graph attention network with correlation-based edge construction (k-NN, MST, TMFG)
  - 🔄 **Code Organization**: Substantial refactoring needed to move bloat code into proper folder structure
  - ❌ **LSTM Module**: Requires formal implementation of sequence models with 60-day lookback windows
  - ❌ **HRP Module**: Needs implementation of hierarchical clustering and recursive bisection allocation

- **GPU Memory Optimization**: Design GAT and LSTM architectures to efficiently utilize 12GB VRAM through batch processing, gradient checkpointing, and mixed-precision training where applicable

- **Local Development Focus**: No hosting infrastructure required; emphasis on reproducible local experimentation with clear documentation for environment setup

## Constraints & Assumptions

### Constraints

- **Budget**: Academic project with no external funding; reliance on open-source tools and free data sources (Wikipedia, Stooq, Yahoo Finance)
- **Timeline**: Implementation and evaluation must be completed within one month or less
- **Resources**: Single developer working part-time; 12GB GPU VRAM limits model complexity and batch sizes; local development environment only
- **Technical**: Monthly rebalancing frequency due to computational constraints; S&P MidCap 400 universe size limited by memory and processing capabilities

### Key Assumptions

- S&P MidCap 400 provides sufficient complexity and liquidity for meaningful ML portfolio optimization research
- 2016-2024 evaluation period captures diverse market conditions including bull markets, corrections, and volatility regimes
- Monthly rebalancing frequency balances signal capture with realistic transaction cost considerations
- Free data sources (Stooq, Yahoo Finance) provide adequate data quality for research purposes despite potential gaps or inaccuracies
- Linear transaction cost modeling approximates real institutional trading costs sufficiently for comparative evaluation
- Existing GAT implementation can be extended and optimized for portfolio allocation without fundamental redesign
- Academic-level computational resources are sufficient for thorough backtesting and model comparison

## Risks & Open Questions

### Key Risks

- **Timeline Risk**: One-month timeline may be insufficient for proper LSTM and HRP implementation, testing, and comprehensive evaluation
- **Data Quality Risk**: Free data sources may contain gaps, errors, or inconsistencies that affect model performance and evaluation validity
- **Overfitting Risk**: Limited evaluation period and potential data snooping through multiple model iterations may lead to false positive results
- **Computational Limitations**: 12GB VRAM constraint may limit GAT model complexity and batch sizes, potentially affecting performance
- **Implementation Risk**: Existing codebase organization issues may slow development and introduce bugs during refactoring

### Open Questions

- Can LSTM and HRP modules be implemented and validated within the compressed timeline?
- What is the optimal GAT architecture configuration given GPU memory constraints?
- How sensitive are results to transaction cost assumptions and data quality issues?
- Will performance improvements be statistically significant given the limited evaluation sample?
- How should hyperparameter tuning be balanced with overfitting prevention in a short timeframe?

### Areas Needing Further Research

- Alternative data augmentation techniques to expand training sample size
- Ensemble methods for combining HRP, LSTM, and GAT predictions
- Robustness testing across different market regimes and stress periods
- Scalability analysis for larger asset universes and higher-frequency rebalancing

## Next Steps

### Immediate Actions

1. **Code Organization** (Week 1): Refactor existing codebase into proper module structure, moving data collection and GAT implementation into organized folders
2. **LSTM Implementation** (Week 1-2): Develop sequence-to-sequence models with 60-day lookback windows and return forecasting capabilities
3. **HRP Implementation** (Week 2): Create hierarchical clustering module with correlation distance matrices and recursive bisection allocation
4. **Unified Evaluation Framework** (Week 2-3): Build rolling backtest engine with consistent constraint application across all approaches
5. **Performance Analysis** (Week 3-4): Run comprehensive backtests, calculate performance metrics, and generate comparative analysis
6. **Documentation and Reporting** (Week 4): Compile results, create visualizations, and prepare final project report

### PM Handoff

This Project Brief provides the full context for the Portfolio Optimization with Machine Learning Techniques project. The next phase requires transitioning from planning to implementation mode, focusing on rapid development of missing components (LSTM, HRP) while leveraging existing data pipeline and GAT framework. Given the compressed timeline, priority should be placed on achieving working implementations rather than extensive optimization, with clear documentation of assumptions and limitations for future enhancement.
