# Scripts Directory

This directory contains production pipeline scripts for the portfolio-ml project.

## Production Pipeline

The production pipeline consists of three main scripts that should be run in sequence:

### 1. Data Collection Pipeline
**File:** `data_collection_pipeline.py`
**Purpose:** Collects S&P MidCap 400 historical data from multiple sources
**Features:**
- Single-ticker sequential approach (API-respectful)
- Smart source selection (tests Stooq availability first)
- Comprehensive logging and progress tracking
- Forward-fill only (no temporal leakage)
- Saves to: `data/final_new_pipeline/`

**Usage:**
```bash
python scripts/data_collection_pipeline.py
```

### 2. Comprehensive Backtest
**File:** `run_comprehensive_backtest.py`
**Purpose:** Trains models and executes rolling backtests with walk-forward analysis
**Features:**
- Trains HRP, LSTM, and GAT models fresh each month
- 96 monthly rolling windows (2016-2024)
- Constraint enforcement and transaction costs
- Saves to: `results/ml_backtest_rolling/`

**Usage:**
```bash
python scripts/run_comprehensive_backtest.py
```

### 3. Performance Analytics
**File:** `run_performance_analytics.py`
**Purpose:** Statistical validation and publication-ready analysis
**Features:**
- Bootstrap confidence intervals (10,000 samples)
- Jobson-Korkie hypothesis testing
- Multiple comparison corrections
- Publication-ready tables in APA format
- Saves to: `results/performance_analytics/`

**Usage:**
```bash
python scripts/run_performance_analytics.py
```

## Legacy Scripts

**Location:** `../legacy_scripts/`
**Purpose:** Historical reference for original implementation
**Status:** Not for production use

## Experiments

**Location:** `../experiments/`
**Purpose:** Experimental frameworks for model variation testing
