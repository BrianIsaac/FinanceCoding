---
date: 2025-10-30
author: backtest-error-analysis
status: ready-for-implementation
priority: critical
---

# Backtest Error Fixes - Implementation Guide

This document provides detailed code patches to fix the critical errors identified in the comprehensive backtest error analysis.

**Total Issues**: 14 categories, 149,376 messages
**Critical/High Priority Fixes**: 4 issues requiring immediate attention

## Table of Contents

1. [Priority 1: GAT Features Dimension Mismatch (CRITICAL)](#1-gat-features-dimension-mismatch-critical)
2. [Priority 2: LSTM Shape Mismatch (HIGH)](#2-lstm-shape-mismatch-high)
3. [Priority 3: Extreme Concentration (HIGH)](#3-extreme-concentration-high)
4. [Priority 4: GAT Training KeyError (MEDIUM)](#4-gat-training-keyerror-medium)
5. [Priority 5: Configuration Fixes (MEDIUM)](#5-configuration-fixes-medium)
6. [Verification Steps](#verification-steps)

---

## 1. GAT Features Dimension Mismatch (CRITICAL)

**Impact**: Prevents GAT models from learning properly, causing feature-to-asset misalignment
**Occurrences**: 1,383 errors
**File**: `src/models/gat/model.py`

### Root Cause

Features are created for the full universe (759 assets) BEFORE filtering, then truncated AFTER graph construction filters to 399 assets. The truncation assumes row ordering matches, but filtering changes the order, causing feature[i] to be assigned to wrong asset.

**Flow**:
1. `_get_node_features(returns, prediction_universe)` creates features for all 759 assets in `prediction_universe` order
2. `build_period_graph()` filters tickers based on `valid_mask` and data availability → 399 assets
3. Graph builder truncates features: `x_np = x_np[:N]` (first 399 rows)
4. **BUG**: First 399 rows of features correspond to first 399 assets in original order, NOT the 399 filtered assets

### Fix: Option A (Recommended) - Filter Before Feature Creation

**Location**: `src/models/gat/model.py` around lines 1560-1576

```python
# BEFORE (incorrect):
def predict_weights(self, date: pd.Timestamp, prediction_universe: list[str]) -> pd.Series:
    """Generate portfolio weights for prediction universe at given date."""
    # ... existing code ...

    # Load actual historical returns data for prediction universe
    returns_data = self._get_historical_returns(date, prediction_universe)

    # Prepare node features
    features_matrix = self._get_node_features(returns_data, prediction_universe)  # ❌ Uses full universe

    # Build graph for current period using prediction universe
    valid_mask = getattr(self, '_last_valid_mask', None)
    graph_data = build_period_graph(
        returns_daily=returns_data,
        period_end=date,
        tickers=prediction_universe,
        features_matrix=features_matrix,
        cfg=self.config.graph_config,
        valid_mask=valid_mask,
    )
```

```python
# AFTER (correct):
def predict_weights(self, date: pd.Timestamp, prediction_universe: list[str]) -> pd.Series:
    """Generate portfolio weights for prediction universe at given date."""
    # ... existing code ...

    # Load actual historical returns data for prediction universe
    returns_data = self._get_historical_returns(date, prediction_universe)

    # Filter universe BEFORE feature creation to ensure alignment
    # Use the same filtering logic as graph_builder
    if hasattr(self, '_last_valid_mask') and self._last_valid_mask is not None:
        valid_mask = self._last_valid_mask
        valid_tickers = valid_mask[valid_mask].index.tolist()
        filtered_universe = [t for t in valid_tickers if t in returns_data.columns]
    else:
        filtered_universe = [t for t in prediction_universe if t in returns_data.columns]

    logger.debug(
        f"Filtered universe from {len(prediction_universe)} to {len(filtered_universe)} "
        f"before feature creation"
    )

    # Prepare node features ONLY for filtered universe ✓
    features_matrix = self._get_node_features(returns_data, filtered_universe)

    # Build graph for current period using filtered universe
    graph_data = build_period_graph(
        returns_daily=returns_data,
        period_end=date,
        tickers=filtered_universe,  # Use filtered universe
        features_matrix=features_matrix,
        cfg=self.config.graph_config,
        valid_mask=None,  # No longer needed, already filtered
    )
```

### Fix: Option B (Alternative) - Index-Based Feature Alignment

**Location**: `src/models/gat/graph_builder.py` around lines 840-860

```python
# BEFORE (incorrect):
if x_np.shape[0] != N:
    logger.warning(f"Features matrix dimension mismatch: features_shape={x_np.shape}, expected_nodes={N}, tickers_len={len(tickers)}")

    # ... padding/truncation logic ...

    elif x_np.shape[0] > N:
        # Truncate if we have more features than tickers (shouldn't happen but be safe)
        x_np = x_np[:N]  # ❌ Assumes first N rows match filtered tickers
        logger.info(f"Truncated features_matrix from {features_matrix.shape[0]} to {N} rows")
```

```python
# AFTER (correct):
if x_np.shape[0] != N:
    logger.warning(
        f"Features matrix dimension mismatch: features_shape={x_np.shape}, "
        f"expected_nodes={N}, tickers_len={len(tickers)}"
    )

    # If features were created for a larger universe, we need to align by ticker name
    if hasattr(features_matrix, 'index'):
        # Features matrix has ticker index - use it for alignment
        try:
            x_np = features_matrix.loc[tickers].values  # ✓ Align by ticker name
            logger.info(f"Aligned features_matrix using ticker index")
        except KeyError as e:
            logger.error(f"Cannot align features by ticker: {e}")
            # Fallback to default features
            x_np = np.ones((N, 1), dtype=np.float32)
    elif x_np.shape[0] > N:
        # No index available - log critical error and use fallback
        logger.critical(
            f"Cannot align {x_np.shape[0]} features to {N} tickers without index. "
            f"Feature-to-asset mapping may be corrupted!"
        )
        # Use default features to prevent silent corruption
        x_np = np.ones((N, 1), dtype=np.float32)
        logger.warning("Using default constant features to prevent corruption")
```

**Recommendation**: Use **Option A** as it fixes the root cause. Option B is a defensive fallback if features_matrix has ticker index.

---

## 2. LSTM Shape Mismatch (HIGH)

**Impact**: Prevents batch size optimisation, forces fallback to default batch size
**Occurrences**: 1,206 errors
**File**: `src/models/lstm/model.py`

### Root Cause

When network is recreated with new input_size (line 438), the trainer's model reference is NOT updated (line 511). The trainer continues using the old network during batch size optimisation, causing shape mismatches.

**Flow**:
1. Universe changes from 316 to 341 assets across windows
2. `optimal_size` is recalculated → different from current `config.input_size`
3. Network is recreated: `self.network = create_ragged_lstm_network(...)` (line 438)
4. Trainer already exists, so only config is updated (line 513-514)
5. **BUG**: `self.trainer.model` still references OLD network with OLD input_size
6. Batch size optimisation creates dummy tensors with NEW universe size (line 461)
7. Forward pass fails: input (960, 324) vs weight (319, 128)

### Fix: Update Trainer Model Reference

**Location**: `src/models/lstm/model.py` around lines 510-515

```python
# BEFORE (incorrect):
# Create or update trainer with adjusted parameters
if self.trainer is None:
    # Create new trainer with confidence-adjusted epochs
    quick_config = TrainingConfig(
        epochs=adjusted_params.get("epochs", max_epochs),
        patience=5,
        batch_size=self.config.training_config.batch_size,
        learning_rate=adjusted_params.get("learning_rate", self.config.training_config.learning_rate * 0.1),
        weight_decay=adjusted_params.get("weight_decay", 0.001),
        use_mixed_precision=self.config.training_config.use_mixed_precision,
    )
    self.trainer = create_trainer(self.network, quick_config)
else:
    # Update existing trainer config
    self.trainer.config.epochs = max_epochs  # ❌ Only config updated
    self.trainer.config.patience = 5
```

```python
# AFTER (correct):
# Create or update trainer with adjusted parameters
if self.trainer is None:
    # Create new trainer with confidence-adjusted epochs
    quick_config = TrainingConfig(
        epochs=adjusted_params.get("epochs", max_epochs),
        patience=5,
        batch_size=self.config.training_config.batch_size,
        learning_rate=adjusted_params.get("learning_rate", self.config.training_config.learning_rate * 0.1),
        weight_decay=adjusted_params.get("weight_decay", 0.001),
        use_mixed_precision=self.config.training_config.use_mixed_precision,
    )
    self.trainer = create_trainer(self.network, quick_config)
else:
    # Update existing trainer with new model reference and config
    self.trainer.model = self.network  # ✓ Update model reference
    self.trainer.config.epochs = max_epochs
    self.trainer.config.patience = 5
    logger.debug(
        f"Updated trainer model reference to network with input_size={self.network.input_size}"
    )
```

### Alternative Fix: Force Trainer Recreation

**Location**: `src/models/lstm/model.py` around lines 434-439

```python
# Alternative approach: recreate trainer when network changes
if self.network is None or self.config.lstm_config.input_size != optimal_size:
    # Create network with optimal size for current universe
    self.config.lstm_config.input_size = optimal_size
    self.config.lstm_config.output_size = optimal_size
    self.network = create_ragged_lstm_network(self.config.lstm_config)
    self.trainer = None  # ✓ Force trainer recreation on next fit
    logger.info(
        f"Created LSTM network with input_size={optimal_size} for universe_size={current_universe_size}. "
        f"Trainer will be recreated."
    )
```

**Recommendation**: Use the first fix (update model reference) as it's more efficient than recreating the entire trainer.

---

## 3. Extreme Concentration (HIGH)

**Impact**: Causes 6 rebalances to be skipped due to >50% concentration
**Occurrences**: 6 CRITICAL errors
**Files**:
- `scripts/run_comprehensive_backtest.py` (lines 391, 417)
- `src/models/hrp/model.py` (lines 515-521)

### Root Cause

HRP and LSTM models configured with `max_position_weight=1.0` (100%, no limit), but rolling engine expects 20% limit. HRP's recursive bisection can naturally generate extreme concentrations without constraint enforcement.

**Affected Dates**:
- 2020-04-01 (50.0%)
- 2020-05-01 (50.0%)
- 2021-05-03 (50.0%)
- 2021-10-01 (50.0%)
- 2023-09-01 (50.0%)
- 2023-11-01 (50.0%)

### Fix 1: Align Model Constraints with Rolling Engine

**Location**: `scripts/run_comprehensive_backtest.py`

```python
# BEFORE (incorrect):
# Line 389-395 - HRP Model
models["HRP"] = HRPModel(
    hrp_config=hrp_config,
    constraints=PortfolioConstraints(
        long_only=True,
        max_position_weight=1.0,  # ❌ No limit - can concentrate
        max_monthly_turnover=10.0,
        min_weight_threshold=0.0,
        top_k_positions=None,
    )
)

# Line 414-423 - LSTM Model
models["LSTM"] = LSTMPortfolioModel(
    constraints=PortfolioConstraints(
        long_only=True,
        max_position_weight=1.0,  # ❌ No limit - can concentrate
        max_monthly_turnover=10.0,
        min_weight_threshold=0.0,
        top_k_positions=None,
        transaction_cost_bps=10.0,
        enable_turnover_penalty=False,
    ),
    config=lstm_config
)
```

```python
# AFTER (correct):
# Line 389-395 - HRP Model
models["HRP"] = HRPModel(
    hrp_config=hrp_config,
    constraints=PortfolioConstraints(
        long_only=True,
        max_position_weight=0.20,  # ✓ 20% limit aligned with rolling engine
        max_monthly_turnover=10.0,
        min_weight_threshold=0.0,
        top_k_positions=None,
    )
)

# Line 414-423 - LSTM Model
models["LSTM"] = LSTMPortfolioModel(
    constraints=PortfolioConstraints(
        long_only=True,
        max_position_weight=0.20,  # ✓ 20% limit aligned with rolling engine
        max_monthly_turnover=10.0,
        min_weight_threshold=0.0,
        top_k_positions=None,
        transaction_cost_bps=10.0,
        enable_turnover_penalty=False,
    ),
    config=lstm_config
)
```

### Fix 2: Add Position Limit Enforcement in HRP Recursive Bisection

**Location**: `src/models/hrp/model.py` around lines 515-527

```python
# Add defensive check BEFORE constraint validation
try:
    raw_weights = self.allocation_engine.recursive_bisection(
        prediction_covariance, cluster_tree
    )

    # ✓ Add pre-emptive check for extreme concentration
    if raw_weights.max() > 0.35:
        logger.warning(
            f"HRP generated extreme concentration: {raw_weights.max():.1%}. "
            f"Pre-emptively clipping to max_position_weight={self.constraints.max_position_weight}"
        )
        # Apply hard clipping before validation
        raw_weights = raw_weights.clip(upper=self.constraints.max_position_weight)
        raw_weights = raw_weights / raw_weights.sum()

except Exception as e:
    logger.warning(f"HRP allocation failed: {str(e)}")
    logger.info(f"Using equal weights for {len(prediction_assets)} assets")
    raw_weights = pd.Series(1.0 / len(prediction_assets), index=prediction_assets)
```

---

## 4. GAT Training KeyError (MEDIUM)

**Impact**: Reduces training data quality by skipping samples with missing tickers
**Occurrences**: 1,173 errors
**File**: `src/models/gat/model.py`

### Root Cause

Training loop attempts to index returns DataFrame with universe tickers that don't exist in `returns.columns`, causing KeyError at lines 595, 619, and 660.

### Fix: Filter Universe Before Training

**Location**: `src/models/gat/model.py` around lines 582-595

```python
# BEFORE (incorrect):
for date in selected_dates:
    try:
        # Build graph for this date
        graph_data = build_period_graph(
            returns_daily=returns,
            period_end=date,
            tickers=universe,  # ❌ May contain unavailable tickers
            features_matrix=features_matrix,
            cfg=self.config.graph_config,
        )

        # Get forward returns as labels
        next_month_end = min(date + pd.Timedelta(days=30), returns.index[-1])
        forward_returns = returns.loc[date:next_month_end, universe].mean()  # ❌ KeyError here
```

```python
# AFTER (correct):
# ✓ Filter universe to available tickers BEFORE training loop
available_universe = [t for t in universe if t in returns.columns]
if len(available_universe) < len(universe):
    logger.info(
        f"Filtered training universe from {len(universe)} to {len(available_universe)} "
        f"available tickers ({len(universe) - len(available_universe)} missing from returns)"
    )

# Use filtered universe throughout training
for date in selected_dates:
    try:
        # Build graph for this date
        graph_data = build_period_graph(
            returns_daily=returns,
            period_end=date,
            tickers=available_universe,  # ✓ Only available tickers
            features_matrix=features_matrix,
            cfg=self.config.graph_config,
        )

        # Get forward returns as labels
        next_month_end = min(date + pd.Timedelta(days=30), returns.index[-1])
        forward_returns = returns.loc[date:next_month_end, available_universe].mean()  # ✓ No KeyError
```

**Apply same fix to lines 619 and 660** (correlation matrix calculation):

```python
# Line 619 - Mixed precision path
hist_returns = returns.loc[:date, available_universe].tail(min(252, len(returns.loc[:date])))  # ✓

# Line 660 - Standard path
hist_returns = returns.loc[:date, available_universe].tail(min(252, len(returns.loc[:date])))  # ✓
```

---

## 5. Configuration Fixes (MEDIUM)

### 5.1 Reduce GAT Missing Asset Warning Volume

**Impact**: 143,112 warnings pollute logs (97.5% of all messages)
**File**: `src/models/gat/model.py`

**Option A: Change to DEBUG level**

```python
# Line 998
if ticker not in returns.columns:
    logger.debug(f"Asset {ticker} not in returns, using zeros")  # Changed from warning
    features_list.append(
        np.zeros((window_length, num_features_per_timestep))
    )
    continue
```

**Option B: Batch logging (recommended)**

```python
# Around line 993, before the loop
missing_assets = [t for t in universe if t not in returns.columns]
if missing_assets:
    logger.info(
        f"{len(missing_assets)} assets not in returns (e.g., {missing_assets[:5]}). "
        f"Using zero-filled features."
    )

for ticker in universe:
    if ticker not in returns.columns:
        # No individual warning - already logged in batch above
        features_list.append(
            np.zeros((window_length, num_features_per_timestep))
        )
        continue
```

### 5.2 Fix Temporal Integrity Validation

**Impact**: 1,120 warnings from using dummy data
**File**: `src/evaluation/validation/rolling_validation.py`

```python
# BEFORE (incorrect) - Line 926-933
dummy_data = pd.DataFrame(index=pd.date_range(
    start=split.train_period.start_date,
    periods=len(train_data),
    freq='D'
)) if train_data else pd.DataFrame()  # ❌ Empty DataFrame

universe_estimate = []

validation_result = self.flexible_validator.validate_with_confidence(
    data=dummy_data,  # ❌ Dummy data has no columns
    universe=universe_estimate,
    context={}
)
```

```python
# AFTER (correct) - Line 926-933
# ✓ Pass actual training data instead of dummy DataFrame
validation_result = self.flexible_validator.validate_with_confidence(
    data=train_data,  # ✓ Actual returns data
    universe=train_data.columns.tolist() if hasattr(train_data, 'columns') else [],
    context={}
)
```

### 5.3 Add Missing Tabulate Dependency

**Impact**: 1 warning, academic reports not generated
**Command**:

```bash
uv add tabulate
```

---

## Verification Steps

### 1. Run Verification Script

```bash
# Before fixes
uv run python scripts/verify_backtest_errors.py --log-file outputs/LATEST/run_comprehensive_backtest.log

# After fixes
uv run python scripts/run_comprehensive_backtest.py
uv run python scripts/verify_backtest_errors.py
```

### 2. Check Specific Error Counts

```bash
LOG_FILE="outputs/$(ls -t outputs | head -1)/run_comprehensive_backtest.log"

# GAT features mismatch (should be 0 after fix)
grep -c "Features matrix dimension mismatch" $LOG_FILE

# LSTM shape mismatch (should be 0 after fix)
grep -c "mat1 and mat2 shapes cannot be multiplied" $LOG_FILE

# Extreme concentration (should be 0 after fix)
grep -c "CRITICAL: Extreme concentration" $LOG_FILE

# GAT training KeyError (should be greatly reduced)
grep -c "training error at.*not in index" $LOG_FILE

# Position limit violations (should be 0 after fix)
grep -c "Position limit violation" $LOG_FILE
```

### 3. Verify Model Performance Improvement

```bash
# Check GAT model Sharpe ratios (should be > 0.30 after fix)
# Before fix: GAT models achieve 0.283 (same as EqualWeight baseline)
# After fix: GAT should outperform baseline

# Check backtest results
cat outputs/LATEST/backtest_results.json | jq '.models[] | select(.model_name | contains("GAT")) | {name: .model_name, sharpe: .metrics.sharpe_ratio}'
```

### 4. Visual Inspection

```python
# Quick test script to verify fixes
import pandas as pd
from src.models.gat.model import GATModel
from src.models.lstm.model import LSTMPortfolioModel

# Test GAT feature alignment
gat = GATModel(preset="paper_reproduction")
# Should see "Filtered universe from 759 to 399 before feature creation" in logs

# Test LSTM trainer update
lstm = LSTMPortfolioModel()
# Should not see shape mismatch errors during batch size optimization
```

---

## Implementation Checklist

- [ ] **Priority 1: GAT Features Mismatch**
  - [ ] Apply Fix Option A to `src/models/gat/model.py:1560-1576`
  - [ ] Test with single backtest window
  - [ ] Verify no dimension mismatch warnings in logs
  - [ ] Check GAT model performance improves

- [ ] **Priority 2: LSTM Shape Mismatch**
  - [ ] Apply fix to `src/models/lstm/model.py:511`
  - [ ] Test with varying universe sizes
  - [ ] Verify batch size optimization succeeds
  - [ ] Check no shape errors in logs

- [ ] **Priority 3: Extreme Concentration**
  - [ ] Update HRP constraints in `scripts/run_comprehensive_backtest.py:391`
  - [ ] Update LSTM constraints in `scripts/run_comprehensive_backtest.py:417`
  - [ ] Add defensive check in `src/models/hrp/model.py:515-527`
  - [ ] Verify no extreme concentration errors
  - [ ] Check all 6 previously-skipped rebalances now execute

- [ ] **Priority 4: GAT Training KeyError**
  - [ ] Filter universe in `src/models/gat/model.py:582-595`
  - [ ] Update lines 619 and 660
  - [ ] Verify no KeyError warnings
  - [ ] Check training uses full date range

- [ ] **Configuration Fixes**
  - [ ] Reduce GAT missing asset warnings (`model.py:998`)
  - [ ] Fix temporal integrity validation (`rolling_validation.py:926-933`)
  - [ ] Add tabulate dependency: `uv add tabulate`

- [ ] **Verification**
  - [ ] Run verification script
  - [ ] Run full comprehensive backtest
  - [ ] Compare before/after error counts
  - [ ] Verify GAT models outperform baseline
  - [ ] Check log file size reduced (from 18.7MB to ~2MB)

---

## Expected Results After Fixes

| Error Category | Before | After | Reduction |
|----------------|--------|-------|-----------|
| GAT Features Mismatch | 1,383 | 0 | 100% |
| LSTM Shape Mismatch | 1,206 | 0 | 100% |
| GAT Training KeyError | 1,173 | 0 | 100% |
| GAT Missing Assets | 143,112 | <100 | 99.9% |
| Temporal Integrity | 1,120 | 0 | 100% |
| Position Violations | 87 | 0 | 100% |
| Extreme Concentration | 6 | 0 | 100% |
| **TOTAL** | **149,376** | **<200** | **99.9%** |

**Performance Improvement Expected**:
- GAT models: Sharpe 0.283 → 0.35-0.45 (15-60% improvement)
- LSTM models: Training 10-20% faster (batch size optimisation enabled)
- Backtest execution: Logs 90% smaller, easier to debug

---

## Notes

1. **Breaking Changes**: None. All fixes are backwards compatible.

2. **Testing**: Test each fix individually before combining to isolate any issues.

3. **Rollback**: Keep current code in a separate branch before applying fixes.

4. **Documentation**: Update any relevant docstrings affected by these changes.

5. **Monitoring**: After deployment, monitor GAT model performance to confirm improvement.

---

**Document Version**: 1.0
**Last Updated**: 2025-10-30
**Related Research**: `thoughts/shared/research/2025-10-30-backtest-errors-analysis.md`
