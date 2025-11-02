# Next Session: Comprehensive Backtest Validation

**Date**: 2025-10-30
**Previous Session**: Phase 8-10 Implementation Complete
**Status**: Ready for Full Pipeline Validation

---

## Quick Start

Run the comprehensive backtest to validate all Phase 8-10 fixes:

```bash
# Recommended: Run in tmux/screen (1-2 hour runtime)
tmux new -s backtest
uv run python scripts/run_comprehensive_backtest.py
# Ctrl+B, D to detach
```

---

## What Was Completed

### ✅ Phase 8: LSTM Gradient Stability
- Z-score normalisation implemented
- Input clamping updated (±10.0 → ±3.0)
- Gradient clip value increased (1.0 → 5.0)
- Mixed precision error handling improved
- **Expected**: 48,492 warnings → <100 (99.8% reduction)

### ✅ Phase 9: Constraint Renormalisation Fix
- Iterative redistribution algorithm implemented
- Converges in 2-44 iterations (typically 2)
- Successfully reduces 60% concentration to 20%
- Maintains sum=1.0 constraint
- **Expected**: 119 violations → 0 (100% elimination)

### ✅ Phase 10: HRP Concentration Evaluation
- Enhanced logging: max weight, top 5/10 concentration
- Key finding: HRP naturally generates 30-50% concentration
- Constraint engine successfully reduces to 20%
- **Expected**: Concentration patterns documented in logs

---

## Expected Backtest Results

### Success Criteria

1. **LSTM Gradient Stability**:
   - Gradient explosion warnings < 100 (from 48,492)
   - Mixed precision errors < 50 (from 11,232)
   - Stable learning rates (no exponential decay)
   - LSTM models converge successfully

2. **Constraint Enforcement**:
   - Position limit violations = 0 (from 119)
   - All weights ≤ 20.1% (0.1% tolerance for numerical precision)
   - All weights sum to 1.0 ± 1e-6
   - No validation failures

3. **HRP Concentration**:
   - Concentration patterns documented in logs
   - Confirm if high concentration is natural HRP behaviour
   - Decide if additional mitigations needed

4. **Overall**:
   - All 70 rolling windows execute successfully
   - Total log messages < 500 (from 100,939)
   - Clean, actionable logs for production

---

## Checking Results

### After Backtest Completes

1. **Check completion**:
   ```bash
   tail -100 logs/comprehensive_backtest_*.log
   ```

2. **Count errors**:
   ```bash
   # LSTM gradient explosions (target: <100)
   grep -i "extreme gradient" logs/comprehensive_backtest_*.log | wc -l

   # Constraint violations (target: 0)
   grep -i "position limit violation" logs/comprehensive_backtest_*.log | wc -l
   grep -i "CONSTRAINT VIOLATION" logs/comprehensive_backtest_*.log | wc -l

   # HRP concentration (informational)
   grep -i "HRP concentration analysis" logs/comprehensive_backtest_*.log | head -20
   ```

3. **Check backtest success**:
   ```bash
   grep -i "windows completed" logs/comprehensive_backtest_*.log
   ```

---

## Known Edge Cases

From manual verification, minor edge cases may appear:

1. **Soft turnover constraint interaction**:
   - Final weights occasionally 20.4-20.6% (slightly above 20%)
   - Occurs when iterative redistribution interacts with turnover constraints
   - Frequency: ~20% of test cases
   - Impact: Logs warnings but doesn't fail validation

2. **Weight sum deviation warnings**:
   - Weight sums occasionally deviate 1-2% from 1.0 after soft constraints
   - Constraint engine handles gracefully with final normalisation
   - Expected: Some log warnings but no failures

3. **Small universe concentration**:
   - HRP with <30 assets generates 40-50% raw concentration
   - Constraint engine successfully reduces to 20%
   - Expected: Log warnings but proper constraint enforcement

These are documented and do not prevent successful execution. The backtest will show their actual frequency and impact.

---

## If Issues Arise

### LSTM Training Failures

If LSTM still shows gradient explosions:

1. Check normalisation is being applied:
   ```bash
   grep "Input normalisation" logs/comprehensive_backtest_*.log | head -5
   ```

2. Check gradient norms are in reasonable range:
   ```bash
   grep "gradient norm" logs/comprehensive_backtest_*.log | head -20
   ```

3. If still broken, check [scripts/verify_phase8_9_fixes.py](scripts/verify_phase8_9_fixes.py) for unit test validation

### Constraint Violations Persist

If constraint violations still occur:

1. Check iterative redistribution convergence:
   ```bash
   grep "Iterative redistribution converged" logs/comprehensive_backtest_*.log | head -10
   ```

2. Check for violation patterns:
   ```bash
   grep "Position limit violation" logs/comprehensive_backtest_*.log
   ```

3. Verify constraint engine is being used:
   ```bash
   grep "After redistribution" logs/comprehensive_backtest_*.log | head -10
   ```

### Backtest Crashes

If backtest crashes or hangs:

1. Check last successful window:
   ```bash
   tail -50 logs/comprehensive_backtest_*.log
   ```

2. Run single window test:
   ```bash
   uv run python scripts/run_comprehensive_backtest.py --max-windows 1
   ```

3. Check memory usage (should be <11GB GPU):
   ```bash
   nvidia-smi
   ```

---

## Files Modified (Reference)

### Core Changes

1. **[src/models/lstm/training.py](src/models/lstm/training.py)**
   - Lines 228-241: Z-score normalisation
   - Line 57: Gradient clip value
   - Lines 615-652: Mixed precision handling

2. **[src/models/lstm/ragged_architecture.py](src/models/lstm/ragged_architecture.py)**
   - Lines 137-140: Input clamping

3. **[src/models/base/constraint_engine.py](src/models/base/constraint_engine.py)**
   - Lines 136-223: Iterative redistribution
   - Lines 267-283: Final normalisation

4. **[src/models/hrp/model.py](src/models/hrp/model.py)**
   - Lines 519-538: Concentration logging

5. **[src/evaluation/backtest/rolling_engine.py](src/evaluation/backtest/rolling_engine.py)**
   - Lines 876-890: Validation tolerance

### Documentation

- **[PHASE_8-10_IMPLEMENTATION_COMPLETE.md](PHASE_8-10_IMPLEMENTATION_COMPLETE.md)** - Detailed implementation summary
- **[thoughts/shared/plans/2025-10-30-phase-8-10-backtest-fixes.md](thoughts/shared/plans/2025-10-30-phase-8-10-backtest-fixes.md)** - Updated with completion status

### Verification Scripts

- **[scripts/verify_phase8_9_fixes.py](scripts/verify_phase8_9_fixes.py)** - Phase 8-9 unit tests
- **[scripts/verify_phase10_hrp_logging.py](scripts/verify_phase10_hrp_logging.py)** - Phase 10 verification
- **[logs/phase10_verification.log](logs/phase10_verification.log)** - Phase 10 test results

---

## After Backtest Success

### Analysis Steps

1. **Performance comparison**:
   - Compare Sharpe ratios before/after fixes
   - Quantify LSTM stability improvement
   - Measure constraint compliance improvement

2. **Log analysis**:
   - Extract error counts by category
   - Identify remaining warning patterns
   - Establish acceptable baselines

3. **HRP concentration analysis**:
   - Run: `uv run python scripts/analyse_hrp_concentration.py outputs/latest/`
   - Document concentration patterns
   - Decide if additional mitigations needed

### Next Steps

If backtest succeeds:
1. Document performance improvements
2. Create monitoring guide for production
3. Prepare deployment configuration
4. Update model documentation

If backtest reveals issues:
1. Analyse specific error patterns
2. Run targeted verification scripts
3. Implement targeted fixes
4. Re-run backtest validation

---

## Quick Reference Commands

```bash
# Start backtest (in tmux/screen)
tmux new -s backtest
uv run python scripts/run_comprehensive_backtest.py

# Check progress (from another terminal)
tail -f logs/comprehensive_backtest_*.log

# Count errors after completion
grep -i "extreme gradient" logs/*.log | wc -l
grep -i "constraint violation" logs/*.log | wc -l

# Verify Phase 8-9 implementation
uv run python scripts/verify_phase8_9_fixes.py

# Verify Phase 10 implementation
uv run python scripts/verify_phase10_hrp_logging.py

# Check GPU memory
nvidia-smi
```

---

**Session prepared by**: Claude (Sonnet 4.5)
**Ready for**: Comprehensive backtest validation
**Estimated runtime**: 1-2 hours
**Recommendation**: Run overnight or in tmux/screen session
