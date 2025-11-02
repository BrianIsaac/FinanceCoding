# Phase 8 Quick Start Guide

**Last Updated**: 2025-10-29

## Current Status

Phases 1-7 are **COMPLETE** and fully verified. Phase 8 is **80% complete** - only one task remains.

## What Needs To Be Done

**Single Task**: Update paper preset to enable time-series node features

**File**: `src/models/gat/model.py` (lines 206-212)

**Estimated Time**: 5 minutes

## The Change

### Current Code (lines 206-212)

```python
# Note: Paper uses time-series features, but we use static as approximation
self.node_feature_type = "static"
logger.warning(
    "Paper preset: Using static node features instead of time-series volatility vectors. "
    "This is a known deviation from the paper specification. "
    "Time-series node features require architectural changes and are not yet implemented."
)
```

### Replace With

```python
# Enable time-series features (paper-accurate)
self.node_feature_type = "timeseries"
self.timeseries_length = 756  # Match paper: 3 years
self.timeseries_features = ["volatility"]  # Paper uses volatility series
self.use_temporal_encoder = True
self.temporal_encoder_type = "lstm"  # Paper uses LSTM
self.temporal_encoder_hidden = 64
self.temporal_encoder_layers = 1

logger.info(
    "Paper preset: Using time-series volatility vectors (756 days) as node features"
)
logger.warning(
    "This increases memory usage ~75x compared to static features. "
    "Reduce timeseries_length to 60-120 if memory constrained."
)
```

## Verification Command

After making the change, verify it works:

```bash
uv run python -c "from src.models.gat.model import GATModelConfig; c = GATModelConfig(preset='paper_reproduction'); print(f'node_feature_type={c.node_feature_type}')"
```

**Expected Output**: `node_feature_type=timeseries`

## Why This is Safe

1. **All infrastructure is ready**: Phases 6-7 implemented temporal encoders and data pipeline
2. **Backward compatible**: Enhanced preset still uses static features (unaffected)
3. **Well tested**: Manual verification completed with 756-day windows
4. **Membership-aware NaN handling**: Implemented and verified in Phase 7

## After Phase 8

Proceed to **Phase 9**: Comprehensive testing and evaluation

See the full plan at: `thoughts/shared/plans/2025-10-29-fix-critical-verification-issues.md`

## Critical Technical Notes

### Membership-Aware NaN Handling (DO NOT CHANGE)

The Phase 7 implementation uses the correct approach:

- ✅ Use `np.nanmean()` and `np.nanstd()` for normalization
- ✅ Fill remaining NaN with 0 after z-scoring (0 = neutral/average)
- ❌ DO NOT use forward fill (breaks membership awareness)
- ❌ DO NOT use dropna on time-series (breaks temporal alignment)

This preserves the membership signal: NaN means "asset wasn't active that day"

## Key Files

1. **Plan Document**: `thoughts/shared/plans/2025-10-29-fix-critical-verification-issues.md`
2. **File to Edit**: `src/models/gat/model.py` (lines 206-212)
3. **Temporal Encoders**: `src/models/gat/temporal_encoders.py` (complete)
4. **Data Pipeline**: `src/models/gat/model.py` - `_prepare_timeseries_features()` (complete)
5. **Manual Tests**: `scripts/manual_verification_phase6_7.py` (all 5 tests passed)

## Implementation Summary (Phases 6-7)

### Phase 6: Temporal Encoder Infrastructure ✅
- Created LSTM, Conv1D, Transformer encoders for time-series node features
- Updated GATPortfolio to support temporal encoding
- All gradient flow tests passed

### Phase 7: Data Pipeline ✅
- Created `_prepare_timeseries_features()` for volatility, returns, momentum
- Created `_get_node_features()` routing method
- Implemented membership-aware NaN handling with `np.nanmean()`/`np.nanstd()`
- All manual verification tests passed (5/5)

## Questions?

Refer to the full plan document for detailed context, implementation notes, and test results.
