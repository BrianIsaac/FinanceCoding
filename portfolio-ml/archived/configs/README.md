# Archived Configuration Files

**Archived Date**: 2025-10-26
**Reason**: Replaced by Hydra-based configuration structure

## Migration Summary

The entire configuration system was migrated from manual YAML loading to Hydra-based configuration management.

### What Was Archived

**Old Structure** (archived to `archived/configs/old_structure/`):
- `model_config.yaml` - Used only 1 field (test_end_date), all other values hardcoded
- `academic_standards.yaml` - Never loaded by any script
- `data/` - 2 files (default.yaml, midcap400.yaml) - Never loaded, values hardcoded
- `evaluation/` - 7 files (performance_config, benchmarks, chart_config, dashboard_config, risk_config, visualization_config, statistical_validation) - Never loaded
- `experiments/` - 3 files (full_evaluation, baseline_comparison, training_config) - Never used
- `models/` - 3 files (gat_default.yaml, hrp_default.yaml, lstm_default.yaml) - Never used

**Total: 17 YAML files** (validation research confirmed all unused)

### What Replaced It

**New Hydra Structure** (in `configs/`):
- `data_collection/` - Hydra configs for data collection pipeline
- `backtest/` - Hydra configs for comprehensive backtest
- `analytics/` - Hydra configs for performance analytics

Each script now uses:
- `@hydra.main` decorator for automatic config loading
- Structured config dataclasses for type safety
- Config composition via defaults lists
- CLI overrides (e.g., `python script.py param=value`)

### Key Improvements

1. **Type Safety**: Structured configs provide runtime validation
2. **CLI Overrides**: Change any config parameter from command line
3. **Composition**: Mix and match config groups (e.g., different universes, models)
4. **No Dead Code**: All config values actually used (unlike old system)
5. **Auto-documentation**: Configs saved with each run in `.hydra/` directory

### Restoration

If you need to reference old configs:
```bash
cat archived/configs/old_structure/<file.yaml>
```

DO NOT restore - migrate to new Hydra structure instead.

### Migration Guide

See: `thoughts/shared/plans/2025-10-26-codebase-cleanup-implementation.md` Phase 3
