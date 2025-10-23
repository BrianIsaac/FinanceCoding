# Portfolio-ML Scripts Cleanup Plan

## Executive Summary

**Problem:** Script duplication and unclear structure after incomplete Sep 14-21 refactoring.

**Solution:** Remove 16+ redundant scripts, establish 3-script production pipeline, archive completed validation scripts.

**Actions:**
1. **DELETE** - 9 training scripts (redundant with run_comprehensive_backtest.py)
2. **DELETE** - 3 duplicate/superseded data collection scripts
3. **DELETE** - 7 Story 5.6 validation scripts (completed Oct 8, 2025, results preserved)
4. **DELETE** - 7 completed/redundant utility scripts (130K of validation/unused code)
5. **DELETE** - docs/ folder (150 files, 2MB - overly comprehensive, not being used)
6. **REORGANISE** - Move experiments to experiments/ directory
7. **DOCUMENT** - Create scripts/README.md and update main README

**Result:** Clean 3-script production pipeline ONLY (125K total).

**Total Deletion:** ~195+ files (~2.3MB) including:
- 26 redundant/completed scripts (~280K)
- 150 documentation files (2MB)

---

## Issue Identified
The repository has script duplication and unclear structure after the refactoring commits (Sep 14-21, 2025). The cleanup in commit f41eab6 was incomplete.

## Current State (Verified: 2025-10-23)

### ✅ Production Pipeline (3 scripts):
1. **Data Collection**: `scripts/data_collection_pipeline.py` (380 lines, comprehensive logging)
2. **Training & Backtest**: `scripts/run_comprehensive_backtest.py` (1057 lines)
3. **Statistical Analysis**: `scripts/run_performance_analytics.py` (1404 lines)

### ❌ Redundant/Unclear Scripts:

#### Duplicate Data Collection:
- `scripts/pipeline_execution/run_complete_new_pipeline.py` - OLDER VERSION (Sep 13), superseded by data_collection_pipeline.py (Sep 21)

#### Why `data_collection_pipeline.py` is the correct one:
**File History Analysis:**
- `data_collection_pipeline.py`: Last modified **Sep 21, 2025** (commit f41eab6 - production cleanup)
- `run_complete_new_pipeline.py`: Last modified **Sep 13, 2025** (commit 17d26f5 - QA framework)

**Technical Comparison:**
- ✅ `data_collection_pipeline.py` (NEWER - USE THIS):
  - Comprehensive logging to file + console
  - Single-ticker sequential approach (more API-respectful)
  - 500ms delay between requests
  - Smart source selection (tests Stooq availability first)
  - Forward-fill ONLY (no temporal leakage)
  - Detailed progress tracking (every 25 tickers)
  - Execution time tracking

- ❌ `run_complete_new_pipeline.py` (OLDER - REMOVE):
  - No logging setup
  - Batch approach with 8 parallel threads (more aggressive)
  - Uses forward + backward fill (TEMPORAL LEAKAGE RISK!)
  - Less detailed error reporting
  - Currently referenced in README (needs update)

#### Individual Training Scripts (9 scripts - DELETE):
These are redundant for production since `run_comprehensive_backtest.py` handles training during rolling backtest:
- `scripts/train_hrp_pipeline.py` (24K)
- `scripts/train_hrp_execution.py` (12K)
- `scripts/train_hrp_aggressive.py` (21K)
- `scripts/train_hrp_pipeline_fixed.py` (19K)
- `scripts/train_lstm_pipeline.py` (35K)
- `scripts/train_lstm_aggressive.py` (32K)
- `scripts/train_gat_pipeline.py` (25K)
- `scripts/train_gat_aggressive.py` (43K)
- `scripts/train_all_models.py` (43K)

**Decision: DELETE** - These scripts were for individual model development/experimentation. The production pipeline (`run_comprehensive_backtest.py`) handles all training during rolling backtests.

#### One-Time Analysis Scripts (COMPLETED - DELETE):
These scripts were run once for specific Story 5.6 validation tasks. Results preserved in `results/`:
- `scripts/run_task_0_risk_mitigation.py` (2.0K) - Validated production readiness risks
- `scripts/run_task_1_pipeline_performance.py` (3.0K) - Validated end-to-end pipeline performance
- `scripts/run_task_2_gpu_memory_validation.py` (3.3K) - Validated GPU memory under production load
- `scripts/run_task_3_constraint_compliance.py` (3.7K) - Validated constraint compliance
- `scripts/run_quick_turnover_fix.py` (11K) - Fixed turnover violations (one-time fix)
- `scripts/run_rebalancing_frequency_solution.py` (12K) - Optimised rebalancing frequency (one-time analysis)
- `scripts/run_enhanced_constraint_validation.py` (3.9K) - Enhanced constraint validation (one-time run)

**Evidence of completion (results preserved):**
```
results/task_0_risk_mitigation_results.json (Oct 8)
results/task_1_pipeline_performance_results.json (Oct 8)
results/task_2_gpu_memory_validation_results.json (Oct 8)
results/task_3_constraint_compliance_results.json (Oct 8)
results/quick_turnover_fix_results.json (Oct 8)
results/rebalancing_frequency_optimization.json (Oct 8)
```

**Decision: DELETE** - These were one-time validation scripts for Story 5.6. Results are preserved in `results/` directory, scripts no longer needed.

#### Experimental Runner Script:
- `scripts/run_experiments.py` (6.7K) - Rolling-window GAT experiment runner for multiple seeds

**Purpose:** Runs rolling experiments with different seeds for GAT model validation. Archives outputs to `outputs/experiments/`.

**Decision: KEEP IN EXPERIMENTS/** - This is a legitimate experimental framework for testing model variations.

#### Documentation Folder (DELETE):
- `docs/` (150 files, 2MB) - Comprehensive documentation generated by previous tooling

**Contents:**
- architecture/ (system design docs)
- prd/ (product requirement docs)
- qa/ (quality assurance assessments)
- stories/ (user stories)
- deployment/ (deployment guides)
- research/ (research documentation)
- api/, development/, models/, templates/, tutorials/

**Decision: DELETE** - Overly comprehensive documentation that is not being actively used or maintained. Essential information is in README.md and inline code documentation.

#### Utility Scripts Analysis (DETAILED REVIEW):

**1. `fix_universe_membership_data.py` (3.5K) - COMPLETED, DELETE**
- **Purpose:** One-time conversion from start/end format to daily format
- **Evidence of completion:**
  - Output exists: `data/processed/universe_membership_daily.csv`
  - Created: Sep 17, 2025
  - One-time data transformation script
- **Decision: DELETE** - Transformation completed, output preserved

**2. `generate_gat_data.py` (14K) - COMPLETED, DELETE**
- **Purpose:** Generate graph snapshots and labels for GAT training
- **Evidence of completion:**
  - Output directories exist: `data/graphs/` and `data/labels/`
  - Used for initial GAT data preparation
  - Not used in production pipeline (run_comprehensive_backtest.py generates data on-the-fly)
- **Decision: DELETE** - Initial generation completed, production pipeline handles this

**3. `download_data.py` (4.1K) - SUPERSEDED, DELETE**
- **Purpose:** Download S&P MidCap 400 data from multiple sources
- **Status:** Superseded by `data_collection_pipeline.py` (16K)
- **Issues:**
  - Older, simpler implementation
  - No logging infrastructure
  - Less comprehensive than data_collection_pipeline.py
  - References old config structure
- **Decision: DELETE** - Superseded by production data_collection_pipeline.py

**4. `create_sample_data.py` (5.6K) - UNUSED, DELETE**
- **Purpose:** Generate synthetic financial data for testing without real data
- **Reality:** Not used in practice, real data is always available
- **Decision: DELETE** - Unlikely to be used

**5. `model_checkpoint_generator.py` (21K) - REDUNDANT, DELETE**
- **Purpose:** Model state serialization system for HRP, LSTM, GAT
- **Reality:** Checkpoint logic is built into `RollingBacktestEngine` directly
- **Evidence:**
  - `src/evaluation/backtest/rolling_engine.py` has `_save_model_checkpoint()` method
  - Not imported by any production scripts
  - Standalone validation framework, not used in production
- **Decision: DELETE** - Functionality already in production code

**6. `pipeline_integrity_validator.py` (30K) - ONE-TIME VALIDATION, DELETE**
- **Purpose:** Story 5.2 Task 6 - End-to-end pipeline integrity validation
- **Evidence of completion:**
  - Output directory exists: `logs/training/pipeline_validation/`
  - Comprehensive validation framework (30K of code)
  - One-time validation for Story 5.2
- **Decision: DELETE** - One-time validation completed, overly complex for ongoing use

**7. `training_convergence_validator.py` (28K) - ONE-TIME VALIDATION, DELETE**
- **Purpose:** Story 5.2 Task 5 - Training convergence and hyperparameter validation
- **Evidence of completion:**
  - Output directory exists: `logs/training/convergence/`
  - Comprehensive validation framework (28K of code)
  - One-time validation for Story 5.2
- **Decision: DELETE** - One-time validation completed, overly complex for ongoing use

**Summary:**
- **DELETE ALL (7 scripts, 130K):**
  - fix_universe (3.5K)
  - generate_gat (14K)
  - download_data (4.1K)
  - create_sample_data (5.6K - unused)
  - model_checkpoint_generator (21K - redundant, functionality in RollingBacktestEngine)
  - pipeline_integrity (30K)
  - training_convergence (28K)
  - pipeline_execution (9.4K - duplicate already marked)
- **KEEP:** None - all utility scripts are redundant or completed

## Recommended Actions

### Action 1: Remove Duplicate Data Collection Script and Update README
```bash
# Remove the older duplicate script
rm scripts/pipeline_execution/run_complete_new_pipeline.py

# Remove the empty directory if nothing else is there
rmdir scripts/pipeline_execution 2>/dev/null || true
```

**Also update README.md** to reference the correct script:
- Change: `python scripts/pipeline_execution/run_complete_new_pipeline.py`
- To: `python scripts/data_collection_pipeline.py`

### Action 2: Delete Redundant Training Scripts and Utility Scripts
The production pipeline handles all training. Delete redundant scripts:

```bash
# Delete individual model training scripts (9 files)
rm scripts/train_hrp_pipeline.py
rm scripts/train_hrp_execution.py
rm scripts/train_hrp_aggressive.py
rm scripts/train_hrp_pipeline_fixed.py
rm scripts/train_lstm_pipeline.py
rm scripts/train_lstm_aggressive.py
rm scripts/train_gat_pipeline.py
rm scripts/train_gat_aggressive.py
rm scripts/train_all_models.py

# Delete superseded/completed/redundant utility scripts (7 files)
rm scripts/fix_universe_membership_data.py
rm scripts/generate_gat_data.py
rm scripts/download_data.py
rm scripts/create_sample_data.py
rm scripts/model_checkpoint_generator.py
rm scripts/pipeline_integrity_validator.py
rm scripts/training_convergence_validator.py

# Remove examples directory if empty
rm -rf scripts/examples/ 2>/dev/null || true
```

### Action 3: Delete Completed One-Time Validation Scripts
These scripts completed their purpose for Story 5.6. Results are preserved in `results/`:

```bash
# Delete completed validation scripts (results preserved in results/)
rm scripts/run_task_0_risk_mitigation.py
rm scripts/run_task_1_pipeline_performance.py
rm scripts/run_task_2_gpu_memory_validation.py
rm scripts/run_task_3_constraint_compliance.py
rm scripts/run_quick_turnover_fix.py
rm scripts/run_rebalancing_frequency_solution.py
rm scripts/run_enhanced_constraint_validation.py
```

### Action 4: Delete Comprehensive Documentation Folder
Remove overly comprehensive docs/ folder (150 files, 2MB):

```bash
# Delete comprehensive documentation folder
rm -rf docs/
```

### Action 5: Move Experimental Runner to Experiments Directory
```bash
# Create experiments directory
mkdir -p experiments

# Move experimental runner
mv scripts/run_experiments.py experiments/
```

### Action 6: Update README to Document Production Pipeline

Add a clear section to README.md:

```markdown
## Production Pipeline

The production pipeline consists of three main scripts:

### 1. Data Collection
```bash
python scripts/data_collection_pipeline.py
```
Collects S&P MidCap 400 data from multiple sources (Stooq, Yahoo Finance), performs quality validation, gap filling, and saves to `data/final_new_pipeline/`.

### 2. Model Training & Backtesting
```bash
python scripts/run_comprehensive_backtest.py
```
Executes rolling backtests for all models (HRP, LSTM, GAT) with walk-forward analysis. Models are trained fresh each month during the rolling backtest. Results saved to `results/ml_backtest_rolling/`.

### 3. Statistical Analysis
```bash
python scripts/run_performance_analytics.py
```
Performs comprehensive statistical validation including bootstrap confidence intervals, hypothesis testing, and publication-ready tables. Results saved to `results/performance_analytics/`.

```

### Action 7: Create scripts/README.md

Create a clear inventory documenting the production pipeline:

```bash
cat > scripts/README.md << 'EOF'
# Scripts Directory

This directory contains production pipeline scripts and utilities for the portfolio-ml project.

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
EOF
```

## Post-Cleanup Verification

After cleanup, verify the structure:

```bash
# Run verification again
python3 verify_pipeline.py

# Check that production pipeline still works
# (dry run or on test data)
```

## Expected Final Structure

```
scripts/
├── README.md                                    # Script inventory and usage guide
├── data_collection_pipeline.py                 # PRODUCTION: Data collection (16K)
├── run_comprehensive_backtest.py               # PRODUCTION: Training + backtest (46K)
└── run_performance_analytics.py                # PRODUCTION: Statistical analysis (63K)

experiments/
├── run_experiments.py                          # Experimental runner for model variations (6.7K)
└── (future experimental scripts)

legacy_scripts/                                 # Historical reference (18 scripts)
└── (original implementation files)
```

## Git Commit Message

```
refactor: consolidate production pipeline and remove redundant code

Major cleanup to establish clear 3-script production pipeline:

**Deleted:**
- 3 duplicate/superseded data collection scripts:
  - run_complete_new_pipeline.py (9.4K - Sep 13, older version)
  - download_data.py (4.1K - superseded by data_collection_pipeline.py)
  - scripts/pipeline_execution/ directory
- 9 individual training scripts (train_*.py) - redundant with run_comprehensive_backtest.py
- 7 one-time Story 5.6 validation scripts (results preserved in results/)
  - Task 0-3 validation scripts (completed Oct 8, 2025)
  - Turnover and rebalancing optimisation scripts
- 7 completed/redundant utility scripts (130K of validation/unused code):
  - fix_universe_membership_data.py (3.5K - transformation completed)
  - generate_gat_data.py (14K - initial generation completed)
  - download_data.py (4.1K - superseded, already listed above)
  - create_sample_data.py (5.6K - unused in practice)
  - model_checkpoint_generator.py (21K - redundant, functionality in RollingBacktestEngine)
  - pipeline_integrity_validator.py (30K - Story 5.2 validation completed)
  - training_convergence_validator.py (28K - Story 5.2 validation completed)
- Comprehensive docs/ folder (150 files, 2MB) - overly comprehensive, not actively used
- scripts/examples/ directory

**Reorganised:**
- Moved run_experiments.py to experiments/ directory
- Created scripts/README.md documenting production pipeline
- Updated main README to reference correct data collection script

**Production Pipeline (3 scripts):**
1. scripts/data_collection_pipeline.py (Sep 21 - newer, production-ready)
   - Single-ticker sequential approach (API-respectful)
   - Forward-fill only (prevents temporal leakage)
   - Comprehensive logging and smart source selection

2. scripts/run_comprehensive_backtest.py
   - Handles all model training during rolling backtest
   - Makes individual training scripts redundant

3. scripts/run_performance_analytics.py
   - Statistical validation and publication-ready analysis

**Rationale:**
- data_collection_pipeline.py is 8 days newer than run_complete_new_pipeline.py
- Created during Sep 21 production cleanup commit (f41eab6)
- Eliminates temporal leakage from backward fill
- Individual training scripts redundant since backtest handles training
- Story 5.6 validation scripts completed their one-time purpose (results preserved)
- Documentation folder was overly comprehensive and not maintained

This cleanup completes the incomplete refactoring from commits f41eab6 and 42dc76b,
establishing clear separation between production pipeline, experiments, and legacy code.
```

## Notes

- Keep `legacy_scripts/` for historical reference only
- Results from deleted validation scripts are preserved in `results/` directory
- The production pipeline is now clearly defined: 3 scripts only
- Essential documentation remains in README.md and inline code comments
- This cleanup removes ~170+ files (scripts + docs) without losing functionality
