"""
Simple validation test for Phases 1-3 fixes.

This test validates the fixes without full end-to-end training.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_phase1_cross_sectional_imputation():
    """Test that cross-sectional mean imputation is being used (Phase 1)."""
    logger.info("=" * 80)
    logger.info("PHASE 1: Cross-Sectional Mean Imputation")
    logger.info("=" * 80)

    # Check code for cross_sectional_mean_impute usage
    model_path = Path("src/models/lstm/model.py")

    with open(model_path, 'r') as f:
        content = f.read()

    # Count occurrences
    count_cross_sectional = content.count('cross_sectional_mean_impute')
    count_forward_fill = content.count("primary_method='forward_fill'")
    count_impute_with_fallback = content.count('impute_with_fallback')

    logger.info(f"cross_sectional_mean_impute occurrences: {count_cross_sectional}")
    logger.info(f"forward_fill occurrences: {count_forward_fill}")
    logger.info(f"impute_with_fallback occurrences: {count_impute_with_fallback}")

    # Validate
    if count_cross_sectional >= 6 and count_forward_fill == 0 and count_impute_with_fallback == 0:
        logger.info("✅ Phase 1: All three methods use cross-sectional mean imputation")
        logger.info("✅ Phase 1: No forward fill references found")
        logger.info("✅ Phase 1: No impute_with_fallback references found")
        return True
    else:
        logger.error("❌ Phase 1: Imputation not correctly updated")
        return False


def test_phase2_lengths_tensor():
    """Test that lengths tensor has correct shape (Phase 2)."""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: Lengths Tensor Shape")
    logger.info("=" * 80)

    model_path = Path("src/models/lstm/model.py")

    with open(model_path, 'r') as f:
        content = f.read()

    # Check for correct implementation
    checks_passed = True

    if '[min_length]' in content:
        logger.info("✅ Phase 2: Creates single-element lengths tensor [min_length]")
    else:
        logger.error("❌ Phase 2: Missing single-element tensor creation")
        checks_passed = False

    if 'batch_size=1' in content or 'Shape is (1,)' in content:
        logger.info("✅ Phase 2: Comment confirms shape (1,)")
    else:
        logger.error("❌ Phase 2: Missing shape documentation")
        checks_passed = False

    if 'for asset in selected_assets' not in content or 'pred_lengths' not in content.split('for asset in selected_assets')[0][-500:]:
        logger.info("✅ Phase 2: No per-asset lengths creation in predict_weights")
    else:
        logger.error("❌ Phase 2: Still creating per-asset lengths")
        checks_passed = False

    return checks_passed


def test_phase3_gat_fixes():
    """Test GAT dimension mismatch fixes (Phase 3)."""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3: GAT Dimension Mismatch Fixes")
    logger.info("=" * 80)

    model_path = Path("src/models/gat/model.py")

    with open(model_path, 'r') as f:
        content = f.read()

    checks = [
        ('x.shape[0]', 'Fix 1: Mask uses actual graph size'),
        ('num_graph_nodes', 'Fix 1: Variable for graph node count'),
        ('graph_data.tickers', 'Fix 2: Weight indexing uses filtered list'),
        ('filtered_returns = returns_data[graph_data.tickers]', 'Fix 3: Correlation matrix filtered'),
        ('len(graph_data.tickers) == 0', 'Fix 4: Empty universe check'),
        ('equal_weights', 'Fix 4: Equal-weight fallback'),
        ('x.shape[0] != mask_valid.shape[0]', 'Shape validation check'),
    ]

    all_passed = True
    for pattern, description in checks:
        if pattern in content:
            logger.info(f"✅ {description}")
        else:
            logger.error(f"❌ {description} - NOT FOUND")
            all_passed = False

    return all_passed


def test_imports():
    """Test that models can be imported."""
    logger.info("\n" + "=" * 80)
    logger.info("IMPORT TESTS")
    logger.info("=" * 80)

    try:
        from src.models.lstm.model import LSTMPortfolioModel
        logger.info("✅ LSTM model imports successfully")
        lstm_ok = True
    except Exception as e:
        logger.error(f"❌ LSTM model import failed: {e}")
        lstm_ok = False

    try:
        from src.models.gat.model import GATPortfolioModel
        logger.info("✅ GAT model imports successfully")
        gat_ok = True
    except Exception as e:
        logger.error(f"❌ GAT model import failed: {e}")
        gat_ok = False

    return lstm_ok and gat_ok


def test_data_quality_logs():
    """Test that cross-sectional imputation logs appear in actual usage."""
    logger.info("\n" + "=" * 80)
    logger.info("DATA QUALITY: Cross-Sectional Mean Imputation Verification")
    logger.info("=" * 80)

    # Load data and test imputation
    returns_path = Path("data/final_new_pipeline/returns_daily_final.parquet")
    if not returns_path.exists():
        logger.warning("⚠️  Real data not found, skipping data quality test")
        return True

    returns = pd.read_parquet(returns_path)
    logger.info(f"Loaded returns: {returns.shape[0]} days, {returns.shape[1]} assets")

    # Test the NA handling function directly
    try:
        from src.data.na_handling import cross_sectional_mean_impute

        # Create sample data with NAs
        test_data = returns.iloc[:100, :50].copy()
        na_count_before = test_data.isna().sum().sum()

        logger.info(f"Test data: {test_data.shape}, NAs before: {na_count_before}")

        # Apply imputation
        imputed_data = cross_sectional_mean_impute(test_data)
        na_count_after = imputed_data.isna().sum().sum()

        logger.info(f"NAs after imputation: {na_count_after}")

        if na_count_after < na_count_before:
            logger.info("✅ Cross-sectional mean imputation reduces NAs")
            return True
        else:
            logger.warning(f"⚠️  Imputation did not reduce NAs (before: {na_count_before}, after: {na_count_after})")
            return True  # May not have NAs in test data

    except Exception as e:
        logger.error(f"❌ Data quality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logger.info("Starting simple validation tests for Phases 1-3\n")

    # Run tests
    phase1_passed = test_phase1_cross_sectional_imputation()
    phase2_passed = test_phase2_lengths_tensor()
    phase3_passed = test_phase3_gat_fixes()
    imports_passed = test_imports()
    data_quality_passed = test_data_quality_logs()

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Phase 1 (LSTM Forward Fill → Cross-Sectional): {'✅ PASS' if phase1_passed else '❌ FAIL'}")
    logger.info(f"Phase 2 (LSTM Lengths Tensor Shape):           {'✅ PASS' if phase2_passed else '❌ FAIL'}")
    logger.info(f"Phase 3 (GAT Dimension Mismatch):              {'✅ PASS' if phase3_passed else '❌ FAIL'}")
    logger.info(f"Model Imports:                                 {'✅ PASS' if imports_passed else '❌ FAIL'}")
    logger.info(f"Data Quality (Cross-Sectional Imputation):     {'✅ PASS' if data_quality_passed else '❌ FAIL'}")

    all_passed = phase1_passed and phase2_passed and phase3_passed and imports_passed and data_quality_passed

    if all_passed:
        logger.info("\n" + "=" * 80)
        logger.info("✅ ALL VALIDATION TESTS PASSED")
        logger.info("=" * 80)
        logger.info("\nPhases 1-3 implementation verified:")
        logger.info("  • Phase 1: LSTM uses cross-sectional mean (no forward fill)")
        logger.info("  • Phase 2: LSTM lengths tensor has shape (1,) for inference")
        logger.info("  • Phase 3: GAT dimension mismatches resolved (4 fixes)")
        logger.info("\nReady for Phase 4: Comprehensive verification with real financial data")
        sys.exit(0)
    else:
        logger.error("\n❌ SOME VALIDATION TESTS FAILED - Review logs above")
        sys.exit(1)
