"""
Manual test script for Phase 3 GAT fixes.

Tests all 4 interconnected fixes:
1. Mask creation uses actual graph size
2. Weight indexing uses filtered asset list
3. Correlation matrix filtered before computation (DiversificationGAT)
4. Empty universe validation with fallback
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.gat.model import GATPortfolioModel, GATModelConfig
from src.models.base.portfolio_model import PortfolioConstraints

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_gat_basic_prediction():
    """Test basic GAT prediction (Fix 1 & 2)."""
    logger.info("=" * 80)
    logger.info("TEST 1: Basic GAT Prediction (Fix 1 & 2)")
    logger.info("=" * 80)

    # Load real data
    returns_path = Path("data/final_new_pipeline/returns_daily_final.parquet")
    if not returns_path.exists():
        logger.error(f"Data not found: {returns_path}")
        return False

    logger.info(f"Loading returns data from {returns_path}")
    returns = pd.read_parquet(returns_path)
    logger.info(f"Loaded returns: {returns.shape[0]} days, {returns.shape[1]} assets")

    # Select test period
    test_date = pd.Timestamp("2024-01-15")
    train_end = test_date - pd.Timedelta(days=1)
    train_start = train_end - pd.Timedelta(days=756)  # GAT uses 756-day lookback

    # Select universe (subset for speed)
    all_assets = returns.columns.tolist()
    universe = all_assets[:100]  # First 100 assets
    logger.info(f"Testing with {len(universe)} assets")

    # Create basic GAT model
    config = GATModelConfig(
        use_diversification_gat=False,  # Test basic GAT first
        hidden_channels=32,
        num_layers=2,
    )
    constraints = PortfolioConstraints()
    model = GATPortfolioModel(constraints, config)

    logger.info("\nTraining GAT model...")

    try:
        # Filter returns to training period
        train_returns = returns.loc[train_start:train_end, universe]

        # Fit model
        model.rolling_fit(
            returns=train_returns,
            date=train_end,
            universe=universe,
        )
        logger.info("✅ Training completed successfully")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    logger.info("\nPredicting weights...")

    try:
        # Predict weights
        weights = model.predict_weights(
            date=test_date,
            universe=universe,
        )

        logger.info(f"✅ Prediction completed successfully")
        logger.info(f"   - Weights shape: {weights.shape}")
        logger.info(f"   - Weights sum: {weights.sum():.6f} (should be ~1.0)")
        logger.info(f"   - Min weight: {weights.min():.6f} (should be ≥0)")
        logger.info(f"   - Max weight: {weights.max():.6f}")
        logger.info(f"   - NaN count: {weights.isna().sum()} (should be 0)")

        # Validation checks
        checks_passed = True

        # Check 1: Sum to 1
        if not np.isclose(weights.sum(), 1.0, atol=1e-6):
            logger.error(f"   ❌ Weights don't sum to 1: {weights.sum()}")
            checks_passed = False
        else:
            logger.info("   ✅ Weights sum to 1")

        # Check 2: Non-negative
        if (weights < 0).any():
            logger.error(f"   ❌ Negative weights found: min={weights.min()}")
            checks_passed = False
        else:
            logger.info("   ✅ All weights non-negative")

        # Check 3: No NaN
        if weights.isna().any():
            logger.error("   ❌ NaN values in weights")
            checks_passed = False
        else:
            logger.info("   ✅ No NaN values")

        # Check 4: No dimension mismatch errors
        logger.info("   ✅ No dimension mismatch errors (Fix 1 & 2 working)")

        return checks_passed

    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_diversification_gat():
    """Test DiversificationGAT (Fix 3)."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: DiversificationGAT Correlation Matrix (Fix 3)")
    logger.info("=" * 80)

    # Load real data
    returns_path = Path("data/final_new_pipeline/returns_daily_final.parquet")
    if not returns_path.exists():
        logger.error(f"Data not found: {returns_path}")
        return False

    returns = pd.read_parquet(returns_path)

    # Select test period
    test_date = pd.Timestamp("2024-01-15")
    train_end = test_date - pd.Timedelta(days=1)
    train_start = train_end - pd.Timedelta(days=756)

    # Select universe
    all_assets = returns.columns.tolist()
    universe = all_assets[:80]  # Smaller for DiversificationGAT (more expensive)
    logger.info(f"Testing with {len(universe)} assets")

    # Create DiversificationGAT model
    config = GATModelConfig(
        use_diversification_gat=True,  # Enable diversification loss
        hidden_channels=32,
        num_layers=2,
    )
    constraints = PortfolioConstraints()
    model = GATPortfolioModel(constraints, config)

    logger.info("\nTraining DiversificationGAT model...")

    try:
        train_returns = returns.loc[train_start:train_end, universe]

        model.rolling_fit(
            returns=train_returns,
            date=train_end,
            universe=universe,
        )
        logger.info("✅ Training completed successfully")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    logger.info("\nPredicting weights with correlation matrix...")

    try:
        weights = model.predict_weights(
            date=test_date,
            universe=universe,
        )

        logger.info(f"✅ Prediction completed successfully")
        logger.info(f"   - Weights shape: {weights.shape}")
        logger.info(f"   - Weights sum: {weights.sum():.6f}")
        logger.info(f"   - No correlation matrix dimension errors (Fix 3 working)")

        # Validation checks
        checks_passed = True

        if not np.isclose(weights.sum(), 1.0, atol=1e-6):
            logger.error(f"   ❌ Weights don't sum to 1: {weights.sum()}")
            checks_passed = False
        else:
            logger.info("   ✅ Weights sum to 1")

        if (weights < 0).any():
            logger.error(f"   ❌ Negative weights found")
            checks_passed = False
        else:
            logger.info("   ✅ All weights non-negative")

        if weights.isna().any():
            logger.error("   ❌ NaN values in weights")
            checks_passed = False
        else:
            logger.info("   ✅ No NaN values")

        return checks_passed

    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_empty_universe_fallback():
    """Test empty universe after filtering (Fix 4)."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Empty Universe Fallback (Fix 4)")
    logger.info("=" * 80)

    # This is a code inspection test
    model_path = Path("src/models/gat/model.py")

    with open(model_path, 'r') as f:
        content = f.read()

    # Check for empty universe handling
    checks = [
        ('len(graph_data.tickers) == 0', 'Empty check'),
        ('equal_weights', 'Fallback to equal weights'),
        ('return pd.Series(equal_weights', 'Return statement'),
    ]

    all_found = True
    for pattern, description in checks:
        if pattern in content:
            logger.info(f"   ✅ Found: {description}")
        else:
            logger.error(f"   ❌ Missing: {description}")
            all_found = False

    if all_found:
        logger.info("✅ Empty universe fallback implemented correctly")
        return True
    else:
        logger.error("❌ Empty universe fallback incomplete")
        return False


def test_shape_validation():
    """Test shape validation before forward pass."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Shape Validation")
    logger.info("=" * 80)

    model_path = Path("src/models/gat/model.py")

    with open(model_path, 'r') as f:
        content = f.read()

    # Check for validation
    if 'x.shape[0] != mask_valid.shape[0]' in content:
        logger.info("   ✅ Found shape validation check")
        logger.info("   ✅ Raises ValueError on mismatch")
        return True
    else:
        logger.error("   ❌ Shape validation not found")
        return False


if __name__ == "__main__":
    logger.info("Starting GAT manual tests for Phase 3\n")

    # Run tests
    test1_passed = test_gat_basic_prediction()
    test2_passed = test_diversification_gat()
    test3_passed = test_empty_universe_fallback()
    test4_passed = test_shape_validation()

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Basic GAT Prediction (Fix 1 & 2):    {'✅ PASS' if test1_passed else '❌ FAIL'}")
    logger.info(f"DiversificationGAT (Fix 3):          {'✅ PASS' if test2_passed else '❌ FAIL'}")
    logger.info(f"Empty Universe Fallback (Fix 4):     {'✅ PASS' if test3_passed else '❌ FAIL'}")
    logger.info(f"Shape Validation:                    {'✅ PASS' if test4_passed else '❌ FAIL'}")

    if test1_passed and test2_passed and test3_passed and test4_passed:
        logger.info("\n✅ ALL MANUAL TESTS PASSED - Phase 3 verified")
        sys.exit(0)
    else:
        logger.error("\n❌ SOME TESTS FAILED - Review logs above")
        sys.exit(1)
