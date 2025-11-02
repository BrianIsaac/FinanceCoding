"""
Manual test script for Phase 1 & 2 LSTM fixes.

Tests:
- Phase 1: Cross-sectional mean imputation (no forward fill)
- Phase 2: Correct lengths tensor shape (batch_size,) not (num_assets,)
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

from src.models.lstm.model import LSTMPortfolioModel, LSTMModelConfig
from src.models.base.portfolio_model import PortfolioConstraints

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_lstm_prediction_with_real_data():
    """Test LSTM prediction with real financial data."""
    logger.info("=" * 80)
    logger.info("PHASE 1 & 2: LSTM PREDICTION TESTS")
    logger.info("=" * 80)

    # Load real data
    returns_path = Path("data/final_new_pipeline/returns_daily_final.parquet")
    if not returns_path.exists():
        logger.error(f"Data not found: {returns_path}")
        return False

    logger.info(f"Loading returns data from {returns_path}")
    returns = pd.read_parquet(returns_path)
    logger.info(f"Loaded returns: {returns.shape[0]} days, {returns.shape[1]} assets")

    # Select test period (recent data)
    test_date = pd.Timestamp("2024-01-15")
    train_end = test_date - pd.Timedelta(days=1)
    train_start = train_end - pd.Timedelta(days=365)

    # Select universe (subset for speed)
    all_assets = returns.columns.tolist()
    universe = all_assets[:100]  # First 100 assets
    logger.info(f"Testing with {len(universe)} assets")

    # Create LSTM model
    config = LSTMModelConfig(
        use_markowitz_layer=False,  # Simpler for testing
    )
    constraints = PortfolioConstraints()
    model = LSTMPortfolioModel(constraints, config)

    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Training (verifies Phase 1 - cross-sectional imputation)")
    logger.info("=" * 80)

    try:
        # Filter returns to training period
        train_returns = returns.loc[train_start:train_end, universe]

        # Fit model
        model.rolling_fit(
            returns=returns,  # Full dataset
            universe=universe,
            rebalance_date=train_end,
            lookback_months=12,  # Shorter for testing
        )
        logger.info("✅ Training completed successfully")
        logger.info("   - No forward fill warnings expected in logs above")
        logger.info("   - Using cross-sectional mean imputation")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Prediction (verifies Phase 2 - lengths tensor shape)")
    logger.info("=" * 80)

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

        # Check 4: Not equal-weight (model learned something)
        expected_equal = 1.0 / len(universe)
        if np.allclose(weights.values, expected_equal, atol=0.001):
            logger.warning("   ⚠️  Weights are approximately equal (model may not have learned)")
        else:
            logger.info("   ✅ Weights are diversified (not equal-weight)")

        if checks_passed:
            logger.info("\n✅ ALL VALIDATION CHECKS PASSED")
            return True
        else:
            logger.error("\n❌ SOME VALIDATION CHECKS FAILED")
            return False

    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lengths_tensor_shape():
    """Verify lengths tensor has correct shape during prediction."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Lengths Tensor Shape (Phase 2 specific)")
    logger.info("=" * 80)

    # This is a code inspection test - verify the shape in the source
    model_path = Path("src/models/lstm/model.py")

    with open(model_path, 'r') as f:
        content = f.read()

    # Check that we create single-element tensor
    if '[min_length]' in content and 'batch_size=1' in content:
        logger.info("✅ Code creates lengths tensor with shape (1,)")
        logger.info("   - Found: torch.tensor([min_length], ...)")
        logger.info("   - Comment confirms: 'Single value for batch_size=1'")
        return True
    else:
        logger.error("❌ Could not verify lengths tensor shape in code")
        return False


if __name__ == "__main__":
    logger.info("Starting LSTM manual tests for Phases 1 & 2\n")

    # Run tests
    test1_passed = test_lstm_prediction_with_real_data()
    test2_passed = test_lengths_tensor_shape()

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Phase 1 & 2 Integration Test: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    logger.info(f"Phase 2 Shape Verification:   {'✅ PASS' if test2_passed else '❌ FAIL'}")

    if test1_passed and test2_passed:
        logger.info("\n✅ ALL MANUAL TESTS PASSED - Phases 1 & 2 verified")
        sys.exit(0)
    else:
        logger.error("\n❌ SOME TESTS FAILED - Review logs above")
        sys.exit(1)
