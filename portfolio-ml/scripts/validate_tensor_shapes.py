"""Validate all tensor shapes are correct."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.base.portfolio_model import PortfolioConstraints
from src.models.gat.model import GATPortfolioModel, GATModelConfig
from src.models.lstm.model import LSTMPortfolioModel, LSTMModelConfig

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_lstm_shapes():
    """Validate LSTM tensor shapes during inference."""
    logger.info("=" * 80)
    logger.info("VALIDATING LSTM TENSOR SHAPES")
    logger.info("=" * 80)

    returns_path = Path("data/final_new_pipeline/returns_daily_final.parquet")
    if not returns_path.exists():
        logger.error(f"Returns data not found at {returns_path}")
        return False

    returns = pd.read_parquet(returns_path)
    logger.info(f"Loaded returns: {returns.shape[0]} days, {returns.shape[1]} assets")

    config = LSTMModelConfig(use_markowitz_layer=False)
    model = LSTMPortfolioModel(PortfolioConstraints(), config)

    test_date = pd.Timestamp("2023-06-01")
    universe = returns.columns.tolist()[:100]
    logger.info(f"Testing with {len(universe)} assets on {test_date.strftime('%Y-%m-%d')}")

    # Train
    logger.info("\nTraining model...")
    model.rolling_fit(returns=returns, universe=universe, rebalance_date=test_date)

    # Predict
    logger.info("\nGenerating prediction...")
    try:
        weights_series = model.predict_weights(test_date, universe)
        weights = weights_series.values

        # Check shapes
        assert weights.shape == (len(universe),), f"Wrong weights shape: {weights.shape}"
        assert not pd.isna(weights).any(), "Weights contain NaN"
        assert abs(weights.sum() - 1.0) < 1e-6, f"Weights sum != 1: {weights.sum()}"

        logger.info("✅ LSTM shapes validated")
        logger.info(f"   Weights shape: {weights.shape} (expected: ({len(universe)},))")
        logger.info(f"   Weights sum: {weights.sum():.6f}")
        logger.info(f"   Non-zero weights: {(weights > 1e-6).sum()}")
        logger.info("\nNote: The lengths tensor shape (1,) is validated internally")
        logger.info("      and cannot be easily inspected without code modification.")
        return True

    except Exception as e:
        logger.error(f"❌ LSTM shape validation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_gat_shapes():
    """Validate GAT tensor shapes during inference."""
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATING GAT TENSOR SHAPES")
    logger.info("=" * 80)

    returns_path = Path("data/final_new_pipeline/returns_daily_final.parquet")
    if not returns_path.exists():
        logger.error(f"Returns data not found at {returns_path}")
        return False

    returns = pd.read_parquet(returns_path)
    logger.info(f"Loaded returns: {returns.shape[0]} days, {returns.shape[1]} assets")

    config = GATModelConfig(preset='enhanced')
    model = GATPortfolioModel(PortfolioConstraints(), config)

    test_date = pd.Timestamp("2023-06-01")
    universe = returns.columns.tolist()[:400]
    logger.info(f"Testing with {len(universe)} assets on {test_date.strftime('%Y-%m-%d')}")

    # Train
    logger.info("\nTraining model...")
    model.rolling_fit(returns=returns, universe=universe, rebalance_date=test_date)

    # Predict - should not raise dimension mismatch
    logger.info("\nGenerating prediction...")
    try:
        weights_series = model.predict_weights(test_date, universe)
        weights = weights_series.values

        # Check shapes
        assert weights.shape == (len(universe),), f"Wrong weights shape: {weights.shape}"
        assert not pd.isna(weights).any(), "Weights contain NaN"
        assert abs(weights.sum() - 1.0) < 1e-6, f"Weights sum != 1: {weights.sum()}"

        logger.info("✅ GAT shapes validated (no dimension mismatch errors)")
        logger.info(f"   Weights shape: {weights.shape} (expected: ({len(universe)},))")
        logger.info(f"   Weights sum: {weights.sum():.6f}")
        logger.info(f"   Non-zero weights: {(weights > 1e-6).sum()}")
        logger.info("\nNote: Mask size now correctly matches graph node count")
        logger.info("      (validated by absence of dimension mismatch errors)")
        return True

    except RuntimeError as e:
        if "Size mismatch" in str(e) or "dimension" in str(e).lower():
            logger.error(f"❌ GAT dimension mismatch still present: {e}")
            return False
        else:
            logger.error(f"❌ GAT shape validation FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    except Exception as e:
        logger.error(f"❌ GAT shape validation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logger.info("Starting tensor shape validation tests\n")

    lstm_passed = validate_lstm_shapes()
    gat_passed = validate_gat_shapes()

    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"LSTM tensor shapes: {'✅ PASS' if lstm_passed else '❌ FAIL'}")
    logger.info(f"GAT tensor shapes:  {'✅ PASS' if gat_passed else '❌ FAIL'}")

    if lstm_passed and gat_passed:
        logger.info("\n✅ All shape validations PASSED")
        sys.exit(0)
    else:
        logger.error("\n❌ Some shape validations FAILED")
        sys.exit(1)
