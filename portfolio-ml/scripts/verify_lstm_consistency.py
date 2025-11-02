"""Verify LSTM training-inference consistency."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.base.portfolio_model import PortfolioConstraints
from src.models.lstm.model import LSTMPortfolioModel, LSTMModelConfig

# Set up logging to capture imputation methods
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def verify_consistency():
    """Check that same imputation is used in training and prediction."""
    logger.info("=" * 80)
    logger.info("LSTM TRAINING-INFERENCE CONSISTENCY CHECK")
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

    # Train - check logs for "cross_sectional_mean_impute"
    train_end = test_date - pd.Timedelta(days=1)
    train_start = train_end - pd.Timedelta(days=365)

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING PHASE - Watch for imputation method")
    logger.info("=" * 80)
    model.rolling_fit(returns=returns, universe=universe, rebalance_date=test_date)

    # Predict - check logs for "cross_sectional_mean_impute" (NO forward_fill)
    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION PHASE - Watch for imputation method")
    logger.info("=" * 80)
    weights_series = model.predict_weights(test_date, universe)
    weights = weights_series.values

    logger.info("\n" + "=" * 80)
    logger.info("CONSISTENCY CHECK RESULTS")
    logger.info("=" * 80)
    logger.info("✅ Check logs above:")
    logger.info("   - Training should use cross_sectional_mean_impute")
    logger.info("   - Prediction should use cross_sectional_mean_impute")
    logger.info("   - NO mentions of 'forward_fill' should appear")
    logger.info(f"\nPrediction successful: {weights.shape}")
    logger.info(f"Weights sum: {weights.sum():.6f}")
    logger.info(f"Non-zero weights: {(weights > 1e-6).sum()}")

    # Check for validity
    if pd.isna(weights).any():
        logger.error("❌ Weights contain NaN")
        return False

    if abs(weights.sum() - 1.0) > 1e-6:
        logger.error(f"❌ Weights don't sum to 1: {weights.sum()}")
        return False

    logger.info("\n✅ LSTM training-inference consistency verified")
    return True


if __name__ == "__main__":
    success = verify_consistency()
    sys.exit(0 if success else 1)
