"""Quick functional test for HRP to verify no regressions."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.base.portfolio_model import PortfolioConstraints
from src.models.hrp.model import HRPModel, HRPConfig

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def quick_test_hrp():
    """Run quick HRP test on real data to ensure no regressions."""
    logger.info("=" * 80)
    logger.info("HRP QUICK TEST (Regression Check)")
    logger.info("=" * 80)

    returns_path = Path("data/final_new_pipeline/returns_daily_final.parquet")
    if not returns_path.exists():
        logger.error(f"Returns data not found at {returns_path}")
        return False

    returns = pd.read_parquet(returns_path)
    logger.info(f"Loaded returns: {returns.shape[0]} days, {returns.shape[1]} assets")

    config = HRPConfig()
    model = HRPModel(PortfolioConstraints(), config)

    # Test on 3 months
    test_dates = pd.date_range("2023-01-01", "2023-04-01", freq="MS")
    universe = returns.columns.tolist()[:200]  # Subset for speed
    logger.info(f"Testing on {len(test_dates)} dates with {len(universe)} assets")

    all_weights = []
    for i, date in enumerate(test_dates):
        logger.info(f"\nPrediction {i+1}/{len(test_dates)}: {date.strftime('%Y-%m-%d')}")

        train_end = date - pd.Timedelta(days=1)
        train_start = train_end - pd.Timedelta(days=365)

        logger.info(f"  Training: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
        model.rolling_fit(
            returns=returns,
            universe=universe,
            rebalance_date=date
        )

        logger.info(f"  Predicting for {date.strftime('%Y-%m-%d')}")
        weights_series = model.predict_weights(date, universe)
        weights = weights_series.values
        all_weights.append(weights)

        non_zero = (weights > 1e-6).sum()
        logger.info(f"  Non-zero weights: {non_zero}, Sum: {weights.sum():.6f}")

    # Check all predictions are valid
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION")
    logger.info("=" * 80)

    all_passed = True
    for i, w in enumerate(all_weights):
        date = test_dates[i]
        checks = [
            (not pd.isna(w).any(), f"Date {date}: Contains NaN"),
            (abs(w.sum() - 1.0) < 1e-6, f"Date {date}: Sum != 1 (sum={w.sum():.6f})"),
            ((w >= 0).all(), f"Date {date}: Contains negative weights"),
            ((w > 1e-6).sum() > 0, f"Date {date}: All weights are zero"),
        ]

        for check_passed, error_msg in checks:
            if not check_passed:
                logger.error(f"❌ {error_msg}")
                all_passed = False

    if all_passed:
        logger.info("✅ HRP quick test PASSED (no regressions)")
        logger.info(f"   All {len(test_dates)} predictions are valid")
        logger.info(f"   Mean non-zero weights: {sum((w > 1e-6).sum() for w in all_weights) / len(all_weights):.1f}")
        return True
    else:
        logger.error("❌ HRP quick test FAILED")
        return False


if __name__ == "__main__":
    success = quick_test_hrp()
    sys.exit(0 if success else 1)
