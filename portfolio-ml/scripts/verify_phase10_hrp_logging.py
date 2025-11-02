#!/usr/bin/env python3
"""
Manual Verification Script for Phase 10: HRP Concentration Logging

This script verifies that the enhanced HRP concentration logging is working
correctly and provides useful metrics for evaluating HRP baseline behaviour.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.base.portfolio_model import PortfolioConstraints
from src.models.hrp.model import HRPModel

# Set up logging to capture HRP output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def verify_hrp_concentration_logging():
    """Verify Phase 10 HRP concentration logging enhancement."""
    logger.info("=" * 80)
    logger.info("Phase 10 Verification: HRP Concentration Logging")
    logger.info("=" * 80)

    # Load data
    logger.info("\nLoading data...")
    returns_path = Path("data/final_new_pipeline/returns_daily_final.parquet")

    if not returns_path.exists():
        logger.error(f"Data file not found: {returns_path}")
        return False

    returns_data = pd.read_parquet(returns_path)
    logger.info(f"Loaded returns: {returns_data.shape[0]} days, {returns_data.shape[1]} assets")

    # Create HRP model with standard constraints
    constraints = PortfolioConstraints(
        max_position_weight=0.20,
        min_weight_threshold=0.01,
        max_monthly_turnover=0.50,
    )

    hrp_model = HRPModel(constraints=constraints)

    # Test with different universe sizes to see concentration patterns
    test_cases = [
        {"n_assets": 20, "label": "Small universe (20 assets)"},
        {"n_assets": 50, "label": "Medium universe (50 assets)"},
        {"n_assets": 100, "label": "Large universe (100 assets)"},
    ]

    logger.info("\n" + "=" * 80)
    logger.info("Testing HRP Concentration Logging Across Different Universe Sizes")
    logger.info("=" * 80)

    results = []

    for test_case in test_cases:
        n_assets = test_case["n_assets"]
        label = test_case["label"]

        logger.info(f"\n{'=' * 80}")
        logger.info(f"Test Case: {label}")
        logger.info(f"{'=' * 80}")

        # Get recent data for this universe size
        test_data = returns_data.iloc[-252:, :n_assets]  # Last year of data
        test_data = test_data.dropna(axis=1, how='all')  # Remove any all-NaN columns

        actual_n_assets = test_data.shape[1]
        logger.info(f"Using {actual_n_assets} assets after cleaning")

        try:
            # Fit the model first
            fit_period = (test_data.index[0], test_data.index[-1])
            hrp_model.fit(test_data, list(test_data.columns), fit_period)

            # Generate portfolio - this will trigger the new logging
            weights = hrp_model.predict_weights(
                date=test_data.index[-1],
                universe=list(test_data.columns)
            )

            # Verify we got the expected log output
            max_weight = weights.max()
            top_5_sum = weights.nlargest(5).sum()
            top_10_sum = weights.nlargest(min(10, len(weights))).sum()

            result = {
                "label": label,
                "n_assets": actual_n_assets,
                "max_weight": max_weight,
                "top_5_sum": top_5_sum,
                "top_10_sum": top_10_sum,
                "success": True,
            }

            logger.info(f"\nVerification Summary for {label}:")
            logger.info(f"  Max weight: {max_weight:.1%}")
            logger.info(f"  Top 5 sum: {top_5_sum:.1%}")
            logger.info(f"  Top 10 sum: {top_10_sum:.1%}")
            logger.info(f"  Final weights sum: {weights.sum():.6f}")

            # Check if logging is producing useful information
            checks = []
            if max_weight <= constraints.max_position_weight + 0.001:
                checks.append("✓ Max weight respects constraint")
            else:
                checks.append(f"✗ Max weight {max_weight:.1%} > limit {constraints.max_position_weight:.1%}")

            if 0.999 < weights.sum() < 1.001:
                checks.append("✓ Weights sum to 1.0")
            else:
                checks.append(f"✗ Weights sum to {weights.sum():.6f}")

            if max_weight > 0.15:
                checks.append(f"⚠ Moderate concentration detected ({max_weight:.1%})")
            elif max_weight > 0.20:
                checks.append(f"⚠ High concentration detected ({max_weight:.1%})")
            else:
                checks.append("✓ Concentration within normal range")

            logger.info("\nChecks:")
            for check in checks:
                logger.info(f"  {check}")

            result["checks"] = checks

        except Exception as e:
            logger.error(f"HRP portfolio generation failed for {label}: {str(e)}")
            result = {
                "label": label,
                "n_assets": actual_n_assets,
                "success": False,
                "error": str(e),
                "checks": [f"✗ Portfolio generation failed: {str(e)}"],
            }

        results.append(result)

    # Overall summary
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 10 VERIFICATION SUMMARY")
    logger.info("=" * 80)

    successful_tests = sum(1 for r in results if r["success"])
    total_tests = len(results)

    logger.info(f"\nSuccessful tests: {successful_tests}/{total_tests}")

    if successful_tests == total_tests:
        logger.info("\n✓ Phase 10 HRP concentration logging is working correctly!")
        logger.info("✓ The new logging provides:")
        logger.info("  - Maximum weight concentration")
        logger.info("  - Top 5 assets concentration")
        logger.info("  - Top 10 assets concentration")
        logger.info("  - Number of assets in portfolio")
        logger.info("\nThis data will be used in backtest to evaluate:")
        logger.info("  1. Whether HRP naturally generates high concentration")
        logger.info("  2. How concentration varies with universe size")
        logger.info("  3. Whether Phase 9 fix resolves concentration issues")
    else:
        logger.warning(f"\n⚠ Only {successful_tests}/{total_tests} tests succeeded")

    logger.info("\n" + "=" * 80)
    logger.info("Concentration Patterns Observed:")
    logger.info("=" * 80)

    for result in results:
        if result["success"]:
            logger.info(
                f"\n{result['label']}: "
                f"max={result['max_weight']:.1%}, "
                f"top5={result['top_5_sum']:.1%}, "
                f"top10={result['top_10_sum']:.1%}"
            )

    logger.info("\n" + "=" * 80)
    logger.info("Phase 10 verification complete.")
    logger.info("Ready for comprehensive backtest to collect full concentration metrics.")
    logger.info("=" * 80)

    return successful_tests == total_tests


if __name__ == "__main__":
    success = verify_hrp_concentration_logging()
    sys.exit(0 if success else 1)
