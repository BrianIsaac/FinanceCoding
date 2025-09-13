#!/usr/bin/env python3
"""
Test GAT model integration and basic functionality.

This script validates that the GAT model can be instantiated, trained,
and used for portfolio prediction with sample data.
"""

import logging
from pathlib import Path

import pandas as pd

from src.models.base.constraints import PortfolioConstraints
from src.models.gat.model import GATModelConfig, GATPortfolioModel


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def test_gat_integration():
    """Test GAT model integration with sample data."""
    logger = setup_logging()
    logger.info("Testing GAT model integration...")

    # Load sample data
    data_dir = Path("data/sample")
    if not data_dir.exists():
        logger.error("Sample data not found. Run scripts/create_sample_data.py first.")
        return False

    try:
        logger.info("Loading sample data...")
        returns = pd.read_parquet(data_dir / "returns_daily.parquet")
        universe = pd.read_parquet(data_dir / "universe_membership.parquet")

        logger.info(f"Data loaded: {returns.shape} returns, {universe.shape} universe")

        # Select subset for testing (first 50 assets, recent 2 years)
        test_assets = returns.columns[:50]
        test_start = returns.index[-500]  # ~2 years
        test_end = returns.index[-1]

        test_returns = returns.loc[test_start:test_end, test_assets].dropna(how='all')
        test_universe = universe.loc[test_start:test_end, test_assets].fillna(False)

        logger.info(f"Test data: {test_returns.shape} returns, {test_universe.sum().sum()} active positions")

        # Create GAT model configuration
        config = GATModelConfig(
            input_features=10,
            hidden_dim=32,  # Smaller for testing
            num_layers=2,   # Fewer layers for testing
            num_attention_heads=4,  # Fewer heads for testing
            dropout=0.2,
        )

        # Create constraints
        constraints = PortfolioConstraints(
            long_only=True,
            top_k_positions=20,  # Limited positions for testing
            max_position_weight=0.15,
            max_monthly_turnover=0.3
        )

        logger.info("Initializing GAT model...")
        gat_model = GATPortfolioModel(constraints=constraints, config=config)

        # Test model fitting (on subset of data)
        fit_start = test_returns.index[0]
        fit_end = test_returns.index[-100]  # Leave some for testing

        active_assets = test_universe.loc[fit_start:fit_end].any().loc[lambda x: x].index.tolist()
        logger.info(f"Fitting model on {len(active_assets)} assets from {fit_start} to {fit_end}")

        gat_model.fit(
            returns=test_returns,
            universe=active_assets,
            fit_period=(fit_start, fit_end)
        )

        logger.info("Model fitting completed successfully!")

        # Test prediction
        pred_date = test_returns.index[-50]
        active_assets_pred = test_universe.loc[pred_date].loc[lambda x: x].index.tolist()

        logger.info(f"Testing prediction for {len(active_assets_pred)} assets on {pred_date}")

        weights = gat_model.predict_weights(
            date=pred_date,
            universe=active_assets_pred
        )

        logger.info("Model prediction completed successfully!")

        # Validate results
        logger.info("Validating results...")
        logger.info(f"  Prediction shape: {weights.shape}")
        logger.info(f"  Weight sum: {weights.sum():.6f}")
        logger.info(f"  Non-zero weights: {(weights > 0).sum()}/{len(weights)}")
        logger.info(f"  Max weight: {weights.max():.4f}")
        logger.info(f"  Min weight: {weights.min():.4f}")

        # Check constraints
        assert abs(weights.sum() - 1.0) < 0.01, f"Weights don't sum to 1: {weights.sum()}"
        assert (weights >= 0).all(), "Found negative weights in long-only portfolio"
        assert weights.max() <= constraints.max_position_weight + 0.01, "Max position weight violated"
        assert (weights > 0).sum() <= constraints.top_k_positions, "Too many positions"

        logger.info("✅ All GAT model tests passed successfully!")
        logger.info("GAT model is functional and ready for production use.")
        return True

    except Exception as e:
        logger.error(f"❌ GAT integration test failed: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_gat_integration()
    exit(0 if success else 1)
