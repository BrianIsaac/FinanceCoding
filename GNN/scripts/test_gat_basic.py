#!/usr/bin/env python3
"""
Basic GAT model test to validate functionality.
"""

import sys

sys.path.append('/home/brian-isaac/Documents/personal/FinanceCoding/GNN')

import logging

from src.models.base.constraints import PortfolioConstraints
from src.models.gat.model import GATModelConfig, GATPortfolioModel

# Setup simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gat_basic():
    """Basic test of GAT model instantiation."""
    try:
        logger.info("Testing GAT model instantiation...")

        # Create simple configuration
        config = GATModelConfig(
            input_features=10,
            hidden_dim=16,  # Very small for testing
            num_layers=2,
            num_attention_heads=2,
            dropout=0.1,
        )

        # Create simple constraints
        constraints = PortfolioConstraints(
            long_only=True,
            top_k_positions=10,
            max_position_weight=0.2
        )

        logger.info("Initializing GAT model...")
        model = GATPortfolioModel(constraints=constraints, config=config)

        logger.info("Testing get_model_info...")
        model_info = model.get_model_info()
        logger.info(f"Model info keys: {list(model_info.keys())}")

        logger.info("✅ GAT model basic instantiation successful!")
        return True

    except Exception as e:
        logger.error(f"❌ GAT basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gat_basic()
    sys.exit(0 if success else 1)
