"""Simple GAT training test to identify the issue."""

import sys

sys.path.append('/home/brian-isaac/Documents/personal/FinanceCoding/GNN')

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.gat.gat_model import GATPortfolio
from src.models.gat.graph_builder import GraphBuildConfig, build_period_graph


def main():

    # Load a small subset of data
    data_path = Path("data/final_new_pipeline")
    returns_path = data_path / "returns_daily_final.parquet"
    returns_data = pd.read_parquet(returns_path)
    returns_data.index = pd.to_datetime(returns_data.index)

    # Use a small universe for testing
    tickers = returns_data.columns[:20].tolist()

    # Create simple features
    features_matrix = np.random.randn(len(tickers), 10).astype(np.float32)

    # Test graph configuration
    try:
        graph_config = GraphBuildConfig(
            filter_method='mst',
            knn_k=8,
            lookback_days=252,
            use_edge_attr=True
        )

        # Test graph building
        test_date = pd.Timestamp('2020-01-01')
        historical_returns = returns_data.loc[:test_date].tail(300)

        graph_data = build_period_graph(
            returns_daily=historical_returns,
            period_end=test_date,
            tickers=tickers,
            features_matrix=features_matrix,
            cfg=graph_config
        )

        # Test GAT model creation
        model = GATPortfolio(
            in_dim=10,
            hidden_dim=64,
            heads=8,
            num_layers=3,
            dropout=0.3,
            residual=True,
            use_gatv2=True,
            use_edge_attr=True,
            head="direct",
            activation="sparsemax"
        )

        # Test forward pass
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        x = graph_data.x.to(device)
        edge_index = graph_data.edge_index.to(device)
        edge_attr = graph_data.edge_attr.to(device) if graph_data.edge_attr is not None else None
        mask_valid = torch.ones(x.shape[0], dtype=torch.bool, device=device)

        with torch.no_grad():
            result = model(x, edge_index, mask_valid, edge_attr)
            if isinstance(result, tuple):
                result[0]
            else:
                pass


    except Exception:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
