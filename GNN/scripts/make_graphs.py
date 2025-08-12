"""Builds quarterly graph snapshots from 30-day volatility using distance correlation.

This script:
- Loads processed prices/returns and quarterly rebalance dates (from make_dataset.py).
- For each rebalance date t:
  - Uses a 3-year lookback (configurable) of daily returns up to t.
  - Computes 30-day realised volatility per asset within that window.
  - Computes pairwise distance correlation across those volatility series.
  - Sparsifies to a planar-like or thin graph (TMFG preferred; MST/KNN as fallback).
  - Packs node features (from features/features_YYYY-MM-DD.parquet) and edges into a PyG Data object.

Outputs:
  processed/graphs/graph_YYYY-MM-DD.pt for each rebalance date.
"""

from __future__ import annotations

import os

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from src.graph import build_quarterly_graphs


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint that builds and saves PyG graph snapshots.

    Args:
        cfg: Hydra DictConfig composed from configs/config.yaml (includes data, graph).
    """
    print(OmegaConf.to_yaml(cfg))

    prices = pd.read_parquet(cfg.paths.prices_processed)
    returns = pd.read_parquet(cfg.paths.returns_processed)
    rb_dates = pd.read_csv(cfg.paths.rebalance_csv)["rebalance_date"].astype("datetime64[ns]")

    os.makedirs(cfg.graph.save_dir, exist_ok=True)

    graphs = build_quarterly_graphs(
        prices=prices,
        returns=returns,
        rebalance_dates=rb_dates,
        lookback_days=int(cfg.graph.lookback_days),
        vol_window_days=int(cfg.graph.vol_window_days),
        similarity=str(cfg.graph.similarity),
        filter_method=str(cfg.graph.filter.method),
        k_neighbors=int(cfg.graph.filter.k_neighbors),
        undirected=bool(cfg.graph.make_undirected),
        features_dir=str(cfg.paths.features_dir),
        membership_csv=str(cfg.membership.csv) if "membership" in cfg and cfg.membership.csv else None,
    )

    for ts, data_obj in graphs.items():
        out_path = os.path.join(cfg.graph.save_dir, f"graph_{ts.date()}.pt")
        torch.save(data_obj, out_path)

    print(f"Saved {len(graphs)} graphs to: {os.path.abspath(cfg.graph.save_dir)}")


if __name__ == "__main__":
    main()
