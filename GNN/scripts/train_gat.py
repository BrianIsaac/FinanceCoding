#!/usr/bin/env python3
"""
Hydra entrypoint for GAT training/backtest.
"""
from __future__ import annotations

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hydra
from omegaconf import DictConfig
from src.train import train_gat

@hydra.main(version_base="1.3", config_path="../config", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    print(cfg)
    train_gat(cfg)

if __name__ == "__main__":
    main()
