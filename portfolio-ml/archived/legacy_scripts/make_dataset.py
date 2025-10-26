"""Entry point to build market data and per-rebalance features using Hydra."""

from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hydra
from omegaconf import DictConfig

from src.data.processors.data_pipeline import run_data_pipeline
from src.data.processors.features import rolling_features
from src.utils.environment import load_project_dotenv


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Builds prices/returns/rebalance artefacts and per-rebalance features.

    Args:
        cfg: Hydra configuration composed from configs in `../configs`.

    Side Effects:
        Writes parquet/CSV artefacts into the current Hydra run directory.
    """
    # Ensure NASDAQ_DATA_LINK_API_KEY from .env is available (if present).
    load_project_dotenv(".env", override=False)

    # Log resolved config for reproducibility.

    artefacts: dict[str, object] = run_data_pipeline(cfg)

    feats = rolling_features(
        px=artefacts["prices"],  # type: ignore[arg-type]
        rets=artefacts["returns"],  # type: ignore[arg-type]
        rebal_dates=artefacts["rebalance_dates"],  # type: ignore[arg-type]
    )

    out_dir = cfg.paths.features_dir
    os.makedirs(out_dir, exist_ok=True)
    for t, df in feats.items():
        df.to_parquet(os.path.join(out_dir, f"features_{t.date()}.parquet"))


if __name__ == "__main__":
    main()
