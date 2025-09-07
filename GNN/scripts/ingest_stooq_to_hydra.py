"""
Ingest pre-downloaded Stooq parquet panels into the Hydra pipeline outputs.

- Reads wide daily panels:
    prices: Date x Tickers (Close)
    volume: Date x Tickers (Volume)
- Aligns/cleans using your Hydra config (calendar, ffill, history/missing thresholds).
- Computes daily log returns and period-end rebalance dates (e.g., quarterly).
- Builds per-rebalance features and writes them under the current Hydra run dir.

Usage (defaults to data/stooq/*.parquet):
    python scripts/ingest_stooq_to_hydra.py

Override Stooq paths with Hydra:
    python scripts/ingest_stooq_to_hydra.py ingest.prices_path=data/stooq/prices.parquet ingest.volume_path=data/stooq/volume.parquet
    # Clip date range (optional):
    python scripts/ingest_stooq_to_hydra.py data.start=2010-01-01 data.end=2024-12-31

Notes:
    - This script mirrors the "processed" artefacts produced by make_dataset.py,
      but uses your already-downloaded Stooq data instead of hitting any API.
"""

from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hydra
import pandas as pd
from omegaconf import DictConfig

from src.data.processors.data_pipeline import (
    _align_calendar,
    _basic_clean,
    _compute_daily_returns,
    _make_rebalance_dates,
)

# Reuse your existing helpers
from src.data.processors.features import rolling_features


def _read_panel(path: str, start: str | None, end: str | None) -> pd.DataFrame:
    """Read a wide parquet panel and subset by date if requested.

    Args:
        path: Parquet file path (Date index expected).
        start: Optional ISO start date (YYYY-MM-DD).
        end: Optional ISO end date (YYYY-MM-DD).

    Returns:
        DataFrame with DatetimeIndex and sorted columns.
    """
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index().sort_index(axis=1)
    if start:
        df = df.loc[pd.to_datetime(start) :]
    if end:
        df = df.loc[: pd.to_datetime(end)]
    return df


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint: ingest Stooq panels and generate processed artefacts.

    Args:
        cfg: Hydra DictConfig composed from your configs.
             Reads optional keys under cfg.ingest: prices_path, volume_path.
    """

    # Defaults; override via Hydra: ingest.prices_path=..., ingest.volume_path=...
    prices_path: str = getattr(cfg.get("ingest", {}), "prices_path", "data/stooq/prices.parquet")
    volume_path: str = getattr(cfg.get("ingest", {}), "volume_path", "data/stooq/volume.parquet")

    # Optional date clipping from your existing config
    start = cfg.data.start if "start" in cfg.data else None
    end = cfg.data.end if "end" in cfg.data else None

    # 1) Load Stooq panels
    prices_raw = _read_panel(prices_path, start, end)
    volume_raw = _read_panel(volume_path, start, end)

    # 2) Persist "interim" caches into this Hydra run dir (for provenance)
    os.makedirs(os.path.dirname(cfg.paths.prices_cache), exist_ok=True)
    prices_raw.to_parquet(cfg.paths.prices_cache)
    volume_raw.to_parquet(cfg.paths.volume_cache)

    # 3) Align & clean using your pipeline rules
    prices_aligned = _align_calendar(
        prices_raw,
        how=cfg.cleaning.calendar,
        ffill_limit=cfg.cleaning.ffill_limit,
    )
    prices_clean = _basic_clean(
        prices_aligned,
        min_history_days=cfg.cleaning.min_history_days,
        max_missing_ratio=cfg.cleaning.max_missing_ratio,
    )

    # 4) Returns & save processed panels
    returns = _compute_daily_returns(prices_clean)
    os.makedirs(os.path.dirname(cfg.paths.prices_processed), exist_ok=True)
    prices_clean.to_parquet(cfg.paths.prices_processed)
    returns.to_parquet(cfg.paths.returns_processed)

    # 5) Rebalance dates (e.g., quarterly) & save
    freq = cfg.rebalance.freq if "rebalance" in cfg and "freq" in cfg.rebalance else "Q"
    rb = _make_rebalance_dates(prices_clean, freq=freq)
    os.makedirs(os.path.dirname(cfg.paths.rebalance_csv), exist_ok=True)
    rb.to_frame().to_csv(cfg.paths.rebalance_csv, index=False)

    # 6) Per-rebalance features
    feats = rolling_features(px=prices_clean, rets=returns, rebal_dates=rb)
    out_dir = cfg.paths.features_dir
    os.makedirs(out_dir, exist_ok=True)
    for t, df in feats.items():
        df.to_parquet(os.path.join(out_dir, f"features_{t.date()}.parquet"))


if __name__ == "__main__":
    main()
