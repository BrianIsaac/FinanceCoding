#!/usr/bin/env python3
"""
Train GAT with optional rolling-window evaluation.

Usage (single window, same as before):
  python scripts/train_gat.py

Usage (rolling windows, example):
  python scripts/train_gat.py roll.enabled=true roll.train_months=36 roll.val_months=12 roll.test_months=12 roll.step_months=6

What this adds (non-breaking):
- Rolling evaluation that slides (train/val/test) windows across time.
- Each roll writes to its own out_dir subfolder: <cfg.train.out_dir>/roll_<k>/
- Early stopping patience can be controlled from config: train.early_stop_patience (default: 0 = disabled).
- After all rolls finish, an aggregate CSV is written to <cfg.train.out_dir>/rolling_summary.csv
  averaging the key metrics across rolls (and across seeds if seeds > 1).

Notes:
- This script only orchestrates the rolls. The per-run training logic remains in src.train.train_gat.
- No changes are required to your existing configs to keep single-window behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
from typing import List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# We call your existing entrypoint.
from src.train import train_gat as _train_single  # noqa: E402

# ----------------------
# Helpers
# ----------------------

def _to_path(p) -> Path:
    return Path(hydra.utils.to_absolute_path(str(p)))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _month_end_series(daily_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Collapse business-daily index to month-end index present in the data."""
    s = pd.Series(1, index=daily_index)
    # Keep last business day per month that exists in the data.
    mi = s.groupby([daily_index.year, daily_index.month]).tail(1).index
    return pd.DatetimeIndex(mi)


@dataclass
class Roll:
    k: int
    train_start: str
    val_start: str
    test_start: str
    test_end: Optional[str]  # hint for downstream (some impls ignore); safe to include
    out_dir: Path


def _make_rolls(
    month_ends: pd.DatetimeIndex,
    base_out_dir: Path,
    train_m: int,
    val_m: int,
    test_m: int,
    step_m: int,
    start_on: Optional[pd.Timestamp] = None,
    stop_on: Optional[pd.Timestamp] = None,
) -> List[Roll]:
    """
    Build sliding (train/val/test) windows along the month-end timeline.

    Windows:
      [train_start, val_start) -> train
      [val_start, test_start)  -> val
      [test_start, test_end]   -> test (inclusive end for clarity)
    """
    assert train_m > 0 and val_m > 0 and test_m > 0 and step_m > 0
    me = month_ends.sort_values()
    if start_on is not None:
        me = me[me >= start_on]
    if stop_on is not None:
        me = me[me <= stop_on]

    rolls: List[Roll] = []
    k = 0
    i = 0
    while True:
        train_end_idx = i + train_m - 1
        val_end_idx = train_end_idx + val_m
        test_end_idx = val_end_idx + test_m

        if test_end_idx >= len(me):
            break

        train_start = me[i]
        val_start = me[train_end_idx + 1]
        test_start = me[val_end_idx + 1]
        test_end = me[test_end_idx]

        k += 1
        out_dir = base_out_dir / f"roll_{k:02d}"
        rolls.append(
            Roll(
                k=k,
                train_start=train_start.strftime("%Y-%m-%d"),
                val_start=val_start.strftime("%Y-%m-%d"),
                test_start=test_start.strftime("%Y-%m-%d"),
                test_end=test_end.strftime("%Y-%m-%d"),
                out_dir=out_dir,
            )
        )
        i += step_m

    return rolls


def _read_compare_csv(roll_dir: Path) -> Optional[pd.DataFrame]:
    p = roll_dir / "compare_gat_vs_baselines.csv"
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    # fallback: try strategy_metrics.csv (GAT-only)
    q = roll_dir / "strategy_metrics.csv"
    if q.exists():
        try:
            df = pd.read_csv(q)
            df["strategy"] = df.get("strategy", pd.Series(["GAT"] * len(df)))
            return df
        except Exception:
            return None
    return None


def _aggregate_rolls(rolls: List[Roll], out_dir: Path) -> None:
    """
    Build a simple aggregate CSV across rolls:
    - If compare_gat_vs_baselines.csv exists per roll, stack and average per strategy.
    - Else fallback to strategy_metrics.csv (GAT only).
    """
    frames = []
    for r in rolls:
        df = _read_compare_csv(r.out_dir)
        if df is None:
            continue
        df = df.copy()
        df["roll"] = r.k
        frames.append(df)

    if not frames:
        print("[Rolling] No per-roll metrics found to aggregate.")
        return

    all_df = pd.concat(frames, ignore_index=True)
    # Keep common metric columns if present.
    metric_cols = [c for c in ["CAGR", "AnnMean", "AnnVol", "Sharpe", "MDD"] if c in all_df.columns]
    group_cols = ["strategy"]
    agg = (
        all_df.groupby(group_cols, as_index=False)[metric_cols].mean()
        if metric_cols else all_df.groupby(group_cols, as_index=False).mean(numeric_only=True)
    )

    agg_path = out_dir / "rolling_summary.csv"
    agg.to_csv(agg_path, index=False)
    print(f"[Rolling] Wrote aggregate metrics -> {agg_path}")


# ----------------------
# Hydra main
# ----------------------

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Orchestrate either:
    - single-window training (default, exact same behavior as before), or
    - rolling-window training when roll.enabled=true
    """

    # Ensure absolute base out_dir
    base_out_dir = _to_path(cfg.train.out_dir)
    _ensure_dir(base_out_dir)

    # Default: no rolling, behave as before
    roll_cfg = cfg.get("roll")
    rolling_enabled = bool(roll_cfg.get("enabled")) if roll_cfg is not None else False

    if not rolling_enabled:
        # Optional: pass early-stop patience down into cfg.train.* so src.train can see it.
        # (If src.train ignores it, it's still harmless.)
        print("[Mode] Single window training")
        _train_single(cfg)
        return

    # ------------- Rolling path -------------
    print("[Mode] Rolling windows enabled")

    # Read the returns index to infer month-ends
    returns_path = _to_path(cfg.data.returns_daily)
    if not returns_path.exists():
        raise FileNotFoundError(f"returns_daily parquet not found: {returns_path}")

    df_ret = pd.read_parquet(returns_path)
    if "date" in df_ret.columns:
        idx = pd.to_datetime(df_ret["date"])
    elif df_ret.index.name is not None:
        idx = pd.to_datetime(df_ret.index)
    else:
        # if saved with a default RangeIndex but a 'Date' column exists
        for col in ["Date", "DATE", "dt", "timestamp"]:
            if col in df_ret.columns:
                idx = pd.to_datetime(df_ret[col])
                break
        else:
            raise ValueError("Could not infer date index from returns_daily parquet.")

    idx = pd.DatetimeIndex(idx).sort_values().unique()
    month_ends = _month_end_series(idx)

    # Parameters (with sane defaults)
    train_m = int(roll_cfg.get("train_months", 36))
    val_m   = int(roll_cfg.get("val_months", 12))
    test_m  = int(roll_cfg.get("test_months", 12))
    step_m  = int(roll_cfg.get("step_months", 12))
    # Optional bounds
    start_on = pd.to_datetime(roll_cfg.get("start")) if roll_cfg.get("start") else None
    stop_on  = pd.to_datetime(roll_cfg.get("stop")) if roll_cfg.get("stop") else None

    # Make the rolls
    rolls = _make_rolls(
        month_ends=month_ends,
        base_out_dir=base_out_dir,
        train_m=train_m,
        val_m=val_m,
        test_m=test_m,
        step_m=step_m,
        start_on=start_on,
        stop_on=stop_on,
    )
    if not rolls:
        raise RuntimeError("No rolling windows produced. Check your (train/val/test/step) months and available data range.")

    print(f"[Rolling] Will run {len(rolls)} rolls "
          f"({train_m}m train / {val_m}m val / {test_m}m test, step={step_m}m).")

    # Run each roll by cloning cfg and overriding split/out_dir
    for r in rolls:
        print(f"\n=== Roll {r.k:02d} ===")
        cfg_roll = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))  # deep copy

        # Ensure unique out_dir per roll
        cfg_roll.train.out_dir = str(r.out_dir)
        _ensure_dir(r.out_dir)

        # Override splits
        cfg_roll.split.train_start = r.train_start
        cfg_roll.split.val_start   = r.val_start
        cfg_roll.split.test_start  = r.test_start
        # Provide optional test_end hint for downstream (harmless if ignored)
        if "test_end" not in cfg_roll.split:
            OmegaConf.set_struct(cfg_roll.split, False)  # allow new key temporarily
            cfg_roll.split.test_end = r.test_end
            OmegaConf.set_struct(cfg_roll.split, True)

        # Propagate early-stop patience if present at roll.*
        if roll_cfg.get("early_stop_patience") is not None:
            cfg_roll.train.early_stop_patience = int(roll_cfg.get("early_stop_patience"))

        # Optionally propagate seed sweep if present at roll.seeds (comma or list)
        seeds = None
        if roll_cfg.get("seeds") is not None:
            val = roll_cfg.get("seeds")
            if isinstance(val, (list, tuple)):
                seeds = [int(x) for x in val]
            elif isinstance(val, str):
                seeds = [int(x) for x in val.split(",") if str(x).strip()]
            else:
                seeds = [int(val)]
        if seeds:
            # Run multiple seeds for this roll, writing into subfolders roll_k/seed_x/
            for s in seeds:
                cfg_seed = OmegaConf.create(OmegaConf.to_container(cfg_roll, resolve=True))
                seed_out = r.out_dir / f"seed_{s}"
                _ensure_dir(seed_out)
                cfg_seed.train.out_dir = str(seed_out)
                cfg_seed.train.seed = s
                _train_single(cfg_seed)
        else:
            _train_single(cfg_roll)

    # Aggregate
    _aggregate_rolls(rolls, base_out_dir)


if __name__ == "__main__":
    main()
