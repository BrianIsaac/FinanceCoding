# src/rolling_eval.py
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src import train as train_mod  # we will call train_mod.train_gat()

# local imports
from src.evaluation.metrics.portfolio_metrics import dsr_from_returns

# ------------------------------------------------------------
# Rolling split helpers
# ------------------------------------------------------------


@dataclass(frozen=True)
class RollSplit:
    train_start: pd.Timestamp
    val_start: pd.Timestamp
    test_start: pd.Timestamp

    def to_datestr_tuple(self) -> Tuple[str, str, str]:
        return (
            self.train_start.date().isoformat(),
            self.val_start.date().isoformat(),
            self.test_start.date().isoformat(),
        )


def _month_add(ts: pd.Timestamp, months: int) -> pd.Timestamp:
    return (ts + pd.DateOffset(months=months)).normalize()


def make_rolling_splits(
    sample_timestamps: List[pd.Timestamp],
    *,
    train_months: int = 36,
    val_months: int = 12,
    test_months: int = 12,
    step_months: int = 12,
) -> List[RollSplit]:
    """
    Build rolling (train/val/test) splits by months.
    Uses sample_timestamps (e.g., graph_YYYY-MM-DD.pt dates) as anchors.
    """
    if not sample_timestamps:
        return []

    ts_sorted = sorted(sample_timestamps)
    t_min = ts_sorted[0].normalize()
    t_max = ts_sorted[-1].normalize()

    splits: List[RollSplit] = []
    cur_train_start = t_min

    while True:
        v0 = _month_add(cur_train_start, train_months)
        t0 = _month_add(v0, val_months)

        # we need at least the test_start to be <= last sample date
        if v0 > t_max or t0 > t_max:
            break

        splits.append(RollSplit(train_start=cur_train_start, val_start=v0, test_start=t0))

        # advance
        cur_train_start = _month_add(cur_train_start, step_months)
        if cur_train_start >= t0:
            # do not allow overlap errors; if step is too large we still progress
            cur_train_start = _month_add(t0, 0)

        # guard infinite loops
        if cur_train_start > t_max:
            break

    return splits


# ------------------------------------------------------------
# Execution per roll & aggregation
# ------------------------------------------------------------


@dataclass
class RollResult:
    roll_id: int
    split: RollSplit
    seed: int
    out_dir: Path
    metrics_row: Optional[Dict[str, float]]  # loaded from strategy_metrics.csv if found
    daily_returns_path: Optional[Path]  # gat_daily_returns.csv if found
    note: Optional[str] = None


def _find_sample_dates(graph_dir: Path) -> List[pd.Timestamp]:
    pats = list(graph_dir.glob("graph_*.pt"))
    out = []
    for p in pats:
        # train.py has _infer_ts; we mimic here to avoid importing private helper
        m = pd.to_datetime(p.stem.split("_")[-1], errors="coerce")
        if pd.isna(m):
            continue
        out.append(pd.Timestamp(m.date()))
    return sorted(set(out))


def _clone_cfg_for_roll(
    base_cfg: DictConfig, split: RollSplit, seed: int, out_dir: Path
) -> DictConfig:
    cfg = copy.deepcopy(base_cfg)
    cfg.split.train_start = split.train_start.date().isoformat()
    cfg.split.val_start = split.val_start.date().isoformat()
    cfg.split.test_start = split.test_start.date().isoformat()
    cfg.train.seed = int(seed)
    cfg.train.out_dir = str(out_dir.as_posix())
    return cfg


def _load_metrics_if_any(out_dir: Path) -> Tuple[Optional[Dict[str, float]], Optional[Path]]:
    sm_path = out_dir / "strategy_metrics.csv"
    r_path = out_dir / "gat_daily_returns.csv"
    metrics_row = None
    if sm_path.exists():
        try:
            df = pd.read_csv(sm_path)
            if not df.empty:
                row = df.iloc[0].to_dict()
                # ensure floats
                for k, v in list(row.items()):
                    if k != "strategy":
                        try:
                            row[k] = float(v)
                        except Exception:
                            row[k] = np.nan
                metrics_row = row
        except Exception:
            metrics_row = None
    return metrics_row, (r_path if r_path.exists() else None)


def run_rolling(
    cfg: DictConfig,
    *,
    out_root: Path,
    train_months: int = 36,
    val_months: int = 12,
    test_months: int = 12,
    step_months: int = 12,
    seeds: Iterable[int] = (42,),
    early_stop: bool = True,
    es_patience: int = 5,
    es_min_delta: float = 0.0,
) -> Dict[str, object]:
    """
    Orchestrate multiple rolling train/val/test runs.
    For each roll:
      - adjust cfg.split.* to the roll dates
      - set cfg.train.seed, cfg.train.out_dir to a roll/seed-specific folder
      - optionally enable early stopping parameters (train.py will consume these)
      - call src.train.train_gat()
      - collect metrics & returns
    Finally:
      - aggregate metrics across rolls & seeds
      - compute conservative Deflated Sharpe across all trials

    Returns a dict payload with:
      {
        "splits": [ ... ],
        "rows": [ per roll/seed metrics... ],
        "summary": { "avg": ..., "std": ..., "DSR": ... }
      }
    """
    out_root.mkdir(parents=True, exist_ok=True)

    # inject early-stop hints into cfg (train.py should honor them)
    if early_stop:
        # We'll add these keys to cfg.train; the refactored train.py will look for them.
        cfg.train.setdefault("early_stop_on_val_sharpe", True)
        cfg.train.setdefault("early_stop_patience", int(es_patience))
        cfg.train.setdefault("early_stop_min_delta", float(es_min_delta))

    # Discover all sample dates and build rolls
    graph_dir = Path(cfg.data.graph_dir)
    sample_dates = _find_sample_dates(graph_dir)
    splits = make_rolling_splits(
        sample_dates,
        train_months=train_months,
        val_months=val_months,
        test_months=test_months,
        step_months=step_months,
    )

    if not splits:
        raise RuntimeError(
            "No rolling splits could be created; check your graph dates and parameters."
        )

    # Execute
    results: List[RollResult] = []
    for ridx, split in enumerate(splits):
        roll_dir = (
            out_root / f"roll_{ridx:02d}_{split.train_start.date()}_{split.test_start.date()}"
        )
        for seed in seeds:
            sub_out = roll_dir / f"seed_{int(seed)}"
            sub_out.mkdir(parents=True, exist_ok=True)

            cfg_roll = _clone_cfg_for_roll(cfg, split, seed=int(seed), out_dir=sub_out)

            # also drop any previous outputs if you want clean runs
            # (we intentionally do not delete to allow resuming)

            # run training & backtest
            try:
                train_mod.train_gat(cfg_roll)  # <-- relies on refactored train.py
                note = None
            except Exception as e:
                # still record the failure but continue
                note = f"FAIL: {type(e).__name__}: {e}"

            # load metrics/returns if available
            met, r_path = _load_metrics_if_any(sub_out)
            results.append(
                RollResult(
                    roll_id=ridx,
                    split=split,
                    seed=int(seed),
                    out_dir=sub_out,
                    metrics_row=met,
                    daily_returns_path=r_path,
                    note=note,
                )
            )

    # Aggregate across rolls/seeds
    rows: List[Dict[str, object]] = []
    r_all_concat: List[pd.Series] = []

    for rr in results:
        base = {
            "roll_id": rr.roll_id,
            "train_start": rr.split.train_start.date().isoformat(),
            "val_start": rr.split.val_start.date().isoformat(),
            "test_start": rr.split.test_start.date().isoformat(),
            "seed": rr.seed,
            "out_dir": str(rr.out_dir.as_posix()),
        }
        if rr.metrics_row is not None:
            rows.append({**base, **rr.metrics_row})
        else:
            rows.append(
                {
                    **base,
                    "strategy": "GAT",
                    "CAGR": np.nan,
                    "AnnMean": np.nan,
                    "AnnVol": np.nan,
                    "Sharpe": np.nan,
                    "MDD": np.nan,
                }
            )

        if rr.daily_returns_path is not None and rr.daily_returns_path.exists():
            try:
                r = pd.read_csv(rr.daily_returns_path, parse_dates=[0], index_col=0).iloc[:, 0]
                r.index = pd.to_datetime(r.index)
                r_all_concat.append(r.rename(f"roll{rr.roll_id}_seed{rr.seed}"))
            except Exception:
                pass

    # compute average metrics across rows
    df_rows = pd.DataFrame(rows)
    # drop non-numeric for averaging
    metric_cols = ["CAGR", "AnnMean", "AnnVol", "Sharpe", "MDD"]
    df_metrics = df_rows[metric_cols].apply(pd.to_numeric, errors="coerce")
    avg_metrics = df_metrics.mean(skipna=True).to_dict()
    std_metrics = df_metrics.std(skipna=True).to_dict()

    # conservative Deflated Sharpe across all concatenated daily return streams
    # approach: stack all streams (align by time, fillna=0), average equal-weight,
    # then compute DSR with num_trials = count of roll/seed streams.
    dsr = np.nan
    if r_all_concat:
        R = pd.concat(r_all_concat, axis=1).fillna(0.0)
        # equal-weight ensemble
        r_ens = R.mean(axis=1)
        dsr = dsr_from_returns(r_ens, sr_benchmark=0.0, num_trials=len(r_all_concat))

        # also persist ensemble daily returns
        (1.0 + r_ens).cumprod().rename("equity").to_frame().to_csv(
            out_root / "ensemble_equity_daily.csv"
        )
        r_ens.rename("r").to_frame().to_csv(out_root / "ensemble_daily_returns.csv")

    # Save artifacts
    df_rows.to_csv(out_root / "rolling_results_detailed.csv", index=False)
    summary_payload = {
        "num_rolls": len(splits),
        "num_streams": len(r_all_concat),
        "avg": avg_metrics,
        "std": std_metrics,
        "deflated_sharpe_conservative": float(dsr) if np.isfinite(dsr) else np.nan,
    }
    with (out_root / "rolling_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"[Rolling] wrote detailed -> {out_root/'rolling_results_detailed.csv'}")
    print(f"[Rolling] wrote summary  -> {out_root/'rolling_summary.json'}")

    return {
        "splits": [s.to_datestr_tuple() for s in splits],
        "rows": rows,
        "summary": summary_payload,
    }
