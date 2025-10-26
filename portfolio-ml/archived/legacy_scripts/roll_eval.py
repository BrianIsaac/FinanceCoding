#!/usr/bin/env python3
"""
Rolling window training/evaluation harness.

- Default schedule: 36m train / 12m val / 12m test, step = 12m
- For each roll:
    * override cfg.split with window boundaries
    * set a unique out_dir (e.g., outputs/gat/roll_00_2018-04-01_to_2019-03-31)
    * call train_gat(cfg)
    * read produced *_daily_returns.csv files, trim to [test_start, test_end], compute metrics
- Aggregate metrics across rolls into a summary CSV.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.train import list_samples, train_gat  # type: ignore

# ----------------------- time helpers -----------------------


def _to_ts(d: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(d).normalize()


def _add_months(ts: pd.Timestamp, months: int) -> pd.Timestamp:
    return (ts + pd.DateOffset(months=months)).normalize()


def _clamp_right(ts: pd.Timestamp, max_ts: pd.Timestamp) -> pd.Timestamp:
    return min(ts, max_ts)


# ----------------------- metrics -----------------------


def _metrics_from_daily_returns(r: pd.Series) -> dict[str, float]:
    """Compute CAGR/AnnMean/AnnVol/Sharpe/MDD from DAILY returns series."""
    r = r.dropna()
    if r.empty:
        return {
            "CAGR": np.nan,
            "AnnMean": np.nan,
            "AnnVol": np.nan,
            "Sharpe": np.nan,
            "MDD": np.nan,
        }
    eq = (1.0 + r).cumprod()
    ann = 252.0
    n = int(r.shape[0])
    cagr = float(eq.iloc[-1] ** (ann / max(n, 1)) - 1.0)
    annmean = float(r.mean() * ann)
    annvol = float(r.std(ddof=0) * np.sqrt(ann))
    sharpe = float(annmean / annvol) if annvol > 0 else np.nan
    dd = eq / eq.cummax() - 1.0
    mdd = float(dd.min())
    return {"CAGR": cagr, "AnnMean": annmean, "AnnVol": annvol, "Sharpe": sharpe, "MDD": mdd}


# ----------------------- roll plan -----------------------


@dataclass
class RollWindow:
    idx: int
    train_start: pd.Timestamp
    val_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _infer_available_range(graph_dir: Path, labels_dir: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    samples = list_samples(graph_dir, labels_dir)
    if not samples:
        raise RuntimeError(f"No samples found under {graph_dir} / {labels_dir}")
    dates = sorted(s.ts for s in samples)
    return dates[0], dates[-1]


def _build_rolls(
    graph_dir: Path,
    labels_dir: Path,
    train_months: int,
    val_months: int,
    test_months: int,
    step_months: int,
    min_buffer_days: int = 0,
    explicit_start: pd.Timestamp | None = None,
    explicit_end: pd.Timestamp | None = None,
) -> list[RollWindow]:
    start_all, end_all = _infer_available_range(graph_dir, labels_dir)
    if explicit_start is not None:
        start_all = max(start_all, explicit_start)
    if explicit_end is not None:
        end_all = min(end_all, explicit_end)

    rolls: list[RollWindow] = []
    t0 = start_all
    idx = 0
    while True:
        tr_start = t0
        val_start = _add_months(tr_start, train_months)
        te_start = _add_months(val_start, val_months)
        te_end = _add_months(te_start, test_months) - pd.Timedelta(days=1)

        if te_start > end_all:
            break
        te_end = _clamp_right(te_end, end_all)

        if (val_start - tr_start).days <= min_buffer_days or (
            te_start - val_start
        ).days <= min_buffer_days:
            break

        rolls.append(
            RollWindow(
                idx=idx,
                train_start=tr_start,
                val_start=val_start,
                test_start=te_start,
                test_end=te_end,
            )
        )

        t0 = _add_months(tr_start, step_months)
        idx += 1

        if _add_months(t0, train_months + val_months + test_months) > end_all + pd.DateOffset(
            days=1
        ):
            break

    if not rolls:
        raise RuntimeError(
            "No valid rolling windows constructed — check your months/step and data coverage."
        )
    return rolls


# ----------------------- main runner -----------------------


def run_rolls(
    base_cfg_path: Path,
    out_root: Path,
    train_months: int = 36,
    val_months: int = 12,
    test_months: int = 12,
    step_months: int = 12,
    start: str | None = None,
    end: str | None = None,
):
    cfg = OmegaConf.load(str(base_cfg_path))

    graph_dir = Path(cfg.data.graph_dir)
    labels_dir = Path(cfg.data.labels_dir)

    rolls = _build_rolls(
        graph_dir=graph_dir,
        labels_dir=labels_dir,
        train_months=train_months,
        val_months=val_months,
        test_months=test_months,
        step_months=step_months,
        explicit_start=_to_ts(start) if start else None,
        explicit_end=_to_ts(end) if end else None,
    )

    out_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    for r in rolls:
        roll_tag = f"roll_{r.idx:02d}_{r.test_start.date()}_{r.test_end.date()}"
        roll_out = out_root / roll_tag

        cfg_roll = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg_roll.split.train_start = str(r.train_start.date())
        cfg_roll.split.val_start = str(r.val_start.date())
        cfg_roll.split.test_start = str(r.test_start.date())
        cfg_roll.train.out_dir = str(roll_out)

        roll_out.mkdir(parents=True, exist_ok=True)

        # train + test (writes daily returns)
        train_gat(cfg_roll)

        # ---- collect and TRIM to test window ----
        def _read_daily(path: Path) -> pd.Series | None:
            if not path.exists():
                return None
            s = pd.read_csv(path, parse_dates=[0], index_col=0).iloc[:, 0]
            s.index = pd.to_datetime(s.index)
            return s.loc[(s.index >= r.test_start) & (s.index <= r.test_end)]

        # GAT
        gat_path = roll_out / "gat_daily_returns.csv"
        gat_r = _read_daily(gat_path)
        if gat_r is not None and not gat_r.empty:
            m = _metrics_from_daily_returns(gat_r)
            all_rows.append({"roll": r.idx, "strategy": "GAT", **m})

        # Baselines (if present)
        for strat in ["ew", "mv", "hrp", "minvar"]:
            p = roll_out / f"{strat}_daily_returns.csv"
            s = _read_daily(p)
            if s is None or s.empty:
                continue
            m = _metrics_from_daily_returns(s)
            all_rows.append({"roll": r.idx, "strategy": strat.upper(), **m})

    per_roll_df = pd.DataFrame(all_rows)
    per_roll_csv = out_root / "per_roll_metrics.csv"
    per_roll_df.to_csv(per_roll_csv, index=False)

    def _agg(df: pd.DataFrame) -> pd.DataFrame:
        cols = ["CAGR", "AnnMean", "AnnVol", "Sharpe", "MDD"]
        g = df.groupby("strategy")[cols]
        mean = g.mean().add_suffix("_mean")
        std = g.std(ddof=0).add_suffix("_std")
        n = g.count().iloc[:, :1].rename(columns={g.count().columns[0]: "N"})
        return pd.concat([n, mean, std], axis=1).reset_index()

    summary = _agg(per_roll_df)
    summary_csv = out_root / "summary_across_rolls.csv"
    summary.to_csv(summary_csv, index=False)


# ----------------------- CLI -----------------------


def parse_args():
    p = argparse.ArgumentParser(description="Rolling window GAT evaluation")
    p.add_argument("--config", required=True, type=Path, help="Path to base config.yaml")
    p.add_argument(
        "--out_root",
        type=Path,
        default=Path("outputs/rolls"),
        help="Root folder to store roll outputs",
    )
    p.add_argument("--train_months", type=int, default=36)
    p.add_argument("--val_months", type=int, default=12)
    p.add_argument("--test_months", type=int, default=12)
    p.add_argument(
        "--step_months", type=int, default=12, help="Slide length between successive rolls"
    )
    p.add_argument(
        "--start", type=str, default=None, help="Optional YYYY-MM-DD to clamp earliest roll start"
    )
    p.add_argument(
        "--end", type=str, default=None, help="Optional YYYY-MM-DD to clamp latest roll end"
    )
    return p.parse_args()


def main():
    args = parse_args()
    run_rolls(
        base_cfg_path=args.config,
        out_root=args.out_root,
        train_months=args.train_months,
        val_months=args.val_months,
        test_months=args.test_months,
        step_months=args.step_months,
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":
    main()
