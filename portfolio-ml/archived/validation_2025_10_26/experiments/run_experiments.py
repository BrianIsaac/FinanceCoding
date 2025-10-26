#!/usr/bin/env python3
"""
Rolling-window experiment runner for GAT.
- Generates (36m train / 12m val / 12m test) windows, sliding by STEP months.
- For each roll and seed, calls scripts/train_gat.py with overrides.
- After each run, archives outputs/gat -> outputs/experiments/<roll>/<seed>/
- Parses compare_gat_vs_baselines.csv, gat_test_stats.csv, train_log.csv
- Aggregates per-run rows + roll/seed averages into outputs/experiments/summary.csv
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta as rd

ROOT = Path(__file__).resolve().parent.parent  # project root (scripts/..)
TRAIN = ROOT / "scripts" / "train_gat.py"
RET_PATH = ROOT / "processed" / "returns_daily.parquet"

OUT_FIXED = ROOT / "outputs" / "gat"
ARCHIVE_ROOT = ROOT / "outputs" / "experiments"
SUMMARY_CSV = ARCHIVE_ROOT / "summary.csv"

SEEDS = [1, 2, 3]  # keep it quick; change to 5 if you like
TOPK = 40  # consistent with your recent runs

# rolling window config (months)
TRAIN_M = 36
VAL_M = 12
TEST_M = 12
STEP_M = 12  # slide by 12 months


# --------- helpers ---------
def read_csv_safe(path: Path) -> list[dict[str, str]] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def last_row_csv(path: Path) -> dict[str, str] | None:
    rows = read_csv_safe(path)
    return rows[-1] if rows else None


def month_floor(d: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=d.year, month=d.month, day=1)


def gen_rolls(dates: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Return list of (train_start, val_start, test_start) monthly stamps."""
    if len(dates) == 0:
        raise RuntimeError("returns_daily.parquet has no rows")
    start = month_floor(dates[0])
    end = month_floor(dates[-1])
    rolls = []
    t0 = start
    while True:
        tr0 = t0
        va0 = tr0 + rd(months=TRAIN_M)
        te0 = va0 + rd(months=VAL_M)
        te1 = te0 + rd(months=TEST_M)
        if te1 > end:
            break
        rolls.append((tr0, va0, te0))
        t0 = t0 + rd(months=STEP_M)
    return rolls


def fmt(d: pd.Timestamp) -> str:
    return d.strftime("%Y-%m-%d")


def run_once(overrides: list[str]) -> None:
    cmd = [sys.executable, str(TRAIN), *overrides]
    if OUT_FIXED.exists():
        shutil.rmtree(OUT_FIXED)
    OUT_FIXED.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)


def archive_run(roll_name: str, seed_name: str) -> Path:
    dest = ARCHIVE_ROOT / roll_name / seed_name
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(OUT_FIXED), str(dest))
    return dest


def collect_metrics(exp_dir: Path, tags: dict) -> dict:
    out: dict = {**tags}
    # final_equity
    ts = read_csv_safe(exp_dir / "gat_test_stats.csv")
    if ts:
        out["final_equity"] = float(ts[0].get("final_equity", "nan"))

    # log tail
    last = last_row_csv(exp_dir / "train_log.csv")
    if last:
        out["epoch"] = int(float(last.get("epoch", "0")))
        out["val_mean_last"] = float(last.get("val_mean", "nan"))
        out["best_val"] = float(last.get("best_val", "nan"))
        out["train_loss_last"] = float(last.get("train_loss", "nan"))

    # strategies
    comp = read_csv_safe(exp_dir / "compare_gat_vs_baselines.csv")
    if comp:
        for row in comp:
            strat = row.get("strategy", "").strip()
            for k in ("CAGR", "AnnMean", "AnnVol", "Sharpe", "MDD", "DeflatedSharpe"):
                if k in row and row[k] != "":
                    out[f"{strat}_{k}"] = float(row[k])
    return out


def write_summary(rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        [w.writerow(r) for r in rows]


# --------- main ---------
def main() -> None:
    df = pd.read_parquet(RET_PATH)  # index=date
    df.index = pd.to_datetime(df.index)
    rolls = gen_rolls(df.index)

    rows: list[dict] = []
    failures: list[dict] = []

    for i, (tr0, va0, te0) in enumerate(rolls, start=1):
        roll_name = f"roll_{i:02d}_{fmt(tr0)}_{fmt(va0)}_{fmt(te0)}"
        for seed in SEEDS:
            seed_name = f"seed_{seed}"
            try:
                overrides = [
                    "train.out_dir=outputs/gat",
                    # objectives + regularizers
                    "loss.objective=sharpe_rnext",
                    "loss.sharpe_eps=1e-6",
                    "loss.turnover_bps=10",
                    "loss.entropy_coef=0.0",
                    "loss.l1_coef=0.002",
                    # model/head
                    "model.head=markowitz",
                    "model.markowitz_mode=chol",
                    f"model.markowitz_topk={TOPK}",
                    "model.markowitz_gamma=3.0",
                    "model.use_edge_attr=true",
                    # covariance
                    "model.cov_estimator=lw",  # ← new: Ledoit–Wolf
                    "model.cov_ridge_eps=1e-5",  # fallback if lw is unavailable
                    # training
                    "train.epochs=60",
                    "train.early_stop_patience=8",  # monitor val Sharpe
                    "train.grad_clip=1.0",
                    "train.ordered_when_memory=true",
                    f"train.seed={seed}",
                    # splits per roll
                    f"split.train_start={fmt(tr0)}",
                    f"split.val_start={fmt(va0)}",
                    f"split.test_start={fmt(te0)}",
                    # evaluation add-ons
                    f"eval.topk_k={TOPK}",
                    "eval.enable_topk_ew=true",
                ]
                run_once(overrides)
                exp_dir = archive_run(roll_name, seed_name)
                tags = {"roll": roll_name, "seed": seed, "topk": TOPK}
                rows.append(collect_metrics(exp_dir, tags))
            except subprocess.CalledProcessError as e:
                failures.append(
                    {"roll": roll_name, "seed": seed, "error": f"returncode={e.returncode}"}
                )
            except Exception as e:
                failures.append(
                    {"roll": roll_name, "seed": seed, "error": f"{type(e).__name__}: {e}"}
                )

    write_summary(rows)

    if failures:
        fail_path = ARCHIVE_ROOT / "failures.json"
        fail_path.write_text(json.dumps(failures, indent=2))


if __name__ == "__main__":
    main()
