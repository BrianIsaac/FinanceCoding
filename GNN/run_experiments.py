#!/usr/bin/env python3
"""
Multi-seed runner for ANL488 GAT best config.

What it does
------------
- Runs the best-found config (gamma=3.0, ridge=5e-5, topk=40) across seeds [1..5].
- Uses fewer epochs (default 40) for faster turnaround.
- Archives outputs/gat -> outputs/experiments/seed_<n>/.
- Aggregates results into outputs/experiments/summary.csv.
- Appends mean ± std row for GAT Sharpe, CAGR, MDD.

Run
---
python run_experiments.py
"""

from __future__ import annotations
import csv
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional

# --------- Paths ---------
ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "scripts" / "train_gat.py"
OUT_FIXED = ROOT / "outputs" / "gat"
ARCHIVE_ROOT = ROOT / "outputs" / "experiments"
SUMMARY_CSV = ARCHIVE_ROOT / "summary.csv"

# --------- Base config ---------
BASE_OVERRIDES = [
    "train.out_dir=outputs/gat",
    "loss.turnover_bps=10",
    "model.head=markowitz",
    "model.markowitz_mode=chol",
    "model.use_edge_attr=true",
    "loss.objective=daily_log_utility",
    "model.markowitz_topk=40",
    "model.weight_cap=0.02",
    "model.markowitz_gamma=3.0",
    "model.cov_shrinkage_alpha=0.25",
    "model.cov_ridge_eps=5e-05",
    "train.epochs=40",   # shorten for speed
]

SEEDS = [1, 2, 3, 4, 5]

# --------- Helpers ---------
def run_once(overrides: List[str]) -> None:
    cmd = [sys.executable, str(TRAIN), *overrides]
    print("\n==> Running:", " ".join(cmd))
    if OUT_FIXED.exists():
        shutil.rmtree(OUT_FIXED)
    OUT_FIXED.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)

def archive_run(name: str) -> Path:
    dest = ARCHIVE_ROOT / name
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(OUT_FIXED), str(dest))
    return dest

def read_csv_safe(path: Path) -> Optional[List[Dict[str,str]]]:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def collect_metrics(exp_dir: Path, tags: Dict) -> Dict:
    out: Dict = {"exp_name": exp_dir.name, **tags}
    comp = read_csv_safe(exp_dir / "compare_gat_vs_baselines.csv")
    if comp:
        for row in comp:
            strat = row.get("strategy","").strip()
            for k in ("CAGR","AnnMean","AnnVol","Sharpe","MDD"):
                val = row.get(k)
                if val: 
                    out[f"{strat}_{k}"] = float(val)
    return out

def write_summary(rows: List[Dict]) -> None:
    keys = sorted({k for r in rows for k in r.keys()})
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows: 
            w.writerow(r)

        # aggregate mean ± std for GAT
        sharpe = [r["GAT_Sharpe"] for r in rows if "GAT_Sharpe" in r]
        cagr   = [r["GAT_CAGR"]   for r in rows if "GAT_CAGR" in r]
        mdd    = [r["GAT_MDD"]    for r in rows if "GAT_MDD" in r]
        agg = {"exp_name":"mean±std"}
        if sharpe: 
            agg["GAT_Sharpe"] = f"{mean(sharpe):.3f} ± {stdev(sharpe):.3f}"
        if cagr:   
            agg["GAT_CAGR"]   = f"{mean(cagr):.3f} ± {stdev(cagr):.3f}"
        if mdd:    
            agg["GAT_MDD"]    = f"{mean(mdd):.3f} ± {stdev(mdd):.3f}"
        w.writerow(agg)
    print(f"\nWrote summary -> {SUMMARY_CSV}")

# --------- Main ---------
def main() -> None:
    results: List[Dict] = []
    for seed in SEEDS:
        name = f"seed_{seed}"
        overrides = BASE_OVERRIDES + [f"train.seed={seed}"]
        try:
            run_once(overrides)
            exp_dir = archive_run(name)
            metrics = collect_metrics(exp_dir, {"seed":seed})
            results.append(metrics)
            print(f"[OK] {name}")
        except Exception as e:
            print(f"[FAIL] {name} -> {e}")
    if results:
        write_summary(results)

if __name__ == "__main__":
    main()
