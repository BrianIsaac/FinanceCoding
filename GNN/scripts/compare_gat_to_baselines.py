#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare GAT vs baselines across many runs (seeds / rolls) and report:
- CAGR, AnnMean, AnnVol, Sharpe, MDD
- Deflated Sharpe (z, p) with multiple-trial correction
Aggregates over directories (each one = a single run's outputs).
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Small, dependency-free normal CDF and inverse CDF (Acklam)
# ------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _norm_ppf(p: float) -> float:
    # Acklam’s approximation
    if not (0.0 < p < 1.0):
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        raise ValueError("p must be in (0,1)")

    # Coefficients in rational approximations
    a = [ -3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00 ]
    b = [ -5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01 ]
    c = [ -7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00 ]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00 ]

    plow  = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2*math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if phigh < p:
        q = math.sqrt(-2*math.log(1-p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p - 0.5
    r = q*q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)

# ------------------------------------------------------------
# Metrics & Deflated Sharpe
# ------------------------------------------------------------

def _compute_metrics_from_returns(r: pd.Series) -> Dict[str, float]:
    r = r.dropna()
    if r.empty:
        return {"CAGR": np.nan, "AnnMean": np.nan, "AnnVol": np.nan, "Sharpe": np.nan, "MDD": np.nan, "N": 0}
    eq = (1.0 + r).cumprod()
    ann = 252.0
    n = int(r.shape[0])
    cagr    = float(eq.iloc[-1] ** (ann / max(n, 1)) - 1.0)
    annmean = float(r.mean() * ann)
    annvol  = float(r.std(ddof=0) * math.sqrt(ann))
    sharpe  = float(annmean / annvol) if annvol > 0 else float("nan")
    dd = eq / eq.cummax() - 1.0
    mdd = float(dd.min())
    return {"CAGR": cagr, "AnnMean": annmean, "AnnVol": annvol, "Sharpe": sharpe, "MDD": mdd, "N": n}

def _sample_skew_kurt(r: pd.Series) -> Tuple[float, float]:
    x = r.dropna().values.astype(np.float64)
    n = x.size
    if n < 3:
        return 0.0, 3.0
    m = x.mean()
    s = x.std(ddof=0)
    if s <= 0:
        return 0.0, 3.0
    z = (x - m) / s
    skew = float((z**3).mean())
    kurt = float((z**4).mean())  # normal => 3.0
    return skew, kurt

def deflated_sharpe(sr: float, n: int, skew: float, kurt: float, num_trials: int) -> Tuple[float, float]:
    """
    Bailey & López de Prado (DSR) style deflation:
      sigma_SR = sqrt( (1 - skew*SR + ((kurt-1)/4)*SR^2) / (n - 1) )
      SR*      = z_(1 - 1/T) * sigma_SR
      z_DSR    = (SR - SR*) / sigma_SR = SR/sigma_SR - z_(1 - 1/T)
      p_DSR    = Phi(z_DSR)
    Returns (z_DSR, p_DSR). T = num_trials (number of configs/strategies you sifted through).
    """
    if not np.isfinite(sr) or n <= 1:
        return float("nan"), float("nan")
    # Guard kurtosis
    if kurt <= 1.0:
        kurt = 3.0
    sigma_sr_num = (1.0 - skew * sr + ((kurt - 1.0) / 4.0) * (sr ** 2))
    sigma_sr_den = max(n - 1, 1)
    sigma_sr = math.sqrt(max(sigma_sr_num, 1e-12) / sigma_sr_den)
    T = max(int(num_trials), 1)
    z_star = _norm_ppf(1.0 - 1.0 / T)
    z_dsr = (sr / sigma_sr) - z_star
    p_dsr = _norm_cdf(z_dsr)
    return float(z_dsr), float(p_dsr)

# ------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------

_STRAT_FILES = {
    "GAT":   "gat_daily_returns.csv",
    "EW":    "ew_daily_returns.csv",
    "MV":    "mv_daily_returns.csv",
    "HRP":   "hrp_daily_returns.csv",
    "MinVar":"minvar_daily_returns.csv",
    "TopK_EW":"topk_ew_daily_returns.csv",   # optional if you later add this path
}

def _read_daily_returns(path: Path) -> Optional[pd.Series]:
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    if df.shape[1] == 0:
        return None
    s = df.iloc[:, 0].astype(float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index()

def _collect_run_metrics(run_dir: Path, strategies: List[str]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for strat in strategies:
        f = _STRAT_FILES.get(strat)
        if f is None:
            continue
        s = _read_daily_returns(run_dir / f)
        if s is None or s.empty:
            # fallback: some earlier pipelines saved “*_equity_daily.csv”; we can derive returns
            eq_path = run_dir / f.replace("_daily_returns", "_equity_daily")
            if eq_path.exists():
                eq = pd.read_csv(eq_path, parse_dates=[0], index_col=0).iloc[:, 0]
                eq = eq.astype(float).sort_index()
                s = eq.pct_change().dropna()
            else:
                continue

        m = _compute_metrics_from_returns(s)
        skew, kurt = _sample_skew_kurt(s)
        row = {
            "run_dir": str(run_dir),
            "strategy": strat,
            **m,
            "Skew": skew,
            "Kurt": kurt,
        }
        rows.append(row)
    return rows

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Compare GAT vs baselines across many output folders.")
    p.add_argument("--dirs", nargs="+", required=True,
                   help="One or more run directories (each contains *_daily_returns.csv produced by training/eval). "
                        "You can also pass a glob via your shell.")
    p.add_argument("--strategies", nargs="+",
                   default=["GAT", "EW", "MV", "HRP", "MinVar"],
                   help="Which strategies to include (files must exist in each run dir).")
    p.add_argument("--num_trials", type=int, default=20,
                   help="T in DSR. Set to the number of alternatives you considered "
                        "(configs × strategies × rolls, roughly). Default 20.")
    p.add_argument("--out", type=str, default="compare_across_runs.csv",
                   help="Output CSV (aggregated across dirs).")
    p.add_argument("--per_run_out", type=str, default="compare_per_run.csv",
                   help="Per-run CSV (one row per run×strategy).")
    args = p.parse_args()

    run_dirs = [Path(d).resolve() for d in args.dirs]
    strategies = list(args.strategies)

    # Collect per-run metrics
    per_run: List[Dict[str, float]] = []
    for rd in run_dirs:
        per_run.extend(_collect_run_metrics(rd, strategies))

    if not per_run:
        print("[!] No metrics found. Ensure your run dirs contain *_daily_returns.csv files.")
        return

    df = pd.DataFrame(per_run)
    # Deflated Sharpe per row
    z_list, p_list = [], []
    for _, row in df.iterrows():
        z, pval = deflated_sharpe(
            sr=float(row["Sharpe"]),
            n=int(row["N"]),
            skew=float(row.get("Skew", 0.0)),
            kurt=float(row.get("Kurt", 3.0)),
            num_trials=int(args.num_trials),
        )
        z_list.append(z)
        p_list.append(pval)
    df["DSR_z"] = z_list
    df["DSR_p"] = p_list

    # Save per-run table
    per_run_out = Path(args.per_run_out).resolve()
    df.to_csv(per_run_out, index=False)
    print(f"[OK] Wrote per-run table -> {per_run_out}")

    # Aggregate across runs (mean ± std)
    agg_funcs = {
        "CAGR": ["mean", "std"],
        "AnnMean": ["mean", "std"],
        "AnnVol": ["mean", "std"],
        "Sharpe": ["mean", "std"],
        "MDD": ["mean", "std"],
        "DSR_z": ["mean", "std"],
        "DSR_p": ["mean", "std"],
        "N": ["mean"],  # avg sample size (roughly similar across runs)
    }
    g = df.groupby("strategy").agg(agg_funcs)
    # Flatten columns
    g.columns = ["_".join([c for c in col if c]).strip("_") for col in g.columns.values]
    g = g.reset_index()

    out_path = Path(args.out).resolve()
    g.to_csv(out_path, index=False)
    print(f"[OK] Wrote aggregated comparison -> {out_path}")
    print("\n=== Aggregated (mean ± std) ===")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(g)

if __name__ == "__main__":
    main()
