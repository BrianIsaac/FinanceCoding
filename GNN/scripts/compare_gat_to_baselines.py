#!/usr/bin/env python3
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("processed/baselines/results.csv")
GAT  = Path("outputs/gat/gat_equity.csv")
OUT  = Path("outputs/gat/compare_gat_vs_baselines.csv")

def ann_stats(equity: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    total_days = (end - start).days
    # assume equity starts at 1.0 at the beginning of the test
    cagr = (equity.iloc[-1]) ** (365.25 / total_days) - 1.0

    # include the first period's return correctly (was 0 before)
    r = equity.div(equity.shift(1).fillna(1.0)) - 1.0

    periods_per_year = len(equity) / (total_days / 365.25)
    ann_mean = r.mean() * periods_per_year
    ann_vol  = r.std(ddof=0) * np.sqrt(periods_per_year) if len(r) > 1 else np.nan
    sharpe   = ann_mean / ann_vol if ann_vol and ann_vol > 0 else np.nan

    mdd = (equity / equity.cummax() - 1.0).min()
    return {"CAGR": cagr, "AnnMean": ann_mean, "AnnVol": ann_vol, "Sharpe": sharpe, "MDD": mdd}

def main():
    b = pd.read_csv(BASE, parse_dates=["rebalance"])
    g = pd.read_csv(GAT, parse_dates=["rebalance"])

    # Align test window
    start = g["rebalance"].iloc[0]
    end   = g["rebalance"].iloc[-1]
    bte = b[(b["rebalance"] >= start) & (b["rebalance"] <= end)].copy()

    # Build a joint frame
    out = pd.DataFrame({"rebalance": g["rebalance"], "equity_GAT": g["equity_gat"]}).set_index("rebalance")
    for col in ["equity_EW","equity_MinVar","equity_HRP","equity_MV"]:
        if col in bte.columns:
            out[col] = bte.set_index("rebalance")[col].reindex(out.index).ffill()

    out = out.dropna(how="any")  # ensure aligned
    # Summaries
    rows = []
    for col in out.columns:
        rows.append({"strategy": col.replace("equity_",""), **ann_stats(out[col], out.index[0], out.index[-1])})
    summ = pd.DataFrame(rows).sort_values("CAGR", ascending=False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT.parent / "equity_curves_aligned.csv")
    summ.to_csv(OUT, index=False)
    print(f"Saved curves -> {OUT.parent/'equity_curves_aligned.csv'}")
    print(f"Saved summary -> {OUT}")

if __name__ == "__main__":
    main()
