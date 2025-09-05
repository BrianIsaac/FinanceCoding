# scripts/sweep_confirm_outperformance.py
#!/usr/bin/env python3
"""
Grid-tune GAT-Direct, apply Top-K, ensemble with HRP, sweep transaction costs,
and report Sharpe with CIs/Deflated Sharpe + win-rates vs baselines.

Usage (defaults are modest to keep runtime reasonable):
  python scripts/sweep_confirm_outperformance.py

Example (narrower grid, different out dir, 2 TC levels, TopK set):
  python scripts/sweep_confirm_outperformance.py ^
      --out-base outputs/confirm ^
      --max-runs 12 ^
      --lrs 0.001,0.002 ^
      --wds 1e-4 ^
      --dropouts 0.1,0.2 ^
      --caps 0.02,0.03 ^
      --entropies 1e-3,5e-3 ^
      --l1s 0,2e-3 ^
      --topk 50,100 ^
      --tc-bps 5,10 ^
      --rebalance-stride 1

Notes
- Assumes your current config + data/graphs are already set up (as in recent runs).
- Trains with model.head=direct and evaluates with (optional) Top-K at inference.
- Reads HRP (and other baselines) daily returns written by train_gat.py for ensembling/comparison.
- “Rebalance stride” 2 means “use every 2nd snapshot” at evaluation (approx. lower rebal freq).
- Feature ablation (optional): pass --feature-mask like 11110111 to zero selected feature channels at EVAL ONLY.
"""

from __future__ import annotations
import argparse
import os  # noqa: F401
import sys
import subprocess
import json  # noqa: F401
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# safe allowlist for PyTorch 2.6+ when loading graphs
from torch.serialization import add_safe_globals  # noqa: E402
try:
    from torch_geometric.data.data import Data, DataEdgeAttr  # type: ignore
    add_safe_globals([Data, DataEdgeAttr])
except Exception:
    try:
        from torch_geometric.data import Data  # type: ignore
        add_safe_globals([Data])
    except Exception:
        pass

# import utilities from your training module
from src.train import (  # noqa: E402
    list_samples, split_samples, load_graph, load_label_vec,  # noqa: F401
    _eval_periods_daily_returns, _annualized_sharpe_from_series,  # noqa: F401
    _backtest_baseline, evaluate_daily_compound,  # for consistency # noqa: F401
)

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def compute_metrics_from_returns(r: pd.Series) -> Dict[str, float]:
    r = r.copy().dropna()
    if r.empty:
        return dict(CAGR=np.nan, AnnMean=np.nan, AnnVol=np.nan, Sharpe=np.nan, MDD=np.nan)
    eq = (1.0 + r).cumprod()
    ann = 252.0
    n = int(r.shape[0])
    cagr    = float(eq.iloc[-1] ** (ann / max(n, 1)) - 1.0)
    annmean = float(r.mean() * ann)
    annvol  = float(r.std(ddof=0) * math.sqrt(ann))
    sharpe  = float(annmean / annvol) if annvol > 0 else float("nan")
    mdd = float((eq / eq.cummax() - 1.0).min())
    return dict(CAGR=cagr, AnnMean=annmean, AnnVol=annvol, Sharpe=sharpe, MDD=mdd)

def block_bootstrap_sharpe_ci(r: pd.Series, alpha=0.05, B=2000, block_len=10, seed=123) -> Tuple[float,float]:
    """Simple moving block bootstrap CI for Sharpe (annualized)."""
    rng = np.random.default_rng(seed)
    r = r.dropna().values
    n = len(r)
    if n < 2:
        return (np.nan, np.nan)
    sh = []
    for _ in range(B):
        res = []
        while len(res) < n:
            start = rng.integers(0, max(1, n - block_len))
            end = min(n, start + block_len)
            res.extend(r[start:end].tolist())
        res = np.array(res[:n], float)
        ann = 252.0
        m = res.mean() * ann
        v = res.std(ddof=0) * math.sqrt(ann)
        sh.append(m / v if v > 1e-12 else np.nan)
    arr = np.array([x for x in sh if np.isfinite(x)])
    if len(arr) == 0:
        return (np.nan, np.nan)
    lo = float(np.quantile(arr, alpha/2))
    hi = float(np.quantile(arr, 1 - alpha/2))
    return (lo, hi)

def deflated_sharpe_ratio(r: pd.Series, sr_hat: Optional[float] = None, n_trials: int = 1) -> float:
    """
    Approximate Deflated Sharpe Ratio (Bailey & López de Prado, 2014).
    Uses Probabilistic Sharpe Ratio with a deflation threshold for multiple trials.
    Treat as a conservative *approximation*.

    DSR = Φ( (SR - SR_defl) * sqrt(T) / sqrt( 1 - γ3*SR + 0.5*(γ4-3)*SR^2 ) )
    where SR_defl = z_(1 - 1/n_trials) * sqrt( (1 - γ3*SR + 0.5*(γ4-3)*SR^2) / T )

    Returns the probability (0..1).
    """
    x = r.dropna().values
    T = len(x)
    if T < 3:
        return float("nan")
    if sr_hat is None:
        m = x.mean() * 252.0
        v = x.std(ddof=0) * math.sqrt(252.0)
        sr_hat = float(m / v) if v > 1e-12 else 0.0
    # sample skew (γ3) and Pearson kurtosis (γ4)
    gamma3 = float(pd.Series(x).skew())
    gamma4_fisher = float(pd.Series(x).kurt())  # Fisher (normal==0)
    gamma4 = gamma4_fisher + 3.0

    denom = math.sqrt(max(1e-12, 1 - gamma3 * sr_hat + 0.5 * (gamma4 - 3.0) * sr_hat**2))
    if n_trials < 1:
        n_trials = 1
    z = 0.0
    # N(0,1) quantile for 1 - 1/n_trials
    if n_trials > 1:
        from scipy.stats import norm
        z = float(norm.ppf(1.0 - 1.0 / n_trials))
    sr_defl = z * (denom / math.sqrt(T))
    from scipy.stats import norm
    dsr = float(norm.cdf((sr_hat - sr_defl) * math.sqrt(T) / denom))
    return dsr

# ------------------------------------------------------------
# Eval helpers (Top-K + optional feature ablation + stride)
# ------------------------------------------------------------

def _zero_features_inplace(g, mask_bits: Optional[str]):
    """Mask is an 8-char string of 1/0; 0 columns are zeroed (eval-only)."""
    if not mask_bits:
        return g
    mask_bits = mask_bits.strip()
    if len(mask_bits) != g.x.shape[1] or any(c not in "01" for c in mask_bits):
        return g
    keep = torch.tensor([c == "1" for c in mask_bits], dtype=torch.bool)
    g.x[:, ~keep] = 0.0
    return g

@torch.no_grad()
def eval_direct_with_topk_and_stride(
    ckpt_path: Path,
    split: Dict[str,str],
    returns_daily_path: Path,
    tc_bps: float,
    topk: Optional[int],
    rebalance_stride: int = 1,
    feature_mask: Optional[str] = None,
) -> pd.Series:
    """
    Load a trained Direct-head model and stitch daily returns across TEST windows,
    applying Top-K at inference and skipping snapshots by 'rebalance_stride'.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(str(ckpt_path), map_location=device)
    cfg_dict = payload.get("cfg", {})
    # build model the same way train.py did
    from src.model_gat import GATPortfolio
    model = GATPortfolio(
        in_dim=cfg_dict["model"]["in_dim"],
        hidden_dim=cfg_dict["model"]["hidden_dim"],
        heads=cfg_dict["model"]["heads"],
        num_layers=cfg_dict["model"]["num_layers"],
        dropout=cfg_dict["model"]["dropout"],
        residual=True,
        use_gatv2=cfg_dict["model"]["use_gatv2"],
        use_edge_attr=cfg_dict["model"]["use_edge_attr"],
        head="direct",
        mem_hidden=None,
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    graph_dir  = Path(cfg_dict["data"]["graph_dir"])
    labels_dir = Path(cfg_dict["data"]["labels_dir"])
    returns_daily: pd.DataFrame = pd.read_parquet(returns_daily_path).sort_index()

    # samples
    samples_all = list_samples(graph_dir, labels_dir)
    _, _, test_s_full = split_samples(samples_all, split["train_start"], split["val_start"], split["test_start"])
    # stride (1 = default)
    test_s = [s for i, s in enumerate(sorted(test_s_full, key=lambda s: s.ts)) if (i % max(1, rebalance_stride) == 0)]
    if not test_s:
        return pd.Series(dtype=float)

    dates: pd.DatetimeIndex = returns_daily.index  # type: ignore
    prev_w: Optional[pd.Series] = None
    chunks: List[pd.Series] = []

    for i, s in enumerate(test_s):
        # trading window bounds
        if i + 1 == len(test_s):
            end_trading = dates[-1]
        else:
            end_trading = dates[dates.searchsorted(test_s[i+1].ts, "left") - 1]
        start_idx = dates.searchsorted(s.ts, "right")
        if start_idx >= len(dates) or end_trading < dates[start_idx]:
            continue
        start_trading = dates[start_idx]

        g = load_graph(s.graph_path)
        g = _zero_features_inplace(g, feature_mask)
        tickers: List[str] = list(getattr(g, "tickers", []))
        x = g.x.to(device)
        eidx = g.edge_index.to(device)
        eattr = getattr(g, "edge_attr", None)
        eattr = eattr.to(device).to(x.dtype) if eattr is not None else None
        mask_all = torch.ones(len(tickers), dtype=torch.bool, device=device)

        # Direct head -> raw weights (assumed >=0 and sum ~1 by model)
        w_raw, _ = model(x, eidx, mask_all, eattr, None)

        # Top-K: keep K largest, renormalize
        if topk is not None and topk > 0 and topk < w_raw.numel():
            idx = torch.topk(w_raw, k=topk).indices
            w_top = torch.zeros_like(w_raw)
            w_top[idx] = w_raw[idx]
            ssum = float(w_top.sum().item())
            if ssum > 1e-12:
                w = w_top / ssum
            else:
                w = torch.full_like(w_raw, 1.0 / len(w_raw))
        else:
            w = w_raw

        curr_w = pd.Series(w.detach().cpu().numpy().astype(float), index=tickers)

        # transaction cost on transition
        if prev_w is None:
            tc = 0.0
        else:
            aligned = pd.concat([prev_w.rename("prev"), curr_w.rename("curr")], axis=1).fillna(0.0)
            tc = float((aligned["curr"] - aligned["prev"]).abs().sum()) * (tc_bps / 10000.0)

        window = returns_daily.loc[start_trading:end_trading]
        valid_mask = ~window.isna().all(axis=0)
        curr_w = curr_w[valid_mask.index[valid_mask]]
        ssum = float(curr_w.sum())
        if ssum > 0:
            curr_w = curr_w / ssum

        daily = window.reindex(columns=curr_w.index, fill_value=0.0).mul(curr_w, axis=1).sum(axis=1)
        if len(daily) > 0 and tc > 0:
            r0 = daily.iloc[0]
            daily.iloc[0] = (1.0 + r0) * (1.0 - tc) - 1.0

        chunks.append(daily)
        prev_w = curr_w

    if not chunks:
        return pd.Series(dtype=float)
    r_all = pd.concat(chunks).sort_index()
    r_all = r_all[~r_all.index.duplicated(keep="first")]
    return r_all.rename("r")

# ------------------------------------------------------------
# Runner
# ------------------------------------------------------------

@dataclass
class RunCfg:
    lr: float
    wd: float
    dropout: float
    cap: float
    ent: float
    l1: float
    tc_bps: int
    topk: Optional[int]
    stride: int
    feat_mask: Optional[str]

def run_train_once(out_dir: Path, cfg: RunCfg, base_cmd: List[str]) -> None:
    """Call scripts/train_gat.py with overrides for a Direct-head run."""
    args = [
        sys.executable, "scripts/train_gat.py",   # use current interpreter/venv
        "roll.enabled=false",
        f"train.out_dir={str(out_dir)}",
        "model.head=direct",
        "train.epochs=12",
        f"train.lr={cfg.lr}",
        f"train.weight_decay={cfg.wd}",
        f"model.dropout={cfg.dropout}",
        f"model.weight_cap={cfg.cap}",
        f"loss.entropy_coef={cfg.ent}",
        f"loss.l1_coef={cfg.l1}",
        f"loss.turnover_bps={cfg.tc_bps}",
    ]
    # allow user pass-through
    args.extend(base_cmd)
    print("[train] ", " ".join(args))
    subprocess.run(args, check=True)

def main():
    from itertools import product

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-base", type=str, default="outputs/confirm")
    ap.add_argument("--max-runs", type=int, default=12)

    ap.add_argument("--lrs", type=str, default="0.001,0.002")
    ap.add_argument("--wds", type=str, default="1e-4,5e-5")
    ap.add_argument("--dropouts", type=str, default="0.1,0.2")
    ap.add_argument("--caps", type=str, default="0.02,0.03")
    ap.add_argument("--entropies", type=str, default="1e-3,5e-3")
    ap.add_argument("--l1s", type=str, default="0,2e-3")

    ap.add_argument("--topk", type=str, default="50,100")   # inference Top-K; can be "" for none
    ap.add_argument("--tc-bps", type=str, default="5,10,25")
    ap.add_argument("--rebalance-stride", type=int, default=1)
    ap.add_argument("--feature-mask", type=str, default=None, help="8-char 0/1 string; 0 columns zeroed at EVAL ONLY")
    ap.add_argument("--shuffle-seed", type=int, default=123, help="seed for shuffling the hyperparameter grid")

    # pass-through overrides to train_gat.py (e.g., different split.* or train.seed)
    ap.add_argument("--train-overrides", type=str, default="", help="extra hydra overrides, space-separated")

    args = ap.parse_args()
    out_base = Path(args.out_base)
    out_base.mkdir(parents=True, exist_ok=True)
    passthru = args.train_overrides.strip().split() if args.train_overrides.strip() else []

    # parse grids
    def _floats(s): return [float(x) for x in str(s).split(",") if x != ""]
    def _ints(s):   return [int(float(x)) for x in str(s).split(",") if x != ""]
    lrs   = _floats(args.lrs)
    wds   = _floats(args.wds)
    drops = _floats(args.dropouts)
    caps  = _floats(args.caps)
    ents  = _floats(args.entropies)
    l1s   = _floats(args.l1s)
    tcs   = _ints(args.tc_bps)
    topks = [int(x) for x in str(args.topk).split(",") if x.strip().isdigit()] if (args.topk and args.topk.strip()) else [None]

    # build and shuffle the full grid, then sample first max-runs for balanced coverage
    grid = list(product(tcs, lrs, wds, drops, caps, ents, l1s, topks))
    rng = np.random.default_rng(args.shuffle_seed)
    rng.shuffle(grid)

    leaderboard = []
    run_idx = 0

    for tc, lr, wd, dr, cap, ent, l1, k in grid:
        if run_idx >= args.max_runs:
            break
        run_idx += 1
        tag = f"tc{tc}_lr{lr}_wd{wd}_do{dr}_cap{cap}_ent{ent}_l1{l1}_k{(k if k else 'all')}_s{args.rebalance_stride}"
        out_dir = out_base / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        rcfg = RunCfg(
            lr=lr, wd=wd, dropout=dr, cap=cap, ent=ent, l1=l1,
            tc_bps=tc, topk=k, stride=args.rebalance_stride,
            feat_mask=args.feature_mask
        )

        # persist the run recipe for reproducibility
        with open(out_dir / "run_cfg.json", "w") as fh:
            json.dump({
                "tag": tag,
                "lr": lr, "wd": wd, "dropout": dr, "cap": cap, "entropy": ent, "l1": l1,
                "tc_bps": tc, "topk": (k if k else 0), "stride": args.rebalance_stride,
                "feature_mask": args.feature_mask,
                "train_overrides": passthru,
            }, fh, indent=2)

        # 1) Train
        try:
            run_train_once(out_dir, rcfg, passthru)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] training failed for {tag}: {e}")
            leaderboard.append({
                "tag": tag, "tc_bps": tc, "topk": (k if k else 0),
                "lr": lr, "wd": wd, "do": dr, "cap": cap, "ent": ent, "l1": l1,
                "Sharpe_GAT": np.nan, "Sharpe_Blend": np.nan, "Sharpe_HRP": np.nan,
                "DSR_GAT": np.nan, "CAGR_GAT": np.nan, "MDD_GAT": np.nan,
            })
            continue

        # 2) Gather paths & cfg for eval
        ckpt = out_dir / "gat_best.pt"
        if not ckpt.exists():
            print(f"[WARN] checkpoint missing for {tag}; skipping eval.")
            continue
        payload = torch.load(str(ckpt), map_location="cpu")
        cfg = payload["cfg"]
        rets_path = Path(cfg["data"]["returns_daily"])
        # split dates (single-window path used by train_gat.py)
        split = dict(
            train_start=cfg["split"]["train_start"],
            val_start=cfg["split"]["val_start"],
            test_start=cfg["split"]["test_start"],
        )

        # 3) Evaluate Direct with Top-K + stride (this recomputes returns)
        r_gat = eval_direct_with_topk_and_stride(
            ckpt_path=ckpt,
            split=split,
            returns_daily_path=rets_path,
            tc_bps=tc,
            topk=k,
            rebalance_stride=rcfg.stride,
            feature_mask=rcfg.feat_mask,
        )

        # Save our eval series
        if not r_gat.empty:
            r_gat.to_frame().to_csv(out_dir / f"gat_direct_topk{(k if k else 0)}_daily_returns.csv")

        # 4) Read baselines daily (already written by train_gat.py)
        baselines: Dict[str, pd.Series] = {}
        for strat in ["EW","HRP","MinVar","MV"]:
            p = out_dir / f"{strat.lower()}_daily_returns.csv"
            if p.exists():
                baselines[strat] = pd.read_csv(p, parse_dates=[0], index_col=0).iloc[:,0].rename(strat)

        # 5) Ensemble with HRP (returns-level blend)
        r_hrp = baselines.get("HRP", pd.Series(dtype=float))
        if not r_gat.empty and not r_hrp.empty:
            idx = r_hrp.index.union(r_gat.index)
            blend = (r_gat.reindex(idx, fill_value=0) + r_hrp.reindex(idx, fill_value=0)) / 2.0
            blend.rename("blend").to_frame().to_csv(out_dir / "blend_hrp_daily_returns.csv")
        else:
            blend = pd.Series(dtype=float)

        # 6) Metrics + CIs + DSR
        rows = []
        for name, series in [("GAT-Direct", r_gat), ("HRP", r_hrp), ("Blend", blend)]:
            if series.empty:
                rows.append({"strategy": name})
                continue
            m = compute_metrics_from_returns(series)
            lo, hi = block_bootstrap_sharpe_ci(series, alpha=0.05, B=1500, block_len=10, seed=123)
            dsr = deflated_sharpe_ratio(series, sr_hat=m["Sharpe"], n_trials=max(1, args.max_runs))
            rows.append({
                "strategy": name,
                **m,
                "Sharpe_CI95_lo": lo,
                "Sharpe_CI95_hi": hi,
                "DeflatedSharpeProb": dsr,
            })

        # 7) Win-rates vs baselines (per-day)
        if not r_gat.empty:
            for bname, bser in baselines.items():
                idx = r_gat.index.intersection(bser.index)
                if len(idx) > 0:
                    wr = float((r_gat.loc[idx] > bser.loc[idx]).mean())
                else:
                    wr = float("nan")
                rows.append({"strategy": f"WinRate_vs_{bname}", "Sharpe": wr})

        df_out = pd.DataFrame(rows)
        df_out.to_csv(out_dir / "summary_metrics.csv", index=False)

        # 8) Leaderboard row
        gat_row = next((r for r in rows if r.get("strategy")=="GAT-Direct" and "Sharpe" in r), None)
        blend_row = next((r for r in rows if r.get("strategy")=="Blend" and "Sharpe" in r), None)
        hrp_row = next((r for r in rows if r.get("strategy")=="HRP" and "Sharpe" in r), None)
        leaderboard.append({
            "tag": tag,
            "tc_bps": tc,
            "topk": (k if k else 0),
            "lr": lr, "wd": wd, "do": dr, "cap": cap, "ent": ent, "l1": l1,
            "Sharpe_GAT": (gat_row or {}).get("Sharpe", np.nan),
            "Sharpe_Blend": (blend_row or {}).get("Sharpe", np.nan),
            "Sharpe_HRP": (hrp_row or {}).get("Sharpe", np.nan),
            "DSR_GAT": (gat_row or {}).get("DeflatedSharpeProb", np.nan),
            "CAGR_GAT": (gat_row or {}).get("CAGR", np.nan),
            "MDD_GAT": (gat_row or {}).get("MDD", np.nan),
        })

    # all done
    board = pd.DataFrame(leaderboard).sort_values(["Sharpe_Blend","Sharpe_GAT"], ascending=False)
    board.to_csv(out_base / "leaderboard.csv", index=False)
    print("\n=== Leaderboard (top 12 by Blend Sharpe then GAT Sharpe) ===")
    if not board.empty:
        print(board.head(12).to_string(index=False))
    else:
        print("(no successful runs)")

if __name__ == "__main__":
    main()
