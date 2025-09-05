#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run baseline portfolio strategies aligned to the GAT evaluation windows.

Outputs per strategy:
  - <strat>_daily_returns.csv  (indexed by trading day)
  - <strat>_equity_daily.csv   (cumulative equity curve)
  - <strat>_equity.csv         (per-rebalance window log incl. turnover)

Strategies:
  - EW, MV, MinVar, HRP, TopK_EW

Covariance:
  - Uses src.cov.build_cov_estimator: ledoit_wolf | oas | linear | diag
"""

from __future__ import annotations
import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Optional SciPy (HRP); falls back gracefully
try:
    from scipy.cluster.hierarchy import linkage, leaves_list  # type: ignore
    from scipy.spatial.distance import squareform  # type: ignore
except Exception:
    linkage = leaves_list = squareform = None

# --- our covariance factory
try:
    from src.cov import build_cov_estimator  # Follows the earlier refactor
except Exception:
    build_cov_estimator = None  # type: ignore

# ---------------- IO + windowing ----------------

@dataclass
class Sample:
    ts: pd.Timestamp
    graph_path: Path
    label_path: Path

def _infer_ts(p: Path) -> pd.Timestamp:
    import re
    m = re.search(r"(\d{4}-\d{2}-\d{2})", p.name)
    if not m:
        raise ValueError(f"Cannot parse date from {p.name}")
    return pd.Timestamp(m.group(1))

def list_samples(graph_dir: Path, labels_dir: Path) -> List[Sample]:
    gfiles = sorted([p for p in graph_dir.glob("graph_*.pt")], key=lambda x: _infer_ts(x))
    out: List[Sample] = []
    for gp in gfiles:
        ts = _infer_ts(gp)
        lp = labels_dir / f"labels_{ts.date()}.parquet"
        if lp.exists():
            out.append(Sample(ts=ts, graph_path=gp, label_path=lp))
    return out

def split_samples(samples: List[Sample], train_start: str, val_start: str, test_start: str
                  ) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    t0 = pd.Timestamp(train_start)
    v0 = pd.Timestamp(val_start)
    te0 = pd.Timestamp(test_start)
    train = [s for s in samples if t0 <= s.ts < v0]
    val   = [s for s in samples if v0 <= s.ts < te0]
    test  = [s for s in samples if s.ts >= te0]
    return train, val, test

def load_graph(path: Path):
    # torch geometric Data saved via torch.save
    return torch.load(str(path), map_location="cpu", weights_only=False)

# ----------------- math utils -----------------

def _project_capped_simplex(v: torch.Tensor, cap: float, s: float = 1.0, iters: int = 50) -> torch.Tensor:
    """
    Project v onto { w: 0<=w<=cap, sum w = s } using bisection in tau:
        w_i = clip(v_i - tau, 0, cap)
    """
    vmax = v.detach().max().item()
    hi = vmax
    lo = hi - cap
    for _ in range(64):
        if torch.clamp(v - lo, 0.0, cap).sum().item() >= s:
            break
        lo -= cap
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        w = torch.clamp(v - mid, 0.0, cap)
        if w.sum().item() > s:
            lo = mid
        else:
            hi = mid
    return torch.clamp(v - hi, 0.0, cap)

def _auto_feasible_cap(cap: float, n_eff: int) -> float:
    # Ensure feasibility: need n_eff * cap >= 1
    if n_eff <= 0:
        return cap
    min_cap = 1.0 / n_eff + 1e-9
    return cap if cap >= min_cap else min_cap

def _baseline_weight_ew(n: int, cap: float) -> np.ndarray:
    w0 = np.full(n, 1.0 / max(n, 1), dtype=np.float32)
    w = _project_capped_simplex(torch.from_numpy(w0), cap=_auto_feasible_cap(cap, n), s=1.0, iters=50)
    return w.detach().cpu().numpy()

def _ivp_weights(S: np.ndarray) -> np.ndarray:
    v = np.clip(np.diag(S), 1e-12, None)
    w = 1.0 / v
    w /= w.sum()
    return w

def _corr_from_cov(S: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    invd = np.where(d > 0, 1.0 / d, 0.0)
    return (S * invd).T * invd

def _hrp_weights(S: np.ndarray) -> np.ndarray:
    n = S.shape[0]
    if linkage is None or leaves_list is None or squareform is None or n < 2:
        return _ivp_weights(S)

    C = np.clip(_corr_from_cov(S), -0.9999, 0.9999)
    D = np.sqrt(0.5 * (1.0 - C))
    z = linkage(squareform(D, checks=False), method="single")
    order = leaves_list(z)
    S_ = S[np.ix_(order, order)]

    w = np.ones(n)
    clusters = [np.arange(n)]
    while clusters:
        cl = clusters.pop()
        if len(cl) <= 1:
            continue
        split = int(math.ceil(len(cl) / 2))
        c1, c2 = cl[:split], cl[split:]

        def _cluster_var(idxs):
            Sc = S_[np.ix_(idxs, idxs)]
            wc = _ivp_weights(Sc)
            return float(wc @ Sc @ wc)

        v1, v2 = _cluster_var(c1), _cluster_var(c2)
        a2 = v1 / (v1 + v2 + 1e-12)
        a1 = 1.0 - a2
        w[c1] *= a1
        w[c2] *= a2
        clusters.extend([c1, c2])

    w_ord = np.zeros(n)
    w_ord[order] = w
    s = w_ord.sum()
    return (w_ord / s) if s > 0 else _ivp_weights(S)

def _project_cap_and_sum_to_one(w_np: np.ndarray, cap: float) -> np.ndarray:
    t = torch.from_numpy(w_np.astype(np.float32))
    w = _project_capped_simplex(t, cap=_auto_feasible_cap(cap, len(w_np)), s=1.0, iters=50)
    return w.detach().cpu().numpy()

# --- PGD Markowitz (diag/full SPD) ---

def _pgd_markowitz(mu: np.ndarray, S: np.ndarray, cap: float, gamma: float, mode: str = "diag",
                   steps: int = 60) -> np.ndarray:
    """
    minimize  gamma * w' S w - mu' w
    s.t.      0 <= w <= cap, sum w = 1
    """
    mu_t = torch.from_numpy(mu.astype(np.float32))
    N = mu_t.numel()
    cap_eff = _auto_feasible_cap(cap, int(N))
    w = torch.full_like(mu_t, 1.0 / max(int(N), 1))

    if mode == "diag":
        sdiag = np.clip(np.diag(S), 1e-10, None)
        S_diag = torch.from_numpy(sdiag).to(dtype=mu_t.dtype)
        L = (2.0 * gamma * S_diag.max()).clamp_min(1e-8)
        eta = 1.0 / float(L)
        for _ in range(steps):
            grad = 2.0 * gamma * (S_diag * w) - mu_t
            w = _project_capped_simplex(w - eta * grad, cap_eff, s=1.0)
        return w.detach().cpu().numpy()

    # full SPD
    St = torch.from_numpy(S.astype(np.float32))
    v = torch.randn_like(mu_t)
    for _ in range(8):
        v = St @ v
        v = v / (v.norm() + 1e-12)
    L = (2.0 * gamma * (v @ (St @ v))).clamp_min(1e-8)
    eta = 1.0 / float(L)
    for _ in range(steps):
        grad = 2.0 * gamma * (St @ w) - mu_t
        w = _project_capped_simplex(w - eta * grad, cap_eff, s=1.0)
    return w.detach().cpu().numpy()

# ----------------- scoring for TopK_EW -----------------

def _momentum_score(hist: pd.DataFrame, lookback: int = 126, method: str = "mean") -> pd.Series:
    """
    Simple price momentum on returns:
      method="mean": average daily return over lookback
      method="prod": cumulative (1+r).prod()-1 over lookback
    """
    h = hist.tail(lookback).fillna(0.0)
    if h.empty:
        return pd.Series(dtype=float)
    if method == "prod":
        return (1.0 + h).prod(axis=0) - 1.0
    return h.mean(axis=0)

def _risk_scale(vol: pd.Series) -> pd.Series:
    v = vol.replace(0.0, np.nan)
    w = 1.0 / v
    w = w.fillna(0.0)
    s = w.sum()
    return w / s if s > 0 else pd.Series(np.zeros_like(w), index=w.index)

# ----------------- per-window engine -----------------

def _compute_window_weights(
    strategy: str,
    tickers: List[str],
    hist: pd.DataFrame,
    cap: float,
    mv_gamma: float,
    cov_method: str,
    cov_kwargs: Dict,
    mode: str,
    topk: Optional[int],
    topk_lookback: int,
    topk_method: str,
    topk_risk_scale: bool,
) -> np.ndarray:

    X = hist.reindex(columns=tickers, fill_value=0.0).values
    n = len(tickers)

    if strategy == "EW":
        return _baseline_weight_ew(n, cap)

    # Covariance for MV/MinVar/HRP and for TopK_EW risk-scaling
    if build_cov_estimator is None:
        # fallback: simple linear shrinkage if src.cov is unavailable
        def _linear_shrink(S, alpha=0.1, ridge=1e-6):
            D = np.diag(np.diag(S))
            return (1.0 - alpha) * S + alpha * D + ridge * np.eye(S.shape[0], dtype=S.dtype)
        S = np.cov(X, rowvar=False)
        S = _linear_shrink(S, alpha=float(cov_kwargs.get("alpha", 0.1)), ridge=float(cov_kwargs.get("ridge", 1e-6)))
    else:
        est = build_cov_estimator(
            method=cov_method,
            alpha=float(cov_kwargs.get("alpha", 0.10)),
            ridge=float(cov_kwargs.get("ridge", 1e-6)),
            lw_shrink=float(cov_kwargs.get("lw_shrink", 0.0)),
        )
        S = est.fit(X).get()

    if strategy == "HRP":
        w = _hrp_weights(S)
        return _project_cap_and_sum_to_one(w, cap)

    if strategy == "MinVar":
        mu0 = np.zeros(n, dtype=np.float32)
        w = _pgd_markowitz(mu0, S, cap=cap, gamma=mv_gamma, mode=mode, steps=60)
        return w

    if strategy == "MV":
        mu = X.mean(axis=0).astype(np.float32)
        # optional top-k on mean returns
        if topk is not None and 0 < topk < n:
            idx = np.argsort(mu)[-topk:]
            S_sub = S[np.ix_(idx, idx)]
            w_sub = _pgd_markowitz(mu[idx], S_sub, cap=_auto_feasible_cap(cap, len(idx)), gamma=mv_gamma, mode=mode, steps=60)
            w = np.zeros(n, dtype=np.float32)
            w[idx] = w_sub
            return w
        return _pgd_markowitz(mu, S, cap=cap, gamma=mv_gamma, mode=mode, steps=60)

    if strategy == "TopK_EW":
        scores = _momentum_score(hist, lookback=topk_lookback, method=topk_method)
        scores = scores.reindex(tickers).fillna(-np.inf)
        k = int(topk or max(1, n // 10))
        idx = np.argsort(scores.values)[-k:]
        chosen = [tickers[i] for i in idx]

        if topk_risk_scale:
            # scale by 1/vol among the selected names, then project to cap and sum=1
            sel = hist.reindex(columns=chosen).fillna(0.0)
            vol = sel.std(ddof=0)
            w_series = _risk_scale(vol)
            w = np.zeros(n, dtype=np.float32)
            w[idx] = w_series.reindex(chosen).fillna(0.0).values.astype(np.float32)
            return _project_cap_and_sum_to_one(w, cap)
        # equal-weight among selected, then project
        w = np.zeros(n, dtype=np.float32)
        w[idx] = 1.0 / k
        return _project_cap_and_sum_to_one(w, cap)

    raise ValueError(f"Unknown strategy: {strategy}")

# ----------------- backtest loop -----------------

def run_baseline_for_samples(
    strategy: str,
    samples: List[Sample],
    returns_daily: pd.DataFrame,
    tc_decimal: float,
    cap: float,
    mv_gamma: float,
    cov_method: str,
    cov_kwargs: Dict,
    mode: str,
    topk: Optional[int],
    topk_lookback: int,
    topk_method: str,
    topk_risk_scale: bool,
    out_dir: Optional[Path] = None,
) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
    dates: pd.DatetimeIndex = returns_daily.index
    prev_w: Optional[pd.Series] = None
    daily_chunks: List[pd.Series] = []
    rows: List[dict] = []
    equity = 1.0

    for i, s in enumerate(samples):
        # trading window: (first day > s.ts) ... (day before next snapshot, or last date)
        end_trading = dates[-1] if i + 1 == len(samples) else dates[dates.searchsorted(samples[i + 1].ts, "left") - 1]
        start_idx = dates.searchsorted(s.ts, "right")
        if start_idx >= len(dates) or end_trading < dates[start_idx]:
            continue
        start_trading = dates[start_idx]

        g = load_graph(s.graph_path)
        tickers: List[str] = list(getattr(g, "tickers", []))

        hist = returns_daily.loc[:start_trading].iloc[:-1].tail(252).fillna(0.0)
        if hist.shape[1] < 2 or hist.shape[0] < 2:
            continue

        w_np = _compute_window_weights(
            strategy=strategy,
            tickers=tickers,
            hist=hist,
            cap=cap,
            mv_gamma=mv_gamma,
            cov_method=cov_method,
            cov_kwargs=cov_kwargs,
            mode=mode,
            topk=topk,
            topk_lookback=topk_lookback,
            topk_method=topk_method,
            topk_risk_scale=topk_risk_scale,
        )
        curr_w = pd.Series(w_np.astype(float), index=tickers)

        # turnover cost vs previous portfolio
        if prev_w is None:
            tc = 0.0
        else:
            aligned = pd.concat([prev_w.rename("prev"), curr_w.rename("curr")], axis=1).fillna(0.0)
            tc = float((aligned["curr"] - aligned["prev"]).abs().sum()) * tc_decimal

        win = returns_daily.loc[start_trading:end_trading]
        valid_mask = ~win.isna().all(axis=0)
        curr_w = curr_w[valid_mask.index[valid_mask]]
        ssum = float(curr_w.sum())
        if ssum > 0:
            curr_w = curr_w / ssum

        r_p = win.reindex(columns=curr_w.index, fill_value=0.0).mul(curr_w, axis=1).sum(axis=1)
        if len(r_p) > 0 and tc > 0:
            r0 = r_p.iloc[0]
            r_p.iloc[0] = (1.0 + r0) * (1.0 - tc) - 1.0

        daily_chunks.append(r_p)

        period_gross = float((1.0 + r_p).prod())
        equity *= period_gross
        rows.append({
            "rebalance": s.ts.date(),
            "start_trading": start_trading.date(),
            "end_trading": end_trading.date(),
            "n_nodes": len(tickers),
            "turnover_cost": tc,
            "period_gross": period_gross,
            f"equity_{strategy.lower()}": equity,
        })
        prev_w = curr_w

    r_all = None
    log_df = None
    if daily_chunks:
        r_all = pd.concat(daily_chunks).sort_index()
        r_all = r_all[~r_all.index.duplicated(keep="first")]
        log_df = pd.DataFrame(rows)

        if out_dir is not None:
            nm = strategy.lower()
            r_all.rename("r").to_frame().to_csv(out_dir / f"{nm}_daily_returns.csv")
            (1.0 + r_all).cumprod().rename("equity").to_frame().to_csv(out_dir / f"{nm}_equity_daily.csv")
            log_df.to_csv(out_dir / f"{nm}_equity.csv", index=False)

    return (r_all if r_all is not None else pd.Series(dtype=float), log_df)

# ----------------- metrics -----------------

def _metrics_from_returns(r: pd.Series) -> Dict[str, float]:
    r = r.dropna()
    if r.empty:
        return {"CAGR": np.nan, "AnnMean": np.nan, "AnnVol": np.nan, "Sharpe": np.nan, "MDD": np.nan}
    eq = (1.0 + r).cumprod()
    ann = 252.0
    n = max(int(r.shape[0]), 1)
    cagr    = float(eq.iloc[-1] ** (ann / n) - 1.0)
    annmean = float(r.mean() * ann)
    annvol  = float(r.std(ddof=0) * math.sqrt(ann))
    sharpe  = float(annmean / annvol) if annvol > 0 else float("nan")
    dd = eq / eq.cummax() - 1.0
    mdd = float(dd.min())
    return {"CAGR": cagr, "AnnMean": annmean, "AnnVol": annvol, "Sharpe": sharpe, "MDD": mdd}

# ----------------- CLI -----------------

def main():
    ap = argparse.ArgumentParser(description="Run baseline strategies aligned with GAT evaluation windows.")
    ap.add_argument("--graph_dir", type=str, required=True)
    ap.add_argument("--labels_dir", type=str, required=True)
    ap.add_argument("--returns_daily", type=str, required=True)
    ap.add_argument("--train_start", type=str, required=True)
    ap.add_argument("--val_start", type=str, required=True)
    ap.add_argument("--test_start", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--strategies", nargs="+", default=["EW", "MV", "HRP", "MinVar", "TopK_EW"])
    ap.add_argument("--cap", type=float, default=0.02, help="Per-name weight cap.")
    ap.add_argument("--gamma", type=float, default=3.0, help="Risk aversion for MV/MinVar.")
    ap.add_argument("--mode", type=str, default="diag", choices=["diag", "chol"], help="Quadratic model type for PGD.")
    ap.add_argument("--topk", type=int, default=0, help="Top-K selection for MV (0 = disabled).")

    ap.add_argument("--turnover_bps", type=float, default=10.0)

    # Covariance options (passed to src.cov)
    ap.add_argument("--cov_method", type=str, default="ledoit_wolf",
                    choices=["ledoit_wolf", "oas", "linear", "diag"])
    ap.add_argument("--cov_alpha", type=float, default=0.10, help="Linear shrink alpha (if method=linear).")
    ap.add_argument("--cov_ridge", type=float, default=1e-6, help="Ridge epsilon added to Σ.")
    ap.add_argument("--cov_lw_shrink", type=float, default=0.0, help="Extra shrink toward diag for LW/OAS (0..1).")

    # TopK_EW scoring options
    ap.add_argument("--topk_ew_k", type=int, default=40, help="K for TopK_EW.")
    ap.add_argument("--topk_lookback", type=int, default=126, help="Lookback days for momentum scoring.")
    ap.add_argument("--topk_method", type=str, default="mean", choices=["mean", "prod"],
                    help="Momentum metric: mean daily return vs cumulative product.")
    ap.add_argument("--topk_risk_scale", action="store_true",
                    help="If set, 1/vol scaling inside TopK_EW before projection.")

    args = ap.parse_args()

    graph_dir = Path(args.graph_dir)
    labels_dir = Path(args.labels_dir)
    rets_path = Path(args.returns_daily)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load universe & windows
    samples_all = list_samples(graph_dir, labels_dir)
    _, _, test_s = split_samples(samples_all, args.train_start, args.val_start, args.test_start)
    print(f"[Data] test windows = {len(test_s)}")

    # Load daily returns
    returns_daily = pd.read_parquet(rets_path).sort_index()
    tc = float(args.turnover_bps) / 10000.0

    cov_kwargs = {"alpha": args.cov_alpha, "ridge": args.cov_ridge, "lw_shrink": args.cov_lw_shrink}

    for strat in args.strategies:
        r, _ = run_baseline_for_samples(
            strategy=strat,
            samples=test_s,
            returns_daily=returns_daily,
            tc_decimal=tc,
            cap=float(args.cap),
            mv_gamma=float(args.gamma),
            cov_method=str(args.cov_method),
            cov_kwargs=cov_kwargs,
            mode=str(args.mode),
            topk=(int(args.topk) if args.topk > 0 else None) if strat == "MV" else None,
            topk_lookback=int(args.topk_lookback),
            topk_method=str(args.topk_method),
            topk_risk_scale=bool(args.topk_risk_scale),
            out_dir=out_dir,
        )
        if r is None or r.empty:
            print(f"[{strat}] no returns generated.")
            continue
        m = _metrics_from_returns(r)
        print(f"[{strat}] {m}")

    print(f"[OK] Baseline files written to {out_dir}")

if __name__ == "__main__":
    main()
