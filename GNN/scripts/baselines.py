#!/usr/bin/env python3
"""
Quarterly portfolio baselines for the SP400 universe.

Strategies:
  • EW  : Equal-Weight with per-name cap + turnover costs
  • MinVar : Heuristic long-only min-variance (projected GD) + cap + costs
  • HRP : Hierarchical Risk Parity (corr-distance clustering + recursive bisection)
  • MV  : Mean-Variance via cvxpy (long-only, sum=1, cap)

Inputs:
  processed/returns_daily.parquet                     (Date x Tickers, arithmetic returns)
  processed/rebalance_dates.csv                       (column: rebalance_date)
  processed/labels/labels_YYYY-MM-DD.parquet          (column: r_next) — used to ensure tradable pool
  data/processed/universe_membership_wiki_sp400_seeded.csv

Output:
  processed/baselines/results.csv  (equity and stats per rebalance)

Run:
  python scripts/baselines.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Optional libs for HRP + MV
try:
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

try:
    import cvxpy as cp  # type: ignore
    HAVE_CVXPY = True
except Exception:
    HAVE_CVXPY = False


# ---------- Paths ---------- #

PROCESSED = Path("processed")
DATA = Path("data/processed")
RET = PROCESSED / "returns_daily.parquet"
REB = PROCESSED / "rebalance_dates.csv"
LAB_DIR = PROCESSED / "labels"
MEM = DATA / "universe_membership_wiki_sp400_seeded.csv"
OUT = PROCESSED / "baselines"


# ---------- Hyperparameters ---------- #

W_CAP: float = 0.02        # per-name cap (e.g., 2%)
TC_BP: float = 10.0        # transaction costs (per side, in bps)
TC: float = TC_BP / 10000.0
COV_LOOKBACK_DAYS: int = 252
MU_LOOKBACK_DAYS: int = 63
RIDGE_EPS: float = 1e-6     # small PSD ridge for covariances
MV_GAMMA: float = 5.0       # risk aversion for MV objective: mu'w - gamma w'Sw


# ---------- Helpers ---------- #

def active_at(mem: pd.DataFrame, ts: pd.Timestamp) -> List[str]:
    """Return active SP400 tickers at timestamp ts."""
    m = mem.copy()
    m["ticker"] = m["ticker"].astype(str).str.upper()
    m["start"] = pd.to_datetime(m["start"], errors="coerce")
    m["end"] = pd.to_datetime(m.get("end"), errors="coerce")
    mask = (m["start"] <= ts) & (m["end"].isna() | (m["end"] >= ts))
    return sorted(set(m.loc[mask, "ticker"].tolist()))


def equal_weight(tickers: List[str]) -> pd.Series:
    """Equal-weight portfolio with cap and renorm."""
    if not tickers:
        return pd.Series(dtype=float)
    w_raw = 1.0 / len(tickers)
    w_capped = min(w_raw, W_CAP)
    w = pd.Series(w_capped, index=tickers, dtype=float)
    s = w.sum()
    w = w / s if s > 0 else w
    return w


def realized_stats(
    rets: pd.DataFrame,
    cov_end_inclusive: pd.Timestamp,
    cov_lookback: int = COV_LOOKBACK_DAYS,
    mu_lookback: int = MU_LOOKBACK_DAYS,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Compute trailing mean vector (mu) and covariance (S)."""
    win_cov = rets.loc[:cov_end_inclusive].tail(cov_lookback).fillna(0.0)
    win_mu  = rets.loc[:cov_end_inclusive].tail(mu_lookback)  # leave NaNs -> mean ignores
    tickers = list(win_cov.columns)

    # Mean of arithmetic returns (per day)
    mu = win_mu.mean(axis=0).fillna(0.0).values

    # Covariance with Ledoit-Wolf shrinkage if available, else sample
    X = win_cov.values
    try:
        from sklearn.covariance import LedoitWolf  # type: ignore
        lw = LedoitWolf().fit(X)
        S = lw.covariance_
    except Exception:
        S = np.cov(X, rowvar=False)
        # force PSD
        S = (S + S.T) / 2.0 + RIDGE_EPS * np.eye(S.shape[0], dtype=float)

    # Stabilize
    S = S + RIDGE_EPS * np.eye(S.shape[0], dtype=float)
    return mu, S, tickers


def min_var_cov(cov: np.ndarray, tickers: List[str], max_iters: int = 200, step: float = 0.01) -> pd.Series:
    """Heuristic long-only MinVar with cap and renorm (projected GD)."""
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)
    S = cov
    w = np.full(n, 1.0 / n, dtype=float)
    for _ in range(max_iters):
        grad = 2.0 * (S @ w)
        w -= step * grad
        w = np.clip(w, 0.0, W_CAP)
        s = w.sum()
        w = (w / s) if s > 0 else np.full(n, 1.0 / n, dtype=float)
    return pd.Series(w, index=tickers, dtype=float)


# ---------- HRP ---------- #

def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(cov), RIDGE_EPS, None))
    corr = cov / np.outer(d, d)
    corr[np.isnan(corr)] = 0.0
    corr = np.clip(corr, -1.0, 1.0)
    return corr

def _corr_distance(corr: np.ndarray) -> np.ndarray:
    # López de Prado’s distance: d_ij = sqrt(0.5 * (1 - corr_ij))
    return np.sqrt(0.5 * (1.0 - corr))

def _cluster_order(corr: np.ndarray) -> np.ndarray:
    if not HAVE_SCIPY:
        # Fallback: identity order
        return np.arange(corr.shape[0])
    dist = _corr_distance(corr)
    # Convert to condensed form for linkage
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="single")  # 'single' works well for HRP
    order = leaves_list(Z)
    return order

def _cluster_var(cov: np.ndarray, idx: np.ndarray) -> float:
    sub = cov[np.ix_(idx, idx)]
    inv_diag = 1.0 / np.clip(np.diag(sub), RIDGE_EPS, None)
    ivp = inv_diag / inv_diag.sum()
    var = float(ivp @ sub @ ivp)
    return var

def hrp_weights(cov: np.ndarray, tickers: List[str]) -> pd.Series:
    """Hierarchical Risk Parity weights (long-only, no explicit cap)."""
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)
    corr = _cov_to_corr(cov)
    order = _cluster_order(corr)
    cov_sorted = cov[np.ix_(order, order)]

    # Recursive bisection
    w = np.ones(n, dtype=float)
    clusters = [np.arange(n)]
    while clusters:
        cl = clusters.pop(0)
        if len(cl) <= 1:
            continue
        split = len(cl) // 2
        left, right = cl[:split], cl[split:]
        var_l = _cluster_var(cov_sorted, left)
        var_r = _cluster_var(cov_sorted, right)
        alpha_l = 1.0 - var_l / (var_l + var_r)
        alpha_r = 1.0 - alpha_l
        w[left] *= alpha_l
        w[right] *= alpha_r
        clusters.extend([left, right])

    # Map back to original order
    w_full = np.zeros(n, dtype=float)
    w_full[order] = w
    # Apply per-name cap then renormalize
    w_full = np.clip(w_full, 0.0, W_CAP)
    s = w_full.sum()
    if s > 0:
        w_full /= s
    else:
        w_full[:] = 1.0 / n
    return pd.Series(w_full, index=tickers, dtype=float)


# ---------- Mean–Variance via cvxpy ---------- #

def mv_cvxpy_weights(mu: np.ndarray, cov: np.ndarray, tickers: List[str],
                     gamma: float = MV_GAMMA) -> pd.Series:
    """Mean–Variance optimizer: maximize mu'w - gamma w'Sw with caps & long-only."""
    n = len(tickers)
    if n == 0 or not HAVE_CVXPY:
        return pd.Series(dtype=float)
    # Ensure PSD
    S = (cov + cov.T) / 2.0 + RIDGE_EPS * np.eye(n)
    w = cp.Variable(n)
    objective = cp.Maximize(mu @ w - gamma * cp.quad_form(w, S))
    constraints = [w >= 0, w <= W_CAP, cp.sum(w) == 1]
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
        if w.value is None:
            # fallback solvers
            prob.solve(solver=cp.OSQP, verbose=False)
    except Exception:
        pass

    if w.value is None:
        # Fallback to MinVar heuristic if optimization failed
        return min_var_cov(S, tickers)

    ww = np.clip(np.array(w.value, dtype=float), 0.0, W_CAP)
    s = ww.sum()
    ww = ww / s if s > 0 else np.full(n, 1.0 / n)
    return pd.Series(ww, index=tickers, dtype=float)


# ---------- Main ---------- #

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    rets = pd.read_parquet(RET).sort_index()
    mem = pd.read_csv(MEM)
    rebal = pd.read_csv(REB, parse_dates=["rebalance_date"]).sort_values("rebalance_date")
    dates: pd.DatetimeIndex = rets.index

    # Equity tracks
    equity: Dict[str, float] = {"EW": 1.0, "MinVar": 1.0, "HRP": 1.0, "MV": 1.0}
    history: List[Dict] = []

    # Previous weights for turnover
    template = pd.Series(0.0, index=rets.columns, dtype=float)
    prev_w = {
        "EW": template.copy(),
        "MinVar": template.copy(),
        "HRP": template.copy(),
        "MV": template.copy(),
    }

    for i, t in enumerate(rebal["rebalance_date"]):
        # Determine forward trading window
        if i + 1 < len(rebal):
            next_rb = rebal["rebalance_date"].iloc[i + 1]
            end_idx = dates.searchsorted(next_rb, side="left") - 1
            if end_idx < 0:
                continue
            end_trading = dates[end_idx]
        else:
            end_trading = dates[-1]

        start_idx = dates.searchsorted(t, side="right")
        if start_idx >= len(dates):
            continue
        start_trading = dates[start_idx]
        if end_trading < start_trading:
            continue

        # Universe + tradable pool
        active = active_at(mem, t)
        lab_path = LAB_DIR / f"labels_{t.date()}.parquet"
        if not lab_path.exists():
            continue
        labels = pd.read_parquet(lab_path)["r_next"]
        pool = sorted(set(active) & set(rets.columns) & set(labels.index))
        if len(pool) < 50:
            continue

        # Trailing stats up to the day BEFORE the forward window starts
        cov_end = dates[start_idx - 1] if start_idx - 1 >= 0 else dates[0]
        mu, S, _ = realized_stats(rets[pool], cov_end,
                                  cov_lookback=COV_LOOKBACK_DAYS,
                                  mu_lookback=MU_LOOKBACK_DAYS)

        # Build weights
        w_EW = equal_weight(pool)
        w_MVmin = min_var_cov(S, pool)
        w_HRP = hrp_weights(S, pool) if HAVE_SCIPY else w_EW
        w_MV = mv_cvxpy_weights(mu, S, pool) if HAVE_CVXPY else w_MVmin

        weights = {"EW": w_EW, "MinVar": w_MVmin, "HRP": w_HRP, "MV": w_MV}

        # Turnover costs at rebalance
        tc_this: Dict[str, float] = {}
        for name, w in weights.items():
            w_full = w.reindex(rets.columns, fill_value=0.0)
            tc = float((w_full - prev_w[name]).abs().sum()) * TC
            tc_this[name] = tc
            prev_w[name] = w_full  # persist for next period

        # Realize returns over forward window
        window = rets.loc[start_trading:end_trading].fillna(0.0)
        for name, w in weights.items():
            w_full = w.reindex(rets.columns, fill_value=0.0)
            daily = (window * w_full).sum(axis=1)
            gross = float((1.0 + daily).prod())
            equity[name] *= (gross * (1.0 - tc_this[name]))

        history.append({
            "rebalance": t.date(),
            "n_pool": len(pool),
            "start_trading": start_trading.date(),
            "end_trading": end_trading.date(),
            "turnover_cost_EW": tc_this["EW"],
            "turnover_cost_MinVar": tc_this["MinVar"],
            "turnover_cost_HRP": tc_this["HRP"],
            "turnover_cost_MV": tc_this["MV"],
            "equity_EW": equity["EW"],
            "equity_MinVar": equity["MinVar"],
            "equity_HRP": equity["HRP"],
            "equity_MV": equity["MV"],
            "mean_label_in_pool": float(labels.loc[pool].mean()),
        })

    out = pd.DataFrame(history)
    out.to_csv(OUT / "results.csv", index=False)
    print(f"Saved baseline results -> {OUT / 'results.csv'}")

    # Friendly hints if deps were missing
    if not HAVE_SCIPY:
        print("[INFO] scipy not found — HRP fell back to EW. `pip install scipy` to enable HRP.")
    if not HAVE_CVXPY:
        print("[INFO] cvxpy not found — MV fell back to MinVar. `pip install cvxpy` to enable MV.")


if __name__ == "__main__":
    main()
