#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import re
import time
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import AdamW

# --- Optional CVXPY backend (not used with PGD) ---
try:
    import cvxpy as cp  # type: ignore
    from cvxpylayers.torch import CvxpyLayer  # type: ignore
    cp.settings.DEBUG = True  # optional
except Exception:
    cp = None
    CvxpyLayer = None  # type: ignore[assignment]

from torch.serialization import add_safe_globals
try:
    from torch_geometric.data.data import Data, DataEdgeAttr  # type: ignore
    add_safe_globals([Data, DataEdgeAttr, DictConfig])
except Exception:
    try:
        from torch_geometric.data import Data  # type: ignore
        add_safe_globals([Data, DictConfig])
    except Exception:
        add_safe_globals([DictConfig])

# (optional) HRP clustering; will gracefully fall back to IVP if SciPy is missing
try:
    from scipy.cluster.hierarchy import linkage, leaves_list  # type: ignore
    from scipy.spatial.distance import squareform  # type: ignore
except Exception:
    linkage = None
    leaves_list = None
    squareform = None

from src.model_gat import GATPortfolio
from src.loss import (
    sharpe_loss,
    turnover_penalty_indexed,
    entropy_penalty,
    neg_daily_log_utility,
)
from src.cov import ledoit_wolf_shrinkage  # NEW: robust Σ option

# ---------------------- globals ----------------------
_MARKOWITZ_BACKEND = "pgd"       # "pgd" | "cvxpy"
_MARKOWITZ_PGD_STEPS = 80
_MARKOWITZ_NORM_MU = True

# ---------------------- utilities ----------------------

@dataclass
class Sample:
    ts: pd.Timestamp
    graph_path: Path
    label_path: Path

def _load_ckpt(path: Path, device: torch.device):
    try:
        return torch.load(str(path), map_location=device)
    except Exception:
        from omegaconf import DictConfig as _DictConfig
        add_safe_globals([_DictConfig])
        return torch.load(str(path), map_location=device, weights_only=False)

def _infer_ts(p: Path) -> pd.Timestamp:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", p.name)
    if not m:
        raise ValueError(f"Cannot parse date from {p.name}")
    return pd.Timestamp(m.group(1))

def _to_psd(cov: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    M = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(M)
    vals_clipped = np.clip(vals, eps, None)
    M_psd = (vecs * vals_clipped) @ vecs.T
    m = M_psd.shape[0]
    return M_psd + eps * np.eye(m, dtype=M_psd.dtype)

# ----------------- CVXPY fallback -----------------

@lru_cache(maxsize=None)
def _build_markowitz_diag_layer(n: int, cap: float, gamma: float):
    if cp is None or CvxpyLayer is None:
        raise RuntimeError("cvxpy/cvxpylayers not available; use model.markowitz_backend=pgd")
    w  = cp.Variable(n)
    mu = cp.Parameter(n)
    d  = cp.Parameter(n)  # std-devs (>=0)
    obj  = cp.Minimize(gamma * cp.sum_squares(cp.multiply(d, w)) - mu @ w)
    cons = [w >= 0, w <= cap, cp.sum(w) == 1]
    prob = cp.Problem(obj, cons)
    return CvxpyLayer(prob, parameters=[mu, d], variables=[w])

@lru_cache(maxsize=None)
def _build_markowitz_chol_layer(n: int, cap: float, gamma: float):
    if cp is None or CvxpyLayer is None:
        raise RuntimeError("cvxpy/cvxpylayers not available; use model.markowitz_backend=pgd")
    w  = cp.Variable(n)
    mu = cp.Parameter(n)
    L  = cp.Parameter((n, n))
    obj  = cp.Minimize(gamma * cp.sum_squares(L @ w) - mu @ w)
    cons = [w >= 0, w <= cap, cp.sum(w) == 1]
    prob = cp.Problem(obj, cons)
    return CvxpyLayer(prob, parameters=[mu, L], variables=[w])

def _chol_spd(M: np.ndarray, base_eps: float = 1e-8, max_tries: int = 5) -> np.ndarray:
    eps = base_eps
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(M + eps * np.eye(M.shape[0], dtype=M.dtype))
        except np.linalg.LinAlgError:
            eps *= 10.0
    M2 = _to_psd(M, eps)
    return np.linalg.cholesky(M2 + 1e-3 * np.eye(M2.shape[0], dtype=M2.dtype))

# ----------------- PGD backend -----------------

def _project_capped_simplex(v: torch.Tensor, cap: float, s: float = 1.0, iters: int = 50) -> torch.Tensor:
    """
    Project v onto { w: 0<=w<=cap, sum w = s } by bisection in tau, where
      w_i = clip(v_i - tau, 0, cap).
    """
    vmax = v.detach().max().item()
    hi = vmax                      # sum -> 0 when tau = max(v)
    lo = hi - cap
    for _ in range(64):            # expand lower bracket until sum >= s
        if torch.clamp(v - lo, 0.0, cap).sum().item() >= s:
            break
        lo -= cap
    for _ in range(iters):         # bisection
        mid = 0.5 * (lo + hi)
        w = torch.clamp(v - mid, 0.0, cap)
        if w.sum().item() > s:
            lo = mid
        else:
            hi = mid
    return torch.clamp(v - hi, 0.0, cap)

def _auto_feasible_cap(cap: float, n_eff: int) -> float:
    """
    Ensure feasibility of sum(w)=1 with per-name cap:
      need n_eff * cap >= 1  -> cap >= 1/n_eff
    If infeasible, raise cap minimally to 1/n_eff (+tiny eps).
    """
    if n_eff <= 0:
        return cap
    min_cap = 1.0 / n_eff + 1e-9
    return cap if cap >= min_cap else min_cap

def _cpu64(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(device="cpu", dtype=torch.float64).contiguous()

def markowitz_layer_pgd(
    mu: torch.Tensor,
    cov_np: np.ndarray,
    cap: float,
    gamma: float,
    mode: str = "diag",
    topk: int | None = None,
    steps: int = 60,
) -> torch.Tensor:
    """
    Unrolled PGD on:
        minimize_w  gamma * wᵀΣw - muᵀw
        s.t.        0 <= w <= cap,  sum w = 1
    Auto-raises cap if n_eff * cap < 1 to keep the constraint feasible.
    """
    N = int(mu.numel())

    # Optional top-K; make cap feasible for the selected dimension.
    if topk is not None and 0 < topk < N:
        idx = torch.topk(mu, k=topk).indices
        idx_cpu = idx.detach().cpu().numpy()
        n_eff = int(len(idx_cpu))
        cap_eff = _auto_feasible_cap(cap, n_eff)
        S_sub  = cov_np[np.ix_(idx_cpu, idx_cpu)]
        w_sub  = markowitz_layer_pgd(mu[idx], S_sub, cap_eff, gamma, mode=mode, topk=None, steps=steps)
        w_full = torch.zeros_like(mu)
        w_full[idx] = w_sub
        return w_full

    # No top-k: ensure feasibility with the full dimension.
    cap_eff = _auto_feasible_cap(cap, N)

    dtype, device = mu.dtype, mu.device
    if mode == "diag":
        sdiag = np.clip(np.diag(cov_np), 1e-10, None)
        S_diag = torch.from_numpy(sdiag).to(device=device, dtype=dtype)
        L = (2.0 * gamma * S_diag.max()).clamp_min(1e-8)  # Lipschitz
        eta = 1.0 / L
        w = torch.full_like(mu, 1.0 / N)
        for _ in range(steps):
            grad = 2.0 * gamma * (S_diag * w) - mu
            w = _project_capped_simplex(w - eta * grad, cap_eff, s=1.0)
        return w

    # general SPD case
    S = torch.from_numpy(cov_np).to(device=device, dtype=dtype)
    v = torch.randn_like(mu)
    for _ in range(8):  # power iteration for spectral norm
        v = S @ v
        v = v / (v.norm() + 1e-12)
    L = (2.0 * gamma * (v @ (S @ v))).clamp_min(1e-8)
    eta = 1.0 / L
    w = torch.full_like(mu, 1.0 / N)
    for _ in range(steps):
        grad = 2.0 * gamma * (S @ w) - mu
        w = _project_capped_simplex(w - eta * grad, cap_eff, s=1.0)
    return w

def markowitz_layer_torch(
    mu: torch.Tensor,            # (N,)
    cov_np: np.ndarray,          # (N, N)
    cap: float,
    gamma: float,
    mode: str = "diag",          # "diag" | "chol"
    topk: int | None = None,
) -> torch.Tensor:
    backend = globals().get("_MARKOWITZ_BACKEND", "pgd")
    if backend == "pgd":
        steps = int(globals().get("_MARKOWITZ_PGD_STEPS", 60))
        return markowitz_layer_pgd(mu, cov_np, cap, gamma, mode=mode, topk=topk, steps=steps)

    # ---- CVXPY fallback ----
    N = int(mu.shape[0])
    if topk is not None and 0 < topk < N:
        idx = torch.topk(mu, k=topk).indices
        mu_sub = mu[idx]
        S_sub  = cov_np[np.ix_(idx.cpu().numpy(), idx.cpu().numpy())]
        w_sub  = markowitz_layer_torch(mu_sub, S_sub, cap, gamma, mode=mode, topk=None)
        w_full = torch.zeros_like(mu)
        w_full[idx] = w_sub
        return w_full

    if mode == "diag":
        if cp is None or CvxpyLayer is None:
            raise RuntimeError("cvxpy/cvxpylayers not available; use model.markowitz_backend=pgd")
        S_psd = _to_psd(cov_np).astype(np.float64, copy=False)
        d_np  = np.sqrt(np.clip(np.diag(S_psd), 1e-10, None))

        layer = _build_markowitz_diag_layer(N, cap, gamma)

        mu64 = _cpu64(mu)
        d_t  = torch.from_numpy(d_np).to(device="cpu", dtype=torch.float64).contiguous()

        if not torch.isfinite(mu64).all():
            raise ValueError("CVXPy(mu) contains non-finite values")
        if not torch.isfinite(d_t).all():
            raise ValueError("CVXPy(d) contains non-finite values")

        try:
            (w64,) = layer(
                mu64, d_t,
                solver_args={"verbose": False, "max_iters": 10000, "eps": 1e-5}
            )
            return w64.to(device=mu.device, dtype=mu.dtype)
        except Exception as e:
            print(f"[CVXPY->PGD fallback(diag)] {type(e).__name__}: {e}", flush=True)
            steps = int(globals().get("_MARKOWITZ_PGD_STEPS", 60))
            return markowitz_layer_pgd(mu, cov_np, cap, gamma, mode=mode, topk=None, steps=steps)

    if cp is None or CvxpyLayer is None:
        raise RuntimeError("cvxpy/cvxpylayers not available; use model.markowitz_backend=pgd")
    S_psd = _to_psd(cov_np).astype(np.float64, copy=False)
    L_np  = _chol_spd(S_psd)

    layer = _build_markowitz_chol_layer(N, cap, gamma)

    mu64 = _cpu64(mu)
    L_t  = torch.from_numpy(L_np).to(device="cpu", dtype=torch.float64).contiguous()

    if not torch.isfinite(mu64).all():
        raise ValueError("CVXPy(mu) contains non-finite values")
    if not torch.isfinite(L_t).all():
        raise ValueError("CVXPy(L) contains non-finite values")

    try:
        (w64,) = layer(
            mu64, L_t,
            solver_args={"verbose": False, "max_iters": 10000, "eps": 1e-5}
        )
        return w64.to(device=mu.device, dtype=mu.dtype)
    except Exception as e:
        print(f"[CVXPY->PGD fallback(chol)] {type(e).__name__}: {e}", flush=True)
        steps = int(globals().get("_MARKOWITZ_PGD_STEPS", 60))
        return markowitz_layer_pgd(mu, cov_np, cap, gamma, mode=mode, topk=None, steps=steps)

# ----------------- memory utils -----------------

def build_prev_mem(tickers: list[str], mem_store: dict[str, torch.Tensor], dim: int, device: torch.device, decay: float) -> torch.Tensor:
    out = torch.zeros((len(tickers), dim), device=device)
    for i, t in enumerate(tickers):
        if t in mem_store:
            out[i] = mem_store[t] * decay
    return out

def write_new_mem(tickers: list[str], new_mem: torch.Tensor, mem_store: dict[str, torch.Tensor]) -> None:
    for i, t in enumerate(tickers):
        mem_store[t] = new_mem[i].detach()

# ----------------- data helpers -----------------

def list_samples(graph_dir: Path, labels_dir: Path) -> List[Sample]:
    gfiles = sorted([p for p in graph_dir.glob("graph_*.pt")], key=lambda x: _infer_ts(x))
    samples: List[Sample] = []
    for gp in gfiles:
        ts = _infer_ts(gp)
        lp = labels_dir / f"labels_{ts.date()}.parquet"
        if lp.exists():
            samples.append(Sample(ts=ts, graph_path=gp, label_path=lp))
    return samples

def split_samples(samples: List[Sample], train_start: str, val_start: str, test_start: str) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    t0 = pd.Timestamp(train_start)
    v0 = pd.Timestamp(val_start)
    te0 = pd.Timestamp(test_start)
    train = [s for s in samples if t0 <= s.ts < v0]
    val   = [s for s in samples if v0 <= s.ts < te0]
    test  = [s for s in samples if s.ts >= te0]
    return train, val, test

def load_graph(path: Path) -> "Data":
    return torch.load(str(path), map_location="cpu", weights_only=False)

def load_label_vec(label_path: Path, tickers: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    df = pd.read_parquet(label_path)
    df = df["r_next"].astype(float)
    s = df.reindex([t for t in tickers])
    mask = ~s.isna()
    y = s.fillna(0.0).astype(np.float32).values
    return torch.from_numpy(y), torch.from_numpy(mask.values)

# ----------------- covariance helpers -----------------

def _shrink_cov_linear_ridge(S: np.ndarray, alpha: float, ridge_eps: float) -> np.ndarray:
    """Linear shrinkage to diag plus optional ridge."""
    if alpha <= 0.0 and ridge_eps <= 0.0:
        return S
    D = np.diag(np.diag(S))
    S_shrunk = (1.0 - alpha) * S + alpha * D
    if ridge_eps > 0.0:
        S_shrunk = S_shrunk + ridge_eps * np.eye(S.shape[0], dtype=S.dtype)
    return S_shrunk

def _cov_from_hist(X: np.ndarray, method: str, alpha: float, ridge_eps: float) -> np.ndarray:
    """
    Return a PSD covariance matrix using the requested method.
    method in {"lw","ledoit_wolf"} -> Ledoit-Wolf shrinkage (from src.cov)
    else -> linear shrinkage to diag (+ridge)
    """
    if X.shape[0] < 2:
        v = np.var(X, axis=0, ddof=1) if X.shape[0] >= 2 else np.ones(X.shape[1], dtype=np.float64)
        return np.diag(np.clip(v, 1e-8, None))

    if method.lower() in {"lw", "ledoit_wolf", "ledoitwolf"}:
        res = ledoit_wolf_shrinkage(X)
        # accept ndarray or tuple of length 2/3
        if isinstance(res, tuple):
            S_hat = res[0]
        else:
            S_hat = res
        return S_hat

    # else: linear shrinkage + optional ridge
    S = np.cov(X, rowvar=False)
    return _shrink_cov_linear_ridge(S, alpha=alpha, ridge_eps=ridge_eps)

# ----------------- baseline helpers -----------------

def _corr_from_cov(S: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    invd = np.where(d > 0, 1.0 / d, 0.0)
    return (S * invd).T * invd

def _ivp_weights(S: np.ndarray) -> np.ndarray:
    v = np.clip(np.diag(S), 1e-12, None)
    w = 1.0 / v
    return (w / w.sum())

def _hrp_weights(S: np.ndarray) -> np.ndarray:
    """
    Minimal HRP:
    - build tree on correlation distance
    - quasi-diagonalize
    - recursive bisection with IVP cluster variance
    Falls back to IVP if SciPy isn't available.
    """
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
    while len(clusters) > 0:
        cl = clusters.pop()
        if len(cl) <= 1:
            continue
        split = int(np.ceil(len(cl) / 2))
        c1, c2 = cl[:split], cl[split:]

        def _cluster_var(idxs):
            S_c = S_[np.ix_(idxs, idxs)]
            w_c = _ivp_weights(S_c)
            return float(w_c @ S_c @ w_c)

        v1, v2 = _cluster_var(c1), _cluster_var(c2)
        alloc2 = v1 / (v1 + v2 + 1e-12)
        alloc1 = 1.0 - alloc2
        w[c1] *= alloc1
        w[c2] *= alloc2
        clusters.extend([c1, c2])

    w_ord = np.zeros(n)
    w_ord[order] = w
    w_ord = np.maximum(w_ord, 0.0)
    s = w_ord.sum()
    return (w_ord / s) if s > 0 else _ivp_weights(S)

def _project_cap_and_sum_to_one(w_np: np.ndarray, cap: float) -> np.ndarray:
    t = torch.from_numpy(w_np.astype(np.float32))
    w = _project_capped_simplex(t, cap=_auto_feasible_cap(cap, len(w_np)), s=1.0, iters=50)
    return w.detach().cpu().numpy()

def _baseline_weight_ew(n: int, cap: float) -> np.ndarray:
    """Equal-weight with per-name cap via capped-simplex projection."""
    w0 = np.full(n, 1.0 / n, dtype=np.float32)
    w  = _project_capped_simplex(
        torch.from_numpy(w0),
        cap=_auto_feasible_cap(cap, n),
        s=1.0,
        iters=50,
    ).detach().cpu().numpy()
    return w

def _baseline_weight_mv(mu: np.ndarray, S: np.ndarray, cap: float, gamma: float,
                        mode: str, topk: Optional[int]) -> np.ndarray:
    mu_t = torch.from_numpy(mu.astype(np.float32))
    if globals().get("_MARKOWITZ_NORM_MU", True):
        scale = float(np.max(np.abs(mu)) + 1e-12)
        if scale > 0:
            mu_t = mu_t / scale
    w_t = markowitz_layer_torch(mu_t, S, cap=cap, gamma=gamma, mode=mode, topk=(int(topk) if topk else None))
    return w_t.detach().cpu().numpy()

def _baseline_weight_minvar(S: np.ndarray, cap: float, gamma: float,
                            mode: str, topk: Optional[int]) -> np.ndarray:
    mu0 = np.zeros(S.shape[0], dtype=np.float32)
    return _baseline_weight_mv(mu0, S, cap, gamma, mode, topk)

def _baseline_weight_hrp(S: np.ndarray, cap: float) -> np.ndarray:
    w = _hrp_weights(S)
    return _project_cap_and_sum_to_one(w, cap)

def _compute_metrics_from_returns(r: pd.Series) -> Dict[str, float]:
    r = r.copy().dropna()
    if r.empty:
        return {"CAGR": np.nan, "AnnMean": np.nan, "AnnVol": np.nan, "Sharpe": np.nan, "MDD": np.nan}
    eq = (1.0 + r).cumprod()
    ann = 252.0
    n = int(r.shape[0])
    cagr    = float(eq.iloc[-1] ** (ann / max(n, 1)) - 1.0)
    annmean = float(r.mean() * ann)
    annvol  = float(r.std(ddof=0) * np.sqrt(ann))
    sharpe  = float(annmean / annvol) if annvol > 0 else float("nan")
    dd = eq / eq.cummax() - 1.0
    mdd = float(dd.min())
    return {"CAGR": cagr, "AnnMean": annmean, "AnnVol": annvol, "Sharpe": sharpe, "MDD": mdd}

def _backtest_baseline(
    strategy: str,
    samples: List[Sample],
    returns_daily: pd.DataFrame,
    tc_decimal: float,
    cap: float,
    mv_gamma: float,
    cov_alpha: float,
    cov_ridge: float,
    mode: str,
    topk: Optional[int],
    cov_method: str,
    out_dir: Optional[Path] = None,
) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
    """
    Re-creates the same rebalance windows you use for GAT evaluation,
    computes baseline weights per window, applies the same TC, and stitches daily returns.
    Returns (daily_returns, per_window_log) where per_window_log mirrors gat_equity.csv columns.
    """
    dates: pd.DatetimeIndex = returns_daily.index
    prev_w: Optional[pd.Series] = None
    daily_chunks: list[pd.Series] = []
    rows: list[dict] = []
    equity = 1.0

    for i, s in enumerate(samples):
        end_trading = dates[-1] if i + 1 == len(samples) else dates[dates.searchsorted(samples[i+1].ts, "left") - 1]
        start_idx = dates.searchsorted(s.ts, "right")
        if start_idx >= len(dates) or end_trading < dates[start_idx]:
            continue
        start_trading = dates[start_idx]

        g = load_graph(s.graph_path)
        tickers: list[str] = list(getattr(g, "tickers", []))

        hist = returns_daily.loc[:start_trading].iloc[:-1].tail(252).fillna(0.0)
        X = hist.reindex(columns=tickers, fill_value=0.0).values.astype(np.float64, copy=False)
        if X.shape[1] < 2:
            continue
        S = _cov_from_hist(X, method=cov_method, alpha=cov_alpha, ridge_eps=cov_ridge)

        if strategy == "EW":
            w_np = _baseline_weight_ew(len(tickers), cap)
        elif strategy == "MV":
            mu = X.mean(axis=0)
            w_np = _baseline_weight_mv(mu, S, cap, mv_gamma, mode, topk)
        elif strategy == "MinVar":
            w_np = _baseline_weight_minvar(S, cap, mv_gamma, mode, topk)
        elif strategy == "HRP":
            w_np = _baseline_weight_hrp(S, cap)
        else:
            raise ValueError(f"Unknown baseline: {strategy}")

        curr_w = pd.Series(w_np.astype(float), index=tickers)

        if prev_w is None:
            tc = 0.0
        else:
            aligned = pd.concat([prev_w.rename("prev"), curr_w.rename("curr")], axis=1).fillna(0.0)
            tc = float((aligned["curr"] - aligned["prev"]).abs().sum()) * tc_decimal

        win = returns_daily.loc[start_trading:end_trading]
        valid_mask = ~win.isna().all(axis=0)
        curr_w = curr_w[valid_mask.index[valid_mask]]
        w_sum = float(curr_w.sum())
        if w_sum > 0:
            curr_w = curr_w / w_sum

        r_p = win.reindex(columns=curr_w.index, fill_value=0.0).mul(curr_w, axis=1).sum(axis=1)
        if len(r_p) > 0 and tc > 0:
            r0 = r_p.iloc[0]
            r_p.iloc[0] = (1.0 + r0) * (1.0 - tc) - 1.0

        daily_chunks.append(r_p)

        period_gross = float((1.0 + r_p).prod())
        equity *= period_gross  # turnover already folded into r_p[0] above
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
            r_all.rename("r").to_frame().to_csv(out_dir / f"{strategy.lower()}_daily_returns.csv")
            (1.0 + r_all).cumprod().rename("equity").to_frame().to_csv(out_dir / f"{strategy.lower()}_equity_daily.csv")
            log_df.to_csv(out_dir / f"{strategy.lower()}_equity.csv", index=False)

    return (r_all if r_all is not None else pd.Series(dtype=float), log_df)

# ----------------- evaluation -----------------

def _annualized_sharpe_from_series(r: pd.Series) -> float:
    r = r.dropna()
    if r.empty:
        return float("nan")
    annmean = float(r.mean() * 252.0)
    annvol  = float(r.std(ddof=0) * np.sqrt(252.0))
    return float(annmean / annvol) if annvol > 0 else float("nan")

def _eval_periods_daily_returns(
    model: nn.Module,
    device: torch.device,
    samples: List[Sample],
    returns_daily: pd.DataFrame,
    tc_decimal: float,
    head: str,
    markowitz_args: Optional[dict],
) -> pd.Series:
    """Return a single concatenated daily return series across the provided samples."""
    dates: pd.DatetimeIndex = returns_daily.index
    prev_w: Optional[pd.Series] = None
    daily_chunks: list[pd.Series] = []

    with torch.no_grad():
        for i, s in enumerate(samples):
            # end of trading window: day before next rebal date (or last day)
            end_trading = dates[-1] if i + 1 == len(samples) else dates[dates.searchsorted(samples[i+1].ts, "left") - 1]
            start_idx = dates.searchsorted(s.ts, "right")
            if start_idx >= len(dates) or end_trading < dates[start_idx]:
                continue
            start_trading = dates[start_idx]

            g = load_graph(s.graph_path)
            tickers: list[str] = list(getattr(g, "tickers", []))
            x = g.x.to(device)
            edge_index = g.edge_index.to(device)
            eattr = getattr(g, "edge_attr", None)
            if eattr is not None:
                eattr = eattr.to(device).to(x.dtype)

            mask_all = torch.ones(len(tickers), dtype=torch.bool, device=device)
            out, _ = model(x, edge_index, mask_all, eattr)

            if head == "direct":
                w_t = out.detach().cpu()
            else:
                # Σ from rolling window up to start_trading (exclude day 0)
                hist = returns_daily.loc[:start_trading].iloc[:-1].tail(252).fillna(0.0)
                X = hist.reindex(columns=tickers, fill_value=0.0).values.astype(np.float64, copy=False)
                if X.shape[1] < 2:
                    continue
                method = str(markowitz_args.get("cov_method", "lw") if markowitz_args else "lw")
                S = _cov_from_hist(
                    X,
                    method=method,
                    alpha=float(markowitz_args.get("cov_shrinkage_alpha", 0.10)) if markowitz_args else 0.10,
                    ridge_eps=float(markowitz_args.get("cov_ridge_eps", 1e-6)) if markowitz_args else 1e-6,
                )

                mu_in = out
                if globals().get("_MARKOWITZ_NORM_MU", True):
                    scale = mu_in.abs().max().clamp_min(1e-8)
                    mu_in = mu_in / scale

                w_m = markowitz_layer_torch(
                    mu_in, S,
                    cap=float(markowitz_args.get("weight_cap", 0.02)) if markowitz_args else 0.02,
                    gamma=float(markowitz_args.get("gamma", 5.0)) if markowitz_args else 5.0,
                    mode=str(markowitz_args.get("mode", "diag")) if markowitz_args else "diag",
                    topk=(int(markowitz_args["topk"]) if (markowitz_args and markowitz_args.get("topk", 0)) else None),
                )
                w_t = w_m.detach().cpu()

            curr_w = pd.Series(w_t.numpy(), index=tickers, dtype=float)

            # TC on transition
            if prev_w is None:
                tc = 0.0
            else:
                aligned = pd.concat([prev_w.rename("prev"), curr_w.rename("curr")], axis=1).fillna(0.0)
                tc = float((aligned["curr"] - aligned["prev"]).abs().sum()) * tc_decimal

            window = returns_daily.loc[start_trading:end_trading]
            valid_mask = ~window.isna().all(axis=0)
            curr_w = curr_w[valid_mask.index[valid_mask]]
            w_sum = float(curr_w.sum())
            if w_sum > 0:
                curr_w = curr_w / w_sum

            daily = window.reindex(columns=curr_w.index, fill_value=0.0).mul(curr_w, axis=1).sum(axis=1)
            if len(daily) > 0 and tc > 0:
                r0 = daily.iloc[0]
                daily.iloc[0] = (1.0 + r0) * (1.0 - tc) - 1.0

            daily_chunks.append(daily)
            prev_w = curr_w

    if not daily_chunks:
        return pd.Series(dtype=float)
    r_all = pd.concat(daily_chunks).sort_index()
    r_all = r_all[~r_all.index.duplicated(keep="first")]
    return r_all

def evaluate_daily_compound(
    model: nn.Module,
    device: torch.device,
    samples: List[Sample],
    returns_daily: pd.DataFrame,
    tc_decimal: float,
    out_dir: Path | None = None,
    head: str = "direct",
    markowitz_args: Optional[dict] = None,
) -> Dict[str, float]:
    r = _eval_periods_daily_returns(
        model=model,
        device=device,
        samples=samples,
        returns_daily=returns_daily,
        tc_decimal=tc_decimal,
        head=head,
        markowitz_args=markowitz_args,
    )
    equity = float((1.0 + r).prod()) if not r.empty else 1.0

    # write per-window equity & weights already handled inside train loop; here we keep API stable
    if out_dir is not None:
        # store stitched daily returns/equity for GAT
        if not r.empty:
            r.rename("r").to_frame().to_csv(out_dir / "gat_daily_returns.csv")
            (1.0 + r).cumprod().rename("equity").to_frame().to_csv(out_dir / "gat_equity_daily.csv")
    return {"final_equity": equity}

# ----------------- training -----------------

def train_gat(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("medium")

    graph_dir = Path(cfg.data.graph_dir)
    labels_dir = Path(cfg.data.labels_dir)
    rets_path = Path(cfg.data.returns_daily)
    out_dir   = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Globals from cfg
    backend = str(getattr(cfg.model, "markowitz_backend", "pgd")).lower()
    assert backend in ("pgd", "cvxpy"), "model.markowitz_backend must be 'pgd' or 'cvxpy'"

    # Use PGD during training if CVXPy was requested (avoid diffcp autograd on Windows)
    train_backend = "pgd" if (backend == "cvxpy" and int(getattr(cfg.train, "epochs", 0)) > 0) else backend
    globals()["_MARKOWITZ_BACKEND"] = train_backend
    globals()["_MARKOWITZ_PGD_STEPS"] = int(getattr(cfg.model, "markowitz_pgd_steps", 80))
    globals()["_MARKOWITZ_NORM_MU"]   = bool(getattr(cfg.model, "markowitz_normalize_mu", True))

    cov_alpha  = float(getattr(cfg.model, "cov_shrinkage_alpha", 0.10))
    cov_ridge  = float(getattr(cfg.model, "cov_ridge_eps", 1.0e-6))
    cov_method = str(getattr(cfg.model, "cov_method", "lw")).lower()  # NEW: "lw" | "linear"

    samples_all = list_samples(graph_dir, labels_dir)
    train_s, val_s, test_s = split_samples(samples_all, cfg.split.train_start, cfg.split.val_start, cfg.split.test_start)

    print(f"[Data] train={len(train_s)}  val={len(val_s)}  test={len(test_s)}", flush=True)
    print(f"[Info] backend={backend} epochs={cfg.train.epochs}", flush=True)

    # We now ALWAYS load daily returns (needed for validation Sharpe)
    returns_daily: pd.DataFrame = pd.read_parquet(rets_path).sort_index()
    dates: pd.DatetimeIndex = returns_daily.index  # type: ignore[assignment]

    model = GATPortfolio(
        in_dim=cfg.model.in_dim,
        hidden_dim=cfg.model.hidden_dim,
        heads=cfg.model.heads,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        residual=True,
        use_gatv2=cfg.model.use_gatv2,
        use_edge_attr=cfg.model.use_edge_attr,
        head=cfg.model.head,
        mem_hidden=(cfg.temporal.mem_hidden if cfg.temporal.mem_hidden is not None else None),
    ).to(device)

    opt = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    torch.manual_seed(cfg.train.seed)
    np.random.seed(int(cfg.train.seed))

    def reset_memory() -> dict[str, torch.Tensor]:
        return {}

    # --- training control / early stopping on VALIDATION SHARPE ---
    patience = int(getattr(cfg.train, "patience", 12))
    best_val_sharpe = -1e9
    best_path = out_dir / "gat_best.pt"
    epochs_no_improve = 0

    log_path = out_dir / "train_log.csv"
    if not log_path.exists():
        pd.DataFrame([{"epoch": 0, "train_loss": np.nan, "val_sharpe": np.nan, "best_val_sharpe": np.nan}]).iloc[0:0] \
          .to_csv(log_path, index=False)

    # ---------- warmup / announcement ----------
    if cfg.model.head == "markowitz":
        if len(train_s) > 0:
            g0 = load_graph(train_s[0].graph_path)
            N0 = int(len(getattr(g0, "tickers", [])))
            topk = int(getattr(cfg.model, "markowitz_topk", 0) or 0)
            n_eff = N0 if (topk <= 0 or topk >= N0) else topk
            mode = str(getattr(cfg.model, "markowitz_mode", "diag"))
            if backend == "cvxpy":
                print(f"[Warmup] Building Markowitz layer (mode={mode}) for n={n_eff} ...", flush=True)
                t0 = time.time()
                if mode == "diag":
                    _ = _build_markowitz_diag_layer(n_eff, cfg.model.weight_cap, cfg.model.markowitz_gamma)
                else:
                    _ = _build_markowitz_chol_layer(n_eff, cfg.model.weight_cap, cfg.model.markowitz_gamma)
                print(f"[Warmup] Done in {time.time()-t0:.2f}s.", flush=True)
            else:
                print(f"[Warmup] Using PGD backend (mode={mode}) for n={n_eff} — no CVXPY compile.", flush=True)

    # ------------ training loop ------------
    for epoch in range(cfg.train.epochs):
        model.train()
        mem_store: dict[str, torch.Tensor] = reset_memory()
        train_loss_sum = 0.0
        train_updates  = 0

        if cfg.temporal.use_memory and cfg.train.ordered_when_memory:
            train_seq = sorted(train_s, key=lambda s: s.ts)
        else:
            train_seq = train_s[:]
            if not (cfg.temporal.use_memory and cfg.train.ordered_when_memory):
                np.random.shuffle(train_seq)

        sharpe_r: list[torch.Tensor] = []
        sharpe_ws: list[torch.Tensor] = []
        sharpe_tickers: list[list[str]] = []

        last_w_detached: Optional[torch.Tensor] = None
        last_tickers: list[str] | None = None

        last_heartbeat = time.time()

        for i, s in enumerate(train_seq):
            g = load_graph(s.graph_path)
            tickers: list[str] = list(getattr(g, "tickers", []))
            y, mask = load_label_vec(s.label_path, tickers)
            if mask.sum().item() < 10:
                continue

            x = g.x.to(device)
            eidx = g.edge_index.to(device)
            eattr = getattr(g, "edge_attr", None)
            eattr = eattr.to(device).to(x.dtype) if (eattr is not None and cfg.model.use_edge_attr) else None
            mvalid = mask.to(device)

            prev_mem = None
            if cfg.temporal.use_memory:
                mem_dim = model.mem_dim
                prev_mem = build_prev_mem(tickers, mem_store, mem_dim, device, cfg.temporal.decay)

            if cfg.model.head == "direct":
                w, new_mem = model(x, eidx, mvalid, eattr, prev_mem)
            else:
                mu_hat, new_mem = model(x, eidx, mvalid, eattr, prev_mem)

            if cfg.temporal.use_memory:
                write_new_mem(tickers, new_mem, mem_store)

            # ----------- three training objectives -----------
            if cfg.loss.objective == "daily_log_utility":
                all_ts = [ss.ts for ss in train_seq]
                try:
                    idx = all_ts.index(s.ts)
                    next_ts = all_ts[idx + 1].to_pydatetime() if idx + 1 < len(all_ts) else None
                except ValueError:
                    next_ts = None

                start_idx = dates.searchsorted(s.ts, side="right")
                end_trading = dates[-1] if next_ts is None else dates[dates.searchsorted(next_ts, side="left") - 1]
                if start_idx >= len(dates) or end_trading < dates[start_idx]:
                    continue
                start_trading = dates[start_idx]

                window = returns_daily.loc[start_trading:end_trading].reindex(columns=tickers, fill_value=0.0)
                R = torch.from_numpy(window.values).to(device).to(x.dtype)

                if cfg.model.head == "direct":
                    w_for_turn = w
                    loss = neg_daily_log_utility(R, w_for_turn)
                else:
                    hist = returns_daily.loc[:start_trading].iloc[:-1].tail(252).fillna(0.0)
                    X = hist.reindex(columns=tickers, fill_value=0.0).values.astype(np.float64, copy=False)
                    S = _cov_from_hist(X, method=cov_method, alpha=cov_alpha, ridge_eps=cov_ridge)

                    mu_in = mu_hat
                    if _MARKOWITZ_NORM_MU:
                        scale = mu_in.abs().max().clamp_min(1e-8)
                        mu_in = mu_in / scale

                    w_m = markowitz_layer_torch(
                        mu_in, S,
                        cap=cfg.model.weight_cap,
                        gamma=cfg.model.markowitz_gamma,
                        mode=getattr(cfg.model, "markowitz_mode", "diag"),
                        topk=(cfg.model.markowitz_topk if cfg.model.markowitz_topk else None),
                    )
                    w_for_turn = w_m
                    loss = neg_daily_log_utility(R, w_for_turn)

                # regularizers
                if last_w_detached is not None and last_tickers is not None:
                    loss = loss + turnover_penalty_indexed(
                        [last_w_detached, w_for_turn], [last_tickers, tickers],
                        tc_decimal=cfg.loss.turnover_bps / 10000.0,
                    )
                loss = loss + entropy_penalty([w_for_turn], coef=cfg.loss.entropy_coef)
                l1c = float(getattr(cfg.loss, "l1_coef", 0.0))
                if l1c > 0:
                    loss = loss + l1c * w_for_turn.abs().mean()

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                train_loss_sum += float(loss.detach().item())
                train_updates  += 1

                last_w_detached = w_for_turn.detach()
                last_tickers = tickers

            elif cfg.loss.objective == "sharpe_rnext":
                if cfg.model.head == "direct":
                    w_for_turn = w
                    r_p = (w_for_turn * y.to(device)).sum()
                else:
                    cov_win = returns_daily.loc[:s.ts].tail(252).fillna(0.0)
                    X = cov_win.reindex(columns=tickers, fill_value=0.0).values.astype(np.float64, copy=False)
                    S = _cov_from_hist(X, method=cov_method, alpha=cov_alpha, ridge_eps=cov_ridge)

                    mu_in = mu_hat
                    if _MARKOWITZ_NORM_MU:
                        scale = mu_in.abs().max().clamp_min(1e-8)
                        mu_in = mu_in / scale

                    w_m = markowitz_layer_torch(
                        mu_in, S,
                        cap=cfg.model.weight_cap,
                        gamma=cfg.model.markowitz_gamma,
                        mode=getattr(cfg.model, "markowitz_mode", "diag"),
                        topk=(cfg.model.markowitz_topk if cfg.model.markowitz_topk else None),
                    )
                    w_for_turn = w_m
                    r_p = (w_for_turn * y.to(device)).sum()

                sharpe_r.append(r_p)
                sharpe_ws.append(w_for_turn)
                sharpe_tickers.append(tickers)

                if len(sharpe_r) == cfg.train.batch_size:
                    loss = sharpe_loss(torch.stack(sharpe_r), eps=cfg.loss.sharpe_eps)
                    loss = loss + turnover_penalty_indexed(
                        sharpe_ws, sharpe_tickers, tc_decimal=cfg.loss.turnover_bps / 10000.0,
                    )
                    loss = loss + entropy_penalty(sharpe_ws, coef=cfg.loss.entropy_coef)
                    l1c = float(getattr(cfg.loss, "l1_coef", 0.0))
                    if l1c > 0:
                        loss = loss + l1c * torch.stack([w.abs().mean() for w in sharpe_ws]).mean()

                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    train_loss_sum += float(loss.detach().item())
                    train_updates  += 1

                    sharpe_r.clear()
                    sharpe_ws.clear()
                    sharpe_tickers.clear()

            else:
                # simple next-step return objective
                if cfg.model.head == "direct":
                    w_for_turn = w
                    loss = -(w_for_turn * y.to(device)).sum()
                else:
                    loss = -(mu_hat * y.to(device)).sum()
                    w_for_turn = None

                if w_for_turn is not None and last_w_detached is not None and last_tickers is not None:
                    loss = loss + turnover_penalty_indexed(
                        [last_w_detached, w_for_turn], [last_tickers, tickers],
                        tc_decimal=cfg.loss.turnover_bps / 10000.0,
                    )
                    loss = loss + entropy_penalty([w_for_turn], coef=cfg.loss.entropy_coef)
                    l1c = float(getattr(cfg.loss, "l1_coef", 0.0))
                    if l1c > 0:
                        loss = loss + l1c * w_for_turn.abs().mean()

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                train_loss_sum += float(loss.detach().item())
                train_updates  += 1

                if w_for_turn is not None:
                    last_w_detached = w_for_turn.detach()
                    last_tickers = tickers

            # heartbeat every ~5s
            if time.time() - last_heartbeat > 5.0:
                print(f"[Epoch {epoch+1:03d}] progress {i+1}/{len(train_seq)} ...", flush=True)
                last_heartbeat = time.time()

            # sanity print every 10 epochs at first step
            if (i == 0) and (epoch % 10 == 0) and cfg.model.head == "markowitz" and last_w_detached is not None:
                with torch.no_grad():
                    s1 = last_w_detached.sum().item()
                    vmax = last_w_detached.max().item()
                    vmin = last_w_detached.min().item()
                print(f"[Chk e{epoch+1:03d}] sum={s1:.6f} min={vmin:.6f} max={vmax:.6f} cap={cfg.model.weight_cap}", flush=True)

        # flush partial Sharpe minibatch
        if cfg.loss.objective == "sharpe_rnext" and len(sharpe_r) > 0:
            loss = sharpe_loss(torch.stack(sharpe_r), eps=cfg.loss.sharpe_eps)
            loss = loss + turnover_penalty_indexed(
                sharpe_ws, sharpe_tickers, tc_decimal=cfg.loss.turnover_bps / 10000.0,
            )
            loss = loss + entropy_penalty(sharpe_ws, coef=cfg.loss.entropy_coef)
            l1c = float(getattr(cfg.loss, "l1_coef", 0.0))
            if l1c > 0:
                loss = loss + l1c * torch.stack([w.abs().mean() for w in sharpe_ws]).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss_sum += float(loss.detach().item())
            train_updates  += 1

        # ---------- Validation: DAILY SHARPE ----------
        model.eval()
        with torch.no_grad():
            marko_args = dict(
                gamma=float(cfg.model.markowitz_gamma),
                weight_cap=float(cfg.model.weight_cap),
                mode=str(getattr(cfg.model, "markowitz_mode", "diag")),
                topk=int(getattr(cfg.model, "markowitz_topk", 0) or 0),
                cov_shrinkage_alpha=cov_alpha,
                cov_ridge_eps=cov_ridge,
                cov_method=cov_method,
            ) if cfg.model.head == "markowitz" else None

            val_r = _eval_periods_daily_returns(
                model=model,
                device=device,
                samples=sorted(val_s, key=lambda ss: ss.ts),
                returns_daily=returns_daily,
                tc_decimal=cfg.loss.turnover_bps / 10000.0,
                head=cfg.model.head,
                markowitz_args=marko_args,
            )
            val_sharpe = _annualized_sharpe_from_series(val_r)

        train_loss_avg = float(train_loss_sum / max(train_updates, 1))
        print(f"[Epoch {epoch+1:03d}] steps={len(train_seq)}  updates={train_updates}  "
              f"train_loss={train_loss_avg:.6f}  val_sharpe={val_sharpe:.6f}", flush=True)

        improved = (not np.isnan(val_sharpe)) and (val_sharpe > best_val_sharpe + 1e-8)
        if improved:
            best_val_sharpe = val_sharpe
            epochs_no_improve = 0
            payload = {"state_dict": model.state_dict(), "cfg": OmegaConf.to_container(cfg, resolve=True)}
            torch.save(payload, best_path)
            print(f"[EarlyStop] ↑ new best val Sharpe {best_val_sharpe:.4f}; saved -> {best_path}", flush=True)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[EarlyStop] patience reached ({patience}); stopping at epoch {epoch+1}.", flush=True)
                break

        # append to CSV log each epoch
        pd.DataFrame([{
            "epoch": epoch + 1,
            "train_loss": train_loss_avg,
            "val_sharpe": float(val_sharpe),
            "best_val_sharpe": float(best_val_sharpe),
        }]).to_csv(log_path, mode="a", header=False, index=False)

    print(f"Saved best model -> {best_path}")
    # Stable default: evaluate with PGD unless explicitly forced.
    eval_backend = backend if bool(getattr(cfg.model, "eval_use_cvxpy", False)) else "pgd"
    globals()["_MARKOWITZ_BACKEND"] = eval_backend
    print(f"[Eval] using backend={eval_backend}", flush=True)

    # ---------- Test backtest ----------
    ckpt = _load_ckpt(best_path, device)
    model.load_state_dict(ckpt["state_dict"])

    returns_daily_test = returns_daily  # already loaded/sorted

    marko_args = dict(
        gamma=float(cfg.model.markowitz_gamma),
        weight_cap=float(cfg.model.weight_cap),
        mode=str(getattr(cfg.model, "markowitz_mode", "diag")),
        topk=int(getattr(cfg.model, "markowitz_topk", 0) or 0),
        cov_shrinkage_alpha=cov_alpha,
        cov_ridge_eps=cov_ridge,
        cov_method=cov_method,
    ) if cfg.model.head == "markowitz" else None

    test_stats = evaluate_daily_compound(
        model=model,
        device=device,
        samples=sorted(test_s, key=lambda s: s.ts),
        returns_daily=returns_daily_test,
        tc_decimal=cfg.loss.turnover_bps / 10000.0,
        out_dir=out_dir,
        head=cfg.model.head,
        markowitz_args=marko_args,
    )
    pd.DataFrame([test_stats]).to_csv(out_dir / "gat_test_stats.csv", index=False)
    print(f"Test stats: {test_stats}  -> {out_dir/'gat_test_stats.csv'}")
    print(f"Saved GAT equity daily -> {out_dir/'gat_equity_daily.csv'} and returns -> gat_daily_returns.csv")

    # === Full metrics (CAGR / AnnMean / AnnVol / Sharpe / MDD) ===
    metrics_row: Optional[Dict[str, float]] = None
    try:
        # read stitched daily returns we just wrote
        p = out_dir / "gat_daily_returns.csv"
        if not p.exists():
            print("[Metrics] Missing gat_daily_returns.csv; skipping metrics.", flush=True)
        else:
            r = pd.read_csv(p, parse_dates=[0], index_col=0).iloc[:, 0]
            r.index = pd.to_datetime(r.index)
            m = _compute_metrics_from_returns(r)
            metrics_row = {"strategy": "GAT", **m}
            pd.DataFrame([metrics_row]).to_csv(out_dir / "strategy_metrics.csv", index=False)
            print(f"Metrics: {metrics_row}  -> {out_dir/'strategy_metrics.csv'}", flush=True)
    except Exception as e:
        print(f"[Metrics] failed: {e}", flush=True)
    # === end metrics ===

    # === Baselines computed in-place (aligned to GAT settings/windows) ===
    try:
        strategies = ["EW", "MV", "HRP", "MinVar"]

        cap   = float(cfg.model.weight_cap)
        gamma = float(cfg.model.markowitz_gamma)
        mode  = str(getattr(cfg.model, "markowitz_mode", "diag"))
        topk  = int(getattr(cfg.model, "markowitz_topk", 0) or 0)
        tc    = cfg.loss.turnover_bps / 10000.0

        baseline_metrics: List[Dict[str, float]] = []
        for strat in strategies:
            r_s, _ = _backtest_baseline(
                strategy=strat,
                samples=sorted(test_s, key=lambda s: s.ts),
                returns_daily=returns_daily_test,
                tc_decimal=tc,
                cap=cap,
                mv_gamma=gamma,
                cov_alpha=cov_alpha,
                cov_ridge=cov_ridge,
                mode=mode,
                topk=(topk if topk > 0 else None),
                cov_method=cov_method,
                out_dir=out_dir,  # writes *_daily.csv alongside GAT files
            )
            if r_s.empty:
                print(f"Metrics: {{'strategy': '{strat}', 'note': 'no returns assembled'}}", flush=True)
                continue
            m = _compute_metrics_from_returns(r_s)
            payload = {"strategy": strat, **m}
            baseline_metrics.append(payload)
            print(f"Metrics: {payload}", flush=True)

        # summary CSV
        try:
            summary_rows = []
            if metrics_row is not None:
                summary_rows.append({"strategy": "GAT", **{k: v for k, v in metrics_row.items() if k != "strategy"}})
            for strat in strategies:
                p = out_dir / f"{strat.lower()}_daily_returns.csv"
                if p.exists():
                    r = pd.read_csv(p, parse_dates=[0], index_col=0).iloc[:, 0]
                    r.index = pd.to_datetime(r.index)
                    summary_rows.append({"strategy": strat, **_compute_metrics_from_returns(r)})
            if summary_rows:
                pd.DataFrame(summary_rows).to_csv(out_dir / "compare_gat_vs_baselines.csv", index=False)
                print(f"Saved summary -> {out_dir/'compare_gat_vs_baselines.csv'}", flush=True)
        except Exception:
            pass

    except Exception as e:
        print(f"[Baselines-inline] failed: {e}", flush=True)
