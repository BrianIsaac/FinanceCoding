#!/usr/bin/env python3
# scripts/make_graphs.py — build monthly graph snapshots

from __future__ import annotations

import importlib

# --- ensure repo root on sys.path so imports resolve correctly ---
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from torch_geometric.data import Data  # type: ignore
except Exception as e:
    raise RuntimeError("torch_geometric is required to build graph snapshots") from e

# Do NOT use networkx (slow for our dense MST). We'll provide a fast Prim MST below.
nx = None  # sentinel to avoid accidental use

# Prefer your central implementations if available
try:
    from src.models.gat import graph_builder as graph_lib  # type: ignore

    # reload to avoid stale bytecode if the user just edited src/models/gat/graph_builder.py
    graph_lib = importlib.reload(graph_lib)
except Exception:
    graph_lib = None  # type: ignore

try:
    from src.data.processors import features as feat_lib  # type: ignore
except Exception:
    feat_lib = None  # type: ignore


# -----------------------------------------------------------------------------
# Config model
# -----------------------------------------------------------------------------


@dataclass
class GraphBuildCfg:
    window_days: int
    expanding: bool
    min_overlap: int
    rebalance: str
    calendar: str
    method: str
    graph_filter: str
    tmfg_keep: str | int
    mst_abs: bool
    edge_attr: Sequence[str]
    dtype: str


def _resolve_cfg(cfg: DictConfig) -> GraphBuildCfg:
    g = cfg.graph
    tmfg_keep = "auto"
    if "tmfg" in g and "keep_n_edges" in g.tmfg:
        k = g.tmfg.keep_n_edges
        tmfg_keep = int(k) if (isinstance(k, (int, float)) and int(k) > 0) else "auto"
    mst_abs = bool(getattr(getattr(g, "mst", {}), "use_absolute_corr", True))
    edge_attr = list(getattr(g, "edge_attr", ["corr", "strength", "sign"]))
    return GraphBuildCfg(
        window_days=int(getattr(g, "window_days", 252)),
        expanding=bool(getattr(g, "expanding", False)),
        min_overlap=int(getattr(g, "min_overlap", 126)),
        rebalance=str(getattr(g, "rebalance", "monthly")).lower(),
        calendar=str(getattr(g, "calendar", "month_end")).lower(),
        method=str(getattr(g, "method", "correlation")).lower(),
        graph_filter=str(getattr(g, "graph_filter", "tmfg")).lower(),
        tmfg_keep=tmfg_keep,
        mst_abs=mst_abs,
        edge_attr=edge_attr,
        dtype=str(getattr(g, "dtype", "float32")),
    )


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _find_returns_path(cfg: DictConfig) -> Path:
    if "data" in cfg and "returns_daily" in cfg.data:
        p = Path(cfg.data.returns_daily)
        if p.exists():
            return p
    if "paths" in cfg and "returns_processed" in cfg.paths:
        p = Path(cfg.paths.returns_processed)
        if p.exists():
            return p
    return Path("processed/returns_daily.parquet")


def _find_graph_dir(cfg: DictConfig) -> Path:
    if "paths" in cfg and "graph_dir" in cfg.paths:
        return Path(cfg.paths.graph_dir)
    return Path("processed/graphs")


def _load_returns(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).sort_index()
    df.columns = [str(c) for c in df.columns]
    return df.astype(float)


def _rebalance_dates(returns: pd.DataFrame, freq: str, calendar: str) -> list[pd.Timestamp]:
    idx = returns.index.to_series()
    if freq in ("monthly", "m"):
        dates = idx.groupby(idx.index.to_period("M")).tail(1).index
    elif freq in ("quarterly", "q"):
        dates = idx.groupby(idx.index.to_period("Q")).tail(1).index
    else:
        dates = returns.index[::21]  # ~monthly fallback
    return [pd.Timestamp(d) for d in dates]


# -----------------------------------------------------------------------------
# Core calculations
# -----------------------------------------------------------------------------


def _window(
    returns: pd.DataFrame,
    end_date: pd.Timestamp,
    lookback_days: int,
    start_anchor: pd.Timestamp | None,
    expanding: bool,
) -> pd.DataFrame:
    right = returns.loc[:end_date].iloc[:-1]  # exclude end_date to avoid leakage
    if expanding and start_anchor is not None:
        left = right.loc[start_anchor:]
    else:
        left = right.tail(int(lookback_days))
    return left


def _prune_min_overlap(win: pd.DataFrame, min_overlap: int) -> pd.DataFrame:
    if min_overlap <= 0:
        return win
    mask = win.notna().sum(axis=0) >= int(min_overlap)
    return win.loc[:, mask]


def _corr_from_window(win: pd.DataFrame, use_abs: bool) -> np.ndarray:
    x = win.fillna(0.0).values
    c = np.corrcoef(x, rowvar=False)
    c = np.clip(c, -0.9999, 0.9999)
    c = 0.5 * (c + c.T)
    np.fill_diagonal(c, 1.0)
    return np.abs(c) if use_abs else c


def _mst_edges_from_corr(c: np.ndarray) -> list[tuple[int, int, float]]:
    """Fast vectorized Prim's MST on correlation-derived distances."""
    n = c.shape[0]
    if n <= 1:
        return []
    d = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - c)))  # Mantegna distance

    selected = np.zeros(n, dtype=bool)
    selected[0] = True
    best_dist = d[0].copy()
    best_from = np.zeros(n, dtype=int)

    edges: list[tuple[int, int, float]] = []
    for _ in range(n - 1):
        masked = np.where(selected, np.inf, best_dist)
        j = int(np.argmin(masked))
        i = int(best_from[j])
        if np.isfinite(masked[j]):
            u, v = (i, j) if i < j else (j, i)
            edges.append((u, v, float(c[u, v])))
        selected[j] = True

        new_best = d[j]
        update = new_best < best_dist
        best_from = np.where(update, j, best_from)
        best_dist = np.where(update, new_best, best_dist)

    return edges


def _tmfg_keep_edges(n: int, cfg_keep: str | int) -> int:
    if isinstance(cfg_keep, int) and cfg_keep > 0:
        return int(cfg_keep)
    return max(0, 3 * (n - 2))  # TMFG triangulation


def _edges_from_corr(c: np.ndarray, gcfg: GraphBuildCfg) -> tuple[np.ndarray, np.ndarray | None]:
    n = c.shape[0]

    # Prefer central implementation (uses TMFG/MST/kNN/threshold and builds attrs)
    if graph_lib is not None and hasattr(graph_lib, "edges_from_corr"):
        try:
            ei_np, eattr_np = graph_lib.edges_from_corr(  # type: ignore
                c,
                method=gcfg.graph_filter,
                undirected=True,
                include_strength=("strength" in gcfg.edge_attr),
                include_sign=("sign" in gcfg.edge_attr),
                tmfg_keep_n=_tmfg_keep_edges(n, gcfg.tmfg_keep),
                mst_use_abs=gcfg.mst_abs,
            )
            return ei_np, eattr_np
        except Exception as err:
            if gcfg.graph_filter == "tmfg":
                raise RuntimeError(f"src.graph.edges_from_corr failed for TMFG: {err}") from err
            warnings.warn(
                f"[graph_lib] edges_from_corr failed; local fallback in use: {err}", stacklevel=2
            )

    # If TMFG was requested but central impl not available, fail loudly
    if gcfg.graph_filter == "tmfg":
        raise RuntimeError(
            "graph_filter=tmfg requested but src.graph.edges_from_corr() "
            f"is not available/importable. graph_lib={graph_lib!r}"
        )

    # Local MST fallback (fast Prim)
    if gcfg.graph_filter == "mst":
        edges = _mst_edges_from_corr(c)
    else:
        raise ValueError(f"Unknown graph_filter='{gcfg.graph_filter}'")

    # Build edge_index and edge_attr
    rows: list[tuple[int, int]] = []
    attrs: list[list[float]] = []
    include_corr = "corr" in gcfg.edge_attr
    include_strength = "strength" in gcfg.edge_attr
    include_sign = "sign" in gcfg.edge_attr

    for i, j, c in edges:
        rows.append((i, j))
        rows.append((j, i))  # undirected as bidirectional edges
        a: list[float] = []
        if include_corr:
            a.append(float(c))
        if include_strength:
            a.append(float(abs(c)))
        if include_sign:
            a.append(1.0 if c >= 0.0 else -1.0)
        if a:
            attrs.append(a)
            attrs.append(a)

    edge_index = np.asarray(rows, dtype=np.int64).T if rows else np.zeros((2, 0), dtype=np.int64)
    edge_attr = np.asarray(attrs, dtype=np.float32) if attrs else None
    return edge_index, edge_attr


def _default_features(win: pd.DataFrame) -> pd.DataFrame:
    r = win.fillna(0.0)
    out = pd.DataFrame(index=r.columns)
    out["ret_1m"] = (1.0 + r.tail(21)).prod() - 1.0
    out["ret_3m"] = (1.0 + r.tail(63)).prod() - 1.0
    out["ret_6m"] = (1.0 + r.tail(126)).prod() - 1.0
    out["vol_3m"] = r.tail(63).std(ddof=0)
    out["vol_6m"] = r.tail(126).std(ddof=0)
    out["mom_12m"] = (1.0 + r.tail(252)).prod() - 1.0
    out["turnover_3m"] = 0.0
    out["size"] = 0.0
    return out.fillna(0.0)


def _winsorize_df(
    df: pd.DataFrame, method: str, c: float, q_low: float, q_high: float
) -> pd.DataFrame:
    x = df.copy()
    m = (method or "mad").lower()
    if m == "mad":
        med = x.median(axis=0)
        mad = (x - med).abs().median(axis=0).replace(0, 1e-9)
        lo = med - c * 1.4826 * mad
        hi = med + c * 1.4826 * mad
        return x.clip(lower=lo, upper=hi, axis=1)
    elif m in ("quantile", "quantiles", "quant"):
        lo = x.quantile(q_low)
        hi = x.quantile(q_high)
        return x.clip(lower=lo, upper=hi, axis=1)
    return x


def _zscore_df(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=0)
    sd = df.std(ddof=0, axis=0).replace(0, 1e-9)
    return (df - mu) / sd


def _build_features(cfg: DictConfig, asof: pd.Timestamp, win: pd.DataFrame) -> pd.DataFrame:
    try:
        if feat_lib is not None and hasattr(feat_lib, "compute_features_from_window"):
            feats = feat_lib.compute_features_from_window(win)  # type: ignore
        else:
            feats = _default_features(win)
    except Exception as err:
        warnings.warn(
            f"[features] compute_features_from_window failed; using defaults: {err}", stacklevel=2
        )
        feats = _default_features(win)

    gf = getattr(cfg.graph, "features", None)
    if gf is not None:
        wz = getattr(gf, "winsorize", None)
        if wz is not None:
            feats = _winsorize_df(
                feats,
                method=str(getattr(wz, "method", "mad")),
                c=float(getattr(wz, "c", 3.5)),
                q_low=float(getattr(wz, "q_low", 0.01)),
                q_high=float(getattr(wz, "q_high", 0.99)),
            )
        impute = str(getattr(gf, "impute", "median"))
        if impute == "median":
            feats = feats.fillna(feats.median())
        elif impute == "mean":
            feats = feats.fillna(feats.mean())
        else:
            feats = feats.fillna(0.0)

        stdz = str(getattr(gf, "standardize", "zscore"))
        if stdz.lower() == "zscore":
            feats = _zscore_df(feats)
    else:
        feats = feats.fillna(0.0)

    return feats


def _save_snapshot(
    out_dir: Path,
    asof: pd.Timestamp,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None,
    tickers: list[str],
) -> None:
    payload: dict[str, object] = {
        "x": x.contiguous(),
        "edge_index": edge_index.contiguous(),
        "tickers": tickers,
    }
    if edge_attr is not None:
        payload["edge_attr"] = edge_attr.contiguous()
    data_obj = Data(**payload)
    target = out_dir / f"graph_{asof.date()}.pt"
    torch.save(data_obj, target)


# -----------------------------------------------------------------------------
# Entry
# -----------------------------------------------------------------------------


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:

    gcfg = _resolve_cfg(cfg)
    returns_path = _find_returns_path(cfg)
    out_dir = _find_graph_dir(cfg)
    _ensure_dir(out_dir)

    returns = _load_returns(returns_path)
    dates = _rebalance_dates(returns, gcfg.rebalance, gcfg.calendar)
    if not dates:
        raise RuntimeError("No rebalance dates found from returns index.")

    dtype = torch.float32 if gcfg.dtype == "float32" else torch.float64

    start_anchor: pd.Timestamp | None = None

    for asof in dates:
        win = _window(returns, asof, gcfg.window_days, start_anchor, gcfg.expanding)
        if start_anchor is None:
            start_anchor = win.index.min() if len(win) else None

        win = _prune_min_overlap(win, gcfg.min_overlap)
        win = win.loc[:, ~(win.isna().all(axis=0))]

        tickers = [str(c) for c in win.columns]
        if len(tickers) < 2 or len(win) < 2:
            continue

        # Prefer robust shrinkage corr from src.graph if available
        use_abs = gcfg.mst_abs if gcfg.graph_filter == "mst" else False
        if graph_lib is not None and hasattr(graph_lib, "corr_from_returns"):
            c = graph_lib.corr_from_returns(win)  # type: ignore
            if use_abs:
                c = np.abs(c)
        else:
            c = _corr_from_window(win, use_abs=use_abs)

        # Build edges (TMFG via src.graph when requested; else fast MST fallback)
        ei_np, eattr_np = _edges_from_corr(c, gcfg)

        feats = _build_features(cfg, asof, win).reindex(tickers).fillna(0.0)
        x_np = feats.values.astype(np.float32, copy=False)

        x = torch.from_numpy(x_np).to(dtype=dtype)
        edge_index = torch.from_numpy(ei_np).long()
        edge_attr = torch.from_numpy(eattr_np).to(dtype=dtype) if eattr_np is not None else None

        _save_snapshot(out_dir, asof, x, edge_index, edge_attr, tickers)


if __name__ == "__main__":
    main()
