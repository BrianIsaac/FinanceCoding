from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, List
import math
import numpy as np
import pandas as pd
import torch

try:
    # torch_geometric >= 2.6
    from torch_geometric.data import Data  # type: ignore
except Exception:
    # older import path fallback
    from torch_geometric.data.data import Data  # type: ignore

from .cov import robust_covariance, to_correlation

__all__ = [
    "GraphBuildConfig",
    "build_graph_from_returns",
    "build_period_graph",
    "edges_from_corr",
    "corr_from_returns",
]


@dataclass
class GraphBuildConfig:
    lookback_days: int = 252
    cov_method: Literal["lw", "oas", "sample"] = "lw"   # Ledoit–Wolf (default)
    shrink_to: Literal["diag", "identity"] = "diag"
    min_var: float = 1e-10                               # variance floor
    corr_method: Literal["from_cov"] = "from_cov"
    filter_method: Literal["mst", "tmfg", "knn", "threshold"] = "mst"
    knn_k: int = 8
    threshold_abs_corr: float = 0.30                     # used for threshold mode
    use_edge_attr: bool = True                           # include [rho, |rho|, sign] if True


# ----------------------------- helpers -----------------------------

def _safe_nan_to_num(a: np.ndarray, fill: float = 0.0) -> np.ndarray:
    out = np.array(a, dtype=float, copy=True)
    out[~np.isfinite(out)] = fill
    return out


def _corr_to_dist(C: np.ndarray) -> np.ndarray:
    """
    Map correlation in [-1, 1] to a distance in [0, 2] using
      d_ij = sqrt(2 * (1 - rho_ij)).
    """
    C_clip = np.clip(C, -1.0, 1.0)
    D = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - C_clip)))
    np.fill_diagonal(D, 0.0)
    return D


def _mst_edges_from_corr(C: np.ndarray) -> List[Tuple[int, int]]:
    """
    Build an MST over correlation-derived distances with a vectorised Prim's algorithm.
    Returns an edge list of undirected pairs (i, j), i<j.
    """
    n = C.shape[0]
    if n <= 1:
        return []
    D = _corr_to_dist(C)

    selected = np.zeros(n, dtype=bool)
    selected[0] = True
    edges: List[Tuple[int, int]] = []

    best_dist = D[0].copy()
    best_from = np.zeros(n, dtype=int)

    for _ in range(n - 1):
        # pick the nearest non-selected node
        masked = np.where(selected, np.inf, best_dist)
        j = int(np.argmin(masked))
        i = int(best_from[j])
        if math.isfinite(masked[j]):
            a, b = (i, j) if i < j else (j, i)
            edges.append((a, b))
        selected[j] = True

        # relax distances from the newly added node j
        new_best = D[j]
        update = new_best < best_dist
        best_dist = np.where(update, new_best, best_dist)
        best_from = np.where(update, j, best_from)

    return edges


def _greedy_init_k4(A: np.ndarray) -> List[int]:
    """
    Greedy K4 initializer for TMFG on |correlation| matrix A (diagonal should be -inf).
    Heuristic: pick node with largest degree (row sum), then add nodes that maximize
    cumulative connection strength to the current set.
    """
    n = A.shape[0]
    if n < 4:
        return list(range(min(n, 4)))

    chosen: List[int] = []
    remaining = set(range(n))

    # 1) max row-sum
    i0 = int(np.argmax(np.sum(A, axis=1)))
    chosen.append(i0)
    remaining.remove(i0)

    # 2..4) greedily add the node with max sum to current chosen set
    for _ in range(3):
        scores = []
        for j in remaining:
            scores.append((float(np.sum(A[j, chosen])), j))
        j_best = max(scores)[1]
        chosen.append(j_best)
        remaining.remove(j_best)

    return chosen


def _tmfg_edges_from_corr(C: np.ndarray) -> List[Tuple[int, int]]:
    """
    Triangulated Maximally Filtered Graph (TMFG) on |correlation|.
    Simplified O(N^2) greedy implementation:

    - Start from a K4 (greedy-selected).
    - Maintain face list (triangles of current planar triangulation).
    - Iteratively insert each remaining node:
        choose the face (a,b,c) that maximizes A[r,a]+A[r,b]+A[r,c],
        connect r to (a,b,c), split face into three.

    Returns undirected edges (i<j). For N nodes, TMFG yields 3N-6 edges (N>=3).
    """
    n = C.shape[0]
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    if n == 3:
        return [(0, 1), (0, 2), (1, 2)]

    A = np.abs(C).copy()
    np.fill_diagonal(A, -np.inf)

    # Initial K4
    seed = _greedy_init_k4(A)
    if len(seed) < 4:
        # fallback to MST if we couldn't seed K4
        return _mst_edges_from_corr(C)

    i0, i1, i2, i3 = seed
    edges: set[Tuple[int, int]] = set()

    def _add(u: int, v: int):
        a, b = (u, v) if u < v else (v, u)
        if a != b:
            edges.add((a, b))

    # K4 edges
    k4 = [i0, i1, i2, i3]
    for a in range(4):
        for b in range(a + 1, 4):
            _add(k4[a], k4[b])

    # faces of tetrahedron (order doesn't matter for this scoring)
    faces: List[Tuple[int, int, int]] = [
        (i0, i1, i2),
        (i0, i1, i3),
        (i0, i2, i3),
        (i1, i2, i3),
    ]

    remaining = [j for j in range(n) if j not in seed]

    # Insert nodes one-by-one
    for r in remaining:
        best_gain = -np.inf
        best_face_idx = -1
        for t, (a, b, c) in enumerate(faces):
            gain = A[r, a] + A[r, b] + A[r, c]
            if gain > best_gain:
                best_gain = gain
                best_face_idx = t

        if best_face_idx < 0:
            # should not happen; fallback safe
            continue

        a, b, c = faces.pop(best_face_idx)
        # Connect r to the face
        _add(r, a)
        _add(r, b)
        _add(r, c)
        # Split face into three new faces
        faces.extend([(r, a, b), (r, a, c), (r, b, c)])

    return sorted(list(edges))


def _knn_edges_from_corr(C: np.ndarray, k: int) -> List[Tuple[int, int]]:
    """
    Symmetric k-NN graph on absolute correlation.
    Keep the top-|rho| neighbors for each node; symmetrise.
    """
    n = C.shape[0]
    if n <= 1 or k <= 0:
        return []
    A = np.abs(C).copy()
    np.fill_diagonal(A, -np.inf)

    nbrs: List[set] = [set() for _ in range(n)]
    for i in range(n):
        kk = min(k, n - 1)
        if kk <= 0:
            continue
        idx = np.argpartition(-A[i], kth=kk - 1)[:kk]
        for j in idx:
            if i != j:
                nbrs[i].add(int(j))

    edges: set[Tuple[int, int]] = set()
    for i in range(n):
        for j in nbrs[i]:
            if i in nbrs[j]:  # symmetric
                a, b = (i, j) if i < j else (j, i)
                edges.add((a, b))
    return sorted(list(edges))


def _threshold_edges_from_corr(C: np.ndarray, thr: float) -> List[Tuple[int, int]]:
    """
    Keep edges with |rho| >= threshold.
    """
    n = C.shape[0]
    E: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(C[i, j]) >= thr:
                E.append((i, j))
    return E


def _edge_attr_from_corr(C: np.ndarray, E: List[Tuple[int, int]]) -> np.ndarray:
    """
    Build edge attributes:
      [rho, abs_rho, sign]
    """
    attrs = []
    for i, j in E:
        rho = float(np.clip(C[i, j], -1.0, 1.0))
        attrs.append([rho, abs(rho), float(np.sign(rho))])
    return np.asarray(attrs, dtype=np.float32)


# --------------------------- main entry ----------------------------

def build_graph_from_returns(
    returns_window: pd.DataFrame,
    features_matrix: Optional[np.ndarray],
    tickers: List[str],
    ts: pd.Timestamp,
    cfg: GraphBuildConfig,
) -> Data:
    """
    Parameters
    ----------
    returns_window : (T x N) DataFrame
        Daily returns (aligned to `tickers`) for the lookback window (no future leakage).
    features_matrix : Optional[np.ndarray]
        Node features matrix X of shape (N, d). If None, we create a 1-dim dummy feature.
    tickers : List[str]
        Node ordering. Must match the columns of `returns_window` after reindex.
    ts : pd.Timestamp
        The snapshot date for this graph (rebalance date).
    cfg : GraphBuildConfig
        Rolling-corr & filtering options.

    Returns
    -------
    torch_geometric.data.Data
        Data(x, edge_index, edge_attr?, tickers=<list[str]>, ts=<timestamp>)
    """
    # Align and clean
    rets = returns_window.reindex(columns=tickers, fill_value=np.nan).astype(float)
    rets = rets.fillna(0.0)  # conservative fill
    X = rets.values  # (T, N)

    # Robust covariance -> correlation
    S = robust_covariance(X, method=cfg.cov_method, shrink_to=cfg.shrink_to, min_var=cfg.min_var)
    C = to_correlation(S)

    # Filtering -> edge list
    if cfg.filter_method == "mst":
        E = _mst_edges_from_corr(C)
    elif cfg.filter_method == "tmfg":
        E = _tmfg_edges_from_corr(C)
    elif cfg.filter_method == "knn":
        E = _knn_edges_from_corr(C, k=int(cfg.knn_k))
    elif cfg.filter_method == "threshold":
        E = _threshold_edges_from_corr(C, thr=float(cfg.threshold_abs_corr))
    else:
        raise ValueError(f"Unknown filter_method={cfg.filter_method}")

    # Edge index & attributes
    if len(E) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr_t = None
    else:
        edge_index = torch.tensor(np.array(E, dtype=np.int64).T, dtype=torch.long)
        if cfg.use_edge_attr:
            edge_attr = _edge_attr_from_corr(C, E)
            edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)
        else:
            edge_attr_t = None

    # Node features
    N = len(tickers)
    if features_matrix is None:
        x_np = np.ones((N, 1), dtype=np.float32)  # trivial constant feature
    else:
        x_np = _safe_nan_to_num(features_matrix).astype(np.float32)
        if x_np.shape[0] != N:
            raise ValueError(f"features_matrix has {x_np.shape[0]} rows but N={N} tickers")

    x = torch.tensor(x_np, dtype=torch.float32)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr_t,
    )
    # helpful metadata for downstream code
    data.tickers = tickers
    data.ts = ts.to_pydatetime()

    return data


# ----------------------- convenience wrapper -----------------------

def build_period_graph(
    returns_daily: pd.DataFrame,
    period_end: pd.Timestamp,
    tickers: List[str],
    features_matrix: Optional[np.ndarray],
    cfg: GraphBuildConfig,
) -> Data:
    """
    Pick the rolling window that ends the **day before** `period_end` (to avoid leakage),
    then call `build_graph_from_returns`.
    """
    lookback = int(cfg.lookback_days)
    # end index is the last index strictly < period_end
    idx_end = returns_daily.index.searchsorted(period_end, side="left") - 1
    if idx_end < 0:
        raise ValueError("Not enough history before period_end to build a graph.")
    start = max(0, idx_end - lookback + 1)
    window = returns_daily.iloc[start:idx_end + 1]
    return build_graph_from_returns(window, features_matrix, tickers, period_end, cfg)


# --- adapter for scripts/make_graphs.py ---------------------------------------
def edges_from_corr(
    C: np.ndarray,
    method: str = "tmfg",
    undirected: bool = True,
    include_strength: bool = True,
    include_sign: bool = True,
    tmfg_keep_n: int | str = "auto",
    mst_use_abs: bool = True,
    knn_k: int | None = None,
    threshold_abs_corr: float | None = None,
):
    """
    Adapter for scripts/make_graphs.py.

    Returns:
      edge_index: np.ndarray shape (2, E)
      edge_attr : Optional[np.ndarray] shape (E, D) with columns:
                  [rho] (+[|rho|] if include_strength, +[sign] if include_sign)
    """
    method = (method or "tmfg").lower().strip()

    # Build edge list (i<j) according to method
    if method == "mst":
        # Use |rho| for MST stability if requested
        E_pairs = _mst_edges_from_corr(np.abs(C)) if mst_use_abs else _mst_edges_from_corr(C)
    elif method == "tmfg":
        E_pairs = _tmfg_edges_from_corr(C)
        # NOTE: We intentionally ignore tmfg_keep_n < full TMFG to preserve planarity.
    elif method == "knn":
        k = int(knn_k) if knn_k is not None else 8
        E_pairs = _knn_edges_from_corr(C, k=k)
    elif method == "threshold":
        thr = float(threshold_abs_corr) if threshold_abs_corr is not None else 0.30
        E_pairs = _threshold_edges_from_corr(C, thr=thr)
    else:
        raise ValueError(f"Unknown method '{method}'")

    # Attributes
    if len(E_pairs) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = None
        return edge_index, edge_attr

    full_attrs = _edge_attr_from_corr(C, E_pairs)  # [rho, |rho|, sign]
    cols = [0]  # rho always
    if include_strength:
        cols.append(1)
    if include_sign:
        cols.append(2)
    comp_attrs = full_attrs[:, cols].astype(np.float32, copy=False)

    rows = []
    attrs = []
    for (i, j), a in zip(E_pairs, comp_attrs):
        rows.append((i, j))
        if undirected:
            rows.append((j, i))
        attrs.append(a)
        if undirected:
            attrs.append(a)

    edge_index = np.asarray(rows, dtype=np.int64).T
    edge_attr = np.asarray(attrs, dtype=np.float32) if attrs else None
    return edge_index, edge_attr


def corr_from_returns(
    returns_window: pd.DataFrame,
    cov_method: str = "lw",
    shrink_to: str = "diag",
    min_var: float = 1e-10,
) -> np.ndarray:
    """
    Convenience: robust covariance -> correlation using src.cov helpers.
    """
    X = returns_window.astype(float).fillna(0.0).values  # (T, N)
    S = robust_covariance(X, method=cov_method, shrink_to=shrink_to, min_var=min_var)
    C = to_correlation(S)
    C = np.clip(0.5 * (C + C.T), -0.9999, 0.9999)
    np.fill_diagonal(C, 1.0)
    return C
