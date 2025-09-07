from __future__ import annotations

import hashlib
import math
import os
import pickle
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch

try:
    # torch_geometric >= 2.6
    from torch_geometric.data import Data  # type: ignore
except Exception:
    # older import path fallback
    from torch_geometric.data.data import Data  # type: ignore

from src.data.processors.covariance import robust_covariance, to_correlation

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
    cov_method: Literal["lw", "oas", "sample"] = "lw"  # Ledoit–Wolf (default)
    shrink_to: Literal["diag", "identity"] = "diag"
    min_var: float = 1e-10  # variance floor
    corr_method: Literal["from_cov"] = "from_cov"
    filter_method: Literal["mst", "tmfg", "knn", "threshold"] = "mst"
    knn_k: int = 8
    threshold_abs_corr: float = 0.30  # used for threshold mode
    use_edge_attr: bool = True  # include [rho, |rho|, sign] if True

    # Enhanced parameters for dynamic universe handling
    enable_caching: bool = True  # Enable graph snapshot caching
    cache_ttl_days: int = 30  # Time-to-live for cached graphs
    adaptive_knn: bool = True  # Adaptive k based on universe size
    min_knn_k: int = 5  # Minimum k for small universes
    max_knn_k: int = 50  # Maximum k for large universes
    universe_stability_threshold: float = 0.05  # Threshold for universe change detection

    # Memory optimization parameters
    max_edges_per_node: int = 50  # Maximum edges per node for memory efficiency
    edge_pruning_threshold: float = 0.01  # Minimum edge weight to keep

    # Multiple graph construction support
    multi_graph_methods: list[str] | None = None  # ['knn', 'mst', 'tmfg'] for ensemble
    ensemble_weights: list[float] | None = None  # Weights for ensemble combination


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


def _mst_edges_from_corr(C: np.ndarray) -> list[tuple[int, int]]:
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
    edges: list[tuple[int, int]] = []

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


def _greedy_init_k4(A: np.ndarray) -> list[int]:
    """
    Greedy K4 initializer for TMFG on |correlation| matrix A (diagonal should be -inf).
    Heuristic: pick node with largest degree (row sum), then add nodes that maximize
    cumulative connection strength to the current set.
    """
    n = A.shape[0]
    if n < 4:
        return list(range(min(n, 4)))

    chosen: list[int] = []
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


def _tmfg_edges_from_corr(C: np.ndarray) -> list[tuple[int, int]]:
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
    edges: set[tuple[int, int]] = set()

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
    faces: list[tuple[int, int, int]] = [
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

    return sorted(edges)


def _knn_edges_from_corr(C: np.ndarray, k: int) -> list[tuple[int, int]]:
    """
    Symmetric k-NN graph on absolute correlation.
    Keep the top-|rho| neighbors for each node; symmetrise.
    """
    n = C.shape[0]
    if n <= 1 or k <= 0:
        return []
    A = np.abs(C).copy()
    np.fill_diagonal(A, -np.inf)

    nbrs: list[set] = [set() for _ in range(n)]
    for i in range(n):
        kk = min(k, n - 1)
        if kk <= 0:
            continue
        idx = np.argpartition(-A[i], kth=kk - 1)[:kk]
        for j in idx:
            if i != j:
                nbrs[i].add(int(j))

    edges: set[tuple[int, int]] = set()
    for i in range(n):
        for j in nbrs[i]:
            if i in nbrs[j]:  # symmetric
                a, b = (i, j) if i < j else (j, i)
                edges.add((a, b))
    return sorted(edges)


def _threshold_edges_from_corr(C: np.ndarray, thr: float) -> list[tuple[int, int]]:
    """
    Keep edges with |rho| >= threshold.
    """
    n = C.shape[0]
    E: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(C[i, j]) >= thr:
                E.append((i, j))
    return E


def _edge_attr_from_corr(C: np.ndarray, E: list[tuple[int, int]]) -> np.ndarray:
    """
    Build edge attributes:
      [rho, abs_rho, sign]
    """
    attrs = []
    for i, j in E:
        rho = float(np.clip(C[i, j], -1.0, 1.0))
        attrs.append([rho, abs(rho), float(np.sign(rho))])
    return np.asarray(attrs, dtype=np.float32)


# --------------------------- dynamic universe helpers ----------------------------


def _compute_adaptive_knn_k(universe_size: int, cfg: GraphBuildConfig) -> int:
    """
    Compute adaptive k-NN parameter based on universe size.
    
    Args:
        universe_size: Number of assets in universe
        cfg: Graph build configuration
        
    Returns:
        Optimal k for k-NN graph construction
    """
    if not cfg.adaptive_knn:
        return cfg.knn_k

    # Scale k based on universe size with logarithmic scaling
    base_k = max(cfg.min_knn_k, int(np.sqrt(universe_size)))
    scaled_k = min(cfg.max_knn_k, base_k)

    # Ensure we don't exceed universe constraints
    max_possible_k = max(1, universe_size - 1)
    return min(scaled_k, max_possible_k)


def _prune_edges_by_weight(
    edges: list[tuple[int, int]],
    edge_attrs: np.ndarray,
    cfg: GraphBuildConfig
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """
    Prune edges based on correlation strength and memory constraints.
    
    Args:
        edges: List of edge tuples
        edge_attrs: Edge attributes array
        cfg: Graph build configuration
        
    Returns:
        Pruned edges and attributes
    """
    if not edges or cfg.edge_pruning_threshold <= 0:
        return edges, edge_attrs

    # Calculate edge strengths (absolute correlation)
    edge_strengths = np.abs(edge_attrs[:, 0])  # Use correlation magnitude

    # Keep edges above threshold
    keep_mask = edge_strengths >= cfg.edge_pruning_threshold

    if keep_mask.sum() == 0:
        # If no edges pass threshold, keep the strongest ones
        n_keep = min(len(edges), len(edges) // 2)
        top_indices = np.argsort(edge_strengths)[-n_keep:]
        keep_mask = np.zeros(len(edges), dtype=bool)
        keep_mask[top_indices] = True

    pruned_edges = [edges[i] for i in range(len(edges)) if keep_mask[i]]
    pruned_attrs = edge_attrs[keep_mask]

    return pruned_edges, pruned_attrs


def _enforce_max_edges_per_node(
    edges: list[tuple[int, int]],
    edge_attrs: np.ndarray,
    n_nodes: int,
    max_edges: int
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """
    Enforce maximum edges per node constraint for memory efficiency.
    
    Args:
        edges: List of edge tuples
        edge_attrs: Edge attributes array  
        n_nodes: Number of nodes
        max_edges: Maximum edges per node
        
    Returns:
        Filtered edges and attributes
    """
    if max_edges <= 0 or not edges:
        return edges, edge_attrs

    # Count edges per node
    edge_counts = [0] * n_nodes
    node_edges = [[] for _ in range(n_nodes)]

    for idx, (i, j) in enumerate(edges):
        edge_counts[i] += 1
        edge_counts[j] += 1
        node_edges[i].append((idx, j, np.abs(edge_attrs[idx, 0])))  # Store strength
        node_edges[j].append((idx, i, np.abs(edge_attrs[idx, 0])))

    # Keep strongest edges for each overconnected node
    keep_indices = set()

    for node in range(n_nodes):
        if edge_counts[node] <= max_edges:
            # Add all edges for this node
            for edge_idx, _, _ in node_edges[node]:
                keep_indices.add(edge_idx)
        else:
            # Keep only the strongest edges
            sorted_edges = sorted(node_edges[node], key=lambda x: x[2], reverse=True)
            for edge_idx, _, _ in sorted_edges[:max_edges]:
                keep_indices.add(edge_idx)

    # Filter edges and attributes
    filtered_edges = [edges[i] for i in sorted(keep_indices)]
    filtered_attrs = edge_attrs[sorted(keep_indices)]

    return filtered_edges, filtered_attrs


def _build_ensemble_graph(
    C: np.ndarray,
    methods: list[str],
    weights: list[float] | None,
    cfg: GraphBuildConfig
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """
    Build ensemble graph combining multiple construction methods.
    
    Args:
        C: Correlation matrix
        methods: List of graph construction methods
        weights: Weights for combining methods (None for equal weights)
        cfg: Graph build configuration
        
    Returns:
        Combined edges and attributes
    """
    if not methods:
        methods = ["mst"]

    if weights is None:
        weights = [1.0 / len(methods)] * len(methods)
    elif len(weights) != len(methods):
        raise ValueError("Number of weights must match number of methods")

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    all_edges: set[tuple[int, int]] = set()
    edge_scores: dict[tuple[int, int], float] = {}

    # Build graphs with different methods
    for method, weight in zip(methods, weights):
        if method == "mst":
            edges = _mst_edges_from_corr(C)
        elif method == "tmfg":
            edges = _tmfg_edges_from_corr(C)
        elif method == "knn":
            k = _compute_adaptive_knn_k(C.shape[0], cfg)
            edges = _knn_edges_from_corr(C, k)
        elif method == "threshold":
            edges = _threshold_edges_from_corr(C, cfg.threshold_abs_corr)
        else:
            continue

        # Add edges with weighted scores
        for edge in edges:
            all_edges.add(edge)
            if edge in edge_scores:
                edge_scores[edge] += weight
            else:
                edge_scores[edge] = weight

    # Sort edges by combined score
    sorted_edges = sorted(all_edges, key=lambda e: edge_scores[e], reverse=True)

    # Build attributes
    edge_attrs = _edge_attr_from_corr(C, sorted_edges)

    return sorted_edges, edge_attrs


# --------------------------- graph caching ----------------------------


class GraphCache:
    """Graph snapshot caching for efficient monthly rebalancing."""

    def __init__(self, cache_dir: str = ".cache/graphs", ttl_days: int = 30):
        """
        Initialize graph cache.
        
        Args:
            cache_dir: Directory to store cached graphs
            ttl_days: Time-to-live for cached graphs in days
        """
        self.cache_dir = cache_dir
        self.ttl_days = ttl_days
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(
        self,
        tickers: list[str],
        ts: pd.Timestamp,
        cfg: GraphBuildConfig,
        returns_hash: str
    ) -> str:
        """Generate cache key for graph configuration."""
        # Create deterministic hash from configuration
        config_str = (
            f"{sorted(tickers)}-{ts.strftime('%Y%m%d')}-{cfg.lookback_days}-"
            f"{cfg.filter_method}-{cfg.knn_k}-{cfg.threshold_abs_corr}-"
            f"{cfg.cov_method}-{cfg.shrink_to}-{returns_hash}"
        )
        return hashlib.md5(config_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> str:
        """Get file path for cached graph."""
        return os.path.join(self.cache_dir, f"graph_{cache_key}.pkl")

    def _is_cache_valid(self, cache_path: str) -> bool:
        """Check if cached graph is still valid based on TTL."""
        if not os.path.exists(cache_path):
            return False

        # Check file age
        file_age_days = (pd.Timestamp.now() - pd.Timestamp.fromtimestamp(os.path.getmtime(cache_path))).days
        return file_age_days < self.ttl_days

    def _hash_returns_window(self, returns_window: pd.DataFrame) -> str:
        """Create hash of returns data for cache validation."""
        # Use a small sample of the returns data to create hash
        sample_data = returns_window.iloc[::max(1, len(returns_window)//100)].values  # Sample every 100th row
        return hashlib.md5(sample_data.tobytes()).hexdigest()[:16]

    def get_cached_graph(
        self,
        returns_window: pd.DataFrame,
        tickers: list[str],
        ts: pd.Timestamp,
        cfg: GraphBuildConfig
    ) -> Data | None:
        """Retrieve cached graph if available and valid."""
        if not cfg.enable_caching:
            return None

        returns_hash = self._hash_returns_window(returns_window)
        cache_key = self._get_cache_key(tickers, ts, cfg, returns_hash)
        cache_path = self._get_cache_path(cache_key)

        if not self._is_cache_valid(cache_path):
            return None

        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                # Validate that cached data matches current request
                if (cached_data.get('tickers') == tickers and
                    cached_data.get('timestamp') == ts):
                    return cached_data['graph']
        except Exception:
            # If cache is corrupted, ignore and rebuild
            pass

        return None

    def cache_graph(
        self,
        graph: Data,
        returns_window: pd.DataFrame,
        tickers: list[str],
        ts: pd.Timestamp,
        cfg: GraphBuildConfig
    ) -> None:
        """Cache graph for future use."""
        if not cfg.enable_caching:
            return

        returns_hash = self._hash_returns_window(returns_window)
        cache_key = self._get_cache_key(tickers, ts, cfg, returns_hash)
        cache_path = self._get_cache_path(cache_key)

        cache_data = {
            'graph': graph,
            'tickers': tickers,
            'timestamp': ts,
            'config_hash': cache_key
        }

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception:
            # Fail silently if caching doesn't work
            pass

    def clear_expired_cache(self) -> int:
        """Clear expired cache entries and return number of files removed."""
        removed_count = 0

        try:
            for filename in os.listdir(self.cache_dir):
                if filename.startswith("graph_") and filename.endswith(".pkl"):
                    file_path = os.path.join(self.cache_dir, filename)
                    if not self._is_cache_valid(file_path):
                        os.remove(file_path)
                        removed_count += 1
        except Exception:
            pass

        return removed_count


# Global cache instance
_graph_cache = GraphCache()


# --------------------------- main entry ----------------------------


def build_graph_from_returns(
    returns_window: pd.DataFrame,
    features_matrix: np.ndarray | None,
    tickers: list[str],
    ts: pd.Timestamp,
    cfg: GraphBuildConfig,
) -> Data:
    """
    Build graph from returns data with enhanced dynamic universe handling and caching.
    
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
        Enhanced configuration with dynamic universe and caching options.

    Returns
    -------
    torch_geometric.data.Data
        Data(x, edge_index, edge_attr?, tickers=<list[str]>, ts=<timestamp>)
        Enhanced with dynamic universe handling and caching support.
    """
    # Check cache first
    if cfg.enable_caching:
        cached_graph = _graph_cache.get_cached_graph(returns_window, tickers, ts, cfg)
        if cached_graph is not None:
            return cached_graph
    # Align and clean
    rets = returns_window.reindex(columns=tickers, fill_value=np.nan).astype(float)
    rets = rets.fillna(0.0)  # conservative fill
    X = rets.values  # (T, N)
    N = X.shape[1]  # Number of assets

    # Robust covariance -> correlation
    S = robust_covariance(X, method=cfg.cov_method, shrink_to=cfg.shrink_to, min_var=cfg.min_var)
    C = to_correlation(S)

    # Enhanced filtering with dynamic universe handling
    if cfg.multi_graph_methods is not None:
        # Use ensemble graph construction
        E, edge_attr = _build_ensemble_graph(C, cfg.multi_graph_methods, cfg.ensemble_weights, cfg)
    else:
        # Single method graph construction with adaptive parameters
        if cfg.filter_method == "mst":
            E = _mst_edges_from_corr(C)
        elif cfg.filter_method == "tmfg":
            E = _tmfg_edges_from_corr(C)
        elif cfg.filter_method == "knn":
            k = _compute_adaptive_knn_k(N, cfg)  # Use adaptive k
            E = _knn_edges_from_corr(C, k=k)
        elif cfg.filter_method == "threshold":
            E = _threshold_edges_from_corr(C, thr=float(cfg.threshold_abs_corr))
        else:
            raise ValueError(f"Unknown filter_method={cfg.filter_method}")

        # Build edge attributes
        edge_attr = _edge_attr_from_corr(C, E) if cfg.use_edge_attr else None

    # Apply edge pruning and memory constraints
    if E and cfg.use_edge_attr and edge_attr is not None:
        # Prune weak edges
        E, edge_attr = _prune_edges_by_weight(E, edge_attr, cfg)

        # Enforce memory constraints
        if cfg.max_edges_per_node > 0:
            E, edge_attr = _enforce_max_edges_per_node(E, edge_attr, N, cfg.max_edges_per_node)

    # Edge index & attributes
    if len(E) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr_t = None
    else:
        edge_index = torch.tensor(np.array(E, dtype=np.int64).T, dtype=torch.long)
        if cfg.use_edge_attr and edge_attr is not None:
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

    # Cache the graph for future use
    if cfg.enable_caching:
        _graph_cache.cache_graph(data, returns_window, tickers, ts, cfg)

    return data


# ----------------------- convenience wrapper -----------------------


def build_period_graph(
    returns_daily: pd.DataFrame,
    period_end: pd.Timestamp,
    tickers: list[str],
    features_matrix: np.ndarray | None,
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
    window = returns_daily.iloc[start : idx_end + 1]
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
