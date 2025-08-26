"""Graph construction utilities for portfolio GNN experiments (paper-aligned).

Implements:
- Quarterly graph snapshots at rebalance dates.
- Node selection via (optional) dynamic membership file.
- Edge weights from distance correlation computed over 30-day realised volatility series.
- Graph sparsification via TMFG (preferred), MST, or KNN.

Notes:
    * TMFG requires an external implementation. See `_tmfg_edges` docstring.
      Until plugged in, use MST/KNN to move forward.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

try:
    import dcor  # distance correlation
except Exception as _:
    dcor = None  # handled gracefully below


@dataclass
class Membership:
    """Represents per-ticker active intervals for a dynamic universe."""
    ticker: str
    start: pd.Timestamp
    end: pd.Timestamp


def _load_membership(membership_csv: Optional[str]) -> Optional[List[Membership]]:
    """Loads a membership file describing active intervals for each ticker.

    Args:
        membership_csv: Path to CSV with columns: ticker,start,end (YYYY-MM-DD).

    Returns:
        A list of Membership records, or None if `membership_csv` is None or not found.
    """
    if not membership_csv or not os.path.exists(membership_csv):
        return None
    df = pd.read_csv(membership_csv)
    df.columns = [c.lower() for c in df.columns]
    if not {"ticker", "start", "end"}.issubset(df.columns):
        raise ValueError("membership_csv must contain columns: ticker,start,end")
    df["ticker"] = df["ticker"].str.upper().str.strip()
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    recs = [Membership(r.ticker, r.start, r.end) for r in df.itertuples(index=False)]
    return recs


def _active_tickers_at(
    ts: pd.Timestamp,
    all_tickers: Sequence[str],
    membership: Optional[List[Membership]],
    present_mask: Optional[dict[str, bool]] = None,
) -> List[str]:
    """Returns the list of active tickers at timestamp `ts`.

    Args:
        ts: Rebalance timestamp.
        all_tickers: Full columns set in the price panel.
        membership: Optional membership intervals per ticker.

    Returns:
        Subset of tickers active at `ts`. If `membership` is None, all non-NaN tickers at `ts` are used.
    """
    if membership is None:
        # fall back to "has price at ts" if provided
        if present_mask is None:
            return list(all_tickers)
        return [t for t in all_tickers if present_mask.get(t, False)]
    active = []
    ms = {}
    # Build quick lookup of intervals per ticker.
    for m in membership:
        ms.setdefault(m.ticker, []).append((m.start, m.end))
    for t in all_tickers:
        ok = False
        for s, e in ms.get(t, []):
            if s <= ts <= e:
                ok = True
                break
        if ok:
            active.append(t)
    return active


def _realised_volatility(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Computes rolling realised volatility (std of daily log returns).

    Args:
        returns: Daily log returns (Date × Tickers).
        window: Window (in trading days) for the rolling standard deviation.

    Returns:
        DataFrame of realised vol with same shape as `returns`.
    """
    return returns.rolling(window=window, min_periods=window).std()


def _pearson_sim(X: np.ndarray) -> np.ndarray:
    """Pearson correlation similarity (N×N).

    Args:
        X: Matrix of shape (T, N) where columns are time series.

    Returns:
        Symmetric similarity matrix in [-1, 1] with ones on the diagonal.
    """
    C = np.corrcoef(X, rowvar=False)
    np.fill_diagonal(C, 1.0)
    return C


def _distance_corr_sim(X: np.ndarray) -> np.ndarray:
    """Distance correlation similarity (N×N).

    Args:
        X: Matrix of shape (T, N) where columns are time series.

    Returns:
        Symmetric matrix with diagonal = 1.0 and off-diagonals ∈ [0, 1].

    Raises:
        RuntimeError: If `dcor` is not installed.
    """
    if dcor is None:
        raise RuntimeError("dcor is not installed. `pip install dcor` or set graph.similarity=pearson.")
    N = X.shape[1]
    S = np.eye(N, dtype=float)
    for i in range(N):
        xi = X[:, i]
        for j in range(i + 1, N):
            xj = X[:, j]
            dc = float(dcor.distance_correlation(xi, xj))
            S[i, j] = S[j, i] = dc
    return S


def _mst_edges(sim: np.ndarray) -> List[Tuple[int, int, float]]:
    """Minimum Spanning Tree edges built from a similarity matrix.

    Args:
        sim: Similarity matrix (N×N).

    Returns:
        List of (u, v, weight) where weight is original similarity.
    """
    dist = 1.0 - np.clip(sim, -1.0, 1.0)
    G = nx.from_numpy_array(dist)
    T = nx.minimum_spanning_tree(G, weight="weight", algorithm="kruskal")
    edges: List[Tuple[int, int, float]] = []
    for u, v in T.edges():
        edges.append((u, v, float(sim[u, v])))
    return edges


def _knn_edges(sim: np.ndarray, k: int) -> List[Tuple[int, int, float]]:
    """KNN graph on similarities (symmetrised).

    Args:
        sim: Similarity matrix (N×N).
        k: Number of neighbors per node.

    Returns:
        List of unique undirected edges (u, v, sim).
    """
    N = sim.shape[0]
    k = max(1, min(k, N - 1))
    S = np.copy(sim)
    np.fill_diagonal(S, -np.inf)
    edges: set[Tuple[int, int]] = set()
    for i in range(N):
        nbrs = np.argsort(-S[i])[:k]
        for j in nbrs:
            u, v = (i, j) if i < j else (j, i)
            edges.add((u, v))
    return [(u, v, float(sim[u, v])) for (u, v) in sorted(edges)]


def _tmfg_edges(sim: np.ndarray) -> List[Tuple[int, int, float]]:
    """Edges via Triangulated Maximally Filtered Graph (TMFG).

    IMPORTANT:
        This function expects an external TMFG implementation. Options:
        - Install a dedicated TMFG package (if available in your environment).
        - Call out to a command-line tool and parse the resulting edges.
        - Replace this function once you choose a concrete TMFG implementation.

    Args:
        sim: Similarity matrix (N×N).

    Returns:
        List of (u, v, weight) edges.

    Raises:
        NotImplementedError: Always, until a TMFG implementation is wired.
    """
    raise NotImplementedError(
        "TMFG filter not implemented. "
        "Set graph.filter.method=mst or knn for now, or plug in a TMFG library here."
    )


def _edge_index(
    edges: List[Tuple[int, int, float]],
    undirected: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Builds PyG edge_index and edge_attr tensors.

    Args:
        edges: List of (u, v, weight).
        undirected: Whether to add reverse edges.

    Returns:
        Tuple (edge_index [2,E], edge_attr [E,1]).
    """
    rows: List[int] = []
    cols: List[int] = []
    weights: List[float] = []
    for u, v, w in edges:
        rows.append(u)
        cols.append(v)
        weights.append(w)
        if undirected:
            rows.append(v)
            cols.append(u)
            weights.append(w)
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_attr = torch.tensor(weights, dtype=torch.float32).unsqueeze(-1)
    return edge_index, edge_attr


def build_quarterly_graphs(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    rebalance_dates: Iterable[pd.Timestamp],
    lookback_days: int,
    vol_window_days: int,
    similarity: str,
    filter_method: str,
    k_neighbors: int,
    undirected: bool,
    features_dir: str,
    membership_csv: Optional[str] = None,
) -> Dict[pd.Timestamp, Data]:
    """Builds PyG graph snapshots aligned with the paper’s methodology.

    For each rebalance date t:
      1) Select a 3-year lookback slice of daily returns up to t.
      2) Compute 30-day realised volatility per asset within that slice.
      3) Compute pairwise distance correlation across those volatility series.
      4) Sparsify via TMFG (preferred) or MST/KNN.
      5) Attach per-asset feature vectors from `features/features_t.parquet`.

    Args:
        prices: Cleaned price panel (Date × Tickers).
        returns: Daily log returns (Date × Tickers).
        rebalance_dates: Quarterly rebalance timestamps.
        lookback_days: Number of daily observations for similarity (≈756).
        vol_window_days: Window for rolling volatility (≈30).
        similarity: "distance_corr" (preferred) or "pearson".
        filter_method: "tmfg" | "mst" | "knn".
        k_neighbors: K for KNN sparsification (if used).
        undirected: Whether to add reverse edges to edge_index.
        features_dir: Directory where per-rebalance features are stored.
        membership_csv: Optional CSV (ticker,start,end) to enforce a dynamic universe.

    Returns:
        Mapping from rebalance timestamp to PyG Data object:
            Data.x: [N,F] node features,
            Data.edge_index: [2,E] edges,
            Data.edge_attr: [E,1] weights,
            Data.tickers: list[str] (kept as a Python attribute),
            Data.ts: str (YYYY-MM-DD).
    """
    membership = _load_membership(membership_csv)
    graphs: Dict[pd.Timestamp, Data] = {}

    all_tickers = list(prices.columns)

    for ts in pd.to_datetime(list(rebalance_dates)):
        # 1) Lookback window (skip early quarters)
        hist_r = returns.loc[:ts].tail(lookback_days)
        if hist_r.shape[0] < lookback_days:
            continue

        # 2) Dynamic universe: which tickers are active at ts?
        last_row = prices.loc[:ts].iloc[-1]
        present_mask = {c: pd.notna(last_row.get(c, np.nan)) for c in all_tickers}
        if membership is not None:
            active = _active_tickers_at(ts, all_tickers, membership, present_mask)
        else:
            active = [c for c in all_tickers if present_mask.get(c, False)]
        if len(active) < 5:
            continue  # too few assets to form a meaningful graph

        # Restrict to active tickers
        R = hist_r[active]

        # 3) 30-day realised volatility series within lookback
        vol = _realised_volatility(R, window=vol_window_days).dropna(how="any")
        if vol.shape[0] < 20:  # sanity guard
            continue

        X = vol.to_numpy()  # (T, N)
        if similarity.lower() == "distance_corr":
            S = _distance_corr_sim(X)
        elif similarity.lower() == "pearson":
            S = _pearson_sim(X)
        else:
            raise ValueError(f"Unknown similarity: {similarity}")

        # 4) Sparsify to obtain a thin, stable topology
        method = filter_method.lower()
        if method == "tmfg":
            edges = _tmfg_edges(S)  # requires a concrete TMFG implementation
        elif method == "mst":
            edges = _mst_edges(S)
        elif method == "knn":
            edges = _knn_edges(S, k_neighbors)
        else:
            raise ValueError(f"Unknown filter_method: {filter_method}")

        # 5) Node features at ts
        fpath = os.path.join(features_dir, f"features_{ts.date()}.parquet")
        if not os.path.exists(fpath):
            # If features missing for this quarter, skip gracefully
            continue
        feats_df = pd.read_parquet(fpath).reindex(active).fillna(0.0)
        X_node = torch.tensor(feats_df.to_numpy(), dtype=torch.float32)

        edge_index, edge_attr = _edge_index(edges, undirected=undirected)

        data_obj = Data(
            x=X_node,
            edge_index=edge_index,
            edge_attr=edge_attr,
            tickers=active,   # PyG will keep this as an attribute
            ts=str(ts.date())
        )
        graphs[ts] = data_obj

    return graphs
