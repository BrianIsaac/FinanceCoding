# src/utils.py
"""Quality-assurance helpers for prices, features, and graph snapshots.

This module provides functions you'll call from a notebook to:
1) Inspect the cleaned prices/returns panel (shape & missingness).
2) Validate per-rebalance feature files (existence, shape, NaNs).
3) Validate graph snapshots (nodes/edges, self-loops, duplicates, connectivity).
4) Check graph node list alignment against membership (active tickers at t).
5) Batch QA for all graphs and export a CSV summary.

Notes:
    - Graph snapshots are assumed to be torch-saved PyG `Data` objects with
      attributes:
        * x: [N, F] float32
        * edge_index: [2, E] long
        * edge_attr: [E, ?] (optional)
        * tickers: List[str] (or 1D tensor of strings/bytes)
        * ts: pandas.Timestamp-like (optional; we also infer from filename)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    from torch_geometric.data import Data  # type: ignore
except Exception:  # pragma: no cover
    Data = Any  # fallback typing if PyG not installed

# Optional: connectivity checks
try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None


# ----------------------------- Prices / Returns ----------------------------- #


def qa_prices_panel(
    prices_path: str,
    returns_path: str | None = None,
) -> dict[str, Any]:
    """Quick QA for the cleaned price (and optional returns) panel.

    Args:
        prices_path: Parquet path with Date × Tickers (floats), DatetimeIndex.
        returns_path: Optional parquet path with daily log returns.

    Returns:
        Dict with shape, date range, column stats, and missingness metrics.
    """
    px = pd.read_parquet(prices_path)
    if not isinstance(px.index, pd.DatetimeIndex):
        px.index = pd.to_datetime(px.index, errors="coerce")
    px = px.sort_index().sort_index(axis=1)

    out: dict[str, Any] = {
        "prices_shape": tuple(px.shape),
        "prices_start": str(px.index.min().date()) if not px.empty else None,
        "prices_end": str(px.index.max().date()) if not px.empty else None,
        "prices_cols": len(px.columns),
        "prices_overall_missing_frac": (
            float(px.isna().sum().sum() / px.size) if px.size else np.nan
        ),
        "prices_share_cols_with_any_na": (
            float((px.isna().mean() > 0).mean()) if px.shape[1] else np.nan
        ),
    }

    if returns_path and Path(returns_path).exists():
        rets = pd.read_parquet(returns_path)
        if not isinstance(rets.index, pd.DatetimeIndex):
            rets.index = pd.to_datetime(rets.index, errors="coerce")
        rets = rets.sort_index().sort_index(axis=1)
        out.update(
            {
                "returns_shape": tuple(rets.shape),
                "returns_start": str(rets.index.min().date()) if not rets.empty else None,
                "returns_end": str(rets.index.max().date()) if not rets.empty else None,
                "returns_any_inf": bool(np.isinf(rets.values).any()) if rets.size else False,
                "returns_overall_missing_frac": (
                    float(rets.isna().sum().sum() / rets.size) if rets.size else np.nan
                ),
            }
        )

        # Basic alignment sanity
        if not px.empty and not rets.empty:
            out["aligned_dates"] = bool(px.index.equals(rets.index))
            out["aligned_cols"] = bool(list(px.columns) == list(rets.columns))
    return out


# ------------------------------- Features QA -------------------------------- #


def expected_feature_path(features_dir: str, ts: pd.Timestamp) -> Path:
    """Return the expected per-rebalance feature file path for timestamp `ts`."""
    return Path(features_dir) / f"features_{ts.date()}.parquet"


def qa_features_for_date(
    features_dir: str,
    ts: pd.Timestamp,
) -> dict[str, Any]:
    """Check that a feature file exists for a given rebalance date and is clean.

    Args:
        features_dir: Directory where per-rebalance feature parquet files live.
        ts: Rebalance timestamp.

    Returns:
        Dict with presence, shape, and NaN/inf counts.
    """
    path = expected_feature_path(features_dir, ts)
    out: dict[str, Any] = {"ts": str(ts.date()), "feature_file": str(path), "exists": path.exists()}
    if not path.exists():
        return out

    feat = pd.read_parquet(path)
    out.update(
        {
            "shape": tuple(feat.shape),
            "columns": list(map(str, feat.columns)),
            "index_is_tickers": "ticker" not in feat.columns and feat.index.dtype == "object",
            "any_nan": bool(feat.isna().any().any()),
            "any_inf": (
                bool(np.isinf(feat.select_dtypes(include=[float, int]).values).any())
                if feat.size
                else False
            ),
        }
    )
    return out


# ------------------------------- Graphs QA ---------------------------------- #

_DATE_RE = re.compile(r"graph_(\d{4}-\d{2}-\d{2})\.pt$")


def list_graph_files(save_dir: str) -> list[Path]:
    """List all graph snapshot files in a directory, sorted by date if possible."""
    files = sorted(Path(save_dir).glob("graph_*.pt"))
    return files


def _infer_ts_from_name(path: Path) -> pd.Timestamp | None:
    m = _DATE_RE.search(path.name)
    if not m:
        return None
    try:
        return pd.to_datetime(m.group(1))
    except Exception:
        return None


def load_graph(path: str | Path) -> Any:
    """Load a torch-saved PyG Data object (works with PyTorch ≥2.6).

    Tries to load with `weights_only=False` (needed for full objects).
    Falls back to allowlisting PyG classes if the safe loader complains.
    """
    p = str(path)
    # 1) Preferred: explicit weights_only=False (safe—these files are yours)
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        # Older torch doesn't have weights_only kwarg
        return torch.load(p, map_location="cpu")
    except Exception:
        # 2) Allowlist PyG types and try again
        try:
            from torch.serialization import add_safe_globals  # PyTorch ≥2.6

            try:
                from torch_geometric.data.data import Data, DataEdgeAttr  # type: ignore

                add_safe_globals([Data, DataEdgeAttr])
            except Exception:
                # Fallback: at least allowlist Data
                from torch_geometric.data import Data  # type: ignore

                add_safe_globals([Data])
        except Exception:
            pass
        # Final attempt (explicitly disable weights-only again)
        return torch.load(p, map_location="cpu", weights_only=False)


@dataclass
class GraphQAResult:
    """Container of per-graph QA metrics."""

    file: str
    ts: str | None
    num_nodes: int
    num_edges: int
    num_edges_unique: int
    self_loops: int
    duplicate_edges: int
    isolated_nodes: int
    mean_degree: float
    connected: bool | None
    mst_tree_ok: bool | None
    x_shape: tuple[int, int] | None
    x_has_nan: bool
    x_has_inf: bool
    edge_attr_shape: tuple[int, int] | None
    edge_attr_has_nan: bool
    edge_attr_has_inf: bool
    membership_coverage: float | None
    extra_in_graph: int | None
    missing_from_graph: int | None


def _edges_unique_undirected(edge_index: torch.Tensor) -> tuple[int, int, int]:
    """Compute unique undirected edges, self-loops, and duplicates.

    Returns:
        (num_edges_unique, self_loops, duplicate_edges)
    """
    ei = edge_index.cpu().numpy()
    if ei.size == 0:
        return 0, 0, 0
    # For undirected, collapse (i,j) and (j,i) to a canonical tuple (min,max)
    pairs = np.vstack([ei[0], ei[1]]).T
    self_loops = int(np.sum(pairs[:, 0] == pairs[:, 1]))
    canon = np.sort(pairs, axis=1)
    # Count unique undirected edges (excluding direction duplicates)
    uniq, counts = np.unique(canon, axis=0, return_counts=True)
    # duplicates = any pair that appears more than once
    duplicate_edges = int(np.sum(counts > 1))
    num_edges_unique = int(len(uniq))
    return num_edges_unique, self_loops, duplicate_edges


def _degree_stats(
    num_nodes: int, edge_index: torch.Tensor, undirected: bool = True
) -> tuple[int, float]:
    """Compute isolated node count and mean degree."""
    deg = np.zeros(num_nodes, dtype=np.int64)
    ei = edge_index.cpu().numpy()
    if ei.size == 0:
        return num_nodes, 0.0
    # Treat as undirected by incrementing both ends
    for u, v in ei.T:
        deg[u] += 1
        deg[v] += 1
    isolated = int(np.sum(deg == 0))
    mean_deg = float(np.mean(deg))
    return isolated, mean_deg


def _to_str_list(tickers_attr: Any) -> list[str]:
    """Normalize various `tickers` attribute encodings to List[str]."""
    if tickers_attr is None:
        return []
    if isinstance(tickers_attr, list):
        return [str(x) for x in tickers_attr]
    if isinstance(tickers_attr, np.ndarray):
        return [str(x) for x in tickers_attr.tolist()]
    if torch.is_tensor(tickers_attr):
        try:
            return [str(x) for x in tickers_attr.tolist()]
        except Exception:
            return [str(x) for x in tickers_attr.cpu().numpy().tolist()]
    return [str(tickers_attr)]


def active_at(membership: pd.DataFrame, ts: pd.Timestamp) -> list[str]:
    """Return active tickers at time ts per membership intervals."""
    mem = membership.copy()
    mem["start"] = pd.to_datetime(mem["start"], errors="coerce")
    if "end" in mem.columns:
        mem["end"] = pd.to_datetime(mem["end"], errors="coerce")
    else:
        mem["end"] = pd.NaT
    mask = (mem["start"] <= ts) & ((mem["end"].isna()) | (mem["end"] >= ts))
    return sorted(set(mem.loc[mask, "ticker"].astype(str).str.upper().tolist()))


def qa_single_graph(
    path: str | Path,
    expect_undirected: bool = True,
    filter_method: str | None = "mst",
    membership_csv: str | None = None,
) -> GraphQAResult:
    """Run QA on a single graph snapshot.

    Args:
        path: Path to torch-saved PyG Data object.
        expect_undirected: If True, verify undirected properties.
        filter_method: If 'mst', check E == N-1 and connected.
        membership_csv: Optional membership CSV to check node set alignment.

    Returns:
        GraphQAResult with structural and data quality metrics.
    """
    p = Path(path)
    ts = _infer_ts_from_name(p)
    g: Data = load_graph(p)  # type: ignore

    # Core shapes
    num_nodes = int(getattr(g, "num_nodes", g.x.shape[0] if hasattr(g, "x") else 0))
    edge_index: torch.Tensor = getattr(g, "edge_index", torch.empty(2, 0, dtype=torch.long))
    num_edges_total = int(edge_index.shape[1])

    # Undirected uniqueness/self-loops/duplicates
    uniq_e, self_loops, dup_e = _edges_unique_undirected(edge_index)

    # Degree stats
    isolated, mean_deg = _degree_stats(num_nodes, edge_index, undirected=expect_undirected)

    # Connectivity / MST checks (optional)
    connected: bool | None = None
    mst_tree_ok: bool | None = None
    if nx is not None and num_nodes > 0:
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        # add undirected edges
        ei = edge_index.cpu().numpy()
        G.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
        connected = nx.is_connected(G) if G.number_of_nodes() > 0 else None
        if filter_method == "mst":
            mst_tree_ok = (uniq_e == max(0, num_nodes - 1)) and bool(connected)

    # Feature & edge_attr stats
    x = getattr(g, "x", None)
    x_shape = tuple(x.shape) if x is not None else None
    x_has_nan = bool(torch.isnan(x).any().item()) if isinstance(x, torch.Tensor) else False
    x_has_inf = bool(torch.isinf(x).any().item()) if isinstance(x, torch.Tensor) else False

    ea = getattr(g, "edge_attr", None)
    ea_shape = tuple(ea.shape) if ea is not None else None
    edge_attr_has_nan = (
        bool(torch.isnan(ea).any().item()) if isinstance(ea, torch.Tensor) else False
    )
    edge_attr_has_inf = (
        bool(torch.isinf(ea).any().item()) if isinstance(ea, torch.Tensor) else False
    )

    # Membership alignment (optional)
    membership_coverage = None
    extra_in_graph = None
    missing_from_graph = None
    if membership_csv and ts is not None and Path(membership_csv).exists():
        mem = pd.read_csv(membership_csv)
        mem["ticker"] = mem["ticker"].astype(str).str.upper()
        active = set(active_at(mem, ts))
        graph_tickers = {t.upper() for t in _to_str_list(getattr(g, "tickers", []))}
        inter = active & graph_tickers
        if active:
            membership_coverage = float(len(inter) / len(active))
        extra_in_graph = int(len(graph_tickers - active))
        missing_from_graph = int(len(active - graph_tickers))

    return GraphQAResult(
        file=str(p),
        ts=str(ts.date()) if ts is not None else None,
        num_nodes=num_nodes,
        num_edges=num_edges_total,
        num_edges_unique=uniq_e,
        self_loops=self_loops,
        duplicate_edges=dup_e,
        isolated_nodes=isolated,
        mean_degree=mean_deg,
        connected=connected,
        mst_tree_ok=mst_tree_ok,
        x_shape=x_shape,
        x_has_nan=x_has_nan,
        x_has_inf=x_has_inf,
        edge_attr_shape=ea_shape,
        edge_attr_has_nan=edge_attr_has_nan,
        edge_attr_has_inf=edge_attr_has_inf,
        membership_coverage=membership_coverage,
        extra_in_graph=extra_in_graph,
        missing_from_graph=missing_from_graph,
    )


def qa_all_graphs(
    save_dir: str,
    expect_undirected: bool = True,
    filter_method: str | None = "mst",
    membership_csv: str | None = None,
    export_csv: str | None = None,
) -> pd.DataFrame:
    """Run QA across all graph snapshots in a directory.

    Args:
        save_dir: Directory containing graph_YYYY-MM-DD.pt files.
        expect_undirected: If True, check undirected assumptions.
        filter_method: If 'mst', assert E == N-1 and connected; 'knn' skips this.
        membership_csv: Optional path to membership CSV for alignment checks.
        export_csv: Optional path to write the QA table as CSV.

    Returns:
        DataFrame of per-graph QA metrics (one row per file).
    """
    rows: list[dict[str, Any]] = []
    for f in list_graph_files(save_dir):
        res = qa_single_graph(
            f,
            expect_undirected=expect_undirected,
            filter_method=filter_method,
            membership_csv=membership_csv,
        )
        rows.append(res.__dict__)
    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)

    if export_csv:
        Path(export_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(export_csv, index=False)
    return df
