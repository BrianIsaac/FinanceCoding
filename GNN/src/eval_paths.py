from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import pandas as pd
import torch
from torch import nn

# local helpers
from src.metrics import compute_metrics_from_returns


# ---------- basic utility: capped-simplex projection ----------

def _project_capped_simplex(v: torch.Tensor, cap: float, s: float = 1.0, iters: int = 50) -> torch.Tensor:
    """
    Project v onto { w: 0<=w<=cap, sum w = s } via bisection in tau, where w_i = clip(v_i - tau, 0, cap).
    Deterministic, differentiable almost everywhere (we only use it for evaluation here).
    """
    vmax = v.detach().max().item()
    hi = vmax
    lo = hi - cap
    for _ in range(64):  # ensure sum >= s bracket
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
    """
    Ensure feasibility of sum(w)=1 with per-name cap: need n_eff * cap >= 1.
    If infeasible, raise cap minimally to 1/n_eff (+tiny eps).
    """
    if n_eff <= 0:
        return cap
    min_cap = 1.0 / n_eff + 1e-9
    return cap if cap >= min_cap else min_cap


# ---------- stitching helper (daily-return stream across windows) ----------

def stitch_daily_returns(
    weights_by_window: List[pd.Series],
    window_dates: List[Tuple[pd.Timestamp, pd.Timestamp]],
    returns_daily: pd.DataFrame,
    turnover_bps: float,
) -> pd.Series:
    """
    Apply the given per-window weights to the daily returns, charging turnover cost on the
    first trading day of each window. Windows must be ordered and non-overlapping.
    """
    assert len(weights_by_window) == len(window_dates)
    daily_chunks: List[pd.Series] = []
    prev_w: Optional[pd.Series] = None
    tc_decimal = float(turnover_bps) / 10000.0

    for w, (start_trading, end_trading) in zip(weights_by_window, window_dates):
        win = returns_daily.loc[start_trading:end_trading]
        # drop columns that are all-NaN in the window
        valid_mask = ~win.isna().all(axis=0)
        w = w[valid_mask.index[valid_mask]]
        w_sum = float(w.sum())
        if w_sum > 0:
            w = w / w_sum

        # turnover vs previous window
        if prev_w is None:
            tc = 0.0
        else:
            aligned = pd.concat([prev_w.rename("prev"), w.rename("curr")], axis=1).fillna(0.0)
            tc = float((aligned["curr"] - aligned["prev"]).abs().sum()) * tc_decimal

        r_p = win.reindex(columns=w.index, fill_value=0.0).mul(w, axis=1).sum(axis=1)
        if len(r_p) > 0 and tc > 0:
            r0 = r_p.iloc[0]
            r_p.iloc[0] = (1.0 + r0) * (1.0 - tc) - 1.0

        daily_chunks.append(r_p)
        prev_w = w

    if not daily_chunks:
        return pd.Series(dtype=float)

    r_all = pd.concat(daily_chunks).sort_index()
    r_all = r_all[~r_all.index.duplicated(keep="first")]
    return r_all


# ---------- Top-k EW evaluation ----------

def _scores_from_model_output(
    head: str,
    model_out: torch.Tensor,
) -> torch.Tensor:
    """
    Use the model output as a ranking score. For 'direct', the output is a weight-like vector;
    for 'markowitz', it's mu_hat (expected returns). Both are sensible for ranking.
    """
    return model_out.detach()


def _build_topk_equal_weight(
    scores: torch.Tensor,
    tickers: List[str],
    k: int,
    cap: float,
) -> pd.Series:
    """
    Select top-k by score and assign equal weights (then project to respect per-name cap).
    """
    k_eff = max(1, min(k, len(tickers)))
    idx = torch.topk(scores, k=k_eff).indices
    w = torch.zeros_like(scores)
    w[idx] = 1.0 / float(k_eff)

    # respect cap and sum-to-one
    cap_eff = _auto_feasible_cap(float(cap), k_eff)
    w = _project_capped_simplex(w, cap=cap_eff, s=1.0, iters=50).cpu().numpy()
    return pd.Series(w.astype(float), index=tickers)


def evaluate_topk_ew(
    model: nn.Module,
    device: torch.device,
    samples: List,  # expects objects with .ts, .graph_path; we don't import the Sample dataclass to avoid cycles
    returns_daily: pd.DataFrame,
    turnover_bps: float,
    *,
    head: str,
    topk: int,
    cap: float = 0.02,
    out_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Mirror the Markowitz evaluation, but allocate by Top-k Equal-Weight (cap-respecting).
    Saves:
      - topk_ew_equity.csv              (per-window log with equity progression)
      - topk_ew_daily_returns.csv       (stitched daily returns)
      - topk_ew_equity_daily.csv        (stitched equity curve)
      - topk_ew_weights/weights_*.parquet (per-window weights)
    Returns summary dict with final equity and annualized metrics.
    """
    model.eval()
    dates: pd.DatetimeIndex = returns_daily.index

    equity = 1.0
    rows: List[Dict] = []
    weights_by_window: List[pd.Series] = []
    window_dates: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

    with torch.no_grad():
        samples_sorted = sorted(samples, key=lambda s: s.ts)
        for i, s in enumerate(samples_sorted):
            # map this rebalance snapshot to trading span
            end_trading = dates[-1] if i + 1 == len(samples_sorted) else dates[dates.searchsorted(samples_sorted[i+1].ts, "left") - 1]
            start_idx = dates.searchsorted(s.ts, "right")
            if start_idx >= len(dates) or end_trading < dates[start_idx]:
                continue

            # load graph snapshot (torch_geometric Data)
            g = torch.load(str(s.graph_path), map_location="cpu")
            tickers: List[str] = list(getattr(g, "tickers", []))

            x = g.x.to(device)
            edge_index = g.edge_index.to(device)
            eattr = getattr(g, "edge_attr", None)
            if eattr is not None:
                eattr = eattr.to(device).to(x.dtype)

            mask_all = torch.ones(len(tickers), dtype=torch.bool, device=device)
            out, _ = model(x, edge_index, mask_all, eattr)

            scores = _scores_from_model_output(head, out)
            w_series = _build_topk_equal_weight(scores.cpu(), tickers, k=topk, cap=cap)

            # accumulate for stitched daily returns
            weights_by_window.append(w_series)
            window_dates.append((dates[start_idx], end_trading))

            # compute period gross (only for logging/equity progression here)
            win = returns_daily.loc[dates[start_idx]:end_trading]
            valid_mask = ~win.isna().all(axis=0)
            w = w_series[valid_mask.index[valid_mask]]
            w = w / float(w.sum()) if float(w.sum()) > 0 else w
            daily = win.reindex(columns=w.index, fill_value=0.0).mul(w, axis=1).sum(axis=1)
            period_gross = float((1.0 + daily).prod())  # TC applied later in stitching

            equity *= period_gross
            rows.append({
                "rebalance": s.ts.date(),
                "start_trading": dates[start_idx].date(),
                "end_trading": end_trading.date(),
                "n_nodes": len(tickers),
                "period_gross": period_gross,
                "equity_topk_ew": equity,
            })

            if out_dir is not None:
                wdir = out_dir / "topk_ew_weights"
                wdir.mkdir(parents=True, exist_ok=True)
                w.rename("weight").to_frame().to_parquet(wdir / f"weights_{s.ts.date()}.parquet")

    # stitched daily returns (with turnover costs window-to-window)
    r_all = stitch_daily_returns(weights_by_window, window_dates, returns_daily, turnover_bps=turnover_bps)

    # persist outputs
    if out_dir is not None:
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / "topk_ew_equity.csv", index=False)
        if not r_all.empty:
            r_all.rename("r").to_frame().to_csv(out_dir / "topk_ew_daily_returns.csv")
            (1.0 + r_all).cumprod().rename("equity").to_frame().to_csv(out_dir / "topk_ew_equity_daily.csv")

    # summary metrics
    if r_all.empty:
        return {"final_equity": float("nan"), "note": "no returns stitched"}
    metrics = compute_metrics_from_returns(r_all)
    final_equity = float((1.0 + r_all).prod())
    return {"final_equity": final_equity, **metrics}
