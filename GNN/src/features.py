# src/features.py
"""
Feature engineering utilities for node (asset) features at each rebalance date.

We compute a compact set of 8 hygienic features per asset, using ONLY data
available *up to and including* the rebalance date `t`. This aligns with the
labeling convention where y = next-period return.

Features (per asset, later z-scored cross-sectionally at each date):
  - r_1, r_5, r_21, r_63: cumulative returns over 1/5/21/63 trading days
  - vol_21, vol_63: realised volatility (stdev of daily returns) over 21/63d
  - mom_252: 12-month momentum (252 trading days)
  - dd_max:   worst drawdown over the past 252 trading days (≤0)

All features are computed from a daily returns matrix. Prices are optional and
only used if you prefer to recompute returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


_EPS = 1e-9


@dataclass
class FeatureConfig:
    """Configuration for feature computation."""
    winsor_q: float = 0.01          # clip each feature to [q, 1-q] cross-sectionally
    zscore: bool = True             # cross-sectional z-scoring per date
    require_min_days: int = 60      # minimum history to compute a feature; otherwise NaN


def _ensure_returns(px: Optional[pd.DataFrame], rets: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Return a daily returns DataFrame with DatetimeIndex and aligned columns."""
    if rets is None:
        if px is None:
            raise ValueError("Either `rets` or `px` must be provided.")
        px = px.sort_index()
        rets = px.pct_change().iloc[1:]
    else:
        rets = rets.sort_index()
    # Clean column names to strings (tickers)
    rets.columns = [str(c) for c in rets.columns]
    return rets


def _cumret(window: pd.DataFrame) -> pd.Series:
    """(1+r).prod - 1 per column; returns NaN where insufficient data."""
    if window.shape[0] == 0:
        return pd.Series(index=window.columns, dtype=float)
    return (1.0 + window).prod(axis=0) - 1.0


def _stdev(window: pd.DataFrame) -> pd.Series:
    if window.shape[0] == 0:
        return pd.Series(index=window.columns, dtype=float)
    return window.std(ddof=0, axis=0)


def _max_drawdown(window: pd.DataFrame) -> pd.Series:
    """Max drawdown (≤0) over the window per column."""
    if window.shape[0] == 0:
        return pd.Series(index=window.columns, dtype=float)
    eq = (1.0 + window).cumprod(axis=0)
    roll_max = eq.cummax(axis=0)
    dd = (eq / (roll_max + _EPS)) - 1.0
    return dd.min(axis=0)


def _winsorize_cs(df: pd.DataFrame, q: float) -> pd.DataFrame:
    """Clip each column cross-sectionally to its [q, 1-q] quantiles."""
    if q <= 0:
        return df
    lo = df.quantile(q, axis=0, interpolation="linear")
    hi = df.quantile(1 - q, axis=0, interpolation="linear")
    return df.clip(lower=lo, upper=hi, axis=1)


def _zscore_cs(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score (per column across assets at the same date)."""
    return (df - df.mean(axis=0)) / (df.std(ddof=0, axis=0) + _EPS)


def compute_features_at_date(
    rets: pd.DataFrame,
    t: pd.Timestamp,
    cfg: FeatureConfig,
) -> pd.DataFrame:
    """Compute features using data up to and including date `t`.

    Parameters
    ----------
    rets : pd.DataFrame
        Daily returns, index=dates, columns=tickers.
    t : pd.Timestamp
        Rebalance date (must be within the returns index).
    cfg : FeatureConfig
        Controls winsorisation and standardisation.

    Returns
    -------
    pd.DataFrame
        index=tickers, columns=[r_1, r_5, r_21, r_63, vol_21, vol_63, mom_252, dd_max].
        NaNs filled with 0 AFTER z-scoring to avoid leakage.
    """
    if t not in rets.index:
        # allow nearest previous trading day
        loc = rets.index.searchsorted(t, side="right") - 1
        if loc < 0:
            raise ValueError(f"No returns available on/before {t}.")
        t_eff = rets.index[loc]
    else:
        t_eff = t

    # Windows (inclusive of t_eff)
    w1   = rets.loc[:t_eff].tail(1)
    w5   = rets.loc[:t_eff].tail(5)
    w21  = rets.loc[:t_eff].tail(21)
    w63  = rets.loc[:t_eff].tail(63)
    w252 = rets.loc[:t_eff].tail(252)

    # Require minimum data or return NaN for that feature
    def _mask_insufficient(win: pd.DataFrame, s: pd.Series) -> pd.Series:
        ok = win.notna().sum(axis=0) >= max(1, cfg.require_min_days if len(win) >= cfg.require_min_days else len(win))
        s = s.copy()
        s[~ok] = np.nan
        return s

    r_1   = _mask_insufficient(w1,   _cumret(w1))
    r_5   = _mask_insufficient(w5,   _cumret(w5))
    r_21  = _mask_insufficient(w21,  _cumret(w21))
    r_63  = _mask_insufficient(w63,  _cumret(w63))
    vol_21 = _mask_insufficient(w21, _stdev(w21))
    vol_63 = _mask_insufficient(w63, _stdev(w63))
    mom_252 = _mask_insufficient(w252, _cumret(w252))
    dd_max   = _mask_insufficient(w252, _max_drawdown(w252))

    df = pd.DataFrame(
        {
            "r_1": r_1,
            "r_5": r_5,
            "r_21": r_21,
            "r_63": r_63,
            "vol_21": vol_21,
            "vol_63": vol_63,
            "mom_252": mom_252,
            "dd_max": dd_max,
        }
    )

    # Basic cleaning
    df = df.replace([np.inf, -np.inf], np.nan)

    # Winsorise each FEATURE cross-sectionally (per column across assets)
    if cfg.winsor_q and cfg.winsor_q > 0:
        df = _winsorize_cs(df, cfg.winsor_q)

    # Cross-sectional z-score (per feature)
    if cfg.zscore:
        df = _zscore_cs(df)

    # Fill residual NaNs with 0 for downstream tensors
    df = df.fillna(0.0)

    return df.astype(np.float32)


def rolling_features(
    px: Optional[pd.DataFrame],
    rets: Optional[pd.DataFrame],
    rebal_dates: Iterable[pd.Timestamp],
    cfg: Optional[FeatureConfig] = None,
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """Compute features for all rebalancing dates.

    Parameters
    ----------
    px : Optional[pd.DataFrame]
        (Optional) Prices; if provided and `rets` is None, returns are computed as pct_change.
    rets : Optional[pd.DataFrame]
        Daily returns. If given, `px` is ignored for returns.
    rebal_dates : Iterable[pd.Timestamp]
        Rebalancing dates (month-ends, or any dates). We will snap to the
        nearest previous trading day if the exact date is not present.
    cfg : Optional[FeatureConfig]
        Controls winsorisation and standardisation.

    Returns
    -------
    Dict[pd.Timestamp, pd.DataFrame]
        Mapping date -> features DataFrame (index=tickers, 8 columns).
    """
    cfg = cfg or FeatureConfig()
    rets = _ensure_returns(px, rets)

    feats_by_t: Dict[pd.Timestamp, pd.DataFrame] = {}
    for t in rebal_dates:
        t = pd.Timestamp(t)
        feats_by_t[t] = compute_features_at_date(rets, t, cfg)

    return feats_by_t
