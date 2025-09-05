# src/metrics.py
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import skew as _skew, kurtosis as _kurtosis, norm


# ===============================================================
# Annualised performance metrics from daily returns
# ===============================================================

def compute_metrics_from_returns(
    r_daily: pd.Series,
    *,
    trading_days: int = 252,
) -> Dict[str, float]:
    """
    Compute standard metrics from a daily return series.
    Expects percentage (not log) returns, e.g., 0.01 for +1%.
    """
    r = pd.Series(r_daily).dropna().astype(float)
    if r.empty:
        return {"CAGR": np.nan, "AnnMean": np.nan, "AnnVol": np.nan, "Sharpe": np.nan, "MDD": np.nan}

    equity = (1.0 + r).cumprod()
    n = len(r)
    cagr    = float(equity.iloc[-1] ** (trading_days / max(n, 1)) - 1.0)
    annmean = float(r.mean() * trading_days)
    annvol  = float(r.std(ddof=0) * np.sqrt(trading_days))
    sharpe  = float(annmean / annvol) if annvol > 0 else np.nan

    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    mdd = float(dd.min())

    return {"CAGR": cagr, "AnnMean": annmean, "AnnVol": annvol, "Sharpe": sharpe, "MDD": mdd}


# ===============================================================
# Probabilistic Sharpe & a conservative "deflated" variant
# (Bailey & López de Prado style PSR; DSR via multiple-testing adj.)
# ===============================================================

def probabilistic_sharpe_ratio(
    sharpe: float,
    *,
    n_obs: int,
    sr_benchmark: float = 0.0,
    skew: Optional[float] = None,
    kurt: Optional[float] = None,
) -> float:
    """
    Probabilistic Sharpe Ratio (PSR): probability Sharpe > sr_benchmark,
    accounting for finite-sample effects, skewness, and kurtosis.

    Formula (Bailey 2012):
      z = (SR - SR*) * sqrt(n - 1) / sqrt(1 - γ3*SR + (γ4 - 1)/4 * SR^2)
      PSR = Φ(z)

    where γ3 is skewness and γ4 is kurtosis (not excess).

    Parameters
    ----------
    sharpe : observed Sharpe (annualised, consistent with others)
    n_obs : number of return observations (daily count)
    sr_benchmark : comparison Sharpe, default 0
    skew : sample skewness of returns (if None, estimated)
    kurt : sample kurtosis (NOT excess; if None, estimated)

    Returns
    -------
    PSR in [0,1]
    """
    if n_obs is None or n_obs < 2 or not np.isfinite(sharpe):
        return np.nan

    # If skew/kurt not provided, assume normal returns as a fallback:
    g3 = 0.0 if skew is None or not np.isfinite(skew) else float(skew)
    g4 = 3.0 if kurt is None or not np.isfinite(kurt) else float(kurt)

    denom = np.sqrt(max(1e-12, 1.0 - g3 * sharpe + 0.25 * (g4 - 1.0) * (sharpe ** 2)))
    z = (sharpe - sr_benchmark) * np.sqrt(max(1.0, n_obs - 1)) / denom
    return float(norm.cdf(z))


def deflated_sharpe_ratio(
    sharpe: float,
    *,
    n_obs: int,
    num_trials: int = 1,
    sr_benchmark: float = 0.0,
    skew: Optional[float] = None,
    kurt: Optional[float] = None,
) -> float:
    """
    A conservative "deflated" probability that SR > sr_benchmark,
    by Bonferroni-adjusting the PSR for multiple trials (model configs, rolls, seeds).

      PSR  = probabilistic_sharpe_ratio(...)
      p    = 1 - PSR
      p*   = min(1, num_trials * p)          # Bonferroni
      DSR  = 1 - p*

    If you have a more precise N_eff (effective number of independent trials),
    pass it in `num_trials`. This is conservative and simple to interpret.
    """
    psr = probabilistic_sharpe_ratio(
        sharpe, n_obs=n_obs, sr_benchmark=sr_benchmark, skew=skew, kurt=kurt
    )
    if not np.isfinite(psr):
        return np.nan
    p = max(0.0, 1.0 - psr)
    p_adj = min(1.0, num_trials * p)
    return float(1.0 - p_adj)


# ===============================================================
# Helpers to estimate skew/kurt and feed PSR/DSR
# ===============================================================

def psr_from_returns(
    r_daily: pd.Series,
    *,
    sr_benchmark: float = 0.0,
) -> float:
    """
    Convenience: compute PSR directly from daily returns.
    """
    r = pd.Series(r_daily).dropna().astype(float)
    n = len(r)
    if n < 2:
        return np.nan
    ann = 252.0
    mu_a = r.mean() * ann
    sig_a = r.std(ddof=0) * np.sqrt(ann)
    sharpe = float(mu_a / sig_a) if sig_a > 0 else np.nan
    if not np.isfinite(sharpe):
        return np.nan
    return probabilistic_sharpe_ratio(
        sharpe, n_obs=n, sr_benchmark=sr_benchmark, skew=float(_skew(r)), kurt=float(_kurtosis(r, fisher=False))
    )


def dsr_from_returns(
    r_daily: pd.Series,
    *,
    sr_benchmark: float = 0.0,
    num_trials: int = 1,
) -> float:
    """
    Convenience: compute our conservative deflated Sharpe (Bonferroni PSR).
    """
    r = pd.Series(r_daily).dropna().astype(float)
    n = len(r)
    if n < 2:
        return np.nan
    ann = 252.0
    mu_a = r.mean() * ann
    sig_a = r.std(ddof=0) * np.sqrt(ann)
    sharpe = float(mu_a / sig_a) if sig_a > 0 else np.nan
    if not np.isfinite(sharpe):
        return np.nan
    return deflated_sharpe_ratio(
        sharpe,
        n_obs=n,
        num_trials=max(1, int(num_trials)),
        sr_benchmark=sr_benchmark,
        skew=float(_skew(r)),
        kurt=float(_kurtosis(r, fisher=False)),
    )
