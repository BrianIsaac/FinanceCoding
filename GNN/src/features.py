"""Feature engineering utilities for node (asset) features at each rebalance date."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def rolling_features(px: pd.DataFrame, rets: pd.DataFrame, rebal_dates: pd.Series) -> Dict[pd.Timestamp, pd.DataFrame]:
    """Computes per-asset features using information up to each rebalance date.

    Features (per asset, z-scored cross-sectionally at each date):
      - - r_1, r_5, r_21, r_63: cumulative returns over 1/5/21/63 trading days.
      - vol_21, vol_63: realised volatility over 21/63 trading days.
      - mom_252: 12-1 style momentum (252-day minus last 21-day return).
      - dd_max: maximum drawdown up to the rebalance date.

    Args:
        px: Cleaned price panel (Date × Tickers).
        rets: Daily log returns aligned to ``px``.
        rebal_dates: Series of rebalance dates (DatetimeIndex).

    Returns:
        Mapping from rebalance timestamp to a features DataFrame
        (index = tickers active at that date, columns = features).

    Notes:
        - Requires at least 252 observations before a rebalance date to emit features.
        - Inf/NaN values are replaced with 0 before z-scoring.
    """
    feats_by_t: Dict[pd.Timestamp, pd.DataFrame] = {}

    for t in rebal_dates:
        hist_r = rets.loc[:t]
        if hist_r.shape[0] < 252:
            continue

        r_1 = hist_r.tail(1).sum()
        r_5 = hist_r.tail(5).sum()
        r_21 = hist_r.tail(21).sum()
        r_63 = hist_r.tail(63).sum()

        vol_21 = hist_r.tail(21).std()
        vol_63 = hist_r.tail(63).std()

        mom_252 = hist_r.tail(252).sum() - hist_r.tail(21).sum()

        dd_path = px.loc[:t].div(px.loc[:t].cummax())
        dd_max = dd_path.min() - 1.0  # negative

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

        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df = (df - df.mean()) / (df.std(ddof=0) + 1e-9)
        feats_by_t[pd.Timestamp(t)] = df

    return feats_by_t
