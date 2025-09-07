# src/loss.py
from __future__ import annotations

import torch
from torch import Tensor


def sharpe_loss(returns: Tensor, eps: float = 1e-6) -> Tensor:
    """
    -(mean / std) over a vector of period returns.
    returns: shape (..., T) or (T,). We aggregate over the last dim.
    """
    if returns.ndim == 0:
        returns = returns.unsqueeze(0)
    mu = returns.mean(dim=-1)
    sd = returns.std(dim=-1, unbiased=False).clamp_min(eps)
    return -(mu / sd).mean()


def turnover_penalty_indexed(
    weights: list[Tensor],
    tickers_list: list[list[str]],
    tc_decimal: float,
) -> Tensor:
    """
    L1 turnover between consecutive weight vectors whose names (tickers) may differ.
    We align by ticker symbols, assume missing names have weight 0, and sum
    |w_t - w_{t-1}| across the *union* of names. The result is multiplied by
    tc_decimal (e.g. 10 bps => 0.001) and summed across all transitions.

    Args
    ----
    weights:       [w_0, w_1, ..., w_K] where each w_i is shape (Ni,)
    tickers_list:  parallel list of tickers for each w_i (len Ni)
    tc_decimal:    proportional transaction-cost multiplier

    Returns
    -------
    scalar Tensor (on the same device as the first weight)
    """
    if len(weights) <= 1:
        dev = weights[0].device if weights else "cpu"
        return torch.tensor(0.0, device=dev)

    dev = weights[0].device
    total = torch.tensor(0.0, device=dev)

    for prev_w, curr_w, prev_names, curr_names in zip(
        weights[:-1], weights[1:], tickers_list[:-1], tickers_list[1:]
    ):
        # Quick exits for empty cases
        if prev_w.numel() == 0 and curr_w.numel() == 0:
            continue

        # Build index maps
        prev_idx = {n: i for i, n in enumerate(prev_names)}
        curr_idx = {n: i for i, n in enumerate(curr_names)}

        # Part A: names present in current window (compare to prev, or 0 if new)
        acc = torch.tensor(0.0, device=dev)
        for n, i in curr_idx.items():
            w_curr = curr_w[i]
            if n in prev_idx:
                w_prev = prev_w[prev_idx[n]]
                acc = acc + (w_curr - w_prev).abs()
            else:
                acc = acc + w_curr.abs()  # bought from 0

        # Part B: names that disappeared (sell to 0)
        for n, j in prev_idx.items():
            if n not in curr_idx:
                acc = acc + prev_w[j].abs()

        total = total + acc * tc_decimal

    return total


def entropy_penalty(weights: list[Tensor], coef: float, eps: float = 1e-12) -> Tensor:
    """
    Sum of w * log(w) across provided weight vectors (negative for simplex weights).
    Adding this term with a *positive* coef encourages *higher* entropy (more
    diversified portfolios). Set coef=0 to disable.
    """
    if coef <= 0.0 or not weights:
        dev = weights[0].device if weights else "cpu"
        return torch.tensor(0.0, device=dev)
    acc = torch.tensor(0.0, device=weights[0].device)
    for w in weights:
        w_ = w.clamp_min(eps)
        acc = acc + (w_ * w_.log()).sum()
    return coef * acc


def neg_daily_log_utility(daily_panel: Tensor, w: Tensor, eps: float = 1e-9) -> Tensor:
    """
    -sum_d log(1 + r_d^T w)  (minimize).
    daily_panel: (D, N) daily asset returns for the window
    w:           (N,)    portfolio weights

    We clamp 1 + r_p to be >= eps for numerical stability, which is equivalent
    to capping the loss in windows with extreme drawdowns.
    """
    # (D,)
    port = daily_panel @ w
    one_plus = (1.0 + port).clamp_min(eps)
    return -torch.log(one_plus).sum()


# --- Optional sparsity helper (not wired by default) --------------------------
def pseudo_l0_sparsity_penalty(
    weights: list[Tensor], coef: float, p: float = 0.5, eps: float = 1e-12
) -> Tensor:
    """
    A differentiable 'pseudo-L0' penalty: sum (w + eps)^p with 0 < p < 1.
    Larger coef and smaller p encourage sparser weights. Use with care; on
    a capped-simplex this competes with turnover/Markowitz constraints.

    Returns scalar Tensor.
    """
    if coef <= 0.0 or not weights:
        dev = weights[0].device if weights else "cpu"
        return torch.tensor(0.0, device=dev)
    acc = torch.tensor(0.0, device=weights[0].device)
    for w in weights:
        acc = acc + (w.clamp_min(0.0) + eps).pow(p).sum()
    return coef * acc
