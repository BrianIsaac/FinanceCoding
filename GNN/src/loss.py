from __future__ import annotations
from typing import List
import torch
from torch import Tensor

def sharpe_loss(returns: Tensor, eps: float = 1e-6) -> Tensor:
    mu = returns.mean()
    sd = returns.std(unbiased=False).clamp_min(eps)
    return -(mu / sd)

def turnover_penalty_indexed(weights: List[Tensor], tickers_list: List[List[str]], tc_decimal: float) -> Tensor:
    if len(weights) <= 1:
        dev = weights[0].device if weights else "cpu"
        return torch.tensor(0.0, device=dev)
    dev = weights[0].device
    acc = torch.tensor(0.0, device=dev)
    for t in range(1, len(weights)):
        w_prev, w_curr = weights[t - 1], weights[t]
        tick_prev, tick_curr = tickers_list[t - 1], tickers_list[t]
        union = list(set(tick_prev) | set(tick_curr))
        pos = {sym: i for i, sym in enumerate(union)}
        idx_prev = torch.tensor([pos[s] for s in tick_prev], dtype=torch.long, device=dev)
        idx_curr = torch.tensor([pos[s] for s in tick_curr], dtype=torch.long, device=dev)
        uprev = torch.zeros(len(union), device=dev, dtype=w_prev.dtype)
        ucurr = torch.zeros(len(union), device=dev, dtype=w_curr.dtype)
        uprev[idx_prev] = w_prev
        ucurr[idx_curr] = w_curr
        acc = acc + (ucurr - uprev).abs().sum()
    return acc * tc_decimal

def entropy_penalty(weights: List[Tensor], coef: float) -> Tensor:
    if coef <= 0.0 or not weights:
        dev = weights[0].device if weights else "cpu"
        return torch.tensor(0.0, device=dev)
    acc = torch.tensor(0.0, device=weights[0].device)
    for w in weights:
        w_ = w.clamp_min(1e-12)
        acc = acc + (w_ * w_.log()).sum()
    return coef * acc

def neg_daily_log_utility(daily_panel: Tensor, w: Tensor, eps: float = 1e-9) -> Tensor:
    """-sum_d log(1 + r_d^T w)  (minimize).  daily_panel: (D, N), w: (N,)"""
    port = daily_panel @ w
    return -(torch.log1p(port).sum())
