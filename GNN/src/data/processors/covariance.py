# src/cov.py
from __future__ import annotations

import numpy as np

try:
    from sklearn.covariance import OAS as SKOAS  # type: ignore
    from sklearn.covariance import LedoitWolf as SKLedoitWolf  # type: ignore
except Exception:
    SKLedoitWolf = None  # type: ignore
    SKOAS = None  # type: ignore

__all__ = [
    "robust_covariance",
    "to_correlation",
    "ledoit_wolf_shrinkage",
    "oas_shrinkage",
]

# --------- basic helpers ---------


def _sample_cov(X: np.ndarray, ddof: int = 0) -> np.ndarray:
    """
    X: (T, N) observations in rows. Returns (N, N) covariance.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D (T, N)")
    Xc = X - X.mean(axis=0, keepdims=True)
    S = np.cov(Xc, rowvar=False, ddof=ddof)
    S = np.asarray(S, dtype=float)
    S = 0.5 * (S + S.T)
    return S


def _identity_target(S: np.ndarray) -> np.ndarray:
    p = S.shape[0]
    avg_var = float(np.trace(S) / max(p, 1))
    return np.eye(p, dtype=float) * avg_var


def _diag_target(S: np.ndarray) -> np.ndarray:
    return np.diag(np.diag(S).astype(float))


def _blend_to_target(S: np.ndarray, alpha: float, target: str = "diag") -> np.ndarray:
    target = (target or "diag").lower()
    if target == "identity":
        T = _identity_target(S)
    elif target == "diag":
        T = _diag_target(S)
    else:
        raise ValueError(f"Unknown shrink_to='{target}'")
    return (1.0 - float(alpha)) * S + float(alpha) * T


def _floor_variances(S: np.ndarray, min_var: float) -> np.ndarray:
    S = 0.5 * (S + S.T)
    d = np.diag(S).copy()
    d = np.where(d < min_var, min_var, d)
    np.fill_diagonal(S, d)
    return S


# --------- public API used across the project ---------


def robust_covariance(
    X: np.ndarray,
    method: str = "lw",
    shrink_to: str = "diag",
    min_var: float = 1e-10,
) -> np.ndarray:
    """
    Robust covariance with shrinkage.
    method ∈ {'lw','oas','sample'}, shrink_to ∈ {'diag','identity'}.
    """
    X = np.asarray(X, dtype=float)
    S = _sample_cov(X, ddof=0)

    m = (method or "lw").lower()
    tgt = (shrink_to or "diag").lower()

    if m == "sample":
        S_hat = S
    elif m == "lw":
        if SKLedoitWolf is None:
            alpha = 0.1
        else:
            try:
                lw = SKLedoitWolf(store_precision=False, assume_centered=False)
                lw.fit(X)
                alpha = float(getattr(lw, "shrinkage_", 0.1))
            except Exception:
                alpha = 0.1
        S_hat = _blend_to_target(S, alpha=alpha, target=tgt)
    elif m == "oas":
        if SKOAS is None:
            alpha = 0.1
        else:
            try:
                oas = SKOAS(store_precision=False, assume_centered=False)
                oas.fit(X)
                alpha = float(getattr(oas, "shrinkage_", 0.1))
            except Exception:
                alpha = 0.1
        S_hat = _blend_to_target(S, alpha=alpha, target=tgt)
    else:
        raise ValueError(f"Unknown method='{method}'. Use 'lw', 'oas', or 'sample'.")

    return _floor_variances(S_hat, min_var=min_var)


def to_correlation(S: np.ndarray) -> np.ndarray:
    """
    Covariance → correlation with clipping/symmetrisation.
    """
    S = np.asarray(S, dtype=float)
    d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    denom = np.outer(d, d)
    C = S / denom
    C = np.clip(C, -0.9999, 0.9999)
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)
    return C


# --------- compatibility shims expected by src.train ---------


def ledoit_wolf_shrinkage(
    X: np.ndarray,
    shrink_to: str = "diag",
    min_var: float = 1e-10,
):
    """
    Return (S_shrunk, alpha) using Ledoit–Wolf.
    Kept separate for code that explicitly wants the alpha.
    """
    X = np.asarray(X, dtype=float)
    S = _sample_cov(X, ddof=0)
    if SKLedoitWolf is None:
        alpha = 0.1
    else:
        try:
            lw = SKLedoitWolf(store_precision=False, assume_centered=False)
            lw.fit(X)
            alpha = float(getattr(lw, "shrinkage_", 0.1))
        except Exception:
            alpha = 0.1
    S_hat = _blend_to_target(S, alpha=alpha, target=shrink_to)
    S_hat = _floor_variances(S_hat, min_var=min_var)
    return S_hat, alpha


def oas_shrinkage(
    X: np.ndarray,
    shrink_to: str = "diag",
    min_var: float = 1e-10,
):
    """
    Return (S_shrunk, alpha) using OAS.
    """
    X = np.asarray(X, dtype=float)
    S = _sample_cov(X, ddof=0)
    if SKOAS is None:
        alpha = 0.1
    else:
        try:
            oas = SKOAS(store_precision=False, assume_centered=False)
            oas.fit(X)
            alpha = float(getattr(oas, "shrinkage_", 0.1))
        except Exception:
            alpha = 0.1
    S_hat = _blend_to_target(S, alpha=alpha, target=shrink_to)
    S_hat = _floor_variances(S_hat, min_var=min_var)
    return S_hat, alpha
