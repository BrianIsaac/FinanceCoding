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


def _sample_cov(data: np.ndarray, ddof: int = 0) -> np.ndarray:
    """
    data: (T, N) observations in rows. Returns (N, N) covariance.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be 2D (T, N)")
    centered_data = data - data.mean(axis=0, keepdims=True)
    cov_matrix = np.cov(centered_data, rowvar=False, ddof=ddof)
    cov_matrix = np.asarray(cov_matrix, dtype=float)
    cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
    return cov_matrix


def _identity_target(cov_matrix: np.ndarray) -> np.ndarray:
    p = cov_matrix.shape[0]
    avg_var = float(np.trace(cov_matrix) / max(p, 1))
    return np.eye(p, dtype=float) * avg_var


def _diag_target(cov_matrix: np.ndarray) -> np.ndarray:
    return np.diag(np.diag(cov_matrix).astype(float))


def _blend_to_target(cov_matrix: np.ndarray, alpha: float, target: str = "diag") -> np.ndarray:
    target = (target or "diag").lower()
    if target == "identity":
        target_matrix = _identity_target(cov_matrix)
    elif target == "diag":
        target_matrix = _diag_target(cov_matrix)
    else:
        raise ValueError(f"Unknown shrink_to='{target}'")
    return (1.0 - float(alpha)) * cov_matrix + float(alpha) * target_matrix


def _floor_variances(cov_matrix: np.ndarray, min_var: float) -> np.ndarray:
    cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
    d = np.diag(cov_matrix).copy()
    d = np.where(d < min_var, min_var, d)
    np.fill_diagonal(cov_matrix, d)
    return cov_matrix


# --------- public API used across the project ---------


def robust_covariance(
    data: np.ndarray,
    method: str = "lw",
    shrink_to: str = "diag",
    min_var: float = 1e-10,
) -> np.ndarray:
    """
    Robust covariance with shrinkage.
    method ∈ {'lw','oas','sample'}, shrink_to ∈ {'diag','identity'}.
    """
    data = np.asarray(data, dtype=float)
    cov_matrix = _sample_cov(data, ddof=0)

    m = (method or "lw").lower()
    tgt = (shrink_to or "diag").lower()

    if m == "sample":
        shrunk_cov = cov_matrix
    elif m == "lw":
        if SKLedoitWolf is None:
            alpha = 0.1
        else:
            try:
                lw = SKLedoitWolf(store_precision=False, assume_centered=False)
                lw.fit(data)
                alpha = float(getattr(lw, "shrinkage_", 0.1))
            except Exception:
                alpha = 0.1
        shrunk_cov = _blend_to_target(cov_matrix, alpha=alpha, target=tgt)
    elif m == "oas":
        if SKOAS is None:
            alpha = 0.1
        else:
            try:
                oas = SKOAS(store_precision=False, assume_centered=False)
                oas.fit(data)
                alpha = float(getattr(oas, "shrinkage_", 0.1))
            except Exception:
                alpha = 0.1
        shrunk_cov = _blend_to_target(cov_matrix, alpha=alpha, target=tgt)
    else:
        raise ValueError(f"Unknown method='{method}'. Use 'lw', 'oas', or 'sample'.")

    return _floor_variances(shrunk_cov, min_var=min_var)


def to_correlation(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Covariance → correlation with clipping/symmetrisation.
    """
    cov_matrix = np.asarray(cov_matrix, dtype=float)
    d = np.sqrt(np.clip(np.diag(cov_matrix), 1e-12, None))
    denom = np.outer(d, d)
    corr_matrix = cov_matrix / denom
    corr_matrix = np.clip(corr_matrix, -0.9999, 0.9999)
    corr_matrix = 0.5 * (corr_matrix + corr_matrix.T)
    np.fill_diagonal(corr_matrix, 1.0)
    return corr_matrix


# --------- compatibility shims expected by src.train ---------


def ledoit_wolf_shrinkage(
    data: np.ndarray,
    shrink_to: str = "diag",
    min_var: float = 1e-10,
):
    """
    Return (shrunk_cov, alpha) using Ledoit–Wolf.
    Kept separate for code that explicitly wants the alpha.
    """
    data = np.asarray(data, dtype=float)
    cov_matrix = _sample_cov(data, ddof=0)
    if SKLedoitWolf is None:
        alpha = 0.1
    else:
        try:
            lw = SKLedoitWolf(store_precision=False, assume_centered=False)
            lw.fit(data)
            alpha = float(getattr(lw, "shrinkage_", 0.1))
        except Exception:
            alpha = 0.1
    shrunk_cov = _blend_to_target(cov_matrix, alpha=alpha, target=shrink_to)
    shrunk_cov = _floor_variances(shrunk_cov, min_var=min_var)
    return shrunk_cov, alpha


def oas_shrinkage(
    data: np.ndarray,
    shrink_to: str = "diag",
    min_var: float = 1e-10,
):
    """
    Return (shrunk_cov, alpha) using OAS.
    """
    data = np.asarray(data, dtype=float)
    cov_matrix = _sample_cov(data, ddof=0)
    if SKOAS is None:
        alpha = 0.1
    else:
        try:
            oas = SKOAS(store_precision=False, assume_centered=False)
            oas.fit(data)
            alpha = float(getattr(oas, "shrinkage_", 0.1))
        except Exception:
            alpha = 0.1
    shrunk_cov = _blend_to_target(cov_matrix, alpha=alpha, target=shrink_to)
    shrunk_cov = _floor_variances(shrunk_cov, min_var=min_var)
    return shrunk_cov, alpha
