"""CCPP kernel implementation backed by the shared USDR+ circuit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from joblib import Memory

from usdr_plus.ccpp import config as cfg
from usdr_plus.utils.cache import _safe_psd_hygiene
from usdr_plus.quantum.circuit import usdr_plus_state

cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)
_memory = Memory(location=cfg.CACHE_DIR, verbose=0, mmap_mode="r")


def fidelity_2d(x1: np.ndarray, x2: np.ndarray, theta: np.ndarray) -> float:
    x1 = np.asarray(x1, dtype=float).ravel()
    x2 = np.asarray(x2, dtype=float).ravel()
    theta = np.asarray(theta, dtype=float).ravel()
    if x1.shape != (2,) or x2.shape != (2,) or theta.shape != (4,):
        raise ValueError("fidelity_2d expects x1/x2 shape (2,) and theta shape (4,)")
    s1 = usdr_plus_state(x1, theta)
    s2 = usdr_plus_state(x2, theta)
    return float(np.abs(np.vdot(s1, s2)) ** 2)


@_memory.cache
def build_kernel_matrix(
    X1: np.ndarray,
    X2: np.ndarray,
    theta: np.ndarray,
    *,
    apply_psd_hygiene_for_square: bool = True,
) -> np.ndarray:
    """Build K(X1, X2) with state reuse, optional PSD hygiene for square blocks."""
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    theta = np.asarray(theta, dtype=float).ravel()
    if X1.ndim != 2 or X1.shape[1] != 2 or X2.ndim != 2 or X2.shape[1] != 2:
        raise ValueError("X1 and X2 must have shape (n, 2)")
    if theta.shape != (4,):
        raise ValueError("theta must have shape (4,)")

    states1 = [usdr_plus_state(x, theta) for x in X1]
    states2 = [usdr_plus_state(x, theta) for x in X2]

    n1, n2 = len(states1), len(states2)
    K = np.empty((n1, n2), dtype=float)
    for i in range(n1):
        s1 = states1[i]
        for j in range(n2):
            K[i, j] = float(np.abs(np.vdot(s1, states2[j])) ** 2)

    if apply_psd_hygiene_for_square and n1 == n2:
        K, _ = _safe_psd_hygiene(K, eps_floor=1e-10, report=False)
    return K
