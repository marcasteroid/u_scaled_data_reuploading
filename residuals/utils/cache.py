"""
usdr_plus/utils/cache.py
========================
Disk-backed joblib cache initialisation, PSD hygiene utilities,
and the low-level cached kernel block helper.

The module-level `memory` object is imported by every sub-module
that uses ``@memory.cache``.
"""

import hashlib
from typing import Tuple

import numpy as np
from joblib import Memory

from usdr_plus.config import CACHE_DIR, axes_high, axes_low

# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------
CACHE_DIR.mkdir(parents=True, exist_ok=True)

memory = Memory(location=CACHE_DIR, verbose=0, mmap_mode="r")

# ---------------------------------------------------------------------------
# PSD helpers
# ---------------------------------------------------------------------------


def _hash_array(arr: np.ndarray) -> str:
    """Stable content hash of a NumPy array (shape + dtype + bytes)."""
    arr = np.ascontiguousarray(arr)
    h = hashlib.sha1()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def _safe_psd_hygiene(
    K: np.ndarray,
    eps_floor: float = 1e-10,
    report: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Symmetrize + tiny jitter if negative eigenvalues; return (K_fixed, jitter_used).
    """
    K = 0.5 * (K + K.T)
    mineig = float(np.min(np.linalg.eigvalsh(K)))
    if mineig < 0:
        eps = abs(mineig) + eps_floor
        if report:
            print(f"[PSD hygiene] Added jitter ε={eps:.2e} to ensure PSD.")
        K = K + eps * np.eye(K.shape[0])
        return K, eps
    return K, 0.0


# ---------------------------------------------------------------------------
# Low-level cached kernel block  (alternative path – kept for completeness)
# ---------------------------------------------------------------------------


@memory.cache
def _kernel_block_cached(
    XA_hash: str,
    XB_hash: str,
    XA: np.ndarray,
    XB: np.ndarray,
    lam_tuple: Tuple[float, ...],
    L: int,
    entangler: str,
    axes: str,
) -> np.ndarray:
    """
    Compliant with U_SDR+ formalism: uses fixed L=2, CNOT, axes_low/high from global config.
    """
    # NOTE: fidelity_2d is imported here to avoid a circular import at module load.
    from usdr_plus.quantum.kernel import fidelity_2d  # noqa: PLC0415

    lam = np.array(lam_tuple, dtype=np.float64)
    NA, NB = XA.shape[0], XB.shape[0]
    K = np.empty((NA, NB), dtype=np.float64)

    for i in range(NA):
        for j in range(NB):
            K[i, j] = fidelity_2d(
                XA[i],
                XB[j],
                lam=lam,
                L=L,
                entangler=entangler,
                axes_low=axes_low,    # from global config
                axes_high=axes_high,  # from global config
            )
    K, _ = _safe_psd_hygiene(K)
    return K
