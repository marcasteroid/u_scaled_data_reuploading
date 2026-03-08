"""
usdr_plus/training/krr.py
==========================
Kernel Ridge Regression prediction helpers and PSD-hygiene utilities
used during model training and evaluation.
"""

import numpy as np

from usdr_plus.analysis.spectrum import analyze_kernel_matrix


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def krr_predict(K_pred: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return K_pred @ alpha  # (n_pred, n_train) @ (n_train,) → (n_pred,)


# ---------------------------------------------------------------------------
# PSD hygiene + τ-regularisation
# ---------------------------------------------------------------------------


def apply_psd_hygiene(K: np.ndarray, tau: float, name: str = "K_train"):
    """
    Apply τ-regularization (and optional jitter if κ > 1e12).

    Returns:
      K_reg, jitter_used
    """
    print(f"\n[PSD] Analyzing {name} before regularization:")
    K_stats = analyze_kernel_matrix(K, name=name, plot=False)

    K_reg = K + tau * np.eye(K.shape[0])
    kappa_before = np.linalg.cond(K_reg)
    print(
        f"[PSD] {name} + τI: τ={tau:.3e}, "
        f"cond(K+τI)={kappa_before:.3e}"
    )

    jitter = 0.0
    if kappa_before > 1e12:
        jitter = 1e-10 * np.trace(K) / K.shape[0]
        K_reg = K_reg + jitter * np.eye(K.shape[0])
        kappa_after = np.linalg.cond(K_reg)
        print(
            f"[PSD] Added jitter {jitter:.2e} to {name}+τI "
            f"(κ_before={kappa_before:.3e} → κ_after={kappa_after:.3e})"
        )
    else:
        kappa_after = kappa_before

    return K_reg, jitter, K_stats, kappa_after
