"""
usdr_plus/quantum/kernel.py
============================
Fidelity-based quantum kernel for U_{SDR+}.

    k(x₁, x₂; θ) = |⟨φ(x₁; θ) | φ(x₂; θ)⟩|²

where |φ(x; θ)⟩ is the state prepared by U_{SDR+}.

Kernel matrix construction is joblib-cached for speed.
"""

import numpy as np

from usdr_plus.quantum.circuit import usdr_plus_state
from usdr_plus.utils.cache import _safe_psd_hygiene, memory

# ---------------------------------------------------------------------------
# Scalar fidelity kernel
# ---------------------------------------------------------------------------


def fidelity_2d(x1, x2, theta) -> float:
    """
    Fidelity kernel k(x1, x2; θ) = |⟨φ(x1; θ) | φ(x2; θ)⟩|² for U_{SDR+}.

    Assumptions (USDR+ protocol):
      • x1, x2 are *preprocessed* 2D feature vectors:
          - if NORMALIZE="minmax": x ∈ [0, 1]^2
          - if NORMALIZE="zscore": standardized
      • usdr_plus_state(x, θ) prepares |φ(x; θ)⟩ with:
          - L = 2,
          - entangler = CNOT(0→1),
          - axes_low = (X, Z),
          - axes_high = (Z, X),
          - β-scaling only (x̂ = x / β).
      • Statevector simulation (default.qubit) is used → fast & exact.
    """
    x1 = np.asarray(x1, dtype=np.float64).ravel()
    x2 = np.asarray(x2, dtype=np.float64).ravel()
    assert x1.shape == (2,), f"Expected x1 shape (2,), got {x1.shape}"
    assert x2.shape == (2,), f"Expected x2 shape (2,), got {x2.shape}"

    psi1 = usdr_plus_state(x1, theta)
    psi2 = usdr_plus_state(x2, theta)

    return float(np.abs(np.vdot(psi1, psi2)) ** 2)


# ---------------------------------------------------------------------------
# Full kernel matrix (statevector, cached)
# ---------------------------------------------------------------------------


@memory.cache
def build_quantum_kernel_matrix(
    X1: np.ndarray,
    X2: np.ndarray,
    theta,
) -> np.ndarray:
    """
    Build the fidelity kernel matrix

        K[i, j] = |⟨φ(X1[i]; θ) | φ(X2[j]; θ)⟩|²

    for the U_{SDR+} feature map.

    Assumptions (USDR+ protocol):
      • X1, X2 are arrays of preprocessed inputs:
          - shape (n1, 2) and (n2, 2)
          - if NORMALIZE="minmax": entries in [0, 1]
          - if NORMALIZE="zscore": standardized (unbounded)
      • usdr_plus_state(x, θ) prepares |φ(x; θ)⟩ with:
          - L = 2, CNOT entangler,
          - axes_low = (X, Z), axes_high = (Z, X),
          - β-scaling only (x̂ = x / β).
      • Statevector simulation (default.qubit) → fast & exact.

    Works for:
      - train–train (square K)
      - train–test / val–test (rectangular K)
    """
    X1 = np.asarray(X1, dtype=np.float64)
    X2 = np.asarray(X2, dtype=np.float64)
    assert X1.ndim == 2 and X1.shape[1] == 2, f"X1 must be (n1, 2), got {X1.shape}"
    assert X2.ndim == 2 and X2.shape[1] == 2, f"X2 must be (n2, 2), got {X2.shape}"

    n1, n2 = X1.shape[0], X2.shape[0]
    K = np.zeros((n1, n2), dtype=np.float64)

    # Precompute states for each input (fast reuse)
    states1 = [usdr_plus_state(x, theta) for x in X1]
    states2 = [usdr_plus_state(x, theta) for x in X2]

    for i in range(n1):
        psi_i = states1[i]
        for j in range(n2):
            psi_j = states2[j]
            K[i, j] = float(np.abs(np.vdot(psi_i, psi_j)) ** 2)

    # PSD hygiene – only for square Gram matrices
    if n1 == n2:
        K, _ = _safe_psd_hygiene(K)

    return K


# ---------------------------------------------------------------------------
# Thin wrapper (cached) used by the training pipeline
# ---------------------------------------------------------------------------


@memory.cache
def build_kernel_matrix(X1: np.ndarray, X2: np.ndarray, theta):
    """
    Thin wrapper around build_quantum_kernel_matrix for U_{SDR+}.

    Assumptions:
      • X1, X2 are preprocessed inputs (MinMax or Z-score, as per NORMALIZE).
      • theta = (lambda1, lambda2, gamma, beta).
    """
    X1 = np.asarray(X1, dtype=np.float64)
    X2 = np.asarray(X2, dtype=np.float64)
    K = build_quantum_kernel_matrix(X1, X2, theta)  # uses usdr_plus_state + PSD hygiene
    return K
