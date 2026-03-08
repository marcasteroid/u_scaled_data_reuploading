"""
usdr_plus/analysis/spectrum.py
================================
Kernel-matrix diagnostics:
  • PSD check, eigenvalue spectrum, condition number
  • Effective rank (Roy & Vetterli entropy formulation)
  • compute_spectrum_metrics – compact summary used by the training loop
"""

from pathlib import Path
import re
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from usdr_plus.utils.cache import _safe_psd_hygiene


# ---------------------------------------------------------------------------
# Full kernel matrix analysis
# ---------------------------------------------------------------------------


def analyze_kernel_matrix(
    K: np.ndarray,
    name: str = "K_train",
    width: float = 10,
    height: float = 5,
    dpi: int = 300,
    plot: bool = True,
    save: bool = True,
    save_dir: str | Path = "figures/usdr/diagnostics",
) -> dict:
    """
    Analyze a quantum kernel matrix built from U_{SDR+}.

    Assumptions (USDR+ protocol):
      • Square K (e.g. K_train, K_train + τI) is a Gram matrix:
          K[i,j] = |⟨φ(x_i; θ) | φ(x_j; θ)⟩|²
        and should be PSD up to numerical noise.
      • Rectangular K (e.g. K_val, K_test) is a kernel block (n_val/test × n_train)
        and is never inverted, so only the SVD-based condition number is meaningful.
    """
    K = np.asarray(K, dtype=np.float64)

    print(f"\n=== {name} ANALYSIS ===")
    print(f"Shape: {K.shape}")

    # --- Square Gram matrix: full PSD + spectrum analysis -----------------
    if K.shape[0] == K.shape[1]:
        # Apply the same PSD hygiene used in the protocol
        K_fixed, jitter = _safe_psd_hygiene(K, eps_floor=1e-10, report=False)

        # Symmetrize again just to be extra safe for eigvalsh
        K_sym = 0.5 * (K_fixed + K_fixed.T)

        eigvals = np.linalg.eigvalsh(K_sym)
        min_eig = float(eigvals.min())
        max_eig = float(eigvals.max())
        cond_number = float(np.linalg.cond(K_sym))

        is_psd = min_eig >= -1e-10

        print(f"Eigenvalue range: [{min_eig:.3e}, {max_eig:.3e}]")
        print(f"Min eigenvalue:   {min_eig:.3e} → PSD: {'YES' if is_psd else 'NO (jitter needed)'}")
        print(f"Condition number κ: {cond_number:.3e}")
        if jitter > 0.0:
            print(f"Jitter applied during PSD hygiene: ε = {jitter:.2e}")

        if plot:
            plt.figure(figsize=(width, height), dpi=dpi)
            plt.semilogy(
                np.sort(eigvals)[::-1],
                "o-",
                markersize=4,
                color="tab:blue",
            )
            plt.title(f"Eigenvalue Spectrum of {name}")
            plt.xlabel("Index (largest to smallest)")
            plt.ylabel("Eigenvalue")
            plt.grid(True, alpha=0.3, which="both", ls="--")
            plt.tight_layout()
            if save:
                slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip()).strip("_").lower()
                out = Path(save_dir) / f"eigen_spectrum_{slug or 'k_train'}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(out, dpi=dpi, bbox_inches="tight")
                print(f"[PLOT] Saved eigenvalue spectrum -> {out}")
            plt.show()

        return {
            "min_eig": min_eig,
            "max_eig": max_eig,
            "cond": cond_number,
            "is_psd": is_psd,
            "eigvals": eigvals,
            "jitter": jitter,
        }

    # --- Rectangular kernel block: only SVD-based cond is meaningful ------
    else:
        cond_number = float(np.linalg.cond(K))
        print("Rectangular matrix → only SVD-based condition number available")
        print(f"Condition number κ: {cond_number:.3e}")
        print("  (No eigenvalue spectrum or PSD check – matrix is not square)")

        if plot:
            print("  Warning: Skipping plot for non-square matrix")

        return {
            "cond": cond_number,
            "note": "rectangular_matrix",
        }


# ---------------------------------------------------------------------------
# Effective rank
# ---------------------------------------------------------------------------


def _effective_rank_from_eigvals(
    eigvals: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Compute the effective rank r_eff of a spectrum λ_i, following
    Roy & Vetterli's definition:

        r_eff = exp( - sum_i p_i log p_i ),
        with p_i = λ_i / sum_j λ_j.

    Notes
    -----
    • Tiny negative eigenvalues (numerical noise) are clipped to 0.
    • Probabilities are floored at `eps` to avoid log(0).
    • If the trace is (numerically) zero, return 0.0.
    """
    eigvals = np.asarray(eigvals, dtype=np.float64)
    eigvals_clipped = np.clip(eigvals, a_min=0.0, a_max=None)
    trace = float(eigvals_clipped.sum())

    if trace <= eps:
        # Degenerate case: matrix is numerically zero
        return 0.0

    p = eigvals_clipped / trace
    p = np.clip(p, a_min=eps, a_max=1.0)

    entropy = -float(np.sum(p * np.log(p)))
    r_eff = float(np.exp(entropy))

    return r_eff


def compute_spectrum_metrics(
    K: np.ndarray,
    *,
    name: Optional[str] = None,
    psd_tol: float = 1e-10,
    log_prefix: str = "[SPEC]",
) -> Dict[str, float]:
    """
    Compute spectral diagnostics for a (square) kernel / Gram matrix K:

        • min_eig, max_eig
        • 2-norm condition number κ
        • effective rank r_eff

    Parameters
    ----------
    K : np.ndarray
        Square kernel (Gram) matrix.
    name : str, optional
        Name used in log messages.
    psd_tol : float, optional
        Tolerance for considering the matrix PSD. Eigenvalues below
        -psd_tol are treated as a warning.
    log_prefix : str, optional
        Prefix for printed log messages (e.g. "[SPEC-FREE]", "[SPEC-CONST]").

    Returns
    -------
    Dict[str, float]
        Dictionary with keys:
            - "min_eig"
            - "max_eig"
            - "kappa"
            - "rank_eff"
    """
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("compute_spectrum_metrics expects a square matrix.")

    # Symmetrize to be robust against tiny asymmetries
    K_sym = 0.5 * (K + K.T)

    # Eigenvalues (real, since K_sym is symmetric)
    eigvals = np.linalg.eigvalsh(K_sym)
    min_eig = float(eigvals.min())
    max_eig = float(eigvals.max())

    # Condition number (2-norm)
    kappa = float(np.linalg.cond(K_sym))

    # Effective rank
    rank_eff = _effective_rank_from_eigvals(eigvals)

    if name is None:
        name = "K"

    is_psd = min_eig >= -psd_tol

    print(
        f"{log_prefix} {name}: "
        f"min_eig={min_eig:.3e}, max_eig={max_eig:.3e}, "
        f"κ={kappa:.3e}, rank_eff={rank_eff:.2f}, "
        f"PSD={'YES' if is_psd else 'NO'}"
    )

    return {
        "min_eig": min_eig,
        "max_eig": max_eig,
        "kappa": kappa,
        "rank_eff": rank_eff,
    }
