#!/usr/bin/env python
"""
Auto-generated from /Users/marco/Desktop/hybrid.ipynb.
This script preserves notebook execution order cell-by-cell.
"""

from __future__ import annotations

import logging
try:
    from IPython.display import display
except Exception:  # pragma: no cover
    def display(obj):
        print(obj)


# ===== CELL 000 (markdown) =====
# # Hybrid Quantum–Classical $U_{\text{SDR}^+}$ Kernel – Small-Sample Regression
# 
# In this section we extend the bounded–hyperparameter study of the $U_{\text{SDR}^+}$ kernel by introducing a **hybrid quantum–classical kernel** for small-sample regression.
# 
# We start from two similarity measures on the same inputs $x, x'$:
# 
# - A **quantum kernel** $k_{\text{USDR}^+}(x,x'; \theta^*)$, given by the fidelity between feature states prepared by the bounded $U_{\text{SDR}^+}$ circuit:
#   $$
#   k_{\text{USDR}^+}(x,x'; \theta^*) 
#   = \bigl|\langle \phi_{\text{USDR}^+}(x; \theta^*) \mid \phi_{\text{USDR}^+}(x'; \theta^*) \rangle\bigr|^2.
#   $$
# - A **classical kernel** $k_C(x,x'; \eta^*)$, e.g. an RBF kernel with fixed hyperparameters from the classical baseline.
# 
# The **hybrid kernel** is a convex combination of these two geometries:
# $$
# k_H(x,x'; \omega)
# = \omega\,k_{\text{USDR}^+}(x,x'; \theta^*)
# + (1 - \omega)\,k_C(x,x'; \eta^*),
# \qquad \omega \in [0,1].
# $$
# 
# At the level of Gram matrices this corresponds to
# $$
# K_H(\omega) = \omega\,K_Q + (1 - \omega)\,K_C,
# $$
# where $K_Q$ and $K_C$ are the (normalized) quantum and classical Gram matrices built on the same train/val/test splits.
# 
# In the notebook we:
# 
# - **Reuse** the best-behaved bounded configuration $(\theta^*, \tau^*)$ found for $U_{\text{SDR}^+}$, and the tuned classical hyperparameters $\eta^*$.
# - Introduce only **one new scalar knob** $\omega$ controlling the trade-off between classical and quantum geometry.
# - For a grid of $\omega$ values, run Kernel Ridge Regression with $K_H(\omega)$ on the 2D smooth-interaction synthetic dataset, computing:
#   - train/val/test MSE,
#   - spectral diagnostics (eigenvalues, effective rank, condition numbers),
#   - and, on the 2D grid, predicted surfaces and residuals.
# 
# This section therefore probes a **one-dimensional quantum–classical trade-off** on top of the existing bounded $U_{\text{SDR}^+}$ setup, allowing us to characterize when and how the USDR$^+$ quantum geometry adds value beyond a strong classical kernel.

# ===== CELL 001 (code) =====
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: None  # disable interactive popups
import seaborn as sns
from pathlib import Path
import os, warnings, json, time
from tqdm import tqdm
from scipy import stats
from scipy.optimize import minimize
import scipy.linalg as la
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Optional, Any, Literal
from functools import partial
import random
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# === QUANTUM and ML ===
import pennylane as qml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# === PARALLEL and CACHING ===
from joblib import Memory, Parallel, delayed

# === PARALLEL and CACHING ===
from matplotlib import animation
from IPython.display import HTML

from matplotlib.ticker import MaxNLocator

# ===== CELL 002 (markdown) =====
# **Environment hygiene (Colab)**

# ===== CELL 003 (code) =====
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ===== CELL 004 (markdown) =====
# **Disk cache setup**

# ===== CELL 005 (code) =====
CACHE_DIR = Path("cache/hybrid")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

memory = Memory(location=CACHE_DIR, verbose=1, mmap_mode='r')  # Read-only for safety
print(f"[CACHE] Initialized at: {CACHE_DIR.resolve()}")

# ===== CELL 006 (markdown) =====
# **Configuration parameters**

# ===== CELL 007 (code) =====
# === EXPERIMENTAL CONFIGURATION (USDR+ PROTOCOL) ===
RAW_DOMAIN = (0.0, 2 * np.pi)      # x1, x2 ∈ [0, 2π] (USDR+ §2.1)

noise_std = 0.05                   # ε ~ N(0, 0.05)
sample_sizes = [50, 100, 200]      # N ∈ {50, 100, 200}
grid_size = 60                     # 60×60 visualization grid
SEEDS = [0, 1, 2]                  # 3 independent seeds

OUTPUT_DIR = Path("preprocessed/hybrid")
BASE_PATH = OUTPUT_DIR             # keep a single canonical base path

# Preprocessing:
# - raw x ∈ [0, 2π] are sampled
# - if NORMALIZE == "minmax": x̃ = x / (2π)  (USDR+ §2.2)
# - circuit then uses x̂ = x̃ / β
NORMALIZE = "minmax"               # "minmax" or "zscore"

depth = 2                          # Fixed L = 2
entangler = "cnot"                 # Fixed for USDR+
axes_low = ("X", "Z")              # Low-freq block
axes_high = ("Z", "X")             # High-freq block

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === HYPERPARAMETERS TO OPTIMIZE (θ) ===
theta_bounds = {
    "lambda1": (0.1, 10.0),
    "lambda2": (0.1, 10.0),
    "gamma":   (1.0, 5.0),         # γ ≥ 1
    "beta":    (0.1, 10.0)
}

def set_all_seeds(seed: int) -> None:
    """Set deterministic seeds for NumPy and Python's random module."""
    random.seed(seed)
    np.random.seed(seed)

print(f"[CONFIG] USDR+ protocol loaded. SEEDS={SEEDS}, N={sample_sizes}, L={depth}")

# ===== CELL 008 (code) =====
# ======================================================================
# - RBF kernel implementation
# - Small grid search for η* (length-scale) and τ* on classical-only model
# - Integration helper to compute K_C_train/val/test and log performance
# ======================================================================

def get_logger(name: str = "usdr_plus.classical_kernel") -> logging.Logger:
    """
    Return a module-level logger configured with a sensible default
    if no handlers are attached yet.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


LOGGER = get_logger()


# ----------------------------------------------------------------------
# RBF classical kernel
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class RBFKernelConfig:
    """
    Configuration for an RBF (Gaussian) kernel.

    Attributes
    ----------
    length_scale : float
        Positive length-scale ℓ controlling the smoothness of the kernel.
        Smaller values -> more local, higher-frequency structure.
    """
    length_scale: float

    def __post_init__(self) -> None:
        if self.length_scale <= 0.0:
            raise ValueError(f"length_scale must be positive, got {self.length_scale!r}")


def rbf_kernel_matrix(
    X1: np.ndarray,
    X2: np.ndarray,
    config: RBFKernelConfig,
) -> np.ndarray:
    """
    Compute the RBF kernel matrix K(x_i, z_j) for two sets of points.

    Parameters
    ----------
    X1 : np.ndarray, shape (n_samples_1, n_features)
        First set of input points.
    X2 : np.ndarray, shape (n_samples_2, n_features)
        Second set of input points.
    config : RBFKernelConfig
        Kernel configuration (length-scale ℓ).

    Returns
    -------
    K : np.ndarray, shape (n_samples_1, n_samples_2)
        RBF Gram matrix with entries:
        K_ij = exp(-||x_i - z_j||^2 / (2 ℓ^2))
    """
    if X1.ndim != 2 or X2.ndim != 2:
        raise ValueError("X1 and X2 must be 2D arrays (n_samples, n_features).")
    if X1.shape[1] != X2.shape[1]:
        raise ValueError(
            f"X1 and X2 must have the same number of features: "
            f"{X1.shape[1]} != {X2.shape[1]}"
        )

    # Squared Euclidean distances ||x_i - z_j||^2 via (x^2 + z^2 - 2 x·z)
    X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)       # (n1, 1)
    X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)       # (1, n2)
    cross = X1 @ X2.T                                    # (n1, n2)

    sq_dists = X1_sq + X2_sq - 2.0 * cross
    # Numerical safety: small negative values -> 0
    sq_dists = np.maximum(sq_dists, 0.0)

    ell2 = float(config.length_scale) ** 2
    K = np.exp(-0.5 * sq_dists / ell2)
    return K


# ----------------------------------------------------------------------
# Simple KRR solver (classical-only; kernel-agnostic)
# ----------------------------------------------------------------------

def krr_predict(
    K_train: np.ndarray,
    y_train: np.ndarray,
    K_eval: np.ndarray,
    tau: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve Kernel Ridge Regression and make predictions on eval points.

    Parameters
    ----------
    K_train : np.ndarray, shape (n_train, n_train)
        Gram matrix on training data.
    y_train : np.ndarray, shape (n_train,)
        Training targets.
    K_eval : np.ndarray, shape (n_eval, n_train)
        Cross-kernel between evaluation points and training points.
    tau : float
        Ridge regularization parameter (λ in KRR).

    Returns
    -------
    alpha : np.ndarray, shape (n_train,)
        Dual coefficients.
    y_pred : np.ndarray, shape (n_eval,)
        Predictions on evaluation points.
    """
    if K_train.shape[0] != K_train.shape[1]:
        raise ValueError("K_train must be square.")
    if K_train.shape[0] != y_train.shape[0]:
        raise ValueError("K_train and y_train must have compatible shapes.")
    if K_eval.shape[1] != K_train.shape[0]:
        raise ValueError("K_eval second dimension must match K_train size.")
    if tau <= 0.0:
        raise ValueError(f"tau must be strictly positive, got {tau!r}")

    n_train = K_train.shape[0]
    # Regularized system: (K + τ I) α = y
    A = K_train + tau * np.eye(n_train, dtype=K_train.dtype)
    # Use solve rather than explicit inverse for numerical stability
    alpha = np.linalg.solve(A, y_train)
    y_pred = K_eval @ alpha
    return alpha, y_pred


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the mean squared error between true and predicted targets.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    diff = y_true - y_pred
    return float(np.mean(diff ** 2))


# ----------------------------------------------------------------------
# Grid search for classical-only model (to pick η* and τ* once)
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class ClassicalGridSearchResult:
    """
    Result of a small grid search over classical RBF kernel hyperparameters
    and ridge regularization.
    """
    best_config: RBFKernelConfig
    best_tau: float
    best_val_mse: float
    results_frame: pd.DataFrame


def grid_search_classical_rbf_krr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    length_scales: Sequence[float],
    tau_grid: Sequence[float],
    logger: Optional[logging.Logger] = None,
) -> ClassicalGridSearchResult:
    """
    Perform a small, one-off grid search for the classical RBF kernel.

    Parameters
    ----------
    X_train, X_val : np.ndarray
        Training and validation inputs, shape (n_samples, n_features).
    y_train, y_val : np.ndarray
        Training and validation targets, shape (n_samples,).
    length_scales : Sequence[float]
        Candidate length-scales ℓ to try.
    tau_grid : Sequence[float]
        Candidate ridge parameters τ to try.
    logger : logging.Logger, optional
        Logger to report progress; if None, uses module-level LOGGER.

    Returns
    -------
    ClassicalGridSearchResult
        Dataclass containing the best config, τ, val MSE, and a DataFrame
        of all tried combinations.
    """
    logger = logger or LOGGER

    records: list[Dict[str, Any]] = []
    best_val_mse = np.inf
    best_config: Optional[RBFKernelConfig] = None
    best_tau: Optional[float] = None

    for ell in length_scales:
        config = RBFKernelConfig(length_scale=ell)
        logger.info("Evaluating classical RBF with length_scale=%.4f", ell)

        K_train = rbf_kernel_matrix(X_train, X_train, config)
        K_val = rbf_kernel_matrix(X_val, X_train, config)

        for tau in tau_grid:
            _, y_val_pred = krr_predict(K_train, y_train, K_val, tau)
            val_mse = mean_squared_error(y_val, y_val_pred)

            records.append(
                {
                    "length_scale": ell,
                    "tau": tau,
                    "val_mse": val_mse,
                }
            )

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_config = config
                best_tau = tau
                logger.info(
                    "New best classical config: ℓ=%.4f, τ=%.3e, Val MSE=%.4e",
                    ell,
                    tau,
                    val_mse,
                )

    if best_config is None or best_tau is None:
        raise RuntimeError("Grid search did not evaluate any configuration.")

    results_df = pd.DataFrame.from_records(records)
    logger.info(
        "Classical grid search completed. Best ℓ=%.4f, τ=%.3e, Val MSE=%.4e",
        best_config.length_scale,
        best_tau,
        best_val_mse,
    )

    return ClassicalGridSearchResult(
        best_config=best_config,
        best_tau=best_tau,
        best_val_mse=best_val_mse,
        results_frame=results_df,
    )


# ----------------------------------------------------------------------
# Per-(N, seed) classical kernel integration into the experiment loop
# ----------------------------------------------------------------------

@dataclass
class ClassicalKernelRunResult:
    """
    Container for classical kernel Gram matrices and performance metrics
    for a single (N, seed) experiment.
    """
    K_train: np.ndarray
    K_val: np.ndarray
    K_test: np.ndarray
    config: RBFKernelConfig
    tau: float
    val_mse: float
    test_mse: float


def run_classical_baseline_for_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: RBFKernelConfig,
    tau: float,
    experiment_meta: Optional[Dict[str, Any]] = None,
    results_df: Optional[pd.DataFrame] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[ClassicalKernelRunResult, pd.DataFrame]:
    """
    Compute classical RBF Gram matrices for (X_train, X_val, X_test),
    run KRR with fixed (config, tau), and record performance.

    This is intended to be called *inside* your existing main experiment
    loop, for each (N, seed) combination, right after you build the
    quantum Gram matrices K_Q_train/K_Q_val/K_Q_test.

    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray
        Inputs for train/val/test splits.
    y_train, y_val, y_test : np.ndarray
        Corresponding targets.
    config : RBFKernelConfig
        Fixed classical kernel hyperparameters η* (e.g. ℓ*).
    tau : float
        Fixed ridge parameter τ* for the classical baseline.
    experiment_meta : dict, optional
        Dictionary with metadata for this run (e.g. {"N": N, "seed": seed}).
        Its keys will be included in the results DataFrame.
    results_df : pd.DataFrame, optional
        Existing results DataFrame to append a new classical-only row to.
        If None, a new DataFrame is created.
    logger : logging.Logger, optional
        Logger instance; if None, uses module-level LOGGER.

    Returns
    -------
    run_result : ClassicalKernelRunResult
        Dataclass with Gram matrices and performance metrics.
    updated_results_df : pd.DataFrame
        Updated results DataFrame including this classical-only row.
    """
    logger = logger or LOGGER
    experiment_meta = experiment_meta or {}

    # --- 1. Build classical Gram matrices ---
    logger.info(
        "Building classical Gram matrices for config ℓ=%.4f (N=%s, seed=%s)",
        config.length_scale,
        experiment_meta.get("N", "NA"),
        experiment_meta.get("seed", "NA"),
    )

    K_train = rbf_kernel_matrix(X_train, X_train, config)
    K_val = rbf_kernel_matrix(X_val, X_train, config)
    K_test = rbf_kernel_matrix(X_test, X_train, config)

    # --- 2. Run KRR for val and test sets ---
    _, y_val_pred = krr_predict(K_train, y_train, K_val, tau)
    _, y_test_pred = krr_predict(K_train, y_train, K_test, tau)

    val_mse = mean_squared_error(y_val, y_val_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    logger.info(
        "Classical baseline (ℓ=%.4f, τ=%.3e) -> Val MSE=%.4e, Test MSE=%.4e",
        config.length_scale,
        tau,
        val_mse,
        test_mse,
    )

    run_result = ClassicalKernelRunResult(
        K_train=K_train,
        K_val=K_val,
        K_test=K_test,
        config=config,
        tau=tau,
        val_mse=val_mse,
        test_mse=test_mse,
    )

    # --- 3. Append metrics to results DataFrame ---
    row = {
        "model_type": "classical_rbf",
        "length_scale": config.length_scale,
        "tau": tau,
        "val_mse": val_mse,
        "test_mse": test_mse,
    }
    # Merge in experiment metadata such as N, seed, etc.
    row.update(experiment_meta)

    if results_df is None:
        results_df = pd.DataFrame([row])
    else:
        results_df = pd.concat(
            [results_df, pd.DataFrame([row])],
            axis=0,
            ignore_index=True,
        )

    return run_result, results_df

# ===== CELL 009 (code) =====
# ======================================================================
# Gram matrix normalization & PSD hygiene for USDR⁺ hybrid experiments
# ======================================================================

# ----------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------

def get_norm_logger(name: str = "usdr_plus.gram_normalization") -> logging.Logger:
    """
    Return a logger configured with a sensible default if no handlers
    are attached yet.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


NORM_LOGGER = get_norm_logger()


# ----------------------------------------------------------------------
# Configuration & results dataclasses
# ----------------------------------------------------------------------

NormalizationMethod = Literal["unit_diag", "trace", "none"]


@dataclass(frozen=True)
class GramNormalizationConfig:
    """
    Configuration for Gram matrix normalization and PSD hygiene.

    Attributes
    ----------
    method : {"unit_diag", "trace", "none"}
        - "unit_diag": scale to have unit diagonal using per-sample factors.
        - "trace": scale all entries so trace(K_train) = n_train (or similar).
        - "none": do not normalize.
    min_diag : float
        Minimum allowed diagonal value for unit-diagonal scaling. Any smaller
        diagonal entries are clipped up to this value to avoid division by zero.
    project_psd : bool
        If True, symmetrize and project K_train onto the PSD cone by
        zeroing out (up to a tolerance) negative eigenvalues.
    psd_tolerance : float
        Eigenvalues ≥ -psd_tolerance are treated as numerically zero in the
        PSD projection.
    compute_eigenspectrum : bool
        If True, compute and return eigenvalues of the normalized K_train.
    """
    method: NormalizationMethod = "unit_diag"
    min_diag: float = 1e-12
    project_psd: bool = False
    psd_tolerance: float = 1e-10
    compute_eigenspectrum: bool = False

    def __post_init__(self) -> None:
        if self.method not in ("unit_diag", "trace", "none"):
            raise ValueError(f"Unknown normalization method: {self.method!r}")
        if self.min_diag <= 0.0:
            raise ValueError(f"min_diag must be positive, got {self.min_diag!r}")
        if self.psd_tolerance < 0.0:
            raise ValueError(f"psd_tolerance must be non-negative, got {self.psd_tolerance!r}")


@dataclass
class NormalizedGramTriplet:
    """
    Normalized Gram matrices for a single (N, seed) split, along with
    diagnostic information.

    Attributes
    ----------
    K_train : np.ndarray
        Normalized Gram matrix on training data.
    K_val : np.ndarray
        Normalized Gram matrix between validation and training data.
    K_test : np.ndarray
        Normalized Gram matrix between test and training data.
    scaling_vector : Optional[np.ndarray]
        Per-sample scaling factors applied on the training axis
        (only used for "unit_diag" method). None otherwise.
    scale_scalar : float
        Global scalar normalization factor (used for "trace" method;
        equals 1.0 for "unit_diag" and "none").
    eigenvalues : Optional[np.ndarray]
        Eigenvalues of the *normalized* K_train (if requested).
    meta : Dict[str, Any]
        Additional metadata about the normalization (method, flags, etc.).
    """
    K_train: np.ndarray
    K_val: np.ndarray
    K_test: np.ndarray
    scaling_vector: Optional[np.ndarray]
    scale_scalar: float
    eigenvalues: Optional[np.ndarray]
    meta: Dict[str, Any]


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _symmetrize(K: np.ndarray) -> np.ndarray:
    """Return the symmetric part of K."""
    return 0.5 * (K + K.T)


def _project_to_psd(
    K: np.ndarray,
    tol: float,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Symmetrize and project K onto the PSD cone by zeroing out small
    negative eigenvalues.

    Parameters
    ----------
    K : np.ndarray, shape (n, n)
        Input Gram matrix (not necessarily PSD).
    tol : float
        Eigenvalue tolerance. Eigenvalues above -tol are treated as zero.
    logger : logging.Logger, optional
        Logger to report projection details.

    Returns
    -------
    K_psd : np.ndarray
        PSD-projected Gram matrix.
    eigvals : np.ndarray
        Eigenvalues of the symmetrized K before clipping.
    """
    logger = logger or NORM_LOGGER
    K_sym = _symmetrize(K)
    eigvals, eigvecs = np.linalg.eigh(K_sym)

    min_eig = float(eigvals[0])
    num_negative = int(np.sum(eigvals < -tol))
    logger.info(
        "PSD projection: min eigenvalue=%.3e, #eigvals < -tol= %d",
        min_eig,
        num_negative,
    )

    # Clip eigenvalues below -tol up to 0
    eigvals_clipped = np.where(eigvals < -tol, 0.0, eigvals)
    K_psd = (eigvecs * eigvals_clipped) @ eigvecs.T
    K_psd = _symmetrize(K_psd)  # enforce symmetry numerically

    return K_psd, eigvals


# ----------------------------------------------------------------------
# Public API: normalize train / val / test triplet
# ----------------------------------------------------------------------

def normalize_gram_triplet(
    K_train: np.ndarray,
    K_val: np.ndarray,
    K_test: np.ndarray,
    config: GramNormalizationConfig,
    logger: Optional[logging.Logger] = None,
    label: str = "",
) -> NormalizedGramTriplet:
    """
    Normalize a Gram-matrix triplet (train, val, test) using statistics
    from the training set only, in a way that is suitable for constructing
    hybrid kernels.

    Parameters
    ----------
    K_train : np.ndarray, shape (n_train, n_train)
        Gram matrix on training data.
    K_val : np.ndarray, shape (n_val, n_train)
        Gram matrix between validation and training data.
    K_test : np.ndarray, shape (n_test, n_train)
        Gram matrix between test and training data.
    config : GramNormalizationConfig
        Normalization and PSD hygiene configuration.
    logger : logging.Logger, optional
        Logger to use; if None, uses module-level NORM_LOGGER.
    label : str, optional
        Label used for logging (e.g. "quantum" or "classical").

    Returns
    -------
    NormalizedGramTriplet
        Dataclass containing normalized Gram matrices plus diagnostics.

    Notes
    -----
    - For "unit_diag":
        * We compute per-sample scaling factors s_i from the diagonal
          of K_train and apply: K_train' = S K_train S, K_val' = K_val S,
          K_test' = K_test S, where S = diag(1 / sqrt(s_i)).
        * This ensures diag(K_train') ≈ 1.
    - For "trace":
        * We compute a scalar factor α so that trace(α K_train) = n_train
          (i.e. average diagonal ~ 1), and scale train/val/test by α.
    - For "none":
        * We return the matrices unchanged (but may still run PSD hygiene).
    """
    logger = logger or NORM_LOGGER

    # Basic shape checks
    n_train = K_train.shape[0]
    if K_train.shape[0] != K_train.shape[1]:
        raise ValueError("K_train must be square.")
    if K_val.shape[1] != n_train or K_test.shape[1] != n_train:
        raise ValueError(
            "K_val and K_test must have n_train columns, "
            f"got {K_val.shape[1]} and {K_test.shape[1]}, expected {n_train}."
        )

    method = config.method
    meta: Dict[str, Any] = {
        "method": method,
        "project_psd": config.project_psd,
        "psd_tolerance": config.psd_tolerance,
        "min_diag": config.min_diag,
    }

    label_str = f" [{label}]" if label else ""
    logger.info(
        "Normalizing Gram triplet%s with method='%s' (n_train=%d).",
        label_str,
        method,
        n_train,
    )

    # Work on copies to avoid mutating caller's matrices
    K_tr = np.array(K_train, copy=True)
    K_va = np.array(K_val, copy=True)
    K_te = np.array(K_test, copy=True)

    scaling_vector: Optional[np.ndarray] = None
    scale_scalar: float = 1.0

    # ---------------------------
    # 1) Apply chosen normalization
    # ---------------------------
    if method == "unit_diag":
        # Diagonal-based scaling: S K S so diag(K') ≈ 1
        diag = np.diag(K_tr).astype(float)
        if np.any(diag < 0.0):
            logger.warning(
                "Negative diagonal entries found before unit-diag scaling%s. "
                "This may indicate a non-PSD Gram matrix.",
                label_str,
            )

        # Clip diagonal to avoid division by zero
        diag_clipped = np.maximum(diag, config.min_diag)
        logger.info(
            "Unit-diag scaling%s: min diag=%.3e, max diag=%.3e (after clipping).",
            label_str,
            float(diag_clipped.min()),
            float(diag_clipped.max()),
        )

        inv_sqrt_diag = 1.0 / np.sqrt(diag_clipped)  # shape (n_train,)
        scaling_vector = inv_sqrt_diag

        # K_train' = S K_train S, where S = diag(inv_sqrt_diag)
        K_tr = (K_tr * inv_sqrt_diag) * inv_sqrt_diag[:, None]

        # K_val' and K_test': multiply each column by inv_sqrt_diag
        K_va = K_va * inv_sqrt_diag
        K_te = K_te * inv_sqrt_diag

        scale_scalar = 1.0  # scalar is not used in this mode

    elif method == "trace":
        # Scalar scaling so trace(K_tr') = n_train
        tr = float(np.trace(K_tr))
        if tr <= 0.0:
            raise ValueError(
                f"Trace of K_train is non-positive ({tr:.3e}), cannot trace-normalize."
            )

        # We want average diagonal ~ 1 => trace / n_train ~ 1
        # So factor = n_train / trace(K)
        factor = n_train / tr
        logger.info(
            "Trace scaling%s: trace(K_train)=%.3e, n_train=%d, factor=%.3e.",
            label_str,
            tr,
            n_train,
            factor,
        )

        K_tr *= factor
        K_va *= factor
        K_te *= factor

        scaling_vector = None
        scale_scalar = factor

    elif method == "none":
        logger.info(
            "No normalization applied to Gram triplet%s (method='none').",
            label_str,
        )
        scaling_vector = None
        scale_scalar = 1.0

    else:  # Should never happen due to __post_init__
        raise RuntimeError(f"Unsupported normalization method: {method!r}")

    # ---------------------------
    # 2) Optional PSD projection & eigenspectrum on normalized K_train
    # ---------------------------
    eigenvalues: Optional[np.ndarray] = None

    if config.project_psd:
        logger.info("Applying PSD projection to K_train%s.", label_str)
        K_tr_psd, eigvals = _project_to_psd(K_tr, tol=config.psd_tolerance, logger=logger)
        K_tr = K_tr_psd
        eigenvalues = eigvals
        meta["psd_projected"] = True
        meta["psd_min_eig_before"] = float(eigvals[0])
        meta["psd_max_eig_before"] = float(eigvals[-1])
    else:
        meta["psd_projected"] = False
        if config.compute_eigenspectrum:
            logger.info("Computing eigenspectrum of normalized K_train%s.", label_str)
            eigvals = np.linalg.eigvalsh(_symmetrize(K_tr))
            eigenvalues = eigvals
            meta["eigs_min"] = float(eigvals[0])
            meta["eigs_max"] = float(eigvals[-1])

    return NormalizedGramTriplet(
        K_train=K_tr,
        K_val=K_va,
        K_test=K_te,
        scaling_vector=scaling_vector,
        scale_scalar=scale_scalar,
        eigenvalues=eigenvalues,
        meta=meta,
    )

# ===== CELL 010 (markdown) =====
# # Hybrid quantum–classical Gram matrices
# 
# We now define the hybrid kernel at the matrix level. Given normalized
# quantum Gram matrices (USDR⁺) and normalized classical Gram matrices (RBF),
# we construct
# \[
#   K_H(\omega) = \omega \,\tilde K_Q + (1-\omega)\,\tilde K_C
# \]
# for a small grid of ω values. We also include optional PSD / spectral
# diagnostics for each K_H_train(ω).

# ===== CELL 011 (code) =====
# ======================================================================
# Hybrid Gram matrices for USDR⁺ + classical kernel
# - ω grid definition
# - Construction of K_H_train/val/test for each ω
# - Optional PSD / eigenspectrum checks for K_H_train(ω)
# ======================================================================

# ----------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------

def get_hybrid_logger(name: str = "usdr_plus.hybrid_kernel") -> logging.Logger:
    """
    Return a logger configured with a sensible default if no handlers
    are attached yet.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


HYBRID_LOGGER = get_hybrid_logger()


# ----------------------------------------------------------------------
# Dataclasses for configuration and results
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class HybridKernelConfig:
    """
    Configuration for constructing hybrid quantum–classical Gram matrices.

    Attributes
    ----------
    omega_grid : Sequence[float]
        Values of ω in [0, 1] to explore. Typically a small fixed set such as
        [0.0, 0.25, 0.5, 0.75, 1.0].
        - ω = 0   -> pure classical kernel
        - ω = 1   -> pure USDR⁺ kernel (bounded regime)
        - 0 < ω < 1 -> hybrid kernel
    psd_check : bool
        If True, perform PSD checks and eigenvalue diagnostics on K_H_train(ω).
    psd_tolerance : float
        Numerical tolerance for declaring eigenvalues as "non-negative". Any
        eigenvalue < -psd_tolerance will be treated as a violation.
    compute_eigenspectrum : bool
        If True, store all eigenvalues for K_H_train(ω). Otherwise, only
        store summary statistics (min, max).
    """
    omega_grid: Sequence[float]
    psd_check: bool = True
    psd_tolerance: float = 1e-10
    compute_eigenspectrum: bool = False

    def __post_init__(self) -> None:
        if not self.omega_grid:
            raise ValueError("omega_grid must contain at least one value.")
        for w in self.omega_grid:
            if not (0.0 <= w <= 1.0):
                raise ValueError(
                    f"All ω must lie in [0, 1]. Invalid value: ω={w!r}"
                )
        if self.psd_tolerance < 0.0:
            raise ValueError(
                f"psd_tolerance must be non-negative, got {self.psd_tolerance!r}"
            )


@dataclass
class HybridGramMatrices:
    """
    Hybrid Gram matrices for a single ω.

    Attributes
    ----------
    omega : float
        Mixing weight ω used for this hybrid kernel.
    K_train : np.ndarray
        Hybrid train Gram matrix K_H_train(ω).
    K_val : np.ndarray
        Hybrid val Gram matrix K_H_val(ω).
    K_test : np.ndarray
        Hybrid test Gram matrix K_H_test(ω).
    eigenvalues : Optional[np.ndarray]
        Eigenvalues of K_train (if requested via config).
    min_eig : Optional[float]
        Minimum eigenvalue of K_train (if PSD check or spectrum computed).
    max_eig : Optional[float]
        Maximum eigenvalue of K_train (if PSD check or spectrum computed).
    is_psd : Optional[bool]
        True if all eigenvalues >= -psd_tolerance; False if violation detected;
        None if no PSD check was requested.
    meta : Dict[str, Any]
        Additional metadata (e.g. psd_tolerance, n_train, etc.).
    """
    omega: float
    K_train: np.ndarray
    K_val: np.ndarray
    K_test: np.ndarray
    eigenvalues: Optional[np.ndarray]
    min_eig: Optional[float]
    max_eig: Optional[float]
    is_psd: Optional[bool]
    meta: Dict[str, Any]


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _psd_check_and_eigs(
    K: np.ndarray,
    psd_tolerance: float,
    compute_eigenspectrum: bool,
    logger: Optional[logging.Logger] = None,
    label: str = "",
) -> (Optional[np.ndarray], Optional[float], Optional[float], Optional[bool]):
    """
    Run PSD / eigenspectrum diagnostics on a Gram matrix.

    Parameters
    ----------
    K : np.ndarray, shape (n, n)
        Symmetric Gram matrix (or nearly symmetric).
    psd_tolerance : float
        Eigenvalues below -psd_tolerance are considered negative.
    compute_eigenspectrum : bool
        If True, return all eigenvalues; otherwise, only min/max.
    logger : logging.Logger, optional
        Logger to use; if None, uses HYBRID_LOGGER.
    label : str, optional
        Label for logging (e.g. f"ω={omega:.2f}").

    Returns
    -------
    eigenvalues : Optional[np.ndarray]
        Eigenvalues (ascending) if requested, else None.
    min_eig : Optional[float]
        Minimum eigenvalue.
    max_eig : Optional[float]
        Maximum eigenvalue.
    is_psd : Optional[bool]
        True if eigenvalues >= -psd_tolerance, else False; None if no check.
    """
    logger = logger or HYBRID_LOGGER
    label_str = f" [{label}]" if label else ""

    # Ensure symmetry numerically before eigendecomposition
    K_sym = _symmetrize(K)
    eigvals = np.linalg.eigvalsh(K_sym)

    min_eig = float(eigvals[0])
    max_eig = float(eigvals[-1])
    is_psd = bool(np.all(eigvals >= -psd_tolerance))

    logger.info(
        "Hybrid PSD check%s: min_eig=%.3e, max_eig=%.3e, psd_tolerance=%.1e, is_psd=%s",
        label_str,
        min_eig,
        max_eig,
        psd_tolerance,
        is_psd,
    )

    if not is_psd:
        num_violations = int(np.sum(eigvals < -psd_tolerance))
        logger.warning(
            "Hybrid Gram%s shows %d eigenvalues < -psd_tolerance=%.1e.",
            label_str,
            num_violations,
            psd_tolerance,
        )

    if compute_eigenspectrum:
        return eigvals, min_eig, max_eig, is_psd
    else:
        return None, min_eig, max_eig, is_psd


# ----------------------------------------------------------------------
# Public API: build hybrid Gram matrices for each ω
# ----------------------------------------------------------------------

def build_hybrid_gram_matrices(
    Ktilde_Q_train: np.ndarray,
    Ktilde_Q_val: np.ndarray,
    Ktilde_Q_test: np.ndarray,
    Ktilde_C_train: np.ndarray,
    Ktilde_C_val: np.ndarray,
    Ktilde_C_test: np.ndarray,
    config: HybridKernelConfig,
    logger: Optional[logging.Logger] = None,
    experiment_meta: Optional[Dict[str, Any]] = None,
) -> Dict[float, HybridGramMatrices]:
    """
    Construct hybrid Gram matrices K_H(ω) = ω K_Q + (1 - ω) K_C for each
    ω in config.omega_grid, using *normalized* quantum and classical
    Gram matrices.

    Parameters
    ----------
    Ktilde_Q_train, Ktilde_Q_val, Ktilde_Q_test : np.ndarray
        Normalized quantum Gram matrices for train/val/test splits.
        Shapes: (n_train, n_train), (n_val, n_train), (n_test, n_train).
    Ktilde_C_train, Ktilde_C_val, Ktilde_C_test : np.ndarray
        Normalized classical Gram matrices for train/val/test splits.
        Same shapes as quantum counterparts.
    config : HybridKernelConfig
        Hybrid configuration (ω grid, PSD checks, etc.).
    logger : logging.Logger, optional
        Logger instance; if None, uses module-level HYBRID_LOGGER.
    experiment_meta : dict, optional
        Metadata about this (N, seed) experiment; stored in result.meta.

    Returns
    -------
    hybrid_mats : Dict[float, HybridGramMatrices]
        Dictionary mapping each ω to its corresponding HybridGramMatrices.

    Notes
    -----
    - This function assumes that quantum and classical Gram matrices are
      already normalized (e.g. via unit-diagonal scaling) and have
      compatible shapes.
    - Raw (unnormalized) K_Q, K_C should be kept separately if you need
      them for reference; this function only deals with the normalized
      versions Ktilde_*.
    """
    logger = logger or HYBRID_LOGGER
    experiment_meta = experiment_meta or {}

    # Shape checks (basic safety)
    n_train = Ktilde_Q_train.shape[0]
    if Ktilde_Q_train.shape != Ktilde_C_train.shape:
        raise ValueError(
            "Quantum and classical K_train must have the same shape: "
            f"{Ktilde_Q_train.shape} != {Ktilde_C_train.shape}"
        )
    if Ktilde_Q_train.shape[0] != Ktilde_Q_train.shape[1]:
        raise ValueError("Ktilde_Q_train must be square.")
    if Ktilde_C_train.shape[0] != Ktilde_C_train.shape[1]:
        raise ValueError("Ktilde_C_train must be square.")

    if Ktilde_Q_val.shape != Ktilde_C_val.shape:
        raise ValueError(
            "Quantum and classical K_val must have the same shape: "
            f"{Ktilde_Q_val.shape} != {Ktilde_C_val.shape}"
        )
    if Ktilde_Q_test.shape != Ktilde_C_test.shape:
        raise ValueError(
            "Quantum and classical K_test must have the same shape: "
            f"{Ktilde_Q_test.shape} != {Ktilde_C_test.shape}"
        )

    if Ktilde_Q_val.shape[1] != n_train or Ktilde_Q_test.shape[1] != n_train:
        raise ValueError(
            "Ktilde_*_val/test should have n_train columns equal to "
            f"Ktilde_*_train size ({n_train}). Got "
            f"{Ktilde_Q_val.shape[1]} and {Ktilde_Q_test.shape[1]}."
        )

    logger.info(
        "Building hybrid Gram matrices for ω grid=%s (n_train=%d, meta=%s).",
        list(config.omega_grid),
        n_train,
        experiment_meta,
    )

    hybrid_mats: Dict[float, HybridGramMatrices] = {}

    for omega in config.omega_grid:
        # 1) Construct hybrid Gram matrices at the matrix level
        logger.info("Constructing hybrid Gram for ω=%.3f.", omega)

        K_H_train = (
            omega * Ktilde_Q_train + (1.0 - omega) * Ktilde_C_train
        )
        K_H_val = (
            omega * Ktilde_Q_val + (1.0 - omega) * Ktilde_C_val
        )
        K_H_test = (
            omega * Ktilde_Q_test + (1.0 - omega) * Ktilde_C_test
        )

        # 2) Optional PSD / eigenvalue diagnostics on K_H_train
        if config.psd_check:
            eigvals, min_eig, max_eig, is_psd = _psd_check_and_eigs(
                K=K_H_train,
                psd_tolerance=config.psd_tolerance,
                compute_eigenspectrum=config.compute_eigenspectrum,
                logger=logger,
                label=f"ω={omega:.3f}",
            )
        else:
            eigvals = None
            min_eig = None
            max_eig = None
            is_psd = None

        # 3) Build meta-info dictionary
        meta: Dict[str, Any] = {
            "omega": omega,
            "psd_checked": config.psd_check,
            "psd_tolerance": config.psd_tolerance,
            "compute_eigenspectrum": config.compute_eigenspectrum,
            "n_train": n_train,
        }
        meta.update(experiment_meta)

        hybrid_mats[omega] = HybridGramMatrices(
            omega=omega,
            K_train=K_H_train,
            K_val=K_H_val,
            K_test=K_H_test,
            eigenvalues=eigvals,
            min_eig=min_eig,
            max_eig=max_eig,
            is_psd=is_psd,
            meta=meta,
        )

    return hybrid_mats

# ===== CELL 012 (code) =====
# ======================================================================
# KRR with hybrid Gram matrices K_H(ω) for USDR⁺ + classical kernel
# ======================================================================

# ----------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------

def get_hybrid_krr_logger(name: str = "usdr_plus.hybrid_krr") -> logging.Logger:
    """
    Return a logger configured with a sensible default if no handlers
    are attached yet.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


HYBRID_KRR_LOGGER = get_hybrid_krr_logger()


# ----------------------------------------------------------------------
# Core KRR solver (kernel-agnostic)
# ----------------------------------------------------------------------

@dataclass
class KRRMetrics:
    """
    Results of a single KRR fit/evaluation.

    Attributes
    ----------
    alpha : np.ndarray, shape (n_train,)
        Dual coefficients.
    y_pred_train : np.ndarray, shape (n_train,)
        Predictions on the training set.
    y_pred_val : np.ndarray, shape (n_val,)
        Predictions on the validation set.
    y_pred_test : np.ndarray, shape (n_test,)
        Predictions on the test set.
    mse_train : float
        Mean squared error on the training set.
    mse_val : float
        Mean squared error on the validation set.
    mse_test : float
        Mean squared error on the test set.
    """
    alpha: np.ndarray
    y_pred_train: np.ndarray
    y_pred_val: np.ndarray
    y_pred_test: np.ndarray
    mse_train: float
    mse_val: float
    mse_test: float


def krr_fit_and_evaluate(
    K_train: np.ndarray,
    y_train: np.ndarray,
    K_val: np.ndarray,
    y_val: np.ndarray,
    K_test: np.ndarray,
    y_test: np.ndarray,
    tau: float,
) -> KRRMetrics:
    """
    Solve Kernel Ridge Regression with kernel matrix K and evaluate on
    train/val/test sets.

    Parameters
    ----------
    K_train : np.ndarray, shape (n_train, n_train)
        Gram matrix on training data.
    y_train : np.ndarray, shape (n_train,)
        Training targets.
    K_val : np.ndarray, shape (n_val, n_train)
        Gram matrix between validation and training data.
    y_val : np.ndarray, shape (n_val,)
        Validation targets.
    K_test : np.ndarray, shape (n_test, n_train)
        Gram matrix between test and training data.
    y_test : np.ndarray, shape (n_test,)
        Test targets.
    tau : float
        Ridge regularization parameter.

    Returns
    -------
    KRRMetrics
        Dataclass containing dual coefficients, predictions, and MSEs.
    """
    if K_train.shape[0] != K_train.shape[1]:
        raise ValueError("K_train must be square.")
    n_train = K_train.shape[0]

    if y_train.shape[0] != n_train:
        raise ValueError("y_train must have length n_train.")
    if K_val.shape[1] != n_train or K_test.shape[1] != n_train:
        raise ValueError("K_val and K_test must have n_train columns.")
    if y_val.shape[0] != K_val.shape[0]:
        raise ValueError("y_val length must match K_val rows.")
    if y_test.shape[0] != K_test.shape[0]:
        raise ValueError("y_test length must match K_test rows.")
    if tau <= 0.0:
        raise ValueError(f"tau must be strictly positive, got {tau!r}")

    # Regularized linear system: (K + τ I) α = y
    A = K_train + tau * np.eye(n_train, dtype=K_train.dtype)
    alpha = np.linalg.solve(A, y_train)

    # Predictions
    y_pred_train = K_train @ alpha
    y_pred_val = K_val @ alpha
    y_pred_test = K_test @ alpha

    # MSEs
    mse_train = float(np.mean((y_train - y_pred_train) ** 2))
    mse_val = float(np.mean((y_val - y_pred_val) ** 2))
    mse_test = float(np.mean((y_test - y_pred_test) ** 2))

    return KRRMetrics(
        alpha=alpha,
        y_pred_train=y_pred_train,
        y_pred_val=y_pred_val,
        y_pred_test=y_pred_test,
        mse_train=mse_train,
        mse_val=mse_val,
        mse_test=mse_test,
    )


# ----------------------------------------------------------------------
# Spectral diagnostics: effective rank & condition numbers
# ----------------------------------------------------------------------

def _get_eigenvalues_from_matrix(
    K: np.ndarray,
) -> np.ndarray:
    """
    Compute eigenvalues of a symmetric Gram matrix in ascending order.
    """
    # Ensure symmetry for numerical robustness
    K_sym = 0.5 * (K + K.T)
    eigvals = np.linalg.eigvalsh(K_sym)
    return eigvals


def effective_rank(
    eigvals: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """
    Compute the "entropy-based" effective rank of a PSD matrix from its
    eigenvalues.

    Definition:
        p_i = λ_i / sum_j λ_j
        H = - sum_i p_i log p_i
        r_eff = exp(H)

    Parameters
    ----------
    eigvals : np.ndarray
        Eigenvalues (can be unsorted; non-negative expected).
    eps : float
        Numerical epsilon to avoid log(0).

    Returns
    -------
    r_eff : float
        Effective rank (in [1, n]).
    """
    eigvals = np.asarray(eigvals, dtype=float)
    eigvals = np.clip(eigvals, 0.0, None)  # enforce non-negative
    trace = float(np.sum(eigvals))
    if trace <= 0.0:
        return 0.0
    p = eigvals / trace
    p = np.clip(p, eps, 1.0)  # avoid log(0)
    H = -float(np.sum(p * np.log(p)))
    r_eff = float(np.exp(H))
    return r_eff


def condition_numbers_from_eigvals(
    eigvals: np.ndarray,
    tau: float,
    eps: float = 1e-15,
) -> Tuple[float, float]:
    """
    Compute raw and regularized condition numbers from eigenvalues.

    Parameters
    ----------
    eigvals : np.ndarray
        Eigenvalues of K (not K+τI), expected non-negative.
    tau : float
        Ridge parameter.
    eps : float
        Numerical epsilon to avoid division by zero.

    Returns
    -------
    kappa_raw : float
        Condition number κ(K) = λ_max / max(λ_min, eps).
    kappa_reg : float
        Condition number κ(K + τI) = (λ_max + τ) / (λ_min + τ).
    """
    eigvals = np.asarray(eigvals, dtype=float)
    eigvals = np.clip(eigvals, 0.0, None)

    lam_min = float(np.min(eigvals)) if eigvals.size > 0 else 0.0
    lam_max = float(np.max(eigvals)) if eigvals.size > 0 else 0.0

    kappa_raw = lam_max / max(lam_min, eps)
    kappa_reg = (lam_max + tau) / (lam_min + tau)

    return kappa_raw, kappa_reg


# ----------------------------------------------------------------------
# Configuration & result containers for hybrid KRR
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class HybridKRRConfig:
    """
    Configuration for running KRR on hybrid Gram matrices K_H(ω).

    Attributes
    ----------
    tau_star : float
        Fixed ridge parameter τ* obtained from the bounded USDR⁺ regime.
        This is reused for all ω to isolate the effect of hybrid mixing.
    compute_eigvals_if_missing : bool
        If True and hybrid Gram entries do not contain eigenvalues,
        compute them from K_H_train(ω).
    eff_rank_eps : float
        Numerical epsilon in effective-rank computation.
    """
    tau_star: float
    compute_eigvals_if_missing: bool = True
    eff_rank_eps: float = 1e-15

    def __post_init__(self) -> None:
        if self.tau_star <= 0.0:
            raise ValueError(f"tau_star must be positive, got {self.tau_star!r}")


@dataclass
class HybridKRROmegaResult:
    """
    Full KRR + spectral diagnostics for a given ω.

    Attributes
    ----------
    omega : float
        Mixing weight.
    metrics : KRRMetrics
        KRR metrics (α, predictions, MSEs).
    effective_rank : Optional[float]
        Entropy-based effective rank of K_H_train(ω).
    kappa_raw : Optional[float]
        Raw condition number κ(K_H_train(ω)).
    kappa_reg : Optional[float]
        Regularized condition number κ(K_H_train(ω) + τ*I).
    eigvals : Optional[np.ndarray]
        Eigenvalues of K_H_train(ω), if available or computed.
    meta : Dict[str, Any]
        Additional metadata (e.g. N, seed, etc.).
    """
    omega: float
    metrics: KRRMetrics
    effective_rank: Optional[float]
    kappa_raw: Optional[float]
    kappa_reg: Optional[float]
    eigvals: Optional[np.ndarray]
    meta: Dict[str, Any]


# ----------------------------------------------------------------------
# Main API: run KRR for all ω and append to global results DataFrame
# ----------------------------------------------------------------------

def run_hybrid_krr_for_omegas(
    hybrid_mats: Mapping[float, Any],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    config: HybridKRRConfig,
    logger: Optional[logging.Logger] = None,
    experiment_meta: Optional[Dict[str, Any]] = None,
    global_results_df: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[float, HybridKRROmegaResult], pd.DataFrame]:
    """
    Run KRR with fixed τ* on all hybrid Gram matrices K_H(ω), compute
    spectral diagnostics, and append rows to a global results DataFrame.

    Parameters
    ----------
    hybrid_mats : Mapping[float, Any]
        Mapping from ω to a hybrid Gram container. Each value is expected
        to have at least the attributes:
            - K_train : np.ndarray
            - K_val   : np.ndarray
            - K_test  : np.ndarray
        Optionally, it may also provide:
            - eigenvalues : np.ndarray or None
    y_train, y_val, y_test : np.ndarray
        Targets for train/val/test splits.
    config : HybridKRRConfig
        KRR configuration (fixed τ*, etc.).
    logger : logging.Logger, optional
        Logger instance; if None, uses HYBRID_KRR_LOGGER.
    experiment_meta : dict, optional
        Metadata for this (N, seed) experiment (e.g. {"N": N, "seed": seed}).
    global_results_df : pd.DataFrame, optional
        Existing results DataFrame to append rows to. If None, a new
        DataFrame is created.

    Returns
    -------
    omega_results : Dict[float, HybridKRROmegaResult]
        Per-ω KRR + spectral results.
    updated_df : pd.DataFrame
        Updated results DataFrame including one row per (ω, experiment).
    """
    logger = logger or HYBRID_KRR_LOGGER
    experiment_meta = experiment_meta or {}

    tau_star = config.tau_star
    logger.info(
        "Running hybrid KRR for %d ω values with fixed τ*=%.3e (meta=%s).",
        len(hybrid_mats),
        tau_star,
        experiment_meta,
    )

    omega_results: Dict[float, HybridKRROmegaResult] = {}

    # Ensure we have a DataFrame to append to
    if global_results_df is None:
        global_results_df = pd.DataFrame()

    for omega, mats in hybrid_mats.items():
        logger.info("Hybrid KRR: processing ω=%.3f.", omega)

        K_train = mats.K_train
        K_val = mats.K_val
        K_test = mats.K_test

        # --- 1) Run KRR with fixed τ* ---
        metrics = krr_fit_and_evaluate(
            K_train=K_train,
            y_train=y_train,
            K_val=K_val,
            y_val=y_val,
            K_test=K_test,
            y_test=y_test,
            tau=tau_star,
        )

        # --- 2) Spectral diagnostics on K_H_train(ω) ---
        eigvals: Optional[np.ndarray] = getattr(mats, "eigenvalues", None)
        if eigvals is None and config.compute_eigvals_if_missing:
            eigvals = _get_eigenvalues_from_matrix(K_train)

        if eigvals is not None:
            r_eff = effective_rank(eigvals, eps=config.eff_rank_eps)
            kappa_raw, kappa_reg = condition_numbers_from_eigvals(
                eigvals, tau=tau_star, eps=config.eff_rank_eps
            )
        else:
            r_eff = None
            kappa_raw = None
            kappa_reg = None

        # --- 3) Build meta info and result object ---
        meta: Dict[str, Any] = {
            "omega": omega,
            "tau_star": tau_star,
        }
        meta.update(experiment_meta)

        omega_result = HybridKRROmegaResult(
            omega=omega,
            metrics=metrics,
            effective_rank=r_eff,
            kappa_raw=kappa_raw,
            kappa_reg=kappa_reg,
            eigvals=eigvals,
            meta=meta,
        )
        omega_results[omega] = omega_result

        # --- 4) Append row to global results DataFrame ---
        row = {
            # experiment identifiers
            "N": experiment_meta.get("N", None),
            "seed": experiment_meta.get("seed", None),
            # hybrid-specific
            "omega": omega,
            "tau_star": tau_star,
            # performance
            "mse_train": metrics.mse_train,
            "mse_val": metrics.mse_val,
            "mse_test": metrics.mse_test,
            # spectral diagnostics
            "effective_rank": r_eff,
            "kappa_raw": kappa_raw,
            "kappa_reg": kappa_reg,
            # flags (could be extended later)
            "model_type": "hybrid_usdr_plus",
        }

        global_results_df = pd.concat(
            [global_results_df, pd.DataFrame([row])],
            axis=0,
            ignore_index=True,
        )

    return omega_results, global_results_df

# ===== CELL 013 (code) =====
# ======================================================================
# Post-processing for hybrid USDR⁺ kernel:
# - Find ω* per (N, seed) by minimizing val MSE
# - Extract metrics at ω*
# - Aggregate over seeds per N (mean/std tables)
# ======================================================================

# ----------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------

def get_hybrid_post_logger(name: str = "usdr_plus.hybrid_postprocessing") -> logging.Logger:
    """
    Return a logger configured with a sensible default if no handlers
    are attached yet.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


HYBRID_POST_LOGGER = get_hybrid_post_logger()


# ----------------------------------------------------------------------
# Dataclasses for results
# ----------------------------------------------------------------------

@dataclass
class OmegaStarRunSummary:
    """
    Per-(N, seed) summary at ω* (best ω by val MSE).

    Attributes
    ----------
    N : int
        Number of training samples in this run.
    seed : int
        Random seed for this run.
    omega_star : float
        ω* value minimizing validation MSE within this (N, seed) block.
    mse_val_star : float
        Validation MSE at ω*.
    mse_test_star : float
        Test MSE at ω*.
    effective_rank_star : Optional[float]
        Effective rank of K_H_train(ω*) (if available).
    kappa_raw_star : Optional[float]
        Raw condition number κ(K_H_train(ω*)) (if available).
    kappa_reg_star : Optional[float]
        Regularized condition number κ(K_H_train(ω*) + τ*I) (if available).
    """
    N: int
    seed: int
    omega_star: float
    mse_val_star: float
    mse_test_star: float
    effective_rank_star: Optional[float]
    kappa_raw_star: Optional[float]
    kappa_reg_star: Optional[float]


# ----------------------------------------------------------------------
# Utility: validate required columns
# ----------------------------------------------------------------------

REQUIRED_HYBRID_COLUMNS = {
    "N",
    "seed",
    "omega",
    "mse_val",
    "mse_test",
    "effective_rank",
    "kappa_raw",
    "kappa_reg",
}


def _validate_hybrid_results_frame(df: pd.DataFrame) -> None:
    """
    Ensure that the global results DataFrame has the columns needed
    for ω* post-processing.
    """
    missing = REQUIRED_HYBRID_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            f"Global results DataFrame is missing required columns: {sorted(missing)}"
        )


# ----------------------------------------------------------------------
# 1) Per-(N, seed) ω* computation and extraction of metrics
# ----------------------------------------------------------------------

def compute_omega_star_per_run(
    global_results_df: pd.DataFrame,
    logger: Optional[logging.Logger] = None,
    hybrid_model_type: str = "hybrid_usdr_plus",
) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], OmegaStarRunSummary]]:
    """
    For each (N, seed) in the global results DataFrame, find the ω that
    minimizes validation MSE (ω*), and extract metrics at ω*.

    Parameters
    ----------
    global_results_df : pd.DataFrame
        Results DataFrame with one row per (N, seed, ω) and at least the
        columns listed in REQUIRED_HYBRID_COLUMNS. It may also contain
        rows for other model types (e.g. "classical_rbf"); this function
        will automatically filter to `model_type == hybrid_model_type` if
        such a column exists.
    logger : logging.Logger, optional
        Logger instance; if None, uses HYBRID_POST_LOGGER.
    hybrid_model_type : str
        Value of `model_type` column identifying hybrid runs. Ignored if
        `model_type` column is not present.

    Returns
    -------
    omega_star_df : pd.DataFrame
        DataFrame with one row per (N, seed) containing:
            - N
            - seed
            - omega_star
            - mse_val_star
            - mse_test_star
            - effective_rank_star
            - kappa_raw_star
            - kappa_reg_star
    omega_star_map : Dict[(N, seed), OmegaStarRunSummary]
        Mapping from (N, seed) to an OmegaStarRunSummary dataclass.
    """
    logger = logger or HYBRID_POST_LOGGER

    if "model_type" in global_results_df.columns:
        df_hybrid = global_results_df[
            global_results_df["model_type"] == hybrid_model_type
        ].copy()
        logger.info(
            "Filtering global results for hybrid model_type='%s': "
            "%d hybrid rows found (out of %d total).",
            hybrid_model_type,
            len(df_hybrid),
            len(global_results_df),
        )
    else:
        df_hybrid = global_results_df.copy()
        logger.info(
            "Global results DataFrame has no 'model_type' column; "
            "assuming all rows correspond to hybrid runs."
        )

    if df_hybrid.empty:
        raise ValueError(
            "No hybrid rows found in global_results_df. "
            "Check 'model_type' values or hybrid_model_type argument."
        )

    _validate_hybrid_results_frame(df_hybrid)

    # Ensure N and seed are integers (for grouping/printing)
    df_hybrid["N"] = df_hybrid["N"].astype(int)
    df_hybrid["seed"] = df_hybrid["seed"].astype(int)

    omega_star_records: list[Dict[str, Any]] = []
    omega_star_map: Dict[Tuple[int, int], OmegaStarRunSummary] = {}

    # Group by (N, seed) and find ω* = argmin_ω mse_val
    grouped = df_hybrid.groupby(["N", "seed"], sort=True)

    logger.info(
        "Computing ω* per (N, seed) for %d distinct (N, seed) combinations.",
        len(grouped),
    )

    for (N_val, seed_val), block in grouped:
        # Find index of minimal validation MSE in this block
        idx_min = block["mse_val"].idxmin()
        best_row = df_hybrid.loc[idx_min]

        omega_star = float(best_row["omega"])
        mse_val_star = float(best_row["mse_val"])
        mse_test_star = float(best_row["mse_test"])
        eff_rank_star = (
            float(best_row["effective_rank"])
            if not pd.isna(best_row["effective_rank"])
            else None
        )
        kappa_raw_star = (
            float(best_row["kappa_raw"])
            if not pd.isna(best_row["kappa_raw"])
            else None
        )
        kappa_reg_star = (
            float(best_row["kappa_reg"])
            if not pd.isna(best_row["kappa_reg"])
            else None
        )

        logger.info(
            "For (N=%d, seed=%d): ω* = %.3f, "
            "Val MSE=%.4e, Test MSE=%.4e, "
            "r_eff*=%.3f, κ_raw*=%.3e, κ_reg*=%.3e",
            N_val,
            seed_val,
            omega_star,
            mse_val_star,
            mse_test_star,
            eff_rank_star if eff_rank_star is not None else float("nan"),
            kappa_raw_star if kappa_raw_star is not None else float("nan"),
            kappa_reg_star if kappa_reg_star is not None else float("nan"),
        )

        summary = OmegaStarRunSummary(
            N=N_val,
            seed=seed_val,
            omega_star=omega_star,
            mse_val_star=mse_val_star,
            mse_test_star=mse_test_star,
            effective_rank_star=eff_rank_star,
            kappa_raw_star=kappa_raw_star,
            kappa_reg_star=kappa_reg_star,
        )
        omega_star_map[(N_val, seed_val)] = summary

        omega_star_records.append(
            {
                "N": N_val,
                "seed": seed_val,
                "omega_star": omega_star,
                "mse_val_star": mse_val_star,
                "mse_test_star": mse_test_star,
                "effective_rank_star": eff_rank_star,
                "kappa_raw_star": kappa_raw_star,
                "kappa_reg_star": kappa_reg_star,
            }
        )

    omega_star_df = pd.DataFrame.from_records(omega_star_records)
    omega_star_df.sort_values(by=["N", "seed"], inplace=True, ignore_index=True)

    return omega_star_df, omega_star_map


# ----------------------------------------------------------------------
# 2) Aggregate over seeds per N (mean/std tables)
# ----------------------------------------------------------------------

def aggregate_omega_star_by_N(
    omega_star_df: pd.DataFrame,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Given the per-(N, seed) ω* summary DataFrame, compute mean and std
    across seeds for each N.

    Specifically, for each unique N, compute:
        - mean_omega_star, std_omega_star
        - mean_mse_test_star, std_mse_test_star
        - mean_effective_rank_star, std_effective_rank_star
        - mean_kappa_reg_star, std_kappa_reg_star

    Parameters
    ----------
    omega_star_df : pd.DataFrame
        DataFrame produced by compute_omega_star_per_run, with columns:
            - N
            - seed
            - omega_star
            - mse_val_star
            - mse_test_star
            - effective_rank_star
            - kappa_raw_star
            - kappa_reg_star
    logger : logging.Logger, optional
        Logger instance; if None, uses HYBRID_POST_LOGGER.

    Returns
    -------
    agg_df : pd.DataFrame
        Aggregated table with one row per N, columns:
            - N
            - mean_omega_star, std_omega_star
            - mean_mse_test_star, std_mse_test_star
            - mean_effective_rank_star, std_effective_rank_star
            - mean_kappa_reg_star, std_kappa_reg_star
    """
    logger = logger or HYBRID_POST_LOGGER

    required_cols = {
        "N",
        "seed",
        "omega_star",
        "mse_test_star",
        "effective_rank_star",
        "kappa_reg_star",
    }
    missing = required_cols.difference(omega_star_df.columns)
    if missing:
        raise ValueError(
            f"omega_star_df is missing required columns: {sorted(missing)}"
        )

    # Convert to numeric types (if not already)
    omega_star_df = omega_star_df.copy()
    omega_star_df["N"] = omega_star_df["N"].astype(int)
    omega_star_df["seed"] = omega_star_df["seed"].astype(int)

    logger.info(
        "Aggregating ω* summaries across seeds for %d distinct N values.",
        omega_star_df["N"].nunique(),
    )

    grouped = omega_star_df.groupby("N", sort=True)

    agg_records: list[Dict[str, Any]] = []

    for N_val, block in grouped:
        # Use .mean() and .std(ddof=0) for population-like stats
        mean_omega = float(block["omega_star"].mean())
        std_omega = float(block["omega_star"].std(ddof=0))

        mean_mse_test = float(block["mse_test_star"].mean())
        std_mse_test = float(block["mse_test_star"].std(ddof=0))

        # effective_rank_star and kappa_reg_star may contain NaNs; handle gracefully
        mean_rank = float(block["effective_rank_star"].mean(skipna=True))
        std_rank = float(block["effective_rank_star"].std(ddof=0, skipna=True))

        mean_kappa_reg = float(block["kappa_reg_star"].mean(skipna=True))
        std_kappa_reg = float(block["kappa_reg_star"].std(ddof=0, skipna=True))

        logger.info(
            "Aggregate for N=%d: "
            "ω*: mean=%.3f, std=%.3f; "
            "Test MSE*: mean=%.4e, std=%.4e; "
            "r_eff*: mean=%.3f, std=%.3f; "
            "κ_reg*: mean=%.3e, std=%.3e.",
            N_val,
            mean_omega,
            std_omega,
            mean_mse_test,
            std_mse_test,
            mean_rank,
            std_rank,
            mean_kappa_reg,
            std_kappa_reg,
        )

        agg_records.append(
            {
                "N": N_val,
                "mean_omega_star": mean_omega,
                "std_omega_star": std_omega,
                "mean_mse_test_star": mean_mse_test,
                "std_mse_test_star": std_mse_test,
                "mean_effective_rank_star": mean_rank,
                "std_effective_rank_star": std_rank,
                "mean_kappa_reg_star": mean_kappa_reg,
                "std_kappa_reg_star": std_kappa_reg,
            }
        )

    agg_df = pd.DataFrame.from_records(agg_records)
    agg_df.sort_values(by="N", inplace=True, ignore_index=True)

    return agg_df

# ===== CELL 014 (code) =====
# ======================================================================
# Visualization utilities for hybrid USDR⁺ kernel experiments
# ======================================================================

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

def get_vis_logger(name: str = "usdr_plus.hybrid_visualization") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


VIS_LOGGER = get_vis_logger()


# ----------------------------------------------------------------------
# 7.1 Geometry-level plots
# ----------------------------------------------------------------------

def plot_kernel_geometry_heatmaps(
    Ktilde_C_train: np.ndarray,
    Ktilde_Q_train: np.ndarray,
    hybrid_mats: Mapping[float, Any],
    omegas_to_show: Sequence[float] = (0.25, 0.5, 0.75),
    title: str = "Hybrid kernel geometry (train Gram matrices)",
    logger: Optional[logging.Logger] = None,
    width: float = 20.0,
    height: float = 20.0,
    dpi: int = 300,
) -> None:
    """
    Plot side-by-side heatmaps for:
        - classical train Gram (Ktilde_C_train),
        - USDR⁺ train Gram (Ktilde_Q_train),
        - hybrid train Gram K_H_train(ω) for selected ω values.

    All heatmaps share the same color scale.

    Parameters
    ----------
    ...
    width : float
        Figure width PER PANEL (inches). Total width = width * n_plots.
    height : float
        Figure height (inches).
    dpi : int
        Figure DPI for export-quality images.
    """
    logger = logger or VIS_LOGGER

    # Collect matrices to set a common vmin / vmax
    mats: List[np.ndarray] = [Ktilde_C_train, Ktilde_Q_train]
    for w in omegas_to_show:
        if w not in hybrid_mats:
            logger.warning("ω=%.3f not found in hybrid_mats; skipping in heatmaps.", w)
            continue
        mats.append(hybrid_mats[w].K_train)

    all_vals = np.concatenate([m.ravel() for m in mats])
    vmin = float(all_vals.min())
    vmax = float(all_vals.max())

    n_plots = 2 + len(omegas_to_show)
    fig, axes = plt.subplots(
        1,
        n_plots,
        figsize=(width * n_plots, height),
        dpi=dpi,
        constrained_layout=True,
    )

    if n_plots == 1:
        axes = [axes]

    ax = axes[0]
    im0 = ax.imshow(Ktilde_C_train, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(r"$K_{\mathrm{C}}\ (\omega=0)$")
    ax.set_xlabel("train index")
    ax.set_ylabel("train index")

    ax = axes[1]
    ax.imshow(Ktilde_Q_train, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(r"$K_{\mathrm{Q}}\ (\omega=1)$")
    ax.set_xlabel("train index")
    ax.set_ylabel("train index")

    for i, w in enumerate(omegas_to_show, start=2):
        if w not in hybrid_mats:
            continue
        ax = axes[i]
        ax.imshow(hybrid_mats[w].K_train, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(rf"$K_{{\mathrm{{H}}}}(\omega={w:.2f})$")
        ax.set_xlabel("train index")
        ax.set_ylabel("train index")

    fig.suptitle(title)
    # Single shared colorbar
    cbar = fig.colorbar(im0, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label("kernel value")


def plot_eigenvalue_spectra_for_case(
    Ktilde_C_train: np.ndarray,
    Ktilde_Q_train: np.ndarray,
    hybrid_mats: Mapping[float, Any],
    omegas_to_show: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    title: str = "Eigenvalue spectra of classical, quantum, and hybrid kernels",
    width: float = 16.0,
    height: float = 12.0,
    dpi: int = 300,
) -> None:
    """
    Plot sorted eigenvalue spectra (log scale) for:
        - classical train Gram,
        - USDR⁺ train Gram,
        - hybrid K_H_train(ω) for selected ω.

    width, height, dpi as usual refer to full figure.
    """
    def _eigvals_sorted(K: np.ndarray) -> np.ndarray:
        K_sym = 0.5 * (K + K.T)
        vals = np.linalg.eigvalsh(K_sym)
        vals = np.clip(vals, 0.0, None)
        vals_sorted = np.sort(vals)[::-1]  # descending
        return vals_sorted

    eig_c = _eigvals_sorted(Ktilde_C_train)
    eig_q = _eigvals_sorted(Ktilde_Q_train)

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    x_c = np.arange(1, len(eig_c) + 1)
    x_q = np.arange(1, len(eig_q) + 1)

    ax.semilogy(x_c, eig_c, marker="o", linestyle="-", label="classical (ω=0)")
    ax.semilogy(x_q, eig_q, marker="o", linestyle="-", label="USDR⁺ (ω=1)")

    for w in omegas_to_show:
        if w not in hybrid_mats:
            continue
        eig_h = _eigvals_sorted(hybrid_mats[w].K_train)
        x_h = np.arange(1, len(eig_h) + 1)
        ax.semilogy(
            x_h,
            eig_h,
            marker="o",
            linestyle="-",
            label=rf"hybrid $\omega={w:.2f}$",
        )

    ax.set_xlabel("eigenvalue index (sorted)")
    ax.set_ylabel("eigenvalue (log scale)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)


def plot_effective_rank_vs_omega(
    global_results_df: pd.DataFrame,
    N_values: Optional[Sequence[int]] = None,
    model_type: str = "hybrid_usdr_plus",
    width: float = 16.0,
    height: float = 8.0,
    dpi: int = 300,
) -> None:
    """
    Plot effective rank vs ω (mean ± std across seeds) for each N.

    Parameters
    ----------
    width : float
        Base figure width (inches).
    height : float
        Height PER ROW (inches). Total height = height * n_rows.
    dpi : int
        Figure DPI.
    """
    df = global_results_df.copy()
    if "model_type" in df.columns:
        df = df[df["model_type"] == model_type]

    if N_values is not None:
        df = df[df["N"].isin(N_values)]

    N_unique = sorted(df["N"].unique())
    n_rows = len(N_unique)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(width, height * n_rows),
        dpi=dpi,
        sharex=True,
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = [axes]

    for ax, N in zip(axes, N_unique):
        block = df[df["N"] == N]
        grp = block.groupby("omega")

        omegas = sorted(grp.groups.keys())
        mean_rank = [grp.get_group(w)["effective_rank"].mean() for w in omegas]
        std_rank = [grp.get_group(w)["effective_rank"].std(ddof=0) for w in omegas]

        ax.errorbar(omegas, mean_rank, yerr=std_rank, marker="o", linestyle="-")
        ax.set_title(f"Effective rank vs ω (N={N})")
        ax.set_ylabel("effective rank")
        ax.grid(True)

    axes[-1].set_xlabel("ω")


def plot_condition_number_vs_omega(
    global_results_df: pd.DataFrame,
    N_values: Optional[Sequence[int]] = None,
    model_type: str = "hybrid_usdr_plus",
    use_regularized: bool = True,
    width: float = 16.0,
    height: float = 8.0,
    dpi: int = 300,
) -> None:
    """
    Plot condition number κ(ω) vs ω (mean ± std) for each N.

    height is per-row, like in plot_effective_rank_vs_omega.
    """
    df = global_results_df.copy()
    if "model_type" in df.columns:
        df = df[df["model_type"] == model_type]

    if N_values is not None:
        df = df[df["N"].isin(N_values)]

    col = "kappa_reg" if use_regularized else "kappa_raw"

    N_unique = sorted(df["N"].unique())
    n_rows = len(N_unique)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(width, height * n_rows),
        dpi=dpi,
        sharex=True,
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = [axes]

    for ax, N in zip(axes, N_unique):
        block = df[df["N"] == N]
        grp = block.groupby("omega")

        omegas = sorted(grp.groups.keys())
        mean_kappa = [grp.get_group(w)[col].mean() for w in omegas]
        std_kappa = [grp.get_group(w)[col].std(ddof=0) for w in omegas]

        ax.errorbar(omegas, mean_kappa, yerr=std_kappa, marker="o", linestyle="-")
        ax.set_title(f"{col} vs ω (N={N})")
        ax.set_ylabel(col)
        ax.grid(True)

    axes[-1].set_xlabel("ω")


# ----------------------------------------------------------------------
# 7.2 Performance plots
# ----------------------------------------------------------------------

def plot_mse_vs_omega(
    global_results_df: pd.DataFrame,
    omega_star_df: Optional[pd.DataFrame] = None,
    N_values: Optional[Sequence[int]] = None,
    model_type: str = "hybrid_usdr_plus",
    width: float = 12.0,
    height: float = 8.0,
    dpi: int = 300,
) -> None:
    """
    Plot val/test MSE vs ω (mean ± std across seeds) for each N.
    Also optionally plot a vertical line at mean omega_star per N.

    width is per column; total width = width * 2.
    height is per row; total height = height * n_rows.
    """
    df = global_results_df.copy()
    if "model_type" in df.columns:
        df = df[df["model_type"] == model_type]

    if N_values is not None:
        df = df[df["N"].isin(N_values)]

    if omega_star_df is not None:
        omega_star_df = omega_star_df.copy()
        # one mean ω* per N
        omega_star_mean = (
            omega_star_df.groupby("N")["omega_star"].mean().to_dict()
        )
    else:
        omega_star_mean = {}

    N_unique = sorted(df["N"].unique())
    n_rows = len(N_unique)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(width * 2, height * n_rows),
        dpi=dpi,
        sharex=True,
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = np.array([axes])

    for row_idx, N in enumerate(N_unique):
        block = df[df["N"] == N]
        grp = block.groupby("omega")

        omegas = sorted(grp.groups.keys())
        mean_val = [grp.get_group(w)["mse_val"].mean() for w in omegas]
        std_val = [grp.get_group(w)["mse_val"].std(ddof=0) for w in omegas]
        mean_test = [grp.get_group(w)["mse_test"].mean() for w in omegas]
        std_test = [grp.get_group(w)["mse_test"].std(ddof=0) for w in omegas]

        ax_val = axes[row_idx, 0]
        ax_test = axes[row_idx, 1]

        # Val MSE
        ax_val.errorbar(omegas, mean_val, yerr=std_val, marker="o", linestyle="-")
        ax_val.set_title(f"Val MSE vs ω (N={N})")
        ax_val.set_ylabel("Val MSE")
        ax_val.grid(True)

        # Test MSE
        ax_test.errorbar(omegas, mean_test, yerr=std_test, marker="o", linestyle="-")
        ax_test.set_title(f"Test MSE vs ω (N={N})")
        ax_test.set_ylabel("Test MSE")
        ax_test.grid(True)

        # Highlight ω=0 and ω=1 (if present)
        for ax in (ax_val, ax_test):
            if 0.0 in omegas:
                idx0 = omegas.index(0.0)
                ax.scatter(
                    omegas[idx0],
                    [mean_val, mean_test][(ax is ax_test)][idx0],
                )
            if 1.0 in omegas:
                idx1 = omegas.index(1.0)
                ax.scatter(
                    omegas[idx1],
                    [mean_val, mean_test][(ax is ax_test)][idx1],
                )

            # Vertical line at mean omega_star (if available)
            if N in omega_star_mean:
                w_star_mean = omega_star_mean[N]
                ax.axvline(
                    w_star_mean,
                    linestyle="--",
                )

    axes[-1, 0].set_xlabel("ω")
    axes[-1, 1].set_xlabel("ω")


def plot_summary_bars_per_N(
    global_results_df: pd.DataFrame,
    omega_star_df: pd.DataFrame,
    model_type: str = "hybrid_usdr_plus",
    width: float = 24.0,
    height: float = 10.0,
    dpi: int = 300,
) -> None:
    """
    For each N, make a bar chart with:
        - test MSE at ω=0 (pure classical, as seen by hybrid kernel)
        - test MSE at ω=1 (pure USDR⁺ bounded)
        - test MSE at ω=omega_star (best hybrid)

    width, height, dpi as usual refer to full figure.
    """
    df = global_results_df.copy()
    if "model_type" in df.columns:
        df = df[df["model_type"] == model_type]

    # Mean test MSE at ω=0 and ω=1
    mean_test_by_N_omega = (
        df.groupby(["N", "omega"])["mse_test"]
        .mean()
        .reset_index()
    )

    def _get_mean_test_for_omega(N: int, omega: float) -> float:
        mask = (mean_test_by_N_omega["N"] == N) & (
            mean_test_by_N_omega["omega"] == omega
        )
        sub = mean_test_by_N_omega[mask]
        return float(sub["mse_test"].iloc[0]) if not sub.empty else np.nan

    # Mean test MSE at ω* from omega_star_df
    mean_test_star_by_N = (
        omega_star_df.groupby("N")["mse_test_star"]
        .mean()
        .to_dict()
    )

    N_unique = sorted(df["N"].unique())
    n_N = len(N_unique)

    bar_width = 0.25
    x = np.arange(n_N)

    classical_vals = [_get_mean_test_for_omega(N, 0.0) for N in N_unique]
    usdr_vals = [_get_mean_test_for_omega(N, 1.0) for N in N_unique]
    hybrid_vals = [mean_test_star_by_N.get(N, np.nan) for N in N_unique]

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

    ax.bar(x - bar_width, classical_vals, width=bar_width, label="ω=0 (classical)")
    ax.bar(x, usdr_vals, width=bar_width, label="ω=1 (USDR⁺)")
    ax.bar(x + bar_width, hybrid_vals, width=bar_width, label="ω=ω* (hybrid best)")

    ax.set_xticks(x)
    ax.set_xticklabels([str(N) for N in N_unique])
    ax.set_xlabel("N")
    ax.set_ylabel("Mean Test MSE")
    ax.set_title("Test MSE comparison per N: classical vs USDR⁺ vs hybrid (ω*)")
    ax.legend()
    ax.grid(True, axis="y")
    ax.yaxis.set_major_locator(MaxNLocator(integer=False))


# ----------------------------------------------------------------------
# 7.3 Predictions & residuals
# ----------------------------------------------------------------------

def plot_2d_true_vs_pred_surfaces(
    X1_grid: np.ndarray,
    X2_grid: np.ndarray,
    y_true_grid: np.ndarray,
    y_pred_classical_grid: np.ndarray,
    y_pred_usdr_grid: np.ndarray,
    y_pred_hybrid_grid: np.ndarray,
    title_prefix: str = "2D regression surfaces",
    width: float = 16.0,
    height: float = 12.0,
    dpi: int = 300,
) -> None:
    """
    Plot 3D surfaces (or contour maps) for:
        - true function,
        - classical prediction,
        - USDR⁺ prediction,
        - hybrid prediction.

    width/height are per row/column: total width = width * 2, total height = height * 2.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.subplots(
        2,
        2,
        figsize=(width * 2, height * 2),
        dpi=dpi,
        subplot_kw={"projection": "3d"},
    )[0]
    axes = np.array(fig.axes).reshape(2, 2)

    surfaces = [
        (y_true_grid, "True function"),
        (y_pred_classical_grid, "Prediction (ω=0, classical)"),
        (y_pred_usdr_grid, "Prediction (ω=1, USDR⁺)"),
        (y_pred_hybrid_grid, "Prediction (ω=ω*, hybrid)"),
    ]

    for ax, (Z, name) in zip(axes.ravel(), surfaces):
        ax.plot_surface(X1_grid, X2_grid, Z)
        ax.set_title(name)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("y")

    fig.suptitle(title_prefix)


def plot_true_vs_pred_scatter(
    y_true_test: np.ndarray,
    y_pred_classical: np.ndarray,
    y_pred_usdr: np.ndarray,
    y_pred_hybrid: np.ndarray,
    title_prefix: str = "True vs predicted (test set)",
    width: float = 10.0,
    height: float = 8.0,
    dpi: int = 300,
) -> None:
    """
    Plot 1x3 scatter: y_true vs y_pred for classical, USDR⁺, hybrid.

    Total width = width * 3, total height = height.
    """
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(width * 3, height),
        dpi=dpi,
        constrained_layout=True,
    )
    configs = [
        (y_pred_classical, "Classical (ω=0)"),
        (y_pred_usdr, "USDR⁺ (ω=1)"),
        (y_pred_hybrid, "Hybrid (ω=ω*)"),
    ]

    for ax, (y_pred, name) in zip(axes, configs):
        ax.scatter(y_true_test, y_pred, s=20)
        min_val = min(y_true_test.min(), y_pred.min())
        max_val = max(y_true_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val])
        ax.set_title(name)
        ax.set_xlabel("y_true")
        ax.set_ylabel("y_pred")
        ax.grid(True)

    fig.suptitle(title_prefix)


def plot_residual_distributions(
    y_true_test: np.ndarray,
    y_pred_classical: np.ndarray,
    y_pred_usdr: np.ndarray,
    y_pred_hybrid: np.ndarray,
    bins: int = 20,
    title: str = "Residual distributions (test set)",
    width: float = 10.0,
    height: float = 8.0,
    dpi: int = 300,
) -> None:
    """
    Plot histograms of residuals for classical, USDR⁺, and hybrid.

    Total width = width * 3, total height = height.
    """
    res_classical = y_pred_classical - y_true_test
    res_usdr = y_pred_usdr - y_true_test
    res_hybrid = y_pred_hybrid - y_true_test

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(width * 3, height),
        dpi=dpi,
        constrained_layout=True,
    )
    configs = [
        (res_classical, "Classical (ω=0)"),
        (res_usdr, "USDR⁺ (ω=1)"),
        (res_hybrid, "Hybrid (ω=ω*)"),
    ]

    for ax, (res, name) in zip(axes, configs):
        ax.hist(res, bins=bins, density=True)
        ax.set_title(name)
        ax.set_xlabel("residual (y_pred - y_true)")
        ax.set_ylabel("density")
        ax.grid(True)

    fig.suptitle(title)

# ===== CELL 015 (code) =====
# ======================================================================
# Summary statistics & interpretation helpers for hybrid USDR⁺ kernel
# ======================================================================

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

def get_hybrid_summary_logger(name: str = "usdr_plus.hybrid_summary") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


HYBRID_SUMMARY_LOGGER = get_hybrid_summary_logger()


# ----------------------------------------------------------------------
# 30. Summary table per N
# ----------------------------------------------------------------------

def build_hybrid_summary_table_per_N(
    global_results_df: pd.DataFrame,
    omega_star_df: pd.DataFrame,
    model_type: str = "hybrid_usdr_plus",
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Build a summary table with one row per N, containing:
        - mean_test_MSE(ω=0)
        - mean_test_MSE(ω=1)
        - mean_test_MSE(ω_star)
        - mean_omega_star
        - mean_rank_eff(ω_star)
        - mean_cond_reg(ω_star)

    Parameters
    ----------
    global_results_df : pd.DataFrame
        Global results with columns at least:
            - N
            - omega
            - mse_test
            - model_type (optional)
    omega_star_df : pd.DataFrame
        Per-(N, seed) ω* summary from compute_omega_star_per_run, with:
            - N
            - seed
            - omega_star
            - mse_test_star
            - effective_rank_star
            - kappa_reg_star
    model_type : str
        If 'model_type' exists in global_results_df, filter to this value.
    logger : logging.Logger, optional

    Returns
    -------
    summary_df : pd.DataFrame
        One row per N, with columns:
            - N
            - mean_test_mse_omega0
            - mean_test_mse_omega1
            - mean_test_mse_omega_star
            - mean_omega_star
            - mean_effective_rank_star
            - mean_kappa_reg_star
    """
    logger = logger or HYBRID_SUMMARY_LOGGER

    df = global_results_df.copy()
    if "model_type" in df.columns:
        df = df[df["model_type"] == model_type]

    # Sanity checks
    required_cols_global = {"N", "omega", "mse_test"}
    missing_global = required_cols_global.difference(df.columns)
    if missing_global:
        raise ValueError(
            f"global_results_df missing required columns: {sorted(missing_global)}"
        )

    required_cols_star = {
        "N",
        "omega_star",
        "mse_test_star",
        "effective_rank_star",
        "kappa_reg_star",
    }
    missing_star = required_cols_star.difference(omega_star_df.columns)
    if missing_star:
        raise ValueError(
            f"omega_star_df missing required columns: {sorted(missing_star)}"
        )

    # Means at ω=0 and ω=1 from global_results_df
    mean_test_by_N_omega = (
        df.groupby(["N", "omega"])["mse_test"]
        .mean()
        .reset_index()
    )

    def _get_mean_test_for_omega(N: int, omega: float) -> float:
        mask = (mean_test_by_N_omega["N"] == N) & (
            mean_test_by_N_omega["omega"] == omega
        )
        sub = mean_test_by_N_omega[mask]
        return float(sub["mse_test"].iloc[0]) if not sub.empty else np.nan

    # Means at ω* from omega_star_df
    omega_star_stats = (
        omega_star_df.groupby("N")
        .agg(
            mean_omega_star=("omega_star", "mean"),
            mean_mse_test_star=("mse_test_star", "mean"),
            mean_effective_rank_star=("effective_rank_star", "mean"),
            mean_kappa_reg_star=("kappa_reg_star", "mean"),
        )
        .reset_index()
    )

    N_unique = sorted(df["N"].unique())
    records: list[Dict[str, Any]] = []

    logger.info(
        "Building hybrid summary table per N for %d distinct N values.",
        len(N_unique),
    )

    for N_val in N_unique:
        mean_test_omega0 = _get_mean_test_for_omega(N_val, 0.0)
        mean_test_omega1 = _get_mean_test_for_omega(N_val, 1.0)

        # Get ω* stats for this N (if present)
        row_star = omega_star_stats[omega_star_stats["N"] == N_val]
        if not row_star.empty:
            r = row_star.iloc[0]
            mean_omega_star = float(r["mean_omega_star"])
            mean_test_omega_star = float(r["mean_mse_test_star"])
            mean_eff_rank_star = float(r["mean_effective_rank_star"])
            mean_kappa_reg_star = float(r["mean_kappa_reg_star"])
        else:
            mean_omega_star = np.nan
            mean_test_omega_star = np.nan
            mean_eff_rank_star = np.nan
            mean_kappa_reg_star = np.nan
            logger.warning(
                "No ω* statistics found for N=%d in omega_star_df.", N_val
            )

        logger.info(
            "N=%d: mean_test(ω=0)=%.4e, mean_test(ω=1)=%.4e, "
            "mean_test(ω*)=%.4e, mean_ω*=%.3f, "
            "mean_r_eff(ω*)=%.3f, mean_κ_reg(ω*)=%.3e",
            N_val,
            mean_test_omega0,
            mean_test_omega1,
            mean_test_omega_star,
            mean_omega_star,
            mean_eff_rank_star,
            mean_kappa_reg_star,
        )

        records.append(
            {
                "N": N_val,
                "mean_test_mse_omega0": mean_test_omega0,
                "mean_test_mse_omega1": mean_test_omega1,
                "mean_test_mse_omega_star": mean_test_omega_star,
                "mean_omega_star": mean_omega_star,
                "mean_effective_rank_star": mean_eff_rank_star,
                "mean_kappa_reg_star": mean_kappa_reg_star,
            }
        )

    summary_df = pd.DataFrame.from_records(records)
    summary_df.sort_values(by="N", inplace=True, ignore_index=True)
    return summary_df


# ----------------------------------------------------------------------
# 31. Correlations & scatter plots (test MSE vs rank / condition)
# ----------------------------------------------------------------------

@dataclass
class HybridCorrelationResults:
    """
    Correlations of test MSE with spectral quantities across all (N, seed, ω).

    Attributes
    ----------
    corr_mse_rank : Optional[float]
        Pearson correlation between mse_test and effective_rank.
    corr_mse_kappa_reg : Optional[float]
        Pearson correlation between mse_test and kappa_reg.
    corr_mse_kappa_raw : Optional[float]
        Pearson correlation between mse_test and kappa_raw.
    """
    corr_mse_rank: Optional[float]
    corr_mse_kappa_reg: Optional[float]
    corr_mse_kappa_raw: Optional[float]


def compute_hybrid_correlations(
    global_results_df: pd.DataFrame,
    model_type: str = "hybrid_usdr_plus",
    logger: Optional[logging.Logger] = None,
) -> HybridCorrelationResults:
    """
    Compute correlations across all (N, seed, ω) between:
        - test MSE and effective rank,
        - test MSE and kappa_reg,
        - test MSE and kappa_raw.

    Parameters
    ----------
    global_results_df : pd.DataFrame
        Global results with columns:
            - mse_test
            - effective_rank
            - kappa_reg
            - kappa_raw
            - model_type (optional)
    model_type : str
        If 'model_type' exists, filter to this value.
    logger : logging.Logger, optional

    Returns
    -------
    HybridCorrelationResults
    """
    logger = logger or HYBRID_SUMMARY_LOGGER

    df = global_results_df.copy()
    if "model_type" in df.columns:
        df = df[df["model_type"] == model_type]

    required_cols = {
        "mse_test",
        "effective_rank",
        "kappa_reg",
        "kappa_raw",
        "omega",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"global_results_df missing required columns for correlations: {sorted(missing)}"
        )

    # Drop rows with NaNs in the relevant columns
    df_corr = df[["mse_test", "effective_rank", "kappa_reg", "kappa_raw"]].dropna()
    if df_corr.empty:
        logger.warning(
            "No valid rows for correlation computation (all NaNs?). "
            "Returning NaNs."
        )
        return HybridCorrelationResults(
            corr_mse_rank=np.nan,
            corr_mse_kappa_reg=np.nan,
            corr_mse_kappa_raw=np.nan,
        )

    corr_matrix = df_corr.corr(method="pearson")
    corr_mse_rank = float(corr_matrix.loc["mse_test", "effective_rank"])
    corr_mse_kappa_reg = float(corr_matrix.loc["mse_test", "kappa_reg"])
    corr_mse_kappa_raw = float(corr_matrix.loc["mse_test", "kappa_raw"])

    logger.info("Correlation(mse_test, effective_rank) = %.3f", corr_mse_rank)
    logger.info("Correlation(mse_test, kappa_reg)     = %.3f", corr_mse_kappa_reg)
    logger.info("Correlation(mse_test, kappa_raw)     = %.3f", corr_mse_kappa_raw)

    return HybridCorrelationResults(
        corr_mse_rank=corr_mse_rank,
        corr_mse_kappa_reg=corr_mse_kappa_reg,
        corr_mse_kappa_raw=corr_mse_kappa_raw,
    )


def plot_mse_vs_rank_and_condition_scatter(
    global_results_df: pd.DataFrame,
    model_type: str = "hybrid_usdr_plus",
    use_regularized: bool = True,
) -> None:
    """
    Scatter plots:
        - test MSE vs effective rank,
        - test MSE vs condition number,
    with points colored by ω.

    Parameters
    ----------
    global_results_df : pd.DataFrame
        DataFrame with columns:
            - mse_test
            - effective_rank
            - kappa_raw
            - kappa_reg
            - omega
            - model_type (optional)
    model_type : str
        If 'model_type' column exists, filter to this value.
    use_regularized : bool
        If True, use kappa_reg; otherwise use kappa_raw.
    """
    df = global_results_df.copy()
    if "model_type" in df.columns:
        df = df[df["model_type"] == model_type]

    col_kappa = "kappa_reg" if use_regularized else "kappa_raw"

    required_cols = {"mse_test", "effective_rank", col_kappa, "omega"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"global_results_df missing required columns for scatter: {sorted(missing)}"
        )

    df = df.dropna(subset=["mse_test", "effective_rank", col_kappa, "omega"])
    if df.empty:
        HYBRID_SUMMARY_LOGGER.warning(
            "No valid rows for scatter plots (NaNs after filtering)."
        )
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Left: MSE vs effective rank
    ax_rank = axes[0]
    sc1 = ax_rank.scatter(df["effective_rank"], df["mse_test"], c=df["omega"])
    ax_rank.set_xlabel("effective rank")
    ax_rank.set_ylabel("test MSE")
    ax_rank.set_title("Test MSE vs effective rank")
    ax_rank.grid(True)
    cbar1 = fig.colorbar(sc1, ax=ax_rank)
    cbar1.set_label("ω")

    # Right: MSE vs condition number
    ax_kappa = axes[1]
    sc2 = ax_kappa.scatter(df[col_kappa], df["mse_test"], c=df["omega"])
    ax_kappa.set_xlabel(col_kappa)
    ax_kappa.set_ylabel("test MSE")
    ax_kappa.set_title(f"Test MSE vs {col_kappa}")
    ax_kappa.grid(True)
    cbar2 = fig.colorbar(sc2, ax=ax_kappa)
    cbar2.set_label("ω")

    fig.suptitle("Hybrid kernel: test MSE vs spectral complexity")

# ===== CELL 016 (markdown) =====
# **True function**

# ===== CELL 017 (code) =====
def true_function(x1, x2, add_noise: bool = True):
    """
    Ground truth:
        f(x1, x2) = sin(x1) + cos(x2) + 0.1 * x1 * x2
    Target:
        y = f(x1, x2) + ε,  ε ~ N(0, noise_std)
    """
    base = np.sin(x1) + np.cos(x2) + 0.1 * x1 * x2

    if not add_noise:
        return base

    eps = np.random.normal(loc=0.0,
                           scale=noise_std,
                           size=np.broadcast(x1, x2).shape)
    return base + eps

# ===== CELL 018 (markdown) =====
# **Dataset generation**

# ===== CELL 019 (code) =====
def generate_datasets(sample_sizes, noise_std, domain=RAW_DOMAIN):
    """
    Generate raw datasets for each N in sample_sizes.

    For each N:
      - Sample x1, x2 ~ Uniform(domain[0], domain[1])
      - Compute noiseless ground truth f(x1, x2)
      - Add Gaussian noise ε ~ N(0, noise_std) to obtain y

    Returns
    -------
    datasets : dict
        {
          N: {
            "x1": np.ndarray shape (N,),
            "x2": np.ndarray shape (N,),
            "y":  np.ndarray shape (N,),
            "y_true": np.ndarray shape (N,)
          },
          ...
        }
    """
    datasets = {}

    for N in sample_sizes:
        x1 = np.random.uniform(domain[0], domain[1], size=N)
        x2 = np.random.uniform(domain[0], domain[1], size=N)

        # noiseless ground truth
        y_true = true_function(x1, x2, add_noise=False)

        # add Gaussian noise
        eps = np.random.normal(loc=0.0, scale=noise_std, size=N)
        y_noisy = y_true + eps

        datasets[N] = {
            "x1": x1,
            "x2": x2,
            "y": y_noisy,
            "y_true": y_true,
        }

    return datasets

# ===== CELL 020 (markdown) =====
# **Visualization grid**

# ===== CELL 021 (code) =====
def generate_test_grid(grid_size: int = grid_size, domain=RAW_DOMAIN):
    """
    Generate a 2D evaluation grid over the raw domain.

    Returns
    -------
    X1, X2 : np.ndarray shape (grid_size, grid_size)
        Meshgrid coordinates in [domain[0], domain[1]].
    Y_true : np.ndarray shape (grid_size, grid_size)
        Noiseless ground-truth values f(x1, x2).
    """
    x1 = np.linspace(domain[0], domain[1], grid_size)
    x2 = np.linspace(domain[0], domain[1], grid_size)
    X1, X2 = np.meshgrid(x1, x2)

    # Noiseless ground truth on the grid
    Y_true = true_function(X1, X2, add_noise=False)

    return X1, X2, Y_true

# ===== CELL 022 (markdown) =====
# **Split and preprocess (70/15/15, fit on train only)**

# ===== CELL 023 (code) =====
def save_split(folder: Path, name: str, X: np.ndarray, y: np.ndarray) -> None:
    """
    Save a split (train/val/test) to CSV with columns x1, x2, y.
    X is assumed to be already preprocessed (e.g. in [0,1]^2 if MinMax).
    """
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["y"] = y
    df.to_csv(folder / f"{name}.csv", index=False)


def preprocess_and_save_2d_datasets(
    sample_sizes,
    noise_std,
    output_dir: str = "preprocessed/hybrid",
    normalize: str = NORMALIZE,
):
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    for N in sample_sizes:
        for seed in SEEDS:
            # 1) Independent random draw for each (N, seed)
            set_all_seeds(seed)
            x1 = np.random.uniform(RAW_DOMAIN[0], RAW_DOMAIN[1], size=N)
            x2 = np.random.uniform(RAW_DOMAIN[0], RAW_DOMAIN[1], size=N)

            y_true = true_function(x1, x2, add_noise=False)
            eps    = np.random.normal(loc=0.0, scale=noise_std, size=N)
            y      = y_true + eps

            X_raw = np.stack([x1, x2], axis=1)

            # 2) 70/15/15 split (using the *same* RNG stream is fine)
            n_train = int(0.70 * N)
            n_val   = int(0.15 * N)
            n_test  = N - n_train - n_val

            idx = np.random.permutation(N)
            train_idx = idx[:n_train]
            val_idx   = idx[n_train:n_train + n_val]
            test_idx  = idx[n_train + n_val:]

            X_train_raw, y_train = X_raw[train_idx], y[train_idx]
            X_val_raw,   y_val   = X_raw[val_idx],   y[val_idx]
            X_test_raw,  y_test  = X_raw[test_idx],  y[test_idx]

            # 3) Normalization (MinMax or Z-score, train-only)
            if normalize == "minmax":
                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(X_train_raw)
                X_val   = np.clip(scaler.transform(X_val_raw),  0.0, 1.0)
                X_test  = np.clip(scaler.transform(X_test_raw), 0.0, 1.0)
            elif normalize == "zscore":
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train_raw)
                X_val   = scaler.transform(X_val_raw)
                X_test  = scaler.transform(X_test_raw)
            else:
                raise ValueError(f"Unknown NORMALIZE mode: {normalize}")

            # 4) Save
            folder = output_dir_path / f"N{N}_seed{seed}"
            folder.mkdir(parents=True, exist_ok=True)

            save_split(folder, "train", X_train, y_train)
            save_split(folder, "val",   X_val,   y_val)
            save_split(folder, "test",  X_test,  y_test)

            print(
                f"[USDR+] Saved N={N}, seed={seed} → "
                f"{n_train}/{n_val}/{n_test} train/val/test "
                f"(NORMALIZE={normalize})"
            )

# ===== CELL 024 (code) =====
preprocess_and_save_2d_datasets(
    sample_sizes=sample_sizes,
    noise_std=noise_std,
    output_dir=str(OUTPUT_DIR),
    normalize=NORMALIZE,
)

# ===== CELL 025 (markdown) =====
# **Utilities**

# ===== CELL 026 (code) =====
def plot_2d_dataset(
    datasets: dict,
    X1: np.ndarray,
    X2: np.ndarray,
    Y_true_grid: np.ndarray,
    N: int = 100,
    width: float = 14.0,
    height: float = 6.0,
    dpi: int = 300,
    theme: str = "whitegrid",
) -> None:
    """
    Visualise the **true function surface** (sin(x₁)+cos(x₂)+0.1·x₁·x₂)
    together with the **noisy raw samples** (y) for a given N.

    Fully compliant with U_{SDR+} protocol:
      • uses *raw* data from generate_datasets (before any scaling)
      • 60×60 grid is supplied externally (X1, X2, Y_true_grid)
      • N ∈ {50,100,200}
    """
    sns.set_theme(style=theme, context="talk")
    fig = plt.figure(figsize=(width, height), dpi=dpi)

    # ---- 1. True surface -------------------------------------------------
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax1.plot_surface(
        X1, X2, Y_true_grid,
        cmap="viridis", edgecolor="none", alpha=0.9, antialiased=True
    )
    ax1.set_title(r"True $f(x_1,x_2)$", fontsize=14, pad=12)
    ax1.set_xlabel(r"$x_1$")
    ax1.set_ylabel(r"$x_2$")
    ax1.set_zlabel(r"$y$")
    fig.colorbar(surf, ax=ax1, shrink=0.6, aspect=12)

    # ---- 2. Noisy raw points --------------------------------------------
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    d = datasets[N]  # raw dict from generate_datasets: x1, x2, y, y_true
    ax2.scatter(
        d["x1"],
        d["x2"],
        d["y"],
        s=60,
        c="tab:red",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85,
        label=f"Noisy samples ($N={N}$)",
    )
    ax2.set_title(f"Raw noisy data ($N={N}$)", fontsize=14, pad=12)
    ax2.set_xlabel(r"$x_1$")
    ax2.set_ylabel(r"$x_2$")
    ax2.set_zlabel(r"$y$")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


def plot_all_resolutions(
    datasets: dict,
    X1: np.ndarray,
    X2: np.ndarray,
    Y_true_grid: np.ndarray,
    Ns: list[int] = None,
) -> None:
    """
    Plot true surface + raw noisy samples for all N in Ns (default {50,100,200}).
    """
    if Ns is None:
        Ns = [50, 100, 200]

    n = len(Ns)
    fig, axes = plt.subplots(
        1, n, figsize=(28, 6), dpi=300,
        subplot_kw=dict(projection="3d")
    )

    for ax, N in zip(axes, Ns):
        # true surface
        ax.plot_surface(
            X1, X2, Y_true_grid,
            cmap="viridis", edgecolor="none", alpha=0.9
        )
        # noisy raw points
        d = datasets[N]
        ax.scatter(
            d["x1"], d["x2"], d["y"],
            s=50, c="tab:red", edgecolor="k", alpha=0.8
        )
        ax.set_title(f"$N={N}$")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("$y$")

    plt.tight_layout()
    plt.show()


def visualize_dataset_splits(
    folder: Path,
    n_rows: int = 5,
    width: float = 18,
    height: float = 6,
    dpi: int = 200,
    theme: str = "darkgrid",
    palette: str = "viridis",
) -> None:
    """
    Visualize **preprocessed** train/val/test splits
    (MinMax or Z-score scaled according to NORMALIZE).

    Compliant with protocol:
      • scaler fitted on train only (see preprocessing function)
      • here we just inspect the saved CSVs.
    """
    sns.set_theme(style=theme, palette=palette, context="talk")

    # Load processed CSVs
    train_full = pd.read_csv(folder / "train.csv")
    val_full   = pd.read_csv(folder / "val.csv")
    test_full  = pd.read_csv(folder / "test.csv")

    train_df = train_full.head(n_rows)
    val_df   = val_full.head(n_rows)
    test_df  = test_full.head(n_rows)

    # Print tables
    print(f"\n=== Training Split (N={len(train_full)}) ===")
    display(train_df.style.background_gradient(cmap="Blues"))
    print(f"\n=== Validation Split (N={len(val_full)}) ===")
    display(val_df.style.background_gradient(cmap="Greens"))
    print(f"\n=== Test Split (N={len(test_full)}) ===")
    display(test_df.style.background_gradient(cmap="Oranges"))

    # Scatter in feature space
    fig, axes = plt.subplots(1, 3, figsize=(width, height), dpi=dpi)
    scatter = dict(s=80, edgecolor='k', linewidth=0.7, alpha=0.8)

    for ax, df, name in zip(
        axes,
        [train_df, val_df, test_df],
        ["Train", "Val", "Test"],
    ):
        sc = ax.scatter(
            df["x1"], df["x2"],
            c=df["y"], cmap=palette, **scatter
        )
        ax.set_title(f"{name} (first {n_rows})", weight="bold")
        ax.set_xlabel("x1 (scaled)")
        ax.set_ylabel("x2 (scaled)")
        ax.grid(True, ls="--", alpha=0.6)
        fig.colorbar(sc, ax=ax, shrink=0.7)

    plt.suptitle("Preprocessed Dataset Splits", fontsize=18, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_gram_matrix(
    matrix: np.ndarray,
    title: str = "Gram Matrix",
    cmap: str = "coolwarm",
    dpi: int = 300,
    width: float = 8,
    height: float = 6,
    annotate: bool = True,
    save: bool = True,
    save_dir: str | Path = "figures/hybrid/diagnostics",
) -> None:
    """
    Plot a Gram matrix as a heatmap with diagonal highlighting.
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

    sns.heatmap(
        matrix,
        annot=annotate,
        fmt=".2f",
        cmap=cmap,
        square=True,
        cbar=True,
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
    )

    # Highlight diagonal
    for i in range(matrix.shape[0]):
        ax.add_patch(
            plt.Rectangle(
                (i, i), 1, 1, fill=False, edgecolor='red', lw=2
            )
        )

    ax.set_title(title, fontsize=16, pad=12)
    plt.tight_layout()
    if save:
        slug = "_".join(title.strip().lower().split())
        out = Path(save_dir) / f"gram_heatmap_{slug or 'matrix'}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"[PLOT] Saved Gram matrix heatmap -> {out}")
    plt.show()


def plot_krr_predictions_vs_true_value(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    width: float = 7,
    height: float = 5,
    dpi: int = 300,
    theme: str = "whitegrid",
    point_color: str = "tab:blue",
    title: str = "KRR Predictions vs True Values",
) -> None:
    """
    Plot Kernel Ridge Regression predictions vs true values.
    """
    sns.set_theme(style=theme, context="talk")

    plt.figure(figsize=(width, height), dpi=dpi)

    # Scatter plot of predictions vs true values
    plt.scatter(
        y_true, y_pred,
        color=point_color, alpha=0.7,
        edgecolors="k", s=70, linewidth=0.6,
        label="Predictions",
    )

    # Ideal line (y = x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--", lw=2, label="Ideal",
    )

    plt.xlabel("True Values", fontsize=12, weight="bold")
    plt.ylabel("Predicted Values", fontsize=12, weight="bold")
    plt.title(title, fontsize=14, weight="bold")

    plt.legend(frameon=True, fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()


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

# ===== CELL 027 (markdown) =====
# **Plot results**

# ===== CELL 028 (code) =====
# If you don't have this in the current session, run it first:
raw_datasets = generate_datasets(
    sample_sizes=sample_sizes,
    noise_std=noise_std,
    domain=RAW_DOMAIN,
)

# Create the 2D evaluation grid over [0, 2π]²
X1, X2, Y_true_grid = generate_test_grid(
    grid_size=grid_size,
    domain=RAW_DOMAIN,
)

# Now this will work
plot_2d_dataset(
    datasets=raw_datasets,  # <-- was datasets=datasets
    X1=X1,
    X2=X2,
    Y_true_grid=Y_true_grid,
    N=100,
    width=28,
    height=12,
    dpi=300,
    theme="darkgrid",
)

# ===== CELL 029 (code) =====
plot_all_resolutions(
    datasets=raw_datasets,
    X1=X1,
    X2=X2,
    Y_true_grid=Y_true_grid,
)

# ===== CELL 030 (markdown) =====
# # **Implement the $U_{\text{SDR}+}$ feature map**

# ===== CELL 031 (markdown) =====
# **Load processed 2D dataset**

# ===== CELL 032 (code) =====
def load_processed_2d_dataset(
    base_path: str | Path = OUTPUT_DIR,
    N: int = 100,
    seed: int = 0,
    eps: float = 1e-12,
    normalize: str = NORMALIZE,
) -> dict:
    """
    Fully U_{SDR+}-compliant loader with safe numerical tolerance.

    Assumptions (USDR+ protocol):
      • Data were generated with x1,x2 ∈ [0, 2π] and then preprocessed via
        - MinMax (train-only) → X ∈ [0,1]^2, or
        - Z-score (train-only) → X standardized (unbounded).
      • This function just loads the *preprocessed* CSVs produced by
        `preprocess_and_save_2d_datasets`.

    Parameters
    ----------
    base_path : str or Path
        Root folder where processed datasets live (e.g. OUTPUT_DIR).
    N : int
        Sample size (one of {50, 100, 200}).
    seed : int
        Split seed (0, 1, 2).
    eps : float
        Numerical tolerance for MinMax clipping checks.
    normalize : {"minmax", "zscore"}
        Preprocessing mode used when creating the dataset.
    """
    base_path = Path(base_path)
    folder = base_path / f"N{N}_seed{seed}"
    if not folder.exists():
        raise FileNotFoundError(f"[USDR+] Dataset folder not found: {folder}")

    splits = ["train", "val", "test"]
    data: dict[str, object] = {
        "metadata": {
            "N": N,
            "seed": seed,
            "normalize": normalize,
            "base_path": str(base_path),
        }
    }

    for split in splits:
        file = folder / f"{split}.csv"
        if not file.is_file():
            raise FileNotFoundError(f"[USDR+] Missing file: {file}")

        df = pd.read_csv(file)
        required = {"x1", "x2", "y"}
        if not required.issubset(df.columns):
            raise ValueError(f"[USDR+] Missing columns in {file}: "
                             f"expected {required}, got {set(df.columns)}")

        X = df[["x1", "x2"]].values.astype(np.float64)
        y = df["y"].values.astype(np.float64)

        if normalize == "minmax":
            # === FINAL FIX: safe clipping + tolerance in [0,1]² ===
            X = np.clip(X, 0.0, 1.0)  # removes 1.0000000000000002 issues
            if not np.all((X >= 0.0 - eps) & (X <= 1.0 + eps)):
                print(f"[USDR+] WARNING: X slightly outside [0,1] in {split} (clipped)")
        elif normalize == "zscore":
            # For Z-score, no clipping: X is standardized, unbounded.
            pass
        else:
            raise ValueError(f"[USDR+] Unknown NORMALIZE mode: {normalize}")

        data[f"X_{split}"] = X
        data[f"y_{split}"] = y
        data["metadata"][f"n_{split}"] = len(y)

        print(
            f"[USDR+] Loaded {split:>5} | "
            f"X shape {X.shape} | y shape {y.shape} | "
            f"NORMALIZE={normalize}"
        )

    print(
        f"\n[USDR+] Dataset N={N}, seed={seed} loaded from {base_path} → "
        f"{data['metadata']['n_train']}/"
        f"{data['metadata']['n_val']}/"
        f"{data['metadata']['n_test']} "
        f"(NORMALIZE={normalize})\n"
    )
    return data

# ===== CELL 033 (code) =====
for N in sample_sizes:       # [50, 100, 200]
    for SEED in SEEDS:       # [0, 1, 2]
        data = load_processed_2d_dataset(
            base_path=OUTPUT_DIR,
            N=N,
            seed=SEED,
            normalize=NORMALIZE,
        )
        print(
            f"N={N}, seed={SEED} → "
            f"train {data['X_train'].shape}, "
            f"val {data['X_val'].shape}, "
            f"test {data['X_test'].shape}"
        )

# ===== CELL 034 (code) =====
visualize_dataset_splits(
    OUTPUT_DIR / "N50_seed0",   # ← folder Path
    n_rows=15, width=20, height=6, dpi=300, theme="whitegrid", palette="coolwarm"
)

# ===== CELL 035 (markdown) =====
# # **Define $U_{\text{SDR}+}$**

# ===== CELL 036 (code) =====
def U_SDR_plus(
    x: np.ndarray,
    theta,
    *,
    L: int = 2,
    entangler: str = "cnot",
) -> None:
    """
    U_{SDR+} feature map.

    Assumptions (USDR+ protocol):
      • x is a *preprocessed* 2D feature vector:
          - if NORMALIZE="minmax": x ∈ [0, 1]^2
          - if NORMALIZE="zscore": x is standardized (unbounded)
      • This routine ONLY applies the bandwidth scaling via β:
            x̂ = x / β
      • L = 2, entangler = "cnot", axes_low = (X,Z), axes_high = (Z,X).

    Parameters
    ----------
    x : array-like of shape (2,)
        Preprocessed input features (x1_tilde, x2_tilde).
    theta : iterable
        (lambda1, lambda2, gamma, beta).
    L : int, default=2
        Number of data-reuploading layers (fixed to 2 in USDR+).
    entangler : {"cnot"}, default="cnot"
        Entangling gate; USDR+ uses CNOT(0→1).
    """
    lambda1, lambda2, gamma, beta = theta

    # USDR+: x̂ = x̃ / β
    x_hat = x / beta

    # Fixed axes as per protocol
    low_axes  = ("X", "Z")  # low-freq block
    high_axes = ("Z", "X")  # high-freq block

    for _ in range(L):
        # --- Low-frequency block (smooth structure) ---
        # axes_low = (X, Z)
        qml.RX(lambda1 * x_hat[0], wires=0)
        qml.RZ(lambda2 * x_hat[1], wires=1)
        if entangler == "cnot":
            qml.CNOT(wires=[0, 1])

        # --- High-frequency block (γ-boost for interactions) ---
        # axes_high = (Z, X)
        qml.RZ(gamma * lambda1 * x_hat[0], wires=0)
        qml.RX(gamma * lambda2 * x_hat[1], wires=1)
        if entangler == "cnot":
            qml.CNOT(wires=[0, 1])

# ===== CELL 037 (code) =====
def visualize_U_SDR_plus_2D(
    x_example: np.ndarray,
    theta,
    width: float = 12,
    height: float = 6,
    dpi: int = 300,
    L: int = 2,
    entangler: str = "cnot",
    save: bool = True,
    save_dir: str | Path = "figures/hybrid/diagnostics",
    plot_name: str = "usdr_plus_circuit",
) -> None:
    """
    Visualize the U_{SDR+} feature map for a single 2D input.

    Assumptions (USDR+ protocol):
      • x_example is a *preprocessed* feature vector:
          - if NORMALIZE="minmax": x_example ∈ [0, 1]^2
          - if NORMALIZE="zscore": x_example is standardized
      • U_SDR_plus will apply only the β scaling: x̂ = x_example / β
      • L = 2, entangler = "cnot", axes_low=(X,Z), axes_high=(Z,X).
    """
    x_example = np.asarray(x_example, dtype=np.float64).ravel()
    assert x_example.shape == (2,), f"Expected x_example shape (2,), got {x_example.shape}"

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit():
        U_SDR_plus(x_example, theta, L=L, entangler=entangler)
        return qml.state()

    fig, ax = qml.draw_mpl(circuit, decimals=3, expansion_strategy="device")()
    fig.set_size_inches(width, height)
    fig.set_dpi(dpi)
    plt.suptitle(
        r"U$_{\mathrm{SDR}+}$ (2D) – L=2, CNOT, $\gamma$-boost, $\beta$-scaling",
        fontsize=16,
    )
    if save:
        slug = "_".join(plot_name.strip().lower().split())
        out = Path(save_dir) / f"circuit_{slug or 'usdr_plus'}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"[PLOT] Saved circuit visualization -> {out}")
    plt.show()

# ===== CELL 038 (code) =====
N_example = 100
SEED_example = 0

data_example = load_processed_2d_dataset(
    base_path=OUTPUT_DIR,
    N=N_example,
    seed=SEED_example,
    normalize=NORMALIZE,
)

X_train = data_example["X_train"]   # shape (n_train, 2)
y_train = data_example["y_train"]   # shape (n_train,)

# ===== CELL 039 (code) =====
x_example = X_train[0]
theta_example = [2.1, 1.8, 2.3, 1.5]
visualize_U_SDR_plus_2D(
    x_example=x_example,
    theta=theta_example,
    width=24,
    height=8,
)

# ===== CELL 040 (markdown) =====
# # **Quantum Kernel Computation**

# ===== CELL 041 (markdown) =====
# **Feature map circuit**

# ===== CELL 042 (code) =====
memory = Memory("cache/hybrid", verbose=0)

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev, interface="numpy")
def usdr_plus_state(x, theta):
    """
    U_{SDR+} state preparation QNode.

    Assumptions (USDR+ protocol):
      • x is already *preprocessed*:
          - if NORMALIZE="minmax": x ∈ [0, 1]^2
          - if NORMALIZE="zscore": standardized
      • Circuit applies only β scaling: x̂ = x / β
      • L = 2, entangler = CNOT, axes_low=(X,Z), axes_high=(Z,X).
    """
    # Ensure proper shape
    x = np.asarray(x, dtype=np.float64).ravel()
    assert x.shape == (2,), f"Expected x shape (2,), got {x.shape}"

    # Delegate to the canonical USDR+ feature map
    U_SDR_plus(x, theta, L=2, entangler="cnot")

    return qml.state()

# ===== CELL 043 (markdown) =====
# **Fidelity 2D**

# ===== CELL 044 (code) =====
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

# ===== CELL 045 (markdown) =====
# **Quantum Kernel Matrix Construction**

# ===== CELL 046 (code) =====
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

# ===== CELL 047 (code) =====
# 1) Choose dataset and load preprocessed splits
N_example = 100
SEED_example = 0

data = load_processed_2d_dataset(
    base_path=OUTPUT_DIR,
    N=N_example,
    seed=SEED_example,
    normalize=NORMALIZE,
)

X_train = data["X_train"]
X_val   = data["X_val"]
X_test  = data["X_test"]

# 2) Choose θ (optimized or example)
theta = theta_example   # or theta_opt from your KRR tuning

# 3) Build Gram matrices
K_train = build_quantum_kernel_matrix(X_train, X_train, theta)
K_val   = build_quantum_kernel_matrix(X_val,   X_train, theta)  # rectangular
K_test  = build_quantum_kernel_matrix(X_test,  X_train, theta)  # rectangular

# ===== CELL 048 (markdown) =====
# **Condition numbers**

# ===== CELL 049 (code) =====
print(f"K_train shape: {K_train.shape}, "
      f"condition number: {np.linalg.cond(K_train):.2e}")

# For K_val and K_test, cond is SVD-based (rectangular),
# not the PSD "Gram" cond we care about for inversion:
print(f"K_val   shape: {K_val.shape},   cond (SVD): {np.linalg.cond(K_val):.2e}")
print(f"K_test  shape: {K_test.shape},  cond (SVD): {np.linalg.cond(K_test):.2e}")

# ===== CELL 050 (markdown) =====
# **True object used in KRR**

# ===== CELL 051 (code) =====
tau = 1e-3  # e.g. from θ_opt
K_reg = K_train + tau * np.eye(K_train.shape[0])
print(f"K_train + τI cond: {np.linalg.cond(K_reg):.2e}")

# ===== CELL 052 (markdown) =====
# **Analyze kernel matrix**

# ===== CELL 053 (code) =====
def analyze_kernel_matrix(
    K: np.ndarray,
    name: str = "K_train",
    width: float = 10,
    height: float = 5,
    dpi: int = 300,
    plot: bool = True,
    save: bool = True,
    save_dir: str | Path = "figures/hybrid/diagnostics",
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
                slug = "_".join(name.strip().lower().split())
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

# === Spectral diagnostics: effective rank, κ, eigen-range (USDR+ constrained) ===

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

# ===== CELL 054 (code) =====
analyze_kernel_matrix(K_train, name="K_train (N=100, seed=0)", width=20, height=10, dpi=300)
analyze_kernel_matrix(K_val,   name="K_val (rectangular)", plot=False)
analyze_kernel_matrix(K_test,  name="K_test (rectangular)", plot=False)

# ===== CELL 055 (code) =====
plot_gram_matrix(
    K_train,
    title="Gram matrix heatmap",
    cmap="viridis",
    dpi=300,
    width=20,
    height=16,
    annotate=False
)

# ===== CELL 056 (markdown) =====
# # **KRR training pipeline**

# ===== CELL 057 (markdown) =====
# **Hyperparameter Optimization (Joint θ + τ)**

# ===== CELL 058 (code) =====
# === 4.1 Kernel Matrix Construction (cached, USDR+ formalism) ===
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


# === 4.2 Hyperparameter Optimization (Joint θ + τ) ===
def krr_val_objective(log_params, X_train, y_train, X_val, y_val):
    """
    Validation MSE for KRR with U_{SDR+}.

    log_params = log([lambda1, lambda2, gamma, beta, tau])
    """
    theta = np.exp(log_params[:4])
    tau   = np.exp(log_params[4])

    # Build Gram and cross-kernel
    K_train = build_kernel_matrix(X_train, X_train, theta)        # (n_tr, n_tr)
    K_val   = build_kernel_matrix(X_val,   X_train, theta)        # (n_val, n_tr)

    # Regularized system
    K_reg = K_train + tau * np.eye(K_train.shape[0])

    try:
        alpha  = np.linalg.solve(K_reg, y_train)
        y_pred = K_val @ alpha
        mse    = mean_squared_error(y_val, y_pred)
        if not np.isfinite(mse):
            return 1e10
        return mse
    except np.linalg.LinAlgError:
        # Very ill-conditioned → heavy penalty
        return 1e10


def optimize_theta_tau(X_train, y_train, X_val, y_val):
    """
    Joint optimization of θ = (λ₁, λ₂, γ, β) and τ.

    Strategy
    --------
    • Optimize in log-space for numerical stability.
    • Primary optimizer: L-BFGS-B with box constraints.
    • Fallback: Optuna (TPE) with the *same* prior ranges.

    Interpretability priors (USDR+ protocol)
    ----------------------------------------
    • λ₁, λ₂ ∈ [0.1, 5.0]
    • γ       ∈ [1.5, 5.0]
    • β       ∈ [0.5, 3.0]
    • τ       ∈ [1e-8, 1e2]  (regularization strength)

    Returns
    -------
    theta_opt : np.ndarray, shape (4,)
        Optimal (λ₁, λ₂, γ, β).
    tau_opt   : float
        Optimal τ.
    val_mse   : float
        Validation MSE at (θ_opt, τ_opt).
    """

    # ----- 1. Hyperparameter ranges (linear space) -------------------------
    LAMBDA_MIN, LAMBDA_MAX = 0.1, 5.0       # for λ₁, λ₂
    GAMMA_MIN,  GAMMA_MAX  = 1.5, 5.0       # for γ
    BETA_MIN,   BETA_MAX   = 0.5, 3.0       # for β
    TAU_MIN,    TAU_MAX    = 1e-8, 1e2      # for τ

    # Convert to log-space bounds for L-BFGS-B
    lambda_bounds_log = (np.log(LAMBDA_MIN), np.log(LAMBDA_MAX))
    gamma_bounds_log  = (np.log(GAMMA_MIN),  np.log(GAMMA_MAX))
    beta_bounds_log   = (np.log(BETA_MIN),   np.log(BETA_MAX))
    tau_bounds_log    = (np.log(TAU_MIN),    np.log(TAU_MAX))

    # Order: [log λ₁, log λ₂, log γ, log β, log τ]
    bounds = [
        lambda_bounds_log,  # λ₁
        lambda_bounds_log,  # λ₂
        gamma_bounds_log,   # γ
        beta_bounds_log,    # β
        tau_bounds_log,     # τ
    ]

    # Sensible initial guess strictly inside the box (linear space)
    lambda_init = 1.0
    gamma_init  = 2.0
    beta_init   = 1.0
    tau_init    = 1e-3

    x0 = np.log([lambda_init, lambda_init, gamma_init, beta_init, tau_init])

    # ----- 2. L-BFGS-B over log-parameters --------------------------------
    print(
        "[OPT] Starting L-BFGS-B over log-params "
        f"(λ∈[{LAMBDA_MIN},{LAMBDA_MAX}], γ∈[{GAMMA_MIN},{GAMMA_MAX}], "
        f"β∈[{BETA_MIN},{BETA_MAX}], τ∈[{TAU_MIN:.0e},{TAU_MAX:.0e}])"
    )

    res = minimize(
        krr_val_objective,
        x0,
        args=(X_train, y_train, X_val, y_val),
        method="L-BFGS-B",
        bounds=bounds,
        tol=1e-6,
    )

    if res.success:
        theta_opt = np.exp(res.x[:4])   # (λ₁, λ₂, γ, β)
        tau_opt   = np.exp(res.x[4])    # τ
        val_mse   = krr_val_objective(res.x, X_train, y_train, X_val, y_val)

        print(
            "[OPT] L-BFGS-B succeeded (constrained)\n"
            f"      λ1={theta_opt[0]:.3f}, λ2={theta_opt[1]:.3f}, "
            f"γ={theta_opt[2]:.3f}, β={theta_opt[3]:.3f}, τ={tau_opt:.3e}\n"
            f"      Val MSE = {val_mse:.4e}"
        )
        return theta_opt, tau_opt, val_mse

    # ----- 3. Fallback: Optuna (TPE) with same priors ---------------------
    print("[OPT] L-BFGS-B failed → fallback to Optuna (TPE, 100 trials)")

    def objective(trial: optuna.Trial) -> float:
        # Sample directly in linear space with the SAME bounds as L-BFGS
        lambda1 = trial.suggest_float("lambda1", LAMBDA_MIN, LAMBDA_MAX, log=True)
        lambda2 = trial.suggest_float("lambda2", LAMBDA_MIN, LAMBDA_MAX, log=True)
        gamma   = trial.suggest_float("gamma",   GAMMA_MIN,  GAMMA_MAX,  log=True)
        beta    = trial.suggest_float("beta",    BETA_MIN,   BETA_MAX,   log=True)
        tau     = trial.suggest_float("tau",     TAU_MIN,    TAU_MAX,    log=True)

        log_params = np.log([lambda1, lambda2, gamma, beta, tau])
        return krr_val_objective(log_params, X_train, y_train, X_val, y_val)

    study = optuna.create_study(
        sampler=TPESampler(seed=42),
        direction="minimize",
    )
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    best = study.best_params
    theta_opt = np.array(
        [best["lambda1"], best["lambda2"], best["gamma"], best["beta"]],
        dtype=float,
    )
    tau_opt = float(best["tau"])
    val_mse = float(study.best_value)

    print(
        "[OPT] Optuna best (constrained):\n"
        f"      λ1={theta_opt[0]:.3f}, λ2={theta_opt[1]:.3f}, "
        f"γ={theta_opt[2]:.3f}, β={theta_opt[3]:.3f}, τ={tau_opt:.3e}\n"
        f"      Best Val MSE = {val_mse:.4e}"
    )

    return theta_opt, tau_opt, val_mse


# === 4.3 PSD Hygiene & Logging ===
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

# ===== CELL 059 (code) =====
def summarize_usdr_plus_constrained_results(
    csv_path: str | Path = "csv/usdr_plus/usdr_plus_final_results_constrained.csv",
    df: Optional[pd.DataFrame] = None,
) -> None:
    # --- 1) Load results ---
    if df is None:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"[SUMMARY-CONSTR] Could not find CSV at: {csv_path}"
            )
        df = pd.read_csv(csv_path)

    # Ensure we only look at the constrained experiment
    if "experiment" in df.columns:
        df = df[df["experiment"] == "usdr_plus_constrained"]

    if df.empty:
        print("[SUMMARY-CONSTR] No rows found for experiment='usdr_plus_constrained'.")
        return

    required_cols = {
        "N",
        "val_mse",
        "test_mse",
        "kappa_reg",
        "rank_eff_train",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            "[SUMMARY-CONSTR] Missing expected columns in results DataFrame: "
            + ", ".join(sorted(missing))
        )

    # --- 2) Metrics to summarize ---
    metrics = [
        "val_mse",
        "test_mse",
        "kappa_reg",
        "rank_eff_train",   # we treat raw K_train rank_eff as canonical
    ]

    # --- 3) Per-N aggregates (mean ± std over SEEDS) ---
    print("=== USDR+ (CONSTRAINED RANGES) – PER-N SUMMARY ===\n")
    for N, dfN in df.groupby("N"):
        print(f"--- N = {N} ---")
        for m in metrics:
            mu  = dfN[m].mean()
            std = dfN[m].std(ddof=1)  # sample std
            print(f"{m:15s}: {mu:.4e} ± {std:.4e}")
        print()

summarize_usdr_plus_constrained_results("csv/usdr_plus/usdr_plus_final_results_constrained.csv")

# ===== CELL 060 (code) =====
# === 4.x Hybrid evaluation loop: USDR+ (bounded) + classical RBF + hybrid K_H(ω) ===

# 0) One-off classical RBF grid search on a reference split
#    (pick a representative N and SEED; here we use the largest N and first SEED)

N_ref = max(sample_sizes)
SEED_ref = SEEDS[0]

print("\n[CLASSICAL-REF] Running one-off RBF grid search "
      f"on reference split N={N_ref}, SEED={SEED_ref}")

set_all_seeds(SEED_ref)
ref_data = load_processed_2d_dataset(
    base_path=OUTPUT_DIR,
    N=N_ref,
    seed=SEED_ref,
    normalize=NORMALIZE,
)

# Candidate grids (adjust if you want finer search)
length_scales_grid = np.logspace(-1, 1, 7)       # ℓ in [0.1, 10]
tau_classical_grid = np.logspace(-8, 2, 7)       # τ in [1e-8, 1e2]

classical_search_result = grid_search_classical_rbf_krr(
    X_train=ref_data["X_train"],
    y_train=ref_data["y_train"],
    X_val=ref_data["X_val"],
    y_val=ref_data["y_val"],
    length_scales=length_scales_grid,
    tau_grid=tau_classical_grid,
)

CLASSICAL_RBF_CONFIG = classical_search_result.best_config
CLASSICAL_BASELINE_TAU = classical_search_result.best_tau

print(
    "[CLASSICAL-REF] Using fixed classical config for all splits: "
    f"ℓ*={CLASSICAL_RBF_CONFIG.length_scale:.4f}, "
    f"τ*_classical={CLASSICAL_BASELINE_TAU:.3e}"
)

# 1) Normalization and hybrid configs
QUANTUM_NORM_CONFIG = GramNormalizationConfig(
    method="unit_diag",
    min_diag=1e-12,
    project_psd=False,
    psd_tolerance=1e-10,
    compute_eigenspectrum=False,
)

CLASSICAL_NORM_CONFIG = GramNormalizationConfig(
    method="unit_diag",
    min_diag=1e-12,
    project_psd=False,
    psd_tolerance=1e-10,
    compute_eigenspectrum=False,
)

HYBRID_KERNEL_CONFIG = HybridKernelConfig(
    omega_grid=[0.0, 0.25, 0.5, 0.75, 1.0],
    psd_check=True,
    psd_tolerance=1e-10,
    compute_eigenspectrum=False,
)

# 2) Containers for global results
global_results_df: Optional[pd.DataFrame] = None
per_run_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}

print("\n[HYBRID] Starting full hybrid evaluation loop over N and SEEDS...")
for N in sample_sizes:      # e.g. [50, 100, 200]
    for SEED in SEEDS:      # e.g. [0, 1, 2]
        print("\n====================================================")
        print(f"=== N={N}, SEED={SEED} (USDR+ bounded + hybrid) ===")
        print("====================================================")

        # 2.0) Deterministic seeding for reproducibility
        set_all_seeds(SEED)

        # 2.1) Load processed dataset splits
        data = load_processed_2d_dataset(
            base_path=OUTPUT_DIR,
            N=N,
            seed=SEED,
            normalize=NORMALIZE,
        )

        X_train = data["X_train"]
        y_train = data["y_train"]
        X_val   = data["X_val"]
        y_val   = data["y_val"]
        X_test  = data["X_test"]
        y_test  = data["y_test"]

        meta = {
            "N": N,
            "seed": SEED,
        }

        print(
            f"[DATA] n_train={data['metadata']['n_train']}, "
            f"n_val={data['metadata']['n_val']}, "
            f"n_test={data['metadata']['n_test']}"
        )

        # 2.2) Joint optimization of USDR+ θ and τ on (train, val)
        theta_opt, tau_opt, val_mse_usdr = optimize_theta_tau(
            X_train, y_train,
            X_val,   y_val,
        )

        print(
            "[USDR+] Bounded optimum for this split:\n"
            f"        λ1={theta_opt[0]:.3f}, λ2={theta_opt[1]:.3f}, "
            f"γ={theta_opt[2]:.3f}, β={theta_opt[3]:.3f}, τ*={tau_opt:.3e}\n"
            f"        Val MSE (USDR+ only) = {val_mse_usdr:.4e}"
        )

        # 2.3) Build USDR+ (quantum) Gram matrices (train/val/test)
        K_Q_train = build_kernel_matrix(X_train, X_train, theta_opt)
        K_Q_val   = build_kernel_matrix(X_val,   X_train, theta_opt)
        K_Q_test  = build_kernel_matrix(X_test,  X_train, theta_opt)

        print(
            f"[K_Q] K_Q_train shape={K_Q_train.shape}, "
            f"K_Q_val shape={K_Q_val.shape}, "
            f"K_Q_test shape={K_Q_test.shape}"
        )

        # Optional: analyze raw quantum kernel spectrum (pure USDR+ diagnosis)
        try:
            spec_train_usdr = compute_spectrum_metrics(
                K_Q_train,
                name=f"K_Q_train (N={N}, seed={SEED})",
                log_prefix="[SPEC-USDR]",
            )
        except NameError:
            # If compute_spectrum_metrics is not defined in this notebook,
            # you can safely ignore this block or plug in your own analyzer.
            spec_train_usdr = None

        # 2.4) Classical RBF Gram matrices and baseline performance
        classical_run, global_results_df = run_classical_baseline_for_split(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            config=CLASSICAL_RBF_CONFIG,
            tau=CLASSICAL_BASELINE_TAU,
            experiment_meta={
                "N": N,
                "seed": SEED,
            },
            results_df=global_results_df,
        )

        K_C_train = classical_run.K_train
        K_C_val   = classical_run.K_val
        K_C_test  = classical_run.K_test

        print(
            f"[K_C] K_C_train shape={K_C_train.shape}, "
            f"K_C_val shape={K_C_val.shape}, "
            f"K_C_test shape={K_C_test.shape}"
        )

        # 2.5) Normalize quantum and classical Gram triplets (unit diagonal)
        norm_Q = normalize_gram_triplet(
            K_train=K_Q_train,
            K_val=K_Q_val,
            K_test=K_Q_test,
            config=QUANTUM_NORM_CONFIG,
            label="quantum",
        )

        norm_C = normalize_gram_triplet(
            K_train=K_C_train,
            K_val=K_C_val,
            K_test=K_C_test,
            config=CLASSICAL_NORM_CONFIG,
            label="classical",
        )

        # 2.6) Build hybrid Gram matrices K_H(ω) for this split
        hybrid_mats = build_hybrid_gram_matrices(
            Ktilde_Q_train=norm_Q.K_train,
            Ktilde_Q_val=norm_Q.K_val,
            Ktilde_Q_test=norm_Q.K_test,
            Ktilde_C_train=norm_C.K_train,
            Ktilde_C_val=norm_C.K_val,
            Ktilde_C_test=norm_C.K_test,
            config=HYBRID_KERNEL_CONFIG,
            experiment_meta=meta,
        )

        # 2.7) Run KRR on all ω with fixed τ* from bounded USDR+ regime
        hybrid_krr_config = HybridKRRConfig(
            tau_star=float(tau_opt),
            compute_eigvals_if_missing=True,
            eff_rank_eps=1e-15,
        )

        omega_results, global_results_df = run_hybrid_krr_for_omegas(
            hybrid_mats=hybrid_mats,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            config=hybrid_krr_config,
            experiment_meta=meta,
            global_results_df=global_results_df,
        )

        # 2.8) Cache per-run objects for later visualization if needed
        per_run_cache[(N, SEED)] = {
            "theta_opt": theta_opt,
            "tau_opt": tau_opt,
            "val_mse_usdr": val_mse_usdr,
            "K_Q_train": K_Q_train,
            "K_Q_val": K_Q_val,
            "K_Q_test": K_Q_test,
            "K_C_train": K_C_train,
            "K_C_val": K_C_val,
            "K_C_test": K_C_test,
            "norm_Q": norm_Q,
            "norm_C": norm_C,
            "hybrid_mats": hybrid_mats,
            "omega_results": omega_results,
            "spec_train_usdr": spec_train_usdr,
        }

# 3) Save full hybrid results to CSV
global_results_df = global_results_df.reset_index(drop=True)
hybrid_results_csv = Path("csv/hybrid/usdr_plus_hybrid_results_constrained.csv")
hybrid_results_csv.parent.mkdir(parents=True, exist_ok=True)
global_results_df.to_csv(hybrid_results_csv, index=False)

print(f"\n[SUMMARY] Hybrid results saved → {hybrid_results_csv}")
display(global_results_df.head())

# ===== CELL 061 (code) =====
# === 5. Hybrid post-processing: ω* per (N, seed) and aggregation by N ===

# Safety checks
if global_results_df is None or global_results_df.empty:
    raise RuntimeError(
        "[HYBRID-POST] global_results_df is empty. "
        "Run the hybrid evaluation loop before calling this cell."
    )

# 5.1 Compute ω* per (N, seed) using validation MSE
omega_star_df, omega_star_map = compute_omega_star_per_run(
    global_results_df=global_results_df,
    hybrid_model_type="hybrid_usdr_plus",
)

print("\n[HYBRID-POST] Per-(N, seed) ω* summary:")
display(omega_star_df)

# 5.2 Aggregate ω* stats across seeds, grouped by N
agg_omega_star_df = aggregate_omega_star_by_N(
    omega_star_df=omega_star_df,
)

print("\n[HYBRID-POST] Aggregated ω* statistics per N:")
display(agg_omega_star_df)

# ===== CELL 062 (code) =====
# === 6. Hybrid summary table per N + global correlations ===

# 6.1 Build summary table per N:
#     - mean_test_MSE(ω=0) classical
#     - mean_test_MSE(ω=1) USDR⁺
#     - mean_test_MSE(ω*)   hybrid best
#     - mean_omega_star
#     - mean_effective_rank_star
#     - mean_kappa_reg_star
hybrid_summary_df = build_hybrid_summary_table_per_N(
    global_results_df=global_results_df,
    omega_star_df=omega_star_df,
    model_type="hybrid_usdr_plus",
)

print("\n[HYBRID-SUMMARY] Hybrid summary table per N:")
display(hybrid_summary_df)

# 6.2 Global correlations: test MSE vs effective rank / κ
hybrid_corrs = compute_hybrid_correlations(
    global_results_df=global_results_df,
    model_type="hybrid_usdr_plus",
)

print("\n[HYBRID-SUMMARY] Global correlations (across all N, seeds, ω):")
print(f"  corr(test MSE, effective_rank) = {hybrid_corrs.corr_mse_rank:.3f}")
print(f"  corr(test MSE, kappa_reg)     = {hybrid_corrs.corr_mse_kappa_reg:.3f}")
print(f"  corr(test MSE, kappa_raw)     = {hybrid_corrs.corr_mse_kappa_raw:.3f}")

# ===== CELL 063 (code) =====
# === 7. Global hybrid plots: geometry + MSE vs ω ===
plt.rcParams["font.family"] = "DejaVu Sans"
# Optionally restrict to a subset of N for plotting
N_values_to_plot = sorted(global_results_df["N"].unique())

print("\n[HYBRID-PLOTS] Plotting effective rank vs ω...")
plot_effective_rank_vs_omega(
    global_results_df=global_results_df,
    N_values=N_values_to_plot,
    model_type="hybrid_usdr_plus",
)

print("\n[HYBRID-PLOTS] Plotting condition number vs ω (regularized κ)...")
plot_condition_number_vs_omega(
    global_results_df=global_results_df,
    N_values=N_values_to_plot,
    model_type="hybrid_usdr_plus",
    use_regularized=True,
)

print("\n[HYBRID-PLOTS] Plotting val/test MSE vs ω...")
plot_mse_vs_omega(
    global_results_df=global_results_df,
    omega_star_df=omega_star_df,
    N_values=N_values_to_plot,
    model_type="hybrid_usdr_plus",
)

print("\n[HYBRID-PLOTS] Plotting summary bar charts per N "
      "(ω=0 classical vs ω=1 USDR⁺ vs ω* hybrid)...")
plot_summary_bars_per_N(
    global_results_df=global_results_df,
    omega_star_df=omega_star_df,
    model_type="hybrid_usdr_plus",
)

print("\n[HYBRID-PLOTS] Scatter: test MSE vs effective rank / condition, colored by ω...")
plot_mse_vs_rank_and_condition_scatter(
    global_results_df=global_results_df,
    model_type="hybrid_usdr_plus",
    use_regularized=True,
)

# ===== CELL 064 (code) =====
# === 8. Per-run test scatter & residuals: classical vs USDR⁺ vs hybrid (ω*) ===

# Choose a representative (N, SEED) for detailed prediction plots
N_vis = max(sample_sizes)   # e.g. largest N
SEED_vis = SEEDS[0]         # e.g. first seed

key = (N_vis, SEED_vis)
if key not in per_run_cache:
    raise KeyError(
        f"[HYBRID-PRED] per_run_cache does not contain key {key}. "
        "Ensure the hybrid evaluation loop has run for this (N, SEED)."
    )

cache = per_run_cache[key]

# Reload data to get y_true on test set
data_vis = load_processed_2d_dataset(
    base_path=OUTPUT_DIR,
    N=N_vis,
    seed=SEED_vis,
    normalize=NORMALIZE,
)

X_train = data_vis["X_train"]
y_train = data_vis["y_train"]
X_val   = data_vis["X_val"]
y_val   = data_vis["y_val"]
X_test  = data_vis["X_test"]
y_test  = data_vis["y_test"]

# --- 8.1 Classical predictions on test (ω=0) ---
K_C_train = cache["K_C_train"]
K_C_val   = cache["K_C_val"]
K_C_test  = cache["K_C_test"]

classical_metrics = krr_fit_and_evaluate(
    K_train=K_C_train,
    y_train=y_train,
    K_val=K_C_val,
    y_val=y_val,
    K_test=K_C_test,
    y_test=y_test,
    tau=float(CLASSICAL_BASELINE_TAU),
)
y_pred_test_classical = classical_metrics.y_pred_test

# --- 8.2 USDR⁺ predictions on test (ω=1, pure quantum) ---
K_Q_train = cache["K_Q_train"]
K_Q_val   = cache["K_Q_val"]
K_Q_test  = cache["K_Q_test"]
tau_usdr  = float(cache["tau_opt"])

usdr_metrics = krr_fit_and_evaluate(
    K_train=K_Q_train,
    y_train=y_train,
    K_val=K_Q_val,
    y_val=y_val,
    K_test=K_Q_test,
    y_test=y_test,
    tau=tau_usdr,
)
y_pred_test_usdr = usdr_metrics.y_pred_test

# --- 8.3 Hybrid predictions on test at ω* for this (N, SEED) ---
omega_star_entry = omega_star_map.get((N_vis, SEED_vis), None)
if omega_star_entry is None:
    raise KeyError(
        f"[HYBRID-PRED] No ω* entry found for (N={N_vis}, seed={SEED_vis}). "
        "Check omega_star_df / omega_star_map."
    )

omega_star = float(omega_star_entry.omega_star)
omega_results_for_run = cache["omega_results"]

if omega_star not in omega_results_for_run:
    # In case of tiny numerical mismatch, fall back to closest ω in grid
    candidate_omegas = np.array(list(omega_results_for_run.keys()), dtype=float)
    idx_closest = int(np.argmin(np.abs(candidate_omegas - omega_star)))
    omega_star_effective = float(candidate_omegas[idx_closest])
    print(
        f"[HYBRID-PRED] ω*={omega_star:.3f} not exactly in omega_results; "
        f"using closest grid ω={omega_star_effective:.3f}."
    )
    omega_star = omega_star_effective

hybrid_star_metrics = omega_results_for_run[omega_star].metrics
y_pred_test_hybrid = hybrid_star_metrics.y_pred_test

# --- 8.4 Plots: scatter + residual distributions on test set ---
print(
    f"\n[HYBRID-PRED] Plotting test-set scatter & residuals for "
    f"(N={N_vis}, SEED={SEED_vis}), ω*≈{omega_star:.3f}"
)

plot_true_vs_pred_scatter(
    y_true_test=y_test,
    y_pred_classical=y_pred_test_classical,
    y_pred_usdr=y_pred_test_usdr,
    y_pred_hybrid=y_pred_test_hybrid,
    title_prefix=f"True vs predicted (test) – N={N_vis}, SEED={SEED_vis}",
)

plot_residual_distributions(
    y_true_test=y_test,
    y_pred_classical=y_pred_test_classical,
    y_pred_usdr=y_pred_test_usdr,
    y_pred_hybrid=y_pred_test_hybrid,
    bins=20,
    title=f"Residual distributions (test) – N={N_vis}, SEED={SEED_vis}",
)

# ===== CELL 065 (code) =====
# ======================================================================
# Geometry-level diagnostics for one split (N_plot, SEED_plot)
# ======================================================================

N_plot = 200
SEED_plot = 0

case = per_run_cache[(N_plot, SEED_plot)]

norm_Q = case["norm_Q"]
norm_C = case["norm_C"]
hybrid_mats = case["hybrid_mats"]

print(f"[GEOM] Geometry-level plots for N={N_plot}, SEED={SEED_plot}")

# Heatmaps: K_C, K_Q, and a few K_H(ω)
plot_kernel_geometry_heatmaps(
    Ktilde_C_train=norm_C.K_train,
    Ktilde_Q_train=norm_Q.K_train,
    hybrid_mats=hybrid_mats,
    omegas_to_show=[0.25, 0.5, 0.75],
    width=12.0,   # per panel
    height=12.0,
    dpi=300,
)

# Eigenvalue spectra: classical vs USDR⁺ vs hybrid
plot_eigenvalue_spectra_for_case(
    Ktilde_C_train=norm_C.K_train,
    Ktilde_Q_train=norm_Q.K_train,
    hybrid_mats=hybrid_mats,
    omegas_to_show=[0.25, 0.5, 0.75, 1.0],
    title=f"Eigenvalue spectra (N={N_plot}, SEED={SEED_plot})",
    width=32.0,
    height=18.0,
    dpi=300,
)

# ===== CELL 066 (markdown) =====
# **Prediction surface**

# ===== CELL 067 (code) =====
from pathlib import Path

def plot_hybrid_prediction_surface_60x60(
    data_dict: dict,
    theta_opt,
    tau_star: float,
    classical_config,
    omega_hybrid: float,
    N: int,
    SEED: int,
    output_dir: str | Path | None = None,
    show: bool = True,
    dpi: int = 300,
    width: float = 30.0,
    height: float = 20.0,
):
    """
    60×60 prediction surfaces for:
        • Classical RBF kernel (baseline)
        • USDR⁺ kernel (bounded regime)
        • Hybrid kernel K_H(ω) = (1-ω) K_C + ω K_Q (with normalization)

    For each model (classical, USDR⁺, hybrid), show:
        – TRUE surface f(x₁, x₂)
        – predicted surface ŷ(x)
        – absolute error surface |ŷ(x) - f(x)|
    arranged in a 3 × 3 grid: (rows = models, columns = true / pred / error).
    """
    omega_hybrid = float(omega_hybrid)
    if not (0.0 <= omega_hybrid <= 1.0):
        raise ValueError(f"omega_hybrid must be in [0,1], got {omega_hybrid}")

    # --- 0) classical τ* (fallback: τ_star) ---
    try:
        tau_classical = float(CLASSICAL_BASELINE_TAU)
    except NameError:
        tau_classical = float(tau_star)
        print(
            "[GRID-HYB] WARNING: CLASSICAL_BASELINE_TAU not defined; "
            "using tau_star for classical RBF as well."
        )

    # --- 1) RAW 60×60 grid ---
    X1_raw, X2_raw, Y_true = generate_test_grid(
        grid_size=60,
        domain=RAW_DOMAIN,
    )
    X_grid_raw = np.column_stack([X1_raw.ravel(), X2_raw.ravel()])

    # --- 2) Preprocess like training ---
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_val   = data_dict["X_val"]
    y_val   = data_dict["y_val"]

    if NORMALIZE == "minmax":
        if "X_train_raw" in data_dict:
            X_train_raw = data_dict["X_train_raw"]
            scaler = MinMaxScaler().fit(X_train_raw)
            X_grid_proc = np.clip(scaler.transform(X_grid_raw), 0.0, 1.0)
            print("[GRID-HYB] Using MinMaxScaler fitted on X_train_raw.")
        else:
            X_grid_proc = X_grid_raw / (RAW_DOMAIN[1] - RAW_DOMAIN[0])
            print(
                "[GRID-HYB] WARNING: X_train_raw not provided; "
                "using domain-based MinMax x/(2π) for grid."
            )
    elif NORMALIZE == "zscore":
        if "X_train_raw" not in data_dict:
            raise ValueError(
                "[GRID-HYB] X_train_raw required for Z-score normalization."
            )
        X_train_raw = data_dict["X_train_raw"]
        scaler = StandardScaler().fit(X_train_raw)
        X_grid_proc = scaler.transform(X_grid_raw)
        print("[GRID-HYB] Using StandardScaler fitted on X_train_raw.")
    else:
        raise ValueError(f"[GRID-HYB] Unknown NORMALIZE mode: {NORMALIZE}")

    print(
        f"[GRID-HYB] X_train shape={X_train.shape}, "
        f"X_grid_proc shape={X_grid_proc.shape}"
    )

    # --- 3) Raw classical + USDR⁺ Gram matrices ---
    # USDR⁺
    K_Q_train = build_kernel_matrix(X_train, X_train, theta_opt)
    K_Q_val   = build_kernel_matrix(X_val,   X_train, theta_opt)
    K_Q_grid  = build_kernel_matrix(X_grid_proc, X_train, theta_opt)

    # Classical RBF
    K_C_train = rbf_kernel_matrix(X_train, X_train, classical_config)
    K_C_val   = rbf_kernel_matrix(X_val,   X_train, classical_config)
    K_C_grid  = rbf_kernel_matrix(X_grid_proc, X_train, classical_config)

    print(
        f"[GRID-HYB] K_Q_train shape={K_Q_train.shape}, "
        f"K_C_train shape={K_C_train.shape}, "
        f"K_grid (Q/C) shape={K_Q_grid.shape}/{K_C_grid.shape}"
    )

    # --- 4) Normalize (same configs as in hybrid loop) ---
    norm_Q = normalize_gram_triplet(
        K_train=K_Q_train,
        K_val=K_Q_val,
        K_test=K_Q_grid,
        config=QUANTUM_NORM_CONFIG,
        label="quantum-grid",
    )
    norm_C = normalize_gram_triplet(
        K_train=K_C_train,
        K_val=K_C_val,
        K_test=K_C_grid,
        config=CLASSICAL_NORM_CONFIG,
        label="classical-grid",
    )

    Ktilde_Q_train, Ktilde_Q_grid = norm_Q.K_train, norm_Q.K_test
    Ktilde_C_train, Ktilde_C_grid = norm_C.K_train, norm_C.K_test

    def _mix(KC, KQ, omega: float) -> np.ndarray:
        return (1.0 - omega) * KC + omega * KQ

    K_H_train = _mix(Ktilde_C_train, Ktilde_Q_train, omega_hybrid)
    K_H_grid  = _mix(Ktilde_C_grid,  Ktilde_Q_grid,  omega_hybrid)

    # --- 5) Helper: solve KRR and predict on grid ---
    def _predict_on_grid(K_train, K_grid, y_train, tau, tag: str):
        # PSD hygiene
        K_reg, jitter, K_stats, kappa_after = apply_psd_hygiene(
            K_train,
            tau,
            name=f"{tag} (N={N}, seed={SEED})",
        )
        # Solve and predict
        alpha = np.linalg.solve(K_reg, y_train)
        y_pred_flat = K_grid @ alpha            # (n_grid,)
        Y_pred = y_pred_flat.reshape(X1_raw.shape)

        error = np.abs(Y_pred - Y_true)
        grid_mse = mean_squared_error(Y_true.ravel(), Y_pred.ravel())

        print(
            f"[GRID-HYB] {tag}: "
            f"grid MSE={grid_mse:.6e}, "
            f"|err| mean={error.mean():.4e}, max={error.max():.4e}, "
            f"κ(K+τI)={kappa_after:.3e}, jitter={jitter:.2e}"
        )

        stats = {
            "grid_mse":       float(grid_mse),
            "mean_abs_error": float(error.mean()),
            "max_abs_error":  float(error.max()),
            "jitter":         float(jitter),
            "kappa_train":    float(K_stats["cond"]),
            "kappa_reg":      float(kappa_after),
            "tau":            float(tau),
        }
        return Y_pred, error, stats

    # --- 6) Classical / USDR⁺ / Hybrid predictions ---
    Y_pred_classical, err_classical, stats_classical = _predict_on_grid(
        K_C_train, K_C_grid, y_train, tau_classical, tag="[CLASSICAL-RBF]"
    )

    Y_pred_usdr, err_usdr, stats_usdr = _predict_on_grid(
        K_Q_train, K_Q_grid, y_train, tau_star, tag="[USDR+]"
    )

    Y_pred_hybrid, err_hybrid, stats_hybrid = _predict_on_grid(
        K_H_train, K_H_grid, y_train, tau_star,
        tag=f"[HYBRID ω={omega_hybrid:.2f}]",
    )

    # --- 7) 3×3 plot: TRUE + pred + error for each model ---
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(width, height),
        dpi=dpi,
        constrained_layout=True,
    )

    rows = [
        (Y_pred_classical, err_classical, "Classical RBF"),
        (Y_pred_usdr,      err_usdr,      r"USDR$^{+}$"),
        (Y_pred_hybrid,    err_hybrid,    r"Hybrid $K_{\mathrm{H}}(\omega)$"),
    ]

    for row_idx, (Y_pred, Y_err, label) in enumerate(rows):
        # Column 0: TRUE f(x1,x2)
        ax_true = axes[row_idx, 0]
        im_true = ax_true.contourf(
            X1_raw,
            X2_raw,
            Y_true,
            levels=60,
            cmap="viridis",
        )
        ax_true.set_title(f"{label} – True $f(x_1,x_2)$", fontsize=12)
        ax_true.set_xlabel(r"$x_1$")
        ax_true.set_ylabel(r"$x_2$")
        fig.colorbar(im_true, ax=ax_true, fraction=0.046, pad=0.04)

        # Column 1: Prediction
        ax_pred = axes[row_idx, 1]
        im_pred = ax_pred.contourf(
            X1_raw,
            X2_raw,
            Y_pred,
            levels=60,
            cmap="viridis",
        )
        ax_pred.set_title(f"{label} – Prediction", fontsize=12)
        ax_pred.set_xlabel(r"$x_1$")
        ax_pred.set_ylabel(r"$x_2$")
        fig.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)

        # Column 2: |Error|
        ax_err = axes[row_idx, 2]
        im_err = ax_err.contourf(
            X1_raw,
            X2_raw,
            Y_err,
            levels=60,
            cmap="Reds",
            vmin=0.0,
        )
        ax_err.set_title(f"{label} – |Error|", fontsize=12)
        ax_err.set_xlabel(r"$x_1$")
        ax_err.set_ylabel(r"$x_2$")
        fig.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"2D True / Prediction / Error Surfaces – N={N}, SEED={SEED}, "
        rf"$\omega = {omega_hybrid:.2f}$",
        fontsize=16,
        weight="bold",
    )

    diagnostics = {
        "classical": stats_classical,
        "usdr_plus": stats_usdr,
        "hybrid":    stats_hybrid,
        "omega_hybrid": float(omega_hybrid),
    }

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "prediction_surface_hybrid_3x3.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"[GRID-HYB] Saved hybrid 3×3 prediction surface → {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return diagnostics

# ===== CELL 068 (code) =====
N_plot, SEED_plot = 200, 0
data = load_processed_2d_dataset(
    base_path=OUTPUT_DIR,
    N=N_plot,
    seed=SEED_plot,
    normalize=NORMALIZE,
)

run_cache = per_run_cache[(N_plot, SEED_plot)]
theta_opt = run_cache["theta_opt"]
tau_opt   = run_cache["tau_opt"]          # τ* used in hybrid
omega_star = omega_star_map[(N_plot, SEED_plot)].omega_star

diag = plot_hybrid_prediction_surface_60x60(
    data_dict=data,
    theta_opt=theta_opt,
    tau_star=tau_opt,
    classical_config=CLASSICAL_RBF_CONFIG,
    omega_hybrid=omega_star,
    N=N_plot,
    SEED=SEED_plot,
    output_dir="figures/hybrid",
    show=True,
)

# ===== CELL 069 (markdown) =====
# **1D slices**

# ===== CELL 070 (code) =====
def plot_hybrid_1d_slices(
    data_dict: dict,
    theta_opt,
    tau_star: float,
    classical_config,
    omega_hybrid: float,
    N: int,
    SEED: int,
    output_dir: str | Path | None = None,
    show: bool = True,
    grid_size: int = 200,
    dpi: int = 300,
    width: float = 28.0,
    height: float = 12.0,
):
    """
    1D slices of the learned *hybrid* predictor:

      • Slice 1:  x1 = π,  x2 ∈ [0, 2π]
      • Slice 2:  x2 = π,  x1 ∈ [0, 2π]

    We compare:
      • ω = 0   → classical RBF-only (after normalization)
      • ω = 1   → pure USDR⁺ kernel
      • ω = ω*  → hybrid convex combination (typically with 0 < ω* < 1)

    All slices:
      • defined in RAW domain [0, 2π];
      • preprocessed exactly like training (NORMALIZE);
      • use the same τ* = tau_star as in the hybrid experiments.

    Returns a dict with MSE / error stats for each slice & kernel.
    """
    omega_hybrid = float(omega_hybrid)
    if not (0.0 <= omega_hybrid <= 1.0):
        raise ValueError(f"omega_hybrid must be in [0,1], got {omega_hybrid}")

    # ---------- 1. Raw 1D grids ----------
    x_grid_raw = np.linspace(RAW_DOMAIN[0], RAW_DOMAIN[1], grid_size)

    # Slice points in [0, 2π]²
    X_slice1_raw = np.stack([np.full(grid_size, np.pi), x_grid_raw], axis=1)
    X_slice2_raw = np.stack([x_grid_raw, np.full(grid_size, np.pi)], axis=1)

    # True function (noiseless)
    y_true1 = true_function(np.pi * np.ones_like(x_grid_raw), x_grid_raw)
    y_true2 = true_function(x_grid_raw, np.pi * np.ones_like(x_grid_raw))

    # ---------- 2. Preprocess slices like training ----------
    X_train = data_dict["X_train"]    # already preprocessed
    y_train = data_dict["y_train"]

    if NORMALIZE == "minmax":
        if "X_train_raw" in data_dict:
            X_train_raw = data_dict["X_train_raw"]
            scaler = MinMaxScaler().fit(X_train_raw)
            X_slice1 = np.clip(scaler.transform(X_slice1_raw), 0.0, 1.0)
            X_slice2 = np.clip(scaler.transform(X_slice2_raw), 0.0, 1.0)
            print("[SLICES-HYB] Using MinMaxScaler fitted on X_train_raw.")
        else:
            X_slice1 = X_slice1_raw / (RAW_DOMAIN[1] - RAW_DOMAIN[0])
            X_slice2 = X_slice2_raw / (RAW_DOMAIN[1] - RAW_DOMAIN[0])
            print(
                "[SLICES-HYB] WARNING: X_train_raw not provided; "
                "using domain-based MinMax x/(2π) for slices."
            )
    elif NORMALIZE == "zscore":
        if "X_train_raw" not in data_dict:
            raise ValueError(
                "[SLICES-HYB] X_train_raw required for Z-score normalization."
            )
        X_train_raw = data_dict["X_train_raw"]
        scaler = StandardScaler().fit(X_train_raw)
        X_slice1 = scaler.transform(X_slice1_raw)
        X_slice2 = scaler.transform(X_slice2_raw)
        print("[SLICES-HYB] Using StandardScaler fitted on X_train_raw.")
    else:
        raise ValueError(f"Unknown NORMALIZE mode: {NORMALIZE}")

    print(
        f"[SLICES-HYB] X_train shape={X_train.shape}, "
        f"X_slice1 shape={X_slice1.shape}, X_slice2 shape={X_slice2.shape}"
    )

    # ---------- 3. Raw quantum & classical Gram matrices ----------
    # USDR⁺ fidelity kernel
    K_Q_train = build_kernel_matrix(X_train, X_train, theta_opt)
    K_Q_s1    = build_kernel_matrix(X_slice1, X_train, theta_opt)
    K_Q_s2    = build_kernel_matrix(X_slice2, X_train, theta_opt)

    # Classical RBF kernel
    K_C_train = rbf_kernel_matrix(X_train, X_train, classical_config)
    K_C_s1    = rbf_kernel_matrix(X_slice1, X_train, classical_config)
    K_C_s2    = rbf_kernel_matrix(X_slice2, X_train, classical_config)

    print(
        f"[SLICES-HYB] K_Q_train {K_Q_train.shape}, K_Q_s1 {K_Q_s1.shape}, K_Q_s2 {K_Q_s2.shape}"
    )
    print(
        f"[SLICES-HYB] K_C_train {K_C_train.shape}, K_C_s1 {K_C_s1.shape}, K_C_s2 {K_C_s2.shape}"
    )

    # ---------- 4. Normalize (unit-diagonal) ----------
    norm_Q = normalize_gram_triplet(
        K_train=K_Q_train,
        K_val=K_Q_s1,   # dummy
        K_test=K_Q_s2,  # dummy
        config=QUANTUM_NORM_CONFIG,
        label="quantum-slices",
    )
    norm_C = normalize_gram_triplet(
        K_train=K_C_train,
        K_val=K_C_s1,
        K_test=K_C_s2,
        config=CLASSICAL_NORM_CONFIG,
        label="classical-slices",
    )

    Ktilde_Q_train, Ktilde_Q_s1, Ktilde_Q_s2 = norm_Q.K_train, norm_Q.K_val, norm_Q.K_test
    Ktilde_C_train, Ktilde_C_s1, Ktilde_C_s2 = norm_C.K_train, norm_C.K_val, norm_C.K_test

    # ---------- 5. Hybrid mixtures for ω=0,1,ω* ----------
    def _mix(KC, KQ, omega: float) -> np.ndarray:
        return (1.0 - omega) * KC + omega * KQ

    # Train kernels
    K_train_omega0 = _mix(Ktilde_C_train, Ktilde_Q_train, 0.0)
    K_train_omega1 = _mix(Ktilde_C_train, Ktilde_Q_train, 1.0)
    K_train_omegah = _mix(Ktilde_C_train, Ktilde_Q_train, omega_hybrid)

    # Slice-1 kernels (x1=π)
    K_s1_omega0 = _mix(Ktilde_C_s1, Ktilde_Q_s1, 0.0)
    K_s1_omega1 = _mix(Ktilde_C_s1, Ktilde_Q_s1, 1.0)
    K_s1_omegah = _mix(Ktilde_C_s1, Ktilde_Q_s1, omega_hybrid)

    # Slice-2 kernels (x2=π)
    K_s2_omega0 = _mix(Ktilde_C_s2, Ktilde_Q_s2, 0.0)
    K_s2_omega1 = _mix(Ktilde_C_s2, Ktilde_Q_s2, 1.0)
    K_s2_omegah = _mix(Ktilde_C_s2, Ktilde_Q_s2, omega_hybrid)

    # ---------- 6. PSD hygiene + solve for each ω ----------
    def _solve_and_predict(K_train, K_s1, K_s2, tag: str):
        K_reg, jitter, K_stats, kappa_after = apply_psd_hygiene(
            K_train,
            tau_star,
            name=f"{tag} (N={N}, seed={SEED}) [slices]",
        )
        alpha = np.linalg.solve(K_reg, y_train)
        y_hat1 = K_s1 @ alpha
        y_hat2 = K_s2 @ alpha
        return (y_hat1, y_hat2), {
            "jitter": float(jitter),
            "kappa_train": float(K_stats["cond"]),
            "kappa_reg": float(kappa_after),
        }

    (y1_omega0, y2_omega0), stats_omega0 = _solve_and_predict(
        K_train_omega0, K_s1_omega0, K_s2_omega0, tag="ω=0"
    )
    (y1_omega1, y2_omega1), stats_omega1 = _solve_and_predict(
        K_train_omega1, K_s1_omega1, K_s2_omega1, tag="ω=1"
    )
    (y1_omegah, y2_omegah), stats_omegah = _solve_and_predict(
        K_train_omegah, K_s1_omegah, K_s2_omegah, tag=f"ω={omega_hybrid:.2f}"
    )

    # ---------- 7. Diagnostics ----------
    def _slice_stats(y_true, y_hat):
        err = y_hat - y_true
        mse = mean_squared_error(y_true, y_hat)
        mean_abs_err = float(np.mean(np.abs(err)))
        max_abs_err = float(np.max(np.abs(err)))
        return mse, mean_abs_err, max_abs_err

    mse1_0, mae1_0, maxe1_0 = _slice_stats(y_true1, y1_omega0)
    mse2_0, mae2_0, maxe2_0 = _slice_stats(y_true2, y2_omega0)

    mse1_1, mae1_1, maxe1_1 = _slice_stats(y_true1, y1_omega1)
    mse2_1, mae2_1, maxe2_1 = _slice_stats(y_true2, y2_omega1)

    mse1_h, mae1_h, maxe1_h = _slice_stats(y_true1, y1_omegah)
    mse2_h, mae2_h, maxe2_h = _slice_stats(y_true2, y2_omegah)

    print(
        f"[SLICES-HYB] Slice1 (x1=π): "
        f"MSE(ω=0)={mse1_0:.4e}, MSE(ω=1)={mse1_1:.4e}, "
        f"MSE(ω={omega_hybrid:.2f})={mse1_h:.4e}"
    )
    print(
        f"[SLICES-HYB] Slice2 (x2=π): "
        f"MSE(ω=0)={mse2_0:.4e}, MSE(ω=1)={mse2_1:.4e}, "
        f"MSE(ω={omega_hybrid:.2f})={mse2_h:.4e}"
    )

    # ---------- 8. Plot ----------
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(width, height), dpi=dpi
    )

    # Slice 1: x1 = π, vary x2
    ax1.plot(x_grid_raw, y_true1, "k-", linewidth=2, label="True")
    ax1.plot(x_grid_raw, y1_omega0, "-",  linewidth=2, label="ω=0 (classical)")
    ax1.plot(x_grid_raw, y1_omega1, "--", linewidth=2, label="ω=1 (USDR⁺)")
    ax1.plot(
        x_grid_raw,
        y1_omegah,
        "-.",
        linewidth=2,
        label=rf"ω={omega_hybrid:.2f} (hybrid)",
    )
    ax1.set_title(
        rf"Slice: $x_1 = \pi$ (vary $x_2$) | $N={N}$, SEED={SEED}",
        fontsize=14,
    )
    ax1.set_xlabel(r"$x_2$")
    ax1.set_ylabel(r"$f(x)$")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Slice 2: x2 = π, vary x1
    ax2.plot(x_grid_raw, y_true2, "k-", linewidth=2, label="True")
    ax2.plot(x_grid_raw, y2_omega0, "-",  linewidth=2, label="ω=0 (classical)")
    ax2.plot(x_grid_raw, y2_omega1, "--", linewidth=2, label="ω=1 (USDR⁺)")
    ax2.plot(
        x_grid_raw,
        y2_omegah,
        "-.",
        linewidth=2,
        label=rf"ω={omega_hybrid:.2f} (hybrid)",
    )
    ax2.set_title(
        rf"Slice: $x_2 = \pi$ (vary $x_1$) | $N={N}$, SEED={SEED}",
        fontsize=14,
    )
    ax2.set_xlabel(r"$x_1$")
    ax2.set_ylabel(r"$f(x)$")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle(
        rf"1D Slices – Hybrid kernel (N={N}, SEED={SEED}, τ*={tau_star:.1e})",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # ---------- 9. Save + show ----------
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "slices_hybrid.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"[SLICES-HYB] Saved 1D slices figure → {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    # ---------- 10. Return diagnostics ----------
    return {
        "omega_hybrid": omega_hybrid,
        # Slice 1
        "mse_slice1_omega0": float(mse1_0),
        "mse_slice1_omega1": float(mse1_1),
        "mse_slice1_omegah": float(mse1_h),
        "mean_abs_err1_omega0": float(mae1_0),
        "mean_abs_err1_omega1": float(mae1_1),
        "mean_abs_err1_omegah": float(mae1_h),
        "max_abs_err1_omega0": float(maxe1_0),
        "max_abs_err1_omega1": float(maxe1_1),
        "max_abs_err1_omegah": float(maxe1_h),
        # Slice 2
        "mse_slice2_omega0": float(mse2_0),
        "mse_slice2_omega1": float(mse2_1),
        "mse_slice2_omegah": float(mse2_h),
        "mean_abs_err2_omega0": float(mae2_0),
        "mean_abs_err2_omega1": float(mae2_1),
        "mean_abs_err2_omegah": float(mae2_h),
        "max_abs_err2_omega0": float(maxe2_0),
        "max_abs_err2_omega1": float(maxe2_1),
        "max_abs_err2_omegah": float(maxe2_h),
        # PSD hygiene summaries
        "stats_omega0": stats_omega0,
        "stats_omega1": stats_omega1,
        "stats_omegah": stats_omegah,
    }


# -------------------------------------------------------------------
# Convenience block for thesis plot: pick a run with 0 < ω* < 1
# and call plot_hybrid_1d_slices on it
# -------------------------------------------------------------------

# 1) Find candidate runs with 0 < omega_star < 1
candidates = []
for (N, SEED), rec in omega_star_map.items():
    w = float(rec.omega_star)
    if 0.0 < w < 1.0:
        candidates.append((N, SEED, w))

if not candidates:
    raise RuntimeError("No runs found with 0 < omega_star < 1.0")

# 2) Pick the most illustrative one: smallest N, then smallest SEED
candidates.sort(key=lambda t: (t[0], t[1]))  # sort by N, then SEED
N_plot, SEED_plot, omega_star = candidates[0]

print(f"[PLOT-HYB-SLICES] Using run with 0 < ω* < 1:")
print(f"    N={N_plot}, SEED={SEED_plot}, omega_star={omega_star:.3f}")

# 3) Load data & pull cached θ*, τ*
data = load_processed_2d_dataset(
    base_path=OUTPUT_DIR,
    N=N_plot,
    seed=SEED_plot,
    normalize=NORMALIZE,
)

run_cache = per_run_cache[(N_plot, SEED_plot)]
theta_opt = run_cache["theta_opt"]
tau_opt   = run_cache["tau_opt"]  # τ* used in hybrid experiments

# 4) Plot hybrid 1D slices (classical vs USDR+ vs hybrid ω*)
diag_slices = plot_hybrid_1d_slices(
    data_dict=data,
    theta_opt=theta_opt,
    tau_star=tau_opt,
    classical_config=CLASSICAL_RBF_CONFIG,
    omega_hybrid=omega_star,
    N=N_plot,
    SEED=SEED_plot,
    output_dir="figures/hybrid",
    show=True,
)

diag_slices

# ===== CELL 071 (markdown) =====
# **Interaction ridge**

# ===== CELL 072 (code) =====
def plot_hybrid_interaction_ridge(
    data_dict: dict,
    theta_opt,
    tau_star: float,
    classical_config,
    omega_hybrid: float,
    N: int,
    SEED: int,
    output_dir: str | Path | None = None,
    show: bool = True,
    grid_size: int = 100,
    dpi: int = 300,
    width: float = 36.0,
    height: float = 10.0,
):
    """
    Interaction ridge visualization for the *hybrid* kernel.

    Compares how well the learned predictors (ω=0 classical, ω=1 USDR⁺,
    ω=ω* hybrid) recover the interaction term 0.1 x1 x2 after removing
    the smooth part (sin x1 + cos x2).

    Protocol:
      • RAW grid in [0, 2π]^2.
      • Grid preprocessed with SAME scheme as training (NORMALIZE).
      • Quantum part: USDR⁺ fidelity kernel via build_kernel_matrix.
      • Classical part: RBF kernel via rbf_kernel_matrix(classical_config).
      • Both kernels are normalized (unit diagonal) and mixed as
            K_H(ω) = (1-ω) K_C + ω K_Q
        using the same QUANTUM_NORM_CONFIG / CLASSICAL_NORM_CONFIG.
      • KRR uses a shared τ* = tau_star for all ω.

    Produces:
      • 1×3 contour plots of residuals for ω=0, ω=1, ω=ω*.
      • Diagnostics (MSE, correlation with 0.1 x1 x2, etc.) for each ω.
    """
    omega_hybrid = float(omega_hybrid)
    if not (0.0 <= omega_hybrid <= 1.0):
        raise ValueError(f"omega_hybrid must be in [0,1], got {omega_hybrid}")

    # ---------- 1. RAW grid over [0, 2π]^2 ----------
    x_grid_raw = np.linspace(RAW_DOMAIN[0], RAW_DOMAIN[1], grid_size)
    X1_raw, X2_raw = np.meshgrid(x_grid_raw, x_grid_raw)
    X_full_raw = np.column_stack([X1_raw.ravel(), X2_raw.ravel()])

    # True components
    smooth_true = np.sin(X1_raw) + np.cos(X2_raw)
    interaction_true = 0.1 * X1_raw * X2_raw  # 0.1 x1 x2

    # ---------- 2. Preprocess grid like training ----------
    X_train = data_dict["X_train"]    # preprocessed (MinMax/Z-score)
    y_train = data_dict["y_train"]

    if NORMALIZE == "minmax":
        if "X_train_raw" in data_dict:
            X_train_raw = data_dict["X_train_raw"]
            scaler = MinMaxScaler().fit(X_train_raw)
            X_full_proc = np.clip(scaler.transform(X_full_raw), 0.0, 1.0)
            print("[RIDGE-HYB] Using MinMaxScaler fitted on X_train_raw.")
        else:
            X_full_proc = X_full_raw / (RAW_DOMAIN[1] - RAW_DOMAIN[0])
            print(
                "[RIDGE-HYB] WARNING: X_train_raw not provided; "
                "using domain-based MinMax x/(2π) for grid."
            )

    elif NORMALIZE == "zscore":
        if "X_train_raw" not in data_dict:
            raise ValueError(
                "[RIDGE-HYB] X_train_raw required for Z-score normalization."
            )
        X_train_raw = data_dict["X_train_raw"]
        scaler = StandardScaler().fit(X_train_raw)
        X_full_proc = scaler.transform(X_full_raw)
        print("[RIDGE-HYB] Using StandardScaler fitted on X_train_raw.")
    else:
        raise ValueError(f"Unknown NORMALIZE mode: {NORMALIZE}")

    print(
        f"[RIDGE-HYB] X_train shape={X_train.shape}, "
        f"X_full_proc shape={X_full_proc.shape}"
    )

    # ---------- 3. Raw quantum & classical Gram matrices ----------
    # USDR⁺ block
    K_Q_train = build_kernel_matrix(X_train,     X_train, theta_opt)
    K_Q_full  = build_kernel_matrix(X_full_proc, X_train, theta_opt)

    # Classical RBF block
    K_C_train = rbf_kernel_matrix(X_train,     X_train, classical_config)
    K_C_full  = rbf_kernel_matrix(X_full_proc, X_train, classical_config)

    # ---------- 4. Normalize (unit-diagonal) ----------
    norm_Q = normalize_gram_triplet(
        K_train=K_Q_train,
        K_val=K_Q_full,
        K_test=K_Q_full,  # dummy
        config=QUANTUM_NORM_CONFIG,
        label="quantum-ridge",
    )
    norm_C = normalize_gram_triplet(
        K_train=K_C_train,
        K_val=K_C_full,
        K_test=K_C_full,  # dummy
        config=CLASSICAL_NORM_CONFIG,
        label="classical-ridge",
    )

    Ktilde_Q_train, Ktilde_Q_full = norm_Q.K_train, norm_Q.K_val
    Ktilde_C_train, Ktilde_C_full = norm_C.K_train, norm_C.K_val

    # ---------- 5. Hybrid mixtures for ω=0,1,ω* ----------
    def _mix(KC, KQ, omega: float) -> np.ndarray:
        return (1.0 - omega) * KC + omega * KQ

    # Train kernels
    K_train_omega0 = _mix(Ktilde_C_train, Ktilde_Q_train, 0.0)
    K_train_omega1 = _mix(Ktilde_C_train, Ktilde_Q_train, 1.0)
    K_train_omegah = _mix(Ktilde_C_train, Ktilde_Q_train, omega_hybrid)

    # Full-grid kernels
    K_full_omega0 = _mix(Ktilde_C_full, Ktilde_Q_full, 0.0)
    K_full_omega1 = _mix(Ktilde_C_full, Ktilde_Q_full, 1.0)
    K_full_omegah = _mix(Ktilde_C_full, Ktilde_Q_full, omega_hybrid)

    # ---------- 6. PSD hygiene + solve KRR for each ω ----------
    def _solve_and_predict(K_train, K_full, tag: str):
        K_reg, jitter, K_stats, kappa_after = apply_psd_hygiene(
            K_train,
            tau_star,
            name=f"{tag} (N={N}, seed={SEED}) [ridge]",
        )
        alpha = np.linalg.solve(K_reg, y_train)
        y_pred_flat = K_full @ alpha
        Y_pred = y_pred_flat.reshape(X1_raw.shape)
        return Y_pred, {
            "jitter": float(jitter),
            "kappa_train": float(K_stats["cond"]),
            "kappa_reg": float(kappa_after),
        }

    Y_pred_omega0, stats_omega0 = _solve_and_predict(
        K_train_omega0, K_full_omega0, tag="ω=0"
    )
    Y_pred_omega1, stats_omega1 = _solve_and_predict(
        K_train_omega1, K_full_omega1, tag="ω=1"
    )
    Y_pred_omegah, stats_omegah = _solve_and_predict(
        K_train_omegah, K_full_omegah, tag=f"ω={omega_hybrid:.2f}"
    )

    # ---------- 7. Residuals ≈ interaction ridge ----------
    residual_omega0 = Y_pred_omega0 - smooth_true
    residual_omega1 = Y_pred_omega1 - smooth_true
    residual_omegah = Y_pred_omegah - smooth_true

    def _ridge_stats(residual):
        corr = np.corrcoef(residual.ravel(), interaction_true.ravel())[0, 1]
        mse_ridge = mean_squared_error(
            interaction_true.ravel(), residual.ravel()
        )
        mean_abs_res = float(np.mean(np.abs(residual)))
        max_abs_res  = float(np.max(np.abs(residual)))
        return mse_ridge, corr, mean_abs_res, max_abs_res

    mse0, corr0, mean0, max0 = _ridge_stats(residual_omega0)
    mse1, corr1, mean1, max1 = _ridge_stats(residual_omega1)
    mseh, corrh, meanh, maxh = _ridge_stats(residual_omegah)

    print(
        f"[RIDGE-HYB] ω=0:  MSE={mse0:.4e}, corr={corr0:.4f}, mean|res|={mean0:.4e}"
    )
    print(
        f"[RIDGE-HYB] ω=1:  MSE={mse1:.4e}, corr={corr1:.4f}, mean|res|={mean1:.4e}"
    )
    print(
        f"[RIDGE-HYB] ω={omega_hybrid:.2f}: "
        f"MSE={mseh:.4e}, corr={corrh:.4f}, mean|res|={meanh:.4e}"
    )

    # ---------- 8. Plot ----------
    fig, axes = plt.subplots(
        1, 3, figsize=(width, height), dpi=dpi, constrained_layout=True
    )

    residuals = [residual_omega0, residual_omega1, residual_omegah]
    titles = [
        "Residual (ω=0, classical)",
        "Residual (ω=1, USDR⁺)",
        rf"Residual (ω={omega_hybrid:.2f}, hybrid)",
    ]

    # common symmetric color scale
    all_res = np.concatenate([r.ravel() for r in residuals])
    r_abs = float(np.max(np.abs(all_res)))
    vmin, vmax = -r_abs, r_abs

    for ax, R, title_str in zip(axes, residuals, titles):
        cf = ax.contourf(
            X1_raw,
            X2_raw,
            R,
            levels=50,
            cmap="RdBu",
            vmin=vmin,
            vmax=vmax,
        )
        # overlay true interaction contours
        cs = ax.contour(
            X1_raw,
            X2_raw,
            interaction_true,
            levels=[0.5, 1.0, 1.5, 2.0],
            colors="black",
            alpha=0.6,
        )
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

        ax.set_title(title_str, fontsize=12)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")

    cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
    cbar.set_label(
        r"Residual $f_{\theta}(x) - (\sin x_1 + \cos x_2)$",
        fontsize=12,
    )

    fig.suptitle(
        rf"Interaction Ridge Recovery ($0.1 x_1 x_2$) | N={N}, SEED={SEED}",
        fontsize=16,
    )

    # ---------- 9. Save + show/close ----------
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "interaction_ridge_hybrid.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"[RIDGE-HYB] Saved interaction ridge figure → {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    # ---------- 10. Return diagnostics ----------
    return {
        "omega_hybrid": float(omega_hybrid),

        "mse_ridge_omega0": float(mse0),
        "corr_ridge_omega0": float(corr0),
        "mean_abs_residual_omega0": float(mean0),
        "max_abs_residual_omega0": float(max0),

        "mse_ridge_omega1": float(mse1),
        "corr_ridge_omega1": float(corr1),
        "mean_abs_residual_omega1": float(mean1),
        "max_abs_residual_omega1": float(max1),

        "mse_ridge_omegah": float(mseh),
        "corr_ridge_omegah": float(corrh),
        "mean_abs_residual_omegah": float(meanh),
        "max_abs_residual_omegah": float(maxh),

        "stats_omega0": stats_omega0,
        "stats_omega1": stats_omega1,
        "stats_omegah": stats_omegah,
    }

# ===== CELL 073 (code) =====
ridge_diag = plot_hybrid_interaction_ridge(
    data_dict=data,
    theta_opt=theta_opt,
    tau_star=tau_opt,
    classical_config=CLASSICAL_RBF_CONFIG,
    omega_hybrid=omega_star,
    N=N_plot,
    SEED=SEED_plot,
    output_dir="figures/hybrid",
)

# ===== CELL 074 (markdown) =====
# **True vs predicted**

# ===== CELL 075 (code) =====
def plot_hybrid_true_vs_predicted(
    data_dict: dict,
    theta_opt,
    tau_star: float,
    classical_config,
    omega_hybrid: float,
    N: int,
    SEED: int,
    output_dir: str | Path | None = None,
    show: bool = True,
    dpi: int = 300,
    width: float = 28.0,
    height: float = 10.0,
):
    """
    True vs Predicted scatter on the TEST set for the *hybrid* kernel.

    Compares three predictors on the same split:
      • ω = 0   → pure classical RBF (normalized)
      • ω = 1   → pure USDR⁺ (normalized)
      • ω = ω*  → hybrid K_H(ω) = (1-ω) K_C + ω K_Q

    Protocol:
      • Uses preprocessed X_train/X_test (MinMax or Z-score).
      • Quantum part via build_kernel_matrix (USDR⁺ fidelity kernel).
      • Classical part via rbf_kernel_matrix(classical_config).
      • Both kernels are normalized via QUANTUM_NORM_CONFIG / CLASSICAL_NORM_CONFIG.
      • Same τ* = tau_star for all ω, with PSD hygiene via apply_psd_hygiene.

    Extended behaviour:
      • Saves the figure as 'true_vs_pred_hybrid.png' inside output_dir (if provided).
      • Optionally shows the figure in the notebook (show=True).
      • Returns a dict with test MSEs and numerical diagnostics for each ω.
    """
    omega_hybrid = float(omega_hybrid)
    if not (0.0 <= omega_hybrid <= 1.0):
        raise ValueError(f"omega_hybrid must be in [0,1], got {omega_hybrid}")

    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_test  = data_dict["X_test"]
    y_test  = data_dict["y_test"]

    print(f"[TVP-HYB] N={N}, SEED={SEED}")
    print(
        f"[TVP-HYB] X_train shape={X_train.shape}, "
        f"X_test shape={X_test.shape}"
    )

    # --- 1. Raw quantum & classical Gram matrices (train/test) ------------
    # USDR⁺ block
    K_Q_train = build_kernel_matrix(X_train, X_train, theta_opt)
    K_Q_test  = build_kernel_matrix(X_test,  X_train, theta_opt)

    # Classical RBF block
    K_C_train = rbf_kernel_matrix(X_train, X_train, classical_config)
    K_C_test  = rbf_kernel_matrix(X_test,  X_train, classical_config)

    print(
        f"[TVP-HYB] K_Q_train shape={K_Q_train.shape}, "
        f"K_Q_test shape={K_Q_test.shape}"
    )
    print(
        f"[TVP-HYB] K_C_train shape={K_C_train.shape}, "
        f"K_C_test shape={K_C_test.shape}"
    )

    # --- 2. Normalize both blocks (unit diagonal) -------------------------
    norm_Q = normalize_gram_triplet(
        K_train=K_Q_train,
        K_val=K_Q_test,
        K_test=K_Q_test,   # dummy reuse
        config=QUANTUM_NORM_CONFIG,
        label="quantum-tvp",
    )
    norm_C = normalize_gram_triplet(
        K_train=K_C_train,
        K_val=K_C_test,
        K_test=K_C_test,   # dummy reuse
        config=CLASSICAL_NORM_CONFIG,
        label="classical-tvp",
    )

    Ktilde_Q_train, Ktilde_Q_test = norm_Q.K_train, norm_Q.K_val
    Ktilde_C_train, Ktilde_C_test = norm_C.K_train, norm_C.K_val

    # --- 3. Hybrid mixtures for ω=0, 1, ω* -------------------------------
    def _mix(KC, KQ, omega: float) -> np.ndarray:
        return (1.0 - omega) * KC + omega * KQ

    K_train_omega0 = _mix(Ktilde_C_train, Ktilde_Q_train, 0.0)
    K_train_omega1 = _mix(Ktilde_C_train, Ktilde_Q_train, 1.0)
    K_train_omegah = _mix(Ktilde_C_train, Ktilde_Q_train, omega_hybrid)

    K_test_omega0 = _mix(Ktilde_C_test, Ktilde_Q_test, 0.0)
    K_test_omega1 = _mix(Ktilde_C_test, Ktilde_Q_test, 1.0)
    K_test_omegah = _mix(Ktilde_C_test, Ktilde_Q_test, omega_hybrid)

    # --- 4. PSD hygiene + solve KRR + predict for each ω ------------------
    def _solve_and_predict(K_train, K_test, tag: str):
        K_reg, jitter, K_stats, kappa_after = apply_psd_hygiene(
            K_train,
            tau_star,
            name=f"{tag} (N={N}, seed={SEED}) [TVP-HYB]",
        )
        alpha = np.linalg.solve(K_reg, y_train)
        y_pred = K_test @ alpha
        test_mse = mean_squared_error(y_test, y_pred)
        print(
            f"[TVP-HYB] {tag}: Test MSE={test_mse:.4e}, "
            f"κ(K+τI)={kappa_after:.3e}, jitter={jitter:.2e}"
        )
        return y_pred, test_mse, {
            "jitter": float(jitter),
            "kappa_train": float(K_stats["cond"]),
            "kappa_reg": float(kappa_after),
        }

    y_pred_omega0, mse_omega0, stats_omega0 = _solve_and_predict(
        K_train_omega0, K_test_omega0, tag="ω=0 (classical)"
    )
    y_pred_omega1, mse_omega1, stats_omega1 = _solve_and_predict(
        K_train_omega1, K_test_omega1, tag="ω=1 (USDR⁺)"
    )
    y_pred_omegah, mse_omegah, stats_omegah = _solve_and_predict(
        K_train_omegah, K_test_omegah, tag=f"ω={omega_hybrid:.2f} (hybrid)"
    )

    # --- 5. Scatter plot true vs predicted for all three ------------------
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(width, height),
        dpi=dpi,
        constrained_layout=True,
    )

    configs = [
        (y_pred_omega0, f"Classical (ω=0)\nMSE={mse_omega0:.4e}"),
        (y_pred_omega1, f"USDR⁺ (ω=1)\nMSE={mse_omega1:.4e}"),
        (y_pred_omegah, f"Hybrid (ω={omega_hybrid:.2f})\nMSE={mse_omegah:.4e}"),
    ]

    min_val = float(min(y_test.min(), y_pred_omega0.min(),
                        y_pred_omega1.min(), y_pred_omegah.min()))
    max_val = float(max(y_test.max(), y_pred_omega0.max(),
                        y_pred_omega1.max(), y_pred_omegah.max()))

    for ax, (y_pred, title_str) in zip(axes, configs):
        ax.scatter(
            y_test,
            y_pred,
            alpha=0.7,
            edgecolor="k",
            linewidth=0.6,
            s=60,
        )
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            lw=2,
        )
        ax.set_xlabel("True y", fontsize=11, weight="bold")
        ax.set_ylabel("Predicted y", fontsize=11, weight="bold")
        ax.set_title(title_str, fontsize=12, weight="bold")
        ax.grid(alpha=0.3, linestyle="--")

    fig.suptitle(
        f"True vs Predicted (Test) – Hybrid Comparison | N={N}, SEED={SEED}",
        fontsize=14,
        weight="bold",
    )

    # --- 6. Save + show/close ---------------------------------------------
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "true_vs_pred_hybrid.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"[TVP-HYB] Saved true-vs-pred hybrid figure → {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    # --- 7. Return diagnostics --------------------------------------------
    return {
        "omega_hybrid": float(omega_hybrid),

        "test_mse_omega0": float(mse_omega0),
        "test_mse_omega1": float(mse_omega1),
        "test_mse_omegah": float(mse_omegah),

        "stats_omega0": stats_omega0,
        "stats_omega1": stats_omega1,
        "stats_omegah": stats_omegah,
    }

# ===== CELL 076 (code) =====
# -------------------------------------------------------------
# Pick an illustrative run with 0 < omega_star < 1 for TVP plot
# -------------------------------------------------------------

# 1) Find candidate runs with 0 < omega_star < 1
candidates = []
for (N, SEED), rec in omega_star_map.items():
    w = float(rec.omega_star)
    if 0.0 < w < 1.0:
        candidates.append((N, SEED, w))

if not candidates:
    raise RuntimeError(
        "[TVP-HYB] No runs found with 0 < omega_star < 1.0; "
        "cannot produce a genuinely hybrid TVP plot."
    )

# 2) Pick the most illustrative one: smallest N, then smallest SEED
candidates.sort(key=lambda t: (t[0], t[1]))  # sort by N, then SEED
N_plot, SEED_plot, omega_star = candidates[0]

print("[TVP-HYB] Using run with 0 < ω* < 1 for true-vs-pred plot:")
print(f"    N={N_plot}, SEED={SEED_plot}, omega_star={omega_star:.3f}")

# 3) Load data & pull cached θ*, τ* for that run
data = load_processed_2d_dataset(
    base_path=OUTPUT_DIR,
    N=N_plot,
    seed=SEED_plot,
    normalize=NORMALIZE,
)

run_cache = per_run_cache[(N_plot, SEED_plot)]
theta_opt = run_cache["theta_opt"]
tau_opt   = run_cache["tau_opt"]   # τ* used in hybrid experiments

# 4) Plot hybrid true vs predicted (classical vs USDR+ vs hybrid ω*)
tvp_diag = plot_hybrid_true_vs_predicted(
    data_dict=data,
    theta_opt=theta_opt,
    tau_star=tau_opt,
    classical_config=CLASSICAL_RBF_CONFIG,
    omega_hybrid=omega_star,   # 0 < ω* < 1
    N=N_plot,
    SEED=SEED_plot,
    output_dir="figures/hybrid",
    show=True,
)

tvp_diag

# ===== CELL 077 (markdown) =====
# **Error plots**

# ===== CELL 078 (code) =====
def plot_hybrid_residual_distributions(
    data_dict: dict,
    theta_opt,
    tau_star: float,
    classical_config,
    omega_hybrid: float,
    N: int,
    SEED: int,
    output_dir: str | Path | None = None,
    show: bool = True,
    dpi: int = 300,
    width: float = 10.0,
    height: float = 6.0,
):
    """
    Residual distributions (train / val / test) for a *single hybrid model*
    on a given split (N, SEED).

    Hybrid KRR model:
      • Classical block: RBF with `classical_config`
      • Quantum block: USDR⁺ via `build_kernel_matrix(X, X, theta_opt)`
      • Normalization: QUANTUM_NORM_CONFIG / CLASSICAL_NORM_CONFIG
      • Hybrid kernel: K_H(ω) = (1-ω) K_C + ω K_Q
      • Regularization: τ* = tau_star from bounded USDR⁺ protocol

    For each split, residuals are:
        r_split = y_split - y_hat_split(ω)

    and visualized as a violin + jittered scatter.

    Parameters
    ----------
    data_dict : dict
        Output of `load_processed_2d_dataset` for this (N, SEED).
        Must contain X_train, y_train, X_val, y_val, X_test, y_test.
    theta_opt : array-like, shape (4,)
        Optimal USDR⁺ parameters (lambda1, lambda2, gamma, beta).
    tau_star : float
        Regularization parameter τ* (same used in hybrid KRR loop).
    classical_config :
        Best classical RBF config (e.g. CLASSICAL_RBF_CONFIG).
    omega_hybrid : float
        Hybrid mixing coefficient ω ∈ [0,1] (e.g. ω* for this run).
    N, SEED : int
        Run identifiers.
    output_dir : str | Path | None
        If provided, save PNG to output_dir / "residuals_violin_hybrid.png".
    show : bool
        Whether to display the figure.
    dpi, width, height : plotting parameters.
    """
    omega_hybrid = float(omega_hybrid)
    if not (0.0 <= omega_hybrid <= 1.0):
        raise ValueError(f"omega_hybrid must be in [0,1], got {omega_hybrid}")

    # ---------- 1. Unpack splits ----------
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_val   = data_dict["X_val"]
    y_val   = data_dict["y_val"]
    X_test  = data_dict["X_test"]
    y_test  = data_dict["y_test"]

    print(f"[RESID-HYB] N={N}, SEED={SEED}, ω={omega_hybrid:.3f}")
    print(
        f"[RESID-HYB] shapes – "
        f"train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}"
    )

    # ---------- 2. Build raw quantum & classical Gram matrices ----------
    # USDR⁺ block
    K_Q_train = build_kernel_matrix(X_train, X_train, theta_opt)
    K_Q_val   = build_kernel_matrix(X_val,   X_train, theta_opt)
    K_Q_test  = build_kernel_matrix(X_test,  X_train, theta_opt)

    # Classical RBF block
    K_C_train = rbf_kernel_matrix(X_train, X_train, classical_config)
    K_C_val   = rbf_kernel_matrix(X_val,   X_train, classical_config)
    K_C_test  = rbf_kernel_matrix(X_test,  X_train, classical_config)

    print(
        f"[RESID-HYB] K_Q_train shape={K_Q_train.shape}, "
        f"K_C_train shape={K_C_train.shape}"
    )

    # ---------- 3. Normalize (unit-diagonal) both blocks ----------
    norm_Q = normalize_gram_triplet(
        K_train=K_Q_train,
        K_val=K_Q_val,
        K_test=K_Q_test,
        config=QUANTUM_NORM_CONFIG,
        label="quantum-resid",
    )
    norm_C = normalize_gram_triplet(
        K_train=K_C_train,
        K_val=K_C_val,
        K_test=K_C_test,
        config=CLASSICAL_NORM_CONFIG,
        label="classical-resid",
    )

    Ktilde_Q_train, Ktilde_Q_val, Ktilde_Q_test = norm_Q.K_train, norm_Q.K_val, norm_Q.K_test
    Ktilde_C_train, Ktilde_C_val, Ktilde_C_test = norm_C.K_train, norm_C.K_val, norm_C.K_test

    # ---------- 4. Build hybrid kernels at ω_hybrid ----------
    def _mix(KC, KQ, omega: float) -> np.ndarray:
        return (1.0 - omega) * KC + omega * KQ

    K_H_train = _mix(Ktilde_C_train, Ktilde_Q_train, omega_hybrid)
    K_H_val   = _mix(Ktilde_C_val,   Ktilde_Q_val,   omega_hybrid)
    K_H_test  = _mix(Ktilde_C_test,  Ktilde_Q_test,  omega_hybrid)

    # ---------- 5. PSD hygiene + solve KRR on hybrid K_H_train ----------
    K_reg, jitter, K_stats, kappa_after = apply_psd_hygiene(
        K_H_train,
        tau_star,
        name=f"K_H_train(ω={omega_hybrid:.2f}) (N={N}, seed={SEED}) [resid-hyb]",
    )

    # Solve (K_reg) alpha = y_train
    alpha = np.linalg.solve(K_reg, y_train)

    # ---------- 6. Predictions and residuals on each split ----------
    y_hat_train = K_H_train @ alpha
    r_train = y_train - y_hat_train

    y_hat_val = K_H_val @ alpha
    r_val = y_val - y_hat_val

    y_hat_test = K_H_test @ alpha
    r_test = y_test - y_hat_test

    # ---------- 7. Basic residual diagnostics ----------
    def summarize_residuals(name: str, y_true, y_hat, r):
        mse = mean_squared_error(y_true, y_hat)
        mae = float(np.mean(np.abs(r)))
        print(
            f"[RESID-HYB] {name}: "
            f"MSE={mse:.4e}, MAE={mae:.4e}, "
            f"mean(r)={np.mean(r):.4e}, std(r)={np.std(r):.4e}"
        )
        return mse, mae

    train_mse, train_mae = summarize_residuals("train", y_train, y_hat_train, r_train)
    val_mse,   val_mae   = summarize_residuals("val",   y_val,   y_hat_val,   r_val)
    test_mse,  test_mae  = summarize_residuals("test",  y_test,  y_hat_test,  r_test)

    # ---------- 8. Violin + jittered scatter plot ----------
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

    data = [r_train, r_val, r_test]
    labels = ["Train", "Val", "Test"]
    positions = np.arange(1, 4)

    parts = ax.violinplot(
        dataset=data,
        positions=positions,
        showmeans=True,
        showmedians=False,
        showextrema=False,
        widths=0.8,
    )

    for pc in parts["bodies"]:
        pc.set_alpha(0.5)

    if "cmeans" in parts:
        parts["cmeans"].set_linewidth(1.5)

    rng = np.random.default_rng(42)  # fixed for reproducibility
    for i, residuals in enumerate(data):
        x_jitter = rng.normal(loc=positions[i], scale=0.03, size=residuals.shape[0])
        ax.scatter(
            x_jitter,
            residuals,
            s=30,
            alpha=0.7,
            edgecolor="k",
            linewidth=0.5,
        )

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(r"Residual $r = y - \hat{y}$", fontsize=12)
    ax.set_title(
        f"Hybrid Residual Distributions (Train / Val / Test)\n"
        f"N={N}, SEED={SEED}, ω={omega_hybrid:.2f}",
        fontsize=14,
        weight="bold",
    )

    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()

    # ---------- 9. Save + show ----------
    stats = {
        "N": N,
        "SEED": SEED,
        "omega_hybrid": float(omega_hybrid),

        "train_mse": float(train_mse),
        "train_mae": float(train_mae),
        "val_mse":   float(val_mse),
        "val_mae":   float(val_mae),
        "test_mse":  float(test_mse),
        "test_mae":  float(test_mae),

        "jitter":       float(jitter),
        "kappa_reg":    float(kappa_after),
        "kappa_train":  float(K_stats["cond"]),
    }

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / "residuals_violin_hybrid.png"
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[RESID-HYB] Saved hybrid residuals violin plot → {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return stats

# ===== CELL 079 (code) =====
run_cache = per_run_cache[(N_plot, SEED_plot)]
data      = load_processed_2d_dataset(
    base_path=OUTPUT_DIR,
    N=N_plot,
    seed=SEED_plot,
    normalize=NORMALIZE,
)

theta_opt  = run_cache["theta_opt"]
tau_opt    = run_cache["tau_opt"]        # τ*
omega_star = omega_star_map[(N_plot, SEED_plot)].omega_star

resid_stats = plot_hybrid_residual_distributions(
    data_dict=data,
    theta_opt=theta_opt,
    tau_star=tau_opt,
    classical_config=CLASSICAL_RBF_CONFIG,
    omega_hybrid=omega_star,
    N=N_plot,
    SEED=SEED_plot,
    output_dir="figures/hybrid",
    show=True,
)

# ===== CELL 080 (code) =====
pass
