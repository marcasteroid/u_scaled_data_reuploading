"""Hybrid kernel construction: convex mix of classical RBF and USDR+ quantum kernels."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from joblib import Memory
from sklearn.metrics import mean_squared_error

from usdr_plus.hybrid import config as cfg
from usdr_plus.quantum.kernel import build_kernel_matrix as build_quantum_kernel_matrix
from usdr_plus.utils.cache import _safe_psd_hygiene

cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)
_memory = Memory(location=cfg.CACHE_DIR, verbose=0, mmap_mode="r")


@dataclass(frozen=True)
class RBFKernelConfig:
    length_scale: float


@dataclass(frozen=True)
class ClassicalGridSearchResult:
    best_config: RBFKernelConfig
    best_tau: float
    best_val_mse: float


def _median_sigma(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n < 2:
        return 1.0
    d2 = []
    for i in range(min(n, 200)):
        for j in range(i + 1, min(n, 200)):
            d2.append(float(np.sum((X[i] - X[j]) ** 2)))
    med = np.median(d2) if d2 else 1.0
    return float(max(np.sqrt(max(med, 1e-12)), 1e-6))


def build_classical_rbf_kernel(X1: np.ndarray, X2: np.ndarray, sigma: float | None = None) -> np.ndarray:
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    if sigma is None:
        sigma = _median_sigma(X1)
    gamma = 1.0 / (2.0 * sigma * sigma)
    sq = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
    return np.exp(-gamma * sq)


def grid_search_classical_rbf_krr(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    length_scales: list[float],
    tau_grid: list[float],
) -> ClassicalGridSearchResult:
    best_cfg: RBFKernelConfig | None = None
    best_tau = 0.0
    best_mse = float("inf")

    for ell in length_scales:
        cfg_ell = RBFKernelConfig(length_scale=float(ell))
        K_train = build_classical_rbf_kernel(X_train, X_train, sigma=cfg_ell.length_scale)
        K_val = build_classical_rbf_kernel(X_val, X_train, sigma=cfg_ell.length_scale)
        for tau in tau_grid:
            tau = float(tau)
            alpha = np.linalg.solve(K_train + tau * np.eye(K_train.shape[0]), y_train)
            y_val_pred = K_val @ alpha
            mse = float(mean_squared_error(y_val, y_val_pred))
            if mse < best_mse:
                best_mse = mse
                best_cfg = cfg_ell
                best_tau = tau

    if best_cfg is None:
        raise RuntimeError("Classical RBF grid search failed to find a valid configuration.")
    return ClassicalGridSearchResult(
        best_config=best_cfg,
        best_tau=best_tau,
        best_val_mse=best_mse,
    )


def normalize_gram(K: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=float)
    d = np.sqrt(np.clip(np.diag(K), 1e-12, None))
    return K / np.outer(d, d)


def normalize_train_test_gram(K_train: np.ndarray, K_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    K_train = np.asarray(K_train, dtype=float)
    K_test = np.asarray(K_test, dtype=float)
    d = np.sqrt(np.clip(np.diag(K_train), 1e-12, None))
    return K_train / np.outer(d, d), K_test / d[None, :]


@_memory.cache
def build_hybrid_train_test_kernels(
    X_train: np.ndarray,
    X_test: np.ndarray,
    theta: np.ndarray,
    omega: float,
    classical_length_scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (K_train_hybrid, K_test_hybrid) for a given omega in [0,1]."""
    omega = float(omega)
    if not (0.0 <= omega <= 1.0):
        raise ValueError(f"omega must be in [0,1], got {omega}")
    Kq_train = build_quantum_kernel_matrix(X_train, X_train, theta)
    Kq_test = build_quantum_kernel_matrix(X_test, X_train, theta)

    sigma = classical_length_scale
    if sigma is None:
        sigma = _median_sigma(np.asarray(X_train, dtype=float))
    Kc_train = build_classical_rbf_kernel(X_train, X_train, sigma=sigma)
    Kc_test = build_classical_rbf_kernel(X_test, X_train, sigma=sigma)

    Kq_train, Kq_test = normalize_train_test_gram(Kq_train, Kq_test)
    Kc_train, Kc_test = normalize_train_test_gram(Kc_train, Kc_test)

    Kh_train = omega * Kq_train + (1.0 - omega) * Kc_train
    Kh_test = omega * Kq_test + (1.0 - omega) * Kc_test
    Kh_train, _ = _safe_psd_hygiene(Kh_train, eps_floor=1e-10, report=False)
    return Kh_train, Kh_test


__all__ = [
    "ClassicalGridSearchResult",
    "RBFKernelConfig",
    "build_classical_rbf_kernel",
    "build_hybrid_train_test_kernels",
    "grid_search_classical_rbf_krr",
    "normalize_train_test_gram",
    "normalize_gram",
]
