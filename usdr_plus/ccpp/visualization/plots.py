"""CCPP plotting functions aligned with the notebook Plots section."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from usdr_plus.ccpp import config as cfg
from usdr_plus.ccpp.quantum.kernel import build_kernel_matrix
from usdr_plus.quantum.circuit import visualize_U_SDR_plus_2D
from usdr_plus.visualization.dataset_plots import plot_gram_matrix
from usdr_plus.analysis.spectrum import analyze_kernel_matrix

matplotlib.use("Agg")
plt.show = lambda *args, **kwargs: None


def _get_input_scaler(data: dict) -> Any:
    direct_keys = ("x_scaler", "scaler_X", "scaler")
    for k in direct_keys:
        if k in data and data[k] is not None:
            return data[k]
    meta = data.get("metadata", {})
    if isinstance(meta, dict):
        for k in direct_keys:
            if k in meta and meta[k] is not None:
                return meta[k]
    return None


def _regularize_kernel_ccpp(
    K_train: np.ndarray,
    tau: float,
    *,
    name: str,
    jitter_factor: float = 1e-10,
    cond_threshold: float = 1e12,
) -> tuple[np.ndarray, float]:
    n = K_train.shape[0]
    K_reg = K_train + tau * np.eye(n)
    try:
        kappa = float(np.linalg.cond(K_reg))
    except np.linalg.LinAlgError:
        kappa = np.inf
    print(f"[PSD-CCPP] {name}: tau={tau:.3e}, cond(K+tauI)={kappa:.3e}")
    jitter = 0.0
    if np.isfinite(kappa) and kappa > cond_threshold:
        jitter = jitter_factor * float(np.trace(K_train)) / float(max(n, 1))
        K_reg = K_reg + jitter * np.eye(n)
        try:
            k_after = float(np.linalg.cond(K_reg))
        except np.linalg.LinAlgError:
            k_after = np.inf
        print(f"[PSD-CCPP] Added jitter eps={jitter:.2e}, cond_after={k_after:.3e}")
    return K_reg, float(jitter)


def save_diagnostics(
    K_train: np.ndarray,
    *,
    x_example: np.ndarray,
    theta: np.ndarray,
    run_tag: str,
    diagnostics_dir: Path = cfg.DIAGNOSTICS_DIR,
) -> None:
    """Save spectrum, Gram heatmap, and circuit visualization in diagnostics dir."""
    spectrum_width = 10
    spectrum_height = 5
    gram_width = 8
    gram_height = 6
    circuit_width = 12
    circuit_height = 6
    dpi = 300

    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    analyze_kernel_matrix(
        K_train,
        name=f"K_train_{run_tag}",
        width=spectrum_width,
        height=spectrum_height,
        dpi=dpi,
        plot=True,
        save=True,
        save_dir=diagnostics_dir,
    )
    plot_gram_matrix(
        K_train,
        title=f"Gram Matrix {run_tag}",
        cmap="viridis",
        width=gram_width,
        height=gram_height,
        dpi=dpi,
        annotate=False,
        save=True,
        save_dir=diagnostics_dir,
    )
    visualize_U_SDR_plus_2D(
        x_example=x_example,
        theta=theta,
        width=circuit_width,
        height=circuit_height,
        dpi=dpi,
        save=True,
        save_dir=diagnostics_dir,
        plot_name=run_tag,
    )


def plot_prediction_surface_ccpp(
    data: Dict[str, Any],
    theta: np.ndarray,
    tau: float,
    *,
    run_dir: Path,
    N: int,
    seed: int,
    grid_size: int = 200,
    width: float = 24.0,
    height: float = 10.0,
    dpi: int = 300,
    show: bool = False,
    raw_domain_at: Optional[Tuple[float, float]] = None,
    raw_domain_v: Optional[Tuple[float, float]] = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    X_train = np.asarray(data["X_train"], dtype=float)
    y_train = np.asarray(data["y_train"], dtype=float).ravel()
    X_val = np.asarray(data.get("X_val", []), dtype=float)
    y_val = np.asarray(data.get("y_val", []), dtype=float).ravel() if np.asarray(data.get("y_val", [])).size else np.array([])
    X_test = np.asarray(data.get("X_test", []), dtype=float)
    y_test = np.asarray(data.get("y_test", []), dtype=float).ravel() if np.asarray(data.get("y_test", [])).size else np.array([])

    X_all = np.vstack([x for x in (X_train, X_val, X_test) if x.size > 0])
    y_all = np.concatenate([y for y in (y_train, y_val, y_test) if y.size > 0])

    scaler_X = _get_input_scaler(data)
    if scaler_X is not None:
        X_train_raw = scaler_X.inverse_transform(X_train)
        X_all_raw = scaler_X.inverse_transform(X_all)
    else:
        X_train_raw = X_train
        X_all_raw = X_all

    if raw_domain_at is None:
        raw_domain_at = (float(X_train_raw[:, 0].min()), float(X_train_raw[:, 0].max()))
    if raw_domain_v is None:
        raw_domain_v = (float(X_train_raw[:, 1].min()), float(X_train_raw[:, 1].max()))

    at_grid = np.linspace(raw_domain_at[0], raw_domain_at[1], grid_size)
    v_grid = np.linspace(raw_domain_v[0], raw_domain_v[1], grid_size)
    AT_grid, V_grid = np.meshgrid(at_grid, v_grid)
    grid_raw = np.column_stack([AT_grid.ravel(), V_grid.ravel()])
    X_grid = scaler_X.transform(grid_raw) if scaler_X is not None else grid_raw

    K_train = build_kernel_matrix(X_train, X_train, theta, apply_psd_hygiene_for_square=False)
    K_reg, _ = _regularize_kernel_ccpp(K_train, tau, name=f"K_train (CCPP surface N={N}, seed={seed})")
    alpha = np.linalg.solve(K_reg, y_train)
    K_grid = build_kernel_matrix(X_grid, X_train, theta, apply_psd_hygiene_for_square=False)
    y_grid_pred = (K_grid @ alpha).reshape(AT_grid.shape)

    fig, axes = plt.subplots(1, 2, figsize=(width, height), dpi=dpi)
    ax0, ax1 = axes
    sc = ax0.scatter(X_all_raw[:, 0], X_all_raw[:, 1], c=y_all, s=20, alpha=0.8)
    ax0.set_xlabel("AT")
    ax0.set_ylabel("V")
    ax0.set_title(f"CCPP Data (N={N}, seed={seed})")
    fig.colorbar(sc, ax=ax0).set_label("EP (observed)")

    cs = ax1.contourf(AT_grid, V_grid, y_grid_pred, levels=30)
    ax1.set_xlabel("AT")
    ax1.set_ylabel("V")
    ax1.set_title("USDR+ KRR Prediction Surface")
    fig.colorbar(cs, ax=ax1).set_label("EP (predicted)")
    fig.suptitle(f"USDR+ CCPP 2D - N={N}, seed={seed}")
    out_path = run_dir / "prediction_surface.png"
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"[PLOT-CCPP] Saved prediction surface -> {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_1d_slices_ccpp(
    data: Dict[str, Any],
    theta: np.ndarray,
    tau: float,
    *,
    run_dir: Path,
    N: int,
    seed: int,
    grid_size: int = 200,
    width: float = 24.0,
    height: float = 10.0,
    dpi: int = 300,
    show: bool = False,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    X_train = np.asarray(data["X_train"], dtype=float)
    y_train = np.asarray(data["y_train"], dtype=float).ravel()
    X_val = np.asarray(data.get("X_val", []), dtype=float)
    y_val = np.asarray(data.get("y_val", []), dtype=float).ravel() if np.asarray(data.get("y_val", [])).size else np.array([])
    X_test = np.asarray(data.get("X_test", []), dtype=float)
    y_test = np.asarray(data.get("y_test", []), dtype=float).ravel() if np.asarray(data.get("y_test", [])).size else np.array([])
    X_all = np.vstack([x for x in (X_train, X_val, X_test) if x.size > 0])
    y_all = np.concatenate([y for y in (y_train, y_val, y_test) if y.size > 0])

    scaler_X = _get_input_scaler(data)
    X_train_raw = scaler_X.inverse_transform(X_train) if scaler_X is not None else X_train.copy()
    X_all_raw = scaler_X.inverse_transform(X_all) if scaler_X is not None else X_all.copy()

    AT_train, V_train = X_train_raw[:, 0], X_train_raw[:, 1]
    AT_fixed = float(np.median(AT_train))
    V_fixed = float(np.median(V_train))
    at_min, at_max = float(AT_train.min()), float(AT_train.max())
    v_min, v_max = float(V_train.min()), float(V_train.max())

    at_grid = np.linspace(at_min, at_max, grid_size)
    v_grid = np.linspace(v_min, v_max, grid_size)
    s1_raw = np.column_stack([at_grid, np.full_like(at_grid, V_fixed)])
    s2_raw = np.column_stack([np.full_like(v_grid, AT_fixed), v_grid])
    s1 = scaler_X.transform(s1_raw) if scaler_X is not None else s1_raw
    s2 = scaler_X.transform(s2_raw) if scaler_X is not None else s2_raw

    K_train = build_kernel_matrix(X_train, X_train, theta, apply_psd_hygiene_for_square=True)
    K_reg, _ = _regularize_kernel_ccpp(K_train, tau, name=f"K_train (CCPP slices N={N}, seed={seed})")
    alpha = np.linalg.solve(K_reg, y_train)
    y_pred_s1 = build_kernel_matrix(s1, X_train, theta, apply_psd_hygiene_for_square=False) @ alpha
    y_pred_s2 = build_kernel_matrix(s2, X_train, theta, apply_psd_hygiene_for_square=False) @ alpha

    band_v = 0.05 * (v_max - v_min) if v_max > v_min else 0.0
    band_at = 0.05 * (at_max - at_min) if at_max > at_min else 0.0
    m1 = np.abs(X_all_raw[:, 1] - V_fixed) <= band_v if band_v > 0 else np.ones(len(X_all_raw), dtype=bool)
    m2 = np.abs(X_all_raw[:, 0] - AT_fixed) <= band_at if band_at > 0 else np.ones(len(X_all_raw), dtype=bool)

    fig, axes = plt.subplots(1, 2, figsize=(width, height), dpi=dpi, constrained_layout=True)
    axes[0].scatter(X_all_raw[m1, 0], y_all[m1], alpha=0.8, s=30, label="Data (V band)")
    axes[0].plot(at_grid, y_pred_s1, "--", lw=2.0, label="USDR+ prediction")
    axes[0].set_xlabel("AT")
    axes[0].set_ylabel("EP")
    axes[0].set_title(f"AT-slice at V~{V_fixed:.2f}")
    axes[0].grid(True, linestyle=":")
    axes[0].legend(loc="best")

    axes[1].scatter(X_all_raw[m2, 1], y_all[m2], alpha=0.8, s=30, label="Data (AT band)")
    axes[1].plot(v_grid, y_pred_s2, "--", lw=2.0, label="USDR+ prediction")
    axes[1].set_xlabel("V")
    axes[1].set_ylabel("EP")
    axes[1].set_title(f"V-slice at AT~{AT_fixed:.2f}")
    axes[1].grid(True, linestyle=":")
    axes[1].legend(loc="best")

    fig.suptitle(f"USDR+ 1D Slices on CCPP 2D (N={N}, seed={seed})", fontsize=14)
    out_path = run_dir / "slices.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"[SLICES-CCPP] Saved 1D slice plot -> {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_true_vs_predicted_ccpp(
    data: Dict[str, Any],
    theta: np.ndarray,
    tau: float,
    *,
    run_dir: Path,
    N: int,
    seed: int,
    width: float = 24.0,
    height: float = 10.0,
    dpi: int = 300,
    show: bool = False,
) -> float:
    run_dir.mkdir(parents=True, exist_ok=True)
    X_train = np.asarray(data["X_train"], dtype=float)
    y_train = np.asarray(data["y_train"], dtype=float).ravel()
    X_test = np.asarray(data["X_test"], dtype=float)
    y_test = np.asarray(data["y_test"], dtype=float).ravel()

    K_train = build_kernel_matrix(X_train, X_train, theta, apply_psd_hygiene_for_square=True)
    K_reg, jitter = _regularize_kernel_ccpp(K_train, tau, name=f"K_train (TVP N={N}, seed={seed})")
    alpha = np.linalg.solve(K_reg, y_train)
    K_test = build_kernel_matrix(X_test, X_train, theta, apply_psd_hygiene_for_square=False)
    y_pred = K_test @ alpha
    test_mse = float(mean_squared_error(y_test, y_pred))

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    ax.scatter(y_test, y_pred, alpha=0.8, s=40, label="Test points")
    y_min = float(min(y_test.min(), y_pred.min()))
    y_max = float(max(y_test.max(), y_pred.max()))
    pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    y_min -= pad
    y_max += pad
    ax.plot([y_min, y_max], [y_min, y_max], "--", lw=2.0, label="y=x")
    ax.set_xlabel("True EP")
    ax.set_ylabel("Predicted EP")
    ax.set_xlim(y_min, y_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"CCPP 2D - True vs Predicted (N={N}, seed={seed})")
    ax.grid(True, linestyle=":")
    ax.legend(loc="best")
    ax.text(
        0.05, 0.95,
        f"Test MSE={test_mse:.3f}\nJitter={jitter:.2e}",
        transform=ax.transAxes, va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.15, edgecolor="none"),
    )
    out_path = run_dir / "true_vs_pred.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"[TVP-CCPP] Saved true-vs-predicted plot -> {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return test_mse


def plot_residual_distributions_ccpp(
    data: Dict[str, Any],
    theta: np.ndarray,
    tau: float,
    *,
    run_dir: Path,
    N: int,
    seed: int,
    width: float = 20.0,
    height: float = 12.0,
    dpi: int = 300,
    show: bool = False,
) -> Dict[str, Dict[str, float]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    X_train = np.asarray(data["X_train"], dtype=float)
    y_train = np.asarray(data["y_train"], dtype=float).ravel()
    X_val = np.asarray(data["X_val"], dtype=float)
    y_val = np.asarray(data["y_val"], dtype=float).ravel()
    X_test = np.asarray(data["X_test"], dtype=float)
    y_test = np.asarray(data["y_test"], dtype=float).ravel()

    K_train = build_kernel_matrix(X_train, X_train, theta, apply_psd_hygiene_for_square=True)
    K_reg, jitter = _regularize_kernel_ccpp(K_train, tau, name=f"K_train (RES N={N}, seed={seed})")
    alpha = np.linalg.solve(K_reg, y_train)
    K_val = build_kernel_matrix(X_val, X_train, theta, apply_psd_hygiene_for_square=False)
    K_test = build_kernel_matrix(X_test, X_train, theta, apply_psd_hygiene_for_square=False)
    y_train_pred = K_train @ alpha
    y_val_pred = K_val @ alpha
    y_test_pred = K_test @ alpha

    r_train = y_train - y_train_pred
    r_val = y_val - y_val_pred
    r_test = y_test - y_test_pred

    stats = {
        "train": {
            "mse": float(mean_squared_error(y_train, y_train_pred)),
            "mae": float(mean_absolute_error(y_train, y_train_pred)),
            "mean": float(np.mean(r_train)),
            "std": float(np.std(r_train, ddof=1)) if r_train.size > 1 else 0.0,
        },
        "val": {
            "mse": float(mean_squared_error(y_val, y_val_pred)),
            "mae": float(mean_absolute_error(y_val, y_val_pred)),
            "mean": float(np.mean(r_val)),
            "std": float(np.std(r_val, ddof=1)) if r_val.size > 1 else 0.0,
        },
        "test": {
            "mse": float(mean_squared_error(y_test, y_test_pred)),
            "mae": float(mean_absolute_error(y_test, y_test_pred)),
            "mean": float(np.mean(r_test)),
            "std": float(np.std(r_test, ddof=1)) if r_test.size > 1 else 0.0,
        },
    }

    residual_groups = [r_train, r_val, r_test]
    labels = ["Train", "Val", "Test"]
    positions = np.arange(1, 4)
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    parts = ax.violinplot(residual_groups, positions=positions, showmeans=True, showextrema=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.7)
    rng = np.random.default_rng(seed)
    for i, res in enumerate(residual_groups):
        x = positions[i] + 0.08 * rng.normal(size=res.size)
        ax.scatter(x, res, s=15, alpha=0.7, linewidths=0.5, edgecolors="k")
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Split")
    ax.set_ylabel("Residual (y_true - y_pred)")
    ax.set_title(f"CCPP 2D - Residual distributions (N={N}, seed={seed})")
    ax.grid(True, axis="y", linestyle=":")
    t = stats["test"]
    ax.text(
        0.98, 0.98,
        f"Test stats:\nMSE={t['mse']:.3f}\nMAE={t['mae']:.3f}\nmean={t['mean']:.3f}\nstd={t['std']:.3f}",
        transform=ax.transAxes, va="top", ha="right", fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.15, edgecolor="none"),
    )
    out_path = run_dir / "residuals_violin.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"[RES-CCPP] Saved residual distribution plot -> {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return stats


def plot_interaction_ridge_ccpp(
    data: Dict[str, Any],
    theta: np.ndarray,
    tau: float,
    *,
    run_dir: Path,
    N: int,
    seed: int,
    grid_size: int = 200,
    width: float = 16.0,
    height: float = 12.0,
    dpi: int = 300,
    show: bool = False,
) -> None:
    """
    Save interaction-ridge-style residual surface for CCPP in run_dir/interaction_ridge.png.
    Residual is computed as the non-additive component of the predicted surface:
        R = Y_pred - (row_mean + col_mean - global_mean)
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    X_train = np.asarray(data["X_train"], dtype=float)
    y_train = np.asarray(data["y_train"], dtype=float).ravel()
    scaler_X = _get_input_scaler(data)
    X_train_raw = scaler_X.inverse_transform(X_train) if scaler_X is not None else X_train

    at_min, at_max = float(X_train_raw[:, 0].min()), float(X_train_raw[:, 0].max())
    v_min, v_max = float(X_train_raw[:, 1].min()), float(X_train_raw[:, 1].max())
    at_grid = np.linspace(at_min, at_max, grid_size)
    v_grid = np.linspace(v_min, v_max, grid_size)
    AT_grid, V_grid = np.meshgrid(at_grid, v_grid)
    grid_raw = np.column_stack([AT_grid.ravel(), V_grid.ravel()])
    X_grid = scaler_X.transform(grid_raw) if scaler_X is not None else grid_raw

    K_train = build_kernel_matrix(X_train, X_train, theta, apply_psd_hygiene_for_square=True)
    K_reg, _ = _regularize_kernel_ccpp(K_train, tau, name=f"K_train (RIDGE N={N}, seed={seed})")
    alpha = np.linalg.solve(K_reg, y_train)
    K_grid = build_kernel_matrix(X_grid, X_train, theta, apply_psd_hygiene_for_square=False)
    Y_pred = (K_grid @ alpha).reshape(AT_grid.shape)

    row_mean = Y_pred.mean(axis=1, keepdims=True)
    col_mean = Y_pred.mean(axis=0, keepdims=True)
    global_mean = Y_pred.mean()
    residual = Y_pred - (row_mean + col_mean - global_mean)

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    cf = ax.contourf(AT_grid, V_grid, residual, levels=50, cmap="RdBu")
    fig.colorbar(cf, ax=ax)
    ax.set_xlabel("AT")
    ax.set_ylabel("V")
    ax.set_title(f"Interaction Ridge Recovery (CCPP) N={N}, seed={seed}")
    out_path = run_dir / "interaction_ridge.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"[RIDGE-CCPP] Saved interaction ridge plot -> {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
