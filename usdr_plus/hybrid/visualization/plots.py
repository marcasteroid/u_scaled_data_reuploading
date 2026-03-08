"""Hybrid plotting utilities (all outputs saved in figures/hybrid/diagnostics)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from usdr_plus.hybrid import config as cfg
import usdr_plus.config as core_cfg
from usdr_plus.data.generator import generate_test_grid, true_function
from usdr_plus.quantum.kernel import build_kernel_matrix
from usdr_plus.training.krr import apply_psd_hygiene
from usdr_plus.hybrid.quantum.kernel import build_classical_rbf_kernel
from usdr_plus.quantum.circuit import visualize_U_SDR_plus_2D
from usdr_plus.visualization.dataset_plots import plot_gram_matrix
from usdr_plus.analysis.spectrum import analyze_kernel_matrix

matplotlib.use("Agg")
plt.show = lambda *args, **kwargs: None


def save_diagnostics(
    K_train: np.ndarray,
    *,
    theta: np.ndarray,
    x_example: np.ndarray,
    N: int,
    seed: int,
    diagnostics_dir: Path = cfg.DIAGNOSTICS_DIR,
) -> None:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    tag = f"n{N}_seed{seed}"
    analyze_kernel_matrix(
        K_train,
        name=f"K_train_{tag}",
        width=10,
        height=5,
        dpi=300,
        plot=True,
        save=True,
        save_dir=diagnostics_dir,
    )
    plot_gram_matrix(
        K_train,
        title=f"Gram Matrix {tag}",
        cmap="viridis",
        width=8,
        height=6,
        dpi=300,
        annotate=False,
        save=True,
        save_dir=diagnostics_dir,
    )
    visualize_U_SDR_plus_2D(
        x_example=x_example,
        theta=theta,
        width=12,
        height=6,
        dpi=300,
        save=True,
        save_dir=diagnostics_dir,
        plot_name=tag,
    )


def _save_fig(fig: plt.Figure, out_path: Path, dpi: int = 300) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_prediction_surface(
    *,
    y_pred_grid: np.ndarray,
    N: int,
    seed: int,
    output_dir: Path,
    omega_label: str | None = None,
    filename: str = "prediction_surface.png",
) -> None:
    x1, x2, y_true = generate_test_grid(grid_size=60)
    err = np.abs(y_true - y_pred_grid)
    # Wide layout to avoid squashed contour subplots in 1x3 arrangement.
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=300)
    im0 = axes[0].contourf(x1, x2, y_true, levels=60, cmap="viridis")
    axes[0].set_title("True")
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].contourf(x1, x2, y_pred_grid, levels=60, cmap="viridis")
    pred_title = f"Hybrid pred N={N}, seed={seed}"
    if omega_label is not None:
        pred_title += f", {omega_label}"
    axes[1].set_title(pred_title)
    fig.colorbar(im1, ax=axes[1])
    im2 = axes[2].contourf(x1, x2, err, levels=60, cmap="Reds")
    axes[2].set_title("|error|")
    fig.colorbar(im2, ax=axes[2])

    # Enforce identical x/y scale and bounds to avoid geometric distortion.
    x_lo, x_hi = 0.0, 2.0 * np.pi
    y_lo, y_hi = 0.0, 2.0 * np.pi
    for ax in axes:
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle("Hybrid Prediction Surface")
    _save_fig(fig, output_dir / filename)


def save_slices(
    *,
    x_line: np.ndarray,
    y_true1: np.ndarray,
    y_pred1: np.ndarray,
    y_true2: np.ndarray,
    y_pred2: np.ndarray,
    N: int,
    seed: int,
    output_dir: Path,
) -> None:
    # Wide layout to preserve horizontal structure of 1D curves.
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), dpi=300)
    axes[0].plot(x_line, y_true1, "k-", lw=2, label="true")
    axes[0].plot(x_line, y_pred1, "r--", lw=2, label="hybrid")
    axes[0].set_title("Slice x1=pi")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(x_line, y_true2, "k-", lw=2, label="true")
    axes[1].plot(x_line, y_pred2, "r--", lw=2, label="hybrid")
    axes[1].set_title("Slice x2=pi")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"Hybrid 1D Slices N={N}, seed={seed}")
    _save_fig(fig, output_dir / "slices.png")


def normalize_gram_triplet(
    *,
    K_train: np.ndarray,
    K_val: np.ndarray,
    K_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.sqrt(np.clip(np.diag(K_train), 1e-12, None))
    Kt = K_train / np.outer(d, d)
    Kv = K_val / d[None, :]
    Ks = K_test / d[None, :]
    return Kt, Kv, Ks


def rbf_kernel_matrix(X1: np.ndarray, X2: np.ndarray, classical_config) -> np.ndarray:
    sigma = None
    if isinstance(classical_config, dict):
        sigma = classical_config.get("length_scale", classical_config.get("sigma"))
    elif hasattr(classical_config, "length_scale"):
        sigma = getattr(classical_config, "length_scale")
    return build_classical_rbf_kernel(X1, X2, sigma=sigma)


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
    """Notebook-aligned hybrid 1D slices (omega=0,1,omega*)."""
    omega_hybrid = float(omega_hybrid)
    if not (0.0 <= omega_hybrid <= 1.0):
        raise ValueError(f"omega_hybrid must be in [0,1], got {omega_hybrid}")

    x_grid_raw = np.linspace(core_cfg.RAW_DOMAIN[0], core_cfg.RAW_DOMAIN[1], grid_size)
    X_slice1_raw = np.stack([np.full(grid_size, np.pi), x_grid_raw], axis=1)
    X_slice2_raw = np.stack([x_grid_raw, np.full(grid_size, np.pi)], axis=1)
    y_true1 = true_function(np.pi * np.ones_like(x_grid_raw), x_grid_raw)
    y_true2 = true_function(x_grid_raw, np.pi * np.ones_like(x_grid_raw))

    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]

    if cfg.NORMALIZE == "minmax":
        if "X_train_raw" in data_dict:
            X_train_raw = data_dict["X_train_raw"]
            scaler = MinMaxScaler().fit(X_train_raw)
            X_slice1 = np.clip(scaler.transform(X_slice1_raw), 0.0, 1.0)
            X_slice2 = np.clip(scaler.transform(X_slice2_raw), 0.0, 1.0)
        else:
            X_slice1 = X_slice1_raw / (core_cfg.RAW_DOMAIN[1] - core_cfg.RAW_DOMAIN[0])
            X_slice2 = X_slice2_raw / (core_cfg.RAW_DOMAIN[1] - core_cfg.RAW_DOMAIN[0])
    elif cfg.NORMALIZE == "zscore":
        if "X_train_raw" not in data_dict:
            raise ValueError("[SLICES-HYB] X_train_raw required for Z-score normalization.")
        scaler = StandardScaler().fit(data_dict["X_train_raw"])
        X_slice1 = scaler.transform(X_slice1_raw)
        X_slice2 = scaler.transform(X_slice2_raw)
    else:
        raise ValueError(f"Unknown NORMALIZE mode: {cfg.NORMALIZE}")

    K_Q_train = build_kernel_matrix(X_train, X_train, theta_opt)
    K_Q_s1 = build_kernel_matrix(X_slice1, X_train, theta_opt)
    K_Q_s2 = build_kernel_matrix(X_slice2, X_train, theta_opt)
    K_C_train = rbf_kernel_matrix(X_train, X_train, classical_config)
    K_C_s1 = rbf_kernel_matrix(X_slice1, X_train, classical_config)
    K_C_s2 = rbf_kernel_matrix(X_slice2, X_train, classical_config)

    KtQ, Ks1Q, Ks2Q = normalize_gram_triplet(K_train=K_Q_train, K_val=K_Q_s1, K_test=K_Q_s2)
    KtC, Ks1C, Ks2C = normalize_gram_triplet(K_train=K_C_train, K_val=K_C_s1, K_test=K_C_s2)

    def _mix(KC, KQ, omega: float) -> np.ndarray:
        return (1.0 - omega) * KC + omega * KQ

    K_train_omega0 = _mix(KtC, KtQ, 0.0)
    K_train_omega1 = _mix(KtC, KtQ, 1.0)
    K_train_omegah = _mix(KtC, KtQ, omega_hybrid)
    K_s1_omega0 = _mix(Ks1C, Ks1Q, 0.0)
    K_s1_omega1 = _mix(Ks1C, Ks1Q, 1.0)
    K_s1_omegah = _mix(Ks1C, Ks1Q, omega_hybrid)
    K_s2_omega0 = _mix(Ks2C, Ks2Q, 0.0)
    K_s2_omega1 = _mix(Ks2C, Ks2Q, 1.0)
    K_s2_omegah = _mix(Ks2C, Ks2Q, omega_hybrid)

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

    (y1_omega0, y2_omega0), stats_omega0 = _solve_and_predict(K_train_omega0, K_s1_omega0, K_s2_omega0, "omega=0")
    (y1_omega1, y2_omega1), stats_omega1 = _solve_and_predict(K_train_omega1, K_s1_omega1, K_s2_omega1, "omega=1")
    (y1_omegah, y2_omegah), stats_omegah = _solve_and_predict(K_train_omegah, K_s1_omegah, K_s2_omegah, f"omega={omega_hybrid:.2f}")

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width, height), dpi=dpi)
    ax1.plot(x_grid_raw, y_true1, "k-", linewidth=2, label="True")
    ax1.plot(x_grid_raw, y1_omega0, "-", linewidth=2, label="omega=0 (classical)")
    ax1.plot(x_grid_raw, y1_omega1, "--", linewidth=2, label="omega=1 (USDR+)")
    ax1.plot(x_grid_raw, y1_omegah, "-.", linewidth=2, label=rf"omega={omega_hybrid:.2f} (hybrid)")
    ax1.set_title(rf"Slice: $x_1=\pi$ | N={N}, SEED={SEED}", fontsize=14)
    ax1.set_xlabel(r"$x_2$")
    ax1.set_ylabel(r"$f(x)$")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(x_grid_raw, y_true2, "k-", linewidth=2, label="True")
    ax2.plot(x_grid_raw, y2_omega0, "-", linewidth=2, label="omega=0 (classical)")
    ax2.plot(x_grid_raw, y2_omega1, "--", linewidth=2, label="omega=1 (USDR+)")
    ax2.plot(x_grid_raw, y2_omegah, "-.", linewidth=2, label=rf"omega={omega_hybrid:.2f} (hybrid)")
    ax2.set_title(rf"Slice: $x_2=\pi$ | N={N}, SEED={SEED}", fontsize=14)
    ax2.set_xlabel(r"$x_1$")
    ax2.set_ylabel(r"$f(x)$")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.suptitle(rf"1D Slices - Hybrid kernel (N={N}, SEED={SEED}, tau*={tau_star:.1e})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "slices_hybrid.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"[SLICES-HYB] Saved 1D slices figure -> {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "omega_hybrid": omega_hybrid,
        "mse_slice1_omega0": float(mse1_0),
        "mse_slice1_omega1": float(mse1_1),
        "mse_slice1_omegah": float(mse1_h),
        "mean_abs_err1_omega0": float(mae1_0),
        "mean_abs_err1_omega1": float(mae1_1),
        "mean_abs_err1_omegah": float(mae1_h),
        "max_abs_err1_omega0": float(maxe1_0),
        "max_abs_err1_omega1": float(maxe1_1),
        "max_abs_err1_omegah": float(maxe1_h),
        "mse_slice2_omega0": float(mse2_0),
        "mse_slice2_omega1": float(mse2_1),
        "mse_slice2_omegah": float(mse2_h),
        "mean_abs_err2_omega0": float(mae2_0),
        "mean_abs_err2_omega1": float(mae2_1),
        "mean_abs_err2_omegah": float(mae2_h),
        "max_abs_err2_omega0": float(maxe2_0),
        "max_abs_err2_omega1": float(maxe2_1),
        "max_abs_err2_omegah": float(maxe2_h),
        "stats_omega0": stats_omega0,
        "stats_omega1": stats_omega1,
        "stats_omegah": stats_omegah,
    }


def save_interaction_ridge(
    *,
    x1: np.ndarray,
    x2: np.ndarray,
    residual: np.ndarray,
    N: int,
    seed: int,
    output_dir: Path,
    width: float = 16.0,
    height: float = 12.0,
    dpi: int = 300,
) -> None:
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    cf = ax.contourf(x1, x2, residual, levels=50, cmap="RdBu")
    # Add contour lines on top of filled map to improve ridge readability.
    cl = ax.contour(x1, x2, residual, levels=12, colors="black", linewidths=0.6, alpha=0.45)
    ax.clabel(cl, inline=True, fontsize=8, fmt="%.2f")
    fig.colorbar(cf, ax=ax)
    ax.set_title(f"Interaction Ridge N={N}, seed={seed}")
    _save_fig(fig, output_dir / "interaction_ridge.png", dpi=dpi)


def save_true_vs_pred(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    N: int,
    seed: int,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 12), dpi=300)
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolor="k", s=60)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "r--", lw=2)
    ax.set_title(f"True vs Pred N={N}, seed={seed}")
    ax.grid(alpha=0.3)
    _save_fig(fig, output_dir / "true_vs_pred.png")


def save_residual_violin(
    *,
    r_train: np.ndarray,
    r_val: np.ndarray,
    r_test: np.ndarray,
    N: int,
    seed: int,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(20, 12), dpi=300)
    groups = [r_train, r_val, r_test]
    positions = np.array([1, 2, 3], dtype=float)
    parts = ax.violinplot(
        groups,
        positions=positions,
        showmeans=True,
        showmedians=True,
        showextrema=True,
    )
    for body in parts["bodies"]:
        body.set_alpha(0.55)

    # Overlay residual samples with horizontal jitter so point density is visible.
    rng = np.random.default_rng(seed)
    for idx, residuals in enumerate(groups):
        x = positions[idx] + 0.06 * rng.normal(size=residuals.size)
        ax.scatter(
            x,
            residuals,
            s=20,
            alpha=0.75,
            color="black",
            edgecolors="white",
            linewidths=0.3,
            zorder=3,
        )

    ax.axhline(0.0, ls="--", lw=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(["Train", "Val", "Test"])
    ax.set_title(f"Residuals N={N}, seed={seed}")
    _save_fig(fig, output_dir / "residuals_violin.png")


def _effective_rank(K: np.ndarray, eps: float = 1e-12) -> float:
    e = np.linalg.eigvalsh(0.5 * (K + K.T))
    e = np.clip(e, 0.0, None)
    s = float(e.sum())
    if s <= eps:
        return 0.0
    p = np.clip(e / s, eps, 1.0)
    return float(np.exp(-np.sum(p * np.log(p))))


def plot_kernel_geometry_heatmaps_case(
    *,
    Ktilde_C_train: np.ndarray,
    Ktilde_Q_train: np.ndarray,
    hybrid_mats: dict[float, np.ndarray],
    omegas_to_show: list[float],
    N: int,
    seed: int,
    output_dir: Path,
    width: float = 12.0,
    height: float = 12.0,
    dpi: int = 300,
) -> None:
    mats = [("Classical", Ktilde_C_train), ("USDR+", Ktilde_Q_train)]
    for w in omegas_to_show:
        if w in hybrid_mats:
            mats.append((f"Hybrid w={w:.2f}", hybrid_mats[w]))
    n = len(mats)
    fig, axes = plt.subplots(1, n, figsize=(width * n, height), dpi=dpi, constrained_layout=True)
    if n == 1:
        axes = [axes]
    vmin = min(float(np.min(m)) for _, m in mats)
    vmax = max(float(np.max(m)) for _, m in mats)
    for ax, (name, M) in zip(axes, mats):
        im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap="viridis", aspect="equal")
        ax.set_box_aspect(1)
        ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Kernel Geometry Heatmaps (N={N}, seed={seed})")
    _save_fig(fig, output_dir / f"geometry_heatmaps_N{N}_seed{seed}.png", dpi=dpi)


def plot_eigenvalue_spectra_for_case(
    *,
    Ktilde_C_train: np.ndarray,
    Ktilde_Q_train: np.ndarray,
    hybrid_mats: dict[float, np.ndarray],
    omegas_to_show: list[float],
    title: str,
    output_dir: Path,
    width: float = 32.0,
    height: float = 18.0,
    dpi: int = 300,
) -> None:
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    def _eig_sorted(M: np.ndarray) -> np.ndarray:
        return np.sort(np.linalg.eigvalsh(0.5 * (M + M.T)))[::-1]
    ax.semilogy(_eig_sorted(Ktilde_C_train), label="classical", lw=2, marker="o", ms=3)
    ax.semilogy(_eig_sorted(Ktilde_Q_train), label="usdr+", lw=2, marker="o", ms=3)
    for w in omegas_to_show:
        if w in hybrid_mats:
            ax.semilogy(_eig_sorted(hybrid_mats[w]), label=f"hybrid w={w:.2f}", lw=1.5, marker="o", ms=3)
    ax.set_title(title)
    ax.set_xlabel("index")
    ax.set_ylabel("eigenvalue")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    _save_fig(fig, output_dir / f"eigen_spectra_{title.replace(' ', '_').replace('(', '').replace(')', '')}.png", dpi=dpi)


def plot_effective_rank_vs_omega(df: np.ndarray | "pd.DataFrame", output_dir: Path, dpi: int = 300) -> None:
    import pandas as pd
    d = pd.DataFrame(df).copy()
    g = d.groupby("omega", as_index=False)["effective_rank"].mean()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    ax.plot(g["omega"], g["effective_rank"], "o-", lw=2)
    ax.set_title("Effective rank vs omega")
    ax.set_xlabel("omega")
    ax.set_ylabel("effective rank")
    ax.grid(alpha=0.3)
    _save_fig(fig, output_dir / "effective_rank_vs_omega.png", dpi=dpi)


def plot_condition_number_vs_omega(df: np.ndarray | "pd.DataFrame", output_dir: Path, dpi: int = 300) -> None:
    import pandas as pd
    d = pd.DataFrame(df).copy()
    g = d.groupby("omega", as_index=False)["kappa_reg"].mean()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    ax.plot(g["omega"], g["kappa_reg"], "o-", lw=2)
    ax.set_yscale("log")
    ax.set_title("Condition number vs omega (regularized)")
    ax.set_xlabel("omega")
    ax.set_ylabel("kappa_reg")
    ax.grid(alpha=0.3, which="both")
    _save_fig(fig, output_dir / "condition_vs_omega.png", dpi=dpi)


def plot_mse_vs_omega(df: np.ndarray | "pd.DataFrame", output_dir: Path, dpi: int = 300) -> None:
    import pandas as pd
    d = pd.DataFrame(df).copy()
    g = d.groupby("omega", as_index=False)[["val_mse", "test_mse"]].mean()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    ax.plot(g["omega"], g["val_mse"], "o-", lw=2, label="val")
    ax.plot(g["omega"], g["test_mse"], "o-", lw=2, label="test")
    ax.set_title("Val/Test MSE vs omega")
    ax.set_xlabel("omega")
    ax.set_ylabel("MSE")
    ax.grid(alpha=0.3)
    ax.legend()
    _save_fig(fig, output_dir / "mse_vs_omega.png", dpi=dpi)


def plot_summary_bars_per_N(df_best: "pd.DataFrame", output_dir: Path, dpi: int = 300) -> None:
    import pandas as pd
    d = pd.DataFrame(df_best).copy()
    g = d.groupby("N", as_index=False).agg(
        test_mse_classical=("test_mse_omega0", "mean"),
        test_mse_usdr=("test_mse_omega1", "mean"),
        test_mse_hybrid=("test_mse", "mean"),
    )
    x = np.arange(len(g))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 7), dpi=dpi)
    ax.bar(x - w, g["test_mse_classical"], width=w, label="omega=0 classical")
    ax.bar(x, g["test_mse_usdr"], width=w, label="omega=1 usdr+")
    ax.bar(x + w, g["test_mse_hybrid"], width=w, label="omega*=hybrid")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in g["N"]])
    ax.set_xlabel("N")
    ax.set_ylabel("Test MSE")
    ax.set_title("Test MSE comparison per N: classical usdr+ hybrid")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, output_dir / "test_mse_comparison_per_N.png", dpi=dpi)


def plot_mse_vs_rank_condition_scatter(df: "pd.DataFrame", output_dir: Path, dpi: int = 300) -> None:
    import pandas as pd
    d = pd.DataFrame(df).copy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=dpi)
    sc1 = axes[0].scatter(d["effective_rank"], d["test_mse"], c=d["omega"], cmap="viridis", alpha=0.8)
    axes[0].set_title("Test MSE vs effective rank")
    axes[0].set_xlabel("effective rank")
    axes[0].set_ylabel("test MSE")
    fig.colorbar(sc1, ax=axes[0], label="omega")
    sc2 = axes[1].scatter(d["kappa_reg"], d["test_mse"], c=d["omega"], cmap="viridis", alpha=0.8)
    axes[1].set_xscale("log")
    axes[1].set_title("Test MSE vs kappa_reg")
    axes[1].set_xlabel("kappa_reg")
    axes[1].set_ylabel("test MSE")
    fig.colorbar(sc2, ax=axes[1], label="omega")
    _save_fig(fig, output_dir / "hybrid_kernel_test_mse_vs_spectral_complexity.png", dpi=dpi)
