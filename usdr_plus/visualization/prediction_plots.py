"""
usdr_plus/visualization/prediction_plots.py
=============================================
Post-training visualisation routines for U_{SDR+} KRR results:
  • 60×60 prediction surface
  • 1-D slices at x₁=π and x₂=π
  • Interaction ridge recovery
  • True vs predicted scatter (test set)
  • Residual violin + jittered scatter (train/val/test)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from usdr_plus.config import NORMALIZE, RAW_DOMAIN
from usdr_plus.data.generator import generate_test_grid, true_function
from usdr_plus.quantum.kernel import build_kernel_matrix
from usdr_plus.training.krr import apply_psd_hygiene, krr_predict


# ---------------------------------------------------------------------------
# 60×60 prediction surface
# ---------------------------------------------------------------------------


def plot_prediction_surface_60x60(
    data_dict: dict,
    theta_opt,
    tau_opt: float,
    N: int,
    SEED: int,
    output_dir: "str | Path | None" = None,
    show: bool = True,
    dpi: int = 300,
    width: float = 18,
    height: float = 5.5,
):
    """
    U_{SDR+} prediction surface on a 60×60 grid (constrained hyperparams).

    Protocol-compliant:
      • Grid is in RAW domain [0, 2π]².
      • Grid inputs are preprocessed with the *same scheme* as train
        (MinMax or Z-score), then fed to U_{SDR+}.
      • Kernel is the fidelity kernel built from usdr_plus_state
        (L=2, CNOT, (X,Z)/(Z,X), β-scaling).
      • KRR uses θ_opt, τ_opt learned on (train,val).

    Extended behaviour for this experiment:
      • Saves the figure to `prediction_surface.png` inside `output_dir`
        if provided.
      • Optionally shows the figure in the notebook (show=True).
      • Returns a dict with grid diagnostics for logging/reporting.
    """
    # ---------- 1. Build raw 60×60 grid ----------
    X1_raw, X2_raw, Y_true = generate_test_grid(
        grid_size=60,
        domain=RAW_DOMAIN,        # (0, 2π)
    )                             # Y_true is noiseless f(x1,x2)
    X_grid_raw = np.column_stack(
        [X1_raw.ravel(), X2_raw.ravel()]
    )

    # ---------- 2. Preprocess grid like training ----------
    X_train = data_dict["X_train"]      # preprocessed (MinMax/Z-score)
    y_train = data_dict["y_train"]

    # If you still have raw train, we can reconstruct the scaler exactly.
    # Otherwise, we approximate MinMax via the known [0, 2π] domain.
    if NORMALIZE == "minmax":
        if "X_train_raw" in data_dict:
            # Exact: fit MinMax on the same raw train used in preprocessing
            X_train_raw = data_dict["X_train_raw"]
            scaler = MinMaxScaler().fit(X_train_raw)
            X_grid_proc = np.clip(
                scaler.transform(X_grid_raw), 0.0, 1.0
            )
            print("[GRID] Using MinMaxScaler fitted on X_train_raw.")
        else:
            # Approximate: map [0, 2π] → [0,1] per coordinate
            X_grid_proc = X_grid_raw / (RAW_DOMAIN[1] - RAW_DOMAIN[0])
            print(
                "[GRID] WARNING: X_train_raw not provided; "
                "using domain-based MinMax x/(2π) for grid."
            )

    elif NORMALIZE == "zscore":
        if "X_train_raw" not in data_dict:
            raise ValueError(
                "[GRID] X_train_raw required for Z-score normalization."
            )
        X_train_raw = data_dict["X_train_raw"]
        scaler = StandardScaler().fit(X_train_raw)
        X_grid_proc = scaler.transform(X_grid_raw)
        print("[GRID] Using StandardScaler fitted on X_train_raw.")
    else:
        raise ValueError(f"Unknown NORMALIZE mode: {NORMALIZE}")

    print(
        f"[GRID] X_train shape={X_train.shape}, "
        f"X_grid_proc shape={X_grid_proc.shape}"
    )

    # ---------- 3. Build kernel matrices with U_{SDR+} ----------
    K_train = build_kernel_matrix(X_train, X_train, theta_opt)
    K_grid  = build_kernel_matrix(X_grid_proc, X_train, theta_opt)

    print(
        f"[GRID] K_train shape={K_train.shape}, "
        f"K_grid shape={K_grid.shape}"
    )

    # ---------- 4. PSD hygiene + solve KRR ----------
    K_reg, jitter, K_stats, kappa_after = apply_psd_hygiene(
        K_train,
        tau_opt,
        name=f"K_train (N={N}, seed={SEED})",
    )

    alpha = np.linalg.solve(K_reg, y_train)

    # Predict on grid: (n_grid × n_train) @ (n_train,) → (n_grid,)
    y_pred_grid = krr_predict(K_grid, alpha)
    Y_pred = y_pred_grid.reshape(X1_raw.shape)

    # ---------- 5. Diagnostics ----------
    error = np.abs(Y_pred - Y_true)

    print(
        f"[GRID] Y_true range: [{Y_true.min():.3f}, {Y_true.max():.3f}]"
    )
    print(
        f"[GRID] Y_pred range: [{Y_pred.min():.3f}, {Y_pred.max():.3f}]"
    )
    print(
        f"[GRID] |error| mean={error.mean():.4e}, "
        f"max={error.max():.4e}"
    )

    grid_mse = mean_squared_error(Y_true.ravel(), Y_pred.ravel())
    print(
        f"[GRID] MSE over 60×60 grid: {grid_mse:.6e} "
        f"(N={N}, SEED={SEED})"
    )

    # ---------- 6. Plot ----------
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(width, height), dpi=dpi
    )

    im1 = ax1.contourf(
        X1_raw, X2_raw, Y_true, levels=60, cmap="viridis"
    )
    ax1.set_title("True Function $f(x_1,x_2)$")
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.contourf(
        X1_raw, X2_raw, Y_pred, levels=60, cmap="viridis"
    )
    ax2.set_title(
        r"$U_{\mathrm{SDR}+}$ Prediction" + f"\nN={N}, SEED={SEED}"
    )
    ax2.set_xlabel("$x_1$")
    ax2.set_ylabel("$x_2$")
    plt.colorbar(im2, ax=ax2)

    im3 = ax3.contourf(
        X1_raw, X2_raw, error, levels=60, cmap="Reds", vmin=0
    )
    ax3.set_title("Absolute Error")
    ax3.set_xlabel("$x_1$")
    ax3.set_ylabel("$x_2$")
    plt.colorbar(im3, ax=ax3)

    plt.suptitle(
        r"$U_{\mathrm{SDR}+}$ Learned Predictor vs True Function",
        fontsize=16,
    )
    plt.tight_layout()

    # ---------- 7. Save + show/close ----------
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "prediction_surface.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"[GRID] Saved prediction surface → {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    # ---------- 8. Return diagnostics for optional logging ----------
    return {
        "grid_mse":     float(grid_mse),
        "jitter":       float(jitter),
        "kappa_train":  float(K_stats["cond"]),
        "kappa_reg":    float(kappa_after),
    }


# ---------------------------------------------------------------------------
# 1-D slices
# ---------------------------------------------------------------------------


def plot_1d_slices(
    data_dict: dict,
    theta_opt,
    tau_opt: float,
    N: int,
    SEED: int,
    output_dir: "str | Path | None" = None,
    show: bool = True,
    grid_size: int = 200,
):
    """
    1D slices of the learned predictor:

      • Slice 1:  x1 = π,  x2 ∈ [0, 2π]
      • Slice 2:  x2 = π,  x1 ∈ [0, 2π]

    Protocol-compliant:
      • Slices are defined in RAW domain [0, 2π].
      • Slice inputs are preprocessed with the SAME scheme
        as training (MinMax or Z-score), then passed to U_{SDR+}.
      • KRR uses θ_opt, τ_opt as learned from (train, val).

    Extended behaviour for this experiment:
      • Saves the figure as 'slices.png' inside output_dir (if provided).
      • Optionally shows the figure in the notebook (show=True).
      • Returns a dict with slice diagnostics (MSE, errors, etc.).
    """
    # ---------- 1. Raw 1D grids ----------
    x_grid_raw = np.linspace(RAW_DOMAIN[0], RAW_DOMAIN[1], grid_size)

    # Raw slice points in [0, 2π]²
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
            # Exact scaler from raw train
            X_train_raw = data_dict["X_train_raw"]
            scaler = MinMaxScaler().fit(X_train_raw)
            X_slice1 = np.clip(scaler.transform(X_slice1_raw), 0.0, 1.0)
            X_slice2 = np.clip(scaler.transform(X_slice2_raw), 0.0, 1.0)
            print("[SLICES] Using MinMaxScaler fitted on X_train_raw.")
        else:
            # Domain-based MinMax approximation [0,2π] → [0,1]
            X_slice1 = X_slice1_raw / (RAW_DOMAIN[1] - RAW_DOMAIN[0])
            X_slice2 = X_slice2_raw / (RAW_DOMAIN[1] - RAW_DOMAIN[0])
            print(
                "[SLICES] WARNING: X_train_raw not provided; "
                "using domain-based MinMax x/(2π) for slices."
            )

    elif NORMALIZE == "zscore":
        if "X_train_raw" not in data_dict:
            raise ValueError(
                "[SLICES] X_train_raw required for Z-score normalization."
            )
        X_train_raw = data_dict["X_train_raw"]
        scaler = StandardScaler().fit(X_train_raw)
        X_slice1 = scaler.transform(X_slice1_raw)
        X_slice2 = scaler.transform(X_slice2_raw)
        print("[SLICES] Using StandardScaler fitted on X_train_raw.")
    else:
        raise ValueError(f"Unknown NORMALIZE mode: {NORMALIZE}")

    print(
        f"[SLICES] X_train shape={X_train.shape}, "
        f"X_slice1 shape={X_slice1.shape}, X_slice2 shape={X_slice2.shape}"
    )

    # ---------- 3. Build kernels and solve KRR ----------
    # Train Gram matrix with U_{SDR+}
    K_train = build_kernel_matrix(X_train, X_train, theta_opt)

    # PSD hygiene + τ-regularization
    K_reg, jitter, K_stats, kappa_after = apply_psd_hygiene(
        K_train,
        tau_opt,
        name=f"K_train (N={N}, seed={SEED}) [slices]",
    )

    alpha = np.linalg.solve(K_reg, y_train)

    # Kernel between slices and train (shape: grid_size × n_train)
    K_slice1 = build_kernel_matrix(X_slice1, X_train, theta_opt)
    K_slice2 = build_kernel_matrix(X_slice2, X_train, theta_opt)

    # Predictions on slices
    y_pred1 = krr_predict(K_slice1, alpha)  # (grid_size,)
    y_pred2 = krr_predict(K_slice2, alpha)

    # ---------- 4. Slice diagnostics ----------
    err1 = y_pred1 - y_true1
    err2 = y_pred2 - y_true2

    mse1 = mean_squared_error(y_true1, y_pred1)
    mse2 = mean_squared_error(y_true2, y_pred2)

    mean_abs_err1 = np.mean(np.abs(err1))
    mean_abs_err2 = np.mean(np.abs(err2))

    max_abs_err1 = np.max(np.abs(err1))
    max_abs_err2 = np.max(np.abs(err2))

    print(
        f"[SLICES] Slice1 (x1=π):  "
        f"MSE={mse1:.4e}, "
        f"mean|err|={mean_abs_err1:.4e}, "
        f"max|err|={max_abs_err1:.4e}"
    )
    print(
        f"[SLICES] Slice2 (x2=π):  "
        f"MSE={mse2:.4e}, "
        f"mean|err|={mean_abs_err2:.4e}, "
        f"max|err|={max_abs_err2:.4e}"
    )

    # ---------- 5. Plot ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 10), dpi=300)

    # Slice 1: x1 = π, vary x2
    ax1.plot(x_grid_raw, y_true1, "k-", linewidth=2, label="True")
    ax1.plot(x_grid_raw, y_pred1, "r--", linewidth=2, label=r"$U_{\mathrm{SDR}+}$")
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
    ax2.plot(x_grid_raw, y_pred2, "r--", linewidth=2, label=r"$U_{\mathrm{SDR}+}$")
    ax2.set_title(
        rf"Slice: $x_2 = \pi$ (vary $x_1$) | $N={N}$, SEED={SEED}",
        fontsize=14,
    )
    ax2.set_xlabel(r"$x_1$")
    ax2.set_ylabel(r"$f(x)$")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle(
        rf"1D Slices of $U_{{\mathrm{{SDR}}+}}$ Predictor (N={N}, SEED={SEED})",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # ---------- 6. Save + show/close ----------
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "slices.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[SLICES] Saved 1D slices figure → {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    # ---------- 7. Return diagnostics for optional logging ----------
    return {
        "mse_slice1":        float(mse1),
        "mse_slice2":        float(mse2),
        "mean_abs_err1":     float(mean_abs_err1),
        "mean_abs_err2":     float(mean_abs_err2),
        "max_abs_err1":      float(max_abs_err1),
        "max_abs_err2":      float(max_abs_err2),
        "jitter":            float(jitter),
        "kappa_train_slices": float(K_stats["cond"]),
        "kappa_reg_slices":   float(kappa_after),
    }


# ---------------------------------------------------------------------------
# Interaction ridge recovery
# ---------------------------------------------------------------------------


def plot_interaction_ridge(
    data_dict: dict,
    theta_opt,
    tau_opt: float,
    N: int,
    SEED: int,
    output_dir: "str | Path | None" = None,
    show: bool = True,
    grid_size: int = 100,
):
    """
    Interaction ridge visualization for U_{SDR+}.

    Shows how well the learned predictor recovers the interaction term
        0.1 x1 x2
    after removing the smooth part (sin x1 + cos x2).

    Protocol-compliant:
      • Build a RAW grid in [0, 2π]^2.
      • Preprocess the grid with the SAME normalization as training
        (MinMax or Z-score), then apply U_{SDR+}.
      • Use θ_opt, τ_opt from the (train,val) KRR fit.

    Extended behaviour for this experiment:
      • Saves the figure as 'interaction_ridge.png' inside output_dir (if provided).
      • Optionally shows the figure in the notebook (show=True).
      • Returns a dict with ridge diagnostics (MSE, correlation, etc.).
    """
    # ---------- 1. RAW grid over [0, 2π]^2 ----------
    x_grid_raw = np.linspace(RAW_DOMAIN[0], RAW_DOMAIN[1], grid_size)
    X1_raw, X2_raw = np.meshgrid(x_grid_raw, x_grid_raw)
    X_full_raw = np.column_stack([X1_raw.ravel(), X2_raw.ravel()])

    # True components
    smooth_true = np.sin(X1_raw) + np.cos(X2_raw)
    interaction_true = 0.1 * X1_raw * X2_raw  # 0.1 x1 x2

    # ---------- 2. Preprocess grid like training ----------
    X_train = data_dict["X_train"]    # preprocessed train
    y_train = data_dict["y_train"]

    if NORMALIZE == "minmax":
        if "X_train_raw" in data_dict:
            X_train_raw = data_dict["X_train_raw"]
            scaler = MinMaxScaler().fit(X_train_raw)
            X_full_proc = np.clip(scaler.transform(X_full_raw), 0.0, 1.0)
            print("[RIDGE] Using MinMaxScaler fitted on X_train_raw.")
        else:
            # Domain-based MinMax approximation: [0,2π] → [0,1]
            X_full_proc = X_full_raw / (RAW_DOMAIN[1] - RAW_DOMAIN[0])
            print(
                "[RIDGE] WARNING: X_train_raw not provided; "
                "using domain-based MinMax x/(2π) for grid."
            )

    elif NORMALIZE == "zscore":
        if "X_train_raw" not in data_dict:
            raise ValueError(
                "[RIDGE] X_train_raw required for Z-score normalization."
            )
        X_train_raw = data_dict["X_train_raw"]
        scaler = StandardScaler().fit(X_train_raw)
        X_full_proc = scaler.transform(X_full_raw)
        print("[RIDGE] Using StandardScaler fitted on X_train_raw.")
    else:
        raise ValueError(f"Unknown NORMALIZE mode: {NORMALIZE}")

    print(
        f"[RIDGE] X_train shape={X_train.shape}, "
        f"X_full_proc shape={X_full_proc.shape}"
    )

    # ---------- 3. Build kernels & solve KRR ----------
    K_train = build_kernel_matrix(X_train, X_train, theta_opt)

    K_reg, jitter, K_stats, kappa_after = apply_psd_hygiene(
        K_train,
        tau_opt,
        name=f"K_train (N={N}, seed={SEED}) [ridge]",
    )

    alpha = np.linalg.solve(K_reg, y_train)

    # K_full: grid_size^2 × n_train
    K_full = build_kernel_matrix(X_full_proc, X_train, theta_opt)
    y_pred_flat = krr_predict(K_full, alpha)    # (grid_size^2,)
    Y_pred = y_pred_flat.reshape(X1_raw.shape)  # (grid_size, grid_size)

    # ---------- 4. Residual ≈ interaction ridge ----------
    residual = Y_pred - smooth_true            # ≈ 0.1 x1 x2

    # Diagnostics: how well residual matches 0.1 x1 x2?
    corr = np.corrcoef(residual.ravel(), interaction_true.ravel())[0, 1]
    mse_ridge = mean_squared_error(
        interaction_true.ravel(), residual.ravel()
    )
    mean_abs_res = float(np.mean(np.abs(residual)))
    max_abs_res  = float(np.max(np.abs(residual)))

    print(
        f"[RIDGE] Residual vs 0.1 x1 x2: "
        f"MSE={mse_ridge:.4e}, corr={corr:.4f}, "
        f"mean|res|={mean_abs_res:.4e}"
    )

    # ---------- 5. Plot ----------
    fig = plt.figure(figsize=(16, 12), dpi=300)
    ax = plt.gca()

    cf = ax.contourf(
        X1_raw,
        X2_raw,
        residual,
        levels=50,
        cmap="RdBu",
        vmin=-2,
        vmax=2,
    )
    cbar = plt.colorbar(cf)
    cbar.set_label(
        r"Residual $f_{\theta}(x) - (\sin x_1 + \cos x_2)$",
        fontsize=12,
    )

    # Overlay true interaction contours
    cs = ax.contour(
        X1_raw,
        X2_raw,
        interaction_true,
        levels=[0.5, 1.0, 1.5, 2.0],
        colors="black",
        alpha=0.6,
    )
    ax.clabel(cs, inline=True, fontsize=10, fmt="%.1f")

    ax.set_title(
        rf"Interaction Ridge Recovery ($0.1 x_1 x_2$) | N={N}, SEED={SEED}",
        fontsize=16,
    )
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    plt.tight_layout()

    # ---------- 6. Save + show/close ----------
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "interaction_ridge.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[RIDGE] Saved interaction ridge figure → {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    # ---------- 7. Return diagnostics for optional logging ----------
    return {
        "mse_ridge":          float(mse_ridge),
        "corr_ridge":         float(corr),
        "mean_abs_residual":  mean_abs_res,
        "max_abs_residual":   max_abs_res,
        "jitter":             float(jitter),
        "kappa_train_ridge":  float(K_stats["cond"]),
        "kappa_reg_ridge":    float(kappa_after),
    }


# ---------------------------------------------------------------------------
# True vs predicted scatter
# ---------------------------------------------------------------------------


def plot_true_vs_predicted(
    data_dict: dict,
    theta_opt,
    tau_opt: float,
    N: int,
    SEED: int,
    output_dir: "str | Path | None" = None,
    show: bool = True,
    dpi: int = 300,
    width: float = 14,
    height: float = 12,
):
    """
    True vs Predicted scatter on the TEST set.

    Protocol-compliant:
      • Uses preprocessed X_train/X_test (MinMax or Z-score).
      • Kernel is the U_{SDR+} fidelity kernel via build_kernel_matrix.
      • KRR uses θ_opt, τ_opt with the same PSD hygiene as the main pipeline.

    Extended behaviour for this experiment:
      • Saves the figure as 'true_vs_pred.png' inside output_dir (if provided).
      • Optionally shows the figure in the notebook (show=True).
      • Returns a dict with test MSE and numerical diagnostics.
    """
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_test  = data_dict["X_test"]
    y_test  = data_dict["y_test"]

    print(f"[TVP] N={N}, SEED={SEED}")
    print(f"[TVP] X_train shape={X_train.shape}, X_test shape={X_test.shape}")

    # --- 1. Build Gram and test kernel ---
    K_train = build_kernel_matrix(X_train, X_train, theta_opt)
    K_test  = build_kernel_matrix(X_test,  X_train, theta_opt)  # (n_test, n_train)

    print(
        f"[TVP] K_train shape={K_train.shape}, "
        f"K_test shape={K_test.shape}"
    )

    # --- 2. PSD hygiene + τ-regularization (same as main KRR) ---
    K_reg, jitter, K_stats, kappa_after = apply_psd_hygiene(
        K_train,
        tau_opt,
        name=f"K_train (N={N}, seed={SEED}) [TVP]",
    )

    # --- 3. Solve KRR and predict on test ---
    alpha = np.linalg.solve(K_reg, y_train)
    y_pred = krr_predict(K_test, alpha)   # (n_test,)

    test_mse = mean_squared_error(y_test, y_pred)
    print(
        f"[TVP] Test MSE={test_mse:.4e}, "
        f"κ(K+τI)={kappa_after:.3e}, jitter={jitter:.2e}"
    )

    # --- 4. Scatter plot true vs predicted ---
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    ax = plt.gca()

    ax.scatter(
        y_test,
        y_pred,
        alpha=0.7,
        edgecolor="k",
        linewidth=0.6,
        s=70,
        label="Test samples",
    )

    min_val = float(min(y_test.min(), y_pred.min()))
    max_val = float(max(y_test.max(), y_pred.max()))
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        lw=2,
        label="Ideal y = x",
    )

    ax.set_xlabel("True y", fontsize=12, weight="bold")
    ax.set_ylabel("Predicted y", fontsize=12, weight="bold")
    ax.set_title(
        f"True vs Predicted (Test) | N={N}, SEED={SEED}\n"
        f"MSE = {test_mse:.4f}",
        fontsize=14,
        weight="bold",
    )
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(frameon=True, fontsize=11)
    plt.tight_layout()

    # --- 5. Save + show/close ---
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "true_vs_pred.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"[TVP] Saved true-vs-pred figure → {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    # --- 6. Return diagnostics for optional logging ---
    return {
        "test_mse":        float(test_mse),
        "jitter":          float(jitter),
        "kappa_train_tvp": float(K_stats["cond"]),
        "kappa_reg_tvp":   float(kappa_after),
    }


# ---------------------------------------------------------------------------
# Residual violin plots
# ---------------------------------------------------------------------------


def plot_residual_distributions(
    data_dict: dict,
    theta_opt,
    tau_opt: float,
    N: int,
    SEED: int,
    output_dir: "str | Path | None" = None,
    show: bool = True,
    dpi: int = 300,
    width: float = 10.0,
    height: float = 6.0,
):
    """
    Residual distributions for train / val / test for a single run (N, SEED).

    For each split, compute:
        r_split = y_split - y_hat_split

    and visualize them as violin + jittered scatter, which is more informative
    than histograms in the small-sample regime (especially val / test).

    Parameters
    ----------
    data_dict : dict
        Output of `load_processed_2d_dataset` for this (N, SEED).
        Must contain X_train, y_train, X_val, y_val, X_test, y_test.
    theta_opt : array-like, shape (4,)
        Optimal USDR+ parameters (lambda1, lambda2, gamma, beta).
    tau_opt : float
        Optimal KRR regularization parameter τ.
    N : int
        Training set size for this run.
    SEED : int
        Seed for this run.
    output_dir : str | Path | None, optional
        If provided, PNG is saved under:
            output_dir / "residuals_violin.png"
    show : bool, optional
        Whether to display the figure in the notebook.
    dpi, width, height : plotting parameters.
    """
    # ---------- 1. Unpack splits ----------
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_val   = data_dict["X_val"]
    y_val   = data_dict["y_val"]
    X_test  = data_dict["X_test"]
    y_test  = data_dict["y_test"]

    print(f"[RESID] N={N}, SEED={SEED}")
    print(
        f"[RESID] shapes – "
        f"train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}"
    )

    # ---------- 2. Build train Gram matrix + PSD hygiene ----------
    K_train = build_kernel_matrix(X_train, X_train, theta_opt)

    K_reg, jitter, K_stats, kappa_after = apply_psd_hygiene(
        K_train,
        tau_opt,
        name=f"K_train (N={N}, seed={SEED}) [resid]",
    )

    # Solve (K_reg) alpha = y_train
    alpha = np.linalg.solve(K_reg, y_train)

    # ---------- 3. Predictions and residuals on each split ----------
    # Train: predictions using K_train
    y_hat_train = krr_predict(K_train, alpha)
    r_train = y_train - y_hat_train

    # Val
    K_val = build_kernel_matrix(X_val, X_train, theta_opt)
    y_hat_val = krr_predict(K_val, alpha)
    r_val = y_val - y_hat_val

    # Test
    K_test = build_kernel_matrix(X_test, X_train, theta_opt)
    y_hat_test = krr_predict(K_test, alpha)
    r_test = y_test - y_hat_test

    # ---------- 4. Basic residual diagnostics ----------
    def summarize_residuals(name: str, y_true, y_hat, r):
        mse = mean_squared_error(y_true, y_hat)
        mae = float(np.mean(np.abs(r)))
        print(
            f"[RESID] {name}: "
            f"MSE={mse:.4e}, MAE={mae:.4e}, "
            f"mean(r)={np.mean(r):.4e}, std(r)={np.std(r):.4e}"
        )
        return mse, mae

    train_mse, train_mae = summarize_residuals("train", y_train, y_hat_train, r_train)
    val_mse,   val_mae   = summarize_residuals("val",   y_val,   y_hat_val,   r_val)
    test_mse,  test_mae  = summarize_residuals("test",  y_test,  y_hat_test,  r_test)

    # ---------- 5. Violin + jittered scatter plot ----------
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

    data = [r_train, r_val, r_test]
    labels = ["Train", "Val", "Test"]
    positions = np.arange(1, 4)

    # Violin plots (show means, hide extrema for cleaner look)
    parts = ax.violinplot(
        dataset=data,
        positions=positions,
        showmeans=True,
        showmedians=False,
        showextrema=False,
        widths=0.8,
    )

    # Light styling for violins
    for pc in parts["bodies"]:
        pc.set_alpha(0.5)

    # Means line style
    if "cmeans" in parts:
        parts["cmeans"].set_linewidth(1.5)

    # Jittered scatter for individual points
    rng = np.random.default_rng(42)  # fixed for reproducible jitter shape
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

    # Horizontal zero line to visualize bias
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(r"Residual $r = y - \hat{y}$", fontsize=12)
    ax.set_title(
        f"Residual Distributions (Train / Val / Test)\n"
        f"N={N}, SEED={SEED}",
        fontsize=14,
        weight="bold",
    )

    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()

    # ---------- 6. Save + show ----------
    stats = {
        "N": N,
        "SEED": SEED,
        "train_mse": float(train_mse),
        "train_mae": float(train_mae),
        "val_mse":   float(val_mse),
        "val_mae":   float(val_mae),
        "test_mse":  float(test_mse),
        "test_mae":  float(test_mae),
        "jitter":    float(jitter),
        "kappa_reg": float(kappa_after),
        "kappa_train": float(K_stats["cond"]),
    }

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / "residuals_violin.png"
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[RESID] Saved residuals violin plot → {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return stats
