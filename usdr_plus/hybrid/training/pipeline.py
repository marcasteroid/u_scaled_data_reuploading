"""Hybrid training loop for all N and seeds with full diagnostics plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from usdr_plus.hybrid import config as cfg
from usdr_plus.hybrid.data.datasets import load_processed_hybrid_dataset
from usdr_plus.hybrid.quantum.kernel import (
    build_classical_rbf_kernel,
    build_hybrid_train_test_kernels,
    grid_search_classical_rbf_krr,
    normalize_gram,
)
from usdr_plus.hybrid.visualization.plots import (
    _effective_rank,
    plot_eigenvalue_spectra_for_case,
    plot_kernel_geometry_heatmaps_case,
    plot_mse_vs_rank_condition_scatter,
    plot_summary_bars_per_N,
    plot_hybrid_1d_slices,
    save_diagnostics,
    save_interaction_ridge,
    save_prediction_surface,
    save_residual_violin,
    save_true_vs_pred,
)
from usdr_plus.training.optimizer import optimize_theta_tau
from usdr_plus.data.generator import generate_test_grid, true_function
from usdr_plus.quantum.kernel import build_kernel_matrix as build_quantum_kernel_matrix


def _predict_grid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    theta: np.ndarray,
    tau: float,
    omega: float,
    classical_length_scale: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    x1, x2, _ = generate_test_grid(grid_size=60)
    X_grid = np.column_stack([x1.ravel(), x2.ravel()])
    X_grid_scaled = X_grid / (2 * np.pi)  # matches minmax preprocessing used in core pipeline
    K_train, K_grid = build_hybrid_train_test_kernels(
        X_train,
        X_grid_scaled,
        theta,
        omega,
        classical_length_scale=classical_length_scale,
    )
    alpha = np.linalg.solve(K_train + tau * np.eye(K_train.shape[0]), y_train)
    y_pred = (K_grid @ alpha).reshape(x1.shape)
    return x1, x2, y_pred


def run_hybrid_experiments(
    *,
    sample_sizes: list[int] | None = None,
    seeds: list[int] | None = None,
    normalize: str = cfg.NORMALIZE,
    base_path: Path = cfg.PREPROCESSED_DIR,
    csv_out: Path = cfg.CSV_PATH,
    progress_cb: Callable[[int, int, int, int], None] | None = None,
) -> pd.DataFrame:
    sample_sizes = list(sample_sizes or cfg.SAMPLE_SIZES)
    seeds = list(seeds or cfg.SEEDS)
    rows: list[dict[str, Any]] = []
    omega_rows: list[dict[str, Any]] = []
    total = len(sample_sizes) * len(seeds)
    done = 0
    classical_ref_data = load_processed_hybrid_dataset(
        base_path=base_path,
        N=max(sample_sizes),
        seed=seeds[0],
        normalize=normalize,
    )
    classical_search = grid_search_classical_rbf_krr(
        X_train=classical_ref_data["X_train"],
        y_train=classical_ref_data["y_train"],
        X_val=classical_ref_data["X_val"],
        y_val=classical_ref_data["y_val"],
        length_scales=cfg.CLASSICAL_LENGTH_SCALES_GRID,
        tau_grid=cfg.CLASSICAL_TAU_GRID,
    )
    classical_cfg = {"length_scale": float(classical_search.best_config.length_scale)}

    for n in sample_sizes:
        for seed in seeds:
            data = load_processed_hybrid_dataset(base_path=base_path, N=n, seed=seed, normalize=normalize)
            X_train, y_train = data["X_train"], data["y_train"]
            X_val, y_val = data["X_val"], data["y_val"]
            X_test, y_test = data["X_test"], data["y_test"]

            theta_opt, tau_opt, _ = optimize_theta_tau(X_train, y_train, X_val, y_val)

            # choose best omega on validation and keep full omega diagnostics
            best = None
            omega_metrics: dict[float, dict[str, float]] = {}
            for omega in cfg.OMEGA_GRID:
                K_train_h, K_val_h = build_hybrid_train_test_kernels(
                    X_train,
                    X_val,
                    theta_opt,
                    omega,
                    classical_length_scale=classical_cfg["length_scale"],
                )
                _, K_test_h_tmp = build_hybrid_train_test_kernels(
                    X_train,
                    X_test,
                    theta_opt,
                    omega,
                    classical_length_scale=classical_cfg["length_scale"],
                )
                alpha = np.linalg.solve(K_train_h + tau_opt * np.eye(K_train_h.shape[0]), y_train)
                y_val_pred = K_val_h @ alpha
                y_test_pred_tmp = K_test_h_tmp @ alpha
                val_mse = float(mean_squared_error(y_val, y_val_pred))
                test_mse_tmp = float(mean_squared_error(y_test, y_test_pred_tmp))
                kappa_raw = float(np.linalg.cond(K_train_h))
                kappa_reg = float(np.linalg.cond(K_train_h + tau_opt * np.eye(K_train_h.shape[0])))
                rank_eff = float(_effective_rank(K_train_h))
                omega_metrics[float(omega)] = {
                    "val_mse": val_mse,
                    "test_mse": test_mse_tmp,
                    "kappa_raw": kappa_raw,
                    "kappa_reg": kappa_reg,
                    "effective_rank": rank_eff,
                }
                if best is None or val_mse < best["val_mse"]:
                    best = {"omega": omega, "val_mse": val_mse}

            omega_star = float(best["omega"])
            K_train_h, K_test_h = build_hybrid_train_test_kernels(
                X_train,
                X_test,
                theta_opt,
                omega_star,
                classical_length_scale=classical_cfg["length_scale"],
            )
            alpha = np.linalg.solve(K_train_h + tau_opt * np.eye(K_train_h.shape[0]), y_train)
            y_test_pred = K_test_h @ alpha
            test_mse = float(mean_squared_error(y_test, y_test_pred))

            # save diagnostics for each run in single diagnostics folder
            save_diagnostics(
                K_train_h,
                theta=theta_opt,
                x_example=X_train[0],
                N=n,
                seed=seed,
                diagnostics_dir=cfg.DIAGNOSTICS_DIR,
            )
            run_dir = cfg.FIGURES_DIR / f"N{n}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            geo_dir = cfg.GEOMETRY_DIR / f"N{n}_seed{seed}"
            geo_dir.mkdir(parents=True, exist_ok=True)

            gx1, gx2, y_grid_pred = _predict_grid(
                X_train,
                y_train,
                theta_opt,
                tau_opt,
                omega_star,
                classical_cfg["length_scale"],
            )
            save_prediction_surface(
                y_pred_grid=y_grid_pred,
                N=n,
                seed=seed,
                output_dir=run_dir,
                omega_label=f"omega*={omega_star:.2f}",
                filename="prediction_surface.png",
            )

            plot_hybrid_1d_slices(
                data_dict=data,
                theta_opt=theta_opt,
                tau_star=tau_opt,
                classical_config=classical_cfg,
                omega_hybrid=omega_star,
                N=n,
                SEED=seed,
                output_dir=run_dir,
                show=False,
                grid_size=200,
                dpi=300,
                width=28.0,
                height=12.0,
            )

            smooth_true = np.sin(gx1) + np.cos(gx2)
            residual_grid = y_grid_pred - smooth_true
            save_interaction_ridge(
                x1=gx1,
                x2=gx2,
                residual=residual_grid,
                N=n,
                seed=seed,
                output_dir=run_dir,
                width=16.0,
                height=12.0,
                dpi=300,
            )
            save_true_vs_pred(y_true=y_test, y_pred=y_test_pred, N=n, seed=seed, output_dir=run_dir)

            K_train_h, K_val_h = build_hybrid_train_test_kernels(
                X_train,
                X_val,
                theta_opt,
                omega_star,
                classical_length_scale=classical_cfg["length_scale"],
            )
            alpha_r = np.linalg.solve(K_train_h + tau_opt * np.eye(K_train_h.shape[0]), y_train)
            y_train_pred = K_train_h @ alpha_r
            y_val_pred = K_val_h @ alpha_r
            r_train = y_train - y_train_pred
            r_val = y_val - y_val_pred
            r_test = y_test - y_test_pred
            save_residual_violin(
                r_train=r_train,
                r_val=r_val,
                r_test=r_test,
                N=n,
                seed=seed,
                output_dir=run_dir,
            )

            # Geometry plots for all N/seed
            Kq_train = build_quantum_kernel_matrix(X_train, X_train, theta_opt)
            Kc_train = build_classical_rbf_kernel(
                X_train,
                X_train,
                sigma=classical_cfg["length_scale"],
            )
            Ktilde_Q = normalize_gram(Kq_train)
            Ktilde_C = normalize_gram(Kc_train)
            hybrid_mats: dict[float, np.ndarray] = {}
            for w in [0.25, 0.5, 0.75, 1.0]:
                K_h, _ = build_hybrid_train_test_kernels(
                    X_train,
                    X_train,
                    theta_opt,
                    float(w),
                    classical_length_scale=classical_cfg["length_scale"],
                )
                hybrid_mats[float(w)] = K_h
            plot_kernel_geometry_heatmaps_case(
                Ktilde_C_train=Ktilde_C,
                Ktilde_Q_train=Ktilde_Q,
                hybrid_mats=hybrid_mats,
                omegas_to_show=[0.25, 0.5, 0.75],
                N=n,
                seed=seed,
                output_dir=geo_dir,
                width=12.0,
                height=12.0,
                dpi=300,
            )
            plot_eigenvalue_spectra_for_case(
                Ktilde_C_train=Ktilde_C,
                Ktilde_Q_train=Ktilde_Q,
                hybrid_mats=hybrid_mats,
                omegas_to_show=[0.25, 0.5, 0.75, 1.0],
                title=f"Eigenvalue spectra N={n} SEED={seed}",
                output_dir=geo_dir,
                width=32.0,
                height=18.0,
                dpi=300,
            )

            # collect omega rows
            for omega, m in omega_metrics.items():
                omega_rows.append(
                    {
                        "N": int(n),
                        "SEED": int(seed),
                        "omega": float(omega),
                        "val_mse": float(m["val_mse"]),
                        "test_mse": float(m["test_mse"]),
                        "effective_rank": float(m["effective_rank"]),
                        "kappa_raw": float(m["kappa_raw"]),
                        "kappa_reg": float(m["kappa_reg"]),
                        "is_best": int(abs(float(omega) - omega_star) < 1e-12),
                    }
                )

            rows.append(
                {
                    "dataset": "hybrid",
                    "experiment": "usdr_plus_hybrid_constrained",
                    "normalize": normalize,
                    "N": int(n),
                    "SEED": int(seed),
                    "lambda1": float(theta_opt[0]),
                    "lambda2": float(theta_opt[1]),
                    "gamma": float(theta_opt[2]),
                    "beta": float(theta_opt[3]),
                    "tau": float(tau_opt),
                    "omega_star": float(omega_star),
                    "val_mse": float(best["val_mse"]),
                    "test_mse": float(test_mse),
                    "test_mse_omega0": float(omega_metrics[0.0]["test_mse"]),
                    "test_mse_omega1": float(omega_metrics[1.0]["test_mse"]),
                }
            )

            done += 1
            if progress_cb is not None:
                progress_cb(done, total, n, seed)

    df = pd.DataFrame(rows)
    df_omega = pd.DataFrame(omega_rows)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)
    omega_csv = csv_out.with_name(csv_out.stem + "_omega_grid.csv")
    df_omega.to_csv(omega_csv, index=False)

    # Global hybrid plots
    cfg.COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    print("[HYBRID-PLOTS] Plotting summary bar charts per N (omega=0 classical vs omega=1 USDR+ vs omega* hybrid)...")
    plot_summary_bars_per_N(df, cfg.COMPARISON_DIR, dpi=300)
    print("[HYBRID-PLOTS] Scatter: test MSE vs effective rank / condition, colored by omega...")
    plot_mse_vs_rank_condition_scatter(df_omega, cfg.COMPARISON_DIR, dpi=300)

    print(f"[HYBRID] All runs completed. Results saved to: {csv_out}")
    return df
