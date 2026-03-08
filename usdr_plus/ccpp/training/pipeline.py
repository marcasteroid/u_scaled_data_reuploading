"""Training pipeline for modular CCPP USDR+ experiments."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

from usdr_plus.ccpp import config as cfg
from usdr_plus.ccpp.data.datasets import load_processed_ccpp_2d_dataset
from usdr_plus.ccpp.quantum.kernel import build_kernel_matrix
from usdr_plus.ccpp.visualization.plots import (
    plot_1d_slices_ccpp,
    plot_interaction_ridge_ccpp,
    plot_prediction_surface_ccpp,
    plot_residual_distributions_ccpp,
    plot_true_vs_predicted_ccpp,
    save_diagnostics,
)
from usdr_plus.analysis.spectrum import compute_spectrum_metrics
from usdr_plus.training.krr import apply_psd_hygiene


@dataclass
class OptimizerResult:
    theta_opt: np.ndarray
    tau_opt: float
    val_mse: float
    method: str
    success: bool
    message: str


def krr_val_objective(
    log_params: np.ndarray,
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    large_penalty: float = 1e9,
) -> float:
    """Validation objective in log-space for [lambda1, lambda2, gamma, beta, tau]."""
    p = np.asarray(log_params, dtype=float).ravel()
    if p.size != 5:
        raise ValueError(f"Expected 5 log parameters, got shape {p.shape}")

    lambda1, lambda2, gamma, beta, tau = np.exp(p)
    gamma = max(gamma, 1.0)
    tau = max(tau, 1e-12)
    theta = np.array([lambda1, lambda2, gamma, beta], dtype=float)

    try:
        K_train = build_kernel_matrix(X_train, X_train, theta, apply_psd_hygiene_for_square=True)
        K_val = build_kernel_matrix(X_val, X_train, theta, apply_psd_hygiene_for_square=False)
        K_reg = 0.5 * (K_train + K_train.T) + tau * np.eye(K_train.shape[0], dtype=float)
        alpha = np.linalg.solve(K_reg, np.asarray(y_train, dtype=float).ravel())
        pred = K_val @ alpha
        mse = float(np.mean((pred - np.asarray(y_val, dtype=float).ravel()) ** 2))
        return mse if np.isfinite(mse) else large_penalty
    except Exception:
        return large_penalty


def optimize_theta_tau(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> OptimizerResult:
    """L-BFGS-B constrained optimization with notebook-equivalent bounds."""
    b = cfg.THETA_BOUNDS
    bounds_log = [
        (np.log(b["lambda1"][0]), np.log(b["lambda1"][1])),
        (np.log(b["lambda2"][0]), np.log(b["lambda2"][1])),
        (np.log(b["gamma"][0]), np.log(b["gamma"][1])),
        (np.log(b["beta"][0]), np.log(b["beta"][1])),
        (np.log(b["tau"][0]), np.log(b["tau"][1])),
    ]
    x0 = np.log(np.array([1.0, 1.0, 2.0, 1.0, 1e-3], dtype=float))

    res = minimize(
        lambda p: krr_val_objective(p, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val),
        x0,
        method="L-BFGS-B",
        bounds=bounds_log,
        tol=cfg.OPT_TOL,
        options={"maxiter": cfg.OPT_MAXITER, "disp": False},
    )

    if res.success and np.isfinite(res.fun):
        lam1, lam2, gamma, beta, tau = np.exp(res.x)
        theta = np.array(
            [
                np.clip(lam1, *b["lambda1"]),
                np.clip(lam2, *b["lambda2"]),
                np.clip(gamma, *b["gamma"]),
                np.clip(beta, *b["beta"]),
            ],
            dtype=float,
        )
        tau = float(np.clip(tau, *b["tau"]))
        return OptimizerResult(
            theta_opt=theta,
            tau_opt=tau,
            val_mse=float(res.fun),
            method="lbfgs",
            success=True,
            message=str(res.message),
        )

    # Match notebook fallback behavior
    def objective_optuna(trial: optuna.trial.Trial) -> float:
        lambda1 = trial.suggest_float("lambda1", *b["lambda1"])
        lambda2 = trial.suggest_float("lambda2", *b["lambda2"])
        gamma = trial.suggest_float("gamma", *b["gamma"])
        beta = trial.suggest_float("beta", *b["beta"])
        tau = trial.suggest_float("tau", *b["tau"], log=True)
        p = np.log([lambda1, lambda2, gamma, beta, tau])
        return krr_val_objective(p, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_optuna, n_trials=100, show_progress_bar=False)
    bp = study.best_params
    theta = np.array([bp["lambda1"], bp["lambda2"], bp["gamma"], bp["beta"]], dtype=float)
    tau = float(bp["tau"])
    return OptimizerResult(
        theta_opt=theta,
        tau_opt=tau,
        val_mse=float(study.best_value),
        method="optuna",
        success=True,
        message=f"lbfgs_failed: {res.message}",
    )


def run_constrained_experiments(
    *,
    sample_sizes: list[int] | None = None,
    seeds: list[int] | None = None,
    normalize: str = cfg.NORMALIZE,
    base_path: Path = cfg.PREPROCESSED_DIR,
    csv_out: Path = cfg.CSV_PATH,
    progress_cb: Callable[[int, int, int, int], None] | None = None,
) -> pd.DataFrame:
    """Run CCPP training/eval loop and persist CSV metrics."""
    sample_sizes = list(sample_sizes or cfg.SAMPLE_SIZES)
    seeds = list(seeds or cfg.SEEDS)
    rows: list[dict[str, Any]] = []
    total = len(sample_sizes) * len(seeds)
    done = 0

    for n in sample_sizes:
        for seed in seeds:
            print("\n" + "=" * 80)
            print(f"[CCPP-USDR+] N={n}, SEED={seed}")
            print("=" * 80)

            data = load_processed_ccpp_2d_dataset(base_path=base_path, N=n, seed=seed, normalize=normalize)
            X_train, y_train = data["X_train"], data["y_train"]
            X_val, y_val = data["X_val"], data["y_val"]
            X_test, y_test = data["X_test"], data["y_test"]
            meta = data.get("metadata", {})

            opt = optimize_theta_tau(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
            print(
                "[CCPP-USDR+] Optimized hyperparameters: "
                f"lambda1={opt.theta_opt[0]:.3f}, lambda2={opt.theta_opt[1]:.3f}, "
                f"gamma={opt.theta_opt[2]:.3f}, beta={opt.theta_opt[3]:.3f}, tau={opt.tau_opt:.3e}, "
                f"val_mse={opt.val_mse:.4e}"
            )
            print(f"[CCPP-USDR+] Optimizer info: success={opt.success}, message={opt.message}")

            K_train = build_kernel_matrix(X_train, X_train, opt.theta_opt, apply_psd_hygiene_for_square=True)
            K_test = build_kernel_matrix(X_test, X_train, opt.theta_opt, apply_psd_hygiene_for_square=False)
            run_fig_dir = cfg.FIGURES_DIR / f"N{n}_seed{seed}"
            run_fig_dir.mkdir(parents=True, exist_ok=True)

            save_diagnostics(
                K_train,
                x_example=X_train[0],
                theta=opt.theta_opt,
                run_tag=f"n{n}_seed{seed}",
            )

            # Save per-run plots from the notebook's CCPP "Plots" section.
            plot_prediction_surface_ccpp(
                data=data,
                theta=opt.theta_opt,
                tau=opt.tau_opt,
                run_dir=run_fig_dir,
                N=n,
                seed=seed,
                show=False,
            )
            plot_1d_slices_ccpp(
                data=data,
                theta=opt.theta_opt,
                tau=opt.tau_opt,
                run_dir=run_fig_dir,
                N=n,
                seed=seed,
                show=False,
            )
            plot_interaction_ridge_ccpp(
                data=data,
                theta=opt.theta_opt,
                tau=opt.tau_opt,
                run_dir=run_fig_dir,
                N=n,
                seed=seed,
                width=16.0,
                height=12.0,
                dpi=300,
                show=False,
            )
            plot_true_vs_predicted_ccpp(
                data=data,
                theta=opt.theta_opt,
                tau=opt.tau_opt,
                run_dir=run_fig_dir,
                N=n,
                seed=seed,
                show=False,
            )
            plot_residual_distributions_ccpp(
                data=data,
                theta=opt.theta_opt,
                tau=opt.tau_opt,
                run_dir=run_fig_dir,
                N=n,
                seed=seed,
                show=False,
            )
            print(f"[PLOTS-CCPP] Saved plots to: {run_fig_dir}")

            spec_train = compute_spectrum_metrics(K_train, name=f"K_train (N={n}, seed={seed})", log_prefix="[SPEC-CCPP-RAW]")
            K_reg, jitter, _, _ = apply_psd_hygiene(K_train, tau=opt.tau_opt, name=f"K_train (N={n}, seed={seed})")
            spec_reg = compute_spectrum_metrics(K_reg, name=f"K_reg (N={n}, seed={seed})", log_prefix="[SPEC-CCPP-REG]")

            alpha = np.linalg.solve(K_reg, y_train)
            y_pred_test = K_test @ alpha
            test_mse = float(mean_squared_error(y_test, y_pred_test))
            print(f"[CCPP-USDR+] test_mse={test_mse:.4e}, kappa_reg={spec_reg['kappa']:.3e}")

            rows.append(
                {
                    "dataset": "ccpp",
                    "experiment": "usdr_plus_constrained_ccpp_2d",
                    "normalize": normalize,
                    "N": int(n),
                    "SEED": int(seed),
                    "n_train": int(meta.get("n_train", len(X_train))),
                    "n_val": int(meta.get("n_val", len(X_val))),
                    "n_test": int(meta.get("n_test", len(X_test))),
                    "lambda1": float(opt.theta_opt[0]),
                    "lambda2": float(opt.theta_opt[1]),
                    "gamma": float(opt.theta_opt[2]),
                    "beta": float(opt.theta_opt[3]),
                    "tau": float(opt.tau_opt),
                    "val_mse": float(opt.val_mse),
                    "test_mse": test_mse,
                    "min_eig_train": float(spec_train["min_eig"]),
                    "max_eig_train": float(spec_train["max_eig"]),
                    "kappa_train": float(spec_train["kappa"]),
                    "rank_eff_train": float(spec_train["rank_eff"]),
                    "min_eig_reg": float(spec_reg["min_eig"]),
                    "max_eig_reg": float(spec_reg["max_eig"]),
                    "kappa_reg": float(spec_reg["kappa"]),
                    "rank_eff_reg": float(spec_reg["rank_eff"]),
                    "jitter": float(jitter),
                }
            )

            done += 1
            if progress_cb is not None:
                with contextlib.suppress(Exception):
                    progress_cb(done, total, n, seed)

    df = pd.DataFrame(rows)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)
    print(f"[CCPP-USDR+] All runs completed. Results saved to: {csv_out}")
    return df
