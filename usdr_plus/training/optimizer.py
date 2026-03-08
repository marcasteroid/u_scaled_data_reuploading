"""
usdr_plus/training/optimizer.py
=================================
Joint optimisation of θ = (λ₁, λ₂, γ, β) and τ for Kernel Ridge Regression.

Strategy
--------
• Primary  : L-BFGS-B in log-space with explicit box constraints.
• Fallback : Optuna TPE sampler (100 trials) with the same prior ranges.
"""

import numpy as np
import optuna
from optuna.samplers import TPESampler
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

from usdr_plus.quantum.kernel import build_kernel_matrix


# ---------------------------------------------------------------------------
# Validation objective
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Joint optimisation
# ---------------------------------------------------------------------------


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
