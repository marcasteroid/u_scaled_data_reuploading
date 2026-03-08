#!/usr/bin/env python
"""
Auto-generated from /Users/marco/Desktop/ccpp.ipynb.
Notebook execution order is preserved cell-by-cell.
"""

from __future__ import annotations

import logging
try:
    from IPython.display import display
except Exception:  # pragma: no cover
    def display(obj):
        print(obj)


# ===== CELL 000 (markdown) =====
# # $U_{\text{SDR}+}$ with Hyperparameter Bounds on Combined Cycle Power Plant (CCPP) (AT, V → EP)
# 
# - Dataset: Combined Cycle Power Plant (UCI).
# - 2D subproblem:
#   - $x_1$ = AT (ambient temperature, °C)
#   - $x_2$ = V (exhaust vacuum, cm Hg)
# - y = EP (net hourly electrical energy output, MW)
# - Goal: apply USDR+ + KRR in the **small-N regime** (e.g. $N \in \{50,100,200\}$) and study:
# - Generalisation (val/test MSE),
# - Spectral diagnostics of the Gram matrix (κ, eigenvalues, effective rank),
# - Behaviour of prediction surfaces / slices and residuals.

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
from typing import Dict, Tuple, List, Optional, Any
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
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error


# === PARALLEL and CACHING ===
from joblib import Memory, Parallel, delayed

# === PARALLEL and CACHING ===
from matplotlib import animation
from IPython.display import HTML

# ===== CELL 002 (code) =====
# [notebook-only skipped] %pip install openpyxl

# ===== CELL 003 (markdown) =====
# **Environment hygiene (Colab)**

# ===== CELL 004 (code) =====
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ===== CELL 005 (markdown) =====
# **Disk cache setup**

# ===== CELL 006 (code) =====
CACHE_DIR = Path("cache/ccpp")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

memory = Memory(location=CACHE_DIR, verbose=1, mmap_mode='r')  # Read-only for safety
print(f"[CACHE] Initialized at: {CACHE_DIR.resolve()}")

# ===== CELL 007 (markdown) =====
# **Configuration parameters**

# ===== CELL 008 (code) =====
# === EXPERIMENTAL CONFIGURATION (USDR+ ON CCPP 2D SUBPROBLEM) ===

# 2D inputs for USDR+:
#   x1 = AT  (ambient temperature, °C)
#   x2 = V   (exhaust vacuum, cm Hg)
# target:
#   y  = EP  (net hourly electrical energy output, MW)

# Empirical raw domain for (AT, V).
# These will be filled AFTER loading the CCPP dataset, e.g.:
#   RAW_DOMAIN_AT = (AT_min, AT_max)
#   RAW_DOMAIN_V  = (V_min, V_max)
RAW_DOMAIN_AT = None   # placeholder, set later from data
RAW_DOMAIN_V  = None   # placeholder, set later from data

sample_sizes = [50, 100, 200]      # N ∈ {50, 100, 200}
grid_size    = 60                  # 60×60 visualization grid
SEEDS        = [0, 1, 2]           # 3 independent subsampling seeds

# Output directory for processed splits and results
OUTPUT_DIR = Path("preprocessed/ccpp")
BASE_PATH  = OUTPUT_DIR            # keep a single canonical base path
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Preprocessing strategy for (AT, V):
#   - raw x = (AT, V) in physical units
#   - if NORMALIZE == "minmax":
#         fit MinMaxScaler on train only → x̃ ∈ [0,1]^2
#   - if NORMALIZE == "zscore":
#         fit StandardScaler on train only → x̃ standardized
# The circuit then uses x̂ = x̃ / β internally.
NORMALIZE = "minmax"               # "minmax" or "zscore"

# USDR+ circuit configuration (unchanged)
depth      = 2                     # Fixed L = 2
entangler  = "cnot"                # Fixed for USDR+
axes_low   = ("X", "Z")            # Low-freq block
axes_high  = ("Z", "X")            # High-freq block

# === HYPERPARAMETERS TO OPTIMIZE (θ) ===
theta_bounds = {
    "lambda1": (0.1, 10.0),
    "lambda2": (0.1, 10.0),
    "gamma":   (1.0,  5.0),        # γ ≥ 1
    "beta":    (0.1, 10.0),
}

def set_all_seeds(seed: int) -> None:
    """Set deterministic seeds for NumPy and Python's random module."""
    random.seed(seed)
    np.random.seed(seed)

print(
    "[CONFIG] USDR+ CCPP 2D protocol loaded. "
    f"SEEDS={SEEDS}, N={sample_sizes}, L={depth}, NORMALIZE={NORMALIZE}"
)

# ===== CELL 009 (markdown) =====
# **True function**

# ===== CELL 010 (code) =====
# === TRUE FUNCTION (CCPP VERSION) ===========================================
#
# In the synthetic experiment we had an analytic "true_function(x1, x2)".
# For the CCPP dataset, the mapping
#
#     (AT, V)  →  EP
#
# is *not* available in closed form: we only observe it through data.
#

def true_function(*args, **kwargs):
    """
    Placeholder for the 'true function' in the CCPP experiment.

    In the synthetic 2D setup, this returned the noiseless function value
    f(x1, x2). For the real CCPP dataset, there is no analytic ground truth,
    so any code relying on true_function must be adapted to work directly
    with the dataset (e.g., visualizing USDR+ predictions vs data).

    Calling this function will raise a RuntimeError on purpose.
    """
    raise RuntimeError(
        "No analytic 'true_function' for the CCPP dataset. "
        "Use data-driven plots (predictions vs EP) instead."
    )

# ===== CELL 011 (markdown) =====
# **Dataset generation**

# ===== CELL 012 (code) =====
# -----------------------------------------------------------------------------
# 1. Loader for the CCPP dataset (Excel)
# -----------------------------------------------------------------------------

def load_ccpp_dataset(
    path: str | Path = "ccpp_dataset.xlsx",
) -> pd.DataFrame:
    """
    Load the Combined Cycle Power Plant (CCPP) dataset from an Excel file
    and normalise column names.

    Expected logical columns (case-insensitive, common aliases handled):
      - AT : Ambient Temperature (°C)
      - V  : Exhaust Vacuum (cm Hg)
      - AP : Ambient Pressure (millibar)
      - RH : Relative Humidity (%)
      - EP or PE : Net hourly electrical energy output (MW)

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns: ['AT', 'V', 'AP', 'RH', 'EP'].
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"[CCPP] Dataset file not found: {path}")

    # Read the Excel file
    df = pd.read_excel(path)

    # Normalise column names (lowercase, strip spaces)
    col_map = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=col_map)

    # Map common aliases to canonical names
    rename_map = {}

    # Ambient Temperature
    for c in df.columns:
        if c in {"at", "temperature"}:
            rename_map[c] = "AT"
    # Exhaust Vacuum
    for c in df.columns:
        if c in {"v", "vacuum"}:
            rename_map[c] = "V"
    # Ambient Pressure
    for c in df.columns:
        if c in {"ap", "pressure"}:
            rename_map[c] = "AP"
    # Relative Humidity
    for c in df.columns:
        if c in {"rh", "humidity"}:
            rename_map[c] = "RH"
    # Power Output: EP or PE
    for c in df.columns:
        if c in {"ep", "pe", "power", "output"}:
            rename_map[c] = "EP"

    df = df.rename(columns=rename_map)

    required = {"AT", "V", "AP", "RH", "EP"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"[CCPP] Missing expected columns after renaming: {missing}. "
            f"Got columns: {list(df.columns)}"
        )

    # Keep only the relevant columns in a canonical order
    df = df[["AT", "V", "AP", "RH", "EP"]].copy()

    print(
        f"[CCPP] Loaded dataset from {path} with shape {df.shape}: "
        f"{len(df)} rows, columns={list(df.columns)}"
    )
    return df


# -----------------------------------------------------------------------------
# 2. Build 2D subproblem (x1 = AT, x2 = V, y = EP) and set RAW_DOMAIN
# -----------------------------------------------------------------------------

def build_ccpp_2d_view(
    df: pd.DataFrame,
    trim_quantiles: tuple[float, float] = (0.0, 1.0),
) -> dict:
    """
    Construct the 2D regression subproblem:
        x1 = AT, x2 = V, y = EP

    Optionally trims the dataset to a central region by quantiles on AT and V.

    Parameters
    ----------
    df : pd.DataFrame
        Full CCPP DataFrame with columns ['AT', 'V', 'AP', 'RH', 'EP'].
    trim_quantiles : (float, float), optional
        Lower and upper quantiles for trimming AT and V, e.g. (0.05, 0.95).
        Use (0.0, 1.0) to disable trimming.

    Returns
    -------
    data_2d : dict
        {
          "X_full_raw": np.ndarray shape (M, 2),  # (AT, V) in physical units
          "y_full":     np.ndarray shape (M,),    # EP
          "AT_domain":  (AT_min, AT_max),
          "V_domain":   (V_min, V_max),
          "df_2d":      pd.DataFrame with columns ["AT", "V", "EP"],
        }
    """
    q_low, q_high = trim_quantiles
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError(
            "[CCPP] trim_quantiles must satisfy 0 <= q_low < q_high <= 1."
        )

    df_2d = df[["AT", "V", "EP"]].copy()

    # Optional trimming to central region
    if (q_low, q_high) != (0.0, 1.0):
        at_low, at_high = df_2d["AT"].quantile([q_low, q_high]).values
        v_low,  v_high  = df_2d["V"].quantile([q_low, q_high]).values

        mask = (
            (df_2d["AT"] >= at_low) & (df_2d["AT"] <= at_high) &
            (df_2d["V"]  >= v_low)  & (df_2d["V"]  <= v_high)
        )
        df_2d = df_2d[mask].reset_index(drop=True)
        print(
            f"[CCPP] Trimmed to quantiles [{q_low}, {q_high}] on (AT, V): "
            f"{len(df_2d)} rows remaining."
        )
    else:
        print("[CCPP] No trimming applied to (AT, V).")

    # Extract raw 2D inputs and target
    X_full_raw = df_2d[["AT", "V"]].values.astype(np.float64)
    y_full     = df_2d["EP"].values.astype(np.float64)

    AT_min, AT_max = float(df_2d["AT"].min()), float(df_2d["AT"].max())
    V_min,  V_max  = float(df_2d["V"].min()),  float(df_2d["V"].max())

    print(
        f"[CCPP] 2D view: X_full_raw shape={X_full_raw.shape}, "
        f"y_full shape={y_full.shape}"
    )
    print(
        f"[CCPP] AT domain ≈ [{AT_min:.3f}, {AT_max:.3f}], "
        f"V domain ≈ [{V_min:.3f}, {V_max:.3f}]"
    )

    return {
        "X_full_raw": X_full_raw,
        "y_full":     y_full,
        "AT_domain":  (AT_min, AT_max),
        "V_domain":   (V_min, V_max),
        "df_2d":      df_2d,
    }


# -----------------------------------------------------------------------------
# 3. Helper: save a split to CSV (2D: x1 = AT, x2 = V, y = EP)
# -----------------------------------------------------------------------------

def save_ccpp_split(folder: Path, name: str, X: np.ndarray, y: np.ndarray) -> None:
    """
    Save a split (train/val/test) to CSV with columns x1, x2, y.

    X is assumed to be already preprocessed (e.g. normalized for USDR+).
    """
    if X.shape[1] != 2:
        raise ValueError(
            f"[CCPP] Expected X with 2 columns (AT, V), got shape {X.shape}"
        )

    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["y"] = y
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / f"{name}.csv", index=False)


# -----------------------------------------------------------------------------
# 4. Preprocess & save small-N 2D CCPP splits for USDR+ (AT, V → EP)
# -----------------------------------------------------------------------------

def preprocess_and_save_ccpp_2d_datasets(
    X_full_raw: np.ndarray,
    y_full: np.ndarray,
    sample_sizes: list[int],
    seeds: list[int],
    output_dir: str | Path = OUTPUT_DIR,
    normalize: str = NORMALIZE,
) -> None:
    """
    For each (N, seed), build a small-N 2D regression dataset (AT, V → EP)
    from the full CCPP data, apply preprocessing, and save splits to CSV.

    Protocol per (N, seed):
      1) Randomly subsample N points from (X_full_raw, y_full).
      2) 70/15/15 train/val/test split.
      3) Fit scaler on train inputs (2D) according to `normalize`:
           - "minmax": MinMaxScaler → X ∈ [0,1]^2
           - "zscore": StandardScaler → X standardized
      4) Transform val/test.
      5) Save preprocessed splits in:
           output_dir / f"N{N}_seed{seed}" / {train,val,test}.csv
         with columns [x1, x2, y].

    Parameters
    ----------
    X_full_raw : np.ndarray, shape (M, 2)
        Full 2D inputs (AT, V) in physical units.
    y_full : np.ndarray, shape (M,)
        Corresponding targets EP.
    sample_sizes : list of int
        List of N values, e.g. [50, 100, 200].
    seeds : list of int
        List of seeds for reproducible subsampling, e.g. [0, 1, 2].
    output_dir : str or Path
        Root directory where processed splits will be saved.
    normalize : {"minmax", "zscore"}
        Preprocessing mode for inputs.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    M = X_full_raw.shape[0]
    if y_full.shape[0] != M:
        raise ValueError(
            f"[CCPP] X_full_raw and y_full have inconsistent lengths: "
            f"{M} vs {y_full.shape[0]}"
        )

    for N in sample_sizes:
        if N > M:
            raise ValueError(
                f"[CCPP] Requested N={N} but only {M} samples available."
            )

        for seed in seeds:
            set_all_seeds(seed)

            # 1) Subsample N points
            idx = np.random.permutation(M)[:N]
            X_raw = X_full_raw[idx]
            y     = y_full[idx]

            # 2) 70/15/15 split
            n_train = int(0.70 * N)
            n_val   = int(0.15 * N)
            n_test  = N - n_train - n_val

            perm = np.random.permutation(N)
            train_idx = perm[:n_train]
            val_idx   = perm[n_train:n_train + n_val]
            test_idx  = perm[n_train + n_val:]

            X_train_raw, y_train = X_raw[train_idx], y[train_idx]
            X_val_raw,   y_val   = X_raw[val_idx],   y[val_idx]
            X_test_raw,  y_test  = X_raw[test_idx],  y[test_idx]

            # 3) Normalization (train-only, 2D)
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
                raise ValueError(f"[CCPP] Unknown NORMALIZE mode: {normalize}")

            # 4) Save preprocessed splits
            folder = output_dir_path / f"N{N}_seed{seed}"
            save_ccpp_split(folder, "train", X_train, y_train)
            save_ccpp_split(folder, "val",   X_val,   y_val)
            save_ccpp_split(folder, "test",  X_test,  y_test)

            print(
                f"[CCPP] Saved 2D splits (AT, V → EP) for N={N}, seed={seed} → "
                f"{n_train}/{n_val}/{n_test} train/val/test "
                f"(NORMALIZE={normalize}) at {folder}"
            )


# -----------------------------------------------------------------------------
# 5. Run the CCPP 2D preprocessing pipeline once
# -----------------------------------------------------------------------------

# 5.1 Load full dataset
df_ccpp = load_ccpp_dataset("ccpp_dataset.xlsx")

# 5.2 Build 2D view (optionally trim extremes)
ccpp_2d = build_ccpp_2d_view(
    df_ccpp,
    trim_quantiles=(0.0, 1.0),   # or e.g. (0.05, 0.95) to remove extremes
)

X_full_raw = ccpp_2d["X_full_raw"]
y_full     = ccpp_2d["y_full"]

# 5.3 Set RAW_DOMAIN_AT and RAW_DOMAIN_V from the data (for plotting grids later)
AT_min, AT_max = ccpp_2d["AT_domain"]
V_min,  V_max  = ccpp_2d["V_domain"]

RAW_DOMAIN_AT = (AT_min, AT_max)
RAW_DOMAIN_V  = (V_min,  V_max)

print(f"[CONFIG] RAW_DOMAIN_AT set to {RAW_DOMAIN_AT}")
print(f"[CONFIG] RAW_DOMAIN_V  set to {RAW_DOMAIN_V}")

# 5.4 Preprocess and save small-N datasets for USDR+ (AT, V → EP)
preprocess_and_save_ccpp_2d_datasets(
    X_full_raw=X_full_raw,
    y_full=y_full,
    sample_sizes=sample_sizes,
    seeds=SEEDS,
    output_dir=OUTPUT_DIR,
    normalize=NORMALIZE,
)

# ===== CELL 013 (markdown) =====
# **Visualization grid**

# ===== CELL 014 (code) =====
RAW_DOMAIN_2D = (RAW_DOMAIN_AT, RAW_DOMAIN_V)

def generate_ccpp_grid(
    grid_size: int = 60,
    domain_2d=RAW_DOMAIN_2D,
):
    """
    Generate a 2D evaluation grid over the raw CCPP domain
    for (AT, V).

    Parameters
    ----------
    grid_size : int
        Number of points per axis (total grid_size × grid_size points).
    domain_2d : ((float, float), (float, float))
        Pair of intervals:
          - domain_2d[0] = (AT_min, AT_max)
          - domain_2d[1] = (V_min,  V_max)

    Returns
    -------
    X1, X2 : np.ndarray of shape (grid_size, grid_size)
        Meshgrid coordinates:
          - X1: AT values
          - X2: V values

    Notes
    -----
    • For CCPP there is no analytic “true surface” EP(AT, V), so this
      helper **does not** return Y_true.
    • Use (X1, X2) to build a grid of raw inputs, preprocess them with
      the same scaler used for training, and then feed them to the
      USDR⁺ + KRR predictor to plot the learned surface.
    """
    (at_min, at_max), (v_min, v_max) = domain_2d

    at_vals = np.linspace(at_min, at_max, grid_size)
    v_vals  = np.linspace(v_min,  v_max,  grid_size)

    X1, X2 = np.meshgrid(at_vals, v_vals)
    return X1, X2

# ===== CELL 015 (markdown) =====
# **Utilities**

# ===== CELL 016 (code) =====
def plot_ccpp_2d_dataset(
    X_full_raw: np.ndarray,
    y_full: np.ndarray,
    *,
    width: float = 14.0,
    height: float = 6.0,
    dpi: int = 300,
    theme: str = "whitegrid",
    max_points: int | None = 2000,
) -> None:
    """
    CCPP-ready visualization of the 2D subproblem:

        x1 = AT (ambient temperature)
        x2 = V  (exhaust vacuum)
        y  = EP (net hourly electrical energy output)

    This replaces the old `plot_2d_dataset` used for the analytic toy
    function. For CCPP there is no analytic true surface EP(AT,V), so
    we show:

      • Left panel: 3D scatter of (AT, V, EP)
      • Right panel: 2D scatter of (AT, V) coloured by EP

    Parameters
    ----------
    X_full_raw : np.ndarray of shape (N_all, 2)
        Raw 2D inputs with columns [AT, V] in physical units.
    y_full : np.ndarray of shape (N_all,)
        Targets, EP in MW.
    width, height, dpi : float
        Figure size and resolution.
    theme : str
        seaborn theme (e.g. 'whitegrid', 'darkgrid').
    max_points : int or None
        If not None and N_all > max_points, randomly subsample this
        many points for plotting, to keep the 3D scatter readable.
    """
    assert X_full_raw.shape[1] == 2, f"Expected X_full_raw shape (N,2), got {X_full_raw.shape}"
    assert X_full_raw.shape[0] == y_full.shape[0], "X_full_raw and y_full must have same number of rows"

    # Optional subsampling for readability
    N_all = X_full_raw.shape[0]
    if (max_points is not None) and (N_all > max_points):
        idx = np.random.permutation(N_all)[:max_points]
        X_plot = X_full_raw[idx]
        y_plot = y_full[idx]
        print(f"[CCPP-PLOT] Subsampled {max_points} / {N_all} points for plotting.")
    else:
        X_plot = X_full_raw
        y_plot = y_full

    at = X_plot[:, 0]  # AT
    v  = X_plot[:, 1]  # V

    sns.set_theme(style=theme, context="talk")
    fig = plt.figure(figsize=(width, height), dpi=dpi)

    # ---- 1. 3D scatter: (AT, V, EP) -------------------------------------
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    sc3d = ax1.scatter(
        at, v, y_plot,
        c=y_plot,
        cmap="viridis",
        s=30,
        edgecolor="k",
        linewidth=0.4,
        alpha=0.85,
    )
    ax1.set_title("CCPP raw data: 3D view", fontsize=14, pad=12)
    ax1.set_xlabel("AT (°C)")
    ax1.set_ylabel("V (cm Hg)")
    ax1.set_zlabel("EP (MW)")
    fig.colorbar(sc3d, ax=ax1, shrink=0.6, aspect=12, label="EP (MW)")

    # ---- 2. 2D scatter with colour = EP ---------------------------------
    ax2 = fig.add_subplot(1, 2, 2)
    sc2d = ax2.scatter(
        at, v,
        c=y_plot,
        cmap="viridis",
        s=30,
        edgecolor="k",
        linewidth=0.4,
        alpha=0.85,
    )
    ax2.set_title("CCPP raw data: EP over (AT, V)", fontsize=14, pad=12)
    ax2.set_xlabel("AT (°C)")
    ax2.set_ylabel("V (cm Hg)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    fig.colorbar(sc2d, ax=ax2, shrink=0.8, aspect=20, label="EP (MW)")

    plt.tight_layout()
    plt.show()

# ===== CELL 017 (markdown) =====
# **Plot results**

# ===== CELL 018 (code) =====
X_full_raw = df_ccpp[["AT", "V"]].values.astype(np.float64)
y_full     = df_ccpp["EP"].values.astype(np.float64)

plot_ccpp_2d_dataset(
    X_full_raw=X_full_raw,
    y_full=y_full,
    width=32,
    height=18,
    dpi=300,
    max_points=2000,
)

# ===== CELL 019 (markdown) =====
# **Plot all resolutions**

# ===== CELL 020 (code) =====
def plot_ccpp_all_resolutions(
    subsamples: dict[int, dict[str, np.ndarray]],
    *,
    Ns: list[int] | None = None,
    width: float = 30.0,
    height: float = 15.0,
    dpi: int = 300,
) -> None:
    """
    CCPP version of 'plot_all_resolutions'.

    Expects 'subsamples' of the form:
        subsamples[N] = {
            "X_raw": np.ndarray shape (N,2) with columns [AT, V],
            "y":     np.ndarray shape (N,)   with EP values,
        }

    For each N, plots a 3D scatter (AT, V, EP).
    """
    if Ns is None:
        Ns = sorted(subsamples.keys())

    n = len(Ns)
    fig, axes = plt.subplots(
        1, n, figsize=(width, height), dpi=dpi,
        subplot_kw=dict(projection="3d")
    )

    for ax, N in zip(axes, Ns):
        if N not in subsamples:
            print(f"[CCPP-RES] Skipping N={N}: not in subsamples.")
            continue

        X_raw = subsamples[N]["X_raw"]
        y     = subsamples[N]["y"]

        at = X_raw[:, 0]
        v  = X_raw[:, 1]

        sc = ax.scatter(
            at, v, y,
            c=y, cmap="viridis",
            s=35, edgecolor="k", alpha=0.85,
        )
        ax.set_title(f"N = {N}")
        ax.set_xlabel("AT (°C)")
        ax.set_ylabel("V (cm Hg)")
        ax.set_zlabel("EP (MW)")

    plt.tight_layout()
    plt.show()

# ===== CELL 021 (code) =====
subsamples = {}
for N in sample_sizes:
    # choose a random subset of size N from the full data
    idx = np.random.permutation(X_full_raw.shape[0])[:N]
    subsamples[N] = {
        "X_raw": X_full_raw[idx],
        "y":     y_full[idx],
    }

plot_ccpp_all_resolutions(subsamples, Ns=[50, 100, 200])

# ===== CELL 022 (markdown) =====
# **Visualize dataset splits**

# ===== CELL 023 (code) =====
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
    Visualize **preprocessed** train/val/test splits for the CCPP 2D
    subproblem:

        x1 = scaled AT (ambient temperature)
        x2 = scaled V  (exhaust vacuum)
        y  = EP (net hourly electrical energy output)

    Assumes that CSVs in `folder` have columns ['x1','x2','y'] created by
    the CCPP preprocessing pipeline.

    Compliant with protocol:
      • scaler fitted on train only
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
        ax.set_xlabel("AT (scaled)")
        ax.set_ylabel("V (scaled)")
        ax.grid(True, ls="--", alpha=0.6)
        fig.colorbar(sc, ax=ax, shrink=0.7, label="EP (MW)")

    plt.suptitle("Preprocessed CCPP 2D Splits (AT, V → EP)", fontsize=18, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# ===== CELL 024 (code) =====
N_example = 100
SEED_example = 0
folder = Path("preprocessed/ccpp") / f"N{N_example}_seed{SEED_example}"

visualize_dataset_splits(
    folder=folder,
    n_rows=10,
    width=40,
    height=12,
    dpi=300,
)

# ===== CELL 025 (markdown) =====
# ## **Preprocessing**

# ===== CELL 026 (code) =====
@dataclass(frozen=True)
class RawDomain2D:
    """
    Raw 2D domain for the CCPP subproblem (AT, V).

    x1 ≡ AT (ambient temperature, °C)
    x2 ≡ V  (exhaust vacuum, cm Hg)

    This is the analogue of [0, 2π]² or [0, 1]² in the synthetic notebook.
    """
    x1_min: float  # AT_min
    x1_max: float  # AT_max
    x2_min: float  # V_min
    x2_max: float  # V_max

    @property
    def as_tuple(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Return ((x1_min, x1_max), (x2_min, x2_max))."""
        return ((self.x1_min, self.x1_max),
                (self.x2_min, self.x2_max))


# Instantiate the global raw domain using the values already computed
RAW_DOMAIN_CCPP_2D = RawDomain2D(
    x1_min=float(RAW_DOMAIN_AT[0]),
    x1_max=float(RAW_DOMAIN_AT[1]),
    x2_min=float(RAW_DOMAIN_V[0]),
    x2_max=float(RAW_DOMAIN_V[1]),
)

print(
    "[CCPP-CONFIG] RAW_DOMAIN_CCPP_2D = "
    f"AT∈[{RAW_DOMAIN_CCPP_2D.x1_min:.3f}, {RAW_DOMAIN_CCPP_2D.x1_max:.3f}], "
    f"V∈[{RAW_DOMAIN_CCPP_2D.x2_min:.3f}, {RAW_DOMAIN_CCPP_2D.x2_max:.3f}]"
)


# ---------------------------------------------------------------------
# 2. Helper: 2D raw grid over (AT, V)
# ---------------------------------------------------------------------

def generate_ccpp_raw_grid(
    grid_size: int = 60,
    domain: RawDomain2D = RAW_DOMAIN_CCPP_2D,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a 2D evaluation grid over the **raw** CCPP domain (AT, V).

    Parameters
    ----------
    grid_size : int
        Number of points per axis (grid_size × grid_size total points).
    domain : RawDomain2D
        Raw domain with separate ranges for AT and V.

    Returns
    -------
    AT_grid, V_grid : np.ndarray, shape (grid_size, grid_size)
        Meshgrid of raw AT (°C) and V (cm Hg) values.

    Notes
    -----
    - Use this to build X_grid_raw for KRR / USDR⁺:
          X_grid_raw = np.column_stack(
              [AT_grid.ravel(), V_grid.ravel()]
          )
      then preprocess X_grid_raw with the same scaler as X_train.
    """
    at_vals = np.linspace(domain.x1_min, domain.x1_max, grid_size)
    v_vals  = np.linspace(domain.x2_min, domain.x2_max, grid_size)

    AT_grid, V_grid = np.meshgrid(at_vals, v_vals, indexing="xy")
    return AT_grid, V_grid


# ---------------------------------------------------------------------
# 3. Helper: 1D raw slice grids for (AT, V)
# ---------------------------------------------------------------------

def generate_ccpp_slice_raw_grid(
    num_points: int = 200,
    axis: str = "AT",
    fixed_value: float | None = None,
    domain: RawDomain2D = RAW_DOMAIN_CCPP_2D,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate raw 1D slice points for (AT, V).

    Parameters
    ----------
    num_points : int
        Number of points along the varying axis.
    axis : {"AT", "V"}
        Which variable to vary:
          - "AT": vary AT, keep V fixed.
          - "V" : vary V, keep AT fixed.
    fixed_value : float or None
        Raw value for the fixed coordinate. If None, use the mid-point
        of its domain.
    domain : RawDomain2D
        Raw domain with ranges for AT and V.

    Returns
    -------
    x_var : np.ndarray, shape (num_points,)
        Values along the varying axis (AT or V, in raw units).
    x_fix : np.ndarray, shape (num_points,)
        Values of the fixed axis (constant array).
    X_slice_raw : np.ndarray, shape (num_points, 2)
        2D raw points (AT, V) along the slice, ready to be preprocessed.

    Examples
    --------
    - Slice at fixed V:
          at_vals, v_vals, X_slice = generate_ccpp_slice_raw_grid(
              num_points=200, axis="AT", fixed_value=40.0
          )

    - Slice at fixed AT (mid-domain V chosen automatically):
          v_vals, at_vals, X_slice = generate_ccpp_slice_raw_grid(
              num_points=200, axis="V"
          )
    """
    axis_norm = axis.strip().upper()
    if axis_norm not in {"AT", "V"}:
        raise ValueError(f"axis must be 'AT' or 'V', got {axis!r}")

    # Midpoints used when fixed_value is not provided
    at_mid = 0.5 * (domain.x1_min + domain.x1_max)
    v_mid  = 0.5 * (domain.x2_min + domain.x2_max)

    if axis_norm == "AT":
        x_var = np.linspace(domain.x1_min, domain.x1_max, num_points)
        v_fix = v_mid if fixed_value is None else float(fixed_value)
        x_fix = np.full_like(x_var, v_fix, dtype=float)
        X_slice_raw = np.column_stack([x_var, x_fix])
    else:  # axis_norm == "V"
        x_var = np.linspace(domain.x2_min, domain.x2_max, num_points)
        at_fix = at_mid if fixed_value is None else float(fixed_value)
        x_fix = np.full_like(x_var, at_fix, dtype=float)
        X_slice_raw = np.column_stack([x_fix, x_var])

    return x_var, x_fix, X_slice_raw

# ===== CELL 027 (markdown) =====
# **CCPP dataset loader**

# ===== CELL 028 (code) =====
CCPP_OUTPUT_DIR = Path("preprocessed/ccpp")

def load_processed_ccpp_2d_dataset(
    base_path: str | Path = CCPP_OUTPUT_DIR,
    N: int = 100,
    seed: int = 0,
    normalize: str = None,
    eps: float = 1e-12,
) -> dict:
    """
    Loader for the preprocessed CCPP 2D datasets (x1=AT, x2=V, y=EP).

    Assumptions
    ----------
    • Datasets were created by preprocess_and_save_ccpp_2d_datasets(...)
      and saved under:
          base_path / f"N{N}_seed{seed}" / {train,val,test}.csv
    • Each CSV has columns: ["x1", "x2", "y"].
    • Inputs x1,x2 are already preprocessed according to `NORMALIZE`
      (MinMax → [0,1]^2 or Z-score standardisation).

    Parameters
    ----------
    base_path : str or Path
        Root folder where the processed splits live.
    N : int
        Subsample size (e.g. 50, 100, 200).
    seed : int
        Seed used for that subsample.
    normalize : {"minmax", "zscore"} or None
        Preprocessing mode. If None, falls back to global NORMALIZE.
    eps : float
        Numerical tolerance used for MinMax clipping checks.

    Returns
    -------
    data_dict : dict
        {
          "X_train", "y_train",
          "X_val",   "y_val",
          "X_test",  "y_test",
          "metadata": {
              "N", "seed", "normalize", "base_path",
              "n_train", "n_val", "n_test",
          }
        }
    """
    if normalize is None:
        normalize = NORMALIZE

    base_path = Path(base_path)
    folder = base_path / f"N{N}_seed{seed}"

    if not folder.exists():
        raise FileNotFoundError(
            f"[CCPP-2D] Dataset folder not found for N={N}, seed={seed}: {folder}"
        )

    splits = ["train", "val", "test"]
    data: dict[str, object] = {
        "metadata": {
            "N": N,
            "seed": seed,
            "normalize": normalize,
            "base_path": str(base_path),
        }
    }

    required_cols = {"x1", "x2", "y"}

    for split in splits:
        csv_path = folder / f"{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"[CCPP-2D] Missing {split}.csv for N={N}, seed={seed}: {csv_path}"
            )

        df = pd.read_csv(csv_path)

        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"[CCPP-2D] Missing columns in {csv_path}: "
                f"expected {required_cols}, got {set(df.columns)}"
            )

        X = df[["x1", "x2"]].values.astype(np.float64)
        y = df["y"].values.astype(np.float64)

        if normalize == "minmax":
            # Inputs should already be in [0,1]^2. We clip and warn if they are slightly off.
            X_clipped = np.clip(X, 0.0, 1.0)
            if not np.all((X >= -eps) & (X <= 1.0 + eps)):
                print(
                    f"[CCPP-2D] WARNING: X_{split} slightly outside [0,1] "
                    f"(N={N}, seed={seed}); values clipped."
                )
            X = X_clipped
        elif normalize == "zscore":
            # Standardized, unbounded → no clipping.
            pass
        else:
            raise ValueError(f"[CCPP-2D] Unknown normalize mode: {normalize}")

        data[f"X_{split}"] = X
        data[f"y_{split}"] = y
        data["metadata"][f"n_{split}"] = len(y)

        print(
            f"[CCPP-2D] Loaded {split:>5} | "
            f"X shape={X.shape}, y shape={y.shape} | NORMALIZE={normalize}"
        )

    print(
        f"\n[CCPP-2D] Dataset N={N}, seed={seed} loaded from {base_path} → "
        f"{data['metadata']['n_train']}/"
        f"{data['metadata']['n_val']}/"
        f"{data['metadata']['n_test']} "
        f"(NORMALIZE={normalize})\n"
    )

    return data

# ===== CELL 029 (code) =====
N_test   = 100
SEED_test = 0

ccpp_data = load_processed_ccpp_2d_dataset(
    base_path=CCPP_OUTPUT_DIR,
    N=N_test,
    seed=SEED_test,
)

X_train = ccpp_data["X_train"]
X_val   = ccpp_data["X_val"]
X_test  = ccpp_data["X_test"]

print("[CHECK] Shapes:")
print(f"  X_train: {X_train.shape}, y_train: {ccpp_data['y_train'].shape}")
print(f"  X_val:   {X_val.shape},   y_val:   {ccpp_data['y_val'].shape}")
print(f"  X_test:  {X_test.shape},  y_test:  {ccpp_data['y_test'].shape}")

print("\n[CHECK] Metadata:")
for k, v in ccpp_data["metadata"].items():
    print(f"  {k}: {v}")

assert (
    ccpp_data["metadata"]["n_train"]
    + ccpp_data["metadata"]["n_val"]
    + ccpp_data["metadata"]["n_test"]
    == N_test
), "[CHECK] n_train + n_val + n_test != N"
print("\n[CHECK] Split sizes sum to N — OK.")

# ===== CELL 030 (markdown) =====
# **2D visualization helpers**

# ===== CELL 031 (code) =====
def visualize_dataset_splits_ccpp(
    data_dict: dict,
    n_rows: int = 5,
    width: float = 36.0,
    height: float = 12.0,
    dpi: int = 300,
    theme: str = "darkgrid",
    palette: str = "viridis",
) -> None:
    """
    Visualize **preprocessed** CCPP 2D dataset splits (AT, V → EP)
    using the standard `data_dict` returned by `load_processed_ccpp_2d_dataset`.

    Expects:
        data_dict = {
            "X_train": np.ndarray of shape (n_train, 2),
            "y_train": np.ndarray of shape (n_train,),
            "X_val":   np.ndarray of shape (n_val, 2),
            "y_val":   np.ndarray of shape (n_val,),
            "X_test":  np.ndarray of shape (n_test, 2),
            "y_test":  np.ndarray of shape (n_test,),
            "metadata": {
                "N", "seed", "normalize", "base_path",
                "n_train", "n_val", "n_test",
            }
        }

    Behaviour:
      • Prints small tables (first `n_rows`) for train / val / test.
      • Plots x1 vs x2 (scaled) with colour = y for each split.
      • Uses the same interface style as your other USDR+ notebooks.
    """
    sns.set_theme(style=theme, palette=palette, context="talk")

    X_train = np.asarray(data_dict["X_train"], dtype=np.float64)
    y_train = np.asarray(data_dict["y_train"], dtype=np.float64)
    X_val   = np.asarray(data_dict["X_val"],   dtype=np.float64)
    y_val   = np.asarray(data_dict["y_val"],   dtype=np.float64)
    X_test  = np.asarray(data_dict["X_test"],  dtype=np.float64)
    y_test  = np.asarray(data_dict["y_test"],  dtype=np.float64)

    meta = data_dict.get("metadata", {})
    N_total   = meta.get("N", None)
    seed      = meta.get("seed", None)
    normalize = meta.get("normalize", None)
    base_path = meta.get("base_path", None)
    n_train   = meta.get("n_train", X_train.shape[0])
    n_val     = meta.get("n_val",   X_val.shape[0])
    n_test    = meta.get("n_test",  X_test.shape[0])

    # --- 1. Build small DataFrames for tabular view ------------------------
    def _df_from_split(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(X, columns=["x1", "x2"])
        df["y"] = y
        return df

    train_df_full = _df_from_split(X_train, y_train)
    val_df_full   = _df_from_split(X_val,   y_val)
    test_df_full  = _df_from_split(X_test,  y_test)

    train_df = train_df_full.head(n_rows)
    val_df   = val_df_full.head(n_rows)
    test_df  = test_df_full.head(n_rows)

    # --- 2. Print metadata + styled tables ---------------------------------
    print("\n=== [CCPP-2D] Processed dataset summary ===")
    print(f"  N total:   {N_total}")
    print(f"  seed:      {seed}")
    print(f"  normalize: {normalize}")
    print(f"  base_path: {base_path}")
    print(f"  split sizes → train/val/test = {n_train}/{n_val}/{n_test}")

    print(f"\n=== Training Split (n_train={n_train}) – first {len(train_df)} rows ===")
    display(train_df.style.background_gradient(cmap="Blues"))

    print(f"\n=== Validation Split (n_val={n_val}) – first {len(val_df)} rows ===")
    display(val_df.style.background_gradient(cmap="Greens"))

    print(f"\n=== Test Split (n_test={n_test}) – first {len(test_df)} rows ===")
    display(test_df.style.background_gradient(cmap="Oranges"))

    # --- 3. Scatter plots in (x1, x2) scaled space -------------------------
    fig, axes = plt.subplots(1, 3, figsize=(width, height), dpi=dpi)
    scatter_kwargs = dict(
        s=70,
        edgecolor="k",
        linewidth=0.6,
        alpha=0.85,
    )

    splits = [
        (train_df_full, "Train"),
        (val_df_full,   "Val"),
        (test_df_full,  "Test"),
    ]

    for ax, (df_split, name) in zip(axes, splits):
        sc = ax.scatter(
            df_split["x1"],
            df_split["x2"],
            c=df_split["y"],
            cmap=palette,
            **scatter_kwargs,
        )
        ax.set_title(
            f"{name} split in scaled (x1, x2)\n"
            f"({len(df_split)} points)",
            weight="bold",
        )
        ax.set_xlabel("x1 (scaled AT)")
        ax.set_ylabel("x2 (scaled V)")
        ax.grid(True, linestyle="--", alpha=0.5)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label("y = EP (MW)", fontsize=10)

    plt.suptitle(
        "Processed CCPP 2D splits – scaled (AT, V) → EP",
        fontsize=18,
        weight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


def visualize_dataset_splits_ccpp_raw(
    raw_data_dict: dict,
    width: float = 18.0,
    height: float = 6.0,
    dpi: int = 200,
    theme: str = "whitegrid",
    palette: str = "viridis",
) -> None:
    """
    OPTIONAL helper: visualise **raw** (AT, V → EP) splits in physical units.

    This is useful if you decide to keep raw coordinates around for interpretation.
    Expected structure (you can adapt to your actual raw container):

        raw_data_dict = {
            "X_train_raw": np.ndarray shape (n_train, 2),  # [AT, V] in original units
            "X_val_raw":   np.ndarray shape (n_val, 2),
            "X_test_raw":  np.ndarray shape (n_test, 2),
            "y_train":     np.ndarray shape (n_train,),
            "y_val":       np.ndarray shape (n_val,),
            "y_test":      np.ndarray shape (n_test,),
            "metadata": { ... }   # optional, same style as processed
        }

    Behaviour:
      • Scatter plots of (AT, V) with colour = EP for train / val / test.
      • All in raw units (°C, cm Hg, MW).
    """
    sns.set_theme(style=theme, palette=palette, context="talk")

    X_train_raw = np.asarray(raw_data_dict["X_train_raw"], dtype=np.float64)
    X_val_raw   = np.asarray(raw_data_dict["X_val_raw"],   dtype=np.float64)
    X_test_raw  = np.asarray(raw_data_dict["X_test_raw"],  dtype=np.float64)
    y_train     = np.asarray(raw_data_dict["y_train"],     dtype=np.float64)
    y_val       = np.asarray(raw_data_dict["y_val"],       dtype=np.float64)
    y_test      = np.asarray(raw_data_dict["y_test"],      dtype=np.float64)

    meta = raw_data_dict.get("metadata", {})
    N_total = meta.get("N", None)
    seed    = meta.get("seed", None)

    print("\n=== [CCPP-2D-RAW] Raw-space splits summary ===")
    print(f"  N total: {N_total}")
    print(f"  seed:    {seed}")
    print(
        f"  split sizes → train/val/test = "
        f"{X_train_raw.shape[0]}/{X_val_raw.shape[0]}/{X_test_raw.shape[0]}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(width, height), dpi=dpi)
    scatter_kwargs = dict(
        s=70,
        edgecolor="k",
        linewidth=0.6,
        alpha=0.85,
    )

    splits = [
        (X_train_raw, y_train, "Train (raw AT, V)"),
        (X_val_raw,   y_val,   "Val (raw AT, V)"),
        (X_test_raw,  y_test,  "Test (raw AT, V)"),
    ]

    for ax, (X_raw, y, title) in zip(axes, splits):
        sc = ax.scatter(
            X_raw[:, 0],   # AT
            X_raw[:, 1],   # V
            c=y,
            cmap=palette,
            **scatter_kwargs,
        )
        ax.set_title(f"{title}\n({X_raw.shape[0]} points)", weight="bold")
        ax.set_xlabel("AT (°C)")
        ax.set_ylabel("V (cm Hg)")
        ax.grid(True, linestyle="--", alpha=0.5)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label("EP (MW)", fontsize=10)

    plt.suptitle(
        "Raw CCPP splits – (AT, V) → EP",
        fontsize=18,
        weight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

# ===== CELL 032 (code) =====
data = load_processed_ccpp_2d_dataset(
    base_path=CCPP_OUTPUT_DIR,
    N=100,
    seed=0,
    normalize=NORMALIZE,
)

visualize_dataset_splits_ccpp(data)

# ===== CELL 033 (markdown) =====
# **Experimental config check**

# ===== CELL 034 (code) =====
# ---------------------------------------------------------------------
# 1) Preprocessing choice (fixed for all CCPP 2D experiments)
# ---------------------------------------------------------------------
# We use the same convention as in the USDR+ synthetic notebook:
#   - raw inputs in physical units (AT in °C, V in cm Hg)
#   - if NORMALIZE == "minmax": scale each coordinate to [0, 1] using train only
#   - these scaled coordinates are what go into the USDR+ feature map
NORMALIZE: str = "minmax"      # or "zscore", but we freeze "minmax" here

# ---------------------------------------------------------------------
# 2) Small-N regime and seeds (fixed)
# ---------------------------------------------------------------------
# Sample sizes for the small-N study
sample_sizes: list[int] = [50, 100, 200]

# Independent random seeds for subsampling / splits
SEEDS: list[int] = [0, 1, 2]

# ---------------------------------------------------------------------
# 3) Processed CCPP 2D splits location
# ---------------------------------------------------------------------
# This is where preprocess_and_save_ccpp_2d_datasets(...) wrote:
#   preprocessed/ccpp/N{N}_seed{seed}/train.csv, val.csv, test.csv
OUTPUT_DIR_CCPP = Path("preprocessed/ccpp")
OUTPUT_DIR_CCPP.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 4) Raw 2D domain for (AT, V)
# ---------------------------------------------------------------------
# These should match what you printed during EDA / preprocessing, e.g.:
#   [CCPP] AT domain ≈ [1.810, 37.110], V domain ≈ [25.360, 81.560]
# If your values are slightly different, update them here.
RAW_DOMAIN_AT: tuple[float, float] = (1.810, 37.110)   # AT (°C)
RAW_DOMAIN_V:  tuple[float, float] = (25.360, 81.560)  # V (cm Hg)

# Pack a convenient 2D-domain object for grid / slice helpers
RAW_DOMAIN_CCPP_2D = {
    "AT": RAW_DOMAIN_AT,
    "V":  RAW_DOMAIN_V,
}

# ---------------------------------------------------------------------
# 5) Config summary (for sanity + documentation in the notebook)
# ---------------------------------------------------------------------
print("[CCPP-CONFIG] Experimental configuration:")
print(f"  NORMALIZE       = {NORMALIZE}")
print(f"  sample_sizes    = {sample_sizes}")
print(f"  SEEDS           = {SEEDS}")
print(f"  OUTPUT_DIR_CCPP = {OUTPUT_DIR_CCPP}")
print(
    f"  RAW_DOMAIN_CCPP_2D = "
    f"AT∈[{RAW_DOMAIN_AT[0]:.3f}, {RAW_DOMAIN_AT[1]:.3f}], "
    f"V∈[{RAW_DOMAIN_V[0]:.3f}, {RAW_DOMAIN_V[1]:.3f}]"
)

# ===== CELL 035 (markdown) =====
# # **Implement the $U_{\text{SDR}+}$ feature map**

# ===== CELL 036 (code) =====
# === USDR+ CIRCUIT CONFIG (2 qubits, 2D input) ==================================

# Global config (already consistent with your notebook, repeated here for clarity)
depth = 2                      # L = 2 data re-uploading layers
entangler_default = "cnot"     # fixed entangler for USDR+
axes_low_default = ("X", "Z")  # low-frequency block: (wire 0, wire 1)
axes_high_default = ("Z", "X") # high-frequency block: (wire 0, wire 1)


def _apply_single_rotation(axis: str, angle: float, wire: int) -> None:
    """
    Helper: apply a single-qubit rotation on the given wire along the chosen axis.

    Parameters
    ----------
    axis : {"X","Y","Z"}
        Rotation axis.
    angle : float
        Rotation angle (radians).
    wire : int
        Qubit index (0 or 1 in this USDR+ setup).
    """
    if axis == "X":
        qml.RX(angle, wires=wire)
    elif axis == "Y":
        qml.RY(angle, wires=wire)
    elif axis == "Z":
        qml.RZ(angle, wires=wire)
    else:
        raise ValueError(f"[U_SDR_plus] Unsupported axis '{axis}'. Use 'X', 'Y', or 'Z'.")


def _apply_entangler(entangler: str, wires=(0, 1)) -> None:
    """
    Helper: apply the chosen 2-qubit entangling gate.

    Parameters
    ----------
    entangler : {"cnot","cz"}
        Name of the entangling gate.
    wires : tuple[int,int]
        Control and target qubits (default: (0,1)).
    """
    if entangler.lower() == "cnot":
        qml.CNOT(wires=wires)
    elif entangler.lower() == "cz":
        qml.CZ(wires=wires)
    else:
        raise ValueError(f"[U_SDR_plus] Unsupported entangler '{entangler}'. Use 'cnot' or 'cz'.")


def U_SDR_plus(
    x: np.ndarray,
    theta: np.ndarray,
    L: int = depth,
    entangler: str = entangler_default,
    axes_low: tuple[str, str] = axes_low_default,
    axes_high: tuple[str, str] = axes_high_default,
) -> None:
    """
    Canonical implementation of the 2-qubit USDR⁺ feature map U_{SDR+}.

    Inputs
    ------
    x : array-like, shape (2,)
        Preprocessed 2D input (x1, x2).
        • If NORMALIZE == "minmax": x ∈ [0, 1]^2
        • If NORMALIZE == "zscore": x is standardized (unbounded)

    theta : array-like, shape (4,)
        Trainable USDR⁺ parameters (λ1, λ2, γ, β):
            λ1, λ2 > 0   : per-feature scaling of x1, x2
            γ    ≥ 1     : high-frequency boost
            β    > 0     : global bandwidth (x̂ = x / β)

    L : int, optional (default: depth = 2)
        Number of data re-uploading layers.

    entangler : {"cnot","cz"}, optional (default: "cnot")
        Two-qubit entangling gate applied between low and high blocks.

    axes_low : tuple[str,str], optional (default: ("X","Z"))
        Rotation axes for the low-frequency block:
            axes_low[0] : axis on wire 0 (for x1)
            axes_low[1] : axis on wire 1 (for x2)

    axes_high : tuple[str,str], optional (default: ("Z","X"))
        Rotation axes for the high-frequency block:
            axes_high[0] : axis on wire 0 (for x1, boosted by γ)
            axes_high[1] : axis on wire 1 (for x2, boosted by γ)

    Circuit structure (per layer ℓ = 1..L)
    --------------------------------------
        1. Bandwidth scaling: x̂ = x / β
        2. Low-frequency block:
               R_{axes_low[0]}( λ1 · x̂1 ) on wire 0
               R_{axes_low[1]}( λ2 · x̂2 ) on wire 1
        3. Entangler: CNOT(0→1)  (by default)
        4. High-frequency block:
               R_{axes_high[0]}( γ · λ1 · x̂1 ) on wire 0
               R_{axes_high[1]}( γ · λ2 · x̂2 ) on wire 1
        5. Entangler: CNOT(0→1) again

    Notes
    -----
    • This function is a *template* to be called inside a PennyLane QNode.
    • It does not return anything; it just applies gates on the active device.
    """
    # --- Input checks ------------------------------------------------------
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.shape != (2,):
        raise ValueError(f"[U_SDR_plus] Expected x shape (2,), got {x.shape}.")

    theta = np.asarray(theta, dtype=np.float64).ravel()
    if theta.shape != (4,):
        raise ValueError(f"[U_SDR_plus] Expected theta shape (4,), got {theta.shape}.")

    lambda1, lambda2, gamma, beta = theta

    if beta <= 0:
        raise ValueError(f"[U_SDR_plus] β must be > 0, got β={beta}.")
    if lambda1 <= 0 or lambda2 <= 0:
        raise ValueError(
            f"[U_SDR_plus] λ1, λ2 must be > 0, got λ1={lambda1}, λ2={lambda2}."
        )
    if gamma < 1.0:
        raise ValueError(f"[U_SDR_plus] γ should be ≥ 1, got γ={gamma}.")

    # --- Bandwidth scaling -------------------------------------------------
    # x̂ = x / β
    x_hat = x / beta
    x1_hat, x2_hat = float(x_hat[0]), float(x_hat[1])

    # --- Data re-uploading layers -----------------------------------------
    for layer in range(L):
        # Low-frequency block
        _apply_single_rotation(
            axis=axes_low[0],
            angle=lambda1 * x1_hat,
            wire=0,
        )
        _apply_single_rotation(
            axis=axes_low[1],
            angle=lambda2 * x2_hat,
            wire=1,
        )

        # Entangler
        _apply_entangler(entangler, wires=(0, 1))

        # High-frequency block (boosted by γ)
        _apply_single_rotation(
            axis=axes_high[0],
            angle=gamma * lambda1 * x1_hat,
            wire=0,
        )
        _apply_single_rotation(
            axis=axes_high[1],
            angle=gamma * lambda2 * x2_hat,
            wire=1,
        )

        # Second entangler
        _apply_entangler(entangler, wires=(0, 1))


# === Convenience QNode: prepare USDR+ state |φ(x; θ)⟩ =======================

# You can re-use your existing device; here is a canonical one:
dev_usdr_plus = qml.device("default.qubit", wires=2)


@qml.qnode(dev_usdr_plus, interface="numpy")
def usdr_plus_state(x: np.ndarray, theta: np.ndarray):
    """
    QNode that prepares the USDR⁺ feature state |φ(x; θ)⟩ on 2 qubits and returns
    the full statevector.

    Assumptions
    -----------
    • x is already *preprocessed* 2D input:
        - if NORMALIZE == "minmax": x ∈ [0,1]^2
        - if NORMALIZE == "zscore": standardized
    • This QNode applies only the β scaling (via U_SDR_plus).
    • Global config: L = 2, entangler = CNOT, axes_low=("X","Z"), axes_high=("Z","X").

    Returns
    -------
    state : np.ndarray, shape (4,)
        Statevector of the 2-qubit system after applying U_{SDR+}(x; θ).
    """
    # Ensure proper shape
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.shape != (2,):
        raise ValueError(f"[usdr_plus_state] Expected x shape (2,), got {x.shape}.")

    theta = np.asarray(theta, dtype=np.float64).ravel()
    if theta.shape != (4,):
        raise ValueError(
            f"[usdr_plus_state] Expected theta shape (4,), got {theta.shape}."
        )

    # Prepare |00⟩ (default initial state of default.qubit), then apply U_{SDR+}
    U_SDR_plus(
        x=x,
        theta=theta,
        L=depth,
        entangler=entangler_default,
        axes_low=axes_low_default,
        axes_high=axes_high_default,
    )

    return qml.state()

# ===== CELL 037 (code) =====
def visualize_U_SDR_plus_2D(
    x_example: np.ndarray,
    theta,
    width: float = 12.0,
    height: float = 6.0,
    dpi: int = 300,
    L: int = 2,
    entangler: str = "cnot",
    save: bool = True,
    save_dir: str | Path = "figures/ccpp/diagnostics",
    plot_name: str = "usdr_plus_circuit",
) -> None:
    """
    Visualize the U_{SDR+} feature map for a single **CCPP 2D input**.

    CCPP + USDR+ setting
    --------------------
    • We work on the 2D subproblem:
          x_1 = AT (ambient temperature)
          x_2 = V  (exhaust vacuum)
      after preprocessing.

    • x_example is a *preprocessed* feature vector (one CCPP sample):
          - if NORMALIZE == "minmax": x_example ∈ [0, 1]^2
          - if NORMALIZE == "zscore": x_example is standardized

    • U_SDR_plus will apply only the bandwidth scaling:
          x̂ = x_example / β

    • Circuit structure:
          - 2 qubits
          - L = 2 data re-uploading layers
          - entangler = "cnot"
          - axes_low  = (X, Z)
          - axes_high = (Z, X)
    """
    # --- Input checks ------------------------------------------------------
    x_example = np.asarray(x_example, dtype=np.float64).ravel()
    if x_example.shape != (2,):
        raise ValueError(
            f"[visualize_U_SDR_plus_2D] Expected x_example shape (2,), "
            f"got {x_example.shape}."
        )

    theta = np.asarray(theta, dtype=np.float64).ravel()
    if theta.shape != (4,):
        raise ValueError(
            f"[visualize_U_SDR_plus_2D] Expected theta shape (4,), "
            f"got {theta.shape}."
        )

    # Local device for drawing (keeps this helper self-contained)
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit():
        # Apply USDR+ with the same structure used in experiments
        U_SDR_plus(
            x=x_example,
            theta=theta,
            L=L,
            entangler=entangler,
            axes_low=axes_low_default,
            axes_high=axes_high_default,
        )
        return qml.state()

    # Draw circuit with PennyLane's matplotlib helper
    fig, ax = qml.draw_mpl(circuit, decimals=3, expansion_strategy="device")()
    fig.set_size_inches(width, height)
    fig.set_dpi(dpi)

    plt.suptitle(
        r"U$_{\mathrm{SDR}+}$ on CCPP 2D input $(\mathrm{AT}, \mathrm{V})$ "
        r"– L=2, CNOT, $\gamma$-boost, $\beta$-scaling",
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

# Load preprocessed CCPP 2D splits (AT, V → EP) for this (N, SEED)
data_example = load_processed_ccpp_2d_dataset(
    base_path=OUTPUT_DIR_CCPP,
    N=N_example,
    seed=SEED_example,
    normalize=NORMALIZE,
)

X_train = data_example["X_train"]  # shape (n_train, 2), scaled (AT, V)
y_train = data_example["y_train"]  # shape (n_train,)

# Pick one example point from the (AT, V) training inputs
x_example = X_train[0]

# Example USDR+ parameters (λ1, λ2, γ, β) just for visualisation
theta_example = np.array([2.1, 1.8, 2.3, 1.5], dtype=float)

visualize_U_SDR_plus_2D(
    x_example=x_example,
    theta=theta_example,
    width=20,
    height=5.5,
)

# ===== CELL 039 (markdown) =====
# # **Quantum Kernel Computation**

# ===== CELL 040 (markdown) =====
# **Feature map circuit**

# ===== CELL 041 (code) =====
usdr_plus_dev = qml.device("default.qubit", wires=2)


# ------------------------------------------------------------------
# Low-level QNode: takes all circuit-structure arguments explicitly
# ------------------------------------------------------------------
@qml.qnode(usdr_plus_dev)
def _usdr_plus_state_qnode(
    x: np.ndarray,
    theta: np.ndarray,
    L: int,
    entangler: str,
    axes_low,
    axes_high,
):
    """
    Internal QNode that applies U_SDR_plus and returns the full statevector.

    Parameters
    ----------
    x : np.ndarray, shape (2,)
        Preprocessed 2D input (e.g., CCPP (AT, V) after scaling).
    theta : np.ndarray, shape (4,)
        USDR+ parameters (lambda1, lambda2, gamma, beta).
    L : int
        Number of data-reuploading layers (typically L=2).
    entangler : str
        Entangling gate pattern ("cnot" in the USDR+ protocol).
    axes_low, axes_high
        Tuples specifying the Pauli axes for low/high-frequency blocks,
        e.g. ("X", "Z") and ("Z", "X").

    Returns
    -------
    state : np.ndarray, shape (4,)
        Full statevector of the 2-qubit system after U_SDR_plus.
    """
    # Apply the USDR+ feature map
    U_SDR_plus(
        x=x,
        theta=theta,
        L=L,
        entangler=entangler,
        axes_low=axes_low,
        axes_high=axes_high,
    )
    return qml.state()


# ------------------------------------------------------------------
# High-level convenience wrapper: usdr_plus_state
# ------------------------------------------------------------------
def usdr_plus_state(
    x: np.ndarray,
    theta,
    L: int | None = None,
    entangler: str | None = None,
    axes_low=None,
    axes_high=None,
):
    """
    Public helper to get the USDR+ statevector for a single 2D input.

    CCPP + USDR+ setting
    --------------------
    • Input x is a *preprocessed* 2D feature vector:
        - For CCPP: x = (AT, V) after MinMax or Z-score scaling.
        - Shape must be (2,).

    • theta = (lambda1, lambda2, gamma, beta) is the USDR+ parameter vector.

    • Circuit structure defaults:
        - L          = depth (global, typically 2),
        - entangler  = entangler_default ("cnot"),
        - axes_low   = axes_low_default  (e.g. ("X", "Z")),
        - axes_high  = axes_high_default (e.g. ("Z", "X")).

    Returns
    -------
    state : np.ndarray, shape (4,)
        2-qubit statevector |psi_theta(x)> prepared by U_SDR_plus.
    """
    # ---- Defaults from global config --------------------------------------
    global depth, entangler_default, axes_low_default, axes_high_default

    if L is None:
        L = depth
    if entangler is None:
        entangler = entangler_default
    if axes_low is None:
        axes_low = axes_low_default
    if axes_high is None:
        axes_high = axes_high_default

    # ---- Input validation / shaping ---------------------------------------
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.shape != (2,):
        raise ValueError(
            f"[usdr_plus_state] Expected x shape (2,), got {x.shape}."
        )

    theta = np.asarray(theta, dtype=np.float64).ravel()
    if theta.shape != (4,):
        raise ValueError(
            f"[usdr_plus_state] Expected theta shape (4,), got {theta.shape}."
        )

    # ---- Call the QNode ----------------------------------------------------
    state = _usdr_plus_state_qnode(
        x=x,
        theta=theta,
        L=L,
        entangler=entangler,
        axes_low=axes_low,
        axes_high=axes_high,
    )

    # PennyLane returns an array with dtype=complex; make sure it's NumPy
    return np.asarray(state)

# ===== CELL 042 (code) =====
data_ccpp_example = load_processed_ccpp_2d_dataset(
    base_path=OUTPUT_DIR_CCPP,
    N=100,
    seed=0,
    normalize=NORMALIZE,
)

x_example = data_ccpp_example["X_train"][0]          # shape (2,)
theta_example = np.array([2.0, 1.5, 2.5, 1.2])      # (λ1, λ2, γ, β)

psi = usdr_plus_state(x_example, theta_example)      # shape (4,)
print("State norm:", np.linalg.norm(psi))
print("Statevector:", psi)

# ===== CELL 043 (markdown) =====
# **Fidelity 2D**

# ===== CELL 044 (code) =====
def fidelity_2d(
    x_a: np.ndarray,
    x_b: np.ndarray,
    theta,
    *,
    L: int = 2,
    entangler: str = "cnot",
) -> float:
    """
    Fidelity kernel between two preprocessed 2D inputs for USDR+.

    k_θ(x_a, x_b) = |⟨ψ_θ(x_a) | ψ_θ(x_b)⟩|²

    Assumptions
    -----------
    - x_a, x_b : preprocessed inputs (AT, V), shape (2,)
      • If NORMALIZE=="minmax": x ∈ [0,1]^2
      • If NORMALIZE=="zscore": standardized R²
    - theta : array-like of length 4
        (lambda1, lambda2, gamma, beta)
    - usdr_plus_state(x, theta, L, entangler) returns a normalized
      2-qubit statevector of shape (4,).
    """
    x_a = np.asarray(x_a, dtype=np.float64).ravel()
    x_b = np.asarray(x_b, dtype=np.float64).ravel()

    if x_a.shape != (2,):
        raise ValueError(f"[fidelity_2d] Expected x_a shape (2,), got {x_a.shape}")
    if x_b.shape != (2,):
        raise ValueError(f"[fidelity_2d] Expected x_b shape (2,), got {x_b.shape}")

    theta = np.asarray(theta, dtype=np.float64).ravel()
    if theta.shape != (4,):
        raise ValueError(f"[fidelity_2d] Expected theta shape (4,), got {theta.shape}")

    # Prepare states via USDR+ feature map
    psi_a = usdr_plus_state(x_a, theta, L=L, entangler=entangler)
    psi_b = usdr_plus_state(x_b, theta, L=L, entangler=entangler)

    # Sanity: ensure normalized states (numerical tolerance)
    # (not strictly necessary every call, but helpful during development)
    # You can comment this out later for speed if you wish.
    norm_a = np.linalg.norm(psi_a)
    norm_b = np.linalg.norm(psi_b)
    if not (np.isfinite(norm_a) and np.isfinite(norm_b)):
        raise RuntimeError("[fidelity_2d] Non-finite state norm encountered.")
    if not (np.allclose(norm_a, 1.0, atol=1e-6) and np.allclose(norm_b, 1.0, atol=1e-6)):
        print(
            f"[fidelity_2d] WARNING: state norms deviate from 1.0 "
            f"(‖ψ_a‖={norm_a:.6f}, ‖ψ_b‖={norm_b:.6f})"
        )

    # Inner product ⟨ψ_a | ψ_b⟩ = vdot(ψ_a, ψ_b)
    overlap = np.vdot(psi_a, psi_b)
    fidelity = float(np.abs(overlap) ** 2)

    return fidelity

# ===== CELL 045 (markdown) =====
# **Quantum Kernel Matrix Construction**

# ===== CELL 046 (code) =====
def _safe_psd_hygiene(
    K: np.ndarray,
    eps_floor: float = 1e-10,
    report: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Symmetrise K and ensure it is positive semi-definite by adding the smallest
    necessary jitter on the diagonal if negative eigenvalues are present.

    Parameters
    ----------
    K : np.ndarray
        Input kernel / Gram matrix, assumed square. Will be symmetrised.
    eps_floor : float, optional
        Additional safety margin added on top of |min_eig| when jittering.
    report : bool, optional
        If True, prints the amount of jitter added.

    Returns
    -------
    K_fixed : np.ndarray
        Symmetrised (and possibly jittered) kernel matrix.
    jitter_used : float
        The jitter value ε such that K_fixed = K_sym + ε I.
        Zero if no jitter was necessary.
    """
    K = np.asarray(K, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(
            f"[_safe_psd_hygiene] Expected square matrix, got shape {K.shape}"
        )

    # Symmetrise
    K_sym = 0.5 * (K + K.T)

    # Eigenvalues
    eigvals = np.linalg.eigvalsh(K_sym)
    min_eig = float(eigvals.min())

    if min_eig < 0.0:
        eps = abs(min_eig) + eps_floor
        if report:
            print(
                f"[PSD hygiene] Added jitter ε={eps:.2e} "
                f"(min_eig={min_eig:.2e}) to ensure PSD."
            )
        K_fixed = K_sym + eps * np.eye(K_sym.shape[0])
        return K_fixed, eps

    # Already PSD (within numerical tolerance)
    return K_sym, 0.0

# ===== CELL 047 (code) =====
def build_quantum_kernel_matrix(
    X1: np.ndarray,
    X2: np.ndarray,
    theta,
    *,
    L: int = 2,
    entangler: str = "cnot",
    apply_psd_hygiene_for_square: bool = True,
) -> np.ndarray:
    """
    Build the USDR+ fidelity kernel matrix between two sets of 2D inputs.

    Parameters
    ----------
    X1 : np.ndarray, shape (n1, 2)
        First set of preprocessed inputs (AT, V).
    X2 : np.ndarray, shape (n2, 2)
        Second set of preprocessed inputs (AT, V).
    theta : array-like of length 4
        (lambda1, lambda2, gamma, beta).
    L : int, optional
        Circuit depth (default: 2).
    entangler : str, optional
        Entangler type ("cnot" for this protocol).
    apply_psd_hygiene_for_square : bool, optional
        If True and X1, X2 have the same shape (square Gram matrix),
        apply PSD hygiene (symmetrize + jitter) before returning.

    Returns
    -------
    K : np.ndarray, shape (n1, n2)
        Kernel matrix, with
            K[i, j] = |⟨ψ_θ(X1[i]) | ψ_θ(X2[j])⟩|².
    """
    X1 = np.asarray(X1, dtype=np.float64)
    X2 = np.asarray(X2, dtype=np.float64)

    if X1.ndim != 2 or X1.shape[1] != 2:
        raise ValueError(f"[build_quantum_kernel_matrix] X1 must have shape (n1, 2), got {X1.shape}")
    if X2.ndim != 2 or X2.shape[1] != 2:
        raise ValueError(f"[build_quantum_kernel_matrix] X2 must have shape (n2, 2), got {X2.shape}")

    theta = np.asarray(theta, dtype=np.float64).ravel()
    if theta.shape != (4,):
        raise ValueError(f"[build_quantum_kernel_matrix] theta must have shape (4,), got {theta.shape}")

    n1, _ = X1.shape
    n2, _ = X2.shape
    K = np.empty((n1, n2), dtype=np.float64)

    # Simple double loop; N <= 200 so this is acceptable.
    for i in range(n1):
        for j in range(n2):
            K[i, j] = fidelity_2d(
                X1[i],
                X2[j],
                theta=theta,
                L=L,
                entangler=entangler,
            )

    # If this is a square Gram matrix, optionally apply PSD hygiene
    if apply_psd_hygiene_for_square and n1 == n2:
        K, jitter = _safe_psd_hygiene(K, eps_floor=1e-10, report=False)
        if jitter > 0.0:
            print(
                f"[build_quantum_kernel_matrix] PSD hygiene applied: "
                f"added jitter ε={jitter:.2e} to square kernel."
            )

    return K

@memory.cache
def build_kernel_matrix(
    X1: np.ndarray,
    X2: np.ndarray,
    theta,
    *,
    L: int = 2,
    entangler: str = "cnot",
    apply_psd_hygiene_for_square: bool = True,
) -> np.ndarray:
    """
    Cached wrapper around build_quantum_kernel_matrix for USDR+.

    This is the canonical entry point used by the KRR / spectral pipeline.

    Parameters
    ----------
    X1, X2 : np.ndarray
        Preprocessed 2D inputs (AT, V).
    theta : array-like of length 4
        (lambda1, lambda2, gamma, beta).
    L : int, optional
        Circuit depth (default: 2).
    entangler : str, optional
        Entangler type ("cnot" in this protocol).
    apply_psd_hygiene_for_square : bool, optional
        Whether to apply PSD hygiene for square matrices.

    Returns
    -------
    K : np.ndarray
        USDR+ fidelity kernel matrix between X1 and X2.
    """
    X1 = np.asarray(X1, dtype=np.float64)
    X2 = np.asarray(X2, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64).ravel()

    return build_quantum_kernel_matrix(
        X1,
        X2,
        theta=theta,
        L=L,
        entangler=entangler,
        apply_psd_hygiene_for_square=apply_psd_hygiene_for_square,
    )

# ===== CELL 048 (code) =====
# === 1) Choose CCPP 2D dataset and load preprocessed splits ===
N_example   = 100
SEED_example = 0

data_ccpp = load_processed_ccpp_2d_dataset(
    base_path=OUTPUT_DIR_CCPP,
    N=N_example,
    seed=SEED_example,
    normalize=NORMALIZE,
)

X_train = data_ccpp["X_train"]   # shape (n_train, 2) – (AT, V) preprocessed
y_train = data_ccpp["y_train"]   # shape (n_train,)
X_val   = data_ccpp["X_val"]     # shape (n_val, 2)
y_val   = data_ccpp["y_val"]     # shape (n_val,)
X_test  = data_ccpp["X_test"]    # shape (n_test, 2)
y_test  = data_ccpp["y_test"]    # shape (n_test,)

print(
    f"[CCPP-KERNEL] N={N_example}, SEED={SEED_example} | "
    f"X_train shape={X_train.shape}, X_val shape={X_val.shape}, "
    f"X_test shape={X_test.shape}"
)

# === 2) Choose θ (either a toy example or the result of KRR hyperparameter tuning) ===
# Example fixed params (replace with theta_opt from your optimiser later):
theta_example = np.array([2.1, 1.8, 2.3, 1.5], dtype=float)  # (lambda1, lambda2, gamma, beta)
theta = theta_example  # or theta_opt from optimize_theta_tau

# === 3) Build USDR+ fidelity kernel matrices (CCPP 2D) ===
# We use the cached wrapper build_kernel_matrix, which internally calls build_quantum_kernel_matrix.

K_train = build_kernel_matrix(
    X1=X_train,
    X2=X_train,
    theta=theta,
    L=2,
    entangler="cnot",
    apply_psd_hygiene_for_square=True,   # K_train is square → PSD hygiene
)
K_val = build_kernel_matrix(
    X1=X_val,
    X2=X_train,
    theta=theta,
    L=2,
    entangler="cnot",
    apply_psd_hygiene_for_square=False,  # rectangular, no PSD hygiene
)
K_test = build_kernel_matrix(
    X1=X_test,
    X2=X_train,
    theta=theta,
    L=2,
    entangler="cnot",
    apply_psd_hygiene_for_square=False,  # rectangular, no PSD hygiene
)

print(
    f"[CCPP-KERNEL] K_train shape={K_train.shape}, "
    f"K_val shape={K_val.shape}, K_test shape={K_test.shape}"
)

# ===== CELL 049 (markdown) =====
# **Condition numbers**

# ===== CELL 050 (code) =====
print(f"K_train shape: {K_train.shape}, "
      f"condition number: {np.linalg.cond(K_train):.2e}")

# For K_val and K_test, cond is SVD-based (rectangular),
# not the PSD "Gram" cond we care about for inversion:
print(f"K_val   shape: {K_val.shape},   cond (SVD): {np.linalg.cond(K_val):.2e}")
print(f"K_test  shape: {K_test.shape},  cond (SVD): {np.linalg.cond(K_test):.2e}")

# ===== CELL 051 (markdown) =====
# **True object used in KRR**

# ===== CELL 052 (code) =====
# --- 1) Load a CCPP 2D split (AT, V → EP) ---
N_example    = 100
SEED_example = 0

data_ccpp = load_processed_ccpp_2d_dataset(
    base_path=OUTPUT_DIR_CCPP,   # e.g. Path("preprocessed/ccpp")
    N=N_example,
    seed=SEED_example,
    normalize=NORMALIZE,         # "minmax" in your config
)

X_train = data_ccpp["X_train"]   # shape (n_train, 2)
y_train = data_ccpp["y_train"]   # shape (n_train,)

print(
    f"[CCPP-KRR] N={N_example}, SEED={SEED_example} | "
    f"X_train shape={X_train.shape}, y_train shape={y_train.shape}"
)

# --- 2) Build the USDR+ Gram matrix on CCPP (AT, V) ---
theta_example = np.array([2.1, 1.8, 2.3, 1.5], dtype=float)  # or theta_opt from optimisation

K_train = build_kernel_matrix(
    X1=X_train,
    X2=X_train,
    theta=theta_example,
    L=2,
    entangler="cnot",
    apply_psd_hygiene_for_square=True,  # ensures PSD + tiny jitter if needed
)

print(f"[CCPP-KRR] K_train shape={K_train.shape}")

# --- 3) Add τI regularisation and inspect condition number ---
tau = 1e-3  # or your optimised tau_opt for this run

K_reg = K_train + tau * np.eye(K_train.shape[0])
cond_K_reg = np.linalg.cond(K_reg)

print(f"[CCPP-KRR] cond(K_train + τI) with τ={tau:.1e}: {cond_K_reg:.2e}")

# ===== CELL 053 (markdown) =====
# **Analyze kernel matrix**

# ===== CELL 054 (code) =====
def analyze_kernel_matrix(
    K: np.ndarray,
    name: str = "K_train",
    width: float = 20,
    height: float = 10,
    dpi: int = 300,
    plot: bool = True,
    save: bool = True,
    save_dir: str | Path = "figures/ccpp/diagnostics",
) -> Dict[str, Any]:
    """
    Analyze a quantum kernel matrix built from U_{SDR+} in the CCPP 2D setup.

    CCPP + USDR+ context
    --------------------
    • K is typically one of:
        - K_train (square Gram matrix over CCPP (AT, V) train points),
        - K_train + τI (regularised Gram),
        - K_val, K_test (rectangular kernel blocks against the train set).

    • Square K (e.g. K_train, K_train + τI) is a Gram matrix:
        K[i,j] = |⟨ψ_θ(x_i) | ψ_θ(x_j)⟩|²
      and should be PSD up to numerical noise.

    • Rectangular K (e.g. K_val, K_test) is a kernel block (n_val/test × n_train)
      and is never inverted, so only the SVD-based condition number is meaningful.

    Parameters
    ----------
    K : np.ndarray
        Kernel matrix (square or rectangular).
    name : str, optional
        Label used in logs and plot titles.
    width, height : float, optional
        Figure size if plot=True.
    dpi : int, optional
        Figure DPI if plot=True.
    plot : bool, optional
        If True, plot eigenvalue spectrum (for square matrices).

    Returns
    -------
    stats : dict
        For square matrices:
            {
                "min_eig", "max_eig", "cond", "is_psd", "eigvals", "jitter"
            }
        For rectangular matrices:
            {
                "cond", "note": "rectangular_matrix"
            }
    """
    K = np.asarray(K, dtype=np.float64)

    print(f"\n=== {name} ANALYSIS ===")
    print(f"Shape: {K.shape}")

    # --- Square Gram matrix: full PSD + spectrum analysis -----------------
    if K.shape[0] == K.shape[1]:
        # Apply the same PSD hygiene used in the USDR+ protocol
        K_fixed, jitter = _safe_psd_hygiene(K, eps_floor=1e-10, report=False)

        # Symmetrise again just to be extra safe for eigvalsh
        K_sym = 0.5 * (K_fixed + K_fixed.T)

        eigvals = np.linalg.eigvalsh(K_sym)
        min_eig = float(eigvals.min())
        max_eig = float(eigvals.max())
        cond_number = float(np.linalg.cond(K_sym))

        is_psd = min_eig >= -1e-10

        print(f"Eigenvalue range: [{min_eig:.3e}, {max_eig:.3e}]")
        print(
            f"Min eigenvalue:   {min_eig:.3e} → "
            f"PSD: {'YES' if is_psd else 'NO (jitter needed)'}"
        )
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


# === Spectral diagnostics: effective rank, κ, eigen-range (CCPP + USDR+) ===

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

    This is used to characterise the "complexity regime" of USDR+ kernels
    on the CCPP 2D (AT, V) regression problem.
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
    Compute spectral diagnostics for a (square) kernel / Gram matrix K
    in the CCPP + USDR+ experiment:

        • min_eig, max_eig
        • 2-norm condition number κ
        • effective rank r_eff

    Parameters
    ----------
    K : np.ndarray
        Square kernel (Gram) matrix, e.g. K_train or K_train + τI.
    name : str, optional
        Name used in log messages (e.g. "K_train", "K_reg").
    psd_tol : float, optional
        Tolerance for considering the matrix PSD. Eigenvalues below
        -psd_tol are treated as a warning.
    log_prefix : str, optional
        Prefix for printed log messages (e.g. "[SPEC-CCPP]").

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

    # Symmetrise to be robust against tiny asymmetries
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

# ===== CELL 055 (code) =====
K_train_stats = analyze_kernel_matrix(K_train, name="K_train_CCPP")
spec_metrics  = compute_spectrum_metrics(K_train, name="K_train_CCPP", log_prefix="[SPEC-CCPP]")

# ===== CELL 056 (markdown) =====
# **Plot gram matrix**

# ===== CELL 057 (code) =====
def plot_gram_matrix(
    K: np.ndarray,
    title: str = "Gram matrix heatmap",
    cmap: str = "viridis",
    dpi: int = 300,
    width: float = 10.0,
    height: float = 8.0,
    annotate: bool = False,
    save: bool = True,
    save_dir: str | Path = "figures/ccpp/diagnostics",
) -> None:
    """
    Plot a Gram / kernel matrix as a heatmap and highlight the diagonal.

    Parameters
    ----------
    K : np.ndarray
        Kernel (Gram) matrix, typically K_train or K_train + τI.
    title : str
        Plot title.
    cmap : str
        Matplotlib / seaborn colormap name.
    dpi : int
        Figure DPI.
    width, height : float
        Figure size in inches.
    annotate : bool
        If True, write the numeric values in each cell (only practical for
        small matrices).
    """
    K = np.asarray(K, dtype=float)

    sns.set_theme(style="white", context="talk")

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

    sns.heatmap(
        K,
        ax=ax,
        cmap=cmap,
        annot=annotate,
        fmt=".2f",
        square=True,
        cbar=True,
        linewidths=0.2,
        linecolor="gray",
    )

    # Highlight diagonal with red boxes
    n = K.shape[0]
    for i in range(n):
        ax.add_patch(
            plt.Rectangle(
                (i, i),          # (x, y) of lower-left corner in heatmap coords
                1, 1,            # width, height (one cell)
                fill=False,
                edgecolor="red",
                linewidth=2.0,
            )
        )

    ax.set_title(title, fontsize=18, pad=14)
    ax.set_xlabel("Column index (j)")
    ax.set_ylabel("Row index (i)")

    plt.tight_layout()
    if save:
        slug = "_".join(title.strip().lower().split())
        out = Path(save_dir) / f"gram_heatmap_{slug or 'matrix'}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"[PLOT] Saved Gram matrix heatmap -> {out}")
    plt.show()

# ===== CELL 058 (code) =====
plot_gram_matrix(
    K_train,
    title="Gram matrix heatmap – CCPP, K_train",
    cmap="viridis",
    dpi=300,
    width=20,
    height=16,
    annotate=False,
)

# ===== CELL 059 (code) =====
def apply_psd_hygiene(
    K: np.ndarray,
    tau: float,
    name: str = "K_train",
    psd_tol: float = 1e-10,
    eps_floor: float = 1e-10,
    analyze_raw: bool = True,
    plot_raw: bool = False,
    plot_reg: bool = False,
) -> tuple[np.ndarray, float, dict, float]:
    """
    Apply PSD hygiene and τ-regularisation to a Gram matrix K for KRR.

    Steps
    -----
    1. Optionally analyse the *raw* K with `analyze_kernel_matrix`
       (logs + optional eigenvalue spectrum).
    2. Form K_reg = K + tau * I.
    3. Run `_safe_psd_hygiene` on K_reg to fix any tiny negative
       eigenvalues (adds extra jitter ε if needed).
    4. Compute spectrum metrics on K_reg with `compute_spectrum_metrics`.

    Parameters
    ----------
    K : np.ndarray
        Square Gram matrix (kernel on the training set).
    tau : float
        KRR regularisation parameter τ (added on the diagonal).
    name : str, optional
        Name used in logs and plots.
    psd_tol : float, optional
        Tolerance for PSD check in `compute_spectrum_metrics`.
    eps_floor : float, optional
        Floor used by `_safe_psd_hygiene` when repairing K_reg.
    analyze_raw : bool, optional
        If True, call `analyze_kernel_matrix` on raw K.
    plot_raw : bool, optional
        If True, plot eigenvalue spectrum for raw K.
    plot_reg : bool, optional
        If True, plot eigenvalue spectrum for K_reg after hygiene.

    Returns
    -------
    K_reg : np.ndarray
        Regularised and PSD-fixed Gram matrix ready for inversion.
    jitter : float
        Extra jitter ε added by `_safe_psd_hygiene` on top of τ.
    K_stats : dict
        Spectral diagnostics of the *raw* K (min_eig, max_eig, cond, rank_eff).
    kappa_after : float
        Condition number κ of K_reg.
    """
    K = np.asarray(K, dtype=np.float64)
    n = K.shape[0]
    if K.ndim != 2 or n != K.shape[1]:
        raise ValueError("apply_psd_hygiene expects a square Gram matrix K.")

    # 1) Analyse raw K (logs + optional spectrum)
    if analyze_raw:
        raw_analysis = analyze_kernel_matrix(
            K,
            name=name,
            width=30,
            height=12,
            dpi=300,
            plot=plot_raw,
        )
        # raw_analysis contains min_eig, max_eig, cond, eigvals, jitter (used inside)
        K_stats = compute_spectrum_metrics(
            K,
            name=name,
            psd_tol=psd_tol,
            log_prefix="[SPEC-RAW]",
        )
    else:
        K_stats = compute_spectrum_metrics(
            K,
            name=name,
            psd_tol=psd_tol,
            log_prefix="[SPEC-RAW]",
        )

    # 2) Add τI
    tau = float(tau)
    print(f"[PSD] Applying τ-regularisation with τ = {tau:.3e} to {name}")
    K_reg = K + tau * np.eye(n, dtype=np.float64)

    # 3) Ensure PSD of K_reg via tiny extra jitter if needed
    K_reg_fixed, jitter = _safe_psd_hygiene(
        K_reg,
        eps_floor=eps_floor,
        report=True,
    )
    K_reg = K_reg_fixed

    # 4) Spectrum metrics AFTER regularisation
    reg_name = f"{name} + τI"
    reg_stats = compute_spectrum_metrics(
        K_reg,
        name=reg_name,
        psd_tol=psd_tol,
        log_prefix="[SPEC-REG]",
    )
    kappa_after = reg_stats["kappa"]

    # Optional: plot eigenvalue spectrum of K_reg
    if plot_reg:
        eigvals_reg = np.linalg.eigvalsh(0.5 * (K_reg + K_reg.T))
        plt.figure(figsize=(30, 12), dpi=300)
        plt.semilogy(
            np.sort(eigvals_reg)[::-1],
            "o-",
            markersize=4,
        )
        plt.title(f"Eigenvalue Spectrum of {reg_name}")
        plt.xlabel("Index (largest to smallest)")
        plt.ylabel("Eigenvalue")
        plt.grid(True, alpha=0.3, which="both", ls="--")
        plt.tight_layout()
        plt.show()

    return K_reg, float(jitter), K_stats, float(kappa_after)

# ===== CELL 060 (markdown) =====
# **PSD hygiene test**

# ===== CELL 061 (code) =====
N_example = 100
SEED_example = 0

# Load preprocessed CCPP 2D dataset
data_example = load_processed_ccpp_2d_dataset(
    base_path=OUTPUT_DIR_CCPP,
    N=N_example,
    seed=SEED_example,
    normalize=NORMALIZE,
)

X_train = data_example["X_train"]
y_train = data_example["y_train"]

print(
    f"[CCPP-TEST] X_train shape={X_train.shape}, "
    f"y_train shape={y_train.shape}"
)

# Example USDR+ parameters (or replace with theta_opt once you optimise)
theta_example = np.array([2.1, 1.8, 2.3, 1.5], dtype=float)

# Build training Gram matrix
K_train = build_quantum_kernel_matrix(
    X_train,
    X_train,
    theta_example,
)

# Apply PSD hygiene with a provisional τ
tau_test = 1e-3

K_reg, jitter, K_stats, kappa_after = apply_psd_hygiene(
    K_train,
    tau=tau_test,
    name="K_train_CCPP",
    analyze_raw=True,
    plot_raw=True,   # set False if you don't want the raw spectrum plot
    plot_reg=True,   # this will plot the spectrum of K_reg
)

print("\n[CCPP-TEST] Raw K_stats:")
for k, v in K_stats.items():
    print(f"  {k}: {v:.3e}" if isinstance(v, float) else f"  {k}: {v}")

print(f"\n[CCPP-TEST] Extra jitter ε added on top of τ: {jitter:.3e}")
print(f"[CCPP-TEST] Condition number κ(K_reg): {kappa_after:.3e}")

# ===== CELL 062 (markdown) =====
# # **KRR training pipeline**

# ===== CELL 063 (markdown) =====
# **Hyperparameter Optimization (Joint θ + τ)**

# ===== CELL 064 (code) =====
def krr_val_objective(
    log_params: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    L: int = 2,
    entangler: str = "cnot",
    large_penalty: float = 1e6,
) -> float:
    """
    Kernel Ridge Regression validation objective for the USDR+ fidelity kernel
    on the 2D CCPP subproblem (AT, V -> EP).

    Parameters
    ----------
    log_params : array-like, shape (5,)
        Log-parameters:
            log([lambda1, lambda2, gamma, beta, tau]).
        The optimizer works in log-space; here we exponentiate.
    X_train : np.ndarray, shape (n_train, 2)
        Preprocessed training inputs (e.g. MinMax-scaled AT, V).
    y_train : np.ndarray, shape (n_train,)
        Training targets (EP).
    X_val : np.ndarray, shape (n_val, 2)
        Preprocessed validation inputs.
    y_val : np.ndarray, shape (n_val,)
        Validation targets.
    L : int, optional
        Circuit depth for U_SDR_plus (default: 2).
    entangler : str, optional
        Entangling gate used in U_SDR_plus (default: "cnot").
    large_penalty : float, optional
        Value returned if the objective cannot be evaluated
        (e.g. linear system is singular, NaNs, etc.).

    Returns
    -------
    float
        Validation MSE for the given parameters, or `large_penalty`
        if something goes wrong numerically.
    """
    # --- 1. Unpack and exponentiate parameters -----------------------------
    log_params = np.asarray(log_params, dtype=float).ravel()
    if log_params.size != 5:
        raise ValueError(
            f"krr_val_objective expects 5 log-params, got shape {log_params.shape}"
        )

    lambda1, lambda2, gamma, beta, tau = np.exp(log_params)

    # Safety guards (optional, light clipping)
    # gamma should be >= 1 by design
    gamma = max(gamma, 1.0)
    # tau must be strictly positive to regularise the system
    tau = max(tau, 1e-12)

    theta = np.array([lambda1, lambda2, gamma, beta], dtype=float)

    # --- 2. Build kernel (Gram) matrices -----------------------------------
    try:
        # K_train is square; we let build_kernel_matrix apply PSD hygiene
        K_train = build_kernel_matrix(
            X_train,
            X_train,
            theta=theta,
            L=L,
            entangler=entangler,
            apply_psd_hygiene_for_square=True,
        )

        # K_val is rectangular: (n_val x n_train), no PSD hygiene needed
        K_val = build_kernel_matrix(
            X_val,
            X_train,
            theta=theta,
            L=L,
            entangler=entangler,
            apply_psd_hygiene_for_square=False,
        )
    except Exception as exc:
        print(f"[KRR-VAL] Kernel computation failed: {exc}")
        return float(large_penalty)

    # --- 3. Form regularised Gram matrix -----------------------------------
    n_train = K_train.shape[0]
    K_reg = K_train + tau * np.eye(n_train, dtype=float)

    # Symmetrise before solving to reduce numerical asymmetries
    K_reg = 0.5 * (K_reg + K_reg.T)

    # --- 4. Solve for alpha -------------------------------------------------
    y_train = np.asarray(y_train, dtype=float).ravel()
    y_val = np.asarray(y_val, dtype=float).ravel()

    if y_train.shape[0] != n_train:
        raise ValueError(
            f"y_train length {y_train.shape[0]} does not match K_train size {n_train}"
        )

    try:
        alpha = np.linalg.solve(K_reg, y_train)
    except np.linalg.LinAlgError as exc:
        print(f"[KRR-VAL] Linear solve failed (possibly ill-conditioned): {exc}")
        return float(large_penalty)

    # --- 5. Predict on validation set --------------------------------------
    y_val_pred = K_val @ alpha

    # --- 6. Compute validation MSE -----------------------------------------
    diff = y_val_pred - y_val
    mse = float(np.mean(diff * diff))

    # Guard against NaNs / inf
    if not np.isfinite(mse):
        print(f"[KRR-VAL] Non-finite MSE encountered: {mse}")
        return float(large_penalty)

    return mse

# ===== CELL 065 (code) =====
data = load_processed_ccpp_2d_dataset(
    base_path=OUTPUT_DIR_CCPP,
    N=100,
    seed=0,
    normalize=NORMALIZE,
)

X_train = data["X_train"]
y_train = data["y_train"]
X_val   = data["X_val"]
y_val   = data["y_val"]

def objective(log_params):
    return krr_val_objective(
        log_params,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        L=depth,
        entangler=entangler,
    )

# ===== CELL 066 (code) =====
def optimize_theta_tau(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    L: int = 2,
    entangler: str = "cnot",
    n_optuna_trials: int = 100,
) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
    """
    Constrained optimisation of USDR+ hyperparameters (λ1, λ2, γ, β, τ)
    for the CCPP 2D subproblem using KRR validation MSE.

    Phase 1: L-BFGS-B in log-space with box constraints.
    Phase 2: (optional) Optuna TPE fallback with same ranges.

    Parameters
    ----------
    X_train, y_train, X_val, y_val :
        Preprocessed inputs and targets for train / validation.
    L : int, optional
        Circuit depth for U_SDR_plus (default: 2).
    entangler : str, optional
        Entangling gate for U_SDR_plus (default: "cnot").
    n_optuna_trials : int, optional
        Number of Optuna trials if the L-BFGS-B phase fails or is unsatisfactory.

    Returns
    -------
    theta_opt : np.ndarray, shape (4,)
        Optimised quantum kernel parameters [λ1, λ2, γ, β].
    tau_opt : float
        Optimised KRR regularisation parameter τ.
    val_mse_opt : float
        Best validation MSE achieved.
    info : dict
        Diagnostic information:
            {
              "method": "lbfgs" or "optuna",
              "lbfgs_success": bool,
              "lbfgs_message": str,
              "lbfgs_val_mse": float or None,
              "optuna_used": bool,
              "optuna_val_mse": float or None,
            }
    """

    # ------------------------------------------------------------------ #
    # 1. Hyperparameter ranges (constrained regime)
    # ------------------------------------------------------------------ #
    bounds_linear = {
        "lambda1": (0.1, 5.0),
        "lambda2": (0.1, 5.0),
        "gamma":   (1.5, 5.0),
        "beta":    (0.5, 3.0),
        "tau":     (1e-8, 1e2),
    }

    # Log-space bounds for L-BFGS-B
    bounds_log = [
        (np.log(bounds_linear["lambda1"][0]), np.log(bounds_linear["lambda1"][1])),
        (np.log(bounds_linear["lambda2"][0]), np.log(bounds_linear["lambda2"][1])),
        (np.log(bounds_linear["gamma"][0]),   np.log(bounds_linear["gamma"][1])),
        (np.log(bounds_linear["beta"][0]),    np.log(bounds_linear["beta"][1])),
        (np.log(bounds_linear["tau"][0]),     np.log(bounds_linear["tau"][1])),
    ]

    # Initial guess (linear space), then log them
    lambda1_0 = 1.0
    lambda2_0 = 1.0
    gamma_0   = 2.0
    beta_0    = 1.0
    tau_0     = 1e-3

    x0_linear = np.array(
        [lambda1_0, lambda2_0, gamma_0, beta_0, tau_0],
        dtype=float,
    )
    x0_log = np.log(x0_linear)

    # ------------------------------------------------------------------ #
    # 2. Define objective wrapper for L-BFGS-B
    # ------------------------------------------------------------------ #
    def objective_log_params(log_params: np.ndarray) -> float:
        return krr_val_objective(
            log_params,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            L=L,
            entangler=entangler,
        )

    # ------------------------------------------------------------------ #
    # 3. Run L-BFGS-B optimisation in log-space
    # ------------------------------------------------------------------ #
    print("\n[OPT] Starting L-BFGS-B optimisation in log-space for (λ1, λ2, γ, β, τ)...")

    res = minimize(
        objective_log_params,
        x0_log,
        method="L-BFGS-B",
        bounds=bounds_log,
        tol=1e-6,
        options={"maxiter": 200, "disp": False},
    )

    info: Dict[str, Any] = {
        "method": "lbfgs",
        "lbfgs_success": bool(res.success),
        "lbfgs_message": str(res.message),
        "lbfgs_val_mse": float(res.fun) if np.isfinite(res.fun) else None,
        "optuna_used": False,
        "optuna_val_mse": None,
    }

    theta_opt = None
    tau_opt = None
    val_mse_opt = np.inf

    if res.success and np.isfinite(res.fun):
        # Exponentiate to get back to linear scale
        lambda1_opt, lambda2_opt, gamma_opt, beta_opt, tau_opt_val = np.exp(res.x)

        # Safety clipping into linear bounds in case of tiny numerical drift
        lambda1_opt = np.clip(lambda1_opt, *bounds_linear["lambda1"])
        lambda2_opt = np.clip(lambda2_opt, *bounds_linear["lambda2"])
        gamma_opt   = np.clip(gamma_opt,   *bounds_linear["gamma"])
        beta_opt    = np.clip(beta_opt,    *bounds_linear["beta"])
        tau_opt_val = np.clip(tau_opt_val, *bounds_linear["tau"])

        theta_opt = np.array(
            [lambda1_opt, lambda2_opt, gamma_opt, beta_opt],
            dtype=float,
        )
        tau_opt = float(tau_opt_val)
        val_mse_opt = float(res.fun)

        print(
            "[OPT] L-BFGS-B succeeded\n"
            f"      λ1={theta_opt[0]:.3f}, λ2={theta_opt[1]:.3f}, "
            f"γ={theta_opt[2]:.3f}, β={theta_opt[3]:.3f}, τ={tau_opt:.3e}\n"
            f"      Val MSE = {val_mse_opt:.4e}"
        )
    else:
        print("[OPT] L-BFGS-B failed or returned non-finite MSE.")
        print(f"      message: {res.message}")

    # ------------------------------------------------------------------ #
    # 4. Optional Optuna fallback (only if L-BFGS-B unsatisfactory)
    # ------------------------------------------------------------------ #
    use_fallback = not (res.success and np.isfinite(res.fun))

    if use_fallback:
        try:
            import optuna

            info["method"] = "optuna"
            info["optuna_used"] = True

            print("\n[OPT] Falling back to Optuna TPE optimisation...")

            def optuna_objective(trial: "optuna.trial.Trial") -> float:
                # Sample in linear space within the same constrained ranges
                lambda1 = trial.suggest_float("lambda1", *bounds_linear["lambda1"])
                lambda2 = trial.suggest_float("lambda2", *bounds_linear["lambda2"])
                gamma   = trial.suggest_float("gamma",   *bounds_linear["gamma"])
                beta    = trial.suggest_float("beta",    *bounds_linear["beta"])
                tau     = trial.suggest_float("tau",     *bounds_linear["tau"], log=True)

                log_params = np.log([lambda1, lambda2, gamma, beta, tau])

                mse = krr_val_objective(
                    log_params,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    L=L,
                    entangler=entangler,
                )
                return mse

            study = optuna.create_study(direction="minimize")
            study.optimize(optuna_objective, n_trials=n_optuna_trials, show_progress_bar=False)

            best_params = study.best_params
            val_mse_opt = float(study.best_value)

            theta_opt = np.array(
                [
                    best_params["lambda1"],
                    best_params["lambda2"],
                    best_params["gamma"],
                    best_params["beta"],
                ],
                dtype=float,
            )
            tau_opt = float(best_params["tau"])

            info["optuna_val_mse"] = val_mse_opt

            print(
                "[OPT] Optuna best parameters\n"
                f"      λ1={theta_opt[0]:.3f}, λ2={theta_opt[1]:.3f}, "
                f"γ={theta_opt[2]:.3f}, β={theta_opt[3]:.3f}, τ={tau_opt:.3e}\n"
                f"      Val MSE = {val_mse_opt:.4e}"
            )

        except ImportError:
            print(
                "[OPT] Optuna is not installed; cannot run fallback optimisation.\n"
                "      Please install `optuna` or rely on the L-BFGS-B result."
            )
            # If L-BFGS-B result is unusable, raise to signal a hard failure
            if theta_opt is None or tau_opt is None or not np.isfinite(val_mse_opt):
                raise RuntimeError(
                    "Both L-BFGS-B optimisation failed and Optuna is unavailable."
                )

    return theta_opt, tau_opt, val_mse_opt, info

# ===== CELL 067 (code) =====
# Load one (N, SEED) split
data = load_processed_ccpp_2d_dataset(
    base_path=OUTPUT_DIR_CCPP,
    N=100,
    seed=0,
    normalize=NORMALIZE,
)

X_train = data["X_train"]
y_train = data["y_train"]
X_val   = data["X_val"]
y_val   = data["y_val"]

theta_opt, tau_opt, val_mse_opt, opt_info = optimize_theta_tau(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    L=depth,
    entangler=entangler,
)

print("\n[SUMMARY] Optimisation info:", opt_info)

# ===== CELL 068 (code) =====
def run_usdr_plus_ccpp_2d_constrained_experiments(
    sample_sizes: List[int],
    seeds: List[int],
    base_path: Path,
    normalize: str = "minmax",
    csv_out: str = "csv/ccpp/usdr_plus_ccpp_2d_results_constrained.csv",
) -> pd.DataFrame:
    """
    Run USDR+ (constrained hyperparameter ranges) on the CCPP 2D subproblem
    (inputs: AT, V; target: EP) for all N × SEED combinations.

    Parameters
    ----------
    sample_sizes : list of int
        e.g. [50, 100, 200]
    seeds : list of int
        e.g. [0, 1, 2]
    base_path : Path
        Base directory where processed CCPP 2D splits live,
        e.g. Path("preprocessed/ccpp")
    normalize : str
        "minmax" or "zscore" (must match how splits were created).
    csv_out : str
        Path/name of the CSV file to write the aggregated results.

    Returns
    -------
    df_results : pd.DataFrame
        DataFrame with one row per (N, SEED) containing:
          - N, SEED, lambda1, lambda2, gamma, beta, tau
          - val_mse, test_mse
          - min_eig_train, max_eig_train, kappa_train, rank_eff_train
          - min_eig_reg,   max_eig_reg,   kappa_reg,   rank_eff_reg
          - jitter
          - dataset, experiment, normalize, n_train, n_val, n_test
    """
    results: List[Dict[str, Any]] = []

    for N in sample_sizes:
        for seed in seeds:
            print("\n" + "=" * 80)
            print(f"[CCPP-USDR+] N={N}, SEED={seed} | constrained hyperparams, 2D subproblem")
            print("=" * 80)

            # 1) Reproducibility
            set_all_seeds(seed)

            # 2) Load preprocessed 2D dataset (AT, V → EP)
            data = load_processed_ccpp_2d_dataset(
                base_path=base_path,
                N=N,
                seed=seed,
                normalize=normalize,
            )

            X_train = data["X_train"]
            y_train = data["y_train"]
            X_val   = data["X_val"]
            y_val   = data["y_val"]
            X_test  = data["X_test"]
            y_test  = data["y_test"]
            meta    = data["metadata"]

            n_train = meta["n_train"]
            n_val   = meta["n_val"]
            n_test  = meta["n_test"]

            print(
                f"[CCPP-USDR+] Shapes – "
                f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}"
            )
            print(
                f"[CCPP-USDR+] Split sizes – "
                f"n_train={n_train}, n_val={n_val}, n_test={n_test}"
            )

            # 3) Hyperparameter optimisation (λ1, λ2, γ, β, τ)
            theta_opt, tau_opt, val_mse, opt_info = optimize_theta_tau(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
            )

            lambda1, lambda2, gamma, beta = theta_opt
            print(
                "[CCPP-USDR+] Optimised hyperparameters:\n"
                f"  λ1={lambda1:.3f}, λ2={lambda2:.3f}, γ={gamma:.3f}, β={beta:.3f}, "
                f"τ={tau_opt:.3e}\n"
                f"  Val MSE = {val_mse:.4e}"
            )
            print(f"[CCPP-USDR+] Optimiser info: {opt_info}")

            # 4) Build Gram matrices with optimal θ
            print("[CCPP-USDR+] Building Gram matrices with θ_opt...")
            K_train = build_kernel_matrix(
                X_train,
                X_train,
                theta_opt,
                L=2,
                entangler="cnot",
                apply_psd_hygiene_for_square=True,   # ensure PSD for Gram
            )
            K_test = build_kernel_matrix(
                X_test,
                X_train,
                theta_opt,
                L=2,
                entangler="cnot",
                apply_psd_hygiene_for_square=False,  # rectangular block
            )

            print(
                f"[CCPP-USDR+] K_train shape={K_train.shape}, "
                f"K_test shape={K_test.shape}"
            )

            # 5) Spectral diagnostics on raw K_train
            spec_train = compute_spectrum_metrics(
                K_train,
                name=f"K_train (N={N}, SEED={seed})",
                log_prefix="[SPEC-CCPP-RAW]",
            )

            # 6) PSD hygiene + τ-regularisation (K + τI)
            K_reg, jitter, K_stats_raw, kappa_after = apply_psd_hygiene(
                K_train,
                tau=tau_opt,
                name=f"K_train (N={N}, SEED={seed}) [CCPP]",
            )

            # Optional: spectrum of K_reg as well
            spec_reg = compute_spectrum_metrics(
                K_reg,
                name=f"K_reg (N={N}, SEED={seed})",
                log_prefix="[SPEC-CCPP-REG]",
            )

            # 7) Solve KRR system and evaluate on test
            print("[CCPP-USDR+] Solving KRR system and evaluating on test...")
            alpha = np.linalg.solve(K_reg, y_train)

            # If you have krr_predict(K_block, alpha) you can use that instead:
            # y_pred_test = krr_predict(K_test, alpha)
            y_pred_test = K_test @ alpha

            test_mse = mean_squared_error(y_test, y_pred_test)

            print(
                "[CCPP-USDR+] Summary for this run:\n"
                f"  Val MSE      = {val_mse:.4e}\n"
                f"  Test MSE     = {test_mse:.4e}\n"
                f"  κ(K_train)   = {spec_train['kappa']:.3e}\n"
                f"  κ(K+τI)      = {spec_reg['kappa']:.3e}\n"
                f"  jitter (ε)   = {jitter:.2e}"
            )

            # 8) Collect all metrics for this (N, SEED)
            results.append(
                {
                    "N": N,
                    "SEED": seed,
                    "lambda1": float(lambda1),
                    "lambda2": float(lambda2),
                    "gamma": float(gamma),
                    "beta": float(beta),
                    "tau": float(tau_opt),
                    "val_mse": float(val_mse),
                    "test_mse": float(test_mse),

                    # Raw K_train metrics
                    "min_eig_train": float(spec_train["min_eig"]),
                    "max_eig_train": float(spec_train["max_eig"]),
                    "kappa_train": float(spec_train["kappa"]),
                    "rank_eff_train": float(spec_train["rank_eff"]),

                    # Regularised K_reg metrics
                    "min_eig_reg": float(spec_reg["min_eig"]),
                    "max_eig_reg": float(spec_reg["max_eig"]),
                    "kappa_reg": float(spec_reg["kappa"]),
                    "rank_eff_reg": float(spec_reg["rank_eff"]),

                    # Extra numerical info
                    "jitter": float(jitter),

                    # Dataset / experiment identifiers
                    "dataset": "ccpp",
                    "experiment": "usdr_plus_constrained_ccpp_2d",
                    "normalize": normalize,
                    "n_train": int(n_train),
                    "n_val": int(n_val),
                    "n_test": int(n_test),
                }
            )

    # 9) Aggregate into DataFrame and save
    df_results = pd.DataFrame(results)
    csv_path = Path(csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print(f"[CCPP-USDR+] All runs completed. Results saved to: {csv_path}")
    print("=" * 80)

    return df_results

# ===== CELL 069 (code) =====
df_results_ccpp = run_usdr_plus_ccpp_2d_constrained_experiments(
    sample_sizes=sample_sizes,
    seeds=SEEDS,
    base_path=OUTPUT_DIR_CCPP,
    normalize=NORMALIZE,
    csv_out="csv/ccpp/usdr_plus_ccpp_2d_results_constrained.csv",
)

df_results_ccpp.head()

# ===== CELL 070 (markdown) =====
# **Save results to CSV**

# ===== CELL 071 (code) =====
from pathlib import Path
from typing import Optional
import pandas as pd

def summarize_usdr_plus_ccpp_constrained_results(
    csv_path: str | Path = "csv/ccpp/usdr_plus_ccpp_2d_results_constrained.csv",
    df: Optional[pd.DataFrame] = None,
    experiment_name: str = "usdr_plus_constrained_ccpp_2d",
) -> None:
    """
    Summarize USDR+ constrained results for the CCPP 2D experiment.

    Parameters
    ----------
    csv_path
        Path to the CSV file containing the results.
    df
        Optional pre-loaded DataFrame. If provided, `csv_path` is ignored.
    experiment_name
        Value of the 'experiment' column to filter on (if present).
        For the current CCPP 2D constrained run this should be
        'usdr_plus_constrained_ccpp_2d'.
    """
    # --- 1) Load results ---
    if df is None:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"[SUMMARY-CCPP-CONSTR] Could not find CSV at: {csv_path}"
            )
        df = pd.read_csv(csv_path)

    # Ensure we only look at the desired experiment, if the column exists
    if "experiment" in df.columns:
        df = df[df["experiment"] == experiment_name]

    if df.empty:
        print(
            f"[SUMMARY-CCPP-CONSTR] No rows found for experiment='{experiment_name}'."
        )
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
            "[SUMMARY-CCPP-CONSTR] Missing expected columns in results DataFrame: "
            + ", ".join(sorted(missing))
        )

    # --- 2) Metrics to summarize ---
    metrics = [
        "val_mse",
        "test_mse",
        "kappa_reg",
        "rank_eff_train",
    ]

    # --- 3) Per-N aggregates (mean ± std over SEEDS) ---
    print("=== USDR+ (CONSTRAINED RANGES, CCPP 2D) – PER-N SUMMARY ===\n")
    for N, dfN in df.groupby("N"):
        print(f"--- N = {N} ---")
        for m in metrics:
            mu = dfN[m].mean()
            std = dfN[m].std(ddof=1)  # sample std
            print(f"{m:15s}: {mu:.4e} ± {std:.4e}")
        print()

# ===== CELL 072 (code) =====
summarize_usdr_plus_ccpp_constrained_results()

# ===== CELL 073 (markdown) =====
# # $U_{\text{SDR}+}$ Results Analysis and Empirical Evaluation on the 2D Smooth-Interaction Regression Task

# ===== CELL 074 (markdown) =====
# **Prediction surface**

# ===== CELL 075 (code) =====
def plot_prediction_surface_60x60(
    data_dict: dict,
    theta_opt,
    tau_opt: float,
    N: int,
    SEED: int,
    output_dir: str | Path | None = None,
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

# ===== CELL 076 (markdown) =====
# **1D slices**

# ===== CELL 077 (code) =====
def plot_1d_slices(
    data_dict: dict,
    theta_opt,
    tau_opt: float,
    N: int,
    SEED: int,
    output_dir: str | Path | None = None,
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

# ===== CELL 078 (markdown) =====
# **Interaction ridge**

# ===== CELL 079 (code) =====
def plot_interaction_ridge(
    data_dict: dict,
    theta_opt,
    tau_opt: float,
    N: int,
    SEED: int,
    output_dir: str | Path | None = None,
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

# ===== CELL 080 (markdown) =====
# **True vs predicted**

# ===== CELL 081 (code) =====
def plot_true_vs_predicted(
    data_dict: dict,
    theta_opt,
    tau_opt: float,
    N: int,
    SEED: int,
    output_dir: str | Path | None = None,
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

# ===== CELL 082 (markdown) =====
# **Error plots**

# ===== CELL 083 (code) =====
def plot_residual_distributions(
    data_dict: dict,
    theta_opt,
    tau_opt: float,
    N: int,
    SEED: int,
    output_dir: str | Path | None = None,
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

# ===== CELL 084 (code) =====
# === PLOTTING PHASE: USDR+ (HYPERPARAMETER BOUNDS), ALL N × SEEDS ===

# 1) Load the constrained metrics so we reuse θ_opt, τ_opt instead of re-optimizing
df_results_constr = pd.read_csv("csv/usdr_plus/usdr_plus_final_results_constrained.csv")

# If you stored an "experiment" column, filter explicitly for safety
if "experiment" in df_results_constr.columns:
    df_results_constr = df_results_constr[
        df_results_constr["experiment"] == "usdr_plus_constrained"
    ]

base_fig_dir_constr = Path("figures/ccpp")

for N in sample_sizes:      # e.g. [50, 100, 200]
    for SEED in SEEDS:      # e.g. [0, 1, 2]
        print("\n=======================================")
        print(f"[PLOTS-CONSTR] Generating plots for N={N}, SEED={SEED}")
        print("=======================================")

        # 2) Select the row corresponding to this (N, SEED)
        row = df_results_constr[
            (df_results_constr["N"] == N) & (df_results_constr["SEED"] == SEED)
        ]
        if row.empty:
            print(
                f"[PLOTS-CONSTR] WARNING: "
                f"No constrained results found for N={N}, SEED={SEED}, skipping."
            )
            continue

        row = row.iloc[0]

        # 3) Reconstruct θ_opt and τ_opt from the stored metrics
        theta_opt = np.array(
            [
                row["lambda1"],
                row["lambda2"],
                row["gamma"],
                row["beta"],
            ],
            dtype=float,
        )
        tau_opt = float(row["tau"])

        # 4) Load the dataset splits for this (N, SEED)
        data = load_processed_2d_dataset(
            base_path=OUTPUT_DIR,
            N=N,
            seed=SEED,
            normalize=NORMALIZE,
        )

        # 5) Per-run figure directory: figures/ccpp/N{N}_seed{SEED}/
        run_dir = base_fig_dir_constr / f"N{N}_seed{SEED}"

        # Only show interactively for one canonical run, e.g. N=100, SEED=0
        show_canonical = (N == 100 and SEED == 0)

        # 6) Call all plotting functions (they will save PNGs + optionally show)
        prediction_surface_stats = plot_prediction_surface_60x60(
            data_dict=data,
            theta_opt=theta_opt,
            tau_opt=tau_opt,
            N=N,
            SEED=SEED,
            output_dir=run_dir,
            show=show_canonical,
        )

        one_d_slices_stats = plot_1d_slices(
            data_dict=data,
            theta_opt=theta_opt,
            tau_opt=tau_opt,
            N=N,
            SEED=SEED,
            output_dir=run_dir,
            show=show_canonical,
        )

        ridge_stats = plot_interaction_ridge(
            data_dict=data,
            theta_opt=theta_opt,
            tau_opt=tau_opt,
            N=N,
            SEED=SEED,
            output_dir=run_dir,
            show=show_canonical,
        )

        tvp_stats = plot_true_vs_predicted(
            data_dict=data,
            theta_opt=theta_opt,
            tau_opt=tau_opt,
            N=N,
            SEED=SEED,
            output_dir=run_dir,
            show=show_canonical,
        )

        resid_stats = plot_residual_distributions(
            data_dict=data,
            theta_opt=theta_opt,
            tau_opt=tau_opt,
            N=N,
            SEED=SEED,
            output_dir=run_dir,
            show=show_canonical,
            width=20,
            height=10,
        )

        print(f"[PLOTS-CONSTR] Saved plots to: {run_dir}")

# ===== CELL 085 (markdown) =====
# # **Plots**

# ===== CELL 086 (code) =====
def _get_input_scaler(data: dict) -> Optional[BaseEstimator]:
    """
    Try to extract the input scaler from the CCPP data dict.

    We look in:
      - data["x_scaler"], data["scaler_X"], data["scaler"]
      - data["metadata"]["x_scaler"], data["metadata"]["scaler_X"], data["metadata"]["scaler"]

    If nothing is found, we return None instead of raising.
    """
    direct_keys = ("x_scaler", "scaler_X", "scaler")
    meta_keys   = ("x_scaler", "scaler_X", "scaler")

    # 1) Directly on the top-level dict
    for k in direct_keys:
        if k in data and data[k] is not None:
            return data[k]

    # 2) Inside metadata, if present
    meta: dict[str, Any] = data.get("metadata", {})
    if isinstance(meta, dict):
        for k in meta_keys:
            if k in meta and meta[k] is not None:
                return meta[k]

    # 3) No scaler found → gracefully fallback to None
    return None

# ===== CELL 087 (code) =====
def plot_prediction_surface_ccpp(
    data: dict,
    theta: np.ndarray,
    tau: float,
    *,
    L: int = 2,
    entangler: str = "cnot",
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
    """
    Plot CCPP 2D prediction surface for a single (N, SEED) run.

    Behaviour:
    ----------
    • If a scaler is present in `data` (via _get_input_scaler):
        - Treat X_* as preprocessed.
        - Invert to raw (AT, V) for plotting.
        - Build a raw grid over (AT, V), then transform to preprocessed space
          before building kernel matrices.
    • If no scaler is present:
        - Treat X_* as 'raw enough'.
        - Infer domains directly from X_train.
        - Use those coordinates both for plotting and kernel construction.

    Parameters
    ----------
    data : dict
        Output of load_processed_ccpp_2d_dataset, with at least:
        X_train, y_train, X_val, y_val, X_test, y_test.
    theta : np.ndarray
        [lambda1, lambda2, gamma, beta] for USDR⁺.
    tau : float
        KRR regularisation parameter.
    L : int
        Circuit depth for U_SDR_plus.
    entangler : str
        Entangling gate label.
    run_dir : Path
        Directory where the figure will be saved.
    N, seed : int
        Sample size and seed (for titles / naming).
    grid_size : int
        Resolution of the (AT, V) grid.
    width, height : float
        Figure size in inches.
    dpi : int
        Figure DPI.
    show : bool
        If True, show figure; otherwise close after saving.
    raw_domain_at, raw_domain_v : Optional[Tuple[float, float]]
        Optional explicit raw domains for AT and V.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- 1) Collect all splits and scaler (if any) -----------------------
    X_train = np.asarray(data["X_train"], dtype=float)
    y_train = np.asarray(data["y_train"], dtype=float).ravel()

    X_val = None
    y_val = None
    if "X_val" in data and data["X_val"] is not None and len(data["X_val"]) > 0:
        X_val = np.asarray(data["X_val"], dtype=float)
        y_val = np.asarray(data["y_val"], dtype=float).ravel()

    X_test = None
    y_test = None
    if "X_test" in data and data["X_test"] is not None and len(data["X_test"]) > 0:
        X_test = np.asarray(data["X_test"], dtype=float)
        y_test = np.asarray(data["y_test"], dtype=float).ravel()

    X_all_list = [X_train]
    y_all_list = [y_train]
    if X_val is not None:
        X_all_list.append(X_val)
        y_all_list.append(y_val)
    if X_test is not None:
        X_all_list.append(X_test)
        y_all_list.append(y_test)

    X_all = np.vstack(X_all_list)
    y_all = np.concatenate(y_all_list)

    scaler_X = _get_input_scaler(data)  # may be None

    # --- 2) Handle raw vs processed coordinates --------------------------
    if scaler_X is not None:
        # data is in processed space → invert for plotting & domains
        X_train_raw = scaler_X.inverse_transform(X_train)
        X_all_raw = scaler_X.inverse_transform(X_all)

        if raw_domain_at is None:
            raw_domain_at = (
                float(X_train_raw[:, 0].min()),
                float(X_train_raw[:, 0].max()),
            )
        if raw_domain_v is None:
            raw_domain_v = (
                float(X_train_raw[:, 1].min()),
                float(X_train_raw[:, 1].max()),
            )

        at_grid = np.linspace(raw_domain_at[0], raw_domain_at[1], grid_size)
        v_grid = np.linspace(raw_domain_v[0], raw_domain_v[1], grid_size)
        AT_grid, V_grid = np.meshgrid(at_grid, v_grid)

        grid_raw = np.column_stack([AT_grid.ravel(), V_grid.ravel()])
        X_grid = scaler_X.transform(grid_raw)  # back to processed for kernel

        AT_plot = AT_grid
        V_plot = V_grid

    else:
        # No scaler: treat X_* as already raw-like.
        X_train_raw = X_train
        X_all_raw = X_all

        if raw_domain_at is None:
            raw_domain_at = (
                float(X_train[:, 0].min()),
                float(X_train[:, 0].max()),
            )
        if raw_domain_v is None:
            raw_domain_v = (
                float(X_train[:, 1].min()),
                float(X_train[:, 1].max()),
            )

        at_grid = np.linspace(raw_domain_at[0], raw_domain_at[1], grid_size)
        v_grid = np.linspace(raw_domain_v[0], raw_domain_v[1], grid_size)
        AT_grid, V_grid = np.meshgrid(at_grid, v_grid)

        grid_raw = np.column_stack([AT_grid.ravel(), V_grid.ravel()])
        X_grid = grid_raw  # same space as X_train

        AT_plot = AT_grid
        V_plot = V_grid

    # --- 3) Build kernels and KRR solution --------------------------------
    # NOTE: build_kernel_matrix is assumed to accept:
    #   build_kernel_matrix(X1, X2, theta, L=..., entangler=..., apply_psd_hygiene_for_square=False)
    K_train = build_kernel_matrix(
        X_train,
        X_train,
        theta=theta,
        L=L,
        entangler=entangler,
        apply_psd_hygiene_for_square=False,
    )

    K_reg, jitter, _, _ = apply_psd_hygiene(
        K_train,
        tau,
        name=f"K_train (CCPP N={N}, seed={seed})",
    )

    alpha = np.linalg.solve(K_reg, y_train)

    K_grid = build_kernel_matrix(
        X_grid,
        X_train,
        theta=theta,
        L=L,
        entangler=entangler,
        apply_psd_hygiene_for_square=False,
    )
    y_grid_pred = (K_grid @ alpha).reshape(AT_plot.shape)

    # --- 4) Plot: scatter + prediction surface ----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(width, height), dpi=dpi)

    # Left: scattered observed data
    ax0 = axes[0]
    sc = ax0.scatter(
        X_all_raw[:, 0],
        X_all_raw[:, 1],
        c=y_all,
        s=20,
        alpha=0.8,
    )
    ax0.set_xlabel("AT (raw or normalized)")
    ax0.set_ylabel("V (raw or normalized)")
    ax0.set_title(f"CCPP Data (N={N}, seed={seed})")
    cbar0 = fig.colorbar(sc, ax=ax0)
    cbar0.set_label("EP (observed)")

    # Right: prediction surface
    ax1 = axes[1]
    cs = ax1.contourf(
        AT_plot,
        V_plot,
        y_grid_pred,
        levels=30,
    )
    ax1.set_xlabel("AT (raw or normalized)")
    ax1.set_ylabel("V (raw or normalized)")
    ax1.set_title("USDR+ KRR Prediction Surface (EP)")
    cbar1 = fig.colorbar(cs, ax=ax1)
    cbar1.set_label("EP (predicted)")

    fig.suptitle(f"USDR⁺ CCPP 2D – N={N}, seed={seed}")

    out_path = run_dir / "prediction_surface.png"
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=dpi)
    print(f"[PLOT-CCPP] Saved prediction surface → {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

# ===== CELL 088 (code) =====
df_used = generate_ccpp_prediction_surface_plots_from_csv(
    results_csv="csv/ccpp/usdr_plus_ccpp_2d_results_constrained.csv",
    experiment_name="usdr_plus_constrained_ccpp_2d",
    L=2,
    entangler="cnot",
    grid_size=200,
    canonical_run=(100, 0),
    width=24.0,
    height=10.0,
    dpi=300,
)

# ===== CELL 089 (markdown) =====
# **1D slices**

# ===== CELL 090 (code) =====
def _regularize_kernel_ccpp(
    K_train: np.ndarray,
    tau: float,
    name: str = "K_train (CCPP 1D slices)",
    jitter_factor: float = 1e-10,
    cond_threshold: float = 1e12,
) -> Tuple[np.ndarray, float]:
    """
    Apply τ-regularisation and, if necessary, a small jitter to K_train.

    Parameters
    ----------
    K_train : (n_train, n_train) ndarray
        Raw training Gram matrix (already PSD up to numerical noise).
    tau : float
        KRR regularisation strength.
    name : str
        Label used in log messages.
    jitter_factor : float
        Multiplier for the trace-based jitter when condition number is too large.
    cond_threshold : float
        Threshold on the condition number for adding jitter.

    Returns
    -------
    K_reg : (n_train, n_train) ndarray
        Regularised kernel matrix K_train + τI (+ εI if needed).
    jitter : float
        Actual jitter added (0.0 if none).
    """
    n = K_train.shape[0]
    K_reg = K_train + tau * np.eye(n)

    try:
        kappa = np.linalg.cond(K_reg)
    except np.linalg.LinAlgError:
        kappa = np.inf

    print(
        f"[PSD-CCPP] {name}: τ={tau:.3e}, "
        f"cond(K+τI)={kappa:.3e}"
    )

    jitter = 0.0
    if np.isfinite(kappa) and kappa > cond_threshold:
        jitter = jitter_factor * np.trace(K_train) / float(n)
        K_reg = K_reg + jitter * np.eye(n)
        try:
            kappa_after = np.linalg.cond(K_reg)
        except np.linalg.LinAlgError:
            kappa_after = np.inf

        print(
            f"[PSD-CCPP] Added jitter ε={jitter:.2e} "
            f"({name}): κ_before={kappa:.3e} → κ_after={kappa_after:.3e}"
        )
    else:
        print(f"[PSD-CCPP] No extra jitter required for {name}.")

    return K_reg, jitter


def plot_1d_slices_ccpp(
    data: Dict[str, Any],
    theta: np.ndarray,
    tau: float,
    *,
    L: int = 2,
    entangler: str = "cnot",
    run_dir: str | Path,
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
    """
    Plot 1D slices of the USDR⁺ predictor for the CCPP 2D subproblem.

    Slice 1: AT varies, V fixed at median V (train set).
    Slice 2: V varies, AT fixed at median AT (train set).

    For each slice:
      • Build a 1D grid in RAW (AT, V) domain.
      • Preprocess the slice with the *same* scaler used for training.
      • Compute USDR⁺ predictions along the slice (via KRR with (θ, τ)).
      • Overlay data points near the slice (small band around fixed coord).
      • Save the result to `run_dir / "slices.png"`.

    Parameters
    ----------
    data : dict
        Processed dataset as returned by `load_processed_ccpp_2d_dataset`, with
        keys like 'X_train', 'y_train', 'X_val', 'X_test', and optionally
        an input scaler accessible via `_get_input_scaler`.
    theta : ndarray, shape (4,)
        Optimised USDR⁺ kernel parameters [λ1, λ2, γ, β].
    tau : float
        KRR regularisation strength.
    L : int, optional
        Circuit depth for U_SDR_plus (default: 2).
    entangler : str, optional
        Entangling gate (default: "cnot").
    run_dir : str or Path
        Directory where "slices.png" will be saved.
    N : int
        Sample size used in this run (for logging / title).
    seed : int
        Random seed used in this run (for logging / title).
    grid_size : int, optional
        Number of grid points along each 1D slice.
    width, height : float
        Figure size in inches.
    dpi : int
        Figure DPI.
    show : bool
        If True, display the figure with plt.show().
        If False, close the figure after saving.
    raw_domain_at, raw_domain_v : (float, float) or None
        Optional explicit RAW domain for AT and V. If None, the domain is
        inferred from the (inverse-transformed) training inputs.
    """
    # ------------------------------------------------------------------ #
    # 1. Extract TRAIN/VAL/TEST and possibly the input scaler
    # ------------------------------------------------------------------ #
    X_train = np.asarray(data["X_train"], dtype=np.float64)
    y_train = np.asarray(data["y_train"], dtype=np.float64)

    X_val = np.asarray(data.get("X_val", []), dtype=np.float64)
    y_val = np.asarray(data.get("y_val", []), dtype=np.float64)

    X_test = np.asarray(data.get("X_test", []), dtype=np.float64)
    y_test = np.asarray(data.get("y_test", []), dtype=np.float64)

    # All data (for overlay)
    X_all_list = [X_train]
    y_all_list = [y_train]
    if X_val.size > 0:
        X_all_list.append(X_val)
        y_all_list.append(y_val)
    if X_test.size > 0:
        X_all_list.append(X_test)
        y_all_list.append(y_test)

    X_all = np.vstack(X_all_list)
    y_all = np.concatenate(y_all_list)

    if X_train.shape[1] != 2:
        raise ValueError(
            f"[SLICES-CCPP] Expected 2D inputs (AT, V), "
            f"got shape {X_train.shape}"
        )

    scaler_X = _get_input_scaler(data)
    has_scaler = scaler_X is not None

    # ------------------------------------------------------------------ #
    # 2. Recover RAW coordinates (AT, V) and slice anchors
    # ------------------------------------------------------------------ #
    if has_scaler:
        X_train_raw = scaler_X.inverse_transform(X_train)
        X_all_raw = scaler_X.inverse_transform(X_all)
    else:
        # Already in "model domain", treat as RAW for slicing/plotting
        X_train_raw = X_train.copy()
        X_all_raw = X_all.copy()

    # Columns: AT = 0, V = 1
    AT_train = X_train_raw[:, 0]
    V_train = X_train_raw[:, 1]

    AT_fixed = float(np.median(AT_train))
    V_fixed = float(np.median(V_train))

    # Infer raw domain if not explicitly provided
    if raw_domain_at is None:
        at_min = float(AT_train.min())
        at_max = float(AT_train.max())
    else:
        at_min, at_max = raw_domain_at

    if raw_domain_v is None:
        v_min = float(V_train.min())
        v_max = float(V_train.max())
    else:
        v_min, v_max = raw_domain_v

    print(
        f"[SLICES-CCPP] N={N}, SEED={seed} | "
        f"AT_fixed={AT_fixed:.3f}, V_fixed={V_fixed:.3f} | "
        f"AT_range=[{at_min:.3f}, {at_max:.3f}], "
        f"V_range=[{v_min:.3f}, {v_max:.3f}]"
    )

    # ------------------------------------------------------------------ #
    # 3. Build 1D RAW grids for the two slices
    # ------------------------------------------------------------------ #
    AT_grid_raw = np.linspace(at_min, at_max, grid_size)
    V_grid_raw = np.linspace(v_min, v_max, grid_size)

    # Slice 1: vary AT, fix V = V_fixed
    slice1_raw = np.column_stack([
        AT_grid_raw,
        np.full_like(AT_grid_raw, V_fixed),
    ])

    # Slice 2: vary V, fix AT = AT_fixed
    slice2_raw = np.column_stack([
        np.full_like(V_grid_raw, AT_fixed),
        V_grid_raw,
    ])

    # Preprocess with the same scaler used for training
    if has_scaler:
        slice1_proc = scaler_X.transform(slice1_raw)
        slice2_proc = scaler_X.transform(slice2_raw)
        X_train_proc = X_train  # already preprocessed
    else:
        slice1_proc = slice1_raw
        slice2_proc = slice2_raw
        X_train_proc = X_train

    # ------------------------------------------------------------------ #
    # 4. Build K_train, regularise with τ (and jitter if needed)
    # ------------------------------------------------------------------ #
    # NOTE: build_kernel_matrix must be already defined in this notebook/codebase
    K_train = build_kernel_matrix(
        X_train_proc,
        X_train_proc,
        theta=theta,
        L=L,
        entangler=entangler,
        apply_psd_hygiene_for_square=True,  # safe for square K
    )

    K_reg, jitter = _regularize_kernel_ccpp(
        K_train,
        tau=tau,
        name=f"K_train (CCPP slices N={N}, seed={seed})",
    )

    # Solve for α in (K_reg α = y_train)
    alpha = np.linalg.solve(K_reg, y_train)

    # ------------------------------------------------------------------ #
    # 5. Build cross-kernels for slices and get predictions
    # ------------------------------------------------------------------ #
    K_slice1 = build_kernel_matrix(
        slice1_proc,
        X_train_proc,
        theta=theta,
        L=L,
        entangler=entangler,
        apply_psd_hygiene_for_square=False,
    )
    y_pred_slice1 = K_slice1 @ alpha

    K_slice2 = build_kernel_matrix(
        slice2_proc,
        X_train_proc,
        theta=theta,
        L=L,
        entangler=entangler,
        apply_psd_hygiene_for_square=False,
    )
    y_pred_slice2 = K_slice2 @ alpha

    # ------------------------------------------------------------------ #
    # 6. Select data points near each slice for overlay
    # ------------------------------------------------------------------ #
    # Use a small band around the fixed coordinates (5% of total range)
    band_width_v = 0.05 * (v_max - v_min) if v_max > v_min else 0.0
    band_width_at = 0.05 * (at_max - at_min) if at_max > at_min else 0.0

    AT_all_raw = X_all_raw[:, 0]
    V_all_raw = X_all_raw[:, 1]

    # Slice 1: points where V is near V_fixed
    if band_width_v > 0.0:
        mask_slice1 = np.abs(V_all_raw - V_fixed) <= band_width_v
    else:
        mask_slice1 = np.ones_like(V_all_raw, dtype=bool)

    if not np.any(mask_slice1):
        # Fallback: use all points
        print(
            "[SLICES-CCPP] WARNING: no points in V-band around V_fixed, "
            "using all data for slice 1 overlay."
        )
        mask_slice1 = np.ones_like(V_all_raw, dtype=bool)

    AT_near_slice1 = AT_all_raw[mask_slice1]
    y_near_slice1 = y_all[mask_slice1]

    # Slice 2: points where AT is near AT_fixed
    if band_width_at > 0.0:
        mask_slice2 = np.abs(AT_all_raw - AT_fixed) <= band_width_at
    else:
        mask_slice2 = np.ones_like(AT_all_raw, dtype=bool)

    if not np.any(mask_slice2):
        print(
            "[SLICES-CCPP] WARNING: no points in AT-band around AT_fixed, "
            "using all data for slice 2 overlay."
        )
        mask_slice2 = np.ones_like(AT_all_raw, dtype=bool)

    V_near_slice2 = V_all_raw[mask_slice2]
    y_near_slice2 = y_all[mask_slice2]

    # ------------------------------------------------------------------ #
    # 7. Plotting
    # ------------------------------------------------------------------ #
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "slices.png"

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(width, height),
        dpi=dpi,
        constrained_layout=True,
    )

    # --- Slice 1: AT slice at fixed V ---
    ax1 = axes[0]
    ax1.scatter(
        AT_near_slice1,
        y_near_slice1,
        alpha=0.8,
        s=30,
        label="Data (V near V_fixed)",
    )
    ax1.plot(
        AT_grid_raw,
        y_pred_slice1,
        linestyle="--",
        linewidth=2.0,
        label="USDR+ prediction",
    )
    ax1.set_xlabel("AT (raw)")
    ax1.set_ylabel("EP")
    ax1.set_title(f"AT-slice at V ≈ {V_fixed:.2f} (N={N}, seed={seed})")
    ax1.grid(True, linestyle=":")
    ax1.legend(loc="best")

    # --- Slice 2: V slice at fixed AT ---
    ax2 = axes[1]
    ax2.scatter(
        V_near_slice2,
        y_near_slice2,
        alpha=0.8,
        s=30,
        label="Data (AT near AT_fixed)",
    )
    ax2.plot(
        V_grid_raw,
        y_pred_slice2,
        linestyle="--",
        linewidth=2.0,
        label="USDR+ prediction",
    )
    ax2.set_xlabel("V (raw)")
    ax2.set_ylabel("EP")
    ax2.set_title(f"V-slice at AT ≈ {AT_fixed:.2f} (N={N}, seed={seed})")
    ax2.grid(True, linestyle=":")
    ax2.legend(loc="best")

    fig.suptitle(
        f"USDR⁺ 1D Slices on CCPP 2D (N={N}, seed={seed})",
        fontsize=14,
    )

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"[SLICES-CCPP] Saved 1D slice plot → {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

# ===== CELL 091 (code) =====
pass

# ===== CELL 092 (code) =====
for N in [50, 100, 200]:
    for SEED in [0, 1, 2]:
        # ... load data/theta_opt/tau_opt for this (N, SEED) ...
        run_dir = Path("figures/ccpp/results") / f"usdr_plus_ccpp_N{N}_seed{SEED}" / "slices"

        plot_1d_slices_ccpp(
            data=data,
            theta=theta_opt,
            tau=tau_opt,
            L=2,
            entangler="cnot",
            run_dir=run_dir,
            N=N,
            seed=SEED,
            grid_size=200,
            width=24.0,
            height=10.0,
            dpi=300,
            show=(N, SEED) == (100, 0),
        )

# ===== CELL 093 (code) =====

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _regularize_kernel_ccpp(
    K_train: np.ndarray,
    tau: float,
    name: str = "K_train (CCPP 1D slices)",
    jitter_factor: float = 1e-10,
    cond_threshold: float = 1e12,
) -> Tuple[np.ndarray, float]:
    """
    Apply τ-regularisation and, if necessary, a small jitter to K_train.
    """
    n = K_train.shape[0]
    K_reg = K_train + tau * np.eye(n)

    try:
        kappa = np.linalg.cond(K_reg)
    except np.linalg.LinAlgError:
        kappa = np.inf

    print(
        f"[PSD-CCPP] {name}: τ={tau:.3e}, "
        f"cond(K+τI)={kappa:.3e}"
    )

    jitter = 0.0
    if np.isfinite(kappa) and kappa > cond_threshold:
        jitter = jitter_factor * np.trace(K_train) / float(n)
        K_reg = K_reg + jitter * np.eye(n)
        try:
            kappa_after = np.linalg.cond(K_reg)
        except np.linalg.LinAlgError:
            kappa_after = np.inf

        print(
            f"[PSD-CCPP] Added jitter ε={jitter:.2e} "
            f"({name}): κ_before={kappa:.3e} → κ_after={kappa_after:.3e}"
        )
    else:
        print(f"[PSD-CCPP] No extra jitter required for {name}.")

    return K_reg, jitter


def plot_1d_slices_ccpp(
    data: Dict[str, Any],
    theta: np.ndarray,
    tau: float,
    *,
    L: int = 2,
    entangler: str = "cnot",
    run_dir: str | Path,  # kept for backward compatibility, but not used
    N: int,
    seed: int,
    grid_size: int = 200,
    width: float = 34.0,
    height: float = 20.0,
    dpi: int = 300,
    show: bool = False,
    raw_domain_at: Optional[Tuple[float, float]] = None,
    raw_domain_v: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot 1D slices of the USDR+ predictor for the CCPP 2D subproblem and save
    them under figures/ccpp/slices/slices_N{N}_seed{seed}.png.
    """
    # ------------------------------------------------------------------ #
    # 1. Extract TRAIN/VAL/TEST and possibly the input scaler
    # ------------------------------------------------------------------ #
    X_train = np.asarray(data["X_train"], dtype=np.float64)
    y_train = np.asarray(data["y_train"], dtype=np.float64)

    X_val = np.asarray(data.get("X_val", []), dtype=np.float64)
    y_val = np.asarray(data.get("y_val", []), dtype=np.float64)

    X_test = np.asarray(data.get("X_test", []), dtype=np.float64)
    y_test = np.asarray(data.get("y_test", []), dtype=np.float64)

    # All data (for overlay)
    X_all_list = [X_train]
    y_all_list = [y_train]
    if X_val.size > 0:
        X_all_list.append(X_val)
        y_all_list.append(y_val)
    if X_test.size > 0:
        X_all_list.append(X_test)
        y_all_list.append(y_test)

    X_all = np.vstack(X_all_list)
    y_all = np.concatenate(y_all_list)

    if X_train.shape[1] != 2:
        raise ValueError(
            f"[SLICES-CCPP] Expected 2D inputs (AT, V), "
            f"got shape {X_train.shape}"
        )

    scaler_X = _get_input_scaler(data)
    has_scaler = scaler_X is not None

    # ------------------------------------------------------------------ #
    # 2. Recover RAW coordinates (AT, V) and slice anchors
    # ------------------------------------------------------------------ #
    if has_scaler:
        X_train_raw = scaler_X.inverse_transform(X_train)
        X_all_raw = scaler_X.inverse_transform(X_all)
    else:
        X_train_raw = X_train.copy()
        X_all_raw = X_all.copy()

    # Columns: AT = 0, V = 1
    AT_train = X_train_raw[:, 0]
    V_train = X_train_raw[:, 1]

    AT_fixed = float(np.median(AT_train))
    V_fixed = float(np.median(V_train))

    # Infer raw domain if not explicitly provided
    if raw_domain_at is None:
        at_min = float(AT_train.min())
        at_max = float(AT_train.max())
    else:
        at_min, at_max = raw_domain_at

    if raw_domain_v is None:
        v_min = float(V_train.min())
        v_max = float(V_train.max())
    else:
        v_min, v_max = raw_domain_v

    print(
        f"[SLICES-CCPP] N={N}, SEED={seed} | "
        f"AT_fixed={AT_fixed:.3f}, V_fixed={V_fixed:.3f} | "
        f"AT_range=[{at_min:.3f}, {at_max:.3f}], "
        f"V_range=[{v_min:.3f}, {v_max:.3f}]"
    )

    # ------------------------------------------------------------------ #
    # 3. Build 1D RAW grids for the two slices
    # ------------------------------------------------------------------ #
    AT_grid_raw = np.linspace(at_min, at_max, grid_size)
    V_grid_raw = np.linspace(v_min, v_max, grid_size)

    slice1_raw = np.column_stack([
        AT_grid_raw,
        np.full_like(AT_grid_raw, V_fixed),
    ])
    slice2_raw = np.column_stack([
        np.full_like(V_grid_raw, AT_fixed),
        V_grid_raw,
    ])

    # Preprocess with the same scaler used for training
    if has_scaler:
        slice1_proc = scaler_X.transform(slice1_raw)
        slice2_proc = scaler_X.transform(slice2_raw)
        X_train_proc = X_train
    else:
        slice1_proc = slice1_raw
        slice2_proc = slice2_raw
        X_train_proc = X_train

    # ------------------------------------------------------------------ #
    # 4. Build K_train, regularise with τ (and jitter if needed)
    # ------------------------------------------------------------------ #
    K_train = build_kernel_matrix(
        X_train_proc,
        X_train_proc,
        theta=theta,
        L=L,
        entangler=entangler,
        apply_psd_hygiene_for_square=True,
    )

    K_reg, jitter = _regularize_kernel_ccpp(
        K_train,
        tau=tau,
        name=f"K_train (CCPP slices N={N}, seed={seed})",
    )

    alpha = np.linalg.solve(K_reg, y_train)

    # ------------------------------------------------------------------ #
    # 5. Cross-kernels and predictions along slices
    # ------------------------------------------------------------------ #
    K_slice1 = build_kernel_matrix(
        slice1_proc,
        X_train_proc,
        theta=theta,
        L=L,
        entangler=entangler,
        apply_psd_hygiene_for_square=False,
    )
    y_pred_slice1 = K_slice1 @ alpha

    K_slice2 = build_kernel_matrix(
        slice2_proc,
        X_train_proc,
        theta=theta,
        L=L,
        entangler=entangler,
        apply_psd_hygiene_for_square=False,
    )
    y_pred_slice2 = K_slice2 @ alpha

    # ------------------------------------------------------------------ #
    # 6. Select data points near each slice for overlay
    # ------------------------------------------------------------------ #
    band_width_v = 0.05 * (v_max - v_min) if v_max > v_min else 0.0
    band_width_at = 0.05 * (at_max - at_min) if at_max > at_min else 0.0

    AT_all_raw = X_all_raw[:, 0]
    V_all_raw = X_all_raw[:, 1]

    if band_width_v > 0.0:
        mask_slice1 = np.abs(V_all_raw - V_fixed) <= band_width_v
    else:
        mask_slice1 = np.ones_like(V_all_raw, dtype=bool)

    if not np.any(mask_slice1):
        print(
            "[SLICES-CCPP] WARNING: no points in V-band around V_fixed, "
            "using all data for slice 1 overlay."
        )
        mask_slice1 = np.ones_like(V_all_raw, dtype=bool)

    AT_near_slice1 = AT_all_raw[mask_slice1]
    y_near_slice1 = y_all[mask_slice1]

    if band_width_at > 0.0:
        mask_slice2 = np.abs(AT_all_raw - AT_fixed) <= band_width_at
    else:
        mask_slice2 = np.ones_like(AT_all_raw, dtype=bool)

    if not np.any(mask_slice2):
        print(
            "[SLICES-CCPP] WARNING: no points in AT-band around AT_fixed, "
            "using all data for slice 2 overlay."
        )
        mask_slice2 = np.ones_like(AT_all_raw, dtype=bool)

    V_near_slice2 = V_all_raw[mask_slice2]
    y_near_slice2 = y_all[mask_slice2]

    # ------------------------------------------------------------------ #
    # 7. Residuals and RMSE for each slice
    # ------------------------------------------------------------------ #
    y_pred_near_slice1 = np.interp(AT_near_slice1, AT_grid_raw, y_pred_slice1)
    resid_slice1 = y_near_slice1 - y_pred_near_slice1

    y_pred_near_slice2 = np.interp(V_near_slice2, V_grid_raw, y_pred_slice2)
    resid_slice2 = y_near_slice2 - y_pred_near_slice2

    rmse_slice1 = float(np.sqrt(np.mean(resid_slice1**2)))
    rmse_slice2 = float(np.sqrt(np.mean(resid_slice2**2)))

    # ------------------------------------------------------------------ #
    # 8. Plotting: save under figures/ccpp/slices
    # ------------------------------------------------------------------ #
    base_dir = Path("figures") / "ccpp" / "slices"
    base_dir.mkdir(parents=True, exist_ok=True)
    out_path = base_dir / f"slices_N{N}_seed{seed}.png"

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(width, height),
        dpi=dpi,
        constrained_layout=True,
        sharex="col",
    )

    # Top-left: AT slice
    ax1 = axes[0, 0]
    ax1.scatter(
        AT_near_slice1,
        y_near_slice1,
        alpha=0.8,
        s=30,
        label="Data (V in band)",
    )
    ax1.plot(
        AT_grid_raw,
        y_pred_slice1,
        linestyle="--",
        linewidth=2.0,
        label="USDR+ prediction",
    )
    ax1.set_ylabel("EP")
    ax1.set_title(f"AT-slice at V ≈ {V_fixed:.2f} (N={N}, seed={seed})")
    ax1.grid(True, linestyle=":")
    ax1.legend(loc="best")
    ax1.text(
        0.02,
        0.95,
        f"RMSE (band) = {rmse_slice1:.2f}",
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.15, edgecolor="none"),
    )

    # Top-right: V slice
    ax2 = axes[0, 1]
    ax2.scatter(
        V_near_slice2,
        y_near_slice2,
        alpha=0.8,
        s=30,
        label="Data (AT in band)",
    )
    ax2.plot(
        V_grid_raw,
        y_pred_slice2,
        linestyle="--",
        linewidth=2.0,
        label="USDR+ prediction",
    )
    ax2.set_ylabel("EP")
    ax2.set_title(f"V-slice at AT ≈ {AT_fixed:.2f} (N={N}, seed={seed})")
    ax2.grid(True, linestyle=":")
    ax2.legend(loc="best")
    ax2.text(
        0.02,
        0.95,
        f"RMSE (band) = {rmse_slice2:.2f}",
        transform=ax2.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.15, edgecolor="none"),
    )

    # Bottom-left: residuals on AT slice
    ax3 = axes[1, 0]
    ax3.axhline(0.0, linestyle="--", linewidth=1.0)
    ax3.scatter(AT_near_slice1, resid_slice1, alpha=0.8, s=25)
    ax3.set_xlabel("AT (raw)")
    ax3.set_ylabel("Residual (data − pred)")
    ax3.set_title("Residuals on AT-slice")
    ax3.grid(True, linestyle=":")

    # Bottom-right: residuals on V slice
    ax4 = axes[1, 1]
    ax4.axhline(0.0, linestyle="--", linewidth=1.0)
    ax4.scatter(V_near_slice2, resid_slice2, alpha=0.8, s=25)
    ax4.set_xlabel("V (raw)")
    ax4.set_ylabel("Residual (data − pred)")
    ax4.set_title("Residuals on V-slice")
    ax4.grid(True, linestyle=":")

    fig.suptitle(
        f"USDR+ 1D Slices on CCPP 2D (N={N}, seed={seed})",
        fontsize=14,
    )

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"[SLICES-CCPP] Saved 1D slice plot → {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

# ===== CELL 094 (code) =====
for N in [50, 100, 200]:
    for SEED in [0, 1, 2]:
        plot_1d_slices_ccpp(
            data=data,
            theta=theta_opt,
            tau=tau_opt,
            L=2,
            entangler="cnot",
            run_dir=".",
            N=N,
            seed=SEED,
            show=False,
        )

# ===== CELL 095 (markdown) =====
# **True vs predicted**

# ===== CELL 096 (code) =====
def plot_true_vs_predicted_ccpp(
    data: Dict[str, Any],
    theta: np.ndarray,
    tau: float,
    *,
    L: int = 2,
    entangler: str = "cnot",
    run_dir: str | Path,
    N: int,
    seed: int,
    width: float = 24.0,   
    height: float = 10.0,  
    dpi: int = 300,
    show: bool = False,
) -> float:
    """
    Plot true vs predicted EP on the CCPP 2D *test* set using USDR+ KRR.

    Steps
    -----
    1. Build K_train(X_train, X_train; θ) and regularise with τ (and optional jitter).
    2. Solve K_reg α = y_train.
    3. Build K_test(X_test, X_train; θ) and compute y_pred = K_test α.
    4. Compute test MSE.
    5. Scatter plot: true EP vs predicted EP, with identity line y = x.
    6. Save figure to
         figures/ccpp/true_vs_pred/true_vs_pred_N{N}_seed{seed}.png

    Parameters
    ----------
    data : dict
        Processed dataset for a given (N, seed) as returned by
        `load_processed_ccpp_2d_dataset`, with keys:
          - "X_train", "y_train"
          - "X_test", "y_test"
    theta : ndarray, shape (4,)
        USDR+ parameters [λ1, λ2, γ, β].
    tau : float
        KRR regularisation strength.
    L : int, optional
        Circuit depth for U_SDR_plus (default: 2).
    entangler : str, optional
        Entangling gate name (default: "cnot").
    run_dir : str or Path
        Ignored for the output path, kept only for backward compatibility.
    N : int
        Sample size for this run (for logging / title).
    seed : int
        Seed for this run (for logging / title).
    width, height : float
        Figure size in inches.
    dpi : int
        Figure DPI.
    show : bool
        If True, display the plot with `plt.show()`. Otherwise close it.

    Returns
    -------
    test_mse : float
        Mean squared error on the test set.
    """
    # ------------------------------------------------------------------ #
    # 1. Extract train / test splits
    # ------------------------------------------------------------------ #
    if "X_train" not in data or "y_train" not in data:
        raise KeyError("`data` must contain 'X_train' and 'y_train'.")
    if "X_test" not in data or "y_test" not in data:
        raise KeyError("`data` must contain 'X_test' and 'y_test'.")

    X_train = np.asarray(data["X_train"], dtype=np.float64)
    y_train = np.asarray(data["y_train"], dtype=np.float64)
    X_test = np.asarray(data["X_test"], dtype=np.float64)
    y_test = np.asarray(data["y_test"], dtype=np.float64)

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"X_train and y_train have incompatible shapes: "
            f"{X_train.shape} vs {y_train.shape}"
        )
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(
            f"X_test and y_test have incompatible shapes: "
            f"{X_test.shape} vs {y_test.shape}"
        )

    # ------------------------------------------------------------------ #
    # 2. Build K_train and regularise with τ (and jitter if needed)
    # ------------------------------------------------------------------ #
    print(
        f"[TVP-CCPP] Building K_train for true-vs-pred plot "
        f"(N={N}, seed={seed})..."
    )

    K_train = build_kernel_matrix(
        X_train,
        X_train,
        theta=theta,
        L=L,
        entangler=entangler,
        apply_psd_hygiene_for_square=True,  # safe for square Gram matrix
    )

    K_reg, jitter = _regularize_kernel_ccpp(
        K_train,
        tau=tau,
        name=f"K_train (CCPP true-vs-pred N={N}, seed={seed})",
    )

    # Solve K_reg α = y_train
    alpha = np.linalg.solve(K_reg, y_train)

    # ------------------------------------------------------------------ #
    # 3. Build K_test and compute predictions
    # ------------------------------------------------------------------ #
    print(
        f"[TVP-CCPP] Building K_test and predicting on test set "
        f"(N={N}, seed={seed})..."
    )

    K_test = build_kernel_matrix(
        X_test,
        X_train,
        theta=theta,
        L=L,
        entangler=entangler,
        apply_psd_hygiene_for_square=False,
    )

    y_pred = K_test @ alpha
    test_mse = float(mean_squared_error(y_test, y_pred))

    print(
        f"[TVP-CCPP] Test MSE (N={N}, seed={seed}) = {test_mse:.4e}, "
        f"jitter={jitter:.2e}"
    )

    # ------------------------------------------------------------------ #
    # 4. Plot true vs predicted with identity line
    # ------------------------------------------------------------------ #
    out_dir = Path("figures") / "ccpp" / "true_vs_pred"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"true_vs_pred_N{N}_seed{seed}.png"

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

    # Scatter: true vs predicted
    ax.scatter(
        y_test,
        y_pred,
        alpha=0.8,
        s=40,
        label="Test points",
    )

    # Identity line y = x
    y_min = float(min(y_test.min(), y_pred.min()))
    y_max = float(max(y_test.max(), y_pred.max()))
    padding = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    y_min -= padding
    y_max += padding

    ax.plot(
        [y_min, y_max],
        [y_min, y_max],
        linestyle="--",
        linewidth=2.0,
        label="y = x (perfect prediction)",
    )

    ax.set_xlabel("True EP")
    ax.set_ylabel("Predicted EP (USDR+)")
    ax.set_xlim(y_min, y_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(
        f"CCPP 2D – True vs Predicted (N={N}, seed={seed})"
    )
    ax.grid(True, linestyle=":")
    ax.legend(loc="best")

    # Compact MSE annotation to improve readability
    ax.text(
        0.05,
        0.95,
        f"Test MSE = {test_mse:.3f}\nJitter = {jitter:.2e}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.15, edgecolor="none"),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"[TVP-CCPP] Saved true-vs-predicted plot → {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return test_mse

# ===== CELL 097 (code) =====
N_list = [50, 100, 200]
seed_list = [0, 1, 2]

results = {}

for N in N_list:
    for SEED in seed_list:
        print(f"\n=== Running CCPP USDR+ for N={N}, seed={SEED} ===")

        # 1) Load data
        data = load_processed_ccpp_2d_dataset(N=N, seed=SEED)

        # 2) YOUR EXISTING OPTIMISATION CODE GOES HERE
        # -------------------------------------------------
        # Whatever you currently do to get theta_opt, tau_opt.
        # For example, you probably have something like:
        #
        # theta_opt, tau_opt, val_mse = run_usdr_plus_bounds_search_ccpp(
        #     data=data,
        #     N=N,
        #     seed=SEED,
        # )
        #
        # DO NOT literally copy the line above if run_usdr_plus_bounds_search_ccpp
        # is not the name of your function – just paste your real optimisation code
        # here so that at the end you have:
        #
        #   theta_opt  -> np.ndarray, shape (4,)
        #   tau_opt    -> float
        #
        # -------------------------------------------------

        # 3) Slices plot (already working for you)
        plot_1d_slices_ccpp(
            data=data,
            theta=theta_opt,
            tau=tau_opt,
            L=2,
            entangler="cnot",
            run_dir=".",   # ignored in our latest version
            N=N,
            seed=SEED,
            grid_size=200,
            width=24.0,
            height=10.0,
            dpi=300,
            show=False,
        )

        # 4) True vs predicted plot, USING THE SAME theta_opt, tau_opt
        mse = plot_true_vs_predicted_ccpp(
            data=data,
            theta=theta_opt,
            tau=tau_opt,
            L=2,
            entangler="cnot",
            run_dir=".",   # ignored; figures go to figures/ccpp/true_vs_pred
            N=N,
            seed=SEED,
            width=24.0,     # or 24.0, 10.0 if you want a wide figure
            height=10.0,
            dpi=300,
            show=False,
        )

        results[(N, SEED)] = mse
        print(f"[RESULT] N={N}, seed={SEED} → test MSE = {mse:.4e}")

# ===== CELL 098 (markdown) =====
# **Residual distributions**

# ===== CELL 099 (code) =====
def plot_residual_distributions_ccpp(
    data: Dict[str, Any],
    theta: np.ndarray,
    tau: float,
    *,
    L: int = 2,
    entangler: str = "cnot",
    run_dir: str | Path,  # kept for backward compatibility (ignored for path)
    N: int,
    seed: int,
    width: float = 20.0,
    height: float = 12.0,
    dpi: int = 300,
    show: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Plot residual distributions (train/val/test) for USDR+ on CCPP 2D.

    Steps
    -----
    1. Build K_train(X_train, X_train; θ) and regularise with τ (and jitter).
    2. Solve K_reg α = y_train.
    3. Build K_val, K_test and compute predictions.
    4. Compute residuals r = y_true - y_pred for train/val/test.
    5. Compute statistics (MSE, MAE, mean, std) for each split and print them.
    6. Violin plot + jittered scatter for the 3 residual groups.
    7. Save figure to
         figures/ccpp/residuals/residuals_N{N}_seed{seed}.png

    Parameters
    ----------
    data : dict
        Processed dataset for a given (N, seed) as returned by
        `load_processed_ccpp_2d_dataset`, with keys:
          - "X_train", "y_train"
          - "X_val",   "y_val"
          - "X_test",  "y_test"
    theta : ndarray, shape (4,)
        USDR+ parameters [λ1, λ2, γ, β].
    tau : float
        KRR regularisation strength.
    L : int, optional
        Circuit depth for U_SDR_plus (default: 2).
    entangler : str, optional
        Entangling gate name (default: "cnot").
    run_dir : str or Path
        Ignored for output path, kept only for backward compatibility.
    N : int
        Sample size for this run (for logging / title).
    seed : int
        Seed for this run (for logging / title).
    width, height : float
        Figure size in inches.
    dpi : int
        Figure DPI.
    show : bool
        If True, display the plot with `plt.show()`. Otherwise close it.

    Returns
    -------
    stats : dict
        Nested dictionary with residual statistics per split, e.g.:
        {
          "train": {"mse": ..., "mae": ..., "mean": ..., "std": ...},
          "val":   {...},
          "test":  {...},
        }
    """
    # ------------------------------------------------------------------ #
    # 1. Extract splits
    # ------------------------------------------------------------------ #
    required_train = {"X_train", "y_train"}
    required_val   = {"X_val", "y_val"}
    required_test  = {"X_test", "y_test"}

    if not required_train.issubset(data.keys()):
        raise KeyError("`data` must contain 'X_train' and 'y_train'.")
    if not required_val.issubset(data.keys()):
        raise KeyError("`data` must contain 'X_val' and 'y_val'.")
    if not required_test.issubset(data.keys()):
        raise KeyError("`data` must contain 'X_test' and 'y_test'.")

    X_train = np.asarray(data["X_train"], dtype=np.float64)
    y_train = np.asarray(data["y_train"], dtype=np.float64)
    X_val   = np.asarray(data["X_val"],   dtype=np.float64)
    y_val   = np.asarray(data["y_val"],   dtype=np.float64)
    X_test  = np.asarray(data["X_test"],  dtype=np.float64)
    y_test  = np.asarray(data["y_test"],  dtype=np.float64)

    for name, X, y in [
        ("train", X_train, y_train),
        ("val",   X_val,   y_val),
        ("test",  X_test,  y_test),
    ]:
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X_{name} and y_{name} have incompatible shapes: "
                f"{X.shape} vs {y.shape}"
            )

    # ------------------------------------------------------------------ #
    # 2. Build K_train and regularise with τ
    # ------------------------------------------------------------------ #
    print(
        f"[RES-CCPP] Building K_train and regularising (N={N}, seed={seed})..."
    )

    K_train = build_kernel_matrix(
        X_train,
        X_train,
        theta=theta,
        L=L,
        entangler=entangler,
        apply_psd_hygiene_for_square=True,  # PSD hygiene for Gram matrix
    )

    K_reg, jitter = _regularize_kernel_ccpp(
        K_train,
        tau=tau,
        name=f"K_train (CCPP residuals N={N}, seed={seed})",
    )

    alpha = np.linalg.solve(K_reg, y_train)

    # ------------------------------------------------------------------ #
    # 3. Build K_val, K_test and compute predictions
    # ------------------------------------------------------------------ #
    print(
        f"[RES-CCPP] Building K_val and K_test and computing predictions "
        f"(N={N}, seed={seed})..."
    )

    K_val = build_kernel_matrix(
        X_val,
        X_train,
        theta=theta,
        L=L,
        entangler=entangler,
        apply_psd_hygiene_for_square=False,
    )
    K_test = build_kernel_matrix(
        X_test,
        X_train,
        theta=theta,
        L=L,
        entangler=entangler,
        apply_psd_hygiene_for_square=False,
    )

    y_train_pred = K_train @ alpha
    y_val_pred   = K_val @ alpha
    y_test_pred  = K_test @ alpha

    # ------------------------------------------------------------------ #
    # 4. Residuals and statistics
    # ------------------------------------------------------------------ #
    res_train = y_train - y_train_pred
    res_val   = y_val   - y_val_pred
    res_test  = y_test  - y_test_pred

    stats: Dict[str, Dict[str, float]] = {}

    def _compute_stats(name: str, y_true: np.ndarray, y_pred: np.ndarray, res: np.ndarray):
        mse  = float(mean_squared_error(y_true, y_pred))
        mae  = float(mean_absolute_error(y_true, y_pred))
        mean = float(np.mean(res))
        std  = float(np.std(res, ddof=1)) if res.size > 1 else 0.0

        stats[name] = {
            "mse": mse,
            "mae": mae,
            "mean": mean,
            "std": std,
        }

        print(
            f"[RES-CCPP] {name.upper()} residual statistics "
            f"(N={N}, seed={seed}):\n"
            f"           MSE  = {mse:.4e}\n"
            f"           MAE  = {mae:.4e}\n"
            f"           mean = {mean:.4e}\n"
            f"           std  = {std:.4e}\n"
        )

    _compute_stats("train", y_train, y_train_pred, res_train)
    _compute_stats("val",   y_val,   y_val_pred,   res_val)
    _compute_stats("test",  y_test,  y_test_pred,  res_test)

    print(f"[RES-CCPP] Jitter used in regularisation: {jitter:.2e}")

    # ------------------------------------------------------------------ #
    # 5. Violin + jittered scatter plot
    # ------------------------------------------------------------------ #
    out_dir = Path("figures") / "ccpp" / "residuals"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"residuals_N{N}_seed{seed}.png"

    residual_groups = [res_train, res_val, res_test]
    labels = ["Train", "Val", "Test"]
    positions = np.arange(1, len(residual_groups) + 1)

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

    # Violin plot
    parts = ax.violinplot(
        residual_groups,
        positions=positions,
        showmeans=True,
        showextrema=True,
        showmedians=True,
    )

    # Slight transparency on violins
    for pc in parts["bodies"]:
        pc.set_alpha(0.7)

    # Jittered scatter on top of violins
    rng = np.random.default_rng(seed)

    for i, res in enumerate(residual_groups):
        if res.size == 0:
            continue
        x0 = positions[i]
        x_jitter = x0 + 0.08 * rng.normal(size=res.size)  # horizontal jitter
        ax.scatter(
            x_jitter,
            res,
            s=15,
            alpha=0.7,
            linewidths=0.5,
            edgecolors="k",
        )

    # Zero residual reference line
    ax.axhline(0.0, linestyle="--", linewidth=1.0)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Split")
    ax.set_ylabel("Residual (y_true − y_pred)")

    ax.set_title(
        f"CCPP 2D – Residual distributions (N={N}, seed={seed})\n"
        f"USDR+ KRR, τ={tau:.3e}"
    )
    ax.grid(True, axis="y", linestyle=":")

    # Compact stats box for the TEST split (usually the one you care about)
    test_stats = stats["test"]
    ax.text(
        0.98,
        0.98,
        (
            "Test stats:\n"
            f"MSE  = {test_stats['mse']:.3f}\n"
            f"MAE  = {test_stats['mae']:.3f}\n"
            f"mean = {test_stats['mean']:.3f}\n"
            f"std  = {test_stats['std']:.3f}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.15, edgecolor="none"),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"[RES-CCPP] Saved residual distribution plot → {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return stats

# ===== CELL 100 (code) =====
N_list = [50, 100, 200]
seed_list = [0, 1, 2]

all_stats = {}

for N in N_list:
    for SEED in seed_list:
        print(f"\n=== CCPP residuals for N={N}, seed={SEED} ===")

        data = load_processed_ccpp_2d_dataset(N=N, seed=SEED)
        stats = plot_residual_distributions_ccpp(
            data=data,
            theta=theta_opt,
            tau=tau_opt,
            L=2,
            entangler="cnot",
            run_dir=".",
            N=N,
            seed=SEED,
            show=False,
        )

        all_stats[(N, SEED)] = stats

# ===== CELL 101 (code) =====
pass
