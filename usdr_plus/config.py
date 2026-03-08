"""
usdr_plus/config.py
===================
Centralised configuration for the U_{SDR+} experimental protocol.

All hyper-parameter bounds, dataset settings, normalisation mode,
circuit architecture and output paths are defined here so that every
sub-module imports from a single source of truth.
"""

import os
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Thread / BLAS hygiene  (must happen before any numerical imports)
# ---------------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ---------------------------------------------------------------------------
# Domain & noise
# ---------------------------------------------------------------------------
RAW_DOMAIN = (0.0, 2 * np.pi)      # x1, x2 ∈ [0, 2π]  (USDR+ §2.1)

noise_std = 0.05                   # ε ~ N(0, 0.05)
sample_sizes = [50, 100, 200]      # N ∈ {50, 100, 200}
grid_size = 60                     # 60×60 visualisation grid
SEEDS = [0, 1, 2]                  # 3 independent seeds

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("preprocessed/usdr_plus")
BASE_PATH  = OUTPUT_DIR                # single canonical base path
CACHE_DIR  = Path("cache/usdr_plus")
RESULTS_CSV = Path("csv/usdr_plus/usdr_plus_final_results_constrained.csv")
FIGURES_DIR = Path("figures/usdr")

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
# - raw x ∈ [0, 2π] are sampled
# - if NORMALIZE == "minmax": x̃ = x / (2π)   (USDR+ §2.2)
# - circuit then uses  x̂ = x̃ / β
NORMALIZE = "minmax"               # "minmax" | "zscore"

# ---------------------------------------------------------------------------
# Circuit architecture (fixed for USDR+)
# ---------------------------------------------------------------------------
depth     = 2                      # Fixed L = 2
entangler = "cnot"                 # Fixed for USDR+
axes_low  = ("X", "Z")             # Low-freq block
axes_high = ("Z", "X")             # High-freq block

# ---------------------------------------------------------------------------
# Hyperparameter search bounds  θ = (λ₁, λ₂, γ, β)
# ---------------------------------------------------------------------------
theta_bounds = {
    "lambda1": (0.1, 10.0),
    "lambda2": (0.1, 10.0),
    "gamma":   (1.0, 5.0),         # γ ≥ 1
    "beta":    (0.1, 10.0),
}

# ---------------------------------------------------------------------------
# Seed helper
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int) -> None:
    """Set deterministic seeds for NumPy and Python's random module."""
    random.seed(seed)
    np.random.seed(seed)
