"""Configuration for modular hybrid pipeline."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

SAMPLE_SIZES = [50, 100, 200]
SEEDS = [0, 1, 2]
NORMALIZE = "minmax"

PREPROCESSED_DIR = Path("preprocessed/hybrid")
CACHE_DIR = Path("cache/hybrid")
CSV_PATH = Path("csv/hybrid/usdr_plus_hybrid_results_constrained.csv")
FIGURES_DIR = Path("figures/hybrid")
DIAGNOSTICS_DIR = FIGURES_DIR / "diagnostics"
EFFECTIVE_RANK_DIR = FIGURES_DIR / "effective_rank"
COMPARISON_DIR = FIGURES_DIR / "comparison"
PRED_DIR = FIGURES_DIR / "pred"
GEOMETRY_DIR = FIGURES_DIR / "geometry"
LOGS_DIR = Path("logs/hybrid")

OMEGA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
CLASSICAL_RBF_CONFIG = {"length_scale": None}
CLASSICAL_LENGTH_SCALES_GRID = [0.1, 0.215443469, 0.464158883, 1.0, 2.15443469, 4.64158883, 10.0]
CLASSICAL_TAU_GRID = [1e-8, 4.641588833612778e-7, 2.1544346900318867e-5, 1e-3, 0.04641588833612782, 2.1544346900318865, 100.0]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
