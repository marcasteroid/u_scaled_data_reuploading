"""Configuration for the modular CCPP pipeline."""

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

DATASET_XLSX = Path("ccpp_dataset.xlsx")
FEATURE_COLUMNS = ("AT", "V")
TARGET_COLUMN = "EP"

SAMPLE_SIZES = [50, 100, 200]
SEEDS = [0, 1, 2]
NORMALIZE = "minmax"  # minmax | zscore

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

PREPROCESSED_DIR = Path("preprocessed/ccpp")
CACHE_DIR = Path("cache/ccpp")
CSV_PATH = Path("csv/ccpp/usdr_plus_ccpp_2d_results_constrained.csv")
FIGURES_DIR = Path("figures/ccpp")
DIAGNOSTICS_DIR = FIGURES_DIR / "diagnostics"
LOGS_DIR = Path("logs/ccpp")

THETA_BOUNDS = {
    "lambda1": (0.1, 5.0),
    "lambda2": (0.1, 5.0),
    "gamma": (1.5, 5.0),
    "beta": (0.5, 3.0),
    "tau": (1e-8, 1e2),
}

OPT_MAXITER = 200
OPT_TOL = 1e-6


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
