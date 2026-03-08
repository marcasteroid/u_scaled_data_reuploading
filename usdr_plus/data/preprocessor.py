"""
usdr_plus/data/preprocessor.py
================================
Dataset pre-processing pipeline for U_{SDR+}:
  • 70 / 15 / 15 train / val / test splits (per seed)
  • MinMax  → X ∈ [0,1]²  (fitted on train only)
  • Z-score → standardised (fitted on train only)
  • Save to CSVs under OUTPUT_DIR
  • Load back into structured dicts consumed by the kernel pipeline
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from usdr_plus.config import (
    NORMALIZE,
    OUTPUT_DIR,
    RAW_DOMAIN,
    SEEDS,
    noise_std,
)
from usdr_plus.config import set_all_seeds
from usdr_plus.data.generator import true_function


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def save_split(folder: Path, name: str, X: np.ndarray, y: np.ndarray) -> None:
    """
    Save a split (train/val/test) to CSV with columns x1, x2, y.
    X is assumed to be already preprocessed (e.g. in [0,1]^2 if MinMax).
    """
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["y"] = y
    df.to_csv(folder / f"{name}.csv", index=False)


# ---------------------------------------------------------------------------
# Pre-processing & persistence
# ---------------------------------------------------------------------------


def preprocess_and_save_2d_datasets(
    sample_sizes,
    noise_std,
    output_dir: str = "preprocessed/usdr_plus",
    normalize: str = NORMALIZE,
):
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    for N in sample_sizes:
        for seed in SEEDS:
            # 1) Independent random draw for each (N, seed)
            set_all_seeds(seed)
            x1 = np.random.uniform(RAW_DOMAIN[0], RAW_DOMAIN[1], size=N)
            x2 = np.random.uniform(RAW_DOMAIN[0], RAW_DOMAIN[1], size=N)

            y_true = true_function(x1, x2, add_noise=False)
            eps    = np.random.normal(loc=0.0, scale=noise_std, size=N)
            y      = y_true + eps

            X_raw = np.stack([x1, x2], axis=1)

            # 2) 70/15/15 split (using the *same* RNG stream is fine)
            n_train = int(0.70 * N)
            n_val   = int(0.15 * N)
            n_test  = N - n_train - n_val

            idx = np.random.permutation(N)
            train_idx = idx[:n_train]
            val_idx   = idx[n_train:n_train + n_val]
            test_idx  = idx[n_train + n_val:]

            X_train_raw, y_train = X_raw[train_idx], y[train_idx]
            X_val_raw,   y_val   = X_raw[val_idx],   y[val_idx]
            X_test_raw,  y_test  = X_raw[test_idx],  y[test_idx]

            # 3) Normalization (MinMax or Z-score, train-only)
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
                raise ValueError(f"Unknown NORMALIZE mode: {normalize}")

            # 4) Save
            folder = output_dir_path / f"N{N}_seed{seed}"
            folder.mkdir(parents=True, exist_ok=True)

            save_split(folder, "train", X_train, y_train)
            save_split(folder, "val",   X_val,   y_val)
            save_split(folder, "test",  X_test,  y_test)

            print(
                f"[USDR+] Saved N={N}, seed={seed} → "
                f"{n_train}/{n_val}/{n_test} train/val/test "
                f"(NORMALIZE={normalize})"
            )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_processed_2d_dataset(
    base_path: "str | Path" = OUTPUT_DIR,
    N: int = 100,
    seed: int = 0,
    eps: float = 1e-12,
    normalize: str = NORMALIZE,
) -> dict:
    """
    Fully U_{SDR+}-compliant loader with safe numerical tolerance.

    Assumptions (USDR+ protocol):
      • Data were generated with x1,x2 ∈ [0, 2π] and then preprocessed via
        - MinMax (train-only) → X ∈ [0,1]^2, or
        - Z-score (train-only) → X standardized (unbounded).
      • This function just loads the *preprocessed* CSVs produced by
        `preprocess_and_save_2d_datasets`.

    Parameters
    ----------
    base_path : str or Path
        Root folder where processed datasets live (e.g. OUTPUT_DIR).
    N : int
        Sample size (one of {50, 100, 200}).
    seed : int
        Split seed (0, 1, 2).
    eps : float
        Numerical tolerance for MinMax clipping checks.
    normalize : {"minmax", "zscore"}
        Preprocessing mode used when creating the dataset.
    """
    base_path = Path(base_path)
    folder = base_path / f"N{N}_seed{seed}"
    if not folder.exists():
        raise FileNotFoundError(f"[USDR+] Dataset folder not found: {folder}")

    splits = ["train", "val", "test"]
    data: dict = {
        "metadata": {
            "N": N,
            "seed": seed,
            "normalize": normalize,
            "base_path": str(base_path),
        }
    }

    for split in splits:
        file = folder / f"{split}.csv"
        if not file.is_file():
            raise FileNotFoundError(f"[USDR+] Missing file: {file}")

        df = pd.read_csv(file)
        required = {"x1", "x2", "y"}
        if not required.issubset(df.columns):
            raise ValueError(f"[USDR+] Missing columns in {file}: "
                             f"expected {required}, got {set(df.columns)}")

        X = df[["x1", "x2"]].values.astype(np.float64)
        y = df["y"].values.astype(np.float64)

        if normalize == "minmax":
            # === FINAL FIX: safe clipping + tolerance in [0,1]² ===
            X = np.clip(X, 0.0, 1.0)  # removes 1.0000000000000002 issues
            if not np.all((X >= 0.0 - eps) & (X <= 1.0 + eps)):
                print(f"[USDR+] WARNING: X slightly outside [0,1] in {split} (clipped)")
        elif normalize == "zscore":
            # For Z-score, no clipping: X is standardized, unbounded.
            pass
        else:
            raise ValueError(f"[USDR+] Unknown NORMALIZE mode: {normalize}")

        data[f"X_{split}"] = X
        data[f"y_{split}"] = y
        data["metadata"][f"n_{split}"] = len(y)

        print(
            f"[USDR+] Loaded {split:>5} | "
            f"X shape {X.shape} | y shape {y.shape} | "
            f"NORMALIZE={normalize}"
        )

    print(
        f"\n[USDR+] Dataset N={N}, seed={seed} loaded from {base_path} → "
        f"{data['metadata']['n_train']}/"
        f"{data['metadata']['n_val']}/"
        f"{data['metadata']['n_test']} "
        f"(NORMALIZE={normalize})\n"
    )
    return data
