"""CCPP dataset loading, split generation, and persisted preprocessing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from usdr_plus.ccpp import config as cfg


def load_ccpp_dataset(path: str | Path = cfg.DATASET_XLSX) -> pd.DataFrame:
    """Load CCPP Excel and normalize columns to AT, V, AP, RH, EP."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CCPP dataset not found: {path}")

    df = pd.read_excel(path)
    lower = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=lower)

    rename_map: dict[str, str] = {}
    for c in df.columns:
        if c in {"at", "temperature"}:
            rename_map[c] = "AT"
        elif c in {"v", "vacuum"}:
            rename_map[c] = "V"
        elif c in {"ap", "pressure"}:
            rename_map[c] = "AP"
        elif c in {"rh", "humidity"}:
            rename_map[c] = "RH"
        elif c in {"ep", "pe", "power", "output"}:
            rename_map[c] = "EP"
    df = df.rename(columns=rename_map)

    required = {"AT", "V", "AP", "RH", "EP"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required CCPP columns: {sorted(missing)}")
    return df[["AT", "V", "AP", "RH", "EP"]].copy()


def _build_scaler(normalize: str):
    if normalize == "minmax":
        return MinMaxScaler()
    if normalize == "zscore":
        return StandardScaler()
    raise ValueError(f"Unsupported normalize mode: {normalize}")


def preprocess_and_save_ccpp_2d_datasets(
    sample_sizes: list[int] | None = None,
    seeds: list[int] | None = None,
    *,
    normalize: str = cfg.NORMALIZE,
    output_dir: Path = cfg.PREPROCESSED_DIR,
    dataset_path: str | Path = cfg.DATASET_XLSX,
) -> None:
    """Create all N x seed splits and persist train/val/test CSV files."""
    sample_sizes = list(sample_sizes or cfg.SAMPLE_SIZES)
    seeds = list(seeds or cfg.SEEDS)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_ccpp_dataset(dataset_path)
    X_all = df.loc[:, list(cfg.FEATURE_COLUMNS)].to_numpy(dtype=float)
    y_all = df[cfg.TARGET_COLUMN].to_numpy(dtype=float)

    if max(sample_sizes) > len(df):
        raise ValueError(f"Requested N={max(sample_sizes)} but dataset has {len(df)} rows")

    m = len(df)
    for n in sample_sizes:
        for seed in seeds:
            cfg.set_all_seeds(seed)
            run_dir = output_dir / f"N{n}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Match notebook behavior exactly:
            # 1) subsample N rows with a permutation over full dataset
            idx = np.random.permutation(m)[:n]
            X_raw = X_all[idx]
            y = y_all[idx]

            # 2) split 70/15/15 with a fresh permutation over the N subset
            n_train = int(cfg.TRAIN_RATIO * n)
            n_val = int(cfg.VAL_RATIO * n)
            n_test = n - n_train - n_val
            perm = np.random.permutation(n)
            train_idx = perm[:n_train]
            val_idx = perm[n_train:n_train + n_val]
            test_idx = perm[n_train + n_val:]

            X_train_raw, y_train = X_raw[train_idx], y[train_idx]
            X_val_raw, y_val = X_raw[val_idx], y[val_idx]
            X_test_raw, y_test = X_raw[test_idx], y[test_idx]

            scaler = _build_scaler(normalize)
            X_train = scaler.fit_transform(X_train_raw)
            X_val = scaler.transform(X_val_raw)
            X_test = scaler.transform(X_test_raw)
            if normalize == "minmax":
                X_val = np.clip(X_val, 0.0, 1.0)
                X_test = np.clip(X_test, 0.0, 1.0)

            pd.DataFrame({"x1": X_train[:, 0], "x2": X_train[:, 1], "y": y_train}).to_csv(run_dir / "train.csv", index=False)
            pd.DataFrame({"x1": X_val[:, 0], "x2": X_val[:, 1], "y": y_val}).to_csv(run_dir / "val.csv", index=False)
            pd.DataFrame({"x1": X_test[:, 0], "x2": X_test[:, 1], "y": y_test}).to_csv(run_dir / "test.csv", index=False)

            meta: dict[str, Any] = {
                "dataset": "ccpp",
                "features": list(cfg.FEATURE_COLUMNS),
                "target": cfg.TARGET_COLUMN,
                "normalize": normalize,
                "N": int(n),
                "seed": int(seed),
                "n_train": int(n_train),
                "n_val": int(n_val),
                "n_test": int(n_test),
            }
            (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(
                f"[PREPROCESS] Saved N={n}, seed={seed} -> {run_dir} "
                f"({n_train}/{n_val}/{n_test} train/val/test)"
            )


def load_processed_ccpp_2d_dataset(
    *,
    base_path: Path = cfg.PREPROCESSED_DIR,
    N: int,
    seed: int,
    normalize: str = cfg.NORMALIZE,
) -> dict[str, Any]:
    """Load persisted split files from preprocessed/ccpp."""
    run_dir = base_path / f"N{N}_seed{seed}"
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing processed split directory: {run_dir}")

    train_df = pd.read_csv(run_dir / "train.csv")
    val_df = pd.read_csv(run_dir / "val.csv")
    test_df = pd.read_csv(run_dir / "test.csv")
    meta_path = run_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    if normalize and metadata.get("normalize") and metadata["normalize"] != normalize:
        raise ValueError(
            f"Requested normalize={normalize} but split was generated with normalize={metadata['normalize']}"
        )

    return {
        "X_train": train_df[["x1", "x2"]].to_numpy(dtype=float),
        "y_train": train_df["y"].to_numpy(dtype=float),
        "X_val": val_df[["x1", "x2"]].to_numpy(dtype=float),
        "y_val": val_df["y"].to_numpy(dtype=float),
        "X_test": test_df[["x1", "x2"]].to_numpy(dtype=float),
        "y_test": test_df["y"].to_numpy(dtype=float),
        "metadata": metadata,
    }
