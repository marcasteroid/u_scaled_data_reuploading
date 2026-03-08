"""CCPP result aggregation helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from usdr_plus.ccpp import config as cfg


def summarize_results(csv_path: str | Path = cfg.CSV_PATH) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Results CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"N", "val_mse", "test_mse", "kappa_train", "kappa_reg"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Results CSV missing required columns: {sorted(missing)}")

    grouped = (
        df.groupby("N", as_index=False)
        .agg(
            val_mse_mean=("val_mse", "mean"),
            val_mse_std=("val_mse", "std"),
            test_mse_mean=("test_mse", "mean"),
            test_mse_std=("test_mse", "std"),
            kappa_train_mean=("kappa_train", "mean"),
            kappa_reg_mean=("kappa_reg", "mean"),
        )
        .sort_values("N")
    )
    print("\n[CCPP] Aggregated metrics by N:")
    print(grouped.to_string(index=False))
    return grouped
