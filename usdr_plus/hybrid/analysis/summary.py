"""Hybrid results summary."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from usdr_plus.hybrid import config as cfg


def summarize_results(csv_path: str | Path = cfg.CSV_PATH) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Hybrid results CSV not found: {path}")
    df = pd.read_csv(path)
    out = (
        df.groupby("N", as_index=False)
        .agg(
            val_mse_mean=("val_mse", "mean"),
            test_mse_mean=("test_mse", "mean"),
            omega_star_mean=("omega_star", "mean"),
        )
        .sort_values("N")
    )
    print("\n[HYBRID] Summary by N:")
    print(out.to_string(index=False))
    return out
