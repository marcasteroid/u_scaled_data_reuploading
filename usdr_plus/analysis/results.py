"""
usdr_plus/analysis/results.py
==============================
Results aggregation, CSV persistence, and per-N summary reporting
for the U_{SDR+} constrained-hyperparameter experiment.
"""

from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Summary reporter
# ---------------------------------------------------------------------------


def summarize_usdr_plus_constrained_results(
    csv_path: "str | Path" = "usdr_plus_final_results_constrained.csv",
    df: Optional[pd.DataFrame] = None,
) -> None:
    # --- 1) Load results ---
    if df is None:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"[SUMMARY-CONSTR] Could not find CSV at: {csv_path}"
            )
        df = pd.read_csv(csv_path)

    # Ensure we only look at the constrained experiment
    if "experiment" in df.columns:
        df = df[df["experiment"] == "usdr_plus_constrained"]

    if df.empty:
        print("[SUMMARY-CONSTR] No rows found for experiment='usdr_plus_constrained'.")
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
            "[SUMMARY-CONSTR] Missing expected columns in results DataFrame: "
            + ", ".join(sorted(missing))
        )

    # --- 2) Metrics to summarize ---
    metrics = [
        "val_mse",
        "test_mse",
        "kappa_reg",
        "rank_eff_train",   # we treat raw K_train rank_eff as canonical
    ]

    # --- 3) Per-N aggregates (mean ± std over SEEDS) ---
    print("=== USDR+ (CONSTRAINED RANGES) – PER-N SUMMARY ===\n")
    for N, dfN in df.groupby("N"):
        print(f"--- N = {N} ---")
        for m in metrics:
            mu  = dfN[m].mean()
            std = dfN[m].std(ddof=1)  # sample std
            print(f"{m:15s}: {mu:.4e} ± {std:.4e}")
        print()
