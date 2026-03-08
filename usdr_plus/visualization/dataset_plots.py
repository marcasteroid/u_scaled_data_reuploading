"""
usdr_plus/visualization/dataset_plots.py
==========================================
Visualisation utilities for raw datasets, pre-processed splits and
Gram matrices.
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# 3-D surface + scatter
# ---------------------------------------------------------------------------


def plot_2d_dataset(
    datasets: dict,
    X1: np.ndarray,
    X2: np.ndarray,
    Y_true_grid: np.ndarray,
    N: int = 100,
    width: float = 14.0,
    height: float = 6.0,
    dpi: int = 300,
    theme: str = "whitegrid",
) -> None:
    """
    Visualise the **true function surface** (sin(x₁)+cos(x₂)+0.1·x₁·x₂)
    together with the **noisy raw samples** (y) for a given N.

    Fully compliant with U_{SDR+} protocol:
      • uses *raw* data from generate_datasets (before any scaling)
      • 60×60 grid is supplied externally (X1, X2, Y_true_grid)
      • N ∈ {50,100,200}
    """
    sns.set_theme(style=theme, context="talk")
    fig = plt.figure(figsize=(width, height), dpi=dpi)

    # ---- 1. True surface -------------------------------------------------
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax1.plot_surface(
        X1, X2, Y_true_grid,
        cmap="viridis", edgecolor="none", alpha=0.9, antialiased=True
    )
    ax1.set_title(r"True $f(x_1,x_2)$", fontsize=14, pad=12)
    ax1.set_xlabel(r"$x_1$")
    ax1.set_ylabel(r"$x_2$")
    ax1.set_zlabel(r"$y$")
    fig.colorbar(surf, ax=ax1, shrink=0.6, aspect=12)

    # ---- 2. Noisy raw points --------------------------------------------
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    d = datasets[N]  # raw dict from generate_datasets: x1, x2, y, y_true
    ax2.scatter(
        d["x1"],
        d["x2"],
        d["y"],
        s=60,
        c="tab:red",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85,
        label=f"Noisy samples ($N={N}$)",
    )
    ax2.set_title(f"Raw noisy data ($N={N}$)", fontsize=14, pad=12)
    ax2.set_xlabel(r"$x_1$")
    ax2.set_ylabel(r"$x_2$")
    ax2.set_zlabel(r"$y$")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


def plot_all_resolutions(
    datasets: dict,
    X1: np.ndarray,
    X2: np.ndarray,
    Y_true_grid: np.ndarray,
    Ns: list = None,
) -> None:
    """
    Plot true surface + raw noisy samples for all N in Ns (default {50,100,200}).
    """
    if Ns is None:
        Ns = [50, 100, 200]

    n = len(Ns)
    fig, axes = plt.subplots(
        1, n, figsize=(28, 6), dpi=300,
        subplot_kw=dict(projection="3d")
    )

    for ax, N in zip(axes, Ns):
        # true surface
        ax.plot_surface(
            X1, X2, Y_true_grid,
            cmap="viridis", edgecolor="none", alpha=0.9
        )
        # noisy raw points
        d = datasets[N]
        ax.scatter(
            d["x1"], d["x2"], d["y"],
            s=50, c="tab:red", edgecolor="k", alpha=0.8
        )
        ax.set_title(f"$N={N}$")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("$y$")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Pre-processed split inspection
# ---------------------------------------------------------------------------


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
    Visualize **preprocessed** train/val/test splits
    (MinMax or Z-score scaled according to NORMALIZE).

    Compliant with protocol:
      • scaler fitted on train only (see preprocessing function)
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
    print(train_df.to_string())
    print(f"\n=== Validation Split (N={len(val_full)}) ===")
    print(val_df.to_string())
    print(f"\n=== Test Split (N={len(test_full)}) ===")
    print(test_df.to_string())

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
        ax.set_xlabel("x1 (scaled)")
        ax.set_ylabel("x2 (scaled)")
        ax.grid(True, ls="--", alpha=0.6)
        fig.colorbar(sc, ax=ax, shrink=0.7)

    plt.suptitle("Preprocessed Dataset Splits", fontsize=18, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ---------------------------------------------------------------------------
# Gram matrix heatmap
# ---------------------------------------------------------------------------


def plot_gram_matrix(
    matrix: np.ndarray,
    title: str = "Gram Matrix",
    cmap: str = "coolwarm",
    dpi: int = 300,
    width: float = 8,
    height: float = 6,
    annotate: bool = True,
    save: bool = True,
    save_dir: str | Path = "figures/usdr/diagnostics",
) -> None:
    """
    Plot a Gram matrix as a heatmap with diagonal highlighting.
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

    sns.heatmap(
        matrix,
        annot=annotate,
        fmt=".2f",
        cmap=cmap,
        square=True,
        cbar=True,
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
    )

    # Highlight diagonal
    for i in range(matrix.shape[0]):
        ax.add_patch(
            plt.Rectangle(
                (i, i), 1, 1, fill=False, edgecolor='red', lw=2
            )
        )

    ax.set_title(title, fontsize=16, pad=12)
    plt.tight_layout()
    if save:
        slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", title.strip()).strip("_").lower()
        out = Path(save_dir) / f"gram_heatmap_{slug or 'matrix'}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"[PLOT] Saved Gram matrix heatmap -> {out}")
    plt.show()


# ---------------------------------------------------------------------------
# KRR predictions vs true values
# ---------------------------------------------------------------------------


def plot_krr_predictions_vs_true_value(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    width: float = 7,
    height: float = 5,
    dpi: int = 300,
    theme: str = "whitegrid",
    point_color: str = "tab:blue",
    title: str = "KRR Predictions vs True Values",
) -> None:
    """
    Plot Kernel Ridge Regression predictions vs true values.
    """
    sns.set_theme(style=theme, context="talk")

    plt.figure(figsize=(width, height), dpi=dpi)

    # Scatter plot of predictions vs true values
    plt.scatter(
        y_true, y_pred,
        color=point_color, alpha=0.7,
        edgecolors="k", s=70, linewidth=0.6,
        label="Predictions",
    )

    # Ideal line (y = x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--", lw=2, label="Ideal",
    )

    plt.xlabel("True Values", fontsize=12, weight="bold")
    plt.ylabel("Predicted Values", fontsize=12, weight="bold")
    plt.title(title, fontsize=14, weight="bold")

    plt.legend(frameon=True, fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()
