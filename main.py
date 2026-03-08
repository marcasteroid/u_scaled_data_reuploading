#!/usr/bin/env python
"""
main.py
=======
U_{SDR+} – Command-Line Interface
===================================

Runs the complete U_{SDR+} research pipeline from data generation through
training, evaluation, and result plotting — exactly as executed in the
original Jupyter notebook, but as a reproducible command-line workflow.

Usage
-----
    # Full pipeline end-to-end
    python main.py run

    # Individual stages
    python main.py generate            # generate & pre-process datasets
    python main.py visualize-data      # plot raw datasets & splits
    python main.py train               # optimise θ+τ, build KRR, save CSV
    python main.py summarize           # print per-N aggregated metrics
    python main.py plot-results        # save all result figures

    # Override config at runtime
    python main.py run --seeds 0 1 --sizes 50 100 --normalize minmax

Run `python main.py --help` or `python main.py <command> --help` for details.
"""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import itertools
import sys
import threading
import time
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Suppress non-critical warnings for cleaner CLI output
warnings.filterwarnings("ignore", category=UserWarning)
matplotlib.use("Agg")  # non-interactive backend – figures are saved to disk
plt.show = lambda *args, **kwargs: None  # disable interactive popups globally

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
import usdr_plus.config as cfg
from usdr_plus.analysis.results import summarize_usdr_plus_constrained_results
from usdr_plus.analysis.spectrum import analyze_kernel_matrix, compute_spectrum_metrics
from usdr_plus.config import set_all_seeds
from usdr_plus.data.generator import generate_datasets, generate_test_grid
from usdr_plus.data.preprocessor import (
    load_processed_2d_dataset,
    preprocess_and_save_2d_datasets,
)
from usdr_plus.quantum.circuit import visualize_U_SDR_plus_2D
from usdr_plus.quantum.kernel import build_kernel_matrix
from usdr_plus.training.krr import apply_psd_hygiene, krr_predict
from usdr_plus.training.optimizer import optimize_theta_tau
from usdr_plus.visualization.dataset_plots import (
    plot_2d_dataset,
    plot_all_resolutions,
    plot_gram_matrix,
    visualize_dataset_splits,
)
from usdr_plus.visualization.prediction_plots import (
    plot_1d_slices,
    plot_interaction_ridge,
    plot_prediction_surface_60x60,
    plot_residual_distributions,
    plot_true_vs_predicted,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _banner(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}\n")


def _resolve_config(args: argparse.Namespace):
    """
    Apply CLI overrides on top of the module-level config defaults.
    Returns (seeds, sample_sizes, normalize, output_dir, results_csv).
    """
    seeds        = list(args.seeds)        if args.seeds        else cfg.SEEDS
    sample_sizes = list(args.sizes)        if args.sizes        else cfg.sample_sizes
    normalize    = args.normalize          if args.normalize     else cfg.NORMALIZE
    output_dir   = Path(args.output_dir)   if args.output_dir    else cfg.OUTPUT_DIR
    results_csv  = Path(args.results_csv)  if args.results_csv   else cfg.RESULTS_CSV
    return seeds, sample_sizes, normalize, output_dir, results_csv


# ---------------------------------------------------------------------------
# Stage 1 – Data generation & pre-processing
# ---------------------------------------------------------------------------


def stage_generate(args: argparse.Namespace) -> None:
    _banner("STAGE 1 – Dataset Generation & Pre-processing")

    seeds, sample_sizes, normalize, output_dir, _ = _resolve_config(args)

    print(f"[CONFIG] USDR+ protocol loaded. SEEDS={seeds}, N={sample_sizes}, L={cfg.depth}")
    print(f"[CONFIG] NORMALIZE={normalize}, OUTPUT_DIR={output_dir}\n")

    preprocess_and_save_2d_datasets(
        sample_sizes=sample_sizes,
        noise_std=cfg.noise_std,
        output_dir=str(output_dir),
        normalize=normalize,
    )

    print(f"\n[GENERATE] Done. Datasets saved under: {output_dir.resolve()}")


# ---------------------------------------------------------------------------
# Stage 2 – Dataset visualisation
# ---------------------------------------------------------------------------


def stage_visualize_data(args: argparse.Namespace) -> None:
    _banner("STAGE 2 – Dataset Visualisation")

    seeds, sample_sizes, normalize, output_dir, _ = _resolve_config(args)

    # Raw datasets (seed-0 only for visualisation)
    set_all_seeds(0)
    raw_datasets = generate_datasets(
        sample_sizes=sample_sizes,
        noise_std=cfg.noise_std,
        domain=cfg.RAW_DOMAIN,
    )

    X1, X2, Y_true_grid = generate_test_grid(
        grid_size=cfg.grid_size,
        domain=cfg.RAW_DOMAIN,
    )

    print("[VIZ] Plotting true surface + noisy samples for N=100 …")
    plot_2d_dataset(
        datasets=raw_datasets,
        X1=X1,
        X2=X2,
        Y_true_grid=Y_true_grid,
        N=100,
        width=28,
        height=12,
        dpi=150,
        theme="darkgrid",
    )

    print("[VIZ] Plotting all resolutions …")
    plot_all_resolutions(
        datasets=raw_datasets,
        X1=X1,
        X2=X2,
        Y_true_grid=Y_true_grid,
        Ns=sample_sizes,
    )

    print("[VIZ] Visualising pre-processed splits for N=50, seed=0 …")
    folder = output_dir / "N50_seed0"
    if folder.exists():
        visualize_dataset_splits(
            folder,
            n_rows=15,
            width=20,
            height=6,
            dpi=150,
            theme="whitegrid",
            palette="coolwarm",
        )
    else:
        print(f"[VIZ] WARNING: {folder} not found – run `generate` first.")

    print("[VIZ] Done.")


# ---------------------------------------------------------------------------
# Stage 3 – KRR training & optimisation
# ---------------------------------------------------------------------------


def stage_train(args: argparse.Namespace) -> pd.DataFrame:
    _banner("STAGE 3 – KRR Training (θ+τ optimisation, all N × SEEDS)")

    seeds, sample_sizes, normalize, output_dir, results_csv = _resolve_config(args)
    from sklearn.metrics import mean_squared_error

    results = []

    for N in sample_sizes:
        for SEED in seeds:
            print("\n==============================")
            print(f"=== N={N}, SEED={SEED} (USDR+ CONSTRAINED) ===")
            print("==============================")

            # 0) Deterministic seeding for reproducibility
            set_all_seeds(SEED)

            # --- 1) Load processed dataset splits ---
            data = load_processed_2d_dataset(
                base_path=output_dir,
                N=N,
                seed=SEED,
                normalize=normalize,
            )

            print(
                f"[DATA] n_train={data['metadata']['n_train']}, "
                f"n_val={data['metadata']['n_val']}, "
                f"n_test={data['metadata']['n_test']}"
            )

            # --- 2) Hyperparameter optimization on (train, val) ---
            theta_opt, tau_opt, val_mse = optimize_theta_tau(
                data["X_train"], data["y_train"],
                data["X_val"],   data["y_val"],
            )

            # --- 3) Build train + test kernels with optimal θ ---
            K_train = build_kernel_matrix(
                data["X_train"], data["X_train"], theta_opt
            )
            K_test = build_kernel_matrix(
                data["X_test"], data["X_train"], theta_opt
            )

            print(
                f"[KERNEL] K_train shape={K_train.shape}, "
                f"K_test shape={K_test.shape}"
            )

            # --- 3b) Save core diagnostics for every run ---
            analyze_kernel_matrix(
                K_train,
                name=f"K_train (N={N}, seed={SEED})",
                plot=True,
                save=True,
                save_dir="figures/usdr/diagnostics",
            )
            plot_gram_matrix(
                K_train,
                title=f"Gram Matrix (N={N}, seed={SEED})",
                cmap="viridis",
                annotate=False,
                save=True,
                save_dir="figures/usdr/diagnostics",
            )
            visualize_U_SDR_plus_2D(
                x_example=data["X_train"][0],
                theta=theta_opt,
                width=24,
                height=8,
                save=True,
                save_dir="figures/usdr/diagnostics",
                plot_name=f"n{N}_seed{SEED}",
            )

            # --- 4) Spectrum of raw K_train ---
            spec_train = compute_spectrum_metrics(
                K_train,
                name=f"K_train (N={N}, seed={SEED})",
                log_prefix="[SPEC-CONSTR]",
            )

            # --- 5) PSD hygiene + detailed logging ---
            K_reg, jitter, K_stats, kappa_after = apply_psd_hygiene(
                K_train,
                tau_opt,
                name=f"K_train (N={N}, seed={SEED})",
            )

            # --- 6) Spectrum of regularized K + τI ---
            spec_reg = compute_spectrum_metrics(
                K_reg,
                name=f"K_train+τI (N={N}, seed={SEED})",
                log_prefix="[SPEC-CONSTR]",
            )

            # --- 7) Solve system and evaluate on test ---
            alpha    = np.linalg.solve(K_reg, data["y_train"])
            y_pred   = krr_predict(K_test, alpha)
            test_mse = mean_squared_error(data["y_test"], y_pred)

            print(
                f"[RESULT] Val MSE={val_mse:.4e}, "
                f"Test MSE={test_mse:.4e}, "
                f"κ(K+τI)={spec_reg['kappa']:.3e}, jitter={jitter:.2e}"
            )

            # --- 8) Collect per-run metrics ---
            results.append({
                # Experiment / model identifiers
                "experiment": "usdr_plus_constrained",
                "model":      "usdr_plus",

                # Dataset config
                "N":    N,
                "SEED": SEED,

                # Hyperparameters (θ, τ)
                "lambda1": float(theta_opt[0]),
                "lambda2": float(theta_opt[1]),
                "gamma":   float(theta_opt[2]),
                "beta":    float(theta_opt[3]),
                "tau":     float(tau_opt),

                # Performance
                "val_mse":  float(val_mse),
                "test_mse": float(test_mse),

                # Numerical stability / spectrum (raw K_train)
                "kappa_train":    float(spec_train["kappa"]),
                "min_eig_train":  float(spec_train["min_eig"]),
                "max_eig_train":  float(spec_train["max_eig"]),
                "rank_eff_train": float(spec_train["rank_eff"]),

                # Numerical stability / spectrum (regularized K+τI)
                "kappa_reg":    float(spec_reg["kappa"]),
                "min_eig_reg":  float(spec_reg["min_eig"]),
                "max_eig_reg":  float(spec_reg["max_eig"]),
                "rank_eff_reg": float(spec_reg["rank_eff"]),

                # PSD hygiene extras
                "jitter": float(jitter),
            })

    # --- 9) Save summary CSV ---
    df_results = pd.DataFrame(results)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(results_csv, index=False)
    print(f"\n[SUMMARY] Results saved → {results_csv}")
    print(df_results.to_string())

    return df_results


# ---------------------------------------------------------------------------
# Stage 4 – Results summary
# ---------------------------------------------------------------------------


def stage_summarize(args: argparse.Namespace) -> None:
    _banner("STAGE 4 – Results Summary")

    _, _, _, _, results_csv = _resolve_config(args)

    if not results_csv.exists():
        print(
            f"[SUMMARIZE] ERROR: {results_csv} not found. "
            "Run `train` stage first."
        )
        sys.exit(1)

    summarize_usdr_plus_constrained_results(csv_path=results_csv)


# ---------------------------------------------------------------------------
# Stage 5 – Result plotting
# ---------------------------------------------------------------------------


def stage_plot_results(args: argparse.Namespace) -> None:
    _banner("STAGE 5 – Result Plots (all N × SEEDS)")

    seeds, sample_sizes, normalize, output_dir, results_csv = _resolve_config(args)

    if not results_csv.exists():
        print(
            f"[PLOT] ERROR: {results_csv} not found. "
            "Run `train` stage first."
        )
        sys.exit(1)

    # 1) Load constrained metrics
    df_results_constr = pd.read_csv(results_csv)

    if "experiment" in df_results_constr.columns:
        df_results_constr = df_results_constr[
            df_results_constr["experiment"] == "usdr_plus_constrained"
        ]

    base_fig_dir_constr = cfg.FIGURES_DIR

    for N in sample_sizes:
        for SEED in seeds:
            print("\n=======================================")
            print(f"[PLOTS-CONSTR] Generating plots for N={N}, SEED={SEED}")
            print("=======================================")

            # 2) Select row for (N, SEED)
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

            # 3) Reconstruct θ_opt and τ_opt
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

            # 4) Load dataset splits
            data = load_processed_2d_dataset(
                base_path=output_dir,
                N=N,
                seed=SEED,
                normalize=normalize,
            )

            # 5) Per-run figure directory
            run_dir = base_fig_dir_constr / f"N{N}_seed{SEED}"

            # Only show interactively for one canonical run, e.g. N=100, SEED=0
            show_canonical = (N == 100 and SEED == 0)

            # 6) Call all plotting functions
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

    print("\n[PLOT] All figures saved.")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def stage_run(args: argparse.Namespace) -> None:
    """Execute all five pipeline stages in sequence."""
    log_dir = Path("logs/usdr_plus")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    steps = [
        ("Generate", stage_generate, 15),
        ("Visualize", stage_visualize_data, 30),
        ("Train", stage_train, 75),
        ("Summarize", stage_summarize, 85),
        ("Plot", stage_plot_results, 100),
    ]

    print("\nRunning USDR+ pipeline in progress mode (verbose logs are captured).")
    print(f"Live console: progress bar only | Full log: {log_path}\n")

    def _start_spinner(pbar: tqdm) -> tuple[threading.Event, threading.Thread]:
        stop = threading.Event()

        def _spin() -> None:
            for ch in itertools.cycle("|/-\\"):
                if stop.is_set():
                    break
                pbar.set_postfix_str(f"spin:{ch}", refresh=True)
                time.sleep(0.15)

        t = threading.Thread(target=_spin, daemon=True)
        t.start()
        return stop, t

    with open(log_path, "w", encoding="utf-8") as log_fh, tqdm(
        total=100,
        desc="usdr_plus",
        unit="%",
        bar_format="{l_bar}{bar}| {n:3.0f}% [{elapsed}<{remaining}]",
        ncols=100,
    ) as pbar:
        stop_spin, spin_thread = _start_spinner(pbar)
        try:
            for label, fn, target_pct in steps:
                pbar.set_description(f"usdr_plus:{label.lower()}")
                with redirect_stdout(log_fh), redirect_stderr(log_fh):
                    fn(args)
                if target_pct > pbar.n:
                    pbar.update(target_pct - pbar.n)
            print("\nUSDR+ pipeline complete.")
            print(f"Log saved to: {log_path}")
        except Exception:
            if pbar.n < 100:
                pbar.update(100 - pbar.n)
            print("\nUSDR+ pipeline failed.")
            print(f"Check log: {log_path}")
            raise
        finally:
            stop_spin.set()
            spin_thread.join(timeout=0.3)


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="usdr-plus",
        description=(
            "U_{SDR+} Quantum Kernel Regression Pipeline\n"
            "--------------------------------------------\n"
            "Runs the 2D Smooth-Interaction Regression benchmark\n"
            "using the U_{SDR+} data-reuploading feature map and KRR.\n\n"
            "Available commands:\n"
            "  run             Full pipeline (all stages)\n"
            "  generate        Stage 1: Generate & pre-process datasets\n"
            "  visualize-data  Stage 2: Visualise raw datasets and splits\n"
            "  train           Stage 3: KRR training (θ+τ optimisation)\n"
            "  summarize       Stage 4: Print per-N aggregated metrics\n"
            "  plot-results    Stage 5: Save all result figures"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # Shared arguments added to every sub-command
    def _add_shared(sp):
        sp.add_argument(
            "--seeds",
            nargs="+",
            type=int,
            metavar="S",
            default=None,
            help=f"Random seeds (default: {cfg.SEEDS})",
        )
        sp.add_argument(
            "--sizes",
            nargs="+",
            type=int,
            metavar="N",
            default=None,
            help=f"Sample sizes (default: {cfg.sample_sizes})",
        )
        sp.add_argument(
            "--normalize",
            choices=["minmax", "zscore"],
            default=None,
            help=f"Normalisation mode (default: {cfg.NORMALIZE})",
        )
        sp.add_argument(
            "--output-dir",
            default=None,
            metavar="PATH",
            help=f"Dataset output directory (default: {cfg.OUTPUT_DIR})",
        )
        sp.add_argument(
            "--results-csv",
            default=None,
            metavar="PATH",
            help=f"Results CSV path (default: {cfg.RESULTS_CSV})",
        )
        return sp

    # run
    _add_shared(subparsers.add_parser(
        "run",
        help="Execute the full pipeline end-to-end",
    ))

    # generate
    _add_shared(subparsers.add_parser(
        "generate",
        help="Stage 1 – Generate & pre-process datasets",
    ))

    # visualize-data
    _add_shared(subparsers.add_parser(
        "visualize-data",
        help="Stage 2 – Visualise raw datasets and preprocessed splits",
    ))

    # train
    sp_train = _add_shared(subparsers.add_parser(
        "train",
        help="Stage 3 – KRR training (θ+τ optimisation for all N × SEEDS)",
    ))

    # summarize
    _add_shared(subparsers.add_parser(
        "summarize",
        help="Stage 4 – Print per-N aggregated metrics from results CSV",
    ))

    # plot-results
    _add_shared(subparsers.add_parser(
        "plot-results",
        help="Stage 5 – Save all result figures",
    ))

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

COMMANDS = {
    "run":            stage_run,
    "generate":       stage_generate,
    "visualize-data": stage_visualize_data,
    "train":          stage_train,
    "summarize":      stage_summarize,
    "plot-results":   stage_plot_results,
}


PROJECT_ALIASES = {
    "usdr-plus": "usdr_plus",
    "usdr": "usdr_plus",
    "core": "usdr_plus",
    "hybrid": "hybrid",
    "ccpp": "ccpp",
}


def _print_project_launcher_help() -> None:
    print(
        "\nUSDR+ Multi-Project Launcher\n"
        "===========================\n"
        "Choose which project to run:\n"
        "  usdr-plus   Current modular USDR+ project (default)\n"
        "  hybrid      Modular hybrid USDR+ project\n\n"
        "  ccpp        Modular CCPP USDR+ project\n\n"
        "Examples:\n"
        "  python main.py run\n"
        "  python main.py usdr-plus train --sizes 50 100\n"
        "  python main.py hybrid run\n"
        "  python main.py hybrid where\n"
        "  python main.py ccpp run\n"
        "  python main.py ccpp where\n"
    )


def _dispatch_usdr_plus(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    print(
        f"\n{'='*60}\n"
        f"  U_{{SDR+}} Research Pipeline  |  command: {args.command}\n"
        f"{'='*60}"
    )

    handler = COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    handler(args)
    return 0


def _dispatch_hybrid(argv: list[str]) -> int:
    from usdr_plus.hybrid.cli import main as hybrid_main

    return int(hybrid_main(argv))


def _dispatch_ccpp(argv: list[str]) -> int:
    from usdr_plus.ccpp.cli import main as ccpp_main

    return int(ccpp_main(argv))


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help"}:
        _print_project_launcher_help()
        return

    first = argv[0].strip().lower()
    if first in PROJECT_ALIASES:
        project = PROJECT_ALIASES[first]
        project_argv = argv[1:]
    else:
        # Backward compatibility: old-style commands default to USDR+ project.
        project = "usdr_plus"
        project_argv = argv

    if project == "hybrid":
        code = _dispatch_hybrid(project_argv)
        if code != 0:
            sys.exit(code)
        return

    if project == "ccpp":
        code = _dispatch_ccpp(project_argv)
        if code != 0:
            sys.exit(code)
        return

    code = _dispatch_usdr_plus(project_argv)
    if code != 0:
        sys.exit(code)


if __name__ == "__main__":
    main()
