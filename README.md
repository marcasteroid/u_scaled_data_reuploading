# U_{SDR+} — Scaled Data Re-Uploading Quantum Kernel Regression

> **2D Smooth-Interaction Regression Benchmark with Hyperparameter Bounds**

This project implements the **U_{SDR+}** quantum kernel regression pipeline
as a fully reproducible, enterprise-grade Python package.

The target function is:

$$y = \sin(x_1) + \cos(x_2) + 0.1\, x_1 x_2 + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, 0.05)$$

---

## Project Structure

```
u_scaled_data_reuploading/
├── main.py                         # CLI entry point (all pipeline stages)
├── pyproject.toml                  # uv-managed dependencies
├── .python-version                 # Python 3.11
├── README.md
│
└── usdr_plus/                      # Python package
    ├── config.py                   # All experimental constants & paths
    ├── data/
    │   ├── generator.py            # true_function, generate_datasets, generate_test_grid
    │   └── preprocessor.py         # preprocess_and_save, load_processed_2d_dataset
    ├── quantum/
    │   ├── circuit.py              # U_SDR_plus feature map, usdr_plus_state QNode
    │   └── kernel.py               # fidelity_2d, build_quantum_kernel_matrix
    ├── training/
    │   ├── optimizer.py            # optimize_theta_tau (L-BFGS-B + Optuna fallback)
    │   └── krr.py                  # krr_predict, apply_psd_hygiene
    ├── analysis/
    │   ├── spectrum.py             # analyze_kernel_matrix, compute_spectrum_metrics
    │   └── results.py              # summarize_usdr_plus_constrained_results
    ├── visualization/
    │   ├── dataset_plots.py        # 3-D surfaces, split inspection, Gram heatmap
    │   └── prediction_plots.py     # prediction surface, 1D slices, ridge, residuals
    └── utils/
        └── cache.py                # joblib Memory, _safe_psd_hygiene, _hash_array
```

---

## Quick Start

### 1. Install with `uv`

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
cd u_scaled_data_reuploading
uv sync

# Activate (optional — uv run handles this automatically)
source .venv/bin/activate
```

### 2. Run the full pipeline

```bash
uv run python main.py run
```

### 3. Run individual stages

```bash
# Stage 1 – Generate and pre-process datasets
uv run python main.py generate

# Stage 2 – Visualise raw datasets and processed splits
uv run python main.py visualize-data

# Stage 3 – KRR training (θ+τ optimisation for all N × SEEDS)
uv run python main.py train

# Stage 4 – Print per-N aggregated metric summary
uv run python main.py summarize

# Stage 5 – Save all result figures
uv run python main.py plot-results
```

### 4. Override configuration at runtime

```bash
# Use only seeds 0 and 1, sample size N=100 only, Z-score normalisation
uv run python main.py run --seeds 0 1 --sizes 100 --normalize zscore

# Redirect output to a custom directory
uv run python main.py generate --output-dir /path/to/my_datasets
```

---

## CLI Reference

```
usage: usdr-plus COMMAND [OPTIONS]

Commands:
  run             Full pipeline (all stages)
  generate        Stage 1 – Generate & pre-process datasets
  visualize-data  Stage 2 – Visualise raw datasets and preprocessed splits
  train           Stage 3 – KRR training (θ+τ optimisation for all N × SEEDS)
  summarize       Stage 4 – Print per-N aggregated metrics from results CSV
  plot-results    Stage 5 – Save all result figures

Shared Options (all commands):
  --seeds  S [S ...]           Random seeds (default: [0, 1, 2])
  --sizes  N [N ...]           Sample sizes (default: [50, 100, 200])
  --normalize {minmax,zscore}  Normalisation mode (default: minmax)
  --output-dir PATH            Dataset output directory
  --results-csv PATH           Results CSV path
```

---

## Experimental Protocol

| Parameter | Value |
|-----------|-------|
| Domain | $x_1, x_2 \in [0, 2\pi]$ |
| Noise | $\varepsilon \sim \mathcal{N}(0, 0.05)$ |
| Sample sizes | $N \in \{50, 100, 200\}$ |
| Seeds | $\{0, 1, 2\}$ |
| Split | 70 / 15 / 15 (train / val / test) |
| Normalisation | MinMax (train-only fit) |
| Layers | $L = 2$ |
| Entangler | CNOT$(0 \to 1)$ |
| Axes (low-freq) | $(R_X, R_Z)$ |
| Axes (high-freq) | $(R_Z, R_X)$ |

### Hyperparameter Bounds (θ)

| Parameter | Range |
|-----------|-------|
| $\lambda_1, \lambda_2$ | $[0.1, 5.0]$ |
| $\gamma$ | $[1.5, 5.0]$ |
| $\beta$ | $[0.5, 3.0]$ |
| $\tau$ (regularisation) | $[10^{-8}, 10^2]$ |

Optimisation: **L-BFGS-B** in log-space (primary), **Optuna TPE** fallback (100 trials).

---

## Output Files

| Path | Description |
|------|-------------|
| `processed_2D_L2_plus_hyp_bo/N{N}_seed{S}/` | Pre-processed CSV splits |
| `usdr_plus_final_results_constrained.csv` | Full results table |
| `figures/usdr_plus_constrained/N{N}_seed{S}/prediction_surface.png` | 60×60 prediction surface |
| `figures/usdr_plus_constrained/N{N}_seed{S}/slices.png` | 1-D function slices |
| `figures/usdr_plus_constrained/N{N}_seed{S}/interaction_ridge.png` | Interaction term recovery |
| `figures/usdr_plus_constrained/N{N}_seed{S}/true_vs_pred.png` | True vs predicted scatter |
| `figures/usdr_plus_constrained/N{N}_seed{S}/residuals_violin.png` | Residual violin plots |
| `usdr_plus_cache_L2_hyp_bo/` | joblib disk cache |

---

## Dependencies

Core: `numpy`, `pandas`, `scipy`, `scikit-learn`, `pennylane`, `optuna`, `joblib`, `matplotlib`, `seaborn`, `tqdm`

Dev: `pytest`, `ruff`, `mypy`, `ipykernel`, `jupyter`

---

## License

MIT
