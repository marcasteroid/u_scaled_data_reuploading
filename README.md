# $U_{SDR+}$ Quantum Machine Learning Pipelines

Modular research codebase for your MSc thesis experiments in $U_{SDR+}$ quantum kernel regression.

This repository provides three runnable pipelines under one CLI launcher:
- `usdr_plus` (core synthetic benchmark)
- `hybrid` (classical + quantum kernel mixture)
- `ccpp` (real-data benchmark on Combined Cycle Power Plant)

All plots are saved to disk (no interactive windows), with progress bars and logs for full runs.

## 1. Requirements

- Python `>=3.11`
- [uv](https://docs.astral.sh/uv/)

Optional but recommended:
- macOS/Linux shell with `zsh` or `bash`

## 2. Setup

```bash
cd /Users/marco/Research/Quantum/u_scaled_data_reuploading
uv sync
```

Run commands with `uv run ...` (no manual venv activation required).

## 3. Main CLI

Top-level launcher:

```bash
uv run python main.py --help
```

### 3.1 Core USDR+ pipeline (default project)

```bash
uv run python main.py run
```

Equivalent explicit form:

```bash
uv run python main.py usdr-plus run
```

Core subcommands:
- `run`
- `generate`
- `visualize-data`
- `train`
- `summarize`
- `plot-results`

Example:

```bash
uv run python main.py usdr-plus train --seeds 0 1 --sizes 50 100 --normalize minmax
```

### 3.2 Hybrid pipeline

```bash
uv run python main.py hybrid run
```

Extra command:

```bash
uv run python main.py hybrid where
```

### 3.3 CCPP pipeline

```bash
uv run python main.py ccpp run
```

Extra command:

```bash
uv run python main.py ccpp where
```

## 4. Repository Layout

```text
u_scaled_data_reuploading/
├── main.py
├── pyproject.toml
├── usdr_plus/
│   ├── analysis/
│   ├── data/
│   ├── quantum/
│   ├── training/
│   ├── visualization/
│   ├── hybrid/
│   │   ├── analysis/
│   │   ├── data/
│   │   ├── quantum/
│   │   ├── training/
│   │   └── visualization/
│   └── ccpp/
│       ├── analysis/
│       ├── data/
│       ├── quantum/
│       ├── training/
│       └── visualization/
├── preprocessed/
├── cache/
├── csv/
├── figures/
└── logs/
```

Notes:
- `notebook_pipeline.py` files are kept as references, but modular code is used by the runners.
- Generated artifacts are ignored by git through `.gitignore`.

## 5. Output Directories

### 5.1 USDR+
- Preprocessed: `preprocessed/usdr_plus`
- Cache: `cache/usdr_plus`
- CSV: `csv/usdr_plus/usdr_plus_final_results_constrained.csv`
- Figures:
  - Diagnostics: `figures/usdr/diagnostics`
  - Per run: `figures/usdr/N{N}_seed{SEED}`
- Logs: `logs/usdr_plus`

### 5.2 Hybrid
- Preprocessed: `preprocessed/hybrid`
- Cache: `cache/hybrid`
- CSV:
  - `csv/hybrid/usdr_plus_hybrid_results_constrained.csv`
  - `csv/hybrid/usdr_plus_hybrid_results_constrained_omega_grid.csv`
- Figures:
  - Diagnostics: `figures/hybrid/diagnostics`
  - Comparison: `figures/hybrid/comparison`
  - Geometry: `figures/hybrid/geometry/N{N}_seed{SEED}`
  - Per run: `figures/hybrid/N{N}_seed{SEED}`
- Logs: `logs/hybrid`

### 5.3 CCPP
- Preprocessed: `preprocessed/ccpp`
- Cache: `cache/ccpp`
- CSV: `csv/ccpp/usdr_plus_ccpp_2d_results_constrained.csv`
- Figures:
  - Diagnostics: `figures/ccpp/diagnostics`
  - Per run: `figures/ccpp/N{N}_seed{SEED}`
- Logs: `logs/ccpp`

## 6. Plot Policy

- Matplotlib backend is non-interactive (`Agg`).
- `plt.show()` is disabled in runtime paths.
- Figures are saved at high resolution (`dpi=300` where configured).

## 7. Typical Workflow

1. Run core benchmark:
```bash
uv run python main.py run
```

2. Run hybrid benchmark:
```bash
uv run python main.py hybrid run
```

3. Run CCPP benchmark:
```bash
uv run python main.py ccpp run
```

4. Inspect:
- CSV summaries in `csv/`
- per-run and diagnostics plots in `figures/`
- full logs in `logs/`

## 8. Troubleshooting

### `ImportError: Import openpyxl failed`
Install/update dependencies with:

```bash
uv sync
```

Or force reinstall:

```bash
uv pip install --upgrade openpyxl
```

### Long runs or apparent stalls
- Check the corresponding log file in `logs/usdr_plus`, `logs/hybrid`, or `logs/ccpp`.
- Progress bars show global completion percentage; detailed step output is redirected to log files.

## 9. Reproducibility Notes

- Default seeds are fixed in each project config (`[0, 1, 2]`).
- Sample sizes default to `[50, 100, 200]`.
- Preprocessing, caches, CSVs, and figures are stored in deterministic project-specific folders.

## 10. License

MIT (as declared in `pyproject.toml`).
