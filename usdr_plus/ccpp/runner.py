"""Runtime helpers for the modular CCPP pipeline."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import itertools
from pathlib import Path
import threading
import time

from tqdm.auto import tqdm

from usdr_plus.ccpp import config as cfg
from usdr_plus.ccpp.analysis.summary import summarize_results
from usdr_plus.ccpp.data.datasets import preprocess_and_save_ccpp_2d_datasets
from usdr_plus.ccpp.training.pipeline import run_constrained_experiments


def notebook_script_path() -> Path:
    return Path(__file__).resolve().parent / "notebook_pipeline.py"


def run_full_pipeline() -> None:
    cfg.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = cfg.LOGS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    def _set_progress(pbar: tqdm, value: int) -> None:
        value = max(0, min(100, int(value)))
        if value > pbar.n:
            pbar.update(value - pbar.n)

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
        desc="ccpp",
        unit="%",
        bar_format="{l_bar}{bar}| {n:3.0f}% [{elapsed}<{remaining}]",
        ncols=100,
    ) as pbar:
        stop_spin, spin_thread = _start_spinner(pbar)
        _set_progress(pbar, 1)

        def _run_update(done: int, total: int, _n: int, _seed: int) -> None:
            base = 10
            span = 84
            _set_progress(pbar, base + int(span * done / max(total, 1)))

        try:
            with redirect_stdout(log_fh), redirect_stderr(log_fh):
                preprocess_and_save_ccpp_2d_datasets(
                    sample_sizes=cfg.SAMPLE_SIZES,
                    seeds=cfg.SEEDS,
                    normalize=cfg.NORMALIZE,
                    output_dir=cfg.PREPROCESSED_DIR,
                    dataset_path=cfg.DATASET_XLSX,
                )
                _set_progress(pbar, 10)
                run_constrained_experiments(
                    sample_sizes=cfg.SAMPLE_SIZES,
                    seeds=cfg.SEEDS,
                    normalize=cfg.NORMALIZE,
                    base_path=cfg.PREPROCESSED_DIR,
                    csv_out=cfg.CSV_PATH,
                    progress_cb=_run_update,
                )
                _set_progress(pbar, 95)
                summarize_results(cfg.CSV_PATH)
                _set_progress(pbar, 99)
            _set_progress(pbar, 100)
            print(f"CCPP pipeline complete. Log saved to: {log_path}")
        except Exception as exc:
            _set_progress(pbar, 100)
            raise RuntimeError(f"CCPP pipeline failed: {exc}. Check log: {log_path}") from exc
        finally:
            stop_spin.set()
            spin_thread.join(timeout=0.3)
