"""Hybrid dataset pipeline wrappers over the core synthetic dataset utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from usdr_plus.hybrid import config as cfg
import usdr_plus.config as core_cfg
from usdr_plus.data.preprocessor import (
    load_processed_2d_dataset as _load_processed_2d_dataset,
    preprocess_and_save_2d_datasets,
)


def preprocess_and_save_hybrid_datasets(
    sample_sizes: list[int] | None = None,
    *,
    normalize: str = cfg.NORMALIZE,
    output_dir: Path = cfg.PREPROCESSED_DIR,
) -> None:
    sample_sizes = list(sample_sizes or cfg.SAMPLE_SIZES)
    preprocess_and_save_2d_datasets(
        sample_sizes=sample_sizes,
        noise_std=core_cfg.noise_std,
        output_dir=str(output_dir),
        normalize=normalize,
    )


def load_processed_hybrid_dataset(
    *,
    base_path: Path = cfg.PREPROCESSED_DIR,
    N: int,
    seed: int,
    normalize: str = cfg.NORMALIZE,
) -> dict[str, Any]:
    return _load_processed_2d_dataset(
        base_path=base_path,
        N=N,
        seed=seed,
        normalize=normalize,
    )
