"""
usdr_plus/data/generator.py
============================
Ground-truth function, raw dataset generation, and 2-D evaluation grid
for the U_{SDR+} 2D Smooth-Interaction Regression benchmark.

    y = sin(x₁) + cos(x₂) + 0.1·x₁·x₂ + ε,   ε ~ N(0, noise_std)
"""

import numpy as np

from usdr_plus.config import RAW_DOMAIN, grid_size, noise_std, sample_sizes


# ---------------------------------------------------------------------------
# Ground-truth function
# ---------------------------------------------------------------------------


def true_function(x1, x2, add_noise: bool = True):
    """
    Ground truth:
        f(x1, x2) = sin(x1) + cos(x2) + 0.1 * x1 * x2
    Target:
        y = f(x1, x2) + ε,  ε ~ N(0, noise_std)
    """
    base = np.sin(x1) + np.cos(x2) + 0.1 * x1 * x2

    if not add_noise:
        return base

    eps = np.random.normal(loc=0.0,
                           scale=noise_std,
                           size=np.broadcast(x1, x2).shape)
    return base + eps


# ---------------------------------------------------------------------------
# Raw dataset generation
# ---------------------------------------------------------------------------


def generate_datasets(sample_sizes, noise_std, domain=RAW_DOMAIN):
    """
    Generate raw datasets for each N in sample_sizes.

    For each N:
      - Sample x1, x2 ~ Uniform(domain[0], domain[1])
      - Compute noiseless ground truth f(x1, x2)
      - Add Gaussian noise ε ~ N(0, noise_std) to obtain y

    Returns
    -------
    datasets : dict
        {
          N: {
            "x1": np.ndarray shape (N,),
            "x2": np.ndarray shape (N,),
            "y":  np.ndarray shape (N,),
            "y_true": np.ndarray shape (N,)
          },
          ...
        }
    """
    datasets = {}

    for N in sample_sizes:
        x1 = np.random.uniform(domain[0], domain[1], size=N)
        x2 = np.random.uniform(domain[0], domain[1], size=N)

        # noiseless ground truth
        y_true = true_function(x1, x2, add_noise=False)

        # add Gaussian noise
        eps = np.random.normal(loc=0.0, scale=noise_std, size=N)
        y_noisy = y_true + eps

        datasets[N] = {
            "x1": x1,
            "x2": x2,
            "y": y_noisy,
            "y_true": y_true,
        }

    return datasets


# ---------------------------------------------------------------------------
# 2-D evaluation grid
# ---------------------------------------------------------------------------


def generate_test_grid(grid_size: int = grid_size, domain=RAW_DOMAIN):
    """
    Generate a 2D evaluation grid over the raw domain.

    Returns
    -------
    X1, X2 : np.ndarray shape (grid_size, grid_size)
        Meshgrid coordinates in [domain[0], domain[1]].
    Y_true : np.ndarray shape (grid_size, grid_size)
        Noiseless ground-truth values f(x1, x2).
    """
    x1 = np.linspace(domain[0], domain[1], grid_size)
    x2 = np.linspace(domain[0], domain[1], grid_size)
    X1, X2 = np.meshgrid(x1, x2)

    # Noiseless ground truth on the grid
    Y_true = true_function(X1, X2, add_noise=False)

    return X1, X2, Y_true
