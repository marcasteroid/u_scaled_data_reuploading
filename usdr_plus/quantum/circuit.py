"""
usdr_plus/quantum/circuit.py
=============================
U_{SDR+} feature-map circuit definition and PennyLane QNode.

The feature map consists of L=2 data-reuploading layers, each with:
  • Low-freq block  – RX(λ₁ x̂₀) ⊗ RZ(λ₂ x̂₁) then CNOT(0→1)
  • High-freq block – RZ(γ λ₁ x̂₀) ⊗ RX(γ λ₂ x̂₁) then CNOT(0→1)
where x̂ = x̃ / β and x̃ is the pre-processed (MinMax / Z-score) input.
"""

import matplotlib.pyplot as plt
from pathlib import Path
import re
import numpy as np
import pennylane as qml

# ---------------------------------------------------------------------------
# Feature map
# ---------------------------------------------------------------------------


def U_SDR_plus(
    x: np.ndarray,
    theta,
    *,
    L: int = 2,
    entangler: str = "cnot",
) -> None:
    """
    U_{SDR+} feature map.

    Assumptions (USDR+ protocol):
      • x is a *preprocessed* 2D feature vector:
          - if NORMALIZE="minmax": x ∈ [0, 1]^2
          - if NORMALIZE="zscore": x is standardized (unbounded)
      • This routine ONLY applies the bandwidth scaling via β:
            x̂ = x / β
      • L = 2, entangler = "cnot", axes_low = (X,Z), axes_high = (Z,X).

    Parameters
    ----------
    x : array-like of shape (2,)
        Preprocessed input features (x1_tilde, x2_tilde).
    theta : iterable
        (lambda1, lambda2, gamma, beta).
    L : int, default=2
        Number of data-reuploading layers (fixed to 2 in USDR+).
    entangler : {"cnot"}, default="cnot"
        Entangling gate; USDR+ uses CNOT(0→1).
    """
    lambda1, lambda2, gamma, beta = theta

    # USDR+: x̂ = x̃ / β
    x_hat = x / beta

    # Fixed axes as per protocol
    low_axes  = ("X", "Z")  # low-freq block
    high_axes = ("Z", "X")  # high-freq block

    for _ in range(L):
        # --- Low-frequency block (smooth structure) ---
        # axes_low = (X, Z)
        qml.RX(lambda1 * x_hat[0], wires=0)
        qml.RZ(lambda2 * x_hat[1], wires=1)
        if entangler == "cnot":
            qml.CNOT(wires=[0, 1])

        # --- High-frequency block (γ-boost for interactions) ---
        # axes_high = (Z, X)
        qml.RZ(gamma * lambda1 * x_hat[0], wires=0)
        qml.RX(gamma * lambda2 * x_hat[1], wires=1)
        if entangler == "cnot":
            qml.CNOT(wires=[0, 1])


# ---------------------------------------------------------------------------
# QNode
# ---------------------------------------------------------------------------

_dev = qml.device("default.qubit", wires=2)


@qml.qnode(_dev, interface="numpy")
def usdr_plus_state(x, theta):
    """
    U_{SDR+} state preparation QNode.

    Assumptions (USDR+ protocol):
      • x is already *preprocessed*:
          - if NORMALIZE="minmax": x ∈ [0, 1]^2
          - if NORMALIZE="zscore": standardized
      • Circuit applies only β scaling: x̂ = x / β
      • L = 2, entangler = CNOT, axes_low=(X,Z), axes_high=(Z,X).
    """
    # Ensure proper shape
    x = np.asarray(x, dtype=np.float64).ravel()
    assert x.shape == (2,), f"Expected x shape (2,), got {x.shape}"

    # Delegate to the canonical USDR+ feature map
    U_SDR_plus(x, theta, L=2, entangler="cnot")

    return qml.state()


# ---------------------------------------------------------------------------
# Circuit visualisation
# ---------------------------------------------------------------------------


def visualize_U_SDR_plus_2D(
    x_example: np.ndarray,
    theta,
    width: float = 12,
    height: float = 6,
    dpi: int = 300,
    L: int = 2,
    entangler: str = "cnot",
    save: bool = True,
    save_dir: str | Path = "figures/usdr/diagnostics",
    plot_name: str = "usdr_plus_circuit",
) -> None:
    """
    Visualize the U_{SDR+} feature map for a single 2D input.

    Assumptions (USDR+ protocol):
      • x_example is a *preprocessed* feature vector:
          - if NORMALIZE="minmax": x_example ∈ [0, 1]^2
          - if NORMALIZE="zscore": x_example is standardized
      • U_SDR_plus will apply only the β scaling: x̂ = x_example / β
      • L = 2, entangler = "cnot", axes_low=(X,Z), axes_high=(Z,X).
    """
    x_example = np.asarray(x_example, dtype=np.float64).ravel()
    assert x_example.shape == (2,), f"Expected x_example shape (2,), got {x_example.shape}"

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit():
        U_SDR_plus(x_example, theta, L=L, entangler=entangler)
        return qml.state()

    fig, ax = qml.draw_mpl(circuit, decimals=3, expansion_strategy="device")()
    fig.set_size_inches(width, height)
    fig.set_dpi(dpi)
    plt.suptitle(
        r"U$_{\mathrm{SDR}+}$ (2D) – L=2, CNOT, $\gamma$-boost, $\beta$-scaling",
        fontsize=16,
    )
    if save:
        slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", plot_name.strip()).strip("_").lower()
        out = Path(save_dir) / f"circuit_{slug or 'usdr_plus'}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"[PLOT] Saved circuit visualization -> {out}")
    plt.show()
