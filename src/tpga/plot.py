"""Visualisation of GA runs and shipment plans (matplotlib)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from tpga.ga import GAResult  # noqa: E402


def plot_convergence(result: GAResult, lp_optimum: float | None = None) -> Figure:
    """Best cost per generation, with the LP optimum as a reference line if given."""
    fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
    ax.plot(range(len(result.history)), result.history, color="#1f4e79", lw=1.8, label="GA best cost")
    if lp_optimum is not None:
        ax.axhline(lp_optimum, color="#c0392b", ls="--", lw=1.2, label="LP optimum")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Total cost")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def plot_plan(x: np.ndarray, title: str = "Shipment plan") -> Figure:
    """Heat map of a shipment plan; nonzero shipments are annotated."""
    x = np.asarray(x, dtype=float)
    m, n = x.shape
    fig, ax = plt.subplots(figsize=(0.7 * n + 2, 0.6 * m + 1.4), constrained_layout=True)
    im = ax.imshow(x, cmap="Blues", aspect="auto")
    for i in range(m):
        for j in range(n):
            if x[i, j] > 0:
                ax.text(
                    j,
                    i,
                    f"{x[i, j]:g}",
                    ha="center",
                    va="center",
                    color="white" if x[i, j] > 0.6 * x.max() else "black",
                    fontsize=8,
                )
    ax.set_xticks(range(n), [f"D{j + 1}" for j in range(n)])
    ax.set_yticks(range(m), [f"S{i + 1}" for i in range(m)])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Quantity")
    return fig


def save(fig: Figure, path: str | Path, dpi: int = 200) -> None:
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
