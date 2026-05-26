from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.events import SpikeEvents


def plot_raster(spikes: SpikeEvents, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    path = Path(path)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(spikes.times, spikes.ids, "|", markersize=4)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("neuron id")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_traces(sample_times: np.ndarray, traces: np.ndarray, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    path = Path(path)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(sample_times, np.asarray(traces).T, linewidth=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("trace")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
