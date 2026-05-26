from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.events import SpikeEvents
from ..analysis.metrics import firing_rates, spike_counts


@dataclass
class SpikeMonitor:
    spikes: SpikeEvents
    n_neurons: int

    @property
    def count(self) -> np.ndarray:
        return spike_counts(self.spikes, self.n_neurons)


@dataclass
class PopulationRateMonitor:
    spikes: SpikeEvents
    n_neurons: int
    duration: float

    @property
    def rates_hz(self) -> np.ndarray:
        return firing_rates(self.spikes, self.n_neurons, self.duration)

    @property
    def mean_rate_hz(self) -> float:
        return float(np.mean(self.rates_hz))
