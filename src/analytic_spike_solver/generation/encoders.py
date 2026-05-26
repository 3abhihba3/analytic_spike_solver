from __future__ import annotations

import numpy as np

from ..core.events import SpikeEvents
from ..core.random import rng_from_seed


def poisson_events(
    rates_hz: np.ndarray | float,
    *,
    duration: float,
    n_neurons: int | None = None,
    t_start: float = 0.0,
    seed: int | np.random.Generator | None = None,
) -> SpikeEvents:
    rng = rng_from_seed(seed)
    rates = np.asarray(rates_hz, dtype=np.float64)
    if rates.ndim == 0:
        if n_neurons is None:
            n_neurons = 1
        rates = np.full(n_neurons, float(rates))
    times = []
    ids = []
    for i, rate in enumerate(rates):
        if rate < 0:
            raise ValueError("rates must be non-negative")
        t = float(t_start)
        while rate > 0:
            t += float(rng.exponential(1.0 / rate))
            if t >= t_start + duration:
                break
            times.append(t)
            ids.append(i)
    return SpikeEvents(times, ids).sorted()


def regular_events(
    rate_hz: float,
    *,
    duration: float,
    n_neurons: int = 1,
    t_start: float = 0.0,
    phase_s: float = 0.0,
) -> SpikeEvents:
    if rate_hz <= 0:
        return SpikeEvents.empty_events()
    period = 1.0 / rate_hz
    times = []
    ids = []
    for i in range(n_neurons):
        first = t_start + phase_s
        k0 = int(np.ceil((t_start - first) / period))
        k1 = int(np.ceil((t_start + duration - first) / period))
        arr = first + np.arange(k0, k1) * period
        arr = arr[(arr >= t_start) & (arr < t_start + duration)]
        times.extend(arr.tolist())
        ids.extend([i] * len(arr))
    return SpikeEvents(times, ids).sorted()


def jittered_regular_events(
    rate_hz: float,
    *,
    duration: float,
    n_neurons: int = 1,
    jitter_std: float = 0.0,
    seed: int | np.random.Generator | None = None,
) -> SpikeEvents:
    rng = rng_from_seed(seed)
    events = regular_events(rate_hz, duration=duration, n_neurons=n_neurons)
    times = events.times + rng.normal(0.0, jitter_std, len(events))
    mask = (times >= 0.0) & (times < duration)
    return SpikeEvents(times[mask], events.ids[mask], events.amplitudes[mask]).sorted()


def burst_events(
    burst_times: np.ndarray,
    *,
    n_neurons: int,
    spikes_per_burst: int = 3,
    isi: float = 0.001,
) -> SpikeEvents:
    times = []
    ids = []
    for bt in np.asarray(burst_times, dtype=np.float64):
        for k in range(spikes_per_burst):
            t = float(bt) + k * isi
            for i in range(n_neurons):
                times.append(t)
                ids.append(i)
    return SpikeEvents(times, ids).sorted()


def from_rate_function(
    rate_fn,
    *,
    n_neurons: int,
    duration: float,
    dt: float,
    seed: int | np.random.Generator | None = None,
) -> SpikeEvents:
    rng = rng_from_seed(seed)
    times = []
    ids = []
    grid = np.arange(0.0, duration, dt)
    for t in grid:
        rates = np.asarray(rate_fn(float(t)), dtype=np.float64)
        if rates.ndim == 0:
            rates = np.full(n_neurons, float(rates))
        probs = np.clip(rates * dt, 0.0, 1.0)
        fired = np.flatnonzero(rng.random(n_neurons) < probs)
        times.extend([float(t)] * len(fired))
        ids.extend(fired.tolist())
    return SpikeEvents(times, ids)


def current_to_spikes(
    currents: np.ndarray, sample_times: np.ndarray, *, gain_hz: float, seed=None
) -> SpikeEvents:
    currents = np.asarray(currents, dtype=np.float64)
    sample_times = np.asarray(sample_times, dtype=np.float64)
    dt = float(np.median(np.diff(sample_times)))
    rng = rng_from_seed(seed)
    times = []
    ids = []
    for ti, t in enumerate(sample_times):
        rates = np.maximum(currents[ti], 0.0) * gain_hz
        fired = np.flatnonzero(rng.random(rates.shape[0]) < np.clip(rates * dt, 0.0, 1.0))
        times.extend([float(t)] * len(fired))
        ids.extend(fired.tolist())
    return SpikeEvents(times, ids)
