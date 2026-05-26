from __future__ import annotations

import numpy as np

from ..core.events import SpikeEvents


def decode_trace(
    spikes: SpikeEvents,
    *,
    n_neurons: int,
    sample_times: np.ndarray,
    tau: float | np.ndarray,
    weight: float | np.ndarray = 1.0,
) -> np.ndarray:
    sample_times = np.asarray(sample_times, dtype=np.float64)
    if sample_times.ndim != 1:
        raise ValueError("sample_times must be one-dimensional")
    tau_arr = np.asarray(tau, dtype=np.float64)
    if tau_arr.ndim == 0:
        tau_arr = np.full(n_neurons, float(tau_arr))
    if tau_arr.shape != (n_neurons,):
        raise ValueError(f"tau must be scalar or shape ({n_neurons},)")
    weight_arr = np.asarray(weight, dtype=np.float64)
    if weight_arr.ndim == 0:
        weight_arr = np.full(n_neurons, float(weight_arr))
    if weight_arr.shape != (n_neurons,):
        raise ValueError(f"weight must be scalar or shape ({n_neurons},)")
    values = np.zeros((n_neurons, len(sample_times)), dtype=np.float64)
    state = np.zeros(n_neurons, dtype=np.float64)
    current_time = float(sample_times[0]) if len(sample_times) else 0.0
    events = spikes.sorted()
    idx = 0
    for s_idx, t in enumerate(sample_times):
        t = float(t)
        while idx < len(events) and events.times[idx] <= t:
            et = float(events.times[idx])
            if et > current_time:
                state *= np.exp(-(et - current_time) / tau_arr)
                current_time = et
            state[events.ids[idx]] += events.amplitudes[idx] * weight_arr[events.ids[idx]]
            idx += 1
        if t > current_time:
            state *= np.exp(-(t - current_time) / tau_arr)
            current_time = t
        values[:, s_idx] = state
    return values


def lowpass_values(values: np.ndarray, sample_times: np.ndarray, tau: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    sample_times = np.asarray(sample_times, dtype=np.float64)
    out = np.zeros_like(values, dtype=np.float64)
    if len(sample_times) == 0:
        return out
    out[..., 0] = values[..., 0]
    for i in range(1, len(sample_times)):
        decay = np.exp(-(sample_times[i] - sample_times[i - 1]) / tau)
        out[..., i] = values[..., i] + (out[..., i - 1] - values[..., i]) * decay
    return out
