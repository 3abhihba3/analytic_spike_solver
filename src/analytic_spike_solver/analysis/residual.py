from __future__ import annotations

import numpy as np

from ..core.events import SpikeEvents
from ..core.solver import DenseLayer, NetworkResult, SolveControls, _emit_counts
from ..generation.bias import generate_bias_times


def residual_growth(result: NetworkResult, weights_by_layer: list[np.ndarray]) -> list[np.ndarray]:
    """Track effective residuals by v_eff^l = W @ v_eff^(l-1) + v^l at final state."""

    residuals: list[np.ndarray] = []
    prev = None
    for layer_result, weights in zip(result.layer_results, weights_by_layer, strict=True):
        local = layer_result.final_v
        if prev is None:
            eff = local.copy()
        else:
            eff = np.asarray(weights).T @ prev + local
        residuals.append(eff)
        prev = eff
    return residuals


def residual_growth_timeseries(
    input_spikes: SpikeEvents,
    layers: list[DenseLayer],
    result: NetworkResult,
    sample_times: np.ndarray,
    *,
    controls: SolveControls | None = None,
) -> list[np.ndarray]:
    """Track effective residuals through layers on a sample grid.

    Returns one array per layer with shape `(n_neurons, n_times)`.
    """

    controls = controls or SolveControls(track_timing=False)
    sample_times = np.asarray(sample_times, dtype=np.float64)
    local_residuals = []
    incoming = input_spikes
    for layer, layer_result in zip(layers, result.layer_results, strict=True):
        local = _layer_v_trace(incoming, layer, sample_times, controls)
        local_residuals.append(local)
        incoming = layer_result.spikes

    effective = []
    prev = None
    for layer, local in zip(layers, local_residuals, strict=True):
        if prev is None:
            eff = local.copy()
        else:
            eff = layer.weights.T @ prev + local
        effective.append(eff)
        prev = eff
    return effective


def _layer_v_trace(
    input_spikes: SpikeEvents,
    layer: DenseLayer,
    sample_times: np.ndarray,
    controls: SolveControls,
) -> np.ndarray:
    if len(sample_times) == 0:
        return np.zeros((layer.n_post, 0), dtype=np.float64)
    t_start = float(sample_times[0])
    t_stop = float(sample_times[-1]) + 1e-15
    events = input_spikes.clipped(t_start, t_stop).sorted()
    bias_times = (
        generate_bias_times(layer.bias_config, t_start=t_start, t_stop=t_stop)
        if layer.has_bias
        else np.asarray([], dtype=np.float64)
    )
    v = np.zeros(layer.n_post, dtype=np.float64)
    out = np.zeros((layer.n_post, len(sample_times)), dtype=np.float64)
    current_time = t_start
    event_idx = 0
    bias_idx = 0
    for s_idx, sample_t in enumerate(sample_times):
        while True:
            next_event_t = (
                float(events.times[event_idx]) if event_idx < len(events) else float("inf")
            )
            next_bias_t = (
                float(bias_times[bias_idx]) if bias_idx < len(bias_times) else float("inf")
            )
            t = min(next_event_t, next_bias_t)
            if t > sample_t:
                break
            if t > current_time:
                v *= np.exp(-(t - current_time) / layer.tau)
                current_time = t
            if next_event_t == t:
                j = event_idx + 1
                while j < len(events) and events.times[j] == events.times[event_idx]:
                    j += 1
                ids = events.ids[event_idx:j]
                amps = events.amplitudes[event_idx:j]
                v += amps @ layer.weights[ids] if ids.size > 1 else amps[0] * layer.weights[ids[0]]
                event_idx = j
            if next_bias_t == t:
                j = bias_idx + 1
                while j < len(bias_times) and bias_times[j] == t:
                    j += 1
                v += (j - bias_idx) * layer.bias_weight
                bias_idx = j
            emitted = _emit_counts(v, layer.theta, 1e-12, controls.threshold_mode)
            if np.any(emitted):
                if controls.reset_mode == "zero":
                    v[np.flatnonzero(emitted)] = 0.0
                else:
                    v -= emitted * layer.theta
        if sample_t > current_time:
            v *= np.exp(-(sample_t - current_time) / layer.tau)
            current_time = float(sample_t)
        out[:, s_idx] = v
    return out
