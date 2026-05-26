from __future__ import annotations

import numpy as np

from ..core.events import SpikeEvents
from ..core.solver import DenseLayer, SolveControls, _emit_counts


def timestep_solve_layer(
    input_spikes: SpikeEvents,
    layer: DenseLayer,
    *,
    dt: float,
    t_start: float,
    t_stop: float,
    controls: SolveControls | None = None,
) -> SpikeEvents:
    """Slow reference solver for small validation cases."""

    if dt <= 0:
        raise ValueError("dt must be positive")
    controls = controls or SolveControls(track_timing=False)
    events = input_spikes.clipped(t_start, t_stop).sorted()
    v = np.zeros(layer.n_post, dtype=np.float64)
    out_t = []
    out_i = []
    idx = 0
    for t in np.arange(t_start, t_stop, dt):
        v *= np.exp(-dt / layer.tau)
        while idx < len(events) and events.times[idx] < t + dt:
            v += events.amplitudes[idx] * layer.weights[events.ids[idx]]
            idx += 1
        emitted = _emit_counts(v, layer.theta, 1e-12, controls.threshold_mode)
        if np.any(emitted):
            active = np.flatnonzero(emitted)
            ids = np.repeat(active, emitted[active])
            out_t.extend([float(t)] * len(ids))
            out_i.extend(ids.tolist())
            if controls.reset_mode == "zero":
                v[active] = 0.0
            else:
                v -= emitted * layer.theta
    return SpikeEvents(out_t, out_i)
