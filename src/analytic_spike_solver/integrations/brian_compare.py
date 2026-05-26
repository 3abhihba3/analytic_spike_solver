from __future__ import annotations

import numpy as np

from ..core.events import SpikeEvents
from ..core.solver import DenseLayer, SolveControls, solve_layer


def brian2_available() -> bool:
    try:
        import brian2  # noqa: F401
    except Exception:
        return False
    return True


def compare_layer_with_brian2(
    input_spikes: SpikeEvents,
    layer: DenseLayer,
    *,
    t_stop: float,
    dt: float = 1e-4,
) -> dict:
    """Run a small equivalent Brian2 layer and compare spike counts.

    Bias, refractory, delays, and zero-reset are intentionally omitted from the
    Brian model; this harness validates the core impulse/decay/subtractive
    threshold path.
    """

    if not brian2_available():
        return {"available": False, "skipped": True}
    from brian2 import (
        Network,
        NeuronGroup,
        SpikeGeneratorGroup,
        SpikeMonitor,
        Synapses,
        defaultclock,
        second,
    )

    defaultclock.dt = dt * second
    events = input_spikes.sorted()
    source = SpikeGeneratorGroup(layer.n_input, events.ids, events.times * second)
    target = NeuronGroup(
        layer.n_post,
        "dv/dt = -v/tau : 1\ntau : second\ntheta : 1",
        threshold="v > theta / 2",
        reset="v -= theta",
        method="exact",
    )
    target.tau = layer.tau * second
    target.theta = layer.theta
    syn = Synapses(source, target, "w : 1", on_pre="v_post += w")
    pre = np.repeat(np.arange(layer.n_input), layer.n_post)
    post = np.tile(np.arange(layer.n_post), layer.n_input)
    syn.connect(i=pre, j=post)
    syn.w = layer.weights.reshape(-1)
    monitor = SpikeMonitor(target)
    net = Network(source, target, syn, monitor)
    net.run(t_stop * second)

    analytic = solve_layer(
        input_spikes,
        layer,
        t_start=0.0,
        t_stop=t_stop,
        controls=SolveControls(track_timing=False),
    )
    brian_spikes = SpikeEvents(
        np.asarray(monitor.t / second), np.asarray(monitor.i, dtype=np.int64)
    )
    return {
        "available": True,
        "skipped": False,
        "analytic_spikes": len(analytic.spikes),
        "brian_spikes": len(brian_spikes),
        "analytic_times": analytic.spikes.times,
        "brian_times": brian_spikes.times,
        "analytic_ids": analytic.spikes.ids,
        "brian_ids": brian_spikes.ids,
        "count_match": len(analytic.spikes) == len(brian_spikes),
    }


def compare_with_brian2(*args, **kwargs):
    return compare_layer_with_brian2(*args, **kwargs)
