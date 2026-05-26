from __future__ import annotations

from ..generation.encoders import poisson_events
from ..core.solver import DenseNetwork, SolveControls


def stress_run(*, n: int = 512, depth: int = 3, duration: float = 0.2, rate_hz: float = 20.0):
    events = poisson_events(rate_hz, duration=duration, n_neurons=n, seed=10)
    net = DenseNetwork.random([n] * (depth + 1), seed=11, bias=0.0)
    return net.solve(events, t_stop=duration, controls=SolveControls(max_total_spikes=5_000_000))
