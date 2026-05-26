from __future__ import annotations

import csv
from pathlib import Path

from ..core.solver import DenseNetwork, SolveControls
from ..generation.encoders import poisson_events


def run_benchmark(path: str | Path, *, sizes=(128, 512), events=(100, 1000), depth=3) -> Path:
    path = Path(path)
    rows = []
    for n in sizes:
        for e in events:
            duration = 1.0
            rate = e / max(n, 1) / duration
            inputs = poisson_events(rate, duration=duration, n_neurons=n, seed=1)
            net = DenseNetwork.random([n] * (depth + 1), seed=2, bias=0.0)
            result = net.solve(
                inputs, t_stop=duration, controls=SolveControls(max_total_spikes=10_000_000)
            )
            rows.append(
                [
                    n,
                    e,
                    depth,
                    len(inputs),
                    result.runtime_s,
                    result.layer_results[-1].stats.output_spikes,
                ]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["N", "target_E", "depth", "actual_input_events", "runtime_s", "final_output_spikes"]
        )
        writer.writerows(rows)
    return path
