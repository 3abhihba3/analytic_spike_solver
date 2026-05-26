from __future__ import annotations

import json
from pathlib import Path

from ..core.config import NetworkConfig
from ..core.events import SpikeEvents
from ..core.solver import DenseNetwork


def run_experiment(config: NetworkConfig, input_spikes: SpikeEvents, out_dir: str | Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    network = DenseNetwork.from_config(config)
    result = network.solve(input_spikes, t_start=0.0, t_stop=config.duration)
    (out_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2))
    result.to_npz(out_dir / "result.npz")
    result.write_summary_csv(out_dir / "summary.csv")
    for i, spikes in enumerate(result.spikes_by_layer):
        spikes.to_csv(out_dir / f"layer_{i}_spikes.csv")
    return result
