from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

from ..core.events import SpikeEvents
from ..core.solver import DenseNetwork


def solve_batch_parallel(
    input_batches: list[SpikeEvents],
    network: DenseNetwork,
    *,
    max_workers: int | None = None,
    **kwargs,
):
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(network.solve, events, **kwargs) for events in input_batches]
        return [future.result() for future in futures]
