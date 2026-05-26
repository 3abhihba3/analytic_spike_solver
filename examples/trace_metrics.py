from __future__ import annotations

import numpy as np

from analytic_spike_solver import SpikeEvents, decode_trace, trace_summary


def main() -> None:
    spikes = SpikeEvents.regular_train(100.0, 0.1)
    sample_times = np.linspace(0, 0.1, 100)
    traces = decode_trace(spikes, n_neurons=1, sample_times=sample_times, tau=0.02)
    print(trace_summary(traces))


if __name__ == "__main__":
    main()
