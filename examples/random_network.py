from __future__ import annotations

from analytic_spike_solver import Sequential, poisson_events


def main() -> None:
    network = Sequential.random([8, 12, 4], seed=1, bias=0.05)
    inputs = poisson_events(20.0, duration=0.2, n_neurons=8, seed=2)
    result = network.solve(inputs, t_stop=0.2)
    print([layer.stats.output_spikes for layer in result.layer_results])


if __name__ == "__main__":
    main()
