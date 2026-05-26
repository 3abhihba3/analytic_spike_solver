# Analytical Spike Solver

CPU event-streaming solver for dense feedforward spiking networks.

## Status

This is a private, proprietary research package. The public API is still
experimental; new code should prefer top-level imports or grouped subpackages.

The implemented model is:

```text
dv_j/dt = -v_j / tau_j + sum_i W[i, j] S_i(t)
if v_j > theta_j / 2: emit spike and v_j -= theta_j
```

`S_i(t)` is treated as an impulse spike train. Between input events, membrane
voltage decays exactly. At an event time, all simultaneous presynaptic events
are summed first, then threshold/reset is applied. This avoids order-dependent
behavior when excitatory and inhibitory spikes share a timestamp.

## Defaults Chosen

- Times are seconds.
- Weight matrix shape is `(n_pre, n_post)`.
- Spikes are arrays of `(time, neuron_id)` with optional amplitudes.
- Threshold is strict: `v > theta / 2`.
- Subtractive reset can emit multiple same-time spikes if a single jump crosses
  threshold by multiple `theta` units.
- `tau` and `theta` may be scalars or per-output-neuron vectors.
- Bias is represented by one deterministic uniform layer-local spike source,
  not by an extra presynaptic row or a reserved event id.
- Simulation windows are half-open: `[t_start, t_stop)`.
- The helper `theta_proportional_to_tau(tau, theta_per_tau)` implements the
  requested `theta ∝ tau` parameterization, while still allowing explicit
  per-neuron `theta` vectors.

## Bias Spikes

For impulse spikes, a uniform bias source with rate `R` and weight `w_j`
produces mean voltage:

```text
mean(v_j) = tau_j * R * w_j
```

So a target constant bias trace `b_j` is approximated by:

```text
w_j = b_j / (tau_j * R)
```

The default bias generator is deterministic uniform firing at `100 Hz`.
`DenseLayer` computes the needed bias impulse weight internally from the target
mean bias:

```python
layer = DenseLayer(weights=W, tau=tau, theta=theta, bias=target_bias)
```

Bias spikes are generated internally from `t_start` and `t_stop`; callers should
not add them to the input `SpikeEvents` array.

The included bias experiment compares uniform bias against a Poisson reference.
With `tau = 20 ms`, target bias `0.25`, and rate `100 Hz`, uniform bias has RMS
error about `0.0365`, while Poisson at the same rate is substantially noisier.

## Complexity

For one dense layer:

```text
E = dynamic input spikes
B = internally generated bias spikes
N = output neurons
Q = output spikes

time work: O((E + B) * N + Q)
state memory: O(N)
weights memory: O(n_pre * N)
```

The solver is analytical in time but still pays for dense event-neuron
interactions. It avoids timestep cost when event count is much smaller than the
number of simulation steps.

## Minimal Example

Install the package in editable mode before running examples or tests:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

```python
import numpy as np

from analytic_spike_solver import DenseLayer, SpikeEvents, solve_layer

events = SpikeEvents(times=[0.01, 0.02], ids=[0, 1])
layer = DenseLayer(
    weights=np.asarray([[0.35, 0.10], [0.30, 0.55]]),
    tau=0.020,
    theta=1.0,
    bias=[0.05, 0.08],
)

result = solve_layer(events, layer, t_start=0.0, t_stop=0.05)
print(result.spikes.times, result.spikes.ids)
```

## Development

Run the local verification suite from the repository root:

```bash
python -m pytest
python -m ruff check .
```

Optional integrations are installed as extras:

```bash
python -m pip install -e ".[plot,brian,numba]"
```

The command-line entry point is:

```bash
analytic-spike-solver benchmark path/to/output.csv
```

## Experiment Features

- Random `DenseLayer.random(...)` and `DenseNetwork.random(...)` constructors.
- Weight initializers: `positive_mean`, `he_signed`, `he_signed_centered`,
  `he_signed_centered_safe`, `he_abs_rescaled`.
- Spike generators for Poisson, regular, jittered regular, bursts, rate
  functions, and continuous-current encoding.
- `DenseNetwork`, `NetworkState`, `SolveControls`, and `NetworkResult`
  support reusable networks, continuation, safety limits, reset/threshold
  modes, delays, refractory windows, timings, and serialization.
- Trace decoding, metrics, monitors, residual tracking, plotting helpers,
  benchmark/stress utilities, and a timestep reference solver are included.

## Package Layout

- `src/analytic_spike_solver/core/`: `SpikeEvents`, solver, network/state/result objects, configs, tau/theta policies.
- `src/analytic_spike_solver/generation/`: bias spike source, spike encoders, random weight initializers.
- `src/analytic_spike_solver/analysis/`: trace decoding, metrics, monitors, residual tracking, plotting.
- `src/analytic_spike_solver/tools/`: experiment runner, benchmarks, stress tests, reference solver, fixtures, parallel batch helper.
- `src/analytic_spike_solver/integrations/`: optional Brian2 comparison, sparse spec, Numba availability hook.
- `examples/`: small executable examples.
- `experiments/`: reproducible experiment scripts; generated outputs are written to `results/` and ignored by git.
- `tests/`: pytest/unittest regression coverage.
