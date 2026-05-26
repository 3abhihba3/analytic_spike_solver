# Design Notes

## Model

The solver implements the feedforward impulse model:

```text
dv_j/dt = -v_j / tau_j + sum_i W[i, j] S_i(t)
```

where `S_i(t)` is an impulse spike train. For an input event `(t, i, a)`,
postsynaptic voltage jumps by:

```text
v[:] += a * W[i, :]
```

Between events:

```text
v_j(t_next) = v_j(t) * exp(-(t_next - t) / tau_j)
```

Thresholding is centered and strict:

```text
if v_j > theta_j / 2:
    emit spike
    v_j -= theta_j
```

If one event group pushes `v_j` across threshold by multiple reset quanta, the
solver emits multiple spikes at the same timestamp using:

```text
count_j = ceil((v_j - theta_j / 2) / theta_j)
```

## Simultaneous Events

All presynaptic events with exactly the same timestamp are summed before any
postsynaptic reset is applied. This is important for dense signed weights:

```text
events at same t:  +0.6 and -0.2
summed jump:       +0.4
result:            no spike
```

Sequentially thresholding those same-time events would be order-dependent and
would not match the continuous-time equation.

## Bias

The solver does not include an analog `b` term in the layer dynamics. Bias is
represented by a single deterministic uniform layer-local spike source. It is
not represented as a normal `SpikeEvents` entry, so there is no reserved id such
as `-1` and no extra row in the dense input weight matrix.

For a target mean bias trace `b_j`, bias source rate `R`, and impulse
weight `w_j`:

```text
mean(v_j) = tau_j * R * w_j
w_j = b_j / (tau_j * R)
```

`DenseLayer` computes `w_j` at initialization when `bias` is supplied. It
validates the easy regime `b_j < theta_j / 2`.

## Bias Noise Recommendation

At fixed total event rate, deterministic uniform bias trains are much less
noisy than Poisson trains. With the included experiment:

```text
tau = 20 ms
target bias = 0.25
uniform bias rate = 100 Hz
```

the uniform source has RMS error about `0.0365`, while Poisson is much noisier.

## Open Decisions

The implementation currently uses these defaults, which are easy to change:

- Strict threshold: `v > theta / 2`, not `>=`; `SolveControls` can switch to inclusive.
- Half-open simulation windows: `[t_start, t_stop)`.
- Output spikes are sorted by time, then neuron id.
- Bias rate means the single uniform source rate.
- `theta` may be explicit per neuron, or constructed with
  `theta_proportional_to_tau`.
- Optional refractory, zero reset, delays, and spike explosion controls are modeled through `SolveControls`/`DenseLayer`.
- No recurrent or lateral connections are modeled by the dense solver.
