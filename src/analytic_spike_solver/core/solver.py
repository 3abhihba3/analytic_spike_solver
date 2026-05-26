from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from ..generation.bias import BiasSpikeConfig, bias_weight_for_target, generate_bias_times
from .config import NetworkConfig, TauThetaConfig, WeightInitConfig
from .events import SpikeEvents
from .params import make_theta
from ..generation.weights import random_weights


ResetMode = Literal["subtractive", "zero"]
ThresholdMode = Literal["strict", "inclusive"]


@dataclass
class SolveControls:
    max_spikes_per_neuron: int | None = None
    max_total_spikes: int | None = None
    max_spikes_per_event: int | None = None
    threshold_mode: ThresholdMode = "strict"
    reset_mode: ResetMode = "subtractive"
    refractory_s: float = 0.0
    track_timing: bool = True


@dataclass
class DenseLayer:
    """Dense feedforward layer with exact event-driven membrane updates."""

    weights: np.ndarray
    tau: np.ndarray | float
    theta: np.ndarray | float
    bias: np.ndarray | float | None = None
    bias_config: BiasSpikeConfig = field(default_factory=BiasSpikeConfig)
    delay: float = 0.0
    name: str = ""

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=np.float64)
        if self.weights.ndim != 2:
            raise ValueError(f"{self.label}: weights must be 2D, got {self.weights.shape}")
        if self.weights.shape[0] <= 0 or self.weights.shape[1] <= 0:
            raise ValueError(f"{self.label}: weights must have positive shape")
        if not np.all(np.isfinite(self.weights)):
            raise ValueError(f"{self.label}: weights contain non-finite values")
        self.tau = _as_neuron_vector(self.tau, self.n_post, "tau", self.label)
        self.theta = _as_neuron_vector(self.theta, self.n_post, "theta", self.label)
        if np.any(self.tau <= 0) or not np.all(np.isfinite(self.tau)):
            raise ValueError(f"{self.label}: tau values must be positive and finite")
        if np.any(self.theta <= 0) or not np.all(np.isfinite(self.theta)):
            raise ValueError(f"{self.label}: theta values must be positive and finite")
        if self.delay < 0 or not np.isfinite(self.delay):
            raise ValueError(f"{self.label}: delay must be finite and non-negative")
        self.bias_weight = _bias_weight(self.bias, self.tau, self.theta, self.bias_config, self.label)

    @property
    def label(self) -> str:
        return self.name or "DenseLayer"

    @property
    def n_input(self) -> int:
        return int(self.weights.shape[0])

    @property
    def n_post(self) -> int:
        return int(self.weights.shape[1])

    @property
    def has_bias(self) -> bool:
        return self.bias_weight is not None and bool(np.any(self.bias_weight != 0.0))

    @classmethod
    def random(
        cls,
        n_pre: int,
        n_post: int,
        *,
        tau: np.ndarray | float = 0.02,
        theta: np.ndarray | float = 1.0,
        theta_policy: str = "constant",
        theta_per_tau: float | None = None,
        target_gain: float | None = None,
        weight_init: str = "he_signed_centered_safe",
        gain: float = 1.0,
        jitter: float = 0.6,
        max_abs_weight: float | None = None,
        bias: np.ndarray | float | None = None,
        bias_config: BiasSpikeConfig | None = None,
        delay: float = 0.0,
        seed: int | np.random.Generator | None = None,
        name: str = "",
    ) -> DenseLayer:
        tau_arr = np.asarray(tau, dtype=np.float64)
        if tau_arr.ndim == 0:
            tau_value = float(tau_arr)
        else:
            tau_value = tau_arr
        theta_value = make_theta(
            tau_value,
            base_theta=theta,
            policy=theta_policy,
            theta_per_tau=theta_per_tau,
            target_gain=target_gain,
        )
        weights = random_weights(
            n_pre,
            n_post,
            mode=weight_init,
            gain=gain,
            jitter=jitter,
            max_abs_weight=max_abs_weight,
            seed=seed,
        )
        return cls(
            weights=weights,
            tau=tau_value,
            theta=theta_value,
            bias=bias,
            bias_config=bias_config or BiasSpikeConfig(),
            delay=delay,
            name=name,
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "weights": self.weights.tolist(),
            "tau": self.tau.tolist(),
            "theta": self.theta.tolist(),
            "bias": None if self.bias is None else np.asarray(self.bias).tolist(),
            "bias_config": {"rate_hz": self.bias_config.rate_hz, "phase_s": self.bias_config.phase_s},
            "delay": self.delay,
        }


@dataclass
class LayerStats:
    input_events: int
    bias_events: int
    total_events: int
    output_spikes: int
    active_event_times: int
    max_spikes_per_neuron_at_event: int
    max_spikes_per_neuron_total: int
    final_v_mean: float
    final_v_max: float
    output_rate_hz: float
    mean_spikes_per_neuron: float
    active_neuron_fraction: float
    expansion_ratio: float
    runtime_s: float = 0.0


@dataclass
class LayerResult:
    spikes: SpikeEvents
    final_v: np.ndarray
    state_time: float
    spike_counts: np.ndarray
    stats: LayerStats


@dataclass
class NetworkState:
    layer_v: list[np.ndarray]
    state_time: float = 0.0

    @classmethod
    def zeros(cls, layers: list[DenseLayer], state_time: float = 0.0) -> NetworkState:
        return cls([np.zeros(layer.n_post, dtype=np.float64) for layer in layers], state_time)

    def to_npz(self, path: str | Path) -> None:
        data = {"state_time": np.asarray([self.state_time])}
        for i, v in enumerate(self.layer_v):
            data[f"layer_v_{i}"] = v
        np.savez(path, **data)

    @classmethod
    def from_npz(cls, path: str | Path) -> NetworkState:
        with np.load(path) as data:
            keys = sorted(k for k in data.files if k.startswith("layer_v_"))
            return cls([data[k].copy() for k in keys], float(data["state_time"][0]))


@dataclass
class NetworkResult:
    spikes_by_layer: list[SpikeEvents]
    layer_results: list[LayerResult]
    final_state: NetworkState
    runtime_s: float = 0.0

    def to_npz(self, path: str | Path) -> None:
        data = {"runtime_s": np.asarray([self.runtime_s])}
        for i, spikes in enumerate(self.spikes_by_layer):
            data[f"layer_{i}_times"] = spikes.times
            data[f"layer_{i}_ids"] = spikes.ids
            data[f"layer_{i}_amplitudes"] = spikes.amplitudes
            data[f"layer_{i}_final_v"] = self.layer_results[i].final_v
        np.savez(path, **data)

    @classmethod
    def from_npz(cls, path: str | Path) -> NetworkResult:
        with np.load(path) as data:
            layer_ids = sorted(
                int(k.split("_")[1]) for k in data.files if k.endswith("_times")
            )
            spikes = [
                SpikeEvents(
                    data[f"layer_{i}_times"],
                    data[f"layer_{i}_ids"],
                    data[f"layer_{i}_amplitudes"],
                )
                for i in layer_ids
            ]
            dummy_results = [
                LayerResult(
                    s,
                    data[f"layer_{i}_final_v"],
                    0.0,
                    np.bincount(s.ids).astype(np.int64) if len(s) else np.asarray([], dtype=np.int64),
                    LayerStats(0, 0, 0, len(s), 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                )
                for i, s in zip(layer_ids, spikes)
            ]
            return cls(spikes, dummy_results, NetworkState([r.final_v for r in dummy_results]), float(data["runtime_s"][0]))

    def write_summary_csv(self, path: str | Path) -> None:
        rows = []
        for i, result in enumerate(self.layer_results):
            s = result.stats
            rows.append([
                i,
                s.input_events,
                s.bias_events,
                s.total_events,
                s.output_spikes,
                s.expansion_ratio,
                s.output_rate_hz,
                s.runtime_s,
            ])
        with Path(path).open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["layer", "input_events", "bias_events", "total_events", "output_spikes", "expansion_ratio", "output_rate_hz", "runtime_s"])
            writer.writerows(rows)


@dataclass
class DenseNetwork:
    layers: list[DenseLayer]
    name: str = ""

    @classmethod
    def random(
        cls,
        layer_sizes: list[int],
        *,
        tau: float = 0.02,
        theta: float = 1.0,
        weight_init: str = "he_signed_centered_safe",
        gain: float = 1.0,
        bias: float | None = None,
        seed: int | None = None,
        name: str = "",
    ) -> DenseNetwork:
        rng = np.random.default_rng(seed)
        layers = []
        for i, (n_pre, n_post) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layers.append(
                DenseLayer.random(
                    n_pre,
                    n_post,
                    tau=tau,
                    theta=theta,
                    weight_init=weight_init,
                    gain=gain,
                    bias=bias,
                    seed=rng,
                    name=f"L{i}",
                )
            )
        return cls(layers=layers, name=name)

    @classmethod
    def from_config(cls, config: NetworkConfig) -> DenseNetwork:
        rng = np.random.default_rng(config.seed)
        layers = []
        for i, (n_pre, n_post) in enumerate(zip(config.layer_sizes[:-1], config.layer_sizes[1:])):
            tt = config.tau_theta
            wi = config.weight_init
            layers.append(
                DenseLayer.random(
                    n_pre,
                    n_post,
                    tau=tt.tau,
                    theta=tt.theta,
                    theta_policy=tt.theta_policy,
                    theta_per_tau=tt.theta_per_tau,
                    target_gain=tt.target_gain,
                    weight_init=wi.mode,
                    gain=wi.gain,
                    jitter=wi.jitter,
                    max_abs_weight=wi.max_abs_weight,
                    bias=config.bias,
                    bias_config=config.bias_config,
                    seed=rng,
                    name=f"L{i}",
                )
            )
        return cls(layers)

    def solve(self, input_spikes: SpikeEvents, **kwargs) -> NetworkResult:
        return solve_network(input_spikes, self.layers, **kwargs)

    def initial_state(self, state_time: float = 0.0) -> NetworkState:
        return NetworkState.zeros(self.layers, state_time)

    def to_dict(self) -> dict:
        return {"name": self.name, "layers": [layer.to_dict() for layer in self.layers]}


def solve_network(
    input_spikes: SpikeEvents,
    layers: list[DenseLayer],
    *,
    t_start: float = 0.0,
    t_stop: float | None = None,
    initial_states: list[np.ndarray] | None = None,
    state: NetworkState | None = None,
    threshold_eps: float = 1e-12,
    controls: SolveControls | None = None,
    layer_windows: list[tuple[float, float | None]] | None = None,
) -> NetworkResult:
    start = time.perf_counter()
    controls = controls or SolveControls()
    if state is not None:
        initial_states = state.layer_v
        t_start = state.state_time
    if initial_states is not None and len(initial_states) != len(layers):
        raise ValueError(f"initial_states length {len(initial_states)} must match layers {len(layers)}")
    current = input_spikes
    layer_results: list[LayerResult] = []
    spikes_by_layer: list[SpikeEvents] = []
    final_vs: list[np.ndarray] = []
    for layer_idx, layer in enumerate(layers):
        if layer_windows is None:
            layer_t_start, layer_t_stop = t_start, t_stop
        else:
            layer_t_start, layer_t_stop = layer_windows[layer_idx]
        result = solve_layer(
            current,
            layer,
            t_start=layer_t_start,
            t_stop=layer_t_stop,
            initial_v=None if initial_states is None else initial_states[layer_idx],
            threshold_eps=threshold_eps,
            controls=controls,
            layer_index=layer_idx,
        )
        layer_results.append(result)
        current = result.spikes
        spikes_by_layer.append(current)
        final_vs.append(result.final_v)
    runtime = time.perf_counter() - start
    return NetworkResult(spikes_by_layer, layer_results, NetworkState(final_vs, t_stop if t_stop is not None else t_start), runtime)


def solve_layer(
    input_spikes: SpikeEvents,
    layer: DenseLayer,
    *,
    t_start: float = 0.0,
    t_stop: float | None = None,
    initial_v: np.ndarray | None = None,
    threshold_eps: float = 1e-12,
    controls: SolveControls | None = None,
    layer_index: int | None = None,
) -> LayerResult:
    start = time.perf_counter()
    controls = controls or SolveControls()
    label = f"layer {layer_index} ({layer.label})" if layer_index is not None else layer.label
    t_start = float(t_start)
    if t_stop is not None:
        t_stop = float(t_stop)
        if t_stop < t_start:
            raise ValueError(f"{label}: t_stop {t_stop} < t_start {t_start}")
    dynamic = input_spikes.clipped(t_start, t_stop).sorted()
    dynamic.validate_ids(layer.n_input, label=f"{label} input")
    bias_times = np.asarray([], dtype=np.float64)
    if layer.has_bias:
        if t_stop is None:
            raise ValueError(f"{label}: t_stop is required for bias")
        bias_times = generate_bias_times(layer.bias_config, t_start=t_start, t_stop=t_stop)
    v = _initial_state(initial_v, layer.n_post, label)
    refractory_until = np.full(layer.n_post, -np.inf, dtype=np.float64)
    spike_counts = np.zeros(layer.n_post, dtype=np.int64)
    out_times: list[np.ndarray] = []
    out_ids: list[np.ndarray] = []
    current_time = t_start
    max_spikes_per_neuron_at_event = 0
    active_event_times = 0
    dynamic_idx = 0
    bias_idx = 0
    total_emitted = 0

    while dynamic_idx < len(dynamic) or bias_idx < len(bias_times):
        next_dynamic_t = float(dynamic.times[dynamic_idx]) if dynamic_idx < len(dynamic) else float("inf")
        next_bias_t = float(bias_times[bias_idx]) if bias_idx < len(bias_times) else float("inf")
        t = min(next_dynamic_t, next_bias_t)
        if t > current_time:
            v *= np.exp(-(t - current_time) / layer.tau)
            current_time = t
        if next_dynamic_t == t:
            j = dynamic_idx + 1
            while j < len(dynamic) and dynamic.times[j] == dynamic.times[dynamic_idx]:
                j += 1
            ids = dynamic.ids[dynamic_idx:j]
            amps = dynamic.amplitudes[dynamic_idx:j]
            if ids.size == 1:
                v += amps[0] * layer.weights[ids[0]]
            else:
                v += amps @ layer.weights[ids]
            dynamic_idx = j
        if next_bias_t == t:
            j = bias_idx + 1
            while j < len(bias_times) and bias_times[j] == t:
                j += 1
            v += (j - bias_idx) * layer.bias_weight
            bias_idx = j
        if not np.all(np.isfinite(v)):
            raise FloatingPointError(f"{label}: membrane state became non-finite at t={t}")
        emitted = _emit_counts(v, layer.theta, threshold_eps, controls.threshold_mode)
        if controls.refractory_s > 0:
            emitted[current_time < refractory_until] = 0
        if controls.max_spikes_per_event is not None and int(emitted.sum()) > controls.max_spikes_per_event:
            raise RuntimeError(f"{label}: max_spikes_per_event exceeded at t={current_time}")
        if controls.max_spikes_per_neuron is not None and np.any(spike_counts + emitted > controls.max_spikes_per_neuron):
            raise RuntimeError(f"{label}: max_spikes_per_neuron exceeded")
        if controls.max_total_spikes is not None and total_emitted + int(emitted.sum()) > controls.max_total_spikes:
            raise RuntimeError(f"{label}: max_total_spikes exceeded")
        if np.any(emitted):
            active_event_times += 1
            max_spikes_per_neuron_at_event = max(max_spikes_per_neuron_at_event, int(np.max(emitted)))
            active_ids = np.flatnonzero(emitted)
            repeated_ids = np.repeat(active_ids, emitted[active_ids])
            out_times.append(np.full(repeated_ids.shape, current_time + layer.delay, dtype=np.float64))
            out_ids.append(repeated_ids.astype(np.int64, copy=False))
            spike_counts += emitted
            total_emitted += int(emitted.sum())
            refractory_until[active_ids] = current_time + controls.refractory_s
            if controls.reset_mode == "subtractive":
                v -= emitted * layer.theta
            elif controls.reset_mode == "zero":
                v[active_ids] = 0.0
            else:
                raise ValueError(f"unknown reset mode: {controls.reset_mode}")
    state_time = current_time
    if t_stop is not None and t_stop > current_time:
        v *= np.exp(-(t_stop - current_time) / layer.tau)
        state_time = t_stop
    if out_times:
        output_spikes = SpikeEvents(np.concatenate(out_times), np.concatenate(out_ids)).sorted()
    else:
        output_spikes = SpikeEvents.empty_events()
    duration = max((t_stop if t_stop is not None else state_time) - t_start, 1e-12)
    runtime = time.perf_counter() - start if controls.track_timing else 0.0
    stats = LayerStats(
        input_events=len(dynamic),
        bias_events=len(bias_times),
        total_events=len(dynamic) + len(bias_times),
        output_spikes=len(output_spikes),
        active_event_times=active_event_times,
        max_spikes_per_neuron_at_event=max_spikes_per_neuron_at_event,
        max_spikes_per_neuron_total=int(np.max(spike_counts)) if spike_counts.size else 0,
        final_v_mean=float(np.mean(v)),
        final_v_max=float(np.max(v)),
        output_rate_hz=float(len(output_spikes) / duration),
        mean_spikes_per_neuron=float(np.mean(spike_counts)),
        active_neuron_fraction=float(np.mean(spike_counts > 0)),
        expansion_ratio=float(len(output_spikes) / max(len(dynamic), 1)),
        runtime_s=runtime,
    )
    return LayerResult(output_spikes, v, state_time, spike_counts, stats)


def solve_batch(input_batches: list[SpikeEvents], network: DenseNetwork | list[DenseLayer], **kwargs) -> list[NetworkResult]:
    layers = network.layers if isinstance(network, DenseNetwork) else network
    return [solve_network(events, layers, **kwargs) for events in input_batches]


def _as_neuron_vector(value: np.ndarray | float, n: int, name: str, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n, float(arr), dtype=np.float64)
    if arr.shape != (n,):
        raise ValueError(f"{label}: {name} must be scalar or shape ({n},), got {arr.shape}")
    return arr.copy()


def _initial_state(initial_v: np.ndarray | None, n: int, label: str) -> np.ndarray:
    if initial_v is None:
        return np.zeros(n, dtype=np.float64)
    arr = np.asarray(initial_v, dtype=np.float64)
    if arr.shape != (n,):
        raise ValueError(f"{label}: initial_v must have shape ({n},), got {arr.shape}")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{label}: initial_v must be finite")
    return arr.copy()


def _bias_weight(
    bias: np.ndarray | float | None,
    tau: np.ndarray,
    theta: np.ndarray,
    config: BiasSpikeConfig,
    label: str,
) -> np.ndarray | None:
    if bias is None:
        return None
    target = _as_neuron_vector(bias, tau.shape[0], "bias", label)
    if np.all(target == 0.0):
        return np.zeros_like(target)
    weight = bias_weight_for_target(target, tau, config.rate_hz, theta=theta)
    if not np.all(np.isfinite(weight)):
        raise ValueError(f"{label}: bias weights are non-finite")
    return weight


def _emit_counts(v: np.ndarray, theta: np.ndarray, eps: float, threshold_mode: ThresholdMode) -> np.ndarray:
    threshold = theta / 2.0
    if threshold_mode == "strict":
        active = v > threshold + eps
    elif threshold_mode == "inclusive":
        active = v >= threshold - eps
    else:
        raise ValueError(f"unknown threshold mode: {threshold_mode}")
    counts = np.zeros(v.shape, dtype=np.int64)
    if not np.any(active):
        return counts
    ratio = (v[active] - threshold[active]) / theta[active]
    counts[active] = np.maximum(np.floor(ratio + eps).astype(np.int64) + 1, 1)
    return counts
