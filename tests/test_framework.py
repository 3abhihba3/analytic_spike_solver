from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from analytic_spike_solver import (
    Dense,
    DenseLayer,
    DenseNetwork,
    LayerResult,
    LayerStats,
    NetworkConfig,
    Sequential,
    SolveControls,
    SpikeEvents,
    TauThetaConfig,
    WeightInitConfig,
    centered_biases,
    decode_trace,
    firing_rates,
    jittered_regular_events,
    locking_metrics,
    poisson_events,
    regular_events,
    residual_balanced_biases,
    solve_batch,
    solve_layer,
)


class FrameworkFeatureTests(unittest.TestCase):
    def test_sequential_add_call_predict_and_custom_layer(self):
        class PassThroughLayer:
            name = "passthrough"
            input_size = 2
            output_size = 2

            @property
            def label(self):
                return self.name

            def initial_state(self):
                return np.asarray([], dtype=np.float64)

            def forward(
                self,
                input_spikes,
                *,
                t_start=0.0,
                t_stop=None,
                initial_state=None,
                threshold_eps=1e-12,
                controls=None,
                layer_index=None,
            ):
                clipped = input_spikes.clipped(t_start, t_stop).sorted()
                stats = LayerStats(
                    input_events=len(clipped),
                    bias_events=0,
                    total_events=len(clipped),
                    output_spikes=len(clipped),
                    active_event_times=len(np.unique(clipped.times)) if len(clipped) else 0,
                    max_spikes_per_neuron_at_event=0,
                    max_spikes_per_neuron_total=0,
                    final_v_mean=0.0,
                    final_v_max=0.0,
                    output_rate_hz=0.0,
                    mean_spikes_per_neuron=0.0,
                    active_neuron_fraction=0.0,
                    expansion_ratio=1.0,
                    runtime_s=0.0,
                )
                return LayerResult(
                    clipped,
                    np.asarray([], dtype=np.float64),
                    t_stop if t_stop is not None else t_start,
                    np.asarray([], dtype=np.int64),
                    stats,
                )

            def to_dict(self):
                return {"name": self.name, "type": "PassThroughLayer"}

        model = Sequential(name="ffn")
        model.add(PassThroughLayer())
        model.add(Dense(1, input_size=2, weights=np.asarray([[0.6], [0.0]]), name="out"))

        events = SpikeEvents([0.0], [0])
        called = model(events, t_stop=0.01)
        predicted = model.predict(events, t_stop=0.01)

        self.assertEqual(len(called.spikes_by_layer), 2)
        np.testing.assert_allclose(called.output.times, [0.0])
        np.testing.assert_allclose(called.outputs.times, [0.0])
        np.testing.assert_allclose(predicted.output.times, [0.0])
        self.assertEqual(model.summary()[1]["type"], "Dense")

    def test_sequential_validates_adjacent_layer_sizes(self):
        model = Sequential([Dense(3, input_size=2)])
        with self.assertRaises(ValueError):
            model.add(Dense(1, input_size=4))

    def test_random_layer_and_network_are_reproducible(self):
        a = DenseLayer.random(4, 3, seed=12)
        b = DenseLayer.random(4, 3, seed=12)
        np.testing.assert_allclose(a.weights, b.weights)
        net = DenseNetwork.random([4, 5, 2], seed=5, bias=0.0)
        self.assertEqual(len(net.layers), 2)
        self.assertEqual(net.layers[0].weights.shape, (4, 5))

    def test_network_from_config_and_state_continuation(self):
        cfg = NetworkConfig(
            layer_sizes=[2, 2],
            duration=0.05,
            tau_theta=TauThetaConfig(tau=0.02, theta=1.0),
            weight_init=WeightInitConfig(mode="positive_mean", gain=0.6),
            seed=1,
        )
        net = DenseNetwork.from_config(cfg)
        events = SpikeEvents([0.0], [0])
        first = net.solve(events, t_stop=0.02)
        second = net.solve(SpikeEvents.empty_events(), t_stop=0.04, state=first.final_state)
        self.assertEqual(len(second.final_state.layer_v), 1)

    def test_spike_events_io_and_constructors(self):
        events = SpikeEvents.from_pairs([(0.02, 1), (0.01, 0)]).sorted()
        events.validate_ids(2)
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "events.csv"
            npz_path = Path(td) / "events.npz"
            events.to_csv(csv_path)
            events.to_npz(npz_path)
            self.assertEqual(len(SpikeEvents.from_csv(csv_path)), 2)
            self.assertEqual(len(SpikeEvents.from_npz(npz_path)), 2)

    def test_generators_trace_and_metrics(self):
        reg = regular_events(100.0, duration=0.031, n_neurons=1)
        pois = poisson_events(50.0, duration=0.1, n_neurons=2, seed=2)
        jit = jittered_regular_events(50.0, duration=0.1, n_neurons=1, jitter_std=0.0, seed=1)
        self.assertGreaterEqual(len(reg), 3)
        self.assertGreaterEqual(len(jit), 1)
        rates = firing_rates(pois, 2, 0.1)
        self.assertEqual(rates.shape, (2,))
        traces = decode_trace(reg, n_neurons=1, sample_times=np.linspace(0, 0.05, 10), tau=0.02)
        self.assertEqual(traces.shape, (1, 10))
        lock = locking_metrics(reg, reg, window=0.001, duration=0.05)
        self.assertIn("post_after_pre_ratio", lock)

    def test_controls_reset_threshold_refractory_and_limits(self):
        layer = DenseLayer(weights=np.asarray([[0.5]]), tau=0.01, theta=1.0)
        inclusive = solve_layer(
            SpikeEvents([0.0], [0]),
            layer,
            controls=SolveControls(threshold_mode="inclusive"),
        )
        self.assertEqual(len(inclusive.spikes), 1)
        zero = solve_layer(
            SpikeEvents([0.0], [0]),
            DenseLayer(weights=np.asarray([[2.6]]), tau=0.01, theta=1.0),
            controls=SolveControls(reset_mode="zero"),
        )
        self.assertAlmostEqual(zero.final_v[0], 0.0)
        with self.assertRaises(RuntimeError):
            solve_layer(
                SpikeEvents([0.0], [0]),
                DenseLayer(weights=np.asarray([[2.6]]), tau=0.01, theta=1.0),
                controls=SolveControls(max_spikes_per_event=1),
            )

    def test_delays_batch_bias_helpers_and_serialization(self):
        layer = DenseLayer(weights=np.asarray([[0.6]]), tau=0.01, theta=1.0, delay=0.005)
        result = solve_layer(SpikeEvents([0.0], [0]), layer)
        np.testing.assert_allclose(result.spikes.times, [0.005])
        self.assertEqual(centered_biases(3).shape, (3,))
        self.assertEqual(residual_balanced_biases(3).shape, (3,))
        net = DenseNetwork([layer])
        batch = solve_batch([SpikeEvents([0.0], [0]), SpikeEvents.empty_events()], net)
        self.assertEqual(len(batch), 2)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "result.npz"
            batch[0].to_npz(path)
            loaded = type(batch[0]).from_npz(path)
            self.assertEqual(len(loaded.spikes_by_layer[0]), 1)


if __name__ == "__main__":
    unittest.main()
