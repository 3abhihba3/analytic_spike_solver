from __future__ import annotations

import math
import unittest

import numpy as np

from analytic_spike_solver import (
    BiasSpikeConfig,
    DenseLayer,
    SpikeEvents,
    bias_weight_for_target,
    generate_bias_times,
    solve_layer,
    solve_network,
    theta_proportional_to_tau,
)


class SpikeSolverTests(unittest.TestCase):
    def test_strict_centered_threshold_does_not_fire_at_exact_half_theta(self):
        layer = DenseLayer(weights=np.asarray([[0.5]]), tau=0.010, theta=1.0)
        result = solve_layer(SpikeEvents([0.0], [0]), layer, t_stop=0.010)
        self.assertEqual(len(result.spikes), 0)

    def test_same_time_events_are_summed_before_thresholding(self):
        layer = DenseLayer(weights=np.asarray([[0.6], [-0.2]]), tau=0.010, theta=1.0)
        events = SpikeEvents([0.0, 0.0], [0, 1])
        result = solve_layer(events, layer, t_stop=0.010)
        self.assertEqual(len(result.spikes), 0)
        self.assertAlmostEqual(result.final_v[0], 0.4 * math.exp(-1.0), places=12)

    def test_multiple_subtractive_resets_can_emit_same_time_spikes(self):
        layer = DenseLayer(weights=np.asarray([[2.6]]), tau=0.010, theta=1.0)
        result = solve_layer(SpikeEvents([0.0], [0]), layer)
        np.testing.assert_allclose(result.spikes.times, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result.spikes.ids, [0, 0, 0])
        self.assertAlmostEqual(result.final_v[0], -0.4, places=12)

    def test_decay_between_events_is_exact(self):
        tau = 0.010
        layer = DenseLayer(weights=np.asarray([[0.4]]), tau=tau, theta=1.0)
        events = SpikeEvents([0.0, tau * math.log(2.0)], [0, 0])
        result = solve_layer(events, layer)
        self.assertEqual(len(result.spikes), 1)
        self.assertAlmostEqual(result.spikes.times[0], tau * math.log(2.0), places=12)
        self.assertAlmostEqual(result.final_v[0], -0.4, places=12)

    def test_network_propagates_sorted_spikes_between_layers(self):
        layer1 = DenseLayer(weights=np.asarray([[0.6, 0.0], [0.0, 0.6]]), tau=0.01, theta=1.0)
        layer2 = DenseLayer(weights=np.asarray([[0.3], [0.6]]), tau=0.01, theta=1.0)
        events = SpikeEvents([0.02, 0.01], [1, 0])
        result = solve_network(events, [layer1, layer2], t_stop=0.05)
        np.testing.assert_allclose(result.spikes_by_layer[0].times, [0.01, 0.02])
        np.testing.assert_array_equal(result.spikes_by_layer[0].ids, [0, 1])
        np.testing.assert_allclose(result.spikes_by_layer[1].times, [0.02])
        np.testing.assert_array_equal(result.spikes_by_layer[1].ids, [0])

    def test_bias_is_internal_not_an_extra_presynaptic_row(self):
        config = BiasSpikeConfig(rate_hz=100.0)
        weights = np.zeros((1, 2))
        layer = DenseLayer(
            weights=weights,
            tau=np.asarray([0.02, 0.04]),
            theta=np.asarray([1.0, 1.0]),
            bias=np.asarray([0.1, 0.2]),
            bias_config=config,
        )
        self.assertEqual(layer.weights.shape, (1, 2))
        np.testing.assert_allclose(layer.bias_weight, [0.05, 0.05])
        result = solve_layer(SpikeEvents.empty_events(), layer, t_start=0.0, t_stop=0.011)
        self.assertEqual(result.stats.bias_events, 2)
        self.assertEqual(result.stats.input_events, 0)

    def test_bias_weight_helper_uses_single_uniform_source_rate(self):
        weight = bias_weight_for_target(
            target_bias=np.asarray([0.1, 0.2]),
            tau=np.asarray([0.02, 0.04]),
            rate_hz=100.0,
            theta=np.asarray([1.0, 1.0]),
        )
        np.testing.assert_allclose(weight, [0.05, 0.05])

    def test_bias_times_are_uniform_and_have_no_ids(self):
        times = generate_bias_times(
            BiasSpikeConfig(rate_hz=100.0),
            t_start=0.0,
            t_stop=0.031,
        )
        np.testing.assert_allclose(times, [0.0, 0.01, 0.02, 0.03])

    def test_layer_with_bias_requires_stop_time(self):
        layer = DenseLayer(weights=np.zeros((3, 1)), tau=0.02, theta=1.0, bias=0.1)
        with self.assertRaises(ValueError):
            solve_layer(SpikeEvents.empty_events(), layer)

    def test_theta_proportional_to_tau_helper(self):
        tau = np.asarray([0.01, 0.02, 0.05])
        theta = theta_proportional_to_tau(tau, theta_per_tau=50.0)
        np.testing.assert_allclose(theta, [0.5, 1.0, 2.5])


if __name__ == "__main__":
    unittest.main()
