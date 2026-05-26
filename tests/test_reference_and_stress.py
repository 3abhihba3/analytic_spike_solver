from __future__ import annotations

import unittest

import numpy as np

from analytic_spike_solver.tools.reference import timestep_solve_layer
from analytic_spike_solver.analysis.residual import residual_growth_timeseries
from analytic_spike_solver.integrations.sparse import SparseLayerSpec
from analytic_spike_solver.tools.stress import stress_run
from analytic_spike_solver import DenseLayer, SpikeEvents, compare_layer_with_brian2


class ReferenceAndStressTests(unittest.TestCase):
    def test_timestep_reference_smoke(self):
        layer = DenseLayer(weights=np.asarray([[0.6]]), tau=0.01, theta=1.0)
        out = timestep_solve_layer(SpikeEvents([0.0], [0]), layer, dt=0.001, t_start=0.0, t_stop=0.01)
        self.assertGreaterEqual(len(out), 1)

    def test_sparse_spec_from_dense(self):
        spec = SparseLayerSpec.from_dense(np.asarray([[0.0, 1.0], [2.0, 0.0]]))
        self.assertEqual(len(spec.weights), 2)

    def test_small_stress_run(self):
        result = stress_run(n=8, depth=2, duration=0.02, rate_hz=5.0)
        self.assertEqual(len(result.layer_results), 2)

    def test_residual_growth_timeseries(self):
        layer = DenseLayer(weights=np.asarray([[0.6]]), tau=0.01, theta=1.0)
        events = SpikeEvents([0.0], [0])
        from analytic_spike_solver import solve_network

        result = solve_network(events, [layer], t_stop=0.02)
        residual = residual_growth_timeseries(events, [layer], result, np.linspace(0, 0.02, 5))
        self.assertEqual(residual[0].shape, (1, 5))

    def test_brian2_comparison_harness(self):
        layer = DenseLayer(weights=np.asarray([[0.6]]), tau=0.01, theta=1.0)
        comparison = compare_layer_with_brian2(SpikeEvents([0.0], [0]), layer, t_stop=0.005)
        self.assertIn("available", comparison)
        if comparison["available"]:
            self.assertTrue(comparison["count_match"])


if __name__ == "__main__":
    unittest.main()
