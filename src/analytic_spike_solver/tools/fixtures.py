from __future__ import annotations

import numpy as np

from ..core.events import SpikeEvents
from ..core.solver import DenseLayer


def single_neuron_fixture() -> tuple[DenseLayer, SpikeEvents, np.ndarray]:
    layer = DenseLayer(weights=np.asarray([[0.4]]), tau=0.01, theta=1.0)
    events = SpikeEvents([0.0, 0.01 * np.log(2.0)], [0, 0])
    expected = np.asarray([0.01 * np.log(2.0)])
    return layer, events, expected
