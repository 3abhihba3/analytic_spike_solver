"""Core solver data structures and execution APIs."""

from .config import LayerConfig, NetworkConfig, TauThetaConfig, WeightInitConfig
from .events import SpikeEvents
from .params import constant_theta, make_theta, theta_for_target_gain, theta_proportional_to_tau
from .solver import (
    Dense,
    DenseLayer,
    DenseNetwork,
    Layer,
    LayerResult,
    LayerStats,
    NetworkResult,
    NetworkState,
    Sequential,
    SolveControls,
    solve_batch,
    solve_layer,
    solve_network,
)

__all__ = [
    "Dense",
    "DenseLayer",
    "DenseNetwork",
    "Layer",
    "LayerConfig",
    "LayerResult",
    "LayerStats",
    "NetworkConfig",
    "NetworkResult",
    "NetworkState",
    "Sequential",
    "SolveControls",
    "SpikeEvents",
    "TauThetaConfig",
    "WeightInitConfig",
    "constant_theta",
    "make_theta",
    "solve_batch",
    "solve_layer",
    "solve_network",
    "theta_for_target_gain",
    "theta_proportional_to_tau",
]
