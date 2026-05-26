"""CPU analytical spike solver for dense feedforward SNN layers."""

__version__ = "0.1.0"

from .generation.bias import (
    BiasSpikeConfig,
    bias_weight_for_target,
    centered_biases,
    generate_bias_times,
    residual_balanced_biases,
    scalar_bias,
    zero_bias,
)
from .core.config import LayerConfig, NetworkConfig, TauThetaConfig, WeightInitConfig
from .integrations.brian_compare import brian2_available, compare_layer_with_brian2, compare_with_brian2
from .generation.encoders import (
    burst_events,
    current_to_spikes,
    from_rate_function,
    jittered_regular_events,
    poisson_events,
    regular_events,
)
from .core.events import SpikeEvents
from .analysis.metrics import *
from .analysis.monitors import PopulationRateMonitor, SpikeMonitor
from .integrations.numba_accel import numba_available
from .core.params import constant_theta, make_theta, theta_for_target_gain, theta_proportional_to_tau
from .tools.parallel import solve_batch_parallel
from .tools.reference import timestep_solve_layer
from .analysis.residual import residual_growth, residual_growth_timeseries
from .core.solver import (
    DenseLayer,
    DenseNetwork,
    LayerResult,
    LayerStats,
    NetworkState,
    NetworkResult,
    SolveControls,
    solve_batch,
    solve_layer,
    solve_network,
)
from .analysis.trace import decode_trace, lowpass_values
from .generation.weights import (
    cap_incoming_jump,
    center_and_scale_incoming,
    random_weights,
    rescale_incoming,
)

__all__ = [
    "BiasSpikeConfig",
    "DenseLayer",
    "DenseNetwork",
    "LayerConfig",
    "LayerResult",
    "LayerStats",
    "NetworkConfig",
    "NetworkResult",
    "NetworkState",
    "PopulationRateMonitor",
    "SolveControls",
    "SpikeEvents",
    "SpikeMonitor",
    "TauThetaConfig",
    "WeightInitConfig",
    "__version__",
    "burst_events",
    "brian2_available",
    "bias_weight_for_target",
    "cap_incoming_jump",
    "center_and_scale_incoming",
    "centered_biases",
    "constant_theta",
    "current_to_spikes",
    "compare_layer_with_brian2",
    "compare_with_brian2",
    "decode_trace",
    "from_rate_function",
    "generate_bias_times",
    "jittered_regular_events",
    "lowpass_values",
    "make_theta",
    "numba_available",
    "poisson_events",
    "random_weights",
    "regular_events",
    "rescale_incoming",
    "residual_balanced_biases",
    "residual_growth",
    "residual_growth_timeseries",
    "scalar_bias",
    "solve_batch",
    "solve_batch_parallel",
    "solve_layer",
    "solve_network",
    "timestep_solve_layer",
    "theta_for_target_gain",
    "theta_proportional_to_tau",
    "zero_bias",
]
