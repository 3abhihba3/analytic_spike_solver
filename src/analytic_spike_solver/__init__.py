"""CPU analytical spike solver for dense feedforward SNN layers."""

__version__ = "0.1.0"

from .analysis.metrics import (
    corrcoef,
    covariance_similarity,
    cv,
    effective_rank,
    firing_rates,
    locking_metrics,
    mae_error,
    offdiag_mean_corr,
    rms_error,
    spike_counts,
    target_comparison,
    trace_summary,
)
from .analysis.monitors import PopulationRateMonitor, SpikeMonitor
from .analysis.residual import residual_growth, residual_growth_timeseries
from .analysis.trace import decode_trace, lowpass_values
from .core.config import LayerConfig, NetworkConfig, TauThetaConfig, WeightInitConfig
from .core.events import SpikeEvents
from .core.params import (
    constant_theta,
    make_theta,
    theta_for_target_gain,
    theta_proportional_to_tau,
)
from .core.solver import (
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
from .generation.bias import (
    BiasSpikeConfig,
    bias_weight_for_target,
    centered_biases,
    generate_bias_times,
    residual_balanced_biases,
    scalar_bias,
    zero_bias,
)
from .generation.encoders import (
    burst_events,
    current_to_spikes,
    from_rate_function,
    jittered_regular_events,
    poisson_events,
    regular_events,
)
from .generation.weights import (
    cap_incoming_jump,
    center_and_scale_incoming,
    random_weights,
    rescale_incoming,
)
from .integrations.brian_compare import (
    brian2_available,
    compare_layer_with_brian2,
    compare_with_brian2,
)
from .integrations.numba_accel import numba_available
from .tools.parallel import solve_batch_parallel
from .tools.reference import timestep_solve_layer

__all__ = [
    "BiasSpikeConfig",
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
    "PopulationRateMonitor",
    "Sequential",
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
    "corrcoef",
    "covariance_similarity",
    "cv",
    "current_to_spikes",
    "compare_layer_with_brian2",
    "compare_with_brian2",
    "decode_trace",
    "effective_rank",
    "firing_rates",
    "from_rate_function",
    "generate_bias_times",
    "jittered_regular_events",
    "locking_metrics",
    "lowpass_values",
    "make_theta",
    "mae_error",
    "numba_available",
    "offdiag_mean_corr",
    "poisson_events",
    "random_weights",
    "regular_events",
    "rescale_incoming",
    "residual_balanced_biases",
    "residual_growth",
    "residual_growth_timeseries",
    "rms_error",
    "scalar_bias",
    "solve_batch",
    "solve_batch_parallel",
    "solve_layer",
    "solve_network",
    "spike_counts",
    "target_comparison",
    "timestep_solve_layer",
    "theta_for_target_gain",
    "theta_proportional_to_tau",
    "trace_summary",
    "zero_bias",
]
