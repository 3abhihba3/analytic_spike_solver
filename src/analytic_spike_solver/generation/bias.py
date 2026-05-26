from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BiasSpikeConfig:
    """Single layer-local bias spike source.

    The bias source fires one deterministic uniform spike train. Bias spikes are
    not represented as normal `SpikeEvents` and therefore do not need a reserved
    neuron id or an extra weight row.
    """

    rate_hz: float = 100.0
    phase_s: float = 0.0

    def __post_init__(self) -> None:
        if self.rate_hz <= 0:
            raise ValueError("rate_hz must be positive")


def generate_bias_times(
    config: BiasSpikeConfig,
    *,
    t_start: float,
    t_stop: float,
) -> np.ndarray:
    """Generate deterministic uniform bias spike times in [t_start, t_stop)."""

    t_start = float(t_start)
    t_stop = float(t_stop)
    if t_stop < t_start:
        raise ValueError("t_stop must be greater than or equal to t_start")
    if t_stop == t_start:
        return np.asarray([], dtype=np.float64)
    period = 1.0 / float(config.rate_hz)
    first = float(config.phase_s)
    k0 = int(np.ceil((t_start - first) / period))
    k1 = int(np.ceil((t_stop - first) / period))
    if k1 <= k0:
        return np.asarray([], dtype=np.float64)
    times = first + np.arange(k0, k1, dtype=np.float64) * period
    return times[(times >= t_start) & (times < t_stop)]


def bias_weight_for_target(
    target_bias: np.ndarray | float,
    tau: np.ndarray | float,
    rate_hz: float,
    *,
    theta: np.ndarray | float | None = None,
) -> np.ndarray:
    """Return the impulse weight vector for a target mean bias voltage.

    For impulse spikes, one uniform source with rate R and weight w produces
    mean voltage tau * R * w. Therefore w = target_bias / (tau * R).
    """

    if rate_hz <= 0:
        raise ValueError("rate_hz must be positive")
    target = np.asarray(target_bias, dtype=np.float64)
    tau_arr = np.asarray(tau, dtype=np.float64)
    if np.any(tau_arr <= 0):
        raise ValueError("tau must be positive")
    if theta is not None:
        theta_arr = np.asarray(theta, dtype=np.float64)
        if np.any(target >= theta_arr / 2):
            raise ValueError("target_bias must be less than theta / 2")
    return np.asarray(target / (tau_arr * float(rate_hz)), dtype=np.float64)


def zero_bias(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.float64)


def scalar_bias(value: float, n: int) -> np.ndarray:
    return np.full(n, float(value), dtype=np.float64)


def centered_biases(n: int, theta: float = 1.0) -> np.ndarray:
    return theta * ((np.arange(n, dtype=np.float64) + 0.5) / n - 0.5)


def residual_balanced_biases(n: int, alpha: float = 0.19, theta: float = 1.0) -> np.ndarray:
    q = (np.arange(n, dtype=np.float64) + 0.5) / n
    z = 2.0 * q - 1.0
    return theta / 2.0 * np.sign(z) * np.abs(z) ** alpha
