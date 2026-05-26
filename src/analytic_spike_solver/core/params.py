from __future__ import annotations

import numpy as np


def theta_proportional_to_tau(
    tau: np.ndarray | float,
    theta_per_tau: float,
) -> np.ndarray:
    """Construct thresholds using the requested theta ∝ tau convention."""

    if theta_per_tau <= 0:
        raise ValueError("theta_per_tau must be positive")
    tau_arr = np.asarray(tau, dtype=np.float64)
    if np.any(tau_arr <= 0):
        raise ValueError("tau must be positive")
    return tau_arr * float(theta_per_tau)


def constant_theta(tau: np.ndarray | float, theta: float) -> np.ndarray:
    tau_arr = np.asarray(tau, dtype=np.float64)
    if np.any(tau_arr <= 0):
        raise ValueError("tau must be positive")
    return np.full_like(tau_arr, float(theta), dtype=np.float64)


def theta_for_target_gain(tau: np.ndarray | float, target_gain: float) -> np.ndarray:
    if target_gain <= 0:
        raise ValueError("target_gain must be positive")
    tau_arr = np.asarray(tau, dtype=np.float64)
    if np.any(tau_arr <= 0):
        raise ValueError("tau must be positive")
    return 1.0 / (target_gain * tau_arr)


def make_theta(
    tau: np.ndarray | float,
    *,
    base_theta: float | np.ndarray = 1.0,
    policy: str = "constant",
    theta_per_tau: float | None = None,
    target_gain: float | None = None,
) -> np.ndarray:
    if policy == "constant":
        return np.asarray(base_theta, dtype=np.float64)
    if policy == "proportional_tau":
        if theta_per_tau is None:
            theta_per_tau = float(np.asarray(base_theta).mean()) / float(np.asarray(tau).mean())
        return theta_proportional_to_tau(tau, theta_per_tau)
    if policy == "target_gain":
        if target_gain is None:
            raise ValueError("target_gain is required for target_gain theta policy")
        return theta_for_target_gain(tau, target_gain)
    raise ValueError(f"unknown theta policy: {policy}")
