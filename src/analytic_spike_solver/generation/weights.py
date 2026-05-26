from __future__ import annotations

from typing import Literal

import numpy as np

from ..core.random import rng_from_seed

WeightInit = Literal[
    "positive_mean",
    "he_signed",
    "he_signed_centered",
    "he_signed_centered_safe",
    "he_abs_rescaled",
]


def rescale_incoming(weights: np.ndarray, target_sum: float) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64).copy()
    totals = weights.sum(axis=0)
    for j, total in enumerate(totals):
        if abs(total) < 1e-12:
            weights[:, j] = target_sum / max(weights.shape[0], 1)
        else:
            weights[:, j] *= target_sum / total
    return weights


def center_and_scale_incoming(weights: np.ndarray, target_std: float) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64).copy()
    weights -= weights.mean(axis=0, keepdims=True)
    std = weights.std(axis=0)
    mask = std > 1e-12
    weights[:, mask] *= target_std / std[mask]
    return weights


def cap_incoming_jump(weights: np.ndarray, max_abs_weight: float) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64).copy()
    if max_abs_weight <= 0:
        raise ValueError("max_abs_weight must be positive")
    col_max = np.max(np.abs(weights), axis=0)
    mask = col_max > max_abs_weight
    weights[:, mask] *= max_abs_weight / col_max[mask]
    return weights


def random_weights(
    n_pre: int,
    n_post: int,
    *,
    mode: WeightInit = "he_signed_centered_safe",
    gain: float = 1.0,
    jitter: float = 0.6,
    max_abs_weight: float | None = None,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    if n_pre <= 0 or n_post <= 0:
        raise ValueError("n_pre and n_post must be positive")
    rng = rng_from_seed(seed)
    if mode == "positive_mean":
        raw = np.maximum(1.0 + jitter * rng.standard_normal((n_pre, n_post)), 0.0)
        weights = rescale_incoming(raw, gain)
    elif mode == "he_signed":
        weights = gain * rng.standard_normal((n_pre, n_post)) * np.sqrt(2.0 / n_pre)
    elif mode == "he_signed_centered":
        raw = rng.standard_normal((n_pre, n_post))
        weights = center_and_scale_incoming(raw, gain * np.sqrt(2.0 / n_pre))
    elif mode == "he_signed_centered_safe":
        raw = rng.standard_normal((n_pre, n_post))
        weights = center_and_scale_incoming(raw, gain * np.sqrt(2.0 / n_pre))
        cap = gain if max_abs_weight is None else max_abs_weight
        weights = cap_incoming_jump(weights, cap)
    elif mode == "he_abs_rescaled":
        raw = np.abs(rng.standard_normal((n_pre, n_post)) * np.sqrt(2.0 / n_pre))
        weights = rescale_incoming(raw, gain)
    else:
        raise ValueError(f"unknown weight init mode: {mode}")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weight initializer produced non-finite values")
    return weights
