from __future__ import annotations

import numpy as np

from ..core.events import SpikeEvents


def spike_counts(spikes: SpikeEvents, n_neurons: int) -> np.ndarray:
    return np.bincount(spikes.ids, minlength=n_neurons).astype(np.int64)


def firing_rates(spikes: SpikeEvents, n_neurons: int, duration: float) -> np.ndarray:
    if duration <= 0:
        raise ValueError("duration must be positive")
    return spike_counts(spikes, n_neurons) / float(duration)


def cv(mean: float | np.ndarray, var: float | np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(var, 0.0)) / np.maximum(mean, 1e-12)


def rms_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def mae_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size == 0 or b.size == 0 or np.var(a) < 1e-12 or np.var(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def trace_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean_trace": float(np.mean(values)),
        "var_trace": float(np.var(values)),
        "cv_trace": float(cv(float(np.mean(values)), float(np.var(values)))),
    }


def offdiag_mean_corr(values: np.ndarray) -> float:
    values = np.asarray(values)
    if values.shape[0] < 2:
        return float("nan")
    active = values.var(axis=1) > 1e-12
    values = values[active]
    if values.shape[0] < 2:
        return float("nan")
    corr = np.corrcoef(values)
    return float(np.nanmean(corr[~np.eye(corr.shape[0], dtype=bool)]))


def covariance_similarity(a_values: np.ndarray, b_values: np.ndarray) -> float:
    a_values = np.asarray(a_values)
    b_values = np.asarray(b_values)
    if a_values.shape[0] < 2 or b_values.shape[0] < 2:
        return float("nan")
    ca = np.cov(a_values)
    cb = np.cov(b_values)
    n = min(ca.shape[0], cb.shape[0])
    mask = ~np.eye(n, dtype=bool)
    va = ca[:n, :n][mask]
    vb = cb[:n, :n][mask]
    if va.var() < 1e-12 or vb.var() < 1e-12:
        return float("nan")
    return float(np.corrcoef(va, vb)[0, 1])


def effective_rank(values: np.ndarray) -> float:
    values = np.asarray(values)
    if values.shape[0] < 2:
        return float("nan")
    eigvals = np.maximum(np.linalg.eigvalsh(np.cov(values)), 0.0)
    total = eigvals.sum()
    if total <= 1e-12:
        return 0.0
    probs = eigvals[eigvals > 1e-12] / total
    return float(np.exp(-np.sum(probs * np.log(probs))))


def locking_metrics(pre: SpikeEvents, post: SpikeEvents, *, window: float, duration: float) -> dict[str, float]:
    pre_t = np.sort(pre.times)
    post_t = np.sort(post.times)
    if len(pre_t) == 0 or len(post_t) == 0:
        return {
            "mean_post_after_pre": float("nan"),
            "expected_post_after_pre": float("nan"),
            "post_after_pre_ratio": float("nan"),
            "frac_post_preceded": float("nan"),
        }
    counts = np.searchsorted(post_t, pre_t + window, side="right") - np.searchsorted(post_t, pre_t, side="left")
    expected = len(post_t) / max(duration, 1e-12) * window
    pre_before = np.searchsorted(pre_t, post_t, side="left") - np.searchsorted(pre_t, post_t - window, side="left")
    return {
        "mean_post_after_pre": float(np.mean(counts)),
        "expected_post_after_pre": float(expected),
        "post_after_pre_ratio": float(np.mean(counts) / max(expected, 1e-12)),
        "frac_post_preceded": float(np.mean(pre_before > 0)),
    }


def target_comparison(name: str, decoded: np.ndarray, target: np.ndarray) -> dict[str, float | str]:
    return {
        "metric": name,
        "rms_error": rms_error(decoded, target),
        "mae_error": mae_error(decoded, target),
        "target_corr": corrcoef(decoded, target),
    }
