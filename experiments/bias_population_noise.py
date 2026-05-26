from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from analytic_spike_solver.generation.bias import (
    BiasSpikeConfig,
    bias_weight_for_target,
    generate_bias_times,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "results"
DEFAULT_PATH = OUT_DIR / "bias_population_noise.csv"


def sample_trace(
    event_times: np.ndarray,
    *,
    tau: float,
    weight: float,
    sample_times: np.ndarray,
) -> np.ndarray:
    v = 0.0
    out = np.zeros_like(sample_times, dtype=np.float64)
    event_idx = 0
    current_time = float(sample_times[0])
    for sample_idx, sample_time in enumerate(sample_times):
        sample_time = float(sample_time)
        while event_idx < len(event_times) and event_times[event_idx] <= sample_time:
            event_time = float(event_times[event_idx])
            if event_time > current_time:
                v *= np.exp(-(event_time - current_time) / tau)
                current_time = event_time
            v += weight
            event_idx += 1
        if sample_time > current_time:
            v *= np.exp(-(sample_time - current_time) / tau)
            current_time = sample_time
        out[sample_idx] = v
    return out


def run_bias_population_experiment(path: Path = DEFAULT_PATH) -> Path:
    tau = 0.020
    target_bias = 0.25
    duration = 3.0
    warmup = 0.5
    sample_dt = 0.0005
    rates = [25.0, 50.0, 100.0, 200.0]
    sample_times = np.arange(0.0, duration, sample_dt)
    warm_mask = sample_times >= warmup
    rows = []

    for rate in rates:
        weight = float(bias_weight_for_target(target_bias, tau, rate))
        config = BiasSpikeConfig(rate_hz=rate)
        times = generate_bias_times(
            config,
            t_start=0.0,
            t_stop=duration,
        )
        trace = sample_trace(
            times,
            tau=tau,
            weight=weight,
            sample_times=sample_times,
        )
        error = trace[warm_mask] - target_bias
        rows.append(
            [
                "uniform",
                rate,
                weight,
                len(times) / duration,
                float(np.mean(trace[warm_mask])),
                float(np.std(trace[warm_mask])),
                float(np.sqrt(np.mean(error**2))),
                float(np.max(np.abs(error))),
                float(np.std(trace[warm_mask]) / target_bias),
            ]
        )
        rng = np.random.default_rng(1000 + int(rate))
        poisson_metrics = []
        poisson_counts = []
        for _ in range(20):
            t = 0.0
            poisson_times = []
            while True:
                t += float(rng.exponential(1.0 / rate))
                if t >= duration:
                    break
                poisson_times.append(t)
            poisson_times_arr = np.asarray(poisson_times, dtype=np.float64)
            poisson_trace = sample_trace(
                poisson_times_arr,
                tau=tau,
                weight=weight,
                sample_times=sample_times,
            )
            poisson_error = poisson_trace[warm_mask] - target_bias
            poisson_metrics.append(
                [
                    float(np.mean(poisson_trace[warm_mask])),
                    float(np.std(poisson_trace[warm_mask])),
                    float(np.sqrt(np.mean(poisson_error**2))),
                    float(np.max(np.abs(poisson_error))),
                    float(np.std(poisson_trace[warm_mask]) / target_bias),
                ]
            )
            poisson_counts.append(len(poisson_times_arr))
        arr = np.asarray(poisson_metrics)
        rows.append(
            [
                "poisson_reference",
                rate,
                weight,
                float(np.mean(poisson_counts) / duration),
                float(np.mean(arr[:, 0])),
                float(np.mean(arr[:, 1])),
                float(np.mean(arr[:, 2])),
                float(np.mean(arr[:, 3])),
                float(np.mean(arr[:, 4])),
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "mode",
                "target_rate_hz",
                "bias_weight",
                "realized_total_rate_hz",
                "mean_trace",
                "std_trace",
                "rms_error",
                "max_abs_error",
                "cv_vs_target",
            ]
        )
        writer.writerows(rows)
    return path


if __name__ == "__main__":
    output = run_bias_population_experiment()
    print(output)
