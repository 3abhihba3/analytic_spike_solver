from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SpikeEvents:
    """Spike event array.

    Times are seconds. Ids are integer neuron ids in the source layer. Amplitudes
    scale the impulse jump applied through the selected weight row and default
    to one.
    """

    times: np.ndarray
    ids: np.ndarray
    amplitudes: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=np.float64)
        self.ids = np.asarray(self.ids, dtype=np.int64)
        if self.amplitudes is None:
            self.amplitudes = np.ones_like(self.times, dtype=np.float64)
        else:
            self.amplitudes = np.asarray(self.amplitudes, dtype=np.float64)
        self._validate()

    def _validate(self) -> None:
        if self.times.ndim != 1 or self.ids.ndim != 1 or self.amplitudes.ndim != 1:
            raise ValueError("times, ids, and amplitudes must be one-dimensional")
        if not (len(self.times) == len(self.ids) == len(self.amplitudes)):
            raise ValueError("times, ids, and amplitudes must have matching lengths")
        if np.any(~np.isfinite(self.times)):
            raise ValueError("spike times must be finite")
        if np.any(~np.isfinite(self.amplitudes)):
            raise ValueError("spike amplitudes must be finite")
        if np.any(self.ids < 0):
            raise ValueError("spike ids must be non-negative")

    def __len__(self) -> int:
        return int(self.times.size)

    @property
    def empty(self) -> bool:
        return len(self) == 0

    @classmethod
    def empty_events(cls) -> SpikeEvents:
        return cls(np.asarray([], dtype=np.float64), np.asarray([], dtype=np.int64))

    @classmethod
    def from_pairs(cls, pairs, amplitudes: np.ndarray | None = None) -> SpikeEvents:
        arr = np.asarray(list(pairs), dtype=np.float64)
        if arr.size == 0:
            return cls.empty_events()
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("pairs must contain (time, id) rows")
        return cls(arr[:, 0], arr[:, 1].astype(np.int64), amplitudes)

    @classmethod
    def from_csv(cls, path: str | Path) -> SpikeEvents:
        data = np.genfromtxt(path, delimiter=",", names=True)
        if data.size == 0:
            return cls.empty_events()
        times = np.atleast_1d(data["time"])
        ids = np.atleast_1d(data["id"]).astype(np.int64)
        amps = np.atleast_1d(data["amplitude"]) if "amplitude" in data.dtype.names else None
        return cls(times, ids, amps)

    @classmethod
    def from_npz(cls, path: str | Path) -> SpikeEvents:
        with np.load(path) as data:
            return cls(data["times"], data["ids"], data["amplitudes"])

    @classmethod
    def regular_train(cls, rate_hz: float, duration: float, n_neurons: int = 1) -> SpikeEvents:
        from ..generation.encoders import regular_events

        return regular_events(rate_hz, duration=duration, n_neurons=n_neurons)

    @classmethod
    def poisson_train(cls, rates_hz, duration: float, seed=None) -> SpikeEvents:
        from ..generation.encoders import poisson_events

        return poisson_events(rates_hz, duration=duration, seed=seed)

    def copy(self) -> SpikeEvents:
        return SpikeEvents(self.times.copy(), self.ids.copy(), self.amplitudes.copy())

    def is_sorted(self) -> bool:
        if len(self) <= 1:
            return True
        later_time = self.times[1:] > self.times[:-1]
        same_time = self.times[1:] == self.times[:-1]
        later_id = self.ids[1:] >= self.ids[:-1]
        return bool(np.all(later_time | (same_time & later_id)))

    def sorted(self) -> SpikeEvents:
        if self.is_sorted():
            return self.copy()
        order = np.lexsort((self.ids, self.times))
        return SpikeEvents(self.times[order], self.ids[order], self.amplitudes[order])

    def clipped(self, t_start: float | None, t_stop: float | None) -> SpikeEvents:
        mask = np.ones(len(self), dtype=bool)
        if t_start is not None:
            mask &= self.times >= float(t_start)
        if t_stop is not None:
            mask &= self.times < float(t_stop)
        return SpikeEvents(self.times[mask], self.ids[mask], self.amplitudes[mask])

    def with_id_offset(self, offset: int) -> SpikeEvents:
        if offset < 0:
            raise ValueError("offset must be non-negative")
        return SpikeEvents(self.times, self.ids + int(offset), self.amplitudes)

    def validate_ids(self, n: int, label: str = "SpikeEvents") -> None:
        if n <= 0:
            raise ValueError("n must be positive")
        if len(self) and int(np.max(self.ids)) >= n:
            raise ValueError(
                f"{label}: max id {int(np.max(self.ids))} outside expected range [0, {n})"
            )

    def to_csv(self, path: str | Path) -> None:
        arr = np.column_stack([self.times, self.ids, self.amplitudes])
        np.savetxt(path, arr, delimiter=",", header="time,id,amplitude", comments="")

    def to_npz(self, path: str | Path) -> None:
        np.savez(path, times=self.times, ids=self.ids, amplitudes=self.amplitudes)

    @staticmethod
    def concat(events: list[SpikeEvents], sort: bool = True) -> SpikeEvents:
        events = [event for event in events if len(event) > 0]
        if not events:
            return SpikeEvents.empty_events()
        out = SpikeEvents(
            np.concatenate([event.times for event in events]),
            np.concatenate([event.ids for event in events]),
            np.concatenate([event.amplitudes for event in events]),
        )
        return out.sorted() if sort else out
