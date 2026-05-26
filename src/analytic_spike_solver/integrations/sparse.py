from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SparseLayerSpec:
    pre_ids: np.ndarray
    post_ids: np.ndarray
    weights: np.ndarray
    n_pre: int
    n_post: int

    @classmethod
    def from_dense(cls, weights: np.ndarray, *, atol: float = 0.0) -> SparseLayerSpec:
        weights = np.asarray(weights, dtype=np.float64)
        pre, post = np.nonzero(np.abs(weights) > atol)
        return cls(pre, post, weights[pre, post], weights.shape[0], weights.shape[1])
