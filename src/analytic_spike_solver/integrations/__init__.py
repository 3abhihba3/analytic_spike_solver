"""Optional integrations and experimental backends."""

from .brian_compare import brian2_available, compare_layer_with_brian2, compare_with_brian2
from .numba_accel import numba_available
from .sparse import SparseLayerSpec

__all__ = [
    "SparseLayerSpec",
    "brian2_available",
    "compare_layer_with_brian2",
    "compare_with_brian2",
    "numba_available",
]
