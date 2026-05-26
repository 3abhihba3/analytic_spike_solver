"""Trace decoding, metrics, residuals, and plotting utilities."""

from . import metrics as _metrics
from .metrics import *
from .monitors import PopulationRateMonitor, SpikeMonitor
from .plotting import plot_raster, plot_traces
from .residual import residual_growth, residual_growth_timeseries
from .trace import decode_trace, lowpass_values

__all__ = [
    *_metrics.__all__,
    "PopulationRateMonitor",
    "SpikeMonitor",
    "decode_trace",
    "lowpass_values",
    "plot_raster",
    "plot_traces",
    "residual_growth",
    "residual_growth_timeseries",
]
