"""Trace decoding, metrics, residuals, and plotting utilities."""

from .metrics import *
from .monitors import PopulationRateMonitor, SpikeMonitor
from .plotting import plot_raster, plot_traces
from .residual import residual_growth, residual_growth_timeseries
from .trace import decode_trace, lowpass_values
