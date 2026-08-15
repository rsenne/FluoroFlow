"""Public FluoroFlow API."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from fluoroflow.core import (
    BASELINE_CORRECTED,
    FILTERED,
    MEAN_REMOVED,
    MOTION_CORRECTED,
    NORMALIZED,
    RESAMPLED,
    ChannelSpec,
    Events,
    Recording,
    SamplingReport,
    Step,
    Trace,
)
from fluoroflow.exceptions import (
    FluoroFlowError,
    InsufficientSamplesError,
    ProvenanceError,
    ValidationError,
)

try:
    __version__ = version("fluoroflow")
except PackageNotFoundError:  # pragma: no cover; only when running from a bare tree
    __version__ = "0.0.0.dev0"

__all__ = [
    "BASELINE_CORRECTED",
    "FILTERED",
    "MEAN_REMOVED",
    "MOTION_CORRECTED",
    "NORMALIZED",
    "RESAMPLED",
    "ChannelSpec",
    "Events",
    "FluoroFlowError",
    "InsufficientSamplesError",
    "ProvenanceError",
    "Recording",
    "SamplingReport",
    "Step",
    "Trace",
    "ValidationError",
    "__version__",
]
