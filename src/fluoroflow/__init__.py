"""Public FluoroFlow API."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from fluoroflow.core import Events, Recording, SamplingReport, Step, Trace
from fluoroflow.eta import (
    AnimalETA,
    BayesianETA,
    PopulationETA,
    align_to_events,
    animal_eta,
    bayesian_eta,
    population_eta,
)
from fluoroflow.exceptions import FluoroFlowError, InsufficientSamplesError, ValidationError
from fluoroflow.preprocessing import (
    BaselineOptions,
    DffOptions,
    LowpassOptions,
    PreprocessOptions,
    preprocess,
)

try:
    __version__ = version("fluoroflow")
except PackageNotFoundError:  # pragma: no cover; only when running from a bare tree
    __version__ = "0.0.0.dev0"

__all__ = [
    "AnimalETA",
    "BaselineOptions",
    "BayesianETA",
    "DffOptions",
    "Events",
    "FluoroFlowError",
    "InsufficientSamplesError",
    "LowpassOptions",
    "PopulationETA",
    "PreprocessOptions",
    "Recording",
    "SamplingReport",
    "Step",
    "Trace",
    "ValidationError",
    "__version__",
    "align_to_events",
    "animal_eta",
    "bayesian_eta",
    "population_eta",
    "preprocess",
]
