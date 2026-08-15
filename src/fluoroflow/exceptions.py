"""FluoroFlow exception hierarchy."""

from __future__ import annotations

__all__ = [
    "FluoroFlowError",
    "InsufficientSamplesError",
    "ValidationError",
]


class FluoroFlowError(Exception):
    """Base class for every error raised by FluoroFlow."""


class ValidationError(FluoroFlowError, ValueError):
    """A value handed to FluoroFlow violates a documented precondition."""


class InsufficientSamplesError(ValidationError):
    """An operation needs more samples than the data contains."""
