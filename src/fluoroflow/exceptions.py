"""Exception hierarchy for FluoroFlow.

Every exception FluoroFlow raises inherits from :class:`FluoroFlowError`, so
callers can catch everything from this library with one clause. Where a builtin
is a natural fit, the concrete classes also inherit from it, so
``except ValueError`` keeps working for people who never read this module.
"""

from __future__ import annotations

__all__ = [
    "FluoroFlowError",
    "InsufficientSamplesError",
    "ProvenanceError",
    "ValidationError",
]


class FluoroFlowError(Exception):
    """Base class for every error raised by FluoroFlow."""


class ValidationError(FluoroFlowError, ValueError):
    """A value handed to FluoroFlow violates a documented precondition."""


class InsufficientSamplesError(ValidationError):
    """An operation needs more samples than the data contains.

    Raised, for example, when asking a one-sample trace for its sampling rate.
    """


class ProvenanceError(FluoroFlowError, RuntimeError):
    r"""An operation is unsafe given what has already been done to the data.

    The canonical case: computing :math:`\Delta F/F` on a signal whose baseline
    has already been subtracted, which silently produces meaningless numbers.
    """
