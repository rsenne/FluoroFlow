"""Reusable precondition guards.

These exist so that every entry point rejects bad input the same way, with the
same message shape, at construction time rather than three transforms later.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fluoroflow.exceptions import ValidationError

__all__ = [
    "as_series",
    "check_matching_length",
    "check_no_infinities",
    "check_percentage",
    "check_positive",
    "check_time_vector",
    "checked_name",
    "median_dt",
]


def as_series(x: Any, *, label: str) -> NDArray[np.float64]:
    """Coerce ``x`` to a read-only one-dimensional float64 array.

    Read-only is the point: a :class:`~fluoroflow.core.trace.Trace` promises that
    its data never changes, and that promise is worthless if a caller can reach
    in and assign to the underlying buffer.

    Parameters
    ----------
    x
        Anything array-like: list, tuple, :class:`numpy.ndarray`,
        :class:`pandas.Series`.
    label
        Name used in error messages, e.g. ``"time"``.

    Returns
    -------
    numpy.ndarray
        A one-dimensional, ``float64``, non-writeable array. Copies only when it
        has to; an array that is already read-only float64 is passed through.

    Raises
    ------
    ValidationError
        If ``x`` cannot be interpreted as a numeric array, or is not
        one-dimensional.
    """
    try:
        arr = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        msg = f"{label} must be array-like and numeric; got {type(x).__name__} ({exc})."
        raise ValidationError(msg) from exc
    if arr.ndim != 1:
        msg = f"{label} must be one-dimensional, got shape {arr.shape}."
        raise ValidationError(msg)
    if arr.flags.writeable:
        arr = arr.copy()
        arr.flags.writeable = False
    return arr


def check_time_vector(time: NDArray[np.float64], *, label: str = "time") -> None:
    """Require a time vector to be finite and strictly increasing.

    Strictly increasing (not merely sorted) is deliberate: duplicate timestamps
    make interpolation, event alignment, and sampling-rate estimation ambiguous,
    and duplicates in real files usually mean a de-interleaving mistake.

    Parameters
    ----------
    time
        Candidate time vector, in seconds.
    label
        Name used in error messages.

    Raises
    ------
    ValidationError
        If any entry is NaN or infinite, or if the vector is not strictly
        increasing.
    """
    if time.size and not bool(np.all(np.isfinite(time))):
        n_bad = int(np.count_nonzero(~np.isfinite(time)))
        msg = f"{label} must be finite; found {n_bad} NaN or infinite value(s)."
        raise ValidationError(msg)
    if time.size < 2:
        return
    steps = np.diff(time)
    bad = steps <= 0.0
    if bool(np.any(bad)):
        first = int(np.argmax(bad))
        msg = (
            f"{label} must be strictly increasing; "
            f"{label}[{first + 1}]={time[first + 1]!r} does not exceed "
            f"{label}[{first}]={time[first]!r} "
            f"({int(np.count_nonzero(bad))} of {steps.size} steps are non-positive)."
        )
        raise ValidationError(msg)


def check_no_infinities(values: NDArray[np.float64], *, label: str = "values") -> None:
    """Reject infinities while allowing NaN.

    NaN is meaningful in photometry: dropped frames, samples outside a trial
    window, and masked artifacts are all legitimately missing. An infinity is
    never legitimate; it means a division by zero happened upstream.

    Parameters
    ----------
    values
        Array to check.
    label
        Name used in error messages.

    Raises
    ------
    ValidationError
        If any entry is positive or negative infinity.
    """
    if values.size and bool(np.any(np.isinf(values))):
        n_bad = int(np.count_nonzero(np.isinf(values)))
        msg = (
            f"{label} contains {n_bad} infinite value(s). NaN is allowed for missing "
            f"samples, but an infinity means a division by zero happened upstream."
        )
        raise ValidationError(msg)


def check_matching_length(a: NDArray[Any], b: NDArray[Any], *, labels: tuple[str, str]) -> None:
    """Require two arrays to have the same length.

    Parameters
    ----------
    a, b
        Arrays to compare.
    labels
        Names of ``a`` and ``b``, used in error messages.

    Raises
    ------
    ValidationError
        If the lengths differ.
    """
    if a.shape[0] != b.shape[0]:
        msg = (
            f"{labels[0]} and {labels[1]} must have the same length, "
            f"got {a.shape[0]} and {b.shape[0]}."
        )
        raise ValidationError(msg)


def checked_name(name: Any, *, label: str = "name") -> str:
    """Require a non-empty string, returned stripped of surrounding whitespace.

    Parameters
    ----------
    name
        Candidate name.
    label
        Name of the field being checked, used in error messages.

    Returns
    -------
    str
        The stripped name.

    Raises
    ------
    ValidationError
        If ``name`` is not a string, or is empty or whitespace only.
    """
    if not isinstance(name, str) or not name.strip():
        msg = f"{label} must be a non-empty string, got {name!r}."
        raise ValidationError(msg)
    return name.strip()


def check_percentage(value: float, *, label: str = "percentile") -> float:
    """Require a percentile expressed in percent, on the closed interval [0, 100].

    FluoroFlow states percentiles in percent everywhere, without exception. The
    old ``photonsoup`` code called ``np.percentile(raw, 0.08)`` intending the 8th
    percentile and silently got the 0.08th, which is a different number entirely.
    A fraction slipped in where a percent belongs is the single easiest way to get
    a plausible-looking wrong baseline, so a value below 1 is suspicious but legal
    and a value above 100 is rejected outright.

    Parameters
    ----------
    value
        Candidate percentile, in percent.
    label
        Name used in error messages.

    Returns
    -------
    float
        The value as a float.

    Raises
    ------
    ValidationError
        If ``value`` is not finite or falls outside [0, 100].
    """
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        msg = f"{label} must be a real number in percent, got {value!r}."
        raise ValidationError(msg) from exc
    if not math.isfinite(out):
        msg = f"{label} must be finite, got {out!r}."
        raise ValidationError(msg)
    if not 0.0 <= out <= 100.0:
        msg = (
            f"{label} is expressed in percent and must lie in [0, 100], got {out!r}. "
            f"For the 8th percentile pass 8, not 0.08."
        )
        raise ValidationError(msg)
    return out


def check_positive(value: float, *, label: str) -> float:
    """Require a strictly positive finite number.

    Parameters
    ----------
    value
        Candidate value.
    label
        Name used in error messages.

    Returns
    -------
    float
        The value as a float.

    Raises
    ------
    ValidationError
        If ``value`` is not finite or is not greater than zero.
    """
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        msg = f"{label} must be a real number, got {value!r}."
        raise ValidationError(msg) from exc
    if not math.isfinite(out) or out <= 0.0:
        msg = f"{label} must be a finite number greater than zero, got {out!r}."
        raise ValidationError(msg)
    return out


def median_dt(time: NDArray[np.float64]) -> float:
    """Median sample interval of a time vector, or NaN if it is too short.

    The median, not the mean and emphatically not ``diff(time)[1]``, so that a
    single dropped frame or a pause in acquisition does not move the estimate.

    Parameters
    ----------
    time
        Time vector in seconds, assumed already validated.

    Returns
    -------
    float
        Median of the successive differences, or ``nan`` when fewer than two
        samples are present.
    """
    if time.size < 2:
        return math.nan
    return float(np.median(np.diff(time)))
