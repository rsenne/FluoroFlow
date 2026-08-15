"""Reusable input validation."""

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
    "check_positive",
    "check_time_vector",
    "checked_name",
    "median_dt",
]


def as_series(x: Any, *, label: str) -> NDArray[np.float64]:
    """Coerce ``x`` to a read-only one-dimensional float64 array."""
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
    """Require a time vector to be finite and strictly increasing."""
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
    """Reject infinities while allowing NaN."""
    if values.size and bool(np.any(np.isinf(values))):
        n_bad = int(np.count_nonzero(np.isinf(values)))
        msg = (
            f"{label} contains {n_bad} infinite value(s). NaN is allowed for missing "
            f"samples, but an infinity means a division by zero happened upstream."
        )
        raise ValidationError(msg)


def check_matching_length(a: NDArray[Any], b: NDArray[Any], *, labels: tuple[str, str]) -> None:
    """Require two arrays to have the same length."""
    if a.shape[0] != b.shape[0]:
        msg = (
            f"{labels[0]} and {labels[1]} must have the same length, "
            f"got {a.shape[0]} and {b.shape[0]}."
        )
        raise ValidationError(msg)


def checked_name(name: Any, *, label: str = "name") -> str:
    """Require a non-empty string, returned stripped of surrounding whitespace."""
    if not isinstance(name, str) or not name.strip():
        msg = f"{label} must be a non-empty string, got {name!r}."
        raise ValidationError(msg)
    return name.strip()


def check_positive(value: float, *, label: str) -> float:
    """Require a strictly positive finite number."""
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
    """Median sample interval of a time vector, or NaN if it is too short."""
    if time.size < 2:
        return math.nan
    return float(np.median(np.diff(time)))
