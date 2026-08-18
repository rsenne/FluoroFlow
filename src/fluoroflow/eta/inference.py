"""Where an event-triggered average departs from its null.

The confidence band an ETA already carries answers this on its own: the signal
differs from the null wherever the interval excludes it. This module reads that
off pointwise, groups the surviving samples into contiguous epochs, and lets a
minimum-duration criterion throw out runs too short to be worth reporting.

The null is either a fixed level -- zero, the natural choice once dF/F has been
anchored by the isosbestic baseline fit -- or estimated from the ETA's own
pre-event baseline as a mean or median. An estimated null is itself a random
quantity, and treating it as fixed understates uncertainty; prefer zero when the
preprocessing already puts it there.

This is a descriptive reading of an interval, not a multiple-comparison
procedure. Neighbouring timepoints are heavily correlated and there are as many
comparisons as samples, so a handful of isolated significant points is what
noise looks like. ``min_duration`` is the usual guard: require a run to persist.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from numpy.typing import NDArray

from fluoroflow.core.events import Events
from fluoroflow.core.validation import as_series, check_matching_length, check_time_vector
from fluoroflow.exceptions import ValidationError

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["NullSpec", "Significance", "compare_to_null", "resolve_null"]

#: A fixed null level, or a rule for estimating one from the pre-event baseline.
NullSpec = float | Literal["zero", "mean", "median"]

_ESTIMATORS: dict[str, Callable[[NDArray[np.float64]], Any]] = {
    "mean": np.mean,
    "median": np.median,
}

_Scalar = TypeVar("_Scalar", bound=np.generic)


def _frozen(values: NDArray[_Scalar]) -> NDArray[_Scalar]:
    """Return the array made read-only, matching the immutability of the rest of the package."""
    values.flags.writeable = False
    return values


@dataclass(frozen=True, slots=True)
class Significance:
    """Where a confidence band excludes the null, and for how long at a stretch."""

    time: NDArray[np.float64]
    mask: NDArray[np.bool_]
    direction: NDArray[np.int8]
    epochs: Events
    null: float
    null_method: str
    min_duration: float | None
    confidence: float | None

    @property
    def n_significant(self) -> int:
        """Number of timepoints whose interval excludes the null."""
        return int(np.count_nonzero(self.mask))

    @property
    def n_epochs(self) -> int:
        """Number of contiguous runs that survived the duration criterion."""
        return len(self.epochs)

    @property
    def total_duration(self) -> float:
        """Summed duration of those runs, in seconds."""
        return self.epochs.total_duration if len(self.epochs) else 0.0

    @property
    def first_crossing(self) -> float | None:
        """Onset of the earliest surviving run, or :data:`None` if there is none."""
        return float(self.epochs.times[0]) if len(self.epochs) else None

    def to_frame(self) -> pd.DataFrame:
        """Return the pointwise verdict as a :class:`pandas.DataFrame`."""
        import pandas as pd

        return pd.DataFrame(
            {
                "time": np.asarray(self.time).copy(),
                "significant": np.asarray(self.mask).copy(),
                "direction": np.asarray(self.direction).copy(),
            }
        )

    def __repr__(self) -> str:
        """One-line summary: null, pointwise count, epoch count, and total time."""
        return (
            f"Significance(null={self.null:.4g} ({self.null_method}), "
            f"n_significant={self.n_significant}/{self.time.size}, "
            f"epochs={self.n_epochs}, total={self.total_duration:.4g} s)"
        )


def resolve_null(
    time: NDArray[np.float64],
    values: NDArray[np.float64],
    null: NullSpec = 0.0,
    *,
    baseline: tuple[float | None, float | None] | None = None,
) -> tuple[float, str]:
    """Turn a null spec into a level, estimating from the baseline window if asked.

    ``null`` is a number, ``"zero"``, or ``"mean"``/``"median"`` of ``values``
    over ``baseline``. The baseline window defaults to everything before the
    event, ``(None, 0.0)``; ``None`` on either side of it means unbounded.
    """
    if not isinstance(null, str):
        try:
            level = float(null)
        except (TypeError, ValueError) as exc:
            msg = f"null must be a number or a string, got {null!r}."
            raise ValidationError(msg) from exc
        if not math.isfinite(level):
            msg = f"A fixed null must be finite, got {level!r}."
            raise ValidationError(msg)
        return level, "fixed"

    if null == "zero":
        return 0.0, "zero"
    estimator = _ESTIMATORS.get(null)
    if estimator is None:
        options = ["zero", *sorted(_ESTIMATORS)]
        msg = f"null must be a number or one of {options}, got {null!r}."
        raise ValidationError(msg)

    lo, hi = (None, 0.0) if baseline is None else baseline
    low = -math.inf if lo is None else float(lo)
    high = math.inf if hi is None else float(hi)
    if not low < high:
        msg = f"baseline start must be less than baseline stop, got {baseline!r}."
        raise ValidationError(msg)

    selected = (time >= low) & (time < high)
    if not selected.any():
        span = f"[{time[0]!r}, {time[-1]!r}]" if time.size else "<empty>"
        msg = (
            f"The baseline window [{low!r}, {high!r}) s contains no samples of an ETA "
            f"spanning {span} s, so a {null!r} null cannot be estimated. Pass an "
            f"explicit baseline that overlaps the window, or a fixed null such as 0.0."
        )
        raise ValidationError(msg)
    return float(estimator(values[selected])), null


def compare_to_null(
    time: NDArray[np.float64],
    mean: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    *,
    null: NullSpec = 0.0,
    baseline: tuple[float | None, float | None] | None = None,
    min_duration: float | None = None,
    confidence: float | None = None,
    name: str = "significant",
) -> Significance:
    """Mark the timepoints whose confidence interval excludes the null.

    Runs shorter than ``min_duration`` seconds are dropped from both the mask
    and the epochs, so the two always agree on what counts.
    """
    time = as_series(time, label="time")
    check_time_vector(time)
    mean = as_series(mean, label="mean")
    lower = as_series(lower, label="lower")
    upper = as_series(upper, label="upper")
    check_matching_length(time, mean, labels=("time", "mean"))
    check_matching_length(time, lower, labels=("time", "lower"))
    check_matching_length(time, upper, labels=("time", "upper"))

    level, null_method = resolve_null(time, mean, null, baseline=baseline)

    above = lower > level
    below = upper < level
    mask = above | below
    epochs = Events.from_boolean(mask, time, name=name)

    if min_duration is not None:
        if not math.isfinite(min_duration) or min_duration < 0.0:
            msg = f"min_duration must be a finite, non-negative number, got {min_duration!r}."
            raise ValidationError(msg)
        durations = epochs.durations
        if durations is None:  # pragma: no cover; from_boolean always attaches durations
            msg = "Epoch durations are required to apply min_duration."
            raise ValidationError(msg)
        keep = durations >= min_duration
        epochs = Events(epochs.times[keep], name=name, durations=durations[keep])
        mask = epochs.to_boolean(time)
        above = above & mask
        below = below & mask

    direction = np.zeros(time.size, dtype=np.int8)
    direction[above] = 1
    direction[below] = -1

    return Significance(
        time=time,
        mask=_frozen(mask),
        direction=_frozen(direction),
        epochs=epochs,
        null=level,
        null_method=null_method,
        min_duration=min_duration,
        confidence=confidence,
    )
