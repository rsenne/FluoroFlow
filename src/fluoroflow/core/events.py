"""Event onsets and epoch durations stored in seconds."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from fluoroflow.core.validation import (
    as_series,
    check_matching_length,
    check_time_vector,
    checked_name,
    median_dt,
)
from fluoroflow.exceptions import InsufficientSamplesError, ValidationError

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["Events"]

#: Float64 machine epsilon used for epoch boundary comparisons.
_EPS = float(np.finfo(np.float64).eps)


def _boundary_tolerance(grid: NDArray[np.float64]) -> float:
    """How close to a timestamp an epoch boundary must land to count as touching it."""
    step = median_dt(grid)
    if math.isnan(step):
        return 0.0
    magnitude = max(abs(float(grid[0])), abs(float(grid[-1])), 1.0)
    return min(0.25 * step, max(1e-9 * step, 32.0 * _EPS * magnitude))


@dataclass(frozen=True, slots=True, eq=False)
class Events:
    """An immutable set of event onsets, optionally with durations and labels."""

    times: NDArray[np.float64]
    name: str = "events"
    durations: NDArray[np.float64] | None = None
    labels: tuple[str, ...] | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerce, validate, and freeze the inputs."""
        times = as_series(self.times, label="times")
        if times.size and not bool(np.all(np.isfinite(times))):
            n_bad = int(np.count_nonzero(~np.isfinite(times)))
            msg = f"Event times must be finite; found {n_bad} NaN or infinite value(s)."
            raise ValidationError(msg)
        if times.size > 1:
            steps = np.diff(times)
            if bool(np.any(steps < 0.0)):
                first = int(np.argmax(steps < 0.0))
                msg = (
                    f"Event times must be non-decreasing; times[{first + 1}]="
                    f"{times[first + 1]!r} precedes times[{first}]={times[first]!r}. "
                    f"Sort them before constructing Events, so that the ordering you "
                    f"intended is explicit."
                )
                raise ValidationError(msg)

        durations = self.durations
        if durations is not None:
            durations = as_series(durations, label="durations")
            check_matching_length(times, durations, labels=("times", "durations"))
            if durations.size and not bool(np.all(np.isfinite(durations))):
                msg = "Event durations must be finite."
                raise ValidationError(msg)
            if durations.size and bool(np.any(durations < 0.0)):
                n_bad = int(np.count_nonzero(durations < 0.0))
                msg = f"Event durations must be non-negative; found {n_bad} negative value(s)."
                raise ValidationError(msg)

        labels = self.labels
        if labels is not None:
            labels = tuple(str(v) for v in labels)
            if len(labels) != times.size:
                msg = (
                    f"labels and times must have the same length, "
                    f"got {len(labels)} and {times.size}."
                )
                raise ValidationError(msg)

        object.__setattr__(self, "times", times)
        object.__setattr__(self, "name", checked_name(self.name))
        object.__setattr__(self, "durations", durations)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))

    @classmethod
    def from_boolean(
        cls,
        mask: Any,
        time: Any,
        *,
        name: str = "events",
        meta: Mapping[str, Any] | None = None,
    ) -> Events:
        """Convert a per-sample boolean state vector into onsets and durations.

        Each contiguous run of True becomes one epoch. The onset is the timestamp
        of the run's first True sample, and the epoch ends at the timestamp of the
        first following False sample. A run reaching the end of the recording is
        closed off one median sample interval after the final sample.
        """
        times_arr = as_series(time, label="time")
        check_time_vector(times_arr)
        flags = np.asarray(mask)
        if flags.ndim != 1:
            msg = f"mask must be one-dimensional, got shape {flags.shape}."
            raise ValidationError(msg)
        if flags.shape[0] != times_arr.size:
            msg = (
                f"mask and time must have the same length, "
                f"got {flags.shape[0]} and {times_arr.size}."
            )
            raise ValidationError(msg)
        state = flags.astype(bool)

        if state.size == 0:
            return cls(times=np.empty(0), name=name, durations=np.empty(0), meta=meta or {})

        edges = np.diff(state.astype(np.int8), prepend=np.int8(0), append=np.int8(0))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)

        step = median_dt(times_arr)
        if starts.size and math.isnan(step):
            msg = (
                f"Measuring epoch durations needs at least 2 timestamps to establish a "
                f"sample interval; got {times_arr.size}. Rather than invent a width for "
                f"the single True sample, this refuses. A mask with no True samples "
                f"needs no interval and is accepted at any length."
            )
            raise InsufficientSamplesError(msg)

        n = times_arr.size
        last_edge = float(times_arr[-1]) + step
        onset = times_arr[starts]
        offset = np.where(ends < n, times_arr[np.clip(ends, 0, n - 1)], last_edge)

        return cls(
            times=onset,
            name=name,
            durations=offset - onset,
            meta=meta or {},
        )

    def __len__(self) -> int:
        """Number of events."""
        return int(self.times.size)

    def __iter__(self) -> Iterator[float]:
        """Iterate over onset times."""
        return (float(t) for t in self.times)

    @property
    def has_durations(self) -> bool:
        """Whether durations are attached, i.e. whether these are epochs."""
        return self.durations is not None

    @property
    def offsets(self) -> NDArray[np.float64]:
        """End time of each epoch, in seconds."""
        if self.durations is None:
            msg = (
                f"Events {self.name!r} has no durations, so it has no offsets. "
                f"These are instants, not epochs."
            )
            raise ValidationError(msg)
        out: NDArray[np.float64] = self.times + self.durations
        return out

    @property
    def total_duration(self) -> float:
        """Summed duration of all epochs, in seconds."""
        if self.durations is None:
            msg = f"Events {self.name!r} has no durations to total."
            raise ValidationError(msg)
        return float(self.durations.sum())

    def _subset(self, keep: NDArray[np.bool_], *, name: str | None = None) -> Events:
        """Return the subset selected by a boolean mask over events."""
        labels = None if self.labels is None else tuple(np.asarray(self.labels)[keep].tolist())
        return Events(
            times=self.times[keep],
            name=self.name if name is None else name,
            durations=None if self.durations is None else self.durations[keep],
            labels=labels,
            meta=self.meta,
        )

    def within(self, start: float | None = None, stop: float | None = None) -> Events:
        """Return the events whose onset lies in the half-open window ``[start, stop)``."""
        lo = -math.inf if start is None else float(start)
        hi = math.inf if stop is None else float(stop)
        if not lo < hi:
            msg = f"start must be less than stop, got start={start!r}, stop={stop!r}."
            raise ValidationError(msg)
        return self._subset((self.times >= lo) & (self.times < hi))

    def with_label(self, label: str) -> Events:
        """Return only the events carrying ``label``."""
        if self.labels is None:
            msg = f"Events {self.name!r} has no labels to filter on."
            raise ValidationError(msg)
        keep = np.array([v == label for v in self.labels], dtype=bool)
        return self._subset(keep, name=f"{self.name}[{label}]")

    def shift(self, delta: float) -> Events:
        """Return a copy with every onset moved by ``delta`` seconds."""
        return Events(
            times=self.times + float(delta),
            name=self.name,
            durations=self.durations,
            labels=self.labels,
            meta=self.meta,
        )

    def to_boolean(self, time: Any) -> NDArray[np.bool_]:
        """Render the epochs as a per-sample boolean vector on ``time``."""
        if self.durations is None:
            msg = (
                f"Events {self.name!r} has no durations, so it cannot be rendered as a "
                f"per-sample mask. Attach durations, or align to onsets instead."
            )
            raise ValidationError(msg)
        grid = as_series(time, label="time")
        check_time_vector(grid)
        out = np.zeros(grid.size, dtype=bool)
        tol = _boundary_tolerance(grid)
        for onset, offset in zip(self.times, self.offsets, strict=True):
            lo = int(np.searchsorted(grid, onset - tol, side="left"))
            hi = int(np.searchsorted(grid, offset - tol, side="left"))
            out[lo:hi] = True
        return out

    def to_frame(self) -> pd.DataFrame:
        """Return the events as a :class:`pandas.DataFrame`."""
        import pandas as pd

        data: dict[str, Any] = {"onset": self.times.copy()}
        if self.durations is not None:
            data["duration"] = self.durations.copy()
            data["offset"] = np.asarray(self.offsets)
        if self.labels is not None:
            data["label"] = list(self.labels)
        return pd.DataFrame(data)

    def __eq__(self, other: object) -> bool:
        """Compare by value; returns :data:`NotImplemented` for non-events."""
        if not isinstance(other, Events):
            return NotImplemented
        mine, theirs = self.durations, other.durations
        if mine is None or theirs is None:
            if mine is not theirs:
                return False
        elif not np.array_equal(mine, theirs):
            return False
        return (
            self.name == other.name
            and np.array_equal(self.times, other.times)
            and self.labels == other.labels
            and dict(self.meta) == dict(other.meta)
        )

    def __repr__(self) -> str:
        """One-line summary: name, count, time span, and total epoch duration."""
        if not len(self):
            return f"Events({self.name!r}, n=0)"
        span = f"{float(self.times[0]):.4g} to {float(self.times[-1]):.4g} s"
        extra = f", total={self.total_duration:.4g} s" if self.durations is not None else ""
        return f"Events({self.name!r}, n={len(self)}, {span}{extra})"
