"""Fluorescence time series with timestamps and processing history."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from fluoroflow.core.provenance import Step, format_history
from fluoroflow.core.validation import (
    as_series,
    check_matching_length,
    check_no_infinities,
    check_positive,
    check_time_vector,
    checked_name,
    median_dt,
)
from fluoroflow.exceptions import InsufficientSamplesError, ValidationError

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["SamplingReport", "Trace"]


@dataclass(frozen=True, slots=True)
class SamplingReport:
    """Summary of a trace's sampling regularity."""

    dt_median: float
    dt_min: float
    dt_max: float
    cv: float
    n_gaps: int
    n_samples: int

    @property
    def is_uniform(self) -> bool:
        """Whether sampling is regular enough to treat as uniform."""
        return self.n_gaps == 0 and self.cv < 0.01


@dataclass(frozen=True, slots=True, eq=False)
class Trace:
    """An immutable fluorescence time series with its own time base and history."""

    time: NDArray[np.float64]
    values: NDArray[np.float64]
    name: str = "signal"
    units: str = "a.u."
    meta: Mapping[str, Any] = field(default_factory=dict)
    history: tuple[Step, ...] = ()
    _dt: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Coerce, validate, and freeze the inputs."""
        time = as_series(self.time, label="time")
        values = as_series(self.values, label="values")
        check_matching_length(time, values, labels=("time", "values"))
        check_time_vector(time)
        check_no_infinities(values)

        history = tuple(self.history)
        for i, step in enumerate(history):
            if not isinstance(step, Step):
                msg = f"history[{i}] must be a Step, got {type(step).__name__}."
                raise ValidationError(msg)

        object.__setattr__(self, "time", time)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "name", checked_name(self.name))
        object.__setattr__(self, "units", str(self.units))
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "_dt", median_dt(time))

    @classmethod
    def from_uniform(
        cls,
        values: Any,
        *,
        fs: float,
        t0: float = 0.0,
        name: str = "signal",
        units: str = "a.u.",
        meta: Mapping[str, Any] | None = None,
    ) -> Trace:
        """Build a trace from values sampled at a known constant rate."""
        rate = check_positive(fs, label="fs")
        arr = as_series(values, label="values")
        time = float(t0) + np.arange(arr.size, dtype=np.float64) / rate
        return cls(time=time, values=arr, name=name, units=units, meta=meta or {})

    def __len__(self) -> int:
        """Number of samples."""
        return int(self.values.size)

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return int(self.values.size)

    @property
    def dt(self) -> float:
        """Median sample interval in seconds."""
        if math.isnan(self._dt):
            msg = (
                f"Trace {self.name!r} has {len(self)} sample(s); at least 2 are needed "
                f"to define a sample interval."
            )
            raise InsufficientSamplesError(msg)
        return self._dt

    @property
    def fs(self) -> float:
        """Sampling rate in hertz, derived from this trace's own timestamps."""
        return 1.0 / self.dt

    @property
    def t0(self) -> float:
        """Time of the first sample, in seconds."""
        if not len(self):
            msg = f"Trace {self.name!r} is empty and has no start time."
            raise InsufficientSamplesError(msg)
        return float(self.time[0])

    @property
    def duration(self) -> float:
        """Elapsed time from the first to the last sample, in seconds."""
        if not len(self):
            msg = f"Trace {self.name!r} is empty and has no duration."
            raise InsufficientSamplesError(msg)
        return float(self.time[-1] - self.time[0])

    @property
    def n_missing(self) -> int:
        """Number of NaN samples."""
        return int(np.count_nonzero(np.isnan(self.values)))

    @property
    def sampling(self) -> SamplingReport:
        """Regularity statistics for this trace's time base."""
        if math.isnan(self._dt):
            msg = (
                f"Trace {self.name!r} has {len(self)} sample(s); at least 2 are needed "
                f"to describe sampling."
            )
            raise InsufficientSamplesError(msg)
        steps = np.diff(self.time)
        mean = float(steps.mean())
        cv = float(steps.std() / mean) if mean > 0.0 else math.inf
        return SamplingReport(
            dt_median=self._dt,
            dt_min=float(steps.min()),
            dt_max=float(steps.max()),
            cv=cv,
            n_gaps=int(np.count_nonzero(steps > 1.5 * self._dt)),
            n_samples=len(self),
        )

    def has_step(self, name: str) -> bool:
        """Whether an operation named ``name`` appears in the history."""
        return any(step.name == name for step in self.history)

    def describe_history(self) -> str:
        """Render the processing history as a numbered, human-readable list."""
        return format_history(self.history)

    def derive(
        self,
        *,
        step: Step,
        values: Any = None,
        time: Any = None,
        name: str | None = None,
        units: str | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> Trace:
        """Return a new trace descended from this one, with ``step`` appended."""
        if not isinstance(step, Step):
            msg = f"step must be a Step, got {type(step).__name__}."
            raise ValidationError(msg)
        merged: dict[str, Any] = dict(self.meta)
        if meta:
            merged.update(meta)
        return Trace(
            time=self.time if time is None else time,
            values=self.values if values is None else values,
            name=self.name if name is None else name,
            units=self.units if units is None else units,
            meta=merged,
            history=(*self.history, step),
        )

    def rename(self, name: str) -> Trace:
        """Return a copy under a new name, leaving the history untouched."""
        return Trace(
            time=self.time,
            values=self.values,
            name=checked_name(name),
            units=self.units,
            meta=self.meta,
            history=self.history,
        )

    def time_slice(self, start: float | None = None, stop: float | None = None) -> Trace:
        """Return the half-open time window ``[start, stop)`` as a new trace."""
        lo = -math.inf if start is None else float(start)
        hi = math.inf if stop is None else float(stop)
        if not lo < hi:
            msg = f"start must be less than stop, got start={start!r}, stop={stop!r}."
            raise ValidationError(msg)
        keep = (self.time >= lo) & (self.time < hi)
        first = int(np.argmax(keep)) if keep.any() else len(self)
        return Trace(
            time=self.time[keep],
            values=self.values[keep],
            name=self.name,
            units=self.units,
            meta=self.meta,
            history=(
                *self.history,
                Step(
                    "time_slice",
                    {
                        "start": start,
                        "stop": stop,
                        "n_dropped_before": first,
                        "n_dropped_after": len(self) - first - int(keep.sum()),
                        "n_kept": int(keep.sum()),
                    },
                ),
            ),
        )

    def index_at(self, t: float) -> int:
        """Index of the sample nearest in time to ``t``."""
        if not len(self):
            msg = f"Trace {self.name!r} is empty; no sample to index."
            raise InsufficientSamplesError(msg)
        right = int(np.searchsorted(self.time, t))
        if right == 0:
            return 0
        if right >= len(self):
            return len(self) - 1
        left = right - 1
        if (t - self.time[left]) <= (self.time[right] - t):
            return left
        return right

    def to_frame(self) -> pd.DataFrame:
        """Return the trace as a two-column :class:`pandas.DataFrame`."""
        import pandas as pd

        return pd.DataFrame({"time": np.asarray(self.time).copy(), self.name: self.values.copy()})

    def __eq__(self, other: object) -> bool:
        """Compare two traces by value, treating NaN in the same slot as equal."""
        if not isinstance(other, Trace):
            return NotImplemented
        return (
            self.name == other.name
            and self.units == other.units
            and np.array_equal(self.time, other.time)
            and np.array_equal(self.values, other.values, equal_nan=True)
            and dict(self.meta) == dict(other.meta)
            and self.history == other.history
        )

    def __repr__(self) -> str:
        """One-line summary: name, length, rate, span, units, and step count."""
        if math.isnan(self._dt):
            rate = "rate undefined"
            span = "span undefined"
        else:
            rate = f"{1.0 / self._dt:.4g} Hz"
            span = f"{self.duration:.4g} s"
        missing = f", {self.n_missing} NaN" if self.n_missing else ""
        return (
            f"Trace({self.name!r}, n={len(self)}, {rate}, {span}, "
            f"units={self.units!r}, steps={len(self.history)}{missing})"
        )
