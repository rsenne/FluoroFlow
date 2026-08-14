"""The :class:`Trace`: one fluorescence time series and everything known about it.

A trace owns its own time vector. That single decision removes a whole family of
bugs: after de-interleaving an interleaved acquisition, each channel carries the
timestamps it was actually sampled at, so its sampling rate is derived from its
own data and cannot be off by the number of multiplexed LEDs.
"""

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
    """Summary of how regular a trace's sampling actually was.

    Photometry rigs drop frames. Rather than assume uniform sampling and be
    quietly wrong, FluoroFlow measures it and hands you the numbers.

    Attributes
    ----------
    dt_median
        Median sample interval, in seconds. This is what :attr:`Trace.dt` reports.
    dt_min, dt_max
        Extremes of the observed sample intervals, in seconds.
    cv
        Coefficient of variation of the sample intervals (standard deviation over
        mean). Below about 0.01 is a well-behaved recording.
    n_gaps
        Number of intervals longer than 1.5 times the median, i.e. the number of
        places where at least one frame appears to be missing.
    n_samples
        Length of the trace.
    """

    dt_median: float
    dt_min: float
    dt_max: float
    cv: float
    n_gaps: int
    n_samples: int

    @property
    def is_uniform(self) -> bool:
        """Whether the sampling is regular enough to treat as uniform.

        True when there are no detected gaps and the coefficient of variation of
        the sample intervals is below 1 percent.
        """
        return self.n_gaps == 0 and self.cv < 0.01


@dataclass(frozen=True, slots=True, eq=False)
class Trace:
    """An immutable fluorescence time series with its own time base and history.

    Parameters
    ----------
    time
        Sample times in seconds. Must be finite and strictly increasing. Whether
        it starts at zero or at the rig's wall clock is up to you; FluoroFlow
        never assumes.
    values
        Fluorescence values, same length as ``time``. NaN is permitted and means
        "missing"; infinities are rejected.
    name
        Identifier for this trace, e.g. ``"Region0G"``.
    units
        Free-text units, e.g. ``"a.u."``, ``"dF/F"``, ``"z"``. Carried along so a
        plot axis label never has to be guessed.
    meta
        Arbitrary metadata. Copied defensively and exposed read-only.
    history
        Ordered :class:`~fluoroflow.core.provenance.Step` records describing what
        has been done to these values.

    Raises
    ------
    ValidationError
        If the arrays are not one-dimensional, are of unequal length, if ``time``
        is not finite and strictly increasing, or if ``values`` contains an
        infinity.

    Notes
    -----
    Instances are frozen and their arrays are set non-writeable, so a trace is
    safe to share between threads, cache, or hold as a reference point while
    processing. Every transform returns a *new* trace via :meth:`derive`.

    Traces compare by value and are deliberately unhashable, since they wrap
    mutable-sized numeric buffers with no meaningful hash.

    Examples
    --------
    >>> import numpy as np
    >>> from fluoroflow import Trace
    >>> t = Trace(np.arange(5) / 10.0, [1.0, 1.1, 0.9, 1.2, 1.0], name="Region0G")
    >>> round(t.fs, 6)
    10.0
    >>> len(t)
    5
    >>> t.values.flags.writeable
    False
    """

    time: NDArray[np.float64]
    values: NDArray[np.float64]
    name: str = "signal"
    units: str = "a.u."
    meta: Mapping[str, Any] = field(default_factory=dict)
    history: tuple[Step, ...] = ()
    _dt: float = field(init=False, repr=False)

    # ------------------------------------------------------------ construction

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
        """Build a trace from values sampled at a known constant rate.

        Convenience for synthetic data and for rigs that report a nominal rate
        instead of per-sample timestamps. Prefer the normal constructor whenever
        real timestamps exist, because real timestamps record dropped frames and a
        reconstructed time base silently does not.

        Parameters
        ----------
        values
            Fluorescence values.
        fs
            Sampling rate in hertz. Must be finite and positive.
        t0
            Time of the first sample, in seconds.
        name, units, meta
            As in the constructor.

        Returns
        -------
        Trace
            A trace whose time vector is ``t0 + arange(n) / fs``.
        """
        rate = check_positive(fs, label="fs")
        arr = as_series(values, label="values")
        time = float(t0) + np.arange(arr.size, dtype=np.float64) / rate
        return cls(time=time, values=arr, name=name, units=units, meta=meta or {})

    # --------------------------------------------------------------- geometry

    def __len__(self) -> int:
        """Number of samples."""
        return int(self.values.size)

    @property
    def n_samples(self) -> int:
        """Number of samples. Spelled out, for readability at call sites."""
        return int(self.values.size)

    @property
    def dt(self) -> float:
        """Median sample interval in seconds.

        Raises
        ------
        InsufficientSamplesError
            If the trace has fewer than two samples, where no interval exists.
        """
        if math.isnan(self._dt):
            msg = (
                f"Trace {self.name!r} has {len(self)} sample(s); at least 2 are needed "
                f"to define a sample interval."
            )
            raise InsufficientSamplesError(msg)
        return self._dt

    @property
    def fs(self) -> float:
        """Sampling rate in hertz, derived from this trace's own timestamps.

        Raises
        ------
        InsufficientSamplesError
            If the trace has fewer than two samples.
        """
        return 1.0 / self.dt

    @property
    def t0(self) -> float:
        """Time of the first sample, in seconds.

        Raises
        ------
        InsufficientSamplesError
            If the trace is empty.
        """
        if not len(self):
            msg = f"Trace {self.name!r} is empty and has no start time."
            raise InsufficientSamplesError(msg)
        return float(self.time[0])

    @property
    def duration(self) -> float:
        """Elapsed time from the first to the last sample, in seconds.

        Note this is one sample interval shorter than the recorded span; it is
        ``time[-1] - time[0]``, not ``n / fs``.

        Raises
        ------
        InsufficientSamplesError
            If the trace is empty.
        """
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
        """Regularity statistics for this trace's time base.

        Raises
        ------
        InsufficientSamplesError
            If the trace has fewer than two samples.
        """
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

    # ------------------------------------------------------------- provenance

    def has_tag(self, tag: str) -> bool:
        """Whether any recorded step carries ``tag``.

        This is how transforms guard against unsafe composition. For example
        ``dff`` checks :data:`~fluoroflow.core.provenance.MEAN_REMOVED` before
        dividing by a baseline.

        Parameters
        ----------
        tag
            Tag to look for, e.g. :data:`~fluoroflow.core.provenance.MEAN_REMOVED`.

        Returns
        -------
        bool
            True if at least one step in the history carries the tag.
        """
        return any(tag in step.tags for step in self.history)

    def has_step(self, name: str) -> bool:
        """Whether an operation named ``name`` appears in the history.

        Parameters
        ----------
        name
            Step name to look for.

        Returns
        -------
        bool
            True if at least one step has that name.
        """
        return any(step.name == name for step in self.history)

    def describe_history(self) -> str:
        """Render the processing history as a numbered, human-readable list.

        Returns
        -------
        str
            One line per step, or ``"<no processing>"`` if nothing was applied.
        """
        return format_history(self.history)

    # -------------------------------------------------------------- derivation

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
        """Return a new trace descended from this one, with ``step`` appended.

        This is the only supported way to produce a transformed trace, and
        ``step`` is a required keyword argument on purpose: it makes an
        undocumented transform impossible to write by accident. Anything not
        overridden is inherited.

        Parameters
        ----------
        step
            Record of the operation being applied.
        values
            New values. Defaults to the current values, which is what a
            metadata-only or tagging step wants.
        time
            New time vector, for operations that change the time base such as
            resampling or cropping. Defaults to the current time vector.
        name, units
            Overrides for the corresponding fields.
        meta
            Metadata to merge into (not replace) the existing metadata.

        Returns
        -------
        Trace
            The derived trace.

        Examples
        --------
        >>> import numpy as np
        >>> from fluoroflow import Step, Trace
        >>> t = Trace(np.arange(4) / 4.0, [1.0, 2.0, 3.0, 4.0])
        >>> doubled = t.derive(values=t.values * 2, step=Step("double", {"by": 2}))
        >>> doubled.values
        array([2., 4., 6., 8.])
        >>> doubled.describe_history()
        " 1. Step('double', by=2)"
        """
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
        """Return a copy under a new name, leaving the history untouched.

        Renaming is not a transformation of the data, so it records no step.

        Parameters
        ----------
        name
            The new name.

        Returns
        -------
        Trace
            A trace identical except for its name.
        """
        return Trace(
            time=self.time,
            values=self.values,
            name=checked_name(name),
            units=self.units,
            meta=self.meta,
            history=self.history,
        )

    def time_slice(self, start: float | None = None, stop: float | None = None) -> Trace:
        """Return the half-open time window ``[start, stop)`` as a new trace.

        Cropping changes the time base, so it is recorded as a step, including how
        many samples were dropped from each end. Nothing in FluoroFlow discards
        samples silently.

        Parameters
        ----------
        start
            Inclusive lower bound in seconds. ``None`` means the beginning.
        stop
            Exclusive upper bound in seconds. ``None`` means the end.

        Returns
        -------
        Trace
            The cropped trace, possibly empty.

        Raises
        ------
        ValidationError
            If ``start`` is not less than ``stop``.
        """
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
        """Index of the sample nearest in time to ``t``.

        Parameters
        ----------
        t
            Target time in seconds.

        Returns
        -------
        int
            Index of the closest sample. Ties resolve to the earlier sample.

        Raises
        ------
        InsufficientSamplesError
            If the trace is empty.
        """
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

    # ------------------------------------------------------------------ export

    def to_frame(self) -> pd.DataFrame:
        """Return the trace as a two-column :class:`pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            Columns ``time`` and the trace's name. The frame owns writeable
            copies, so editing it cannot corrupt the trace.
        """
        import pandas as pd

        return pd.DataFrame({"time": np.asarray(self.time).copy(), self.name: self.values.copy()})

    # ------------------------------------------------------------------ dunder

    def __eq__(self, other: object) -> bool:
        """Compare two traces by value, treating NaN in the same slot as equal.

        Returns :data:`NotImplemented` for non-traces rather than raising, so that
        Python can fall back to the other operand and ``trace == None`` is False
        instead of an error.
        """
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
