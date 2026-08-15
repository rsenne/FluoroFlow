"""Signal traces, an isosbestic control, and events for one session."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from fluoroflow.core.events import Events
from fluoroflow.core.trace import Trace
from fluoroflow.exceptions import ValidationError

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["Recording"]


@dataclass(frozen=True, slots=True, eq=False)
class Recording:
    """One fiber photometry session: 1-2 signal channels, an isosbestic control, and events."""

    signals: tuple[Trace, ...]
    isosbestic: Trace | None = None
    events: Mapping[str, Events] = field(default_factory=dict)
    subject: str | None = None
    session: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the channel shape and event keys, then freeze."""
        signals = tuple(self.signals)
        if not 1 <= len(signals) <= 2:
            msg = f"A recording needs 1 or 2 signal traces, got {len(signals)}."
            raise ValidationError(msg)
        for i, trace in enumerate(signals):
            if not isinstance(trace, Trace):
                msg = f"signals[{i}] must be a Trace, got {type(trace).__name__}."
                raise ValidationError(msg)

        isosbestic = self.isosbestic
        if isosbestic is not None and not isinstance(isosbestic, Trace):
            msg = f"isosbestic must be a Trace or None, got {type(isosbestic).__name__}."
            raise ValidationError(msg)

        names = [trace.name for trace in signals]
        if isosbestic is not None:
            names.append(isosbestic.name)
        if len(names) != len(set(names)):
            msg = f"Trace names within a recording must be unique, got {names!r}."
            raise ValidationError(msg)

        events = dict(self.events)
        for key, event_set in events.items():
            if not isinstance(event_set, Events):
                msg = f"events[{key!r}] must be an Events, got {type(event_set).__name__}."
                raise ValidationError(msg)
            if event_set.name != key:
                msg = f"events[{key!r}] is named {event_set.name!r}; key and name must agree."
                raise ValidationError(msg)

        object.__setattr__(self, "signals", signals)
        object.__setattr__(self, "isosbestic", isosbestic)
        object.__setattr__(self, "events", MappingProxyType(events))
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))

    @classmethod
    def from_traces(
        cls,
        *signals: Trace,
        isosbestic: Trace | None = None,
        events: Mapping[str, Events] | tuple[Events, ...] = (),
        subject: str | None = None,
        session: str | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> Recording:
        """Build a recording, keying events by their own name."""
        if isinstance(events, Mapping):
            keyed_events = dict(events)
        else:
            keyed_events = {}
            for event_set in events:
                if event_set.name in keyed_events:
                    msg = f"Duplicate event name {event_set.name!r}."
                    raise ValidationError(msg)
                keyed_events[event_set.name] = event_set

        return cls(
            signals=signals,
            isosbestic=isosbestic,
            events=keyed_events,
            subject=subject,
            session=session,
            meta=meta or {},
        )

    def _all_traces(self) -> tuple[Trace, ...]:
        """Every trace this recording holds, signals first, then the isosbestic."""
        if self.isosbestic is None:
            return self.signals
        return (*self.signals, self.isosbestic)

    def __getitem__(self, key: str) -> Trace:
        """Return the trace named ``key``."""
        for trace in self._all_traces():
            if trace.name == key:
                return trace
        available = ", ".join(t.name for t in self._all_traces()) or "<none>"
        msg = f"No trace named {key!r}. Available traces: {available}."
        raise KeyError(msg)

    def __contains__(self, key: object) -> bool:
        """Whether a trace with this key exists."""
        return any(trace.name == key for trace in self._all_traces())

    def __iter__(self) -> Iterator[str]:
        """Iterate over trace names, signals first, then the isosbestic."""
        return (trace.name for trace in self._all_traces())

    def __len__(self) -> int:
        """Number of traces (1-3: signals plus an optional isosbestic)."""
        return len(self._all_traces())

    def event(self, key: str) -> Events:
        """Return the event set named ``key``."""
        try:
            return self.events[key]
        except KeyError:
            available = ", ".join(sorted(self.events)) or "<none>"
            msg = f"No events named {key!r}. Available event sets: {available}."
            raise KeyError(msg) from None

    @property
    def duration(self) -> float:
        """Longest trace duration in the recording, in seconds."""
        spans = [t.duration for t in self._all_traces() if len(t) > 0]
        return max(spans) if spans else 0.0

    def with_events(self, *events: Events) -> Recording:
        """Return a copy with these event sets added, replacing any of the same name."""
        merged = dict(self.events)
        for event_set in events:
            if not isinstance(event_set, Events):
                msg = f"with_events expects Events objects, got {type(event_set).__name__}."
                raise ValidationError(msg)
            merged[event_set.name] = event_set
        return self._replace(events=merged)

    def with_meta(self, **updates: Any) -> Recording:
        """Return a copy with additional metadata merged in."""
        merged = dict(self.meta)
        merged.update(updates)
        return self._replace(meta=merged)

    def map_traces(
        self,
        fn: Callable[[Trace], Trace],
        *,
        keys: tuple[str, ...] | None = None,
    ) -> Recording:
        """Apply a trace-to-trace function to the signals and/or the isosbestic.

        Each trace keeps its role (signal or isosbestic) regardless of whether
        ``fn`` renames it.
        """
        available = {trace.name for trace in self._all_traces()}
        targets = available if keys is None else set(keys)
        for key in targets:
            if key not in available:
                listed = ", ".join(sorted(available)) or "<none>"
                msg = f"No trace named {key!r}. Available traces: {listed}."
                raise KeyError(msg)

        def transform(trace: Trace) -> Trace:
            if trace.name not in targets:
                return trace
            result = fn(trace)
            if not isinstance(result, Trace):
                msg = (
                    f"The function passed to map_traces must return a Trace; for "
                    f"{trace.name!r} it returned {type(result).__name__}."
                )
                raise ValidationError(msg)
            return result

        signals = tuple(transform(trace) for trace in self.signals)
        isosbestic = None if self.isosbestic is None else transform(self.isosbestic)
        return self._replace(signals=signals, isosbestic=isosbestic)

    def _replace(self, **changes: Any) -> Recording:
        """Return a copy with the given fields replaced."""
        fields: dict[str, Any] = {
            "signals": self.signals,
            "isosbestic": self.isosbestic,
            "events": dict(self.events),
            "subject": self.subject,
            "session": self.session,
            "meta": dict(self.meta),
        }
        fields.update(changes)
        return Recording(**fields)

    def describe(self) -> pd.DataFrame:
        """Summarise every trace as one row of a :class:`pandas.DataFrame`."""
        import numpy as np
        import pandas as pd

        columns = ["trace", "role", "n_samples", "fs", "duration", "units", "n_missing", "n_steps"]
        rows = []
        roles = [("signal", trace) for trace in self.signals]
        if self.isosbestic is not None:
            roles.append(("isosbestic", self.isosbestic))
        for role, trace in roles:
            short = len(trace) < 2
            rows.append(
                {
                    "trace": trace.name,
                    "role": role,
                    "n_samples": len(trace),
                    "fs": np.nan if short else trace.fs,
                    "duration": np.nan if not len(trace) else trace.duration,
                    "units": trace.units,
                    "n_missing": trace.n_missing,
                    "n_steps": len(trace.history),
                }
            )
        return pd.DataFrame(rows, columns=columns)

    def __eq__(self, other: object) -> bool:
        """Compare by value; returns :data:`NotImplemented` for non-recordings."""
        if not isinstance(other, Recording):
            return NotImplemented
        return (
            self.signals == other.signals
            and self.isosbestic == other.isosbestic
            and dict(self.events) == dict(other.events)
            and self.subject == other.subject
            and self.session == other.session
            and dict(self.meta) == dict(other.meta)
        )

    def __repr__(self) -> str:
        """One-line summary: subject, session, and channel inventory."""
        who = self.subject or "<no subject>"
        what = self.session or "<no session>"
        iso = "yes" if self.isosbestic is not None else "no"
        return (
            f"Recording({who!r}/{what!r}, signals={len(self.signals)}, isosbestic={iso}, "
            f"events={len(self.events)}, duration={self.duration:.4g} s)"
        )
