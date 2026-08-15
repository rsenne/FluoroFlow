"""Traces, events, and explicit channel relationships for one session."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from fluoroflow.core.events import Events
from fluoroflow.core.trace import Trace
from fluoroflow.core.validation import checked_name
from fluoroflow.exceptions import ValidationError

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["ChannelSpec", "Recording"]


@dataclass(frozen=True, slots=True)
class ChannelSpec:
    """How one signal channel relates to its control channel and to the brain.

    Parameters
    ----------
    signal
        Key of the signal trace within the recording, e.g. ``"Region0G"``.
    control
        Key of the movement-control trace (isosbestic, tdTomato) to regress out,
        or ``None`` when the channel has no control.
    indicator
        Sensor name, e.g. ``"GCaMP6f"`` or ``"dLight1.3b"``.
    excitation_nm
        Excitation wavelength in nanometres, e.g. ``470.0``.
    control_nm
        Excitation wavelength of the control channel, e.g. ``415.0``.
    region
        Anatomical target, e.g. ``"vCA1"``.
    hemisphere
        ``"left"``, ``"right"``, or ``None``.
    meta
        Arbitrary metadata.

    Raises
    ------
    ValidationError
        If ``signal`` is empty, or if ``control`` is equal to ``signal``.

    Examples
    --------
    >>> from fluoroflow import ChannelSpec
    >>> ChannelSpec("Region0G", control="Region0G_iso", indicator="GCaMP6f", region="vCA1")
    ChannelSpec('Region0G', control='Region0G_iso', indicator='GCaMP6f', region='vCA1')
    """

    signal: str
    control: str | None = None
    indicator: str | None = None
    excitation_nm: float | None = None
    control_nm: float | None = None
    region: str | None = None
    hemisphere: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the keys and freeze the metadata."""
        object.__setattr__(self, "signal", checked_name(self.signal, label="signal"))
        if self.control is not None:
            control = checked_name(self.control, label="control")
            if control == self.signal:
                msg = (
                    f"A channel cannot be its own control; both signal and control are "
                    f"{control!r}. Pass control=None if this channel has no control."
                )
                raise ValidationError(msg)
            object.__setattr__(self, "control", control)
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))

    @property
    def has_control(self) -> bool:
        """Whether a movement-control channel is paired with this signal."""
        return self.control is not None

    def __repr__(self) -> str:
        """Compact summary that omits unset fields."""
        parts = [repr(self.signal)]
        for name in ("control", "indicator", "excitation_nm", "control_nm", "region", "hemisphere"):
            value = getattr(self, name)
            if value is not None:
                parts.append(f"{name}={value!r}")
        return f"ChannelSpec({', '.join(parts)})"


@dataclass(frozen=True, slots=True, eq=False)
class Recording:
    """An immutable collection of traces and events from a single session.

    Parameters
    ----------
    traces
        Traces keyed by name. Each trace's own ``name`` must equal its key, so
        that a trace pulled out of a recording still knows what it is called.
        Use :meth:`from_traces` to key them automatically.
    events
        Event sets keyed by name, same rule.
    channels
        Signal-to-control pairings. Referenced keys must exist in ``traces``.
    subject
        Animal identifier.
    session
        Session identifier, e.g. ``"fear-conditioning-day1"``.
    meta
        Arbitrary metadata: treatment group, rig, experimenter, anything.

    Raises
    ------
    ValidationError
        If a key disagrees with the contained object's name, if a value is of the
        wrong type, or if a channel spec references a trace that does not exist.

    Notes
    -----
    Mutator methods return new recordings.

    Examples
    --------
    >>> import numpy as np
    >>> from fluoroflow import Recording, Trace
    >>> sig = Trace(np.arange(4) / 4.0, [1.0, 2.0, 3.0, 4.0], name="Region0G")
    >>> rec = Recording.from_traces(sig, subject="M1")
    >>> rec["Region0G"].name
    'Region0G'
    >>> list(rec)
    ['Region0G']
    """

    traces: Mapping[str, Trace]
    events: Mapping[str, Events] = field(default_factory=dict)
    channels: tuple[ChannelSpec, ...] = ()
    subject: str | None = None
    session: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate keys, types, and cross-references, then freeze."""
        traces = dict(self.traces)
        for key, trace in traces.items():
            if not isinstance(trace, Trace):
                msg = f"traces[{key!r}] must be a Trace, got {type(trace).__name__}."
                raise ValidationError(msg)
            if trace.name != key:
                msg = (
                    f"traces[{key!r}] is named {trace.name!r}. A trace's key and its name "
                    f"must agree; use Trace.rename or Recording.from_traces."
                )
                raise ValidationError(msg)

        events = dict(self.events)
        for key, event_set in events.items():
            if not isinstance(event_set, Events):
                msg = f"events[{key!r}] must be an Events, got {type(event_set).__name__}."
                raise ValidationError(msg)
            if event_set.name != key:
                msg = f"events[{key!r}] is named {event_set.name!r}; key and name must agree."
                raise ValidationError(msg)

        channels = tuple(self.channels)
        for i, spec in enumerate(channels):
            if not isinstance(spec, ChannelSpec):
                msg = f"channels[{i}] must be a ChannelSpec, got {type(spec).__name__}."
                raise ValidationError(msg)
            for role in ("signal", "control"):
                key = getattr(spec, role)
                if key is not None and key not in traces:
                    available = ", ".join(sorted(traces)) or "<none>"
                    msg = (
                        f"channels[{i}] names {key!r} as its {role}, but the recording has "
                        f"no such trace. Available traces: {available}."
                    )
                    raise ValidationError(msg)

        object.__setattr__(self, "traces", MappingProxyType(traces))
        object.__setattr__(self, "events", MappingProxyType(events))
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))

    @classmethod
    def from_traces(
        cls,
        *traces: Trace,
        events: Mapping[str, Events] | tuple[Events, ...] = (),
        channels: tuple[ChannelSpec, ...] = (),
        subject: str | None = None,
        session: str | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> Recording:
        """Build a recording from traces, keying each by its own name.

        Parameters
        ----------
        *traces
            Traces to include. Names must be unique.
        events
            Either a mapping already keyed by name, or a tuple of
            :class:`~fluoroflow.core.events.Events` to key automatically.
        channels, subject, session, meta
            As in the constructor.

        Returns
        -------
        Recording
            The assembled recording.

        Raises
        ------
        ValidationError
            If two traces, or two event sets, share a name.
        """
        keyed: dict[str, Trace] = {}
        for trace in traces:
            if not isinstance(trace, Trace):
                msg = f"from_traces expects Trace objects, got {type(trace).__name__}."
                raise ValidationError(msg)
            if trace.name in keyed:
                msg = (
                    f"Duplicate trace name {trace.name!r}; names must be unique within a recording."
                )
                raise ValidationError(msg)
            keyed[trace.name] = trace

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
            traces=keyed,
            events=keyed_events,
            channels=channels,
            subject=subject,
            session=session,
            meta=meta or {},
        )

    def __getitem__(self, key: str) -> Trace:
        """Return the trace named ``key``.

        Raises
        ------
        KeyError
            If no such trace exists.
        """
        try:
            return self.traces[key]
        except KeyError:
            available = ", ".join(sorted(self.traces)) or "<none>"
            msg = f"No trace named {key!r}. Available traces: {available}."
            raise KeyError(msg) from None

    def __contains__(self, key: object) -> bool:
        """Whether a trace with this key exists."""
        return key in self.traces

    def __iter__(self) -> Iterator[str]:
        """Iterate over trace keys, in insertion order."""
        return iter(self.traces)

    def __len__(self) -> int:
        """Number of traces."""
        return len(self.traces)

    def event(self, key: str) -> Events:
        """Return the event set named ``key``.

        Parameters
        ----------
        key
            Event set name.

        Returns
        -------
        Events
            The requested event set.

        Raises
        ------
        KeyError
            If no such event set exists.
        """
        try:
            return self.events[key]
        except KeyError:
            available = ", ".join(sorted(self.events)) or "<none>"
            msg = f"No events named {key!r}. Available event sets: {available}."
            raise KeyError(msg) from None

    def channel(self, signal: str) -> ChannelSpec:
        """Return the channel spec whose signal is ``signal``.

        Parameters
        ----------
        signal
            Signal trace key.

        Returns
        -------
        ChannelSpec
            The matching spec.

        Raises
        ------
        KeyError
            If no channel spec names that trace as its signal.
        """
        for spec in self.channels:
            if spec.signal == signal:
                return spec
        declared = ", ".join(spec.signal for spec in self.channels) or "<none>"
        msg = f"No channel spec with signal {signal!r}. Declared signals: {declared}."
        raise KeyError(msg)

    def pairs(self) -> Iterator[tuple[ChannelSpec, Trace, Trace | None]]:
        """Iterate over ``(spec, signal_trace, control_trace)`` for each channel.

        The control is ``None`` for channels declared without one.

        Yields
        ------
        tuple
            The spec, its signal trace, and its control trace or ``None``.
        """
        for spec in self.channels:
            control = None if spec.control is None else self.traces[spec.control]
            yield spec, self.traces[spec.signal], control

    @property
    def duration(self) -> float:
        """Longest trace duration in the recording, in seconds.

        Returns
        -------
        float
            Maximum of the individual trace durations, or ``0.0`` when the
            recording holds no usable traces.
        """
        spans = [t.duration for t in self.traces.values() if len(t) > 0]
        return max(spans) if spans else 0.0

    def with_traces(self, *traces: Trace) -> Recording:
        """Return a copy with these traces added, replacing any of the same name.

        Parameters
        ----------
        *traces
            Traces to add or replace.

        Returns
        -------
        Recording
            The updated recording.
        """
        merged = dict(self.traces)
        for trace in traces:
            if not isinstance(trace, Trace):
                msg = f"with_traces expects Trace objects, got {type(trace).__name__}."
                raise ValidationError(msg)
            merged[trace.name] = trace
        return self._replace(traces=merged)

    def with_events(self, *events: Events) -> Recording:
        """Return a copy with these event sets added, replacing any of the same name.

        Parameters
        ----------
        *events
            Event sets to add or replace.

        Returns
        -------
        Recording
            The updated recording.
        """
        merged = dict(self.events)
        for event_set in events:
            if not isinstance(event_set, Events):
                msg = f"with_events expects Events objects, got {type(event_set).__name__}."
                raise ValidationError(msg)
            merged[event_set.name] = event_set
        return self._replace(events=merged)

    def with_channels(self, *channels: ChannelSpec) -> Recording:
        """Return a copy with these channel specs, replacing any for the same signal.

        Parameters
        ----------
        *channels
            Specs to add or replace.

        Returns
        -------
        Recording
            The updated recording.
        """
        merged = {spec.signal: spec for spec in self.channels}
        for spec in channels:
            merged[spec.signal] = spec
        return self._replace(channels=tuple(merged.values()))

    def with_meta(self, **updates: Any) -> Recording:
        """Return a copy with additional metadata merged in.

        Parameters
        ----------
        **updates
            Keys to set or overwrite.

        Returns
        -------
        Recording
            The updated recording.
        """
        merged = dict(self.meta)
        merged.update(updates)
        return self._replace(meta=merged)

    def map_traces(
        self,
        fn: Callable[[Trace], Trace],
        *,
        keys: tuple[str, ...] | None = None,
    ) -> Recording:
        """Apply a trace-to-trace function across the recording.

        Parameters
        ----------
        fn
            Function taking a :class:`~fluoroflow.core.trace.Trace` and returning
            one. If it renames the trace, the result is keyed under the new name
            and the old key is dropped.
        keys
            Which traces to transform. ``None`` means all of them.

        Returns
        -------
        Recording
            A recording with the transformed traces in place.

        Raises
        ------
        KeyError
            If a requested key does not exist.
        ValidationError
            If ``fn`` returns something that is not a trace.
        """
        targets = tuple(self.traces) if keys is None else tuple(keys)
        for key in targets:
            if key not in self.traces:
                available = ", ".join(sorted(self.traces)) or "<none>"
                msg = f"No trace named {key!r}. Available traces: {available}."
                raise KeyError(msg)

        merged: dict[str, Trace] = {}
        renames: dict[str, str] = {}
        for key, trace in self.traces.items():
            if key not in targets:
                merged[key] = trace
                continue
            result = fn(trace)
            if not isinstance(result, Trace):
                msg = (
                    f"The function passed to map_traces must return a Trace; for "
                    f"{key!r} it returned {type(result).__name__}."
                )
                raise ValidationError(msg)
            merged[result.name] = result
            if result.name != key:
                renames[key] = result.name

        channels = self.channels
        if renames:
            channels = tuple(
                ChannelSpec(
                    signal=renames.get(spec.signal, spec.signal),
                    control=(
                        None if spec.control is None else renames.get(spec.control, spec.control)
                    ),
                    indicator=spec.indicator,
                    excitation_nm=spec.excitation_nm,
                    control_nm=spec.control_nm,
                    region=spec.region,
                    hemisphere=spec.hemisphere,
                    meta=spec.meta,
                )
                for spec in channels
            )
        return self._replace(traces=merged, channels=channels)

    def _replace(self, **changes: Any) -> Recording:
        """Return a copy with the given fields replaced."""
        fields: dict[str, Any] = {
            "traces": dict(self.traces),
            "events": dict(self.events),
            "channels": self.channels,
            "subject": self.subject,
            "session": self.session,
            "meta": dict(self.meta),
        }
        fields.update(changes)
        return Recording(**fields)

    def describe(self) -> pd.DataFrame:
        """Summarise every trace as one row of a :class:`pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            Columns ``trace``, ``n_samples``, ``fs``, ``duration``, ``units``,
            ``n_missing``, and ``n_steps``. Sampling rate and duration are ``NaN``
            for traces too short to define them, rather than raising, so that a
            summary of a partly broken session still prints.
        """
        import numpy as np
        import pandas as pd

        columns = ["trace", "n_samples", "fs", "duration", "units", "n_missing", "n_steps"]
        rows = []
        for key, trace in self.traces.items():
            short = len(trace) < 2
            rows.append(
                {
                    "trace": key,
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
            dict(self.traces) == dict(other.traces)
            and dict(self.events) == dict(other.events)
            and self.channels == other.channels
            and self.subject == other.subject
            and self.session == other.session
            and dict(self.meta) == dict(other.meta)
        )

    def __repr__(self) -> str:
        """One-line summary: subject, session, and inventory."""
        who = self.subject or "<no subject>"
        what = self.session or "<no session>"
        return (
            f"Recording({who!r}/{what!r}, traces={len(self.traces)}, "
            f"events={len(self.events)}, channels={len(self.channels)}, "
            f"duration={self.duration:.4g} s)"
        )
