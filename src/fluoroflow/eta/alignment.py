"""Cutting fixed-width trial windows out of a trace around event onsets."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fluoroflow.core.events import Events
from fluoroflow.core.trace import Trace
from fluoroflow.exceptions import InsufficientSamplesError, ValidationError

__all__ = ["align_to_events"]


def align_to_events(
    trace: Trace, events: Events, window: tuple[float, float]
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Cut a ``window`` of samples around each event onset out of ``trace``.

    ``window = (pre, post)`` seconds relative to onset, e.g. ``(-2.0, 5.0)`` for
    2 s before to 5 s after. Trials whose window does not fit fully inside the
    trace are dropped rather than NaN-padded.
    """
    if not window[0] < window[1]:
        msg = f"window start must be less than window stop, got window={window!r}."
        raise ValidationError(msg)

    fs = trace.fs
    n_pre = round(-window[0] * fs)
    n_post = round(window[1] * fs)
    relative_time = np.arange(-n_pre, n_post) / fs

    n_samples = len(trace)
    n_events = len(events)
    valid_rows: list[NDArray[np.float64]] = []
    n_dropped = 0
    for t in events.times:
        center = trace.index_at(float(t))
        lo = center - n_pre
        hi = center + n_post
        if lo < 0 or hi > n_samples:
            n_dropped += 1
            continue
        valid_rows.append(trace.values[lo:hi])

    if not valid_rows:
        msg = (
            f"No trials survived windowing: {n_dropped} of {n_events} event(s) were "
            f"dropped because window={window!r} s falls outside the bounds of trace "
            f"{trace.name!r} for every one of them."
        )
        raise InsufficientSamplesError(msg)

    trials = np.stack(valid_rows)
    return relative_time, trials, n_dropped
