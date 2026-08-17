"""Lowpass filtering."""

from __future__ import annotations

from scipy.signal import butter, filtfilt

from fluoroflow.core.provenance import Step
from fluoroflow.core.trace import Trace
from fluoroflow.core.validation import check_positive
from fluoroflow.exceptions import InsufficientSamplesError, ValidationError

__all__ = ["lowpass_filter"]


def lowpass_filter(trace: Trace, *, cutoff_hz: float = 3.0, order: int = 2) -> Trace:
    """Zero-phase Butterworth lowpass filter."""
    cutoff = check_positive(cutoff_hz, label="cutoff_hz")
    nyquist = trace.fs / 2.0
    if cutoff >= nyquist:
        msg = (
            f"cutoff_hz must be below the Nyquist frequency ({nyquist:.4g} Hz for "
            f"{trace.name!r} at {trace.fs:.4g} Hz), got {cutoff!r}."
        )
        raise ValidationError(msg)
    if trace.n_missing:
        msg = (
            f"Trace {trace.name!r} has {trace.n_missing} missing sample(s); filtering "
            f"cannot run through NaNs. Interpolate or crop first."
        )
        raise ValidationError(msg)

    b, a = butter(order, cutoff / nyquist, btype="low")
    try:
        filtered = filtfilt(b, a, trace.values)
    except ValueError as exc:
        msg = (
            f"Trace {trace.name!r} has too few samples ({len(trace)}) for an "
            f"order-{order} zero-phase filter."
        )
        raise InsufficientSamplesError(msg) from exc

    step = Step("lowpass", {"cutoff_hz": cutoff, "order": order})
    return trace.derive(values=filtered, step=step)
