"""dF/F and its normalized variants."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.stats import median_abs_deviation

from fluoroflow.core.provenance import Step
from fluoroflow.core.trace import Trace
from fluoroflow.exceptions import ValidationError

__all__ = ["DffMethod", "compute_dff"]

DffMethod = Literal["dff", "z", "mad_z", "null_z"]

_UNITS = {"dff": "dF/F", "z": "z", "mad_z": "MAD-z", "null_z": "null-Z"}


def compute_dff(signal: Trace, baseline: Trace, *, method: DffMethod = "dff") -> Trace:
    """Compute dF/F against a fitted baseline, optionally normalized.

    ``"dff"`` is the raw ``(signal - baseline) / baseline`` ratio. ``"z"`` and
    ``"mad_z"`` are the standard and median/MAD-robust z-scores of that ratio.
    ``"null_z"`` instead scales by the root-mean-square deviation from zero
    without recentering, since the IRLS baseline already anchors the zero
    point and re-centering to the sample mean would undo that (Keevers &
    Jean-Richard-dit-Bressel, 2025).
    """
    if method not in _UNITS:
        msg = f"method must be one of {sorted(_UNITS)}, got {method!r}."
        raise ValidationError(msg)
    if len(signal) != len(baseline):
        msg = (
            f"signal and baseline must have the same length, got {len(signal)} and {len(baseline)}."
        )
        raise ValidationError(msg)

    dff = (signal.values - baseline.values) / baseline.values

    if method == "dff":
        values = dff
    elif method == "z":
        sd = float(dff.std())
        if sd == 0.0:
            msg = "z-score requires nonzero variance in the dF/F trace."
            raise ValidationError(msg)
        values = (dff - dff.mean()) / sd
    elif method == "mad_z":
        mad = median_abs_deviation(dff, scale="normal")
        if mad == 0.0:
            msg = "mad_z requires a nonzero median absolute deviation in the dF/F trace."
            raise ValidationError(msg)
        values = (dff - np.median(dff)) / mad
    else:
        rms = float(np.sqrt(np.mean(dff**2)))
        if rms == 0.0:
            msg = "null_z requires a nonzero root-mean-square deviation in the dF/F trace."
            raise ValidationError(msg)
        values = dff / rms

    return signal.derive(
        values=values,
        units=_UNITS[method],
        step=Step("dff", {"method": method, "baseline": baseline.name}),
    )
