"""A toggleable preprocessing pipeline: lowpass filter, isosbestic baseline, dF/F.

Each stage is an options dataclass; set a stage to ``None`` to skip it. Signal
traces flow through the enabled stages in order; the isosbestic (when present)
is lowpass-filtered alongside them since the baseline stage regresses against it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fluoroflow.core.recording import Recording
from fluoroflow.core.trace import Trace
from fluoroflow.exceptions import ValidationError
from fluoroflow.preprocessing.baseline import baseline_correct, fit_isosbestic_baseline
from fluoroflow.preprocessing.dff import DffMethod, compute_dff
from fluoroflow.preprocessing.filtering import lowpass_filter

__all__ = ["BaselineOptions", "DffOptions", "LowpassOptions", "PreprocessOptions", "preprocess"]


@dataclass(frozen=True, slots=True)
class LowpassOptions:
    """Zero-phase Butterworth lowpass filter settings."""

    cutoff_hz: float = 3.0
    order: int = 2


@dataclass(frozen=True, slots=True)
class BaselineOptions:
    """IRLS isosbestic-regression baseline settings (Tukey's bisquare)."""

    tuning_constant: float = 1.4
    max_iter: int = 50
    tol: float = 1e-6


@dataclass(frozen=True, slots=True)
class DffOptions:
    """dF/F normalization settings.

    ``"null_z"`` is the default because it scales dF/F without recentering,
    leaving the zero the isosbestic baseline fit established. That keeps zero
    meaningful as a null, which is what :meth:`AnimalETA.significance` and its
    counterparts test against. Ask for ``"dff"`` to get the raw ratio back.
    """

    method: DffMethod = "null_z"


@dataclass(frozen=True, slots=True)
class PreprocessOptions:
    """Which pipeline stages run, and with what settings.

    Set a field to ``None`` to skip that stage.
    """

    lowpass: LowpassOptions | None = field(default_factory=LowpassOptions)
    baseline: BaselineOptions | None = field(default_factory=BaselineOptions)
    dff: DffOptions | None = field(default_factory=DffOptions)


def preprocess(recording: Recording, options: PreprocessOptions | None = None) -> Recording:
    """Run the enabled stages over every signal channel in ``recording``."""
    opts = options or PreprocessOptions()
    if opts.dff is not None and opts.baseline is None:
        msg = "dF/F requires baseline correction; set options.baseline or options.dff=None."
        raise ValidationError(msg)

    out = recording
    if opts.lowpass is not None:
        lp = opts.lowpass
        out = out.map_traces(lambda t: lowpass_filter(t, cutoff_hz=lp.cutoff_hz, order=lp.order))

    if opts.baseline is None:
        return out

    if out.isosbestic is None:
        msg = "Baseline correction requires an isosbestic control channel."
        raise ValidationError(msg)
    isosbestic = out.isosbestic
    bl = opts.baseline
    dff_method = None if opts.dff is None else opts.dff.method

    def process(signal: Trace) -> Trace:
        baseline = fit_isosbestic_baseline(
            signal,
            isosbestic,
            tuning_constant=bl.tuning_constant,
            max_iter=bl.max_iter,
            tol=bl.tol,
        )
        if dff_method is None:
            return baseline_correct(signal, baseline)
        return compute_dff(signal, baseline, method=dff_method)

    new_signals = tuple(process(s) for s in out.signals)
    return Recording(
        signals=new_signals,
        isosbestic=out.isosbestic,
        events=out.events,
        subject=out.subject,
        session=out.session,
        meta=out.meta,
    )
