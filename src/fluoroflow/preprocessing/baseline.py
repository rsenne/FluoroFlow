"""Baseline and motion correction via robust regression of the isosbestic control.

Following Keevers & Jean-Richard-dit-Bressel (Neurophotonics, 2025): the isosbestic
signal is fit to the experimental signal by iteratively reweighted least squares
(IRLS) with Tukey's bisquare weighting. Samples where the two channels diverge
(real neural transients, absent from the isosbestic) are downweighted each
iteration, so the converged fit tracks only what the two channels share:
photobleaching and motion artifact. That fit is the baseline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import median_abs_deviation

from fluoroflow.core.provenance import Step
from fluoroflow.core.trace import Trace
from fluoroflow.exceptions import InsufficientSamplesError, ValidationError

__all__ = ["baseline_correct", "fit_isosbestic_baseline"]


def fit_isosbestic_baseline(
    signal: Trace,
    isosbestic: Trace,
    *,
    tuning_constant: float = 1.4,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> Trace:
    """Robustly regress the isosbestic onto the signal and return the fitted baseline."""
    if len(signal) != len(isosbestic):
        msg = (
            f"signal and isosbestic must be frame-aligned (same sample count); got "
            f"{len(signal)} for {signal.name!r} and {len(isosbestic)} for {isosbestic.name!r}."
        )
        raise ValidationError(msg)
    if signal.n_missing or isosbestic.n_missing:
        msg = "Baseline fitting cannot run through missing samples; interpolate or crop first."
        raise ValidationError(msg)
    if len(signal) < 3:
        msg = f"At least 3 samples are needed to fit a robust baseline, got {len(signal)}."
        raise InsufficientSamplesError(msg)

    x = isosbestic.values
    y = signal.values
    design = np.column_stack([x, np.ones_like(x)])

    weights = np.ones_like(y)
    coef = np.zeros(2)
    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):  # noqa: B007
        sqrt_w = np.sqrt(weights)
        new_coef, *_ = np.linalg.lstsq(design * sqrt_w[:, None], y * sqrt_w, rcond=None)
        residuals = y - design @ new_coef

        if np.max(np.abs(new_coef - coef)) < tol:
            coef = new_coef
            converged = True
            break
        coef = new_coef

        scale = median_abs_deviation(residuals, scale="normal")
        if scale == 0.0:
            converged = True
            break
        u = residuals / (tuning_constant * scale)
        weights = np.where(np.abs(u) < 1.0, (1.0 - u**2) ** 2, 0.0)

    fitted: NDArray[np.float64] = design @ coef
    return Trace(
        time=signal.time,
        values=fitted,
        name=f"{signal.name}_baseline",
        units=signal.units,
        history=(
            Step(
                "irls_baseline",
                {
                    "reference": isosbestic.name,
                    "tuning_constant": tuning_constant,
                    "iterations": iterations,
                    "converged": converged,
                },
            ),
        ),
    )


def baseline_correct(signal: Trace, baseline: Trace) -> Trace:
    """Subtract a fitted baseline from a signal, in the signal's original units."""
    if len(signal) != len(baseline):
        msg = (
            f"signal and baseline must have the same length, got {len(signal)} and {len(baseline)}."
        )
        raise ValidationError(msg)
    return signal.derive(
        values=signal.values - baseline.values,
        step=Step("baseline_correct", {"baseline": baseline.name}),
    )
