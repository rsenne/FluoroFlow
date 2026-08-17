"""Tests for the IRLS isosbestic baseline fit."""

from __future__ import annotations

import numpy as np
import pytest

from fluoroflow import Trace
from fluoroflow.exceptions import InsufficientSamplesError, ValidationError
from fluoroflow.preprocessing import baseline_correct, fit_isosbestic_baseline


def linear_pair(n: int, *, slope: float, intercept: float) -> tuple[Trace, np.ndarray, np.ndarray]:
    """An isosbestic ramp and a signal that is an exact linear function of it."""
    time = np.arange(n) / 10.0
    iso_values = np.linspace(0.0, 1.0, n)
    true_baseline = slope * iso_values + intercept
    iso = Trace(time, iso_values, name="iso")
    return iso, true_baseline, time


class TestFitIsosbesticBaseline:
    def test_recovers_an_exact_linear_relationship(self) -> None:
        iso, true_baseline, time = linear_pair(20, slope=2.0, intercept=5.0)
        signal = Trace(time, true_baseline, name="sig")
        baseline = fit_isosbestic_baseline(signal, iso)
        np.testing.assert_allclose(baseline.values, true_baseline, atol=1e-8)
        assert baseline.history[-1].params["converged"] is True

    def test_downweights_a_transient_the_isosbestic_does_not_share(self) -> None:
        iso, true_baseline, time = linear_pair(50, slope=3.0, intercept=1.0)
        signal_values = true_baseline.copy()
        signal_values[10] += 20.0
        signal = Trace(time, signal_values, name="sig")

        baseline = fit_isosbestic_baseline(signal, iso)
        ols_a, ols_b = np.polyfit(iso.values, signal_values, 1)
        ols_fitted = ols_a * iso.values + ols_b

        irls_error = np.abs(baseline.values - true_baseline)
        ols_error = np.abs(ols_fitted - true_baseline)
        assert irls_error.mean() < ols_error.mean()
        assert irls_error[10] < ols_error[10]

    def test_output_naming_and_units(self) -> None:
        iso, true_baseline, time = linear_pair(20, slope=1.0, intercept=0.0)
        signal = Trace(time, true_baseline, name="Region0G", units="a.u.")
        baseline = fit_isosbestic_baseline(signal, iso)
        assert baseline.name == "Region0G_baseline"
        assert baseline.units == "a.u."

    def test_interpolates_an_offset_isosbestic_onto_the_signals_own_clock(self) -> None:
        # Simulate interleaved acquisition: the isosbestic's clock is offset from the
        # signal's, and padded well beyond it so no boundary clamping is exercised here.
        signal_time = np.arange(20) / 10.0
        iso_time = np.arange(-5, 25) / 10.0 + 0.02
        iso_values = 2.0 * iso_time + 1.0
        true_baseline = 3.0 * (2.0 * signal_time + 1.0) + 5.0

        iso = Trace(iso_time, iso_values, name="iso")
        signal = Trace(signal_time, true_baseline, name="sig")
        baseline = fit_isosbestic_baseline(signal, iso)
        np.testing.assert_allclose(baseline.values, true_baseline, atol=1e-8)

    def test_rejects_missing_samples(self) -> None:
        iso = Trace(np.arange(10) / 10.0, np.zeros(10), name="iso")
        values = np.zeros(10)
        values[3] = np.nan
        signal = Trace(np.arange(10) / 10.0, values, name="sig")
        with pytest.raises(ValidationError, match="missing samples"):
            fit_isosbestic_baseline(signal, iso)

    def test_rejects_too_few_samples(self) -> None:
        iso = Trace(np.arange(2) / 10.0, np.zeros(2), name="iso")
        signal = Trace(np.arange(2) / 10.0, np.zeros(2), name="sig")
        with pytest.raises(InsufficientSamplesError, match="At least 3"):
            fit_isosbestic_baseline(signal, iso)


class TestBaselineCorrect:
    def test_subtracts_the_baseline(self) -> None:
        time = np.arange(5) / 10.0
        signal = Trace(time, np.array([1.0, 2.0, 3.0, 4.0, 5.0]), name="sig")
        baseline = Trace(time, np.full(5, 0.5), name="sig_baseline")
        out = baseline_correct(signal, baseline)
        np.testing.assert_allclose(out.values, [0.5, 1.5, 2.5, 3.5, 4.5])
        assert out.history[-1].params == {"baseline": "sig_baseline"}

    def test_rejects_a_length_mismatch(self) -> None:
        signal = Trace(np.arange(5) / 10.0, np.zeros(5), name="sig")
        baseline = Trace(np.arange(3) / 10.0, np.zeros(3), name="sig_baseline")
        with pytest.raises(ValidationError, match="same length"):
            baseline_correct(signal, baseline)
