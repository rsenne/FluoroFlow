"""Tests for compute_dff and its normalized variants."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import median_abs_deviation

from fluoroflow import Trace
from fluoroflow.exceptions import ValidationError
from fluoroflow.preprocessing import compute_dff


def signal_and_baseline() -> tuple[Trace, Trace]:
    time = np.arange(5) / 10.0
    signal = Trace(time, np.array([110.0, 120.0, 90.0, 130.0, 100.0]), name="sig")
    baseline = Trace(time, np.full(5, 100.0), name="sig_baseline")
    return signal, baseline


def flat_signal_and_baseline() -> tuple[Trace, Trace]:
    time = np.arange(5) / 10.0
    signal = Trace(time, np.full(5, 100.0), name="sig")
    baseline = Trace(time, np.full(5, 100.0), name="sig_baseline")
    return signal, baseline


class TestComputeDff:
    def test_dff_matches_the_formula(self) -> None:
        signal, baseline = signal_and_baseline()
        out = compute_dff(signal, baseline, method="dff")
        expected = (signal.values - baseline.values) / baseline.values
        np.testing.assert_allclose(out.values, expected)
        assert out.units == "dF/F"

    def test_z_has_zero_mean_and_unit_variance(self) -> None:
        signal, baseline = signal_and_baseline()
        out = compute_dff(signal, baseline, method="z")
        assert out.values.mean() == pytest.approx(0.0, abs=1e-10)
        assert out.values.std() == pytest.approx(1.0)
        assert out.units == "z"

    def test_mad_z_has_zero_median_and_unit_mad(self) -> None:
        signal, baseline = signal_and_baseline()
        out = compute_dff(signal, baseline, method="mad_z")
        assert float(np.median(out.values)) == pytest.approx(0.0, abs=1e-10)
        assert median_abs_deviation(out.values, scale="normal") == pytest.approx(1.0)
        assert out.units == "MAD-z"

    def test_null_z_has_unit_rms(self) -> None:
        signal, baseline = signal_and_baseline()
        out = compute_dff(signal, baseline, method="null_z")
        rms = float(np.sqrt(np.mean(out.values**2)))
        assert rms == pytest.approx(1.0)
        assert out.units == "null-Z"

    def test_unknown_method_raises(self) -> None:
        signal, baseline = signal_and_baseline()
        with pytest.raises(ValidationError, match="must be one of"):
            compute_dff(signal, baseline, method="bogus")

    def test_length_mismatch_raises(self) -> None:
        signal, _baseline = signal_and_baseline()
        short = Trace(np.arange(3) / 10.0, np.zeros(3), name="sig_baseline")
        with pytest.raises(ValidationError, match="same length"):
            compute_dff(signal, short)

    def test_zero_variance_raises_for_z(self) -> None:
        signal, baseline = flat_signal_and_baseline()
        with pytest.raises(ValidationError, match="nonzero variance"):
            compute_dff(signal, baseline, method="z")

    def test_zero_variance_raises_for_mad_z(self) -> None:
        signal, baseline = flat_signal_and_baseline()
        with pytest.raises(ValidationError, match="median absolute deviation"):
            compute_dff(signal, baseline, method="mad_z")

    def test_zero_variance_raises_for_null_z(self) -> None:
        signal, baseline = flat_signal_and_baseline()
        with pytest.raises(ValidationError, match="root-mean-square"):
            compute_dff(signal, baseline, method="null_z")

    def test_step_records_method_and_baseline_name(self) -> None:
        signal, baseline = signal_and_baseline()
        out = compute_dff(signal, baseline, method="dff")
        assert out.history[-1].params == {"method": "dff", "baseline": "sig_baseline"}
