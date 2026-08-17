"""Tests for lowpass_filter."""

from __future__ import annotations

import numpy as np
import pytest

from fluoroflow import Trace
from fluoroflow.exceptions import InsufficientSamplesError, ValidationError
from fluoroflow.preprocessing import lowpass_filter


def sine_plus_noise(fs: float, duration: float, *, low_hz: float, high_hz: float) -> Trace:
    """A slow sine buried in fast sine "noise", for a filter to separate."""
    t = np.arange(int(duration * fs)) / fs
    low = np.sin(2 * np.pi * low_hz * t)
    high = 0.5 * np.sin(2 * np.pi * high_hz * t)
    return Trace(t, low + high, name="signal")


class TestLowpassFilter:
    def test_removes_high_frequency_content(self) -> None:
        trace = sine_plus_noise(200.0, 10.0, low_hz=1.0, high_hz=50.0)
        low_only = np.sin(2 * np.pi * 1.0 * trace.time)
        out = lowpass_filter(trace, cutoff_hz=5.0, order=2)
        residual = out.values - low_only
        assert float(np.std(residual)) < 0.1

    def test_preserves_time_and_length(self) -> None:
        trace = sine_plus_noise(200.0, 5.0, low_hz=1.0, high_hz=50.0)
        out = lowpass_filter(trace, cutoff_hz=5.0)
        np.testing.assert_array_equal(out.time, trace.time)
        assert len(out) == len(trace)

    def test_records_a_step(self) -> None:
        trace = sine_plus_noise(200.0, 5.0, low_hz=1.0, high_hz=50.0)
        out = lowpass_filter(trace, cutoff_hz=5.0, order=3)
        assert out.history[-1].params == {"cutoff_hz": 5.0, "order": 3}

    def test_rejects_a_cutoff_at_or_above_nyquist(self) -> None:
        trace = Trace(np.arange(100) / 100.0, np.zeros(100))
        with pytest.raises(ValidationError, match="Nyquist"):
            lowpass_filter(trace, cutoff_hz=50.0)

    def test_rejects_missing_samples(self) -> None:
        values = np.zeros(100)
        values[10] = np.nan
        trace = Trace(np.arange(100) / 100.0, values)
        with pytest.raises(ValidationError, match="missing sample"):
            lowpass_filter(trace, cutoff_hz=5.0)

    def test_too_few_samples_raises_insufficient_samples(self) -> None:
        trace = Trace(np.arange(4) / 100.0, np.zeros(4))
        with pytest.raises(InsufficientSamplesError, match="too few samples"):
            lowpass_filter(trace, cutoff_hz=5.0, order=3)
