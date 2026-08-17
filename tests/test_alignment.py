"""Tests for align_to_events."""

from __future__ import annotations

import numpy as np
import pytest

from fluoroflow import Events, Trace
from fluoroflow.eta import align_to_events
from fluoroflow.exceptions import InsufficientSamplesError, ValidationError


def make_trace(n: int = 200, fs: float = 10.0) -> Trace:
    time = np.arange(n) / fs
    values = np.arange(n, dtype=np.float64)
    return Trace(time, values, name="sig")


class TestAlignToEvents:
    def test_shape_and_count_for_events_safely_inside_the_trace(self) -> None:
        trace = make_trace(n=200, fs=10.0)
        events = Events(times=np.array([5.0, 10.0, 15.0]), name="events")
        relative_time, trials, n_dropped = align_to_events(trace, events, (-2.0, 5.0))
        n_pre, n_post = 20, 50
        assert relative_time.shape == (n_pre + n_post,)
        assert trials.shape == (3, n_pre + n_post)
        assert n_dropped == 0

    def test_trial_values_match_the_expected_slice(self) -> None:
        trace = make_trace(n=200, fs=10.0)
        events = Events(times=np.array([10.0]), name="events")
        _relative_time, trials, n_dropped = align_to_events(trace, events, (-2.0, 5.0))
        center = trace.index_at(10.0)
        expected = trace.values[center - 20 : center + 50]
        np.testing.assert_allclose(trials[0], expected)
        assert n_dropped == 0

    def test_relative_time_matches_window_bounds(self) -> None:
        trace = make_trace(n=200, fs=10.0)
        events = Events(times=np.array([10.0]), name="events")
        relative_time, _trials, _n_dropped = align_to_events(trace, events, (-2.0, 5.0))
        assert relative_time[0] == pytest.approx(-2.0)
        assert relative_time[-1] == pytest.approx(5.0 - 1.0 / trace.fs)

    def test_events_near_the_edges_are_dropped_and_counted(self) -> None:
        trace = make_trace(n=200, fs=10.0)
        # 10.0 s and 15.0 s are safely inside; 0.5 s and 19.9 s cannot fit a (-2, 5) window.
        events = Events(times=np.array([0.5, 10.0, 15.0, 19.9]), name="events")
        _relative_time, trials, n_dropped = align_to_events(trace, events, (-2.0, 5.0))
        assert trials.shape[0] == 2
        assert n_dropped == 2

    def test_window_start_not_less_than_stop_raises(self) -> None:
        trace = make_trace()
        events = Events(times=np.array([10.0]), name="events")
        with pytest.raises(ValidationError, match="window start must be less than window stop"):
            align_to_events(trace, events, (5.0, -2.0))

    def test_equal_window_bounds_raise(self) -> None:
        trace = make_trace()
        events = Events(times=np.array([10.0]), name="events")
        with pytest.raises(ValidationError, match="window start must be less than window stop"):
            align_to_events(trace, events, (1.0, 1.0))

    def test_zero_surviving_trials_raises(self) -> None:
        trace = make_trace(n=200, fs=10.0)
        events = Events(times=np.array([0.1, 19.9]), name="events")
        with pytest.raises(InsufficientSamplesError, match="No trials survived windowing"):
            align_to_events(trace, events, (-2.0, 5.0))
