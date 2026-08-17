"""Tests for animal_eta and AnimalETA."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from fluoroflow import Events, Trace
from fluoroflow.eta import animal_eta
from fluoroflow.exceptions import InsufficientSamplesError


def make_trace(n: int = 300, fs: float = 10.0) -> Trace:
    time = np.arange(n) / fs
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 1.0, n)
    return Trace(time, values, name="sig")


class TestAnimalEta:
    def test_mean_and_sem_match_manual_computation(self) -> None:
        trace = make_trace()
        events = Events(times=np.array([5.0, 10.0, 15.0, 20.0]), name="events")
        result = animal_eta(trace, events, (-1.0, 1.0), ci=None)

        n_pre, n_post = 10, 10
        rows = []
        for t in events.times:
            center = trace.index_at(float(t))
            rows.append(trace.values[center - n_pre : center + n_post])
        trials = np.stack(rows)
        expected_mean = trials.mean(axis=0)
        expected_sem = trials.std(axis=0, ddof=1) / np.sqrt(trials.shape[0])

        np.testing.assert_allclose(result.mean, expected_mean)
        np.testing.assert_allclose(result.sem, expected_sem)
        assert result.n_trials == 4
        assert result.n_dropped == 0

    def test_t_interval_is_symmetric_around_the_mean(self) -> None:
        trace = make_trace()
        events = Events(times=np.array([5.0, 10.0, 15.0, 20.0]), name="events")
        result = animal_eta(trace, events, (-1.0, 1.0), ci="t", confidence=0.95)
        assert result.ci_lower is not None
        assert result.ci_upper is not None
        np.testing.assert_allclose(result.mean - result.ci_lower, result.ci_upper - result.mean)

    def test_t_interval_matches_manual_formula(self) -> None:
        trace = make_trace()
        events = Events(times=np.array([5.0, 10.0, 15.0, 20.0]), name="events")
        result = animal_eta(trace, events, (-1.0, 1.0), ci="t", confidence=0.9)
        t_crit = stats.t.ppf(0.95, df=result.n_trials - 1)
        assert result.ci_lower is not None
        assert result.ci_upper is not None
        np.testing.assert_allclose(result.ci_lower, result.mean - t_crit * result.sem)
        np.testing.assert_allclose(result.ci_upper, result.mean + t_crit * result.sem)

    def test_t_interval_narrows_as_confidence_decreases(self) -> None:
        trace = make_trace()
        events = Events(times=np.array([5.0, 10.0, 15.0, 20.0, 25.0]), name="events")
        wide = animal_eta(trace, events, (-1.0, 1.0), ci="t", confidence=0.99)
        narrow = animal_eta(trace, events, (-1.0, 1.0), ci="t", confidence=0.5)
        assert wide.ci_upper is not None
        assert wide.ci_lower is not None
        assert narrow.ci_upper is not None
        assert narrow.ci_lower is not None
        wide_width = wide.ci_upper - wide.ci_lower
        narrow_width = narrow.ci_upper - narrow.ci_lower
        assert np.all(narrow_width < wide_width)

    def test_bootstrap_is_reproducible_with_a_fixed_seed(self) -> None:
        trace = make_trace()
        events = Events(times=np.array([5.0, 10.0, 15.0, 20.0, 25.0]), name="events")
        first = animal_eta(trace, events, (-1.0, 1.0), ci="bootstrap", seed=42, n_boot=500)
        second = animal_eta(trace, events, (-1.0, 1.0), ci="bootstrap", seed=42, n_boot=500)
        assert first.ci_lower is not None
        assert first.ci_upper is not None
        np.testing.assert_array_equal(first.ci_lower, second.ci_lower)
        np.testing.assert_array_equal(first.ci_upper, second.ci_upper)

    def test_bootstrap_differs_with_a_different_seed(self) -> None:
        trace = make_trace()
        events = Events(times=np.array([5.0, 10.0, 15.0, 20.0, 25.0]), name="events")
        first = animal_eta(trace, events, (-1.0, 1.0), ci="bootstrap", seed=1, n_boot=500)
        second = animal_eta(trace, events, (-1.0, 1.0), ci="bootstrap", seed=2, n_boot=500)
        assert first.ci_lower is not None
        assert second.ci_lower is not None
        assert not np.array_equal(first.ci_lower, second.ci_lower)

    def test_ci_none_gives_no_interval(self) -> None:
        trace = make_trace()
        events = Events(times=np.array([5.0, 10.0, 15.0]), name="events")
        result = animal_eta(trace, events, (-1.0, 1.0), ci=None)
        assert result.ci_lower is None
        assert result.ci_upper is None
        assert result.method is None
        assert result.confidence is None

    def test_fewer_than_two_trials_raises(self) -> None:
        trace = make_trace()
        events = Events(times=np.array([5.0]), name="events")
        with pytest.raises(InsufficientSamplesError, match="at least 2"):
            animal_eta(trace, events, (-1.0, 1.0))

    def test_zero_trials_raises_from_alignment(self) -> None:
        trace = make_trace(n=200, fs=10.0)
        events = Events(times=np.array([0.1, 19.9]), name="events")
        with pytest.raises(InsufficientSamplesError):
            animal_eta(trace, events, (-2.0, 5.0))

    def test_name_defaults_to_trace_name(self) -> None:
        trace = make_trace()
        events = Events(times=np.array([5.0, 10.0]), name="events")
        result = animal_eta(trace, events, (-1.0, 1.0), ci=None)
        assert result.name == "sig"

    def test_name_override(self) -> None:
        trace = make_trace()
        events = Events(times=np.array([5.0, 10.0]), name="events")
        result = animal_eta(trace, events, (-1.0, 1.0), ci=None, name="animal_1")
        assert result.name == "animal_1"

    def test_to_frame_columns(self) -> None:
        trace = make_trace()
        events = Events(times=np.array([5.0, 10.0, 15.0]), name="events")
        result = animal_eta(trace, events, (-1.0, 1.0), ci="t")
        frame = result.to_frame()
        assert list(frame.columns) == ["time", "mean", "sem", "ci_lower", "ci_upper"]
        assert len(frame) == result.time.size

    def test_to_frame_ci_columns_are_nan_when_ci_is_none(self) -> None:
        trace = make_trace()
        events = Events(times=np.array([5.0, 10.0, 15.0]), name="events")
        result = animal_eta(trace, events, (-1.0, 1.0), ci=None)
        frame = result.to_frame()
        assert frame["ci_lower"].isna().all()
        assert frame["ci_upper"].isna().all()
