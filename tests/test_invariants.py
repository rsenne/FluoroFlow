"""Property-based tests for the core data model."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

from fluoroflow import Events, Step, Trace

# Keep generated values within realistic photometry magnitudes.
values = npst.arrays(
    dtype=np.float64,
    shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=0, max_side=200),
    elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)

sample_counts = st.integers(min_value=0, max_value=200)
rates = st.floats(min_value=0.5, max_value=1000.0, allow_nan=False, allow_infinity=False)

# Epoch durations require at least two timestamps.
boolean_masks = npst.arrays(
    dtype=np.bool_,
    shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=2, max_side=200),
)


def uniform_trace(n: int, fs: float, data: np.ndarray | None = None) -> Trace:
    """A trace on a uniform grid, for properties that do not care about the values."""
    payload = np.zeros(n) if data is None else data[:n]
    return Trace(np.arange(n) / fs, payload, name="p")


class TestTraceInvariants:
    @given(values)
    def test_construction_preserves_the_values_exactly(self, data: np.ndarray) -> None:
        trace = Trace(np.arange(data.size, dtype=float), data)
        np.testing.assert_array_equal(trace.values, data)

    @given(values)
    def test_construction_never_hands_back_a_writeable_buffer(self, data: np.ndarray) -> None:
        trace = Trace(np.arange(data.size, dtype=float), data)
        assert not trace.values.flags.writeable
        assert not trace.time.flags.writeable

    @given(values)
    def test_a_trace_equals_a_second_trace_built_the_same_way(self, data: np.ndarray) -> None:
        time = np.arange(data.size, dtype=float)
        assert Trace(time, data) == Trace(time, data)

    @given(sample_counts, rates)
    def test_the_derived_rate_matches_the_grid_it_was_built_on(self, n: int, fs: float) -> None:
        assume(n >= 2)
        assert uniform_trace(n, fs).fs == pytest.approx(fs, rel=1e-9)

    @given(values)
    def test_derive_appends_exactly_one_step_and_keeps_the_length(self, data: np.ndarray) -> None:
        trace = Trace(np.arange(data.size, dtype=float), data)
        out = trace.derive(values=data * 2.0, step=Step("scale", {"by": 2.0}))
        assert len(out.history) == 1
        assert len(out) == len(trace)

    @given(values, st.integers(min_value=1, max_value=6))
    def test_history_length_equals_the_number_of_derivations(
        self, data: np.ndarray, k: int
    ) -> None:
        trace = Trace(np.arange(data.size, dtype=float), data)
        for i in range(k):
            trace = trace.derive(step=Step(f"step{i}"))
        assert len(trace.history) == k

    @given(sample_counts, st.floats(min_value=0.0, max_value=20.0))
    def test_time_slice_partitions_the_samples(self, n: int, cut: float) -> None:
        assume(n >= 1)
        trace = uniform_trace(n, 10.0)
        out = trace.time_slice(stop=cut)
        params = out.history[-1].params
        assert params["n_kept"] + params["n_dropped_before"] + params["n_dropped_after"] == n
        assert params["n_kept"] == len(out)

    @given(sample_counts, st.floats(min_value=-5.0, max_value=25.0))
    def test_complementary_windows_reconstruct_the_whole_trace(self, n: int, cut: float) -> None:
        assume(n >= 1)
        trace = uniform_trace(n, 10.0)
        left, right = trace.time_slice(stop=cut), trace.time_slice(start=cut)
        assert len(left) + len(right) == n
        np.testing.assert_array_equal(np.concatenate([left.time, right.time]), trace.time)

    @given(sample_counts, st.floats(min_value=-5.0, max_value=25.0, allow_nan=False))
    def test_index_at_returns_the_argmin_of_the_time_distance(self, n: int, t: float) -> None:
        assume(n >= 1)
        trace = uniform_trace(n, 10.0)
        got = trace.index_at(t)
        assert abs(trace.time[got] - t) == np.abs(trace.time - t).min()

    @given(values)
    def test_rename_changes_nothing_but_the_name(self, data: np.ndarray) -> None:
        trace = Trace(np.arange(data.size, dtype=float), data, name="before")
        out = trace.rename("after")
        assert out.name == "after"
        np.testing.assert_array_equal(out.values, trace.values)
        assert out.history == trace.history

    @given(values)
    def test_to_frame_round_trips_the_values(self, data: np.ndarray) -> None:
        trace = Trace(np.arange(data.size, dtype=float), data, name="p")
        np.testing.assert_array_equal(trace.to_frame()["p"].to_numpy(), data)


class TestEventInvariants:
    @given(
        npst.arrays(
            dtype=np.float64,
            shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=0, max_side=100),
            elements=st.floats(min_value=-1e4, max_value=1e4, allow_nan=False),
        )
    )
    def test_sorted_times_are_always_accepted(self, times: np.ndarray) -> None:
        assert len(Events(np.sort(times))) == times.size

    @given(boolean_masks)
    def test_boolean_masks_round_trip_on_a_uniform_grid(self, mask: np.ndarray) -> None:
        time = np.arange(mask.size) / 20.0
        restored = Events.from_boolean(mask, time).to_boolean(time)
        np.testing.assert_array_equal(restored, mask)

    @given(boolean_masks)
    def test_epoch_count_equals_the_number_of_runs(self, mask: np.ndarray) -> None:
        expected = int(np.count_nonzero(np.diff(mask.astype(np.int8), prepend=np.int8(0)) == 1))
        assert len(Events.from_boolean(mask, np.arange(mask.size) / 20.0)) == expected

    @given(boolean_masks)
    def test_total_epoch_duration_equals_the_time_the_mask_was_true(self, mask: np.ndarray) -> None:
        dt = 1.0 / 20.0
        events = Events.from_boolean(mask, np.arange(mask.size) * dt)
        expected = float(np.count_nonzero(mask)) * dt
        assert events.total_duration == pytest.approx(expected, rel=1e-9, abs=1e-12)

    @given(
        npst.arrays(
            dtype=np.float64,
            shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=0, max_side=50),
            elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False),
        ),
        st.floats(min_value=-1e3, max_value=1e3, allow_nan=False),
    )
    def test_shifting_by_a_delta_and_back_is_the_identity(
        self, times: np.ndarray, delta: float
    ) -> None:
        events = Events(np.sort(times))
        np.testing.assert_allclose(events.shift(delta).shift(-delta).times, events.times, atol=1e-9)

    @given(
        npst.arrays(
            dtype=np.float64,
            shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=0, max_side=50),
            elements=st.floats(min_value=0.0, max_value=1e3, allow_nan=False),
        ),
        st.floats(min_value=0.0, max_value=500.0),
        st.floats(min_value=500.0, max_value=1e3),
    )
    def test_within_and_its_complement_partition_the_events(
        self, times: np.ndarray, lo: float, hi: float
    ) -> None:
        assume(lo < hi)
        events = Events(np.sort(times))
        inside = len(events.within(lo, hi))
        outside = len(events.within(stop=lo)) + len(events.within(start=hi))
        assert inside + outside == len(events)
