"""Tests for Trace: construction, immutability, geometry, provenance, derivation."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from fluoroflow import MEAN_REMOVED, NORMALIZED, Step, Trace
from fluoroflow.exceptions import InsufficientSamplesError, ValidationError


class TestConstruction:
    def test_accepts_lists(self) -> None:
        t = Trace([0.0, 0.1, 0.2], [1.0, 2.0, 3.0])
        assert len(t) == 3
        assert t.values.dtype == np.float64

    def test_defaults(self) -> None:
        t = Trace([0.0, 0.1], [1.0, 2.0])
        assert t.name == "signal"
        assert t.units == "a.u."
        assert t.history == ()
        assert dict(t.meta) == {}

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValidationError, match="same length"):
            Trace([0.0, 0.1, 0.2], [1.0, 2.0])

    def test_rejects_non_monotonic_time(self) -> None:
        with pytest.raises(ValidationError, match="strictly increasing"):
            Trace([0.0, 0.2, 0.1], [1.0, 2.0, 3.0])

    def test_rejects_nan_in_time(self) -> None:
        with pytest.raises(ValidationError, match="finite"):
            Trace([0.0, np.nan], [1.0, 2.0])

    def test_allows_nan_in_values(self) -> None:
        t = Trace([0.0, 0.1, 0.2], [1.0, np.nan, 3.0])
        assert t.n_missing == 1

    def test_rejects_infinity_in_values(self) -> None:
        with pytest.raises(ValidationError, match="infinite"):
            Trace([0.0, 0.1], [1.0, np.inf])

    def test_rejects_blank_name(self) -> None:
        with pytest.raises(ValidationError, match="non-empty string"):
            Trace([0.0, 0.1], [1.0, 2.0], name="  ")

    def test_rejects_two_dimensional_values(self) -> None:
        with pytest.raises(ValidationError, match="one-dimensional"):
            Trace([0.0, 0.1], np.zeros((2, 2)))

    def test_rejects_a_history_entry_that_is_not_a_step(self) -> None:
        with pytest.raises(ValidationError, match="must be a Step"):
            Trace([0.0, 0.1], [1.0, 2.0], history=("dff",))

    def test_empty_trace_is_constructible(self) -> None:
        t = Trace([], [])
        assert len(t) == 0

    def test_from_uniform_builds_the_expected_time_base(self) -> None:
        t = Trace.from_uniform([1.0, 2.0, 3.0, 4.0], fs=20.0, t0=5.0)
        np.testing.assert_allclose(t.time, [5.0, 5.05, 5.1, 5.15])
        assert t.fs == pytest.approx(20.0)

    def test_from_uniform_rejects_a_non_positive_rate(self) -> None:
        with pytest.raises(ValidationError, match="greater than zero"):
            Trace.from_uniform([1.0, 2.0], fs=0.0)


class TestImmutability:
    def test_values_are_not_writeable(self, simple_trace: Trace) -> None:
        assert not simple_trace.values.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            simple_trace.values[0] = 99.0

    def test_time_is_not_writeable(self, simple_trace: Trace) -> None:
        assert not simple_trace.time.flags.writeable

    def test_attributes_cannot_be_reassigned(self, simple_trace: Trace) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            simple_trace.name = "other"  # type: ignore[misc]

    def test_later_edits_to_the_source_array_do_not_leak_in(self) -> None:
        values = np.array([1.0, 2.0, 3.0])
        t = Trace([0.0, 0.1, 0.2], values)
        values[0] = 99.0
        assert t.values[0] == 1.0

    def test_metadata_is_a_defensive_copy(self) -> None:
        meta = {"rig": "A"}
        t = Trace([0.0, 0.1], [1.0, 2.0], meta=meta)
        meta["rig"] = "B"
        assert t.meta["rig"] == "A"

    def test_traces_are_unhashable(self, simple_trace: Trace) -> None:
        with pytest.raises(TypeError):
            hash(simple_trace)


class TestGeometry:
    def test_dt_and_fs_come_from_the_traces_own_timestamps(self) -> None:
        t = Trace(np.arange(100) / 30.0, np.zeros(100))
        assert t.dt == pytest.approx(1 / 30.0)
        assert t.fs == pytest.approx(30.0)

    def test_rate_is_unaffected_by_the_interleaving_of_the_source_file(self) -> None:
        interleaved = np.arange(300) / 90.0
        demuxed = Trace(interleaved[::3], np.zeros(100), name="Region0G")
        assert demuxed.fs == pytest.approx(30.0)

    @pytest.mark.parametrize("n", [0, 1])
    def test_rate_raises_rather_than_guessing_when_undefined(self, n: int) -> None:
        t = Trace(np.arange(n, dtype=float), np.zeros(n))
        with pytest.raises(InsufficientSamplesError, match="at least 2"):
            _ = t.fs
        with pytest.raises(InsufficientSamplesError, match="at least 2"):
            _ = t.dt

    def test_t0_and_duration(self) -> None:
        t = Trace(np.arange(10) / 10.0 + 4.0, np.zeros(10))
        assert t.t0 == pytest.approx(4.0)
        assert t.duration == pytest.approx(0.9)

    def test_empty_trace_has_no_start_or_duration(self) -> None:
        t = Trace([], [])
        with pytest.raises(InsufficientSamplesError, match="empty"):
            _ = t.t0
        with pytest.raises(InsufficientSamplesError, match="empty"):
            _ = t.duration

    def test_n_samples_matches_len(self, simple_trace: Trace) -> None:
        assert simple_trace.n_samples == len(simple_trace) == 10


class TestSamplingReport:
    def test_uniform_sampling_is_reported_as_uniform(self) -> None:
        report = Trace(np.arange(100) / 30.0, np.zeros(100)).sampling
        assert report.is_uniform
        assert report.n_gaps == 0
        assert report.cv == pytest.approx(0.0, abs=1e-9)
        assert report.n_samples == 100

    def test_a_dropped_frame_is_reported_as_a_gap(self) -> None:
        time = np.delete(np.arange(100) / 30.0, 50)
        report = Trace(time, np.zeros(99)).sampling
        assert report.n_gaps == 1
        assert not report.is_uniform
        assert report.dt_max == pytest.approx(2 / 30.0)
        assert report.dt_median == pytest.approx(1 / 30.0)

    def test_report_requires_two_samples(self) -> None:
        with pytest.raises(InsufficientSamplesError, match="at least 2"):
            _ = Trace([0.0], [1.0]).sampling


class TestProvenance:
    def test_derive_appends_exactly_one_step(self, simple_trace: Trace) -> None:
        out = simple_trace.derive(values=simple_trace.values * 2, step=Step("scale", {"by": 2}))
        assert len(out.history) == len(simple_trace.history) + 1
        assert out.history[-1] == Step("scale", {"by": 2})

    def test_derive_leaves_the_original_untouched(self, simple_trace: Trace) -> None:
        before = simple_trace.values.copy()
        simple_trace.derive(values=np.zeros(len(simple_trace)), step=Step("zero"))
        np.testing.assert_array_equal(simple_trace.values, before)
        assert simple_trace.history == ()

    def test_derive_requires_a_step(self, simple_trace: Trace) -> None:
        with pytest.raises(TypeError):
            simple_trace.derive(values=simple_trace.values)  # type: ignore[call-arg]

    def test_derive_rejects_a_non_step(self, simple_trace: Trace) -> None:
        with pytest.raises(ValidationError, match="must be a Step"):
            simple_trace.derive(values=simple_trace.values, step="scaled")

    def test_derive_validates_the_new_values(self, simple_trace: Trace) -> None:
        with pytest.raises(ValidationError, match="same length"):
            simple_trace.derive(values=np.zeros(3), step=Step("truncate"))

    def test_derive_can_change_the_time_base(self, simple_trace: Trace) -> None:
        out = simple_trace.derive(
            time=np.arange(5) / 5.0,
            values=np.zeros(5),
            step=Step("resample", {"fs": 5.0}),
        )
        assert out.fs == pytest.approx(5.0)

    def test_derive_merges_rather_than_replaces_metadata(self) -> None:
        t = Trace([0.0, 0.1], [1.0, 2.0], meta={"rig": "A"})
        out = t.derive(step=Step("tag"), meta={"operator": "rs"})
        assert dict(out.meta) == {"rig": "A", "operator": "rs"}

    def test_derive_inherits_name_units_and_values_when_not_overridden(self) -> None:
        t = Trace([0.0, 0.1], [1.0, 2.0], name="Region0G", units="a.u.")
        out = t.derive(step=Step("tag"))
        assert out.name == "Region0G"
        assert out.units == "a.u."
        np.testing.assert_array_equal(out.values, t.values)

    def test_tags_are_queryable_across_the_whole_history(self, simple_trace: Trace) -> None:
        out = simple_trace.derive(step=Step("detrend", tags={MEAN_REMOVED})).derive(
            step=Step("zscore", tags={NORMALIZED})
        )
        assert out.has_tag(MEAN_REMOVED)
        assert out.has_tag(NORMALIZED)
        assert not out.has_tag("motion-corrected")

    def test_step_lookup_by_name(self, simple_trace: Trace) -> None:
        out = simple_trace.derive(step=Step("airpls", {"lam": 1e7}))
        assert out.has_step("airpls")
        assert not out.has_step("butterworth")

    def test_describe_history_is_readable(self, simple_trace: Trace) -> None:
        assert simple_trace.describe_history() == "<no processing>"
        out = simple_trace.derive(step=Step("airpls", {"lam": 100.0}))
        assert out.describe_history() == " 1. Step('airpls', lam=100.0)"

    def test_rename_is_not_a_transformation(self, simple_trace: Trace) -> None:
        out = simple_trace.rename("Region1R")
        assert out.name == "Region1R"
        assert out.history == simple_trace.history


class TestTimeSlice:
    def test_returns_the_half_open_window(self) -> None:
        t = Trace(np.arange(10) / 10.0, np.arange(10, dtype=float))
        out = t.time_slice(0.2, 0.5)
        np.testing.assert_allclose(out.time, [0.2, 0.3, 0.4])
        np.testing.assert_allclose(out.values, [2.0, 3.0, 4.0])

    def test_records_how_many_samples_it_dropped(self) -> None:
        t = Trace(np.arange(10) / 10.0, np.arange(10, dtype=float))
        params = t.time_slice(0.2, 0.5).history[-1].params
        assert params["n_dropped_before"] == 2
        assert params["n_dropped_after"] == 5
        assert params["n_kept"] == 3

    def test_open_bounds(self) -> None:
        t = Trace(np.arange(10) / 10.0, np.arange(10, dtype=float))
        assert len(t.time_slice(stop=0.3)) == 3
        assert len(t.time_slice(start=0.7)) == 3
        assert len(t.time_slice()) == 10

    def test_a_window_containing_nothing_yields_an_empty_trace(self) -> None:
        t = Trace(np.arange(10) / 10.0, np.arange(10, dtype=float))
        out = t.time_slice(50.0, 60.0)
        assert len(out) == 0
        assert out.history[-1].params["n_dropped_before"] == 10

    def test_rejects_an_inverted_window(self) -> None:
        t = Trace(np.arange(10) / 10.0, np.arange(10, dtype=float))
        with pytest.raises(ValidationError, match="start must be less than stop"):
            t.time_slice(0.5, 0.2)


class TestIndexAt:
    @pytest.mark.parametrize(("t", "expected"), [(0.0, 0), (0.31, 3), (0.29, 3), (0.9, 9)])
    def test_finds_the_nearest_sample(self, t: float, expected: int) -> None:
        trace = Trace(np.arange(10) / 10.0, np.zeros(10))
        assert trace.index_at(t) == expected

    def test_clamps_outside_the_recording(self) -> None:
        trace = Trace(np.arange(10) / 10.0, np.zeros(10))
        assert trace.index_at(-5.0) == 0
        assert trace.index_at(500.0) == 9

    def test_ties_go_to_the_earlier_sample(self) -> None:
        trace = Trace([0.0, 1.0], [0.0, 0.0])
        assert trace.index_at(0.5) == 0

    def test_empty_trace_has_no_index(self) -> None:
        with pytest.raises(InsufficientSamplesError, match="empty"):
            Trace([], []).index_at(0.0)


class TestEqualityAndRepr:
    def test_equal_traces_compare_equal(self) -> None:
        a = Trace([0.0, 0.1], [1.0, 2.0], name="x")
        b = Trace([0.0, 0.1], [1.0, 2.0], name="x")
        assert a == b

    def test_nan_in_the_same_slot_counts_as_equal(self) -> None:
        a = Trace([0.0, 0.1], [1.0, np.nan])
        b = Trace([0.0, 0.1], [1.0, np.nan])
        assert a == b

    @pytest.mark.parametrize(
        "other",
        [
            Trace([0.0, 0.1], [1.0, 2.0], name="y"),
            Trace([0.0, 0.2], [1.0, 2.0], name="x"),
            Trace([0.0, 0.1], [1.0, 3.0], name="x"),
            Trace([0.0, 0.1], [1.0, 2.0], name="x", units="z"),
            Trace([0.0, 0.1], [1.0, 2.0], name="x", meta={"a": 1}),
            Trace([0.0, 0.1], [1.0, 2.0], name="x", history=(Step("s"),)),
        ],
    )
    def test_any_differing_field_breaks_equality(self, other: Trace) -> None:
        assert Trace([0.0, 0.1], [1.0, 2.0], name="x") != other

    def test_comparison_with_a_non_trace_is_false_not_an_error(self, simple_trace: Trace) -> None:
        assert simple_trace != None  # noqa: E711
        assert simple_trace != 3
        assert simple_trace.__eq__("Region0G") is NotImplemented

    def test_repr_summarises_without_dumping_the_data(self, simple_trace: Trace) -> None:
        text = repr(simple_trace)
        assert "Region0G" in text
        assert "n=10" in text
        assert "10 Hz" in text

    def test_repr_of_a_trace_too_short_to_have_a_rate(self) -> None:
        assert "rate undefined" in repr(Trace([0.0], [1.0]))

    def test_repr_flags_missing_samples(self) -> None:
        assert "1 NaN" in repr(Trace([0.0, 0.1], [1.0, np.nan]))


class TestExport:
    def test_to_frame_columns(self, simple_trace: Trace) -> None:
        frame = simple_trace.to_frame()
        assert list(frame.columns) == ["time", "Region0G"]
        assert len(frame) == 10

    def test_to_frame_hands_back_writeable_copies(self, simple_trace: Trace) -> None:
        frame = simple_trace.to_frame()
        frame.loc[0, "Region0G"] = 99.0
        assert simple_trace.values[0] != 99.0


def test_dt_cache_is_not_part_of_the_public_repr(simple_trace: Trace) -> None:
    assert "_dt" not in repr(simple_trace)


def test_median_dt_sentinel_is_nan_for_short_traces() -> None:
    assert math.isnan(Trace([0.0], [1.0])._dt)
