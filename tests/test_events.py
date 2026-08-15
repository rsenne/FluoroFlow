"""Tests for Events."""

from __future__ import annotations

import numpy as np
import pytest

from fluoroflow import Events
from fluoroflow.exceptions import InsufficientSamplesError, ValidationError


class TestConstruction:
    def test_accepts_a_list_of_times(self) -> None:
        ev = Events([1.0, 2.0, 3.0], name="tone")
        assert len(ev) == 3
        assert ev.name == "tone"
        assert not ev.has_durations

    def test_allows_simultaneous_events(self) -> None:
        assert len(Events([1.0, 1.0, 2.0])) == 3

    def test_rejects_unsorted_times(self) -> None:
        with pytest.raises(ValidationError, match="non-decreasing"):
            Events([1.0, 3.0, 2.0])

    def test_rejects_nan_times(self) -> None:
        with pytest.raises(ValidationError, match="finite"):
            Events([1.0, np.nan])

    def test_rejects_negative_durations(self) -> None:
        with pytest.raises(ValidationError, match="non-negative"):
            Events([1.0, 2.0], durations=[1.0, -0.5])

    def test_allows_zero_duration(self) -> None:
        assert Events([1.0], durations=[0.0]).total_duration == pytest.approx(0.0)

    def test_rejects_mismatched_durations(self) -> None:
        with pytest.raises(ValidationError, match="same length"):
            Events([1.0, 2.0], durations=[1.0])

    def test_rejects_mismatched_labels(self) -> None:
        with pytest.raises(ValidationError, match="same length"):
            Events([1.0, 2.0], labels=("a",))

    def test_empty_is_constructible(self) -> None:
        assert len(Events([])) == 0

    def test_metadata_is_a_defensive_copy(self) -> None:
        meta = {"source": "anymaze"}
        ev = Events([1.0], meta=meta)
        meta["source"] = "tampered"
        assert ev.meta["source"] == "anymaze"

    def test_events_are_unhashable(self) -> None:
        with pytest.raises(TypeError):
            hash(Events([1.0]))


class TestFromBoolean:
    def test_two_runs(self) -> None:
        time = np.arange(8) * 0.5
        mask = [False, True, True, False, False, True, False, False]
        ev = Events.from_boolean(mask, time, name="freezing")
        np.testing.assert_allclose(ev.times, [0.5, 2.5])
        np.testing.assert_allclose(ev.durations, [1.0, 0.5])

    def test_a_run_starting_at_the_first_sample(self) -> None:
        time = np.arange(5) * 1.0
        ev = Events.from_boolean([True, True, False, False, False], time)
        np.testing.assert_allclose(ev.times, [0.0])
        np.testing.assert_allclose(ev.durations, [2.0])

    def test_a_run_reaching_the_end_of_the_recording_is_closed_off(self) -> None:
        time = np.arange(5) * 1.0
        ev = Events.from_boolean([False, False, False, True, True], time)
        np.testing.assert_allclose(ev.times, [3.0])
        np.testing.assert_allclose(ev.durations, [2.0])

    def test_all_true(self) -> None:
        time = np.arange(4) * 0.25
        ev = Events.from_boolean(np.ones(4, dtype=bool), time)
        np.testing.assert_allclose(ev.times, [0.0])
        np.testing.assert_allclose(ev.durations, [1.0])

    def test_all_false_yields_no_epochs(self) -> None:
        ev = Events.from_boolean(np.zeros(4, dtype=bool), np.arange(4) * 0.25)
        assert len(ev) == 0
        assert ev.has_durations

    def test_alternating_samples(self) -> None:
        time = np.arange(6) * 1.0
        ev = Events.from_boolean([True, False, True, False, True, False], time)
        np.testing.assert_allclose(ev.times, [0.0, 2.0, 4.0])
        np.testing.assert_allclose(ev.durations, [1.0, 1.0, 1.0])

    def test_a_lone_true_sample_has_no_measurable_width_and_refuses(self) -> None:
        with pytest.raises(InsufficientSamplesError, match="at least 2 timestamps"):
            Events.from_boolean([True], [0.0])

    def test_a_lone_false_sample_needs_no_interval_and_is_accepted(self) -> None:
        assert len(Events.from_boolean([False], [0.0])) == 0

    def test_empty_input(self) -> None:
        assert len(Events.from_boolean([], [])) == 0

    def test_two_samples_are_enough(self) -> None:
        ev = Events.from_boolean([True, False], [0.0, 0.5])
        np.testing.assert_allclose(ev.durations, [0.5])

    def test_accepts_integer_zero_one_coding(self) -> None:
        ev = Events.from_boolean([0, 1, 1, 0], np.arange(4) * 1.0)
        np.testing.assert_allclose(ev.times, [1.0])
        np.testing.assert_allclose(ev.durations, [2.0])

    def test_rejects_a_length_mismatch(self) -> None:
        with pytest.raises(ValidationError, match="same length"):
            Events.from_boolean([True, False], [0.0, 1.0, 2.0])

    def test_rejects_a_two_dimensional_mask(self) -> None:
        with pytest.raises(ValidationError, match="one-dimensional"):
            Events.from_boolean(np.zeros((2, 2), dtype=bool), [0.0, 1.0])

    def test_works_on_a_non_uniform_time_base(self) -> None:
        time = np.array([0.0, 0.5, 3.0, 3.5, 4.0])
        ev = Events.from_boolean([False, True, True, False, False], time)
        np.testing.assert_allclose(ev.times, [0.5])
        np.testing.assert_allclose(ev.durations, [3.0])


class TestToBoolean:
    def test_round_trips_on_a_uniform_grid(self) -> None:
        time = np.arange(20) * 0.1
        mask = np.zeros(20, dtype=bool)
        mask[3:7] = True
        mask[12:13] = True
        restored = Events.from_boolean(mask, time).to_boolean(time)
        np.testing.assert_array_equal(restored, mask)

    def test_a_run_ending_one_sample_short_does_not_swallow_the_last_sample(self) -> None:
        time = np.arange(18) / 20.0
        mask = np.ones(18, dtype=bool)
        mask[5] = False
        mask[17] = False
        restored = Events.from_boolean(mask, time).to_boolean(time)
        np.testing.assert_array_equal(restored, mask)

    def test_round_trips_on_a_wall_clock_time_base(self) -> None:
        time = 1.7e9 + np.arange(200) / 30.0
        mask = np.zeros(200, dtype=bool)
        mask[3:50] = True
        mask[120:199] = True
        restored = Events.from_boolean(mask, time).to_boolean(time)
        np.testing.assert_array_equal(restored, mask)

    def test_the_tolerance_never_swallows_a_genuine_one_sample_epoch(self) -> None:
        time = np.arange(10) / 20.0
        mask = np.zeros(10, dtype=bool)
        mask[4] = True
        events = Events.from_boolean(mask, time)
        assert len(events) == 1
        np.testing.assert_array_equal(events.to_boolean(time), mask)

    def test_requires_durations(self) -> None:
        with pytest.raises(ValidationError, match="no durations"):
            Events([1.0]).to_boolean(np.arange(5) * 1.0)

    def test_marks_the_half_open_interval(self) -> None:
        ev = Events([1.0], durations=[2.0])
        got = ev.to_boolean([0.0, 1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(got, [False, True, True, False, False])

    def test_evaluates_onto_a_different_grid(self) -> None:
        ev = Events([1.0], durations=[1.0])
        got = ev.to_boolean(np.arange(0, 3, 0.5))
        np.testing.assert_array_equal(got, [False, False, True, True, False, False])


class TestOffsets:
    def test_offsets_are_onsets_plus_durations(self) -> None:
        ev = Events([1.0, 5.0], durations=[0.5, 2.0])
        np.testing.assert_allclose(ev.offsets, [1.5, 7.0])

    def test_total_duration(self) -> None:
        assert Events([1.0, 5.0], durations=[0.5, 2.0]).total_duration == pytest.approx(2.5)

    def test_instants_have_no_offsets(self) -> None:
        with pytest.raises(ValidationError, match="not epochs"):
            _ = Events([1.0]).offsets

    def test_instants_have_no_total_duration(self) -> None:
        with pytest.raises(ValidationError, match="no durations"):
            _ = Events([1.0]).total_duration


class TestSubsets:
    def test_within_selects_on_onset(self) -> None:
        ev = Events([1.0, 5.0, 9.5], name="tone")
        np.testing.assert_allclose(ev.within(2.0, 10.0).times, [5.0, 9.5])

    def test_within_is_half_open(self) -> None:
        ev = Events([1.0, 2.0, 3.0])
        np.testing.assert_allclose(ev.within(1.0, 3.0).times, [1.0, 2.0])

    def test_within_keeps_an_epoch_that_extends_past_the_window(self) -> None:
        ev = Events([1.0], durations=[100.0])
        np.testing.assert_allclose(ev.within(0.0, 2.0).durations, [100.0])

    def test_within_carries_labels_along(self) -> None:
        ev = Events([1.0, 5.0], labels=("a", "b"))
        assert ev.within(2.0).labels == ("b",)

    def test_within_rejects_an_inverted_window(self) -> None:
        with pytest.raises(ValidationError, match="start must be less than stop"):
            Events([1.0]).within(5.0, 2.0)

    def test_with_label(self) -> None:
        ev = Events([1.0, 2.0, 3.0], name="cue", labels=("tone", "shock", "tone"))
        picked = ev.with_label("tone")
        np.testing.assert_allclose(picked.times, [1.0, 3.0])
        assert picked.name == "cue[tone]"

    def test_with_label_requires_labels(self) -> None:
        with pytest.raises(ValidationError, match="no labels"):
            Events([1.0]).with_label("tone")

    def test_shift_moves_onsets_and_leaves_durations(self) -> None:
        ev = Events([1.0, 2.0], durations=[0.5, 0.5]).shift(-0.25)
        np.testing.assert_allclose(ev.times, [0.75, 1.75])
        np.testing.assert_allclose(ev.durations, [0.5, 0.5])

    def test_shift_round_trips(self) -> None:
        ev = Events([1.0, 2.0])
        np.testing.assert_allclose(ev.shift(3.0).shift(-3.0).times, ev.times)


class TestEqualityAndExport:
    def test_equality(self) -> None:
        assert Events([1.0], name="a") == Events([1.0], name="a")
        assert Events([1.0], name="a") != Events([1.0], name="b")
        assert Events([1.0]) != Events([1.0], durations=[1.0])
        assert Events([1.0], durations=[1.0]) != Events([1.0], durations=[2.0])

    def test_comparison_with_a_non_events_is_false_not_an_error(self) -> None:
        assert Events([1.0]) != None  # noqa: E711
        assert Events([1.0]) != 3

    def test_iteration_yields_floats(self) -> None:
        assert list(Events([1.0, 2.0])) == [1.0, 2.0]

    def test_to_frame_for_instants(self) -> None:
        assert list(Events([1.0, 2.0]).to_frame().columns) == ["onset"]

    def test_to_frame_for_epochs_with_labels(self) -> None:
        ev = Events([1.0], durations=[2.0], labels=("tone",))
        assert list(ev.to_frame().columns) == ["onset", "duration", "offset", "label"]

    def test_repr(self) -> None:
        assert repr(Events([])) == "Events('events', n=0)"
        text = repr(Events([1.0, 9.0], name="shock", durations=[1.0, 1.0]))
        assert "shock" in text
        assert "n=2" in text
        assert "total=2 s" in text
