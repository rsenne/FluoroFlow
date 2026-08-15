"""Tests for invalid inputs and edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from fluoroflow import ChannelSpec, Events, Recording, Trace
from fluoroflow.core.validation import check_percentage, check_positive
from fluoroflow.datasets import synthetic_recording
from fluoroflow.exceptions import ValidationError


class TestNonNumericScalars:
    def test_check_percentage_on_something_that_is_not_a_number(self) -> None:
        with pytest.raises(ValidationError, match="in percent"):
            check_percentage("eight")

    def test_check_positive_on_something_that_is_not_a_number(self) -> None:
        with pytest.raises(ValidationError, match="must be a real number"):
            check_positive([30.0], label="fs")


class TestEventGuards:
    def test_nan_durations_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="durations must be finite"):
            Events([1.0, 2.0], durations=[1.0, np.nan])

    def test_infinite_durations_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="durations must be finite"):
            Events([1.0], durations=[np.inf])

    def test_to_boolean_rejects_an_unsorted_grid(self) -> None:
        with pytest.raises(ValidationError, match="strictly increasing"):
            Events([1.0], durations=[1.0]).to_boolean([0.0, 2.0, 1.0])

    def test_from_boolean_rejects_an_unsorted_time_base(self) -> None:
        with pytest.raises(ValidationError, match="strictly increasing"):
            Events.from_boolean([True, False, True], [0.0, 2.0, 1.0])

    def test_an_epoch_entirely_past_the_grid_marks_nothing(self) -> None:
        got = Events([100.0], durations=[1.0]).to_boolean([0.0, 1.0, 2.0])
        assert not got.any()

    @pytest.mark.parametrize("grid", [[], [0.0]])
    def test_to_boolean_on_a_grid_with_no_sample_interval(self, grid: list[float]) -> None:
        got = Events([0.0], durations=[1.0]).to_boolean(grid)
        assert got.tolist() == [True] * len(grid)


class TestRecordingGuards:
    def test_events_mapping_rejects_a_non_events_value(self) -> None:
        with pytest.raises(ValidationError, match="must be an Events"):
            Recording(traces={}, events={"cue": [1.0, 2.0]})  # type: ignore[dict-item]

    def test_channels_reject_a_non_channelspec(self) -> None:
        with pytest.raises(ValidationError, match="must be a ChannelSpec"):
            Recording(traces={}, channels=("Region0G",))

    def test_with_events_rejects_a_non_events(self) -> None:
        with pytest.raises(ValidationError, match="expects Events objects"):
            Recording(traces={}).with_events(Trace([0.0, 0.1], [1.0, 2.0]))

    def test_from_traces_accepts_events_as_a_mapping(self) -> None:
        rec = Recording.from_traces(
            Trace([0.0, 0.1], [1.0, 2.0], name="A"), events={"cue": Events([0.05], name="cue")}
        )
        assert rec.event("cue").name == "cue"

    def test_channel_lookup_scans_past_non_matching_specs(self) -> None:
        rec = Recording.from_traces(
            Trace([0.0, 0.1], [1.0, 2.0], name="A"),
            Trace([0.0, 0.1], [3.0, 4.0], name="B"),
            channels=(ChannelSpec("A"), ChannelSpec("B", region="BLA")),
        )
        assert rec.channel("B").region == "BLA"


class TestDatasetEdges:
    def test_a_transient_scheduled_past_the_end_is_dropped_not_crashed(self) -> None:
        data = synthetic_recording(
            duration=5.0,
            event_times=[4.99],
            n_transients=0,
            noise_cv=0.0,
            motion_gain_signal=0.0,
            seed=0,
        )
        assert len(data.signal) > 0
        np.testing.assert_array_equal(data.truth.transient_dff, 0.0)

    def test_a_transient_near_the_end_is_truncated_to_fit(self) -> None:
        data = synthetic_recording(
            duration=5.0,
            event_times=[4.0],
            n_transients=0,
            noise_cv=0.0,
            motion_gain_signal=0.0,
            seed=0,
        )
        truth = data.truth
        assert truth.transient_dff[-1] > 0.0, "the tail should still be rising or decaying"
        assert truth.transient_dff.max() <= 0.10 + 1e-12
