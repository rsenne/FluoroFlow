"""Tests for invalid inputs and edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from fluoroflow import Events, Recording, Trace
from fluoroflow.core.validation import check_positive
from fluoroflow.exceptions import ValidationError


class TestNonNumericScalars:
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
            Recording(
                signals=(Trace([0.0, 0.1], [1.0, 2.0], name="A"),),
                events={"cue": [1.0, 2.0]},  # type: ignore[dict-item]
            )

    def test_with_events_rejects_a_non_events(self) -> None:
        with pytest.raises(ValidationError, match="expects Events objects"):
            Recording(signals=(Trace([0.0, 0.1], [1.0, 2.0], name="A"),)).with_events(
                Trace([0.0, 0.1], [1.0, 2.0])
            )
