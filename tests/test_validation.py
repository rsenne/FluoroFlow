"""Tests for the precondition guards."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fluoroflow.core.validation import (
    as_series,
    check_matching_length,
    check_no_infinities,
    check_percentage,
    check_positive,
    check_time_vector,
    checked_name,
    median_dt,
)
from fluoroflow.exceptions import ValidationError


class TestAsSeries:
    def test_coerces_list_to_float64(self) -> None:
        out = as_series([1, 2, 3], label="x")
        assert out.dtype == np.float64
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])

    def test_result_is_read_only(self) -> None:
        assert not as_series([1.0, 2.0], label="x").flags.writeable

    def test_does_not_alias_the_callers_array(self) -> None:
        source = np.array([1.0, 2.0, 3.0])
        out = as_series(source, label="x")
        source[0] = 99.0
        assert out[0] == 1.0

    def test_does_not_freeze_the_callers_array(self) -> None:
        source = np.array([1.0, 2.0, 3.0])
        as_series(source, label="x")
        assert source.flags.writeable, "coercion must not make the caller's array read-only"

    def test_passes_through_a_read_only_float_array_without_copying(self) -> None:
        source = np.array([1.0, 2.0, 3.0])
        source.flags.writeable = False
        assert as_series(source, label="x") is source

    def test_rejects_two_dimensional_input(self) -> None:
        with pytest.raises(ValidationError, match="one-dimensional"):
            as_series(np.zeros((2, 3)), label="x")

    def test_rejects_non_numeric_input(self) -> None:
        with pytest.raises(ValidationError, match="numeric"):
            as_series(["a", "b"], label="x")

    def test_accepts_empty(self) -> None:
        assert as_series([], label="x").size == 0


class TestCheckTimeVector:
    def test_accepts_strictly_increasing(self) -> None:
        check_time_vector(as_series([0.0, 0.1, 0.2], label="t"))

    @pytest.mark.parametrize("short", [[], [1.0]])
    def test_accepts_degenerate_lengths(self, short: list[float]) -> None:
        check_time_vector(as_series(short, label="t"))

    def test_rejects_nan(self) -> None:
        with pytest.raises(ValidationError, match="finite"):
            check_time_vector(as_series([0.0, np.nan, 0.2], label="t"))

    def test_rejects_infinity(self) -> None:
        with pytest.raises(ValidationError, match="finite"):
            check_time_vector(as_series([0.0, np.inf], label="t"))

    def test_rejects_duplicate_timestamps(self) -> None:
        with pytest.raises(ValidationError, match="strictly increasing"):
            check_time_vector(as_series([0.0, 0.1, 0.1, 0.2], label="t"))

    def test_rejects_decreasing(self) -> None:
        with pytest.raises(ValidationError, match="strictly increasing"):
            check_time_vector(as_series([0.0, 0.2, 0.1], label="t"))

    def test_error_points_at_the_first_offending_index(self) -> None:
        with pytest.raises(ValidationError, match=r"time\[3\]"):
            check_time_vector(as_series([0.0, 0.1, 0.2, 0.15, 0.3], label="t"))


class TestCheckNoInfinities:
    def test_allows_nan_because_dropped_frames_are_real(self) -> None:
        check_no_infinities(as_series([1.0, np.nan, 3.0], label="v"))

    @pytest.mark.parametrize("bad", [np.inf, -np.inf])
    def test_rejects_infinity(self, bad: float) -> None:
        with pytest.raises(ValidationError, match="infinite"):
            check_no_infinities(as_series([1.0, bad], label="v"))

    def test_accepts_empty(self) -> None:
        check_no_infinities(as_series([], label="v"))


class TestCheckMatchingLength:
    def test_accepts_equal_lengths(self) -> None:
        check_matching_length(np.zeros(3), np.ones(3), labels=("a", "b"))

    def test_rejects_unequal_lengths_and_reports_both(self) -> None:
        with pytest.raises(ValidationError, match="got 3 and 4"):
            check_matching_length(np.zeros(3), np.ones(4), labels=("a", "b"))


class TestCheckedName:
    def test_strips_surrounding_whitespace(self) -> None:
        assert checked_name("  Region0G ") == "Region0G"

    @pytest.mark.parametrize("bad", ["", "   ", None, 3])
    def test_rejects_non_strings_and_blanks(self, bad: object) -> None:
        with pytest.raises(ValidationError, match="non-empty string"):
            checked_name(bad)


class TestCheckPercentage:
    @pytest.mark.parametrize("good", [0.0, 0.08, 8.0, 50, 100.0])
    def test_accepts_the_closed_unit_percent_interval(self, good: float) -> None:
        assert check_percentage(good) == pytest.approx(float(good))

    @pytest.mark.parametrize("bad", [-0.1, 100.1, 101, 1000])
    def test_rejects_values_outside_the_interval(self, bad: float) -> None:
        with pytest.raises(ValidationError, match=r"\[0, 100\]"):
            check_percentage(bad)

    def test_rejects_nan(self) -> None:
        with pytest.raises(ValidationError, match="finite"):
            check_percentage(math.nan)

    def test_error_message_teaches_the_convention(self) -> None:
        # The whole point of this guard: someone reaching for a fraction should be
        # told, in the traceback, that percentiles here are in percent.
        with pytest.raises(ValidationError, match=r"pass 8, not 0\.08"):
            check_percentage(800.0)


class TestCheckPositive:
    @pytest.mark.parametrize("good", [1e-12, 1.0, 30.0])
    def test_accepts_positive_finite(self, good: float) -> None:
        assert check_positive(good, label="fs") == pytest.approx(good)

    @pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
    def test_rejects_non_positive_or_non_finite(self, bad: float) -> None:
        with pytest.raises(ValidationError, match="greater than zero"):
            check_positive(bad, label="fs")


class TestMedianDt:
    @pytest.mark.parametrize("short", [[], [1.0]])
    def test_returns_nan_when_no_interval_exists(self, short: list[float]) -> None:
        assert math.isnan(median_dt(as_series(short, label="t")))

    def test_uniform_sampling(self) -> None:
        assert median_dt(as_series(np.arange(100) / 30.0, label="t")) == pytest.approx(1 / 30.0)

    def test_a_single_dropped_frame_does_not_move_the_estimate(self) -> None:
        time = np.delete(np.arange(100) / 30.0, 50)
        assert median_dt(as_series(time, label="t")) == pytest.approx(1 / 30.0)

    def test_second_difference_would_be_wrong_where_the_median_is_right(self) -> None:
        # Regression guard for the RamiPho bug: taking `np.diff(t)[1]` off an
        # interleaved timestamp column measures the inter-LED gap, so the reported
        # rate came out ~3x too high for a 3-LED acquisition. Estimating from a
        # de-interleaved channel's own timestamps is what makes it correct.
        per_channel_fs = 30.0
        n_leds = 3
        interleaved = np.arange(300) / (per_channel_fs * n_leds)
        demuxed = interleaved[::n_leds]

        naive_fs_from_interleaved = 1.0 / float(np.diff(interleaved)[1])
        assert naive_fs_from_interleaved == pytest.approx(per_channel_fs * n_leds)

        assert 1.0 / median_dt(as_series(demuxed, label="t")) == pytest.approx(per_channel_fs)
