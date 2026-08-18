"""Tests for comparing an ETA's confidence band against a null."""

from __future__ import annotations

import numpy as np
import pytest

from fluoroflow.eta import AnimalETA, Significance, compare_to_null, resolve_null
from fluoroflow.eta.inference import _ESTIMATORS
from fluoroflow.exceptions import ValidationError

# 10 Hz over [-1, 1) s, so the event lands exactly on index 10.
TIME = np.arange(-10, 10) / 10.0


def band(
    mean: np.ndarray, half_width: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A mean with a symmetric interval of the given half-width."""
    return mean, mean - half_width, mean + half_width


def make_animal(
    mean: np.ndarray,
    half_width: float | np.ndarray,
    *,
    name: str = "a1",
    with_ci: bool = True,
) -> AnimalETA:
    values, lower, upper = band(mean, half_width)
    return AnimalETA(
        name=name,
        time=TIME,
        mean=values,
        sem=np.full(TIME.size, 0.1),
        ci_lower=lower if with_ci else None,
        ci_upper=upper if with_ci else None,
        n_trials=10,
        n_dropped=0,
        method="t" if with_ci else None,
        confidence=0.95 if with_ci else None,
    )


class TestResolveNull:
    def test_a_number_passes_through(self) -> None:
        level, method = resolve_null(TIME, np.ones(TIME.size), 2.5)
        assert level == 2.5
        assert method == "fixed"

    def test_zero_keyword_matches_the_numeric_default(self) -> None:
        values = np.ones(TIME.size)
        assert resolve_null(TIME, values, "zero") == (0.0, "zero")
        assert resolve_null(TIME, values, 0.0)[0] == 0.0

    def test_mean_and_median_use_the_pre_event_window_by_default(self) -> None:
        values = np.where(TIME < 0.0, 3.0, 100.0)
        values[0] = 13.0  # An outlier the mean feels and the median does not.
        mean_level, mean_method = resolve_null(TIME, values, "mean")
        assert mean_level == pytest.approx(4.0)
        assert mean_method == "mean"
        assert resolve_null(TIME, values, "median") == (3.0, "median")

    def test_an_explicit_baseline_window_is_honoured(self) -> None:
        values = np.where(TIME < -0.5, 7.0, 1.0)
        level, method = resolve_null(TIME, values, "mean", baseline=(None, -0.5))
        assert level == 7.0
        assert method == "mean"

    def test_half_open_baseline_excludes_its_upper_edge(self) -> None:
        values = np.arange(TIME.size, dtype=float)
        level, _ = resolve_null(TIME, values, "mean", baseline=(-0.2, 0.0))
        # Only t = -0.2 and t = -0.1, i.e. indices 8 and 9.
        assert level == pytest.approx(8.5)

    def test_empty_baseline_window_raises(self) -> None:
        with pytest.raises(ValidationError, match="cannot be estimated"):
            resolve_null(TIME, np.ones(TIME.size), "mean", baseline=(5.0, 6.0))

    def test_inverted_baseline_window_raises(self) -> None:
        with pytest.raises(ValidationError, match="must be less than"):
            resolve_null(TIME, np.ones(TIME.size), "mean", baseline=(0.5, -0.5))

    def test_unknown_keyword_raises_and_lists_the_options(self) -> None:
        with pytest.raises(ValidationError, match="zero"):
            resolve_null(TIME, np.ones(TIME.size), "mode")

    def test_non_finite_fixed_null_raises(self) -> None:
        with pytest.raises(ValidationError, match="must be finite"):
            resolve_null(TIME, np.ones(TIME.size), np.inf)

    def test_non_numeric_null_raises(self) -> None:
        with pytest.raises(ValidationError, match="must be a number"):
            resolve_null(TIME, np.ones(TIME.size), None)


class TestCompareToNull:
    def test_marks_only_where_the_interval_clears_the_null(self) -> None:
        mean = np.where(TIME >= 0.0, 1.0, 0.0)
        result = compare_to_null(*(TIME, *band(mean, 0.5)))
        np.testing.assert_array_equal(result.mask, TIME >= 0.0)
        assert result.n_significant == 10

    def test_a_band_that_touches_the_null_is_not_significant(self) -> None:
        mean = np.full(TIME.size, 1.0)
        # Lower limit lands exactly on zero: the interval includes the null.
        result = compare_to_null(*(TIME, *band(mean, 1.0)))
        assert result.n_significant == 0

    def test_direction_separates_increases_from_decreases(self) -> None:
        mean = np.where(TIME >= 0.0, 1.0, -1.0)
        result = compare_to_null(*(TIME, *band(mean, 0.25)))
        np.testing.assert_array_equal(result.direction[TIME >= 0.0], 1)
        np.testing.assert_array_equal(result.direction[TIME < 0.0], -1)

    def test_direction_is_zero_where_the_interval_spans_the_null(self) -> None:
        mean = np.zeros(TIME.size)
        result = compare_to_null(*(TIME, *band(mean, 1.0)))
        np.testing.assert_array_equal(result.direction, 0)
        assert result.n_epochs == 0
        assert result.total_duration == 0.0
        assert result.first_crossing is None

    def test_epochs_capture_contiguous_runs(self) -> None:
        mean = np.zeros(TIME.size)
        mean[2:5] = 1.0
        mean[12:16] = 1.0
        result = compare_to_null(*(TIME, *band(mean, 0.25)))
        assert result.n_epochs == 2
        np.testing.assert_allclose(result.epochs.times, [TIME[2], TIME[12]])
        np.testing.assert_allclose(result.epochs.durations, [0.3, 0.4])
        assert result.first_crossing == pytest.approx(TIME[2])
        assert result.total_duration == pytest.approx(0.7)

    def test_min_duration_drops_short_runs_from_mask_and_epochs(self) -> None:
        mean = np.zeros(TIME.size)
        mean[2:4] = 1.0  # 0.2 s -- too short
        mean[12:18] = 1.0  # 0.6 s -- long enough
        result = compare_to_null(*(TIME, *band(mean, 0.25)), min_duration=0.5)
        assert result.n_epochs == 1
        assert result.n_significant == 6
        np.testing.assert_array_equal(np.flatnonzero(result.mask), np.arange(12, 18))
        np.testing.assert_array_equal(result.direction[2:4], 0)

    def test_min_duration_boundary_is_inclusive(self) -> None:
        mean = np.zeros(TIME.size)
        mean[4:9] = 1.0  # exactly 0.5 s
        result = compare_to_null(*(TIME, *band(mean, 0.25)), min_duration=0.5)
        assert result.n_epochs == 1

    def test_min_duration_can_reject_everything(self) -> None:
        mean = np.zeros(TIME.size)
        mean[4:6] = 1.0
        result = compare_to_null(*(TIME, *band(mean, 0.25)), min_duration=5.0)
        assert result.n_epochs == 0
        assert result.n_significant == 0

    def test_negative_min_duration_raises(self) -> None:
        mean = np.ones(TIME.size)
        with pytest.raises(ValidationError, match="non-negative"):
            compare_to_null(*(TIME, *band(mean, 0.25)), min_duration=-1.0)

    def test_an_estimated_null_shifts_the_verdict(self) -> None:
        # A trace sitting at 1.0 throughout: significant against zero, but not
        # against its own pre-event baseline.
        mean = np.full(TIME.size, 1.0)
        against_zero = compare_to_null(*(TIME, *band(mean, 0.25)))
        against_baseline = compare_to_null(*(TIME, *band(mean, 0.25)), null="median")
        assert against_zero.n_significant == TIME.size
        assert against_baseline.n_significant == 0
        assert against_baseline.null == 1.0

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValidationError, match="same length"):
            compare_to_null(TIME, np.ones(3), np.zeros(3), np.ones(3))

    def test_results_are_read_only(self) -> None:
        result = compare_to_null(*(TIME, *band(np.ones(TIME.size), 0.25)))
        with pytest.raises(ValueError, match="read-only"):
            result.mask[0] = False
        with pytest.raises(ValueError, match="read-only"):
            result.direction[0] = 0

    def test_to_frame_columns(self) -> None:
        result = compare_to_null(*(TIME, *band(np.ones(TIME.size), 0.25)))
        frame = result.to_frame()
        assert list(frame.columns) == ["time", "significant", "direction"]
        assert len(frame) == TIME.size

    def test_repr_summarises_the_verdict(self) -> None:
        result = compare_to_null(*(TIME, *band(np.ones(TIME.size), 0.25)))
        assert "Significance(null=0 (fixed)" in repr(result)
        assert "epochs=1" in repr(result)

    def test_every_estimator_is_reachable_by_name(self) -> None:
        mean = np.full(TIME.size, 2.0)
        for keyword in _ESTIMATORS:
            result = compare_to_null(*(TIME, *band(mean, 0.25)), null=keyword)
            assert result.null_method == keyword
            assert result.null == 2.0


class TestEtaSignificanceMethods:
    def test_animal_eta_reports_its_own_band(self) -> None:
        mean = np.where(TIME >= 0.0, 1.0, 0.0)
        result = make_animal(mean, 0.25).significance()
        assert isinstance(result, Significance)
        np.testing.assert_array_equal(result.mask, TIME >= 0.0)
        assert result.confidence == 0.95
        assert result.epochs.name == "a1_significant"

    def test_animal_eta_without_a_band_raises(self) -> None:
        animal = make_animal(np.ones(TIME.size), 0.25, with_ci=False)
        with pytest.raises(ValidationError, match="ci=None"):
            animal.significance()

    def test_population_eta_reports_the_across_animal_band(self) -> None:
        from fluoroflow.eta import population_eta

        animals = [
            make_animal(np.where(TIME >= 0.0, 1.0, 0.0), 0.25, name="a1"),
            make_animal(np.where(TIME >= 0.0, 1.1, 0.0), 0.25, name="a2"),
            make_animal(np.where(TIME >= 0.0, 0.9, 0.0), 0.25, name="a3"),
        ]
        result = population_eta(animals, ci="t").significance()
        np.testing.assert_array_equal(result.mask, TIME >= 0.0)
        assert result.epochs.name == "population_significant"

    def test_bayesian_eta_reports_the_random_effects_band(self) -> None:
        from fluoroflow.eta import bayesian_eta

        animals = [
            make_animal(np.where(TIME >= 0.0, 1.0, 0.0), 0.25, name="a1"),
            make_animal(np.where(TIME >= 0.0, 1.1, 0.0), 0.25, name="a2"),
            make_animal(np.where(TIME >= 0.0, 0.9, 0.0), 0.25, name="a3"),
        ]
        result = bayesian_eta(animals).significance()
        np.testing.assert_array_equal(result.mask, TIME >= 0.0)
        assert result.confidence == 0.95

    def test_null_and_min_duration_reach_through_the_methods(self) -> None:
        mean = np.where(TIME >= 0.0, 1.0, 0.5)
        animal = make_animal(mean, 0.1)
        assert animal.significance().n_significant == TIME.size
        against_baseline = animal.significance(null="mean", min_duration=0.5)
        assert against_baseline.null == 0.5
        np.testing.assert_array_equal(against_baseline.mask, TIME >= 0.0)
