"""Tests for population_eta and PopulationETA."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from fluoroflow.eta import AnimalETA, population_eta
from fluoroflow.exceptions import InsufficientSamplesError, ValidationError

TIME = np.array([0.0, 1.0, 2.0])


def make_animal(
    name: str, mean: list[float], sem: list[float], *, time: np.ndarray = TIME
) -> AnimalETA:
    return AnimalETA(
        name=name,
        time=time,
        mean=np.array(mean),
        sem=np.array(sem),
        ci_lower=None,
        ci_upper=None,
        n_trials=10,
        n_dropped=0,
        method=None,
        confidence=None,
    )


class TestPopulationEta:
    def test_mean_and_sem_match_manual_across_animal_computation(self) -> None:
        animals = [
            make_animal("a1", [1.0, 2.0, 3.0], [0.1, 0.1, 0.1]),
            make_animal("a2", [2.0, 3.0, 4.0], [0.2, 0.2, 0.2]),
            make_animal("a3", [1.5, 2.5, 3.5], [0.15, 0.15, 0.15]),
        ]
        result = population_eta(animals, ci="t")

        stacked = np.stack([a.mean for a in animals])
        expected_mean = stacked.mean(axis=0)
        expected_sem = stacked.std(axis=0, ddof=1) / np.sqrt(3)

        np.testing.assert_allclose(result.mean, expected_mean)
        np.testing.assert_allclose(result.sem, expected_sem)
        assert result.n_animals == 3
        np.testing.assert_array_equal(result.time, TIME)

    def test_t_interval_matches_manual_formula(self) -> None:
        animals = [
            make_animal("a1", [1.0, 2.0, 3.0], [0.1, 0.1, 0.1]),
            make_animal("a2", [2.0, 3.0, 4.0], [0.2, 0.2, 0.2]),
            make_animal("a3", [1.5, 2.5, 3.5], [0.15, 0.15, 0.15]),
        ]
        result = population_eta(animals, ci="t", confidence=0.9)
        t_crit = stats.t.ppf(0.95, df=2)
        np.testing.assert_allclose(result.ci_lower, result.mean - t_crit * result.sem)
        np.testing.assert_allclose(result.ci_upper, result.mean + t_crit * result.sem)

    def test_bootstrap_is_reproducible_with_a_fixed_seed(self) -> None:
        animals = [
            make_animal("a1", [1.0, 2.0], [0.1, 0.1]),
            make_animal("a2", [2.0, 3.0], [0.2, 0.2]),
            make_animal("a3", [1.5, 2.5], [0.15, 0.15]),
            make_animal("a4", [1.2, 2.2], [0.12, 0.12]),
        ]
        first = population_eta(animals, ci="bootstrap", seed=7, n_boot=500)
        second = population_eta(animals, ci="bootstrap", seed=7, n_boot=500)
        np.testing.assert_array_equal(first.ci_lower, second.ci_lower)
        np.testing.assert_array_equal(first.ci_upper, second.ci_upper)

    def test_mismatched_time_raises(self) -> None:
        animals = [
            make_animal("a1", [1.0, 2.0, 3.0], [0.1, 0.1, 0.1]),
            make_animal("a2", [2.0, 3.0, 4.0], [0.2, 0.2, 0.2], time=np.array([0.0, 1.0, 3.0])),
        ]
        with pytest.raises(ValidationError, match="a2"):
            population_eta(animals)

    def test_fewer_than_two_animals_raises(self) -> None:
        animals = [make_animal("a1", [1.0, 2.0, 3.0], [0.1, 0.1, 0.1])]
        with pytest.raises(InsufficientSamplesError, match="at least 2"):
            population_eta(animals)

    def test_to_frame_columns(self) -> None:
        animals = [
            make_animal("a1", [1.0, 2.0, 3.0], [0.1, 0.1, 0.1]),
            make_animal("a2", [2.0, 3.0, 4.0], [0.2, 0.2, 0.2]),
        ]
        result = population_eta(animals)
        frame = result.to_frame()
        assert list(frame.columns) == ["time", "mean", "sem", "ci_lower", "ci_upper"]
        assert len(frame) == TIME.size
