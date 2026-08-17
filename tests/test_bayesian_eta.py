"""Tests for bayesian_eta and BayesianETA."""

from __future__ import annotations

import numpy as np
import pytest

from fluoroflow.eta import AnimalETA, bayesian_eta
from fluoroflow.exceptions import InsufficientSamplesError, ValidationError


def make_animal(name: str, mean: list[float], sem: list[float], *, time: np.ndarray) -> AnimalETA:
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


class TestBayesianEta:
    def test_agreeing_animals_get_heavy_shrinkage_and_near_zero_tau2(self) -> None:
        time = np.array([0.0, 1.0])
        animals = [
            make_animal("a1", [1.00, 2.00], [0.2, 0.2], time=time),
            make_animal("a2", [1.01, 1.99], [0.2, 0.2], time=time),
            make_animal("a3", [0.99, 2.01], [0.2, 0.2], time=time),
            make_animal("a4", [1.00, 2.00], [0.2, 0.2], time=time),
        ]
        result = bayesian_eta(animals)

        np.testing.assert_allclose(result.tau2, 0.0, atol=1e-8)
        # Full shrinkage: every animal's posterior mean collapses onto the population mean.
        for i in range(len(animals)):
            np.testing.assert_allclose(result.animal_means[i], result.population_mean, atol=1e-8)

    def test_disagreeing_animals_get_light_shrinkage_and_positive_tau2(self) -> None:
        time = np.array([0.0])
        animals = [
            make_animal("a1", [0.0], [0.1], time=time),
            make_animal("a2", [10.0], [0.1], time=time),
        ]
        result = bayesian_eta(animals)

        assert np.all(result.tau2 > 0.0)
        theta = np.array([0.0, 10.0])
        # Light shrinkage: each animal's posterior mean stays close to its own raw mean,
        # much closer than it is to the (fully-pooled) population mean.
        for i in range(len(animals)):
            dist_to_own = abs(float(result.animal_means[i, 0]) - theta[i])
            dist_to_pop = abs(theta[i] - float(result.population_mean[0]))
            assert dist_to_own < 0.1 * dist_to_pop

    def test_population_mean_matches_hand_computed_two_animal_case(self) -> None:
        time = np.array([0.0])
        theta = np.array([1.0, 3.0])
        v = np.array([0.04, 0.09])
        animals = [
            make_animal("a1", [float(theta[0])], [float(np.sqrt(v[0]))], time=time),
            make_animal("a2", [float(theta[1])], [float(np.sqrt(v[1]))], time=time),
        ]
        result = bayesian_eta(animals, confidence=0.95)

        w = 1.0 / v
        theta_fe = np.sum(w * theta) / np.sum(w)
        q = np.sum(w * (theta - theta_fe) ** 2)
        df = 1
        c = np.sum(w) - np.sum(w**2) / np.sum(w)
        tau2_expected = max(float((q - df) / c), 0.0)

        w_star = 1.0 / (v + tau2_expected)
        expected_population_mean = np.sum(w_star * theta) / np.sum(w_star)
        expected_var_re = 1.0 / np.sum(w_star)

        np.testing.assert_allclose(result.tau2[0], tau2_expected)
        np.testing.assert_allclose(result.population_mean[0], expected_population_mean)
        np.testing.assert_allclose(
            result.population_ci_upper[0] - result.population_mean[0],
            result.population_mean[0] - result.population_ci_lower[0],
        )
        assert result.population_ci_upper[0] - result.population_mean[0] > 0.0
        assert np.sqrt(expected_var_re) > 0.0

    def test_mismatched_time_raises(self) -> None:
        animals = [
            make_animal("a1", [1.0, 2.0], [0.1, 0.1], time=np.array([0.0, 1.0])),
            make_animal("a2", [2.0, 3.0], [0.2, 0.2], time=np.array([0.0, 2.0])),
        ]
        with pytest.raises(ValidationError, match="a2"):
            bayesian_eta(animals)

    def test_zero_sem_raises(self) -> None:
        time = np.array([0.0, 1.0])
        animals = [
            make_animal("a1", [1.0, 2.0], [0.0, 0.2], time=time),
            make_animal("a2", [1.1, 2.1], [0.1, 0.2], time=time),
        ]
        with pytest.raises(ValidationError, match="nonzero within-animal variance"):
            bayesian_eta(animals)

    def test_fewer_than_two_animals_raises(self) -> None:
        animals = [make_animal("a1", [1.0, 2.0], [0.1, 0.1], time=np.array([0.0, 1.0]))]
        with pytest.raises(InsufficientSamplesError, match="at least 2"):
            bayesian_eta(animals)

    def test_to_frame_columns(self) -> None:
        time = np.array([0.0, 1.0])
        animals = [
            make_animal("a1", [1.0, 2.0], [0.1, 0.1], time=time),
            make_animal("a2", [2.0, 3.0], [0.2, 0.2], time=time),
        ]
        result = bayesian_eta(animals)
        frame = result.to_frame()
        assert list(frame.columns) == [
            "time",
            "population_mean",
            "population_ci_lower",
            "population_ci_upper",
            "tau2",
        ]
        assert len(frame) == time.size
