"""Random-effects (DerSimonian-Laird) hierarchical event-triggered average.

Each timepoint is fit independently (vectorized across time) as a Normal-Normal
random-effects meta-analysis over animals: a closed-form posterior conditional on
a plug-in between-animal variance (tau-squared), estimated by the method of
moments. There is no MCMC here, only per-timepoint closed-form algebra.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from fluoroflow.eta.animal import AnimalETA
from fluoroflow.eta.population import check_matching_time
from fluoroflow.exceptions import InsufficientSamplesError, ValidationError

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["BayesianETA", "bayesian_eta"]


@dataclass(frozen=True, slots=True)
class BayesianETA:
    """Population and per-animal posteriors from a random-effects meta-analysis."""

    time: NDArray[np.float64]
    population_mean: NDArray[np.float64]
    population_ci_lower: NDArray[np.float64]
    population_ci_upper: NDArray[np.float64]
    animal_names: tuple[str, ...]
    animal_means: NDArray[np.float64]
    animal_ci_lower: NDArray[np.float64]
    animal_ci_upper: NDArray[np.float64]
    tau2: NDArray[np.float64]
    n_animals: int
    confidence: float

    def to_frame(self) -> pd.DataFrame:
        """Return the population-level posterior as a :class:`pandas.DataFrame`."""
        import pandas as pd

        return pd.DataFrame(
            {
                "time": self.time.copy(),
                "population_mean": self.population_mean.copy(),
                "population_ci_lower": self.population_ci_lower.copy(),
                "population_ci_upper": self.population_ci_upper.copy(),
                "tau2": self.tau2.copy(),
            }
        )


def bayesian_eta(animals: Sequence[AnimalETA], *, confidence: float = 0.95) -> BayesianETA:
    """Fit a per-timepoint DerSimonian-Laird random-effects model across animals."""
    n_animals = len(animals)
    if n_animals < 2:
        msg = f"bayesian_eta needs at least 2 animals, got {n_animals}."
        raise InsufficientSamplesError(msg)

    time = check_matching_time(animals)
    theta = np.stack([animal.mean for animal in animals])
    v = np.stack([animal.sem**2 for animal in animals])
    if np.any(v == 0.0):
        msg = (
            "bayesian_eta requires nonzero within-animal variance (sem > 0) at every "
            "timepoint for every animal; a zero SEM means every trial for that animal "
            "was identical at that instant, which the inverse-variance weighting can't use."
        )
        raise ValidationError(msg)

    w = 1.0 / v
    sum_w = w.sum(axis=0)
    theta_fe = (w * theta).sum(axis=0) / sum_w
    q = (w * (theta - theta_fe) ** 2).sum(axis=0)
    df = n_animals - 1
    c = sum_w - (w**2).sum(axis=0) / sum_w
    with np.errstate(divide="ignore", invalid="ignore"):
        tau2_raw = (q - df) / c
    tau2: NDArray[np.float64] = np.where(c == 0.0, 0.0, np.clip(tau2_raw, a_min=0.0, a_max=None))

    w_star = 1.0 / (v + tau2)
    sum_w_star = w_star.sum(axis=0)
    population_mean = (w_star * theta).sum(axis=0) / sum_w_star
    var_re = 1.0 / sum_w_star

    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    population_ci_lower = population_mean - z * np.sqrt(var_re)
    population_ci_upper = population_mean + z * np.sqrt(var_re)

    no_heterogeneity = tau2 == 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        shrinkage = np.where(no_heterogeneity, 0.0, tau2 / (tau2 + v))
        var_shrunk = np.where(no_heterogeneity, var_re, 1.0 / (1.0 / v + 1.0 / tau2))
    animal_means: NDArray[np.float64] = np.where(
        no_heterogeneity, population_mean, population_mean + shrinkage * (theta - population_mean)
    )

    animal_ci_lower = animal_means - z * np.sqrt(var_shrunk)
    animal_ci_upper = animal_means + z * np.sqrt(var_shrunk)

    return BayesianETA(
        time=time,
        population_mean=population_mean,
        population_ci_lower=population_ci_lower,
        population_ci_upper=population_ci_upper,
        animal_names=tuple(animal.name for animal in animals),
        animal_means=animal_means,
        animal_ci_lower=animal_ci_lower,
        animal_ci_upper=animal_ci_upper,
        tau2=tau2,
        n_animals=n_animals,
        confidence=confidence,
    )
