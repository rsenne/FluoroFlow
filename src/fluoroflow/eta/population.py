"""Across-animal event-triggered averages, with animals as the resampling unit."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from fluoroflow.eta.animal import AnimalETA
from fluoroflow.exceptions import InsufficientSamplesError, ValidationError

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["PopulationETA", "population_eta"]


@dataclass(frozen=True, slots=True)
class PopulationETA:
    """An across-animal event-triggered average."""

    time: NDArray[np.float64]
    mean: NDArray[np.float64]
    sem: NDArray[np.float64]
    ci_lower: NDArray[np.float64]
    ci_upper: NDArray[np.float64]
    n_animals: int
    method: Literal["t", "bootstrap"]
    confidence: float

    def to_frame(self) -> pd.DataFrame:
        """Return the average as a :class:`pandas.DataFrame`."""
        import pandas as pd

        return pd.DataFrame(
            {
                "time": self.time.copy(),
                "mean": self.mean.copy(),
                "sem": self.sem.copy(),
                "ci_lower": self.ci_lower.copy(),
                "ci_upper": self.ci_upper.copy(),
            }
        )


def check_matching_time(animals: Sequence[AnimalETA]) -> NDArray[np.float64]:
    """Validate that every animal shares the same alignment window, and return it."""
    reference = animals[0].time
    for animal in animals[1:]:
        if not np.array_equal(animal.time, reference):
            msg = (
                f"All animals must share the same alignment window (time base); "
                f"{animal.name!r} does not match {animals[0].name!r}."
            )
            raise ValidationError(msg)
    return reference


def population_eta(
    animals: Sequence[AnimalETA],
    *,
    ci: Literal["t", "bootstrap"] = "t",
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int | np.random.Generator | None = None,
) -> PopulationETA:
    """Average per-animal ETAs into a population-level ETA, with animals as the unit."""
    n_animals = len(animals)
    if n_animals < 2:
        msg = f"population_eta needs at least 2 animals, got {n_animals}."
        raise InsufficientSamplesError(msg)

    time = check_matching_time(animals)
    stacked = np.stack([animal.mean for animal in animals])

    mean = stacked.mean(axis=0)
    sem = stacked.std(axis=0, ddof=1) / np.sqrt(n_animals)

    ci_lower: NDArray[np.float64]
    ci_upper: NDArray[np.float64]
    if ci == "t":
        t_crit = float(stats.t.ppf(0.5 + confidence / 2.0, df=n_animals - 1))
        ci_lower = mean - t_crit * sem
        ci_upper = mean + t_crit * sem
    else:
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, n_animals, size=(n_boot, n_animals))
        boot_means = stacked[draws].mean(axis=1)
        alpha = (1.0 - confidence) / 2.0
        ci_lower = np.quantile(boot_means, alpha, axis=0)
        ci_upper = np.quantile(boot_means, 1.0 - alpha, axis=0)

    return PopulationETA(
        time=time,
        mean=mean,
        sem=sem,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_animals=n_animals,
        method=ci,
        confidence=confidence,
    )
