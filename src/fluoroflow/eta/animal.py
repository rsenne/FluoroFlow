"""Per-animal event-triggered averages with a parametric or bootstrap confidence band."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from fluoroflow.core.events import Events
from fluoroflow.core.trace import Trace
from fluoroflow.eta.alignment import align_to_events
from fluoroflow.exceptions import InsufficientSamplesError

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["AnimalETA", "animal_eta"]


@dataclass(frozen=True, slots=True)
class AnimalETA:
    """A single animal's trial-averaged event-triggered average."""

    name: str
    time: NDArray[np.float64]
    mean: NDArray[np.float64]
    sem: NDArray[np.float64]
    ci_lower: NDArray[np.float64] | None
    ci_upper: NDArray[np.float64] | None
    n_trials: int
    n_dropped: int
    method: Literal["t", "bootstrap"] | None
    confidence: float | None

    def to_frame(self) -> pd.DataFrame:
        """Return the average as a :class:`pandas.DataFrame`."""
        import pandas as pd

        n = self.time.size
        ci_lower = np.full(n, np.nan) if self.ci_lower is None else self.ci_lower.copy()
        ci_upper = np.full(n, np.nan) if self.ci_upper is None else self.ci_upper.copy()
        return pd.DataFrame(
            {
                "time": self.time.copy(),
                "mean": self.mean.copy(),
                "sem": self.sem.copy(),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        )


def animal_eta(
    trace: Trace,
    events: Events,
    window: tuple[float, float],
    *,
    ci: Literal["t", "bootstrap"] | None = "t",
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int | np.random.Generator | None = None,
    name: str | None = None,
) -> AnimalETA:
    """Average single-trial windows aligned to ``events`` into one animal-level ETA."""
    relative_time, trials, n_dropped = align_to_events(trace, events, window)
    n_valid = trials.shape[0]
    if n_valid < 2:
        msg = (
            f"Trace {trace.name!r} has {n_valid} surviving trial(s) after windowing "
            f"(dropped {n_dropped}); at least 2 are needed to estimate trial-to-trial "
            f"variability."
        )
        raise InsufficientSamplesError(msg)

    mean = trials.mean(axis=0)
    sem = trials.std(axis=0, ddof=1) / np.sqrt(n_valid)

    ci_lower: NDArray[np.float64] | None
    ci_upper: NDArray[np.float64] | None
    if ci == "t":
        t_crit = float(stats.t.ppf(0.5 + confidence / 2.0, df=n_valid - 1))
        ci_lower = mean - t_crit * sem
        ci_upper = mean + t_crit * sem
    elif ci == "bootstrap":
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, n_valid, size=(n_boot, n_valid))
        boot_means = trials[draws].mean(axis=1)
        alpha = (1.0 - confidence) / 2.0
        ci_lower = np.quantile(boot_means, alpha, axis=0)
        ci_upper = np.quantile(boot_means, 1.0 - alpha, axis=0)
    else:
        ci_lower = None
        ci_upper = None

    return AnimalETA(
        name=trace.name if name is None else name,
        time=relative_time,
        mean=mean,
        sem=sem,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_trials=n_valid,
        n_dropped=n_dropped,
        method=ci,
        confidence=confidence if ci is not None else None,
    )
