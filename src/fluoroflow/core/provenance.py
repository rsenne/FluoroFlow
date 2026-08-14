"""Processing provenance: what was done to a signal, in order, with what parameters.

Every FluoroFlow transform appends a :class:`Step` to its output's history. This
is not bookkeeping for its own sake; downstream operations *read* the history
to refuse unsafe combinations. See :data:`MEAN_REMOVED` for the motivating case.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

__all__ = [
    "BASELINE_CORRECTED",
    "FILTERED",
    "MEAN_REMOVED",
    "MOTION_CORRECTED",
    "NORMALIZED",
    "RESAMPLED",
    "Step",
]

#: The step removed the signal's absolute offset (detrending, high-pass
#: filtering, regression residuals). Dividing such a signal by its own mean
#: (the classic broken ``dF/F``) is numerically meaningless, so
#: :func:`fluoroflow.preprocess.dff` refuses to run on a trace carrying this tag.
MEAN_REMOVED = "mean-removed"

#: The step estimated and removed a slow baseline (airPLS, asymmetric least
#: squares, biexponential bleach fit).
BASELINE_CORRECTED = "baseline-corrected"

#: The step applied a frequency-domain filter.
FILTERED = "filtered"

#: The step regressed out a movement reference (isosbestic, tdTomato).
MOTION_CORRECTED = "motion-corrected"

#: The step converted the signal to a relative or standardised scale.
NORMALIZED = "normalized"

#: The step changed the time base.
RESAMPLED = "resampled"


@dataclass(frozen=True, slots=True, eq=False)
class Step:
    """One recorded operation in a signal's processing history.

    Parameters
    ----------
    name
        Short identifier for the operation, e.g. ``"airpls"``. By convention this
        matches the name of the function that produced it.
    params
        The parameters the operation actually ran with, including defaults that
        were filled in. Copied defensively and exposed read-only, so a step can
        never be edited after the fact.
    tags
        Semantic markers other operations can query. Prefer the module-level
        constants such as :data:`MEAN_REMOVED` over bare strings.

    Notes
    -----
    :class:`Step` is unhashable by design: it compares by value, and its
    ``params`` mapping has no meaningful hash.
    """

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    tags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        """Freeze ``params`` and ``tags`` against later mutation."""
        if not isinstance(self.name, str) or not self.name.strip():
            msg = f"Step name must be a non-empty string, got {self.name!r}."
            raise ValueError(msg)
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))
        object.__setattr__(self, "tags", frozenset(self.tags))

    def __eq__(self, other: object) -> bool:
        """Compare by name, parameters, and tags."""
        if not isinstance(other, Step):
            return NotImplemented
        return (
            self.name == other.name
            and dict(self.params) == dict(other.params)
            and self.tags == other.tags
        )

    def __repr__(self) -> str:
        """Render as ``Step(name, k=v, ...)``, tags included when present."""
        parts = [repr(self.name)]
        parts += [f"{k}={v!r}" for k, v in self.params.items()]
        if self.tags:
            parts.append(f"tags={sorted(self.tags)!r}")
        return f"Step({', '.join(parts)})"


def format_history(history: Iterable[Step]) -> str:
    """Render a processing history as a numbered, human-readable list.

    Parameters
    ----------
    history
        The steps to render, in the order they were applied.

    Returns
    -------
    str
        One line per step, or ``"<no processing>"`` if the history is empty.
    """
    steps = list(history)
    if not steps:
        return "<no processing>"
    return "\n".join(f"{i:>2}. {step!r}" for i, step in enumerate(steps, start=1))
