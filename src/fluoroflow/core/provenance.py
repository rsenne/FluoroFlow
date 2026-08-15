"""Ordered processing provenance for signals."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

__all__ = ["Step"]


@dataclass(frozen=True, slots=True, eq=False)
class Step:
    """One recorded operation in a signal's processing history."""

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the name and freeze ``params`` against later mutation."""
        if not isinstance(self.name, str) or not self.name.strip():
            msg = f"Step name must be a non-empty string, got {self.name!r}."
            raise ValueError(msg)
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    def __eq__(self, other: object) -> bool:
        """Compare by name and parameters."""
        if not isinstance(other, Step):
            return NotImplemented
        return self.name == other.name and dict(self.params) == dict(other.params)

    def __repr__(self) -> str:
        """Render as ``Step(name, k=v, ...)``."""
        parts = [repr(self.name)]
        parts += [f"{k}={v!r}" for k, v in self.params.items()]
        return f"Step({', '.join(parts)})"


def format_history(history: Iterable[Step]) -> str:
    """Render a processing history as a numbered, human-readable list."""
    steps = list(history)
    if not steps:
        return "<no processing>"
    return "\n".join(f"{i:>2}. {step!r}" for i, step in enumerate(steps, start=1))
