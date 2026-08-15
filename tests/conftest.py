"""Shared fixtures and test configuration."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, settings

from fluoroflow import Trace

# Avoid deadline failures on slow CI runners.
settings.register_profile(
    "fluoroflow",
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("fluoroflow")


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator, so a failing test fails the same way twice."""
    return np.random.default_rng(20260810)


@pytest.fixture
def simple_trace() -> Trace:
    """A short, uniformly sampled, well-behaved trace at 10 Hz."""
    return Trace(
        time=np.arange(10, dtype=float) / 10.0,
        values=np.array([1.0, 1.1, 0.9, 1.2, 1.0, 0.8, 1.3, 1.1, 0.95, 1.05]),
        name="Region0G",
    )
