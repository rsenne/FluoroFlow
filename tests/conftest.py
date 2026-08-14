"""Shared fixtures and test configuration."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, settings

from fluoroflow import Trace
from fluoroflow.datasets import SyntheticDataset, synthetic_recording

# Numerical property tests do real work; a wall-clock deadline turns a slow CI
# runner into a spurious failure.
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


@pytest.fixture
def clean_dataset() -> SyntheticDataset:
    """A noiseless, motion-free, event-locked dataset for exact assertions."""
    return synthetic_recording(
        duration=60.0,
        event_times=[5.0, 15.0, 25.0, 35.0, 45.0],
        n_transients=0,
        noise_cv=0.0,
        motion_gain_signal=0.0,
        motion_gain_control=0.0,
        seed=0,
    )


@pytest.fixture
def realistic_dataset() -> SyntheticDataset:
    """A dataset with noise, motion, jitter, and dropped frames."""
    return synthetic_recording(
        duration=300.0,
        event_times=[30.0, 90.0, 150.0, 210.0, 270.0],
        n_transients=60,
        timestamp_jitter=0.02,
        dropped_fraction=0.01,
        seed=7,
    )
