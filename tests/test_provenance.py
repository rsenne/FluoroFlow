"""Tests for Step and history formatting."""

from __future__ import annotations

import dataclasses

import pytest

from fluoroflow import MEAN_REMOVED, NORMALIZED, Step
from fluoroflow.core.provenance import format_history


class TestStep:
    def test_defaults(self) -> None:
        step = Step("dff")
        assert dict(step.params) == {}
        assert step.tags == frozenset()

    def test_rejects_a_blank_name(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            Step("   ")

    def test_is_frozen(self) -> None:
        step = Step("dff")
        with pytest.raises(dataclasses.FrozenInstanceError):
            step.name = "other"  # type: ignore[misc]

    def test_params_cannot_be_mutated_after_construction(self) -> None:
        step = Step("dff", {"percentile": 8.0})
        with pytest.raises(TypeError):
            step.params["percentile"] = 0.08  # type: ignore[index]

    def test_params_are_a_defensive_copy(self) -> None:
        params = {"percentile": 8.0}
        step = Step("dff", params)
        params["percentile"] = 0.08
        assert step.params["percentile"] == 8.0

    def test_tags_are_normalised_to_a_frozenset(self) -> None:
        assert Step("detrend", tags=[MEAN_REMOVED, MEAN_REMOVED]).tags == frozenset({MEAN_REMOVED})

    def test_equality_is_by_value(self) -> None:
        assert Step("dff", {"p": 8.0}) == Step("dff", {"p": 8.0})
        assert Step("dff", {"p": 8.0}) != Step("dff", {"p": 9.0})
        assert Step("dff") != Step("zscore")
        assert Step("dff", tags={NORMALIZED}) != Step("dff")

    def test_comparison_with_a_non_step_is_false_not_an_error(self) -> None:
        assert Step("dff") != "dff"
        assert Step("dff") != None  # noqa: E711

    def test_steps_are_unhashable(self) -> None:
        # A params mapping has no meaningful hash. Saying so up front beats
        # RamiPho's __hash__, which raised TypeError from inside a dict lookup.
        with pytest.raises(TypeError):
            hash(Step("dff"))

    def test_repr_shows_name_params_and_tags(self) -> None:
        assert repr(Step("dff", {"p": 8.0})) == "Step('dff', p=8.0)"
        text = repr(Step("detrend", {"lam": 1e7}, tags={MEAN_REMOVED}))
        assert "detrend" in text
        assert "lam=10000000.0" in text
        assert "mean-removed" in text


class TestFormatHistory:
    def test_empty(self) -> None:
        assert format_history(()) == "<no processing>"

    def test_numbers_the_steps_in_order(self) -> None:
        text = format_history([Step("airpls"), Step("butterworth", {"cutoff": 4.0})])
        assert text.splitlines() == [" 1. Step('airpls')", " 2. Step('butterworth', cutoff=4.0)"]

    def test_accepts_any_iterable(self) -> None:
        assert format_history(iter([Step("a")])) == " 1. Step('a')"


def test_the_tag_constants_are_distinct() -> None:
    from fluoroflow.core import provenance

    tags = [
        provenance.BASELINE_CORRECTED,
        provenance.FILTERED,
        provenance.MEAN_REMOVED,
        provenance.MOTION_CORRECTED,
        provenance.NORMALIZED,
        provenance.RESAMPLED,
    ]
    assert len(set(tags)) == len(tags)
