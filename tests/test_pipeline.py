"""Integration tests for the toggleable preprocess() pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from fluoroflow import Recording, Trace
from fluoroflow.exceptions import ValidationError
from fluoroflow.preprocessing import BaselineOptions, DffOptions, PreprocessOptions, preprocess


def make_recording(n: int = 200, fs: float = 30.0, seed: int = 0) -> Recording:
    """A two-channel session: a decaying isosbestic and a linearly related signal."""
    rng = np.random.default_rng(seed)
    time = np.arange(n) / fs
    iso_values = np.linspace(500.0, 480.0, n) + rng.normal(0.0, 0.2, n)
    signal_values = 1.5 * iso_values + 20.0 + rng.normal(0.0, 0.2, n)
    signal_values[50:55] += 15.0
    signal_values[150:153] += 10.0
    iso = Trace(time, iso_values, name="Region0G_iso")
    signal = Trace(time, signal_values, name="Region0G")
    return Recording.from_traces(signal, isosbestic=iso, subject="M1", session="s1")


class TestPreprocess:
    def test_default_pipeline_produces_null_z(self) -> None:
        rec = make_recording()
        out = preprocess(rec)
        result = out["Region0G"]
        assert result.units == "null-Z"
        assert len(result) == len(rec["Region0G"])
        assert result.history[-1].params["method"] == "null_z"

    def test_raw_dff_is_still_available(self) -> None:
        rec = make_recording()
        out = preprocess(rec, PreprocessOptions(dff=DffOptions(method="dff")))
        result = out["Region0G"]
        assert result.units == "dF/F"
        assert result.history[-1].params["method"] == "dff"

    def test_lowpass_only_when_baseline_disabled(self) -> None:
        rec = make_recording()
        out = preprocess(rec, PreprocessOptions(baseline=None, dff=None))
        result = out["Region0G"]
        assert result.units == rec["Region0G"].units
        assert result.has_step("lowpass")
        assert not result.has_step("dff")

    def test_stops_after_baseline_correction_when_dff_is_none(self) -> None:
        rec = make_recording()
        out = preprocess(rec, PreprocessOptions(dff=None))
        result = out["Region0G"]
        assert result.has_step("baseline_correct")
        assert not result.has_step("dff")
        assert result.units == rec["Region0G"].units

    def test_each_signal_is_regressed_against_the_shared_isosbestic(self) -> None:
        base = make_recording()
        signal2_values = 0.8 * base["Region0G_iso"].values + 10.0
        signal2 = Trace(base["Region0G"].time, signal2_values, name="Region1G")
        rec = Recording.from_traces(
            base["Region0G"], signal2, isosbestic=base["Region0G_iso"], subject="M1", session="s1"
        )
        out = preprocess(rec)
        assert out.signals[0].units == "null-Z"
        assert out.signals[1].units == "null-Z"

    def test_baseline_without_isosbestic_raises(self) -> None:
        rec = Recording.from_traces(make_recording()["Region0G"])
        with pytest.raises(ValidationError, match="isosbestic control channel"):
            preprocess(rec)

    def test_dff_without_baseline_raises(self) -> None:
        rec = make_recording()
        with pytest.raises(ValidationError, match="requires baseline correction"):
            preprocess(rec, PreprocessOptions(baseline=None))

    def test_custom_baseline_and_dff_options_are_honoured(self) -> None:
        rec = make_recording()
        out = preprocess(
            rec,
            PreprocessOptions(
                baseline=BaselineOptions(tuning_constant=3.0),
                dff=DffOptions(method="null_z"),
            ),
        )
        result = out["Region0G"]
        assert result.units == "null-Z"
