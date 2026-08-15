"""Tests for ChannelSpec and Recording."""

from __future__ import annotations

import numpy as np
import pytest

from fluoroflow import ChannelSpec, Events, Recording, Step, Trace
from fluoroflow.exceptions import ValidationError


def make_trace(name: str, n: int = 10, fs: float = 10.0) -> Trace:
    """A minimal well-formed trace with the given name."""
    return Trace(np.arange(n) / fs, np.arange(n, dtype=float), name=name)


@pytest.fixture
def recording() -> Recording:
    """A two-channel recording with one declared signal/control pair."""
    return Recording.from_traces(
        make_trace("Region0G"),
        make_trace("Region0G_iso"),
        events=(Events([0.2, 0.5], name="cue"),),
        channels=(ChannelSpec("Region0G", control="Region0G_iso", region="vCA1"),),
        subject="M1",
        session="day1",
    )


class TestChannelSpec:
    def test_minimal_spec_has_no_control(self) -> None:
        spec = ChannelSpec("Region0G")
        assert spec.control is None
        assert not spec.has_control

    def test_a_channel_cannot_be_its_own_control(self) -> None:
        with pytest.raises(ValidationError, match="cannot be its own control"):
            ChannelSpec("Region0G", control="Region0G")

    def test_rejects_a_blank_signal(self) -> None:
        with pytest.raises(ValidationError, match="non-empty string"):
            ChannelSpec("  ")

    def test_repr_omits_unset_fields(self) -> None:
        assert repr(ChannelSpec("Region0G")) == "ChannelSpec('Region0G')"

    def test_specs_compare_by_value(self) -> None:
        assert ChannelSpec("A", region="vCA1") == ChannelSpec("A", region="vCA1")
        assert ChannelSpec("A", region="vCA1") != ChannelSpec("A", region="BLA")

    def test_metadata_is_a_defensive_copy(self) -> None:
        meta = {"fiber": "MFC_200"}
        spec = ChannelSpec("A", meta=meta)
        meta["fiber"] = "tampered"
        assert spec.meta["fiber"] == "MFC_200"


class TestConstruction:
    def test_from_traces_keys_by_name(self, recording: Recording) -> None:
        assert set(recording) == {"Region0G", "Region0G_iso"}
        assert recording["Region0G"].name == "Region0G"

    def test_from_traces_rejects_duplicate_names(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate trace name"):
            Recording.from_traces(make_trace("A"), make_trace("A"))

    def test_from_traces_rejects_non_traces(self) -> None:
        with pytest.raises(ValidationError, match="expects Trace objects"):
            Recording.from_traces("not a trace")

    def test_from_traces_keys_events_by_name(self) -> None:
        rec = Recording.from_traces(make_trace("A"), events=(Events([1.0], name="cue"),))
        assert rec.event("cue").name == "cue"

    def test_from_traces_rejects_duplicate_event_names(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate event name"):
            Recording.from_traces(
                make_trace("A"), events=(Events([1.0], name="cue"), Events([2.0], name="cue"))
            )

    def test_a_key_must_match_the_traces_own_name(self) -> None:
        with pytest.raises(ValidationError, match="key and its name must agree"):
            Recording(traces={"wrong_key": make_trace("Region0G")})

    def test_an_event_key_must_match_its_name(self) -> None:
        with pytest.raises(ValidationError, match="key and name must agree"):
            Recording(traces={}, events={"wrong": Events([1.0], name="cue")})

    def test_rejects_a_non_trace_value(self) -> None:
        with pytest.raises(ValidationError, match="must be a Trace"):
            Recording(traces={"A": np.zeros(5)})  # type: ignore[dict-item]

    def test_a_channel_spec_must_reference_traces_that_exist(self) -> None:
        with pytest.raises(ValidationError, match="no such trace"):
            Recording.from_traces(make_trace("Region0G"), channels=(ChannelSpec("Region1R"),))

    def test_a_missing_control_is_caught_too(self) -> None:
        with pytest.raises(ValidationError, match="as its control"):
            Recording.from_traces(
                make_trace("Region0G"),
                channels=(ChannelSpec("Region0G", control="Region0G_iso"),),
            )

    def test_the_error_lists_what_is_available(self) -> None:
        with pytest.raises(ValidationError, match="Available traces: Region0G"):
            Recording.from_traces(make_trace("Region0G"), channels=(ChannelSpec("Region1R"),))

    def test_an_empty_recording_is_valid(self) -> None:
        rec = Recording(traces={})
        assert len(rec) == 0
        assert rec.duration == 0.0


class TestAccess:
    def test_getitem(self, recording: Recording) -> None:
        assert recording["Region0G"].name == "Region0G"

    def test_getitem_error_lists_available_traces(self, recording: Recording) -> None:
        with pytest.raises(KeyError, match="Region0G, Region0G_iso"):
            _ = recording["Region1R"]

    def test_contains_and_len(self, recording: Recording) -> None:
        assert "Region0G" in recording
        assert "Region1R" not in recording
        assert len(recording) == 2

    def test_event_lookup_error_lists_available_sets(self, recording: Recording) -> None:
        with pytest.raises(KeyError, match="Available event sets: cue"):
            recording.event("shock")

    def test_channel_lookup(self, recording: Recording) -> None:
        assert recording.channel("Region0G").region == "vCA1"

    def test_channel_lookup_error_lists_declared_signals(self, recording: Recording) -> None:
        with pytest.raises(KeyError, match="Declared signals: Region0G"):
            recording.channel("Region1R")

    def test_pairs_yields_spec_signal_and_control(self, recording: Recording) -> None:
        (spec, signal, control) = next(iter(recording.pairs()))
        assert spec.signal == "Region0G"
        assert signal.name == "Region0G"
        assert control is not None
        assert control.name == "Region0G_iso"

    def test_pairs_yields_none_for_a_channel_without_a_control(self) -> None:
        rec = Recording.from_traces(make_trace("Region0G"), channels=(ChannelSpec("Region0G"),))
        (_, _, control) = next(iter(rec.pairs()))
        assert control is None

    def test_duration_is_the_longest_trace(self) -> None:
        rec = Recording.from_traces(make_trace("A", n=10), make_trace("B", n=50))
        assert rec.duration == pytest.approx(4.9)

    def test_duration_ignores_empty_traces(self) -> None:
        rec = Recording.from_traces(make_trace("A", n=10), Trace([], [], name="B"))
        assert rec.duration == pytest.approx(0.9)


class TestDerivation:
    def test_with_traces_replaces_by_name(self, recording: Recording) -> None:
        replacement = make_trace("Region0G", n=5)
        out = recording.with_traces(replacement)
        assert len(out["Region0G"]) == 5
        assert len(recording["Region0G"]) == 10, "the original must be untouched"

    def test_with_traces_rejects_non_traces(self, recording: Recording) -> None:
        with pytest.raises(ValidationError, match="expects Trace objects"):
            recording.with_traces(3)

    def test_with_events_adds_and_replaces(self, recording: Recording) -> None:
        out = recording.with_events(Events([9.0], name="shock"))
        assert set(out.events) == {"cue", "shock"}
        assert set(recording.events) == {"cue"}

    def test_with_channels_replaces_by_signal(self, recording: Recording) -> None:
        out = recording.with_channels(ChannelSpec("Region0G", region="BLA"))
        assert len(out.channels) == 1
        assert out.channel("Region0G").region == "BLA"
        assert recording.channel("Region0G").region == "vCA1"

    def test_with_meta_merges(self, recording: Recording) -> None:
        out = recording.with_meta(treatment="ChR2").with_meta(rig="A")
        assert dict(out.meta) == {"treatment": "ChR2", "rig": "A"}
        assert dict(recording.meta) == {}

    def test_derivation_preserves_identity_fields(self, recording: Recording) -> None:
        out = recording.with_meta(x=1)
        assert out.subject == "M1"
        assert out.session == "day1"
        assert out.channels == recording.channels


class TestMapTraces:
    def test_applies_to_every_trace_by_default(self, recording: Recording) -> None:
        out = recording.map_traces(lambda t: t.derive(values=t.values * 2, step=Step("double")))
        np.testing.assert_allclose(out["Region0G"].values, recording["Region0G"].values * 2)
        np.testing.assert_allclose(out["Region0G_iso"].values, recording["Region0G_iso"].values * 2)

    def test_records_the_step_on_each_trace(self, recording: Recording) -> None:
        out = recording.map_traces(lambda t: t.derive(step=Step("tag")))
        assert all(len(t.history) == 1 for t in out.traces.values())

    def test_respects_a_key_subset(self, recording: Recording) -> None:
        out = recording.map_traces(
            lambda t: t.derive(values=t.values * 0, step=Step("zero")), keys=("Region0G",)
        )
        assert out["Region0G"].values.sum() == 0
        np.testing.assert_allclose(out["Region0G_iso"].values, recording["Region0G_iso"].values)

    def test_rejects_an_unknown_key(self, recording: Recording) -> None:
        with pytest.raises(KeyError, match="No trace named 'Region1R'"):
            recording.map_traces(lambda t: t, keys=("Region1R",))

    def test_rejects_a_function_that_does_not_return_a_trace(self, recording: Recording) -> None:
        with pytest.raises(ValidationError, match="must return a Trace"):
            recording.map_traces(lambda t: t.values)  # type: ignore[return-value]

    def test_a_rename_rekeys_the_trace(self, recording: Recording) -> None:
        out = recording.map_traces(lambda t: t.rename(f"{t.name}_dff"))
        assert set(out) == {"Region0G_dff", "Region0G_iso_dff"}

    def test_a_rename_follows_through_into_the_channel_specs(self, recording: Recording) -> None:
        out = recording.map_traces(lambda t: t.rename(f"{t.name}_dff"))
        spec = out.channel("Region0G_dff")
        assert spec.control == "Region0G_iso_dff"
        assert spec.region == "vCA1", "unrelated spec fields must survive the rename"

    def test_preserves_trace_order(self, recording: Recording) -> None:
        out = recording.map_traces(lambda t: t.derive(step=Step("tag")))
        assert list(out) == list(recording)


class TestExportAndRepr:
    def test_describe_has_one_row_per_trace(self, recording: Recording) -> None:
        frame = recording.describe()
        assert len(frame) == 2
        assert list(frame.columns) == [
            "trace",
            "n_samples",
            "fs",
            "duration",
            "units",
            "n_missing",
            "n_steps",
        ]
        assert frame.loc[0, "fs"] == pytest.approx(10.0)

    def test_describe_reports_nan_rather_than_raising_on_a_short_trace(self) -> None:
        rec = Recording.from_traces(Trace([0.0], [1.0], name="A"))
        frame = rec.describe()
        assert np.isnan(frame.loc[0, "fs"])
        assert frame.loc[0, "duration"] == pytest.approx(0.0)

    def test_describe_on_an_empty_trace(self) -> None:
        frame = Recording.from_traces(Trace([], [], name="A")).describe()
        assert np.isnan(frame.loc[0, "fs"])
        assert np.isnan(frame.loc[0, "duration"])

    def test_equality(self, recording: Recording) -> None:
        assert recording == recording.with_meta()
        assert recording != recording.with_meta(x=1)

    def test_comparison_with_a_non_recording_is_false_not_an_error(
        self, recording: Recording
    ) -> None:
        assert recording != None  # noqa: E711

    def test_repr(self, recording: Recording) -> None:
        text = repr(recording)
        assert "M1" in text
        assert "traces=2" in text
        assert "channels=1" in text

    def test_repr_without_identifiers(self) -> None:
        assert "<no subject>" in repr(Recording(traces={}))
