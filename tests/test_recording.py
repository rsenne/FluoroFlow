"""Tests for Recording."""

from __future__ import annotations

import numpy as np
import pytest

from fluoroflow import Events, Recording, Step, Trace
from fluoroflow.exceptions import ValidationError


def make_trace(name: str, n: int = 10, fs: float = 10.0) -> Trace:
    """A minimal well-formed trace with the given name."""
    return Trace(np.arange(n) / fs, np.arange(n, dtype=float), name=name)


@pytest.fixture
def recording() -> Recording:
    """A two-channel recording: one signal, one isosbestic, one event set."""
    return Recording.from_traces(
        make_trace("Region0G"),
        isosbestic=make_trace("Region0G_iso"),
        events=(Events([0.2, 0.5], name="cue"),),
        subject="M1",
        session="day1",
    )


class TestConstruction:
    def test_from_traces_accepts_one_signal(self) -> None:
        rec = Recording.from_traces(make_trace("Region0G"))
        assert rec.signals[0].name == "Region0G"
        assert rec.isosbestic is None

    def test_from_traces_accepts_two_signals(self) -> None:
        rec = Recording.from_traces(make_trace("Region0G"), make_trace("Region1G"))
        assert [t.name for t in rec.signals] == ["Region0G", "Region1G"]

    def test_rejects_zero_signals(self) -> None:
        with pytest.raises(ValidationError, match="1 or 2 signal traces"):
            Recording(signals=())

    def test_rejects_three_signals(self) -> None:
        with pytest.raises(ValidationError, match="1 or 2 signal traces"):
            Recording(signals=(make_trace("A"), make_trace("B"), make_trace("C")))

    def test_rejects_a_non_trace_signal(self) -> None:
        with pytest.raises(ValidationError, match="must be a Trace"):
            Recording(signals=(np.zeros(5),))

    def test_rejects_a_non_trace_isosbestic(self) -> None:
        with pytest.raises(ValidationError, match="must be a Trace or None"):
            Recording(signals=(make_trace("A"),), isosbestic=np.zeros(5))

    def test_rejects_duplicate_names_between_signal_and_isosbestic(self) -> None:
        with pytest.raises(ValidationError, match="must be unique"):
            Recording(signals=(make_trace("A"),), isosbestic=make_trace("A"))

    def test_rejects_duplicate_names_between_signals(self) -> None:
        with pytest.raises(ValidationError, match="must be unique"):
            Recording(signals=(make_trace("A"), make_trace("A")))

    def test_from_traces_keys_events_by_name(self) -> None:
        rec = Recording.from_traces(make_trace("A"), events=(Events([1.0], name="cue"),))
        assert rec.event("cue").name == "cue"

    def test_from_traces_rejects_duplicate_event_names(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate event name"):
            Recording.from_traces(
                make_trace("A"), events=(Events([1.0], name="cue"), Events([2.0], name="cue"))
            )

    def test_an_event_key_must_match_its_name(self) -> None:
        with pytest.raises(ValidationError, match="key and name must agree"):
            Recording(signals=(make_trace("A"),), events={"wrong": Events([1.0], name="cue")})

    def test_from_traces_accepts_events_as_a_mapping(self) -> None:
        rec = Recording.from_traces(make_trace("A"), events={"cue": Events([0.05], name="cue")})
        assert rec.event("cue").name == "cue"


class TestAccess:
    def test_getitem_finds_a_signal(self, recording: Recording) -> None:
        assert recording["Region0G"].name == "Region0G"

    def test_getitem_finds_the_isosbestic(self, recording: Recording) -> None:
        assert recording["Region0G_iso"].name == "Region0G_iso"

    def test_getitem_error_lists_available_traces(self, recording: Recording) -> None:
        with pytest.raises(KeyError, match="Region0G, Region0G_iso"):
            _ = recording["Region1R"]

    def test_contains_and_len(self, recording: Recording) -> None:
        assert "Region0G" in recording
        assert "Region0G_iso" in recording
        assert "Region1R" not in recording
        assert len(recording) == 2

    def test_iter_yields_signals_then_isosbestic(self, recording: Recording) -> None:
        assert list(recording) == ["Region0G", "Region0G_iso"]

    def test_event_lookup_error_lists_available_sets(self, recording: Recording) -> None:
        with pytest.raises(KeyError, match="Available event sets: cue"):
            recording.event("shock")

    def test_duration_is_the_longest_trace(self) -> None:
        rec = Recording.from_traces(make_trace("A", n=10), isosbestic=make_trace("B", n=50))
        assert rec.duration == pytest.approx(4.9)

    def test_duration_ignores_empty_traces(self) -> None:
        rec = Recording.from_traces(make_trace("A", n=10), isosbestic=Trace([], [], name="B"))
        assert rec.duration == pytest.approx(0.9)

    def test_an_empty_isosbestic_free_recording_is_valid(self) -> None:
        rec = Recording.from_traces(make_trace("A"))
        assert len(rec) == 1


class TestDerivation:
    def test_with_events_adds_and_replaces(self, recording: Recording) -> None:
        out = recording.with_events(Events([9.0], name="shock"))
        assert set(out.events) == {"cue", "shock"}
        assert set(recording.events) == {"cue"}

    def test_with_meta_merges(self, recording: Recording) -> None:
        out = recording.with_meta(treatment="ChR2").with_meta(rig="A")
        assert dict(out.meta) == {"treatment": "ChR2", "rig": "A"}
        assert dict(recording.meta) == {}

    def test_derivation_preserves_identity_fields(self, recording: Recording) -> None:
        out = recording.with_meta(x=1)
        assert out.subject == "M1"
        assert out.session == "day1"


class TestMapTraces:
    def test_applies_to_every_trace_by_default(self, recording: Recording) -> None:
        out = recording.map_traces(lambda t: t.derive(values=t.values * 2, step=Step("double")))
        np.testing.assert_allclose(out["Region0G"].values, recording["Region0G"].values * 2)
        np.testing.assert_allclose(out["Region0G_iso"].values, recording["Region0G_iso"].values * 2)

    def test_records_the_step_on_each_trace(self, recording: Recording) -> None:
        out = recording.map_traces(lambda t: t.derive(step=Step("tag")))
        assert all(len(t.history) == 1 for t in out._all_traces())

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

    def test_a_rename_keeps_its_role(self, recording: Recording) -> None:
        out = recording.map_traces(lambda t: t.rename(f"{t.name}_dff"))
        assert out.signals[0].name == "Region0G_dff"
        assert out.isosbestic is not None
        assert out.isosbestic.name == "Region0G_iso_dff"

    def test_preserves_trace_order(self, recording: Recording) -> None:
        out = recording.map_traces(lambda t: t.derive(step=Step("tag")))
        assert list(out) == list(recording)


class TestExportAndRepr:
    def test_describe_has_one_row_per_trace(self, recording: Recording) -> None:
        frame = recording.describe()
        assert len(frame) == 2
        assert list(frame.columns) == [
            "trace",
            "role",
            "n_samples",
            "fs",
            "duration",
            "units",
            "n_missing",
            "n_steps",
        ]
        assert frame.loc[0, "role"] == "signal"
        assert frame.loc[1, "role"] == "isosbestic"
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
        assert "signals=1" in text
        assert "isosbestic=yes" in text

    def test_repr_without_identifiers_or_isosbestic(self) -> None:
        text = repr(Recording.from_traces(make_trace("A")))
        assert "<no subject>" in text
        assert "isosbestic=no" in text
