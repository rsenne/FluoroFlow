"""Tests for the synthetic data generator.

The generator is test infrastructure, so it needs testing harder than the code it
supports: a silently wrong forward model would make every downstream recovery
test agree with itself and be wrong together.
"""

from __future__ import annotations

import numpy as np
import pytest

from fluoroflow.datasets import (
    SyntheticDataset,
    biexponential_bleach,
    synthetic_recording,
    transient_kernel,
)
from fluoroflow.exceptions import ValidationError


class TestTransientKernel:
    def test_peaks_at_exactly_one(self) -> None:
        # Amplitude parameters elsewhere are documented as peak dF/F, which is only
        # true if the kernel peak is exactly 1.
        assert transient_kernel(100.0).max() == pytest.approx(1.0)

    def test_starts_at_zero_and_is_causal(self) -> None:
        k = transient_kernel(100.0)
        assert k[0] == pytest.approx(0.0)
        assert np.all(k >= 0.0)

    def test_rises_then_decays(self) -> None:
        k = transient_kernel(100.0, tau_rise=0.05, tau_decay=0.5)
        peak = int(np.argmax(k))
        assert peak > 0
        assert np.all(np.diff(k[: peak + 1]) > 0)
        assert np.all(np.diff(k[peak:]) < 0)

    def test_peak_latency_matches_the_analytic_value(self) -> None:
        # For a difference of exponentials the peak sits at
        # t = ln(td/tr) / (1/tr - 1/td), which is a closed form worth checking
        # against rather than trusting the array.
        rise, decay, fs = 0.08, 0.60, 500.0
        expected = np.log(decay / rise) / (1.0 / rise - 1.0 / decay)
        k = transient_kernel(fs, tau_rise=rise, tau_decay=decay)
        assert float(np.argmax(k)) / fs == pytest.approx(expected, abs=1.0 / fs)

    def test_longer_decay_makes_a_longer_kernel(self) -> None:
        longer = transient_kernel(50.0, tau_decay=1.0)
        shorter = transient_kernel(50.0, tau_decay=0.5)
        assert longer.size > shorter.size

    def test_rejects_a_rise_slower_than_the_decay(self) -> None:
        with pytest.raises(ValidationError, match="tau_rise must be smaller"):
            transient_kernel(100.0, tau_rise=1.0, tau_decay=0.5)

    def test_rejects_a_non_positive_rate(self) -> None:
        with pytest.raises(ValidationError, match="greater than zero"):
            transient_kernel(0.0)


class TestBiexponentialBleach:
    def test_decays_monotonically(self) -> None:
        time = np.arange(1000) / 30.0
        envelope = biexponential_bleach(
            time, baseline=1000.0, amplitude=0.35, tau_fast=20.0, tau_slow=400.0
        )
        assert np.all(np.diff(envelope) < 0)

    def test_starts_the_requested_fraction_above_baseline(self) -> None:
        time = np.arange(10) / 30.0
        envelope = biexponential_bleach(
            time, baseline=1000.0, amplitude=0.35, tau_fast=20.0, tau_slow=400.0
        )
        assert envelope[0] == pytest.approx(1350.0)

    def test_approaches_baseline_over_many_time_constants(self) -> None:
        time = np.arange(0.0, 5000.0, 1.0)
        envelope = biexponential_bleach(
            time, baseline=1000.0, amplitude=0.35, tau_fast=20.0, tau_slow=400.0
        )
        assert envelope[-1] == pytest.approx(1000.0, rel=1e-3)

    def test_is_invariant_to_the_time_origin(self) -> None:
        # The envelope is defined relative to the first sample, so shifting the
        # clock must not change the curve.
        time = np.arange(100) / 30.0
        kwargs = {"baseline": 1000.0, "amplitude": 0.2, "tau_fast": 5.0, "tau_slow": 50.0}
        np.testing.assert_allclose(
            biexponential_bleach(time, **kwargs),
            biexponential_bleach(time + 1234.5, **kwargs),
        )

    def test_rejects_an_out_of_range_share(self) -> None:
        with pytest.raises(ValidationError, match=r"\[0, 1\]"):
            biexponential_bleach(
                np.arange(10.0),
                baseline=1.0,
                amplitude=0.1,
                tau_fast=1.0,
                tau_slow=2.0,
                fast_share=2.0,
            )


class TestForwardModel:
    def test_dividing_out_the_true_bleach_recovers_the_true_dff_exactly(self) -> None:
        # This is the contract every downstream recovery test leans on. With no
        # noise it must hold to floating-point precision, not approximately.
        data = synthetic_recording(duration=60.0, noise_cv=0.0, seed=1)
        recovered = data.signal.values / data.truth.bleach_signal - 1.0
        np.testing.assert_allclose(recovered, data.truth.observable_dff, atol=1e-12)

    def test_the_control_channel_carries_motion_but_no_transients(self) -> None:
        data = synthetic_recording(duration=60.0, noise_cv=0.0, seed=1)
        recovered = data.control.values / data.truth.bleach_control - 1.0
        expected = data.truth.motion_gain_control * data.truth.motion
        np.testing.assert_allclose(recovered, expected, atol=1e-12)

    def test_noise_has_the_requested_magnitude(self) -> None:
        data = synthetic_recording(duration=600.0, noise_cv=0.004, seed=2)
        residual = data.signal.values - data.truth.bleach_signal * (1.0 + data.truth.observable_dff)
        assert residual.std() == pytest.approx(data.truth.noise_sd_signal, rel=0.05)
        assert residual.mean() == pytest.approx(0.0, abs=0.2 * data.truth.noise_sd_signal)

    def test_motion_is_shared_between_channels_with_different_gains(self) -> None:
        data = synthetic_recording(duration=120.0, noise_cv=0.0, n_transients=0, seed=3)
        sig = data.signal.values / data.truth.bleach_signal - 1.0
        ctl = data.control.values / data.truth.bleach_control - 1.0
        ratio = data.truth.motion_gain_signal / data.truth.motion_gain_control
        np.testing.assert_allclose(sig, ratio * ctl, atol=1e-12)
        assert data.truth.motion_gain_signal != data.truth.motion_gain_control

    def test_motion_is_standardised(self) -> None:
        motion = synthetic_recording(duration=300.0, seed=4).truth.motion
        assert motion.mean() == pytest.approx(0.0, abs=1e-9)
        assert motion.std() == pytest.approx(1.0, rel=1e-9)


class TestTransients:
    def test_event_locked_transients_have_the_requested_peak_amplitude(
        self, clean_dataset: SyntheticDataset
    ) -> None:
        truth = clean_dataset.truth
        for event in clean_dataset.recording.event("cue"):
            window = (truth.time >= event) & (truth.time < event + 3.0)
            assert truth.transient_dff[window].max() == pytest.approx(0.10, rel=1e-9)

    def test_transients_peak_after_the_event_not_before(
        self, clean_dataset: SyntheticDataset
    ) -> None:
        truth = clean_dataset.truth
        for event in clean_dataset.recording.event("cue"):
            before = (truth.time >= event - 2.0) & (truth.time < event)
            assert truth.transient_dff[before].max() == pytest.approx(0.0, abs=1e-12)

    def test_transient_bookkeeping_matches_the_events(
        self, clean_dataset: SyntheticDataset
    ) -> None:
        truth = clean_dataset.truth
        events = clean_dataset.recording.event("cue")
        assert len(truth.transient_times) == len(events)
        np.testing.assert_allclose(truth.transient_times, events.times + 0.15)
        np.testing.assert_allclose(truth.transient_amplitudes, 0.10)

    def test_transient_times_are_sorted(self) -> None:
        truth = synthetic_recording(
            duration=120.0, event_times=[10.0, 50.0], n_transients=30, seed=5
        ).truth
        assert np.all(np.diff(truth.transient_times) >= 0)

    def test_spontaneous_amplitudes_respect_the_requested_range(self) -> None:
        truth = synthetic_recording(
            duration=300.0, n_transients=100, transient_amplitude=(0.05, 0.20), seed=6
        ).truth
        assert truth.transient_amplitudes.min() >= 0.05
        assert truth.transient_amplitudes.max() <= 0.20

    def test_no_transients_means_a_flat_true_dff(self) -> None:
        truth = synthetic_recording(duration=30.0, n_transients=0, seed=7).truth
        np.testing.assert_array_equal(truth.transient_dff, 0.0)


class TestRecordingShape:
    def test_traces_share_the_time_base_with_the_ground_truth(
        self, realistic_dataset: SyntheticDataset
    ) -> None:
        truth = realistic_dataset.truth
        for name in ("Region0G", "Region0G_iso"):
            np.testing.assert_array_equal(realistic_dataset.recording[name].time, truth.time)
        for array in (truth.transient_dff, truth.motion, truth.bleach_signal, truth.bleach_control):
            assert array.shape == truth.time.shape

    def test_the_channel_pairing_is_declared(self, realistic_dataset: SyntheticDataset) -> None:
        spec, signal, control = next(iter(realistic_dataset.recording.pairs()))
        assert spec.excitation_nm == 470.0
        assert spec.control_nm == 415.0
        assert signal.name == "Region0G"
        assert control is not None

    def test_events_are_attached_when_requested(self) -> None:
        data = synthetic_recording(duration=60.0, event_times=[5.0, 25.0], seed=0)
        np.testing.assert_allclose(data.recording.event("cue").times, [5.0, 25.0])

    def test_no_events_are_attached_by_default(self) -> None:
        assert synthetic_recording(duration=10.0).recording.events == {}

    def test_sampling_rate_is_what_was_asked_for(self) -> None:
        assert synthetic_recording(duration=60.0, fs=45.0).signal.fs == pytest.approx(45.0)

    def test_traces_carry_no_processing_history(self) -> None:
        # Synthetic data is raw data. A generator that pre-labelled its output as
        # processed would let a pipeline's provenance guards pass vacuously.
        assert synthetic_recording(duration=10.0).signal.history == ()


class TestAcquisitionArtefacts:
    def test_dropped_frames_shorten_the_recording_and_leave_gaps(self) -> None:
        data = synthetic_recording(duration=300.0, dropped_fraction=0.05, seed=8)
        expected = round(300.0 * 30.0)
        assert len(data.signal) < expected
        assert len(data.signal) == pytest.approx(expected * 0.95, rel=0.05)
        assert data.signal.sampling.n_gaps > 0

    def test_dropped_frames_keep_the_time_base_valid(self) -> None:
        time = synthetic_recording(duration=60.0, dropped_fraction=0.3, seed=9).signal.time
        assert np.all(np.diff(time) > 0)

    def test_the_median_rate_survives_heavy_frame_loss(self) -> None:
        # The median interval is the reason for this: a fifth of the frames gone
        # and the reported rate is still the true per-frame rate.
        data = synthetic_recording(duration=300.0, fs=30.0, dropped_fraction=0.2, seed=10)
        assert data.signal.fs == pytest.approx(30.0, rel=1e-9)

    def test_timestamp_jitter_keeps_the_time_base_strictly_increasing(self) -> None:
        for seed in range(5):
            time = synthetic_recording(duration=30.0, timestamp_jitter=0.5, seed=seed).signal.time
            assert np.all(np.diff(time) > 0)

    def test_jitter_shows_up_in_the_sampling_report(self) -> None:
        report = synthetic_recording(duration=60.0, timestamp_jitter=0.05, seed=11).signal.sampling
        assert report.cv > 0.01
        assert not report.is_uniform


class TestReproducibility:
    def test_the_same_seed_gives_identical_data(self) -> None:
        a = synthetic_recording(duration=30.0, seed=42)
        b = synthetic_recording(duration=30.0, seed=42)
        np.testing.assert_array_equal(a.signal.values, b.signal.values)
        np.testing.assert_array_equal(a.control.values, b.control.values)
        assert a.recording == b.recording

    def test_different_seeds_give_different_data(self) -> None:
        a = synthetic_recording(duration=30.0, seed=1)
        b = synthetic_recording(duration=30.0, seed=2)
        assert not np.array_equal(a.signal.values, b.signal.values)

    def test_the_resolved_parameters_are_recorded(self) -> None:
        params = synthetic_recording(duration=30.0, fs=25.0, seed=3).truth.params
        assert params["fs"] == 25.0
        assert params["seed"] == 3
        assert params["duration"] == 30.0


class TestValidation:
    def test_rejects_a_duration_too_short_for_two_samples(self) -> None:
        with pytest.raises(ValidationError, match="at least 2 are needed"):
            synthetic_recording(duration=0.01, fs=30.0)

    @pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
    def test_rejects_an_impossible_drop_fraction(self, bad: float) -> None:
        with pytest.raises(ValidationError, match=r"\[0, 1\)"):
            synthetic_recording(duration=10.0, dropped_fraction=bad)

    def test_rejects_an_inverted_amplitude_range(self) -> None:
        with pytest.raises(ValidationError, match="low <= high"):
            synthetic_recording(duration=10.0, transient_amplitude=(0.2, 0.1))

    def test_rejects_a_non_positive_rate(self) -> None:
        with pytest.raises(ValidationError, match="greater than zero"):
            synthetic_recording(duration=10.0, fs=-1.0)
