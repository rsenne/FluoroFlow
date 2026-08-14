r"""Synthetic photometry data with known ground truth.

This module is load-bearing for the test suite, not a demo. A preprocessing step
that runs without crashing has proved nothing; the question is whether it
recovers the signal that was actually there. So FluoroFlow generates recordings
from an explicit forward model and keeps the hidden components around, letting
tests assert recovery rather than mere execution.

The forward model is multiplicative, which is how photometry actually behaves:

.. math::

    F_{\mathrm{sig}}(t) &= B_{\mathrm{sig}}(t)\,
        \bigl[1 + a(t) + g_{\mathrm{sig}} m(t)\bigr] + \epsilon_{\mathrm{sig}}(t) \\
    F_{\mathrm{ctl}}(t) &= B_{\mathrm{ctl}}(t)\,
        \bigl[1 + g_{\mathrm{ctl}} m(t)\bigr] + \epsilon_{\mathrm{ctl}}(t)

where :math:`B` is a biexponential bleaching envelope, :math:`a(t)` is the true
transient :math:`\Delta F/F`, and :math:`m(t)` is a shared unit-variance movement
artefact entering both channels with different gains. Two consequences make this
useful for testing: an ideal baseline correction recovers
:math:`a(t) + g_{\mathrm{sig}} m(t)`, and an ideal motion correction on top of it
recovers :math:`a(t)`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

from fluoroflow.core.events import Events
from fluoroflow.core.recording import ChannelSpec, Recording
from fluoroflow.core.trace import Trace
from fluoroflow.core.validation import check_positive
from fluoroflow.exceptions import ValidationError

__all__ = [
    "SyntheticDataset",
    "SyntheticTruth",
    "biexponential_bleach",
    "synthetic_recording",
    "transient_kernel",
]


def transient_kernel(
    fs: float,
    *,
    tau_rise: float = 0.08,
    tau_decay: float = 0.6,
    span: float = 8.0,
) -> NDArray[np.float64]:
    r"""Unit-peak difference-of-exponentials kernel for one calcium transient.

    Normalising to a peak of exactly 1 is what makes the amplitude parameter
    elsewhere in this module mean "peak :math:`\Delta F/F`", so a test can assert
    a recovered peak against the number it asked for.

    Parameters
    ----------
    fs
        Sampling rate in hertz.
    tau_rise
        Rise time constant in seconds. Must be smaller than ``tau_decay``.
    tau_decay
        Decay time constant in seconds.
    span
        Kernel length as a multiple of ``tau_decay``.

    Returns
    -------
    numpy.ndarray
        Causal kernel starting at its onset, peaking at 1.0.

    Raises
    ------
    ValidationError
        If ``tau_rise`` is not strictly less than ``tau_decay``, which would make
        the kernel non-positive.

    Examples
    --------
    >>> from fluoroflow.datasets import transient_kernel
    >>> k = transient_kernel(100.0)
    >>> round(float(k.max()), 12)
    1.0
    >>> float(k[0])
    0.0
    """
    rate = check_positive(fs, label="fs")
    rise = check_positive(tau_rise, label="tau_rise")
    decay = check_positive(tau_decay, label="tau_decay")
    if rise >= decay:
        msg = (
            f"tau_rise must be smaller than tau_decay for a positive transient, "
            f"got tau_rise={rise!r} and tau_decay={decay!r}."
        )
        raise ValidationError(msg)

    n = int(np.ceil(check_positive(span, label="span") * decay * rate)) + 1
    t = np.arange(n, dtype=np.float64) / rate
    k = np.exp(-t / decay) - np.exp(-t / rise)
    peak = float(k.max())
    return np.asarray(k / peak, dtype=np.float64)


def biexponential_bleach(
    time: NDArray[np.float64],
    *,
    baseline: float,
    amplitude: float,
    tau_fast: float,
    tau_slow: float,
    fast_share: float = 0.6,
) -> NDArray[np.float64]:
    """Biexponential photobleaching envelope.

    Parameters
    ----------
    time
        Time vector in seconds, starting at or after zero.
    baseline
        Asymptotic fluorescence the envelope decays towards.
    amplitude
        Total decay as a fraction of ``baseline``, so ``0.35`` means the trace
        starts 35 percent above where it ends.
    tau_fast, tau_slow
        Time constants of the two components, in seconds.
    fast_share
        Fraction of ``amplitude`` carried by the fast component, in [0, 1].

    Returns
    -------
    numpy.ndarray
        The envelope, same length as ``time``.
    """
    fast = check_positive(tau_fast, label="tau_fast")
    slow = check_positive(tau_slow, label="tau_slow")
    share = float(fast_share)
    if not 0.0 <= share <= 1.0:
        msg = f"fast_share must lie in [0, 1], got {share!r}."
        raise ValidationError(msg)
    elapsed = time - time[0] if time.size else time
    decay = share * np.exp(-elapsed / fast) + (1.0 - share) * np.exp(-elapsed / slow)
    return np.asarray(float(baseline) * (1.0 + float(amplitude) * decay), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SyntheticTruth:
    r"""The hidden components of a synthetic recording.

    Every array is on the same time base as the recording's traces, including
    after simulated frame drops, so a test can compare element by element without
    resampling anything.

    Attributes
    ----------
    time
        Time base shared by all arrays here and by the recording's traces.
    transient_dff
        The true transient :math:`\Delta F/F` in the signal channel. This is what
        a perfect end-to-end pipeline should return.
    motion
        Shared movement artefact, normalised to zero mean and unit variance.
    bleach_signal, bleach_control
        The bleaching envelopes actually applied to each channel.
    motion_gain_signal, motion_gain_control
        Gains with which ``motion`` entered each channel.
    transient_times, transient_amplitudes
        Onset time and peak amplitude of every transient injected, sorted by time.
    noise_sd_signal, noise_sd_control
        Standard deviation of the additive Gaussian noise on each channel.
    params
        Every generator argument, resolved, for reproducing the dataset.
    """

    time: NDArray[np.float64]
    transient_dff: NDArray[np.float64]
    motion: NDArray[np.float64]
    bleach_signal: NDArray[np.float64]
    bleach_control: NDArray[np.float64]
    motion_gain_signal: float
    motion_gain_control: float
    transient_times: NDArray[np.float64]
    transient_amplitudes: NDArray[np.float64]
    noise_sd_signal: float
    noise_sd_control: float
    params: Mapping[str, Any]

    @property
    def observable_dff(self) -> NDArray[np.float64]:
        """What a perfect baseline correction alone recovers.

        Transients plus the movement artefact as it appears in the signal channel,
        i.e. the target for a bleach-correction test before motion correction runs.
        """
        return np.asarray(self.transient_dff + self.motion_gain_signal * self.motion)


@dataclass(frozen=True, slots=True)
class SyntheticDataset:
    """A synthetic :class:`~fluoroflow.core.recording.Recording` and its ground truth.

    Attributes
    ----------
    recording
        What a pipeline is allowed to see.
    truth
        What it is being judged against.
    """

    recording: Recording
    truth: SyntheticTruth

    @property
    def signal(self) -> Trace:
        """The signal trace, by convention named ``"Region0G"``."""
        return self.recording["Region0G"]

    @property
    def control(self) -> Trace:
        """The isosbestic control trace, by convention named ``"Region0G_iso"``."""
        return self.recording["Region0G_iso"]


def _make_time(
    *,
    fs: float,
    n: int,
    timestamp_jitter: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Build a time vector, optionally with per-interval jitter.

    Jitter is applied to the intervals and then accumulated, rather than added to
    the timestamps directly, because that construction cannot produce a
    non-increasing time vector no matter how large the jitter.
    """
    if timestamp_jitter <= 0.0:
        return np.arange(n, dtype=np.float64) / fs
    nominal = 1.0 / fs
    steps = nominal * (1.0 + timestamp_jitter * rng.standard_normal(n))
    steps = np.clip(steps, 0.05 * nominal, None)
    time: NDArray[np.float64] = np.cumsum(steps) - steps[0]
    return time


def synthetic_recording(
    *,
    fs: float = 30.0,
    duration: float = 300.0,
    event_times: Sequence[float] | None = None,
    event_response_amplitude: float = 0.10,
    event_response_latency: float = 0.15,
    n_transients: int = 40,
    transient_amplitude: tuple[float, float] = (0.02, 0.15),
    tau_rise: float = 0.08,
    tau_decay: float = 0.60,
    baseline_signal: float = 1000.0,
    baseline_control: float = 600.0,
    bleach_amplitude: float = 0.35,
    bleach_tau_fast: float = 20.0,
    bleach_tau_slow: float = 400.0,
    motion_gain_signal: float = 0.030,
    motion_gain_control: float = 0.040,
    motion_timescale: float = 1.0,
    noise_cv: float = 0.004,
    dropped_fraction: float = 0.0,
    timestamp_jitter: float = 0.0,
    subject: str = "synthetic-01",
    session: str = "synthetic",
    seed: int = 0,
) -> SyntheticDataset:
    r"""Generate a two-channel photometry recording with known hidden components.

    Parameters
    ----------
    fs
        Sampling rate in hertz, per channel. This is the rate after any
        de-interleaving a real rig would require.
    duration
        Recording length in seconds.
    event_times
        Times of an experimental event, in seconds. When given, a transient of
        fixed amplitude is placed after each one and the events are attached to
        the recording under the name ``"cue"``, which is what event-triggered
        average tests align to. When ``None``, no events are attached.
    event_response_amplitude
        Peak :math:`\Delta F/F` of each event-locked transient. Fixed rather than
        random so that the expected event-triggered average is known exactly.
    event_response_latency
        Delay in seconds from event onset to transient onset.
    n_transients
        Number of additional spontaneous transients at uniformly random times.
        Set to zero for a clean event-locked dataset.
    transient_amplitude
        Inclusive ``(low, high)`` range for spontaneous transient peak amplitudes.
    tau_rise, tau_decay
        Transient kernel time constants in seconds.
    baseline_signal, baseline_control
        Asymptotic fluorescence of each channel, in arbitrary units.
    bleach_amplitude
        Total photobleaching decay as a fraction of baseline.
    bleach_tau_fast, bleach_tau_slow
        Bleaching time constants in seconds. The control channel gets slightly
        different ones, as real channels do.
    motion_gain_signal, motion_gain_control
        How strongly the shared movement artefact enters each channel, in
        :math:`\Delta F/F` units per standard deviation of motion. Deliberately
        unequal, since a motion-correction step that assumes a gain of 1 is wrong.
    motion_timescale
        Smoothing timescale of the movement artefact, in seconds.
    noise_cv
        Additive Gaussian noise standard deviation, as a fraction of each
        channel's baseline.
    dropped_fraction
        Fraction of frames to delete, simulating acquisition dropouts. The first
        and last samples are always kept.
    timestamp_jitter
        Relative standard deviation of the sample-interval jitter. ``0.02`` gives
        realistically imperfect timestamps.
    subject, session
        Identifiers stored on the recording.
    seed
        Seed for the random generator. The default makes every call reproducible,
        which is what a test suite needs.

    Returns
    -------
    SyntheticDataset
        The recording and its ground truth.

    Raises
    ------
    ValidationError
        If ``duration`` is too short to hold two samples, if ``dropped_fraction``
        is outside [0, 1), or if any positive-valued parameter is not positive.

    Examples
    --------
    >>> from fluoroflow.datasets import synthetic_recording
    >>> data = synthetic_recording(duration=30.0, event_times=[5.0, 15.0, 25.0], seed=0)
    >>> data.recording
    Recording('synthetic-01'/'synthetic', traces=2, events=1, channels=1, duration=29.97 s)
    >>> round(data.signal.fs, 6)
    30.0
    >>> len(data.recording.event("cue"))
    3
    """
    rate = check_positive(fs, label="fs")
    span = check_positive(duration, label="duration")
    if not 0.0 <= dropped_fraction < 1.0:
        msg = f"dropped_fraction must lie in [0, 1), got {dropped_fraction!r}."
        raise ValidationError(msg)
    low, high = (float(transient_amplitude[0]), float(transient_amplitude[1]))
    if low > high:
        msg = (
            f"transient_amplitude must be (low, high) with low <= high, "
            f"got {transient_amplitude!r}."
        )
        raise ValidationError(msg)

    n = round(span * rate)
    if n < 2:
        msg = f"duration={span!r} s at fs={rate!r} Hz yields {n} sample(s); at least 2 are needed."
        raise ValidationError(msg)

    rng = np.random.default_rng(seed)
    time = _make_time(fs=rate, n=n, timestamp_jitter=float(timestamp_jitter), rng=rng)

    # --- transients ------------------------------------------------------
    kernel = transient_kernel(rate, tau_rise=tau_rise, tau_decay=tau_decay)
    onsets: list[float] = []
    amplitudes: list[float] = []
    if event_times is not None:
        for t_event in event_times:
            onsets.append(float(t_event) + float(event_response_latency))
            amplitudes.append(float(event_response_amplitude))
    if n_transients > 0:
        onsets.extend(rng.uniform(0.0, float(time[-1]), size=n_transients).tolist())
        amplitudes.extend(rng.uniform(low, high, size=n_transients).tolist())

    order = np.argsort(np.asarray(onsets, dtype=np.float64)) if onsets else np.empty(0, dtype=int)
    onset_times = np.asarray(onsets, dtype=np.float64)[order]
    onset_amps = np.asarray(amplitudes, dtype=np.float64)[order]

    transient_dff = np.zeros(n, dtype=np.float64)
    for t_on, amp in zip(onset_times, onset_amps, strict=True):
        start = int(np.searchsorted(time, t_on, side="left"))
        if start >= n:
            continue
        stop = min(n, start + kernel.size)
        transient_dff[start:stop] += amp * kernel[: stop - start]

    # --- movement artefact -----------------------------------------------
    walk = np.cumsum(rng.standard_normal(n))
    sigma_samples = max(float(motion_timescale) * rate, 1e-6)
    motion = gaussian_filter1d(walk, sigma_samples, mode="nearest")
    motion = motion - motion.mean()
    scale = float(motion.std())
    if scale > 0.0:
        motion = motion / scale

    # --- bleaching, noise, and the observed traces -----------------------
    bleach_signal = biexponential_bleach(
        time,
        baseline=baseline_signal,
        amplitude=bleach_amplitude,
        tau_fast=bleach_tau_fast,
        tau_slow=bleach_tau_slow,
    )
    bleach_control = biexponential_bleach(
        time,
        baseline=baseline_control,
        amplitude=bleach_amplitude * 0.8,
        tau_fast=bleach_tau_fast * 1.3,
        tau_slow=bleach_tau_slow * 0.85,
    )

    noise_sd_signal = float(noise_cv) * float(baseline_signal)
    noise_sd_control = float(noise_cv) * float(baseline_control)
    signal_f = bleach_signal * (
        1.0 + transient_dff + float(motion_gain_signal) * motion
    ) + noise_sd_signal * rng.standard_normal(n)
    control_f = bleach_control * (
        1.0 + float(motion_gain_control) * motion
    ) + noise_sd_control * rng.standard_normal(n)

    # --- simulated frame drops -------------------------------------------
    keep = np.ones(n, dtype=bool)
    if dropped_fraction > 0.0:
        keep = rng.random(n) >= float(dropped_fraction)
        keep[0] = True
        keep[-1] = True

    time = time[keep]
    transient_dff = transient_dff[keep]
    motion = motion[keep]
    bleach_signal = bleach_signal[keep]
    bleach_control = bleach_control[keep]
    signal_f = signal_f[keep]
    control_f = control_f[keep]

    # --- assemble ---------------------------------------------------------
    params: dict[str, Any] = {
        "fs": rate,
        "duration": span,
        "event_times": None if event_times is None else tuple(float(v) for v in event_times),
        "event_response_amplitude": float(event_response_amplitude),
        "event_response_latency": float(event_response_latency),
        "n_transients": int(n_transients),
        "transient_amplitude": (low, high),
        "tau_rise": float(tau_rise),
        "tau_decay": float(tau_decay),
        "baseline_signal": float(baseline_signal),
        "baseline_control": float(baseline_control),
        "bleach_amplitude": float(bleach_amplitude),
        "bleach_tau_fast": float(bleach_tau_fast),
        "bleach_tau_slow": float(bleach_tau_slow),
        "motion_gain_signal": float(motion_gain_signal),
        "motion_gain_control": float(motion_gain_control),
        "motion_timescale": float(motion_timescale),
        "noise_cv": float(noise_cv),
        "dropped_fraction": float(dropped_fraction),
        "timestamp_jitter": float(timestamp_jitter),
        "seed": int(seed),
    }

    signal = Trace(
        time=time, values=signal_f, name="Region0G", units="a.u.", meta={"role": "signal"}
    )
    control = Trace(
        time=time, values=control_f, name="Region0G_iso", units="a.u.", meta={"role": "control"}
    )
    spec = ChannelSpec(
        signal="Region0G",
        control="Region0G_iso",
        indicator="GCaMP-synthetic",
        excitation_nm=470.0,
        control_nm=415.0,
        region="synthetic",
    )
    events: tuple[Events, ...] = ()
    if event_times is not None:
        events = (Events(np.asarray(event_times, dtype=np.float64), name="cue"),)

    recording = Recording.from_traces(
        signal,
        control,
        events=events,
        channels=(spec,),
        subject=subject,
        session=session,
        meta={"synthetic": True, "seed": int(seed)},
    )

    truth = SyntheticTruth(
        time=time,
        transient_dff=transient_dff,
        motion=motion,
        bleach_signal=bleach_signal,
        bleach_control=bleach_control,
        motion_gain_signal=float(motion_gain_signal),
        motion_gain_control=float(motion_gain_control),
        transient_times=onset_times,
        transient_amplitudes=onset_amps,
        noise_sd_signal=noise_sd_signal,
        noise_sd_control=noise_sd_control,
        params=params,
    )
    return SyntheticDataset(recording=recording, truth=truth)
