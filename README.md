# FluoroFlow

FluoroFlow analyzes Neurophotometrics-style fiber photometry recordings with
one or two signal channels, an optional shared isosbestic control, and behavioral events.
It provides an immutable data model, isosbestic artifact correction and dF/F,
and event-triggered averages (ETAs) within and across animals.

The public API is not yet stable.

## Install

```bash
git clone https://github.com/rsenne/FluoroFlow
cd FluoroFlow
uv sync --all-extras
```

Requires Python 3.11+.

## Data model

### Trace

A `Trace` contains read-only values and timestamps. Its sampling rate is the
inverse median timestamp interval.

```python
import numpy as np
from fluoroflow import Trace

signal = Trace(np.arange(9000) / 30.0, counts, name="Region0G", units="a.u.")
signal.fs         # 30.0
signal.dt         # 0.0333...
signal.sampling   # interval range, coefficient of variation, and frame gaps
signal.n_missing  # number of NaN samples
```

Transforms return a new `Trace` and record the operation in its history.

```python
cropped = signal.time_slice(10.0, 20.0)
cropped.history[-1]
# Step('time_slice', start=10.0, stop=20.0, n_dropped_before=300,
#      n_dropped_after=8400, n_kept=300)
cropped.has_step("time_slice")  # True
```

Useful methods include:

- `time_slice(start, stop)` selects a half-open time window.
- `interpolate_to(time)` linearly resamples onto new timestamps.
- `index_at(t)` finds the nearest sample.
- `to_frame()` returns a `pandas.DataFrame`.

### Events

`Events` stores onset times and optional durations in seconds, independent of a
trace's sample indices.

```python
from fluoroflow import Events

freezing = Events.from_boolean(freeze_mask, behavior_time, name="freezing")
freezing.total_duration
freezing.to_boolean(signal.time)         # render on the photometry clock
freezing.within(start=60.0, stop=120.0)  # select events by onset
```

### Recording

A `Recording` holds one or two signal channels, an optional shared isosbestic
control, and event sets for one session.

```python
from fluoroflow import Recording

rec = Recording.from_traces(
    signal,
    isosbestic=iso,
    subject="M1",
    session="fear-conditioning-day1",
    events=(freezing,),
)

rec["Region0G"]
rec = rec.map_traces(lambda tr: tr.time_slice(0.0, 1800.0))
```

## Preprocessing

`preprocess()` optionally filters the traces, fits an isosbestic baseline for
each signal, and returns either dF/F or a baseline-subtracted signal.

```python
from fluoroflow import DffOptions, PreprocessOptions, preprocess

out = preprocess(rec)
```

- **Lowpass:** a zero-phase Butterworth filter applied to signals and the
  isosbestic. Defaults: `cutoff_hz=3.0`, `order=2`.
- **Baseline fit:** robustly regress the isosbestic onto each signal
  using iteratively reweighted least squares with Tukey's bisquare weights
  (Keevers & Jean-Richard-dit-Bressel, *Neurophotonics*, 2025). Signal-only
  transients are downweighted, leaving shared bleaching and motion artifacts in
  the fit. The isosbestic is interpolated onto each signal's timestamps before
  fitting. Defaults: `tuning_constant=1.4`, `max_iter=50`, `tol=1e-6`.
- **Output:** when dF/F is enabled, compute `(signal - baseline) / baseline`
  from the filtered signal and fitted baseline. When it is disabled, subtract
  the fitted baseline instead.

| method | formula |
|---|---|
| `"dff"` | `(signal - baseline) / baseline` |
| `"z"` | `(dff - mean) / std` |
| `"mad_z"` | `(dff - median) / MAD` |
| `"null_z"` | `dff / rms(dff)` without recentering |

```python
out = preprocess(rec, PreprocessOptions(dff=DffOptions(method="null_z")))
out = preprocess(rec, PreprocessOptions(baseline=None, dff=None))  # lowpass only
```

Set an options field to `None` to skip it. Enabling dF/F requires the baseline
fit, and enabling the baseline fit requires an isosbestic control. The
isosbestic is filtered but is not baseline-corrected or converted to dF/F.

The default output records `lowpass` and `dff` on each signal:

```python
processed = preprocess(rec)["Region0G"]
[step.name for step in processed.history]
# ['lowpass', 'dff']
```

The stages are also available individually from `fluoroflow.preprocessing`:
`lowpass_filter`, `fit_isosbestic_baseline`, `baseline_correct`, and
`compute_dff`.

## Event-triggered averages

### Per animal

`animal_eta` aligns a window around each event and averages across trials.

```python
from fluoroflow import animal_eta

a1 = animal_eta(signal, freezing, window=(-2.0, 5.0), ci="bootstrap")
a1.mean, a1.sem, a1.ci_lower, a1.ci_upper
```

`ci` accepts `"t"`, `"bootstrap"`, or `None`. Trials whose windows extend
beyond the trace are dropped and counted in `n_dropped`.

### Across animals

`population_eta` averages animal-level ETAs, using animals rather than pooled
trials as the unit of variance.

```python
from fluoroflow import population_eta

pop = population_eta([a1, a2, a3], ci="t")
```

### Random-effects model

`bayesian_eta` fits an independent DerSimonian-Laird random-effects model at
each time point. It returns the population posterior, between-animal variance,
and shrunk estimates for each animal. The calculation is closed form and does
not use MCMC.

```python
from fluoroflow import bayesian_eta

bayes = bayesian_eta([a1, a2, a3])
bayes.population_mean, bayes.population_ci_lower, bayes.population_ci_upper
bayes.tau2          # between-animal variance by time point
bayes.animal_means  # shrunk animal estimates
```

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
uv run mypy
uv run pre-commit install  # install hooks once
```

## License

MIT
