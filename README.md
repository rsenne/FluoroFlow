# FluoroFlow

Fiber photometry analysis that is simple, fast, and hard to get wrong.

> **Status:** the core data model is in place. IO, preprocessing, and analysis are
> next, and the public API is not yet stable.

FluoroFlow is the successor to `photonsoup`/RamiPho. Same science, rebuilt around
three commitments:

1. **An explicit data model.** A recording is data plus metadata plus a
   processing history, not a class whose constructor reads files off disk and
   mutates itself.
2. **Nothing implicit.** LED-state-to-wavelength mappings, channel pairings,
   sampling rates, and baseline windows are parameters you can see, not integers
   hardcoded in an `if`.
3. **Every numerical claim tested.** Preprocessing is validated against synthetic
   signals with known ground truth, plus property-based invariants and
   golden-file regressions.

## Install

```bash
git clone https://github.com/rsenne/FluoroFlow
cd FluoroFlow
uv sync --all-extras
```

## The data model

A `Trace` is an immutable fluorescence time series that owns its own time vector.
Its arrays are read-only and its sampling rate is derived from its own timestamps,
so a de-interleaved channel cannot report the multiplexed frame rate by accident.

```python
import numpy as np
from fluoroflow import Trace

t = Trace(np.arange(9000) / 30.0, counts, name="Region0G", units="a.u.")
t.fs  # 30.0, from this trace's timestamps
t.sampling  # dt spread, coefficient of variation, detected frame gaps
t.values.flags.writeable  # False
```

Transforms never mutate; they return a new trace with one step appended to its
history, and `step` is a required argument so an unrecorded transform cannot be
written by accident.

```python
from fluoroflow import Step, MEAN_REMOVED

detrended = t.derive(values=residual, step=Step("airpls", {"lam": 1e7}, tags={MEAN_REMOVED}))
detrended.describe_history()
# ' 1. Step('airpls', lam=10000000.0, tags=['mean-removed'])'
detrended.has_tag(MEAN_REMOVED)  # True; this is what stops dF/F running on it
```

A `Recording` holds the traces, the events, and the signal-to-control pairings for
one session. The pairing is declared, never inferred from a column name:

```python
from fluoroflow import ChannelSpec, Recording

rec = Recording.from_traces(
    signal,
    isosbestic,
    channels=(
        ChannelSpec(
            "Region0G",
            control="Region0G_iso",
            indicator="GCaMP6f",
            excitation_nm=470.0,
            control_nm=415.0,
            region="vCA1",
        ),
    ),
    subject="M1",
    session="fear-conditioning-day1",
)

for spec, sig, ctl in rec.pairs():
    ...  # ctl is None for channels declared without a control

rec = rec.map_traces(lambda tr: dff(tr))  # pipelines are function composition
```

`Events` stores behaviour as times, never as sample indices, since an index only
means something relative to one channel's clock:

```python
from fluoroflow import Events

freezing = Events.from_boolean(freeze_mask, behaviour_time, name="freezing")
freezing.total_duration
freezing.to_boolean(rec["Region0G"].time)  # resampled onto the photometry clock
```

## Synthetic data with known ground truth

`fluoroflow.datasets` generates recordings from an explicit forward model and keeps
the hidden components, so a preprocessing step can be asked whether it *recovered*
the signal rather than merely whether it ran.

```python
from fluoroflow.datasets import synthetic_recording

data = synthetic_recording(duration=300.0, event_times=[30.0, 90.0, 150.0], seed=0)
data.recording  # what a pipeline is allowed to see
data.truth.transient_dff  # what a perfect pipeline should return
data.truth.observable_dff  # what perfect bleach correction alone recovers
```

Knobs cover the artefacts that break real analyses: correlated movement entering
both channels at different gains, biexponential bleaching, timestamp jitter, and
dropped frames.

## Development

```bash
uv run pytest              # tests, including every docstring example
uv run ruff check .        # lint
uv run ruff format .       # format
uv run mypy                # types (strict)
uv run pre-commit install  # once, to wire up hooks
```

## License

MIT
