# FluoroFlow

Fiber photometry analysis that is simple, fast, and hard to get wrong.

> **Status:** the core data model is in place. Preprocessing and event-triggered
> average (ETA) analysis are next, and the public API is not yet stable.

FluoroFlow is the successor to `photonsoup`/RamiPho, scoped tightly to what a
Neurophotometrics recording actually is: up to two signal channels and one
isosbestic control, plus behavioural events. Two commitments:

1. **An explicit data model.** A recording is data plus metadata plus a
   processing history, not a class whose constructor reads files off disk and
   mutates itself.
2. **Every transform is provenance-tracked.** Preprocessing steps never mutate
   in place; they return a new trace with one step appended to its history, so
   you can always ask what has already been done to a signal.

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
from fluoroflow import Step

detrended = t.derive(values=residual, step=Step("airpls", {"lam": 1e7}))
detrended.describe_history()
# ' 1. Step('airpls', lam=10000000.0)'
detrended.has_step("airpls")  # True
```

A `Recording` holds up to two signal traces, an optional shared isosbestic
control, and the events for one session:

```python
from fluoroflow import Recording

rec = Recording.from_traces(
    signal,
    isosbestic=iso,
    subject="M1",
    session="fear-conditioning-day1",
)

rec["Region0G"]  # look up any trace by name
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

## Development

```bash
uv run pytest              # tests
uv run ruff check .        # lint
uv run ruff format .       # format
uv run mypy                # types (strict)
uv run pre-commit install  # once, to wire up hooks
```

## License

MIT
