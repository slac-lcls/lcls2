# MPI parallel analysis

## What it is

The same psana2 script that runs serially can run under `mpirun`. With
`DataSource(exp=..., run=...)` and more than one MPI rank, `DataSource` returns
an `MPIDataSource` and the ranks split into roles: one rank reads the small-data
(`.smd.xtc2`) files, event-builder ranks group them into batches, and big-data
ranks read the full events and run the user's event loop. Optional "SRV" ranks
collect per-event results into HDF5 files through `ds.smalldata()`.

This needs MPI (`mpi4py`) and, for real experiments, access to LCLS data. The
tests run the same code on synthetic data generated with `xtcwriter` and
`smdwriter`.

## Key concepts

### Roles

The ranks are assigned by `Communicators` in `psana/psana/psexp/node.py`:

| Role | Ranks | What it does |
|---|---|---|
| `smd0` | world rank 0 | Reads the small-data files and sends chunks to the event builders. |
| `eb` | ranks 1 .. `PS_EB_NODES` | Build events from the small data and send batches of event offsets to big-data ranks. |
| `bd` | the remaining psana ranks | Read the big-data files and yield events to `run.events()`; this is where user code sees events. |
| `srv` | the last `PS_SRV_NODES` ranks | Smalldata servers: receive `smd.event(...)` data and write HDF5. |

On `srv` ranks `DataSource` returns a `NullDataSource`, whose runs have no
events, so the same script runs everywhere. `Communicators` requires at least 3
ranks besides the `srv` ranks, and `MPIDataSource` requires more than
`PS_EB_NODES + 1` psana ranks.

With only one MPI rank, or with `PS_PARALLEL=none`, `DataSource(exp=...)`
returns a `SerialDataSource` instead.

### Environment variables

Read in the code (default in parentheses):

| Variable | Read in | Meaning |
|---|---|---|
| `PS_PARALLEL` (`mpi`) | `psana/psana/psexp/tools.py` | `mpi` or `none`. |
| `PS_EB_NODES` (`1`) | `psana/psana/psexp/node.py` | Number of event-builder ranks. |
| `PS_SRV_NODES` (`0`) | `psana/psana/psexp/node.py`, `datasource.py` | Number of smalldata server ranks. |
| `PS_EB_NODE_LOCAL` (`0`) | `psana/psana/datasource.py`, `node.py` | If true, one event builder per compute node (sets `PS_EB_NODES` to the node count). |
| `PS_SMD_N_EVENTS` (`20000`) | `psana/psana/psexp/tools.py` | Events per small-data chunk sent by smd0. |

`DataSource` also changes some defaults for experiment names starting with
`mfx` (`_force_mfx_overrides` in `psana/psana/datasource.py`): `PS_EB_NODES`
is set from the node count, `PS_SMD_N_EVENTS` to 1000 and `batch_size` to 1
unless you set them.

### Small data: `ds.smalldata()`

`DataSourceBase.smalldata(**kwargs)` returns the `SmallData` object
(`psana/psana/smalldata.py`) after calling
`setup_parms(filename=None, batch_size=1000, cache_size=None, callbacks=[], swmr_mode=False)`.
In parallel mode it raises an exception if `PS_SRV_NODES` is 0.

- `smd.event(evt, **kwargs)` (or `smd.event(timestamp, ...)`) stores
  per-event values; dict values are flattened with `/` in the key, and
  `align_group="name"` writes into an HDF5 group. Missing values are filled
  with -99999 (integers) or NaN (floats).
- `smd.summary` is true on the ranks that should do reductions and write
  summary data.
- `smd.sum(value)`, `max`, `min`, ... reduce across client (big-data) ranks
  onto client rank 0.
- `smd.save_summary(...)` writes summary data; `smd.done()` finishes the file.
- Output: each server writes `<base>_part<N>.h5`; in `done()` client rank 0
  joins them into `filename` using HDF5 virtual datasets. In serial mode the
  file is written directly.
- `callbacks=[f]`: the server calls `f(data_dict)` once per event with a flat
  dict of that event's values (`Server.handle`), as used in
  `psana/psana/tests/run_smalldata.py`.

### Other parallel helpers

- `ds.unique_user_rank()` is true on exactly one rank, useful for printing
  or final checks (`psana/psana/psexp/ds_base.py`).
- `smd_callback=` lets a function on the event-builder side choose which
  big-data rank gets each event (`psana/psana/tests/ds.py`, `smd_callback`).
- Calibration constants are fetched once by smd0 and shared through MPI shared
  memory (see [Calibration constants](calibration.md)).

## Minimal example

Adapted from `psana/psana/tests/run_smalldata.py`, which
`psana/psana/tests/byhand_mpi.py` runs as
`PS_SRV_NODES=2 mpirun -n 6 python run_smalldata.py` (1 smd0, 1 eb, 2 bd, 2 srv).
`xpptut15` run 14 is the synthetic test data set (to write it into `.tmp`,
see [Synthetic test data](../getting-started.md#synthetic-test-data)); replace
it with your experiment and run.

```python
import numpy as np
from psana import DataSource

xtc_dir = ".tmp"   # see Getting started
ds = DataSource(exp="xpptut15", run=14, dir=xtc_dir, batch_size=2)
smd = ds.smalldata(filename="smalldata_test.h5", batch_size=5)

for run in ds.runs():
    det = run.Detector("xppcspad")
    for evt in run.events():
        smd.event(evt, oneint=1, arrfloat=np.ones(2, dtype=float))

if smd.summary:
    smd.save_summary({"summary_array": np.arange(3)}, summary_int=1)
smd.done()
```

```bash
PS_SRV_NODES=2 mpirun -n 6 python my_script.py
```

## Where in the code

- `psana/psana/datasource.py`: chooses `MPIDataSource` / `NullDataSource`.
- `psana/psana/psexp/node.py`: `Communicators` (roles), and the smd0, event-builder and big-data node classes.
- `psana/psana/psexp/mpi_ds.py`: `MPIDataSource`, `RunParallel`.
- `psana/psana/psexp/tools.py`: `mode`, `PS_SMD_N_EVENTS`.
- `psana/psana/smdreader.pyx`, `psana/psana/eventbuilder.pyx`: small-data reading and event building (Cython).
- `psana/psana/smalldata.py`: `SmallData`, `Server`.
- Tests: `psana/psana/tests/byhand_mpi.py`, `run_smalldata.py`, `ds.py`.

API pages: [psana.psexp.mpi_ds](../api/python/mpi_ds.md),
[psana.smalldata](../api/python/smalldata.md),
[psana.psexp.ds_base](../api/python/ds_base.md).
