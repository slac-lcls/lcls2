# DataSource, Run and the event loop

## What it is

Every psana2 analysis follows the same pattern: create a `DataSource`, loop
over its runs, get detector objects from the run, and loop over the events of
the run (optionally grouped in steps of a scan):

```text
DataSource(...)            -> data source object (serial, single file, MPI, shmem, ...)
  .runs()                  -> Run objects, one per run
    run.Detector(name)     -> detector interface (see "Detector interface")
    run.events()           -> Event objects (L1Accept only)
    run.steps()            -> Step objects; step.events() -> Event objects
```

The same user code runs serially, under MPI, or on shared memory; the choice is
made by `DataSource` from its keyword arguments and the environment.

## Key concepts

### `DataSource(**kwargs)`

A factory function in `psana/psana/datasource.py` (exported as
`psana.DataSource`). It checks the keywords `shmem`, `exp`, `files`, `drp` in
that order and returns the matching class (details on the
[psana.datasource API page](../api/python/datasource.md)):

| You pass | You get | Typical use |
|---|---|---|
| `exp=`, `run=` (optionally `dir=`) | `SerialDataSource`, or under MPI `MPIDataSource` (see [MPI parallel analysis](mpi-parallel.md)) | offline analysis of a run |
| `files=` | `SingleFileDataSource` | one xtc2 file, no small data needed |
| `shmem=` | `ShmemDataSource` (see [Shared memory and live analysis](shmem-live.md)) | monitoring a running DAQ |
| `drp=` | `DrpDataSource` | Python code running inside the DAQ's DRP |

All of them derive from `DataSourceBase` (`psana/psana/psexp/ds_base.py`),
which reads the keyword arguments. Commonly used ones, with the defaults read
in `DataSourceBase.__init__`:

| Keyword | Default | Meaning (from the code) |
|---|---|---|
| `exp` | `None` | Experiment name. |
| `run` | `None` | Run number or list of run numbers; `None` means every run found in the xtc directory. |
| `dir` | `None` | xtc directory. If not given: `$SIT_PSDM_DATA/<exp[:3]>/<exp>/xtc` (default `SIT_PSDM_DATA` is `/reg/d/psdm`). |
| `files` | `None` | xtc2 file name (or list) for `SingleFileDataSource`. |
| `max_events` | `0` | Stop after this many events (0 = no limit). |
| `batch_size` | `1000` | Number of events per batch passed between MPI ranks. |
| `detectors` / `xdetectors` | `[]` | Keep / drop the stream files of these detectors. |
| `small_xtc` | `[]` | Detectors whose data are read from the big-data file instead of the small-data file. |
| `timestamps` | empty array | Only process these event timestamps (array or `.npy` path). |
| `live` | `False` | Wait for files that are still being written (see [Shared memory and live analysis](shmem-live.md)). |
| `skip_calib_load` | `[]` | Detector names (or `"all"`) whose calibration constants are not fetched. |
| `log_level` | `logging.INFO` | Logging level (a level name string is accepted). |

Unknown keywords are not an error: `DataSourceBase` logs
"Unrecognized kwarg" and ignores them. Some tests still pass a `filter=`
keyword, which is not in the list of known keywords and therefore has no
effect.

For `exp=` the data source expects the usual LCLS layout: big-data files
`<exp>-r<run>-s<stream>-c<chunk>.xtc2` in the xtc directory and small-data
files `smalldata/<exp>-r<run>-s<stream>-c<chunk>.smd.xtc2` next to them (see
[XTC2 data format](xtc2.md)).

### `Run`

`psana/psana/psexp/run.py`. The base class `Run` has the subclasses
`RunSerial`, `RunSingleFile`, `RunShmem`, `RunDrp`, `RunSmallData`, and
`RunParallel` (in `psana/psana/psexp/mpi_ds.py`). What user code typically uses:

- `run.expt`, `run.runnum`, `run.timestamp`: taken from the BeginRun
  transition in the data (so they are also set for `files=`, if the file has
  that information).
- `run.events()`: yields an `Event` for each L1Accept. Other transitions
  (SlowUpdate, BeginStep, ...) are consumed internally and update the EPICS and
  scan values; the loop ends at EndRun.
- `run.steps()`: yields a `Step` for each BeginStep; `step.events()` yields the
  L1Accepts until the matching EndStep.
- `run.Detector(name, accept_missing=False, **kwargs)`: see
  [Detector interface](detector-interface.md).
- `run.detnames`, `run.epicsinfo`, `run.scaninfo`, `run.xtcinfo`: what is in
  the run.
- Random access by timestamp: `RunSerial` and `RunParallel` have a
  `build_table()` context manager and `event(ts)`; see
  `psana/psana/tests/test_run_build_table.py`.

### `Event` and `Step`

`psana/psana/event.py` and `psana/psana/psexp/step.py`.

- `evt.timestamp`: 64-bit integer; upper 32 bits are seconds, lower 32 bits
  nanoseconds.
- `evt.datetime()`: the timestamp as a naive `datetime` (the epoch used by the
  code is 1990-01-01).
- `evt.run()`: returns a small `RunCtx` object (`psana/psana/psexp/run_ctx.py`)
  with only `expt`, `runnum`, `timestamp`, `intg_det`, not the full `Run`.
- `evt.service()`: the transition id of the event.
- `step.evt`: the BeginStep event; scan variables can be read from it with a
  scan detector, for example `run.Detector('motor2')`.

## Minimal example

Adapted from `psana/psana/tests/user_loops.py`. `xpptut15` run 14 is a
synthetic data set that the tests generate with the `xtcwriter` and `smdwriter`
tools; [Synthetic test data](../getting-started.md#synthetic-test-data) shows
how to write it into `.tmp` with `psana/psana/tests/setup_input_files.py`.
With real LCLS data you would use your experiment name and run number and
usually omit `dir`.

```python
from psana import DataSource

xtc_dir = ".tmp"   # written by setup_input_files.py (see Getting started)
ds = DataSource(exp="xpptut15", run=14, dir=xtc_dir)
for run in ds.runs():
    print(run.expt, run.runnum)
    det = run.Detector("xppcspad")                 # area detector
    edet = run.Detector("HX2:DVD:GCC:01:PMON")     # EPICS variable
    sdet = run.Detector("motor2")                  # scan variable
    for step in run.steps():
        for evt in step.events():
            calib = det.raw.calib(evt)
            print(evt.timestamp, sdet(evt), edet(evt), calib.shape)
```

Without steps, iterate `run.events()` directly. Reading one file:
`DataSource(files=".tmp/xpptut15-r0014-s000-c000.xtc2")` works the same way but
only sees the detectors stored in that file.

## Where in the code

- `psana/psana/datasource.py`: `DataSource`, `InvalidDataSource`.
- `psana/psana/psexp/ds_base.py`: `DataSourceBase` (keyword arguments, file discovery, `smalldata()`).
- `psana/psana/psexp/serial_ds.py`, `singlefile_ds.py`, `shmem_ds.py`, `drp_ds.py`, `null_ds.py`, `mpi_ds.py`: the data source classes.
- `psana/psana/psexp/run.py`: `Run` and its subclasses.
- `psana/psana/psexp/step.py`, `psana/psana/event.py`, `psana/psana/psexp/run_ctx.py`: `Step`, `Event`, `RunCtx`.
- `psana/psana/dgrammanager.py`: `DgramManager` (reads dgrams and Configure data).
- `psana/psana/psexp/envstore.py`, `envstore_manager.py`: EPICS and scan values.
- Examples: `psana/psana/tests/user_loops.py`, `psana/psana/tests/ds.py`.

API pages: [psana.datasource](../api/python/datasource.md),
[psana.psexp.ds_base](../api/python/ds_base.md),
[psana.psexp.run](../api/python/run.md),
[psana.event](../api/python/event.md),
[psana.dgrammanager](../api/python/dgrammanager.md).
