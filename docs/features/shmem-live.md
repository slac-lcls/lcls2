# Shared memory and live analysis

## What it is

psana2 can analyze data while it is being taken, in two ways:

1. **Shared memory** (`DataSource(shmem=tag)`): read events from a
   shared-memory server on the same machine. In the DAQ the server is the
   monitoring event builder (MEB, executable `monReqServer`), which publishes
   built events under a tag; for tests the `shmemServer` tool serves events
   from an xtc2 file. This is what online monitoring (AMI) uses.
2. **Live files** (`DataSource(exp=..., run=..., live=True)`): read the xtc2
   files of a run that is still being recorded, waiting for data that has not
   been written yet.

Both need a running DAQ (or the test servers below) on the machine where the
analysis runs.

## Key concepts

### Shared-memory mode

- `DataSource(shmem=tag)` returns a `ShmemDataSource`
  (`psana/psana/psexp/shmem_ds.py`); `runs()` yields `RunShmem` objects
  (`psana/psana/psexp/run.py`).
- `DgramManager(["shmem"], tag=...)` (`psana/psana/dgrammanager.py`) connects a
  `PyShmemClient` (`psana/psana/shmem/shmem.pyx`, a Cython wrapper of the C++
  `psalg::shmem::ShmemClient` in `psalg/psalg/shmem/ShmemClient.hh`) and
  starts a background reader thread. Connecting is retried up to
  `SHMEM_CONN_MAX_RETRIES` (100) times.
- The reader thread queues every transition, but keeps only the most recent
  L1Accept. If the analysis is slower than the data rate, events are skipped:
  shared-memory mode is for monitoring, not for processing every event.
- No event building happens in psana in this mode; the server already
  delivers built events.
- Calibration constants: with the optional `supervisor` keyword one process
  fetches the constants and broadcasts them to the others over ZMQ (see
  `ShmemDataSource` and `RunShmem._setup_run_calibconst`).
- Under MPI, `DataSource(shmem=...)` gives the first `PS_SRV_NODES` ranks a
  `NullDataSource` (smalldata servers) and the others a `ShmemDataSource`
  (`psana/psana/datasource.py`).

### Shared-memory servers

- **MEB**: `monReqServer` (`psdaq/psdaq/monreq/monReqServer.cc`) serves the
  events it receives into shared memory through `XtcMonitorServer`
  (`psalg/psalg/shmem/XtcMonitorServer.hh`). Its `-t` option sets the tag
  (default: the instrument name given with `-P`), `-n` the number of event
  buffers and `-q` the number of event queues. In `psdaq/psdaq/cnf/tmo.cnf`,
  AMI workers read `psana://shmem={hutch}`.
- **Test server**: `shmemServer` (`psalg/psalg/shmem/src/shmemServer.cc`)
  serves xtc2 files:
  `shmemServer -p <partitionTag> -n <numberOfBuffers> -s <sizeOfBuffers> -f <file>`
  (other inputs: `-l` file list, `-x` run prefix, `-d` directory; `-c` number
  of clients, `-r` rate, `-L` loops).

### Live mode (`live=True`)

`DataSourceBase` (`psana/psana/psexp/ds_base.py`) and the small-data reader:

- With `live=True` the number of read retries comes from `PS_R_MAX_RETRIES`
  (default 60); without it, retries are disabled.
- Files may still be named `*.xtc2.inprogress`. When no new data is available
  the small-data reader (`psana/psana/psexp/smdreader_manager.py`) retries
  once per second up to that limit, and stops waiting early once the
  `.xtc2.inprogress` files it saw have been renamed to `.xtc2`.
- The list of files for the run is taken from the logbook web service
  (`.../lgbk/<exp>/ws/<run>/current_files_for_live_mode`), polled until it is
  stable for `PS_LIVE_FILE_LIST_STABLE_POLLS` polls (default 3). In live mode
  the directory is not scanned as a fallback.

## Minimal example

Shared memory, from `psana/psana/app/shmemClientSimple.py` (installed as the
`shmemClientSimple` command). The tag must match the server's (`-t` of
`monReqServer`, `-p` of `shmemServer`):

```python
from psana import DataSource

ds = DataSource(shmem="tst")          # shared-memory tag
run = next(ds.runs())
for evt in run.events():
    print(evt.service(), evt.timestamp)
```

The test `psana/psana/tests/test_shmem.py` sets up the same thing without a
DAQ: it writes a file with `xtcwriter -t -n 64 -f data_shmem.xtc2`, starts
`shmemServer -c 4 -n 10 -f data_shmem.xtc2 -p shmem_test_<pid> -s 0x80000`, and
runs clients with `DataSource(shmem="shmem_test_<pid>")`.

Live files, as in `psana/psana/tests/test_live_transfer.py` (requires a run
that is being recorded and access to the logbook service):

```python
ds = DataSource(exp=exp, run=run, live=True, batch_size=1, max_events=5)
for evt in next(ds.runs()).events():
    ...
```

## Where in the code

- `psana/psana/psexp/shmem_ds.py`: `ShmemDataSource`.
- `psana/psana/psexp/run.py`: `RunShmem`.
- `psana/psana/dgrammanager.py`: shared-memory reader thread.
- `psana/psana/shmem/shmem.pyx`: `PyShmemClient`.
- `psalg/psalg/shmem/`: `ShmemClient`, `XtcMonitorServer`, `shmemServer`, `shmemClient`, `shmemWriter`.
- `psdaq/psdaq/monreq/monReqServer.cc`: the MEB.
- `psana/psana/psexp/ds_base.py`, `psana/psana/psexp/smdreader_manager.py`: live-mode file discovery and retries.

Note: `psana/psana/psexp/mpi_shmem.py` is unrelated to `shmem=`; it manages
MPI-3 shared-memory windows used to share arrays between ranks of an MPI job.

API pages: [psana.psexp.shmem_ds](../api/python/shmem_ds.md),
[psana.dgrammanager](../api/python/dgrammanager.md),
[psana.psexp.run](../api/python/run.md);
C++: [C++ API](../api/cpp/index.html) (`psalg::shmem`).
