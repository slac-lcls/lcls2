# psana.psexp.run

The run classes yielded by `ds.runs()`. `Run` is the base class with the
methods most user code calls: `events()` (one `Event` per L1Accept),
`steps()` (one `Step` per BeginStep), `Detector(name, accept_missing=False, **kwargs)`,
and the properties `detnames`, `epicsinfo`, `scaninfo`, `xtcinfo`. Which
subclass you get depends on the data source:

| Class | Data source |
|---|---|
| `RunSerial` | `SerialDataSource` (`exp=`, one process) |
| `RunSingleFile` | `SingleFileDataSource` (`files=`) |
| `RunShmem` | `ShmemDataSource` (`shmem=`) |
| `RunDrp` | `DrpDataSource` (`drp=`) |
| `RunSmallData` | created internally (`eventbuilder_manager.py`, `smdreader_manager.py`) and passed to `smd_callback` |
| `RunParallel` | `MPIDataSource`; defined in [`psana.psexp.mpi_ds`](mpi_ds.md) |

`Run.__init__` also stores `expt`, `runnum` and `timestamp` (the experiment
name, run number and BeginRun timestamp that the data source passes in; the
data sources read them from the BeginRun dgrams). They have no docstrings, so
they are not listed below. `run.run()` returns `runnum`. `RunSmallData` does
not call `Run.__init__` and does not set these three attributes.

See [DataSource, Run and the event loop](../../features/datasource-run-events.md)
and [Detector interface](../../features/detector-interface.md).

::: psana.psexp.run
